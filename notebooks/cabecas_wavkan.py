# -*- coding: utf-8 -*-
"""Plugin do P3: cabeças Wav-KAN (KAN com wavelets) no protocolo único.

Carregue com ``pc.carregar_plugin(caminho)`` (o SHA-256 deste arquivo vai para cada
``metricas.json``, separado do hash do protocolo, que não muda). Não altera nada do
``protocolo_comum.py``: só registra cabeças e configs e oferece funções de teste,
de medição e de execução.

Camada (Bozorgasl & Chen, arXiv 2405.12832), igual ao ``WavKANLayer`` do
torch-conv-kan (commit fixado em ``TORCH_CONV_KAN_COMMIT``)::

    z = (x - translation) / scale                  # tensor (lote, saída, entrada)
    y = BN( sum_i wavelet_weights * psi(z) + base_weight @ SiLU(x) )

* 4 parâmetros por aresta (``scale``, ``translation``, ``wavelet_weights``,
  ``base_weight``) + a BatchNorm1d de cada camada (2 por saída).
* Única diferença da referência: |scale| >= 1e-3, preservando o sinal. Na
  inicialização (scale = 1) não muda nada.
* Wavelets: chapéu mexicano (padrão do artigo), DoG e Morlet (omega0 = 5).

Cabeça = 2 camadas ``[in, h, n]``, sem ativação entre elas (como o ``WavKAN`` do
torch-conv-kan; o Dropout 0,5 antes da cabeça é o do protocolo). Para in = 2048 e
n = 2: ``8202·h + 4`` parâmetros.

| id | wavelet | h | parâmetros (E/S) | papel |
|----|---------|---|------------------|-------|
| W1 | chapéu mexicano | 128 | 1.049.860 | mesma topologia da KAN spline (d) |
| W2 | chapéu mexicano | 32  |   262.468 | mesmo orçamento da MLP-128 (b) |
| W3 | DoG             | 128 | 1.049.860 | sensibilidade à wavelet |
| W4 | Morlet          | 128 | 1.049.860 | sensibilidade à wavelet |
| W5 | chapéu mexicano | 480 | 3.936.964 | mesmo orçamento da KAN spline (d) e da MLP-iso (c) |

A wavelet dos orçamentos casados (chapéu mexicano) foi fixada antes de rodar.
"""
from __future__ import annotations

import ast
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import protocolo_comum as pc

VERSAO = "1.0.0"

TORCH_CONV_KAN_URL = "https://github.com/IvanDrokin/torch-conv-kan.git"
TORCH_CONV_KAN_COMMIT = "b412a3b839e35d0ddce378f833c2e6aa258cc251"

ESCALA_MIN = 1e-3
OMEGA0_MORLET = 5.0
CONST_CHAPEU = 2.0 / (math.sqrt(3.0) * math.pi**0.25)
WAVELETS = ("mexican_hat", "dog", "morlet")
ROTULO_WAVELET = {"mexican_hat": "mexhat", "dog": "dog", "morlet": "morlet"}

# id -> (wavelet, largura da camada oculta)
W = {
    "W1": ("mexican_hat", 128),
    "W2": ("mexican_hat", 32),
    "W3": ("dog", 128),
    "W4": ("morlet", 128),
    "W5": ("mexican_hat", 480),
}
PRINCIPAIS = ("W1", "W2")
SENSIBILIDADE = ("W3", "W4", "W5")
BACKBONE_RESNET152 = 58_143_808


# ---------------------------------------------------------------------------
# Camada e cabeça
# ---------------------------------------------------------------------------
def wavelet(z: torch.Tensor, tipo: str) -> torch.Tensor:
    if tipo == "mexican_hat":
        return CONST_CHAPEU * (z**2 - 1) * torch.exp(-0.5 * z**2)
    if tipo == "dog":
        return -z * torch.exp(-0.5 * z**2)
    if tipo == "morlet":
        return torch.cos(OMEGA0_MORLET * z) * torch.exp(-0.5 * z**2)
    raise ValueError(f"wavelet desconhecida {tipo!r}; use uma de {WAVELETS}")


def escala_segura(s: torch.Tensor) -> torch.Tensor:
    """|s| >= ESCALA_MIN preservando o sinal (s = 0 vira +ESCALA_MIN)."""
    return torch.where(s >= 0, s.clamp(min=ESCALA_MIN), s.clamp(max=-ESCALA_MIN))


class WavKANLinear(nn.Module):
    """Uma camada Wav-KAN: in_features -> out_features (com BatchNorm1d na saída)."""

    def __init__(self, in_features: int, out_features: int, wavelet: str = "mexican_hat"):
        super().__init__()
        if wavelet not in WAVELETS:
            raise ValueError(f"wavelet desconhecida {wavelet!r}; use uma de {WAVELETS}")
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet = wavelet
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(1) - self.translation) / escala_segura(self.scale)
        saida_wav = (wavelet(z, self.wavelet) * self.wavelet_weights).sum(dim=2)
        saida_base = F.linear(F.silu(x), self.base_weight)
        return self.bn(saida_wav + saida_base)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, wavelet={self.wavelet}"


# Diagnóstico da última unidade (preenchido pelas cabeças; lido por executar_fila_wavkan)
DIAG_ATUAL: dict = {}
_contador = [0]


class CabecaWavKAN(nn.Module):
    """[in, h, n] com duas WavKANLinear em sequência.

    Diagnóstico só dos pesos: média e dp de ``scale`` e ``translation`` ao fim de cada
    fase. A fase é inferida da entrada: na fase 1 o backbone está congelado e as
    features chegam sem gradiente; na fase 2 chegam com gradiente. Isso não muda a
    saída da cabeça.
    """

    def __init__(self, in_features: int, oculta: int, n_saidas: int, wavelet: str = "mexican_hat"):
        super().__init__()
        self.wavelet = wavelet
        self.dims = (in_features, oculta, n_saidas)
        self.camadas = nn.ModuleList(
            [WavKANLinear(in_features, oculta, wavelet), WavKANLinear(oculta, n_saidas, wavelet)]
        )
        self._fase_vista = None
        self.diagnostico = {"dims": list(self.dims), "wavelet": wavelet}
        _contador[0] += 1
        DIAG_ATUAL[_contador[0]] = self.diagnostico

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            fase = 2 if x.requires_grad else 1
            if self._fase_vista == 1 and fase == 2:
                self._registrar("fim_fase1")
            elif self._fase_vista is None and fase == 2:
                self.diagnostico["observacao"] = "retomada na fase 2: o fim da fase 1 não foi observado nesta sessão"
            self._fase_vista = fase
        elif self._fase_vista is not None and f"fim_fase{self._fase_vista}" not in self.diagnostico:
            self._registrar(f"fim_fase{self._fase_vista}")
        for c in self.camadas:
            x = c(x)
        return x

    def como_sequencial(self) -> nn.Sequential:
        """As mesmas camadas (mesmos parâmetros), sem o diagnóstico — para medir tempo."""
        return nn.Sequential(*self.camadas)

    @torch.no_grad()
    def estatisticas(self) -> dict:
        out = {}
        for i, c in enumerate(self.camadas):
            s, t = c.scale.detach().float(), c.translation.detach().float()
            out[f"camada{i}"] = {
                "scale_media": float(s.mean()),
                "scale_dp": float(s.std()),
                "scale_abs_min": float(s.abs().min()),
                "frac_scale_na_guarda": float((s.abs() <= ESCALA_MIN).float().mean()),
                "frac_scale_negativa": float((s < 0).float().mean()),
                "translation_media": float(t.mean()),
                "translation_dp": float(t.std()),
                "translation_min": float(t.min()),
                "translation_max": float(t.max()),
            }
        return out

    def _registrar(self, rotulo: str) -> None:
        self.diagnostico[rotulo] = self.estatisticas()


def n_parametros(in_features: int, oculta: int, n_saidas: int) -> int:
    """4 por aresta + BatchNorm (peso e viés) de cada camada."""
    return 4 * (in_features * oculta + oculta * n_saidas) + 2 * (oculta + n_saidas)


def nome_cabeca(wavelet_: str, oculta: int) -> str:
    return f"wavkan_{ROTULO_WAVELET[wavelet_]}_h{oculta}"


def _fabrica(wavelet_: str, oculta: int):
    def fabrica(in_f, n, **kw):
        cab = CabecaWavKAN(in_f, oculta, n, wavelet_)
        n_par = sum(p.numel() for p in cab.parameters())
        assert n_par == n_parametros(in_f, oculta, n), (n_par, n_parametros(in_f, oculta, n))
        return cab

    return fabrica


# ---------------------------------------------------------------------------
# Registro: cabeças (nome descritivo + apelido W1..W5) e configs
# ---------------------------------------------------------------------------
CABECAS = {w: nome_cabeca(*v) for w, v in W.items()}
CONFIGS_ES = {w: f"resnet152__{CABECAS[w]}" for w in W}
CONFIGS_MLSD = {w: f"mlsd__resnet152__{CABECAS[w]}" for w in PRINCIPAIS}
EXPERIMENTO = {"es": "wavkan", "dist": "wavkan_mlsd"}

for _w, (_wav, _h) in W.items():
    pc.registrar_cabeca(CABECAS[_w], _fabrica(_wav, _h))
    # O apelido é o que a parte 7 do 01 procura (mede só W1 e W5).
    pc.registrar_cabeca(_w, _fabrica(_wav, _h))
    pc.registrar_config(
        CONFIGS_ES[_w],
        {"backbone": "resnet152", "cabecas": {"es": CABECAS[_w]},
         "descricao": f"P3 {_w}: ResNet-152 + Wav-KAN {_wav} [2048,{_h},2]"},
    )
for _w in PRINCIPAIS:
    _wav, _h = W[_w]
    pc.registrar_config(
        CONFIGS_MLSD[_w],
        {"backbone": "resnet152", "cabecas": {"dist": CABECAS[_w]},
         "descricao": f"P3 {_w} MLSD: ResNet-152 + Wav-KAN {_wav} [2048,{_h},3]"},
    )


def id_da_config(cfg: str) -> str | None:
    for w in W:
        if cfg in (CONFIGS_ES[w], CONFIGS_MLSD.get(w)):
            return w
    return None


def checar_parametros() -> dict:
    """Contagens da tabela do plano (E/S, n = 2) e da MLSD (n = 3); levanta erro se divergirem."""
    esperado_es = {"W1": 1_049_860, "W2": 262_468, "W3": 1_049_860, "W4": 1_049_860, "W5": 3_936_964}
    out = {"es": {}, "dist": {}}
    for w, (wav, h) in W.items():
        n = sum(p.numel() for p in pc._CABECAS[CABECAS[w]](2048, 2).parameters())
        assert n == 8202 * h + 4 == esperado_es[w], (w, n)
        out["es"][w] = {"cabeca": n, "treinaveis_fase2": BACKBONE_RESNET152 + n}
    for w in PRINCIPAIS:
        wav, h = W[w]
        n = sum(p.numel() for p in pc._CABECAS[CABECAS[w]](2048, 3).parameters())
        assert n == n_parametros(2048, h, 3) == 8206 * h + 6, (w, n)
        out["dist"][w] = {"cabeca": n, "treinaveis_fase2": BACKBONE_RESNET152 + n}
    assert out["es"]["W1"]["treinaveis_fase2"] == 59_193_668
    assert out["es"]["W2"]["treinaveis_fase2"] == 58_406_276
    assert out["es"]["W5"]["treinaveis_fase2"] == 62_080_772
    return out


# ---------------------------------------------------------------------------
# Referências para a checagem de equivalência
# ---------------------------------------------------------------------------
def referencia_numpy(x: np.ndarray, estado: dict, wavelet_: str, modo_bn: str = "treino", eps: float = 1e-5) -> np.ndarray:
    """A mesma conta em NumPy (float64), a partir do state_dict de uma WavKANLinear.

    modo_bn: "treino" (estatística do lote, variância enviesada) ou "aval" (médias móveis).
    """
    p = {k: v.detach().cpu().double().numpy() for k, v in estado.items()}
    x = np.asarray(x, dtype=np.float64)
    s = p["scale"]
    s = np.where(s >= 0, np.maximum(s, ESCALA_MIN), np.minimum(s, -ESCALA_MIN))
    z = (x[:, None, :] - p["translation"][None]) / s[None]
    g = np.exp(-0.5 * z**2)
    if wavelet_ == "mexican_hat":
        psi = CONST_CHAPEU * (z**2 - 1) * g
    elif wavelet_ == "dog":
        psi = -z * g
    elif wavelet_ == "morlet":
        psi = np.cos(OMEGA0_MORLET * z) * g
    else:
        raise ValueError(wavelet_)
    y = (psi * p["wavelet_weights"][None]).sum(axis=2) + (x / (1 + np.exp(-x))) @ p["base_weight"].T
    if modo_bn == "treino":
        m, v = y.mean(0), y.var(0)
    else:
        m, v = p["bn.running_mean"], p["bn.running_var"]
    return (y - m) / np.sqrt(v + eps) * p["bn.weight"] + p["bn.bias"]


def classes_torch_conv_kan(dir_repo) -> dict:
    """Extrai SÓ as classes WavKANLayer (kans/layers.py) e WavKAN (kans/kan.py) do clone fixado.

    Lê o código-fonte com ``ast`` e executa apenas essas duas classes (contas de torch),
    sem importar o pacote (que puxa einops e outras dependências).
    """
    dir_repo = Path(dir_repo)
    ns = {"torch": torch, "nn": nn, "F": F, "math": math}
    for arq, nome in (("kans/layers.py", "WavKANLayer"), ("kans/kan.py", "WavKAN")):
        fonte = (dir_repo / arq).read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        defs = [n for n in arvore.body if isinstance(n, ast.ClassDef) and n.name == nome]
        if len(defs) != 1:
            raise RuntimeError(f"{nome} não encontrada (ou repetida) em {arq}")
        modulo = ast.Module(body=defs, type_ignores=[])
        exec(compile(modulo, f"{dir_repo.name}/{arq}", "exec"), ns)
    return {"WavKANLayer": ns["WavKANLayer"], "WavKAN": ns["WavKAN"]}


def copiar_para_referencia(camada: WavKANLinear, ref: nn.Module) -> None:
    """Copia os pesos de uma WavKANLinear para um WavKANLayer do torch-conv-kan."""
    sd = {k.replace("base_weight", "weight1"): v for k, v in camada.state_dict().items()}
    ref.load_state_dict(sd, strict=True)


def perturbar_(mod: nn.Module, semente: int = 0) -> None:
    """Afasta scale/translation/BN da inicialização (|scale| em [0,5; 1,5], com sinais trocados)."""
    g = torch.Generator().manual_seed(semente)
    with torch.no_grad():
        for c in [m for m in mod.modules() if isinstance(m, WavKANLinear)]:
            sinal = torch.where(torch.rand(c.scale.shape, generator=g) < 0.2, -1.0, 1.0)
            c.scale.copy_(sinal * (0.5 + torch.rand(c.scale.shape, generator=g)))
            c.translation.copy_(torch.randn(c.translation.shape, generator=g) * 0.5)
            c.bn.weight.copy_(1 + 0.1 * torch.randn(c.out_features, generator=g))
            c.bn.bias.copy_(0.1 * torch.randn(c.out_features, generator=g))
            c.bn.running_mean.copy_(0.2 * torch.randn(c.out_features, generator=g))
            c.bn.running_var.copy_(0.5 + torch.rand(c.out_features, generator=g))


def entrada_teste(lote: int, in_features: int, semente: int = 0) -> torch.Tensor:
    """Entrada parecida com as features da ResNet (>= 0, algumas grandes) com alguns negativos."""
    g = torch.Generator().manual_seed(semente)
    x = torch.relu(torch.randn(lote, in_features, generator=g)) * 2.0
    x[:, :8] = torch.randn(lote, 8, generator=g)
    return x


def checar_equivalencia(dir_repo=None, dispositivo="cpu", atol: float = 1e-5, lote: int = 16, in_f: int = 64,
                        out_f: int = 24) -> dict:
    """Compara WavKANLinear com a referência NumPy e, se ``dir_repo`` for dado, com o torch-conv-kan.

    3 wavelets × BN em modo treino e avaliação, com scale/translation/BN afastados da
    inicialização. Com o torch-conv-kan compara também os gradientes (modo treino) e a
    cabeça [in, 12, 2] inteira contra o ``WavKAN`` dele. fp32; atol 1e-5.
    """
    dispositivo = torch.device(dispositivo)
    ref_cls = classes_torch_conv_kan(dir_repo) if dir_repo else None

    def nova(wav, estado):
        c = WavKANLinear(in_f, out_f, wav)
        c.load_state_dict(estado)
        return c

    linhas = []
    for i, wav in enumerate(WAVELETS):
        torch.manual_seed(100 + i)
        base = WavKANLinear(in_f, out_f, wav)
        perturbar_(base, semente=i)
        estado = {k: v.clone() for k, v in base.state_dict().items()}
        x = entrada_teste(lote, in_f, semente=i)
        for modo in ("treino", "aval"):
            treino = modo == "treino"
            minha = nova(wav, estado).to(dispositivo).train(treino)
            with torch.no_grad():
                y = minha(x.to(dispositivo)).cpu().double().numpy()
            y_np = referencia_numpy(x.numpy(), estado, wav, modo, eps=base.bn.eps)
            linha = {"wavelet": wav, "modo_bn": modo, "max_dif_numpy": float(np.abs(y - y_np).max())}
            if ref_cls is not None:
                ref = ref_cls["WavKANLayer"](in_f, out_f, wavelet_type=wav)
                copiar_para_referencia(nova(wav, estado), ref)
                ref = ref.to(dispositivo).train(treino)
                minha = nova(wav, estado).to(dispositivo).train(treino)
                xa = x.to(dispositivo).clone().requires_grad_(treino)
                xb = x.to(dispositivo).clone().requires_grad_(treino)
                ya, yb = minha(xa), ref(xb)
                linha["max_dif_torch_conv_kan"] = float((ya - yb).detach().abs().max())
                if treino:
                    (ya**2).sum().backward()
                    (yb**2).sum().backward()
                    pares = [(minha.scale, ref.scale), (minha.translation, ref.translation),
                             (minha.wavelet_weights, ref.wavelet_weights), (minha.base_weight, ref.weight1),
                             (minha.bn.weight, ref.bn.weight), (minha.bn.bias, ref.bn.bias), (xa, xb)]
                    linha["max_dif_grad_torch_conv_kan"] = max(float((a.grad - b.grad).abs().max()) for a, b in pares)
            linhas.append(linha)
    cabeca = []
    if ref_cls is not None:
        for i, wav in enumerate(WAVELETS):
            torch.manual_seed(200 + i)
            cab = CabecaWavKAN(in_f, 12, 2, wav)
            perturbar_(cab, semente=10 + i)
            ref = ref_cls["WavKAN"]([in_f, 12, 2], wavelet_type=wav)
            camadas_ref = [m for m in ref.layers if hasattr(m, "weight1")]
            assert len(camadas_ref) == 2
            for a, b in zip(cab.camadas, camadas_ref):
                copiar_para_referencia(a, b)
            x = entrada_teste(lote, in_f, semente=20 + i).to(dispositivo)
            cab, ref = cab.to(dispositivo).eval(), ref.to(dispositivo).eval()
            with torch.no_grad():
                d = float((cab(x) - ref(x)).abs().max())
            cabeca.append({"wavelet": wav, "max_dif_cabeca_torch_conv_kan": d})
    difs = [v for l in linhas + cabeca for k, v in l.items() if k.startswith("max_dif")]
    return {"dispositivo": str(dispositivo), "atol": atol, "camadas": linhas, "cabeca": cabeca,
            "max_dif": max(difs), "ok": bool(max(difs) <= atol),
            "torch_conv_kan_commit": TORCH_CONV_KAN_COMMIT if dir_repo else None}


# ---------------------------------------------------------------------------
# Medição só da cabeça (tensores aleatórios, sem imagens)
# ---------------------------------------------------------------------------
def medir_cabeca(cab: nn.Module, in_features: int, lote: int, modo: str, n_aquec: int = 20, n_medidas: int = 200,
                 dispositivo="cuda", semente: int = 0) -> dict:
    """Tempo por chamada (ms) e memória de pico da cabeça. modo: "fwd_bwd" (treino) ou "inferencia".

    Na GPU usa eventos CUDA por chamada; na CPU, perf_counter. Devolve média, dp,
    mediana (das ``n_medidas``) e o pico de memória alocada além da já ocupada.
    """
    dispositivo = torch.device(dispositivo)
    cab = cab.to(dispositivo)
    g = torch.Generator(device="cpu").manual_seed(semente)
    x = (torch.relu(torch.randn(lote, in_features, generator=g)) * 2).to(dispositivo)
    if modo == "fwd_bwd":
        cab.train()
        for p in cab.parameters():
            p.requires_grad_(True)
    else:
        cab.eval()
    cuda = dispositivo.type == "cuda"

    def passo():
        if modo == "fwd_bwd":
            for p in cab.parameters():
                p.grad = None
            cab(x).float().square().mean().backward()
        else:
            with torch.no_grad():
                cab(x)

    for _ in range(n_aquec):
        passo()
    mem = None
    if cuda:
        torch.cuda.synchronize(dispositivo)
        base = torch.cuda.memory_allocated(dispositivo)
        torch.cuda.reset_peak_memory_stats(dispositivo)
        passo()
        torch.cuda.synchronize(dispositivo)
        mem = (torch.cuda.max_memory_allocated(dispositivo) - base) / 1024**2
    tempos = []
    for _ in range(n_medidas):
        if cuda:
            ini, fim = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ini.record()
            passo()
            fim.record()
            torch.cuda.synchronize(dispositivo)
            tempos.append(ini.elapsed_time(fim))
        else:
            t0 = time.perf_counter()
            passo()
            tempos.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(tempos)
    return {"modo": modo, "lote": lote, "ms_media": float(t.mean()), "ms_dp": float(t.std(ddof=1)),
            "ms_mediana": float(np.median(t)), "ms_por_imagem": float(t.mean() / lote), "mem_pico_mb": mem,
            "n_aquec": n_aquec, "n_medidas": n_medidas}


# ---------------------------------------------------------------------------
# Execução de uma fila (envolve pc.executar_unidade) com diagnóstico e checagem de NaN
# ---------------------------------------------------------------------------
def montar_fila_sementes(experimento: str, configs, sementes, coluna_dobra_de=None, dobras=range(pc.N_DOBRAS)):
    """Ordem semente -> config -> dobra: com o saldo no fim, todas as configs já têm a semente 42.

    ``coluna_dobra_de(k)`` dá a coluna de dobra da dobra k (MLSD: ``fold_dist_dk``);
    None = "fold" (entrada/saída).
    """
    return [
        {"experimento": experimento, "config": c, "semente": int(s), "dobra": int(d),
         "coluna_dobra": coluna_dobra_de(d) if coluna_dobra_de else "fold"}
        for s in sementes
        for c in configs
        for d in dobras
    ]


def dir_unidade(raiz_resultados, unidade) -> Path:
    return pc._caminhos_unidade(raiz_resultados, ".", unidade)[0]


def executar_fila_wavkan(sessao, fila, raiz_resultados, raiz_checkpoints, previsao=None, **kw) -> list:
    """Como pc.executar_fila, mais:

    * ``diagnostico_wavkan.json`` na pasta de cada unidade concluída (scale/translation
      ao fim de cada fase; equivalente ao diagnóstico de grade da KAN spline);
    * se a perda virar NaN/inf, grava ``falha_nan.json`` na pasta da unidade e PARA a fila;
    * ``previsao(unidade) -> str`` (opcional) é impresso antes de cada unidade a rodar.
    """
    feitos = []
    for u in fila:
        d = dir_unidade(raiz_resultados, u)
        if (d / "metricas.json").exists():
            continue
        if previsao is not None:
            print(previsao(u))
        DIAG_ATUAL.clear()
        try:
            r = pc.executar_unidade(sessao, u, raiz_resultados, raiz_checkpoints, **kw)
        except FloatingPointError as e:
            pc.gravar_json_atomico(
                {"erro": str(e), "quando": pc.agora_iso(), "unidade": u, "plugin_versao": VERSAO,
                 "diagnostico_parcial": list(DIAG_ATUAL.values())},
                d / "falha_nan.json",
            )
            raise RuntimeError(
                f"Perda não finita em {u['config']} s{u['semente']} dobra {u['dobra']} ({e}). "
                f"Registrado em {d / 'falha_nan.json'}. Pare aqui e avise a sessão P3."
            ) from e
        if r is None:
            continue
        diag = list(DIAG_ATUAL.values())
        pc.gravar_json_atomico(
            {"unidade": r["unidade"], "plugin_versao": VERSAO, "cabecas": diag, "gravado_em": pc.agora_iso()},
            d / "diagnostico_wavkan.json",
        )
        feitos.append(r["unidade"])
    return feitos


def ler_diagnosticos(raiz_resultados, experimento: str):
    """Uma linha por (unidade, camada, momento) com as estatísticas de scale/translation."""
    import json

    import pandas as pd

    linhas = []
    for arq in sorted(Path(raiz_resultados, experimento).glob("*/s*/dobra*/diagnostico_wavkan.json")):
        r = json.loads(arq.read_text(encoding="utf-8"))
        u = r["unidade"]
        for cab in r["cabecas"]:
            for momento in ("fim_fase1", "fim_fase2"):
                for camada, est in (cab.get(momento) or {}).items():
                    linhas.append({"config": u["config"], "semente": u["semente"], "dobra": u["dobra"],
                                   "momento": momento, "camada": camada, **est})
    return pd.DataFrame(linhas)
