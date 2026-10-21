# -*- coding: utf-8 -*-
"""Plugin do P2: multitarefa no protocolo único (uma ResNet-152, duas cabeças).

Carregue com ``pc.carregar_plugin(caminho)``. O SHA-256 deste arquivo vai para cada
``metricas.json``, separado do hash do protocolo, que não muda. Este arquivo não altera
nada do ``protocolo_comum.py``: só registra configs e oferece funções de checagem, de
execução e de medição.

Uma rede, duas saídas: backbone -> Dropout(0,5) -> {es: cabeça, dist: cabeça}.

* ``es``: entrada/saída (2 classes), em todas as imagens.
* ``dist``: distância do disparo (encostado, curta, distância), só nas entradas.

O que o plano chamava de "perda com máscara" já é o caminho nativo do protocolo, e por
isso a multitarefa (MT) e as redes de tarefa única (ST) do P1 passam pelo mesmo código:

* perda = ``soma_ponderada`` com λ = 1, ou seja, L = L_es + L_dist;
* CE ponderada por tarefa com ``ignore_index = -1``: as saídas (y_dist = -1) não entram
  em L_dist;
* lote sem nenhuma entrada: o protocolo usa L_dist = 0·logits (valor finito, gradiente
  zero na cabeça de distância), em vez da CE, que daria NaN;
* pesos de classe n/(K·n_c) calculados no treino da dobra; os da distância contam só as
  entradas e são idênticos aos da ST de distância do P1 (``checar_dados`` confere);
* Fase 1 treina as duas cabeças com o backbone congelado; Fase 2 treina tudo.

Dobras: coluna ``fold`` (todas as imagens). O teste da dobra k é a dobra k inteira, igual
ao da ST de entrada/saída e ao da ST de distância do P1 (``fold_dist_dk``). As métricas
de distância ignoram y_dist = -1.

| config | cabeças (es e dist) | parâmetros totais | par ST do P1 (soma) |
|--------|---------------------|-------------------|---------------------|
| mt_kan | KAN [2048,128,k], grade 10, ordem 3 (= (d)) | 66.017.728 | 124.161.536 (-46,8 %) |
| mt_mlp | MLP [2048,128,k] com ReLU (= (b))           | 58.668.997 | 116.812.805 (-49,8 %) |

As redes ST não são treinadas aqui: são as do P1 (``CONFIGS_ST``), com as mesmas dobras,
a mesma receita e as sementes 42, 43 e 44.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import protocolo_comum as pc

VERSAO = "1.0.0"

EXPERIMENTO = "multitarefa"
CABECA = {"kan": "kan", "mlp": "mlp128"}            # tipo -> cabeça nativa do protocolo
CONFIGS = {t: f"mt_{t}" for t in CABECA}            # tipo -> config MT
SEMENTES = (42, 43, 44)

# Redes de tarefa única (ST) do P1, que entram na comparação sem treinar de novo
EXPERIMENTOS_ST = {"es": "ablacao_cabeca", "dist": "mlsd_cabeca"}
CONFIGS_ST = {
    "kan": {"es": "resnet152__kan", "dist": "mlsd__resnet152__kan"},
    "mlp": {"es": "resnet152__mlp128", "dist": "mlsd__resnet152__mlp128"},
}

PARAMETROS_BACKBONE = 58_143_808     # ResNet-152 sem a fc (confere com o 01 do P1)
PARAMETROS_CABECA = {                # (tipo, tarefa) -> parâmetros da cabeça
    ("kan", "es"): 3_936_000,
    ("kan", "dist"): 3_937_920,
    ("mlp", "es"): 262_530,
    ("mlp", "dist"): 262_659,
}
PARAMETROS_TOTAIS = {
    **{CONFIGS[t]: PARAMETROS_BACKBONE + PARAMETROS_CABECA[(t, "es")] + PARAMETROS_CABECA[(t, "dist")] for t in CABECA},
    **{CONFIGS_ST[t][k]: PARAMETROS_BACKBONE + PARAMETROS_CABECA[(t, k)] for t in CABECA for k in ("es", "dist")},
}

# Base do P1 (folds_entrada_saida.csv)
DADOS_ESPERADOS = {"n_img": 2554, "n_entradas": 1883, "n_saidas": 671, "n_casos": 486}
ACC_MAJORITARIA_DIST = 1609 / 1883   # prever sempre "distância" (0,854)

for _t, _cab in CABECA.items():
    pc.registrar_config(
        CONFIGS[_t],
        {
            "backbone": "resnet152",
            "cabecas": {"es": _cab, "dist": _cab},
            "pesos_tarefa": {"es": 1.0, "dist": 1.0},
            "perda": "soma_ponderada",
            "descricao": f"P2 multitarefa: ResNet-152 + cabeças {_cab} para entrada/saída e distância (λ = 1)",
        },
    )


def tipo_da_config(cfg: str) -> str | None:
    """'mt_kan', 'resnet152__kan' ou 'mlsd__resnet152__kan' -> 'kan' (None se não for do P2)."""
    for t in CABECA:
        if cfg == CONFIGS[t] or cfg in CONFIGS_ST[t].values():
            return t
    return None


def config_st(tipo: str, tarefa: str) -> dict:
    """Config da rede ST do P1 como dicionário (para medir a inferência; não registra nada).

    Igual à registrada no P1: ``resnet152__{kan,mlp128}`` (nativa) e
    ``mlsd__resnet152__{kan,mlp128}`` (``configs_mlsd.py``).
    """
    return {"nome": CONFIGS_ST[tipo][tarefa], "backbone": "resnet152", "cabecas": {tarefa: CABECA[tipo]}}


# ---------------------------------------------------------------------------
# Dados e fila
# ---------------------------------------------------------------------------
def checar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Confere as dobras do P1 para a multitarefa. Uma linha por dobra; erro se algo divergir.

    Além das contagens da base, confere, dobra a dobra, que a ST de distância do P1
    (coluna ``fold_dist_dk`` de ``configs_mlsd.py``: entradas mantêm a dobra; saídas só
    na dobra k) tem as mesmas imagens de teste e os mesmos pesos de classe que a MT.
    """
    erros = []
    n = {
        "n_img": int(len(df)),
        "n_entradas": int((df["y_es"] == 0).sum()),
        "n_saidas": int((df["y_es"] == 1).sum()),
        "n_casos": int(df["case_id"].nunique()),
    }
    for k, v in DADOS_ESPERADOS.items():
        if n[k] != v:
            erros.append(f"{k} = {n[k]} (esperado {v})")
    entrada = df["y_es"].to_numpy() == 0
    y = df["y_dist"].to_numpy()
    if not np.array_equal(entrada, y >= 0):
        erros.append("y_dist >= 0 deveria valer exatamente nas entradas")
    erros += pc.checar_dobras(df)["avisos"]            # caso em 2 dobras; fração de saída fora de ±3 pp
    erros += pc.regra_composicao(df)[1]                # >= 2 casos de encostado e todas as classes por dobra
    f = df["fold"].to_numpy()
    linhas = []
    for k in range(pc.N_DOBRAS):
        tr, te = (f != k) & (f >= 0), f == k
        col_st = np.where(entrada | (f == k), f, -1)   # = configs_mlsd.adicionar_colunas_dist
        tr_st, te_st = (col_st != k) & (col_st >= 0), col_st == k
        if not np.array_equal(te, te_st):
            erros.append(f"dobra {k}: teste da MT diferente do da ST de distância")
        if not np.array_equal(tr & entrada, tr_st):
            erros.append(f"dobra {k}: entradas de treino da MT diferentes das da ST de distância")
        w_mt = pc.pesos_classe(y[tr], pc.TAREFAS["dist"].n_classes)
        w_st = pc.pesos_classe(y[tr_st], pc.TAREFAS["dist"].n_classes)
        if not np.array_equal(w_mt, w_st):
            erros.append(f"dobra {k}: pesos de classe da distância diferentes ({w_mt} x {w_st})")
        te_ent = te & entrada
        linhas.append({
            "dobra": k,
            "n_treino": int(tr.sum()),
            "entradas_treino": int((tr & entrada).sum()),
            "n_teste": int(te.sum()),
            "entradas_teste": int(te_ent.sum()),
            **{f"teste_{c}": int((y[te_ent] == i).sum()) for i, c in enumerate(pc.TAREFAS["dist"].nomes_classes)},
            "casos_encostado_teste": int(df.loc[te_ent & (y == 0), "case_id"].nunique()),
            "frac_saida_teste": round(float((~entrada[te]).mean()), 4),
            "pesos_es": np.round(pc.pesos_classe(df["y_es"].to_numpy()[tr], 2), 3).tolist(),
            "pesos_dist": np.round(w_mt, 3).tolist(),
        })
    if erros:
        raise RuntimeError("Dados da multitarefa inválidos: " + "; ".join(erros))
    return pd.DataFrame(linhas)


def montar_fila(sementes=SEMENTES, configs=None, dobras=range(pc.N_DOBRAS)) -> list:
    """Ordem semente -> config -> dobra: com o saldo no fim, as configs já têm a semente 42."""
    configs = list(configs) if configs else list(CONFIGS.values())
    desconhecidas = sorted(set(configs) - set(CONFIGS.values()))
    if desconhecidas:
        raise ValueError(f"Configs fora do P2: {desconhecidas}. Use {list(CONFIGS.values())}.")
    return [
        {"experimento": EXPERIMENTO, "config": c, "semente": int(s), "dobra": int(d), "coluna_dobra": "fold"}
        for s in sementes
        for c in configs
        for d in dobras
    ]


def dir_unidade(raiz_resultados, unidade) -> Path:
    return pc._caminhos_unidade(raiz_resultados, ".", unidade)[0]


# ---------------------------------------------------------------------------
# Checagem de uma unidade e execução da fila
# ---------------------------------------------------------------------------
def checar_unidade(r: dict, backbone_esperado=None) -> list:
    """Problemas de um metricas.json da MT (lista vazia = ok).

    ``backbone_esperado``: None usa ``PARAMETROS_BACKBONE``; False não confere o backbone.
    """
    problemas = []
    cfg = r["config"]["nome"]
    tipo = tipo_da_config(cfg)
    if cfg not in CONFIGS.values():
        return [f"{cfg} não é uma config multitarefa do P2"]
    if set(r["metricas"]) != {"es", "dist"}:
        problemas.append(f"{cfg}: métricas de {sorted(r['metricas'])} (esperado es e dist)")
    par = r["parametros"]
    for t in ("es", "dist"):
        if par["cabecas"].get(t) != PARAMETROS_CABECA[(tipo, t)]:
            problemas.append(f"{cfg}: cabeça {t} com {par['cabecas'].get(t)} parâmetros (esperado {PARAMETROS_CABECA[(tipo, t)]})")
    bb = PARAMETROS_BACKBONE if backbone_esperado is None else backbone_esperado
    if bb is not False and par["backbone"] != bb:
        problemas.append(f"{cfg}: backbone com {par['backbone']} parâmetros (esperado {bb})")
    if par["treinaveis_fase1"] != par["cabecas_total"]:
        problemas.append(f"{cfg}: a Fase 1 não treina exatamente as duas cabeças")
    for t, m in r["metricas"].items():
        if not m.get("prob_continua"):
            problemas.append(f"{cfg} {t}: probabilidades não contínuas")
        auc = m.get("auc", float("nan"))
        if not np.isfinite(auc):
            problemas.append(f"{cfg} {t}: AUC inválida ({m.get('auc_erro', 'NaN')})")
        elif abs(auc - m["rec_macro"]) < 1e-12:
            problemas.append(f"{cfg} {t}: AUC = revocação macro")
    if "es" in r["metricas"] and r["metricas"]["es"]["n"] != r["n_teste"]:
        problemas.append(f"{cfg}: a métrica de entrada/saída não usou todas as imagens de teste")
    if {"es", "dist"} <= set(r["metricas"]) and not 0 < r["metricas"]["dist"]["n"] < r["metricas"]["es"]["n"]:
        problemas.append(f"{cfg}: a métrica de distância deveria usar só as entradas do teste")
    sha = (r.get("plugins") or {}).get("multitarefa", {}).get("sha256")
    if sha is None:
        problemas.append(f"{cfg}: SHA do multitarefa.py ausente no metricas.json")
    return problemas


def executar_fila_mt(sessao, fila, raiz_resultados, raiz_checkpoints, previsao=None, **kw) -> list:
    """Como ``pc.executar_fila``, mais:

    * ``previsao(unidade) -> str`` (opcional) é impresso antes de cada unidade a rodar;
    * se a perda virar NaN/inf, grava ``falha_nan.json`` na pasta da unidade e PARA a fila;
    * cada unidade concluída passa por ``checar_unidade`` (problemas são impressos).
    """
    feitos = []
    for u in fila:
        d = dir_unidade(raiz_resultados, u)
        if (d / "metricas.json").exists():
            continue
        if previsao is not None:
            print(previsao(u))
        try:
            r = pc.executar_unidade(sessao, u, raiz_resultados, raiz_checkpoints, **kw)
        except FloatingPointError as e:
            pc.gravar_json_atomico(
                {"erro": str(e), "quando": pc.agora_iso(), "unidade": u, "plugin_versao": VERSAO},
                d / "falha_nan.json",
            )
            raise RuntimeError(
                f"Perda não finita em {u['config']} s{u['semente']} dobra {u['dobra']} ({e}). "
                f"Registrado em {d / 'falha_nan.json'}. Pare aqui e avise a sessão P2."
            ) from e
        if r is None:
            continue
        for p in checar_unidade(r):
            print("⚠️", p)
        feitos.append(r["unidade"])
    return feitos


# ---------------------------------------------------------------------------
# Benchmark de inferência: 1 forward da MT x 2 forwards do par ST
# ---------------------------------------------------------------------------
def _medir_chamada(fn, dispositivo, n_aquec: int, n_medidas: int) -> dict:
    """Tempo por chamada (ms; eventos CUDA na GPU, perf_counter na CPU) e memória de pico extra (MB)."""
    cuda = dispositivo.type == "cuda"
    with torch.no_grad():
        for _ in range(n_aquec):
            fn()
        mem = None
        if cuda:
            torch.cuda.synchronize(dispositivo)
            base = torch.cuda.memory_allocated(dispositivo)
            torch.cuda.reset_peak_memory_stats(dispositivo)
            fn()
            torch.cuda.synchronize(dispositivo)
            mem = (torch.cuda.max_memory_allocated(dispositivo) - base) / 1024**2
        tempos = []
        for _ in range(n_medidas):
            if cuda:
                ini, fim = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ini.record()
                fn()
                fim.record()
                torch.cuda.synchronize(dispositivo)
                tempos.append(ini.elapsed_time(fim))
            else:
                t0 = time.perf_counter()
                fn()
                tempos.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(tempos)
    return {
        "ms_media": float(t.mean()),
        "ms_dp": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        "ms_mediana": float(np.median(t)),
        "mem_pico_mb": mem,
    }


def benchmark_inferencia(precisao: str = "amp", dispositivo=None, lotes=(1, 32), n_aquec: int = 20,
                         n_medidas: int = 200, tipos=tuple(CABECA), semente: int = 0) -> dict:
    """Tempo de inferência para obter as DUAS saídas: MT (1 forward) x par ST (2 forwards).

    Mede, para cada tipo de cabeça e cada tamanho de lote: ``mt`` (a rede multitarefa),
    ``par_st`` (rede ST de entrada/saída + rede ST de distância, em sequência) e cada ST
    sozinha. Modelos com o backbone ImageNet e cabeças recém-inicializadas: o tempo não
    depende dos valores dos pesos. Mesma precisão e mesmo formato de memória do protocolo
    (AMP fp16 no backbone, cabeças em fp32, channels_last). Entrada aleatória 3×224×224
    (o pré-processamento é igual nos dois casos e fica de fora).
    """
    dispositivo = torch.device(dispositivo) if dispositivo else pc.dispositivo_padrao()
    cuda = dispositivo.type == "cuda"
    amp = precisao == "amp" and cuda
    linhas = []
    for tipo in tipos:
        pc.semear(semente)
        modelos = {
            "mt": pc.ModeloProtocolo(pc.obter_config(CONFIGS[tipo])),
            "st_es": pc.ModeloProtocolo(pc.obter_config(config_st(tipo, "es"))),
            "st_dist": pc.ModeloProtocolo(pc.obter_config(config_st(tipo, "dist"))),
        }
        for m in modelos.values():
            m.eval()
            if cuda:
                m.backbone.to(memory_format=torch.channels_last)
            m.to(dispositivo)
        if cuda:
            torch.backends.cudnn.benchmark = True
        npar = {k: int(sum(p.numel() for p in m.parameters())) for k, m in modelos.items()}
        npar["par_st"] = npar["st_es"] + npar["st_dist"]
        for lote in lotes:
            g = torch.Generator(device="cpu").manual_seed(semente)
            x = torch.randn(lote, 3, pc.IMG_SIZE, pc.IMG_SIZE, generator=g).to(dispositivo)
            if cuda:
                x = x.contiguous(memory_format=torch.channels_last)

            def chamada(*ms, x=x):
                def f():
                    with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=amp):
                        return [m(x) for m in ms]
                return f

            variantes = {
                "mt": chamada(modelos["mt"]),
                "par_st": chamada(modelos["st_es"], modelos["st_dist"]),
                "st_es": chamada(modelos["st_es"]),
                "st_dist": chamada(modelos["st_dist"]),
            }
            for nome, fn in variantes.items():
                r = _medir_chamada(fn, dispositivo, n_aquec, n_medidas)
                linhas.append({
                    "tipo": tipo, "variante": nome, "lote": int(lote),
                    "forwards": 2 if nome == "par_st" else 1,
                    "parametros": npar[nome], "pesos_fp32_mb": round(npar[nome] * 4 / 1024**2, 1),
                    **r, "ms_por_imagem": r["ms_media"] / lote,
                })
        del modelos
        if cuda:
            torch.cuda.empty_cache()
    return {
        "plugin_versao": VERSAO,
        "precisao": "amp_fp16" if amp else "fp32",
        "gpu": pc.info_gpu(),
        "torch": torch.__version__,
        "dispositivo": str(dispositivo),
        "lotes": [int(b) for b in lotes],
        "n_aquec": int(n_aquec),
        "n_medidas": int(n_medidas),
        "medido_em": pc.agora_iso(),
        "linhas": linhas,
    }


def tabela_benchmark(bench: dict) -> pd.DataFrame:
    """ms por imagem (média ± dp) da MT e do par ST, e quantas vezes o par é mais lento."""
    b = pd.DataFrame(bench["linhas"])
    linhas = []
    for (tipo, lote), g in b.groupby(["tipo", "lote"], sort=False):
        g = g.set_index("variante")
        mt, par = g.loc["mt"], g.loc["par_st"]
        linhas.append({
            "tipo": tipo, "lote": int(lote),
            "mt_ms_img": mt.ms_por_imagem, "mt_dp_ms_img": mt.ms_dp / lote,
            "par_st_ms_img": par.ms_por_imagem, "par_st_dp_ms_img": par.ms_dp / lote,
            "par_st_sobre_mt": par.ms_media / mt.ms_media,
            "mt_mais_rapida_alem_do_ruido": bool(mt.ms_media + mt.ms_dp < par.ms_media - par.ms_dp),
            "mt_mem_pico_mb": mt.mem_pico_mb, "par_st_mem_pico_mb": par.mem_pico_mb,
            "mt_parametros": int(mt.parametros), "par_st_parametros": int(par.parametros),
        })
    return pd.DataFrame(linhas)
