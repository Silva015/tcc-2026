# -*- coding: utf-8 -*-
"""Protocolo comum de treino e avaliação — TransferKAN / SIBGRAPI 2026.

Um único módulo define, para TODOS os modelos, as mesmas dobras, o mesmo
pré-processamento e augmentation, a mesma receita de 2 fases, a mesma regra de
checkpoint (última época) e as mesmas métricas (AUC sempre a partir de
probabilidades). Todo resultado registra o SHA-256 deste arquivo; por isso ele
só é editado pela sessão P1 e, depois de publicado, não muda.

Extensões (Wav-KAN, multitarefa) entram por arquivos separados ("plugins"), que
importam este módulo e chamam:

    registrar_cabeca(nome, fabrica)       # fabrica(in_features, n_saidas, **kw) -> nn.Module
    registrar_backbone(nome, fabrica)     # fabrica() -> (modulo, n_features, info: dict)
    registrar_perda(nome, fabrica)        # fabrica(tarefas: dict, cfg: dict) -> nn.Module
    registrar_config(nome, cfg)           # cfg: dict (ver `validar_config`)

O dataset devolve sempre (img, y_es, y_dist); y_dist = -1 nas saídas. O modelo é
backbone -> Dropout(0.5) -> {tarefa: cabeça} e devolve um dicionário de logits.
"""
from __future__ import annotations

import copy
import datetime as _dt
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROTOCOLO_VERSAO = "1.0.0"

# ---------------------------------------------------------------------------
# Constantes fixadas (pré-registradas)
# ---------------------------------------------------------------------------
DATASET_URL = "https://github.com/pedrogarciafreitas/FDCPUnBGunshotDB.git"
DATASET_COMMIT = "f6793899b081a5d6de34100498d9345a7db4c951"
LEGUWOI_URL = "https://gitlab.com/lisa-unb/leguwoi.git"
LEGUWOI_COMMIT = "83d82f1f696f6a8d7e2969388ed026274f873ffa"
EFFICIENT_KAN_COMMIT = "7b6ce1c87f18c8bc90c208f6b494042344216b11"
ALBUMENTATIONS_VERSAO = "2.0.8"
DRIVE_PASTA_ID = "1i_o4gxs-nyzZOPFp21gOwI_AMHPzct-4"
ARQUIVO_SENTINELA = ".tcc_final_id"

PIP_DEPENDENCIAS = [
    f"albumentations=={ALBUMENTATIONS_VERSAO}",
    f"git+https://github.com/Blealtan/efficient-kan@{EFFICIENT_KAN_COMMIT}",
]

IMG_SIZE = 224
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DP_IMAGENET = (0.229, 0.224, 0.225)

N_DOBRAS = 5
SEMENTE_DOBRAS = 42
ARQUIVO_DOBRAS = "folds_entrada_saida.csv"
ARQUIVO_META_DOBRAS = "dobras_meta.json"

# Receita de 2 fases (idêntica à do TransferKAN_EntradaSaida.ipynb)
RECEITA = {
    "epocas_f1": 15,
    "epocas_f2": 45,
    "lr_f1": 1e-3,
    "wd_f1": 1e-2,
    "lr_f2": 1e-5,
    "wd_f2": 1e-3,
    "lote": 32,
    "dropout": 0.5,
    "drop_last": True,
}

# strat4: 0 saída, 1 encostado, 2 curta, 3 distância
SUBPASTA_DIST = {"ENCOSTADO": 0, "CURTA": 1, "DISTANCIA": 2}

# Impressão digital do KFold antigo (TransferKAN_EntradaSaida.ipynb): tamanhos
# do teste das dobras 1..5, deduzidos das acurácias impressas no log
# (ex.: 396/496 = 79,84 %). Soma = 2.554.
IMPRESSAO_DIGITAL_ANTIGA = (496, 636, 459, 465, 498)
# Acurácia de teste na época 60, dobras 1..5 do log antigo (%)
ACCS_ANTIGAS_ULTIMA_EPOCA = (79.84, 84.12, 88.89, 87.31, 86.75)

# Critério da ponte fp32 x AMP (definido pelo autor antes de rodar)
PONTE = {"max_delta_acc_pp": 2.0, "max_delta_auc": 0.015, "margem_pp": 3.0}
# Faixa de acurácia aceitável:
#  - dobra 1 antiga reconstruída (impressão digital confere): 79,84 ± 3 pp;
#  - senão, dobra 0 nova: faixa das 5 dobras antigas ± 3 pp (76,84–91,89 %).
FAIXA_PONTE_ANTIGA = (ACCS_ANTIGAS_ULTIMA_EPOCA[0] - 3.0, ACCS_ANTIGAS_ULTIMA_EPOCA[0] + 3.0)
FAIXA_PONTE_NOVA = (min(ACCS_ANTIGAS_ULTIMA_EPOCA) - 3.0, max(ACCS_ANTIGAS_ULTIMA_EPOCA) + 3.0)

# ---------------------------------------------------------------------------
# Utilidades gerais
# ---------------------------------------------------------------------------
ID_SESSAO = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def definir_id_sessao(rotulo: str) -> str:
    """Fixa o dono das travas (ex.: "A" / "B", um por aba do Colab).

    Com um rótulo estável, a aba que caiu e foi reaberta retoma as próprias
    travas na hora, sem esperar a expiração.
    """
    global ID_SESSAO
    ID_SESSAO = f"aba-{rotulo}"
    return ID_SESSAO


def agora_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_arquivo(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def sha256_protocolo() -> str:
    return sha256_arquivo(__file__)


def gravar_json_atomico(obj, caminho) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + f".tmp-{uuid.uuid4().hex[:6]}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_padrao)
    os.replace(tmp, caminho)


def _json_padrao(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def semear(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente % (2**32))
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def dispositivo_padrao() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Ambiente, GPU e repositórios fixados
# ---------------------------------------------------------------------------
def _versao_pacote(nome: str):
    try:
        from importlib.metadata import version

        return version(nome)
    except Exception:
        return None


def _commit_efficient_kan():
    try:
        from importlib.metadata import distribution

        d = distribution("efficient-kan")
        txt = d.read_text("direct_url.json")
        if txt:
            return json.loads(txt).get("vcs_info", {}).get("commit_id")
    except Exception:
        return None
    return None


def info_gpu() -> dict:
    if not torch.cuda.is_available():
        return {"disponivel": False, "nome": None}
    p = torch.cuda.get_device_properties(0)
    return {
        "disponivel": True,
        "nome": p.name,
        "memoria_gb": round(p.total_memory / 1024**3, 2),
        "capacidade": f"{p.major}.{p.minor}",
        "cuda": torch.version.cuda,
    }


def _flags_tf32() -> dict:
    flags = {}
    try:
        flags["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        flags["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    for nome in ("matmul", "cudnn"):
        try:
            mod = torch.backends.cuda.matmul if nome == "matmul" else torch.backends.cudnn
            flags[f"{nome}_fp32_precision"] = str(getattr(mod, "fp32_precision"))
        except Exception:
            pass
    flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    return flags


def info_ambiente() -> dict:
    try:
        import albumentations as A

        alb = A.__version__
    except Exception:
        alb = None
    try:
        import cv2

        cv = cv2.__version__
    except Exception:
        cv = None
    try:
        import sklearn

        skl = sklearn.__version__
    except Exception:
        skl = None
    try:
        import torchvision

        tv = torchvision.__version__
    except Exception:
        tv = None
    return {
        "protocolo_versao": PROTOCOLO_VERSAO,
        "protocolo_sha256": sha256_protocolo(),
        "python": platform.python_version(),
        "plataforma": _plataforma(),
        "host": socket.gethostname(),
        "id_sessao": ID_SESSAO,
        "torch": torch.__version__,
        "torchvision": tv,
        "albumentations": alb,
        "opencv": cv,
        "sklearn": skl,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "efficient_kan_commit": _commit_efficient_kan(),
        "gpu": info_gpu(),
        "tf32": _flags_tf32(),
        "num_cpus": os.cpu_count(),
    }


def _plataforma() -> str:
    if "google.colab" in sys.modules or os.path.exists("/content") and os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return platform.system().lower()


def checar_dependencias() -> dict:
    """Confere as versões fixadas. Levanta erro se algo divergir."""
    import albumentations as A

    erros = []
    if A.__version__ != ALBUMENTATIONS_VERSAO:
        erros.append(f"albumentations {A.__version__} != {ALBUMENTATIONS_VERSAO}")
    commit = _commit_efficient_kan()
    if commit is not None and commit != EFFICIENT_KAN_COMMIT:
        erros.append(f"efficient-kan @{commit} != @{EFFICIENT_KAN_COMMIT}")
    try:
        import efficient_kan  # noqa: F401
    except Exception as e:  # pragma: no cover
        erros.append(f"efficient_kan não importa: {e}")
    if erros:
        raise RuntimeError("Dependências fora do fixado: " + "; ".join(erros))
    return {"albumentations": A.__version__, "efficient_kan_commit": commit}


class ErroGPU(RuntimeError):
    pass


def exigir_gpu(permitidas=("L4",), permitir_reserva: bool = False) -> dict:
    """Rodadas comparáveis só na L4. Outra GPU exige permitir_reserva=True."""
    g = info_gpu()
    if not g["disponivel"]:
        raise ErroGPU("Sem GPU CUDA. Troque o tipo de ambiente de execução para GPU (L4).")
    ok = any(p.lower() in g["nome"].lower() for p in permitidas)
    g["comparavel"] = bool(ok)
    if not ok and not permitir_reserva:
        raise ErroGPU(
            f"GPU {g['nome']!r} não é {permitidas}. As rodadas comparáveis são só na L4. "
            "Para usar como reserva (fica registrada e marcada com † na tabela), "
            "defina PERMITIR_RESERVA = True."
        )
    return g


def clonar_fixado(url: str, commit: str, destino) -> str:
    """git clone + checkout no commit fixado; confere o commit resultante."""
    destino = Path(destino)
    if not (destino / ".git").exists():
        if destino.exists():
            shutil.rmtree(destino)
        r = subprocess.run(["git", "clone", "-q", url, str(destino)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Falha no git clone de {url}: {r.stderr.strip()}")
    r = subprocess.run(["git", "-C", str(destino), "checkout", "-q", commit], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["git", "-C", str(destino), "fetch", "-q", "origin"], capture_output=True)
        r = subprocess.run(["git", "-C", str(destino), "checkout", "-q", commit], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Commit {commit} não encontrado em {url}: {r.stderr.strip()}")
    head = subprocess.run(
        ["git", "-C", str(destino), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if head != commit:
        raise RuntimeError(f"{destino}: HEAD {head} != commit fixado {commit}")
    return head


def checar_sentinela_drive(dir_drive) -> None:
    """Confere que o Drive montado é o da conta dona da pasta tcc-final."""
    p = Path(dir_drive) / ARQUIVO_SENTINELA
    if not p.exists():
        raise RuntimeError(
            f"Não achei {p}. O Colab parece logado em OUTRA conta Google. Use a conta "
            "padrão (authuser=0), dona da pasta tcc-final, e monte o Drive de novo. "
            "Nada foi gravado."
        )
    conteudo = p.read_text(encoding="utf-8").strip()
    if DRIVE_PASTA_ID not in conteudo:
        raise RuntimeError(
            f"O sentinela {p} não contém o id {DRIVE_PASTA_ID}. Pasta errada ou outra conta. "
            "Nada foi gravado."
        )


# ---------------------------------------------------------------------------
# Dados: listagem, anomalias, dobras
# ---------------------------------------------------------------------------
def listar_imagens(raiz_dataset) -> pd.DataFrame:
    """Lista as imagens como nos notebooks originais (rótulos pelas pastas).

    ENTRADAS_EQX/**/*.JPG -> y_es = 0, y_dist pela subpasta; SAIDAS_EQX/*.JPG ->
    y_es = 1, y_dist = -1. A lista sai ORDENADA por rel_path.
    """
    raiz = Path(raiz_dataset)
    linhas = []
    for p in glob.glob(str(raiz / "database" / "ENTRADAS_EQX" / "**" / "*.JPG"), recursive=True):
        rel = Path(p).relative_to(raiz).as_posix()
        sub = Path(p).parent.name
        if sub not in SUBPASTA_DIST:
            raise ValueError(f"Imagem de entrada fora das subpastas de distância: {rel}")
        yd = SUBPASTA_DIST[sub]
        linhas.append((rel, Path(p).name[:9], 0, yd, yd + 1))
    for p in glob.glob(str(raiz / "database" / "SAIDAS_EQX" / "*.JPG")):
        rel = Path(p).relative_to(raiz).as_posix()
        linhas.append((rel, Path(p).name[:9], 1, -1, 0))
    df = pd.DataFrame(linhas, columns=["rel_path", "case_id", "y_es", "y_dist", "strat4"])
    df = df.sort_values("rel_path", kind="mergesort").reset_index(drop=True)
    if df["rel_path"].duplicated().any():
        raise ValueError("rel_path duplicado na listagem")
    return df


def anomalias_csv(raiz_dataset, df: pd.DataFrame) -> dict:
    """Cruza a listagem das pastas com csv_configs/gunshots_eqx.csv do dataset.

    Devolve {rel_path: texto} com as divergências encontradas (imagem fora do
    CSV, linha do CSV sem imagem, rótulo divergente, caso divergente).
    """
    raiz = Path(raiz_dataset)
    csvp = raiz / "csv_configs" / "gunshots_eqx.csv"
    obs: dict[str, list[str]] = {}
    if not csvp.exists():
        return {"__geral__": f"CSV {csvp.name} ausente"}
    c = pd.read_csv(csvp, dtype=str)
    if "path" not in c.columns:
        return {"__geral__": "CSV sem coluna path"}
    nome_csv = c["path"].map(lambda s: Path(str(s).replace("\\", "/")).name)
    c = c.assign(_nome=nome_csv)
    por_nome = {n: g for n, g in c.groupby("_nome")}
    nomes_df = set(df["rel_path"].map(lambda s: Path(s).name))
    for _, r in df.iterrows():
        nome = Path(r.rel_path).name
        if nome not in por_nome:
            obs.setdefault(r.rel_path, []).append("imagem ausente do CSV do dataset")
            continue
        g = por_nome[nome]
        if len(g) > 1:
            obs.setdefault(r.rel_path, []).append(f"{len(g)} linhas no CSV")
        row = g.iloc[0]
        st = str(row.get("shot_type", ""))
        st_es = {"TE": 0, "TS": 1}.get(st)
        if st_es is not None and st_es != r.y_es:
            obs.setdefault(r.rel_path, []).append(f"CSV shot_type={st} diverge da pasta")
        dist = str(row.get("distance", ""))
        dist_map = {"YE": 0, "YC": 1, "YD": 2}
        if r.y_es == 0:
            yd_csv = dist_map.get(dist)
            if yd_csv is None:
                obs.setdefault(r.rel_path, []).append(f"CSV distance={dist} (sem distância válida)")
            elif yd_csv != r.y_dist:
                obs.setdefault(r.rel_path, []).append(f"CSV distance={dist} diverge da subpasta")
        elif dist != "YI":
            obs.setdefault(r.rel_path, []).append(f"CSV distance={dist} numa saída (esperado YI/FALSE)")
        caso_csv = f"{row.get('case_year', '')}.{str(row.get('case_number', '')).zfill(4)}"
        if caso_csv != r.case_id:
            obs.setdefault(r.rel_path, []).append(f"CSV caso={caso_csv} diverge de {r.case_id}")
    for nome in sorted(set(por_nome) - nomes_df):
        obs.setdefault(f"__csv__/{nome}", []).append("linha do CSV sem imagem nas pastas")
    return {k: "; ".join(v) for k, v in obs.items()}


def regra_composicao(df: pd.DataFrame, coluna_dobra: str = "fold") -> tuple[bool, list[str]]:
    """Cada dobra de teste: >=2 casos de encostado e >=1 imagem de cada strat4."""
    falhas = []
    for k in range(N_DOBRAS):
        t = df[df[coluna_dobra] == k]
        n_casos_enc = t.loc[t["strat4"] == 1, "case_id"].nunique()
        if n_casos_enc < 2:
            falhas.append(f"dobra {k}: {n_casos_enc} caso(s) de encostado")
        for cl in range(4):
            if (t["strat4"] == cl).sum() < 1:
                falhas.append(f"dobra {k}: nenhuma imagem strat4={cl}")
    return (len(falhas) == 0, falhas)


def gerar_dobras(df: pd.DataFrame, semente_inicial: int = SEMENTE_DOBRAS, max_tentativas: int = 1000):
    """StratifiedGroupKFold(5, shuffle=True) com y=strat4 e groups=case_id.

    Regra pré-registrada: usa a semente 42; se a composição falhar, a 1ª
    semente >= 43 que passe (decidido só pela composição).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.sort_values("rel_path", kind="mergesort").reset_index(drop=True)
    tentativas = []
    for s in range(semente_inicial, semente_inicial + max_tentativas):
        sgkf = StratifiedGroupKFold(n_splits=N_DOBRAS, shuffle=True, random_state=s)
        fold = np.full(len(df), -1, dtype=int)
        for k, (_, te) in enumerate(sgkf.split(df["rel_path"], df["strat4"], df["case_id"])):
            fold[te] = k
        d = df.assign(fold=fold)
        ok, falhas = regra_composicao(d)
        tentativas.append({"semente": s, "ok": ok, "falhas": falhas})
        if ok:
            return d, {"semente_usada": s, "tentativas": tentativas}
    raise RuntimeError("Nenhuma semente passou na regra de composição")


def checar_dobras(df: pd.DataFrame, coluna_dobra: str = "fold") -> dict:
    """Checagens do notebook 00 (não alteram as dobras)."""
    avisos = []
    caso_dobras = df.groupby("case_id")[coluna_dobra].nunique()
    vazamento = int((caso_dobras > 1).sum())
    if vazamento:
        avisos.append(f"{vazamento} caso(s) em mais de uma dobra")
    frac_global = float((df["y_es"] == 1).mean())
    por_dobra = []
    for k in range(N_DOBRAS):
        t = df[df[coluna_dobra] == k]
        frac = float((t["y_es"] == 1).mean())
        item = {
            "dobra": k,
            "n_img": int(len(t)),
            "n_casos": int(t["case_id"].nunique()),
            "frac_saida": round(frac, 4),
            "dif_pp": round(100 * (frac - frac_global), 2),
            "n_saida": int((t["y_es"] == 1).sum()),
            "n_encostado": int((t["y_dist"] == 0).sum()),
            "n_curta": int((t["y_dist"] == 1).sum()),
            "n_distancia": int((t["y_dist"] == 2).sum()),
            "casos_encostado": int(t.loc[t["y_dist"] == 0, "case_id"].nunique()),
        }
        if abs(item["dif_pp"]) > 3.0:
            avisos.append(f"dobra {k}: fração de saída {frac:.3f} difere {item['dif_pp']} pp da global")
        por_dobra.append(item)
    return {
        "n_img": int(len(df)),
        "n_casos": int(df["case_id"].nunique()),
        "frac_saida_global": round(frac_global, 4),
        "casos_em_mais_de_uma_dobra": vazamento,
        "contagem_y_dist": {int(k): int(v) for k, v in df["y_dist"].value_counts().sort_index().items()},
        "por_dobra": por_dobra,
        "avisos": avisos,
    }


def colisoes_chave_leguwoi(df: pd.DataFrame) -> dict:
    """Quantos casos ano.caso colapsariam usando só o número (chave do leguwoi)."""
    so_numero = df["case_id"].str.slice(5, 9)
    g = df.assign(_num=so_numero).groupby("_num")["case_id"].nunique()
    return {
        "grupos_ano_caso": int(df["case_id"].nunique()),
        "grupos_so_numero": int(so_numero.nunique()),
        "numeros_com_mais_de_um_ano": int((g > 1).sum()),
    }


def dobras_antigas(raiz_dataset) -> pd.DataFrame:
    """Reconstrói o KFold do TransferKAN_EntradaSaida.ipynb original.

    Ordem do glob SEM ordenar (como no notebook), case_id = filename[:9],
    KFold(5, shuffle=True, random_state=42) sobre a lista de casos únicos na
    ordem de primeira aparição. fold_antiga 0 = "FOLD 1" do log antigo.
    """
    from sklearn.model_selection import KFold

    raiz = Path(raiz_dataset)
    caminhos = list(glob.glob(os.path.join(str(raiz), "database", "ENTRADAS_EQX", "**", "*.JPG"), recursive=True))
    caminhos += list(glob.glob(os.path.join(str(raiz), "database", "SAIDAS_EQX", "*.JPG")))
    case_to_idx: dict[str, list[int]] = {}
    for i, p in enumerate(caminhos):
        case_to_idx.setdefault(os.path.basename(p)[:9], []).append(i)
    casos = list(case_to_idx.keys())
    fold = np.full(len(caminhos), -1, dtype=int)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k, (_, te) in enumerate(kf.split(casos)):
        for ci in te:
            fold[case_to_idx[casos[ci]]] = k
    rel = [Path(p).relative_to(raiz).as_posix() for p in caminhos]
    return pd.DataFrame({"rel_path": rel, "fold_antiga": fold})


def conferir_impressao_digital(df_antigo: pd.DataFrame) -> dict:
    tam = tuple(int((df_antigo["fold_antiga"] == k).sum()) for k in range(5))
    return {
        "tamanhos_teste": tam,
        "esperado": IMPRESSAO_DIGITAL_ANTIGA,
        "confere": tam == IMPRESSAO_DIGITAL_ANTIGA,
        "dobra1_confere": tam[0] == IMPRESSAO_DIGITAL_ANTIGA[0],
    }


def ler_dobras(caminho_csv) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, dtype={"case_id": str, "obs": str}, keep_default_na=False)
    for c in ("y_es", "y_dist", "strat4", "fold"):
        df[c] = df[c].astype(int)
    df = df.sort_values("rel_path", kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cache 224x224 (gerado na sessão; nunca gravado no Drive)
# ---------------------------------------------------------------------------
def _resize_protocolo():
    import albumentations as A

    return A.Resize(IMG_SIZE, IMG_SIZE)


def carregar_redimensionada(caminho, resize=None) -> np.ndarray:
    """Igual à 1ª operação do pipeline original: PIL RGB -> np -> A.Resize(224,224)."""
    from PIL import Image

    if resize is None:
        resize = _resize_protocolo()
    with Image.open(caminho) as im:
        arr = np.array(im.convert("RGB"))
    return resize(image=arr)["image"]


def construir_cache(raiz_dataset, rel_paths, n_threads: int | None = None) -> np.ndarray:
    raiz = Path(raiz_dataset)
    n = len(rel_paths)
    cache = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    resize = _resize_protocolo()

    def _um(i):
        cache[i] = carregar_redimensionada(raiz / rel_paths[i], resize)

    n_threads = n_threads or min(16, (os.cpu_count() or 2) * 2)
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(_um, range(n)))
    return cache


def conferir_cache(cache, raiz_dataset, rel_paths, n: int = 5, semente: int = 0) -> bool:
    rng = np.random.default_rng(semente)
    idx = rng.choice(len(rel_paths), size=min(n, len(rel_paths)), replace=False)
    for i in idx:
        ref = carregar_redimensionada(Path(raiz_dataset) / rel_paths[i])
        if not np.array_equal(ref, cache[i]):
            return False
    return True


# ---------------------------------------------------------------------------
# Augmentation (como de fato rodou no TransferKAN_EntradaSaida.ipynb)
# ---------------------------------------------------------------------------
AVISO_SSR = "ShiftScaleRotate is a special case of Affine transform"


def transform_treino():
    """Pipeline do notebook original com os valores EFETIVOS da execução.

    Na execução original (albumentations 2.x) `alpha_affine` e `var_limit`
    foram descartados; aqui ficam explícitos os valores que de fato rodaram:
    ElasticTransform(alpha=1, sigma=50) e GaussNoise(std_range=(0.2, 0.44)).
    ShiftScaleRotate é mantido (só emite aviso de depreciação, que é filtrado).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=AVISO_SSR)
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GaussNoise(std_range=(0.2, 0.44), p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.Normalize(mean=MEDIA_IMAGENET, std=DP_IMAGENET),
                ToTensorV2(),
            ]
        )


def transform_teste():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEDIA_IMAGENET, std=DP_IMAGENET),
            ToTensorV2(),
        ]
    )


def checar_avisos_transforms(cache_amostra: np.ndarray) -> list[str]:
    """Constrói e aplica os transforms; devolve os avisos inesperados."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tt, te = transform_treino(), transform_teste()
        tt.set_random_seed(0)
        for img in cache_amostra:
            for _ in range(4):
                tt(image=img)
            te(image=img)
    return [str(x.message) for x in w if AVISO_SSR not in str(x.message)]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetGuWID(torch.utils.data.Dataset):
    """Devolve (img, y_es, y_dist) a partir do cache uint8 224x224."""

    def __init__(self, cache: np.ndarray, df: pd.DataFrame, indices, transform):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y_es = df["y_es"].to_numpy()[self.indices]
        self.y_dist = df["y_dist"].to_numpy()[self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.cache[self.indices[i]]
        img = self.transform(image=img)["image"]
        return img, int(self.y_es[i]), int(self.y_dist[i])


def _iniciar_worker(_worker_id):
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)
    info = torch.utils.data.get_worker_info()
    tr = getattr(info.dataset, "transform", None)
    if hasattr(tr, "set_random_seed"):
        tr.set_random_seed(int(s))


def semente_epoca(semente: int, dobra: int, epoca: int) -> int:
    return int(semente) * 1_000_000 + int(dobra) * 1_000 + int(epoca)


def num_workers_padrao() -> int:
    return max(0, min(8, (os.cpu_count() or 1)))


def loader_treino(ds, semente: int, dobra: int, epoca: int, lote: int, num_workers: int, drop_last: bool = True):
    g = torch.Generator()
    g.manual_seed(semente_epoca(semente, dobra, epoca))
    if num_workers == 0 and hasattr(ds.transform, "set_random_seed"):
        ds.transform.set_random_seed(semente_epoca(semente, dobra, epoca))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=lote,
        shuffle=True,
        drop_last=drop_last,
        generator=g,
        num_workers=num_workers,
        worker_init_fn=_iniciar_worker if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def loader_teste(ds, lote: int, num_workers: int):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=lote,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Tarefas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tarefa:
    nome: str
    coluna: str
    n_classes: int
    nomes_classes: tuple


TAREFAS = {
    "es": Tarefa("es", "y_es", 2, ("entrada", "saida")),
    "dist": Tarefa("dist", "y_dist", 3, ("encostado", "curta", "distancia")),
}
IGNORAR = -1


def pesos_classe(y: np.ndarray, n_classes: int) -> np.ndarray:
    """n / (K * n_c) sobre os rótulos válidos (= n/(2 n_c) no binário)."""
    y = y[y != IGNORAR]
    n = len(y)
    cont = np.bincount(y, minlength=n_classes).astype(float)
    if (cont == 0).any():
        raise ValueError(f"Classe sem exemplos no treino: {cont}")
    return n / (n_classes * cont)


# ---------------------------------------------------------------------------
# Registros (pontos de extensão)
# ---------------------------------------------------------------------------
_CABECAS: dict = {}
_BACKBONES: dict = {}
_PERDAS: dict = {}
_CONFIGS: dict = {}
PLUGINS_CARREGADOS: dict = {}


def _registrar(reg: dict, tipo: str, nome: str, obj, sobrescrever: bool):
    if nome in reg and not sobrescrever:
        raise KeyError(f"{tipo} {nome!r} já registrada(o); use sobrescrever=True")
    reg[nome] = obj
    return obj


def registrar_cabeca(nome: str, fabrica, sobrescrever: bool = False):
    """fabrica(in_features: int, n_saidas: int, **kw) -> nn.Module (logits)."""
    return _registrar(_CABECAS, "cabeça", nome, fabrica, sobrescrever)


def registrar_backbone(nome: str, fabrica, sobrescrever: bool = False):
    """fabrica() -> (modulo, n_features, info: dict). O módulo devolve as features."""
    return _registrar(_BACKBONES, "backbone", nome, fabrica, sobrescrever)


def registrar_perda(nome: str, fabrica, sobrescrever: bool = False):
    """fabrica(tarefas: dict[str, Tarefa], cfg: dict) -> nn.Module.

    forward(perdas: dict[str, Tensor], epoca: int, fase: int) -> Tensor escalar.
    Parâmetros do módulo (ex.: pesos aprendíveis) entram como parâmetros novos.
    """
    return _registrar(_PERDAS, "perda", nome, fabrica, sobrescrever)


def registrar_config(nome: str, cfg: dict, sobrescrever: bool = False):
    cfg = validar_config(dict(cfg, nome=nome))
    return _registrar(_CONFIGS, "config", nome, cfg, sobrescrever)


def cabecas_registradas():
    return sorted(_CABECAS)


def backbones_registrados():
    return sorted(_BACKBONES)


def configs_registradas():
    return sorted(_CONFIGS)


def obter_config(nome_ou_cfg) -> dict:
    if isinstance(nome_ou_cfg, dict):
        return validar_config(dict(nome_ou_cfg))
    if nome_ou_cfg not in _CONFIGS:
        raise KeyError(f"Config {nome_ou_cfg!r} não registrada. Disponíveis: {configs_registradas()}")
    return copy.deepcopy(_CONFIGS[nome_ou_cfg])


def validar_config(cfg: dict) -> dict:
    """Campos: nome, backbone, cabecas {tarefa: {"nome": cabeça, "kw": {...}}},
    pesos_tarefa {tarefa: float} (padrão 1.0), perda (padrão "soma_ponderada"),
    descricao (opcional). A receita de treino NÃO é configurável."""
    for k in ("nome", "backbone", "cabecas"):
        if k not in cfg:
            raise ValueError(f"config sem {k!r}: {cfg}")
    if "receita" in cfg:
        raise ValueError("A receita de treino é fixa no protocolo e não pode ser sobrescrita")
    cab = {}
    for t, c in cfg["cabecas"].items():
        if t not in TAREFAS:
            raise ValueError(f"tarefa desconhecida {t!r}")
        if isinstance(c, str):
            c = {"nome": c, "kw": {}}
        c = {"nome": c["nome"], "kw": dict(c.get("kw", {}))}
        cab[t] = c
    cfg["cabecas"] = cab
    cfg.setdefault("pesos_tarefa", {t: 1.0 for t in cab})
    cfg.setdefault("perda", "soma_ponderada")
    cfg.setdefault("descricao", "")
    return cfg


def carregar_plugin(caminho) -> str:
    """Importa um plugin por caminho (ele chama os registrar_*). Registra o SHA-256."""
    caminho = Path(caminho)
    nome = caminho.stem
    sha = sha256_arquivo(caminho)
    if nome in PLUGINS_CARREGADOS:
        if PLUGINS_CARREGADOS[nome]["sha256"] != sha:
            raise RuntimeError(f"Plugin {nome} já carregado com outro conteúdo; reinicie o ambiente")
        return sha
    if "protocolo_comum" not in sys.modules:
        sys.modules["protocolo_comum"] = sys.modules[__name__]
    spec = importlib.util.spec_from_file_location(nome, str(caminho))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    PLUGINS_CARREGADOS[nome] = {"caminho": str(caminho), "sha256": sha}
    return sha


# ---------------------------------------------------------------------------
# Cabeças nativas
# ---------------------------------------------------------------------------
def _kan(dims, grid_size=10, spline_order=3):
    from efficient_kan import KAN

    return KAN(list(dims), grid_size=grid_size, spline_order=spline_order)


def parametros_kan(in_f: int, oculta: int, n: int, grid_size: int = 10, spline_order: int = 3) -> int:
    """efficient-kan: por aresta = base (1) + spline (grid+ordem) + escala (1); sem bias."""
    arestas = in_f * oculta + oculta * n
    return arestas * (1 + grid_size + spline_order + 1)


def oculta_mlp_iso(in_f: int, n: int, alvo: int) -> int:
    """Largura H de Linear(in,H)+Linear(H,n) cujo nº de parâmetros é o mais próximo de alvo."""
    return int(round((alvo - n) / (in_f + 1 + n)))


def _cab_linear(in_f, n, **kw):
    return nn.Linear(in_f, n)


def _cab_mlp(in_f, n, oculta=128, **kw):
    return nn.Sequential(nn.Linear(in_f, oculta), nn.ReLU(inplace=True), nn.Linear(oculta, n))


def _cab_mlp_iso(in_f, n, alvo=None, **kw):
    if alvo is None:
        alvo = parametros_kan(in_f, 128, n, 10, 3)
    return _cab_mlp(in_f, n, oculta=oculta_mlp_iso(in_f, n, alvo))


def _cab_kan(in_f, n, oculta=128, grid_size=10, spline_order=3, **kw):
    return _kan([in_f, oculta, n], grid_size=grid_size, spline_order=spline_order)


def _cab_kan_pequena(in_f, n, **kw):
    return _kan([in_f, 8, n], grid_size=10, spline_order=3)


class _CabecaSqueezeNet(nn.Module):
    """Conv2d(512, n, 1) -> ReLU -> AdaptiveAvgPool -> flatten (cabeça original da SqueezeNet)."""

    def __init__(self, in_f, n):
        super().__init__()
        self.conv = nn.Conv2d(in_f, n, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return torch.flatten(self.pool(self.relu(self.conv(x))), 1)


registrar_cabeca("linear", _cab_linear)
registrar_cabeca("mlp128", lambda in_f, n, **kw: _cab_mlp(in_f, n, oculta=128))
registrar_cabeca("mlp_iso", _cab_mlp_iso)
registrar_cabeca("kan", _cab_kan)
registrar_cabeca("kan_pequena", _cab_kan_pequena)
registrar_cabeca("conv1x1_squeezenet", lambda in_f, n, **kw: _CabecaSqueezeNet(in_f, n))


# ---------------------------------------------------------------------------
# Backbones nativos (catálogo de leguwoi/src/models/torchvision_models.py)
# ---------------------------------------------------------------------------
# nome -> (função torchvision, caminho da camada final, caminho do Dropout que a
#          precede e é substituído (ou None))
CATALOGO_TORCHVISION = {
    "resnet152": ("resnet152", ("fc",), None),
    "alexnet": ("alexnet", ("classifier", 6), None),
    "densenet169": ("densenet169", ("classifier",), None),
    "efficientnet_v2_l": ("efficientnet_v2_l", ("classifier", 1), ("classifier", 0)),
    "mnasnet1_3": ("mnasnet1_3", ("classifier", 1), ("classifier", 0)),
    "mobilenet_v2": ("mobilenet_v2", ("classifier", 1), ("classifier", 0)),
    "regnet_y_16gf": ("regnet_y_16gf", ("fc",), None),
    "shufflenet_v2_x0_5": ("shufflenet_v2_x0_5", ("fc",), None),
    "squeezenet1_0": ("squeezenet1_0", None, None),  # tratado à parte
    "swin_b": ("swin_b", ("head",), None),
    "vgg16": ("vgg16", ("classifier", 6), ("classifier", 5)),
    "maxvit_t": ("maxvit_t", ("classifier", 5), None),
}


def _fmt_caminho(caminho) -> str:
    return "".join(f"[{c}]" if isinstance(c, int) else (c if i == 0 else f".{c}") for i, c in enumerate(caminho))


def _obter_sub(mod, caminho):
    for c in caminho:
        mod = mod[c] if isinstance(c, int) else getattr(mod, c)
    return mod


def _trocar_sub(mod, caminho, novo):
    pai = _obter_sub(mod, caminho[:-1]) if len(caminho) > 1 else mod
    ult = caminho[-1]
    if isinstance(ult, int):
        pai[ult] = novo
    else:
        setattr(pai, ult, novo)


def _fabrica_torchvision(nome_cat: str, pretreinado: bool = True):
    def fabrica():
        import torchvision

        fn, cam_final, cam_drop = CATALOGO_TORCHVISION[nome_cat]
        pesos = "DEFAULT" if pretreinado else None
        m = torchvision.models.get_model(fn, weights=pesos)
        enum_pesos = None
        if pretreinado:
            enum_pesos = str(torchvision.models.get_model_weights(fn).DEFAULT)
        info = {"torchvision": fn, "pesos": enum_pesos}
        if nome_cat == "squeezenet1_0":
            # features -> (Dropout do protocolo) -> Conv2d(512,n,1) -> ReLU -> pool
            info["camada_final_trocada"] = "classifier[1] (Conv2d 512->1000) -> cabeça conv1x1_squeezenet"
            info["dropout_original_substituido"] = "classifier[0] (Dropout 0.5)"
            return m.features, 512, info
        final = _obter_sub(m, cam_final)
        n_feat = final.in_features
        _trocar_sub(m, cam_final, nn.Identity())
        info["camada_final_trocada"] = (
            f"{_fmt_caminho(cam_final)} ({final.__class__.__name__} {n_feat}->{final.out_features})"
        )
        if cam_drop is not None:
            drop = _obter_sub(m, cam_drop)
            assert isinstance(drop, nn.Dropout), f"{nome_cat}: esperado Dropout em {cam_drop}"
            info["dropout_original_substituido"] = f"{_fmt_caminho(cam_drop)} (Dropout {drop.p})"
            _trocar_sub(m, cam_drop, nn.Identity())
        return m, n_feat, info

    return fabrica


for _n in CATALOGO_TORCHVISION:
    registrar_backbone(_n, _fabrica_torchvision(_n))


# ---------------------------------------------------------------------------
# Perdas
# ---------------------------------------------------------------------------
class SomaPonderada(nn.Module):
    def __init__(self, tarefas, cfg):
        super().__init__()
        self.pesos = {t: float(cfg.get("pesos_tarefa", {}).get(t, 1.0)) for t in tarefas}

    def forward(self, perdas, epoca, fase):
        return sum(self.pesos[t] * l for t, l in perdas.items())


registrar_perda("soma_ponderada", lambda tarefas, cfg: SomaPonderada(tarefas, cfg))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class ModeloProtocolo(nn.Module):
    """backbone -> Dropout(0.5) -> {tarefa: cabeça}. Cabeças sempre em fp32."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        bb, n_feat, info = _BACKBONES[cfg["backbone"]]()
        self.backbone = bb
        self.n_features = n_feat
        self.info_backbone = info
        self.dropout = nn.Dropout(p=RECEITA["dropout"])
        self.tarefas = {t: TAREFAS[t] for t in cfg["cabecas"]}
        self.cabecas = nn.ModuleDict(
            {
                t: _CABECAS[c["nome"]](n_feat, TAREFAS[t].n_classes, **c["kw"])
                for t, c in cfg["cabecas"].items()
            }
        )
        self.combinador = _PERDAS[cfg["perda"]](self.tarefas, cfg)

    def features(self, x):
        f = self.backbone(x)
        if f.ndim > 2 and self.cfg["backbone"] != "squeezenet1_0":
            f = torch.flatten(f, 1)
        return f

    def forward(self, x):
        f = self.dropout(self.features(x))
        saida = {}
        with torch.autocast(device_type=f.device.type, enabled=False):
            f32 = f.float()
            for t, cab in self.cabecas.items():
                saida[t] = cab(f32)
        return saida

    def parametros_novos(self):
        return list(self.cabecas.parameters()) + list(self.combinador.parameters())

    def nomes_novos(self):
        return {f"cabecas.{n}" for n, _ in self.cabecas.named_parameters()} | {
            f"combinador.{n}" for n, _ in self.combinador.named_parameters()
        }


def contar_parametros(modelo: ModeloProtocolo) -> dict:
    cab = {t: sum(p.numel() for p in c.parameters()) for t, c in modelo.cabecas.items()}
    bb = sum(p.numel() for p in modelo.backbone.parameters())
    comb = sum(p.numel() for p in modelo.combinador.parameters())
    return {
        "cabecas": cab,
        "cabecas_total": int(sum(cab.values())),
        "backbone": int(bb),
        "combinador": int(comb),
        "treinaveis_fase1": int(sum(cab.values()) + comb),
        "treinaveis_fase2": int(bb + sum(cab.values()) + comb),
    }


def _modulos_kan(cab):
    try:
        from efficient_kan.kan import KANLinear
    except Exception:
        return []
    return [m for m in cab.modules() if isinstance(m, KANLinear)]


class DiagnosticoKAN:
    """Fração das entradas de cada camada KAN dentro da grade base e da estendida."""

    def __init__(self, modelo: ModeloProtocolo):
        self.ganchos, self.acum = [], {}
        for t, cab in modelo.cabecas.items():
            for i, m in enumerate(_modulos_kan(cab)):
                chave = f"{t}.kan{i}"
                k = m.spline_order
                g = m.grid.detach()
                base = (float(g[0, k]), float(g[0, -k - 1]))
                ext = (float(g[0, 0]), float(g[0, -1]))
                self.acum[chave] = {"base": base, "estendida": ext, "n": 0, "dentro_base": 0, "dentro_ext": 0}
                self.ganchos.append(m.register_forward_pre_hook(self._gancho(chave)))

    def _gancho(self, chave):
        def h(_m, args):
            x = args[0].detach().float()
            a = self.acum[chave]
            a["n"] += x.numel()
            a["dentro_base"] += int(((x >= a["base"][0]) & (x <= a["base"][1])).sum())
            a["dentro_ext"] += int(((x >= a["estendida"][0]) & (x <= a["estendida"][1])).sum())

        return h

    def resultado(self):
        for g in self.ganchos:
            g.remove()
        out = {}
        for k, a in self.acum.items():
            n = max(1, a["n"])
            out[k] = {
                "grade_base": a["base"],
                "grade_estendida": a["estendida"],
                "frac_dentro_base": a["dentro_base"] / n,
                "frac_dentro_estendida": a["dentro_ext"] / n,
            }
        return out


# ---------------------------------------------------------------------------
# Métricas (AUC sempre de probabilidades)
# ---------------------------------------------------------------------------
def metricas_tarefa(y: np.ndarray, P: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    m = y != IGNORAR
    y, P = y[m], P[m]
    k = P.shape[1]
    pred = P.argmax(1)
    out = {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "prec_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "rec_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "acc_balanceada": float(balanced_accuracy_score(y, pred)),
        "matriz_confusao": confusion_matrix(y, pred, labels=list(range(k))).tolist(),
        "recall_por_classe": recall_score(y, pred, average=None, labels=list(range(k)), zero_division=0).tolist(),
        "prob_continua": bool(len(np.unique(np.round(P[:, -1], 6))) > 2),
    }
    try:
        if k == 2:
            out["auc"] = float(roc_auc_score(y, P[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(k))))
    except ValueError as e:
        out["auc"] = float("nan")
        out["auc_erro"] = str(e)
    if k == 2:
        cm = np.array(out["matriz_confusao"])
        out["sensibilidade_saida"] = float(cm[1, 1] / max(1, cm[1].sum()))
        out["especificidade_saida"] = float(cm[0, 0] / max(1, cm[0].sum()))
    return out


# ---------------------------------------------------------------------------
# Travas (duas sessões no mesmo Drive)
# ---------------------------------------------------------------------------
def _ler_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def adquirir_trava(dir_unidade, espera_s: float = 30.0, expira_min: float = 30.0) -> bool:
    p = Path(dir_unidade) / "lock.json"
    atual = _ler_json(p)
    if atual and atual.get("id_sessao") != ID_SESSAO:
        hb = _dt.datetime.fromisoformat(atual.get("heartbeat"))
        idade = (_dt.datetime.now(_dt.timezone.utc) - hb).total_seconds() / 60
        if idade < expira_min:
            return False
    gravar_json_atomico({"id_sessao": ID_SESSAO, "inicio": agora_iso(), "heartbeat": agora_iso()}, p)
    if espera_s > 0:
        time.sleep(espera_s)
    atual = _ler_json(p)
    return bool(atual and atual.get("id_sessao") == ID_SESSAO)


def atualizar_trava(dir_unidade) -> None:
    p = Path(dir_unidade) / "lock.json"
    atual = _ler_json(p) or {}
    if atual.get("id_sessao") not in (None, ID_SESSAO):
        raise RuntimeError(f"Trava de {dir_unidade} foi tomada por outra sessão ({atual.get('id_sessao')})")
    atual.update({"id_sessao": ID_SESSAO, "heartbeat": agora_iso()})
    atual.setdefault("inicio", agora_iso())
    gravar_json_atomico(atual, p)


def liberar_trava(dir_unidade) -> None:
    p = Path(dir_unidade) / "lock.json"
    atual = _ler_json(p)
    if atual and atual.get("id_sessao") == ID_SESSAO:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Sessão (dados prontos para treinar)
# ---------------------------------------------------------------------------
@dataclass
class Sessao:
    df: pd.DataFrame
    cache: np.ndarray
    raiz_dataset: Path
    meta: dict = field(default_factory=dict)


def preparar_sessao(
    raiz_trabalho="/content",
    caminho_dobras=None,
    caminho_meta=None,
    construir=True,
    exigir_hash_cache=False,
    df_extra: pd.DataFrame | None = None,
) -> Sessao:
    """Clona dataset e leguwoi fixados, lê as dobras e gera o cache em memória."""
    raiz_trabalho = Path(raiz_trabalho)
    raiz_ds = raiz_trabalho / "FDCPUnBGunshotDB"
    meta = {
        "dataset_commit": clonar_fixado(DATASET_URL, DATASET_COMMIT, raiz_ds),
        "leguwoi_commit": clonar_fixado(LEGUWOI_URL, LEGUWOI_COMMIT, raiz_trabalho / "leguwoi"),
    }
    listagem = listar_imagens(raiz_ds)
    if caminho_dobras is not None:
        df = ler_dobras(caminho_dobras)
        meta["dobras_sha256"] = sha256_arquivo(caminho_dobras)
        if list(df["rel_path"]) != list(listagem["rel_path"]):
            raise RuntimeError("As imagens clonadas não batem com o CSV de dobras")
        for c in ("case_id", "y_es", "y_dist", "strat4"):
            if not (df[c].astype(str).values == listagem[c].astype(str).values).all():
                raise RuntimeError(f"Coluna {c} do CSV de dobras diverge da listagem")
    else:
        df = listagem
    if df_extra is not None:
        df = df.merge(df_extra, on="rel_path", how="left", validate="one_to_one")
    meta_dobras = _ler_json(caminho_meta) if caminho_meta else None
    if meta_dobras:
        meta["semente_dobras"] = meta_dobras.get("semente_usada")
    cache = np.empty((0,), dtype=np.uint8)
    if construir:
        t0 = time.time()
        cache = construir_cache(raiz_ds, list(df["rel_path"]))
        meta["cache_s"] = round(time.time() - t0, 1)
        meta["cache_sha256"] = sha256_array(cache)
        esperado = (meta_dobras or {}).get("cache_sha256")
        meta["cache_sha256_esperado"] = esperado
        meta["cache_confere"] = (esperado == meta["cache_sha256"]) if esperado else None
        if esperado and esperado != meta["cache_sha256"]:
            if exigir_hash_cache:
                raise RuntimeError("SHA-256 do cache diverge do registrado no notebook 00")
            meta["cache_divergencia"] = _divergencia_cache(cache, raiz_ds, df)
    return Sessao(df=df, cache=cache, raiz_dataset=raiz_ds, meta=meta)


def _divergencia_cache(cache, raiz_ds, df, n=20):
    # Compara com um resize independente (cv2 direto) só para quantificar
    import cv2

    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    dif = []
    from PIL import Image

    for i in idx:
        with Image.open(raiz_ds / df["rel_path"].iloc[i]) as im:
            arr = np.array(im.convert("RGB"))
        ref = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        dif.append(int(np.abs(ref.astype(int) - cache[i].astype(int)).max()))
    return {"max_dif_pixel_amostra": max(dif), "amostra": int(len(idx))}


# ---------------------------------------------------------------------------
# Treino de uma unidade (modelo, semente, dobra)
# ---------------------------------------------------------------------------
class InterrupcaoSimulada(RuntimeError):
    pass


def _caminhos_unidade(raiz_resultados, raiz_checkpoints, unidade):
    rel = Path(unidade["experimento"]) / unidade["config"] / f"s{unidade['semente']}" / f"dobra{unidade['dobra']}"
    return Path(raiz_resultados) / rel, Path(raiz_checkpoints) / rel


def _estado_rng():
    est = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        est["cuda"] = torch.cuda.get_rng_state_all()
    return est


def _restaurar_rng(est):
    random.setstate(est["python"])
    np.random.set_state(est["numpy"])
    torch.set_rng_state(est["torch"])
    if torch.cuda.is_available() and "cuda" in est:
        torch.cuda.set_rng_state_all(est["cuda"])


def _configurar_fase(modelo, fase):
    if fase == 1:
        for p in modelo.backbone.parameters():
            p.requires_grad = False
        for p in modelo.parametros_novos():
            p.requires_grad = True
        opt = torch.optim.AdamW(modelo.parametros_novos(), lr=RECEITA["lr_f1"], weight_decay=RECEITA["wd_f1"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RECEITA["epocas_f1"])
    else:
        for p in modelo.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(modelo.parameters(), lr=RECEITA["lr_f2"], weight_decay=RECEITA["wd_f2"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RECEITA["epocas_f2"])
    return opt, sch


def _salvar_checkpoint(modelo, opt, sch, scaler, epoca, fase, historico, ident, destino, dir_tmp, tempo_treino):
    sd = modelo.state_dict()
    parcial = fase == 1
    if parcial:
        novos = modelo.nomes_novos()
        buffers = {n for n, _ in modelo.named_buffers()}
        sd = {k: v for k, v in sd.items() if k in novos or k in buffers}
    obj = {
        "protocolo_sha256": sha256_protocolo(),
        "ident": ident,
        "epoca": epoca,
        "fase": fase,
        "parcial": parcial,
        "estado_modelo": sd,
        "otimizador": opt.state_dict(),
        "agendador": sch.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _estado_rng(),
        "historico": historico,
        "tempo_treino_s": tempo_treino,
        "salvo_em": agora_iso(),
    }
    dir_tmp = Path(dir_tmp)
    dir_tmp.mkdir(parents=True, exist_ok=True)
    local = dir_tmp / f"ckpt-{uuid.uuid4().hex[:8]}.pt"
    torch.save(obj, local)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + f".tmp-{uuid.uuid4().hex[:6]}")
    shutil.copyfile(local, tmp)
    os.replace(tmp, destino)
    local.unlink(missing_ok=True)


def executar_unidade(
    sessao: Sessao,
    unidade: dict,
    raiz_resultados,
    raiz_checkpoints,
    precisao: str = "amp",
    dispositivo=None,
    num_workers=None,
    usar_trava: bool = True,
    espera_trava_s: float = 30.0,
    ckpt_intervalo_min: float = 0.0,
    dir_tmp="/tmp/protocolo_ckpt",
    limite_epocas: tuple | None = None,
    limite_imagens: int | None = None,
    parar_apos_epoca: int | None = None,
    verbose: bool = True,
) -> dict | None:
    """Treina e avalia uma unidade; retoma do checkpoint se houver.

    unidade = {"experimento", "config", "semente", "dobra", "coluna_dobra" (opcional)}.
    `limite_epocas=(e1, e2)` e `limite_imagens` existem só para testes rápidos
    (o resultado fica marcado como teste_rapido).
    """
    t_inicio = time.time()
    dispositivo = torch.device(dispositivo) if dispositivo else dispositivo_padrao()
    num_workers = num_workers_padrao() if num_workers is None else num_workers
    cfg = obter_config(unidade["config"])
    coluna_dobra = unidade.get("coluna_dobra", "fold")
    semente, dobra = int(unidade["semente"]), int(unidade["dobra"])
    dir_res, dir_ck = _caminhos_unidade(raiz_resultados, raiz_checkpoints, unidade)
    arq_metricas = dir_res / "metricas.json"
    if arq_metricas.exists():
        if verbose:
            print(f"✔ já concluída: {dir_res}")
        return None
    dir_res.mkdir(parents=True, exist_ok=True)
    if usar_trava and not adquirir_trava(dir_res, espera_s=espera_trava_s):
        if verbose:
            print(f"⏭ outra sessão está com {dir_res}")
        return None

    amp = precisao == "amp" and dispositivo.type == "cuda"
    precisao_efetiva = "amp_fp16" if amp else "fp32"
    teste_rapido = bool(limite_epocas or limite_imagens or parar_apos_epoca)
    e1, e2 = limite_epocas or (RECEITA["epocas_f1"], RECEITA["epocas_f2"])
    total = e1 + e2

    df = sessao.df
    idx_te = np.flatnonzero(df[coluna_dobra].to_numpy() == dobra)
    idx_tr = np.flatnonzero((df[coluna_dobra].to_numpy() != dobra) & (df[coluna_dobra].to_numpy() >= 0))
    if limite_imagens:
        rng = np.random.default_rng(semente)
        idx_tr = np.sort(rng.choice(idx_tr, size=min(limite_imagens, len(idx_tr)), replace=False))
        idx_te = np.sort(rng.choice(idx_te, size=min(max(8, limite_imagens // 4), len(idx_te)), replace=False))
    casos_tr, casos_te = set(df["case_id"].iloc[idx_tr]), set(df["case_id"].iloc[idx_te])
    if casos_tr & casos_te:
        raise RuntimeError("Vazamento: caso em treino e teste")

    ident = {
        "experimento": unidade["experimento"],
        "config": cfg["nome"],
        "semente": semente,
        "dobra": dobra,
        "coluna_dobra": coluna_dobra,
        "precisao": precisao_efetiva,
        "teste_rapido": teste_rapido,
        "limite_epocas": [e1, e2],
    }

    # Modelo (inicialização da cabeça determinada pela semente)
    semear(semente * 1000 + dobra)
    modelo = ModeloProtocolo(cfg)
    if dispositivo.type == "cuda":
        modelo.backbone.to(memory_format=torch.channels_last)
    modelo.to(dispositivo)
    torch.backends.cudnn.benchmark = True if dispositivo.type == "cuda" else torch.backends.cudnn.benchmark

    pesos = {
        t: torch.tensor(pesos_classe(df[tar.coluna].to_numpy()[idx_tr], tar.n_classes), dtype=torch.float32, device=dispositivo)
        for t, tar in modelo.tarefas.items()
    }
    criterios = {t: nn.CrossEntropyLoss(weight=w, ignore_index=IGNORAR) for t, w in pesos.items()}

    ds_tr = DatasetGuWID(sessao.cache, df, idx_tr, transform_treino())
    ds_te = DatasetGuWID(sessao.cache, df, idx_te, transform_teste())

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    historico, tempo_treino = [], 0.0
    inicio_epoca = 0
    arq_ck = dir_ck / "ckpt_last.pt"
    opt = sch = None
    fase_atual = None
    if arq_ck.exists():
        ck = torch.load(arq_ck, map_location="cpu", weights_only=False)
        if ck["ident"] != ident:
            raise RuntimeError(f"Checkpoint de outra configuração em {arq_ck}: {ck['ident']} != {ident}")
        faltando, inesperado = modelo.load_state_dict(ck["estado_modelo"], strict=False)
        if inesperado or (not ck["parcial"] and faltando):
            raise RuntimeError(f"Checkpoint incompatível: faltando={faltando[:5]} inesperado={inesperado[:5]}")
        modelo.to(dispositivo)
        historico, tempo_treino = ck["historico"], ck["tempo_treino_s"]
        inicio_epoca = ck["epoca"] + 1
        fase_prox = 1 if inicio_epoca < e1 else 2
        if fase_prox == ck["fase"]:
            fase_atual = fase_prox
            opt, sch = _configurar_fase(modelo, fase_atual)
            opt.load_state_dict(ck["otimizador"])
            sch.load_state_dict(ck["agendador"])
        scaler.load_state_dict(ck["scaler"])
        _restaurar_rng(ck["rng"])
        if verbose:
            print(f"↻ retomando {dir_res.name} de {cfg['nome']} s{semente} na época {inicio_epoca + 1}/{total}")
        del ck

    ultimo_ckpt = time.time()
    tempo_ckpt = 0.0
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tarefa_log = "es" if "es" in modelo.tarefas else next(iter(modelo.tarefas))
    for epoca in range(inicio_epoca, total):
        fase = 1 if epoca < e1 else 2
        if fase != fase_atual:
            opt, sch = _configurar_fase(modelo, fase)
            fase_atual = fase
        modelo.train()
        dl = loader_treino(ds_tr, semente, dobra, epoca, RECEITA["lote"], num_workers, RECEITA["drop_last"])
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        soma_perda, n_amostras, acertos, n_log = 0.0, 0, 0, 0
        perdas_t = {t: 0.0 for t in modelo.tarefas}
        for img, y_es, y_dist in dl:
            img = img.to(dispositivo, non_blocking=True)
            if dispositivo.type == "cuda":
                img = img.contiguous(memory_format=torch.channels_last)
            ys = {"es": y_es.to(dispositivo, non_blocking=True), "dist": y_dist.to(dispositivo, non_blocking=True)}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=amp):
                saida = modelo(img)
            perdas = {}
            for t, logit in saida.items():
                y = ys[t]
                if (y != IGNORAR).any():
                    perdas[t] = criterios[t](logit.float(), y)
                else:
                    perdas[t] = logit.sum() * 0.0
            perda = modelo.combinador(perdas, epoca, fase)
            if not torch.isfinite(perda):
                raise FloatingPointError(f"Perda não finita na época {epoca + 1} ({precisao_efetiva})")
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            b = img.shape[0]
            soma_perda += float(perda.detach()) * b
            n_amostras += b
            for t in perdas:
                perdas_t[t] += float(perdas[t].detach()) * b
            y = ys[tarefa_log]
            m = y != IGNORAR
            acertos += int((saida[tarefa_log].argmax(1)[m] == y[m]).sum())
            n_log += int(m.sum())
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tempo_treino += dt
        lr = opt.param_groups[0]["lr"]
        sch.step()
        reg = {
            "epoca": epoca + 1,
            "fase": fase,
            "perda": soma_perda / max(1, n_amostras),
            f"acc_treino_{tarefa_log}": acertos / max(1, n_log),
            "lr": lr,
            "tempo_s": round(dt, 3),
            "n_amostras": n_amostras,
        }
        for t, v in perdas_t.items():
            reg[f"perda_{t}"] = v / max(1, n_amostras)
        if dispositivo.type == "cuda":
            reg["mem_pico_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        historico.append(reg)
        if verbose and (epoca + 1 in (1, e1, total) or (epoca + 1) % 10 == 0):
            print(
                f"  {cfg['nome']} s{semente} d{dobra} | época {epoca + 1:02d}/{total} (F{fase}) | "
                f"perda {reg['perda']:.4f} | acc treino {reg[f'acc_treino_{tarefa_log}']:.3f} | {dt:.1f}s"
            )
        fim_fase = epoca + 1 in (e1, total)
        if epoca + 1 < total and (
            fim_fase or (time.time() - ultimo_ckpt) / 60 >= ckpt_intervalo_min or parar_apos_epoca == epoca + 1
        ):
            tc = time.time()
            _salvar_checkpoint(modelo, opt, sch, scaler, epoca, fase, historico, ident, arq_ck, dir_tmp, tempo_treino)
            tempo_ckpt += time.time() - tc
            ultimo_ckpt = time.time()
        if usar_trava:
            atualizar_trava(dir_res)
        if parar_apos_epoca is not None and epoca + 1 == parar_apos_epoca:
            raise InterrupcaoSimulada(f"interrompido após a época {epoca + 1}")

    # Avaliação ÚNICA no teste da dobra (modelo da última época)
    modelo.eval()
    diag = DiagnosticoKAN(modelo)
    probs = {t: [] for t in modelo.tarefas}
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for img, _, _ in loader_teste(ds_te, RECEITA["lote"], num_workers):
            img = img.to(dispositivo, non_blocking=True)
            if dispositivo.type == "cuda":
                img = img.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=amp):
                saida = modelo(img)
            for t, logit in saida.items():
                probs[t].append(F.softmax(logit.float(), dim=1).cpu().numpy())
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    tempo_teste = time.time() - t0
    diag_kan = diag.resultado()

    pred = pd.DataFrame(
        {
            "rel_path": df["rel_path"].to_numpy()[idx_te],
            "case_id": df["case_id"].to_numpy()[idx_te],
            "y_es": df["y_es"].to_numpy()[idx_te],
            "y_dist": df["y_dist"].to_numpy()[idx_te],
        }
    )
    metricas = {}
    for t, lst in probs.items():
        P = np.concatenate(lst)
        tar = TAREFAS[t]
        for c in range(tar.n_classes):
            pred[f"p_{t}_{c}"] = P[:, c]
        pred[f"pred_{t}"] = P.argmax(1)
        metricas[t] = metricas_tarefa(df[tar.coluna].to_numpy()[idx_te], P)

    resultado = {
        "unidade": ident,
        "config": cfg,
        "backbone_info": modelo.info_backbone,
        "receita": RECEITA,
        "parametros": contar_parametros(modelo),
        "metricas": metricas,
        "diagnostico_kan": diag_kan,
        "n_treino": int(len(idx_tr)),
        "n_teste": int(len(idx_te)),
        "n_casos_treino": len(casos_tr),
        "n_casos_teste": len(casos_te),
        "pesos_classe": {t: w.tolist() for t, w in pesos.items()},
        "tempos": {
            "treino_s": round(tempo_treino, 2),
            "teste_s": round(tempo_teste, 2),
            "ckpt_s_nesta_sessao": round(tempo_ckpt, 2),
            "parede_nesta_sessao_s": round(time.time() - t_inicio, 2),
            "f1_s_por_epoca": float(np.mean([h["tempo_s"] for h in historico if h["fase"] == 1] or [np.nan])),
            "f2_s_por_epoca": float(np.mean([h["tempo_s"] for h in historico if h["fase"] == 2] or [np.nan])),
        },
        "num_workers": num_workers,
        "ambiente": info_ambiente(),
        "dados": sessao.meta,
        "plugins": PLUGINS_CARREGADOS,
        "concluido_em": agora_iso(),
    }
    pred.to_csv(dir_res / "pred_teste.csv", index=False)
    pd.DataFrame(historico).to_csv(dir_res / "historico.csv", index=False)
    gravar_json_atomico(resultado, arq_metricas)
    if arq_ck.exists():
        arq_ck.unlink()
    if usar_trava:
        liberar_trava(dir_res)
    if verbose:
        m = metricas[tarefa_log]
        print(
            f"✅ {cfg['nome']} s{semente} d{dobra}: acc {m['acc']:.4f} | AUC {m['auc']:.4f} | "
            f"F1 {m['f1_macro']:.4f} | treino {tempo_treino / 60:.1f} min"
        )
    return resultado


# ---------------------------------------------------------------------------
# Filas
# ---------------------------------------------------------------------------
def montar_fila(experimento: str, configs, sementes=(42,), dobras=range(N_DOBRAS), coluna_dobra: str = "fold"):
    """Ordem: config -> semente -> dobra (cada config completa as 5 dobras antes da próxima)."""
    return [
        {"experimento": experimento, "config": c, "semente": int(s), "dobra": int(d), "coluna_dobra": coluna_dobra}
        for c in configs
        for s in sementes
        for d in dobras
    ]


def executar_fila(sessao: Sessao, fila, raiz_resultados, raiz_checkpoints, plugins=(), **kw):
    for p in plugins:
        carregar_plugin(p)
    feitos = []
    for u in fila:
        r = executar_unidade(sessao, u, raiz_resultados, raiz_checkpoints, **kw)
        if r is not None:
            feitos.append(r["unidade"])
    return feitos


def ler_metricas_unidade(raiz_resultados, unidade: dict) -> dict | None:
    dir_res, _ = _caminhos_unidade(raiz_resultados, ".", unidade)
    return _ler_json(dir_res / "metricas.json")


def ler_resultados(raiz_resultados, experimento: str) -> pd.DataFrame:
    """Uma linha por unidade concluída (modelo x semente x dobra)."""
    linhas = []
    for arq in sorted(Path(raiz_resultados, experimento).glob("*/s*/dobra*/metricas.json")):
        r = _ler_json(arq)
        if not r:
            continue
        u = r["unidade"]
        base = {
            "experimento": u["experimento"],
            "config": u["config"],
            "semente": u["semente"],
            "dobra": u["dobra"],
            "precisao": u["precisao"],
            "teste_rapido": u["teste_rapido"],
            "backbone": r["config"]["backbone"],
            "cabecas": json.dumps({t: c["nome"] for t, c in r["config"]["cabecas"].items()}),
            "params_cabeca": r["parametros"]["cabecas_total"],
            "params_total": r["parametros"]["treinaveis_fase2"],
            "n_treino": r["n_treino"],
            "n_teste": r["n_teste"],
            "treino_min": r["tempos"]["treino_s"] / 60,
            "f1_s_epoca": r["tempos"]["f1_s_por_epoca"],
            "f2_s_epoca": r["tempos"]["f2_s_por_epoca"],
            "gpu": (r["ambiente"]["gpu"] or {}).get("nome"),
            "protocolo_sha256": r["ambiente"]["protocolo_sha256"],
            "dobras_sha256": r["dados"].get("dobras_sha256"),
            "plugins": json.dumps({k: v["sha256"] for k, v in r.get("plugins", {}).items()}),
            "caminho": str(arq.parent),
        }
        for t, m in r["metricas"].items():
            for k in ("acc", "prec_macro", "rec_macro", "f1_macro", "auc", "acc_balanceada"):
                base[f"{t}_{k}"] = m.get(k)
            for k in ("sensibilidade_saida", "especificidade_saida"):
                if k in m:
                    base[f"{t}_{k}"] = m[k]
        for k, d in (r.get("diagnostico_kan") or {}).items():
            base[f"kan_{k}_frac_base"] = d["frac_dentro_base"]
        linhas.append(base)
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Calibração e ponte
# ---------------------------------------------------------------------------
def avaliar_ponte(fp32: dict, amp: dict, amp2: dict | None = None, faixa_acc=FAIXA_PONTE_ANTIGA) -> dict:
    """Regra da ponte (acc em %, AUC em [0,1]); devolve a decisão e os motivos.

    AMP é adotado se |Δacc| <= 2 pp, |ΔAUC| <= 0,015 e as duas acurácias na
    faixa. Se só a parte de acurácia falhar: "rodar_amp2" (2ª execução AMP com
    outra semente), e cai para fp32 se ela também falhar. Se a AUC falhar: fp32.
    Se o próprio fp32 sair da faixa: "parar" (o problema não é numérico).
    decisao ∈ {"amp", "fp32", "rodar_amp2", "parar"}.
    """
    lim = PONTE
    lo, hi = faixa_acc

    def _criterio(a):
        d_acc = abs(fp32["acc"] - a["acc"])
        d_auc = abs(fp32["auc"] - a["auc"])
        ok_acc = d_acc <= lim["max_delta_acc_pp"] and lo <= a["acc"] <= hi
        ok_auc = d_auc <= lim["max_delta_auc"]
        return ok_acc, ok_auc, {"delta_acc_pp": round(d_acc, 3), "delta_auc": round(d_auc, 4), "acc_amp": a["acc"]}

    res = {"fp32": fp32, "amp": amp, "amp2": amp2, "faixa_acc": [lo, hi], "limites": lim}
    if not (lo <= fp32["acc"] <= hi):
        res.update(decisao="parar", motivo=f"fp32 com acc {fp32['acc']:.2f}% fora da faixa [{lo:.2f}, {hi:.2f}]")
        return res
    ok_acc, ok_auc, det = _criterio(amp)
    res["criterio_amp"] = det | {"ok_acc": ok_acc, "ok_auc": ok_auc}
    if ok_acc and ok_auc:
        res.update(decisao="amp", motivo="AMP dentro dos limites")
    elif not ok_auc:
        res.update(decisao="fp32", motivo="ΔAUC acima do limite")
    elif amp2 is None:
        res.update(decisao="rodar_amp2", motivo="só a acurácia falhou; rodar 2ª execução AMP (semente 43)")
    else:
        ok_acc2, ok_auc2, det2 = _criterio(amp2)
        res["criterio_amp2"] = det2 | {"ok_acc": ok_acc2, "ok_auc": ok_auc2}
        if ok_acc2 and ok_auc2:
            res.update(decisao="amp", motivo="2ª execução AMP dentro dos limites")
        else:
            res.update(decisao="fp32", motivo="2ª execução AMP também falhou")
    return res


def medir_vazao(
    sessao: Sessao,
    config,
    dispositivo=None,
    precisao: str = "amp",
    n_lotes: int = 30,
    aquecimento: int = 5,
    num_workers=None,
) -> dict:
    """Mede imagens/s em F1 (backbone congelado), F2 (tudo) e teste, com o pipeline real."""
    dispositivo = torch.device(dispositivo) if dispositivo else dispositivo_padrao()
    num_workers = num_workers_padrao() if num_workers is None else num_workers
    cfg = obter_config(config)
    amp = precisao == "amp" and dispositivo.type == "cuda"
    semear(0)
    modelo = ModeloProtocolo(cfg)
    if dispositivo.type == "cuda":
        modelo.backbone.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    modelo.to(dispositivo)
    df = sessao.df
    idx = np.arange(len(df))
    ds_tr = DatasetGuWID(sessao.cache, df, idx, transform_treino())
    ds_te = DatasetGuWID(sessao.cache, df, idx, transform_teste())
    pesos = {t: torch.ones(tar.n_classes, device=dispositivo) for t, tar in modelo.tarefas.items()}
    crit = {t: nn.CrossEntropyLoss(weight=w, ignore_index=IGNORAR) for t, w in pesos.items()}
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = {"config": cfg["nome"], "precisao": "amp_fp16" if amp else "fp32", "parametros": contar_parametros(modelo)}

    def _rodar(fase):
        opt, _ = _configurar_fase(modelo, fase)
        modelo.train()
        dl = loader_treino(ds_tr, 0, 0, fase, RECEITA["lote"], num_workers, True)
        it = iter(dl)
        if dispositivo.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        n_img, t0 = 0, None
        for i in range(aquecimento + n_lotes):
            if i == aquecimento:
                if dispositivo.type == "cuda":
                    torch.cuda.synchronize()
                t0, n_img = time.time(), 0
            img, y_es, y_dist = next(it)
            img = img.to(dispositivo, non_blocking=True)
            if dispositivo.type == "cuda":
                img = img.contiguous(memory_format=torch.channels_last)
            ys = {"es": y_es.to(dispositivo), "dist": y_dist.to(dispositivo)}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=amp):
                s = modelo(img)
            perdas = {
                t: (crit[t](l.float(), ys[t]) if (ys[t] != IGNORAR).any() else l.sum() * 0) for t, l in s.items()
            }
            scaler.scale(modelo.combinador(perdas, 0, fase)).backward()
            scaler.step(opt)
            scaler.update()
            n_img += img.shape[0]
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        r = {"img_s": n_img / dt}
        if dispositivo.type == "cuda":
            r["mem_pico_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        return r

    out["fase1"] = _rodar(1)
    out["fase2"] = _rodar(2)
    modelo.eval()
    dl = loader_teste(ds_te, RECEITA["lote"], num_workers)
    it = iter(dl)
    n_img = 0
    with torch.no_grad():
        for i in range(aquecimento + n_lotes):
            if i == aquecimento:
                if dispositivo.type == "cuda":
                    torch.cuda.synchronize()
                t0, n_img = time.time(), 0
            img, _, _ = next(it)
            img = img.to(dispositivo, non_blocking=True)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=amp):
                modelo(img)
            n_img += img.shape[0]
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    out["teste"] = {"img_s": n_img / (time.time() - t0)}
    del modelo
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
    return out


def estimar_rodada(vazao: dict, n_img: int = 2554, n_dobras: int = N_DOBRAS) -> dict:
    """Horas de GPU de uma rodada 5-fold x 60 épocas a partir da vazão medida."""
    n_tr = n_img * (n_dobras - 1) / n_dobras
    n_te = n_img / n_dobras
    lote = RECEITA["lote"]
    n_tr_util = (n_tr // lote) * lote  # drop_last
    s_f1 = n_tr_util / vazao["fase1"]["img_s"]
    s_f2 = n_tr_util / vazao["fase2"]["img_s"]
    s_te = n_te / vazao["teste"]["img_s"]
    por_dobra = RECEITA["epocas_f1"] * s_f1 + RECEITA["epocas_f2"] * s_f2 + s_te
    return {
        "s_epoca_f1": s_f1,
        "s_epoca_f2": s_f2,
        "min_por_dobra": por_dobra / 60,
        "horas_rodada_5fold": n_dobras * por_dobra / 3600,
    }


def medir_gravacao(modelo: nn.Module, dir_destino, dir_tmp="/tmp/protocolo_ckpt") -> dict:
    """Tempo para gravar um checkpoint cheio (modelo + AdamW) no destino (ex.: Drive)."""
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-5)
    for p in modelo.parameters():
        opt.state[p] = {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p), "step": torch.tensor(1.0)}
    obj = {"m": modelo.state_dict(), "o": opt.state_dict()}
    Path(dir_tmp).mkdir(parents=True, exist_ok=True)
    local = Path(dir_tmp) / "medir.pt"
    t0 = time.time()
    torch.save(obj, local)
    t_local = time.time() - t0
    tam = local.stat().st_size
    destino = Path(dir_destino) / "medir_gravacao.pt"
    destino.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shutil.copyfile(local, destino)
    t_destino = time.time() - t0
    destino.unlink(missing_ok=True)
    local.unlink(missing_ok=True)
    return {"tamanho_gb": round(tam / 1024**3, 3), "s_local": round(t_local, 2), "s_destino": round(t_destino, 2)}


# ---------------------------------------------------------------------------
# Configs nativas dos Experimentos 1 e 2
# ---------------------------------------------------------------------------
CONFIGS_EXP1 = {
    "a": "resnet152__linear",
    "b": "resnet152__mlp128",
    "c": "resnet152__mlp_iso",
    "d": "resnet152__kan",
    "e": "resnet152__kan_pequena",
}
for _cab in ("linear", "mlp128", "mlp_iso", "kan", "kan_pequena"):
    registrar_config(
        f"resnet152__{_cab}",
        {"backbone": "resnet152", "cabecas": {"es": _cab}, "descricao": f"Exp. 1: ResNet-152 + cabeça {_cab}"},
    )

BASELINES = [
    "alexnet",
    "mobilenet_v2",
    "mnasnet1_3",
    "shufflenet_v2_x0_5",
    "squeezenet1_0",
    "densenet169",
    "maxvit_t",
    "swin_b",
    "efficientnet_v2_l",
    "vgg16",
    "regnet_y_16gf",
]
for _bb in BASELINES:
    _c = "conv1x1_squeezenet" if _bb == "squeezenet1_0" else "linear"
    registrar_config(
        f"{_bb}__{'conv1x1' if _bb == 'squeezenet1_0' else 'linear'}",
        {"backbone": _bb, "cabecas": {"es": _c}, "descricao": f"Exp. 2: baseline {_bb} pré-treinado"},
    )
CONFIGS_EXP2 = {
    _bb: f"{_bb}__{'conv1x1' if _bb == 'squeezenet1_0' else 'linear'}" for _bb in BASELINES
}
