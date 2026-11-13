"""Reprodução das Tabelas 4 e 5 de Lira et al. (2025, IJLM) com e sem vazamento de casos.

Não altera ``protocolo_comum.py`` nem ``configs_mlsd.py``: só usa utilidades deles
(clone fixado, JSON atômico, travas). O treino NÃO é o do protocolo único: aqui se
roda a receita do próprio leguwoi, importando o código deles do clone fixado
(classes de ``models``, ``GunshotDatasetConfig``, ``iterative_split``).

O que é reproduzido (``leguwoi/src/tasks/train_test_validation_fixed_split.py``):

* imagens lidas com PIL e aumentadas UMA vez ao carregar (o conjunto de treino
  aumentado fica fixo em memória); teste só com Resize + Normalize;
* ``skorch.NeuralNetClassifier`` com SGD (lr 0,01, momentum 0,9), CrossEntropyLoss,
  ``ValidSplit(cv=5, stratified=True)`` interno e SEM embaralhar os lotes (padrão do
  skorch); o teste usa os pesos da ÚLTIMA época (o Checkpoint deles não é recarregado);
* a única semente do código, ``GunshotDatasetConfig.SEED = 0x40``, vai só para a
  reamostragem. Onde o original não fixa semente (aumentações, pesos iniciais), este
  módulo fixa a mesma 0x40 para que as rodadas sejam repetíveis.

Receitas (o repositório tem mais de uma versão; ver ``RECEITAS``):

* ``A``  = tag ``V_0.1`` (19/02/2024), a que o texto do artigo descreve: SMOTEENN,
  pesos de classe balanceados, CosineAnnealingWarmRestarts;
* ``B``  = ``master`` (commit 83d82f1): sem reamostragem, pesos uniformes, StepLR(50);
* ``B_smoteenn`` = ``master`` com ``--data_resampler SMOTEENN``.

A única coisa que muda entre "com vazamento" e "sem vazamento" é a partição:

* ``vazado_p{s}``   = ``iterative_split`` do leguwoi com ``np.random.seed(s)``;
* ``agrupado_d{k}`` = dobra k de ``folds_entrada_saida.csv`` como teste (agrupada por
  ano.caso), restrita às imagens do CSV de Lira.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import protocolo_comum as pc

VERSAO = "1.0.0"
SEMENTE = 0x40  # GunshotDatasetConfig.SEED
TAG_V01 = "V_0.1"
PREFIXO_CAMINHO_LIRA = "/home/pedro/databases/tiros/"

TAREFAS = {
    # tarefa -> (CSV do dataset, coluna-alvo)
    "es": ("gunshots_eqx.csv", "shot_type"),
    "dist": ("gunshots_eqx_ONLY_INPUT_SHOTS.csv", "distance"),
}

# As 59 linhas das Tabelas 4 e 5, na ordem do artigo (nomes das classes do leguwoi).
MODELOS = [
    "Alexnet", "Densenet121", "Densenet161", "Densenet169", "Densenet201",
    "EfficientnetB0", "EfficientnetB1", "EfficientnetB2", "EfficientnetB3", "EfficientnetB4",
    "EfficientnetB5", "EfficientnetB6", "EfficientnetB7", "EfficientnetV2L", "EfficientnetV2M",
    "EfficientnetV2S", "MaxVit", "MNASnet05", "MNASnet075", "MNASnet10", "MNASnet13",
    "MobileNetV2", "MobileNetV3Large", "MobileNetV3Small",
    "RegnetX16GF", "RegnetX32GF", "RegnetX400MF", "RegnetX800MF", "RegnetX8GF",
    "RegnetY16GF", "RegnetY32GF", "RegnetY400MF", "RegnetY800MF", "RegnetY8GF",
    "Resnet152", "Resnet101", "Resnet50", "Resnet34", "Resnet18",
    "ShuffleNetV2x05", "ShuffleNetV2x10", "ShuffleNetV2x15", "ShuffleNetV2x20",
    "Squeezenet10", "Squeezenet11",
    "SwinTransformerB", "SwinTransformerS", "SwinTransformerT",
    "SwinTransformerV2B", "SwinTransformerV2S", "SwinTransformerV2T",
    "VGG11", "VGG13", "VGG16", "VGG19", "ViTB16", "ViTB32", "ViTL16", "ViTL32",
]
# As 12 arquiteturas já usadas na dissertação: saem primeiro na fila.
DOZE = [
    "Resnet152", "Alexnet", "MobileNetV2", "MNASnet13", "ShuffleNetV2x05", "Squeezenet10",
    "Densenet169", "MaxVit", "SwinTransformerB", "EfficientnetV2L", "VGG16", "RegnetY16GF",
]
# Artigo, "Experimental setup": lote 32 para estes modelos, 64 para os demais
# (os de lote 16, RegnetY128GF e ViTH14, não estão nas tabelas).
LOTE_32 = {
    "EfficientnetB6", "EfficientnetB7", "EfficientnetV2L", "Squeezenet10",
    "SwinTransformerT", "SwinTransformerS", "SwinTransformerB",
    "SwinTransformerV2T", "SwinTransformerV2S", "SwinTransformerV2B",
    "ShuffleNetV2x20", "ShuffleNetV2x15", "ViTB16", "ViTB32", "ViTL16", "ViTL32", "MaxVit",
}

RECEITAS = {
    "A": {
        "origem": "tag V_0.1 (commit 14956dd8, 19/02/2024)",
        "reamostragem": "SMOTEENN", "pesos_classe": "balanced",
        "scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-7)",
        "epocas": 1000,
        "nota": "modelos e aumentações do master (as aumentações são idênticas na tag; "
                "as classes de modelo da tag tinham erros corrigidos em 10/03/2024)",
    },
    "B": {
        "origem": "master (commit 83d82f1)",
        "reamostragem": None, "pesos_classe": "uniforme",
        "scheduler": "StepLR(step_size=50, gamma=0.1)",
        "epocas": 300,
        "nota": "artigo: 1000 épocas; com StepLR(50) o lr é 1e-8 na época 300 — "
                "equivalência conferida na calibração",
    },
    "B_smoteenn": {
        "origem": "master (commit 83d82f1) com --data_resampler SMOTEENN",
        "reamostragem": "SMOTEENN", "pesos_classe": "uniforme",
        "scheduler": "StepLR(step_size=50, gamma=0.1)",
        "epocas": 300,
        "nota": "como B, com a reamostragem que o artigo descreve",
    },
}

METRICAS = [
    "acc", "prec_M", "prec_m", "prec_W", "rec_M", "rec_m", "rec_W", "f1_M", "f1_m", "f1_W",
    "auc_M", "auc_m", "auc_W", "espec_M", "espec_m", "espec_W",
]


def lote_do_artigo(modelo: str) -> int:
    return 32 if modelo in LOTE_32 else 64


def semear(semente: int = SEMENTE) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


# ---------------------------------------------------------------------------
# Contexto: clones fixados, código do leguwoi, CSVs e partições
# ---------------------------------------------------------------------------
@dataclass
class Contexto:
    raiz_dataset: Path
    raiz_leguwoi: Path
    dobras: pd.DataFrame            # rel_path, case_id (ano.caso), fold
    lw: dict                        # módulos/classes importados do leguwoi
    meta: dict = field(default_factory=dict)
    dir_cache: Path | None = None
    _csv: dict = field(default_factory=dict)
    _arrays: dict = field(default_factory=dict)


def varrer_sementes(raiz_leguwoi) -> list[str]:
    """Todas as linhas .py do leguwoi que mencionam semente (arquivo:linha: texto)."""
    achados = []
    for p in sorted(Path(raiz_leguwoi).rglob("*.py")):
        for i, linha in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            baixo = linha.lower()
            if "seed" in baixo or "random_state" in baixo:
                achados.append(f"{p.relative_to(raiz_leguwoi).as_posix()}:{i}: {linha.strip()}")
    return achados


SEMENTES_ESPERADAS = [
    "src/configs/gunshot_dataset_config.py:8: SEED = 0x40",
    "src/data/gunshot_dataset.py:82: resampler = resampler_class(random_state=GunshotDatasetConfig.SEED)",
    "src/data/gunshot_dataset_kfold.py:98: resampler = resampler_class(random_state=GunshotDatasetConfig.SEED)",
]


def _git(raiz, *args) -> str:
    r = subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def diferencas_v01(raiz_leguwoi) -> dict:
    """Confere, no clone, o que muda da tag V_0.1 para o commit fixado."""
    arqs = {
        "aumentacoes": "src/configs/gunshot_dataset_config.py",
        "modelos": "src/models/torchvision_models.py",
        "otimizacao": "src/configs/optimization_configs.py",
        "particao_e_dados": "src/data/gunshot_dataset.py",
    }
    out = {"commit_v01": _git(raiz_leguwoi, "rev-parse", f"{TAG_V01}^{{commit}}").strip()}
    for nome, arq in arqs.items():
        d = _git(raiz_leguwoi, "diff", TAG_V01, pc.LEGUWOI_COMMIT, "--", arq)
        out[nome] = {"arquivo": arq, "linhas_alteradas": sum(
            1 for l in d.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))}
    v01 = _git(raiz_leguwoi, "show", f"{TAG_V01}:src/data/gunshot_dataset.py")
    out["v01_smoteenn_fixo"] = "resampler = SMOTEENN(random_state=GunshotDatasetConfig.SEED)" in v01
    out["v01_iterative_split_sem_semente"] = "IterativeStratification(" in v01 and "random_state" not in v01.split(
        "def get_gunshow_dataset(")[0].split("def iterative_split")[-1]
    return out


def preparar(raiz_trabalho="/content", caminho_dobras=None, dir_cache=None, clonar=True) -> Contexto:
    """Clona dataset e leguwoi nos commits fixados e importa o código do leguwoi."""
    raiz_trabalho = Path(raiz_trabalho)
    raiz_ds = raiz_trabalho / "FDCPUnBGunshotDB"
    raiz_lw = raiz_trabalho / "leguwoi"
    meta = {"versao_modulo": VERSAO, "semente": SEMENTE}
    if clonar:
        meta["dataset_commit"] = pc.clonar_fixado(pc.DATASET_URL, pc.DATASET_COMMIT, raiz_ds)
        meta["leguwoi_commit"] = pc.clonar_fixado(pc.LEGUWOI_URL, pc.LEGUWOI_COMMIT, raiz_lw)
    else:
        meta["leguwoi_commit"] = _git(raiz_lw, "rev-parse", "HEAD").strip()
        if meta["leguwoi_commit"] != pc.LEGUWOI_COMMIT:
            raise RuntimeError(f"leguwoi em {meta['leguwoi_commit']}, esperado {pc.LEGUWOI_COMMIT}")

    achados = varrer_sementes(raiz_lw)
    if achados != SEMENTES_ESPERADAS:
        raise RuntimeError(
            "O leguwoi usa semente em lugares diferentes do registrado. Achado:\n  " + "\n  ".join(achados))
    meta["sementes_no_leguwoi"] = achados
    meta["diferencas_v01"] = diferencas_v01(raiz_lw)

    src = str(raiz_lw / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import configs as lw_configs  # noqa: E402  (leguwoi)
    import models as lw_models  # noqa: E402
    from data.gunshot_dataset import iterative_split  # noqa: E402

    if Path(lw_models.__file__).resolve().parent != (raiz_lw / "src" / "models").resolve():
        raise RuntimeError(f"'models' importado de {lw_models.__file__}, não do leguwoi")
    if lw_configs.GunshotDatasetConfig.SEED != SEMENTE:
        raise RuntimeError("GunshotDatasetConfig.SEED mudou")
    faltam = [m for m in MODELOS if not hasattr(lw_models, m)]
    if faltam:
        raise RuntimeError(f"Classes ausentes no leguwoi: {faltam}")
    lw = {"models": lw_models, "DatasetConfig": lw_configs.GunshotDatasetConfig,
          "OptimizerConfig": lw_configs.GunshotOptimizerConfig, "iterative_split": iterative_split}

    if caminho_dobras is None:
        raise ValueError("Informe caminho_dobras (folds_entrada_saida.csv)")
    dobras = pc.ler_dobras(caminho_dobras)[["rel_path", "case_id", "fold"]]
    meta["dobras_sha256"] = pc.sha256_arquivo(caminho_dobras)
    ctx = Contexto(raiz_dataset=raiz_ds, raiz_leguwoi=raiz_lw, dobras=dobras, lw=lw, meta=meta,
                   dir_cache=Path(dir_cache) if dir_cache else raiz_trabalho / "cache_lira")
    return ctx


def csv_lira(ctx: Contexto, tarefa: str) -> pd.DataFrame:
    """CSV do dataset como o leguwoi o lê (alvo já codificado), mais rel_path, caso real e dobra."""
    if tarefa in ctx._csv:
        return ctx._csv[tarefa]
    from sklearn.preprocessing import LabelEncoder

    nome, alvo = TAREFAS[tarefa]
    df = pd.read_csv(ctx.raiz_dataset / "csv_configs" / nome)
    lb = LabelEncoder()
    df[alvo] = lb.fit_transform(df[alvo])  # como em get_gunshow_dataset
    rel = df["path"].str.replace(PREFIXO_CAMINHO_LIRA, "database/", regex=False)
    if rel.duplicated().any():
        raise RuntimeError(f"CSV de Lira ({tarefa}) com caminho repetido")
    extra = pd.DataFrame({"rel_path": rel}).merge(ctx.dobras, on="rel_path", how="left", validate="one_to_one")
    if extra["case_id"].isna().any():
        raise RuntimeError(f"Imagens do CSV de Lira ({tarefa}) fora de folds_entrada_saida.csv")
    df["rel_path"] = rel.to_numpy()
    df["caso_real"] = extra["case_id"].to_numpy()
    df["dobra"] = extra["fold"].astype(int).to_numpy()
    df["path"] = [str(ctx.raiz_dataset / r) for r in rel]
    df.attrs["classes"] = {str(c): int(i) for i, c in enumerate(lb.classes_)}
    df.attrs["alvo"] = alvo
    ctx._csv[tarefa] = df
    return df


COLUNAS_ORIGINAIS = ["path", "case_year", "case_number", "sex", "age", "caliber", "position", "shot_type",
                     "accommodation", "trajectory", "distance", "scale", "num_injury", "num_photography",
                     "crop_code"]


def particao(ctx: Contexto, tarefa: str, nome: str):
    """(df_treino, df_teste) na ordem do CSV, como ``df.iloc[indices]`` do leguwoi."""
    df = csv_lira(ctx, tarefa)
    alvo = df.attrs["alvo"]
    if nome.startswith("vazado_p"):
        p = int(nome.removeprefix("vazado_p"))
        # o leguwoi passa o DataFrame lido do CSV: mesmas colunas e mesma ordem
        orig = df[[c for c in COLUNAS_ORIGINAIS if c in df.columns]]
        estado = np.random.get_state()
        np.random.seed(p)
        try:
            tr, te = ctx.lw["iterative_split"](orig, test_size=0.2, stratify_columns=["case_number", alvo])
        finally:
            np.random.set_state(estado)
        return df.loc[tr.index], df.loc[te.index]
    if nome.startswith("agrupado_d"):
        k = int(nome.removeprefix("agrupado_d"))
        te = df["dobra"] == k
        return df[~te], df[te]
    raise ValueError(f"Partição desconhecida: {nome}")


def conferir_com_holdout_do_drive(ctx: Contexto, caminho_holdout, particoes=(42,)) -> dict:
    """Confere que ``particao(vazado_p*)`` é a mesma de dados/holdout_lira2024.csv (notebook 04b)."""
    h = pd.read_csv(caminho_holdout).set_index("rel_path")
    out = {"holdout_sha256": pc.sha256_arquivo(caminho_holdout)}
    for tarefa in TAREFAS:
        for p in particoes:
            tr, te = particao(ctx, tarefa, f"vazado_p{p}")
            col = h[f"vaz_{tarefa}_p{p}"]
            ok = bool((col.loc[tr["rel_path"]] == 1).all() and (col.loc[te["rel_path"]] == 0).all()
                      and int((col >= 0).sum()) == len(tr) + len(te))
            out[f"{tarefa}_p{p}"] = ok
            if not ok:
                raise RuntimeError(f"Partição vazado_p{p} ({tarefa}) difere de {caminho_holdout}")
    return out


def resumo_particao(ctx: Contexto, tarefa: str, nome: str) -> dict:
    tr, te = particao(ctx, tarefa, nome)
    alvo = csv_lira(ctx, tarefa).attrs["alvo"]
    casos_tr = set(tr["caso_real"])
    vaza = te["caso_real"].isin(casos_tr)
    return {
        "particao": nome, "tarefa": tarefa, "n_treino": len(tr), "n_teste": len(te),
        "frac_teste": len(te) / (len(tr) + len(te)),
        "frac_teste_com_caso_no_treino": float(vaza.mean()),
        "casos_teste": int(te["caso_real"].nunique()), "casos_em_ambos": int(te.loc[vaza, "caso_real"].nunique()),
        "classes_treino": tr[alvo].value_counts().sort_index().to_dict(),
        "classes_teste": te[alvo].value_counts().sort_index().to_dict(),
    }


# ---------------------------------------------------------------------------
# Dados em memória (como GunshotsDataset._load_data) e reamostragem
# ---------------------------------------------------------------------------
def _amostra(n: int, limite) -> np.ndarray:
    """Índices igualmente espaçados (só para testes rápidos com ``limite``)."""
    return np.arange(n) if limite is None or limite >= n else np.linspace(0, n - 1, int(limite)).astype(int)


def _carregar(df: pd.DataFrame, transforms, limite=None) -> np.ndarray:
    """Igual a GunshotsDataset._load_single_row: PIL -> np.array -> transforms, uma vez por imagem."""
    from PIL import Image

    caminhos = list(df["path"])
    if limite is not None:
        caminhos = [caminhos[i] for i in _amostra(len(caminhos), limite)]
    out = np.empty((len(caminhos), 3, 224, 224), dtype=np.float32)
    for i, c in enumerate(caminhos):
        img = np.array(Image.open(c))
        t = transforms(image=img)["image"]
        out[i] = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    return out


def arrays(ctx: Contexto, tarefa: str, nome_particao: str, reamostragem=None, limite=None):
    """X_treino, y_treino, X_teste, y_teste (float32 NCHW), com cache em disco local.

    A semente 0x40 é fixada antes das aumentações, então o conjunto de treino
    aumentado é o mesmo para todos os modelos da mesma partição/tarefa.
    """
    chave = (tarefa, nome_particao, reamostragem, limite)
    if chave in ctx._arrays:
        return ctx._arrays[chave]
    alvo = csv_lira(ctx, tarefa).attrs["alvo"]
    tr, te = particao(ctx, tarefa, nome_particao)
    cfg = ctx.lw["DatasetConfig"]
    base = None
    if ctx.dir_cache is not None and limite is None:
        ctx.dir_cache.mkdir(parents=True, exist_ok=True)
        base = ctx.dir_cache / f"{tarefa}__{nome_particao}"
    if base is not None and Path(f"{base}__Xtr.npy").exists() and Path(f"{base}__Xte.npy").exists():
        Xtr, Xte = np.load(f"{base}__Xtr.npy"), np.load(f"{base}__Xte.npy")
    else:
        semear(SEMENTE)
        Xtr = _carregar(tr, cfg.TRAIN_TRANSFORMS, limite)
        Xte = _carregar(te, cfg.TEST_TRANSFORMS, limite)
        if base is not None:
            np.save(f"{base}__Xtr.npy", Xtr)
            np.save(f"{base}__Xte.npy", Xte)
    ytr = tr[alvo].to_numpy()[_amostra(len(tr), limite)].astype(np.int64)
    yte = te[alvo].to_numpy()[_amostra(len(te), limite)].astype(np.int64)
    info = {"n_treino_original": int(len(ytr)), "reamostragem": reamostragem}
    if reamostragem:
        arq = Path(f"{base}__{reamostragem}.npz") if base is not None else None
        if arq is not None and arq.exists():
            z = np.load(arq)
            Xtr, ytr = z["X"], z["y"]
        else:
            # como em get_gunshow_dataset: achata, reamostra com random_state=SEED, desfaz
            from data.common import get_resampler  # leguwoi

            n, c, h, w = Xtr.shape
            res = get_resampler(reamostragem)(random_state=cfg.SEED)
            Xr, ytr = res.fit_resample(Xtr.reshape(n, -1), ytr)
            Xtr = np.ascontiguousarray(Xr.reshape((Xr.shape[0], c, h, w)), dtype=np.float32)
            ytr = np.asarray(ytr).astype(np.int64)
            if arq is not None:
                np.savez(arq, X=Xtr, y=ytr)
    info["n_treino"] = int(len(ytr))
    info["classes_treino"] = {int(k): int(v) for k, v in zip(*np.unique(ytr, return_counts=True))}
    info["classes_teste"] = {int(k): int(v) for k, v in zip(*np.unique(yte, return_counts=True))}
    ctx._arrays = {chave: (Xtr, ytr, Xte, yte, info)}  # guarda só o último (memória)
    return ctx._arrays[chave]


# ---------------------------------------------------------------------------
# Métricas das tabelas (16 colunas)
# ---------------------------------------------------------------------------
def metricas_tabela(y, y_pred, n_classes: int) -> dict:
    """As 16 colunas das Tabelas 4 e 5, calculadas das CLASSES previstas.

    Definições conferidas contra as linhas publicadas (testes em ``testar_metricas``):
    precisão/revocação/F1 do scikit-learn (macro, micro, ponderada; 0 quando indefinida);
    AUC de classes = roc_auc_score sobre rótulos previstos (um-contra-resto em 3 classes);
    especificidade por classe = VN/(VN+FP), com média macro, micro (agregada) e ponderada
    pelo suporte. Na Tabela 4 (binária) a coluna de especificidade ponderada publicada é
    igual à acurácia, o que nenhuma média ponderada padrão reproduz; ``espec_W`` segue o
    artigo nesse caso e ``espec_W_padrao`` guarda a definição padrão.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    y = np.asarray(y).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rot = list(range(n_classes))
    m = {"acc": float(accuracy_score(y, y_pred))}
    for suf, media in (("M", "macro"), ("m", "micro"), ("W", "weighted")):
        p, r, f, _ = precision_recall_fscore_support(y, y_pred, labels=rot, average=media, zero_division=0)
        m[f"prec_{suf}"], m[f"rec_{suf}"], m[f"f1_{suf}"] = float(p), float(r), float(f)
    cm = confusion_matrix(y, y_pred, labels=rot).astype(float)
    vp = np.diag(cm)
    fn = cm.sum(1) - vp
    fp = cm.sum(0) - vp
    vn = cm.sum() - vp - fn - fp
    sup = cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec_c = np.where(sup > 0, vp / (vp + fn), np.nan)
        esp_c = np.where(vn + fp > 0, vn / (vn + fp), np.nan)
    auc_c = (rec_c + esp_c) / 2  # AUC com um único limiar, por classe
    peso = sup / sup.sum()
    presentes = sup > 0
    esp_micro = float(vn.sum() / (vn.sum() + fp.sum()))
    if n_classes == 2:
        auc = float(np.nanmean(auc_c))  # = roc_auc_score(y, y_pred) = revocação macro
        m.update({"auc_M": auc, "auc_m": auc, "auc_W": auc})
        m["espec_W_padrao"] = float(np.nansum(peso * esp_c))
        m.update({"espec_M": float(np.nanmean(esp_c)), "espec_m": esp_micro, "espec_W": m["acc"]})
    else:
        m["auc_M"] = float(np.mean(auc_c[presentes]))
        m["auc_m"] = float((vp.sum() / (vp.sum() + fn.sum()) + esp_micro) / 2)
        m["auc_W"] = float(np.sum(peso[presentes] * auc_c[presentes]))
        m["espec_M"] = float(np.nanmean(esp_c))
        m["espec_m"] = esp_micro
        m["espec_W"] = float(np.nansum(peso * esp_c))
        m["espec_W_padrao"] = m["espec_W"]
    return m


def testar_metricas(tol: float = 0.0015) -> list[str]:
    """Confere ``metricas_tabela`` contra linhas publicadas que fixam as definições."""
    casos = []
    # Tabela 5, EfficientNet B6 / SqueezeNet 1.1: tudo na classe majoritária (0.886 / 0.095 / 0.019)
    y = np.array([0] * 34 + [1] * 318 + [2] * 7)
    casos.append(("T5 só majoritária", y, np.full_like(y, 1), 3, dict(
        acc=.886, prec_M=.295, prec_m=.886, prec_W=.785, rec_M=.333, rec_m=.886, rec_W=.886, f1_M=.313,
        f1_m=.886, f1_W=.832, auc_M=.500, auc_m=.914, auc_W=.500, espec_M=.667, espec_m=.943, espec_W=.114)))
    # Tabela 5, SqueezeNet 1.0: tudo na classe de 9,5 %
    casos.append(("T5 só curta", y, np.full_like(y, 0), 3, dict(
        acc=.095, prec_M=.032, prec_W=.009, rec_M=.333, f1_M=.058, f1_W=.016,
        auc_M=.500, auc_m=.321, auc_W=.500, espec_M=.667, espec_m=.547, espec_W=.905)))
    # Tabela 4, EfficientNet B7 / SqueezeNet: tudo na classe majoritária (0.750)
    y2 = np.array([0] * 358 + [1] * 119)
    casos.append(("T4 só majoritária", y2, np.zeros_like(y2), 2, dict(
        acc=.750, prec_M=.375, prec_m=.750, prec_W=.563, rec_M=.500, rec_m=.750, rec_W=.750, f1_M=.429,
        f1_m=.750, f1_W=.644, auc_M=.500, auc_m=.500, auc_W=.500, espec_M=.500, espec_m=.750, espec_W=.750)))
    erros = []
    for nome, yv, yp, k, esperado in casos:
        m = metricas_tabela(yv, yp, k)
        for c, v in esperado.items():
            if abs(m[c] - v) > tol:
                erros.append(f"{nome}: {c} = {m[c]:.4f}, publicado {v:.3f}")
    return erros


# ---------------------------------------------------------------------------
# Treino (espelha fixed_train_test_validation_split)
# ---------------------------------------------------------------------------
def _scheduler(receita: str):
    from skorch.callbacks import LRScheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

    if receita == "A":  # optimization_configs.py da tag V_0.1
        return LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=10, T_mult=2, eta_min=1e-7)
    return LRScheduler(policy=StepLR, step_size=50, gamma=0.1)  # master


def _classe_rede(precisao: str):
    from skorch import NeuralNetClassifier

    if precisao == "fp32":
        return NeuralNetClassifier
    if precisao != "amp":
        raise ValueError(precisao)
    from skorch.dataset import unpack_data

    class RedeAMP(NeuralNetClassifier):
        """Mesmo laço do skorch, com autocast fp16 e GradScaler (só muda a aritmética)."""

        def initialize(self):
            super().initialize()
            self.escalonador_ = torch.cuda.amp.GradScaler()
            return self

        def infer(self, x, **fit_params):
            with torch.autocast("cuda", dtype=torch.float16):
                out = super().infer(x, **fit_params)
            return out.float()

        def train_step(self, batch, **fit_params):
            self._set_training(True)
            Xi, yi = unpack_data(batch)
            self._zero_grad_optimizer()
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            self.escalonador_.scale(loss).backward()
            self.escalonador_.step(self.optimizer_)
            self.escalonador_.update()
            return {"loss": loss, "y_pred": y_pred}

    return RedeAMP


class InterrupcaoSimulada(RuntimeError):
    pass


def _callbacks_de_controle(X_teste, dir_ckpt, dir_unidade, ckpt_min, instantaneos, parar_apos_epoca, estado):
    from skorch.callbacks import Callback

    class Controle(Callback):
        """Retomada após queda, batimento da trava e predições de teste em épocas escolhidas."""

        def on_train_begin(self, net, **kw):
            self.ultimo_ = time.time()
            if estado.get("scheduler") is not None:  # retomada: repõe o estado exato do scheduler
                sch = dict(net.callbacks_)["lr"].lr_scheduler_
                sch.load_state_dict(estado["scheduler"])
                for g, lr in zip(net.optimizer_.param_groups, sch.get_last_lr()):
                    g["lr"] = lr
                if estado.get("escalonador") is not None and hasattr(net, "escalonador_"):
                    net.escalonador_.load_state_dict(estado["escalonador"])

        def on_epoch_end(self, net, **kw):
            ep = len(net.history)
            if ep in instantaneos:
                estado["instantaneos"][str(ep)] = net.predict(X_teste).astype(int).tolist()
            agora = time.time()
            parar = parar_apos_epoca is not None and ep == parar_apos_epoca
            if dir_ckpt is not None and (parar or (agora - self.ultimo_) / 60 >= ckpt_min):
                self.gravar(net)
                self.ultimo_ = time.time()
                if dir_unidade is not None:
                    pc.atualizar_trava(dir_unidade)
            if parar:
                raise InterrupcaoSimulada(f"parada simulada após a época {ep}")

        def gravar(self, net):
            tmp = Path(str(dir_ckpt) + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            net.save_params(f_params=str(tmp / "params.pt"), f_optimizer=str(tmp / "optimizer.pt"),
                            f_history=str(tmp / "history.json"))
            torch.save({
                "scheduler": dict(net.callbacks_)["lr"].lr_scheduler_.state_dict(),
                "escalonador": net.escalonador_.state_dict() if hasattr(net, "escalonador_") else None,
                "instantaneos": estado["instantaneos"], "tempo_treino_s": estado["tempo_anterior_s"]
                + (time.time() - estado["inicio"]), "lote": estado["lote"],
            }, tmp / "extra.pt")
            shutil.rmtree(dir_ckpt, ignore_errors=True)
            os.replace(tmp, dir_ckpt)

    return Controle()


def caminho_unidade(raiz_resultados, receita, particao_nome, tarefa, modelo) -> Path:
    return Path(raiz_resultados) / receita / particao_nome / tarefa / modelo


def treinar_unidade(
    ctx: Contexto, modelo: str, tarefa: str, particao_nome: str, receita: str,
    raiz_resultados, raiz_checkpoints=None, precisao: str = "fp32", epocas: int | None = None,
    lote: int | None = None, instantaneos=(), ckpt_min: float = 20.0, limite_imagens=None,
    parar_apos_epoca=None, dispositivo=None, rotulo_extra: str = "",
) -> dict:
    """Treina e avalia uma unidade; devolve (e grava) o resultado. Retoma de checkpoint se houver."""
    import skorch
    from sklearn.utils.class_weight import compute_class_weight

    cfg_r = RECEITAS[receita]
    opt = ctx.lw["OptimizerConfig"]
    epocas = int(epocas or cfg_r["epocas"])
    dir_u = caminho_unidade(raiz_resultados, receita + rotulo_extra, particao_nome, tarefa, modelo)
    arq = dir_u / "resultado.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    dir_ck = None
    if raiz_checkpoints is not None:
        dir_ck = caminho_unidade(raiz_checkpoints, receita + rotulo_extra, particao_nome, tarefa, modelo)

    Xtr, ytr, Xte, yte, info = arrays(ctx, tarefa, particao_nome, cfg_r["reamostragem"], limite_imagens)
    # --- daqui em diante, na ordem de fixed_train_test_validation_split ---
    labels = len(np.unique(np.unique(ytr).tolist() + np.unique(yte).tolist()))
    if cfg_r["pesos_classe"] == "balanced":  # V_0.1
        class_weights = torch.tensor(
            compute_class_weight(class_weight="balanced", classes=np.unique(ytr), y=ytr), dtype=torch.float)
    else:  # master: "Criando um tensor com todos os pesos iguais a 1"
        class_weights = torch.ones(labels, dtype=torch.float)

    dispositivo = dispositivo or opt.DEVICE
    estado = {"instantaneos": {}, "scheduler": None, "escalonador": None, "tempo_anterior_s": 0.0,
              "inicio": time.time(), "lote": int(lote or lote_do_artigo(modelo))}
    retomada = dir_ck is not None and (dir_ck / "extra.pt").exists()
    if retomada:
        extra = torch.load(dir_ck / "extra.pt", map_location="cpu", weights_only=False)  # arquivo deste módulo
        estado.update({k: extra[k] for k in ("instantaneos", "scheduler", "escalonador", "lote")})
        estado["tempo_anterior_s"] = float(extra["tempo_treino_s"])

    def construir(tam_lote):
        semear(SEMENTE)  # o original não fixa semente dos pesos iniciais; aqui 0x40
        return _classe_rede(precisao)(
            module=getattr(ctx.lw["models"], modelo),
            module__in_channels=3,
            module__output_features=labels,
            lr=opt.LR,
            criterion=opt.CRITERION,
            criterion__weight=class_weights,
            batch_size=tam_lote,
            max_epochs=epocas,
            optimizer=opt.OPTIMIZER,
            optimizer__momentum=opt.MOMENTUM,
            callbacks=[("lr", _scheduler(receita)),
                       ("controle", _callbacks_de_controle(Xte, dir_ck, dir_u if raiz_checkpoints else None,
                                                           ckpt_min, set(instantaneos), parar_apos_epoca,
                                                           estado))],
            train_split=skorch.dataset.ValidSplit(cv=5, stratified=True),
            device=dispositivo,
            verbose=0,
        )

    desvios = []
    while True:
        net = construir(estado["lote"])
        try:
            if retomada:
                net.initialize()
                net.load_params(f_params=str(dir_ck / "params.pt"), f_optimizer=str(dir_ck / "optimizer.pt"),
                                f_history=str(dir_ck / "history.json"))
                feitas = len(net.history)
                if feitas < epocas:
                    net.partial_fit(Xtr, y=ytr, epochs=epocas - feitas)
            else:
                net.fit(Xtr, y=ytr)
            break
        except torch.cuda.OutOfMemoryError:
            # o lote do artigo foi usado numa RTX 3090 (24 GB); se não couber, metade e registra
            del net
            torch.cuda.empty_cache()
            if estado["lote"] <= 8 or retomada:
                raise
            desvios.append(f"lote {estado['lote']} não coube na GPU; refeito com {estado['lote'] // 2}")
            estado["lote"] //= 2
            estado["instantaneos"].clear()
            estado["inicio"] = time.time()

    y_pred = net.predict(Xte)
    proba = net.predict_proba(Xte)
    tempo = estado["tempo_anterior_s"] + (time.time() - estado["inicio"])
    hist = net.history
    res = {
        "modelo": modelo, "tarefa": tarefa, "particao": particao_nome, "receita": receita + rotulo_extra,
        "receita_def": cfg_r, "epocas": epocas, "epocas_feitas": len(hist), "lote": estado["lote"],
        "lote_do_artigo": lote_do_artigo(modelo), "precisao": precisao, "semente": SEMENTE,
        "n_classes": labels, "classes": csv_lira(ctx, tarefa).attrs["classes"],
        "pesos_classe": class_weights.tolist(), "dados": info, "desvios": desvios, "retomada": bool(retomada),
        "metricas": metricas_tabela(yte, y_pred, labels),
        "y_teste": np.asarray(yte).astype(int).tolist(), "y_pred": np.asarray(y_pred).astype(int).tolist(),
        "proba": np.round(proba.astype(np.float64), 5).tolist(),
        "instantaneos": estado["instantaneos"],
        "historico": {"train_loss": [float(v) for v in hist[:, "train_loss"]],
                      "valid_loss": [float(v) for v in hist[:, "valid_loss"]],
                      "valid_acc": [float(v) for v in hist[:, "valid_acc"]]},
        "treino_min": tempo / 60, "limite_imagens": limite_imagens,
        "teste_rapido": bool(limite_imagens or epocas != cfg_r["epocas"]),
        "n_parametros": int(sum(p.numel() for p in net.module_.parameters())),
        "ambiente": ambiente(), "contexto": ctx.meta, "gravado_em": pc.agora_iso(),
    }
    pc.gravar_json_atomico(res, arq)
    if dir_ck is not None:
        shutil.rmtree(dir_ck, ignore_errors=True)
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def ambiente() -> dict:
    import albumentations
    import imblearn
    import skmultilearn
    import skorch
    import sklearn
    import torchvision

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
            "skorch": skorch.__version__, "albumentations": albumentations.__version__,
            "imblearn": imblearn.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "skmultilearn": getattr(skmultilearn, "__version__", "0.2.0"), "gpu": gpu}


# ---------------------------------------------------------------------------
# Fila
# ---------------------------------------------------------------------------
def montar_fila(receita: str, modelos=None, tarefas=("es", "dist"),
                particoes=("vazado_p42", "agrupado_d0")) -> list[dict]:
    """12 modelos da dissertação primeiro, depois os outros 47; por tarefa/partição agrupadas
    (assim os arrays em cache servem a vários modelos seguidos)."""
    modelos = list(modelos) if modelos is not None else DOZE + [m for m in MODELOS if m not in DOZE]
    blocos = [[m for m in modelos if m in DOZE], [m for m in modelos if m not in DOZE]]
    fila = []
    for bloco in blocos:
        for tarefa in tarefas:
            for part in particoes:
                for m in bloco:
                    fila.append({"receita": receita, "modelo": m, "tarefa": tarefa, "particao": part})
    return fila


def executar_fila(ctx, fila, raiz_resultados, raiz_checkpoints, precisao="fp32", **kw) -> list[dict]:
    feitos = []
    for i, u in enumerate(fila, 1):
        dir_u = caminho_unidade(raiz_resultados, u["receita"], u["particao"], u["tarefa"], u["modelo"])
        rot = f"[{i}/{len(fila)}] {u['receita']} {u['particao']} {u['tarefa']} {u['modelo']}"
        if (dir_u / "resultado.json").exists():
            continue
        dir_u.mkdir(parents=True, exist_ok=True)
        if not pc.adquirir_trava(dir_u, espera_s=5.0, expira_min=90.0):
            print(rot, "— em uso por outra aba, pulando")
            continue
        try:
            r = treinar_unidade(ctx, u["modelo"], u["tarefa"], u["particao"], u["receita"],
                                raiz_resultados, raiz_checkpoints, precisao=precisao, **kw)
            m = r["metricas"]
            print(rot, f"— acc {m['acc']:.3f} F1 {m['f1_M']:.3f} AUCc {m['auc_M']:.3f} | {r['treino_min']:.1f} min")
            feitos.append(u)
        except Exception as e:  # registra e segue: uma arquitetura não derruba a fila
            pc.gravar_json_atomico({**u, "erro": f"{type(e).__name__}: {e}", "quando": pc.agora_iso()},
                                   dir_u / f"erro_{int(time.time())}.json")
            print(rot, "— ERRO:", type(e).__name__, str(e)[:200])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            pc.liberar_trava(dir_u)
    return feitos


def ler_resultados(raiz_resultados, receita=None) -> pd.DataFrame:
    """Uma linha por resultado.json (métricas das tabelas + identificação)."""
    linhas = []
    for arq in sorted(Path(raiz_resultados).glob("*/*/*/*/resultado.json")):
        r = json.loads(arq.read_text(encoding="utf-8"))
        if receita is not None and r["receita"] != receita:
            continue
        linhas.append({
            "receita": r["receita"], "particao": r["particao"], "tarefa": r["tarefa"], "modelo": r["modelo"],
            **r["metricas"], "epocas": r["epocas_feitas"], "lote": r["lote"], "precisao": r["precisao"],
            "treino_min": r["treino_min"], "teste_rapido": r["teste_rapido"],
            "n_treino": r["dados"]["n_treino"], "n_teste": len(r["y_teste"]), "desvios": "; ".join(r["desvios"]),
        })
    return pd.DataFrame(linhas)
