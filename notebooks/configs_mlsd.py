"""Plugin do P1: distância do disparo (MLSD) e hold-out de Lira et al. no protocolo único.

Carregue com ``pc.carregar_plugin(caminho)`` (o SHA-256 deste arquivo vai para cada
``metricas.json``). Não altera nada do ``protocolo_comum.py``: só registra configs e
oferece funções de dados.

* MLSD = 3 classes (encostado, curta, distância), **só nas entradas**, mesmas dobras
  de ``folds_entrada_saida.csv``. Coluna de dobra da unidade k: ``fold_dist_d{k}``
  (treino = entradas das outras dobras; teste = todas as imagens da dobra k — as
  saídas só recebem predição, para a métrica de pipeline do P2, e não entram nas
  métricas, que ignoram ``y_dist = -1``).
* Hold-out de Lira et al. (2025, IJLM): reprodução do ``iterative_split`` de
  ``leguwoi/src/data/gunshot_dataset.py`` sobre os CSVs do próprio dataset, para
  medir o efeito do vazamento de casos sob a mesma receita.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import protocolo_comum as pc

VERSAO = "1.0.0"

# ---------------------------------------------------------------------------
# Configs (prefixo mlsd__ = tarefa "dist" sozinha)
# ---------------------------------------------------------------------------
CABECAS = {"a": "linear", "b": "mlp128", "c": "mlp_iso", "d": "kan", "e": "kan_pequena"}
CONFIGS_CABECA = {}
for _k, _cab in CABECAS.items():
    CONFIGS_CABECA[_k] = f"mlsd__resnet152__{_cab}"
    pc.registrar_config(
        CONFIGS_CABECA[_k],
        {"backbone": "resnet152", "cabecas": {"dist": _cab},
         "descricao": f"MLSD: ResNet-152 + cabeça {_cab} ({_k})"},
    )

CONFIGS_BASELINES = {}
for _bb in pc.BASELINES:
    _sq = _bb == "squeezenet1_0"
    CONFIGS_BASELINES[_bb] = f"mlsd__{_bb}__{'conv1x1' if _sq else 'linear'}"
    pc.registrar_config(
        CONFIGS_BASELINES[_bb],
        {"backbone": _bb, "cabecas": {"dist": "conv1x1_squeezenet" if _sq else "linear"},
         "descricao": f"MLSD: baseline {_bb} pré-treinado"},
    )


# ---------------------------------------------------------------------------
# Dobras da MLSD
# ---------------------------------------------------------------------------
def coluna_dobra_dist(k: int) -> str:
    return f"fold_dist_d{int(k)}"


def adicionar_colunas_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta fold_dist_d0..d4: entradas mantêm a dobra; saídas só na dobra k (teste)."""
    df = df.copy()
    entrada = df["y_es"].to_numpy() == 0
    fold = df["fold"].to_numpy()
    for k in range(pc.N_DOBRAS):
        df[coluna_dobra_dist(k)] = np.where(entrada | (fold == k), fold, -1)
    return df


def checar_colunas_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dobra: tamanhos de treino/teste, classes no teste e casos de encostado."""
    linhas = []
    for k in range(pc.N_DOBRAS):
        c = df[coluna_dobra_dist(k)].to_numpy()
        te, tr = c == k, (c != k) & (c >= 0)
        te_ent = te & (df["y_dist"].to_numpy() >= 0)
        ydt = df.loc[te_ent, "y_dist"]
        linha = {
            "dobra": k,
            "n_treino": int(tr.sum()),
            "saidas_no_treino": int((tr & (df["y_es"].to_numpy() == 1)).sum()),
            "n_teste_total": int(te.sum()),
            "n_teste_entradas": int(te_ent.sum()),
            **{f"teste_{n}": int((ydt == i).sum()) for i, n in enumerate(pc.TAREFAS["dist"].nomes_classes)},
            "casos_encostado_teste": int(df.loc[te_ent & (df["y_dist"].to_numpy() == 0), "case_id"].nunique()),
            "casos_em_treino_e_teste": len(set(df.loc[tr, "case_id"]) & set(df.loc[te, "case_id"])),
        }
        linhas.append(linha)
    t = pd.DataFrame(linhas)
    erros = []
    if t["saidas_no_treino"].any():
        erros.append("há saídas no treino da MLSD")
    if t["casos_em_treino_e_teste"].any():
        erros.append("caso em treino e teste")
    if (t[[f"teste_{n}" for n in pc.TAREFAS["dist"].nomes_classes]] == 0).any().any():
        erros.append("dobra de teste sem alguma classe")
    if (t["casos_encostado_teste"] < 2).any():
        erros.append("dobra de teste com < 2 casos de encostado")
    if int(t["n_teste_entradas"].sum()) != int((df["y_dist"] >= 0).sum()):
        erros.append("as entradas não cobrem exatamente as 5 dobras de teste")
    if erros:
        raise RuntimeError("Dobras da MLSD inválidas: " + "; ".join(erros))
    return t


def montar_fila_dist(experimento: str, configs, sementes=(42,), dobras=range(pc.N_DOBRAS)):
    """Como pc.montar_fila, mas com a coluna de dobra própria de cada dobra."""
    return [
        {"experimento": experimento, "config": c, "semente": int(s), "dobra": int(d),
         "coluna_dobra": coluna_dobra_dist(d)}
        for c in configs
        for s in sementes
        for d in dobras
    ]


# ---------------------------------------------------------------------------
# Hold-out de Lira et al. (2025, IJLM) — reprodução do iterative_split
# ---------------------------------------------------------------------------
CSVS_LIRA = {
    # tarefa -> (CSV do dataset, coluna-alvo usada por Lira)
    "es": ("gunshots_eqx.csv", "shot_type"),
    "dist": ("gunshots_eqx_ONLY_INPUT_SHOTS.csv", "distance"),
}
PREFIXO_CAMINHO_LIRA = "/home/pedro/databases/tiros/"
SEMENTES_PARTICAO = (42, 43, 44)
VERSAO_SKMULTILEARN = "0.2.0"


def coluna_holdout(tarefa: str, particao: int) -> str:
    return f"vaz_{tarefa}_p{int(particao)}"


def _iterative_split_lira(df: pd.DataFrame, test_size: float, stratify_columns):
    """Cópia de leguwoi/src/data/gunshot_dataset.py::iterative_split (devolve índices)."""
    from skmultilearn.model_selection import IterativeStratification

    one_hot_cols = [pd.get_dummies(df[col]) for col in stratify_columns]
    one_hot_cols = pd.concat(one_hot_cols, axis=1).to_numpy()
    stratifier = IterativeStratification(
        n_splits=2, order=len(stratify_columns), sample_distribution_per_fold=[test_size, 1 - test_size]
    )
    train_indices, test_indices = next(stratifier.split(df.to_numpy(), one_hot_cols))
    return train_indices, test_indices


def particao_lira(raiz_dataset, tarefa: str, particao: int, test_size: float = 0.2) -> pd.DataFrame:
    """rel_path + papel (0 teste, 1 treino) de uma partição no estilo de Lira.

    Lira não fixou semente: o IterativeStratification desempata pelo RNG global do
    NumPy. Aqui a partição p usa np.random.seed(p) — reproduz o MÉTODO, não a
    partição exata do artigo. Leitura do CSV e codificação idênticas às do leguwoi.
    """
    from sklearn.preprocessing import LabelEncoder

    nome, alvo = CSVS_LIRA[tarefa]
    c = pd.read_csv(Path(raiz_dataset) / "csv_configs" / nome)
    rel = c["path"].str.replace(PREFIXO_CAMINHO_LIRA, "database/", regex=False)
    c[alvo] = LabelEncoder().fit_transform(c[alvo])
    estado = np.random.get_state()
    np.random.seed(int(particao))
    try:
        tr, te = _iterative_split_lira(c, test_size, ["case_number", alvo])
    finally:
        np.random.set_state(estado)
    papel = np.full(len(c), -1)
    papel[tr], papel[te] = 1, 0
    return pd.DataFrame({"rel_path": rel, "papel": papel})


def tabela_holdout_lira(raiz_dataset, df: pd.DataFrame, particoes=SEMENTES_PARTICAO) -> pd.DataFrame:
    """rel_path + vaz_{es,dist}_p{s} (0 teste, 1 treino, -1 fora do CSV de Lira), na ordem de df."""
    out = df[["rel_path"]].copy()
    for tarefa in CSVS_LIRA:
        for p in particoes:
            part = particao_lira(raiz_dataset, tarefa, p)
            if part["rel_path"].duplicated().any():
                raise RuntimeError(f"CSV de Lira ({tarefa}) com caminho repetido")
            faltam = set(part["rel_path"]) - set(df["rel_path"])
            if faltam:
                raise RuntimeError(f"{len(faltam)} imagens do CSV de Lira ({tarefa}) fora da listagem: {sorted(faltam)[:3]}")
            m = out[["rel_path"]].merge(part, on="rel_path", how="left", validate="one_to_one")
            out[coluna_holdout(tarefa, p)] = m["papel"].fillna(-1).astype(int).to_numpy()
    return out


def resumo_holdout(df: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Taxa de vazamento por partição: imagens de teste cujo caso (ano.caso) está no treino."""
    h = df[["rel_path", "case_id", "y_es", "y_dist"]].merge(holdout, on="rel_path", validate="one_to_one")
    linhas = []
    for col in [c for c in holdout.columns if c.startswith("vaz_")]:
        te, tr = h[col] == 0, h[col] == 1
        casos_tr = set(h.loc[tr, "case_id"])
        vaza = h.loc[te, "case_id"].isin(casos_tr)
        tarefa = col.split("_")[1]
        ycol = pc.TAREFAS[tarefa].coluna
        linhas.append({
            "particao": col, "n_treino": int(tr.sum()), "n_teste": int(te.sum()), "fora_do_csv": int((h[col] == -1).sum()),
            "frac_teste_com_caso_no_treino": float(vaza.mean()),
            "casos_teste": int(h.loc[te, "case_id"].nunique()),
            "casos_teste_so_no_teste": int(h.loc[te & ~vaza, "case_id"].nunique()),
            "classes_teste": h.loc[te, ycol].value_counts().sort_index().to_dict(),
            "rotulo_invalido_no_uso": int(((h[col] >= 0) & (h[ycol] < 0)).sum()),
        })
    return pd.DataFrame(linhas)


def sessao_holdout(sessao: pc.Sessao, holdout: pd.DataFrame, sha_holdout: str | None = None) -> pc.Sessao:
    """Sessão para o hold-out de Lira: mesmas imagens e cache, unidade de agrupamento = imagem.

    O protocolo recusa caso em treino e teste; o split de Lira trata cada imagem como
    independente, então aqui ``case_id`` vira o próprio ``rel_path`` (o caso real fica
    em ``case_id_real`` e é recuperado na análise pelo rel_path). Isto é deliberado e
    fica registrado em ``dados`` de cada ``metricas.json``.
    """
    df = sessao.df.merge(holdout, on="rel_path", how="left", validate="one_to_one")
    df["case_id_real"] = df["case_id"]
    df["case_id"] = df["rel_path"]
    meta = dict(sessao.meta)
    meta.update({
        "holdout_lira": True,
        "unidade_de_agrupamento": "imagem (rel_path): reprodução deliberada do split de Lira, com vazamento de casos",
        "holdout_lira_sha256": sha_holdout,
        "skmultilearn": VERSAO_SKMULTILEARN,
    })
    return pc.Sessao(df=df, cache=sessao.cache, raiz_dataset=sessao.raiz_dataset, meta=meta)


def fila_vazamento(configs_por_tarefa: dict, particoes=SEMENTES_PARTICAO, semente_treino: int = 42):
    """Uma unidade por (partição, tarefa, config): experimento vazamento_lira_p{s}, dobra 0 = teste."""
    return [
        {"experimento": f"vazamento_lira_p{p}", "config": cfg, "semente": int(semente_treino), "dobra": 0,
         "coluna_dobra": coluna_holdout(tarefa, p)}
        for p in particoes
        for tarefa, cfgs in configs_por_tarefa.items()
        for cfg in cfgs
    ]
