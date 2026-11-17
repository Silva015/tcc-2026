import json, glob, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
REPO = Path(sys.argv[1]); TR = sys.argv[2]
sys.path.insert(0, str(REPO/"notebooks"))
import reproducao_lira as rl
un = {}
for f in glob.glob(TR + "/mcp-092684b8-*-blob-17948798*.json"):
    r = json.load(open(f))
    if r.get("receita") != "B_permutada": continue
    k = (r["particao"], r["tarefa"], r["modelo"]); assert k not in un, k; un[k] = r
print("unidades:", len(un), "| épocas:", {r["epocas_feitas"] for r in un.values()}, "| precisão:", {r["precisao"] for r in un.values()},
      "| desvios:", {d for r in un.values() for d in r["desvios"]}, "| retomadas:", sum(r["retomada"] for r in un.values()),
      "| horas:", round(sum(r["treino_min"] for r in un.values())/60, 1), "| módulo:", {r["contexto"]["versao_modulo"] for r in un.values()})
linhas = []
for (p, t, m), r in sorted(un.items()):
    y, P = np.array(r["y_teste"]), np.array(r["proba"])
    # confere as métricas gravadas recalculando das predições
    mm = rl.metricas_tabela(y, r["y_pred"], r["n_classes"])
    assert all(abs(mm[c] - r["metricas"][c]) < 1e-9 for c in rl.METRICAS), (p, t, m)
    try: aucp = roc_auc_score(y, P[:, 1]) if r["n_classes"] == 2 else roc_auc_score(y, P, multi_class="ovr", average="macro")
    except Exception as e: aucp = float("nan")
    linhas.append({"particao": p, "tarefa": t, "modelo_leguwoi": m, **{c: r["metricas"][c] for c in rl.METRICAS}, "auc_prob_M": aucp,
                   "n_treino": r["dados"]["n_treino"], "n_teste": len(y), "lote": r["lote"], "treino_min": r["treino_min"],
                   "uma_classe_so": len(set(r["y_pred"])) == 1, "classes_teste": json.dumps(r["dados"]["classes_teste"])})
res = pd.DataFrame(linhas)
SA = REPO/"resultados"/"reproducao_lira"; SA.mkdir(exist_ok=True)
res.round(6).to_csv(SA/"unidades.csv", index=False)
PUB = pd.read_csv(REPO/"resultados"/"lira2024_tabelas_completas.csv")
ordem = [m for m in rl.MODELOS if m in set(res["modelo_leguwoi"])]
pd.set_option("display.width", 250)
resumo = {"receita": "B_permutada", "precisao": "amp", "epocas": 300, "modelos": ordem, "por_tarefa": {}}
for t, n in (("es", 4), ("dist", 5)):
    pub = PUB[(PUB.tarefa == t) & PUB.modelo_leguwoi.isin(ordem)].set_index("modelo_leguwoi").loc[ordem]
    tab = {p: res[(res.tarefa == t) & (res.particao == p)].set_index("modelo_leguwoi").loc[ordem] for p in ("vazado_p42", "agrupado_d0")}
    vaz, agr = tab["vazado_p42"], tab["agrupado_d0"]
    for nome, d in ((f"tabela{n}_reproduzida", vaz), (f"tabela{n}_sem_vazamento", agr)):
        o = d[rl.METRICAS + ["auc_prob_M"]].copy(); o.insert(0, "rotulo_artigo", pub["rotulo_artigo"]); o.round(4).to_csv(SA/f"{nome}.csv")
    dif = (vaz[rl.METRICAS + ["auc_prob_M"]] - agr[rl.METRICAS + ["auc_prob_M"]]); dif.insert(0, "rotulo_artigo", pub["rotulo_artigo"]); dif.round(4).to_csv(SA/f"tabela{n}_diferenca.csv")
    q = pd.DataFrame({"pub_acc": pub.acc, "vaz_acc": vaz.acc, "agr_acc": agr.acc, "pub_f1": pub.f1_M, "vaz_f1": vaz.f1_M, "agr_f1": agr.f1_M,
                      "pub_aucc": pub.auc_M, "vaz_aucc": vaz.auc_M, "agr_aucc": agr.auc_M, "vaz_aucp": vaz.auc_prob_M, "agr_aucp": agr.auc_prob_M,
                      "1cls_v": vaz.uma_classe_so, "1cls_a": agr.uma_classe_so})
    print(f"\n=== Tabela {n} ({t}) | teste vazado {vaz.classes_teste.iloc[0]} n={vaz.n_teste.iloc[0]} | agrupado {agr.classes_teste.iloc[0]} n={agr.n_teste.iloc[0]}")
    print(q.round(3).to_string())
    aprende = ~(vaz.uma_classe_so | agr.uma_classe_so)
    pre = [m for m in ordem if m.startswith("Resnet")]
    resumo["por_tarefa"][t] = {
        "erro_abs_medio_vs_publicado": {c: float((vaz[c] - pub[c]).abs().mean()) for c in ("acc", "f1_M", "auc_M")},
        "spearman_acc": float(vaz.acc.corr(pub.acc, method="spearman")), "spearman_f1_M": float(vaz.f1_M.corr(pub.f1_M, method="spearman")),
        "media_vazado": {c: float(vaz[c].mean()) for c in ("acc", "f1_M", "auc_M", "auc_prob_M")},
        "media_agrupado": {c: float(agr[c].mean()) for c in ("acc", "f1_M", "auc_M", "auc_prob_M")},
        "queda_media": {c: float((vaz[c] - agr[c]).mean()) for c in ("acc", "f1_M", "auc_M", "auc_prob_M")},
        "queda_media_resnets": {c: float((vaz.loc[pre, c] - agr.loc[pre, c]).mean()) for c in ("acc", "f1_M", "auc_M", "auc_prob_M")},
        "modelos_melhores_com_vazamento": {c: int(((vaz[c] - agr[c]) > 0).sum()) for c in ("acc", "f1_M", "auc_M", "auc_prob_M")},
        "uma_classe_so": {"vazado": int(vaz.uma_classe_so.sum()), "agrupado": int(agr.uma_classe_so.sum())},
        "classes_teste": {"vazado": json.loads(vaz.classes_teste.iloc[0]), "agrupado": json.loads(agr.classes_teste.iloc[0])},
    }
    print(json.dumps(resumo["por_tarefa"][t], indent=1))
resumo["horas_gpu"] = float(res.treino_min.sum()/60)
json.dump(resumo, open(SA/"resumo_reproducao.json", "w"), indent=1, ensure_ascii=False)
