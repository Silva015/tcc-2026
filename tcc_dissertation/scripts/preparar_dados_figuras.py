#!/usr/bin/env python3
"""Prepara os dados das figuras da dissertação (pgfplots) a partir dos resultados do protocolo único.

Entradas:
  --drive DIR   pasta com os arquivos do P1 que ficaram só no Drive (tcc-final/resultados/):
                ablacao_runs.csv, ablacao_oof.csv  (de ablacao_cabeca/)
                mlsd_runs.csv,    mlsd_oof.csv     (de mlsd_cabeca/)
  resultados/   do repositório (só leitura).

Antes de gravar qualquer coisa, o script confere os arquivos do Drive contra resultados/:
  1. AUC de cada unidade (config x semente x dobra) recalculada do oof.csv = runs.csv;
  2. média ± dp entre dobras (cada dobra = média das sementes) = tabela_final.csv / tabela_mlsd.csv;
  3. Δ por dobra das cabeças = decisao.json / decisao_mlsd.json;
  4. AUC OOF (probabilidades médias das sementes) = tabela_wavkan*.csv;
  5. AUC agrupada da semente 42 e AUC no split de Lira = tabela_vazamento.csv.
Se algo não bater, para sem gravar.

Saídas (tcc_dissertation/img/protocolo/): CSVs pequenos para o pgfplots e figuras_macros.tex
com as AUC que aparecem nas legendas. Só usa a biblioteca padrão do Python.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(RAIZ, "resultados")
SAIDA = os.path.join(RAIZ, "tcc_dissertation", "img", "protocolo")

CABECAS = ["a", "b", "c", "d"]
CFG_ES = {"a": "resnet152__linear", "b": "resnet152__mlp128", "c": "resnet152__mlp_iso",
          "d": "resnet152__kan", "e": "resnet152__kan_pequena"}
CFG_DIST = {k: "mlsd__" + CFG_ES[k] for k in CABECAS}
CLASSES_DIST = {0: "enc", 1: "cur", 2: "dis"}  # encostado, curta, distância

falhas = []


def confere(cond, msg):
    print(("ok    " if cond else "FALHA ") + msg)
    if not cond:
        falhas.append(msg)


# ---------------------------------------------------------------- AUC e ROC
def auc_binaria(y, s):
    """AUC de Mann-Whitney com postos médios nos empates (igual ao roc_auc_score)."""
    pares = sorted(zip(s, y))
    n = len(pares)
    postos = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pares[j + 1][0] == pares[i][0]:
            j += 1
        medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            postos[k] = medio
        i = j + 1
    n1 = sum(1 for _, yy in pares if yy == 1)
    n0 = n - n1
    soma = sum(p for p, (_, yy) in zip(postos, pares) if yy == 1)
    return (soma - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ovr_macro(y, probs, classes=(0, 1, 2)):
    return statistics.fmean(auc_binaria([1 if v == c else 0 for v in y], [p[c] for p in probs]) for c in classes)


def curva_roc(y, s):
    """Vértices da ROC (um por limiar distinto), como o roc_curve."""
    pares = sorted(zip(s, y), key=lambda t: -t[0])
    p = sum(y)
    n = len(y) - p
    tp = fp = 0
    pts = [(0.0, 0.0)]
    i = 0
    while i < len(pares):
        lim = pares[i][0]
        while i < len(pares) and pares[i][0] == lim:
            if pares[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        pts.append((fp / n, tp / p))
    return pts


def simplificar(pts, eps=0.002):
    """Ramer-Douglas-Peucker: tira só pontos que ficam a menos de eps da reta (não muda o desenho)."""
    manter = [False] * len(pts)
    manter[0] = manter[-1] = True
    pilha = [(0, len(pts) - 1)]
    while pilha:
        a, b = pilha.pop()
        (x1, y1), (x2, y2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1
        norma = math.hypot(dx, dy) or 1.0
        idx, dmax = -1, 0.0
        for k in range(a + 1, b):
            d = abs(dy * (pts[k][0] - x1) - dx * (pts[k][1] - y1)) / norma
            if d > dmax:
                idx, dmax = k, d
        if dmax > eps:
            manter[idx] = True
            pilha += [(a, idx), (idx, b)]
    return [q for q, m in zip(pts, manter) if m]


# ---------------------------------------------------------------- leitura
def ler_csv(caminho):
    with open(caminho, newline="") as f:
        return list(csv.DictReader(f))


def media_dp(valores):
    return statistics.fmean(valores), statistics.stdev(valores)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def resumo_por_dobra(runs, config, campo, sementes=None):
    """{dobra: média das sementes} para uma config."""
    por = defaultdict(list)
    for r in runs:
        if r["config"] == config and (sementes is None or int(r["semente"]) in sementes):
            por[int(r["dobra"])].append(float(r[campo]))
    return {d: statistics.fmean(v) for d, v in sorted(por.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    runs_es = ler_csv(os.path.join(args.drive, "ablacao_runs.csv"))
    runs_dist = ler_csv(os.path.join(args.drive, "mlsd_runs.csv"))
    oof_es = ler_csv(os.path.join(args.drive, "ablacao_oof.csv"))
    oof_dist = [r for r in ler_csv(os.path.join(args.drive, "mlsd_oof.csv")) if int(r["y_dist"]) >= 0]

    # 1. AUC por unidade: oof.csv x runs.csv
    unid = defaultdict(list)
    for r in oof_es:
        unid[(r["config"], int(r["semente"]), int(r["dobra"]))].append(r)
    dif = 0.0
    for r in runs_es:
        linhas = unid[(r["config"], int(r["semente"]), int(r["dobra"]))]
        a = auc_binaria([int(x["y_es"]) for x in linhas], [float(x["p_es_1"]) for x in linhas])
        dif = max(dif, abs(a - float(r["es_auc"])))
    confere(dif < 1e-9, f"E/S: AUC por unidade do oof = runs.csv ({len(runs_es)} unidades, dif. máx. {dif:.1e})")

    unid = defaultdict(list)
    for r in oof_dist:
        unid[(r["config"], int(r["semente"]), int(r["dobra"]))].append(r)
    dif = 0.0
    for r in runs_dist:
        linhas = unid[(r["config"], int(r["semente"]), int(r["dobra"]))]
        a = auc_ovr_macro([int(x["y_dist"]) for x in linhas],
                          [{c: float(x[f"p_dist_{c}"]) for c in range(3)} for x in linhas])
        dif = max(dif, abs(a - float(r["dist_auc"])))
    confere(dif < 1e-9, f"MLSD: AUC OvR por unidade do oof = runs.csv ({len(runs_dist)} unidades, dif. máx. {dif:.1e})")

    # 2. média ± dp = tabelas
    tab_final = {r["config"]: r for r in ler_csv(os.path.join(RES, "tabela_final.csv")) if r["painel"] == "A"}
    for k, cfg in CFG_ES.items():
        m, s = media_dp(list(resumo_por_dobra(runs_es, cfg, "es_auc").values()))
        confere(fmt(m, s) == tab_final[cfg]["auc"], f"E/S ({k}) AUC {fmt(m, s)} = tabela_final {tab_final[cfg]['auc']}")
    tab_mlsd = {r["config"]: r for r in ler_csv(os.path.join(RES, "tabela_mlsd.csv")) if r["painel"] == "A"}
    for k, cfg in CFG_DIST.items():
        m, s = media_dp(list(resumo_por_dobra(runs_dist, cfg, "dist_auc").values()))
        confere(fmt(m, s) == tab_mlsd[cfg]["auc"], f"MLSD ({k}) AUC {fmt(m, s)} = tabela_mlsd {tab_mlsd[cfg]['auc']}")

    # 3. Δ por dobra = decisao*.json
    for runs, cfgs, arq, campo in [(runs_es, CFG_ES, "decisao.json", "es_auc"),
                                   (runs_dist, CFG_DIST, "decisao_mlsd.json", "dist_auc")]:
        dec = json.load(open(os.path.join(RES, arq)))
        kan = resumo_por_dobra(runs, cfgs["d"], campo)
        for k in ["a", "b", "c"]:
            outra = resumo_por_dobra(runs, cfgs[k], campo)
            calc = {str(d): round(kan[d] - outra[d], 4) for d in kan}
            esper = dec["comparacoes"][k]["delta_auc_dobras"]
            ok = all(abs(calc[d] - esper[d]) <= 1.01e-4 for d in esper)
            confere(ok, f"{arq}: Δ por dobra KAN − ({k}) {calc}")

    # 4. AUC OOF com probabilidades médias das sementes = tabela_wavkan*.csv
    def oof_medio(oof, cfg, campos):
        acc = defaultdict(list)
        y = {}
        for r in oof:
            if r["config"] == cfg:
                acc[r["rel_path"]].append([float(r[c]) for c in campos])
                y[r["rel_path"]] = r
        chaves = sorted(acc)
        return chaves, y, {c: [statistics.fmean(v[i] for v in acc[c]) for i in range(len(campos))] for c in chaves}

    tw = {r["id"]: r for r in ler_csv(os.path.join(RES, "tabela_wavkan.csv"))}
    twd = {r["id"]: r for r in ler_csv(os.path.join(RES, "tabela_wavkan_mlsd.csv"))}
    roc_es, roc_dist, auc_leg = {}, {}, {}
    for k in CABECAS:
        chaves, y, p = oof_medio(oof_es, CFG_ES[k], ["p_es_1"])
        yy = [int(y[c]["y_es"]) for c in chaves]
        ss = [p[c][0] for c in chaves]
        a = auc_binaria(yy, ss)
        confere(f"{a:.4f}" == tw[k]["auc_oof"], f"E/S ({k}) AUC OOF {a:.4f} = tabela_wavkan {tw[k]['auc_oof']}")
        roc_es[k] = simplificar(curva_roc(yy, ss))
        auc_leg[f"Es{k.upper()}"] = f"{a:.3f}"

        chaves, y, p = oof_medio(oof_dist, CFG_DIST[k], ["p_dist_0", "p_dist_1", "p_dist_2"])
        yy = [int(y[c]["y_dist"]) for c in chaves]
        a = auc_ovr_macro(yy, [{i: p[c][i] for i in range(3)} for c in chaves])
        confere(f"{a:.4f}" == twd[k]["auc_oof"], f"MLSD ({k}) AUC OOF {a:.4f} = tabela_wavkan_mlsd {twd[k]['auc_oof']}")
        for c, nome in CLASSES_DIST.items():
            yb = [1 if v == c else 0 for v in yy]
            sb = [p[ch][c] for ch in chaves]
            roc_dist[(nome, k)] = simplificar(curva_roc(yb, sb))
            auc_leg[f"Dist{nome.capitalize()}{k.upper()}"] = f"{auc_binaria(yb, sb):.3f}"

    # 5. vazamento: agrupado s42 e split de Lira = tabela_vazamento.csv
    tv = {(r["tarefa"], r["config"]): r for r in ler_csv(os.path.join(RES, "tabela_vazamento.csv"))}
    lira = ler_csv(os.path.join(RES, "vazamento_lira_runs.csv"))
    pontos_vaz = {}
    for tarefa, runs, cfgs, campo in [("es", runs_es, CFG_ES, "es_auc"), ("dist", runs_dist, CFG_DIST, "dist_auc")]:
        for k in ["a", "d"]:
            cfg = cfgs[k]
            agr = resumo_por_dobra(runs, cfg, campo, sementes={42})
            m, s = media_dp(list(agr.values()))
            confere(fmt(m, s) == tv[(tarefa, cfg)]["agrupado_auc"],
                    f"vazamento {tarefa} ({k}) agrupado s42 {fmt(m, s)} = {tv[(tarefa, cfg)]['agrupado_auc']}")
            lr = [float(r[campo]) for r in lira if r["config"] == cfg]
            m, s = media_dp(lr)
            confere(len(lr) == 3 and fmt(m, s) == tv[(tarefa, cfg)]["lira_auc"],
                    f"vazamento {tarefa} ({k}) split de Lira {fmt(m, s)} = {tv[(tarefa, cfg)]['lira_auc']}")
            pontos_vaz[(tarefa, k, "agrup")] = list(agr.values())
            pontos_vaz[(tarefa, k, "lira")] = lr

    # diagnóstico da grade da KAN (só descritivo; medido nas imagens de teste)
    diag = []
    for tarefa, runs, cfgs in [("es", runs_es, CFG_ES), ("dist", runs_dist, CFG_DIST)]:
        for k in (["d", "e"] if tarefa == "es" else ["d"]):
            for camada in ("kan0", "kan1"):
                v = [float(r[f"kan_{tarefa}.{camada}_frac_base"]) for r in runs
                     if r["config"] == cfgs[k] and r.get(f"kan_{tarefa}.{camada}_frac_base")]
                diag.append({"tarefa": tarefa, "cabeca": k, "camada": camada, "n_unidades": len(v),
                             "min": f"{min(v):.3f}", "media": f"{statistics.fmean(v):.3f}", "max": f"{max(v):.3f}"})

    if falhas:
        sys.exit(f"\n{len(falhas)} conferência(s) falharam; nada foi gravado.")

    # ---------------------------------------------------------------- gravação
    os.makedirs(SAIDA, exist_ok=True)

    def gravar(nome, cab, linhas):
        with open(os.path.join(SAIDA, nome), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cab)
            w.writerows(linhas)

    # vazamento: x = posição do modelo (a = 1, d = 2) ± deslocamento da partição + espalhamento fixo
    for (tarefa, k, part), vals in pontos_vaz.items():
        base = {"a": 1.0, "d": 2.0}[k] + (-0.17 if part == "agrup" else 0.17)
        n = len(vals)
        gravar(f"vaz_{tarefa}_{k}_{part}.csv", ["x", "auc"],
               [[f"{base + (i - (n - 1) / 2) * 0.05:.3f}", f"{v:.4f}"] for i, v in enumerate(vals)])
    for k, pts in roc_es.items():
        gravar(f"roc_es_{k}.csv", ["fpr", "tpr"], [[f"{x:.4f}", f"{y:.4f}"] for x, y in pts])
    for (nome, k), pts in roc_dist.items():
        gravar(f"roc_dist_{nome}_{k}.csv", ["fpr", "tpr"], [[f"{x:.4f}", f"{y:.4f}"] for x, y in pts])

    # AUC x parâmetros da cabeça (de tabela_wavkan*.csv). W3 e W4 têm o nº de parâmetros da W1 e são
    # deslocados na horizontal só para não se sobreporem; "ancora" posiciona o rótulo de cada ponto.
    desloc = {"W3": 0.66, "W4": 1.60}
    ancora = {"es": {"c": "south west", "d": "west", "W5": "north west", "W3": "east"},
              "dist": {"c": "south west", "d": "north west"}}
    for nome, tab in [("es", tw), ("dist", twd)]:
        linhas = []
        for i, r in tab.items():
            m, s = (float(x) for x in r["auc"].split(" ± "))
            grupo = "wav" if i.startswith("W") else ("kan" if i == "d" else "mlp")
            p = int(r["params_cabeca"])
            linhas.append([i, p, f"{p * desloc.get(i, 1.0):.0f}", f"{m:.3f}", f"{s:.3f}", grupo,
                           ancora[nome].get(i, "west")])
        for grupo in ("mlp", "kan", "wav"):
            gravar(f"auc_param_{nome}_{grupo}.csv", ["id", "params", "xplot", "auc", "dp", "grupo", "ancora"],
                   [l for l in linhas if l[5] == grupo])

    gravar("diagnostico_kan.csv", list(diag[0].keys()), [list(d.values()) for d in diag])

    with open(os.path.join(SAIDA, "figuras_macros.tex"), "w") as f:
        f.write("% Gerado por tcc_dissertation/scripts/preparar_dados_figuras.py: AUC OOF das legendas das ROC.\n")
        for nome, v in sorted(auc_leg.items()):
            f.write(f"\\newcommand{{\\aucRoc{nome}}}{{{v}}}\n")
    print(f"\nTudo conferido; arquivos gravados em {SAIDA}")
    for d in diag:
        print("diagnóstico KAN", d)


if __name__ == "__main__":
    main()
