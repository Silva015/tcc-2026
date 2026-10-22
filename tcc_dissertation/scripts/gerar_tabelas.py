#!/usr/bin/env python3
"""Gera as tabelas da dissertação (EN e PT no mesmo arquivo, via \\EP{en}{pt}) a partir de resultados/.

Lê só resultados/*.csv|*.json (nada é editado lá) e grava tcc_dissertation/comum/tabelas/*.tex.
Os contrastes vêm dos JSON em precisão cheia: o contrastes_multitarefa.tex de resultados/ arredondou
duas vezes (−0,0095 → −0,009), e aqui o arredondamento é feito uma só vez.
Só usa a biblioteca padrão do Python.
"""
import csv
import json
import os
import re

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(RAIZ, "resultados")
SAIDA = os.path.join(RAIZ, "tcc_dissertation", "comum", "tabelas")


def ler(nome):
    with open(os.path.join(RES, nome), newline="") as f:
        return list(csv.DictReader(f))


def js(nome):
    return json.load(open(os.path.join(RES, nome)))


def EP(en, pt):
    return f"\\EP{{{en}}}{{{pt}}}"


def pm(txt):
    """'0.904 ± 0.013' -> '$0.904 \\pm 0.013$'."""
    m, s = txt.split(" ± ")
    return f"${m} \\pm {s}$"


def sinal(x, nd=3):
    """Número com sinal; se arredondar para zero sem ser zero, ganha uma casa (ex.: +0.0004)."""
    v = f"{x:+.{nd}f}"
    if float(v) == 0 and x != 0:
        v = f"{x:+.{nd + 1}f}"
    if float(v) == 0:
        v = "0." + "0" * nd
    return f"${v}$"


def ic(lo, hi, nd=3):
    return f"[{sinal(lo, nd)}, {sinal(hi, nd)}]"


def milhar(n):
    return f"{int(n):,}".replace(",", "{,}")


def nome_modelo(txt):
    return (txt.replace("= TransKAN", "= TransferKAN").replace("TransKAN (ResNet152 + KAN)", "TransferKAN (ResNet152 + KAN)")
            .replace("mexican hat", "Mexican hat"))


def sementes(txt):
    n = len(str(txt).split(","))
    return str(n)


def celulas_bilingues(linha):
    """Nas linhas de dados: decimais viram \\dec{...} (vírgula no PT) e milhares 1{,}049{,}860 viram
    \\EP{1{,}049{,}860}{1.049.860}. Nomes como MNASNet1.3 ou x0.5 não são tocados."""
    linha = re.sub(r"(?<![\w{.])(\d+(?:\{,\}\d{3})+)(?![\d.])",
                   lambda m: f"\\EP{{{m.group(1)}}}{{{m.group(1).replace('{,}', '.')}}}", linha)
    return re.sub(r"(?<![\w{.,])(\d+\.\d+)", r"\\dec{\1}", linha)


def tabela(nome, legenda_en, legenda_pt, rotulo, colunas, cabecalho, linhas, tamanho="\\footnotesize",
           larga=False, nota_en="", nota_pt="", espaco="4pt"):
    corpo = []
    corpo.append("% Gerado por tcc_dissertation/scripts/gerar_tabelas.py a partir de resultados/. Não edite à mão.")
    corpo.append("\\begin{table}[!htbp]")
    corpo.append("\\centering")
    corpo.append(f"\\caption{{{EP(legenda_en, legenda_pt)}}}")
    corpo.append(f"\\label{{{rotulo}}}")
    corpo.append(tamanho)
    corpo.append(f"\\setlength{{\\tabcolsep}}{{{espaco}}}")
    if larga:
        corpo.append("\\begin{adjustbox}{max width=\\textwidth}")
    corpo.append(f"\\begin{{tabular}}{{{colunas}}}")
    corpo.append("\\toprule")
    corpo += cabecalho
    corpo.append("\\midrule")
    for l in linhas:
        cru = l.endswith("\\\\") or l.startswith(("\\midrule", "\\cmidrule", "\\hline"))
        if not cru:
            l = celulas_bilingues(l)
        corpo.append(l if cru else l + " \\\\")
    corpo.append("\\bottomrule")
    corpo.append("\\end{tabular}")
    if larga:
        corpo.append("\\end{adjustbox}")
    if nota_en:
        corpo.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {EP(nota_en, nota_pt)}}}")
    corpo.append("\\end{table}")
    with open(os.path.join(SAIDA, nome + ".tex"), "w") as f:
        f.write("\n".join(corpo) + "\n")
    print("gravado", nome)


def secao(texto_en, texto_pt, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{EP(texto_en, texto_pt)}}}}} \\\\"


# ---------------------------------------------------------------------------------------------- tabelas
def t_dobras():
    meta = js("dobras_meta.json")["checagem"]
    linhas = []
    for d in meta["por_dobra"]:
        linhas.append(f"{d['dobra'] + 1} & {milhar(d['n_img'])} & {d['n_casos']} & {d['n_saida']} & "
                      f"{d['n_encostado']} ({d['casos_encostado']}) & {d['n_curta']} & {d['n_distancia']}")
    cy = meta["contagem_y_dist"]
    tot_cont = sum(d["casos_encostado"] for d in meta["por_dobra"])
    linhas.append("\\midrule")
    linhas.append(f"{EP('Total', 'Total')} & {milhar(meta['n_img'])} & {meta['n_casos']} & {cy['-1']} & "
                  f"{cy['0']} ({tot_cont}) & {cy['1']} & {milhar(cy['2'])}")
    tabela("tab_dobras",
           "Composition of the five test folds (\\texttt{StratifiedGroupKFold}, seed 42, groups = case, "
           "stratified by the four-class label). Contact: images (cases). No case appears in more than one fold.",
           "Composição das cinco dobras de teste (\\texttt{StratifiedGroupKFold}, semente 42, grupos = caso, "
           "estratificação pelo rótulo de quatro classes). Encostado: imagens (casos). Nenhum caso aparece em mais de uma dobra.",
           "tab:dobras", "crrrrrr",
           [f"{EP('Fold', 'Dobra')} & {EP('Images', 'Imagens')} & {EP('Cases', 'Casos')} & {EP('Exits', 'Saídas')} & "
            f"{EP('Contact', 'Encostado')} & {EP('Close', 'Curta')} & {EP('Distant', 'Distância')} \\\\"],
           linhas)


def t_cabecas():
    tw = {r["id"]: r for r in ler("tabela_wavkan.csv")}
    twd = {r["id"]: r for r in ler("tabela_wavkan_mlsd.csv")}
    tf = {r["config"]: r for r in ler("tabela_final.csv") if r["painel"] == "A"}
    desc = {
        "a": ("Linear", "Linear$(2048, K)$"),
        "b": ("MLP-128", "$[2048, 128, K]$, ReLU"),
        "c": (EP("MLP, matched to (d)", "MLP isoparamétrica"), "$[2048, 1919, K]$, ReLU"),
        "d": (EP("Spline KAN = TransferKAN", "KAN spline = TransferKAN"), "$[2048, 128, K]$, $G=10$, $k=3$"),
        "e": ("KAN-8", "$[2048, 8, 2]$, $G=10$, $k=3$"),
        "W1": (EP("Wav-KAN, Mexican hat", "Wav-KAN, chapéu mexicano"), "$[2048, 128, K]$"),
        "W2": (EP("Wav-KAN, Mexican hat", "Wav-KAN, chapéu mexicano"), "$[2048, 32, K]$"),
        "W3": ("Wav-KAN, DoG", "$[2048, 128, 2]$"),
        "W4": ("Wav-KAN, Morlet", "$[2048, 128, 2]$"),
        "W5": (EP("Wav-KAN, Mexican hat", "Wav-KAN, chapéu mexicano"), "$[2048, 480, 2]$"),
    }
    papel = {
        "a": EP("reference", "referência"), "b": EP("reference", "referência"),
        "c": EP("same budget as (d)", "mesmo orçamento de (d)"), "d": EP("object of study", "objeto de estudo"),
        "e": EP("sensitivity", "sensibilidade"), "W1": EP("topology of (d)", "topologia de (d)"),
        "W2": EP("same budget as (b)", "mesmo orçamento de (b)"), "W3": EP("sensitivity", "sensibilidade"),
        "W4": EP("sensitivity", "sensibilidade"), "W5": EP("same budget as (d)", "mesmo orçamento de (d)"),
    }
    sem = js("decisao_wavkan.json")["sementes_por_config"]
    linhas = []
    for i in ["a", "b", "c", "d", "e", "W1", "W2", "W3", "W4", "W5"]:
        if i == "e":
            p_es = milhar(round(float(tf["resnet152__kan_pequena"]["params_cabeca_M"]) * 1e6))
            s_es, p_d, s_d = "42", "---", "---"
        else:
            p_es = milhar(tw[i]["params_cabeca"])
            s_es = "42--44" if len(sem["es"][i]) == 3 else "42"
            p_d = milhar(twd[i]["params_cabeca"]) if i in twd else "---"
            s_d = ("42--44" if len(sem["dist"][i]) == 3 else "42") if i in sem["dist"] else "---"
        rot = f"({i})" if len(i) == 1 else i
        linhas.append(f"{rot} & {desc[i][0]} & {desc[i][1]} & {p_es} & {p_d} & {s_es} & {s_d} & {papel[i]}")
    tabela("tab_cabecas",
           "Classification heads compared on the same ResNet-152 backbone (58,143,808 parameters) and the same recipe. "
           "$K$ = number of classes (2 for entry/exit, 3 for MLSD). Parameters of the head only; seeds used in each task.",
           "Cabeças de classificação comparadas sobre o mesmo backbone ResNet-152 (58.143.808 parâmetros) e a mesma receita. "
           "$K$ = número de classes (2 em entrada/saída, 3 na MLSD). Parâmetros só da cabeça; sementes usadas em cada tarefa.",
           "tab:cabecas", "lllrrccl",
           [f" & & & \\multicolumn{{2}}{{c}}{{{EP('Head parameters', 'Parâmetros da cabeça')}}} & "
            f"\\multicolumn{{2}}{{c}}{{{EP('Seeds', 'Sementes')}}} & \\\\",
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
            f"Id & {EP('Head', 'Cabeça')} & {EP('Structure', 'Estrutura')} & E/S & MLSD & E/S & MLSD & {EP('Role', 'Papel')} \\\\"],
           linhas, larga=True, espaco="3pt")


def t_protocolo_es():
    tf = ler("tabela_final.csv")
    linhas = [secao("Head ablation (ResNet-152 backbone)", "Ablação da cabeça (backbone ResNet-152)", 7)]
    painel_b = False
    for r in tf:
        if r["painel"] == "B" and not painel_b:
            linhas.append("\\midrule")
            linhas.append(secao("Backbones (linear head)", "Backbones (cabeça linear)", 7))
            painel_b = True
        linhas.append(f"{nome_modelo(r['modelo'])} & {float(r['params_total_M']):.2f} & {sementes(r['sementes'])} & "
                      f"{pm(r['acc'])} & {pm(r['f1_macro'])} & {pm(r['auc'])} & {float(r['treino_min_dobra']):.1f}")
    tabela("tab_protocolo_es",
           "Entry/exit classification under the single protocol (case-grouped stratified 5-fold, same two-phase recipe, "
           "last epoch). Mean $\\pm$ s.d.\\ across folds; each fold is the mean over the training seeds (column S). "
           "F1 is the macro average (macro precision and recall are in Appendix~\\ref{chap:apendice_mmw}); AUC is computed "
           "from predicted probabilities. "
           "Min: training minutes per fold on an NVIDIA L4.",
           "Classificação entrada/saída no protocolo único (5 dobras estratificadas e agrupadas por caso, mesma receita "
           "de duas fases, última época). Média $\\pm$ dp entre dobras; cada dobra é a média das sementes de treino (coluna S). "
           "F1 é a média macro (precisão e revocação macro estão no Apêndice~\\ref{chap:apendice_mmw}); a AUC é calculada "
           "das probabilidades previstas. "
           "Min: minutos de treino por dobra numa NVIDIA L4.",
           "tab:protocolo_es", "lrccccr",
           [f"{EP('Model', 'Modelo')} & {EP('Params (M)', 'Parâm. (M)')} & S & Acc & F1 & AUC & Min \\\\"],
           linhas, larga=True, espaco="4pt")


def t_protocolo_mlsd():
    tm = ler("tabela_mlsd.csv")
    linhas = [secao("Head ablation (ResNet-152 backbone)", "Ablação da cabeça (backbone ResNet-152)", 10)]
    painel_b = False
    for r in tm:
        if r["painel"] == "B" and not painel_b:
            linhas.append("\\midrule")
            linhas.append(secao("Backbones (linear head)", "Backbones (cabeça linear)", 10))
            painel_b = True
        linhas.append(f"{nome_modelo(r['modelo'])} & {float(r['params_total_M']):.2f} & {sementes(r['sementes'])} & "
                      f"{pm(r['acc'])} & {pm(r['acc_balanceada'])} & {pm(r['f1_macro'])} & {pm(r['auc'])} & "
                      f"{float(r['rec_encostado']):.2f} & {float(r['rec_curta']):.2f} & {float(r['treino_min_dobra']):.1f}")
    tabela("tab_protocolo_mlsd",
           "Medico-legal shooting distance (contact, close range, distant; entry wounds only) under the same protocol as "
           "Table~\\ref{tab:protocolo_es}. Mean $\\pm$ s.d.\\ across folds; S = training seeds (Swin-B and MaxViT-T: "
           "seeds 42 and 43 only). AUC = one-vs-rest macro AUC from probabilities. Rec.: per-class recall from the pooled "
           "out-of-fold predictions; contact has 4--13 test images per fold and is descriptive only.",
           "Distância médico-legal do disparo (encostado, curta, distância; só entradas) no mesmo protocolo da "
           "Tabela~\\ref{tab:protocolo_es}. Média $\\pm$ dp entre dobras; S = sementes de treino (Swin-B e MaxViT-T: "
           "só as sementes 42 e 43). AUC = AUC macro um-contra-o-resto, das probabilidades. Rev.: revocação por classe; "
           "encostado tem de 4 a 13 imagens de teste por dobra e é só descritivo.",
           "tab:protocolo_mlsd", "lrcccccccr",
           [f"{EP('Model', 'Modelo')} & {EP('Params (M)', 'Parâm. (M)')} & S & Acc & {EP('Bal.\\ acc', 'Acc bal.')} & "
            f"F1$_M$ & AUC & {EP('Rec.\\ cont.', 'Rev.\\ enc.')} & {EP('Rec.\\ close', 'Rev.\\ curta')} & Min \\\\"],
           linhas, larga=True, espaco="3pt")


def t_vazamento():
    tv = ler("tabela_vazamento.csv")
    linhas = []
    for r in tv:
        tarefa = EP("Entry/exit", "Entrada/saída") if r["tarefa"] == "es" else "MLSD"
        modelo = nome_modelo(r["modelo"]).replace("ResNet152 + ", "")
        vaz = f"{float(r['vazamento_medio']) * 100:.1f}\\%"
        linhas.append(f"{tarefa} & {modelo} & {EP('Lira et al.', 'Lira et al.')} & {vaz} & {pm(r['lira_acc'])} & "
                      f"{pm(r['lira_f1_macro'])} & {pm(r['lira_acc_balanceada'])} & {pm(r['lira_auc'])}")
        linhas.append(f" & & {EP('Grouped', 'Agrupada')} & 0\\% & {pm(r['agrupado_acc'])} & {pm(r['agrupado_f1_macro'])} & "
                      f"{pm(r['agrupado_acc_balanceada'])} & {pm(r['agrupado_auc'])}")
        linhas.append(f" & & $\\Delta$ & & {sinal(float(r['delta_acc']))} & {sinal(float(r['delta_f1_macro']))} & "
                      f"{sinal(float(r['delta_acc_balanceada']))} & {sinal(float(r['delta_auc']))}")
        if r is not tv[-1]:
            linhas.append("\\midrule")
    tabela("tab_vazamento",
           "Effect of the data split with everything else fixed (ResNet-152, same recipe, seed 42). "
           "\\emph{Lira et al.}: the image-level hold-out of the public code of Lira et al.\\ (iterative stratification "
           "on case number and label) reproduced with three partitions; mean $\\pm$ s.d.\\ over the partitions. "
           "\\emph{Grouped}: the five case-grouped folds of this work; mean $\\pm$ s.d.\\ over folds. "
           "Leak: share of test images whose case also has images in training. $\\Delta$ = Lira $-$ grouped.",
           "Efeito da partição com todo o resto fixo (ResNet-152, mesma receita, semente 42). "
           "\\emph{Lira et al.}: o hold-out por imagem do código público de Lira et al.\\ (estratificação iterativa "
           "sobre o número do caso e o rótulo) reproduzido em três partições; média $\\pm$ dp entre as partições. "
           "\\emph{Agrupada}: as cinco dobras agrupadas por caso deste trabalho; média $\\pm$ dp entre dobras. "
           "Vaz.: fração das imagens de teste cujo caso também tem imagens no treino. $\\Delta$ = Lira $-$ agrupada.",
           "tab:vazamento", "lllrcccc",
           [f"{EP('Task', 'Tarefa')} & {EP('Head', 'Cabeça')} & {EP('Split', 'Partição')} & {EP('Leak', 'Vaz.')} & "
            f"Acc & F1$_M$ & {EP('Bal.\\ acc', 'Acc bal.')} & AUC \\\\"],
           linhas, larga=True, espaco="3pt")


def t_lira(tarefa):
    arq = "tabela_comparacao_2024_es.csv" if tarefa == "es" else "tabela_comparacao_2024_dist.csv"
    rows = ler(arq)
    linhas = []
    for r in rows:
        modelo = nome_modelo(r["modelo"])
        if r["lira_acc"]:
            lira = f"{float(r['lira_acc']):.3f} & {float(r['lira_f1_M']):.3f} & {float(r['lira_auc_M_classes']):.3f}"
        else:
            lira = "--- & --- & ---"
        linhas.append(f"{modelo} & {lira} & {pm(r['nosso_acc'])} & {pm(r['nosso_f1_M'])} & "
                      f"{pm(r['nosso_auc_M_classes'])} & {pm(r['nosso_auc_M_prob'])}")
    if tarefa == "es":
        en_t, pt_t, tab = "Entry/exit", "Entrada/saída", "Table~4"
        en_x, pt_x = "", ""
    else:
        en_t, pt_t, tab = "MLSD (entries only)", "MLSD (só entradas)", "Table~5"
        en_x = " In this task, predicting always ``distant'' already gives an accuracy of 0.854."
        pt_x = " Nesta tarefa, prever sempre ``distância'' já dá acurácia de 0,854."
    tabela(f"tab_lira_{tarefa}",
           f"{en_t}: numbers published by Lira et al.~\\cite{{Lira2025deep}} ({tab}) next to the same backbones under the "
           "single protocol of this work. The two sides differ in three ways at once: the split (image-level hold-out in "
           "their public code $\\times$ case-grouped 5-fold), the initialisation (mostly trained from scratch in their "
           "public code $\\times$ ImageNet weights) and the AUC (from predicted classes, AUC$_c$ $\\times$ from "
           "probabilities, AUC$_p$). Only AUC$_c$ is computed in the same way on both sides; the table compares "
           "evaluation packages, not models. M = macro; ours: mean $\\pm$ s.d.\\ across folds." + en_x,
           f"{pt_t}: números publicados por Lira et al.~\\cite{{Lira2025deep}} ({tab.replace('Table', 'Tabela')}) ao lado dos mesmos backbones no "
           "protocolo único deste trabalho. Os dois lados diferem em três pontos ao mesmo tempo: a partição (hold-out por "
           "imagem no código público deles $\\times$ 5 dobras agrupadas por caso), a inicialização (quase tudo treinado do "
           "zero no código público deles $\\times$ pesos do ImageNet) e a AUC (das classes previstas, AUC$_c$ $\\times$ "
           "das probabilidades, AUC$_p$). Só a AUC$_c$ é calculada do mesmo jeito nos dois lados; a tabela compara "
           "pacotes de avaliação, não modelos. M = macro; nossos: média $\\pm$ dp entre dobras." + pt_x,
           f"tab:lira_{tarefa}", "lccc|cccc",
           [f" & \\multicolumn{{3}}{{c|}}{{{EP('Lira et al.\\ (published)', 'Lira et al.\\ (publicado)')}}} & "
            f"\\multicolumn{{4}}{{c}}{{{EP('Single protocol (this work)', 'Protocolo único (este trabalho)')}}} \\\\",
            f"{EP('Model', 'Modelo')} & Acc & F1$_M$ & AUC$_{{c,M}}$ & Acc & F1$_M$ & AUC$_{{c,M}}$ & AUC$_{{p,M}}$ \\\\"],
           linhas, larga=True, espaco="3pt")


def t_transformers():
    d = js("comparacao_transkan_transformers.json")
    nomes = {"swin_b": "Swin-B", "maxvit_t": "MaxViT-T"}
    linhas = []
    for k, c in d.items():
        tarefa, outro = k.split("__")
        t = EP("Entry/exit", "Entrada/saída") if tarefa == "es" else "MLSD"
        s_b = "42--44" if len(c["sementes_b"]) == 3 else "42, 43"
        linhas.append(f"{t} & TransferKAN $-$ {nomes[outro]} & {sinal(c['delta_auc_oof'])} & "
                      f"{ic(*c['ic95'])} & {c['dobras_a_vence']}/{c['n_dobras']} & {s_b}")
    tabela("tab_transformers",
           "TransferKAN (ResNet-152 + spline KAN, seeds 42--44) against the two transformer backbones with a linear head, "
           "under the single protocol. $\\Delta$AUC on the pooled out-of-fold predictions (probabilities averaged over the "
           "seeds); 95\\% CI from a paired case-level bootstrap (2,000 resamples); wins = folds in which TransferKAN has the "
           "higher AUC.",
           "TransferKAN (ResNet-152 + KAN spline, sementes 42--44) contra os dois backbones transformer com cabeça linear, "
           "no protocolo único. $\\Delta$AUC nas predições fora da dobra (probabilidades médias das sementes); IC 95\\% por "
           "bootstrap pareado de casos (2.000 reamostragens); vence = dobras em que a TransferKAN tem a AUC maior.",
           "tab:transformers", "llcccc",
           [f"{EP('Task', 'Tarefa')} & {EP('Contrast', 'Contraste')} & $\\Delta$AUC & {EP('95\\% CI', 'IC 95\\%')} & "
            f"{EP('Wins', 'Vence')} & {EP('Seeds of the transformer', 'Sementes do transformer')} \\\\"],
           linhas, larga=True)


def t_wavkan(tarefa):
    arq = "tabela_wavkan.csv" if tarefa == "es" else "tabela_wavkan_mlsd.csv"
    linhas = []
    for r in ler(arq):
        i = r["id"]
        rot = f"({i})" if len(i) == 1 else i
        nome = {"a": "Linear", "b": "MLP-128", "c": "MLP-iso", "d": EP("Spline KAN = TransferKAN", "KAN spline = TransferKAN")}.get(i)
        if nome is None:
            onda = r["modelo"].split("Wav-KAN ")[1].split(",")[0]
            onda = {"mexican hat": EP("Mexican hat", "chapéu mexicano"), "DoG": "DoG", "Morlet": "Morlet"}[onda]
            h = re.search(r"h=(\d+)", r["modelo"]).group(1)
            nome = f"Wav-KAN, {onda}, $h={h}$"
        linhas.append(f"{rot} & {nome} & {milhar(r['params_cabeca'])} & {sementes(r['sementes'])} & {pm(r['acc'])} & "
                      f"{pm(r['f1_macro'])} & {pm(r['auc'])} & {float(r['auc_oof']):.3f} & "
                      f"{float(r['min_por_dobra']):.1f} & {float(r['cabeca_fwd_bwd_ms_lote32']):.2f}")
    if tarefa == "es":
        en_t, pt_t, rot = "Entry/exit", "Entrada/saída", "tab:wavkan_es"
    else:
        en_t, pt_t, rot = "MLSD (entries only; AUC = one-vs-rest macro)", "MLSD (só entradas; AUC = macro um-contra-o-resto)", "tab:wavkan_mlsd"
    tabela(f"tab_wavkan_{tarefa}",
           f"{en_t}: wavelet-KAN heads against the heads (a)--(d) on the same ResNet-152 and recipe. Mean $\\pm$ s.d.\\ "
           "across folds (each fold averaged over the S training seeds); F1 is macro; AUC$_{\\mathrm{OOF}}$ pools the "
           "out-of-fold predictions. Min: training minutes per fold (NVIDIA L4). Head ms: forward + backward of the head "
           "alone on a batch of 32 random 2048-d inputs.",
           f"{pt_t}: cabeças Wav-KAN contra as cabeças (a)--(d), na mesma ResNet-152 e com a mesma receita. Média "
           "$\\pm$ dp entre dobras (cada dobra é a média das S sementes); F1 macro; a AUC$_{\\mathrm{OOF}}$ junta as "
           "predições fora da dobra. Min: minutos de treino por dobra (NVIDIA L4). Cabeça ms: forward + backward só da "
           "cabeça, num lote de 32 entradas aleatórias de 2048 dimensões.",
           rot, "llrcccccrr",
           [f"Id & {EP('Head', 'Cabeça')} & {EP('Params', 'Parâm.')} & S & Acc & F1 & AUC & AUC$_{{\\mathrm{{OOF}}}}$ & "
            f"Min & {EP('Head ms', 'Cabeça ms')} \\\\"],
           linhas, larga=True, espaco="3pt")


def t_contrastes_wavkan():
    d = js("decisao_wavkan.json")
    papeis = {"primário 1": EP("primary 1", "primário 1"), "primário 2": EP("primary 2", "primário 2"),
              "secundário": EP("secondary", "secundário"), "descritivo": EP("descriptive", "descritivo"),
              "sensibilidade": EP("sensitivity", "sensibilidade")}
    linhas = []
    tarefa_ant = None
    for k, c in d["contrastes"].items():
        if tarefa_ant and c["tarefa"] != tarefa_ant:
            linhas.append("\\midrule")
        tarefa_ant = c["tarefa"]
        t = "E/S" if c["tarefa"] == "es" else "MLSD"
        b = f"({c['b']})" if len(c["b"]) == 1 else c["b"]
        if c["superior"]:
            res = EP("superior", "superior")
        elif c["inferior"]:
            res = EP("inferior", "inferior")
        elif c["nao_inferior"]:
            res = EP("non-inferior", "não inferior")
        else:
            res = EP("inconclusive", "inconclusivo")
        s = "42--44" if len(c["sementes"]) == 3 else "42"
        linhas.append(f"{t} & {c['a']} $-$ {b} & {papeis[c['papel']]} & {s} & {sinal(c['delta_auc_oof'])} {ic(*c['ic95_auc'])} & "
                      f"{sinal(c['delta_f1_oof'])} {ic(*c['ic95_f1'])} & {c['dobras_a_vence']}/{c['n_dobras']} & {res}")
    tabela("tab_contrastes_wavkan",
           "Pre-registered contrasts of the wavelet-KAN heads ($\\Delta$ = W $-$ reference) on the pooled out-of-fold "
           "predictions (probabilities averaged over the common seeds); 95\\% CI from a paired case-level bootstrap "
           "(2,000 resamples). Non-inferiority margin: $-0.02$ on the lower CI bound of both $\\Delta$AUC and $\\Delta$F1. "
           "Superior: higher AUC in at least 4/5 folds and $\\Delta$AUC CI above zero. Inferior: higher AUC in at most 1/5 folds "
           "and $\\Delta$AUC CI below zero. Inconclusive: none of the three. Wins: folds in which W has the higher AUC.",
           "Contrastes pré-registrados das cabeças Wav-KAN ($\\Delta$ = W $-$ referência) nas predições fora da dobra "
           "(probabilidades médias das sementes em comum); IC 95\\% por bootstrap pareado de casos (2.000 reamostragens). "
           "Margem de não inferioridade: $-0{,}02$ no limite inferior do IC do $\\Delta$AUC e do $\\Delta$F1. "
           "Superior: AUC maior em pelo menos 4/5 dobras e IC do $\\Delta$AUC acima de zero. Inferior: AUC maior em no máximo "
           "1/5 dobras e IC do $\\Delta$AUC abaixo de zero. Inconclusivo: nenhum dos três. Vence: dobras em que W tem a AUC maior.",
           "tab:contrastes_wavkan", "lllcccrl",
           [f"{EP('Task', 'Tarefa')} & {EP('Contrast', 'Contraste')} & {EP('Role', 'Papel')} & "
            f"{EP('Seeds', 'Sementes')} & $\\Delta$AUC [IC 95\\%] & $\\Delta$F1 [IC 95\\%] & "
            f"{EP('Wins', 'Vence')} & {EP('Outcome', 'Desfecho')} \\\\".replace("[IC 95\\%]", f"[{EP('95\\% CI', 'IC 95\\%')}]")],
           linhas, larga=True, espaco="3pt")


SIST = {
    "st_kan": EP("Two single-task nets, KAN heads (d)", "Duas redes de tarefa única, cabeças KAN (d)"),
    "mt_kan": EP("One multitask net, KAN heads", "Uma rede multitarefa, cabeças KAN"),
    "st_mlp": EP("Two single-task nets, MLP-128 heads (b)", "Duas redes de tarefa única, cabeças MLP-128 (b)"),
    "mt_mlp": EP("One multitask net, MLP-128 heads", "Uma rede multitarefa, cabeças MLP-128"),
}


def t_multitarefa():
    tm = ler("tabela_multitarefa.csv")
    custo = js("decisao_multitarefa.json")["custo"]
    bench = {(b["tipo"], b["lote"]): b for b in js("decisao_multitarefa.json")["benchmark_inferencia"]["tabela"]}
    linhas = []
    for r in tm:
        linhas.append(f"{SIST[r['id']]} & {r['redes']} & {float(r['params_total_M']):.2f} & {pm(r['es_auc'])} & "
                      f"{pm(r['es_f1_macro'])} & {pm(r['dist_auc'])} & {pm(r['dist_f1_macro'])} & {pm(r['pipeline_f1_macro_4'])}")
    tabela("tab_mt_desempenho",
           "One multitask ResNet-152 (shared backbone, one head per task, $L = L_{\\mathrm{E/S}} + L_{\\mathrm{dist}}$ with "
           "exits masked out of $L_{\\mathrm{dist}}$) against the pair of single-task networks of the head ablation, "
           "same folds and recipe, seeds 42--44 on both sides. Mean $\\pm$ s.d.\\ across folds; AUC from probabilities "
           "(distance: one-vs-rest macro, entries only); F1 is macro; 4-class F1: exit/contact/close/distant pipeline "
           "(secondary, descriptive).",
           "Uma ResNet-152 multitarefa (backbone compartilhado, uma cabeça por tarefa, $L = L_{\\mathrm{E/S}} + "
           "L_{\\mathrm{dist}}$, com as saídas fora de $L_{\\mathrm{dist}}$) contra o par de redes de tarefa única da "
           "ablação da cabeça, com as mesmas dobras e a mesma receita e sementes 42--44 nos dois lados. Média $\\pm$ dp "
           "entre dobras; AUC das probabilidades (distância: macro um-contra-o-resto, só entradas); F1 macro; F1 de 4 classes: "
           "pipeline saída/encostado/curta/distância (secundário, descritivo).",
           "tab:mt_desempenho", "lcrccccc",
           [f"{EP('System', 'Sistema')} & {EP('Nets', 'Redes')} & {EP('Params (M)', 'Parâm. (M)')} & "
            f"{EP('E/S AUC', 'AUC E/S')} & {EP('E/S F1', 'F1 E/S')} & {EP('Dist.\\ AUC', 'AUC dist.')} & "
            f"{EP('Dist.\\ F1', 'F1 dist.')} & {EP('4-class F1', 'F1 4 classes')} \\\\"],
           linhas, larga=True, espaco="3pt")

    linhas = []
    for r in tm:
        c = custo[r["id"]]
        tipo = "kan" if r["id"].endswith("kan") else "mlp"
        lado = "mt" if r["id"].startswith("mt") else "par_st"
        b1, b32 = bench[(tipo, 1)], bench[(tipo, 32)]
        linhas.append(f"{SIST[r['id']]} & {c['params_total'] / 1e6:.2f} & "
                      f"${c['min_por_dobra']:.1f} \\pm {c['min_por_dobra_dp']:.1f}$ & {c['mem_pico_treino_gb']:.1f} & "
                      f"${b1[lado + '_ms_img']:.1f} \\pm {b1[lado + '_dp_ms_img']:.1f}$ & "
                      f"${b32[lado + '_ms_img']:.2f} \\pm {b32[lado + '_dp_ms_img']:.2f}$ & {b32[lado + '_mem_pico_mb']:.0f}")
    tabela("tab_mt_custo",
           "Cost of obtaining both outputs on an NVIDIA L4 (AMP fp16). Training: minutes per fold (single-task = sum of "
           "the two networks; mean $\\pm$ s.d.\\ over the 15 units) and peak training memory. Inference: ms per image to "
           "obtain both outputs (one forward pass for the multitask network, two for the pair; 20 warm-up and 200 timed "
           "runs with CUDA events, mean $\\pm$ s.d.) and peak inference memory at batch 32. Timing does not depend on "
           "the weights and was measured on freshly built models.",
           "Custo para obter as duas saídas numa NVIDIA L4 (AMP fp16). Treino: minutos por dobra (tarefa única = soma das "
           "duas redes; média $\\pm$ dp nas 15 unidades) e pico de memória do treino. Inferência: ms por imagem para obter "
           "as duas saídas (um forward na rede multitarefa, dois no par; 20 aquecimentos e 200 medidas por eventos CUDA, "
           "média $\\pm$ dp) e pico de memória da inferência com lote 32. O tempo não depende dos pesos e foi medido em "
           "modelos recém-criados.",
           "tab:mt_custo", "lrcccccc".replace("cccccc", "ccccc"),
           [f" & & \\multicolumn{{2}}{{c}}{{{EP('Training', 'Treino')}}} & \\multicolumn{{3}}{{c}}{{{EP('Inference (both outputs)', 'Inferência (as duas saídas)')}}} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
            f"{EP('System', 'Sistema')} & {EP('Params (M)', 'Parâm. (M)')} & {EP('min/fold', 'min/dobra')} & "
            f"{EP('peak GB', 'pico GB')} & {EP('ms/img, batch 1', 'ms/img, lote 1')} & {EP('ms/img, batch 32', 'ms/img, lote 32')} & "
            f"{EP('peak MB', 'pico MB')} \\\\"],
           linhas, larga=True, espaco="3pt")


def t_contrastes_mt():
    d = js("decisao_multitarefa.json")
    nomes = {"mt_kan": "MT-KAN", "mt_mlp": "MT-MLP", "st_kan": "ST-KAN", "st_mlp": "ST-MLP"}
    papeis = {"primário": EP("primary", "primário"), "secundário": EP("secondary", "secundário"),
              "descritivo": EP("descriptive", "descritivo")}
    linhas = []
    tarefa_ant = None
    for k, c in d["contrastes"].items():
        if tarefa_ant and c["tarefa"] != tarefa_ant:
            linhas.append("\\midrule")
        tarefa_ant = c["tarefa"]
        t = "E/S" if c["tarefa"] == "es" else EP("Dist.", "Dist.")
        if c["papel"] == "descritivo":
            res = EP("descriptive", "descritivo")
        elif c["desfecho"] == "pior":
            res = EP("worse, within margin", "pior, dentro da margem") if c["dentro_da_margem"] else EP("worse", "pior")
        elif c["desfecho"] == "nao_inferior":
            res = EP("non-inferior", "não inferior")
        elif c["desfecho"] == "melhora":
            res = EP("better", "melhor")
        else:
            res = EP("inconclusive", "inconclusivo")
        linhas.append(f"{t} & {nomes[c['a']]} $-$ {nomes[c['b']]} & {papeis[c['papel']]} & "
                      f"{sinal(c['delta_auc_oof'])} {ic(*c['ic95_auc'])} & {sinal(c['delta_f1_oof'])} {ic(*c['ic95_f1'])} & "
                      f"{c['dobras_a_vence']}/{c['n_dobras']} & {res}")
    linhas.append("\\midrule")
    for k, c in d["pipeline"].items():
        linhas.append(f"{EP('4 classes', '4 classes')} & {nomes[c['a']]} $-$ {nomes[c['b']]} & {EP('descriptive', 'descritivo')} & --- & "
                      f"{sinal(c['delta_f1_oof'])} {ic(*c['ic95_f1'])} & {c['dobras_a_vence']}/{c['n_dobras']} & "
                      f"{EP('descriptive', 'descritivo')}")
    tabela("tab_contrastes_mt",
           "Pre-registered contrasts of the multitask study ($\\Delta$ = first $-$ second; MT = multitask, ST = pair of "
           "single-task networks), on the pooled out-of-fold predictions (probabilities averaged over seeds 42--44); 95\\% CI "
           "from a paired case-level bootstrap (2,000 resamples). Non-inferiority margin: $-0.02$ on the lower CI bound of "
           "$\\Delta$AUC. Worse: higher AUC in at most 1/5 folds and $\\Delta$AUC CI below zero; ``within margin'': that CI is "
           "still above $-0.02$. "
           "Wins: folds in which the first system has the higher AUC (F1 for the 4-class rows).",
           "Contrastes pré-registrados do estudo multitarefa ($\\Delta$ = primeiro $-$ segundo; MT = multitarefa, ST = par "
           "de redes de tarefa única), nas predições fora da dobra (probabilidades médias das sementes 42--44); IC 95\\% por "
           "bootstrap pareado de casos (2.000 reamostragens). Margem de não inferioridade: $-0{,}02$ no limite inferior do "
           "IC do $\\Delta$AUC. Pior: AUC maior em no máximo 1/5 dobras e IC do $\\Delta$AUC abaixo de zero; ``dentro da margem'': "
           "esse IC ainda fica acima de $-0{,}02$. Vence: dobras em que o primeiro sistema tem a AUC maior (o F1, nas linhas de 4 classes).",
           "tab:contrastes_mt", "lllccrl",
           [f"{EP('Task', 'Tarefa')} & {EP('Contrast', 'Contraste')} & {EP('Role', 'Papel')} & "
            f"$\\Delta$AUC [{EP('95\\% CI', 'IC 95\\%')}] & $\\Delta$F1 [{EP('95\\% CI', 'IC 95\\%')}] & "
            f"{EP('Wins', 'Vence')} & {EP('Outcome', 'Desfecho')} \\\\"],
           linhas, larga=True, espaco="3pt")


def t_mmw(tarefa):
    rows = ler("tabela_mmw_es.csv" if tarefa == "es" else "tabela_mmw_dist.csv")
    met = ["prec", "rec", "f1", "auc", "espec"]
    linhas = []
    for r in rows:
        cel = [f"{float(r['acc_mean']):.3f}"]
        for m in met:
            for a in ["M", "m", "W"]:
                cel.append(f"{float(r[f'{m}_{a}_mean']):.3f}")
        linhas.append(f"{nome_modelo(r['modelo'])} & " + " & ".join(cel))
    en_t = "Entry/exit" if tarefa == "es" else "MLSD (entries only)"
    pt_t = "Entrada/saída" if tarefa == "es" else "MLSD (só entradas)"
    tabela(f"tab_mmw_{tarefa}",
           f"{en_t} under the single protocol, in the format of Lira et al.~\\cite{{Lira2025deep}}: M = macro, m = micro, "
           "W = weighted. Mean across folds (the standard deviations are in the result files of the repository); AUC from predicted "
           "probabilities. Seeds as in the main tables.",
           f"{pt_t} no protocolo único, no formato de Lira et al.~\\cite{{Lira2025deep}}: M = macro, m = micro, "
           "W = ponderada. Média entre dobras (os desvios-padrão estão nos arquivos de resultados do repositório); AUC das probabilidades "
           "previstas. Sementes como nas tabelas principais.",
           f"tab:mmw_{tarefa}", "l" + "c" * 16,
           [f" & Acc & \\multicolumn{{3}}{{c}}{{{EP('Precision', 'Precisão')}}} & \\multicolumn{{3}}{{c}}{{{EP('Recall', 'Revocação')}}} & "
            f"\\multicolumn{{3}}{{c}}{{F1}} & \\multicolumn{{3}}{{c}}{{AUC}} & \\multicolumn{{3}}{{c}}{{{EP('Specificity', 'Especificidade')}}} \\\\",
            f"{EP('Model', 'Modelo')} & & M & m & W & M & m & W & M & m & W & M & m & W & M & m & W \\\\"],
           linhas, larga=True, espaco="2.5pt")


def t_exploratorios():
    """Valores da versão anterior desta dissertação (notebooks ConvKANWithResNet_*, PureConvKAN_*, PureKAN_*).
    Não vêm do protocolo único e não estão em resultados/."""
    es = [
        ("ResKANet-18", "128 px", "0.822 & 0.767 & 0.822 & 0.820 & 0.762 & 0.822 & 0.822 & 0.765 & 0.822 & 0.821 & 0.839 & 0.885 & 0.839 & 0.762 & 0.822 & 0.703"),
        ("PureConvKAN", "128 px", "0.707 & 0.614 & 0.707 & 0.704 & 0.611 & 0.707 & 0.707 & 0.613 & 0.707 & 0.706 & 0.682 & 0.768 & 0.682 & 0.611 & 0.707 & 0.516"),
        ("PureKAN", "64 px", "0.692 & 0.614 & 0.692 & 0.692 & 0.614 & 0.692 & 0.692 & 0.614 & 0.692 & 0.692 & 0.684 & 0.733 & 0.684 & 0.614 & 0.692 & 0.536"),
    ]
    dist = [
        ("PureConvKAN", "128 px", "0.803 & 0.268 & 0.803 & 0.645 & 0.333 & 0.803 & 0.803 & 0.297 & 0.803 & 0.716 & 0.536 & 0.888 & 0.509 & 0.667 & 0.902 & 0.197"),
        ("PureKAN", "64 px", "0.712 & 0.355 & 0.712 & 0.689 & 0.324 & 0.712 & 0.712 & 0.328 & 0.712 & 0.693 & 0.463 & 0.841 & 0.496 & 0.679 & 0.856 & 0.326"),
    ]
    linhas = [secao("Entry/exit", "Entrada/saída", 18)]
    linhas += [f"{n} & {r} & {v}" for n, r, v in es]
    linhas.append("\\midrule")
    linhas.append(secao("MLSD", "MLSD", 18))
    linhas += [f"{n} & {r} & {v}" for n, r, v in dist]
    tabela("tab_exploratorios",
           "Exploratory KAN architectures trained from scratch (no ImageNet pre-training) in an earlier phase of this work. "
           "They used a single 80/20 hold-out split by case, inputs of 64 or 128 px, batches of 8 and their own recipe, "
           "so they are \\emph{not} comparable with any number obtained under the single protocol. M = macro, m = micro, "
           "W = weighted.",
           "Arquiteturas KAN exploratórias treinadas do zero (sem pré-treino no ImageNet) numa fase anterior deste trabalho. "
           "Usaram um único hold-out 80/20 por caso, entradas de 64 ou 128 px, lotes de 8 e receita própria, portanto "
           "\\emph{não} são comparáveis com nenhum número obtido no protocolo único. M = macro, m = micro, W = ponderada.",
           "tab:exploratorios", "ll" + "c" * 16,
           [f" & & Acc & \\multicolumn{{3}}{{c}}{{{EP('Precision', 'Precisão')}}} & \\multicolumn{{3}}{{c}}{{{EP('Recall', 'Revocação')}}} & "
            f"\\multicolumn{{3}}{{c}}{{F1}} & \\multicolumn{{3}}{{c}}{{AUC}} & \\multicolumn{{3}}{{c}}{{{EP('Specificity', 'Especificidade')}}} \\\\",
            f"{EP('Model', 'Modelo')} & {EP('Input', 'Entrada')} & & M & m & W & M & m & W & M & m & W & M & m & W & M & m & W \\\\"],
           linhas, larga=True, espaco="2.5pt")


if __name__ == "__main__":
    os.makedirs(SAIDA, exist_ok=True)
    t_dobras()
    t_cabecas()
    t_protocolo_es()
    t_protocolo_mlsd()
    t_vazamento()
    t_lira("es")
    t_lira("dist")
    t_transformers()
    t_wavkan("es")
    t_wavkan("dist")
    t_contrastes_wavkan()
    t_multitarefa()
    t_contrastes_mt()
    t_mmw("es")
    t_mmw("dist")
    t_exploratorios()
