#!/usr/bin/env python3
"""Confere os números da dissertação contra resultados/ e contra o artigo de Lira et al.

1. Transcrição de Lira: cada linha de resultados/lira2024_tabelas.csv (16 métricas) tem de
   aparecer, na mesma ordem, no texto do artigo (references/2024_paper.txt).
2. Números do texto: cada número decimal (e cada porcentagem) dos capítulos EN e PT e das tabelas
   comuns precisa bater, a menos do arredondamento mostrado, com algum valor de resultados/,
   do artigo de Lira, dos dados das figuras ou de uma lista branca de constantes do protocolo.
   Este teste é FRACO (resultados/ tem milhares de valores e ~98% de números ao acaso passariam):
   só pega número sem origem nenhuma. O teste forte é o 4.
3. Paridade: o multiconjunto de números de cada capítulo EN tem de ser igual ao do capítulo PT.
4. Afirmações: ~220 valores calculados direto das fontes (tabelas, decisões, contrastes, custos,
   faixas) têm de aparecer, já formatados (e faixas inteiras, como "27--39"), no capítulo EN e no PT
   em que são citados. Um número errado no texto faz o valor certo "não ser encontrado".

Uso: python3 scripts/conferir_numeros.py   (só biblioteca padrão; código de saída 1 se algo falhar)
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TCC = os.path.join(RAIZ, "tcc_dissertation")
RES = os.path.join(RAIZ, "resultados")
PAPER = os.path.join(RAIZ, "references", "2024_paper.txt")

PARES = [  # (EN, PT)
    ("tex/1_Introduction.tex", "pt/tex/1_Introducao.tex"),
    ("tex/2_Background.tex", "pt/tex/2_Fundamentacao.tex"),
    ("tex/3_RelatedWork.tex", "pt/tex/3_TrabalhosRelacionados.tex"),
    ("tex/4_ProposedSolution.tex", "pt/tex/4_SolucaoProposta.tex"),
    ("tex/5_methodology.tex", "pt/tex/5_Metodologia.tex"),
    ("tex/6_Results.tex", "pt/tex/6_Resultados.tex"),
    ("tex/7_Conclusions.tex", "pt/tex/7_Conclusoes.tex"),
    ("tex/abstract.tex", "pt/tex/abstract.tex"),
    ("tex/resumo.tex", "pt/tex/resumo.tex"),
    ("tex/Apendice_MMW.tex", "pt/tex/Apendice_MMW.tex"),
    ("tex/Apendice_Exploratorios.tex", "pt/tex/Apendice_Exploratorios.tex"),
]

# Constantes do protocolo, versões, nomes de modelos e valores citados de fontes que não são resultados/
# (cada uma com o motivo). Comparadas pelo valor exato mostrado.
LISTA_BRANCA = {
    "0.5": "Dropout, p de aumento de dados, limiar 0,5, ShuffleNetV2 x0.5",
    "0.2": "aumento de dados (ColorJitter, std do GaussNoise, p)", "0.44": "std máx. do GaussNoise",
    "0.05": "shift do ShiftScaleRotate e matiz do ColorJitter", "0.1": "escala do ShiftScaleRotate; ruído ~0,1 min",
    "0.3": "p do CLAHE e da elástica", "2.0": "clip do CLAHE", "2.0.8": "versão da Albumentations",
    "0.2.0": "versão do scikit-multilearn", "1.0.0": "versão do protocolo", "1.3": "MNASNet 1.3",
    "1.0": "SqueezeNet 1.0", "0.02": "margem de não inferioridade (pré-registrada)",
    "0.015": "critério da ponte AMP (|ΔAUC| <= 0,015)", "0.80": "limite de sanidade da acurácia (P2)",
    "0.01": "ordem de grandeza da perda da MT em E/S (≈ ΔAUC −0,010/−0,011)",
    "0.854": "acurácia de prever sempre 'distância' (1609/1883)", "0.70": "mediana lida da Fig. 4 de Lira et al.",
    "0.3": "rede de 0,3 M parâmetros (ShuffleNetV2: 0,34 M)",
    "3.94": "milhões de parâmetros da cabeça KAN (3.936.000)", "0.26": "0,26 M (W2/MLP-128)",
    "60": "cm (distância curta típica)", "1.6": "grade estendida da efficient-kan (não citada)",
    "0.51": "acurácia balanceada ≈0,51 (agrupada)", "0.78": "acurácia balanceada ≈0,78 (Lira split)",
    "0.71": "AUC da Wav-KAN ≈0,71 na MLSD", "0.80": "AUC das MLPs ≈0,80 na MLSD",
}

falhas = []


def num_csv_json():
    """Todos os números de resultados/ (sem os oof.csv e o csv de dobras) e de img/protocolo/."""
    valores = []
    arquivos = [a for a in glob.glob(os.path.join(RES, "**", "*.csv"), recursive=True)
                if not a.endswith("oof.csv") and "folds_entrada_saida" not in a]
    arquivos += glob.glob(os.path.join(TCC, "img", "protocolo", "*.csv"))
    for a in arquivos:
        with open(a, newline="") as f:
            for linha in csv.reader(f):
                for cel in linha:
                    for t in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", cel):
                        valores.append(float(t))
    def visitar(o):
        if isinstance(o, dict):
            for v in o.values():
                visitar(v)
        elif isinstance(o, list):
            for v in o:
                visitar(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            valores.append(float(o))
        elif isinstance(o, str):
            for t in re.findall(r"-?\d+\.\d+", o):
                valores.append(float(t))
    for a in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True):
        visitar(json.load(open(a)))
    for t in re.findall(r"\\newcommand\{\\\w+\}\{([\d.]+)\}", open(os.path.join(TCC, "img", "protocolo", "figuras_macros.tex")).read()):
        valores.append(float(t))
    return valores


def numeros_paper():
    return [float(t) for t in re.findall(r"\d+\.\d+", open(PAPER, encoding="utf-8", errors="ignore").read())]


def limpar_tex(t):
    t = re.sub(r"\\dec\{([^}]*)\}", r"\1", t)
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\(label|ref|cite|input|texttt|includegraphics|figuraBib|url|refTab|refFig)\{[^}]*\}", " ", t)
    t = re.sub(r"\\figuraBib\[[^\]]*\]\{[^}]*\}", " ", t)
    t = re.sub(r"\{width=[^}]*\}", " ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*\}", " ", t)
    t = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", " ", t)
    t = re.sub(r"\[\d+pt\]|\d+(\.\d+)?(pt|mm|cm|em)\b", " ", t)
    return t


def so_ingles(t):
    """Troca cada \\EP{en}{pt} pelo primeiro argumento (as tabelas comuns são lidas em inglês)."""
    saida, i = [], 0
    while True:
        j = t.find("\\EP{", i)
        if j < 0:
            return "".join(saida) + t[i:]
        saida.append(t[i:j])
        args, k = [], j + 3
        for _ in range(2):
            prof, ini = 0, k
            while True:
                c = t[k]
                prof += (c == "{") - (c == "}")
                k += 1
                if prof == 0:
                    break
            args.append(t[ini + 1:k - 1])
        saida.append(args[0])
        i = k


def extrair(texto, lingua):
    """Lista de (número como texto normalizado, nº de casas, é_porcentagem, linha)."""
    out = []
    if "\\EP{" in texto:
        texto = so_ingles(texto)
    for i, linha in enumerate(texto.splitlines(), 1):
        l = limpar_tex(linha)
        if lingua == "pt":
            l = l.replace("{,}", ",")
            # no PT, ponto entre dígitos é milhar (2.554) ou versão/nome (2.0.8, 1.3): tratados à parte
            for m in re.finditer(r"(?<![\d.,])(\d+(?:\.\d+)+)(?!\d|[.,]\d)", l):
                tok = m.group(1)
                if tok in LISTA_BRANCA:
                    out.append((tok, None, False, i))
                elif re.fullmatch(r"\d{1,3}(\.\d{3})+", tok):
                    continue  # milhar
                else:
                    out.append(("PT_PONTO:" + tok, None, False, i))
            for m in re.finditer(r"(?<![\d.,])(\d+),(\d+)(\\?%)?", l):
                tok = f"{m.group(1)}.{m.group(2)}"
                out.append((tok, len(m.group(2)), bool(m.group(3)), i))
        else:
            l = re.sub(r"(\d)\{,\}(\d{3})", r"\1\2", l)
            l = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", l)
            for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)+)(\\?%)?", l):
                tok = m.group(1)
                if tok.count(".") > 1:
                    out.append((tok, None, False, i))
                    continue
                out.append((tok, len(tok.split(".")[1]), bool(m.group(2)), i))
    return out


def bate(tok, casas, pct, refs, refs_pct):
    if tok in LISTA_BRANCA:
        return True
    if casas is None:
        return False
    x = float(tok)
    tol = 0.5 * 10 ** (-casas) + 1e-9
    alvo = refs_pct if pct else refs
    import bisect
    j = bisect.bisect_left(alvo, x - tol)
    return j < len(alvo) and alvo[j] <= x + tol


def main():
    # 1. transcrição de Lira
    paper = open(PAPER, encoding="utf-8", errors="ignore").read()
    paper_esp = re.sub(r"\s+", " ", paper)
    cols = ["acc", "prec_M", "prec_m", "prec_W", "rec_M", "rec_m", "rec_W", "f1_M", "f1_m", "f1_W",
            "auc_M", "auc_m", "auc_W", "espec_M", "espec_m", "espec_W"]
    lira = list(csv.DictReader(open(os.path.join(RES, "lira2024_tabelas.csv"))))
    ok = 0
    for r in lira:
        seq = " ".join(f"{float(r[c]):.3f}" for c in cols)
        if seq in paper_esp:
            ok += 1
        else:
            falhas.append(f"Lira: linha {r['variante_lira']} ({r['tabela_lira']}) não encontrada no artigo")
    print(f"[Lira] {ok}/{len(lira)} linhas de lira2024_tabelas.csv aparecem idênticas no texto do artigo")

    refs = sorted({abs(v) for v in num_csv_json()} | {abs(v) for v in numeros_paper()})
    refs_pct = sorted({round(abs(v) * 100, 10) for v in refs} | {abs(v) for v in refs})

    # 2. números do texto
    total = sem_origem = 0
    por_arquivo = {}
    def lingua_de(rel, padrao):
        return "en" if rel.endswith("abstract.tex") else ("pt" if rel.endswith("resumo.tex") else padrao)
    arquivos = [(p, lingua_de(p, "en")) for p, _ in PARES] + [(p, lingua_de(p, "pt")) for _, p in PARES]
    arquivos += [(os.path.relpath(a, TCC), "en") for a in sorted(glob.glob(os.path.join(TCC, "comum", "tabelas", "*.tex")))]
    for rel, lingua in arquivos:
        nums = extrair(open(os.path.join(TCC, rel)).read(), lingua)
        por_arquivo[rel] = nums
        for tok, casas, pct, linha in nums:
            total += 1
            if tok.startswith("PT_PONTO:"):
                falhas.append(f"{rel}:{linha}: número com ponto decimal no PT: {tok[9:]}")
                continue
            if rel.endswith("tab_exploratorios.tex") or "Apendice_Exploratorios" in rel:
                continue  # números antigos, fora de resultados/ (Apêndice B)
            if not bate(tok, casas, pct, refs, refs_pct):
                sem_origem += 1
                print(f"  sem origem: {rel}:{linha}: {tok}{'%' if pct else ''}")
    print(f"[Texto] {total} números lidos; {sem_origem} sem origem em resultados/, no artigo ou na lista branca")

    # 3. paridade EN x PT
    difs = 0
    for en, pt in PARES:
        a = Counter(t for t, *_ in por_arquivo[en])
        b = Counter(t for t, *_ in por_arquivo[pt])
        if a != b:
            difs += 1
            print(f"  paridade {en} x {pt}: só EN {dict(a - b)}; só PT {dict(b - a)}")
    print(f"[Paridade] {len(PARES) - difs}/{len(PARES)} pares de capítulos com os mesmos números")

    # 4. afirmações: valores calculados das fontes têm de aparecer, formatados, no capítulo EN e no PT
    faltam = 0
    lista = afirmacoes()
    for cap, tipo, valor, rotulo in lista:
        for rel, lingua in ((PARES[cap][0], "en"), (PARES[cap][1], "pt")):
            if rel.endswith("abstract.tex"):
                lingua = "en"
            elif rel.endswith("resumo.tex"):
                lingua = "pt"
            alvo = formatar(tipo, valor, lingua)
            texto = normalizar(open(os.path.join(TCC, rel)).read(), lingua)
            if alvo not in texto:
                faltam += 1
                print(f"  afirmação não encontrada: {rel}: {rotulo} = {alvo!r}")
    print(f"[Afirmações] {len(lista)} valores conferidos nos capítulos EN e PT; {faltam} não encontrados")

    for f in falhas:
        print("FALHA", f)
    sys.exit(1 if falhas or sem_origem or difs or faltam else 0)


# ------------------------------------------------------------------------------------------ afirmações
def normalizar(t, lingua):
    t = re.sub(r"(?<!\\)%.*", "", t).replace("$", "").replace("\\,", "").replace("~", " ")
    if lingua == "pt":
        t = t.replace("{,}", ",")
    else:
        t = re.sub(r"(\d)\{,\}(\d{3})", r"\1,\2", t)
    return t


def formatar(tipo, v, lingua):
    if tipo == "int":
        s = f"{int(v):,}"
        return s.replace(",", ".") if lingua == "pt" else s
    if tipo.startswith("faixa"):
        sub = tipo.split(":")[1]
        return formatar(sub, v[0], lingua) + "--" + formatar(sub, v[1], lingua)
    if tipo.startswith("f"):
        s = f"{v:.{int(tipo[1:])}f}"
    elif tipo.startswith("s"):
        nd = int(tipo[1:].rstrip("n"))
        s = f"{v:+.{nd}f}"
        if float(s) == 0 and v != 0:
            s = f"{v:+.{nd + 1}f}"
        if s.startswith("+") and tipo.endswith("n"):
            s = s[1:]
    elif tipo.startswith("pct"):
        s = f"{v * 100:.{int(tipo[3:])}f}"
    elif tipo == "pm":
        s = v.replace(" ± ", " \\pm ")
    else:
        raise ValueError(tipo)
    return s.replace(".", ",") if lingua == "pt" else s


def afirmacoes():
    """(índice em PARES, tipo, valor, rótulo). Os valores saem de resultados/ e de img/protocolo/."""
    C1, C2, C3, C4, C5, C6, C7, AB, RE = 0, 1, 2, 3, 4, 5, 6, 7, 8
    ler = lambda n: list(csv.DictReader(open(os.path.join(RES, n))))
    js = lambda n: json.load(open(os.path.join(RES, n)))
    A = []
    add = lambda cap, tipo, v, r: A.append((cap, tipo, v, r))

    # vazamento
    tv = {(r["tarefa"], r["config"]): r for r in ler("tabela_vazamento.csv")}
    es_a, es_d = tv[("es", "resnet152__linear")], tv[("es", "resnet152__kan")]
    di_a, di_d = tv[("dist", "mlsd__resnet152__linear")], tv[("dist", "mlsd__resnet152__kan")]
    for cap in (C6, C7):
        add(cap, "f3", float(es_a["lira_auc"].split()[0]), "AUC E/S no split de Lira")
        add(cap, "f3", float(es_a["agrupado_auc"].split()[0]), "AUC E/S agrupada s42")
        add(cap, "f3", float(es_a["delta_auc"]), "ΔAUC vazamento E/S")
        add(cap, "f3", float(di_a["delta_auc"]), "ΔAUC vazamento MLSD linear")
        add(cap, "f3", float(di_d["delta_auc"]), "ΔAUC vazamento MLSD KAN")
        add(cap, "pct1", float(es_a["vazamento_medio"]), "vazamento E/S %")
        add(cap, "pct1", float(di_a["vazamento_medio"]), "vazamento MLSD %")
    for cap in (C6,):
        for r, campo in ((di_a, "lira_auc"), (di_a, "agrupado_auc"), (di_d, "lira_auc"), (di_d, "agrupado_auc")):
            add(cap, "f3", float(r[campo].split()[0]), f"{r['config']} {campo}")
    for cap in (C1, AB, RE):
        add(cap, "f3", float(es_a["delta_auc"]), "ΔAUC vazamento E/S")
        add(cap, "f3", float(di_d["delta_auc"]), "ΔAUC vazamento MLSD KAN")
        add(cap, "f3", float(di_a["delta_auc"]), "ΔAUC vazamento MLSD linear")
        add(cap, "pct1", float(es_a["vazamento_medio"]), "vazamento E/S %")

    # cabeças E/S e MLSD
    tf = {r["config"]: r for r in ler("tabela_final.csv") if r["painel"] == "A"}
    tfb = {r["config"]: r for r in ler("tabela_final.csv") if r["painel"] == "B"}
    add(C6, "f3", float(tf["resnet152__kan_pequena"]["auc"].split()[0]), "AUC (e)")
    add(C6, "f3", float(tf["resnet152__mlp_iso"]["auc"].split()[0]), "AUC (c)")
    dec = js("decisao.json")["comparacoes"]
    for k in ("a", "b", "c"):
        c = dec[k]
        for cap in ((C6, C7) if k == "c" else (C6,)):
            add(cap, "s3n" if c["delta_auc_oof"] < 0 else "s3", c["delta_auc_oof"], f"ΔAUC KAN−({k}) E/S")
            add(cap, "s3n", c["ic95"][0], f"IC inf KAN−({k}) E/S")
            add(cap, "s3", c["ic95"][1], f"IC sup KAN−({k}) E/S")
    tm = {r["config"]: r for r in ler("tabela_mlsd.csv") if r["painel"] == "A"}
    for k in ("linear", "mlp128", "mlp_iso", "kan"):
        add(C6, "pm", tm[f"mlsd__resnet152__{k}"]["auc"], f"AUC MLSD {k}")
    rec_e = [float(r["rec_encostado"]) for r in tm.values()]
    rec_c = [float(r["rec_curta"]) for r in tm.values()]
    add(C6, "faixa:f2", (min(rec_e), max(rec_e)), "revocação encostado (faixa)")
    add(C7, "faixa:pct0", (min(rec_e), max(rec_e)), "revocação encostado % (faixa)")
    add(C6, "faixa:f2", (min(rec_c), max(rec_c)), "revocação curta (faixa)")
    dm = js("decisao_mlsd.json")["comparacoes"]
    for k in ("a", "b", "c"):
        c = dm[k]
        add(C6, "s3n" if c["delta_auc_oof"] < 0 else "s3", c["delta_auc_oof"], f"ΔAUC KAN−({k}) MLSD")
        add(C6, "s3n", c["ic95"][0], f"IC inf KAN−({k}) MLSD")
        add(C6, "s3" if c["ic95"][1] > 0 else "s3n", c["ic95"][1], f"IC sup KAN−({k}) MLSD")
    add(C7, "s3n", dm["c"]["delta_auc_oof"], "ΔAUC KAN−(c) MLSD")

    # diagnóstico da grade da KAN
    dg = list(csv.DictReader(open(os.path.join(TCC, "img", "protocolo", "diagnostico_kan.csv"))))
    g = {(r["tarefa"], r["cabeca"], r["camada"]): r for r in dg}
    for chave in (("es", "d", "kan0"), ("es", "d", "kan1"), ("es", "e", "kan1")):
        add(C6, "faixa:pct0", (float(g[chave]["min"]), float(g[chave]["max"])), f"grade {chave} (faixa)")

    # backbones
    for cfg, rot in (("swin_b__linear", "Swin-B"), ("maxvit_t__linear", "MaxViT-T"), ("efficientnet_v2_l__linear", "EffV2-L"),
                     ("regnet_y_16gf__linear", "RegNetY"), ("densenet169__linear", "DenseNet"), ("shufflenet_v2_x0_5__linear", "ShuffleNet"),
                     ("vgg16__linear", "VGG16")):
        add(C6, "f3", float(tfb[cfg]["auc"].split()[0]), f"AUC E/S {rot}")
    add(C6, "pm", tfb["swin_b__linear"]["auc"], "AUC E/S Swin-B ± dp")
    add(C6, "f3", float(tfb["swin_b__linear"]["acc"].split()[0]), "acc E/S Swin-B")
    for cap in (C7, AB, RE):
        add(cap, "f3", float(tfb["swin_b__linear"]["auc"].split()[0]), "AUC E/S Swin-B")
    tmb = {r["config"]: r for r in ler("tabela_mlsd.csv") if r["painel"] == "B"}
    add(C6, "pm", tmb["mlsd__swin_b__linear"]["auc"], "AUC MLSD Swin-B")
    add(C7, "f3", float(tmb["mlsd__swin_b__linear"]["auc"].split()[0]), "AUC MLSD Swin-B")
    for cfg in ("maxvit_t__linear", "densenet169__linear", "efficientnet_v2_l__linear"):
        add(C6, "f3", float(tmb["mlsd__" + cfg]["auc"].split()[0]), f"AUC MLSD {cfg}")
    add(C6, "f3", float(tmb["mlsd__efficientnet_v2_l__linear"]["auc"].split()[1 + 1]), "dp MLSD EffV2-L")
    add(C6, "f3", float(tmb["mlsd__mnasnet1_3__linear"]["acc"].split()[0]), "acc MLSD MNASNet")

    # comparação com Lira (números deles e nossos)
    ces = {r["config"]: r for r in ler("tabela_comparacao_2024_es.csv")}
    cdi = {r["config"]: r for r in ler("tabela_comparacao_2024_dist.csv")}
    r = ces["resnet152__linear"]
    for campo in ("lira_acc", "lira_f1_M", "lira_auc_M_classes"):
        add(C6, "f3", float(r[campo]), f"Lira E/S ResNet152 {campo}")
        if campo != "lira_f1_M":
            add(C3, "f3", float(r[campo]), f"Lira E/S {campo} (cap. 3)")
    for campo in ("nosso_acc", "nosso_f1_M", "nosso_auc_M_classes", "nosso_auc_M_prob"):
        add(C6, "f3", float(r[campo].split()[0]), f"nosso E/S ResNet152 {campo}")
    t = ces["resnet152__kan"]
    for campo in ("nosso_f1_M", "nosso_auc_M_classes"):
        add(C6, "f3", float(t[campo].split()[0]), f"TransferKAN E/S {campo}")
    s = ces["swin_b__linear"]
    for campo in ("lira_acc", "lira_auc_M_classes"):
        add(C6, "f3", float(s[campo]), f"Lira Swin-B {campo}")
    add(C6, "f3", float(s["nosso_auc_M_classes"].split()[0]), "nosso Swin-B AUC_c")
    rd = cdi["mlsd__resnet152__linear"]
    for campo in ("lira_acc", "lira_auc_M_classes"):
        add(C3, "f3", float(rd[campo]), f"Lira MLSD {campo}")
    add(C6, "f3", float(rd["lira_auc_M_classes"]), "Lira MLSD ResNet152 AUC_c")
    acc_lira = [float(x["lira_acc"]) for x in cdi.values() if x["lira_acc"] and "squeezenet" not in x["config"]]
    add(C6, "f3", min(acc_lira), "menor acc Lira MLSD sem SqueezeNet")
    add(C6, "f3", max(acc_lira), "maior acc Lira MLSD")
    tk = cdi["mlsd__resnet152__kan"]
    add(C6, "f3", float(tk["nosso_acc"].split()[0]), "TransferKAN MLSD acc")
    add(C6, "f3", float(tk["nosso_auc_M_classes"].split()[0]), "TransferKAN MLSD AUC_c")
    add(C6, "f3", float(cdi["mlsd__squeezenet1_0__conv1x1"]["lira_auc_M_classes"]), "Lira SqueezeNet AUC_c")
    add(C6, "f3", float(cdi["mlsd__shufflenet_v2_x0_5__linear"]["lira_auc_M_classes"]), "Lira ShuffleNet AUC_c")
    lira_csv = {(x["tarefa"], x["backbone"]): x for x in ler("lira2024_tabelas.csv")}
    add(C3, "f3", float(lira_csv[("dist", "resnet152")]["auc_m"]), "Lira MLSD AUC micro")

    # transformers
    tr = js("comparacao_transkan_transformers.json")
    for k, c in tr.items():
        add(C6, "s3n", c["delta_auc_oof"], f"ΔAUC TransferKAN−{k}")
        if k != "dist__maxvit_t":
            add(C6, "s3n", c["ic95"][0], f"IC inf {k}")
            add(C6, "s3n", c["ic95"][1], f"IC sup {k}")

    # Wav-KAN
    w = js("decisao_wavkan.json")["contrastes"]
    c = w["es__W2_vs_b"]
    for v, rot in ((c["delta_auc_oof"], "ΔAUC W2−b"), (c["ic95_auc"][0], "IC inf"), (c["delta_f1_oof"], "ΔF1 W2−b"),
                   (c["ic95_f1"][0], "IC F1 inf"), (c["ic95_f1"][1], "IC F1 sup")):
        add(C6, "s3n", v, rot)
    add(C6, "s3", c["ic95_auc"][1], "IC AUC sup W2−b")
    c = w["es__W4_vs_W1"]
    for v in (c["delta_f1_oof"], c["ic95_f1"][0], c["ic95_f1"][1]):
        add(C6, "s3n", v, "W4−W1 ΔF1")
    c = w["dist__W2_vs_b"]
    for v in (c["delta_auc_oof"], c["ic95_auc"][0], c["ic95_auc"][1]):
        add(C6, "s3n", v, "MLSD W2−b ΔAUC")
    outros = [w[k]["delta_auc_oof"] for k in ("dist__W1_vs_c", "dist__W1_vs_d", "dist__W2_vs_d", "dist__W1_vs_b")]
    add(C6, "s3n", max(outros), "MLSD Wav-KAN contraste menos negativo")
    add(C6, "s3n", min(outros), "MLSD Wav-KAN contraste mais negativo")
    tw = {x["id"]: x for x in ler("tabela_wavkan.csv")}
    twd = {x["id"]: x for x in ler("tabela_wavkan_mlsd.csv")}
    add(C6, "pm", twd["W1"]["auc"], "AUC MLSD W1")
    add(C6, "pm", twd["W2"]["auc"], "AUC MLSD W2")
    aucs = [float(x["auc"].split()[0]) for x in tw.values()]
    add(C6, "faixa:f3", (min(aucs), max(aucs)), "faixa de AUC E/S das cabeças")
    add(C6, "f1", float(tw["W1"]["cabeca_fwd_bwd_ms_lote32"]), "cabeça W1 ms")
    add(C6, "f2", float(tw["c"]["cabeca_fwd_bwd_ms_lote32"]), "cabeça c ms")
    add(C6, "f0", float(tw["W1"]["cabeca_mem_pico_mb"]), "memória W1 MB")
    tempo = js("decisao_wavkan.json")["tempo"]
    add(C6, "f2", tempo["es__W2_vs_d"]["min_por_dobra"][0], "min/dobra W2")
    add(C6, "f2", tempo["es__W2_vs_d"]["min_por_dobra"][2], "min/dobra d")
    add(C6, "f1", tempo["es__W1_vs_c"]["variacao_dobra_pct"], "W1 mais lenta que c (%)")
    add(C6, "f1", (tempo["es__W1_vs_d"]["min_por_dobra"][0] / tempo["es__W1_vs_d"]["min_por_dobra"][2] - 1) * 100,
        "W1 mais lenta que d (%)")
    dw = list(csv.DictReader(open(os.path.join(RES, "diagnostico_wavkan.csv"))))
    add(C6, "f3", max(float(x["scale_media"]) for x in dw), "escala média máx.")
    add(C6, "f3", min(float(x["scale_dp"]) for x in dw), "dp escala mín.")
    add(C6, "f3", max(float(x["scale_dp"]) for x in dw), "dp escala máx.")
    add(C6, "f3", max(float(x["translation_dp"]) for x in dw), "dp translação máx.")

    # multitarefa
    m = js("decisao_multitarefa.json")
    tmt = {x["id"]: x for x in ler("tabela_multitarefa.csv")}
    for k in ("es__mt_kan_vs_st_kan", "es__mt_mlp_vs_st_mlp", "dist__mt_kan_vs_st_kan", "dist__mt_mlp_vs_st_mlp",
              "es__mt_kan_vs_mt_mlp", "dist__mt_kan_vs_mt_mlp"):
        c = m["contrastes"][k]
        add(C6, "s3n" if c["delta_auc_oof"] < 0 else "s3", c["delta_auc_oof"], f"ΔAUC {k}")
        add(C6, "s3n", c["ic95_auc"][0], f"IC inf {k}")
        add(C6, "s3n" if c["ic95_auc"][1] < 0 else "s3", c["ic95_auc"][1], f"IC sup {k}")
    for k in ("es__mt_kan_vs_st_kan", "es__mt_mlp_vs_st_mlp"):
        c = m["contrastes"][k]
        for v in (c["delta_f1_oof"], c["ic95_f1"][0], c["ic95_f1"][1]):
            add(C6, "s3n", v, f"ΔF1 {k}")
    for k in ("pipeline__mt_kan_vs_st_kan", "pipeline__mt_mlp_vs_st_mlp"):
        c = m["pipeline"][k]
        add(C6, "s3n", c["delta_f1_oof"], f"ΔF1 {k}")
        add(C6, "s3n", c["ic95_f1"][0], f"IC inf {k}")
        add(C6, "s3n" if c["ic95_f1"][1] < 0 else "s3", c["ic95_f1"][1], f"IC sup {k}")
    for i in ("st_kan", "mt_kan"):
        add(C6, "pm", tmt[i]["es_auc"], f"AUC E/S {i}")
        add(C6, "pm", tmt[i]["dist_auc"], f"AUC dist {i}")
        add(C6, "f2", float(tmt[i]["params_total_M"]), f"params {i}")
        add(C6, "f1", float(tmt[i]["min_por_dobra"]), f"min/dobra {i}")
        add(C6, "f2", float(tmt[i]["inf_ms_img_lote32"]), f"ms/img lote 32 {i}")
        add(C6, "f1", float(tmt[i]["inf_ms_img_lote1"]), f"ms/img lote 1 {i}")
    for i in ("st_mlp", "mt_mlp"):
        add(C6, "f2", float(tmt[i]["params_total_M"]), f"params {i}")
        add(C6, "f1", float(tmt[i]["min_por_dobra"]), f"min/dobra {i}")
    for i in ("mt_kan", "mt_mlp"):
        add(C6, "f3", float(tmt[i]["es_acc"].split()[0]), f"acc E/S {i}")
        add(C6, "f3", float(tmt[i]["pipeline_f1_macro_4_oof"]), f"F1 4 classes OOF {i}")
    add(C6, "f3", float(tmt["mt_kan"]["dist_acc"].split()[0]), "acc dist MT-KAN")
    add(C6, "f3", float(tmt["st_kan"]["dist_acc"].split()[0]), "acc dist ST-KAN")
    add(C6, "f3", float(tmt["mt_kan"]["dist_acc_balanceada"].split()[0]), "acc bal dist MT-KAN")
    add(C6, "f3", float(tmt["st_kan"]["dist_acc_balanceada"].split()[0]), "acc bal dist ST-KAN")
    for i in ("st_kan", "st_mlp"):
        add(C6, "f3", float(tmt[i]["pipeline_f1_macro_4_oof"]), f"F1 4 classes OOF {i}")
    add(C6, "f1", m["custo"]["mt_kan"]["reducao_params_pct"], "redução de parâmetros KAN (%)")
    add(C6, "f1", m["custo"]["mt_mlp"]["reducao_params_pct"], "redução de parâmetros MLP (%)")
    b32 = [b for b in m["benchmark_inferencia"]["tabela"] if b["tipo"] == "kan" and b["lote"] == 32][0]
    add(C6, "f0", b32["par_st_mem_pico_mb"], "memória inferência par ST")
    add(C6, "f0", b32["mt_mem_pico_mb"], "memória inferência MT")
    pipe = {(x["id"]): x for x in ler("pipeline_multitarefa.csv")}
    add(C6, "f3", float(pipe["mt_kan"]["recall_saida"]), "revocação de saída MT")
    add(C6, "f3", float(pipe["st_kan"]["recall_saida"]), "revocação de saída ST")

    # metodologia
    cal = js("calibracao_l4.json")["ponte"]
    add(C5, "f2", cal["fp32"]["acc"], "acc ponte fp32")
    add(C5, "f2", cal["amp"]["acc"], "acc ponte AMP")
    add(C5, "f4", cal["fp32"]["auc"], "AUC ponte fp32")
    add(C5, "f4", cal["amp"]["auc"], "AUC ponte AMP")
    add(C5, "f1", cal["fp32"]["treino_min"], "min ponte fp32")
    add(C5, "f1", cal["amp"]["treino_min"], "min ponte AMP")
    meta = js("dobras_meta.json")
    fr = [d["n_saida"] / d["n_img"] for d in meta["checagem"]["por_dobra"]]
    add(C5, "pct1", min(fr), "fração de saídas mín.")
    add(C5, "pct1", max(fr), "fração de saídas máx.")
    add(C5, "pct1", meta["checagem"]["frac_saida_global"], "fração de saídas global")
    add(C5, "int", meta["colisoes_chave_leguwoi"]["numeros_com_mais_de_um_ano"], "números em mais de um ano")
    add(C5, "int", meta["colisoes_chave_leguwoi"]["grupos_so_numero"], "grupos sem o ano")
    add(C5, "int", meta["checagem"]["n_img"], "imagens")
    add(C5, "int", meta["checagem"]["n_casos"], "casos")
    vl = ler("vazamento_lira_runs.csv")
    for cfg in ("resnet152__linear", "mlsd__resnet152__linear"):
        x = [r for r in vl if r["config"] == cfg][0]
        add(C5, "int", int(x["n_treino"]), f"treino split Lira {cfg}")
        add(C5, "int", int(x["n_teste"]), f"teste split Lira {cfg}")
    return A


if __name__ == "__main__":
    main()
