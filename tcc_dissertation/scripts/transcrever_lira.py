#!/usr/bin/env python3
"""Transcreve as 59 linhas das Tabelas 4 e 5 de Lira et al. (2025, IJLM).

Lê ``references/2024_paper.txt`` (texto extraído do PDF), grava
``resultados/lira2024_tabelas_completas.csv`` e confere a transcrição:

* 59 linhas por tabela, na ordem do artigo, cada uma com 16 números;
* as 24 linhas de ``resultados/lira2024_tabelas.csv`` (já conferidas à mão) idênticas;
* identidades internas das tabelas (colunas micro = acurácia etc.).

Sai com código 0 só se tudo conferir. Uso: ``python3 -I transcrever_lira.py [--checar]``
(``--checar`` não grava; compara com o CSV existente).
"""

from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ARTIGO = RAIZ / "references" / "2024_paper.txt"
SAIDA = RAIZ / "resultados" / "lira2024_tabelas_completas.csv"
CONFERIDAS = RAIZ / "resultados" / "lira2024_tabelas.csv"

METRICAS = [
    "acc", "prec_M", "prec_m", "prec_W", "rec_M", "rec_m", "rec_W", "f1_M", "f1_m", "f1_W",
    "auc_M", "auc_m", "auc_W", "espec_M", "espec_m", "espec_W",
]

# Ordem das linhas nas Tabelas 4 e 5: (classe do leguwoi, construtor do torchvision que a
# classe chama em src/models/torchvision_models.py, rótulo no artigo).
# Atenção: "Regnet X16GF/X32GF/Y16GF/Y32GF" do artigo são regnet_*_1_6gf e regnet_*_3_2gf.
LINHAS = [
    ("Alexnet", "alexnet", "Alexnet"),
    ("Densenet121", "densenet121", "DenseNet 121"),
    ("Densenet161", "densenet161", "DenseNet 161"),
    ("Densenet169", "densenet169", "DenseNet 169"),
    ("Densenet201", "densenet201", "DenseNet 201"),
    ("EfficientnetB0", "efficientnet_b0", "EfficientNet B0"),
    ("EfficientnetB1", "efficientnet_b1", "EfficientNet B1"),
    ("EfficientnetB2", "efficientnet_b2", "EfficientNet B2"),
    ("EfficientnetB3", "efficientnet_b3", "EfficientNet B3"),
    ("EfficientnetB4", "efficientnet_b4", "EfficientNet B4"),
    ("EfficientnetB5", "efficientnet_b5", "EfficientNet B5"),
    ("EfficientnetB6", "efficientnet_b6", "EfficientNet B6"),
    ("EfficientnetB7", "efficientnet_b7", "EfficientNet B7"),
    ("EfficientnetV2L", "efficientnet_v2_l", "EfficientNet V2L"),
    ("EfficientnetV2M", "efficientnet_v2_m", "EfficientNet V2M"),
    ("EfficientnetV2S", "efficientnet_v2_s", "EfficientNet V2S"),
    ("MaxVit", "maxvit_t", "MaxVit"),
    ("MNASnet05", "mnasnet0_5", "MNASNet 5"),
    ("MNASnet075", "mnasnet0_75", "MNASNet 75"),
    ("MNASnet10", "mnasnet1_0", "MNASNet 10"),
    ("MNASnet13", "mnasnet1_3", "MNASNet 13"),
    ("MobileNetV2", "mobilenet_v2", "MobileNet V2"),
    ("MobileNetV3Large", "mobilenet_v3_large", "MobileNet V3Large"),
    ("MobileNetV3Small", "mobilenet_v3_small", "MobileNet V3Small"),
    ("RegnetX16GF", "regnet_x_1_6gf", "Regnet X16GF"),
    ("RegnetX32GF", "regnet_x_3_2gf", "Regnet X32GF"),
    ("RegnetX400MF", "regnet_x_400mf", "Regnet X400MF"),
    ("RegnetX800MF", "regnet_x_800mf", "Regnet X800MF"),
    ("RegnetX8GF", "regnet_x_8gf", "Regnet X8GF"),
    ("RegnetY16GF", "regnet_y_1_6gf", "Regnet Y16GF"),
    ("RegnetY32GF", "regnet_y_3_2gf", "Regnet Y32GF"),
    ("RegnetY400MF", "regnet_y_400mf", "Regnet Y400MF"),
    ("RegnetY800MF", "regnet_y_800mf", "Regnet Y800MF"),
    ("RegnetY8GF", "regnet_y_8gf", "Regnet Y8GF"),
    ("Resnet152", "resnet152", "ResNet 152"),
    ("Resnet101", "resnet101", "ResNet 101"),
    ("Resnet50", "resnet50", "ResNet 50"),
    ("Resnet34", "resnet34", "ResNet 34"),
    ("Resnet18", "resnet18", "ResNet 18"),
    ("ShuffleNetV2x05", "shufflenet_v2_x0_5", "ShuffleNet V2x05"),
    ("ShuffleNetV2x10", "shufflenet_v2_x1_0", "ShuffleNet V2x10"),
    ("ShuffleNetV2x15", "shufflenet_v2_x1_5", "ShuffleNet V2x15"),
    ("ShuffleNetV2x20", "shufflenet_v2_x2_0", "ShuffleNet V2x20"),
    ("Squeezenet10", "squeezenet1_0", "SqueezeNet 10"),
    ("Squeezenet11", "squeezenet1_1", "SqueezeNet 11"),
    ("SwinTransformerB", "swin_b", "SwinT B"),
    ("SwinTransformerS", "swin_s", "SwinT S"),
    ("SwinTransformerT", "swin_t", "SwinT T"),
    ("SwinTransformerV2B", "swin_v2_b", "SwinT V2B"),
    ("SwinTransformerV2S", "swin_v2_s", "SwinT V2S"),
    ("SwinTransformerV2T", "swin_v2_t", "SwinT V2T"),
    ("VGG11", "vgg11", "VGG 11"),
    ("VGG13", "vgg13", "VGG 13"),
    ("VGG16", "vgg16", "VGG 16"),
    ("VGG19", "vgg19", "VGG 19"),
    ("ViTB16", "vit_b_16", "ViT B16"),
    ("ViTB32", "vit_b_32", "ViT B32"),
    ("ViTL16", "vit_l_16", "ViT L16"),
    ("ViTL32", "vit_l_32", "ViT L32"),
]

# Última parte do rótulo impresso em cada linha (confere a ordem contra o texto).
SUFIXO = {m: r.split()[-1] if " " in r else None for m, _, r in LINHAS}

# backbone em lira2024_tabelas.csv -> classe do leguwoi
DOZE = {
    "alexnet": "Alexnet", "mobilenet_v2": "MobileNetV2", "mnasnet1_3": "MNASnet13",
    "shufflenet_v2_x0_5": "ShuffleNetV2x05", "squeezenet1_0": "Squeezenet10",
    "densenet169": "Densenet169", "maxvit_t": "MaxVit", "swin_b": "SwinTransformerB",
    "efficientnet_v2_l": "EfficientnetV2L", "vgg16": "VGG16", "regnet_y_16gf": "RegnetY16GF",
    "resnet152": "Resnet152",
}

# Linhas em que o texto do artigo não satisfaz as identidades das demais 117 linhas.
INCONSISTENTES_NO_ARTIGO = {
    ("dist", "EfficientnetB5"): (
        "acc 0.869 e AUC micro 0.902 implicam espec. micro 0.934-0.935, mas a linha traz "
        "0.667 / 0.943 / 0.114 nas colunas de especificidade, iguais às da linha B6 logo abaixo"
    ),
}

_LINHA = re.compile(r"^(.*?)((?:\s*\d\.\d{3}){16})\s*$")


def extrair(texto: str) -> dict:
    """{'es': [(rotulo_no_texto, [16 strings])...], 'dist': [...]}."""
    linhas = texto.splitlines()
    ini = {}
    for i, l in enumerate(linhas):
        if l.startswith("Table 4 Performance metrics"):
            ini["es"] = i
        elif l.startswith("Table 5 Performance metrics"):
            ini["dist"] = i
    out = {}
    for tarefa, i0 in ini.items():
        achadas = []
        for l in linhas[i0 + 1:]:
            if l.startswith("Considering each metric separately"):
                break
            m = _LINHA.match(l)
            if m:
                achadas.append((m.group(1).strip(), m.group(2).split()))
        out[tarefa] = achadas
    return out


def montar() -> list[dict]:
    tabelas = extrair(ARTIGO.read_text(encoding="utf-8"))
    registros = []
    for tarefa, nome in (("es", "Table 4"), ("dist", "Table 5")):
        achadas = tabelas[tarefa]
        if len(achadas) != len(LINHAS):
            raise SystemExit(f"{nome}: {len(achadas)} linhas no texto, esperadas {len(LINHAS)}")
        for (modelo, tv, rotulo), (texto, nums) in zip(LINHAS, achadas):
            suf = SUFIXO[modelo]
            if suf is not None and not texto.replace(" ", "").endswith(suf.replace(" ", "")):
                raise SystemExit(f"{nome}: linha de {modelo} tem rótulo '{texto}', esperado final '{suf}'")
            registros.append({
                "tarefa": tarefa, "tabela_lira": nome, "modelo_leguwoi": modelo,
                "torchvision": tv, "rotulo_artigo": rotulo, **dict(zip(METRICAS, nums)),
            })
    return registros


def conferir(registros: list[dict]) -> list[str]:
    erros = []
    por_chave = {(r["tarefa"], r["modelo_leguwoi"]): r for r in registros}
    # 1) as 24 linhas já conferidas
    n = 0
    with CONFERIDAS.open(encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            r = por_chave[(linha["tarefa"], DOZE[linha["backbone"]])]
            n += 1
            for m in METRICAS:
                if float(linha[m]) != float(r[m]):
                    erros.append(f"{linha['tarefa']}/{linha['backbone']}: {m} = {r[m]}, conferido {linha[m]}")
    if n != 24:
        erros.append(f"lira2024_tabelas.csv tem {n} linhas, esperadas 24")
    # 2) identidades internas
    tol = 0.0011
    for r in registros:
        v = {m: float(r[m]) for m in METRICAS}
        quem = f"{r['tarefa']}/{r['modelo_leguwoi']}"
        for m in ("prec_m", "rec_m", "f1_m"):
            if v[m] != v["acc"]:
                erros.append(f"{quem}: {m} != acc")
        if r["tarefa"] == "es":
            for m in ("auc_M", "auc_m", "auc_W", "espec_M"):
                if v[m] != v["rec_M"]:
                    erros.append(f"{quem}: {m} != rec_M")
            for m in ("espec_m", "espec_W"):
                if v[m] != v["acc"]:
                    erros.append(f"{quem}: {m} != acc")
        elif (r["tarefa"], r["modelo_leguwoi"]) in INCONSISTENTES_NO_ARTIGO:
            print("AVISO (inconsistência do próprio artigo):", quem, "-",
                  INCONSISTENTES_NO_ARTIGO[(r["tarefa"], r["modelo_leguwoi"])])
        else:
            # 3 classes: espec. micro = 1 - (1-acc)/2 ; AUC micro de classes = (acc + espec_m)/2
            if abs(v["espec_m"] - (1 - (1 - v["acc"]) / 2)) > tol:
                erros.append(f"{quem}: espec_m não é 1-(1-acc)/2")
            if abs(v["auc_m"] - (v["acc"] + v["espec_m"]) / 2) > tol:
                erros.append(f"{quem}: auc_m não é (acc+espec_m)/2")
    return erros


def para_csv(registros: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(registros[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(registros)
    return buf.getvalue()


def main() -> int:
    registros = montar()
    erros = conferir(registros)
    for e in erros:
        print("ERRO:", e)
    texto = para_csv(registros)
    if "--checar" in sys.argv:
        if not SAIDA.exists() or SAIDA.read_text(encoding="utf-8") != texto:
            print("ERRO: resultados/lira2024_tabelas_completas.csv difere da transcrição")
            return 1
    elif not erros:
        SAIDA.write_text(texto, encoding="utf-8")
        print(f"Gravado {SAIDA.relative_to(RAIZ)}: {len(registros)} linhas")
    print("OK" if not erros else f"{len(erros)} erro(s)")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
