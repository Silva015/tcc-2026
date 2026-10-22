# TransferKAN sob um protocolo sem vazamento

Reavaliação das cabeças de Kolmogorov-Arnold na classificação de ferimentos por arma de fogo.
Trabalho de conclusão de curso na Universidade de Brasília (UnB), de Arthur Silva Carneiro, com orientação de Pedro Garcia Freitas.

> **English summary.** This repository holds the code, results and dissertation of a study on the TransferKAN model (an ImageNet-pre-trained ResNet-152 with a Kolmogorov-Arnold Network head) for classifying gunshot wounds as entry or exit and for estimating the shooting distance from real crime-scene photographs. Every model was retrained under a single leak-free protocol: case-grouped folds, one training recipe and decision rules fixed before the runs. Under that protocol the KAN head brings no gain over linear or MLP heads, Swin-B is the best backbone, and the hold-out used in the 2024 study inflates the AUC because images of the same case fall in training and test.

## A pergunta e o que se encontrou

A TransferKAN troca a cabeça densa de uma ResNet-152 pré-treinada no ImageNet por uma KAN. A pergunta é se essa troca melhora a classificação de ferimentos de **entrada × saída** e da **distância do disparo** (encostado, curta distância, distância), em fotografias reais de locais de crime da base FDCPUnBGunshotDB.

Para responder, todos os modelos foram treinados de novo num **protocolo único**:
- cinco dobras agrupadas por caso;
- a mesma receita de treino para todos;
- AUC calculada das probabilidades;
- regras de decisão escritas antes das rodadas.

| Pergunta | Resultado |
|---|---|
| O hold-out do estudo de 2024 infla os números? | Sim. No código público, ele estratifica pelo caso em vez de agrupar: 98,7% das imagens de teste têm o caso no treino. Com a mesma rede e a mesma receita, a AUC sobe +0,034 em entrada/saída e de +0,155 a +0,158 na distância. |
| A cabeça KAN melhora sobre cabeças linear e MLP? | Não. Em entrada/saída a AUC é 0,905 contra 0,904–0,906 das outras cabeças. Na distância, ela fica abaixo da MLP de mesmo tamanho. |
| Uma KAN com wavelets (Wav-KAN) muda isso? | Não. AUC igual em entrada/saída, bem pior na distância, e não é mais rápida. |
| Qual a melhor arquitetura entre as 12 retreinadas? | Swin-B: AUC 0,929 em entrada/saída e 0,838 na distância, acima da TransferKAN nas duas tarefas. |
| Uma rede para as duas tarefas substitui o par? | Em parte. Tem 47–50% menos parâmetros e é 2× mais rápida na inferência. É não inferior na distância e ~0,01 de AUC pior em entrada/saída, dentro da margem pré-registrada. |

A principal limitação é a classe **encostado**: são 42 imagens de 16 casos, quase nunca reconhecidas quando o caso não está no treino.

## Estrutura

```
notebooks/          código de treino e análise (protocolo único + notebooks históricos)
resultados/         saídas conferidas: tabelas (CSV/LaTeX), decisões, contrastes, figuras
tcc_dissertation/   dissertação em inglês (tex/) e em português (pt/tex/)
sibgrapi2026/       artigo curto e pôster do SIBGRAPI 2026
references/         artigo de Lira et al. (2025) em PDF e em texto
```

### `notebooks/`

O módulo `protocolo_comum.py` (versão 1.0.0) contém dados, dobras, receita, laço de treino, retomada e métricas. Ele foi congelado antes da primeira rodada, e cada execução grava o seu SHA-256. Os outros experimentos entram como plugins, sem editar o módulo:

| Notebooks | Plugin | O que fazem |
|---|---|---|
| `00`–`03`, `05` | — | dobras agrupadas, calibração na L4, ablação da cabeça, 12 backbones, análise de entrada/saída |
| `04`, `04b`, `05b` | `configs_mlsd.py` | distância do disparo, reprodução do hold-out de 2024, comparação com Lira et al. |
| `06`–`08` | `cabecas_wavkan.py` | cabeças Wav-KAN: benchmark, treino e análise |
| `09`, `10` | `multitarefa.py` | rede única com uma cabeça por tarefa: treino (duas abas) e análise |

Os notebooks `TransferKAN_*`, `PureKAN_*`, `PureConvKAN_*` e `ConvKANWithResNet_*`, com os logs `extracted_*.txt`, são da fase anterior ao protocolo único. Ficam como registro e não entram nas comparações.

### `resultados/`

Todos os números da dissertação vêm daqui:
- `tabela_*.csv|tex`, `decisao*.json` e `contrastes_*.csv|tex`;
- `lira2024_tabelas.csv`, a transcrição das Tabelas 4 e 5 de Lira et al.

As subpastas `wavkan/`, `wavkan_mlsd/` e `multitarefa/` guardam as rodadas por unidade e as predições fora da dobra. As predições por unidade do protocolo base (cabeças, backbones e MLSD) ficaram no Google Drive do projeto e não estão aqui.

## Dados

A base FDCPUnBGunshotDB (Lira et al., *International Journal of Legal Medicine*, 2025) não tem licença de redistribuição. Por isso, **nenhuma imagem da base está neste repositório**. Os notebooks clonam a base num commit fixo (`f679389`) em cada sessão. Quem usar a base deve citar Lira et al. (2025).

## Reproduzir os experimentos

Os notebooks rodam no Google Colab com GPU NVIDIA L4 e instalam as próprias dependências, com versões fixadas (Albumentations 2.0.8, `efficient-kan` no commit `7b6ce1c`). A ordem é:
1. `00` (dobras) e `01` (calibração e checagem da precisão mista);
2. `02`, `03` e `04`/`04b` (treino);
3. `05`/`05b` (análise);
4. depois, `06`–`08` para a Wav-KAN e `09`–`10` para a multitarefa.

A retomada é por época, então uma sessão que cair continua de onde parou.

## Compilar a dissertação

Veja [`tcc_dissertation/README.md`](tcc_dissertation/README.md). Em resumo, só compila com **pdfLaTeX**, e as tabelas são geradas a partir de `resultados/` por script.

## Licença

Apache 2.0 (veja [`LICENSE.md`](LICENSE.md)). A licença não cobre a base de imagens nem o artigo de Lira et al. em `references/`.
