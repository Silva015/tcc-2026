# Dissertação (TCC)

*TransferKAN Under a Leak-Free Protocol: Re-evaluating Kolmogorov-Arnold Heads for Gunshot Wound Classification* / *TransferKAN sob um protocolo sem vazamento: reavaliação das cabeças de Kolmogorov-Arnold na classificação de ferimentos por arma de fogo*.

Há duas versões com o mesmo conteúdo, na classe LaTeX do CIC/UnB:

| Versão | Entrada | Capítulos | Classe |
|---|---|---|---|
| Inglês | `monografia.tex` | `tex/` | `UnB-CIC-English.cls` |
| Português | `pt/monografia.tex` | `pt/tex/` | `UnB-CIC.cls` |

## Compilar

Só com **pdfLaTeX**: XeLaTeX e Tectonic falham na classe.

Inglês (o `make` também limpa os arquivos auxiliares):

```bash
cd tcc_dissertation && make
```

Português, de dentro de `pt/`:

```bash
cd tcc_dissertation/pt && pdflatex monografia && bibtex monografia && pdflatex monografia && pdflatex monografia
```

Não defina `TEXINPUTS=..:` ao compilar o português, senão o pdfLaTeX pega o `monografia.tex` em inglês da pasta de cima. A pasta `pt/` usa links simbólicos para `img/`, `comum/`, a classe e a bibliografia.

## Tabelas, figuras e números

Tabelas e figuras são **compartilhadas** pelas duas versões, na pasta `comum/`:
- o texto em cada língua vem do macro `\EP{inglês}{português}`;
- os decimais das tabelas usam `\dec{}`, que vira vírgula no português.

Os scripts ficam em `scripts/` e usam só a biblioteca padrão do Python:

| Script | O que faz |
|---|---|
| `gerar_tabelas.py` | Gera `comum/tabelas/*.tex` a partir de `resultados/`. Não edite as tabelas à mão: rode o script de novo. |
| `preparar_dados_figuras.py --drive DIR` | Gera os dados das figuras em `img/protocolo/`. Lê as predições do protocolo base que ficaram no Drive e confere tudo contra `resultados/` antes de gravar. |
| `conferir_numeros.py` | Confere a transcrição das tabelas de Lira et al. contra o artigo e cerca de 220 valores do texto contra `resultados/`, nas duas línguas. Também confere se o inglês e o português citam os mesmos números. |

Depois de qualquer mudança em `resultados/`:

```bash
python3 tcc_dissertation/scripts/gerar_tabelas.py
python3 tcc_dissertation/scripts/conferir_numeros.py
```

## Pendente

A data da defesa e a banca ainda não foram definidas. A capa mostra só o ano, e a folha de aprovação está desligada em `monografia.tex` e em `pt/monografia.tex`.

Quando houver banca:
1. remova a linha `\renewcommand{\folhaaprovacao}{}` e a linha `\makeatletter…` logo abaixo dela;
2. ponha `\diamesano{dia}{mês}{ano}` com a data da defesa;
3. acrescente um `\membrobanca{nome}{instituição}` para cada membro.
