# Plano P4: reescrever a dissertação (EN e PT) com os resultados do protocolo único

Plano para uma sessão nova. Não treina nada: só reescreve `tcc_dissertation/` (inglês em `tex/`, português em `pt/tex/`) com os números que o P1 já produziu e verificou.

## 0. Antes de começar

**Leia, nesta ordem:**
1. este plano;
2. `resultados/frases_dissertacao.md` e `resultados/frase_manuscrito.md` (rascunhos automáticos; veja as ressalvas da seção 4);
3. `resultados/decisao.json`, `resultados/decisao_mlsd.json`, `resultados/comparacao_transkan_transformers.json`;
4. as tabelas `resultados/tabela_*.csv` e `.tex` e as figuras `resultados/*.png`;
5. `docs/planos/P1_protocolo.md`, que registra a regra de decisão definida **antes** dos resultados (14/10, commit `c2a957c`);
6. `auditoria-consistencia.md` (o que estava errado no texto antigo) e `references/2024_paper.txt` (Lira et al.).

**Como compilar** (a memória do projeto `dissertation-build` também tem isto):
- só **pdfLaTeX**: XeLaTeX e Tectonic quebram na classe;
- EN: `make` em `tcc_dissertation/`;
- PT: `pdflatex monografia` dentro de `pt/`, depois `bibtex`, depois `pdflatex` duas vezes. **Não** defina `TEXINPUTS=..:`, senão ele compila o inglês;
- a chave do artigo de 2024 é `Lira2025deep`;
- macros da classe: `\figuraBib`, `\figura`, `\refFig`, `\refTab`.

**Regras do repositório:**
- `git add` só por caminho;
- confira `.git/index.lock` antes de commitar;
- não mexa em `notebooks/`, `resultados/`, `sibgrapi2026/` nem nos arquivos de outras sessões;
- `tcc_dissertation/monografia_EN.pdf` e `monografia_PT.pdf` aparecem apagados no diretório de trabalho do usuário: não os recrie nem os commite sem perguntar.

## 1. Decisões que o usuário precisa tomar primeiro

Pergunte antes de escrever. A recomendação vem primeiro em cada item.

1. **Modelos exploratórios antigos** (ResKANet-18, PureConvKAN, PureKAN, hoje no Cap. 6):
   - (rec.) mover para um apêndice de "experimentos exploratórios", com o aviso de que usaram hold-out, outra resolução e outra receita e não são comparáveis;
   - ou remover.
2. **Título e tese:**
   - (rec.) manter "TransferKAN" no título, mas com a tese de uma avaliação sem vazamento (ex.: *"TransferKAN under a leak-free protocol: …"*);
   - ou renomear o trabalho inteiro.
3. **P2 (multitarefa) e P3 (Wav-KAN):**
   - (rec.) escrever agora só com o P1 e deixar "trabalhos futuros" citando os dois;
   - ou esperar os resultados deles.
4. **Semente 44 da Swin-B e da MaxViT-T no MLSD:**
   - (rec.) reportar com 2 sementes e dizer isso na legenda;
   - ou rodar antes (~2,6 unidades), o que obriga a rodar o 05b de novo e atualizar `resultados/`.

## 2. A nova tese em cinco pontos

1. **O hold-out de 2024 vaza casos.** O `iterative_split` do código de Lira estratifica por número do caso em vez de agrupar. Reproduzindo o método (3 partições), 98,7% (E/S) e 98,4% (MLSD) das imagens de teste têm o caso no treino. Com a mesma receita, isso infla a AUC em **+0,034** (E/S) e **+0,155 a +0,158** (MLSD).
2. **Protocolo único, sem vazamento:**
   - dobras agrupadas por caso;
   - 12 backbones pré-treinados retreinados com a mesma receita;
   - AUC calculada de probabilidades;
   - regra de decisão definida antes dos resultados.
3. **A cabeça KAN não é a fonte do ganho:**
   - **E/S:** equivalente às cabeças linear e MLP (desfecho B);
   - **MLSD:** abaixo da MLP isoparamétrica (desfecho C) e equivalente à MLP-128 e à linear.
4. **O que explica a diferença para os números de 2024** é a partição, o pré-treino e a receita. A melhor arquitetura no protocolo único é a **Swin-B**, que supera a TransKAN nas duas tarefas.
5. **"Encostado" quase não é aprendido** quando o caso não aparece no treino (recall 0,13–0,17 na ResNet-152). Essa é a principal limitação.

A hipótese antiga ("trocar a MLP pela KAN melhora") vira uma **pergunta de pesquisa com resposta negativa**. Não a apague: mostre o teste dela.

## 3. Fatos que o texto deve usar (todos verificados)

Média ± desvio-padrão entre as 5 dobras; cada dobra é a média das sementes disponíveis. A AUC vem de probabilidades (OvR macro no MLSD). AUC_c = (recall + especificidade)/2, que é a AUC que Lira calcula a partir de classes; use só para comparar com Lira.

**Dados e dobras (`resultados/dobras_meta.json`):**
- 2.554 imagens: 2.551 do CSV de Lira + 3 ausentes dele, com 1 anomalia anotada;
- 486 casos (ano.caso; 417 se o ano for ignorado, como no código de Lira);
- 1.883 entradas e 671 saídas;
- MLSD só nas entradas: 1.609 distância, 232 curta e 42 encostado;
- `StratifiedGroupKFold(5)`, semente 42, `y=strat4`, grupos = caso, com ≥2 casos de encostado por dobra de teste;
- tamanhos de teste: 506 / 459 / 531 / 538 / 520.

**Receita (`notebooks/protocolo_comum.py`, `RECEITA`):**
- igual para todas as configs, inclusive as duas TransferKAN;
- ResNet-152 ImageNet → Dropout 0,5 → cabeça;
- fase 1: 15 épocas só na cabeça, AdamW lr 1e-3, wd 1e-2;
- fase 2: 45 épocas na rede toda, lr 1e-5, wd 1e-3;
- cosseno, lote 32, `drop_last`;
- CE com pesos n/(K·n_c);
- Albumentations com os valores que de fato rodaram;
- avalia-se a **última época**, sem validação interna;
- AMP fp16.
- **Ponte fp32 × AMP** (`calibracao_l4.json`): fp32 deu acc 84,98% e AUC 0,9078 em 14,9 min por dobra; AMP deu 84,39% e 0,9076 em 8,1 min (Δacc 0,59 pp, ΔAUC 0,0002).
- Hardware: NVIDIA L4.

**Cabeças:**

| | Cabeça | Parâmetros E/S | Parâmetros MLSD |
|---|---|---|---|
| (a) | linear | 4.098 | 6.147 |
| (b) | MLP-128 | 262.530 | 262.659 |
| (c) | MLP isoparamétrica | 3.935.871 | 3.937.791 |
| (d) | KAN [2048,128,K], grade 10, ordem 3 = TransferKAN | 3.936.000 | 3.937.920 |
| (e) | KAN-8 (só E/S, s42) | 246.000 | — |

**E/S, cabeças (3 sementes; `tabela_final.csv`):**
- AUC: (a) 0,904±0,013; (b) 0,905±0,014; (c) 0,906±0,014; (d) 0,905±0,013; (e) 0,903±0,014;
- acc 0,846 em todas.
- **Decisão B** (`decisao.json`): a KAN contra a melhor MLP (c) dá ΔAUC OOF −0,001, IC 95% [−0,002; +0,000], e a KAN vence 2 de 5 dobras. Contra (b) e (a), os IC também incluem zero.

**E/S, backbones (cabeça linear; s42; Swin-B e MaxViT-T com 3 sementes):**

| Backbone | AUC |
|---|---|
| Swin-B | **0,929±0,011** (acc 0,884) |
| MaxViT-T | 0,917 |
| EfficientNetV2-L | 0,916 |
| RegNetY-16GF | 0,906 |
| DenseNet169 | 0,904 |
| ResNet152 | 0,904 |
| VGG16 | 0,891 |
| MobileNetV2 | 0,877 |
| MNASNet1.3 | 0,872 |
| SqueezeNet1.0 | 0,866 |
| AlexNet | 0,859 |
| ShuffleNetV2 x0.5 | 0,839 |

**MLSD, cabeças (3 sementes; `tabela_mlsd.csv`):**
- AUC: (a) 0,785±0,087; (b) 0,802±0,062; (c) 0,807±0,055; (d) 0,798±0,058;
- acurácia balanceada 0,51–0,52;
- recall de encostado 0,13–0,17, só descritivo: são 4–13 imagens por dobra.
- **Decisão C** (`decisao_mlsd.json`):
  - KAN × (c): ΔAUC −0,011, IC [−0,021; −0,003], 1 de 5 dobras;
  - KAN × (b): −0,005, IC [−0,012; +0,003];
  - KAN × (a): +0,008, IC [−0,005; +0,023].

**MLSD, backbones:**

| Backbone | AUC |
|---|---|
| Swin-B | **0,838±0,054** (sementes 42 e 43) |
| MaxViT-T | 0,828 (sementes 42 e 43) |
| DenseNet169 | 0,821 |
| AlexNet | 0,806 |
| MobileNetV2 | 0,806 |
| SqueezeNet1.0 | 0,805 |
| ShuffleNetV2 x0.5 | 0,776 |
| VGG16 | 0,776 |
| RegNetY-16GF | 0,773 |
| EfficientNetV2-L | 0,744±0,133 |
| MNASNet1.3 | 0,710 |

A MNASNet colapsou (acc 0,451): comente como instabilidade, não como resultado.

**TransKAN × transformers (`comparacao_transkan_transformers.json`, bootstrap de casos, 2.000 reamostragens):**

| Tarefa | Contra | ΔAUC | IC 95% | Dobras vencidas |
|---|---|---|---|---|
| E/S | Swin-B | −0,023 | [−0,035; −0,011] | 0/5 |
| E/S | MaxViT-T | −0,010 | [−0,020; −0,001] | 0/5 |
| MLSD | Swin-B | −0,033 | [−0,064; −0,0004] | 1/5 |
| MLSD | MaxViT-T | −0,030 | [−0,059; +0,001] | 1/5 |

**Vazamento (`tabela_vazamento.csv`; agrupado = semente 42, por isso 0,906 e não 0,904):**

| Tarefa e modelo | AUC no split de Lira | AUC agrupada |
|---|---|---|
| E/S, linear | 0,940±0,003 | 0,906±0,013 |
| E/S, KAN | 0,940±0,003 | 0,906±0,011 |
| MLSD, linear | 0,937±0,004 | 0,779±0,088 |
| MLSD, KAN | 0,942±0,003 | 0,788±0,063 |

No MLSD, a acurácia balanceada cai de ~0,78 para ~0,51. Figura: `vazamento_vs_agrupado.png`.

**Comparação com Lira (`tabela_comparacao_2024_*.csv`, Tabelas 4 e 5 do artigo, transcrição conferida):**

| Tarefa | Fonte | Acc | F1_M | AUC_c | AUC_p |
|---|---|---|---|---|---|
| E/S | ResNet152, Lira | 0,869 | 0,824 | 0,821 | — |
| E/S | ResNet152, protocolo único | 0,846 | 0,805 | 0,814 | 0,904 |
| MLSD | ResNet152, Lira | 0,925 | 0,701 | 0,733 | — |
| MLSD | ResNet152, protocolo único | 0,815 | 0,488 | 0,670 | 0,785 |

- Quase todos os modelos de Lira foram treinados do zero, por isso, no protocolo único, os pré-treinados sobem muito em E/S (ex.: Swin-B de 0,728 para 0,884 de acurácia).
- Cite também a k-fold do próprio Lira (Fig. 4, mediana ~0,70) como evidência independente de que o hold-out é otimista.

**Números antigos que devem sumir do texto:**
- TransferKAN E/S: acc 0,860, AUC 0,908 (macro) e 0,928 (micro);
- TransferKAN MLSD: acc 0,873, AUC macro 0,749;
- os do pôster: 0,871, 0,847, 0,893, 0,924;
- a frase "melhor época escolhida no teste";
- o MLSD em fase única com grade 5 e torchvision;
- SMOTE+ENN, se aparecer.

## 4. Ressalvas de redação (achadas na verificação de 19/10)

Os números das saídas do 05 e do 05b batem entre si em 401 conferências cruzadas. Os **rascunhos de frase** têm três exageros que o texto final deve corrigir:

1. *"TransKAN's advantage over the baselines of Lira et al. is explained by the pre-trained backbone and the two-phase recipe"* (`decisao.json`, `frase_manuscrito.md`, `frases_dissertacao.md`). Contra os números publicados por Lira (com vazamento), a TransKAN:
   - em E/S, fica acima de 11 das 12 linhas, mas **abaixo da ResNet152 de Lira** (acc 0,846 × 0,869; AUC_c 0,816 × 0,821);
   - no MLSD, fica abaixo de 11 das 12 em acurácia e acima de 11 das 12 em AUC_c.

   **Escreva assim:** "no protocolo único, os backbones pré-treinados igualam ou superam a TransKAN; as diferenças em relação aos números de Lira et al. vêm da partição, do pré-treino e da receita, não da cabeça KAN".
2. **MLSD:** não escreva "abaixo das MLPs". A KAN fica significativamente abaixo só da MLP isoparamétrica; contra a MLP-128 e a linear, os IC incluem zero.
3. **Vazamento:** use as taxas por tarefa (98,7% e 98,4%; 98,6% só como média) e deixe claro que o "agrupado" dessa tabela é a semente 42.

**O que pode ser afirmado** (`frases_dissertacao.md`, "Pode afirmar?"): as quatro afirmações são `false` nas duas tarefas:
- só trocar a cabeça melhora;
- novo estado da arte;
- supera os transformers;
- eficiência de parâmetros.

## 5. O que muda em cada capítulo

Escreva primeiro o EN e espelhe no PT, capítulo a capítulo. Commite cada par EN+PT separadamente.

| Capítulo (EN / PT) | O que fazer |
|---|---|
| `abstract.tex` / `resumo.tex` (os dois existem em `tex/` e em `pt/tex/`) | Reescrever com os 5 pontos da seção 2 e 3–4 números: o +0,155 do vazamento, a decisão B/C, o 0,929 da Swin-B |
| 1 Introduction / Introdução | Objetivos → perguntas de pesquisa (RQ1: o vazamento infla os números de 2024? RQ2: a cabeça KAN melhora sob um protocolo justo? RQ3: como ficam os 12 backbones no mesmo protocolo?). Contribuições reescritas: auditoria e reprodução do split, protocolo único, ablação da cabeça com regra pré-registrada. Remover "pioneering" e "empirical demonstration that TransferKAN…" |
| 2 Background / Fundamentação | Manter KAN e transfer learning. Acrescentar uma subseção curta sobre vazamento por grupo (validação agrupada) e a diferença entre AUC de probabilidades e AUC de classes |
| 3 Related Work / Trabalhos Relacionados | Lira et al.: descrever o hold-out **pelo código** (`iterative_split` estratifica por caso, ver `auditoria-consistencia.md`), citar `Lira2025deep` como fonte da base e das Tabelas 4 e 5, e citar a k-fold deles (Fig. 4) |
| 4 Proposed Solution / Solução Proposta | A TransferKAN continua sendo o objeto de estudo, mas sem promessas ("Superior Discrimination…" vira hipótese). Acrescentar o desenho da ablação (a)–(e) com as contagens de parâmetros |
| 5 Methodology / Metodologia | Reescrever inteiro com a seção 3: dados, dobras, receita, ponte AMP, cabeças, 12 backbones, MLSD (treino só com entradas; teste = dobra inteira), experimento de vazamento, métricas, regra de decisão pré-registrada (data e commit), estatística (bootstrap de casos 2.000; Δ por dobra), sementes 42/43/44, hardware e custo |
| 6 Results / Resultados | Trocar as tabelas e o gráfico de barras antigos (AUC 0,908/0,928; 0,749/0,944) pelas tabelas e figuras novas. Ordem: (1) vazamento, (2) E/S, (3) MLSD, (4) comparação com Lira, (5) TransKAN × transformers. As tabelas M/m/W (`tabela_mmw_*.tex`) vão para o apêndice. Os modelos exploratórios seguem a decisão 1 |
| 7 Conclusions / Conclusões | Reescrever: respostas às RQ; limitações (uma base, encostado, 2 sementes para Swin/MaxViT no MLSD, reprodução do método e não da partição exata de Lira, última época sem validação); trabalhos futuros (multitarefa P2, Wav-KAN P3, validação externa) |
| `Published_Articles.tex`, `Anexo1.tex` | Conferir se citam números antigos do SIBGRAPI. O manuscrito SIBGRAPI em si fica fora deste plano |

**Tabelas e figuras:**
- copie os `.tex` de `resultados/` para `tcc_dissertation/tex/tabelas/` e as figuras para `tcc_dissertation/img/protocolo/`; o `pt/img` é link simbólico para `img`;
- as legendas geradas estão em inglês: no PT, traduza as legendas nas cópias do PT (`pt/tex/tabelas/`);
- não edite os arquivos de `resultados/`.

## 6. Verificação antes de entregar

1. As duas versões compilam com pdfLaTeX, sem `??` e sem citação indefinida (confira o `.log`).
2. Nenhum número antigo no texto. Este comando deve dar zero linhas, tirando o apêndice exploratório se for mantido:
   `grep -rnE "0[.,]860|0[.,]873|0[.,]749|0[.,]908|0[.,]928|0[.,]893|0[.,]924|SMOTE|best epoch|melhor época" tcc_dissertation/tex tcc_dissertation/pt/tex`
3. Cada número citado no texto confere com `resultados/*.csv` e `*.json`. Faça um script que extraia os números do `.tex` e compare.
4. As frases da seção 4 estão corrigidas.
5. EN e PT têm as mesmas seções, tabelas e números.

## 7. Fora do escopo

- novos treinos;
- mudanças em `protocolo_comum.py` ou nos notebooks;
- o manuscrito e o pôster do SIBGRAPI;
- as sessões P2 e P3;
- a reescrita dos guias de estudo e de código, que foram removidos em 19/10 por estarem desatualizados e podem ser recuperados no histórico do git. Um guia novo, se o usuário quiser, fica para depois da dissertação.
