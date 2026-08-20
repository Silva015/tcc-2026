# Guia de Estudo — TransferKAN (TCC)

Guia estruturado para você entender o projeto inteiro (texto + código + teoria) e conversar com o professor com segurança. Siga a ordem das fases; cada uma diz **o que ler**, **o que entender** e **perguntas que o professor pode fazer**.

---

## Visão geral em 3 frases

1. **Problema:** classificar ferimentos por arma de fogo em imagens forenses reais — (a) Entrada vs. Saída e (b) distância do disparo / MLSD (Encostado, Curta, À distância).
2. **Ideia:** pegar uma CNN forte (ResNet152, via *transfer learning*) e trocar a "cabeça" classificadora tradicional (MLP) por uma **KAN** (Kolmogorov-Arnold Network) → arquitetura **TransferKAN**.
3. **Resultado-chave:** sob avaliação honesta (validação cruzada k-fold por caso, sem vazamento), a TransferKAN tem **AUC bem maior** que a ResNet152 do artigo de 2024 (0,928 micro vs 0,821 na tarefa Entrada/Saída), mostrando que a KAN discrimina melhor as classes minoritárias.

---

## Fase 0 — O contexto (30 min)

**Ler:** `references/2024_paper.txt` (o artigo de Lira et al., 2024) — é a base de comparação de todo o TCC.

**Entender:**
- A base de dados **FDCPUnBGunshotDB**: 2.551 imagens reais (1.883 entrada / 668 saída); MLSD só nas de entrada (1.609 à distância / 232 curta / 42 encostado). **É muito desbalanceada** — esse é o fio condutor do TCC.
- O artigo testou **59 arquiteturas** e a **ResNet152 venceu** (acc 86,9% / AUC 82,09% em Entrada-Saída; acc 92,48% em MLSD).
- Limitação que o artigo deixou em aberto: o desbalanceamento prejudica precisão/recall das classes raras → **é a brecha que o seu TCC ataca**.

**Perguntas prováveis:** "Por que comparar com esse artigo?" "Qual a limitação dele que você resolve?"

---

## Fase 1 — Fundamentos teóricos (2–3 h)

**Ler:** Capítulo 2 (Fundamentação) da sua monografia.

### 1.1 Patologia forense (o "domínio")
- Entrada vs. saída: orla de escoriação, inversão/eversão do tecido, formas.
- MLSD: vestígios de pólvora (tatuagem/esfumaçamento, fuligem, marca do cano).
- Você não precisa virar perito, mas saiba **por que é difícil** (putrefação, ângulo, calibre, iluminação).

### 1.2 Deep Learning / Visão Computacional
- **CNN:** extrai características hierárquicas (bordas → texturas → formas).
- **ResNet / conexões residuais:** resolvem o *vanishing gradient*; permitem redes muito profundas (152 camadas). Saber explicar "o que é uma skip connection".
- **Transfer Learning:** usar pesos pré-treinados na ImageNet e adaptar para um problema com poucos dados. **Por que isso importa aqui?** Porque a base forense é pequena.
- **MLP (cabeça tradicional):** camadas densas + ativação fixa (ReLU). É o que a KAN substitui.

### 1.3 KAN — o coração do trabalho (estude com calma!)
- **Teorema de representação de Kolmogorov-Arnold:** qualquer função contínua multivariada = composição de funções de **uma variável** + somas.
- **MLP vs KAN:** MLP tem pesos fixos nas arestas e ativação fixa nos nós; **KAN tem funções aprendíveis nas arestas** (e só soma nos nós).
- **B-splines:** como as funções das arestas são parametrizadas (polinômios por partes que mudam de forma no treino). Parâmetros relevantes no código: *grid size* = 10, *spline order* = 3.
- **Por que ajuda aqui:** mais expressiva com menos parâmetros → menos sobreajuste em dado pequeno/desbalanceado; molda fronteiras ao redor das classes raras.

**Perguntas prováveis:** "Explique a diferença entre MLP e KAN." "O que é uma B-spline?" "Por que a KAN teria vantagem em dados desbalanceados?" "O que é o teorema de Kolmogorov-Arnold?"

> 📌 **Material extra recomendado:** o artigo original da KAN (Liu et al., 2024, `arXiv:2404.19756`) e a implementação `efficient-kan` (GitHub). Veja as figuras do artigo da KAN para a intuição "ativação na aresta".

---

## Fase 2 — A arquitetura proposta (1 h)

**Ler:** Capítulo 4 (Solução Proposta) + **a Figura da arquitetura** (diagrama TransferKAN).

**Entender o fluxo:** Imagem 224×224 → **ResNet152** (sai vetor de 2048) → **Dropout 0,5** → **KAN [2048, 128, N]** → N classes (2 ou 3).
- Por que remover a última camada densa da ResNet? Para usar só o extrator de características.
- Por que Dropout 0,5? Regularização forte contra sobreajuste.
- As 3 variantes alternativas (para comparação): **PureKAN**, **PureConvKAN**, **ConvKAN+ResNet** — servem para provar que **só a combinação ResNet+KAN** funciona bem (ablação).

**Perguntas prováveis:** "Por que 2048?" "O que cada bloco faz?" "Por que testou as outras 3 arquiteturas?" (resposta: estudo de ablação).

---

## Fase 3 — Metodologia (2 h) — *a parte que mais distingue o seu TCC*

**Ler:** Capítulo 5 (Metodologia).

### 3.1 Vazamento de dados (data leakage) — **CONCEITO-CHAVE**
- Um mesmo caso/vítima tem várias fotos do mesmo ferimento. Se dividir as **imagens** aleatoriamente, a mesma lesão pode cair no treino e no teste → métrica inflada e falsa.
- Sua solução: dividir por **Identificador de Caso** (extraído do nome do arquivo) → **5-fold por caso**, com checagem programática de interseção zero.
- **Esse é o seu maior argumento metodológico.** O artigo de 2024 usou *hold-out* simples (mais frouxo).

### 3.2 Data Augmentation (biblioteca Albumentations)
- Geométrica (flip, rotação), elástica (simula pele), CLAHE + ruído gaussiano (iluminação).
- Aplicada **só no treino**; teste só redimensiona + normaliza.

### 3.3 Treinamento em 2 fases
- Fase 1 (épocas 1–15): ResNet **congelada**, treina só a KAN (lr 1e-3).
- Fase 2 (épocas 16–60): tudo **descongelado** / fine-tuning (lr 1e-5).
- Agendador CosineAnnealingLR; **perda ponderada por frequência inversa** (penaliza erro nas classes raras).

**Perguntas prováveis:** "O que é vazamento semântico e como você evitou?" "Por que k-fold por caso e não aleatório?" "Por que congelar o backbone na primeira fase?" "Como lidou com o desbalanceamento?" (3 respostas: augmentation + k-fold honesto + perda ponderada).

---

## Fase 4 — Métricas e resultados (2 h)

**Ler:** Capítulo 6 (Resultados) + as duas tabelas + os 3 gráficos (boxplot, barras de AUC).

### 4.1 Métricas (saiba definir cada uma)
- **Acurácia** — enganosa em dado desbalanceado (basta prever sempre a classe majoritária).
- **Precisão, Revocação (recall), F1** — equilíbrio entre falsos positivos/negativos.
- **Especificidade** — taxa de verdadeiros negativos.
- **AUC (ROC)** — capacidade de separar classes em **todos os limiares**; **a métrica mais importante do seu TCC**.
- **Macro / micro / Ponderada** — macro trata classes igualmente (sensível às raras); micro/ponderada pesam pelo volume.

### 4.2 A leitura correta dos números
- ResNet152 tem acurácia bruta um pouco maior (0,869), **mas** por *hold-out* (sujeito a vazamento).
- Sob k-fold honesto, a mediana de acurácia da ResNet152 cai para **~0,73** (lida do boxplot, Fig. 4 do artigo de 2024 — note: o texto antigo dizia "0,817", foi corrigido para ~0,73 conforme a figura real).
- A TransferKAN ganha **onde importa**: AUC micro 0,928 / macro 0,908 vs 0,821; e em MLSD AUC macro 0,749 vs 0,733.
- **Mensagem central:** "acurácia bruta é métrica fraca aqui; sob avaliação justa e olhando AUC, a TransferKAN é superior e mais estável."

**Perguntas prováveis (as mais importantes!):**
- "Se a acurácia da ResNet é maior, por que dizer que sua proposta é melhor?" → vazamento + acurácia enganosa + AUC.
- "O que AUC mede e por que você foca nela?"
- "Por que a TransferKAN não bateu a acurácia da referência em MLSD?" → avaliação muito mais severa (k-fold por caso) + foco em métricas integradas.

---

## Fase 5 — O código (3–4 h)

> 📘 **Esta fase tem um guia dedicado:** [`GUIA_DO_CODIGO.md`](GUIA_DO_CODIGO.md) explica o `TransferKAN_EntradaSaida.ipynb` linha a linha, traz a tabela de rastreabilidade código ↔ monografia e as respostas prontas para perguntas sobre implementação. O resumo abaixo é o mapa; o guia é o território.

**Pasta:** `notebooks/` — 4 arquiteturas × 2 tarefas = 8 notebooks.

| Arquitetura | Entrada/Saída | MLSD | Papel |
|---|---|---|---|
| **TransferKAN** (proposta) | `TransferKAN_EntradaSaida.ipynb` | `TransferKAN_MLSD.ipynb` | modelo principal (k-fold) |
| ConvKAN+ResNet | `ConvKANWithResNet_EntradaSaida.ipynb` | `ConvKANWithResNet_MLSD.ipynb` | ablação |
| PureConvKAN | `PureConvKAN_EntradaSaida.ipynb` | `PureConvKAN_MLSD.ipynb` | ablação |
| PureKAN | `PureKAN_EntradaSaida.ipynb` | `PureKAN_MLSD.ipynb` | ablação |

**Foque primeiro no `TransferKAN_EntradaSaida.ipynb`.** Leia célula a célula e saiba localizar:
1. **Definição da classe do modelo** (`TransferKAN_Gunshot`): onde a ResNet152 é carregada, a fc removida, o Dropout e a KAN (`efficient-kan`) anexados.
2. **Separação por caso** (`unique_cases`, `KFold`): a prova de que não há vazamento.
3. **Loop de 5 folds** + treino em 2 fases (congelar/descongelar).
4. **Cálculo das métricas OOF** (`accuracy_score`, `precision/recall/f1`, `roc_auc_score` macro/micro/weighted, especificidade via matriz de confusão).
5. **Pesos de classe** (frequência inversa) na função de perda.

> Dica: rode mentalmente o pipeline "dados → augmentation → modelo → treino 2 fases → predições OOF → métricas". Saiba dizer **em que célula** cada etapa acontece.

**Perguntas prováveis:** "Mostre no código onde você garante que não há vazamento." "Onde a KAN substitui a camada densa?" "Como calculou a AUC para multiclasse?"

---

## Fase 6 — Pontos fracos / limitações (30 min) — *prepare-se para a defesa*

Saiba reconhecer (mostra maturidade):
- Base de **uma só instituição** (PCDF/DF) → falta validação externa.
- Classe "Encostado" tem só **42 imagens** → resultados de MLSD são frágeis nas classes raras.
- Comparação não é 100% simétrica: ResNet152 (hold-out, do artigo) vs TransferKAN (k-fold, seu) — você argumenta a favor do k-fold, mas **o ideal seria rodar a ResNet152 sob o mesmo k-fold** (trabalho futuro / possível pergunta dura).
- KAN ainda é recente; custo computacional e interpretabilidade são frentes abertas (você cita XAI como trabalho futuro).

---

## Roteiro rápido de revisão (véspera da conversa)

1. Conte a "história" em 3 frases (topo deste guia).
2. Desenhe o diagrama da TransferKAN de cabeça.
3. Explique MLP vs KAN + B-spline.
4. Explique vazamento de dados e k-fold por caso.
5. Defenda "por que AUC > acurácia aqui".
6. Aponte 2 limitações e 2 trabalhos futuros.

## Glossário-relâmpago
- **Backbone:** extrator de características (a ResNet sem a última camada).
- **OOF (out-of-fold):** predições feitas nos dados que ficaram de fora de cada dobra; geram a métrica honesta.
- **Ablação:** remover/trocar partes para provar que cada componente importa (PureKAN, PureConvKAN, ConvKAN+ResNet).
- **Hold-out vs k-fold:** divisão única vs várias divisões rotativas (mais robusto).
