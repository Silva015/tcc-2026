# Guia do Pôster — TransKAN no SIBGRAPI 2026

Guia para você ficar no estande com segurança. Ele cobre:
- o que o **manuscrito do SIBGRAPI** diz (que **não é igual ao TCC**);
- o que falar para cada tipo de visitante;
- como responder às perguntas difíceis.

O pôster está em [`poster/poster.tex`](poster/poster.tex) e compila com `make` dentro de `poster/`.

---

## 0. Antes de tudo: o paper em 3 frases

1. **Problema:** peritos precisam dizer se um ferimento de arma de fogo é de **entrada ou saída** (e a distância do tiro) a partir de fotos reais. As fotos são ruidosas e muito desbalanceadas, e as CNNs com cabeça MLP convencional ficam aquém nessa tarefa.
2. **Ideia:** **TransKAN** = ResNet-152 pré-treinada (transfer learning) + **cabeça KAN** no lugar da cabeça densa. Na KAN, cada aresta aprende a própria função de ativação (uma B-spline).
3. **Resultado:** na base **GuWID** (2.551 imagens reais da PCDF), com **validação cruzada 5-fold agrupada por caso**, a TransKAN **iguala ou supera a ResNet-152 em todas as métricas principais**:

   | Métrica | TransKAN | ResNet-152 |
   |---|---|---|
   | F1 macro | 0,832 | 0,824 |
   | Precisão macro | 0,847 | 0,827 |
   | AUC macro | 0,893 | 0,821 |

   Foram 12 famílias de arquiteturas comparadas.

---

## 1. ATENÇÃO: paper ≠ TCC — use só os números do paper

| | TCC (`GUIA_DE_ESTUDO.md`) | **Manuscrito SIBGRAPI** |
|---|---|---|
| Nome do modelo | TransferKAN | **TransKAN** |
| Nome da base | FDCPUnBGunshotDB | **GuWID** (Gunshot Wound Image Database) |
| Comparação | ResNet-152 do artigo de 2024 (*hold-out*) | **12 famílias retreinadas por você no mesmo 5-fold agrupado** |
| Desbalanceamento | augmentation + perda ponderada | **SMOTE + ENN** + perda ponderada + augmentation |
| Tarefas relatadas | Entrada/Saída **e** MLSD | **só Entrada/Saída** (MLSD fica para trabalho futuro) |
| AUC TransKAN | 0,928 micro / 0,908 macro | **0,924 micro / 0,893 macro** |

> ⚠️ **Divergência código × manuscrito que você precisa resolver antes do evento.** Os notebooks deste repositório (versão TCC) **não têm SMOTE/ENN**: `grep` por `smote`/`imblearn` não encontra nada. Eles também usam `KFold` sobre os casos, que é **agrupado, mas não estratificado**. O manuscrito descreve *grouped, stratified 5-fold* + SMOTE/ENN. Se os experimentos do paper rodaram em outro código (Colab/Overleaf/outra pasta), saiba dizer **onde** ele está. Se alguém pedir o código, a resposta do paper é: "será liberado após a aceitação".

---

## 2. O paper seção por seção — o que dizer e para onde apontar

### I. Introdução
- **Contexto:** armas de fogo respondem por cerca de ¾ dos homicídios nas Américas. Só no Brasil, são mais de 47 mil mortes por ano.
- **Por que é difícil:** o perito se baseia em forma, dimensão e bordas do ferimento. Isso pode ser obscurecido por velocidade, trajetória, calibre, anteparos e região anatômica.
- **Lacuna:** os trabalhos anteriores usavam poucas amostras e modelos animais, ou não avaliavam MLSD em dados de autópsia. Os que usam deep learning combinam CNN + **MLP**, que tem dificuldade com fronteiras de decisão irregulares em dados ruidosos e desbalanceados.
- **Contribuição:** o paper **não propõe uma KAN nova**. É um estudo empírico, guiado pela aplicação: *uma cabeça KAN ajuda de verdade no regime de poucos dados e muito ruído?*
- 👉 Aponte: o **título** e a **faixa de fotos** (coluna 1).

### II.A Base de dados
- Arquivo da PCDF com **2.012.584 fotos (jan/2012 – mar/2022)**. Aprovação do **CEP/UnB, protocolo 4418221.9.0000.0030**.
- **Critérios de inclusão:**
  - vítima identificada antes da decomposição;
  - ferimento bem fotografado;
  - projétil recuperado e calibre confirmado pela Balística;
  - tipo, local e MLSD descritos pelo legista no laudo.
- Resultado: **2.551 imagens de 489 casos** (465 vítimas homens + 24 mulheres). São 2.437 imagens (95,5%) de homens e 114 (4,5%) de mulheres. Isso reflete o perfil regional das vítimas, não um viés de amostragem.
- Rótulos **transcritos literalmente dos laudos**, sem julgamento novo. **Não houve segunda revisão** (é uma limitação declarada).
- 👉 Aponte: o **funil** e as **barras de proporção** (coluna 1).

### II.B Pré-processamento
- Os ferimentos foram **recortados manualmente no GIMP** em quadrados centrados na lesão.
- Distribuição da base:

  | Rótulo | Quantidade |
  |---|---|
  | **Entrada** | 1.883 |
  | **Saída** | 668 |
  | MLSD (só entradas) — à distância | 1.609 (85,45%) |
  | MLSD — curta distância | 232 (12,32%) |
  | MLSD — encostado | 42 (2,23%) |
  | Região — tronco | 1.310 (51,35%) |
  | Região — cabeça/pescoço | 548 (21,48%) |
  | Região — membros superiores | 503 (19,72%) |
  | Região — membros inferiores | 190 (7,45%) |

- 👉 Aponte: a **Fig. 1 do pôster** (foto bruta com régua forense → recorte quadrado).

### II.C Desbalanceamento + augmentation
- **SMOTE:** cria amostras sintéticas da classe minoritária interpolando vizinhos.
- **ENN** (*Edited Nearest Neighbours*): remove amostras da classe majoritária que ficam perto da fronteira.
- Os dois foram aplicados sobre as imagens RGB 224×224 **achatadas em vetores de pixels** e depois remontadas em imagens.
- Tudo é **reajustado dentro de cada dobra de treino**, para não vazar informação para validação e teste. Além disso há **pesos de classe** na perda.
- **Albumentations:**
  - flips horizontais e verticais;
  - shift/scale/rotate;
  - **elastic** (simula a elasticidade da pele);
  - **CLAHE**, color jitter e **ruído gaussiano** (simulam a iluminação da cena e o ruído do sensor).
- 👉 Aponte: a faixa de **chips** "SMOTE → ENN → Class-weighted → Albumentations" (coluna 2).

### II.D Arquitetura
- **Entrada/saída é classificação binária; MLSD é multiclasse.** A prática forense define a distância por categorias (marca do cano, fuligem, tatuagem de pólvora), e não como número contínuo.
- **Fluxo:** imagem 224² → **ResNet-152** (ImageNet, sem a camada `fc`) → vetor de **2048** dimensões → **Dropout 0,5** → **KAN [2048, 128, 2]**, com B-splines **cúbicas (ordem 3)** e **grade 10**, implementada com **Efficient-KAN**.
- Na Fig. 2, FC e KAN aparecem lado a lado **só para comparação visual**. São **configurações separadas**, não uma arquitetura paralela.
- 👉 Aponte: a **Fig. 3 do pôster** (arquitetura) e a **faixa de blocos** logo abaixo.

### III. Setup experimental
- Por limite de páginas, o paper relata **só Entrada/Saída**.
- **5-fold agrupado e estratificado por ID do caso:** todas as fotos de uma vítima ficam na mesma dobra, o que evita vazamento.
- **Baselines:** extrator + cabeça FC, todos do TorchVision:
  - AlexNet, DenseNet, EfficientNet, MNASNet, MobileNet, RegNet;
  - ResNet, ShuffleNet, SqueezeNet, SwinT, VGG e ViT (MaxViT).
- **Treino:** 60 épocas por dobra.
  - **Fase 1 (épocas 1–15):** backbone congelado, só a cabeça treina. AdamW, lr 1e-3, weight decay 1e-2.
  - **Fase 2 (épocas 16–60):** a rede inteira treina, com lr 1e-5.
  - Durante todo o treino: CosineAnnealingLR e perda ponderada pela frequência inversa das classes.
- Métricas via scikit-learn: precisão, recall, F1, AUC e especificidade, em **macro** (todas as classes pesam igual) e **micro** (global). Guarda-se o melhor checkpoint de validação de cada dobra, e as predições *out-of-fold* são agregadas.
- 👉 Aponte: a **grade de folds** e a **linha do tempo das 2 fases** (colunas 2 e 3).

### IV. Resultados (Tabela I)

| Arquitetura | Prec. macro/micro | Recall macro/micro | F1 macro/micro | AUC macro/micro | Espec. macro/micro |
|---|---|---|---|---|---|
| AlexNet | .695/.765 | .716/.765 | .703/.765 | .716/.716 | .716/.765 |
| DenseNet169 | .700/.777 | .679/.777 | .688/.777 | .679/.679 | .679/.777 |
| EfficientNetV2L | .710/.782 | .638/.782 | .654/.782 | .638/.638 | .638/.782 |
| MNASNet1.3 | .659/.746 | .656/.746 | .657/.746 | .656/.656 | .656/.746 |
| MobileNetV2 | .609/.719 | .593/.719 | .598/.719 | .593/.593 | .593/.719 |
| RegNetY-16GF | .571/.711 | .549/.711 | .549/.711 | .549/.549 | .549/.711 |
| **ResNet152** | .827/.869 | .821/.869 | .824/.869 | .821/.821 | .821/.869 |
| ShuffleNetV2 x0.5 | .628/.730 | .614/.730 | .619/.730 | .614/.614 | .614/.730 |
| SqueezeNet1.0 | .375/.750 | .500/.750 | .429/.750 | .500/.500 | .500/.750 |
| SwinT-B | .614/.728 | .591/.728 | .597/.728 | .591/.591 | .591/.728 |
| VGG16 | .706/.782 | .685/.782 | .694/.782 | .685/.685 | .685/.782 |
| ViT (MaxViT) | .703/.780 | .653/.780 | .668/.780 | .653/.653 | .653/.780 |
| **TransKAN** | **.847/.871** | .821/.871 | **.832/.871** | **.893/.924** | .821/.871 |

**Leitura (o que o paper diz):**
- Os modelos leves (MobileNet, ShuffleNet, SqueezeNet) formam o piso: compressão agressiva perde detalhe de borda.
- ViTs isotrópicos são inconsistentes com pouco dado. O MaxViT, que é hierárquico, vai melhor.
- As residuais são as mais fortes, e a ResNet-152 é a melhor baseline. Por isso ela virou o backbone.
- A TransKAN supera a ResNet-152 em precisão (0,847 vs 0,827), em F1 (0,832 vs 0,824) e, principalmente, em AUC (0,893/0,924 vs 0,821).

**Leitura que VOCÊ precisa saber** (pergunta de especialista):
- Numa tarefa **binária**, **micro precisão = micro recall = micro F1 = micro especificidade = acurácia**. A coluna "micro" inteira é a acurácia: 0,871 vs 0,869.
- Também na tarefa binária, **especificidade macro = recall macro**, porque a especificidade de uma classe é o recall da outra. O "empate" de 0,821 com a ResNet é uma **identidade matemática**, não coincidência.
- ⚠️ **Nos 12 baselines, AUC macro = AUC micro = recall macro, exatamente.** Isso é a assinatura de uma AUC calculada a partir de **rótulos duros (0/1)**: com um único limiar, AUC = (TPR+TNR)/2 = acurácia balanceada = recall macro. Na TransKAN, a AUC (0,893) é diferente do recall (0,821), o que indica que foi calculada a partir das **probabilidades do softmax**. O próprio `GUIA_DO_CODIGO.md` mostra `roc_auc_score` sobre `F.softmax(...)`.
  - **Consequência:** a comparação de AUC provavelmente **favorece a TransKAN de forma injusta**.
  - **Se der tempo antes do evento**, recalcule a AUC dos baselines com probabilidades.
  - **Enquanto isso**, o pôster usa **F1 macro** no gráfico de barras. Para trocar para AUC, mude `\chartaucfalse` para `\chartauctrue` em `poster.tex`.
  - No estande, a vitória que se sustenta sem ressalvas é: **"iguala ou supera em todas as métricas de limiar e ganha em precisão e F1"**.

### V. Conclusão e trabalho futuro
- Resolver entrada/saída e MLSD juntas num pipeline **multitarefa**.
- Trocar as splines por **wavelets** (Wav-KAN) para reduzir custo.
- **Validação externa** em bases internacionais.

---

## 3. Pitches (com a ordem de apontar no pôster)

> Regra de ouro: **comece pelas fotos, termine nos números.** Pergunte antes: *"Você trabalha com visão computacional ou com área forense?"*. Isso decide qual pitch usar.

### 15 s — o "gancho" (para quem está só passando)
**PT:** "A gente ensina uma rede neural a olhar fotos reais de perícia e dizer se o tiro **entrou ou saiu** por esse ferimento. O truque é trocar a última camada da rede por uma **KAN**, uma rede que *aprende as próprias funções de ativação*."
**EN:** "We teach a neural network to look at real forensic photos and tell whether a bullet **entered or exited** through that wound. The trick is replacing the network's last layer with a **KAN**, a network that *learns its own activation functions*."

### 1 min — visitante leigo
1. 👉 **Fotos (col. 1):** "Estas são fotos reais de casos da Polícia Civil do DF, com régua forense. Hoje, quem decide se é entrada ou saída é o perito, e a iluminação e a própria lesão dificultam."
2. 👉 **Barras de proporção:** "A base tem 2.551 imagens, mas é muito desequilibrada: ferimentos de saída são só um quarto das imagens, e tiros encostados, 2% das entradas."
3. 👉 **Arquitetura (col. 2):** "Usamos uma rede já treinada em milhões de imagens para extrair características, e trocamos o classificador final por uma KAN."
4. 👉 **Números (col. 3):** "Comparamos com 12 famílias de redes famosas nas mesmas condições, e a nossa ficou em primeiro em todas as métricas principais."

### 3 min — visitante de visão computacional / ML
1. **Problema e dado** (20 s): GuWID, 2.551 imagens de 489 casos reais, rótulos vindos dos laudos, forte desbalanceamento.
2. 👉 **MLP vs KAN** (60 s): "No MLP, as arestas têm pesos escalares e o nó tem uma ativação fixa. Na KAN, cada aresta é uma **B-spline cúbica aprendível** e o nó só soma (teorema de Kolmogorov–Arnold). A ideia é ganhar flexibilidade para fronteiras irregulares **sem aumentar a profundidade**, e só na cabeça. O backbone continua sendo uma ResNet-152 pré-treinada."
3. 👉 **Avaliação honesta** (40 s): "Uma vítima tem várias fotos da mesma lesão. Se você divide por imagem, vaza. Nós dividimos por **caso**, em 5 dobras, e **retreinamos os 12 baselines no mesmo protocolo**. O treino tem duas fases: cabeça sozinha, depois fine-tuning total com lr baixo."
4. 👉 **Resultados** (40 s): F1 macro 0,832 vs 0,824, precisão 0,847 vs 0,827, acurácia 0,871 vs 0,869, AUC 0,893. "O ganho não é enorme em acurácia, mas é consistente, e a cabeça é a única coisa que muda."
5. **Fecho** (20 s): trabalho futuro (multitarefa com MLSD, Wav-KAN, validação externa) e o QR code.

### 5 min — especialista / avaliador
O pitch de 3 min, **mais**:
- **Por que a cabeça, e não uma KAN inteira?** Com poucos milhares de imagens, uma KAN convolucional do zero sobreajusta. No TCC, as ablações PureKAN, PureConvKAN e ConvKAN+ResNet foram piores.
- **Honestidade sobre as métricas:** em tarefa binária, a coluna micro é a acurácia. O "empate" em recall/especificidade macro é uma identidade. Seja transparente sobre a AUC dos baselines (ver Seção 2.IV).
- **SMOTE/ENN por dobra** + augmentation que simula pele e iluminação.
- **Limitações:** base de uma instituição só, rótulos sem segunda revisão, só 42 casos de tiro encostado, falta de validação externa.

---

## 4. Números para decorar

| O quê | Valor |
|---|---|
| Fotos no arquivo PCDF | 2.012.584 (2012–2022) |
| Imagens na GuWID | **2.551** de **489 casos** |
| Entrada / Saída | **1.883 / 668** (73,8% / 26,2%) |
| MLSD (entradas) | 1.609 distância / 232 curta / **42 encostado** |
| Sexo | 95,5% homens (465 casos) / 4,5% mulheres (24 casos) |
| Entrada da rede | 224×224 RGB |
| Backbone | ResNet-152 (ImageNet), 2048 dimensões, cerca de 60 M parâmetros |
| Cabeça | Dropout 0,5 → KAN [2048, 128, 2], ordem 3, grade 10 |
| Treino | 60 épocas: 15 congeladas (lr 1e-3) + 45 fine-tuning (lr 1e-5) |
| Avaliação | 5-fold agrupado por caso, OOF, 12 famílias de baselines |
| TransKAN | Prec. .847 · F1 .832 · AUC .893 / .924 · Acc .871 |
| ResNet-152 | Prec. .827 · F1 .824 · AUC .821 · Acc .869 |
| Ética | CEP/UnB 4418221.9.0000.0030 |

---

## 5. Banco de perguntas e respostas

### KAN / teoria
1. **"O que é uma KAN?"**
   É uma rede inspirada no teorema de Kolmogorov–Arnold: toda função contínua de várias variáveis pode ser escrita como somas e composições de funções de **uma** variável. Na prática, cada aresta tem uma função 1-D aprendível (uma B-spline) e cada nó só soma. No MLP é o contrário: o peso é escalar e a ativação fica fixa no nó.
2. **"O que é uma B-spline, e o que significam grade 10 e ordem 3?"**
   É um polinômio por partes com emendas suaves. Ordem 3 significa cúbica. Grade 10 significa 10 intervalos, o que dá cerca de 13 coeficientes por aresta. Mais grade dá mais expressividade e mais parâmetros.
3. **"Isso não é só uma cabeça com mais parâmetros?"**
   A cabeça KAN tem mais parâmetros que um MLP da mesma largura. Na Efficient-KAN, cada aresta tem cerca de 15 parâmetros (13 coeficientes de spline + peso base + escala), o que dá cerca de 3,9 M na camada 2048→128. Mesmo assim, é pouco perto dos cerca de 60 M da ResNet-152. **Não fizemos** um controle com um MLP de mesmo número de parâmetros. É uma boa sugestão e vale anotar como trabalho futuro. *(Seja honesto; é uma pergunta boa.)*
4. **"Por que não uma KAN inteira / ConvKAN?"**
   Com cerca de 2,5 mil imagens, aprender filtros do zero sobreajusta. No TCC, as variantes PureKAN, PureConvKAN e ConvKAN+ResNet ficaram abaixo. O ganho está em reaproveitar as features da ImageNet e só flexibilizar a decisão.
5. **"E a interpretabilidade da KAN?"**
   É uma vantagem potencial: dá para plotar as splines aprendidas. Com 2048 entradas, porém, isso não é trivial, e ainda não exploramos. Fica como trabalho futuro.
6. **"Por que Efficient-KAN e não pykan?"**
   A pykan original é lenta e pesada em memória. A Efficient-KAN reformula o cálculo das B-splines como multiplicação de matrizes e roda bem em GPU.

### Metodologia / avaliação
7. **"Como evitou vazamento?"**
   Dividindo por **ID do caso**: todas as fotos da mesma vítima caem na mesma dobra. Com divisão por imagem, a mesma lesão apareceria no treino e no teste, e a métrica ficaria inflada.
8. **"Os baselines foram treinados igual?"**
   Sim. Os 12 foram retreinados no mesmo 5-fold agrupado, com o mesmo augmentation e o mesmo treino em 2 fases. Na Tabela I aparece só a melhor variante de cada família.
9. **"Por que congelar o backbone nas primeiras 15 épocas?"**
   No início a cabeça é aleatória e gera gradientes grandes que estragariam as features pré-treinadas. Primeiro a cabeça se acomoda; depois vem o fine-tuning com lr 100× menor.
10. **"Qual checkpoint você usa? Não é seleção no teste?"**
    O melhor checkpoint por acurácia de validação em cada dobra, com as predições OOF agregadas. *Se insistirem:* o ideal seria um conjunto de validação interno separado do de teste em cada dobra. Reconheça como ponto a melhorar se for o caso.
11. **"Por que não relatam MLSD?"**
    Por limite de páginas; ficou para a versão estendida. O MLSD é mais difícil porque há só 42 imagens de tiro encostado. *(A Fig. 2 mostra saídas de MLSD porque o pipeline é o mesmo para as duas tarefas.)*

### SMOTE / desbalanceamento (pergunta clássica de especialista)
12. **"SMOTE em pixels? Isso gera imagens estranhas."**
    Sim, interpolar pixels cria imagens "fantasma", misturas de duas lesões. Foi uma escolha simples para equilibrar as classes, e é **refeita por dobra**, só no treino. Alternativas melhores seriam SMOTE no espaço de features da CNN ou só oversampling com augmentation. **Não fizemos a ablação** sem SMOTE, e é uma boa sugestão.
13. **"Por que SMOTE + ENN + pesos de classe? Não é redundante?"**
    Cada um ataca uma parte do problema. SMOTE aumenta a classe minoritária, ENN limpa a fronteira, e os pesos cobrem o desbalanceamento residual. Admita que não isolamos o efeito de cada um.

### Resultados / métricas
14. **"O ganho sobre a ResNet é pequeno em acurácia (0,871 vs 0,869)."**
    Concordo, a acurácia é praticamente igual. O ganho aparece em **precisão macro** (+0,020) e **F1 macro** (+0,008), ou seja, menos falsos positivos. Mudamos **só a cabeça**, então é um ganho "de graça" em cima de um backbone forte.
15. **"Por que o recall e a especificidade macro empatam em 0,821?"**
    Em tarefa binária, a especificidade de uma classe é o recall da outra, então especificidade macro = recall macro. Não é coincidência.
16. **"Por que a AUC dos baselines é igual ao recall macro?"**
    Essa é **a pergunta mais perigosa**. Resposta honesta: "Boa observação. Isso indica que a AUC dos baselines foi calculada a partir das predições de classe, e nesse caso a AUC coincide com a acurácia balanceada. A TransKAN usou as probabilidades. Vou recalcular os baselines com probabilidades na versão estendida. Por isso, no pôster, destaco o F1." **Não tente defender a diferença 0,893 vs 0,821 como se fosse limpa.**
17. **"Por que ViTs foram mal?"**
    Transformers isotrópicos precisam de muito dado. Com cerca de 2,5 mil imagens, o viés indutivo das CNNs (localidade, hierarquia) ajuda. O MaxViT, que é hierárquico, foi o melhor dos transformers.
18. **"Tem significância estatística? Desvio entre as dobras?"**
    O paper relata métricas OOF agregadas, sem intervalo de confiança. Um teste pareado por dobra, ou um bootstrap das predições OOF, seria o próximo passo. *(Não invente números.)*

### Dados / ética
19. **"Como conseguiram as fotos? E a ética?"**
    São registros de casos reais da PCDF, com permissão legal e aprovação do CEP da UnB (protocolo 4418221.9.0000.0030). As imagens são recortadas só na lesão, sem rosto nem identificação.
20. **"Os rótulos são confiáveis?"**
    Vêm do laudo do legista, transcritos literalmente. É o padrão-ouro disponível, mas não houve segunda revisão independente, o que declaramos como limitação.
21. **"Viés de sexo (95% homens)?"**
    Reflete o perfil real das vítimas na região. Com só 24 casos femininos, não fizemos avaliação estratificada por sexo.
22. **"Vão liberar a base?"**
    O paper diz que código e dados serão disponibilizados após a aceitação. Os detalhes de anonimização e o acordo com a PCDF ainda precisam ser acertados.
23. **"Isso substitui o perito?"**
    Não. É uma ferramenta de **apoio**: uma segunda opinião objetiva e repetível. Ainda falta validação externa antes de qualquer uso em laudo.

### Futuro
24. **"Próximos passos?"**
    Multitarefa (entrada/saída + MLSD juntos), Wav-KAN para reduzir custo, validação em bases internacionais e, idealmente, um MLP de mesma capacidade como controle.
25. **"Funciona com fotos de celular / outras polícias?"**
    Ainda não sabemos, e é exatamente a validação externa que falta. O augmentation (CLAHE, ruído, color jitter) tenta dar robustez à iluminação e ao sensor.

---

## 6. Logística do estande

- **Abordagem:** faça contato visual e pergunte *"Posso te mostrar em 1 minuto?"*. Deixe a pessoa olhar as fotos primeiro; elas chamam atenção sozinhas.
- **Conteúdo sensível:** as fotos são de lesões reais. Se alguém parecer desconfortável, vá direto para a coluna 2 (arquitetura).
- **Leve:**
  - 10–20 cópias A4 do paper;
  - celular ou tablet com a página de revisão aberta;
  - cartão ou QR para contato;
  - água.
- **Não saiba?** Diga: *"Não testei isso, mas é uma ótima ideia. Posso anotar seu contato?"*. Isso vale mais do que inventar uma resposta.
- **Antes do evento:**
  - Coloque sua foto em `poster/img/author.jpg` e recompile.
  - **Troque as fotos pelos originais em alta resolução.** As extraídas do PDF ficam com cerca de 56–130 dpi no A0; o ideal é ≥150 dpi. Mantenha os mesmos nomes de arquivo em `poster/img/`.
  - Confirme que o QR (`\qrurl`) aponta para um link **público**. Hoje ele aponta para `github.com/Silva015/tcc-2026`.
  - Confirme o tamanho exigido pelo evento. O pôster está em A0 retrato (84,1 × 118,9 cm).
