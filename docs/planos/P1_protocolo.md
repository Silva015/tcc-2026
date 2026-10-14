# Plano: "O ganho vem da KAN ou do extrator + receita?" (ablação da cabeça + baselines no mesmo protocolo)

## Contexto

O artigo SIBGRAPI (TransKAN) compara a ResNet-152 + cabeça KAN com baselines copiados da Tabela 4 de Lira et al. Esses baselines vieram de um *hold-out* com vazamento, e a AUC deles foi calculada a partir das classes previstas. Por isso, hoje não dá para atribuir o ganho à KAN.

O objetivo é rodar todos os modelos sob um protocolo único e auditável. São três partes:
1. trocar só a cabeça da ResNet-152: linear, MLP, MLP com o mesmo nº de parâmetros da KAN, e a própria KAN;
2. retreinar os baselines da Tabela I;
3. montar uma tabela comparável e uma frase honesta para o manuscrito, com a regra de decisão definida **antes** dos resultados.

**Decisões já tomadas (por você):**
- **Infraestrutura:** Colab **com assinatura**. Todas as rodadas comparáveis na **L4**; a T4 fica só de reserva, registrada no JSON. Kaggle fora do plano.
- **Protocolo:**
  - sem SMOTE/ENN; o texto do artigo é corrigido;
  - o modelo avaliado é o da **última época**;
  - sem validação interna;
  - `drop_last=True`;
  - AMP **fp16**, condicionado à checagem de ponte.
- **Dados:**
  - nenhuma imagem é redistribuída: o dataset é clonado, num commit fixado, em cada sessão;
  - citar Lira et al. (2025, IJLM) como origem da base é correção obrigatória;
  - os números do artigo (P 0,847, AUC 0,893/0,924) foram imprecisão e serão substituídos.
- **Entrega:** o Overleaf fica como está; eu entrego o `.tex` da tabela e as frases. A dissertação fica fora deste plano.
- **Auditoria:** `auditoria-consistencia.md` lido; não há conflito com o plano.

## O que a leitura revelou

1. **`TransferKAN_EntradaSaida.ipynb`**, a receita de referência, executada na L4:
   - **dobras:** `KFold` sobre os casos (`filename[:9]`), agrupado mas **não estratificado**; 2.554 imagens em 486 casos;
   - **checkpoint:** escolhido pela **acurácia de teste**;
   - **métricas:** só OOF agregado;
   - **augmentation:** a Albumentations descartou `alpha_affine` e `var_limit`, então o `GaussNoise` rodou com `std_range=(0.2, 0.44)`;
   - **KAN:** `[2048,128,2]`, grade 10, ordem 3, com 15 parâmetros por aresta × 262.400 arestas = **3.936.000**.
2. **`TransferKAN_MLSD.ipynb`** não usa as 2 fases. Fica fora deste prompt (o artigo só reporta entrada/saída).
3. **Os outros 6 notebooks** usam *hold-out*, 128/64 px e lote 8. Ficam como registro histórico.
4. **leguwoi:**
   - `StratifiedGroupKFold` sem semente;
   - `case_number` sem o ano (417 grupos, contra 485 ano.caso);
   - a AUC é calculada a partir de `y_pred`;
   - o *hold-out* estratifica **por** caso, e é daí que vem o vazamento;
   - **só as ResNets usam pesos ImageNet** (e ainda trocam a `conv1` por uma camada aleatória); as demais famílias foram treinadas do zero;
   - no k-fold agrupado, a ResNet-152 de Lira tem mediana ~0,70.
5. **Manuscrito real** (`~/Downloads/manuscript 6.pdf`):
   - a Tabela I tem 12 linhas, contando a ResNet152;
   - o texto afirma SMOTE+ENN, *StratifiedGroupKFold*, checkpoint escolhido na validação e baselines na receita de 2 fases, mas nada disso está no código;
   - diz "three-layer KAN", mas são 2 camadas;
   - a contagem de casos diverge (489 no texto, 485 no CSV, 486 nas pastas).
6. **Binário:** P/R/F1 micro são iguais à acurácia, e a especificidade macro é igual à revocação macro, então essas colunas são redundantes.
7. **Drive:** `MyDrive/Colab Notebooks/tcc-final` (id `1i_o4gxs-nyzZOPFp21gOwI_AMHPzct-4`, dono `tutuscarneiro@gmail.com`), hoje vazia.

## Protocolo único (vale para TODOS os modelos)

| Item | Decisão |
|---|---|
| Dados | `git clone` de FDCPUnBGunshotDB **em cada sessão**, no commit `f679389`. Também clonar o leguwoi (GitLab), no `83d82f1`. **Nenhuma imagem nem *cache* é gravado no Drive** (a base não tem licença de redistribuição). Rótulos vêm das pastas, como nos notebooks. |
| Dobras | **Um único** `folds_entrada_saida.csv`, compartilhado com os Prompts 2 e 3 e gravado uma vez.<br>**Geração:**<ul><li>lista **ordenada por `rel_path`**;</li><li>`StratifiedGroupKFold(5, shuffle=True, random_state=42)` com **y = `strat4`** (0 saída, 1 encostado, 2 curta, 3 distância) e groups = `case_id` (`filename[:9]`).</li></ul>**Regra pré-registrada:**<ul><li>cada dobra de teste precisa de ≥2 casos de encostado e ≥1 imagem de cada classe;</li><li>se a semente 42 falhar, usar a 1ª semente ≥43 que passe (decidido só pela composição).</li></ul>**Colunas:** `rel_path, case_id, y_es, y_dist (−1 nas saídas), strat4, fold, obs`. A coluna `obs` guarda as 4 anomalias do CSV do dataset. |
| Pré-processamento | *Cache* uint8 224×224 (≈384 MB), gerado na sessão em `/content`, com o mesmo `A.Resize(224,224)` que é a 1ª operação do pipeline original. O resultado é idêntico ao on-the-fly. O SHA-256 é comparado com o que o notebook 00 registrou. |
| Augmentation | Igual à do TransferKAN **como de fato rodou**: `albumentations==2.0.8` e os valores efetivos explícitos (`ElasticTransform(alpha=1, sigma=50)`, `GaussNoise(std_range=(0.2,0.44))`, `ShiftScaleRotate` igual). Nenhum `UserWarning`. No teste: Resize + Normalize. |
| Modelo | Backbone torchvision com pesos `DEFAULT` → flatten → `Dropout(0.5)` → `{tarefa: cabeça}` (o padrão é uma única tarefa, `es`). |
| Fase 1 (ép. 1–15) | Backbone congelado, `model.train()` (a BN atualiza as estatísticas, como no original), `AdamW` só nos parâmetros **novos** (lr 1e-3, wd 1e-2), `CosineAnnealingLR(T_max=15)`. |
| Fase 2 (ép. 16–60) | Tudo descongelado, `AdamW` (lr 1e-5, wd 1e-3), `CosineAnnealingLR(T_max=45)`. |
| Lote / perda | Lote 32. Treino com **`drop_last=True`** e **sem validação interna** (a parte de treino inteira da dobra). CE com pesos n/(2·n_c) da dobra de treino. |
| Numérico | **AMP fp16** (não bf16) no backbone, **cabeça sempre em fp32**, `channels_last`, igual para todos. **Só é adotado se passar na ponte**; se falhar, tudo roda em fp32, com os padrões do PyTorch, como no log antigo. |
| GPU | Rodadas comparáveis **só na L4**. A célula de setup confere `nvidia-smi`; outra GPU só roda com `PERMITIR_RESERVA=True`, fica registrada no JSON e é marcada com † na tabela. |
| Checkpoint avaliado | **Última época (60).** O teste da dobra é avaliado uma única vez, depois da época 60. |
| Métricas por dobra | Acc, P/R/F1 macro, **AUC = `roc_auc_score(y, P(saída))`**, sensibilidade/especificidade por classe. Agregadas como **média ± dp entre as 5 dobras**, mais a AUC OOF. |
| Parâmetros / tempo | Parâmetros da cabeça e total treinável (Fase 2). Tempo de treino por época e por dobra na L4, com `cuda.synchronize`, sem contar I/O de checkpoint nem setup. |
| Sementes | As dobras ficam fixas. A semente de treino s ∈ {42, 43, 44} controla a inicialização da cabeça, o dropout, a ordem dos lotes e a augmentation. O gerador depende de (s, dobra, época), então a retomada é reprodutível. |
| Dependências | `efficient-kan@7b6ce1c`, `albumentations==2.0.8`; torch/torchvision do ambiente, com as versões registradas. |
| Rastreabilidade | Cada JSON registra: versões, commits (dataset e leguwoi), SHA-256 de `protocolo_comum.py`, dos plugins e do CSV de dobras, GPU, flags de TF32/AMP e ID da sessão. |

## Experimento 1: ablação da cabeça (ResNet-152 fixa; backbone com 58.143.808 parâmetros)

| id | Cabeça (após Dropout 0,5) | Parâmetros da cabeça |
|---|---|---|
| (a) | `Linear(2048,2)`; é também a linha "ResNet152" do Exp. 2 | 4.098 |
| (b) | MLP `[2048,128,2]`: Linear → ReLU → Linear | 262.530 |
| (c) | MLP isoparamétrica `[2048,1919,2]`: mesma profundidade, só mais larga | 3.935.871 |
| (d) | KAN `[2048,128,2]`, grade 10, ordem 3 (referência) | 3.936.000 |
| (e) opcional | KAN `[2048,8,2]` (orçamento do MLP (b)), só se sobrarem unidades | 246.000 |

Diagnóstico extra na KAN: a fração das entradas que caem dentro da grade [−1,1]. As *features* são ≥0 e o dropout as dobra no treino, então a spline pode estar pouco usada.

## Experimento 2: baselines da Tabela I no mesmo protocolo

**Regra:** o catálogo segue `leguwoi/src/models/torchvision_models.py`: as mesmas funções do torchvision e os mesmos pontos de troca da camada final (`fc`, `classifier[6]` na VGG, `classifier[1]` na SqueezeNet etc.).
- **Desvios declarados:** pesos `DEFAULT` do ImageNet em todos e **sem** a troca da `conv1`.
- **Camada final:** `Dropout(0.5)+Linear(in,2)`; na SqueezeNet, `Conv2d(512,2,1)`.
- **Fases:** a Fase 1 treina só a camada nova; a Fase 2 treina tudo.

O custo é medido em U = uma rodada 5-fold × 60 épocas da ResNet-152 com AMP. Estimativa: U ≈ **1,1 h de L4** (faixa de 0,8 a 1,6 h; a T4 seria ~2 h). A calibração substitui esses valores por medidas.

| Grupo | Modelos | Custo |
|---|---|---|
| Já coberto | ResNet152 = (a) | 0 |
| Baratos | AlexNet, MobileNetV2, MNASNet1.3, ShuffleNetV2 x0.5, SqueezeNet1.0 | ~1,0 U no total |
| Médios + Swin-B | DenseNet169 (~0,5 U), MaxViT-T (~0,7 U), **Swin-B** (~1,5 U) | ~2,7 U |
| Pesados | EfficientNetV2-L (~1,8 U; pré-treinada a 480 px, roda a 224), VGG16 (~1,2 U), RegNetY-16GF (~1,4 U) | ~4,4 U |

O que não rodar sai da tabela, ou aparece marcado como "não reavaliado", e o texto não afirma superioridade sobre esses modelos.

## Execução: Colab com assinatura, na L4

| # | Notebook | O que | L4 (estimativa) |
|---|---|---|---|
| 0 | `00` (CPU, Colab ou Mac) | Clone fixado, dobras e checagens, hash do *cache*. Grava `folds_entrada_saida.csv` no Drive. | 0 |
| 1 | `01` | Calibração e ponte (veja abaixo) | ~1,5–2 h |
| 2 | `02` | Cabeças (d) → (c) → (b) → (a), semente 42. **Isto já responde a pergunta.** | ~4,5 h |
| 3 | `03` | 5 baratos → DenseNet169 → MaxViT-T → **Swin-B** | ~4 h |
| 4 | `02` | Sementes 43 e 44 nas cabeças (b), (c), (d), e (a) se der | ~6,5–9 h |
| 5 | `03` | Pesados, se sobrarem unidades: EffNetV2-L → VGG16 → RegNetY-16GF | ~5 h |
| 6 | `05` (CPU/Mac) | Análise, tabela, bootstrap e frase | 0 |

**Orçamento em unidades de computação:**
- Não há tabela oficial de consumo; para a L4, a estimativa ainda incerta é de ~4–5 un/h.
- Essencial (passos 1–3): ~10 h de L4 ≈ 40–50 un.
- Com as sementes extras: ~19 h ≈ 75–95 un.
- Com os pesados: ~24 h ≈ 100–120 un.
- O Colab Pro dá 100 un/mês (US$ 9,99). É provável que os pesados **não** caibam num mês do Pro. O 01 mede o consumo real e diz o que cabe.

**Unidades de computação no 01:**
1. Você anota o saldo e o consumo/h que o painel "Recursos" mostra, no início e no fim (células de entrada).
2. O notebook calcula un/h = Δsaldo/Δtempo, confere com o valor do painel e estima as **unidades por rodada 5-fold de cada modelo**, a partir dos tempos de época medidos.
3. Escreve em `calibracao_l4.json` uma tabela "cabe / não cabe no saldo".
- **Lembrete:** desconecte o ambiente quando terminar; uma sessão parada também consome unidades.

**Ponte (no 01, na L4, antes de qualquer rodada):**
- **O que roda:** a KAN (d), semente 42, na dobra 1 **antiga**: `KFold(5, shuffle=True, random_state=42)` sobre `unique_cases`, na ordem do `glob` sem ordenar, como no original. Usa o código novo e é avaliada na última época.
- **Como roda:** em fp32 e em AMP, com a mesma ordem de lotes e a mesma augmentation.
- **Impressão digital da dobra:** os logs (396/496 = 79,84%; 401/496 = 80,85%) implicam **496 imagens de teste**. Se a dobra reconstruída não bater, reporto antes de comparar.
- **Critério:**
  1. **AMP adotado** se valerem as três condições:
     - |Δacc| ≤ 2 pp;
     - |ΔAUC| ≤ 0,015;
     - as duas acurácias em 79,84% ± 3 pp.
  2. Se só a parte de acurácia falhar e a AUC passar: uma 2ª execução AMP com a semente 43, avaliada contra o fp32. Se falhar de novo, tudo em fp32.
  3. Se a AUC falhar: fp32 direto.
  4. **Salvaguarda:** se o **próprio fp32** sair de 79,84% ± 3 pp, o problema não é numérico. **Paro e te reporto.**
- **Contexto:** o log antigo também foi feito na L4, então a ponte compara no mesmo tipo de GPU. A execução fp32 mede de quebra o custo fp32 (TF32 nas convoluções).

**O 01 também mede:**
- tempos isolados de época (F1/F2), pico de memória e un/h de cada modelo;
- custo de gravar checkpoint no Drive;
- teste de matar e retomar;
- teste do registro de plugins;
- se os arquivos existirem: memória e tempo das cabeças **W1 e W5** (Wav-KAN h=128 e h=480) e um **smoke test da multitarefa**. Se não existirem, as células pulam com aviso.

## Retomada, conta e trava

- **Conta:** ao subir os notebooks, crio `tcc-final/.tcc_final_id` com o id da pasta. Depois do `drive.mount`, o setup confere esse arquivo e **para antes de gravar**, se o Colab estiver em outra conta (o padrão é `authuser=0`, a dona da pasta). Os links vão com `?authuser=0`.
- **Retomada:**
  - A unidade de trabalho é (modelo, semente, dobra). Se `metricas.json` existe, a unidade é pulada; se existe `ckpt_last.pt`, retoma da época salva; senão, começa do zero.
  - O checkpoint é **por época**, com escrita atômica, e guarda modelo, otimizador, scheduler, GradScaler, época, fase, RNGs e histórico.
  - Na Fase 2 ele tem ~0,7 GB na ResNet-152 e ~1,7 GB no VGG16. Há `CKPT_MIN_INTERVAL_MIN` caso a gravação no Drive pese.
  - O checkpoint é apagado quando a dobra termina. Ficam `historico.csv`, `pred_teste.csv` e `metricas.json`.
  - Depois de uma queda, é só "Executar tudo".
- **Duas sessões simultâneas** (se a assinatura permitir):
  - **Trava:** cada unidade ganha um `lock.json` (id da sessão, horário de início, *heartbeat* atualizado a cada época). Para pegar uma unidade, a sessão grava a trava, espera ~30 s (a sincronização do Drive tem latência), relê e só segue se ainda for a dona. Uma trava com *heartbeat* de mais de 30 min é considerada abandonada.
  - **Redundância:** cada aba recebe uma **fila diferente** no setup (por exemplo, aba A com o Exp. 1 e aba B com o Exp. 2), e a trava é só a rede de segurança.

## Tabela final (planejada)

- **Painel A (cabeça; ResNet-152 fixa):** (a), (b), (c), (d) [e (e)].
- **Painel B (backbones; cabeça linear):** os baselines reavaliados, mais a TransKAN (d).
- **Colunas:** Modelo · Cabeça · Parâmetros treináveis (cabeça/total) · Acc · P macro · R macro · F1 macro · **AUC (prob.)** · Tempo/dobra (min, L4).
  - Os valores são média ± dp (5 dobras).
  - Com 3 sementes, cada dobra entra com a média das sementes, e o dp entre sementes vai em nota.
  - Unidades que rodaram na T4 levam †.
- **Contraste:** ΔAUC (KAN − MLP-iso) e (KAN − MLP-128), com IC95% por *bootstrap* de casos sobre o OOF pareado e o número de dobras em que a KAN vence.
- As colunas micro e a especificidade macro saem, por serem redundantes (vai uma nota na legenda). Os números de Lira saem da tabela.
- Saídas: `resultados/tabela_final.csv` e `.tex` (IEEEtran).

## Regra de decisão pré-registrada e frases

- **Métrica primária:** AUC (probabilidade), média entre dobras.
- **Comparação primária:** (d) contra a **melhor** MLP (b, c), pareada por dobra.
- **"A KAN melhora"** = ΔAUC > 0 em ≥4/5 dobras **e** IC95% do ΔAUC OOF excluindo 0. Com uma semente só, a frase ganha a ressalva "single training seed".

Frases (em inglês, como no manuscrito):

- **A (KAN melhor):** "Under an identical protocol (same ResNet-152 backbone, folds, augmentation and two-phase schedule), the KAN head raised AUC from X±s to Y±s relative to the best MLP head (paired ΔAUC = d, 95% CI [l, u]; better in k/5 folds), including an MLP with a matched 3.9 M-parameter budget. The improvement is therefore attributable to the head, although it is small compared with the effect of ImageNet pre-training."
- **B (empate):** "Under an identical protocol, the KAN head performed on par with linear and MLP heads (ΔAUC = d, 95% CI [l, u] includes zero). TransKAN's advantage over the baselines of Lira et al. is explained by the pre-trained ResNet-152 backbone and the two-phase fine-tuning recipe rather than by the KAN head itself; we present the KAN head as a competitive alternative, not as the source of the gain."
- **C (KAN pior):** a frase de B, dizendo que a cabeça KAN ficou abaixo das MLPs.

**Proibido sem o resultado correspondente:**
- "só trocar a cabeça melhora";
- "new SOTA";
- "outperforms transformer-based baselines" (exige Swin-B e MaxViT reavaliados);
- "parameter efficiency".

**Correções obrigatórias no artigo:**
- **Números:** substituir os da TransKAN.
- **Métodos:**
  - remover SMOTE+ENN;
  - regra de checkpoint = última época;
  - dobras por caso, estratificadas por `strat4` e com semente fixa;
  - baselines pré-treinados no mesmo protocolo;
  - AUC a partir de probabilidades.
- **Descrição do modelo:** "two-layer KAN".
- **Dados:**
  - contagens de imagens e casos tiradas do notebook 00;
  - **citar Lira et al. (2025, IJLM) como origem da base** (GuWID = FDCPUnBGunshotDB) e da tabela de referência.

## Arquivos

**Numeração:** 00, 01, 02, 03 e 05 são deste prompt (o 04 fica vago, já que o Kaggle saiu); 06–08 são do Prompt 3; 09–10 do Prompt 2.

**Novos no repositório:**
- `notebooks/protocolo_comum.py`: **editado só nesta sessão**. Contém dataset/cache, dobras, transforms, catálogo de backbones, laço de 2 fases, checkpoint/retomada/trava, métricas e `executar_fila(fila, plugins=[…])`. Pontos de extensão, para que `cabecas_wavkan.py` e `multitarefa.py` entrem por arquivos separados **sem mudar o hash**:
  - `registrar_cabeca(nome, fabrica)`, com `fabrica(in_features, n_saidas, **kw)`. As cabeças nativas `linear`, `mlp128`, `mlp_iso`, `kan` e `kan_pequena` usam o mesmo registro.
  - `registrar_config(nome, cfg)` e `registrar_perda(nome, fabrica)`.
  - **Dataset:** devolve `(img, y_es, y_dist)`.
  - **Modelo:** `backbone → Dropout → {tarefa: cabeça}`, com saída em dicionário. Cada tarefa tem pesos de classe próprios e `ignore_index=-1`.
  - **Métricas:** por tarefa; na distância, a AUC é OvR macro, só nas entradas.
  - **Plugins:** são importados por caminho, antes de a fila ser lida.
- `notebooks/00_preparar_dados_e_folds.ipynb`
- `notebooks/01_calibracao_L4.ipynb`
- `notebooks/02_ablacao_cabeca_ResNet152.ipynb`
- `notebooks/03_baselines_mesmo_protocolo.ipynb`
- `notebooks/05_analise_tabela_final.ipynb`
- `resultados/`: `folds_entrada_saida.csv`, `calibracao_l4.json`, `ablacao_cabeca/{runs.csv, resumo.json, oof.csv}`, `baselines/{runs.csv, resumo.json, oof.csv}`, `tabela_final.{csv,tex}`. Em cada `runs.csv`, 1 linha = modelo × semente × dobra.

**Alterados:** `.gitignore` (ganha `*.pt`, `*.pth` e `*.npy`). Os 8 notebooks antigos não mudam.

**Drive (`tcc-final/`):** os 5 notebooks, o `protocolo_comum.py` e o sentinela, enviados pelo conector do Drive. As sessões criam `dados/` (só o CSV), `resultados/` e `checkpoints/`.

## Riscos

- **Os números vão mudar.** A TransKAN deve ficar abaixo de 0,860/0,908, e baselines pré-treinados (Swin-B, DenseNet, EffNetV2-L) podem empatar com ela.
- **Pouco poder estatístico.** O dp da AUC entre dobras deve ficar em ~0,02–0,03, e diferenças entre cabeças abaixo de ~0,01 não serão distinguíveis. O desfecho B é plausível.
- **Unidades de computação:** a taxa da L4 não é publicada e varia. O 01 mede antes de qualquer rodada longa.
- **Quedas mesmo com assinatura:** a retomada por época limita a perda a ≤1 época.
- **Duas sessões:** como o Drive tem latência, a trava não é atômica. Por isso cada aba recebe uma fila diferente e a trava fica só como segurança.
- **Espaço no Drive:** ≤1,7 GB de checkpoint por unidade ativa.
- **Ponte:** se cair para fp32, o custo sobe ~1,5–2× (a L4 tem TF32) e as sementes extras e os pesados saem do mês.
- **NaN com AMP:** o GradScaler cuida da maior parte e a cabeça roda em fp32. Se persistir, aquele modelo roda em fp32 e isso é anotado.
- **Falta de memória:** improvável na L4 com 24 GB; se acontecer, acumulação 2×16, como desvio declarado.
- **Comportamento questionável de alguns modelos:** a SqueezeNet pode colapsar (0,500), e a EffNetV2-L roda a 224 px, embora pré-treinada a 480. Os dois casos são reportados como estão.
- **Clone a cada sessão (~735 MB):** se o GitHub ou o GitLab estiverem fora do ar, a sessão não começa (a checagem do commit para com mensagem clara). Versões diferentes do OpenCV podem mudar alguns pixels do *cache*; o hash é comparado e, se divergir, a diferença máxima vai para o JSON.

## Verificação

- **00:**
  - 2.554 imagens, 486 casos e 0 casos compartilhados entre dobras;
  - regra de composição, com a semente usada registrada;
  - fração de saídas por dobra a ±3 pp de 26,3% (se falhar, aviso antes de treinar);
  - `y_dist`: 42/232/1.609, e −1 nas 671 saídas;
  - `obs` com 4 anomalias (se aparecer outro número, aviso);
  - *cache* igual a `A.Resize` em 5 imagens;
  - colisões com a chave do leguwoi.
- **01:**
  - sentinela da conta conferido antes da primeira gravação;
  - GPU = L4;
  - contagem de parâmetros (4.098 / 262.530 / 3.935.871 / 3.936.000);
  - ponte com 496 imagens de teste e o critério acima;
  - nenhum `UserWarning`;
  - teste de matar e retomar com histórico idêntico;
  - teste de plugin sem mudar o hash;
  - un/h e unidades por rodada;
  - W1/W5 e multitarefa, se os arquivos existirem.
- **02/03:**
  - primeiro uma execução curta (1 dobra × 2 épocas, numa pasta de teste);
  - cada `metricas.json` com GPU, hash e commits.
- **05:**
  - recalcular as métricas a partir de `pred_teste.csv` e conferir com o JSON;
  - AUC ≠ revocação macro em toda linha;
  - hashes iguais em todas as execuções;
  - sanidade: a acurácia por dobra de (d), semente 42, entre 80% e 89%.

## Regras para as sessões em nuvem (14/10/2026)

- Cada sessão trabalha no próprio branch e abre um PR para `main`. Não faça push direto em `main`.
- O PR do P1 (protocolo) é mergeado primeiro. P2 e P3 atualizam o branch a partir de `main` depois disso e **nunca editam `notebooks/protocolo_comum.py`** (o SHA-256 precisa ser igual em todos os prompts).
- Adicione só os próprios arquivos pelo caminho (nada de `git add -A`). Não commite `references/leguwoi/` (clone do GitLab no commit `83d82f1` quando precisar), PDFs do `sibgrapi2026/` nem da dissertação.
- O manuscrito (`~/Downloads/manuscript 6.pdf`) não está disponível na nuvem: os fatos necessários já estão neste plano e em `auditoria-consistencia.md` (raiz do repo, leia).
- Os testes descritos como "no Mac" rodam no container da nuvem, em CPU.
- Drive (pasta `tcc-final`, id `1i_o4gxs-nyzZOPFp21gOwI_AMHPzct-4`, conta tutuscarneiro@gmail.com): se o conector do Google Drive estiver disponível, suba **só os seus arquivos**, atualizando o arquivo existente com o mesmo nome em vez de criar outro. Se não estiver, deixe tudo no repo e diga no PR quais arquivos o Arthur deve subir. Nunca suba imagens, cache (`.npy`) ou checkpoints.
- Caminho único do CSV de dobras: `dados/folds_entrada_saida.csv` (repo e Drive).
- Flag de GPU de reserva: `PERMITIR_RESERVA`. Última célula de todo notebook de GPU: `DESLIGAR_AO_FIM=True` → `runtime.unassign()`.
- Uma única trava (`lock.json`, heartbeat de 30 min), implementada no protocolo; os plugins a reutilizam.
- Prioridade global de unidades de computação (Colab Pro, ~100 UC/mês, tudo na L4): 00 → 01 → cabeças (d),(c),(b),(a) s42 → P3: W1, W2 → P2: `st_dist_kan`, `mt_kan` → baselines baratos/médios + Swin-B → sementes 43/44 só de (d) e da melhor MLP com o saldo restante. Pesados, W3–W5 e o par MLP da multitarefa ficam para o mês seguinte.

### Ajustes finais específicos do P1
1. Mantenha o CLI `python protocolo_comum.py executar --fila fila.json --plugins <arquivo.py> [--gpu 0]` além de `executar_fila()`: P2 e P3 usam o CLI.
2. Mova o CSV de dobras para `dados/folds_entrada_saida.csv` (tire de `resultados/`).
3. Não afirme que o log antigo rodou na L4: o notebook só imprime "cuda" e o Arthur usou T4 e L4 na época. A tolerância de ±3 pp da ponte cobre isso.
4. No 01, gere a tabela "cabe / não cabe" seguindo a prioridade global acima.
