# Plano — Prompt 2: rede única para Entrada/Saída + distância do disparo (MLSD)

## Contexto
Hoje são duas redes separadas: `TransferKAN_EntradaSaida` (2 classes) e `TransferKAN_MLSD` (3 classes: encostado/curta/distância). O objetivo é **uma** ResNet-152 com extrator compartilhado e duas cabeças, que preveja as duas coisas. Vamos comparar cabeça KAN × MLP e medir contra as redes de tarefa única (ST), **nas mesmas dobras e no mesmo protocolo**.

Este plano **não altera** o protocolo do Prompt 1 (`…drifting-kettle.md`, aprovado):
- usa o `protocolo_comum.py` como está (o hash é o mesmo em todos os prompts);
- tudo o que é multitarefa (MT) fica no plugin `multitarefa.py`;
- todas as rodadas são na **L4 do Colab com assinatura**, a mesma GPU do Prompt 1;
- o resultado entra **só no TCC**; artigo e pôster mantêm "multitarefa" como trabalho futuro.

## O que a leitura revelou
1. **Não são dois datasets.** As 1.883 imagens da MLSD são exatamente `ENTRADAS_EQX/{ENCOSTADO,CURTA,DISTANCIA}`, um subconjunto das 2.554 de Entrada/Saída. A "junção" é a tabela única `folds_entrada_saida.csv` do Prompt 1, com `y_dist = −1` nas 671 saídas.
2. **Os casos misturam as tarefas.** Pelo CSV do dataset (commit `f679389`):
   - 194 casos têm entrada **e** saída;
   - cerca de 45 casos têm só saída;
   - **encostado aparece em só 16 casos** (42 imagens); curta em 90 e distância em 392.

   Como as dobras são definidas uma vez sobre a união (estratificadas por `strat4`, regra de ≥2 casos de encostado por dobra), a ST de distância usa essas mesmas dobras restritas às entradas. Assim, nenhum caso atravessa dobras em nenhuma tarefa.
3. **CSV × pastas:**
   - 3 saídas estão fora do CSV;
   - 1 saída tem `distance=YD`;
   - todo o resto bate 100%.

   As quatro anomalias já entram na coluna `obs` do `00` (Prompt 1). A pasta continua sendo a fonte de verdade.
4. **O que o Prompt 1 já expõe e eu reaproveito sem editar:**
   - `registrar_cabeca`, `registrar_config` e `registrar_perda`;
   - o dataset multirrótulo `(img, y_es, y_dist)`;
   - o modelo `backbone → Dropout(0.5) → {tarefa: cabeça}`, que devolve um dict de logits;
   - pesos de classe por tarefa e `ignore_index=−1`;
   - Fase 1 treinando todas as cabeças novas;
   - métricas por tarefa (distância: AUC OvR macro só nas entradas);
   - o CLI `executar --fila … --plugins multitarefa.py`.
5. **A MLSD antiga não serve de referência:** era uma fase só, lr 1e-4, grade 5, torchvision e checkpoint escolhido no teste (F1 macro 0,526; AUC 0,749). A ST de distância é **retreinada**.
6. **Auditoria (lida):**
   - nada de "parameter efficiency" da cabeça KAN; aqui o argumento é **uma rede no lugar de duas**, com os parâmetros contados;
   - AUC sempre de probabilidades;
   - a base não tem licença de redistribuição;
   - o TCC deve citar Lira et al. (2025, IJLM) como origem da base.

## Plugin `notebooks/multitarefa.py`
É o único lugar com código MT. Usa os ganchos do Prompt 1 e não reimplementa o backbone nem o laço, para que a ST-ES do Prompt 1 e a MT passem pelo **mesmo caminho de código**.
- **`MultiHeadNet`:** uma fábrica fina sobre o modelo de dicionário do protocolo.
  - Cabeças vindas do registro: **KAN** = `kan` (d) `[2048,128,k]`, grade 10, ordem 3; **MLP** = `mlp128` (b), com ReLU.
  - Cabeças em fp32 e backbone em AMP, como no protocolo.
- **Dataset multitarefa:** uma visão sobre o dataset multirrótulo.
  - `st_dist_*` **treina só nas linhas com `y_dist ≥ 0`**, mas **prevê em todas as linhas de teste da dobra** (a métrica de pipeline precisa disso). As métricas da tarefa usam só as entradas.
  - A MT treina e avalia em todas as linhas.
- **Perda com máscara** (`registrar_perda('soma_mascarada')`): `L = L_es + λ·L_dist`, com **λ = 1 fixo**.
  - CE ponderada por tarefa: n/(K·n_c) na dobra de treino. Para a distância, o cálculo usa **só as entradas** (≈15 / 2,7 / 0,39).
  - `ignore_index=−1`.
  - **Guarda:** se o lote não tiver nenhuma entrada, `L_dist = 0`, para não gerar NaN.
  - Sem oversampling. Ponderação por incerteza e cabeça única de 4 classes ficam fora do núcleo.
- **Configs registradas:** `st_dist_kan`, `st_dist_mlp`, `mt_kan`, `mt_mlp`, todas com semente 42 e `salvar_pesos_finais` só em `mt_kan`.
- **Métrica de pipeline e benchmark de inferência** (detalhes abaixo).
- O SHA-256 do plugin vai no JSON, ao lado do hash do protocolo.
- **Se faltar algum gancho no protocolo congelado** (por exemplo, filtros separados de treino e de avaliação), paro e te aviso. Não edito `protocolo_comum.py`.

| Config | Parâm. totais | Origem |
|---|---|---|
| `st_es_kan` | 62.079.808 | **Prompt 1 (d), s42** (não roda de novo) |
| `st_es_mlp` | 58.406.338 | **Prompt 1 (b), s42** |
| `st_dist_kan` | 62.081.728 | este plano |
| `st_dist_mlp` | 58.406.467 | este plano |
| `mt_kan` | **66.017.728** (o par ST soma 124.161.536 → −47%) | este plano |
| `mt_mlp` | **58.668.997** (o par ST soma 116.812.805 → −50%) | este plano |

## Métricas e comparação
Por dobra, com média ± dp (ddof=1) das 5 dobras e mais o OOF. A AUC sempre sai do softmax.
- **Entrada/Saída** (todas as imagens de teste): acurácia, P/R/F1 macro e AUC = `roc_auc_score(y, P(saída))`.
- **Distância** (as mesmas entradas de teste para ST e MT, conferido com `assert`):
  - acurácia, com a nota de que prever sempre a classe majoritária já dá 0,854;
  - P/R/F1 macro e AUC OvR macro;
  - recall/F1 por classe e matriz de confusão, calculados no OOF (são ~8 imagens de encostado por dobra).
- **Pipeline de 4 classes** (secundária, sem treino extra): F1 macro em {saída, encostado, curta, distância}.
  - ST: a rede ES decide; se ela disser "entrada", entra a rede de distância. A junção com o `pred_teste.csv` do Prompt 1 é por `rel_path`.
  - MT: as duas cabeças da mesma rede.
- **Custo:**
  - parâmetros totais e treináveis por fase;
  - tempo por dobra = 15·t_F1 + 45·t_F2, pela calibração do Prompt 1 na **L4**;
  - tempo de parede real por unidade;
  - **benchmark de inferência na L4** para obter as duas saídas: a MT faz 1 forward e o par ST faz 2. Medido em ms/img com lote 1 e lote 32, com 20 aquecimentos + 200 medições por eventos CUDA;
  - pico de memória.
- **Regra pré-registrada:**
  - Δ = MT − ST, pareado por dobra, com IC95% por bootstrap de **casos** (2.000 reamostragens) sobre o OOF;
  - **não inferior**: o limite inferior do IC de ΔAUC fica acima de −0,02, a mesma margem do Prompt 3;
  - **melhora**: o IC exclui 0 **e** a MT vence em ≥4 das 5 dobras;
  - a mesma regra vale para KAN × MLP dentro da MT;
  - a frase do texto leva a ressalva "single training seed";
  - nada de afirmação específica sobre encostado (são 16 casos).

## Arquivos
**Novos no repositório, com cópia no Drive `tcc-final/` pelo conector:**
- `notebooks/multitarefa.py`: o plugin descrito acima.
- `notebooks/09_multitarefa_entrada_saida_distancia.ipynb` (Colab, **L4**)
  - Setup do Prompt 1:
    - sentinela `.tcc_final_id`;
    - clones fixados e `pip` fixado;
    - **cache gerado na sessão, nunca copiado**;
    - registro do nome da GPU (T4 só como reserva, com a troca anotada no JSON).
  - Modos:
    - `smoke`: 1 dobra, 64 imagens, 2+2 épocas, com desconexão e retomada;
    - `calibracao`: t_F1/t_F2 das 4 configs + benchmark de inferência na L4;
    - `fila`: chama o CLI do Prompt 1 com `--plugins multitarefa.py`, retomando por época no Drive.
  - **`lock.json` por unidade**, porque pode haver duas sessões simultâneas:
    - grava sessão, início e *heartbeat* a cada época;
    - uma unidade com *heartbeat* recente é pulada; uma com *heartbeat* com mais de 20 min de atraso é retomada;
    - o lock é relido antes de cada gravação de checkpoint.
  - Última célula: `DESLIGAR_AO_FIM=True` chama `runtime.unassign()`, porque uma sessão ociosa continua gastando unidades.
- `notebooks/10_analise_multitarefa.ipynb` (CPU/Mac)
  - junta os resultados e o ST-ES do Prompt 1;
  - gera tabelas, bootstrap, matriz de confusão e pipeline;
  - aplica a regra de decisão;
  - grava `tabela_multitarefa.{csv,tex}`;
  - escreve trechos de texto para o TCC (EN + PT).
- `resultados/multitarefa/`:
  - `<config>/{runs.csv, resumo.json, oof.csv}` (1 linha = config × semente × dobra);
  - `calibracao_mt.json` (t_F1, t_F2, benchmark, GPU);
  - `tabela_multitarefa.{csv,tex}`.

**Drive privado:** só os pesos finais da `mt_kan`, em fp16 (~0,65 GB nas 5 dobras), em `tcc-final/pesos/mt_kan/`. Os checkpoints de retomada são apagados ao fim de cada dobra.

**Não mudam:** `protocolo_comum.py`, os notebooks 00–08 e os 8 notebooks antigos.

## Ordem de execução e unidades de computação (UC)
- **Fórmula:** horas_L4(config) = 5 × (15·t_F1 + 45·t_F2) × f_img + ~5 min de setup por sessão, e **UC = horas_L4 × taxa da L4**.
  - t_F1 e t_F2 vêm da calibração do Prompt 1 e do `09`.
  - f_img = 1 com 2.554 imagens e ≈ 0,74 nas configs só de entradas.
  - A taxa da L4 se lê em "Recursos" no Colab; hoje fica em torno de ~4,8 UC/h, confira.
- **Até a calibração existir:** usei U_L4 (5 dobras × 60 épocas da ResNet-152, AMP) ≈ **0,8–1,2 h**, cerca de 2× a T4. Os números abaixo serão substituídos pelos medidos.

| Passo | O quê | L4 (AMP) | UC (~4,8/h) |
|---|---|---|---|
| 0 | Eu, no Mac: plugin, 09 e 10; smoke local em CPU (sintético, ResNet-18, interrupção e retomada) contra o protocolo já implementado; upload para o Drive | 0 | 0 |
| 1 | `09` `smoke` + `calibracao` / benchmark | 0,3–0,5 h | ~2 |
| 2 | **Logo depois das cabeças do Prompt 1:** sessão A com `st_dist_kan`, sessão B com `mt_kan`, em paralelo | 0,6–0,9 h + 0,8–1,2 h | ~7–10 |
| ← | **Já responde "MT-KAN × ST-KAN"** (com a (d) do Prompt 1) | | **~9–12 UC no acumulado** |
| 3 | **Só se sobrarem unidades:** sessão A com `st_dist_mlp`, sessão B com `mt_mlp` | 0,6–0,9 h + 0,8–1,2 h | ~7–10 |
| 4 | `10` análise (Mac) | 0 | 0 |

- **Total do prompt:** ~3,1–4,7 h de L4, **~15–23 UC**. As sessões em paralelo dividem o tempo de parede, mas não o custo em UC.
- **Se a ponte do Prompt 1 cair para fp32:** custo ~2×, ou seja, ~30–45 UC, e o par MLP provavelmente não cabe.
- **Não cabe no núcleo:** 3 sementes (+~2× o total), ponderação por incerteza (~+1 h) e cabeça de 4 classes (~+1 h).

## Riscos
- **Rótulo faltando:** a máscara com guarda resolve. A distância não é propagada do caso para a saída: a saída não mostra fuligem nem tatuagem, e um caso pode ter vários disparos.
- **Conjuntos de imagens diferentes:**
  - na MT, a cabeça ES vê 2.554 imagens e a cabeça de distância vê 1.883;
  - a ST de distância treina só nas entradas, para que a BN e o tamanho efetivo do lote sejam os de uma rede de distância de verdade;
  - a avaliação é pareada nas mesmas entradas.
- **Desbalanceamento:**
  - encostado é 2,2% das entradas (peso ≈ 38× o de distância), com ~3 casos por dobra;
  - a cabeça de distância pode colapsar na classe majoritária. Checagem de sanidade: F1 macro > 0,31, que é o que se obtém prevendo sempre a majoritária;
  - os ICs da distância vão sair largos, e "inconclusivo" é um desfecho plausível.
- **Transferência negativa ou positiva:**
  - a ES tem perda em todas as imagens e pode dominar o extrator;
  - por outro lado, fuligem e tatuagem também são pistas de entrada;
  - os picos das duas tarefas podem cair em épocas diferentes, mas a regra da última época trata ST e MT igualmente;
  - com o Dropout compartilhado do protocolo, as duas cabeças veem a mesma máscara (é a mesma rede).
- **`GaussNoise(std_range=(0.2,0.44))` "como rodou"** pode apagar a tatuagem e prejudicar a classe curta. O efeito é o mesmo em ST e MT, então a comparação continua justa, mas vira limitação no texto.
- **Duas sessões no mesmo Drive:**
  - o Drive sincroniza entre VMs com atraso, então o `lock.json` sozinho não é atômico;
  - por isso as sessões recebem **filas disjuntas** (A e B acima), e o lock é só rede de segurança.
- **L4 indisponível:** a assinatura não garante L4. A unidade roda na T4 de reserva, com a GPU registrada por dobra; a coluna de tempo continua vindo da calibração na L4.
- **Gasto de UC à toa:** uma sessão ociosa conectada continua consumindo, daí o `runtime.unassign()` no fim.
- **Dependência do Prompt 1:**
  - o plugin só roda depois do protocolo congelado e das dobras `strat4` gravadas pelo `00`;
  - se faltar um gancho, aviso antes de qualquer rodada.

## Verificação
- **Mac (CPU):**
  - perda com máscara: lote sem nenhuma entrada dá valor finito e nenhum gradiente na cabeça de distância; com uma tarefa só, a perda é igual à da ST;
  - contagens exatas de parâmetros (tabela acima);
  - AUC ≠ recall, e a AUC OvR sai de probabilidades;
  - interromper e retomar dá histórico idêntico;
  - lock: duas instâncias simuladas não pegam a mesma unidade;
  - o `10` roda com previsões falsas;
  - o hash de `protocolo_comum.py` não muda.
- **Antes do passo 2:** conferir no `00` do Prompt 1:
  - 2.554 imagens, 486 casos, 0 casos compartilhados entre dobras;
  - ≥2 casos de encostado por dobra;
  - saída a ±3 pp de 26,3%;
  - as 4 anomalias em `obs`.
- **`09`:**
  - o `smoke` retoma na época certa;
  - a calibração grava t_F1/t_F2 e o benchmark com a GPU = L4;
  - as UC são recalculadas e eu te reporto antes do passo 2.
- **Depois de cada config:**
  - as métricas recalculadas a partir do `oof.csv` batem com o JSON;
  - acurácia ES por dobra entre 80% e 89%;
  - F1 macro de distância > 0,31;
  - os hashes do protocolo, do plugin e das dobras são iguais em todas as unidades.
- **`10`:** reproduz a tabela a partir dos CSVs e aplica a regra de decisão automaticamente.

## Respostas recebidas
- Auditoria lida: não há conflito; só as correções de texto acima.
- Dobras: `strat4`, ordenadas por `rel_path`, no `folds_entrada_saida.csv` do Prompt 1, com a regra de encostado lá. Não crio outro arquivo.
- `protocolo_comum.py` não é editado. Sem validação interna; `drop_last=True` para todos.
- O cache é gerado em cada sessão e nunca vai para o Drive.
- Infraestrutura: Colab com assinatura, tudo na L4 e a T4 só como reserva registrada. Sem Kaggle e sem Pay-As-You-Go. `lock.json` por unidade.
- Prioridade: `st_dist_kan` → `mt_kan` logo depois das cabeças do Prompt 1; o par MLP só se sobrarem unidades.
- Numeração `09`/`10`. Resultado só no TCC. Pesos: só os da `mt_kan`, no Drive privado.

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

### Ajustes finais específicos do P2
1. Não reimplemente a trava: use a do protocolo.
2. No 09, o modo calibração só mede o que o 01 do P1 não mediu (t_F1/t_F2 das configs só de entradas e o benchmark de inferência); o smoke da MT já roda no 01.
3. Execução: `st_dist_kan` → `mt_kan` depois das W1/W2 do P3; o par MLP fica para o mês seguinte.
