# Plano — Prompt 3: cabeça Wav-KAN (KAN com wavelets) na ResNet-152, no protocolo único do Prompt 1

## Contexto
A cabeça atual da TransferKAN é uma KAN spline da efficient-kan, `[2048,128,2]`, com grade 10 e ordem 3. São **3.936.000 parâmetros** (15 por aresta). A auditoria (`auditoria-consistencia.md`) marca como "questionável" falar em *parameter efficiency* com esse número, já que um MLP equivalente tem ~262 k.

Este plano testa se uma **Wav-KAN** (Bozorgasl & Chen, arXiv 2405.12832) chega a um desempenho parecido com menos parâmetros e menos tempo. A comparação usa **exatamente** o protocolo aprovado do Prompt 1 (plano `drifting-kettle`):
- `protocolo_comum.py` **sem edição**, com o mesmo SHA-256;
- `folds_entrada_saida.csv` (estratificado por `strat4`, ordenado por `rel_path`, grupo `filename[:9]`);
- o cache gerado na própria sessão (nada vai para o Drive);
- o augmentation "como rodou";
- as 2 fases;
- AMP ou fp32, conforme a checagem de ponte do 01;
- `drop_last=True`;
- **última época, com o treino inteiro da dobra (sem validação interna)**;
- AUC calculada a partir de probabilidades.

As referências (a)–(d), com semente 42, vêm do Prompt 1. Aqui só se treinam as cabeças Wav-KAN.

**Infraestrutura:** Colab com assinatura, **sem Kaggle**. Todas as rodadas são na **L4**, o mesmo tipo de GPU do Prompt 1. A T4 fica só como reserva, registrada no JSON. Os notebooks ficam em `notebooks/` e na pasta do Drive `Colab Notebooks/tcc-final` (id `1i_o4gxs-nyzZOPFp21gOwI_AMHPzct-4`).

## Interfaces do Prompt 1 que uso (não edito nenhum arquivo dele)
- `registrar_cabeca(nome, fabrica)`, com `fabrica(in_features, n_saidas, **kw) -> nn.Module`.
- O CLI `python protocolo_comum.py executar --fila fila.json --plugins cabecas_wavkan.py --gpu 0`.
  - Os plugins são importados antes de ler a fila.
  - O SHA-256 do plugin vai no JSON separado do hash do protocolo, que portanto não muda.
- Também do Prompt 1, sem mudança:
  - setup de sessão: sentinela `.tcc_final_id` da conta do Drive, clones fixados (dataset `f679389`, efficient-kan `7b6ce1c`), `albumentations==2.0.8`, cache local com verificação de hash;
  - retomada por época no Drive (`ckpt_last.pt`, apagado ao fim da dobra);
  - saídas `historico.csv`, `pred_teste.csv`, `metricas.json` e `runs.csv`/`resumo.json`/`oof.csv`;
  - o campo de GPU em cada execução.

## Achados específicos deste prompt
- **Auditoria:**
  - *parameter efficiency* é questionável para a spline. Este experimento é o que pode sustentar, ou derrubar, uma frase de eficiência para cabeças KAN.
  - A base não tem licença de redistribuição, então nenhuma imagem nem cache sai da sessão. É obrigatório citar Lira et al. (2025).
  - A última época perde < 1 pp para a "melhor", o que reforça a regra sem validação.
- **Não existe Wav-KAN oficial no pip.** Não há bloqueio no Colab, mas há ressalvas em cada opção:
  - `zavareh1/Wav-KAN`: código só no GitHub, sem licença declarada. O `weight1` dele não entra na saída, mas **infla a contagem de parâmetros**.
  - `IvanDrokin/torch-conv-kan` tem `WavKANLayer`: ramo base SiLU + wavelet + BN, Apache-2.0. Precisa de `git clone`, e o pip `convkan` é outro projeto.
  - `torch-wavelet-kan` (PyPI): de terceiros, só Morlet, com fonte não verificada.
  - → **Implementação própria** em PyTorch puro, sem dependência nova, validada contra o torch-conv-kan e contra uma referência em NumPy.
- **"Menos tempo" quase certamente não aparece ponta a ponta:**
  - a ResNet-152 faz ~11,5 GFLOPs por imagem, e qualquer uma dessas cabeças fica abaixo de 0,1 % disso;
  - a Wav-KAN materializa um tensor (lote × saída × entrada), de custo O(B·in·out), e é limitada por banda de memória;
  - **na L4 isso pesa mais que na T4**: a banda da L4 (~300 GB/s) é parecida com a da T4 (~320 GB/s), mas o backbone em fp16 roda ~2× mais rápido; a fração de tempo da cabeça Wav-KAN cresce, sobretudo com h=480 e na Fase 1;
  - no modelo inteiro, os parâmetros caem só de 62,1 M para 59,2 M (h=128) ou 58,4 M (h=32).
  - → O tempo é medido nos dois níveis, e a frase permitida é "cabeça N× menor".

## Implementação — `notebooks/cabecas_wavkan.py` (único arquivo de código novo)
- **`WavKANLinear(in, out, wavelet)`**, igual ao `WavKANLayer` do torch-conv-kan:
  - parâmetros: `scale` (out,in) iniciado em 1, `translation` (out,in) iniciado em 0, `wavelet_weights` (out,in) com Kaiming e `base_weight` (out,in) aplicado a SiLU(x);
  - saída: `BatchNorm1d(wavelet + base)`;
  - **4 parâmetros por aresta**, mais a BN em cada camada;
  - guarda |scale| ≥ 1e-3, preservando o sinal (não altera nada na inicialização).
- **Wavelets:** Mexican hat (padrão do artigo), DoG e Morlet (ω₀ = 5, fixo).
  - Fica de fora a Meyer: é definida por partes, é lenta e tem gradiente zero nos platôs.
  - Fica de fora a Shannon: na referência, a janela de Hamming é aplicada ao longo do índice das features, o que é um artefato.
- **Registro:** `WavKANHead(2048, h, 2, wavelet)`, com 2 camadas, entra como `wavkan_{mexhat|dog|morlet}_h{32|128|480}` via `registrar_cabeca`. O laço do Prompt 1 roda a cabeça em fp32 e otimiza "os parâmetros novos" na Fase 1.
- **`n_parametros(h) = 8202·h + 4`**, com assert no registro.
- **Diagnóstico:** média e desvio de `scale` e `translation` ao fim de cada fase, vindos só dos pesos. Vai para `metricas.json` e é o equivalente ao diagnóstico "fração dentro da grade" que o Prompt 1 faz para a spline.
- **Sem tuning:** otimizador, lr, wd, dropout e épocas são os da receita.

## Configurações e orçamento de parâmetros (backbone com 58.143.808)
| id | Cabeça | Parâm. da cabeça | Treináveis na fase 2 | Papel |
|---|---|---|---|---|
| (a) | Linear(2048,2) | 4.098 | 58.147.906 | Prompt 1 |
| (b) | MLP [2048,128,2] | 262.530 | 58.406.338 | Prompt 1 |
| (c) | MLP-iso [2048,1919,2] | 3.935.871 | 62.079.679 | Prompt 1 |
| (d) | KAN spline [2048,128,2] G10 k3 | 3.936.000 | 62.079.808 | Prompt 1: **referência principal** |
| **W1** | Wav-KAN Mexican hat, h=128 | 1.049.860 | 59.193.668 | mesma topologia de (d), **3,7× menor** |
| **W2** | Wav-KAN Mexican hat, h=32 | 262.468 | 58.406.276 | **mesmo orçamento de (b)** (Δ = −62), **15× menor** que (d) |
| W3 | Wav-KAN DoG, h=128 | 1.049.860 | 59.193.668 | só sensibilidade à wavelet |
| W4 | Wav-KAN Morlet, h=128 | 1.049.860 | 59.193.668 | só sensibilidade à wavelet |
| W5 | Wav-KAN Mexican hat, h=480 | 3.936.964 | 62.080.772 | mesmo orçamento de (d) (Δ = +964) |

A wavelet dos orçamentos casados é fixada **a priori** (Mexican hat), sem nenhuma seleção pelos dados. W3 e W4 são reportadas só como sensibilidade e não geram novas execuções.

## Regra pré-registrada (no código e no topo do 08, antes de rodar)
- **Métrica primária:** AUC (prob.), média entre dobras. A F1 macro é secundária. Tudo com semente 42 e a ressalva "single training seed".
- **Não inferioridade de W1, W2 e W5 frente a (d):**
  - **margem de −0,02**: o limite inferior do IC95% de ΔAUC e de ΔF1 (W − d) fica acima de −0,02;
  - IC calculado por *bootstrap* pareado de casos (2.000 reamostragens) sobre o OOF, que tem as mesmas imagens;
  - mais o número de dobras em que W não perde.
  - A margem segue o dp entre dobras de ~0,02–0,03 previsto no Prompt 1.
- **W2 frente a (b) (mesmo orçamento):**
  - a regra de superioridade do Prompt 1: ΔAUC > 0 em ≥4/5 dobras **e** IC95% excluindo 0;
  - **só então** vale escrever "*more parameter-efficient than an MLP*";
  - se não, a frase é "*on par with an MLP of equal size*".
- **Tempo:**
  - "mais rápida" só se o benchmark da cabeça na L4 mostrar ganho além do ruído (média ± dp de 200 medições) **e** o tempo por dobra calibrado na L4 também cair;
  - se não, a frase é "*fewer head parameters at equal wall-clock time*", sempre com a ressalva de que o backbone domina o custo.
- **Frase-modelo:** "A wavelet-KAN head (Mexican hat; 1.05 M / 0.26 M parameters) matched the spline-KAN head (3.94 M) within the pre-registered non-inferiority margin (ΔAUC = d, 95% CI [l, u]; single training seed)…"
  - Se não chegar à não inferioridade: "…did not reach non-inferiority…".

## Arquivos
**Novos no repo (e cópias no Drive `tcc-final/`, enviadas pelo conector):**
- `notebooks/cabecas_wavkan.py`: veja acima.
- `notebooks/06_benchmark_cabecas.ipynb` (**Colab L4**, ~0,3 h):
  1. Contagem exata de parâmetros, com assert, de (a)–(d) e W1–W5.
  2. Equivalência numérica com o `WavKANLayer` do torch-conv-kan (`git clone`) e com a referência em NumPy, nas 3 wavelets (atol 1e-5, fp32).
  3. **Só a cabeça**, sobre tensores aleatórios (sem imagens): fwd+bwd com lote 32, inferência com lotes 1 e 256, memória de pico, por eventos CUDA (20 de aquecimento + 200 medidas). `torch.compile` opcional para todas.
  4. **t_F1 e t_F2 ponta a ponta** de W1–W5, com a mesma rotina de calibração do 01. Ela entra na fórmula de tempo por dobra do Prompt 1: 15·t_F1 + 45·t_F2.
     - (b) e (d) são medidos de novo na mesma sessão, como âncoras, para detectar diferença entre máquinas L4.
     - O cache é gerado na sessão.
  5. Converte os tempos em **horas e unidades de computação previstas** para cada W, usando a taxa de UC/h da L4 registrada pelo Prompt 1.
  - Grava `resultados/benchmark_cabecas.json` com GPU, versões e hash do plugin. Para se a GPU não for L4, a menos que `PERMITIR_T4=True`, e nesse caso marca o JSON.
- `notebooks/07_wavkan_cabeca_ResNet152.ipynb` (**Colab L4**, executor principal):
  - Monta a fila no formato `fila.json` do Prompt 1 e roda `executar --fila … --plugins cabecas_wavkan.py`, com retomada por época no Drive (depois de uma queda, é só "Executar tudo").
  - Fila padrão: `["W1", "W2"]`. `INCLUIR_SENSIBILIDADE=True` acrescenta W3 → W4 → W5.
  - Antes de cada unidade, imprime as UC previstas pelo 06.
  - Se a GPU não for L4, pede confirmação (`PERMITIR_T4`). Quando roda na T4, isso fica registrado no `metricas.json` da dobra, porque o Prompt 1 já grava a GPU.
  - Modo `SMOKE=True`: 1 dobra × 2+2 épocas; desconecte no meio e rode de novo para ver a retomada.
- `notebooks/08_analise_wavkan.ipynb` (CPU/Mac):
  - lê `resultados/ablacao_cabeca/` (Prompt 1), `resultados/wavkan/` e `benchmark_cabecas.json`;
  - recalcula as métricas a partir dos `pred_teste.csv`;
  - faz o bootstrap e aplica a regra;
  - gera `resultados/tabela_wavkan.{csv,tex}` (IEEEtran) e `resultados/fig_auc_vs_parametros.pdf` (AUC × parâmetros da cabeça, eixo log, barras = dp entre dobras).
- `resultados/wavkan/{fila.json, runs.csv, resumo.json, oof.csv}`, no formato do Prompt 1 (1 linha = cabeça × semente × dobra, com a GPU).

**Alterados:** nenhum. `protocolo_comum.py`, os notebooks 00–05 e os 8 notebooks antigos continuam intactos.

## Ordem de execução
0. **Eu, no Mac:**
   - escrevo `cabecas_wavkan.py`, o 06, o 07 e o 08;
   - teste local em CPU: forma da saída, `8202·h+4`, equivalência com NumPy nas 3 wavelets, e um plugin carregado pelo CLI;
   - quando o `protocolo_comum.py` existir: 2+2 épocas com dados sintéticos, incluindo interromper e retomar;
   - subo os 3 notebooks e o `.py` para `tcc-final`.
1. **(Prompt 1)** 00 → 01 (calibração e ponte AMP/fp32 na L4, com a taxa de UC/h registrada) → cabeças **(d) e (b)**.
2. **L4:** `06_benchmark_cabecas` e `07` com `SMOKE=True`. São baratos e podem rodar antes ou durante o passo 1.
3. **L4, logo depois de (d)/(b):** `07` com a fila **W1 → W2**.
4. **L4, só se sobrar unidade** depois do que o Prompt 1 priorizar: `07` com `INCLUIR_SENSIBILIDADE=True` (W3 → W4 → W5).
5. **Mac:** `08_analise_wavkan`. Depois eu trago do Drive para o repo os notebooks executados (com saídas, que servem de prova) e `resultados/wavkan/`.

## Estimativa de computação (L4), derivada da calibração do Prompt 1
**Fórmula** (aplicada pelo 06, com valores medidos):
- horas(W) = 5 × (15·t_F1(W) + 45·t_F2(W)) / 3600 + sobrecarga de sessão (clone, pip e cache: ~5 min por sessão; checkpoint no Drive pelo `CKPT_MIN_INTERVAL_MIN`).
- UC(W) = horas(W) × taxa_L4. A taxa é lida no painel "Recursos" do Colab e registrada pelo 01: o Google não publica a taxa da L4.

**Estimativa prévia** (substituída pelo 06):
- U_L4 = uma rodada 5-fold × 60 épocas de (d) na L4 com AMP ≈ **1,0 h** (faixa de 0,8 a 1,4 h; a L4 faz ≈ 2× a T4, em que U ≈ 2 h).
- Custo da cabeça, maior na L4 pelo motivo acima:

| Unidade | Custo relativo | Horas L4 | UC (se taxa ≈ 4–5 UC/h, a confirmar) |
|---|---|---|---|
| 06 + smoke do 07 | — | ~0,5 h | ~2–3 |
| **W1** | ~1,03–1,06 U | ~0,8–1,5 h | ~4–7 |
| **W2** | ~1,00–1,02 U | ~0,8–1,4 h | ~4–7 |
| W3, W4 | ~1,03–1,06 U cada | ~1,6–3,0 h | ~8–15 |
| W5 | ~1,10–1,20 U | ~0,9–1,7 h | ~4–8 |
| **Essencial (06 + smoke + W1 + W2)** | | **~2,1–3,4 h** | **~10–17** |
| Tudo (W1–W5) | | ~4,6–8,1 h | ~22–40 |

- O Colab Pro dá 100 UC/mês. O essencial do Prompt 3 consome ~10–17 % disso, e tudo ~22–40 %. Como o Prompt 1 consome a maior parte, a prioridade W1 → W2 garante a resposta e W3–W5 ficam condicionados ao saldo.
- Se a ponte do 01 decidir **fp32**, as horas sobem pelo fator fp32/AMP medido no 01.
- Se uma unidade cair na **T4**: ~2× mais horas (com taxa por hora menor), a GPU fica registrada, e o tempo da tabela continua vindo só do 06 na L4.
- **Sessões:** o Pro não tem execução em segundo plano (mantenha a aba aberta); o Pro+ tem. Nos dois casos, a retomada por época limita a perda.

## Tabela comparativa (mesmo protocolo, semente 42)
- **Linhas:** (a), (b), (c), (d), W1, W2, mais W3, W4 e W5 se tiverem rodado (o que não rodar fica marcado como "não executado").
- **Colunas:**
  - parâmetros da cabeça / treináveis na fase 2;
  - Acc, P macro, R macro, F1 macro e AUC (prob.), em média ± dp das 5 dobras, mais a AUC OOF;
  - tempo por dobra (min, **L4** isolada, 15·t_F1 + 45·t_F2);
  - cabeça fwd+bwd (ms/lote 32, L4), inferência da cabeça (ms/img, L4) e memória de pico da cabeça.
- **Contrastes:** ΔAUC e ΔF1 com IC95% e dobras em que W não perde, para W1, W2 e W5 contra (d) e para W2 contra (b).
- **Rodapé:**
  - protocolo, GPU (com unidades em T4 marcadas, se houver), uma única semente;
  - as colunas micro saem, por serem redundantes, como no Prompt 1;
  - **não se compara com 0,908 (TCC) nem com 0,893 (artigo)**.

## Riscos
- **A Wav-KAN não sair mais rápida.** É o desfecho mais provável, e na L4 a desvantagem relativa da cabeça é maior. A conclusão honesta passa a ser "menos parâmetros na cabeça".
- **NaN.** Pode vir da divisão por `scale` ou do fp16. Mitigação: guarda |s| ≥ 1e-3, cabeça em fp32 (pelo laço do Prompt 1) e checagem de NaN que aborta a unidade com mensagem, registrada no JSON.
- **A L4 indisponível na hora.** Acontece mesmo com assinatura. Primeiro espere ou tente de novo; a T4 só entra com `PERMITIR_T4=True`, registrada.
  - Entre GPUs diferentes, a métrica muda pouco: as duas rodam em AMP fp16, e só o não determinismo do cuDNN varia.
  - Uma dobra retomada em outra GPU fica marcada no `historico.csv`, porque a GPU é gravada por época.
- **Unidades de computação acabando.** O 07 mostra o custo previsto antes de cada unidade, e a fila padrão tem só W1 e W2.
- **O plugin falhar no CLI do Prompt 1.** O `SMOKE` do 07 pega isso antes das rodadas longas.
- **Confusão na interpretação.** A grade da efficient-kan cobre [−1,1] (estendida até ±1,6), e as features são ≥ 0, com várias acima disso. A spline pode estar meio inativa, enquanto a Wav-KAN ajusta `translation`/`scale`. Uma vantagem da Wav-KAN pode vir daí, não "das wavelets"; os diagnósticos dos dois lados permitem discutir isso.
- **Pouco poder estatístico.** São 5 dobras e 1 semente, então só a não inferioridade com IC é afirmável.

## Verificação
- **Mac:** os testes do `cabecas_wavkan.py` passam (forma, contagem, equivalência com NumPy, plugin carregado pelo CLI). Quando o protocolo existir: smoke sintético com retomada idêntica.
- **06 (L4):**
  - asserts de parâmetros e equivalência com o torch-conv-kan;
  - as âncoras (b) e (d) ficam a ≤10 % da calibração L4 do Prompt 1;
  - o JSON registra GPU = L4 e as UC previstas.
- **07 SMOKE:**
  - as cabeças W aparecem no registro e o CLI executa uma unidade W;
  - a retomada volta na época certa;
  - AUC calculada sobre probabilidades contínuas;
  - **o hash de `protocolo_comum.py` e do CSV de dobras é igual ao das execuções do Prompt 1**, e o hash do plugin fica registrado à parte;
  - nada de `.npy` ou `.JPG` no Drive.
- **08:**
  - as métricas recalculadas dos `pred_teste.csv` batem com o JSON;
  - o OOF de W e o de (d) têm as mesmas imagens e casos;
  - AUC ≠ revocação macro em toda linha;
  - todas as unidades estão em L4 (ou marcadas);
  - a tabela e a figura são geradas.
- **Final:** os notebooks executados e `resultados/wavkan/` vão para o repo. Os 3 notebooks e o `.py` ficam em `tcc-final` no Drive.

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

### Ajustes finais específicos do P3
1. Use `PERMITIR_RESERVA`, não `PERMITIR_T4`.
2. O 01 do P1 já mede memória e tempo de W1/W5: no 06 não repita isso; mantenha as âncoras (b)/(d) só se custarem poucos minutos.
3. Execução: W1 → W2 logo depois das cabeças s42 do P1; W3–W5 ficam para o mês seguinte.
