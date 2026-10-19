# Notas do P1 para as sessões P2 (multitarefa) e P3 (Wav-KAN)

Correções e avisos sobre o `protocolo_comum.py` v1.0.0 (SHA-256 `b7cd9bac…5ef9e13`, congelado) e o que o P1 já entrega pronto. Valem mais que os trechos correspondentes de `P2_multitarefa.md` e `P3_wavkan.md`.

## 1. Como rodar (não existe CLI)

Os planos citam `python protocolo_comum.py executar …`: esse comando **não existe**. Cada notebook roda pela API:

```python
import protocolo_comum as pc
pc.carregar_plugin(f"{DIR_DRIVE}/multitarefa.py")      # registra cabeças/perdas/configs; o SHA vai para o metricas.json
sess = pc.preparar_sessao(RAIZ_TRABALHO, caminho_dobras=..., caminho_meta=...)
fila = pc.montar_fila("multitarefa", ["mt_kan"], sementes=[42])
pc.executar_fila(sess, fila, R, C, precisao=PREC, ckpt_intervalo_min=CKPT)
runs = pc.ler_resultados(R, "multitarefa")
```

Siga o esqueleto dos notebooks 02/03/04: sentinela, SHA do protocolo, `exigir_gpu`, `definir_id_sessao(ABA)`, `TESTE_RAPIDO`, precisão e intervalo de checkpoint lidos de `resultados/calibracao_l4.json`.

## 2. Dobras e a coluna de dobra

- A unidade aceita `"coluna_dobra"` (padrão `"fold"`). Treino = linhas com `coluna != dobra` **e** `coluna >= 0`; teste = `coluna == dobra`. Um valor **−1 exclui a linha** do treino e do teste.
- Para acrescentar colunas, mude `sess.df` depois do `preparar_sessao` (ou use `df_extra=`). O `ident` da unidade grava a coluna usada.
- **MLSD do P1** (`notebooks/configs_mlsd.py`): na unidade da dobra *k*, a coluna é `fold_dist_dk`. O treino usa só as entradas das outras dobras, e o teste é a dobra *k* inteira. As saídas recebem `p_dist_*` no `pred_teste.csv`, mas não entram nas métricas (`y_dist = −1`).
- As métricas por tarefa já ignoram `y = −1`. Na distância, a AUC é OvR macro, calculada das probabilidades.

## 3. O P2 reaproveita o P1 (decisão do usuário)

O P2 **não** treina `st_es_*` nem `st_dist_*`. As redes de tarefa única são estas:

| P2 | rodada do P1 | caminho |
|---|---|---|
| `st_es_kan` | `resnet152__kan` (cabeça d), s42/43/44 | `resultados/ablacao_cabeca/` |
| `st_es_mlp` | `resnet152__mlp128` (cabeça b), s42/43/44 | `resultados/ablacao_cabeca/` |
| `st_dist_kan` | `mlsd__resnet152__kan` (cabeça d), s42/43/44 | `resultados/mlsd_cabeca/` |
| `st_dist_mlp` | `mlsd__resnet152__mlp128` (cabeça b), s42/43/44 | `resultados/mlsd_cabeca/` |

- As contagens de parâmetros conferem com a tabela do P2: 62.081.728 e 58.406.467.
- **Pipeline de 4 classes:** junte por `rel_path` o `pred_teste.csv` da ST-ES e o da ST-dist da mesma dobra. Como o teste da MLSD do P1 é a dobra inteira, as saídas também têm `p_dist_*`.
- **O P2 só treina `mt_kan` e `mt_mlp`**, com `coluna_dobra = "fold"` (todas as imagens), as mesmas dobras e a semente 42 (mais 43/44 se houver saldo). A comparação MT × ST deve usar as mesmas sementes dos dois lados.
- **Prefixo de configs:** o P1 usa `mlsd__`, então não use esse prefixo no `multitarefa.py`.

## 4. O que não existe no protocolo

- **`salvar_pesos_finais`:** o checkpoint é apagado quando a unidade termina. Se precisar dos pesos (benchmark de inferência da MT), meça a inferência dentro do próprio notebook, logo depois de treinar uma unidade com `limite_epocas`, ou peça ao P1 antes de rodar. O protocolo não muda depois de congelado.
- **Validação interna:** não há; a regra é a última época.
- **Receita configurável:** `validar_config` recusa a chave `"receita"`.

## 5. Custos medidos na L4 (notebook 01 e rodadas do P1)

| Item | Valor |
|---|---|
| Taxa | **1,54 UC/h** (o painel chegou a mostrar 1,62) |
| ResNet-152, AMP, entrada/saída | ~8,4 min por dobra (~42 min por config × semente) |
| ResNet-152, MLSD (só entradas) | ~0,74× disso |
| Época da F2 (entrada/saída) | 8,8 s |
| Checkpoint completo | 3,8 s → `CKPT_INTERVALO_MIN = 5` |
| Precisão | **AMP fp16** (decidida pela ponte); as cabeças sempre em fp32 |

## 6. Situação do P1 e lições do Colab (19/10)

- **O P1 terminou.** As análises do 05 e do 05b estão em `resultados/` (commit `0054dcb`). Os resultados por unidade continuam no Drive, em `resultados/<experimento>/`. Só falta a semente 44 da Swin-B e da MaxViT-T no MLSD, que é opcional.
- **Uma aba, um arquivo de notebook.** Duas abas com o mesmo notebook dividem o kernel e salvam uma por cima da outra. Para rodar em paralelo, use uma cópia com outro nome (ex.: `04_mlsd_mesmo_protocolo_abaB.ipynb`) e outro `ABA`.
- **Não comece duas abas no mesmo experimento no mesmo segundo.** O Drive criou duas pastas `mlsd_baselines` com o mesmo nome. A sessão reaberta montou a outra pasta e refez cerca de 3 h de unidades. Comece as abas com um minuto de diferença e, antes de analisar, procure pastas de mesmo nome em `resultados/`. A cópia que sobrou se chama `mlsd_baselines_duplicado_abaA` e fica fora da análise.
- **Unidades:** quando o saldo acaba, a L4 e a A100 ficam cinza e o Colab conecta na T4. O `exigir_gpu` para nesse caso; não ligue `PERMITIR_RESERVA` para rodadas comparáveis.

## 7. Ajustes válidos para os dois planos (P2 e P3), depois do P1

1. **Custo real.** A taxa da L4 é **1,54 UC/h**, e não 4–5 UC/h como os planos estimam. Uma config × semente de 5 dobras na ResNet-152 custa ~0,7 h ≈ **1,1 UC** (E/S ou multitarefa) e ~0,6 h ≈ 0,95 UC no MLSD (pelas rodadas do P1: ~8,4 min por dobra em E/S e ~7,4 min no MLSD, já com avaliação e gravação). Todas as tabelas de UC dos planos ficam ~3–4× menores.
2. **Sementes.** As referências do P1 (cabeças a–d, nas duas tarefas) têm as sementes **42, 43 e 44**. Recomendação: rodar as configs principais também com as 3 sementes. A comparação fica pareada por semente, o bootstrap usa por dobra a média das sementes (como no 05b) e a ressalva "single training seed" deixa de valer. As configs de sensibilidade ficam só com s42. Confirme com o usuário antes de rodar.
3. **O que os resultados do P1 mudam na leitura:**
   - a KAN spline (d) **empata** com as MLPs em E/S e fica **abaixo** da MLP isoparamétrica no MLSD;
   - logo, o contraste que mais informa é contra a **melhor MLP do mesmo orçamento**, e não só contra (d);
   - nenhum texto pode dizer que a KAN spline é "a referência que funciona".
4. **Execução:**
   - não existe CLI (veja a seção 1);
   - **não existe** `salvar_pesos_finais` (seção 4);
   - **caminho das dobras:** no Drive é `dados/folds_entrada_saida.csv`; no repositório é `resultados/folds_entrada_saida.csv`;
   - **antes das rodadas longas**, o usuário roda a parte 7 do `01_calibracao_L4.ipynb` (alguns minutos de L4), quando o plugin da sessão já estiver no Drive. A célula do `cabecas_wavkan.py` mede W1/W5; a do `multitarefa.py` faz o smoke da multitarefa.
5. **Git e branches:**
   - o P1 já está em `main` (enviado em 19/10);
   - se a sessão rodar **no mesmo diretório local** do usuário, não crie branch: commite em `main` por caminho e só envie com o OK dele;
   - se rodar na nuvem, use branch + PR como os planos dizem;
   - o repositório é **público**: nunca commite imagens da base, nem recortes em figuras ou pôsteres.

## 8. Ajustes específicos

**P2 (multitarefa):**
- **Só treina `mt_kan` e `mt_mlp`** (seção 3). Os passos "sessão A com `st_dist_kan`" e "`st_dist_mlp`" do plano saem: a ST de distância é a `mlsd_cabeca` do P1, retreinada no mesmo protocolo (AUC 0,798 para a KAN; 0,802 para a MLP-128).
- **Referências ST com 3 sementes:** `st_es_kan` = `resnet152__kan`, `st_es_mlp` = `resnet152__mlp128`, `st_dist_kan` = `mlsd__resnet152__kan` e `st_dist_mlp` = `mlsd__resnet152__mlp128`.
- **Pesos finais no Drive:** não dá. O benchmark de inferência (1 forward da MT contra 2 do par ST) é medido dentro do 09, sobre um modelo recém-treinado com `limite_epocas` curto. O tempo de inferência não depende dos pesos.
- **Checagens de sanidade, conferidas com o P1:**
  - prever sempre a majoritária dá acurácia 0,854 no MLSD;
  - F1 macro da distância > 0,31 (o P1 dá ~0,48);
  - acc E/S por dobra entre 80% e 89% (o P1 dá ~0,83–0,86).
- **Custo do núcleo com 3 sementes:** 2 configs × 3 sementes ≈ 4,5 h ≈ **7 UC**, mais ~0,5 UC de smoke e calibração.

**P3 (Wav-KAN):**
- As referências (a)–(d) têm 3 sementes. Rode **W1 e W2 com 42/43/44** (≈ 4,5 h ≈ **7 UC**). W3–W5, de sensibilidade, ficam só com s42 (≈ 2,3 h ≈ **3,6 UC**).
- **Contraste que mais informa:** W2 × (b) (mesmo orçamento; a única porta para dizer "mais eficiente que um MLP") e W1/W5 × (c), a MLP isoparamétrica. O contraste W × (d) continua, mas como secundário.
- **Margem de não inferioridade.** O plano usa −0,02. No P1, o IC pareado entre cabeças em E/S teve largura de ~0,003, então −0,02 é folgado. A sessão pode manter a margem (pré-registrada em 14/10) ou propor uma mais estreita, **antes** de rodar e com o OK do usuário.
- **MLSD (opcional, decisão do usuário):** repetir W1/W2 no MLSD com `coluna_dobra = fold_dist_dk`, reaproveitando as funções de `configs_mlsd.py` sem editá-lo. Custa ≈ 3 sementes × 2 configs × 0,62 h ≈ **5,7 UC**. Responde se uma KAN com wavelets corrige o déficit da spline no MLSD.

## 9. Ordem de execução recomendada (antes do P4)

| Passo | Quem | O quê | UC |
|---|---|---|---|
| 0 | usuário | Confirmar o saldo de unidades, que foi a zero em 17/10 | — |
| 1 | sessão P3 (local, CPU) | `cabecas_wavkan.py`, 06, 07 e 08; testes; enviar ao Drive | 0 |
| 2 | sessão P2 (local, CPU) | `multitarefa.py`, 09 e 10; testes; enviar ao Drive. Só depois do commit do P3, para não dividir o diretório de trabalho | 0 |
| 3 | usuário, L4 | `01` parte 7 (W1/W5 e smoke da MT); `06` do P3 e `07` com `SMOKE`; `09` em modo smoke | ~1 |
| 4 | usuário, 2 abas L4 | Aba A: `07` com W1 → W2 (3 sementes). Aba B: `09` com `mt_kan` → `mt_mlp` (3 sementes). Experimentos diferentes, cada aba com o próprio notebook, começando com 1 min de diferença | ~14 |
| 5 | usuário, L4, se houver saldo | W3–W5 (s42); P3 no MLSD; semente 44 da Swin-B e da MaxViT-T no MLSD (P1) | ~3,6 + ~5,7 + ~2,6 |
| 6 | sessões P3 e P2 (CPU) | `08` e `10`; trazer os resultados para o repositório | 0 |
| 7 | sessão P4 | Reescrever a dissertação incluindo P2 e P3 (decisão 3 do plano P4) | 0 |

- **Núcleo (passos 3 e 4):** ~15 UC (~9–10 h de L4, ~5 h de parede com as 2 abas).
- **Tudo:** ~27 UC.
- Some ~20% de folga para quedas e sessões ociosas.

## 10. Drive e git

- **Drive:**
  - cada sessão sobe só os próprios arquivos;
  - o P1 é dono de `protocolo_comum.py`, `configs_mlsd.py`, dos notebooks 00–05b e de `dados/lira2024_tabelas.csv`;
  - nunca suba imagens, cache nem checkpoints.
- **Git:**
  - `git add` por caminho;
  - confira `.git/index.lock` antes de commitar;
  - não mexa em `references/leguwoi/` nem nos 8 notebooks antigos.
