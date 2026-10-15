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

## 6. Drive e git

- **Drive:**
  - cada sessão sobe só os próprios arquivos;
  - o P1 é dono de `protocolo_comum.py`, `configs_mlsd.py`, dos notebooks 00–05b e de `dados/lira2024_tabelas.csv`;
  - nunca suba imagens, cache nem checkpoints.
- **Git:**
  - `git add` por caminho;
  - confira `.git/index.lock` antes de commitar;
  - não mexa em `references/leguwoi/` nem nos 8 notebooks antigos.
