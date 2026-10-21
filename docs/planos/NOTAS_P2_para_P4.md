# Notas do P2 (multitarefa) para o P4

Resultados das rodadas de 21/10/2026 na L4, trazidos do Drive (`tcc-final`) e conferidos. Só as redes multitarefa (MT) foram treinadas: `mt_kan` e `mt_mlp`, sementes 42, 43 e 44, 5 dobras cada. As redes de tarefa única (ST) são as do P1: `resnet152__kan` e `resnet152__mlp128` (entrada/saída) e `mlsd__resnet152__kan` e `mlsd__resnet152__mlp128` (distância).

## Onde está cada coisa

| Arquivo | O que é |
|---|---|
| `resultados/decisao_multitarefa.json` | Contrastes, IC, desfechos da regra pré-registrada, pipeline, custo, tempo de inferência, `pode_afirmar`, frases EN/PT |
| `resultados/frases_multitarefa.md` | As mesmas frases, prontas para copiar, e os lembretes |
| `resultados/tabela_multitarefa.{csv,tex}` | ST × MT, com as duas cabeças: parâmetros, AUC e F1 das duas tarefas, F1 do pipeline de 4 classes, min/dobra, ms/img |
| `resultados/contrastes_multitarefa.{csv,tex}` | Contrastes (ΔAUC e ΔF1 com IC 95 %) e desfechos |
| `resultados/pipeline_multitarefa.csv` | Pipeline de 4 classes (saída, encostado, curta, distância): F1 macro e revocação por classe |
| `resultados/confusao_multitarefa.csv` | Matrizes de confusão da distância e do pipeline (OOF, média das sementes) |
| `resultados/multitarefa/{mt_kan,mt_mlp}/{runs.csv,resumo.json,oof.csv}` | 15 unidades por rede (semente × dobra) e as predições fora da dobra, as duas cabeças |
| `resultados/multitarefa/calibracao_mt.json` | Tempo de inferência: 1 forward da MT × 2 do par ST, lotes 1 e 32 |
| `resultados/multitarefa/teste_rapido_ok.json` | Teste rápido do `09`, com a retomada comparada ao ruído da GPU |
| `resultados/multitarefa/fila_aba{A,B}.json` | Fila de cada aba (A: `mt_kan`, B: `mt_mlp`) |
| `resultados/multitarefa/notebooks_executados/` | Os dois `09` (abas A e B) e o `10`, como rodaram no Colab, sem o nome e o ID do Google nas células |

Ficaram só no Drive: as pastas por unidade (`metricas.json`, `pred_teste.csv`, `historico.csv`) e os `oof.csv` agregados do P1 usados na conferência. Os checkpoints foram apagados ao fim de cada dobra.

## Conferências feitas

- Os tamanhos de todos os arquivos batem com os do Drive. Os arquivos foram gravados a partir do download do conector, sem cópia à mão.
- O código que rodou é o do repositório. A única diferença é `TESTE_RAPIDO = False` na aba A, que é o esperado.
- As 30 unidades usaram a L4 em AMP, com protocolo SHA `b7cd9bac…`, plugin SHA `b9841e0f…` e as dobras SHA `ccef0fdc…`. Nenhuma perda deu NaN e não sobrou checkpoint.
- `runs.csv` bate com o `oof.csv` unidade a unidade: AUC, F1, acurácia e acurácia balanceada das duas tarefas, com diferença máxima de 1e-16. `resumo.json` bate com `runs.csv`, com diferença máxima de 8e-17.
- **Recálculo independente**, num script próprio e não com o código do `10`, a partir dos OOF da MT e dos OOF agregados do P1 (`ablacao_cabeca/oof.csv` e `mlsd_cabeca/oof.csv` do Drive):
  - a `tabela_multitarefa.csv` (ST e MT, as duas tarefas, todas as métricas) é idêntica;
  - os 8 contrastes e os 3 do pipeline batem com o `decisao_multitarefa.json` sem diferença nenhuma, inclusive o IC (mesmo bootstrap pareado de casos, 2000 reamostragens, semente 0), as dobras vencidas e o desfecho;
  - as linhas das ST saem iguais às do `10`, então o `10` usou exatamente as predições do P1.
- As frases do `frases_multitarefa.md` são as do `decisao_multitarefa.json`.
- As duas tabelas `.tex` compilam no pdfLaTeX.
- **Teste rápido.** A fase só das cabeças saiu idêntica (diferença 0). Com duas interrupções, a retomada diferiu 3,0e-4 nas perdas, contra um ruído de 5,2e-4 entre duas rodadas sem interrupção, e 4,6e-3 nas probabilidades, contra 3,0e-3 de ruído (limite 9,0e-3).

## Resultados

Média ± dp entre dobras (cada dobra é a média das 3 sementes). Distância só nas entradas. Custo medido na L4.

| Sistema | Redes | Params (M) | E/S AUC | E/S F1 | Dist. AUC | Dist. F1 | F1 4 classes | min/dobra | ms/img (lote 32) |
|---|---|---|---|---|---|---|---|---|---|
| ST, cabeças KAN (d) | 2 | 124,16 | 0,905 ± 0,013 | 0,806 ± 0,013 | 0,798 ± 0,058 | 0,477 ± 0,045 | 0,500 ± 0,035 | 14,9 | 2,22 |
| **MT, cabeças KAN** | 1 | 66,02 | 0,894 ± 0,014 | 0,784 ± 0,022 | 0,805 ± 0,065 | 0,482 ± 0,058 | 0,491 ± 0,045 | 8,6 | 1,12 |
| ST, cabeças MLP-128 (b) | 2 | 116,81 | 0,905 ± 0,014 | 0,806 ± 0,015 | 0,802 ± 0,062 | 0,482 ± 0,050 | 0,506 ± 0,037 | 14,6 | 2,22 |
| **MT, cabeças MLP-128** | 1 | 58,67 | 0,895 ± 0,014 | 0,787 ± 0,017 | 0,806 ± 0,071 | 0,492 ± 0,063 | 0,499 ± 0,045 | 8,1 | 1,11 |

Contrastes (Δ = a − b no OOF médio das sementes; IC 95 % por bootstrap de casos; margem de não inferioridade −0,02 no ΔAUC):

| Tarefa | Contraste | Papel | ΔAUC [IC 95 %] | ΔF1 [IC 95 %] | a vence | Desfecho |
|---|---|---|---|---|---|---|
| E/S | MT-KAN − ST-KAN | primário | −0,011 [−0,015; −0,006] | −0,023 [−0,034; −0,011] | 0/5 | pior (dentro da margem) |
| E/S | MT-MLP − ST-MLP | primário | −0,010 [−0,014; −0,005] | −0,022 [−0,034; −0,010] | 0/5 | pior (dentro da margem) |
| E/S | MT-KAN − MT-MLP | secundário | −0,001 [−0,003; +0,0004] | −0,002 [−0,008; +0,003] | 2/5 | não inferior |
| Dist. | MT-KAN − ST-KAN | primário | +0,007 [−0,003; +0,018] | −0,002 [−0,029; +0,027] | 3/5 | não inferior |
| Dist. | MT-MLP − ST-MLP | primário | +0,004 [−0,006; +0,014] | +0,007 [−0,021; +0,035] | 3/5 | não inferior |
| Dist. | MT-KAN − MT-MLP | secundário | −0,002 [−0,009; +0,003] | −0,008 [−0,014; −0,001] | 2/5 | não inferior |
| 4 classes | MT-KAN − ST-KAN | descritivo | — | −0,030 [−0,061; −0,004] | 1/5 | — |
| 4 classes | MT-MLP − ST-MLP | descritivo | — | −0,022 [−0,052; +0,004] | 2/5 | — |

ST-KAN × ST-MLP (descritivo) não difere em nenhuma das tarefas: ΔAUC −0,0001 em E/S e −0,005 na distância, os dois IC cruzando o zero.

**Custo** (uma rede no lugar de duas):
- **Parâmetros:** −46,8 % (KAN) e −49,8 % (MLP).
- **Inferência das duas saídas:** 2,0× mais rápida. Com lote 32, 1,12 contra 2,22 ms/img; com lote 1, 27,3 contra 51,0 ms/img. O pico de memória da inferência com lote 32 é de 163 contra 282 MB.
- **Treino:** 8,6 contra 14,9 min por dobra (KAN) e 8,1 contra 14,6 (MLP). O pico de memória do treino é quase igual: 3,7 contra 3,6 GB.

**Em uma frase:** uma ResNet-152 com duas cabeças custa cerca de metade do par de redes e empata na distância. Na entrada/saída perde ~0,01 de AUC em todas as dobras: é pouco, menor que a margem de 0,02, mas consistente.

## Cuidados para o texto

1. **"Pior" e "dentro da margem" ao mesmo tempo.**
   - Em E/S, a regra deu *pior*: o IC do ΔAUC fica todo abaixo de zero e a MT não vence em nenhuma dobra. Ao mesmo tempo, o limite inferior (−0,015) fica acima da margem de −0,02. As frases do `10` já dizem as duas coisas; o texto também deve dizer.
   - No `decisao_multitarefa.json`, `desfecho = "pior"` com `dentro_da_margem = true`. Como os desfechos são exclusivos, `pode_afirmar.mt_non_inferior_to_st.es__* = false`. Não ler esse `false` como "inferior além da margem".
   - No F1 a queda é maior: ΔF1 −0,022 com limite inferior −0,034, além da margem, e a acurácia cai ~2,5 pontos (0,846 → 0,820/0,823). Se o texto citar F1 ou acurácia, dizer isso.
2. **Avisos de sanidade.** Em 5 das 30 unidades a acurácia de E/S ficou abaixo de 0,80 (`mt_kan`: 4; `mt_mlp`: 1; entre 0,792 e 0,797; dobras 2 e 3). O limite de 0,80 veio das ST do P1. Os avisos refletem a queda do item 1, não uma falha do treino: as perdas são finitas e a AUC dessas dobras é normal (0,87–0,88).
3. **Distância: nunca só a acurácia.** Prever sempre "distância" já dá 0,854 de acurácia (F1 macro 0,307). A acurácia da MT é maior que a da ST (0,824 contra 0,813), mas a acurácia balanceada é igual ou menor (0,511 contra 0,515 na KAN; 0,520 nas duas na MLP). Reportar AUC e F1 macro.
4. **Encostado.** No pipeline, a revocação de encostado cai de 0,119 para 0,048 (5 → 2 das 42 imagens). São 16 casos, ~8 imagens por dobra: nada de afirmação específica sobre encostado.
5. **Pipeline de 4 classes é secundário e descritivo.** A queda da MT-KAN (IC exclui o zero) vem sobretudo da cabeça E/S: a MT manda mais entradas para "saída" (no OOF, 283 imagens de "distância" previstas como saída contra 204 na ST), e por isso a revocação de saída sobe (0,788 contra 0,756).
6. **KAN × MLP dentro da MT:** empatam. A comparação é entre tipos de cabeça. Nada de "eficiência em parâmetros" da KAN, que tem 7,87 M parâmetros nas cabeças contra 0,53 M da MLP.
7. **"Mais rápida" vale aqui, ao contrário do P3.**
   - **Treino:** 8,6 contra 14,9 min por dobra, com dp de 0,10–0,15 min.
   - **Inferência:** 2,0× mais rápida, com dp de 1–4 % das medidas (o maior com lote 1).
   - Isso é consequência direta de um forward no lugar de dois. Apresente como custo, não como mérito do método.
   - A inferência foi medida em modelos recém-criados (o tempo não depende dos pesos), em AMP fp16 na L4, com 20 aquecimentos e 200 medidas por eventos CUDA.
8. **O que cada cabeça vê no treino.** A cabeça E/S da MT treina com as 2.554 imagens e a de distância com as 1.883 entradas (as saídas ficam fora da perda de distância). A ST de distância do P1 treinou só nas entradas. Dobras, pesos de classe e imagens de teste são os mesmos, e a avaliação é pareada nas mesmas imagens.
9. **Só no TCC.** O artigo e o pôster do SIBGRAPI mantêm a multitarefa como trabalho futuro.
10. **Não comparar com 0,908 nem com 0,893** (TCC antigo e artigo do SIBGRAPI): os protocolos são outros.
11. **Largura da tabela.** `tabela_multitarefa.tex` tem 12 colunas e passa 104 pt da largura útil de uma A4 com 1,5 cm de margem. Na dissertação, use `\resizebox{\textwidth}{!}{…}` ou divida a tabela, por exemplo desempenho em uma e custo em outra.
12. **Pesos finais não guardados.** O plano P2 previa guardar os pesos da `mt_kan` em fp16 no Drive, mas o protocolo congelado não tem essa função. Nenhum peso foi guardado, e nenhum resultado depende disso.
13. **Mesmo protocolo do P1:** última época, sem validação; AMP; `GaussNoise(std_range=(0,2; 0,44))` como rodou, que vale igualmente para ST e MT e é limitação, não viés da comparação.
14. **Histórico do teste rápido.** O primeiro teste rápido na L4 falhou só na tolerância da retomada. Na Fase 2, duas rodadas iguais já diferem na GPU, porque o backward do `max_pool2d` na CUDA não é determinístico. O commit cc01303 passou a comparar a retomada com esse ruído. Só o teste mudou; o treino e o plugin são os mesmos.
