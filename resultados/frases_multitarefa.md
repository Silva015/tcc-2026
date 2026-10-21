# Frases da multitarefa (geradas pelo 10)

## EN

- On entry/exit classification, under an identical protocol, one multitask ResNet-152 with spline-KAN heads (shared backbone, one head per task) underperformed the dedicated single-task network with the same head (AUC 0.894 ± 0.014 vs. 0.905 ± 0.013; ΔAUC = -0.011, 95% CI [-0.015, -0.006]; better in 0/5 folds), although the difference stays within the pre-registered non-inferiority margin of -0.02 (seeds 42, 43, 44).

- On entry/exit classification, under an identical protocol, one multitask ResNet-152 with MLP-128 heads (shared backbone, one head per task) underperformed the dedicated single-task network with the same head (AUC 0.895 ± 0.014 vs. 0.905 ± 0.014; ΔAUC = -0.010, 95% CI [-0.014, -0.005]; better in 0/5 folds), although the difference stays within the pre-registered non-inferiority margin of -0.02 (seeds 42, 43, 44).

- Within the multitask network, on entry/exit classification, spline-KAN heads (7.87 M head parameters) performed on par with MLP-128 heads (0.53 M), within the pre-registered margin (AUC 0.894 ± 0.014 vs. 0.895 ± 0.014; ΔAUC = -0.001, 95% CI [-0.003, +0.0004]; KAN better in 2/5 folds) (seeds 42, 43, 44). This compares head types, not parameter efficiency.

- On shooting-distance classification (entries only), under an identical protocol, one multitask ResNet-152 with spline-KAN heads (shared backbone, one head per task) matched the dedicated single-task network with the same head within the pre-registered non-inferiority margin of -0.02 (AUC 0.805 ± 0.065 vs. 0.798 ± 0.058; ΔAUC = +0.007, 95% CI [-0.003, +0.018]; better in 3/5 folds; ΔF1 = -0.002, 95% CI [-0.029, +0.027]) (seeds 42, 43, 44).

- On shooting-distance classification (entries only), under an identical protocol, one multitask ResNet-152 with MLP-128 heads (shared backbone, one head per task) matched the dedicated single-task network with the same head within the pre-registered non-inferiority margin of -0.02 (AUC 0.806 ± 0.071 vs. 0.802 ± 0.062; ΔAUC = +0.004, 95% CI [-0.006, +0.014]; better in 3/5 folds; ΔF1 = +0.007, 95% CI [-0.021, +0.035]) (seeds 42, 43, 44).

- Within the multitask network, on shooting-distance classification (entries only), spline-KAN heads (7.87 M head parameters) performed on par with MLP-128 heads (0.53 M), within the pre-registered margin (AUC 0.805 ± 0.065 vs. 0.806 ± 0.071; ΔAUC = -0.002, 95% CI [-0.009, +0.003]; KAN better in 2/5 folds) (seeds 42, 43, 44). This compares head types, not parameter efficiency.

- One multitask network with spline-KAN heads replaces two single-task networks: 66.02 M vs. 124.16 M parameters (-46.8%). Both outputs need one forward pass instead of two: 1.12 vs. 2.22 ms per image at batch 32 and 27.31 vs. 51.03 ms at batch 1 on an NVIDIA L4 (2.0× faster at batch 32). Training takes 8.6 min per fold for the multitask network vs. 14.9 min for the two single-task networks together (NVIDIA L4; faster beyond run-to-run noise).

- One multitask network with MLP-128 heads replaces two single-task networks: 58.67 M vs. 116.81 M parameters (-49.8%). Both outputs need one forward pass instead of two: 1.11 vs. 2.22 ms per image at batch 32 and 24.31 vs. 48.48 ms at batch 1 on an NVIDIA L4 (2.0× faster at batch 32). Training takes 8.1 min per fold for the multitask network vs. 14.6 min for the two single-task networks together (NVIDIA L4; faster beyond run-to-run noise).

- Secondary, descriptive: in the 4-class pipeline (exit, contact, close, distant) with spline-KAN heads, macro-F1 on the pooled out-of-fold predictions is 0.485 for the multitask network vs. 0.515 for the single-task pair (ΔF1 = -0.030, 95% CI [-0.061, -0.004]).

- Secondary, descriptive: in the 4-class pipeline (exit, contact, close, distant) with MLP-128 heads, macro-F1 on the pooled out-of-fold predictions is 0.492 for the multitask network vs. 0.514 for the single-task pair (ΔF1 = -0.022, 95% CI [-0.052, +0.004]).

## PT (dissertação)

- Em entrada/saída, no mesmo protocolo, uma única ResNet-152 multitarefa com cabeças KAN spline (backbone compartilhado, uma cabeça por tarefa) ficou abaixo da rede de tarefa única dedicada, com a mesma cabeça (AUC 0.894 ± 0.014 contra 0.905 ± 0.013; ΔAUC = -0.011, IC 95 % [-0.015, -0.006]; melhor em 0/5 dobras), embora a diferença fique dentro da margem de não inferioridade pré-registrada de -0.02 (sementes 42, 43, 44).

- Em entrada/saída, no mesmo protocolo, uma única ResNet-152 multitarefa com cabeças MLP-128 (backbone compartilhado, uma cabeça por tarefa) ficou abaixo da rede de tarefa única dedicada, com a mesma cabeça (AUC 0.895 ± 0.014 contra 0.905 ± 0.014; ΔAUC = -0.010, IC 95 % [-0.014, -0.005]; melhor em 0/5 dobras), embora a diferença fique dentro da margem de não inferioridade pré-registrada de -0.02 (sementes 42, 43, 44).

- Dentro da rede multitarefa, em entrada/saída, as cabeças KAN spline (7.87 M parâmetros nas cabeças) empataram com as cabeças MLP-128 (0.53 M), dentro da margem pré-registrada (AUC 0.894 ± 0.014 contra 0.895 ± 0.014; ΔAUC = -0.001, IC 95 % [-0.003, +0.0004]; KAN melhor em 2/5 dobras) (sementes 42, 43, 44). Isso compara tipos de cabeça, não eficiência em parâmetros.

- Em distância do disparo (só entradas), no mesmo protocolo, uma única ResNet-152 multitarefa com cabeças KAN spline (backbone compartilhado, uma cabeça por tarefa) foi não inferior à rede de tarefa única dedicada, com a mesma cabeça, dentro da margem pré-registrada de -0.02 (AUC 0.805 ± 0.065 contra 0.798 ± 0.058; ΔAUC = +0.007, IC 95 % [-0.003, +0.018]; melhor em 3/5 dobras; ΔF1 = -0.002, IC 95 % [-0.029, +0.027]) (sementes 42, 43, 44).

- Em distância do disparo (só entradas), no mesmo protocolo, uma única ResNet-152 multitarefa com cabeças MLP-128 (backbone compartilhado, uma cabeça por tarefa) foi não inferior à rede de tarefa única dedicada, com a mesma cabeça, dentro da margem pré-registrada de -0.02 (AUC 0.806 ± 0.071 contra 0.802 ± 0.062; ΔAUC = +0.004, IC 95 % [-0.006, +0.014]; melhor em 3/5 dobras; ΔF1 = +0.007, IC 95 % [-0.021, +0.035]) (sementes 42, 43, 44).

- Dentro da rede multitarefa, em distância do disparo (só entradas), as cabeças KAN spline (7.87 M parâmetros nas cabeças) empataram com as cabeças MLP-128 (0.53 M), dentro da margem pré-registrada (AUC 0.805 ± 0.065 contra 0.806 ± 0.071; ΔAUC = -0.002, IC 95 % [-0.009, +0.003]; KAN melhor em 2/5 dobras) (sementes 42, 43, 44). Isso compara tipos de cabeça, não eficiência em parâmetros.

- Uma rede multitarefa com cabeças KAN spline substitui duas redes de tarefa única: 66.02 M contra 124.16 M parâmetros (-46.8 %). As duas saídas saem de um forward em vez de dois: 1.12 contra 2.22 ms por imagem com lote 32 e 27.31 contra 51.03 ms com lote 1 na NVIDIA L4 (2.0× mais rápido com lote 32). O treino leva 8.6 min por dobra para a rede multitarefa contra 14.9 min para as duas redes de tarefa única juntas (L4; mais rápido além da variação entre rodadas).

- Uma rede multitarefa com cabeças MLP-128 substitui duas redes de tarefa única: 58.67 M contra 116.81 M parâmetros (-49.8 %). As duas saídas saem de um forward em vez de dois: 1.11 contra 2.22 ms por imagem com lote 32 e 24.31 contra 48.48 ms com lote 1 na NVIDIA L4 (2.0× mais rápido com lote 32). O treino leva 8.1 min por dobra para a rede multitarefa contra 14.6 min para as duas redes de tarefa única juntas (L4; mais rápido além da variação entre rodadas).

- Secundário, descritivo: no pipeline de 4 classes (saída, encostado, curta, distância) com cabeças KAN spline, o F1 macro no OOF é 0.485 para a rede multitarefa contra 0.515 para o par de tarefa única (ΔF1 = -0.030, IC 95 % [-0.061, -0.004]).

- Secundário, descritivo: no pipeline de 4 classes (saída, encostado, curta, distância) com cabeças MLP-128, o F1 macro no OOF é 0.492 para a rede multitarefa contra 0.514 para o par de tarefa única (ΔF1 = -0.022, IC 95 % [-0.052, +0.004]).

## Pode afirmar

```json
{
 "mt_non_inferior_to_st": {
  "es__kan": false,
  "es__mlp": false,
  "dist__kan": true,
  "dist__mlp": true
 },
 "mt_better_than_st": {
  "es__kan": false,
  "es__mlp": false,
  "dist__kan": false,
  "dist__mlp": false
 },
 "mt_worse_than_st": {
  "es__kan": true,
  "es__mlp": true,
  "dist__kan": false,
  "dist__mlp": false
 },
 "kan_better_than_mlp_within_mt": {
  "es": false,
  "dist": false
 },
 "one_network_instead_of_two": true,
 "mt_faster_inference": {
  "kan_lote1": true,
  "kan_lote32": true,
  "mlp_lote1": true,
  "mlp_lote32": true
 },
 "mt_faster_training": {
  "kan": true,
  "mlp": true
 },
 "kan_head_parameter_efficiency": false,
 "claims_about_contact_shots": false,
 "compare_with_0.908_or_0.893": false,
 "spline_KAN_is_the_reference_that_works": false
}
```

## Lembretes

- O resultado entra só no TCC; o artigo e o pôster do SIBGRAPI mantêm a multitarefa como trabalho futuro.
- O argumento é uma rede no lugar de duas, com os parâmetros contados; nada de "parameter efficiency" da cabeça KAN.
- Distância: prever sempre "distância" já dá acurácia 0,854 (F1 macro 0,307); reporte AUC e F1 macro, não só a acurácia.
- Nada de afirmação específica sobre encostado: são 16 casos (42 imagens), ~8 imagens por dobra.
- Não comparar com 0,908 (TCC antigo) nem com 0,893 (artigo do SIBGRAPI): protocolos diferentes.
- Na MT, a cabeça E/S vê 2.554 imagens e a de distância 1.883; a ST de distância do P1 treinou só nas entradas. A avaliação é pareada nas mesmas imagens.
- GaussNoise(std_range=(0,2; 0,44)) "como rodou" pode apagar a tatuagem e prejudicar a classe curta; vale igualmente para ST e MT (limitação, não viés da comparação).
- Citar Lira et al. (2025, IJLM) como origem da base; a base não tem licença de redistribuição e nenhuma imagem sai da sessão.
