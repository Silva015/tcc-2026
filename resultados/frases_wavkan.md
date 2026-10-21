# Frases da Wav-KAN (geradas pelo 08)

## EN (manuscrito)

- On entry/exit classification, under an identical protocol, a wavelet-KAN head (Mexican hat, 0.26 M parameters) was not shown non-inferior to an MLP head of the same size (0.26 M; AUC 0.902 ± 0.013 vs. 0.905 ± 0.014; ΔAUC = -0.001, 95% CI [-0.004, +0.002]; better in 0/5 folds; ΔF1 = -0.012, 95% CI [-0.023, -0.003]; the lower bound of the ΔF1 CI is below the pre-registered -0.02 margin), although the pre-registered rule for a difference (≥ 4/5 folds and CI excluding zero) was not met either (seeds 42, 43, 44).

- On entry/exit classification, a wavelet-KAN head (W1: Mexican hat, 1.05 M parameters) matched a parameter-matched MLP head (3.94 M) within the pre-registered non-inferiority margin (ΔAUC = -0.002, 95% CI [-0.005, +0.002]; ΔF1 = -0.003, 95% CI [-0.014, +0.008]) (seeds 42, 43, 44).

- On entry/exit classification, a wavelet-KAN head (W5: Mexican hat, 3.94 M parameters) matched a parameter-matched MLP head (3.94 M) within the pre-registered non-inferiority margin (ΔAUC = -0.003, 95% CI [-0.008, +0.002]; ΔF1 = -0.009, 95% CI [-0.019, +0.003]) (single training seed).

- On entry/exit classification, a wavelet-KAN head (W1: Mexican hat, 1.05 M parameters) matched the spline-KAN head (3.94 M) within the pre-registered non-inferiority margin (ΔAUC = -0.001, 95% CI [-0.004, +0.003]; ΔF1 = -0.005, 95% CI [-0.016, +0.005]) (seeds 42, 43, 44).

- On entry/exit classification, a wavelet-KAN head (W2: Mexican hat, 0.26 M parameters) did not reach non-inferiority against the spline-KAN head (3.94 M; ΔAUC = -0.001, 95% CI [-0.003, +0.002]; ΔF1 = -0.011, 95% CI [-0.021, -0.002]) (seeds 42, 43, 44).

- On entry/exit classification, a wavelet-KAN head (W5: Mexican hat, 3.94 M parameters) matched the spline-KAN head (3.94 M) within the pre-registered non-inferiority margin (ΔAUC = -0.002, 95% CI [-0.007, +0.003]; ΔF1 = -0.008, 95% CI [-0.019, +0.002]) (single training seed).

- On shooting-distance (MLSD) classification, under an identical protocol, a wavelet-KAN head (Mexican hat, 0.26 M parameters) underperformed an MLP head of the same size (0.26 M; AUC 0.715 ± 0.113 vs. 0.802 ± 0.062; ΔAUC = -0.071, 95% CI [-0.098, -0.043]; better in 0/5 folds) (seeds 42, 43, 44).

- On shooting-distance (MLSD) classification, a wavelet-KAN head (W1: Mexican hat, 1.05 M parameters) did not reach non-inferiority against a parameter-matched MLP head (3.94 M; ΔAUC = -0.086, 95% CI [-0.120, -0.052]; ΔF1 = -0.070, 95% CI [-0.121, -0.024]) (seeds 42, 43, 44).

- On shooting-distance (MLSD) classification, a wavelet-KAN head (W1: Mexican hat, 1.05 M parameters) did not reach non-inferiority against the spline-KAN head (3.94 M; ΔAUC = -0.075, 95% CI [-0.105, -0.044]; ΔF1 = -0.067, 95% CI [-0.113, -0.024]) (seeds 42, 43, 44).

- On shooting-distance (MLSD) classification, a wavelet-KAN head (W2: Mexican hat, 0.26 M parameters) did not reach non-inferiority against the spline-KAN head (3.94 M; ΔAUC = -0.066, 95% CI [-0.096, -0.038]; ΔF1 = -0.070, 95% CI [-0.119, -0.022]) (seeds 42, 43, 44).

- On entry/exit classification, W2 has the same head-parameter budget as the MLP-128 head at equal wall-clock time (8.5 vs. 8.3 min per fold on an NVIDIA L4); the ResNet-152 backbone dominates the cost.

- On entry/exit classification, W1 has 3.7× fewer head parameters than the spline-KAN head but trains 4% slower per fold (8.8 vs. 8.5 min on an NVIDIA L4), because each wavelet layer materialises a batch × output × input tensor.

- On shooting-distance (MLSD) classification, W2 has the same head-parameter budget as the MLP-128 head at equal wall-clock time (6.4 vs. 6.3 min per fold on an NVIDIA L4); the ResNet-152 backbone dominates the cost.

- On shooting-distance (MLSD) classification, W1 has 3.7× fewer head parameters than the spline-KAN head but trains 4% slower per fold (6.7 vs. 6.5 min on an NVIDIA L4), because each wavelet layer materialises a batch × output × input tensor.

## PT (dissertação)

- Em entrada/saída, no mesmo protocolo, uma cabeça Wav-KAN (chapéu mexicano, 0.26 M parâmetros) não se mostrou não inferior a uma cabeça MLP do mesmo tamanho (0.26 M; AUC 0.902 ± 0.013 contra 0.905 ± 0.014; ΔAUC = -0.001, IC 95 % [-0.004, +0.002]; melhor em 0/5 dobras; ΔF1 = -0.012, IC 95 % [-0.023, -0.003]; o limite inferior do IC de ΔF1 fica abaixo da margem pré-registrada de -0.02), mas a regra pré-registrada para uma diferença (≥ 4/5 dobras e IC sem o zero) também não foi cumprida (sementes 42, 43, 44).

- Em entrada/saída, uma cabeça Wav-KAN (W1: chapéu mexicano, 1.05 M parâmetros) foi não inferior a uma cabeça MLP isoparamétrica (3.94 M) dentro da margem pré-registrada (ΔAUC = -0.002, IC 95 % [-0.005, +0.002]; ΔF1 = -0.003, IC 95 % [-0.014, +0.008]) (sementes 42, 43, 44).

- Em entrada/saída, uma cabeça Wav-KAN (W5: chapéu mexicano, 3.94 M parâmetros) foi não inferior a uma cabeça MLP isoparamétrica (3.94 M) dentro da margem pré-registrada (ΔAUC = -0.003, IC 95 % [-0.008, +0.002]; ΔF1 = -0.009, IC 95 % [-0.019, +0.003]) (uma única semente de treino).

- Em entrada/saída, uma cabeça Wav-KAN (W1: chapéu mexicano, 1.05 M parâmetros) foi não inferior à cabeça KAN spline (3.94 M) dentro da margem pré-registrada (ΔAUC = -0.001, IC 95 % [-0.004, +0.003]; ΔF1 = -0.005, IC 95 % [-0.016, +0.005]) (sementes 42, 43, 44).

- Em entrada/saída, uma cabeça Wav-KAN (W2: chapéu mexicano, 0.26 M parâmetros) não atingiu a não inferioridade em relação à cabeça KAN spline (3.94 M; ΔAUC = -0.001, IC 95 % [-0.003, +0.002]; ΔF1 = -0.011, IC 95 % [-0.021, -0.002]) (sementes 42, 43, 44).

- Em entrada/saída, uma cabeça Wav-KAN (W5: chapéu mexicano, 3.94 M parâmetros) foi não inferior à cabeça KAN spline (3.94 M) dentro da margem pré-registrada (ΔAUC = -0.002, IC 95 % [-0.007, +0.003]; ΔF1 = -0.008, IC 95 % [-0.019, +0.002]) (uma única semente de treino).

- Em MLSD (distância), no mesmo protocolo, uma cabeça Wav-KAN (chapéu mexicano, 0.26 M parâmetros) ficou abaixo de uma cabeça MLP do mesmo tamanho (0.26 M; AUC 0.715 ± 0.113 contra 0.802 ± 0.062; ΔAUC = -0.071, IC 95 % [-0.098, -0.043]; melhor em 0/5 dobras) (sementes 42, 43, 44).

- Em MLSD (distância), uma cabeça Wav-KAN (W1: chapéu mexicano, 1.05 M parâmetros) não atingiu a não inferioridade em relação a uma cabeça MLP isoparamétrica (3.94 M; ΔAUC = -0.086, IC 95 % [-0.120, -0.052]; ΔF1 = -0.070, IC 95 % [-0.121, -0.024]) (sementes 42, 43, 44).

- Em MLSD (distância), uma cabeça Wav-KAN (W1: chapéu mexicano, 1.05 M parâmetros) não atingiu a não inferioridade em relação à cabeça KAN spline (3.94 M; ΔAUC = -0.075, IC 95 % [-0.105, -0.044]; ΔF1 = -0.067, IC 95 % [-0.113, -0.024]) (sementes 42, 43, 44).

- Em MLSD (distância), uma cabeça Wav-KAN (W2: chapéu mexicano, 0.26 M parâmetros) não atingiu a não inferioridade em relação à cabeça KAN spline (3.94 M; ΔAUC = -0.066, IC 95 % [-0.096, -0.038]; ΔF1 = -0.070, IC 95 % [-0.119, -0.022]) (sementes 42, 43, 44).

- Em entrada/saída, a W2 tem o mesmo orçamento de parâmetros na cabeça que a MLP-128 com o mesmo tempo de treino (8.5 contra 8.3 min por dobra na L4); o backbone ResNet-152 domina o custo.

- Em entrada/saída, a W1 tem 3.7× menos parâmetros na cabeça que a KAN spline mas treina 4% mais devagar por dobra (8.8 contra 8.5 min na L4), porque cada camada com wavelets materializa um tensor lote × saída × entrada.

- Em MLSD (distância), a W2 tem o mesmo orçamento de parâmetros na cabeça que a MLP-128 com o mesmo tempo de treino (6.4 contra 6.3 min por dobra na L4); o backbone ResNet-152 domina o custo.

- Em MLSD (distância), a W1 tem 3.7× menos parâmetros na cabeça que a KAN spline mas treina 4% mais devagar por dobra (6.7 contra 6.5 min na L4), porque cada camada com wavelets materializa um tensor lote × saída × entrada.

## Pode afirmar

```json
{
 "more_parameter_efficient_than_an_MLP": {
  "es": false,
  "dist": false
 },
 "wavkan_faster": {
  "es__W2_vs_b": false,
  "es__W1_vs_c": false,
  "es__W5_vs_c": false,
  "es__W1_vs_d": false,
  "es__W2_vs_d": true,
  "es__W5_vs_d": false,
  "dist__W2_vs_b": false,
  "dist__W1_vs_c": false,
  "dist__W1_vs_d": false,
  "dist__W2_vs_d": true
 },
 "compare_with_0.908_or_0.893": false,
 "spline_KAN_is_the_reference_that_works": false
}
```

## Lembretes

- Não comparar com 0,908 (TCC antigo) nem com 0,893 (artigo do SIBGRAPI): protocolos diferentes.
- O contraste que mais informa é contra a melhor MLP do mesmo orçamento; a KAN spline (d) empata com as MLPs em E/S e fica abaixo da MLP-iso na MLSD.
- Uma vantagem da Wav-KAN pode vir de `translation`/`scale` se ajustarem à faixa das features (≥ 0, várias acima de 1), e não "das wavelets": veja `diagnostico_wavkan.csv` e a fração dentro da grade da spline.
- Citar Lira et al. (2025) como origem da base; nenhuma imagem sai da sessão.
