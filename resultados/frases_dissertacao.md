# Frases para a dissertação (geradas pelo 05b)

## entrada/saída — desfecho B

**PT.** Sob o mesmo protocolo, a cabeça KAN teve desempenho equivalente ao das cabeças linear e MLP na classificação entrada/saída (AUC 0.905±0.013 × 0.906±0.014; ΔAUC = -0.001, IC 95 % [-0.002, +0.000] inclui zero). O ganho da TransKAN sobre os baselines de Lira et al. vem do extrator pré-treinado e da receita de duas fases, não da cabeça KAN.

**EN.** Under an identical protocol, the KAN head performed on par with linear and MLP heads for entry/exit classification (AUC 0.905±0.013 vs. 0.906±0.014; ΔAUC = -0.001, 95% CI [-0.002, +0.000] includes zero). TransKAN's advantage over the baselines of Lira et al. is explained by the pre-trained backbone and the two-phase recipe rather than by the KAN head itself.

## MLSD (distância) — desfecho C

**PT.** Sob o mesmo protocolo, a cabeça KAN ficou abaixo das MLPs na estimativa da distância do disparo (MLSD) (AUC 0.798±0.058 × 0.807±0.055; ΔAUC = -0.011, IC 95 % [-0.021, -0.003]; melhor em 1/5 dobras).

**EN.** Under an identical protocol, the KAN head underperformed the MLP heads for medico-legal shooting distance (MLSD) (AUC 0.798±0.058 vs. 0.807±0.055; ΔAUC = -0.011, 95% CI [-0.021, -0.003]; better in 1/5 folds).

## Comparação com Lira et al. (2024)

- entrada/saída: ResNet152 em Lira = acc 0.869, AUC(classes, M) 0.821 (hold-out com vazamento); no mesmo protocolo = acc 0.846 ± 0.008, AUC(classes, M) 0.814 ± 0.012, AUC(prob, M) 0.904 ± 0.013; TransKAN = acc 0.846 ± 0.010, AUC(prob, M) 0.905 ± 0.013.
- MLSD (distância): ResNet152 em Lira = acc 0.925, AUC(classes, M) 0.733 (hold-out com vazamento); no mesmo protocolo = acc 0.815 ± 0.014, AUC(classes, M) 0.670 ± 0.047, AUC(prob, M) 0.785 ± 0.087; TransKAN = acc 0.813 ± 0.017, AUC(prob, M) 0.798 ± 0.058.

**Cuidado:** os números de Lira vêm de outra partição (com vazamento de casos), outra receita (quase tudo treinado do zero) e AUC de classes previstas; a diferença mede o pacote todo, não um modelo contra o outro. O efeito isolado da partição está na tabela de vazamento.

## Vazamento (mesma receita, semente 42)

- entrada/saída, ResNet152 + Linear (a): AUC hold-out de Lira 0.940 ± 0.003 × agrupado 0.906 ± 0.013 (Δ = +0.034); acc 0.876 ± 0.005 × 0.848 ± 0.010; 98.7 % das imagens de teste com o caso no treino.
- entrada/saída, ResNet152 + KAN (d) = TransKAN: AUC hold-out de Lira 0.940 ± 0.003 × agrupado 0.906 ± 0.011 (Δ = +0.034); acc 0.881 ± 0.006 × 0.846 ± 0.008; 98.7 % das imagens de teste com o caso no treino.
- MLSD (distância), ResNet152 + Linear (a): AUC hold-out de Lira 0.937 ± 0.004 × agrupado 0.779 ± 0.088 (Δ = +0.158); acc 0.857 ± 0.006 × 0.817 ± 0.014; 98.4 % das imagens de teste com o caso no treino.
- MLSD (distância), ResNet152 + KAN (d) = TransKAN: AUC hold-out de Lira 0.942 ± 0.003 × agrupado 0.788 ± 0.063 (Δ = +0.155); acc 0.863 ± 0.002 × 0.810 ± 0.018; 98.4 % das imagens de teste com o caso no treino.

**EN.** Reproducing the hold-out split of Lira et al. (image-level iterative stratification, in which 98.6% of the test images share their case with training images) under our recipe changes AUC by +0.034 (entry/exit, linear), +0.034 (entry/exit, KAN), +0.158 (MLSD, linear), +0.155 (MLSD, KAN) relative to case-grouped folds; numbers obtained with leaky splits are therefore not comparable with ours.

## Pode afirmar?

```json
{
 "es": {
  "so_trocar_a_cabeca_melhora": false,
  "novo_estado_da_arte_no_protocolo": false,
  "supera_transformers": false,
  "eficiencia_de_parametros": false,
  "baselines_reavaliados": [
   "alexnet",
   "mobilenet_v2",
   "mnasnet1_3",
   "shufflenet_v2_x0_5",
   "squeezenet1_0",
   "densenet169",
   "maxvit_t",
   "swin_b",
   "efficientnet_v2_l",
   "vgg16",
   "regnet_y_16gf"
  ],
  "melhor_modelo_auc": [
   "Swin-B",
   0.9286
  ]
 },
 "dist": {
  "so_trocar_a_cabeca_melhora": false,
  "novo_estado_da_arte_no_protocolo": false,
  "supera_transformers": false,
  "eficiencia_de_parametros": false,
  "baselines_reavaliados": [
   "alexnet",
   "mobilenet_v2",
   "mnasnet1_3",
   "shufflenet_v2_x0_5",
   "squeezenet1_0",
   "densenet169",
   "maxvit_t",
   "swin_b",
   "efficientnet_v2_l",
   "vgg16",
   "regnet_y_16gf"
  ],
  "melhor_modelo_auc": [
   "Swin-B",
   0.8381
  ]
 }
}
```

- Correções obrigatórias: substituir os números antigos da TransferKAN-MLSD (acc 0,873; AUC 0,749: fase única, grade 5, melhor época escolhida no teste) pelos do protocolo único; a hipótese "a cabeça KAN melhora" só pode ser mantida no desfecho A; citar Lira et al. (2025, IJLM) como origem da base e das Tabelas 4 e 5.
