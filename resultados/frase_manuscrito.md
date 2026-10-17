# Frase para o manuscrito (desfecho B)

Under an identical protocol, the KAN head performed on par with linear and MLP heads (AUC 0.905±0.013 vs. 0.906±0.014; ΔAUC = -0.001, 95% CI [-0.002, +0.000] includes zero). TransKAN's advantage over the baselines of Lira et al. is explained by the pre-trained ResNet-152 backbone and the two-phase fine-tuning recipe rather than by the KAN head itself; we present the KAN head as a competitive alternative, not as the source of the gain.

- Comparação primária: KAN (d) × ResNet152 + MLP-iso (c); ΔAUC OOF = -0.001, IC 95 % [-0.002, +0.000]; KAN vence em 2/5 dobras.
- Pode afirmar: {"so_trocar_a_cabeca_melhora": false, "new_SOTA": false, "outperforms_transformer_baselines": false, "parameter_efficiency": false}
- Baselines não reavaliados (saem da tabela ou ficam marcados): nenhum
- Correções obrigatórias no texto: substituir os números antigos da TransKAN; remover SMOTE+ENN; checkpoint = última época; dobras por caso estratificadas por strat4 com semente fixa; baselines pré-treinados no mesmo protocolo; AUC por probabilidade; "two-layer KAN"; contagens de imagens e casos do notebook 00; citar Lira et al. (2025, IJLM) como origem da base (GuWID = FDCPUnBGunshotDB) e da tabela de referência.
