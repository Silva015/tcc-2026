# Auditoria de consistência — pôster × manuscript 6 × dissertação × notebooks × Lira 2024 (30/09/2026)

## Pôster × manuscript 6
- Pôster reproduz o manuscript 6 fielmente (título, GuWID, 2.551 img / 465+24 casos, SMOTE+ENN, Albumentations, KAN [2048,128,2] ordem 3 grade 10, treino 2 fases, 5-fold agrupado estratificado, Tabela I idêntica, futuro).
- Diferença menor: manuscrito diz "three-layer KAN classifier"; [2048,128,2] são 2 camadas KAN (3 larguras).

## Manuscript 6 × realidade
- Tabela I (12 baselines + ResNet152) = cópia exata da Tabela 4 de Lira et al. 2025 (hold-out). Manuscrito afirma que todos foram treinados no mesmo protocolo 5-fold/2 fases — não confere.
- Lira et al. aparece só como [14] na introdução, não como fonte da tabela nem da base (GuWID = FDCPUnBGunshotDB).
- AUC dos baselines = recall macro; confirmado no código leguwoi: `roc_auc_score(y_test, y_pred)` com classes previstas. AUC da TransKAN vem de probabilidades → comparação desigual.
- Números da TransKAN no pôster (acc .871, prec .847, F1 .832, AUC .893/.924) não aparecem em nenhum notebook; foi imprecisão. Números reais: acc .860, prec M .817, rec M .835, F1 M .825, AUC .908/.928.
- Notebooks não têm SMOTE/ENN nem estratificação (KFold simples agrupado por caso).

## Dissertação × notebooks
- Tabelas de Entrada/Saída e MLSD batem exatamente com os notebooks.
- Descompasso: notebook MLSD usa torchvision (flip, rotação, brilho/contraste), não Albumentations.
- Checkpoint escolhido no teste de cada dobra; última época dá média ~85,4% vs melhor ~86,2% (efeito < 1 ponto).

## Código de 2024 (repositório leguwoi, clonado em references/leguwoi)
- Hold-out (src/data/gunshot_dataset.py, `iterative_split`): usa IterativeStratification com colunas [case_number, alvo] → ESTRATIFICA por caso em vez de agrupar. Reproduzido com o CSV público: ~99% das imagens de teste têm o mesmo case_number no treino. Ou seja, o 0,869 da Tabela 4 tem vazamento (no código público), apesar do texto do artigo dizer "split por caso".
- K-fold (src/data/gunshot_dataset_kfold.py): StratifiedGroupKFold com grupos = case_number → agrupado de verdade (Fig. 4, ResNet-152 mediana ~0,70).
- Outros: class weights comentados (todos = 1) no hold-out; augmentation aplicada uma única vez no carregamento; case_number sem o ano (417 grupos vs 485 ano.caso).

## Dissertação × Lira 2024 (correções)
- Afirmação de vazamento no hold-out de 2024: DEFENSÁVEL, mas deve citar o código (iterative_split estratifica por caso), não o texto do artigo.
- ERRADO: mediana do boxplot (Fig. 4a) é ~0,70 (caixa ~0,64–0,75), não ~0,73.
- ERRADO: resumo diz melhora em precisão/recall ponderados no MLSD (na verdade .853 vs .917 e .873 vs .925).
- Questionável: "parameter efficiency" — cabeça KAN ~3,9 M params vs ~260 k de um MLP equivalente.

## Take-home messages reais (pôster)
1. Dividir por foto infla resultados: mesma ResNet-152 cai de 0,87 (hold-out com vazamento) para ~0,70 (k-fold por caso).
2. Com 5-fold por caso, ResNet-152 + cabeça KAN (com a nossa receita de treino): acc 0,86, recall macro 0,835, AUC 0,91.
3. KAN precisa de extrator pré-treinado (ablações 0,69–0,82).
- Não afirmar que "só trocar a cabeça" melhora: falta ResNet+MLP no mesmo protocolo (próximo experimento).

## Atualização 14/10/2026
- Notebooks TransferKAN no Colab (EntradaSaida e MLSD) conferidos: idênticos aos do repo, com as mesmas saídas. Não existe outro notebook.
- FDCPUnBGunshotDB (GitHub) não tem licença de redistribuição e o README exige citar Lira et al. (2025, IJLM).
