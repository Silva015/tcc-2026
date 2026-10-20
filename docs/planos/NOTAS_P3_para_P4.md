# Notas do P3 (Wav-KAN) para o P4

Resultados das rodadas de 19–20/10/2026 na L4, trazidos do Drive (`tcc-final`) e conferidos.

## Onde está cada coisa

| Arquivo | O que é |
|---|---|
| `resultados/decisao_wavkan.json` | Contrastes, IC, desfechos da regra pré-registrada, tempo, frases EN/PT |
| `resultados/frases_wavkan.md` | As mesmas frases, prontas para copiar |
| `resultados/tabela_wavkan.{csv,tex}` | Entrada/saída: (a)–(d) do P1 + W1–W5 |
| `resultados/tabela_wavkan_mlsd.{csv,tex}` | MLSD: (a)–(d) + W1, W2 |
| `resultados/contrastes_wavkan.{csv,tex}` | Tabela de contrastes (ΔAUC e ΔF1 com IC 95 %) |
| `resultados/fig_auc_vs_parametros.pdf` | AUC × parâmetros da cabeça (eixo log), barras = dp entre dobras |
| `resultados/diagnostico_wavkan.csv` | `scale`/`translation` das camadas Wav-KAN no fim de cada fase |
| `resultados/benchmark_cabecas.json` | Saída do `06`: equivalência com o torch-conv-kan, tempo e memória só da cabeça |
| `resultados/wavkan/{fila.json,runs.csv,resumo.json,oof.csv}` | Entrada/saída: 45 unidades (cabeça × semente × dobra) e predições fora da dobra |
| `resultados/wavkan/notebooks_executados/` | `06`, `07` e `08` como rodaram no Colab (sem o nome e o ID do Google nas células) |

Ficaram só no Drive: as pastas por unidade, os checkpoints e as rodadas da MLSD (`resultados/wavkan_mlsd/`), que ainda não têm `runs.csv`/`oof.csv` porque a W2 semente 44 não terminou.

## Conferências feitas

- Os tamanhos de todos os arquivos batem com os do Drive.
- `runs.csv` bate com `resumo.json`, e a AUC de cada dobra recalculada a partir do `oof.csv` bate com a do `runs.csv` (diferença máxima de 1e-16).
- Os contrastes de entrada/saída, recalculados à parte com o mesmo bootstrap pareado (2000 reamostragens, semente 0), batem com o `08` até a 6.ª casa decimal.
- `06`: a camada e a cabeça são idênticas às do torch-conv-kan @ b412a3b em CPU e CUDA, com diferença 0 na saída e no gradiente.
- Todas as rodadas usaram a L4, o protocolo SHA b7cd9bac… e o plugin SHA d02f6ca6….

## Resultados

**Entrada/saída** (AUC média ± dp entre dobras; F1 macro; min/dobra na L4):

| Cabeça | Params cabeça | AUC | F1 | Acc | min/dobra | Sementes |
|---|---|---|---|---|---|---|
| (a) Linear | 4 098 | 0,904 ± 0,013 | 0,805 | 0,846 | 8,2 | 42–44 |
| (b) MLP-128 | 262 530 | 0,905 ± 0,014 | 0,806 | 0,846 | 8,3 | 42–44 |
| (c) MLP-iso | 3 935 871 | 0,906 ± 0,014 | 0,807 | 0,846 | 8,3 | 42–44 |
| (d) KAN spline | 3 936 000 | 0,905 ± 0,013 | 0,806 | 0,846 | 8,5 | 42–44 |
| W1 chapéu mexicano h128 | 1 049 860 | 0,902 ± 0,014 | 0,800 | 0,835 | 8,8 | 42–44 |
| W2 chapéu mexicano h32 | 262 468 | 0,902 ± 0,013 | 0,799 | 0,834 | 8,5 | 42–44 |
| W3 DoG h128 | 1 049 860 | 0,908 ± 0,015 | 0,793 | 0,823 | 8,7 | 42 |
| W4 Morlet h128 | 1 049 860 | 0,901 ± 0,015 | 0,779 | 0,807 | 8,7 | 42 |
| W5 chapéu mexicano h480 | 3 936 964 | 0,903 ± 0,013 | 0,799 | 0,834 | 10,6 | 42 |

- **Contraste primário 1 (W2 × b):** *inconclusivo*. ΔAUC = −0,001 [−0,004; +0,002], dentro da margem; ΔF1 = −0,012 [−0,023; −0,003]. O IC do F1 exclui o zero e passa da margem de −0,02, por isso não há não inferioridade. A W2 não vence a (b) em nenhuma dobra.
- **Contraste primário 2:** W1 × (c) e W5 × (c) não inferiores (W5 com uma só semente).
- **Secundários:** W1 × (d) e W5 × (d) não inferiores; W2 × (d) não (limite inferior do ΔF1 −0,0206).
- **Sensibilidade (semente 42):** a DoG (W3) tem a maior AUC da tabela (0,908), mas F1 e acurácia mais baixos. A Morlet (W4) é a pior em F1 (ΔF1 contra W1 = −0,023 [−0,035; −0,012]).

**MLSD** (só entradas):

| Cabeça | AUC | F1 | Acc | Sementes |
|---|---|---|---|---|
| (a) | 0,785 ± 0,087 | 0,488 | 0,815 | 42–44 |
| (b) | 0,802 ± 0,062 | 0,482 | 0,813 | 42–44 |
| (c) | 0,807 ± 0,055 | 0,480 | 0,821 | 42–44 |
| (d) | 0,798 ± 0,058 | 0,477 | 0,813 | 42–44 |
| W1 | 0,711 ± 0,116 | 0,418 | 0,671 | 42–44 |
| W2 | 0,713 ± 0,112 | 0,410 | 0,647 | 42, 43 |

- **Contraste primário 1 (W2 × b):** *inferior* (C). ΔAUC = −0,076 [−0,103; −0,050]; perde nas 5 dobras.
- Todos os contrastes da MLSD (W1 × c, W1 × d, W2 × d, W1 × b) dão inferior, com ΔAUC entre −0,072 e −0,086.

## Cuidados para o texto

1. **Frase corrigida.** A frase do `08` para o resultado *inconclusivo* dizia que "o limite inferior do IC fica abaixo da margem", mas citava só o IC do ΔAUC, que está dentro dela. Quem fica abaixo da margem é o ΔF1. O `notebooks/08_analise_wavkan.ipynb` foi corrigido: agora a frase nomeia a métrica e traz o IC dela. `frases_wavkan.md` e `decisao_wavkan.json` foram regerados com o código novo a partir das mesmas saídas, e só a frase W2 × b (EN e PT) mudou. O `08` executado em `notebooks_executados/` mostra a frase antiga, porque é o registro da rodada.
2. **MLSD W2 sem a semente 44.** A W2 na MLSD tem só as dobras 0–2 da semente 44, então a análise usou as sementes 42 e 43. Para fechar, falta rodar as dobras 3 e 4 (cerca de 15 min de L4) e repetir o `08`. A conclusão (inferior, por ~0,07–0,09 de AUC) dificilmente muda.
3. **"Mais rápida" não vale.** `pode_afirmar.wavkan_faster.es__W2_vs_d = true` sai da regra (a cabeça sozinha é mais rápida que a KAN spline, 2,95 × 3,30 ms, e a dobra encurtou). Mas a dobra encurtou só 0,35 % (8,46 × 8,49 min, dp ≈ 0,1), o que está dentro do ruído. A própria regra manda dizer "mesmo tempo" quando a diferença cabe no ruído. Não escrever que a Wav-KAN treina mais rápido. A W1 é 3,6 % mais lenta por dobra que a KAN spline e 5,3 % mais lenta que a MLP-iso; a W5 é 25–27 % mais lenta. Só a cabeça, com lote 32, a W1 leva 8,8 ms contra 0,73 ms da MLP-iso e usa 223 MB contra 0,24 MB.
4. **As wavelets quase não mudaram de forma.** Nas duas fases e nas duas tarefas, `scale` termina com média entre 0,99 e 1,003 (dp 0,013–0,020) e `translation` com média ≈ 0 (dp 0,013–0,025); os valores iniciais são 1 e 0. Na prática, a cabeça aprendeu os pesos de uma wavelet de forma fixa. Não atribuir resultado nenhum ao ajuste da forma da wavelet.
5. **Mesma AUC, F1/acurácia mais baixos.** Em entrada/saída, a Wav-KAN ordena os casos tão bem quanto as MLPs (AUC igual), mas no limiar 0,5 acerta ~1 ponto a menos de acurácia e de F1, com recall macro um pouco maior. Isso é deslocamento do limiar ou calibração, não falta de separação. Se o texto falar em F1, dizer isso.
6. **MLSD é onde a Wav-KAN perde de verdade.** Além da média mais baixa, a variação entre dobras dobra (dp 0,11–0,12 contra 0,06), e a acurácia cai de ~0,81 para ~0,65.
7. **Uma semente só.** W3, W4 e W5 rodaram só com a semente 42. Tratar como sensibilidade, nunca como resultado principal.
8. **Não comparar com 0,908 nem com 0,893** (TCC antigo e artigo do SIBGRAPI): os protocolos são outros.
9. **Figura.** Em `fig_auc_vs_parametros.pdf`, os rótulos W5, c e d se sobrepõem perto de 4 M de parâmetros. Se a figura entrar na dissertação, ajustar a posição dos rótulos no `08`.
