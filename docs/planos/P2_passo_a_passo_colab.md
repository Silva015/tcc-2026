# P2 (multitarefa): passo a passo no Colab

O que você faz em cada notebook, em ordem. Todos estão na pasta `Colab Notebooks/tcc-final` do Drive.

## Antes de começar

- **Abrir um notebook:** no Drive, entre em `tcc-final` e dê dois cliques no arquivo. Ele abre no Colab.
- **Escolher a placa:** *Ambiente de execução → Alterar o tipo de ambiente de execução*. Escolha **L4** ou **CPU**, conforme o passo.
- **Célula de configuração:** é a primeira célula de código. Ela começa com `# ==== Configuração`. Só mude o que o passo mandar.
- **Executar tudo:** *Ambiente de execução → Executar tudo*.
- **Saldo:** o P2 inteiro gasta **~8 unidades** (~5 h de L4 no total). Se a L4 aparecer cinza, o saldo acabou.
- **Se a aba cair** no meio de uma rodada: abra o mesmo notebook de novo e clique em **Executar tudo**. Ele pula o que já terminou e continua de onde parou.
- **Uma aba, um notebook:** nunca abra o mesmo notebook em duas abas. Por isso existem dois arquivos do 09: um para a aba A e outro para a aba B.
- **Desconexão automática:** os dois notebooks 09 se desconectam sozinhos quando a rodada termina. O 01 não: desconecte você.

## Passo 1. `01_calibracao_L4.ipynb` (L4, ~15 min, ~0,4 unidade)

- **Mudar:** nada.
- **Fazer:** escolha **L4** e clique em **Executar tudo**.
  - O que já foi medido antes é pulado.
  - A parte 7 faz um teste curto da rede multitarefa.
- **Deu certo se,** perto do fim, aparecer uma linha com `'config': 'mt_kan'` e `'ok': True`. Um `nan` na AUC dessa linha é normal: o teste usa poucas imagens.
- **Ao terminar:** *Ambiente de execução → Desconectar e excluir ambiente de execução*.

## Passo 2. `09_multitarefa_entrada_saida_distancia.ipynb`: teste rápido (aba A, L4, ~15 min)

- **Mudar:** nada. O arquivo já vem com `TESTE_RAPIDO = True` e `SO_CONFIGS = ["mt_kan"]`.
- **Fazer:** escolha **L4** e clique em **Executar tudo**.
- **Deu certo se** aparecer `✅ Teste rápido OK`.
  - Logo depois aparece um erro vermelho dizendo `TESTE_RAPIDO está True`. **Esse erro é esperado**: é o notebook parando de propósito.
  - Antes disso aparece uma tabelinha com o tempo de inferência (a rede única contra o par de redes). É só informação.

## Passo 3. O mesmo `09`: rodada de verdade (aba A, L4, ~2,3 h, ~3,5 unidades)

- **Mudar:** `TESTE_RAPIDO = False`. O resto fica como está.
- **Fazer:** clique em **Executar tudo** e deixe a aba aberta.
- **O que roda:** a rede multitarefa com cabeças KAN (`mt_kan`), sementes 42, 43 e 44, 5 dobras cada.
- Antes de cada dobra, ele mostra quanto tempo e quantas unidades ela vai gastar.

## Passo 4. `09_multitarefa_abaB.ipynb` (aba B, L4, ~2,3 h, ~3,5 unidades, ao mesmo tempo que o passo 3)

- **Quando:** **1 minuto depois** de começar o passo 3.
- **Mudar:** nada. Este arquivo já vem com `ABA = "B"`, `SO_CONFIGS = ["mt_mlp"]` e `TESTE_RAPIDO = False`.
- **Fazer:** abra o arquivo numa **nova aba** do navegador, escolha **L4** e clique em **Executar tudo**.
- **O que roda:** a rede multitarefa com cabeças MLP (`mt_mlp`), sementes 42, 43 e 44.
- **Se preferir uma aba só:** pule este passo e, no passo 3, mude também `SO_CONFIGS = None`. A aba A roda as duas redes, uma depois da outra (~4,5 h).

## Passo 5. `10_analise_multitarefa.ipynb` (CPU, ~10 min, não gasta unidades de GPU)

- **Quando:** depois que as duas abas terminarem.
- **Mudar:** nada.
- **Fazer:** escolha **CPU** e clique em **Executar tudo**.
- **Depois:** volte a esta conversa e escreva: `Terminei as rodadas e o 10 no Colab. Confira os resultados no Drive e traga-os para o repositório.`

## Se algo der errado

- **`O teste rápido ainda não passou…`:** você abriu a aba B antes do passo 2 terminar. Termine o passo 2 e depois rode a aba B de novo.
- **`Perda não finita … avise a sessão P2`:** a fila parou sozinha e anotou o problema. Não rode de novo; me avise.
- **`… não é o publicado`:** um arquivo do Drive foi trocado. Não rode nada; me avise.
- **`GPU … não é ('L4',)`:** o Colab te deu outra placa, provavelmente porque o saldo acabou. Espere ou confira o saldo. Não ligue `PERMITIR_RESERVA`.
- **Uma aba caiu:** abra o mesmo arquivo de novo, escolha **L4** e clique em **Executar tudo** (com `TESTE_RAPIDO = False`). A outra aba continua normalmente.

---

## Notas técnicas para a sessão de análise

- **Plugin:** `notebooks/multitarefa.py` v1.0.0, SHA-256 `b9841e0f8c0192bf67df96f318e384659a96659c5eda08737dae740f612c1814`, embutido no 09 (abas A e B) e no 10.
  - Registra só `mt_kan` e `mt_mlp`: ResNet-152 → Dropout 0,5 → {`es`, `dist`}, cabeças nativas `kan` (d) e `mlp128` (b), `pesos_tarefa = {es: 1, dist: 1}`, perda `soma_ponderada`.
  - Parâmetros conferidos com a ResNet-152 de verdade: `mt_kan` 66.017.728 (par ST 124.161.536, −46,8 %); `mt_mlp` 58.668.997 (par ST 116.812.805, −49,8 %).
  - Experimento `multitarefa`; coluna de dobra `fold` (todas as imagens).
- **Diferenças em relação ao plano P2 e por quê:**
  - Não existe CLI. A fila roda pela API, via `mt.executar_fila_mt`, que envolve `pc.executar_unidade`, grava `falha_nan.json` e para a fila se a perda virar NaN.
  - As redes de tarefa única **não** são treinadas: são as do P1 (`resnet152__kan` e `resnet152__mlp128` em `ablacao_cabeca`; `mlsd__resnet152__kan` e `mlsd__resnet152__mlp128` em `mlsd_cabeca`), sementes 42–44 (decisão do Arthur, notas do P1 §3 e §8).
  - Não há perda `soma_mascarada` registrada. O protocolo já faz a máscara: CE ponderada com `ignore_index = −1` e, num lote sem nenhuma entrada, `L_dist = 0·logits`. A `soma_ponderada` com λ = 1 dá L = L_es + L_dist. Assim a MT e as ST passam pelo mesmo código. Testado em CPU no caminho real do protocolo: com 3 lotes sem entrada, o gradiente da cabeça de distância foi zero exatamente nesses lotes, e as perdas ficaram finitas.
  - Pesos de classe e imagens de teste da MT iguais aos das ST do P1, dobra a dobra (`mt.checar_dados`, que roda no 09). Distância: ≈ 13–17 / 2,5–2,9 / 0,39.
  - Não existe `salvar_pesos_finais`. O tempo de inferência (1 forward da MT × 2 do par ST, lotes 1 e 32, 20 aquecimentos + 200 medidas com eventos CUDA) é medido no teste rápido do 09, com modelos recém-criados (o tempo não depende dos pesos), e vai para `resultados/multitarefa/calibracao_mt.json`.
  - O "modo calibração" do plano virou só esse benchmark. t_F1 e t_F2 saem das rodadas de verdade (P1 e MT); o *smoke* da MT na L4 é a parte 7 do 01.
  - Sementes 42, 43 e 44 nas duas configs, ordem semente → config → dobra. Duas abas com filas disjuntas (A: `mt_kan`, B: `mt_mlp`). A trava é a do protocolo, com o `ABA` como dono. Cada aba grava a própria `fila_aba{A,B}.json`; os agregados ficam por config (`resultados/multitarefa/<config>/`), então as abas não escrevem no mesmo arquivo.
  - A aba B só roda depois que o teste rápido da aba A grava `teste_rapido_ok.json` com o SHA do plugin.
  - **Teste de retomada (corrigido em 21/10, depois da 1ª tentativa na L4):** a 1ª versão exigia diferença < 1e-3 entre a rodada interrompida e a de referência. Na L4 a Fase 1 saiu idêntica, mas a Fase 2 diferiu até 6,7e-4 nas perdas e passou de 1e-3 nas probabilidades. É ruído da GPU: na F2 o gradiente atravessa o `max_pool2d` da ResNet, cujo backward na CUDA não é determinístico (o P1 já tinha visto 9,3e-4 no 01). Agora o 09 roda uma 2ª referência sem interrupção, mede esse ruído e exige: F1 igual (≤ 1e-5) e F2 ≤ 3× o ruído, com piso de 1e-3. Só o 09 mudou; o plugin e o SHA são os mesmos.
- **Regra de decisão do 10** (fixada antes de qualquer rodada da MT):
  - primários MT × ST com a mesma cabeça, por tarefa; secundário KAN × MLP dentro da MT; ST-KAN × ST-MLP só descritivo;
  - "melhora" = ≥ 4/5 dobras e IC do ΔAUC > 0; "não inferior" = limite inferior do IC do ΔAUC > −0,02 (só ΔAUC, como no plano P2; ΔF1 vem junto, mas não decide);
  - "pior" (acrescentado, espelho de "melhora", como o P3 fez) = ≤ 1/5 dobras e IC < 0; se ficar dentro da margem, a frase diz isso. Senão, "inconclusivo".
- **Pipeline de 4 classes:** junta por `rel_path` o `pred_teste.csv` da ST de entrada/saída e o da ST de distância da mesma semente e dobra (o teste da ST de distância do P1 é a dobra inteira, então as saídas têm `p_dist_*`). Na MT, as duas cabeças da mesma unidade.
- **Saídas do 10:** `resultados/tabela_multitarefa.{csv,tex}`, `contrastes_multitarefa.{csv,tex}`, `pipeline_multitarefa.csv`, `confusao_multitarefa.csv`, `decisao_multitarefa.json` e `frases_multitarefa.md`; e `resultados/multitarefa/<config>/{runs.csv,resumo.json,oof.csv}`.
- **Testes locais (CPU):**
  - Plugin: 34 checagens (máscara e guarda no caminho real do protocolo, parâmetros, métricas por tarefa, retomada idêntica, trava entre duas abas, NaN, benchmark).
  - 09 (abas A e B) e 10 de ponta a ponta num Drive falso, com GPU e backbone simulados e as dobras reais: a aba B recusa rodar antes do teste da A, as duas abas rodam ao mesmo tempo sem pegar a mesma unidade, e o 10 roda com as ST do P1 simuladas e com predições sintéticas de efeito controlado (MT igual, melhor e pior que a ST). As tabelas `.tex` compilam com pdfLaTeX.
  - O 10 lê `case_id` dos `pred_teste.csv` como texto (`dtype={"case_id": str}`); lido como número, `2014.0100` vira `2014.01`.
  - Não foi testado: o desempenho real na L4 e a montagem do Drive.
- **Drive (`tcc-final`, 21/10/2026):** `multitarefa.py` (id `1jBf8gI0ZaQUAekCVscWlBVbgUvmfWmOc`), `09_multitarefa_entrada_saida_distancia.ipynb` (`1e0JqjKTeihspHWJdqKdhQRtk3VUgXZ2_`), `09_multitarefa_abaB.ipynb` (`1ugny8r6Abqnt9Ql5i87dqvX_PgpHrcTD`) e `10_analise_multitarefa.ipynb` (`1LapBz-Ctd9sPP8aOQmUwhVs0oan446iO`). O plugin, o 09 da aba A e o 10 foram baixados de volta e são idênticos byte a byte aos do repositório; o 09 da aba B confere no tamanho e no cabeçalho e na célula de configuração.
