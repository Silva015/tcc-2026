# P3 (Wav-KAN): passo a passo no Colab

O que você faz em cada notebook, em ordem. Todos estão na pasta `Colab Notebooks/tcc-final` do Drive.

## Antes de começar

- **Abrir um notebook:** no Drive, entre em `tcc-final` e dê dois cliques no arquivo. Ele abre no Colab.
- **Escolher a placa:** *Ambiente de execução → Alterar o tipo de ambiente de execução*. Escolha **L4** ou **CPU**, conforme o passo.
- **Célula de configuração:** é a primeira célula de código. Ela começa com `# ==== Configuração desta sessão ====`. Só mude o que o passo mandar.
- **Executar tudo:** *Ambiente de execução → Executar tudo*.
- **Saldo:** para o P3 inteiro são ~17 unidades. Sem os opcionais, ~8.
- **Se a aba cair** no meio de uma rodada: abra o mesmo notebook de novo e clique em **Executar tudo**. Ele pula o que já terminou e continua de onde parou.
- **Uma aba, um notebook:** nunca abra o mesmo notebook em duas abas ao mesmo tempo.
- **Desconexão automática:** os notebooks 06 e 07 se desconectam sozinhos quando a rodada termina. Isso economiza unidades.

## Passo 1. `01_calibracao_L4.ipynb` (L4, ~15 min, uma vez)

- **Mudar:** nada.
- **Fazer:** escolha **L4** e clique em **Executar tudo**.
  - O que já foi medido em outubro é pulado.
  - A parte 7 mede as cabeças W1 e W5 e testa a multitarefa do P2.
- **Quando rodar:** o ideal é depois que a sessão P2 também subir os arquivos dela, para medir tudo de uma vez. Se quiser começar o P3 antes, rode agora e de novo quando o P2 estiver pronto (custa pouco).
- **Ao terminar:** *Ambiente de execução → Desconectar e excluir ambiente de execução*.

## Passo 2. `06_benchmark_cabecas.ipynb` (L4, ~10 min)

- **Mudar:** nada.
- **Fazer:** escolha **L4** e clique em **Executar tudo**.
- **Resultado:** ele confere as contas da Wav-KAN e mede o tempo das cabeças. Também calcula quanto cada rodada do 07 vai custar.
- No fim, ele se desconecta sozinho.

## Passo 3. `07_wavkan_cabeca_ResNet152.ipynb`: teste rápido (aba A, L4, ~10 min)

- **Mudar:**
  - `TAREFA = "es"`
  - `ETAPA = "principal"`
  - `TESTE_RAPIDO = True`
- **Fazer:** escolha **L4** e clique em **Executar tudo**.
- **Deu certo se** aparecer `✅ Teste rápido OK`.
  - Logo depois aparece um erro vermelho dizendo `TESTE_RAPIDO está True`. **Esse erro é esperado**: é o notebook parando de propósito.

## Passo 4. `07`: rodada de verdade, entrada/saída (aba A, L4, ~4,5 h, ~7 unidades)

- **Mudar:** `TESTE_RAPIDO = False`. O resto fica como no passo 3.
- **Fazer:** clique em **Executar tudo** e deixe a aba aberta.
- **Ordem da rodada:** W1 e W2 com a semente 42, depois com a 43, depois com a 44.
- **Antes de cada parte**, ele mostra quanto tempo e quantas unidades ela vai gastar.
- **Ao mesmo tempo, se quiser:** espere 1 minuto e comece o `09` do P2 na **aba B**, com o passo a passo da sessão P2.

## Passo 5. `07`: distância do disparo (MLSD) (L4, ~3,5 h, ~5 unidades)

Rode quando o passo 4 terminar, no mesmo notebook.

1. **Teste rápido:**
   - **Mudar:** `TAREFA = "mlsd"` e `TESTE_RAPIDO = True`. `ETAPA` continua `"principal"`.
   - **Fazer:** **Executar tudo** (~10 min).
   - **Deu certo se** aparecer `✅ Teste rápido OK`, seguido do erro vermelho esperado.
2. **Rodada:**
   - **Mudar:** `TESTE_RAPIDO = False`.
   - **Fazer:** **Executar tudo**.

## Passo 6 (opcional, por último, só se sobrar saldo). `07`: sensibilidade (L4, ~2,5 h, ~4 unidades)

1. **Teste rápido:**
   - **Mudar:** `TAREFA = "es"`, `ETAPA = "sensibilidade"` e `TESTE_RAPIDO = True`.
   - **Fazer:** **Executar tudo**.
2. **Rodada:**
   - **Mudar:** `TESTE_RAPIDO = False`.
   - **Fazer:** **Executar tudo**.

A rodada treina W3, W4 e W5, só com a semente 42.

## Passo 7. `08_analise_wavkan.ipynb` (CPU, ~5 min, não gasta unidades de GPU)

- **Mudar:** nada.
- **Fazer:** escolha **CPU** e clique em **Executar tudo**.
- **Depois:** volte a esta conversa e escreva: `Terminei as rodadas e o 08 no Colab. Confira os resultados no Drive e traga-os para o repositório.`

## Se algo der errado

- **`Perda não finita … avise a sessão P3`:** a fila parou sozinha e anotou o problema. Não rode de novo; me avise.
- **`… não é o publicado`:** um arquivo do Drive foi trocado. Não rode nada; me avise.
- **`GPU … não é ('L4',)`:** o Colab te deu outra placa, provavelmente porque o saldo acabou. Espere ou confira o saldo. Não ligue `PERMITIR_RESERVA`.

---

## Notas técnicas para a sessão de análise

- **Plugin:** `notebooks/cabecas_wavkan.py` v1.0.0. O SHA-256 está embutido no 06, no 07 e no 08.
  - Registra as cabeças `wavkan_{mexhat,dog,morlet}_h{32,128,480}` (as 5 usadas) e os apelidos `W1`–`W5`. A parte 7 do 01 procura os apelidos e mede só W1 e W5.
  - Configs `resnet152__wavkan_*` (E/S, W1–W5) e `mlsd__resnet152__wavkan_mexhat_h{128,32}` (MLSD, W1 e W2).
  - Experimentos `wavkan` e `wavkan_mlsd`.
- **Equivalência:** idêntica bit a bit ao `WavKANLayer` e ao `WavKAN` do torch-conv-kan @ `b412a3b`, na saída e nos gradientes. O repositório é MIT, e não Apache-2.0 como diz o plano; o cabeçalho do arquivo cita trechos Apache-2.0.
  - Contra a referência NumPy (float64), a diferença máxima é ≤ 5e-6 com 2048→128, em fp32 na CPU.
  - O 06 só executa as duas classes extraídas do clone fixado. Não importa o pacote.
- **Diferenças em relação ao plano P3 e por quê:**
  - Não existe CLI. A fila roda pela API, via `cw.executar_fila_wavkan`, que envolve `pc.executar_unidade`.
    - O diagnóstico de `scale`/`translation` vai para `diagnostico_wavkan.json` em cada unidade, e não para o `metricas.json`, que é do protocolo.
    - NaN grava `falha_nan.json` e para a fila.
  - A ordem da fila é semente → cabeça → dobra. Se o saldo acabar, W1 e W2 já terão a semente 42.
  - O 06 não repete a medida de ponta a ponta de W1/W5, que vem da parte 7 do 01.
    - As âncoras (a)–(d) entram só no tempo da cabeça, que custa segundos.
    - O tempo por dobra da tabela vem das rodadas reais (15·t_F1 + 45·t_F2).
  - Os contrastes primários seguem as notas do P1 (§8): W2 × (b) e W1/W5 × (c). W × (d) virou secundário.
    - A regra de W2 × (b) ganhou o desfecho "inferior", espelhando o desfecho C do 05, e o "inconclusivo".
    - O tempo ganhou "mais lenta", porque "*at equal wall-clock time*" seria falso se a Wav-KAN atrasar a dobra.
  - Margem de não inferioridade: **−0,02**, a pré-registrada em 14/10. No P1, o IC pareado teve largura de ~0,003, então ela é folgada. Pode ser estreitada, com o OK do Arthur, **antes** das rodadas (`MARGEM_NI` no 08).
- **Testes locais (CPU):**
  - O plugin passou nas contagens, na equivalência e em uma fila sintética com backbone mínimo: retomada idêntica, NaN, MLSD com `fold_dist_dk`, diagnóstico e nenhuma imagem nos resultados.
  - O 06, o 07 (E/S, MLSD e sensibilidade, com teste rápido e rodada) e o 08 rodaram de ponta a ponta num Drive falso, com GPU e ResNet simuladas.
  - Não foi testado: o desempenho real na L4 e a montagem do Drive.
