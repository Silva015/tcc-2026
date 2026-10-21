# Roteiro: como rodar o P3 e o P2 antes da dissertação (P4)

Passo a passo para o Arthur. Os detalhes técnicos estão em `NOTAS_P1_para_P2_P3.md` (seções 7–9). Aqui está só o que você faz.

## Palavras usadas aqui

- **Sessão**: uma conversa nova com o Claude, no app, na aba Code, com a pasta `tcc-2026` aberta. Cada parte (P2, P3, P4) é uma sessão separada.
- **Colab**: os notebooks que rodam na nuvem do Google, abertos a partir da pasta `tcc-final` do Drive.
- **L4**: o tipo de placa de vídeo (GPU) que usamos no Colab. Escolha em *Ambiente de execução → Alterar o tipo de ambiente de execução*.
- **Unidades (UC)**: o crédito do Colab Pro. A L4 gasta ~1,5 por hora.
- **Aba**: uma guia do navegador com um notebook do Colab aberto. Duas abas = dois treinos ao mesmo tempo.
- **Teste rápido**: uma rodada curta (poucos minutos) para ver se tudo funciona antes da rodada de verdade.

## Etapa 1: preparar o código (no computador, sem gastar unidades)

### 1.1. Sessão P3 (Wav-KAN)

Abra uma sessão nova e cole:

```
Você é a sessão P3 (Wav-KAN). Execute o plano docs/planos/P3_wavkan.md com os ajustes das seções 7, 8 e 9 de docs/planos/NOTAS_P1_para_P2_P3.md (os ajustes valem mais que o plano). Decisões já tomadas: W1 e W2 com as sementes 42, 43 e 44; W3, W4 e W5 só com a semente 42, rodados por último e só se sobrar saldo; Wav-KAN no MLSD: não. Trabalhe nesta pasta, na branch main, sem criar outra branch; adicione só os seus arquivos pelo caminho e não envie ao GitHub sem me perguntar. Não altere protocolo_comum.py, configs_mlsd.py nem os notebooks 00 a 05b. Suba seus arquivos para a pasta tcc-final do Drive. No fim, me dê um passo a passo simples, sem jargão, do que eu faço em cada notebook no Colab: o que mudar na célula de configuração, em qual aba e em que ordem.
```

- Se quiser a Wav-KAN também no MLSD (custa mais ~6 unidades), troque "Wav-KAN no MLSD: não" por "sim".
- A sessão vai escrever o código e os notebooks 06, 07 e 08, testar no computador e subir tudo para o Drive. Responda às perguntas dela.
- Espere ela dizer que terminou e que fez o commit.

### 1.2. Sessão P2 (multitarefa): só depois que a P3 terminar

Abra outra sessão nova e cole:

```
Você é a sessão P2 (multitarefa). Execute o plano docs/planos/P2_multitarefa.md com os ajustes das seções 7, 8 e 9 de docs/planos/NOTAS_P1_para_P2_P3.md (os ajustes valem mais que o plano). Decisões já tomadas: treinar só mt_kan e mt_mlp, com as sementes 42, 43 e 44; as redes de tarefa única vêm do P1. Trabalhe nesta pasta, na branch main, sem criar outra branch; adicione só os seus arquivos pelo caminho e não envie ao GitHub sem me perguntar. Não altere protocolo_comum.py, configs_mlsd.py, os notebooks 00 a 05b nem os arquivos da sessão P3. Suba seus arquivos para a pasta tcc-final do Drive. No fim, me dê um passo a passo simples, sem jargão, do que eu faço em cada notebook no Colab: o que mudar na célula de configuração, em qual aba e em que ordem.
```

As duas sessões usam a mesma pasta, então não rode as duas ao mesmo tempo.

## Etapa 2: rodar no Colab (gasta unidades)

### 2.0. Saldo
Abra `colab.research.google.com/signup` ou, num notebook, clique em *RAM/Disco → Ver recursos*.
- O mínimo é **~18 unidades**.
- Com tudo o que é opcional, **~32 unidades**.
- Se a L4 aparecer cinza, o saldo acabou.

### 2.1. Calibração (uma vez, ~15 min)
1. No Drive, em `tcc-final`, abra `01_calibracao_L4.ipynb`.
2. Escolha a **L4** e clique em **Executar tudo**. Não mude nada.
   - O que já foi medido em outubro é pulado; ele só mede as cabeças novas do P3 e testa a rede do P2.
3. Ao terminar: *Ambiente de execução → Desconectar e excluir ambiente de execução*.

### 2.2. Testes do P3 (~30 min)
- Rode o `06` e depois o `07` em modo de teste rápido.
- O que mudar em cada um vem no passo a passo final da sessão P3.

### 2.3. As rodadas de verdade (~5 h, duas abas)

| Aba | Notebook | Treina | Tempo |
|---|---|---|---|
| A | `07` do P3 | W1 e W2, 3 sementes | ~4,5 h |
| B | `09` do P2 | `mt_kan` e `mt_mlp`, 3 sementes | ~4,5 h |

1. **Aba A:** primeiro em teste rápido; depois no modo de verdade, **Executar tudo**.
2. Espere **1 minuto**.
3. **Aba B:** primeiro em teste rápido; depois no modo de verdade, **Executar tudo**.

**Cuidados (aprendidos no P1):**
- cada aba precisa de um notebook **diferente**; nunca abra o mesmo notebook em duas abas;
- se uma aba cair, reabra e clique em **Executar tudo**: ela continua de onde parou;
- quando as duas terminarem, **desconecte as duas**; aberta e parada, a sessão continua gastando unidades.

### 2.4. Opcionais (só se sobrar saldo)

| O quê | Como | Unidades |
|---|---|---|
| W3, W4 e W5 do P3 | `07` com a opção de sensibilidade, conforme a sessão P3 explicar | ~4 |
| Semente 44 do MLSD (P1) | Abra `04_mlsd_mesmo_protocolo.ipynb` e mude `ETAPA = "baselines_sementes_extras"`, `FILTRO_SEMENTES = [44]` e `TESTE_RAPIDO = False`. Clique em Executar tudo. Depois rode o `05b` com o tipo **CPU** | ~3 |

## Etapa 3: análises (sem gastar unidades)

1. Abra o `08` (P3) e o `10` (P2), escolha o tipo **CPU** e clique em **Executar tudo** em cada um.
2. Volte à conversa da **sessão P3** e escreva: `Terminei as rodadas e o 08 no Colab. Confira os resultados no Drive e traga-os para o repositório.`
3. Faça o mesmo na **sessão P2**, citando o `10`.

## Etapa 4: dissertação (P4)

Abra uma sessão nova e cole:

```
Você é a sessão P4. Execute o plano docs/planos/P4_reescrita_dissertacao.md. Os resultados das três sessões experimentais (P1: protocolo único, cabeças, backbones, MLSD e vazamento; P3: Wav-KAN; P2: multitarefa) já estão em resultados/, e o P2 e o P3 deixaram notas para você em docs/planos/NOTAS_P2_para_P4.md e docs/planos/NOTAS_P3_para_P4.md. Leia tudo o que a seção 0 do plano manda, comece pelas decisões da seção 1 e me pergunte as que ainda estão em aberto antes de escrever.
```
