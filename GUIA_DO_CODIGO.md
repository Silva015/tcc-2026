# Guia do Código — TransferKAN (TCC)

Este documento explica **o código** do TCC, do primeiro `import` até a última métrica impressa. Ele é o par prático do [`GUIA_DE_ESTUDO.md`](GUIA_DE_ESTUDO.md): lá está a teoria (o que é uma KAN, por que AUC importa), aqui está a implementação (em que célula, em que linha, e por quê).

O objetivo é que, ao terminar, você consiga responder a qualquer variação de **"me mostre no código onde você faz isso"** sem hesitar.

---

## Sumário

1. [Como usar este guia](#1-como-usar-este-guia)
2. [Mapa do repositório](#2-mapa-do-repositório)
3. [O pipeline em 7 etapas](#3-o-pipeline-em-7-etapas)
4. [Glossário de PyTorch e scikit-learn](#4-glossário-de-pytorch-e-scikit-learn)
5. [`TransferKAN_EntradaSaida.ipynb` — linha a linha](#5-transferkan_entradasaidaipynb--linha-a-linha)
6. [`TransferKAN_MLSD.ipynb` — só as diferenças](#6-transferkan_mlsdipynb--só-as-diferenças)
7. [As três arquiteturas de ablação](#7-as-três-arquiteturas-de-ablação)
8. [Do código para a monografia](#8-do-código-para-a-monografia)
9. [Perguntas de banca sobre o código](#9-perguntas-de-banca-sobre-o-código)
10. [Limitações reais e como respondê-las](#10-limitações-reais-e-como-respondê-las)
11. [Como rodar hoje](#11-como-rodar-hoje)

---

## 1. Como usar este guia

**Ordem sugerida de leitura:**

| Se você tem… | Leia |
|---|---|
| 30 minutos | Seções 2, 3 e 9 |
| 2 horas | Seções 2, 3, 4, 5 |
| Uma tarde inteira | Tudo, na ordem |

**A regra de ouro:** a Seção 5 é a que importa. Os outros 7 notebooks compartilham 70% do código com o `TransferKAN_EntradaSaida.ipynb`. Domine esse, e os demais viram "o mesmo, mas com X trocado por Y" — que é exatamente como as Seções 6 e 7 os apresentam.

---

## 2. Mapa do repositório

```
tcc-2026/
├── notebooks/          ← o código dos experimentos (8 notebooks)
├── tcc_dissertation/   ← a monografia em LaTeX (EN em tex/, PT em pt/tex/)
├── references/         ← o artigo de Lira et al. (2024), sua base de comparação
└── sibgrapi2026/       ← template do artigo curto (não usado ainda)
```

Os 8 notebooks são **4 arquiteturas × 2 tarefas**. As duas tarefas são:

- **Entrada/Saída (E/S):** binária — o ferimento é de entrada (0) ou de saída (1)?
- **MLSD:** três classes — a que distância o tiro foi disparado? Encostado (0), Curta (1), À distância (2).

| Arquitetura | Notebook E/S | Notebook MLSD | O que é | Papel |
|---|---|---|---|---|
| **TransferKAN** | `TransferKAN_EntradaSaida.ipynb` | `TransferKAN_MLSD.ipynb` | ResNet152 pré-treinada + cabeça KAN | **A proposta do TCC** |
| ResKANet-18 | `ConvKANWithResNet_EntradaSaida.ipynb` | `ConvKANWithResNet_MLSD.ipynb` | ResNet convolucional-KAN do zero | Ablação |
| PureConvKAN | `PureConvKAN_EntradaSaida.ipynb` | `PureConvKAN_MLSD.ipynb` | CNN rasa com convoluções KAN | Ablação |
| PureKAN | `PureKAN_EntradaSaida.ipynb` | `PureKAN_MLSD.ipynb` | KAN direto sobre pixels | Ablação |

E os hiperparâmetros **reais** de cada um — vale decorar a linha da TransferKAN:

| Notebook | Entrada | Lote | Validação | Épocas / Otimização |
|---|---|---|---|---|
| `TransferKAN_EntradaSaida` | 224×224 | 32 | **5-fold por caso** | 15 congelado (lr 1e-3) + 45 fine-tune (lr 1e-5) |
| `TransferKAN_MLSD` | 224×224 | 32 | **5-fold por caso** | 60 end-to-end (lr 1e-4) — **sem congelamento** |
| `ConvKANWithResNet_EntradaSaida` | 128×128 | 8 | hold-out 80/20 | 60 end-to-end (lr 1e-3) |
| `ConvKANWithResNet_MLSD` | 128×128 | 8 | hold-out 80/20 | 60, backbone congelado (lr 1e-3) |
| `PureConvKAN_*` | 128×128 | 8 | hold-out 80/20 | 60 end-to-end |
| `PureKAN_*` | 64×64 | 8 | hold-out 80/20 | 60 end-to-end |

> ⚠️ **Cuidado com uma pegadinha de nomenclatura.** O notebook `ConvKANWithResNet_EntradaSaida` **não** usa ResNet152 pré-treinada: ele usa a `reskanet_18x32p`, uma ResNet-18 cujas *convoluções* são KANs, treinada do zero. Já o `ConvKANWithResNet_MLSD` faz outra coisa completamente diferente (ResNet152 congelada + cabeça KAN), e o resultado dele **nunca foi reportado na monografia**. Por isso a monografia agora chama a linha da tabela de **`ResKANet-18`**, que é o nome correto do que gerou aquele número.

---

## 3. O pipeline em 7 etapas

Todo notebook segue o mesmo esqueleto. Guarde este mapa: ele responde "em que célula está X?" instantaneamente.

```
  [1] Instalar e importar          → célula 2   (!pip, !git clone, imports, device)
       ↓
  [2] Rotular as imagens           → célula 4   (class GunshotDataset)
       ↓
  [3] Separar por CASO             → célula 6   (case_to_indices, unique_cases)
       ↓
  [4] Definir as transformações    → célula 6   (transform_treino / transform_teste)
       ↓
  [5] Montar o modelo              → célula 8   (class TransferKAN_Gunshot)
       ↓
  [6] Treinar em 5 dobras          → célula 10  (for fold in kf.split(...))
       ↓
  [7] Agregar OOF e medir          → célula 10  (oof_y_true/pred/prob → métricas)
```

As etapas [3] e [4] moram na mesma célula, e as [6] e [7] também — por isso são 5 células de código para 7 etapas.

**Se lhe pedirem para desenhar o fluxo de dados:**

```
imagem .JPG (tamanho original, variável)
   → PIL.Image.open().convert('RGB')          # vira RGB
   → np.array()                                # vira matriz numpy
   → transform (resize 224 + augment + norm)   # vira tensor (3, 224, 224)
   → DataLoader empilha 32                     # vira tensor (32, 3, 224, 224)
   → ResNet152 sem a fc                        # vira tensor (32, 2048, 1, 1)
   → .view(batch, -1)                          # vira tensor (32, 2048)
   → Dropout(0.5)                              # mesma forma, metade zerada
   → KAN([2048, 128, 2])                       # vira tensor (32, 2)  ← logits
   → softmax                                   # vira tensor (32, 2)  ← probabilidades
   → argmax                                    # vira tensor (32,)    ← classe prevista
```

---

## 4. Glossário de PyTorch e scikit-learn

Só o necessário para ler o código. Se um destes termos aparecer numa pergunta e você travar, volte aqui.

| Termo | O que é, em uma frase |
|---|---|
| **`Dataset`** | Classe do PyTorch que representa "uma coleção de amostras". Você é obrigado a implementar `__len__` (quantas tem) e `__getitem__` (me dê a de índice *i*). |
| **`__getitem__`** | O método chamado quando você escreve `dataset[5]`. É onde a imagem é aberta do disco — **uma de cada vez**, sob demanda. |
| **`DataLoader`** | Envolve um `Dataset` e cuida de agrupar amostras em lotes (*batches*), embaralhar e paralelizar a leitura. |
| **`Subset`** | Uma "janela" sobre um `Dataset`: `Subset(ds, [3, 7, 9])` é um dataset com só aquelas 3 amostras. Não copia nada, só guarda os índices. |
| **`nn.Module`** | A classe-base de todo modelo/camada. Herdar dela é o que faz o PyTorch reconhecer seus parâmetros e saber treiná-los. |
| **`forward`** | O método que define o caminho dos dados pelo modelo. Você nunca o chama diretamente: escreve `model(x)`, e o PyTorch chama `forward(x)` por baixo. |
| **`.to(device)`** | Move tensores/modelos para a GPU (`cuda`) ou CPU. Modelo e dados precisam estar no **mesmo** device, ou o PyTorch dá erro. |
| **`optimizer.zero_grad()`** | Zera os gradientes acumulados. Sem isso, os gradientes de lotes anteriores se somam aos atuais e o treino vira lixo. |
| **`loss.backward()`** | Calcula, por retropropagação, o gradiente da perda em relação a cada parâmetro. |
| **`optimizer.step()`** | Aplica esses gradientes, atualizando os pesos. A tríade `zero_grad → backward → step` é o coração de todo treino em PyTorch. |
| **`model.train()`** | Liga o modo treino: o Dropout passa a zerar neurônios, a BatchNorm passa a atualizar estatísticas. |
| **`model.eval()`** | Desliga esses comportamentos. **Esquecer isso na avaliação é um bug clássico** — o Dropout continuaria ativo e as predições ficariam aleatórias. |
| **`torch.no_grad()`** | Bloco que desliga o rastreamento de gradientes. Na avaliação, economiza memória e tempo — não vamos treinar nada ali. |
| **`state_dict()`** | Um dicionário com todos os pesos do modelo. É o que se salva em `.pth` e se recarrega com `load_state_dict`. |
| **`softmax`** | Converte os *logits* (números crus, podem ser negativos) em probabilidades que somam 1. |
| **`torch.max(out, 1)`** | Devolve `(maior_valor, índice_do_maior)` por linha. O índice é a classe prevista — por isso o código escreve `_, predicted = torch.max(...)`, descartando o valor. |
| **`KFold`** | Do scikit-learn: divide uma lista em *k* partes e gera *k* combinações "treine em k−1, teste em 1". |
| **OOF (*out-of-fold*)** | As predições que cada dobra faz sobre os dados que ela **não** viu no treino. Juntando as 5, você tem uma predição honesta para *cada* amostra da base. |

---

## 5. `TransferKAN_EntradaSaida.ipynb` — linha a linha

Este é **o** notebook do TCC. Os números da linha "TransferKAN [Nossa]" da Tabela 5 da monografia saem daqui.

O notebook tem 11 células: as de índice 0, 1, 3, 5, 7 e 9 são texto (markdown), e as de índice **2, 4, 6, 8 e 10** são código. São essas cinco que vamos dissecar.

---

### Célula 2 — Ambiente, dados e imports

```python
!pip install git+https://github.com/Blealtan/efficient-kan -q
!pip install albumentations -q
```

A `!` no início manda a linha para o terminal do Colab, e não para o Python.

- **`efficient-kan`** é a implementação de KAN que o TCC usa. Não é a oficial (`pykan`, dos autores do artigo) — é uma reimplementação da comunidade, otimizada para rodar rápido em GPU. É de onde vem a classe `KAN` do modelo.
- **`albumentations`** é a biblioteca de *data augmentation*. Escolhida em vez do `torchvision.transforms` porque tem transformações que o torchvision não tem — em especial a `ElasticTransform`, que é a que simula a elasticidade da pele.
- **`-q`** é só "quiet", para não poluir a saída.

```python
!git clone https://gitlab.com/lisa-unb/leguwoi.git -q
!git clone https://github.com/pedrogarciafreitas/FDCPUnBGunshotDB.git -q
```

Baixa os dados. O segundo repositório é a **FDCPUnBGunshotDB**, a base de 2.551 imagens da PCDF — é dele que tudo depende. O primeiro (`leguwoi`) é o repositório de código do laboratório; na prática este notebook **não usa nada dele**, é uma herança do template. Se lhe perguntarem, diga isso: é honesto e mostra que você leu o código.

```python
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
from efficient_kan import KAN
import copy
```

Nada de especial, mas vale saber o papel de três deles:

- **`glob`** faz busca de arquivos por padrão (`*.JPG`). É o que evita listar as 2.551 imagens à mão.
- **`copy`** aparece depois em `copy.deepcopy(...)`, usado para congelar uma cópia dos melhores pesos — sem o `deepcopy`, você guardaria uma *referência* que continuaria mudando junto com o modelo.
- **`torchvision.models`** é de onde vem a ResNet152 já pré-treinada.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`device` é uma variável que vale `cuda` se houver GPU e `cpu` caso contrário. Todo `.to(device)` no resto do notebook usa isso. **Sem GPU, este notebook leva dias** — a ResNet152 tem 60 milhões de parâmetros e são 5 dobras × 60 épocas.

```python
def inspecionar_diretorio(caminho_base):
    for raiz, diretorios, arquivos in os.walk(caminho_base):
        ...
inspecionar_diretorio('./FDCPUnBGunshotDB')
```

Função puramente diagnóstica: percorre as pastas e imprime quantos arquivos há em cada uma. **Não afeta o resultado em nada** — serve para você conferir que o `git clone` funcionou e ver os nomes das pastas antes de escrever os caminhos da próxima célula. Se perguntarem, é um *sanity check* de ingestão.

---

### Célula 4 — A classe `GunshotDataset`

Esta célula é a **ponte** entre arquivos `.JPG` no disco e tensores que a rede consegue consumir.

```python
class GunshotDataset(Dataset):
```

Herda de `Dataset`, do PyTorch. Esse contrato exige implementar `__len__` e `__getitem__` — e, em troca, o `DataLoader` passa a saber trabalhar com a sua classe.

```python
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
```

O construtor guarda a pasta-raiz, a política de transformação, e cria **duas listas paralelas**: `image_paths[i]` é o caminho da imagem *i*, e `labels[i]` é o rótulo dela. O paralelismo entre as duas é a estrutura de dados central do notebook inteiro — o k-fold da célula 10 opera sobre *índices* dessas listas.

```python
        caminho_entradas = os.path.join(root_dir, 'database', 'ENTRADAS_EQX', '**', '*.JPG')
        for path in glob(caminho_entradas, recursive=True):
            self.image_paths.append(path)
            self.labels.append(0)
```

**Aqui está a rotulagem.** Não existe planilha nem CSV de rótulos: **a pasta é o rótulo**. Tudo que estiver dentro de `ENTRADAS_EQX` é ferimento de entrada e recebe a etiqueta `0`.

O `'**'` combinado com `recursive=True` diz "procure em qualquer subpasta, em qualquer profundidade". Isso é necessário porque as entradas são subdivididas em `ENCOSTADO/`, `CURTA/` e `DISTANCIA/` — e para esta tarefa binária essa subdivisão é irrelevante, todas são "entrada".

```python
        caminho_saidas = os.path.join(root_dir, 'database', 'SAIDAS_EQX', '*.JPG')
        for path in glob(caminho_saidas):
            self.image_paths.append(path)
            self.labels.append(1)
```

O mesmo para as saídas, rótulo `1`. Note que **não há `recursive=True`** aqui: as saídas estão todas soltas numa pasta só, sem subdivisão por distância — coerente com o fato de que ferimento de saída não carrega vestígio de distância.

> **Resultado esperado:** 1.883 imagens com rótulo 0 e 668 com rótulo 1, totalizando 2.551. É exatamente o que a monografia reporta na Seção 5.1. Se o número impresso for outro, o `git clone` falhou.

```python
    def __len__(self):
        return len(self.image_paths)
```

Quantas amostras há. O `DataLoader` usa isso para saber quando a época acabou.

```python
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
```

**O coração do carregamento sob demanda (*lazy loading*).** A imagem só é lida do disco **no instante em que é pedida**. Se o código carregasse as 2.551 imagens no `__init__`, estouraria a RAM do Colab.

O `.convert('RGB')` normaliza o formato: algumas fotos podem estar em escala de cinza (1 canal) ou com canal alfa (4 canais), e a ResNet exige exatamente 3 canais. Sem essa linha, uma única foto fora do padrão derruba o treino no meio.

```python
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented['image']

        return image, label
```

Aplica a transformação, **se houver uma**. Três detalhes:

1. `np.array(image)` converte de PIL para NumPy porque a Albumentations trabalha com matrizes NumPy (o torchvision trabalharia direto com PIL — é a principal diferença de interface entre as duas).
2. `self.transform(image=image_np)` — o argumento **tem que se chamar `image=`**; a Albumentations usa argumentos nomeados porque ela também sabe transformar máscaras e caixas delimitadoras junto.
3. O retorno é um dicionário; `augmented['image']` extrai a imagem transformada.

O retorno `image, label` é o que o `DataLoader` vai empilhar em lotes — daí virem sempre em pares no loop de treino: `for images, labels in trainloader`.

---

### Célula 6 — Transformações e separação por caso

A célula mais importante do ponto de vista **metodológico**. É aqui que mora o argumento anti-vazamento do TCC.

#### 6a. O pipeline de treino

```python
transform_treino = A.Compose([
    A.Resize(224, 224),
```
Padroniza tudo para 224×224 — o tamanho que a ResNet152 espera, por ter sido pré-treinada na ImageNet nessa resolução.

```python
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
```
Espelhamentos, cada um com 50% de chance. Justificativa forense: um ferimento não tem orientação canônica — a foto poderia ter sido tirada de cabeça para baixo. Ensinar isso ao modelo o impede de decorar orientações.

```python
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
```
Desloca até 5%, redimensiona até 10% e gira até 30°. Simula variação de ângulo e distância do fotógrafo.

```python
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
```
**CLAHE** = *Contrast Limited Adaptive Histogram Equalization*. Equalização de contraste feita por blocos (aqui, uma grade 8×8) em vez de na imagem inteira, com um teto (`clip_limit`) para não amplificar ruído. Realça detalhe de textura do tecido — que é justamente onde está a orla de escoriação.

```python
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
```
Deforma a imagem como se fosse borracha. **É a transformação com melhor justificativa no TCC**: pele humana é elástica, e o mesmo ferimento aparece com formatos ligeiramente diferentes conforme a posição do corpo.

```python
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
```
Ruído gaussiano — simula granulação de foto tirada com pouca luz, comum em cena de crime.

```python
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
```
Perturba brilho, contraste, saturação e matiz. Impede o modelo de se apoiar no balanço de cor específico de uma câmera. Note que o `hue` é bem mais conservador (0,05) que os demais (0,2) — mudar demais o matiz transformaria tecido em algo de outra cor, destruindo a informação clínica.

```python
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```
`Normalize` subtrai a média e divide pelo desvio-padrão, **canal a canal**. Esses seis números não são arbitrários: são as estatísticas da **ImageNet**. Usá-los é obrigatório ao empregar pesos pré-treinados — a ResNet foi treinada esperando entradas nessa escala. `ToTensorV2()` converte a matriz NumPy `(altura, largura, canais)` no tensor PyTorch `(canais, altura, largura)`.

> **A ordem importa.** `Normalize` vem depois de todas as transformações visuais e antes do `ToTensorV2`. Se você normalizasse antes do CLAHE, por exemplo, o CLAHE operaria sobre valores negativos e o resultado não faria sentido.

#### 6b. O pipeline de teste

```python
transform_teste = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**Só redimensiona e normaliza. Nenhuma aleatoriedade.** Esta é uma das perguntas de banca mais prováveis, e a resposta é: *augmentation* no teste tornaria a métrica não determinística e artificialmente mais difícil ou mais fácil, dependendo do sorteio. A avaliação tem que medir o modelo, não a loteria.

#### 6c. Sementes

```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

Fixa os geradores pseudoaleatórios para que a partição e a ordem dos lotes sejam reprodutíveis. **Note o que *não* está aqui: `np.random.seed()`.** A Albumentations sorteia com NumPy, então o *augmentation* especificamente não é reprodutível bit a bit. Isso está honestamente registrado na Seção 5.3.2 da monografia — e é melhor você mesmo apontar isso do que ser pego.

#### 6d. O truque das duas instâncias

```python
dataset_base_treino = GunshotDataset(root_dir='./FDCPUnBGunshotDB', transform=transform_treino)
dataset_base_teste  = GunshotDataset(root_dir='./FDCPUnBGunshotDB', transform=transform_teste)
```

**Duas instâncias sobre exatamente os mesmos arquivos, com políticas de transformação diferentes.**

Por que não uma só? Porque a transformação é fixada na construção do objeto. Se houvesse um único dataset, não haveria como fazer as mesmas imagens serem aumentadas no treino e determinísticas no teste — a menos que você trocasse `dataset.transform` no meio do loop, o que é frágil e fácil de errar.

Com duas instâncias, os subconjuntos são extraídos por índice de uma ou de outra, e a garantia vira **estrutural**: é impossível uma imagem de teste receber *augmentation*. Se perguntarem "como você garante que não aplicou augmentation no teste?", esta é a resposta, e ela é forte.

#### 6e. O agrupamento por caso — **o argumento central do TCC**

```python
case_to_indices = {}
for idx, path in enumerate(dataset_base_treino.image_paths):
    filename = os.path.basename(path)
    case_id = filename[:9]
    if case_id not in case_to_indices:
        case_to_indices[case_id] = []
    case_to_indices[case_id].append(idx)
```

Leia com calma, porque é o trecho que você mais vai precisar explicar.

- `os.path.basename(path)` descarta as pastas e fica só com o nome do arquivo.
- **`filename[:9]`** — pega os **9 primeiros caracteres** do nome. Esse prefixo é o **identificador do caso** (a vítima/laudo). Duas fotos do mesmo ferimento têm o mesmo prefixo.
- O dicionário resultante mapeia `"123456789" → [12, 13, 14, 87]`, ou seja, **caso → lista de índices das imagens daquele caso**.

**Por que isso é o coração do trabalho:** uma mesma vítima tem várias fotos do mesmo ferimento, de ângulos ligeiramente diferentes. Se você dividisse as *imagens* ao acaso, a foto A do caso X cairia no treino e a foto B do mesmo caso X no teste. O modelo veria, na prática, uma imagem quase idêntica a uma que ele decorou — e a métrica subiria sem que ele tivesse aprendido nada generalizável. Isso é **vazamento semântico**.

A solução do código é sortear **casos**, nunca imagens. Todas as fotos de um caso vão juntas para o mesmo lado.

```python
unique_cases = list(case_to_indices.keys())
print(f"📊 Total de Casos Únicos ENTRADA/SAÍDA para o K-Fold: {len(unique_cases)}")
```

`unique_cases` é a lista sobre a qual o `KFold` vai operar na célula 10. Guarde este nome: quando perguntarem "onde exatamente você evita o vazamento?", a resposta é *"o `KFold` recebe `unique_cases`, não a lista de imagens"*.

#### 6f. A partição vestigial (não se assuste ao ver isto)

```python
cases_treino, cases_teste = train_test_split(unique_cases, test_size=0.2, random_state=RANDOM_SEED)
...
trainloader = DataLoader(dataset_treino, batch_size=32, shuffle=True)
testloader  = DataLoader(dataset_teste, batch_size=32, shuffle=False)
```

Aqui o notebook faz uma partição simples 80/20 por caso e monta os *loaders*.

> 🔍 **Ponto de atenção — e você precisa saber disto antes da banca.** Este bloco é **código morto**. A célula 10 refaz a partição dentro do laço de dobras e **reatribui** `cases_treino`, `cases_teste`, `dataset_treino`, `dataset_teste`, `trainloader` e `testloader`. Nenhum valor calculado aqui sobrevive à primeira iteração do k-fold.
>
> É resíduo de uma versão anterior do experimento, que usava *hold-out* antes de migrar para validação cruzada. **Não afeta nenhum resultado** — mas se o professor apontar, a resposta correta é: *"é uma partição legada, sobrescrita pelo laço k-fold da célula seguinte; os resultados reportados vêm exclusivamente das 5 dobras."* Reconhecer isso vale muito mais que tentar justificar.

Uma coisa deste bloco, porém, **é** útil: `shuffle=True` no treino e `shuffle=False` no teste. Embaralhar o treino evita que o modelo aprenda a ordem dos dados; no teste seria inútil e só dificultaria a depuração.

#### 6g. O *sanity check*

```python
train_cases_check = set([os.path.basename(dataset_base_treino.image_paths[i])[:9] for i in indices_treino])
test_cases_check  = set([os.path.basename(dataset_base_teste.image_paths[i])[:9] for i in indices_teste])
intersecao = train_cases_check.intersection(test_cases_check)

if len(intersecao) == 0:
    print("\n✅ SUCESSO! Zero interseção semântica identificada pelo código.")
else:
    print(f"\n❌ ALERTA DE VAZAMENTO! {len(intersecao)} casos vazados: {intersecao}")
```

Recalcula os IDs de caso a partir dos índices efetivamente usados, transforma em conjuntos e verifica se a interseção é vazia.

**Por que isso vale ouro numa defesa:** você não *afirma* que não há vazamento — você **prova programaticamente**, e a prova roda junto com o experimento. Se alguém perguntar "como você tem certeza?", a resposta é "o código verifica, e a saída está no notebook".

---

### Célula 8 — A arquitetura `TransferKAN_Gunshot`

```python
class TransferKAN_Gunshot(nn.Module):
    def __init__(self):
        super(TransferKAN_Gunshot, self).__init__()
```

Herda de `nn.Module` e chama o construtor da classe-mãe — obrigatório, é o que inicializa a máquina interna de registro de parâmetros do PyTorch.

```python
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
```

Baixa a ResNet152 **com os pesos treinados na ImageNet**. É aqui que o *transfer learning* acontece: em vez de aprender do zero o que é uma borda ou uma textura, o modelo já chega sabendo — aprendido em 1,2 milhão de imagens. Essencial, porque a base forense tem só 2.551.

```python
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flat_size = 2048
```

**A linha mais citada em bancas.** Desmonte-a da direita para a esquerda:

1. `resnet.children()` devolve as camadas de primeiro nível da ResNet, em ordem.
2. `list(...)` materializa isso numa lista.
3. `[:-1]` **descarta a última** — que é a `fc`, a camada densa que classificava nas 1.000 classes da ImageNet. Ela não nos serve: queremos 2 classes, não 1.000.
4. `nn.Sequential(*...)` remonta as camadas restantes numa rede que aplica uma após a outra. O `*` desempacota a lista em argumentos posicionais.

O que sobra é o **extrator de características** (*backbone*). Ele termina no *average pooling* global, e por isso sua saída tem 2048 valores — que é a largura do último bloco convolucional da ResNet152. **"Por que 2048?"** Porque é isso que a arquitetura da ResNet-152 produz; não foi uma escolha sua, foi uma consequência.

```python
        self.dropout = nn.Dropout(p=0.5)
```

Durante o treino, zera aleatoriamente 50% dos 2048 valores a cada passo. Força a rede a não depender de nenhuma característica isolada, distribuindo a decisão. `p=0.5` é agressivo — justificado pela base pequena e pelo risco alto de sobreajuste.

```python
        self.kan = KAN([self.flat_size, 128, 2], grid_size=10, spline_order=3)
```

**A inovação do TCC.** Onde uma arquitetura tradicional colocaria `nn.Linear(2048, 2)` (ou um MLP com ReLU no meio), aqui entra uma KAN.

- **`[2048, 128, 2]`** é a topologia: 2048 entradas → 128 neurônios ocultos → 2 saídas (entrada/saída).
- **`spline_order=3`** — as funções aprendidas nas arestas são splines **cúbicas**. Grau 3 é o padrão consagrado: suave o bastante para ser derivável duas vezes, flexível o bastante para curvar.
- **`grid_size=10`** — cada spline é definida sobre uma grade de 10 intervalos, em vez dos 5 padrão. **Mais intervalos = mais pontos de controle = curvas mais expressivas**, ao custo de mais parâmetros. Foi dobrado justamente porque as fronteiras entre entrada e saída são visualmente sutis.

> **Se perguntarem "onde a KAN substitui a MLP?"** — é literalmente esta linha. Trocar `self.kan = KAN([2048, 128, 2], ...)` por `self.kan = nn.Sequential(nn.Linear(2048,128), nn.ReLU(), nn.Linear(128,2))` daria a arquitetura de referência tradicional. Todo o resto do notebook ficaria idêntico.

```python
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.kan(x)
        return x
```

O caminho dos dados, em 4 linhas:

1. `self.features(x)` — a imagem `(32, 3, 224, 224)` vira `(32, 2048, 1, 1)`.
2. `x.view(x.size(0), -1)` — **achata**. `x.size(0)` é o tamanho do lote (32), e `-1` diz ao PyTorch "calcule esta dimensão sozinho". Resultado: `(32, 2048)`. Sem isso, a KAN receberia um tensor 4-D e quebraria.
3. `self.dropout(x)` — zera metade dos valores (só no modo `train()`).
4. `self.kan(x)` — classifica, devolvendo `(32, 2)`.

O retorno são **logits**, não probabilidades: não há softmax aqui. Isso é intencional — a `nn.CrossEntropyLoss` do PyTorch **já aplica o softmax internamente**. Aplicá-lo duas vezes seria um bug silencioso (o treino ainda funcionaria, mas pior).

```python
model_gunshot = TransferKAN_Gunshot().to(device)
```

Instancia e move para a GPU. (Esta instância específica é descartada: a célula 10 recria o modelo dentro de cada dobra.)

---

### Célula 10 — Validação cruzada, treino em duas fases e métricas

A célula mais longa. Vamos por blocos.

#### 10a. Configuração

```python
K_FOLDS = 5
epochs_fase1 = 15
epochs_fase2 = 45
total_epochs = epochs_fase1 + epochs_fase2   # = 60

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
```

5 dobras, 60 épocas por dobra — logo, **300 épocas de treino no total**. `shuffle=True` embaralha os casos antes de cortá-los em 5 partes (sem isso, a divisão seguiria a ordem alfabética dos arquivos, que pode ter estrutura escondida).

```python
oof_y_true = []
oof_y_pred = []
oof_y_prob_both = []
```

As três listas que vão acumular, ao longo das 5 dobras, respectivamente: o rótulo verdadeiro, a classe prevista e o vetor de probabilidades. **É daqui que sai toda a Tabela 5.**

#### 10b. O laço das dobras

```python
for fold, (train_case_idx, test_case_idx) in enumerate(kf.split(unique_cases)):
```

`kf.split(unique_cases)` gera 5 pares de índices. **Note o argumento: `unique_cases`, não `image_paths`.** Esta é a linha exata para apontar quando perguntarem onde o vazamento é evitado.

```python
    cases_treino = [unique_cases[i] for i in train_case_idx]
    cases_teste = [unique_cases[i] for i in test_case_idx]

    indices_treino = []
    for case in cases_treino: indices_treino.extend(case_to_indices[case])
    indices_teste = []
    for case in cases_teste: indices_teste.extend(case_to_indices[case])
```

Traduz "índices de casos" → "IDs de casos" → "índices de imagens". O `extend` (e não `append`) é o que **desdobra** cada caso em suas múltiplas fotos.

> É por causa deste passo que as dobras **não** têm o mesmo número de imagens: cada dobra tem 1/5 dos *casos*, mas casos têm quantidades diferentes de fotos.

```python
    dataset_treino = Subset(dataset_base_treino, indices_treino)
    dataset_teste  = Subset(dataset_base_teste, indices_teste)

    trainloader = DataLoader(dataset_treino, batch_size=32, shuffle=True)
    testloader  = DataLoader(dataset_teste, batch_size=32, shuffle=False)
```

Aqui as duas instâncias da célula 6 se pagam: o treino sai da instância **com** *augmentation*, o teste da instância **sem**.

#### 10c. Pesos de classe

```python
    labels_treino = [dataset_base_treino.labels[i] for i in indices_treino]
    peso_entrada = len(labels_treino) / (2.0 * labels_treino.count(0))
    peso_saida = len(labels_treino) / (2.0 * labels_treino.count(1))
    pesos_classes = torch.tensor([peso_entrada, peso_saida], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos_classes)
```

A fórmula é `N / (n_classes × n_da_classe)` — **frequência inversa**. Com 1.883 entradas e 668 saídas, o peso da classe minoritária (saída) fica cerca de 2,8× maior que o da majoritária.

**Efeito prático:** errar uma "saída" dói quase três vezes mais que errar uma "entrada". Sem isso, o modelo aprenderia rapidamente que chutar "entrada" sempre garante 74% de acerto — e pararia por aí.

**Detalhe crucial:** os pesos são recalculados **dentro** do laço, a partir só do treino daquela dobra. Calculá-los sobre a base inteira seria uma forma sutil de vazamento (a distribuição do teste influenciaria o treino).

#### 10d. Reinicialização por dobra

```python
    model_gunshot = TransferKAN_Gunshot().to(device)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model_gunshot.state_dict())
```

**O modelo é reconstruído do zero a cada dobra.** Se ele fosse reaproveitado, a dobra 2 começaria já tendo visto os dados de teste da dobra 1 — e a validação cruzada perderia todo o sentido. Esta é uma pergunta de banca clássica: *"por que você reinstancia o modelo dentro do laço?"*

#### 10e. Congelamento e descongelamento

```python
    for epoch in range(total_epochs):
        if epoch == 0:
            for param in model_gunshot.features.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(model_gunshot.kan.parameters(), lr=1e-3, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_fase1)
```

**Fase 1 (épocas 1–15).**

- `requires_grad = False` no `features` **congela o backbone**: ele continua processando as imagens, mas seus pesos não são atualizados.
- O otimizador recebe **só** `model_gunshot.kan.parameters()` — literalmente não tem acesso aos pesos da ResNet.
- `lr=1e-3` é uma taxa alta, apropriada para pesos aleatórios (as splines da KAN começam do zero).

**Por que congelar?** No início, a KAN está aleatória e produz gradientes enormes e desorganizados. Se o backbone estivesse solto, esses gradientes destruiriam representações que levaram semanas de GPU para serem aprendidas na ImageNet. Congelar dá tempo à KAN de "se acomodar" antes de mexer no que já é bom.

```python
        elif epoch == epochs_fase1:
            for param in model_gunshot.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model_gunshot.parameters(), lr=1e-5, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_fase2)
```

**Fase 2 (épocas 16–60).** Tudo é descongelado e o otimizador passa a receber `model_gunshot.parameters()` — a rede inteira.

`lr=1e-5` é **cem vezes menor** que a da fase 1. É o ponto central do *fine-tuning*: ajustes minúsculos, para adaptar os filtros ao domínio forense sem apagar o que eles já sabiam.

`CosineAnnealingLR` faz a taxa decair suavemente seguindo meia onda de cosseno ao longo de `T_max` épocas — passos largos no começo, milimétricos no fim.

#### 10f. O laço de treino

```python
        model_gunshot.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_gunshot(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

O ciclo canônico do PyTorch, e você deve saber recitá-lo:

1. `zero_grad()` — limpa gradientes do lote anterior.
2. `model(images)` — passagem para frente, produz os logits.
3. `criterion(outputs, labels)` — calcula a perda (com os pesos de classe embutidos).
4. `backward()` — retropropaga, calculando o gradiente de cada parâmetro.
5. `step()` — atualiza os pesos.

```python
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
```

Contabiliza acertos. `torch.max(outputs, 1)` devolve `(valor, índice)` por linha; só o índice interessa, daí o `_`. `(predicted == labels)` produz um tensor de booleanos, `.sum()` conta os `True`, `.item()` extrai o número Python.

#### 10g. O laço de avaliação

```python
        model_gunshot.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                ...
        acc_test = 100 * correct_test / total_test
        scheduler.step()
```

`eval()` desliga o Dropout; `no_grad()` desliga o rastreamento de gradientes. **`scheduler.step()` fica fora do laço de lotes**, sendo chamado uma vez por época — se estivesse dentro, a taxa despencaria 60× mais rápido que o previsto.

#### 10h. O checkpoint

```python
        if acc_test > best_acc:
            best_acc = acc_test
            best_model_wts = copy.deepcopy(model_gunshot.state_dict())
            torch.save(best_model_wts, f'melhor_modelo_ent_sai_fold{fold+1}.pth')
```

Guarda os pesos sempre que a acurácia de teste bate o recorde. O `deepcopy` é indispensável: `state_dict()` devolve referências aos tensores vivos, que continuariam mudando junto com o treino.

> ⚠️ **Esta é a fragilidade metodológica real do trabalho, e você precisa conhecê-la.** A época é escolhida olhando o conjunto de teste — e é esse mesmo conjunto que depois gera as métricas OOF. Isso é **seleção sobre o teste**, e torna a acurácia reportada otimista. O certo seria separar, dentro do treino de cada dobra, uma partição interna de validação para escolher a época.
>
> A monografia agora declara isso abertamente na Seção 5.3.6 ("Ameaças à Validade"). Veja a Seção 10 deste guia para a resposta completa a essa pergunta.

#### 10i. Coleta das predições OOF

```python
    model_gunshot.load_state_dict(best_model_wts)
    model_gunshot.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model_gunshot(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            oof_y_true.extend(labels.cpu().numpy())
            oof_y_pred.extend(predicted.cpu().numpy())
            oof_y_prob_both.extend(probabilities.cpu().numpy())
```

Terminada a dobra, recarrega os melhores pesos e passa uma última vez pelo teste — agora guardando tudo.

- **`F.softmax(outputs, dim=1)`** — aqui, sim, aplica-se o softmax. Os logits viram probabilidades que somam 1 por amostra. **Isso é indispensável para a AUC**, que precisa de escores contínuos, não de decisões binárias.
- `dim=1` significa "normalize ao longo das classes" (cada linha soma 1), não ao longo do lote.
- `.cpu().numpy()` traz da GPU para a CPU e converte para NumPy, porque o scikit-learn não entende tensores CUDA.

Ao fim das 5 dobras, as três listas contêm **uma predição para cada uma das 2.551 imagens**, sempre feita por um modelo que nunca viu aquele caso no treino. Isso é a predição *out-of-fold*.

#### 10j. As métricas

```python
oof_y_true_onehot = np.eye(2)[oof_y_true]
```

Truque idiomático de NumPy: `np.eye(2)` é a matriz identidade 2×2, e indexá-la com o vetor de rótulos produz a codificação *one-hot* (`0 → [1,0]`, `1 → [0,1]`). O `roc_auc_score` precisa desse formato para calcular médias macro/micro.

```python
cm = confusion_matrix(oof_y_true, oof_y_pred)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
```

Extrai os quatro contadores da matriz de confusão, **por classe**:

- `np.diag(cm)` — a diagonal são os acertos → **TP** (verdadeiros positivos).
- `cm.sum(axis=0)` soma as **colunas** (tudo que foi *previsto* como aquela classe); menos os acertos → **FP**.
- `cm.sum(axis=1)` soma as **linhas** (tudo que *era* aquela classe); menos os acertos → **FN**.
- O que sobra do total → **TN**.

```python
for avg in ['macro', 'micro', 'weighted']:
    metrics[f'Prec_{avg}'] = precision_score(oof_y_true, oof_y_pred, average=avg, zero_division=0)
    ...
    metrics[f'AUC_{avg}'] = roc_auc_score(oof_y_true_onehot, oof_y_prob_both, average=avg)
```

Calcula precisão, revocação, F1 e AUC nas três médias:

- **macro** — média simples entre classes. **Trata a classe rara como igualmente importante** → é a que denuncia mau desempenho nas minoritárias.
- **micro** — agrega todas as amostras antes de calcular. Dominada pela classe majoritária.
- **weighted** — média ponderada pelo número de amostras de cada classe.

`zero_division=0` evita erro caso alguma classe não receba nenhuma predição (aconteceria divisão por zero).

```python
specificity_per_class = TN / (TN + FP)
metrics['Spec_macro'] = np.mean(specificity_per_class)
metrics['Spec_micro'] = TN.sum() / (TN.sum() + FP.sum())
metrics['Spec_weighted'] = np.average(specificity_per_class, weights=support_per_class)
```

A **especificidade** (taxa de verdadeiros negativos) não existe pronta no scikit-learn, por isso é calculada à mão a partir dos contadores. Note como as três médias são construídas de forma diferente: macro é média simples, micro soma antes de dividir, weighted usa o suporte como peso.

O resto da célula só imprime a tabela formatada — é a fonte literal da Tabela 5 da monografia.

---

## 6. `TransferKAN_MLSD.ipynb` — só as diferenças

Este notebook faz a segunda tarefa: dadas **apenas as imagens de entrada**, dizer a que distância o tiro foi disparado. Ele é 80% igual ao anterior. Só o que muda importa.

### 6.1. A classe do dataset muda de nome e de lógica

```python
class GunshotDistanceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        ...
        caminho_encostado = os.path.join(root_dir, 'database', 'ENTRADAS_EQX', 'ENCOSTADO', '*.JPG')
        for path in glob(caminho_encostado):
            self.image_paths.append(path); self.labels.append(0)

        caminho_curta = os.path.join(root_dir, 'database', 'ENTRADAS_EQX', 'CURTA', '*.JPG')
        ... self.labels.append(1)

        caminho_distancia = os.path.join(root_dir, 'database', 'ENTRADAS_EQX', 'DISTANCIA', '*.JPG')
        ... self.labels.append(2)
```

Agora as **subpastas** de `ENTRADAS_EQX` viram os rótulos — exatamente as subpastas que o notebook anterior ignorava com `'**'`. E a pasta `SAIDAS_EQX` some inteiramente: ferimento de saída não tem vestígio de distância, então não entra nesta tarefa.

O desbalanceamento aqui é brutal: **1.609 / 232 / 42**. A classe `ENCOSTADO` tem 42 imagens — menos de 3% da base. Esse número explica quase todos os resultados fracos da tarefa.

### 6.2. A KAN tem 3 saídas e grade padrão

```python
self.kan = KAN([self.flat_size, 128, 3])
```

Duas mudanças em relação ao notebook E/S:

- `3` em vez de `2` na saída — três classes de distância.
- **Não há `grid_size=10`.** Sem o argumento, a `efficient-kan` usa o padrão, que é **5**. A monografia registra isso explicitamente na Seção 5.2.2: com apenas 42 imagens na classe mais rara, uma grade mais densa criaria mais coeficientes de spline do que há amostras para estimá-los.

### 6.3. **Não há treino em duas fases**

```python
model_mlsd = TransferKAN_MLSD().to(device)
optimizer = torch.optim.AdamW(model_mlsd.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

Compare com a célula 10e do notebook E/S: **não existe `requires_grad = False` em lugar nenhum**. O otimizador recebe `model_mlsd.parameters()` — a rede inteira — desde a época 0, com uma taxa única de `1e-4`.

Ou seja: **a tarefa MLSD treina de ponta a ponta, sem congelamento, com uma taxa intermediária** entre as duas usadas na tarefa E/S (`1e-3` e `1e-5`).

> Esta era uma das divergências entre o código e o texto. A monografia foi corrigida: a Seção 5.3.4 agora tem duas subseções separadas, uma para cada regime, com a justificativa técnica — as texturas finas que distinguem as distâncias (fuligem, tatuagem) vivem nas camadas convolucionais intermediárias e não estão bem representadas nas características genéricas da ImageNet, de modo que congelá-las privaria o classificador justamente do sinal da tarefa.

### 6.4. Os pesos de classe usam o scikit-learn

```python
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
```

Em vez da fórmula manual do outro notebook, usa a função pronta. **É a mesma matemática**: `compute_class_weight('balanced')` calcula exatamente `n_amostras / (n_classes × contagem_da_classe)`. Se perguntarem por que os dois notebooks fazem diferente, a resposta honesta é: são duas formas de escrever a mesma coisa.

### 6.5. A AUC multiclasse

```python
oof_y_true_bin = label_binarize(oof_y_true, classes=[0, 1, 2])
...
metrics[f'AUC_{avg}'] = roc_auc_score(oof_y_true_bin, oof_y_prob_both, multi_class='ovr', average=avg)
```

Duas diferenças em relação ao caso binário:

- **`label_binarize`** substitui o truque do `np.eye(2)`. Faz a mesma coisa (codificação *one-hot*), mas para 3 classes.
- **`multi_class='ovr'`** = *one-vs-rest*. A AUC é definida para problemas binários; com 3 classes, o scikit-learn calcula três curvas ROC — "Encostado contra o resto", "Curta contra o resto", "À distância contra o resto" — e depois as combina segundo o `average`.

**Essa é a resposta para "como você calculou a AUC para multiclasse?"**: uma-contra-todas, com média macro, micro e ponderada.

```python
print(classification_report(oof_y_true, oof_y_pred, target_names=classes, zero_division=0))
```

Só o notebook MLSD imprime o relatório por classe. **Vale rodar e olhar**: é ali que se vê, número a número, o desempenho na classe `Encostado` — e ele é fraco. Melhor você conhecer esse número antes da banca.

---

## 7. As três arquiteturas de ablação

Um **estudo de ablação** responde: "cada peça da minha arquitetura é realmente necessária?". A resposta se dá removendo ou trocando peças e vendo o que acontece.

A TransferKAN tem duas peças: **(a)** um extrator de características profundo e pré-treinado, e **(b)** um classificador KAN. As três referências abaixo removem ou enfraquecem (a), mantendo alguma forma de (b).

### 7.1. PureKAN — remove todo o estágio convolucional

```python
class PureKAN_Gunshot(nn.Module):
    def __init__(self, img_size=64):
        self.flat_size = 3 * img_size * img_size     # 3 × 64 × 64 = 12.288
        self.dropout = nn.Dropout(p=0.5)
        self.kan = KAN([self.flat_size, 256, 64, 2], grid_size=5, spline_order=3)

    def forward(self, x):
        x = x.view(x.size(0), -1)     # achata a imagem inteira em um vetor
        x = self.dropout(x)
        x = self.kan(x)
        return x
```

**A linha de base mais austera.** A imagem é achatada em um vetor de 12.288 pixels e entregue direto à KAN. Não há convolução nenhuma.

- **O que testa:** a KAN, sozinha, consegue classificar imagens?
- **Por que 64×64:** um vetor de 3×224×224 = 150.528 entradas tornaria a primeira camada da KAN gigantesca. Reduzir para 64×64 é o que torna o experimento executável.
- **Resultado:** 0,692 em E/S e 0,712 em MLSD — com AUC macro de **0,463** em MLSD, ou seja, **abaixo do acaso**. A rede simplesmente não extraiu sinal sobre as classes raras.
- **O que isso prova:** achatar pixels destrói a estrutura espacial. Uma KAN não tem noção de vizinhança — para ela, o pixel (0,0) e o pixel (63,63) são apenas as entradas nº 1 e nº 12.288, sem qualquer relação. Convolução existe justamente para capturar essa relação.

### 7.2. PureConvKAN — convolução, mas rasa e do zero

```python
class CustomPureConvKAN_Gunshot(nn.Module):
    def __init__(self):
        self.conv1 = ConvKAN(in_channels=3,  out_channels=16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1   = nn.BatchNorm2d(16)
        # ... blocos 2 (16→32) e 3 (32→64), idênticos em estrutura ...
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64, 2)
```

Uma CNN clássica de três blocos, mas com uma diferença: as convoluções são **`ConvKAN`** — filtros convolucionais cujas ativações são splines aprendíveis, em vez de multiplicação seguida de ReLU.

Cada bloco é `ConvKAN → MaxPool → BatchNorm`, com os canais crescendo `3 → 16 → 32 → 64` — o padrão de toda CNN: quanto mais fundo, mais canais e menos resolução.

```python
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
```

*Global average pooling*: reduz cada canal a um único número, seja qual for a resolução que chegou ali. É um truque útil — dispensa calcular à mão o tamanho do achatamento (o comentário no código diz exatamente isso).

> **Ironia que vale notar:** o classificador final desta arquitetura "Pure**KAN**" é um `nn.Linear` comum, não uma KAN. Ou seja, ela tem convoluções KAN e cabeça linear — o **oposto** da TransferKAN, que tem convoluções comuns e cabeça KAN. Se o professor notar, essa é a leitura correta.

- **Resultado:** 0,707 em E/S. Melhor que a PureKAN, pior que tudo mais.
- **O que isso prova:** convolução ajuda (0,692 → 0,707), mas três blocos treinados do zero em 2.551 imagens não chegam nem perto de um extrator treinado em 1,2 milhão.

### 7.3. ResKANet-18 — profunda e residual, mas ainda do zero

```python
from models.reskanet import reskanet_18x32p

class ConvKANWithResNet_Gunshot(nn.Module):
    def __init__(self):
        self.backbone = reskanet_18x32p(input_channels=3, num_classes=2)

    def forward(self, x):
        return self.backbone(x)
```

Uma ResNet-18 inteira na qual **todas** as convoluções são KANs, vinda da biblioteca `torch-conv-kan`. Tem conexões residuais, profundidade real — tudo, **menos o pré-treinamento**.

- **Resultado:** 0,822 em E/S — o melhor entre as ablações.
- **O que isso prova:** é a peça decisiva do argumento. Esta arquitetura tem profundidade e resíduos, e ainda assim fica abaixo da TransferKAN. **A diferença entre elas é o pré-treinamento na ImageNet.** Ou seja: o ganho não vem de "usar KAN em todo lugar", vem de **combinar** características pré-treinadas com uma cabeça KAN.

> ⚠️ **Este é o notebook com nome enganoso.** Chama-se `ConvKANWithResNet`, mas não usa a ResNet152 pré-treinada em lugar algum. Por isso a monografia agora nomeia a linha da tabela como **`ResKANet-18 [do zero]`**. Já o `ConvKANWithResNet_MLSD.ipynb` faz outra coisa — ResNet152 congelada + `KAN([2048, 64, 3])` — e o resultado dele nunca foi reportado na monografia. Se perguntarem, é um experimento executado mas não incluído.

### 7.4. A leitura conjunta

| Modelo | Extrator | Pré-treinado? | ACC (E/S) |
|---|---|---|---|
| PureKAN | nenhum | — | 0,692 |
| PureConvKAN | 3 blocos ConvKAN | não | 0,707 |
| ResKANet-18 | ResNet-18 KAN completa | **não** | 0,822 |
| **TransferKAN** | **ResNet152** | **sim** | **0,860** |

A progressão é monotônica e conta uma história limpa: **quanto mais o modelo se aproxima de um extrator profundo e pré-treinado, melhor ele fica.** A cabeça KAN sozinha não substitui a extração de características — é a combinação das duas coisas que funciona.

> **Ressalva que você deve fazer antes que a façam por você:** as três ablações rodam em resolução menor (128 ou 64 px), lote menor (8) e sob *hold-out* simples, enquanto a TransferKAN roda em 224 px, lote 32 e sob 5-fold. **A comparação não é isométrica.** A tendência é clara e consistente, mas não é uma ablação controlada. A monografia registra isso na Seção 5.3.6, e a Seção 10 deste guia traz a resposta pronta.

---

## 8. Do código para a monografia

Tabela de rastreabilidade. Use-a para responder a qualquer "me mostre onde".

| A monografia afirma… | Notebook | Célula | O trecho exato |
|---|---|---|---|
| ResNet152 pré-treinada como *backbone* (§5.2.1) | `TransferKAN_EntradaSaida` | 8 | `models.resnet152(weights=...DEFAULT)` |
| Camada `fc` final removida, vetor de 2048 (§5.2.1) | `TransferKAN_EntradaSaida` | 8 | `nn.Sequential(*list(resnet.children())[:-1])` |
| Dropout $p=0{,}5$ (§5.2.1) | `TransferKAN_EntradaSaida` | 8 | `nn.Dropout(p=0.5)` |
| KAN `[2048, 128, 2]`, grade 10, ordem 3 (§5.2.2) | `TransferKAN_EntradaSaida` | 8 | `KAN([2048, 128, 2], grid_size=10, spline_order=3)` |
| KAN `[2048, 128, 3]`, grade padrão 5 (§5.2.2) | `TransferKAN_MLSD` | 5 | `KAN([self.flat_size, 128, 3])` |
| Ampliação só no treino (§5.3.1) | `TransferKAN_EntradaSaida` | 6 | duas instâncias: `dataset_base_treino` / `dataset_base_teste` |
| Transformações geométricas, elástica, CLAHE, ruído, cor (§5.3.1) | `TransferKAN_EntradaSaida` | 6 | o `A.Compose([...])` de `transform_treino` |
| Teste só redimensiona e normaliza (§5.3.1) | `TransferKAN_EntradaSaida` | 6 | `transform_teste` |
| Normalização ImageNet (§5.3.1) | `TransferKAN_EntradaSaida` | 6 | `A.Normalize(mean=[0.485,...], std=[0.229,...])` |
| Lotes de 32 (§5.3.1) | `TransferKAN_EntradaSaida` | 10 | `DataLoader(..., batch_size=32, ...)` |
| Semente 42 (§5.3.2) | `TransferKAN_EntradaSaida` | 6 | `torch.manual_seed(42)`, `random.seed(42)` |
| Divisão por Identificador de Caso (§5.3.3) | `TransferKAN_EntradaSaida` | 6 | `case_id = filename[:9]` → `case_to_indices` |
| Validação cruzada de 5 dobras por caso (§5.3.3) | `TransferKAN_EntradaSaida` | 10 | `kf.split(unique_cases)` |
| Verificação programática de interseção zero (§5.3.3) | `TransferKAN_EntradaSaida` | 6 | `train_cases_check.intersection(test_cases_check)` |
| Fase 1: 15 épocas, *backbone* congelado, lr 1e-3 (§5.3.4) | `TransferKAN_EntradaSaida` | 10 | bloco `if epoch == 0:` |
| Fase 2: 45 épocas, tudo solto, lr 1e-5 (§5.3.4) | `TransferKAN_EntradaSaida` | 10 | bloco `elif epoch == epochs_fase1:` |
| MLSD em fase única, lr 1e-4 (§5.3.4) | `TransferKAN_MLSD` | 6 | `AdamW(model_mlsd.parameters(), lr=1e-4, ...)` |
| Agendador CosineAnnealingLR (§5.3.4) | ambos | 10 / 6 | `lr_scheduler.CosineAnnealingLR(...)` |
| Perda ponderada por frequência inversa (§5.3.4) | `TransferKAN_EntradaSaida` | 10 | `peso = len(labels) / (2.0 * labels.count(c))` |
| Pesos recalculados por dobra (§5.3.4) | `TransferKAN_EntradaSaida` | 10 | o cálculo está **dentro** do `for fold ...` |
| Modelo reinicializado a cada dobra (§5.3.4) | `TransferKAN_EntradaSaida` | 10 | `model_gunshot = TransferKAN_Gunshot().to(device)` dentro do laço |
| Checkpoint pela maior acurácia de teste (§5.3.4 e §5.3.6) | `TransferKAN_EntradaSaida` | 10 | `if acc_test > best_acc:` |
| Predições OOF agregadas (§5.3.4) | `TransferKAN_EntradaSaida` | 10 | `oof_y_true/pred/prob_both.extend(...)` |
| Métricas macro/micro/ponderada (Cap. 6) | `TransferKAN_EntradaSaida` | 10 | laço `for avg in ['macro','micro','weighted']` |
| AUC multiclasse uma-contra-todas (Cap. 6) | `TransferKAN_MLSD` | 6 | `roc_auc_score(..., multi_class='ovr', average=avg)` |
| Especificidade (Cap. 6) | `TransferKAN_EntradaSaida` | 10 | `specificity_per_class = TN / (TN + FP)` |
| Ablações em 128/64 px, lote 8, *hold-out* (§5.3.5) | notebooks de ablação | 3–5 | `A.Resize(128,128)`, `batch_size=8`, `train_test_split` |

---

## 9. Perguntas de banca sobre o código

Respostas curtas, para você ler em voz alta na véspera.

**1. "Por que 2048?"**
Não foi escolha minha — é a dimensão da saída do último bloco convolucional da ResNet152, depois do *average pooling* global. Ao remover a camada `fc`, é o que sobra naturalmente.

**2. "Onde exatamente a KAN substitui a MLP?"**
Célula 8, linha `self.kan = KAN([self.flat_size, 128, 2], grid_size=10, spline_order=3)`. Numa arquitetura tradicional, ali estaria um `nn.Linear` ou uma pilha `Linear → ReLU → Linear`. Todo o resto do notebook seria idêntico.

**3. "Me mostre no código onde você garante que não há vazamento."**
Em dois lugares. Primeiro na célula 6, onde construo `case_to_indices` agrupando as imagens pelos 9 primeiros caracteres do nome do arquivo, que são o identificador do caso. Depois na célula 10, onde o `KFold` recebe `unique_cases` — a lista de **casos**, não de imagens. E há uma terceira prova: o *sanity check* no fim da célula 6 calcula a interseção entre os conjuntos de IDs e alerta se não for vazia.

**4. "Por que 9 caracteres?"**
Porque é o comprimento do prefixo que identifica o caso no padrão de nomenclatura da FDCPUnBGunshotDB. É uma convenção da base, não um número arbitrário.

**5. "Por que dois objetos de dataset em vez de um?"**
Porque a transformação é fixada na construção do objeto. Com duas instâncias — uma com *augmentation*, outra determinística — a garantia de que nenhuma imagem de teste é aumentada passa a ser estrutural, e não uma disciplina que eu preciso lembrar de manter.

**6. "Por que você reinstancia o modelo dentro do laço de dobras?"**
Porque, se eu reaproveitasse o modelo, a dobra 2 começaria já tendo treinado sobre os dados que são teste da dobra 1. A validação cruzada perderia todo o sentido. Cada dobra tem que começar dos pesos da ImageNet, virgem em relação à base forense.

**7. "O que é OOF?"**
*Out-of-fold*: a predição que cada dobra faz sobre os dados que ela **não** viu no treino. Juntando as 5 dobras, tenho uma predição para cada uma das 2.551 imagens, sempre feita por um modelo que nunca viu aquele caso. As métricas saem desse conjunto agregado.

**8. "Como você calculou a AUC para 3 classes?"**
Uma-contra-todas (`multi_class='ovr'`): três curvas ROC, uma por classe contra todas as demais, combinadas em média macro, micro e ponderada. Antes disso, os rótulos passam por `label_binarize` e as saídas por `softmax`, porque a AUC precisa de escores contínuos, não de decisões.

**9. "Por que o `forward` não tem softmax?"**
Porque a `nn.CrossEntropyLoss` do PyTorch já aplica softmax internamente. Aplicá-lo duas vezes seria um erro. O softmax explícito aparece só na hora de coletar probabilidades para a AUC, com `F.softmax`.

**10. "Como você tratou o desbalanceamento?"**
Por três frentes: pesos de classe por frequência inversa na função de perda; *data augmentation* agressivo no treino; e a escolha de reportar métricas macro e AUC, que não se deixam enganar pela classe majoritária. O k-fold por caso não trata o desbalanceamento, mas impede que ele seja mascarado por vazamento.

**11. "Por que congelar o backbone nas primeiras 15 épocas?"**
Porque a KAN começa com pesos aleatórios e produz gradientes grandes e desorganizados. Se o *backbone* estivesse solto, esses gradientes danificariam representações que custaram muito para serem aprendidas na ImageNet. Congelar dá tempo às splines de se estabilizarem antes de mexer no que já é bom.

**12. "Por que a taxa cai de 1e-3 para 1e-5?"**
Porque as duas fases têm objetivos diferentes. Na fase 1 estou treinando pesos aleatórios do zero, e uma taxa alta acelera. Na fase 2 estou ajustando pesos já bons, e uma taxa cem vezes menor garante que o ajuste seja fino em vez de destrutivo.

**13. "Por que a tarefa MLSD não usa as duas fases?"**
Porque o que distingue as três distâncias são texturas de granulação fina — fuligem, tatuagem, resíduo de pólvora — que vivem nas camadas convolucionais intermediárias e não estão bem representadas nas características genéricas da ImageNet. Congelar essas camadas por um quarto do treino privaria o classificador justamente do sinal da tarefa. Por isso adotei fase única com taxa intermediária de 1e-4.

**14. "Qual a diferença entre macro, micro e ponderada?"**
Macro tira média simples entre as classes, tratando a classe de 42 imagens como igualmente importante à de 1.609 — é a que denuncia mau desempenho nas raras. Micro agrega todas as amostras antes de calcular, e por isso é dominada pela majoritária. Ponderada faz média pesada pelo número de amostras de cada classe.

**15. "O `train_test_split` da célula 6 conflita com o k-fold?"**
Não. É código legado de uma versão anterior, que usava *hold-out*. A célula 10 reatribui todas aquelas variáveis dentro do laço de dobras, então nada calculado ali sobrevive. Os resultados reportados vêm exclusivamente das 5 dobras.

**16. "Se a acurácia da ResNet152 é maior, por que a sua proposta é melhor?"**
Por três razões. Primeira: os 0,869 da referência vêm de *hold-out*, sujeito a vazamento; sob k-fold agrupado, a mediana dela cai para cerca de 0,73, contra os meus 0,860. Segunda: acurácia é métrica fraca numa base 74/26 — basta chutar sempre a majoritária para chegar perto. Terceira, e principal: em AUC, que mede a capacidade de separar as classes em todos os limiares, a TransferKAN faz 0,928 micro e 0,908 macro contra 0,821 da referência.

---

## 10. Limitações reais e como respondê-las

Depois das correções feitas na monografia, o texto e o código estão alinhados. Sobram estas quatro limitações — todas **legítimas** e todas já declaradas na Seção 5.3.6 ("Ameaças à Validade"). Reconhecê-las de antemão é sinal de maturidade; ser pego por elas é o oposto.

### 10.1. O checkpoint é escolhido no conjunto de teste

**O que o código faz:** dentro de cada dobra, `if acc_test > best_acc:` guarda os pesos da época com melhor acurácia **no teste daquela dobra** — e é esse mesmo conjunto que depois entra nas métricas OOF.

**Por que é um problema:** a época foi escolhida olhando a resposta. A métrica reportada é a da *melhor* época naqueles casos, não a de uma época escolhida às cegas. Isso torna a acurácia otimista.

**Resposta honesta:**
> "Sim, esse é um viés real e está declarado na monografia. O procedimento correto seria reservar, dentro do treino de cada dobra e também separada por caso, uma partição interna de validação exclusiva para escolher a época. Duas ponderações, porém: primeiro, o viés incide sobre as métricas dependentes de limiar — acurácia, precisão, revocação —, e a métrica sobre a qual meu argumento se apoia é a AUC, que integra o desempenho sobre todos os limiares e é estruturalmente menos sensível a isso. Segundo, a magnitude do efeito não explicaria sozinha uma diferença de 0,86 contra 0,73. Corrigir isso está listado como trabalho futuro prioritário."

### 10.2. A comparação com as ablações não é isométrica

**O que o código faz:** TransferKAN roda em 224 px, lote 32, 5-fold. As ablações rodam em 128 ou 64 px, lote 8, *hold-out*.

**Resposta honesta:**
> "As três referências não são uma ablação controlada — elas diferem da TransferKAN em resolução, tamanho de lote e protocolo de validação, além da arquitetura. O que a comparação sustenta é uma tendência, não uma atribuição causal: a progressão 0,692 → 0,707 → 0,822 → 0,860 é monotônica e acompanha exatamente o quanto o modelo se aproxima de um extrator profundo e pré-treinado. Rodar as três sob as mesmas condições da TransferKAN é o que transformaria isso numa ablação de verdade, e está entre os trabalhos futuros."

### 10.3. A referência não foi re-executada localmente

**O que acontece:** os números da ResNet152 vêm do artigo de Lira et al. (2024), não de uma execução minha. Logo, comparo dois protocolos distintos — *hold-out* lá, k-fold por caso aqui.

**Resposta honesta:**
> "Correto, e é por isso que eu não me apoio na comparação de acurácia bruta. A comparação legítima que faço é: sob o protocolo mais severo, minha AUC é maior que a AUC deles sob o protocolo mais frouxo. O ideal seria re-executar a ResNet152 sob o meu k-fold por caso — é o primeiro item de trabalho futuro da monografia."

### 10.4. A classe `Encostado` tem 42 imagens

**Resposta honesta:**
> "É a limitação mais dura do trabalho, e ela é da base, não do método. Com 42 imagens divididas em 5 dobras, cada dobra de teste tem cerca de 8 exemplos de disparo encostado. Nenhuma métrica calculada sobre 8 amostras é estatisticamente confiável. Por isso a tarefa MLSD é apresentada como secundária, e por isso não uso a grade de spline ampliada nela: mais parâmetros por aresta do que amostras disponíveis seria receita de sobreajuste."

### 10.5. Duas ressalvas menores, para não serem surpresa

- **Reprodutibilidade parcial.** A semente 42 está fixada para `torch` e `random`, mas não para `numpy` — e a Albumentations sorteia com NumPy. Além disso, convoluções em GPU não são bit a bit determinísticas. Logo, uma re-execução dá números próximos, não idênticos. Está declarado na Seção 5.3.2.
- **O repositório `leguwoi`** é clonado na primeira célula mas nunca é usado. É herança de template. Melhor dizer isso que inventar uma função para ele.

---

## 11. Como rodar hoje

> ✅ **Antes de mais nada: os 8 notebooks rodaram, e a prova está neles.** Todos guardam as saídas de execução, e nenhuma célula de nenhum deles contém erro. No `TransferKAN_EntradaSaida.ipynb`, a saída da célula 10 mostra as 5 dobras completas e termina com a tabela cujos valores são **exatamente** os da Tabela 6.1 da monografia: ACC 0,860 · AUC macro 0,908 · AUC micro 0,928 · Revocação macro 0,835 · F1 macro 0,825. Ou seja, os números que você defende vieram de uma execução real, e ela está registrada no arquivo. Se quiser conferir, é só abrir o notebook e rolar até o fim.

### 11.1. Onde rodar

**Google Colab com GPU.** Não é preferência, é requisito: a ResNet152 tem ~60 milhões de parâmetros, e são 5 dobras × 60 épocas. Em CPU, o `TransferKAN_EntradaSaida` levaria dias.

No Colab: `Ambiente de execução → Alterar o tipo de ambiente de execução → Acelerador de hardware: GPU`. Confirme na saída da célula 2 que `device` imprimiu `CUDA`.

### 11.2. O que a primeira célula faz por você

```bash
pip install git+https://github.com/Blealtan/efficient-kan -q
pip install albumentations -q
git clone https://github.com/pedrogarciafreitas/FDCPUnBGunshotDB.git -q
```

Nada precisa ser baixado à mão. Rode as células **em ordem**, de cima para baixo — o notebook tem estado, e pular a célula 6 quebra a célula 10.

### 11.3. ⚠️ O problema do Albumentations — e ele é mais sutil do que parece

Os notebooks foram escritos contra a interface anterior à versão 2.0 da biblioteca. Nas versões atuais, três parâmetros que o código usa **não existem mais**:

| No código | Situação hoje |
|---|---|
| `A.ElasticTransform(alpha_affine=50)` | `alpha_affine` foi removido |
| `A.GaussNoise(var_limit=(10.0, 50.0))` | `var_limit` foi removido (hoje é `std_range`) |
| `A.ShiftScaleRotate(...)` | descontinuada em favor de `A.Affine` |

**O código não quebra — e é justamente por isso que o problema é perigoso.** A Albumentations moderna não levanta exceção: ela emite um `UserWarning` e **descarta silenciosamente o argumento inválido**, seguindo com o valor padrão.

E isso **já aconteceu na execução que gerou os resultados do TCC**. As saídas salvas na célula 6 do `TransferKAN_EntradaSaida.ipynb` mostram exatamente estes três avisos:

```
UserWarning: ShiftScaleRotate is a special case of Affine transform.
UserWarning: Argument(s) 'alpha_affine' are not valid for transform ElasticTransform
UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
```

Portanto, na prática:

- `ElasticTransform` rodou **sem** o `alpha_affine=50`, no padrão da biblioteca;
- `GaussNoise` rodou **sem** o `var_limit=(10, 50)` — o padrão atual é `std_range=(0.2, 0.44)`, uma escala completamente diferente;
- `ShiftScaleRotate` funcionou normalmente, apenas depreciada.

**O que isso significa para a defesa:** o *conjunto* de transformações aplicado é o que a monografia descreve — flip, rotação, CLAHE, elástica, ruído gaussiano, jitter de cor. Mas **duas delas rodaram com magnitudes diferentes das que estão escritas no código**. Se perguntarem, a resposta honesta é:

> "As transformações aplicadas são as descritas. Dois parâmetros de intensidade, porém, foram descartados silenciosamente por uma mudança de interface da Albumentations entre a escrita do código e a execução, de modo que `ElasticTransform` e `GaussNoise` operaram nas magnitudes padrão da biblioteca. Isso está registrado na Seção 5.3.2 da monografia. Não invalida os resultados — a ampliação continuou ativa e diversificada —, mas significa que a reprodução exata exige fixar a versão da biblioteca."

Se for **re-executar**, escolha conscientemente:

1. **Reproduzir o experimento como ele de fato ocorreu** — use uma Albumentations 2.x e aceite os avisos. É o que corresponde aos números da monografia.
2. **Rodar com os parâmetros que o código declara** — migre para a interface nova (`A.Affine`, `std_range`) e ajuste as escalas à mão. **Isso muda o *augmentation* e portanto muda os números**, que deixarão de bater com a Tabela 6.1.

### 11.4. O que você deve ver

| Célula | Saída esperada |
|---|---|
| 2 | a "radiografia" das pastas, com as contagens de arquivos |
| 4/6 | `Total de imagens encontradas: 2554` e `Total de Casos Únicos: 486` |
| 6 | `✅ SUCESSO! Zero interseção semântica identificada pelo código.` |
| 8 | `🚀 Modelo TransferKAN_Gunshot inicializado e enviado para: CUDA` |
| 10 | 5 blocos de FOLD e, ao final, a tabela de métricas |

Se a contagem de imagens não der 2.554, o `git clone` falhou ou a estrutura de pastas mudou — rode `inspecionar_diretorio` de novo e confira os caminhos da célula 4.

> 📌 **Note a diferença: o código conta 2.554, a monografia diz 2.551.** Os 2.551 são a contagem publicada por Lira et al. (1.883 entradas + 668 saídas); a cópia atual do repositório público tem **três imagens de saída a mais**, totalizando 671 saídas e 2.554 imagens, em 486 casos. A tarefa MLSD bate exatamente: 1.883 imagens de entrada, em 441 casos. A diferença é de 0,12% e não muda nenhuma conclusão, mas está declarada na Seção 5.1 da monografia — porque a banca pode simplesmente comparar o número impresso no notebook com o número no texto.

---

## Onde continuar

- Teoria (KAN, B-splines, AUC, patologia forense) → [`GUIA_DE_ESTUDO.md`](GUIA_DE_ESTUDO.md)
- O texto que este código sustenta → `tcc_dissertation/pt/monografia.pdf` (Caps. 4, 5 e 6)
- A base de comparação → `references/2024_paper.txt`
