# Projeto Final - Visão Computacional - Grupo 2

Sistema inteligente estereoscópico para classificação de materiais recicláveis e estimativa de distância.

**Integrantes:**

- Cesar de Jesus Carvalho
- Mariana Chiara Travassos Sarinho
- Vinícius de Marchi Costa

## Objetivo

O projeto utiliza duas webcams fixadas no mesmo suporte para combinar duas tarefas:

1. classificação do material por uma ResNet-50;
2. estimativa da distância por visão estéreo e mapa de disparidade.

A interface mostra o material previsto, confiança, lixeira recomendada, distância, orientação para aproximar ou afastar o objeto, FPS, latência e mapa de disparidade.

As seis classes utilizadas são `cardboard`, `glass`, `metal`, `paper`, `plastic` e `trash`.

## Resultados principais

O conjunto TrashNet foi dividido de forma estratificada em 70% para treinamento, 15% para validação e 15% para teste. Todas as classes estão presentes nas três partes.

| Métrica final | Resultado |
|---|---:|
| Imagens de treinamento | 1.768 |
| Imagens de validação | 379 |
| Imagens de teste | 380 |
| Acurácia no teste | 87,11% |
| F1-score macro | 84,63% |
| F1-score ponderado | 87,27% |
| Loss no teste | 0,4271 |

Nos testes práticos, o sistema reconheceu corretamente exemplos de papel, papelão, plástico, vidro e metal. As principais dificuldades apareceram em objetos transparentes ou muito reflexivos, como garrafa PET, colher e algumas posições da panela metálica. Esses casos podem causar confusão entre `plastic`, `glass` e `metal`.

## Calibração estéreo final

A calibração incluída foi realizada em 02/08/2026 com 25 pares de imagens e tabuleiro de 6 x 8 cantos internos.

| Parâmetro | Resultado |
|---|---:|
| Resolução | 640 x 480 |
| Lado do quadrado | 0,030 m |
| Baseline estimado | 0,0647 m |
| RMS estéreo | 2,0238 px |
| Erro epipolar vertical médio | 1,2713 px |

Depois da calibração, as câmeras não devem se mover uma em relação à outra.

## Estrutura do repositório

```text
Projeto_Final_Grupo2/
├── 01_treinamento_modelo.ipynb
├── 02_execucao_estereo.ipynb
├── executar_projeto.py
├── verificar_ambiente.py
├── listar_cameras.py
├── capturar_calibracao.py
├── calibrar_estereo.py
├── validar_calibracao.py
├── converter_modelo_onnx.py
├── config_sgbm.json
├── configurar_windows.bat
├── configurar_linux.sh
├── calibracao/
├── modelos/
├── resultados/
└── documentacao/
```

## Modelo treinado

O arquivo `resnet50_waste.keras` possui aproximadamente 210 MB e, por isso, não está incluído diretamente no repositório. O limite normal do GitHub é de 100 MiB por arquivo.

Para reproduzir o modelo, execute o notebook `01_treinamento_modelo.ipynb` no Google Colab. Ao final, coloque estes arquivos na pasta `modelos/`:

```text
modelos/
├── resnet50_waste.keras
└── class_names.json
```

O notebook publicado mantém os resultados finais, incluindo curvas de treinamento, relatório de classificação, matriz de confusão e exemplos de predição.

## Execução no Windows

Foi utilizado Python 3.12.10 e TensorFlow 2.21.0 nos testes finais.

No Prompt de Comando, dentro da pasta do projeto, execute:

```bat
configurar_windows.bat
```

O script cria o ambiente `.venv312` e instala as bibliotecas. Se o Python 3.12 ainda não estiver instalado:

```bat
py install 3.12
```

Para localizar os índices das webcams:

```bat
python listar_cameras.py
```

Na montagem testada no Windows, a câmera esquerda foi `1` e a direita foi `0`:

```bat
python executar_projeto.py --modo verificar --cam-esq 1 --cam-dir 0
python executar_projeto.py --modo executar --cam-esq 1 --cam-dir 0
```

O modelo Keras já é o padrão. Portanto, não é necessário informar `--modelo`. Se os índices mudarem, substitua os valores de `--cam-esq` e `--cam-dir`.

## Nova calibração

Uma nova calibração só deve ser feita se a posição relativa das câmeras for alterada.

```bat
python capturar_calibracao.py --cam-esq 1 --cam-dir 0
python calibrar_estereo.py
python validar_calibracao.py --cam-esq 1 --cam-dir 0
```

## Modos do programa

| Modo | Finalidade |
|---|---|
| `listar` | procura índices de webcams disponíveis |
| `verificar` | mostra as duas câmeras sem procurar o tabuleiro |
| `capturar` | salva pares do tabuleiro detectado nas duas câmeras |
| `calibrar` | calcula calibração, retificação e métricas |
| `validar` | mostra imagens retificadas com linhas epipolares |
| `executar` | integra classificação, disparidade, distância e interface |

## Controles

- `q` ou `Esc`: encerra o programa;
- `Espaço`: salva um par no modo de captura;
- `d`: mostra ou oculta o mapa de disparidade;
- `s`: salva imagens e resultados da execução.

## Linux e ONNX

O programa também aceita modelos `.onnx`, que podem ser executados pelo módulo DNN do OpenCV. Essa alternativa permanece opcional e ainda deve ser validada no ambiente CV26 da faculdade.

Depois de gerar e testar `modelos/resnet50_waste.onnx`, use:

```bash
python3 executar_projeto.py --modo executar --modelo modelos/resnet50_waste.onnx --cam-esq 0 --cam-dir 2
```

Os índices das câmeras podem ser diferentes no Linux.

## Limitações observadas

- superfícies transparentes, refletivas ou sem textura prejudicam a classificação e a disparidade;
- objetos pequenos podem fazer o modelo considerar principalmente o fundo da imagem;
- confiança alta não garante que a classe esteja correta;
- a base TrashNet é diferente das imagens produzidas pelas webcams da montagem;
- novos exemplos capturados na própria montagem podem melhorar principalmente as classes `metal`, `plastic` e `glass`.

## Arquivos produzidos

Cada execução cria uma pasta em `resultados/` com:

- imagens esquerda e direita retificadas;
- tela final com classe e distância;
- mapa de disparidade;
- arquivo `metricas_tempo_real.csv` com classe, confiança, distância, FPS e latência.
