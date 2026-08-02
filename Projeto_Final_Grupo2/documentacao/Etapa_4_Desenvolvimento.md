# Etapa 4 - Desenvolvimento do Projeto: hardware e software

Disciplina: ESZA019 - Visão Computacional  
Grupo 2: Cesar de Jesus Carvalho, Mariana Chiara Travassos Sarinho e Vinícius de Marchi Costa  
Entrega: 5 de agosto de 2026

## 1. Título do projeto

Sistema inteligente estereoscópico para classificação de materiais recicláveis e estimativa de distância.

## 2. Objetivo

Desenvolver um sistema em tempo real capaz de receber um objeto por vez, classificar seu material, indicar a lixeira correspondente e estimar sua distância em relação às câmeras. O projeto integra recursos trabalhados na disciplina: formação da imagem, parâmetros intrínsecos e extrínsecos, distorção, calibração monocular e estéreo, retificação, disparidade, reconstrução 3D simplificada e CNN.

## 3. Hardware e maquete

O protótipo utiliza duas webcams USB conectadas ao mesmo computador. As câmeras estão fixadas em uma caixa de papelão reforçada, no mesmo suporte utilizado no Lab 6. A base nominal de montagem foi planejada próxima de 63 mm entre os centros ópticos. A estimativa obtida anteriormente pela calibração do Lab 6 foi de 55,07 mm; essa diferença será discutida como resultado da montagem real e da estimação dos parâmetros extrínsecos.

Cuidados adotados:

- lentes na mesma altura;
- câmeras aproximadamente paralelas;
- suporte sem folga;
- identificação física de esquerda e direita;
- resolução comum de 640 x 480;
- proibição de mover as câmeras depois da calibração.

Fotos que devem ser adicionadas antes da entrega:

1. visão frontal da maquete;
2. visão superior mostrando a separação das câmeras;
3. detalhe da fixação de cada webcam;
4. computador com as duas câmeras conectadas;
5. tabuleiro físico de calibração.

## 4. Calibração das câmeras

O tabuleiro do grupo possui 7 x 9 quadrados, equivalentes a 6 x 8 cantos internos. Cada quadrado mede 30 mm. Serão capturados 25 pares de imagens com variação de distância, rotação, inclinação e posição no campo de visão.

O programa executa:

1. detecção dos cantos com `cv2.findChessboardCorners` e refinamento subpixel;
2. calibração monocular de cada câmera com `cv2.calibrateCamera`, obtendo `K1`, `D1`, `K2` e `D2`;
3. calibração estéreo com `cv2.stereoCalibrate`, obtendo `R`, `T`, `E` e `F`;
4. retificação com `cv2.stereoRectify`, obtendo `R1`, `R2`, `P1`, `P2` e `Q`;
5. criação dos mapas com `cv2.initUndistortRectifyMap`;
6. validação do erro de reprojeção e do desalinhamento epipolar vertical.

Resultados anteriores do Lab 6, usados apenas como referência:

| Medida | Resultado |
|---|---:|
| Pares válidos | 25 |
| RMS esquerdo | 0,1734 px |
| RMS direito | 0,1679 px |
| RMS estéreo | 1,5284 px |
| Baseline estimado | 5,507 cm |
| Erro absoluto médio das distâncias | 0,85 cm |
| Erro percentual médio | 1,16% |

Esses valores demonstram que a montagem já produziu boas medidas de distância, mas a calibração própria do projeto será registrada separadamente. O arquivo do Lab 6 permanece apenas como contingência.

Tabela para os resultados da nova calibração:

| Medida | Resultado do projeto final |
|---|---:|
| Pares válidos | preencher após calibrar |
| RMS esquerdo | preencher |
| RMS direito | preencher |
| RMS estéreo | preencher |
| Erro de reprojeção esquerdo | preencher |
| Erro de reprojeção direito | preencher |
| Erro epipolar vertical | preencher |
| Baseline estimado | preencher |

## 5. Processamento estéreo

Os frames são capturados quase simultaneamente usando `grab()` nas duas câmeras antes de `retrieve()`. Depois da retificação, as imagens em escala de cinza são processadas pelo StereoSGBM. A matriz `Q` converte disparidade em coordenadas 3D. Para reduzir ruído, a distância exibida é a mediana dos pontos válidos na região central do objeto e também é suavizada temporalmente.

Quando a profundidade não é confiável, o sistema não inventa uma medida: mostra “Distância indisponível” e mantém a classificação funcionando.

## 6. Classificação dos resíduos

Foi escolhida uma ResNet-50 pré-treinada no ImageNet, ajustada por transfer learning e fine-tuning para seis classes:

- `cardboard`;
- `glass`;
- `metal`;
- `paper`;
- `plastic`;
- `trash`.

O usuário posiciona um objeto por vez dentro do quadrado central. Somente o recorte da imagem esquerda retificada é enviado ao classificador. A decisão é aceita com confiança mínima de 60%; abaixo desse valor, a tela apresenta “Objeto não reconhecido”.

Correspondência das lixeiras:

| Classe | Indicação |
|---|---|
| cardboard / paper | lixeira azul |
| plastic | lixeira vermelha |
| glass | lixeira verde |
| metal | lixeira amarela |
| trash | lixeira cinza para rejeitos |

## 7. Integração e interface

A interface mostra a imagem esquerda retificada, a área usada pela ResNet, o mapa de disparidade, a classe prevista, a confiança, a lixeira, a distância, a orientação de aproximar/afastar, o FPS e a latência de inferência.

O sistema salva automaticamente um CSV por sessão com as métricas necessárias para a análise final.

## 8. Software entregue

O arquivo principal é `executar_projeto.py`. Ele possui seis modos: listar, verificar, capturar, calibrar, validar e executar. Também foram incluídos scripts curtos para cada etapa, verificador de ambiente, conversor Keras para ONNX, dependências separadas para Windows/Linux, notebooks corrigidos e manual de execução.

O modelo Keras foi validado no Windows com Python 3.12 e TensorFlow 2.21. A alternativa ONNX pode ser executada pelo módulo DNN do OpenCV, mas deve ser validada no ambiente Linux antes da apresentação.

## 9. Execução resumida

```bash
conda activate CV26
cd Projeto_Final_Grupo2
bash configurar_linux.sh
python3 verificar_ambiente.py --testar-cameras --cam-esq 0 --cam-dir 2
python3 validar_calibracao.py --cam-esq 0 --cam-dir 2
python3 executar_projeto.py --modo executar --cam-esq 0 --cam-dir 2
```

## 10. Limitações conhecidas

- transparência e reflexos do vidro prejudicam a disparidade;
- superfícies lisas fornecem poucas correspondências;
- movimentar o suporte invalida a calibração;
- o classificador pressupõe um único objeto centralizado;
- iluminação e semelhança visual entre papel e papelão podem alterar a confiança;
- índices de câmera podem mudar entre computadores.

## 11. Situação dos testes finais

- calibração concluída com 25 pares;
- classificação, distância, alertas e mapa de disparidade validados no Windows;
- métricas e imagens reais organizadas em `resultados/`;
- avaliação final da ResNet-50 concluída com 87,11% de acurácia;
- conversão e teste do ONNX no ambiente CV26 ainda pendentes;
- teste formal com voluntários permanece como extensão opcional.

## 12. Uso de inteligência artificial generativa

Foi utilizada uma ferramenta de inteligência artificial generativa como apoio na organização do código, revisão técnica e redação inicial da documentação. Os integrantes são responsáveis por revisar, executar, validar e apresentar o conteúdo final e seus resultados experimentais.
