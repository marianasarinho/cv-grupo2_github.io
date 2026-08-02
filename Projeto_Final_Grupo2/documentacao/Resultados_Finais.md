# Resultados finais

Data dos testes: 02/08/2026.

## 1. Avaliação da ResNet-50

O dataset foi dividido de forma estratificada, garantindo que as seis classes aparecessem nos três conjuntos.

| Conjunto | Imagens |
|---|---:|
| Treinamento | 1.768 |
| Validação | 379 |
| Teste | 380 |

Métricas obtidas no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Loss | 0,4271 |
| Acurácia | 87,11% |
| Precision macro | 84,07% |
| Recall macro | 86,06% |
| F1-score macro | 84,63% |
| F1-score ponderado | 87,27% |

Desempenho por classe:

| Classe | Precision | Recall | F1-score | Imagens |
|---|---:|---:|---:|---:|
| cardboard | 90,00% | 90,00% | 90,00% | 60 |
| glass | 85,71% | 94,74% | 90,00% | 76 |
| metal | 87,93% | 82,26% | 85,00% | 62 |
| paper | 93,02% | 89,89% | 91,43% | 89 |
| plastic | 90,62% | 79,45% | 84,67% | 73 |
| trash | 57,14% | 80,00% | 66,67% | 20 |

A classe `paper` obteve o maior F1-score. A classe `trash` apresentou o menor resultado e também possui a menor quantidade de imagens no teste.

## 2. Calibração estéreo

| Parâmetro | Resultado |
|---|---:|
| Pares válidos | 25 |
| Resolução | 640 x 480 |
| Cantos internos | 6 x 8 |
| Lado do quadrado | 0,030 m |
| RMS câmera esquerda | 1,1493 px |
| RMS câmera direita | 1,1281 px |
| RMS estéreo | 2,0238 px |
| Erro de reprojeção esquerdo | 0,9067 px |
| Erro de reprojeção direito | 0,9061 px |
| Erro epipolar vertical médio | 1,2713 px |
| Baseline estimado | 0,0647 m |

## 3. Testes práticos com as webcams

Foram testados objetos com diferentes tamanhos, materiais, posições, reflexos e distâncias. As imagens selecionadas estão em `resultados/testes_finais/`.

| Objeto | Saída observada | Resultado |
|---|---|---|
| Garrafa de vidro | vidro, 96,6% | correto |
| Garrafa PET | vidro, 95,7% | incorreto |
| Embalagem pequena | não reconhecido, 56,7% | rejeitada pelo limiar |
| Papel-alumínio | metal, 82,7% em um teste | correto em uma posição |
| Papel-alumínio | não reconhecido, 32,0% em outro teste | baixa confiança |
| Colher | vidro, 97,8% | incorreto |
| Panela pela lateral | metal, 88,6% | correto |
| Fundo da panela | vidro, 91,0% | incorreto |
| Papel amassado | papel, 66,5% | correto |
| Folheto | papel, 94,9% | correto |
| Objeto plástico | plástico, 90,3% | correto |
| Objeto plástico branco | plástico, 86,8% | correto |
| Caderno | papel, 74,6% | correto |
| Caixa de tênis | papelão, 92,8% | correto |

As distâncias registradas nos exemplos selecionados ficaram aproximadamente entre 0,29 m e 1,17 m. Nos objetos a 0,29 m, o aviso para afastar apareceu corretamente.

## 4. Análise

Os testes mostram que a classificação e a profundidade funcionam de forma integrada, mas o resultado depende da aparência do objeto. Vidro, PET e metal podem produzir brilho e reflexos parecidos. Por isso, a garrafa PET e a colher foram classificadas como vidro com confiança alta.

A panela mostrou bem essa limitação: a lateral foi reconhecida como metal, enquanto o fundo muito reflexivo foi classificado como vidro. O papel-alumínio também mudou de resultado conforme o formato e o ângulo.

Os mapas de disparidade apresentaram ruído e regiões sem correspondência em superfícies lisas, claras, transparentes ou refletivas. Esse comportamento é esperado em visão estéreo, pois o algoritmo precisa encontrar pontos semelhantes nas duas imagens.

Como melhoria, o treinamento pode ser complementado com imagens capturadas pelas próprias webcams, principalmente para as classes `metal`, `plastic` e `glass`, variando fundo, iluminação, distância e orientação.
