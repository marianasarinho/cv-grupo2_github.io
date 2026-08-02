# Etapa 5 - Roteiro de teste voluntário

Projeto: Sistema inteligente estereoscópico para classificação de materiais recicláveis e estimativa de distância  
Grupo 2: Cesar de Jesus Carvalho, Mariana Chiara Travassos Sarinho e Vinícius de Marchi Costa  
Entrega: 5 de agosto de 2026

## 1. Objetivo do teste

Avaliar se uma pessoa sem contato prévio com o projeto consegue posicionar um resíduo, compreender a classificação, interpretar a distância e escolher a lixeira indicada. O teste também medirá acertos, confiança, erro de distância, disponibilidade da profundidade, FPS, latência e clareza da interface.

## 2. Participantes

Planejamento: 5 a 10 voluntários adultos. Não serão coletados nome completo, documento, imagem do rosto ou outro dado sensível. Cada participante receberá apenas um código, como V01, V02 e V03.

## 3. Consentimento simples

Antes do início, o responsável lerá:

> Este é um teste acadêmico de um protótipo de visão computacional. Sua participação é voluntária e dura aproximadamente 8 minutos. Serão registrados apenas o código do participante, o resultado das tarefas e suas respostas sobre a interface. Você pode interromper o teste a qualquer momento. Nenhuma imagem pessoal será publicada sem autorização específica.

Registrar: `[ ] aceitou participar` ou `[ ] não aceitou`.

## 4. Preparação do ambiente

1. Fixar as câmeras e não alterar o suporte após a calibração.
2. Manter iluminação suficiente e fundo simples.
3. Marcar no chão as distâncias de 0,25 m, 0,50 m, 1,00 m e 1,60 m.
4. Separar pelo menos um exemplo de papel, papelão, plástico, vidro, metal e rejeito.
5. Confirmar a execução de `verificar_ambiente.py`.
6. Abrir o sistema e iniciar uma nova sessão de resultados.
7. Não ensinar a resposta correta durante a tarefa.

## 5. Instrução dada ao voluntário

“Posicione um objeto por vez dentro do quadrado mostrado na tela. Observe a classe, a lixeira indicada e a mensagem de distância. Faça os ajustes que o sistema solicitar.”

## 6. Tarefas

| Tarefa | Ação do voluntário | Resultado esperado |
|---|---|---|
| T1 | colocar papel ou papelão a 0,50 m | classe correspondente, lixeira azul e distância válida |
| T2 | colocar plástico a 1,00 m | plástico, lixeira vermelha e distância próxima de 1,00 m |
| T3 | testar vidro ou metal na faixa válida | classe/lixeira correta; registrar possível falha por reflexão |
| T4 | colocar um objeto a 0,25 m | mensagem “AFASTE O OBJETO” |
| T5 | colocar um objeto a 1,60 m | mensagem “APROXIME O OBJETO” |
| T6 | colocar um exemplo da classe `trash` | indicação de rejeito/lixeira cinza |
| T7 | escolher livremente outro objeto | observar se o voluntário entende confiança e falha de profundidade |

Para reduzir efeito de aprendizagem, T1, T2 e T3 devem ter a ordem alternada entre os participantes.

## 7. Dados registrados por tentativa

| Campo | Descrição |
|---|---|
| código | participante anônimo |
| tarefa | T1 a T7 |
| material real | classe esperada |
| classe prevista | saída da ResNet-50 |
| confiança | probabilidade da classe prevista |
| distância real | posição marcada no chão |
| distância medida | valor do sistema |
| profundidade válida | sim/não |
| tempo da tarefa | segundos até concluir |
| indicação compreendida | sim/não |
| observação | reflexo, pouca textura, hesitação ou erro |

Modelo de tabela:

| Código | Tarefa | Real | Prevista | Conf. | Dist. real (m) | Dist. medida (m) | Prof. válida | Tempo (s) | Compreendeu | Observação |
|---|---|---|---|---:|---:|---:|---|---:|---|---|
| V01 | T1 |  |  |  | 0,50 |  |  |  |  |  |
| V01 | T2 |  |  |  | 1,00 |  |  |  |  |  |
| V01 | T3 |  |  |  |  |  |  |  |  |  |
| V01 | T4 |  |  |  | 0,25 |  |  |  |  |  |
| V01 | T5 |  |  |  | 1,60 |  |  |  |  |  |
| V01 | T6 |  |  |  |  |  |  |  |  |  |
| V01 | T7 |  |  |  |  |  |  |  |  |  |

## 8. Perguntas após as tarefas

Escala: 1 = discordo totalmente; 5 = concordo totalmente.

1. Foi fácil entender onde posicionar o objeto.
2. A classe prevista ficou clara.
3. A cor e o nome da lixeira ficaram claros.
4. A indicação de aproximar ou afastar foi útil.
5. O sistema respondeu rápido o suficiente.
6. Eu usaria um sistema semelhante para ajudar no descarte.

Perguntas abertas:

1. O que mais confundiu você?
2. Qual informação da tela foi mais útil?
3. O que você mudaria na interface?

## 9. Métricas calculadas

- acurácia de classificação: acertos / tentativas válidas;
- acurácia por classe e matriz de confusão;
- precisão, recall e F1-score;
- taxa de profundidade válida: medições válidas / tentativas;
- erro absoluto médio da distância: média de `|medida - real|`;
- erro percentual médio para distâncias reais não nulas;
- tempo médio de conclusão;
- FPS e latência média da inferência;
- taxa de compreensão: tarefas com indicação compreendida / tarefas;
- média e distribuição das respostas de 1 a 5.

## 10. Critério de sucesso de uma tentativa

Uma tentativa será considerada totalmente bem-sucedida quando:

1. a classe/lixeira estiver correta;
2. a profundidade for válida quando esperada;
3. o erro de distância não ultrapassar 10 cm ou 10%, usando o maior limite;
4. o voluntário concluir sem intervenção direta.

Falhas serão mantidas na análise e discutidas, especialmente em vidro, superfícies refletivas e objetos sem textura.

## 11. Encerramento

Agradecer ao voluntário, perguntar se deseja retirar sua participação e salvar a planilha sem identificação pessoal. Os registros automáticos em CSV serão comparados com as anotações do observador.

