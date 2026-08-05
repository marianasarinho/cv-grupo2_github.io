# Etapa 5 - Roteiro de teste voluntário

Projeto: Sistema inteligente estereoscópico para classificação de materiais recicláveis e estimativa de distância  
Grupo 2: Cesar de Jesus Carvalho, Mariana Chiara Travassos Sarinho e Vinícius de Marchi Costa  
Entrega: 5 de agosto de 2026

## 1. Objetivo do teste

Avaliar se uma pessoa sem contato prévio com o projeto consegue posicionar um resíduo, compreender a classificação, interpretar a distância e identificar a lixeira indicada.

## 2. Participantes

Planejamento: 5 a 10 voluntários adultos. Não serão coletados nome completo, documento, imagem do rosto ou outro dado sensível. Será utilizado o mesmo modelo de roteiro para todos os participantes, com uma cópia separada para cada teste.

## 3. Consentimento simples

Antes do início, o responsável lerá:

> Este é um teste acadêmico de um protótipo de visão computacional. Sua participação é voluntária e dura aproximadamente 8 minutos. Serão registrados apenas os resultados das tarefas e suas respostas sobre a interface. Você pode interromper o teste a qualquer momento. Nenhuma imagem pessoal será publicada sem autorização específica.

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

Para reduzir efeito de aprendizagem, T1, T2 e T3 devem ter a ordem alternada entre os participantes.

## 7. Tabela de anotações

| Tarefa | O que era esperado | O que apareceu | Observação |
|---|---|---|---|
| T1 | classe correspondente, lixeira azul e distância próxima de 0,50 m |  |  |
| T2 | plástico, lixeira vermelha e distância próxima de 1,00 m |  |  |
| T3 | classe e lixeira corretas para vidro ou metal |  |  |
| T4 | mensagem “AFASTE O OBJETO” |  |  |
| T5 | mensagem “APROXIME O OBJETO” |  |  |

## 8. Perguntas após as tarefas

Escala: 1 = discordo totalmente; 5 = concordo totalmente.

1. Foi fácil entender onde posicionar o objeto.
2. A classe prevista ficou clara.
3. A cor e o nome da lixeira ficaram claros.
4. A indicação de aproximar ou afastar foi útil.
5. O sistema respondeu rápido o suficiente.
6. Eu usaria um sistema semelhante para ajudar no descarte.

## 9. Métricas calculadas

- quantidade de tarefas em que o resultado esperado apareceu;
- quantidade de classificações e indicações de lixeira corretas;
- quantidade de avisos de distância apresentados corretamente;
- média das respostas de 1 a 5 sobre a interface.

## 10. Critério de sucesso de uma tentativa

Uma tentativa será considerada totalmente bem-sucedida quando:

1. a classe/lixeira estiver correta;
2. a distância ou o aviso esperado aparecer corretamente;
3. o voluntário concluir sem intervenção direta.

Falhas serão mantidas na análise e discutidas, especialmente em vidro, superfícies refletivas e objetos sem textura.

## 11. Encerramento

Agradecer ao voluntário, perguntar se deseja retirar sua participação e guardar o roteiro preenchido sem identificação pessoal.
