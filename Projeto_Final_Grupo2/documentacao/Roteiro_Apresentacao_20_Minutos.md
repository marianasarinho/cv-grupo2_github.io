# Roteiro de apresentação e contingência - 20 minutos

## Antes de sair de casa

- copiar `resnet50_waste.keras` e `class_names.json`;
- manter `calibracao_estereo.npz` e o XML de backup;
- levar uma cópia em pendrive e outra em nuvem;
- levar papel/papelão, plástico, metal e um objeto com textura;
- conferir que as webcams continuam firmes e identificadas;
- manter as imagens dos testes finais e um vídeo curto como contingência.

## Cronograma em sala

| Tempo | Ação |
|---:|---|
| 0-2 min | conectar as duas webcams e abrir o terminal |
| 2-4 min | ativar o ambiente já preparado no computador usado |
| 4-6 min | executar `verificar_ambiente.py --testar-cameras` |
| 6-8 min | conferir os índices e a imagem das câmeras |
| 8-10 min | mostrar a retificação com linhas horizontais |
| 10-15 min | classificar 3 materiais e mostrar distância/lixeira |
| 15-17 min | demonstrar objeto próximo e aviso para afastar |
| 17-19 min | abrir o CSV com FPS, latência e validade da profundidade |
| 19-20 min | mostrar os arquivos de calibração e concluir |

## Comandos

```bat
cd Projeto_Final_Grupo2
.venv312\Scripts\activate
python verificar_ambiente.py --testar-cameras --cam-esq 1 --cam-dir 0
python validar_calibracao.py --cam-esq 1 --cam-dir 0
python executar_projeto.py --modo executar --cam-esq 1 --cam-dir 0
```

## Se algo falhar

- índices diferentes: `python3 listar_cameras.py`;
- calibração nova ausente: o programa usa o XML do Lab 6 automaticamente;
- profundidade inválida: usar objeto com textura, boa iluminação e distância entre 0,30 m e 1,50 m;
- modelo não abre: confirmar se `resnet50_waste.keras` está em `modelos/` e se o ambiente `.venv312` está ativo;
- uma webcam não abre: trocar a porta USB, evitar hub e executar novamente a listagem;
- nunca treinar o modelo nem reinstalar TensorFlow durante a apresentação;
- se a apresentação for feita no Linux, validar o ONNX e os índices das câmeras antes de sair de casa.
