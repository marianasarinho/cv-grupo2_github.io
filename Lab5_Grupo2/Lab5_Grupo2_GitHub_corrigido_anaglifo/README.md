# Lab 5 — Câmera Estéreo, Calibração e Vídeo 3D

**Disciplina:** ESZA019 — Visão Computacional  
**Grupo:** Grupo 2  
**Integrantes:** preencher nomes completos  

Este repositório contém o relatório em Jupyter Notebook, os códigos usados no laboratório, os dados de calibração, os parâmetros obtidos e o vídeo anáglifo gerado.

## Arquivos principais

- `Relatorio_Lab5_Grupo2.ipynb`: relatório principal para visualizar no GitHub.
- `Relatorio_Lab5_Grupo2.html`: versão HTML do relatório.
- `codigos/`: scripts Python usados no experimento.
- `dados/data/stereoL/`: imagens da câmera esquerda.
- `dados/data/stereoR/`: imagens da câmera direita.
- `dados/data/corners_detected/`: imagens com cantos detectados.
- `resultados/calibracao_resultados_abc.txt`: resultados numéricos da calibração.
- `resultados/stereo_params_abc.xml`: parâmetros salvos da câmera estéreo.
- `resultados/video3d_anaglifo_abc.mp4`: vídeo 3D em anáglifo.
- `figuras/`: figuras usadas no relatório.

## Ordem de execução

```bash
conda activate CV26
python3 codigos/capture_images_abc.py
python3 codigos/calibrate_abc.py
python3 codigos/movie3d_abc.py
python3 codigos/movie3d_abc_gravacao.py
```

## Resultados resumidos

| Parâmetro | Valor |
|---|---:|
| Tamanho da imagem | 640 x 480 pixels |
| Tabuleiro | 6 x 8 cantos internos |
| Pares válidos | 15 |
| RMS câmera esquerda | 0.224408 |
| RMS câmera direita | 0.214041 |
| RMS estéreo | 0.638500 |
| Baseline | 1.885733 |

## Observação sobre o anáglifo

Após o teste com os óculos 3D, o vídeo anáglifo foi corrigido invertendo a atribuição vermelho/ciano. A calibração não precisou ser refeita; apenas a composição visual do anáglifo foi alterada. O arquivo `resultados/video3d_anaglifo_abc.mp4` já está na versão corrigida. A versão anterior foi mantida como `resultados/video3d_anaglifo_abc_original_invertido.mp4`.
