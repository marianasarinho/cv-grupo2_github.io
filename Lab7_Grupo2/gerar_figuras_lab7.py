"""Recria as figuras numéricas do relatório do Laboratório 7.

Os valores abaixo foram transcritos das execuções realizadas pelo Grupo 2
no Google Colab em 29/07/2026.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PASTA_ASSETS = Path(__file__).resolve().parent / "assets"
PASTA_ASSETS.mkdir(exist_ok=True)

EPOCAS = np.arange(1, 11)

ACURACIA_TREINO = np.array(
    [0.4270, 0.5648, 0.6221, 0.6580, 0.6816,
     0.6983, 0.7186, 0.7386, 0.7486, 0.7617]
)
PERDA_TREINO = np.array(
    [1.5733, 1.2237, 1.0703, 0.9691, 0.8996,
     0.8510, 0.7936, 0.7406, 0.7062, 0.6637]
)
ACURACIA_VALIDACAO = np.array(
    [0.5552, 0.6162, 0.6588, 0.6809, 0.6872,
     0.6971, 0.6934, 0.7020, 0.7111, 0.7115]
)
PERDA_VALIDACAO = np.array(
    [1.2564, 1.0817, 0.9644, 0.9129, 0.8938,
     0.8715, 0.8832, 0.8592, 0.8435, 0.8426]
)

ACURACIA_AUMENTO_TREINO = np.array(
    [0.4001, 0.4975, 0.5329, 0.5579, 0.5716,
     0.5858, 0.5928, 0.6026, 0.6074, 0.6174]
)
PERDA_AUMENTO_TREINO = np.array(
    [1.6497, 1.4007, 1.3025, 1.2441, 1.2056,
     1.1690, 1.1477, 1.1210, 1.1058, 1.0846]
)
ACURACIA_AUMENTO_VALIDACAO = np.array(
    [0.5283, 0.5505, 0.5939, 0.6134, 0.6333,
     0.6337, 0.6475, 0.6425, 0.6213, 0.6467]
)
PERDA_AUMENTO_VALIDACAO = np.array(
    [1.3001, 1.2284, 1.1377, 1.0871, 1.0302,
     1.0403, 0.9926, 1.0107, 1.1160, 1.0412]
)

CLASSES = [
    "Avião", "Automóvel", "Pássaro", "Gato", "Cervo",
    "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão",
]

MATRIZ_CONFUSAO = np.array(
    [
        [780, 23, 24, 12, 24, 4, 10, 16, 74, 33],
        [20, 817, 4, 5, 4, 5, 6, 5, 37, 97],
        [83, 7, 504, 56, 154, 69, 57, 43, 13, 14],
        [26, 11, 57, 483, 114, 158, 62, 43, 30, 16],
        [21, 5, 42, 43, 739, 22, 39, 72, 16, 1],
        [18, 9, 43, 170, 70, 580, 19, 72, 12, 7],
        [10, 7, 32, 64, 71, 21, 771, 9, 8, 7],
        [19, 3, 21, 21, 62, 50, 4, 800, 11, 9],
        [70, 37, 8, 7, 7, 5, 5, 6, 832, 23],
        [35, 68, 8, 13, 7, 5, 3, 16, 36, 809],
    ]
)


def aplicar_estilo():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def salvar_curvas_principais():
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))

    eixos[0].plot(EPOCAS, 100 * ACURACIA_TREINO, marker="o", label="Treino")
    eixos[0].plot(
        EPOCAS, 100 * ACURACIA_VALIDACAO, marker="o", label="Validação"
    )
    eixos[0].set(
        title="Acurácia ao longo do treinamento",
        xlabel="Época",
        ylabel="Acurácia (%)",
        xticks=EPOCAS,
    )
    eixos[0].legend()

    eixos[1].plot(EPOCAS, PERDA_TREINO, marker="o", label="Treino")
    eixos[1].plot(EPOCAS, PERDA_VALIDACAO, marker="o", label="Validação")
    eixos[1].set(
        title="Perda ao longo do treinamento",
        xlabel="Época",
        ylabel="Perda",
        xticks=EPOCAS,
    )
    eixos[1].legend()

    fig.suptitle("CNN original — CIFAR-10", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PASTA_ASSETS / "curvas_treinamento.png", bbox_inches="tight")
    plt.close(fig)


def salvar_matriz_confusao():
    fig, eixo = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        MATRIZ_CONFUSAO,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={"label": "Número de imagens"},
        ax=eixo,
    )
    eixo.set(
        title="Matriz de confusão — conjunto de teste",
        xlabel="Classe predita",
        ylabel="Classe real",
    )
    eixo.tick_params(axis="x", rotation=45)
    eixo.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(PASTA_ASSETS / "matriz_confusao.png", bbox_inches="tight")
    plt.close(fig)


def salvar_curvas_aumento():
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))

    eixos[0].plot(
        EPOCAS, 100 * ACURACIA_AUMENTO_TREINO, marker="o", label="Treino"
    )
    eixos[0].plot(
        EPOCAS,
        100 * ACURACIA_AUMENTO_VALIDACAO,
        marker="o",
        label="Validação",
    )
    eixos[0].set(
        title="Acurácia com aumento de dados",
        xlabel="Época",
        ylabel="Acurácia (%)",
        xticks=EPOCAS,
    )
    eixos[0].legend()

    eixos[1].plot(
        EPOCAS, PERDA_AUMENTO_TREINO, marker="o", label="Treino"
    )
    eixos[1].plot(
        EPOCAS, PERDA_AUMENTO_VALIDACAO, marker="o", label="Validação"
    )
    eixos[1].set(
        title="Perda com aumento de dados",
        xlabel="Época",
        ylabel="Perda",
        xticks=EPOCAS,
    )
    eixos[1].legend()

    fig.suptitle("CNN com data augmentation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PASTA_ASSETS / "curvas_aumento_dados.png", bbox_inches="tight")
    plt.close(fig)


def salvar_comparacao():
    fig, eixo = plt.subplots(figsize=(7.5, 4.5))
    eixo.plot(
        EPOCAS,
        100 * ACURACIA_VALIDACAO,
        marker="o",
        label="Modelo original",
    )
    eixo.plot(
        EPOCAS,
        100 * ACURACIA_AUMENTO_VALIDACAO,
        marker="o",
        label="Com aumento de dados",
    )
    eixo.set(
        title="Comparação da acurácia de validação",
        xlabel="Época",
        ylabel="Acurácia (%)",
        xticks=EPOCAS,
    )
    eixo.legend()
    fig.tight_layout()
    fig.savefig(PASTA_ASSETS / "comparacao_modelos.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    aplicar_estilo()
    salvar_curvas_principais()
    salvar_matriz_confusao()
    salvar_curvas_aumento()
    salvar_comparacao()
    acuracia = np.trace(MATRIZ_CONFUSAO) / MATRIZ_CONFUSAO.sum()
    print(f"Acurácia calculada da matriz: {100 * acuracia:.2f}%")
