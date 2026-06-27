import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# LAB 3 - SIFT, HOMOGRAFIA, RANSAC E MOSAICO
#
# Arquivos de entrada:
#   imagem1.jpg
#   imagem2.jpg
#
# Arquivos gerados:
#   correspondencias_ransac.jpg
#   mosaico_minimos_quadrados.jpg
#   mosaico_ransac.jpg
#
# Execute este arquivo na mesma pasta das imagens.
# ============================================================

IMAGEM_1 = "imagem1.jpg"
IMAGEM_2 = "imagem2.jpg"

LIMIAR_LOWE = 0.75
LIMIAR_REPROJECAO = 5.0
MINIMO_MATCHES = 4


def normalizar_homografia(H):
    """Normaliza H para que H[2, 2] seja igual a 1."""
    if H is None:
        return None

    if abs(H[2, 2]) < 1e-12:
        return H

    return H / H[2, 2]


def criar_mosaico(imagem1, imagem2, H):
    """
    Projeta a imagem1 no plano da imagem2 usando H.
    O canvas é calculado para evitar cortes.
    """
    altura1, largura1 = imagem1.shape[:2]
    altura2, largura2 = imagem2.shape[:2]

    cantos1 = np.float32([
        [0, 0],
        [largura1, 0],
        [largura1, altura1],
        [0, altura1]
    ]).reshape(-1, 1, 2)

    cantos2 = np.float32([
        [0, 0],
        [largura2, 0],
        [largura2, altura2],
        [0, altura2]
    ]).reshape(-1, 1, 2)

    cantos1_transformados = cv2.perspectiveTransform(cantos1, H)
    todos_cantos = np.concatenate(
        (cantos1_transformados, cantos2),
        axis=0
    )

    x_min, y_min = np.floor(
        todos_cantos.min(axis=0).ravel()
    ).astype(int)

    x_max, y_max = np.ceil(
        todos_cantos.max(axis=0).ravel()
    ).astype(int)

    deslocamento_x = -x_min
    deslocamento_y = -y_min

    matriz_deslocamento = np.array([
        [1, 0, deslocamento_x],
        [0, 1, deslocamento_y],
        [0, 0, 1]
    ], dtype=np.float64)

    largura_canvas = x_max - x_min
    altura_canvas = y_max - y_min

    mosaico = cv2.warpPerspective(
        imagem1,
        matriz_deslocamento @ H,
        (largura_canvas, altura_canvas)
    )

    mosaico[
        deslocamento_y:deslocamento_y + altura2,
        deslocamento_x:deslocamento_x + largura2
    ] = imagem2

    return mosaico


def main():
    imagem1 = cv2.imread(IMAGEM_1)
    imagem2 = cv2.imread(IMAGEM_2)

    if imagem1 is None:
        raise FileNotFoundError(
            f"Não foi possível abrir {IMAGEM_1}. "
            "Verifique se está na mesma pasta do código."
        )

    if imagem2 is None:
        raise FileNotFoundError(
            f"Não foi possível abrir {IMAGEM_2}. "
            "Verifique se está na mesma pasta do código."
        )

    cinza1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
    cinza2 = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)

    # PASSO 1 - SIFT
    sift = cv2.SIFT_create()

    pontos1, descritores1 = sift.detectAndCompute(cinza1, None)
    pontos2, descritores2 = sift.detectAndCompute(cinza2, None)

    print(f"Pontos SIFT na imagem 1: {len(pontos1)}")
    print(f"Pontos SIFT na imagem 2: {len(pontos2)}")

    if descritores1 is None or descritores2 is None:
        raise RuntimeError(
            "Não foi possível gerar descritores SIFT. "
            "Use imagens com mais detalhes visuais."
        )

    # PASSO 2 - MATCHING E TESTE DE LOWE
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches_brutos = matcher.knnMatch(
        descritores1,
        descritores2,
        k=2
    )

    bons_matches = []

    for par in matches_brutos:
        if len(par) < 2:
            continue

        primeiro, segundo = par

        if primeiro.distance < LIMIAR_LOWE * segundo.distance:
            bons_matches.append(primeiro)

    print(f"Matches após o teste de Lowe: {len(bons_matches)}")

    if len(bons_matches) < MINIMO_MATCHES:
        raise RuntimeError(
            f"Foram encontrados apenas {len(bons_matches)} bons matches. "
            f"São necessários pelo menos {MINIMO_MATCHES}. "
            "Tire novas fotos com maior sobreposição e mais textura."
        )

    pontos_origem = np.float32([
        pontos1[m.queryIdx].pt for m in bons_matches
    ]).reshape(-1, 1, 2)

    pontos_destino = np.float32([
        pontos2[m.trainIdx].pt for m in bons_matches
    ]).reshape(-1, 1, 2)

    # PASSO 3 - HOMOGRAFIA POR MÍNIMOS QUADRADOS
    H_ls, _ = cv2.findHomography(
        pontos_origem,
        pontos_destino,
        method=0
    )

    if H_ls is None:
        raise RuntimeError(
            "Não foi possível calcular a homografia por mínimos quadrados."
        )

    H_ls = normalizar_homografia(H_ls)

    # PASSO 4 - HOMOGRAFIA POR RANSAC
    H_ransac, mascara_ransac = cv2.findHomography(
        pontos_origem,
        pontos_destino,
        cv2.RANSAC,
        LIMIAR_REPROJECAO
    )

    if H_ransac is None or mascara_ransac is None:
        raise RuntimeError(
            "Não foi possível calcular a homografia com RANSAC."
        )

    H_ransac = normalizar_homografia(H_ransac)
    mascara_ransac = mascara_ransac.ravel().astype(bool)

    inliers = int(np.count_nonzero(mascara_ransac))
    outliers = int(len(bons_matches) - inliers)
    percentual_outliers = 100.0 * outliers / len(bons_matches)

    # RESULTADOS NUMÉRICOS
    np.set_printoptions(precision=6, suppress=True)

    print("\n--- Matriz de Homografia por Mínimos Quadrados ---")
    print(H_ls)

    print("\n--- Matriz de Homografia por RANSAC ---")
    print(H_ransac)

    print("\n--- Resultado do RANSAC ---")
    print(f"Total de bons matches: {len(bons_matches)}")
    print(f"Inliers: {inliers}")
    print(f"Outliers: {outliers}")
    print(f"Porcentagem de outliers: {percentual_outliers:.2f}%")

    diferenca_horizontal = abs(H_ls[0, 2] - H_ransac[0, 2])
    diferenca_vertical = abs(H_ls[1, 2] - H_ransac[1, 2])

    print("\n--- Comparação das translações ---")
    print(f"H_ls[0,2]: {H_ls[0, 2]:.6f}")
    print(f"H_ransac[0,2]: {H_ransac[0, 2]:.6f}")
    print(f"Diferença horizontal: {diferenca_horizontal:.6f}")

    print(f"H_ls[1,2]: {H_ls[1, 2]:.6f}")
    print(f"H_ransac[1,2]: {H_ransac[1, 2]:.6f}")
    print(f"Diferença vertical: {diferenca_vertical:.6f}")

    # PASSO 5 - CORRESPONDÊNCIAS
    imagem_matches = cv2.drawMatches(
        imagem1,
        pontos1,
        imagem2,
        pontos2,
        bons_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mascara_ransac.astype(np.uint8).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        "correspondencias_ransac.jpg",
        imagem_matches
    )

    # PASSO 6 - MOSAICOS
    mosaico_ls = criar_mosaico(
        imagem1,
        imagem2,
        H_ls
    )

    mosaico_ransac = criar_mosaico(
        imagem1,
        imagem2,
        H_ransac
    )

    cv2.imwrite(
        "mosaico_minimos_quadrados.jpg",
        mosaico_ls
    )

    cv2.imwrite(
        "mosaico_ransac.jpg",
        mosaico_ransac
    )

    # EXIBIÇÃO
    figura, eixos = plt.subplots(3, 1, figsize=(14, 18))

    eixos[0].imshow(
        cv2.cvtColor(imagem_matches, cv2.COLOR_BGR2RGB)
    )
    eixos[0].set_title(
        "Correspondências aceitas pelo RANSAC"
    )
    eixos[0].axis("off")

    eixos[1].imshow(
        cv2.cvtColor(mosaico_ls, cv2.COLOR_BGR2RGB)
    )
    eixos[1].set_title(
        "Mosaico por Mínimos Quadrados Tradicionais"
    )
    eixos[1].axis("off")

    eixos[2].imshow(
        cv2.cvtColor(mosaico_ransac, cv2.COLOR_BGR2RGB)
    )
    eixos[2].set_title(
        "Mosaico com Homografia Robusta por RANSAC"
    )
    eixos[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nArquivos gerados com sucesso:")
    print("- correspondencias_ransac.jpg")
    print("- mosaico_minimos_quadrados.jpg")
    print("- mosaico_ransac.jpg")


if __name__ == "__main__":
    main()
