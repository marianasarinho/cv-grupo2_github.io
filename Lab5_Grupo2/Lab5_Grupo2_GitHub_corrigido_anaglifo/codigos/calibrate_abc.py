import os
import glob
import cv2
import numpy as np

# ============================================================
# LAB 5 - CALIBRACAO ESTEREO COM IMAGENS PROPRIAS
# Salve este arquivo como: calibrate_abc.py
# Execute no Linux com:    python3 calibrate_abc.py
# ============================================================

# Numero de cantos internos do tabuleiro: colunas x linhas.
CHECKERBOARD = (6, 8)

# Tamanho real do lado do quadrado do tabuleiro.
# Se voce nao souber, deixe 1.0. Nesse caso, a translacao T sai em "unidades de quadrado".
# Se souber, use por exemplo 25.0 para 25 mm.
SQUARE_SIZE = 1.0

LEFT_DIR = "./data/stereoL"
RIGHT_DIR = "./data/stereoR"

OUTPUT_XML = "stereo_params_abc.xml"
OUTPUT_TXT = "calibracao_resultados_abc.txt"
OUTPUT_CORNERS_DIR = "./data/corners_detected"


def listar_imagens(pasta):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(pasta, ext)))
    return sorted(files)


def matriz_para_texto(nome, M):
    return f"{nome} =\n{np.array2string(M, precision=8, suppress_small=False)}\n\n"


def main():
    os.makedirs(OUTPUT_CORNERS_DIR, exist_ok=True)

    images_l = listar_imagens(LEFT_DIR)
    images_r = listar_imagens(RIGHT_DIR)

    if len(images_l) == 0 or len(images_r) == 0:
        raise RuntimeError(
            "Nao encontrei imagens. Verifique se existem arquivos em "
            "./data/stereoL e ./data/stereoR."
        )

    if len(images_l) != len(images_r):
        raise RuntimeError(
            f"Quantidade diferente de imagens: esquerda={len(images_l)}, direita={len(images_r)}."
        )

    print(f"Encontrados {len(images_l)} pares de imagens.")

    cols, rows = CHECKERBOARD
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    image_size = None

    criteria_subpix = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for idx, (file_l, file_r) in enumerate(zip(images_l, images_r), start=1):
        img_l = cv2.imread(file_l)
        img_r = cv2.imread(file_r)

        if img_l is None or img_r is None:
            print(f"Par {idx:02d}: erro ao ler imagem. Ignorando.")
            continue

        if img_l.shape[:2] != img_r.shape[:2]:
            print(f"Par {idx:02d}: tamanhos diferentes. Ignorando.")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        image_size = (gray_l.shape[1], gray_l.shape[0])

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, flags)

        if found_l and found_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)

            objpoints.append(objp.copy())
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

            draw_l = img_l.copy()
            draw_r = img_r.copy()
            cv2.drawChessboardCorners(draw_l, CHECKERBOARD, corners_l, found_l)
            cv2.drawChessboardCorners(draw_r, CHECKERBOARD, corners_r, found_r)
            cv2.imwrite(os.path.join(OUTPUT_CORNERS_DIR, f"left_{idx:02d}.png"), draw_l)
            cv2.imwrite(os.path.join(OUTPUT_CORNERS_DIR, f"right_{idx:02d}.png"), draw_r)

            print(f"Par {idx:02d}: cantos detectados.")
        else:
            print(f"Par {idx:02d}: cantos NAO detectados nas duas imagens. Ignorando.")

    if len(objpoints) < 8:
        raise RuntimeError(
            f"Foram aceitos apenas {len(objpoints)} pares. "
            "Para calibrar bem, tente pelo menos 10 pares bons."
        )

    print("\nCalibrando camera esquerda...")
    rms_l, M1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )

    print("Calibrando camera direita...")
    rms_r, M2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )

    print("Calibrando par estereo...")
    criteria_stereo = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )

    flags_stereo = cv2.CALIB_FIX_INTRINSIC

    rms_stereo, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        M1,
        D1,
        M2,
        D2,
        image_size,
        criteria=criteria_stereo,
        flags=flags_stereo,
    )

    print("Calculando retificacao estereo...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        M1, D1, M2, D2, image_size, R, T, alpha=0
    )

    baseline = float(np.linalg.norm(T))

    fs = cv2.FileStorage(OUTPUT_XML, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Nao foi possivel criar {OUTPUT_XML}")

    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("board_cols", int(cols))
    fs.write("board_rows", int(rows))
    fs.write("square_size", float(SQUARE_SIZE))
    fs.write("num_valid_pairs", int(len(objpoints)))
    fs.write("rms_left", float(rms_l))
    fs.write("rms_right", float(rms_r))
    fs.write("rms_stereo", float(rms_stereo))
    fs.write("M1", M1)
    fs.write("D1", D1)
    fs.write("M2", M2)
    fs.write("D2", D2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.write("roi1", np.array(roi1, dtype=np.int32).reshape(1, 4))
    fs.write("roi2", np.array(roi2, dtype=np.int32).reshape(1, 4))
    fs.release()

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("LAB 5 - RESULTADOS DA CALIBRACAO ESTEREO\n")
        f.write("================================================\n\n")
        f.write(f"Tamanho da imagem: {image_size[0]} x {image_size[1]} pixels\n")
        f.write(f"Tabuleiro: {cols} x {rows} cantos internos\n")
        f.write(f"Tamanho do quadrado usado: {SQUARE_SIZE}\n")
        f.write(f"Pares validos usados: {len(objpoints)}\n\n")
        f.write(f"Erro RMS camera esquerda: {rms_l}\n")
        f.write(f"Erro RMS camera direita: {rms_r}\n")
        f.write(f"Erro RMS stereoCalibrate: {rms_stereo}\n")
        f.write(f"Baseline ||T||: {baseline}\n\n")
        f.write(matriz_para_texto("M1 - matriz intrinseca da camera esquerda", M1))
        f.write(matriz_para_texto("D1 - coeficientes de distorcao da esquerda", D1))
        f.write(matriz_para_texto("M2 - matriz intrinseca da camera direita", M2))
        f.write(matriz_para_texto("D2 - coeficientes de distorcao da direita", D2))
        f.write(matriz_para_texto("R - rotacao entre as cameras", R))
        f.write(matriz_para_texto("T - translacao entre as cameras", T))
        f.write(matriz_para_texto("E - matriz essencial", E))
        f.write(matriz_para_texto("F - matriz fundamental", F))
        f.write(matriz_para_texto("R1 - retificacao esquerda", R1))
        f.write(matriz_para_texto("R2 - retificacao direita", R2))
        f.write(matriz_para_texto("P1 - nova matriz de projecao esquerda", P1))
        f.write(matriz_para_texto("P2 - nova matriz de projecao direita", P2))
        f.write(matriz_para_texto("Q - matriz de reprojecao 3D", Q))
        f.write(f"roi1 = {roi1}\n")
        f.write(f"roi2 = {roi2}\n\n")
        f.write("Parametros salvos no XML:\n")
        f.write("image_width, image_height, board_cols, board_rows, square_size, num_valid_pairs,\n")
        f.write("rms_left, rms_right, rms_stereo, M1, D1, M2, D2, R, T, E, F,\n")
        f.write("R1, R2, P1, P2, Q, roi1, roi2.\n")

    print("\nCalibracao concluida!")
    print(f"Arquivo XML salvo em: {OUTPUT_XML}")
    print(f"Resultados em texto salvos em: {OUTPUT_TXT}")
    print(f"Imagens com cantos desenhados em: {OUTPUT_CORNERS_DIR}")


if __name__ == "__main__":
    main()
