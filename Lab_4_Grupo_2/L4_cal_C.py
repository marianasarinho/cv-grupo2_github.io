import cv2
import numpy as np
import glob
import re
from pathlib import Path


CHECKERBOARD = (6, 8)
PASTA_IMAGENS = Path("Capturas_2")
PASTA_CANTOS = Path("cam2_cantos_detectados")
PASTA_CANTOS.mkdir(exist_ok=True)

ARQUIVO_PARAMETROS = "cam2_parametros_calibracao.npz"
ARQUIVO_RESULTADOS = "cam2_resultados_calibracao.txt"


def ordem_numerica(caminho):
    """Ordena cam2_frm2 antes de cam2_frm10."""
    numeros = re.findall(r"\d+", Path(caminho).stem)
    return int(numeros[-1]) if numeros else -1


criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

objpoints = []
imgpoints = []
nomes_validos = []

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

images = sorted(
    glob.glob(str(PASTA_IMAGENS / "cam2_frm*.jpg")),
    key=ordem_numerica
)

if len(images) == 0:
    print("Nenhuma imagem da segunda câmera foi encontrada.")
    print(f"Verifique a pasta: {PASTA_IMAGENS.resolve()}")
    raise SystemExit

image_size = None

flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
)

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print("Não foi possível abrir:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]
    elif image_size != gray.shape[::-1]:
        print(f"{fname} ignorada: resolução diferente.")
        continue

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags
    )

    if ret:
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        nomes_validos.append(fname)

        imagem_cantos = img.copy()
        cv2.drawChessboardCorners(
            imagem_cantos,
            CHECKERBOARD,
            corners2,
            ret
        )

        nome_saida = PASTA_CANTOS / f"cantos_{Path(fname).name}"
        cv2.imwrite(str(nome_saida), imagem_cantos)

        print(Path(fname).name, "- cantos encontrados")
    else:
        print(Path(fname).name, "- tabuleiro não detectado")

print()
print("Imagens analisadas:", len(images))
print("Imagens utilizadas:", len(nomes_validos))

if len(objpoints) < 10:
    print("Foram encontradas menos de 10 imagens válidas.")
    print("Capture novas imagens antes de calibrar.")
    raise SystemExit

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
skew = camera_matrix[0, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]
aspect_ratio = fx / fy

matrizes_rotacao = []
erros_por_imagem = []

for i in range(len(objpoints)):
    matriz_R, _ = cv2.Rodrigues(rvecs[i])
    matrizes_rotacao.append(matriz_R)

    pontos_reprojetados, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    # O OpenCV 5 pode devolver os dois conjuntos com formatos/canais
    # diferentes, mesmo representando os mesmos pontos 2D.
    # Por isso, ambos são convertidos para matrizes N x 2 float64.
    pontos_detectados = (
        np.asarray(imgpoints[i])
        .reshape(-1, 2)
        .astype(np.float64)
    )

    pontos_reprojetados_2d = (
        np.asarray(pontos_reprojetados)
        .reshape(-1, 2)
        .astype(np.float64)
    )

    # Mesma definição usada no tutorial do OpenCV:
    # norma L2 total dividida pelo número de pontos.
    erro = (
        np.linalg.norm(
            pontos_detectados - pontos_reprojetados_2d
        )
        / len(pontos_reprojetados_2d)
    )

    erros_por_imagem.append(float(erro))

erro_medio = float(np.mean(erros_por_imagem))

print()
print("Erro RMS da calibração:")
print(rms)

print()
print("Matriz intrínseca K:")
print(camera_matrix)

print()
print("Coeficientes de distorção [k1, k2, p1, p2, k3]:")
print(dist_coeffs)

print()
print("Focal length em pixels:")
print("fx =", fx)
print("fy =", fy)

print()
print("Aspect ratio fx/fy:")
print(aspect_ratio)

print()
print("Skew:")
print(skew)

print()
print("Principal point:")
print("cx =", cx)
print("cy =", cy)

for i, (nome, matriz_R, tvec) in enumerate(
    zip(nomes_validos, matrizes_rotacao, tvecs)
):
    print()
    print(f"Imagem {i}: {Path(nome).name}")
    print("Matriz de rotação R:")
    print(matriz_R)
    print("Vetor de translação t:")
    print(tvec)

print()
print("Erro médio de reprojeção:")
print(erro_medio)

np.savez(
    ARQUIVO_PARAMETROS,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=np.asarray(rvecs),
    rotation_matrices=np.asarray(matrizes_rotacao),
    tvecs=np.asarray(tvecs),
    rms=rms,
    erro_medio=erro_medio,
    erros_por_imagem=np.asarray(erros_por_imagem),
    image_size=np.asarray(image_size),
    nomes_imagens=np.asarray(nomes_validos)
)

with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as arquivo:
    arquivo.write("CALIBRAÇÃO DA SEGUNDA CÂMERA\n")
    arquivo.write("=" * 50 + "\n\n")
    arquivo.write(f"Imagens analisadas: {len(images)}\n")
    arquivo.write(f"Imagens utilizadas: {len(nomes_validos)}\n")
    arquivo.write(f"Resolução: {image_size[0]} x {image_size[1]} pixels\n")
    arquivo.write(f"Erro RMS: {rms}\n")
    arquivo.write(f"Erro médio de reprojeção: {erro_medio}\n\n")

    arquivo.write("Matriz intrínseca K:\n")
    arquivo.write(np.array2string(camera_matrix, precision=8))
    arquivo.write("\n\n")

    arquivo.write("Coeficientes de distorção [k1, k2, p1, p2, k3]:\n")
    arquivo.write(np.array2string(dist_coeffs, precision=8))
    arquivo.write("\n\n")

    arquivo.write(f"fx = {fx}\n")
    arquivo.write(f"fy = {fy}\n")
    arquivo.write(f"Aspect ratio fx/fy = {aspect_ratio}\n")
    arquivo.write(f"Skew = {skew}\n")
    arquivo.write(f"Principal point = ({cx}, {cy})\n\n")

    for i, (nome, matriz_R, tvec, erro) in enumerate(
        zip(nomes_validos, matrizes_rotacao, tvecs, erros_por_imagem)
    ):
        arquivo.write(f"Imagem {i}: {Path(nome).name}\n")
        arquivo.write("Matriz R:\n")
        arquivo.write(np.array2string(matriz_R, precision=8))
        arquivo.write("\nVetor t:\n")
        arquivo.write(np.array2string(tvec, precision=8))
        arquivo.write(f"\nErro de reprojeção: {erro}\n\n")

print()
print("Arquivos gerados:")
print(ARQUIVO_PARAMETROS)
print(ARQUIVO_RESULTADOS)
print(PASTA_CANTOS)

imagem_teste = cv2.imread(nomes_validos[0])
h, w = imagem_teste.shape[:2]

nova_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

imagem_corrigida = cv2.undistort(
    imagem_teste,
    camera_matrix,
    dist_coeffs,
    None,
    nova_camera_matrix
)

cv2.imwrite("cam2_imagem_original_calibracao.jpg", imagem_teste)
cv2.imwrite("cam2_imagem_corrigida_calibracao.jpg", imagem_corrigida)

print("cam2_imagem_original_calibracao.jpg")
print("cam2_imagem_corrigida_calibracao.jpg")
