import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ============================================================
# LAB 2 - PARTE 2A
# SIFT + Feature Matching + Homografia usando imagens salvas
#
# IMPORTANTE:
# Este codigo apenas LE as imagens foto1.png e foto2.png.
# Ele NAO sobrescreve as imagens originais.
# O resultado e salvo em resultado_parte2_A.png.
# ============================================================

MIN_MATCH_COUNT = 10

# Imagens tiradas no laboratorio
IMG1_PATH = "foto1.png"   # imagem de consulta
IMG2_PATH = "foto2.png"   # imagem de busca
OUT_PATH = "resultado_parte2_A.png"

# Ler imagens em escala de cinza
img1 = cv.imread(IMG1_PATH, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(IMG2_PATH, cv.IMREAD_GRAYSCALE)

# Verificar se as imagens foram carregadas
if img1 is None:
    print(f"Erro: nao foi possivel abrir {IMG1_PATH}")
    exit()

if img2 is None:
    print(f"Erro: nao foi possivel abrir {IMG2_PATH}")
    exit()

# Criar detector SIFT
sift = cv.SIFT_create()

# Detectar pontos-chave e descritores
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("Erro: nao foram encontrados descritores suficientes.")
    exit()

# Configurar FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Encontrar correspondencias entre os descritores
matches = flann.knnMatch(des1, des2, k=2)

# Aplicar o teste de Lowe para manter boas correspondencias
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print(f"Pontos-chave imagem 1: {len(kp1)}")
print(f"Pontos-chave imagem 2: {len(kp2)}")
print(f"Boas correspondencias: {len(good)}")

# Calcular homografia se houver matches suficientes
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    if M is not None and mask is not None:
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        img2_result = cv.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Homografia nao foi calculada corretamente.")
        matchesMask = None
        img2_result = img2.copy()
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
    img2_result = img2.copy()

# Desenhar correspondencias
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matchesMask,
    flags=2
)

img3 = cv.drawMatches(img1, kp1, img2_result, kp2, good, None, **draw_params)

# Salvar resultado sem alterar as imagens originais
cv.imwrite(OUT_PATH, img3)
print(f"Resultado salvo em: {OUT_PATH}")

# Mostrar resultado
plt.figure(figsize=(14, 7))
plt.imshow(img3, cmap="gray")
plt.title("SIFT + Feature Matching + Homografia")
plt.axis("off")
plt.show()
