import numpy as np
import cv2 as cv

# ============================================================
# LAB 2 - PARTE 2B
# SIFT + Feature Matching usando duas webcams
#
# IMPORTANTE:
# Este codigo NAO sobrescreve as imagens originais.
# Ele mostra o resultado em video.
# Para salvar um frame do resultado, pressione 's'.
# O arquivo salvo sera resultado_parte2_B_frame.png.
#
# Para sair, pressione 'q'.
# ============================================================

MIN_MATCH_COUNT = 10
CAMERA_0 = 0
CAMERA_1 = 1
OUT_FRAME = "resultado_parte2_B_frame.png"

# Criar detector SIFT
sift = cv.SIFT_create()

# Configurar FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)


def aplicar_sift_matching(frame0, frame1):
    """Aplica SIFT + matching em dois frames e retorna a imagem final."""

    gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)

    if des0 is None or des1 is None:
        return np.hstack((gray0, gray1)), 0, 0, 0

    matches = flann.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    matchesMask = None
    gray1_result = gray1.copy()

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is not None and mask is not None:
            matchesMask = mask.ravel().tolist()

            h, w = gray0.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            gray1_result = cv.polylines(gray1.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2
    )

    resultado = cv.drawMatches(gray0, kp0, gray1_result, kp1, good, None, **draw_params)

    texto = f"kp0={len(kp0)}  kp1={len(kp1)}  good={len(good)}"
    cv.putText(resultado, texto, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return resultado, len(kp0), len(kp1), len(good)


# Abrir as duas cameras
cam0 = cv.VideoCapture(CAMERA_0)
cam1 = cv.VideoCapture(CAMERA_1)

if not cam0.isOpened():
    print("Erro: nao foi possivel abrir a camera 0.")
    exit()

if not cam1.isOpened():
    print("Erro: nao foi possivel abrir a camera 1.")
    print("Tente trocar CAMERA_1 para 2 ou 3.")
    cam0.release()
    exit()

print("Cameras abertas com sucesso.")
print("Pressione 's' para salvar apenas o frame do resultado.")
print("Pressione 'q' para sair.")

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0 or not ret1:
        print("Erro ao capturar frame de uma das cameras.")
        break

    resultado, n0, n1, ngood = aplicar_sift_matching(frame0, frame1)

    cv.imshow("Parte 2B - SIFT em duas webcams", resultado)

    tecla = cv.waitKey(1) & 0xFF

    if tecla == ord('s'):
        cv.imwrite(OUT_FRAME, resultado)
        print(f"Frame do resultado salvo em: {OUT_FRAME}")

    if tecla == ord('q'):
        print("Programa encerrado.")
        break

cam0.release()
cam1.release()
cv.destroyAllWindows()
