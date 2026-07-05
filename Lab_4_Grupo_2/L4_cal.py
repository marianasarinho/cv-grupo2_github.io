import cv2
import numpy as np
import glob


CHECKERBOARD = (6, 8)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

objpoints = []
imgpoints = []

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

images = sorted(glob.glob("frm*.jpg"))

if len(images) == 0:
    print("Nenhuma imagem frm*.jpg foi encontrada.")
    exit()

image_size = None
imagens_validas = []

flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_FAST_CHECK
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

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags
    )

    if ret:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners2)
        imagens_validas.append(fname)

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners2,
            ret
        )

        print(fname, "- cantos encontrados")

        cv2.imshow("Detecção dos cantos", img)
        cv2.waitKey(300)

    else:
        print(fname, "- tabuleiro não detectado")

cv2.destroyAllWindows()

print()
print("Imagens analisadas:", len(images))
print("Imagens utilizadas:", len(imagens_validas))

if len(objpoints) < 5:
    print("Poucas imagens válidas para realizar a calibração.")
    exit()

rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
)

print()
print("Erro RMS da calibração:")
print(rms)

print()
print("Matriz intrínseca da câmera:")
print(camera_matrix)

print()
print("Coeficientes de distorção:")
print(dist_coeffs)

print()
print("Vetores de rotação:")
for i, rvec in enumerate(rvecs):
    print("Imagem", i, ":")
    print(rvec)

print()
print("Vetores de translação:")
for i, tvec in enumerate(tvecs):
    print("Imagem", i, ":")
    print(tvec)


erro_total = 0

for i in range(len(objpoints)):

    pontos_reprojetados, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    erro = cv2.norm(
        imgpoints[i],
        pontos_reprojetados,
        cv2.NORM_L2
    ) / len(pontos_reprojetados)

    erro_total += erro

erro_medio = erro_total / len(objpoints)

print()
print("Erro médio de reprojeção:")
print(erro_medio)


np.savez(
    "parametros_calibracao.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=np.asarray(rvecs),
    tvecs=np.asarray(tvecs),
    rms=rms,
    erro_medio=erro_medio
)

print()
print("Parâmetros salvos em parametros_calibracao.npz")


imagem_teste = cv2.imread(imagens_validas[0])

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

cv2.imwrite(
    "imagem_original_calibracao.jpg",
    imagem_teste
)

cv2.imwrite(
    "imagem_corrigida_calibracao.jpg",
    imagem_corrigida
)

print("Imagem original salva em imagem_original_calibracao.jpg")
print("Imagem corrigida salva em imagem_corrigida_calibracao.jpg")

cv2.imshow("Imagem original", imagem_teste)
cv2.imshow("Imagem corrigida", imagem_corrigida)

cv2.waitKey(0)
cv2.destroyAllWindows()
