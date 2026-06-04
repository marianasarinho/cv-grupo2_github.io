import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cam0 = cv.VideoCapture(0)
cam1 = cv.VideoCapture(1)

if not cam0.isOpened():
    print("Erro: nao foi possivel abrir a camera 0.")
    exit()

if not cam1.isOpened():
    print("Erro: nao foi possivel abrir a camera 1.")
    print("Tente trocar o indice da segunda camera para 2 ou 3.")
    cam0.release()
    exit()

print("Cameras abertas com sucesso.")
print("Pressione 's' para salvar as imagens.")
print("Pressione 'q' para sair sem salvar.")

while True:

    # Ler frame da camera 0
    ret0, frame0 = cam0.read()

    # Ler frame da camera 1
    ret1, frame1 = cam1.read()

    # Verificar se a camera 0 capturou corretamente
    if not ret0:
        print("Erro ao capturar imagem da camera 0.")
        break

    # Verificar se a camera 1 capturou corretamente
    if not ret1:
        print("Erro ao capturar imagem da camera 1.")
        break

    # Mostrar as imagens em janelas separadas
    cv.imshow("Camera 0", frame0)
    cv.imshow("Camera 1", frame1)

    # Capturar tecla pressionada
    tecla = cv.waitKey(1) & 0xFF

    # Se apertar 's', salvar as duas imagens
    if tecla == ord('s'):

        cv.imwrite("camera0_objeto.png", frame0)
        cv.imwrite("camera1_objeto.png", frame1)

        print("Imagens salvas com sucesso:")
        print("camera0_objeto.png")
        print("camera1_objeto.png")

        break

    # Se apertar 'q', sair sem salvar
    if tecla == ord('q'):

        print("Programa encerrado sem salvar imagens.")

        break

cam0.release()
cam1.release()

cv.destroyAllWindows()

MIN_MATCH_COUNT = 10

img1 = cv.imread('camera0_objeto.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('camera1_objeto.png', cv.IMREAD_GRAYSCALE)  # trainImage
 
sift = cv.SIFT_create()
 
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
 
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
 
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
 
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None        

draw_params = dict(matchColor = (0,255,0), 
                   singlePointColor = None,
                   matchesMask = matchesMask, 
                   flags = 2)
 
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
 
plt.imshow(img3, 'gray'),plt.show()