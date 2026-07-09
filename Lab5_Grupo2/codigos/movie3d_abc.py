import time
import cv2
import numpy as np

# ============================================================
# LAB 5 - VISUALIZACAO 3D AO VIVO COM ANAGLIFO E DISPARIDADE
# Salve este arquivo como: movie3d_abc.py
# Execute no Linux com:    python3 movie3d_abc.py
# ============================================================

CAM_LEFT_ID = 0
CAM_RIGHT_ID = 2
CALIB_FILE = "stereo_params_abc.xml"

FPS = 30

# Parametros simples para o mapa de disparidade.
# numDisparities precisa ser multiplo de 16.
NUM_DISPARITIES = 16 * 8
BLOCK_SIZE = 7


def abrir_camera(camera_id, width, height):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def carregar_calibracao(nome_arquivo):
    fs = cv2.FileStorage(nome_arquivo, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o arquivo de calibracao: {nome_arquivo}")

    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())

    M1 = fs.getNode("M1").mat()
    D1 = fs.getNode("D1").mat()
    M2 = fs.getNode("M2").mat()
    D2 = fs.getNode("D2").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    Q = fs.getNode("Q").mat()
    fs.release()

    return (width, height), M1, D1, M2, D2, R1, R2, P1, P2, Q


def criar_anaglifo(left_rect, right_rect):
    """Anaglifo vermelho-ciano: esquerda no canal vermelho, direita em azul+verde."""
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    return cv2.merge([right_gray, right_gray, left_gray])


def main():
    image_size, M1, D1, M2, D2, R1, R2, P1, P2, Q = carregar_calibracao(CALIB_FILE)
    width, height = image_size

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        M1, D1, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        M2, D2, R2, P2, image_size, cv2.CV_16SC2
    )

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 1 * BLOCK_SIZE ** 2,
        P2=32 * 1 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    cam_l = abrir_camera(CAM_LEFT_ID, width, height)
    cam_r = abrir_camera(CAM_RIGHT_ID, width, height)

    if cam_l is None or cam_r is None:
        if cam_l is not None:
            cam_l.release()
        if cam_r is not None:
            cam_r.release()
        raise RuntimeError("Nao foi possivel abrir as duas cameras.")

    print("Visualizacao ao vivo iniciada.")
    print("Teclas:")
    print("  ESC ou Q = sair")
    print("  S = salvar frame anaglifo")

    frame_id = 0

    try:
        while True:
            ret_l, frame_l = cam_l.read()
            ret_r, frame_r = cam_r.read()

            if not ret_l or frame_l is None or not ret_r or frame_r is None:
                print("Falha ao ler uma das cameras.")
                time.sleep(0.1)
                continue

            if frame_l.shape[1] != width or frame_l.shape[0] != height:
                frame_l = cv2.resize(frame_l, image_size)
            if frame_r.shape[1] != width or frame_r.shape[0] != height:
                frame_r = cv2.resize(frame_r, image_size)

            rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

            anaglyph = criar_anaglifo(rect_l, rect_r)

            gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)

            cv2.imshow("Anaglifo 3D ao vivo", anaglyph)
            cv2.imshow("Disparidade", disp_vis)
            cv2.imshow("Esquerda retificada", rect_l)
            cv2.imshow("Direita retificada", rect_r)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                frame_id += 1
                nome = f"anaglifo_frame_{frame_id:02d}.png"
                cv2.imwrite(nome, anaglyph)
                print(f"Frame salvo: {nome}")

    finally:
        cam_l.release()
        cam_r.release()
        cv2.destroyAllWindows()

    print("Finalizado.")


if __name__ == "__main__":
    main()
