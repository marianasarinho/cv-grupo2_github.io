import os
import time
import cv2

# ============================================================
# LAB 5 - CAPTURA DE IMAGENS PARA CALIBRACAO ESTEREO
# Salve este arquivo como: capture_images_abc.py
# Execute no Linux com:    python3 capture_images_abc.py
# ============================================================

# Ajuste se necessario. No PC do laboratorio, normalmente os IDs sao 0, 1, 2...
CAM_LEFT_ID = 0
CAM_RIGHT_ID = 2

# Numero de cantos internos do tabuleiro: colunas x linhas.
# No exemplo visto em aula estava sendo usado (9, 6).
CHECKERBOARD = (6, 8)

# Pasta de saida. O programa cria data/stereoL e data/stereoR.
OUTPUT_PATH = "./data"

# Quantidade desejada pelo enunciado: entre 10 e 15 pares.
TARGET_PAIRS = 15

# Intervalo minimo entre pares salvos.
CAPTURE_INTERVAL = 5

# Resolucao usada tambem na calibracao e na visualizacao depois.
WIDTH = 640
HEIGHT = 480
FPS = 30


def abrir_camera(camera_id):
    """Abre uma webcam no Linux reduzindo risco de travamento por banda USB."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        return None

    # MJPG ajuda quando usamos duas webcams USB ao mesmo tempo.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def fechar(cam_left, cam_right):
    if cam_left is not None:
        cam_left.release()
    if cam_right is not None:
        cam_right.release()
    cv2.destroyAllWindows()


def verificar_ids(cam_left_id, cam_right_id):
    """Confirma se esquerda/direita estao corretas. Use Y para manter e N para trocar."""
    cam_left = abrir_camera(cam_left_id)
    cam_right = abrir_camera(cam_right_id)

    if cam_left is None:
        raise RuntimeError(f"Nao foi possivel abrir a camera esquerda ID={cam_left_id}")
    if cam_right is None:
        cam_left.release()
        raise RuntimeError(f"Nao foi possivel abrir a camera direita ID={cam_right_id}")

    print("\nVerificacao das cameras")
    print("Clique em uma janela do OpenCV e pressione:")
    print("  Y = manter IDs")
    print("  N = trocar esquerda/direita")
    print("  ESC = sair\n")

    # Aquecimento inicial.
    for _ in range(20):
        cam_left.read()
        cam_right.read()

    while True:
        ret_l, frame_l = cam_left.read()
        ret_r, frame_r = cam_right.read()

        if not ret_l or frame_l is None:
            fechar(cam_left, cam_right)
            raise RuntimeError("Falha ao ler a camera esquerda.")
        if not ret_r or frame_r is None:
            fechar(cam_left, cam_right)
            raise RuntimeError("Falha ao ler a camera direita.")

        prev_l = frame_l.copy()
        prev_r = frame_r.copy()

        cv2.putText(prev_l, f"ESQUERDA ID={cam_left_id}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(prev_r, f"DIREITA ID={cam_right_id}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera esquerda", prev_l)
        cv2.imshow("Camera direita", prev_r)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("y"), ord("Y")):
            print("IDs mantidos.")
            break
        if key in (ord("n"), ord("N")):
            cam_left_id, cam_right_id = cam_right_id, cam_left_id
            print("IDs trocados.")
            break
        if key == 27:
            fechar(cam_left, cam_right)
            raise KeyboardInterrupt("Encerrado pelo usuario.")

    fechar(cam_left, cam_right)
    return cam_left_id, cam_right_id


def capturar(cam_left_id, cam_right_id):
    left_dir = os.path.join(OUTPUT_PATH, "stereoL")
    right_dir = os.path.join(OUTPUT_PATH, "stereoR")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    cam_left = abrir_camera(cam_left_id)
    cam_right = abrir_camera(cam_right_id)

    if cam_left is None or cam_right is None:
        fechar(cam_left, cam_right)
        raise RuntimeError("Nao foi possivel abrir as duas cameras para captura.")

    print("\nCaptura iniciada.")
    print("Mostre o tabuleiro simultaneamente para as duas cameras.")
    print("O par so sera salvo quando os cantos forem detectados nas duas imagens.")
    print("Pressione ESC para parar antes do limite.\n")

    count = 0
    last_capture_time = time.time() - CAPTURE_INTERVAL

    try:
        while True:
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            if not ret_l or frame_l is None or not ret_r or frame_r is None:
                print("Falha em uma leitura. Tentando novamente...")
                time.sleep(0.1)
                continue

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            found_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
            found_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

            show_l = frame_l.copy()
            show_r = frame_r.copy()

            if found_l:
                cv2.drawChessboardCorners(show_l, CHECKERBOARD, corners_l, found_l)
            if found_r:
                cv2.drawChessboardCorners(show_r, CHECKERBOARD, corners_r, found_r)

            elapsed = time.time() - last_capture_time
            remaining = max(0, int(CAPTURE_INTERVAL - elapsed))

            msg = f"Salvas: {count}/{TARGET_PAIRS} | Proxima: {remaining}s"
            cv2.putText(show_l, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
            cv2.putText(show_r, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)

            cv2.imshow("Camera esquerda", show_l)
            cv2.imshow("Camera direita", show_r)

            if elapsed >= CAPTURE_INTERVAL:
                if found_l and found_r:
                    count += 1
                    name = f"img{count:02d}.png"
                    file_l = os.path.join(left_dir, name)
                    file_r = os.path.join(right_dir, name)
                    cv2.imwrite(file_l, frame_l)
                    cv2.imwrite(file_r, frame_r)
                    print(f"Par {count:02d} salvo: {file_l} | {file_r}")
                    last_capture_time = time.time()

                    if count >= TARGET_PAIRS:
                        print("Quantidade alvo atingida.")
                        break
                else:
                    last_capture_time = time.time()
                    print("Tabuleiro nao detectado nas duas cameras. Par nao salvo.")

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Captura encerrada pelo usuario.")
                break

    finally:
        fechar(cam_left, cam_right)

    print(f"Total de pares salvos: {count}")


def main():
    print("LAB 5 - captura para calibracao estereo")
    print("Pressione ENTER para iniciar.")
    input()
    left_id, right_id = verificar_ids(CAM_LEFT_ID, CAM_RIGHT_ID)
    capturar(left_id, right_id)


if __name__ == "__main__":
    main()
