import cv2 as cv
from pathlib import Path

# Troque para 1 ou 2 se a câmera desejada não abrir.
CAMERA_INDEX = 0

# Pasta exclusiva para as imagens da segunda câmera.
PASTA_SAIDA = Path("Capturas_2")
PASTA_SAIDA.mkdir(exist_ok=True)

# No Windows, CAP_DSHOW costuma abrir a webcam com menos atraso.
cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    print("Tente alterar CAMERA_INDEX para 1 ou 2.")
    raise SystemExit

# Fixamos uma resolução comum. Se a câmera não aceitar, ela usará outra.
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

i = 0

print("Pressione 's' para salvar uma imagem.")
print("Pressione 'q' para encerrar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Não foi possível receber o quadro da câmera.")
        break

    cv.putText(
        frame,
        f"Camera 2 | imagens salvas: {i}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv.imshow("Captura - Camera 2", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord("s"):
        nome = PASTA_SAIDA / f"cam2_frm{i}.jpg"

        if cv.imwrite(str(nome), frame):
            print(f"Imagem salva: {nome}")
            i += 1
        else:
            print(f"Erro ao salvar: {nome}")

    elif key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

print(f"Captura encerrada. Total de imagens salvas: {i}")
