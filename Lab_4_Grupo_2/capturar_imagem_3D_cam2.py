import cv2 as cv
from pathlib import Path

CAMERA_INDEX = 0
ARQUIVO_SAIDA = Path("cam2_imagem_colorida.jpg")

cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    print("Tente alterar CAMERA_INDEX para 1 ou 2.")
    raise SystemExit

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

print("Posicione a câmera diante de uma cena colorida com linhas retas.")
print("Pressione 's' para salvar a imagem e encerrar.")
print("Pressione 'q' para sair sem salvar.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Não foi possível receber a imagem da câmera.")
        break

    preview = frame.copy()
    cv.putText(
        preview,
        "s: salvar | q: sair",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv.imshow("Imagem colorida - Camera 2", preview)

    tecla = cv.waitKey(1) & 0xFF

    if tecla == ord("s"):
        if cv.imwrite(str(ARQUIVO_SAIDA), frame):
            print(f"Imagem salva em: {ARQUIVO_SAIDA.resolve()}")
        else:
            print("Erro ao salvar a imagem.")
        break

    if tecla == ord("q"):
        print("Captura cancelada.")
        break

cap.release()
cv.destroyAllWindows()
