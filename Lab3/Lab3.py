import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# LAB 3 - CAPTURA SIMULTÂNEA COM DUAS WEBCAMS
#
# Teclas:
#   S -> salva imagem1.jpg e imagem2.jpg
#   Q -> encerra o programa
#
# Caso as câmeras não abram, troque os índices abaixo.
# Exemplos comuns:
#   câmera integrada = 0
#   câmera USB = 1 ou 2
# ============================================================

CAMERA_1 = 0
CAMERA_2 = 1

NOME_IMAGEM_1 = "imagem1.jpg"
NOME_IMAGEM_2 = "imagem2.jpg"

LARGURA = 640
ALTURA = 480


def abrir_camera(indice):
    """
    Tenta abrir uma câmera pelo índice informado.
    No Windows, utiliza inicialmente o backend DirectShow.
    """
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(indice)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)

    return cap


def ajustar_altura(imagem, altura_alvo):
    """
    Redimensiona a imagem mantendo a proporção.
    """
    altura, largura = imagem.shape[:2]

    if altura == altura_alvo:
        return imagem

    nova_largura = int(largura * altura_alvo / altura)
    return cv2.resize(imagem, (nova_largura, altura_alvo))


def adicionar_texto(imagem, texto):
    """
    Adiciona uma legenda na parte superior da imagem.
    """
    resultado = imagem.copy()

    cv2.rectangle(
        resultado,
        (0, 0),
        (resultado.shape[1], 40),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        resultado,
        texto,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return resultado


def main():
    cap1 = abrir_camera(CAMERA_1)
    cap2 = abrir_camera(CAMERA_2)

    if not cap1.isOpened():
        print(f"Erro: não foi possível abrir a câmera {CAMERA_1}.")
        print("Tente trocar CAMERA_1 para outro índice, como 1 ou 2.")

    if not cap2.isOpened():
        print(f"Erro: não foi possível abrir a câmera {CAMERA_2}.")
        print("Tente trocar CAMERA_2 para outro índice, como 0, 2 ou 3.")

    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        return

    print("Duas câmeras abertas com sucesso.")
    print("Pressione S para salvar as duas imagens.")
    print("Pressione Q para sair.")

    while True:
        sucesso1, frame1 = cap1.read()
        sucesso2, frame2 = cap2.read()

        if not sucesso1:
            print("Erro ao capturar imagem da câmera 1.")
            break

        if not sucesso2:
            print("Erro ao capturar imagem da câmera 2.")
            break

        frame1_exibicao = adicionar_texto(
            frame1,
            "Camera 1 - imagem1.jpg"
        )

        frame2_exibicao = adicionar_texto(
            frame2,
            "Camera 2 - imagem2.jpg"
        )

        altura_comum = min(
            frame1_exibicao.shape[0],
            frame2_exibicao.shape[0]
        )

        frame1_exibicao = ajustar_altura(
            frame1_exibicao,
            altura_comum
        )

        frame2_exibicao = ajustar_altura(
            frame2_exibicao,
            altura_comum
        )

        visualizacao = np.hstack(
            (frame1_exibicao, frame2_exibicao)
        )

        cv2.putText(
            visualizacao,
            "S: salvar as duas imagens | Q: sair",
            (10, visualizacao.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Lab 3 - Captura com duas webcams",
            visualizacao
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("s"):
            caminho1 = Path(NOME_IMAGEM_1)
            caminho2 = Path(NOME_IMAGEM_2)

            salvou1 = cv2.imwrite(str(caminho1), frame1)
            salvou2 = cv2.imwrite(str(caminho2), frame2)

            if salvou1 and salvou2:
                horario = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                print("\nImagens salvas com sucesso!")
                print(f"{NOME_IMAGEM_1}: {caminho1.resolve()}")
                print(f"{NOME_IMAGEM_2}: {caminho2.resolve()}")
                print(f"Horário da captura: {horario}")

                aviso = visualizacao.copy()

                cv2.rectangle(
                    aviso,
                    (0, 0),
                    (aviso.shape[1], 65),
                    (0, 120, 0),
                    -1
                )

                cv2.putText(
                    aviso,
                    "IMAGENS SALVAS COM SUCESSO",
                    (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(
                    "Lab 3 - Captura com duas webcams",
                    aviso
                )

                cv2.waitKey(1200)

            else:
                print("Erro ao salvar uma ou ambas as imagens.")

        elif tecla == ord("q"):
            print("Programa encerrado pelo usuário.")
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
