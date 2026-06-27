import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# LAB 3 - PARTE B
# DUAS WEBCAMS + SIFT + MATCHING + RANSAC + MOSAICO EM TEMPO REAL
#
# Teclas:
#   Q -> sair
#   S -> salvar as imagens e os resultados atuais
#
# Caso uma câmera não abra, altere:
#   CAMERA_1 = 0
#   CAMERA_2 = 1
# ============================================================

CAMERA_1 = 0
CAMERA_2 = 1

LARGURA = 640
ALTURA = 480

# Processa SIFT a cada N quadros para reduzir travamentos.
PROCESSAR_A_CADA = 3

# Parâmetros
LIMIAR_LOWE = 0.75
LIMIAR_RANSAC = 5.0
MINIMO_MATCHES = 4

# Limites para impedir mosaicos absurdamente grandes.
MAX_LARGURA_MOSAICO = 3000
MAX_ALTURA_MOSAICO = 2000


def abrir_camera(indice):
    """
    Abre uma webcam pelo índice informado.
    """
    cap = cv2.VideoCapture(indice)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def redimensionar_por_largura(imagem, largura_alvo):
    """
    Redimensiona uma imagem mantendo a proporção.
    """
    altura, largura = imagem.shape[:2]

    if largura == largura_alvo:
        return imagem

    escala = largura_alvo / largura
    nova_altura = max(1, int(altura * escala))

    return cv2.resize(
        imagem,
        (largura_alvo, nova_altura),
        interpolation=cv2.INTER_AREA
    )


def redimensionar_por_altura(imagem, altura_alvo):
    """
    Redimensiona uma imagem mantendo a proporção.
    """
    altura, largura = imagem.shape[:2]

    if altura == altura_alvo:
        return imagem

    escala = altura_alvo / altura
    nova_largura = max(1, int(largura * escala))

    return cv2.resize(
        imagem,
        (nova_largura, altura_alvo),
        interpolation=cv2.INTER_AREA
    )


def escrever_texto(imagem, texto, posicao=(10, 30),
                   cor=(255, 255, 255), escala=0.65):
    """
    Escreve texto com contorno para facilitar a leitura.
    """
    resultado = imagem.copy()

    cv2.putText(
        resultado,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        resultado,
        texto,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        cor,
        2,
        cv2.LINE_AA
    )

    return resultado


def tela_duas_cameras(frame1, frame2, mensagem):
    """
    Mostra as duas câmeras lado a lado.
    É usada quando ainda não existe uma homografia válida.
    """
    altura_comum = min(frame1.shape[0], frame2.shape[0])

    lado1 = redimensionar_por_altura(frame1, altura_comum)
    lado2 = redimensionar_por_altura(frame2, altura_comum)

    tela = np.hstack((lado1, lado2))

    tela = escrever_texto(
        tela,
        mensagem,
        posicao=(10, 30),
        cor=(0, 255, 255),
        escala=0.65
    )

    tela = escrever_texto(
        tela,
        "Q: sair | S: salvar",
        posicao=(10, tela.shape[0] - 15),
        cor=(255, 255, 255),
        escala=0.6
    )

    return tela


def criar_mosaico(imagem1, imagem2, H):
    """
    Projeta a imagem1 no plano da imagem2 usando a homografia H.
    """
    altura1, largura1 = imagem1.shape[:2]
    altura2, largura2 = imagem2.shape[:2]

    cantos1 = np.float32([
        [0, 0],
        [largura1, 0],
        [largura1, altura1],
        [0, altura1]
    ]).reshape(-1, 1, 2)

    cantos2 = np.float32([
        [0, 0],
        [largura2, 0],
        [largura2, altura2],
        [0, altura2]
    ]).reshape(-1, 1, 2)

    cantos1_transformados = cv2.perspectiveTransform(
        cantos1,
        H
    )

    todos_cantos = np.concatenate(
        (cantos1_transformados, cantos2),
        axis=0
    )

    x_min, y_min = np.floor(
        todos_cantos.min(axis=0).ravel()
    ).astype(int)

    x_max, y_max = np.ceil(
        todos_cantos.max(axis=0).ravel()
    ).astype(int)

    largura_canvas = x_max - x_min
    altura_canvas = y_max - y_min

    if largura_canvas <= 0 or altura_canvas <= 0:
        raise ValueError("Canvas inválido.")

    if (
        largura_canvas > MAX_LARGURA_MOSAICO
        or altura_canvas > MAX_ALTURA_MOSAICO
    ):
        raise ValueError("Homografia instável: mosaico muito grande.")

    deslocamento_x = -x_min
    deslocamento_y = -y_min

    matriz_deslocamento = np.array([
        [1.0, 0.0, deslocamento_x],
        [0.0, 1.0, deslocamento_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    mosaico = cv2.warpPerspective(
        imagem1,
        matriz_deslocamento @ H,
        (largura_canvas, altura_canvas)
    )

    x1 = deslocamento_x
    y1 = deslocamento_y
    x2 = x1 + largura2
    y2 = y1 + altura2

    if (
        x1 < 0
        or y1 < 0
        or x2 > mosaico.shape[1]
        or y2 > mosaico.shape[0]
    ):
        raise ValueError("Imagem 2 ficou fora do canvas.")

    mosaico[y1:y2, x1:x2] = imagem2

    return mosaico


def montar_tela_resultado(imagem_matches, mosaico, status):
    """
    Coloca as correspondências na parte superior
    e o mosaico na parte inferior.
    """
    largura_final = max(
        imagem_matches.shape[1],
        mosaico.shape[1]
    )

    matches_ajustado = redimensionar_por_largura(
        imagem_matches,
        largura_final
    )

    mosaico_ajustado = redimensionar_por_largura(
        mosaico,
        largura_final
    )

    matches_ajustado = escrever_texto(
        matches_ajustado,
        "Correspondencias SIFT aceitas pelo RANSAC",
        posicao=(10, 30),
        cor=(0, 255, 0),
        escala=0.65
    )

    mosaico_ajustado = escrever_texto(
        mosaico_ajustado,
        "Mosaico em tempo real",
        posicao=(10, 30),
        cor=(0, 255, 255),
        escala=0.65
    )

    tela = np.vstack(
        (matches_ajustado, mosaico_ajustado)
    )

    tela = escrever_texto(
        tela,
        status,
        posicao=(10, tela.shape[0] - 15),
        cor=(255, 255, 255),
        escala=0.58
    )

    return tela


def salvar_resultados(frame1, frame2, tela,
                      imagem_matches=None, mosaico=None):
    """
    Salva os resultados atuais em uma nova pasta.
    """
    horario = datetime.now().strftime("%Y%m%d_%H%M%S")

    pasta = Path(
        f"resultado_lab3_tempo_real_{horario}"
    )

    pasta.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(pasta / "camera1.jpg"),
        frame1
    )

    cv2.imwrite(
        str(pasta / "camera2.jpg"),
        frame2
    )

    cv2.imwrite(
        str(pasta / "tela_completa.jpg"),
        tela
    )

    if imagem_matches is not None:
        cv2.imwrite(
            str(pasta / "correspondencias.jpg"),
            imagem_matches
        )

    if mosaico is not None:
        cv2.imwrite(
            str(pasta / "mosaico.jpg"),
            mosaico
        )

    print(f"Resultados salvos em: {pasta.resolve()}")


def main():
    cap1 = abrir_camera(CAMERA_1)
    cap2 = abrir_camera(CAMERA_2)

    if not cap1.isOpened():
        print(f"Erro: câmera {CAMERA_1} não abriu.")

    if not cap2.isOpened():
        print(f"Erro: câmera {CAMERA_2} não abriu.")

    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        print("Altere CAMERA_1 e CAMERA_2 no início do código.")
        return

    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("Erro: sua instalação do OpenCV não possui SIFT.")
        print("Instale com: python3 -m pip install opencv-contrib-python")
        cap1.release()
        cap2.release()
        return

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    contador = 0

    tela_atual = None
    frame1_atual = None
    frame2_atual = None
    matches_atual = None
    mosaico_atual = None

    print("Duas câmeras abertas com sucesso.")
    print("Q: sair")
    print("S: salvar resultados")

    while True:
        sucesso1, frame1 = cap1.read()
        sucesso2, frame2 = cap2.read()

        if not sucesso1 or not sucesso2:
            print("Erro ao ler uma das câmeras.")
            break

        frame1_atual = frame1.copy()
        frame2_atual = frame2.copy()

        contador += 1

        if contador % PROCESSAR_A_CADA == 0 or tela_atual is None:
            cinza1 = cv2.cvtColor(
                frame1,
                cv2.COLOR_BGR2GRAY
            )

            cinza2 = cv2.cvtColor(
                frame2,
                cv2.COLOR_BGR2GRAY
            )

            pontos1, descritores1 = sift.detectAndCompute(
                cinza1,
                None
            )

            pontos2, descritores2 = sift.detectAndCompute(
                cinza2,
                None
            )

            status = (
                f"SIFT1: {len(pontos1)} | "
                f"SIFT2: {len(pontos2)}"
            )

            if descritores1 is None or descritores2 is None:
                tela_atual = tela_duas_cameras(
                    frame1,
                    frame2,
                    status + " | Sem descritores suficientes"
                )

                matches_atual = None
                mosaico_atual = None

            else:
                pares = matcher.knnMatch(
                    descritores1,
                    descritores2,
                    k=2
                )

                bons_matches = []

                for par in pares:
                    if len(par) < 2:
                        continue

                    primeiro, segundo = par

                    if (
                        primeiro.distance
                        < LIMIAR_LOWE * segundo.distance
                    ):
                        bons_matches.append(primeiro)

                status += f" | Matches: {len(bons_matches)}"

                if len(bons_matches) < MINIMO_MATCHES:
                    tela_atual = tela_duas_cameras(
                        frame1,
                        frame2,
                        status + " | Poucos matches"
                    )

                    matches_atual = None
                    mosaico_atual = None

                else:
                    origem = np.float32([
                        pontos1[m.queryIdx].pt
                        for m in bons_matches
                    ]).reshape(-1, 1, 2)

                    destino = np.float32([
                        pontos2[m.trainIdx].pt
                        for m in bons_matches
                    ]).reshape(-1, 1, 2)

                    H, mascara = cv2.findHomography(
                        origem,
                        destino,
                        cv2.RANSAC,
                        LIMIAR_RANSAC
                    )

                    if H is None or mascara is None:
                        tela_atual = tela_duas_cameras(
                            frame1,
                            frame2,
                            status + " | Homografia não encontrada"
                        )

                        matches_atual = None
                        mosaico_atual = None

                    else:
                        mascara = mascara.ravel().astype(np.uint8)

                        inliers = int(
                            np.count_nonzero(mascara)
                        )

                        outliers = int(
                            len(bons_matches) - inliers
                        )

                        status += (
                            f" | Inliers: {inliers}"
                            f" | Outliers: {outliers}"
                        )

                        imagem_matches = cv2.drawMatches(
                            frame1,
                            pontos1,
                            frame2,
                            pontos2,
                            bons_matches,
                            None,
                            matchColor=(0, 255, 0),
                            singlePointColor=None,
                            matchesMask=mascara.tolist(),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )

                        try:
                            mosaico = criar_mosaico(
                                frame1,
                                frame2,
                                H
                            )

                            tela_atual = montar_tela_resultado(
                                imagem_matches,
                                mosaico,
                                status + " | Q: sair | S: salvar"
                            )

                            matches_atual = imagem_matches.copy()
                            mosaico_atual = mosaico.copy()

                        except ValueError as erro:
                            tela_atual = tela_duas_cameras(
                                frame1,
                                frame2,
                                status + f" | {erro}"
                            )

                            matches_atual = imagem_matches.copy()
                            mosaico_atual = None

        tela_exibicao = tela_atual

        altura_maxima = 900

        if tela_exibicao.shape[0] > altura_maxima:
            tela_exibicao = redimensionar_por_altura(
                tela_exibicao,
                altura_maxima
            )

        cv2.imshow(
            "Lab 3 - Parte B - Tempo Real",
            tela_exibicao
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            print("Programa encerrado.")
            break

        if tecla == ord("s"):
            salvar_resultados(
                frame1_atual,
                frame2_atual,
                tela_atual,
                matches_atual,
                mosaico_atual
            )

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
