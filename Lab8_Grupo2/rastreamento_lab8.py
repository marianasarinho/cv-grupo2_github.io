"""
LAB 8 - Rastreamento de objetos em video com OpenCV

Este programa possui dois modos:
1. video: rastreia uma ROI selecionada manualmente em um video gravado;
2. webcam: rastreia uma ROI selecionada manualmente ao vivo.

Instalacao:
    python -m pip install opencv-contrib-python

Exemplos:
    python rastreamento_lab8.py --modo video --entrada "video.mp4"
    python rastreamento_lab8.py --modo webcam --camera 0

No modo video, antes da selecao:
    S ou ENTER - selecionar a ROI no quadro atual
    N ou ESPACO - avancar alguns quadros
    B           - voltar aproximadamente 1 segundo
    Q ou ESC    - cancelar

Durante o rastreamento:
    R           - selecionar novamente a ROI
    Q ou ESC    - encerrar
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2


RASTREADORES = ("CSRT", "KCF", "MIL", "MOSSE", "GOTURN")


def analisar_argumentos():
    parser = argparse.ArgumentParser(
        description="Rastreamento de objetos em video ou pela webcam."
    )
    parser.add_argument(
        "--modo",
        required=True,
        choices=("video", "webcam"),
        help="Fonte das imagens: video gravado ou webcam.",
    )
    parser.add_argument(
        "--entrada",
        help="Caminho do video de entrada. Obrigatorio no modo video.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice da webcam. Padrao: 0.",
    )
    parser.add_argument(
        "--saida",
        help=(
            "Caminho do MP4 de saida. Se omitido, sera criado dentro "
            "da pasta resultados."
        ),
    )
    parser.add_argument(
        "--rastreador",
        type=str.upper,
        choices=RASTREADORES,
        default="CSRT",
        help="Algoritmo de rastreamento. Padrao: CSRT.",
    )
    parser.add_argument(
        "--inicio",
        type=float,
        default=0.0,
        help="Instante inicial do video, em segundos. Padrao: 0.",
    )
    parser.add_argument(
        "--goturn-prototxt",
        default="goturn.prototxt",
        help="Arquivo .prototxt do GOTURN.",
    )
    parser.add_argument(
        "--goturn-modelo",
        default="goturn.caffemodel",
        help="Arquivo .caffemodel do GOTURN.",
    )
    args = parser.parse_args()

    if args.modo == "video" and not args.entrada:
        parser.error("--entrada e obrigatorio quando --modo video.")
    if args.inicio < 0:
        parser.error("--inicio nao pode ser negativo.")

    return args


def obter_criador(nome):
    """Procura o construtor do rastreador na API atual ou em cv2.legacy."""
    nome_funcao = f"Tracker{nome}_create"

    if hasattr(cv2, nome_funcao):
        return getattr(cv2, nome_funcao)

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, nome_funcao):
        return getattr(cv2.legacy, nome_funcao)

    return None


def criar_goturn(caminho_prototxt, caminho_modelo):
    """Cria o GOTURN usando os dois arquivos de rede neural."""
    if not os.path.isfile(caminho_prototxt):
        raise FileNotFoundError(
            f"Arquivo do GOTURN nao encontrado: {caminho_prototxt}"
        )
    if not os.path.isfile(caminho_modelo):
        raise FileNotFoundError(
            f"Arquivo do GOTURN nao encontrado: {caminho_modelo}"
        )

    criador = obter_criador("GOTURN")
    if criador is None:
        raise RuntimeError(
            "Esta instalacao do OpenCV nao possui o rastreador GOTURN."
        )

    # As versoes recentes aceitam os caminhos por TrackerGOTURN_Params.
    if hasattr(cv2, "TrackerGOTURN_Params"):
        parametros = cv2.TrackerGOTURN_Params()
        parametros.modelTxt = caminho_prototxt
        parametros.modelBin = caminho_modelo
        return criador(parametros)

    # Em versoes antigas, o construtor usa os nomes padrao no diretorio atual.
    diretorio_anterior = os.getcwd()
    diretorio_modelo = os.path.dirname(os.path.abspath(caminho_modelo))
    nome_modelo = os.path.basename(caminho_modelo)
    nome_prototxt = os.path.basename(caminho_prototxt)

    if nome_modelo != "goturn.caffemodel" or nome_prototxt != "goturn.prototxt":
        raise RuntimeError(
            "Nesta versao do OpenCV, renomeie os arquivos para "
            "goturn.caffemodel e goturn.prototxt e coloque-os na mesma pasta."
        )

    if os.path.dirname(os.path.abspath(caminho_prototxt)) != diretorio_modelo:
        raise RuntimeError("Os dois arquivos do GOTURN devem estar na mesma pasta.")

    try:
        os.chdir(diretorio_modelo)
        return criador()
    finally:
        os.chdir(diretorio_anterior)


def criar_rastreador(nome, args):
    if nome == "GOTURN":
        return criar_goturn(args.goturn_prototxt, args.goturn_modelo)

    criador = obter_criador(nome)
    if criador is None:
        raise RuntimeError(
            f"O rastreador {nome} nao esta disponivel. "
            "Instale opencv-contrib-python e remova opencv-python, se necessario."
        )
    return criador()


def criar_captura(args):
    if args.modo == "video":
        caminho = Path(args.entrada)
        if not caminho.is_file():
            raise FileNotFoundError(f"Video nao encontrado: {caminho}")
        captura = cv2.VideoCapture(str(caminho))
    else:
        captura = cv2.VideoCapture(args.camera)

    if not captura.isOpened():
        fonte = args.entrada if args.modo == "video" else f"camera {args.camera}"
        raise RuntimeError(f"Nao foi possivel abrir: {fonte}")

    return captura


def caminho_saida(args):
    if args.saida:
        saida = Path(args.saida)
    else:
        nome = (
            "rastreamento_video.mp4"
            if args.modo == "video"
            else "rastreamento_webcam.mp4"
        )
        saida = Path("resultados") / nome

    saida.parent.mkdir(parents=True, exist_ok=True)
    return saida


def texto(frame, mensagem, posicao, cor, escala=0.65):
    cv2.putText(
        frame,
        mensagem,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        cor,
        2,
        cv2.LINE_AA,
    )


def quadro_para_selecao_video(captura, fps, inicio):
    """
    Permite escolher o quadro em que a ROI sera marcada.
    Isso e importante quando o objeto ainda nao aparece no primeiro quadro.
    """
    captura.set(cv2.CAP_PROP_POS_MSEC, inicio * 1000.0)
    sucesso, quadro = captura.read()
    if not sucesso:
        raise RuntimeError("Nao foi possivel ler o quadro inicial do video.")

    passo = max(1, int(round(fps / 3.0)))
    janela = "Escolha do quadro inicial"

    while True:
        exibicao = quadro.copy()
        instante = captura.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        texto(exibicao, f"Tempo: {instante:.2f} s", (15, 30), (0, 255, 255))
        texto(
            exibicao,
            "S/ENTER: selecionar | N/ESPACO: avancar",
            (15, exibicao.shape[0] - 45),
            (255, 255, 255),
            0.55,
        )
        texto(
            exibicao,
            "B: voltar 1 s | Q/ESC: cancelar",
            (15, exibicao.shape[0] - 18),
            (255, 255, 255),
            0.55,
        )
        cv2.imshow(janela, exibicao)
        tecla = cv2.waitKey(0) & 0xFF

        if tecla in (ord("s"), ord("S"), 13):
            cv2.destroyWindow(janela)
            return quadro

        if tecla in (ord("n"), ord("N"), 32):
            novo_quadro = quadro
            for _ in range(passo):
                sucesso, candidato = captura.read()
                if not sucesso:
                    break
                novo_quadro = candidato
            quadro = novo_quadro
            continue

        if tecla in (ord("b"), ord("B")):
            posicao_atual = captura.get(cv2.CAP_PROP_POS_MSEC)
            nova_posicao = max(0.0, posicao_atual - 1000.0)
            captura.set(cv2.CAP_PROP_POS_MSEC, nova_posicao)
            sucesso, candidato = captura.read()
            if sucesso:
                quadro = candidato
            continue

        if tecla in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(janela)
            raise KeyboardInterrupt


def quadro_para_selecao_webcam(captura):
    """Le alguns quadros para estabilizar exposicao e foco da webcam."""
    quadro = None
    for _ in range(20):
        sucesso, quadro = captura.read()
        if not sucesso:
            raise RuntimeError("Nao foi possivel ler um quadro da webcam.")
    return quadro


def selecionar_roi(quadro):
    janela = "Selecione a ROI e pressione ENTER ou ESPACO"
    caixa = cv2.selectROI(
        janela,
        quadro,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow(janela)

    x, y, largura, altura = [int(valor) for valor in caixa]
    if largura <= 0 or altura <= 0:
        raise KeyboardInterrupt
    return x, y, largura, altura


def inicializar_rastreador(nome, args, quadro, caixa):
    rastreador = criar_rastreador(nome, args)
    resultado = rastreador.init(quadro, caixa)
    if resultado is False:
        raise RuntimeError("O rastreador nao conseguiu inicializar a ROI.")
    return rastreador


def criar_gravador(saida, fps, largura, altura):
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(
        str(saida),
        codec,
        fps,
        (largura, altura),
    )
    if not gravador.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video de saida: {saida}")
    return gravador


def desenhar_caixa(quadro, caixa, cor=(0, 255, 0)):
    x, y, largura, altura = [int(valor) for valor in caixa]
    cv2.rectangle(
        quadro,
        (x, y),
        (x + largura, y + altura),
        cor,
        2,
    )


def executar(args):
    captura = criar_captura(args)
    gravador = None
    saida = caminho_saida(args)

    try:
        fps_fonte = captura.get(cv2.CAP_PROP_FPS)
        if fps_fonte <= 1 or fps_fonte > 240:
            fps_fonte = 30.0

        if args.modo == "video":
            quadro = quadro_para_selecao_video(captura, fps_fonte, args.inicio)
        else:
            quadro = quadro_para_selecao_webcam(captura)

        caixa = selecionar_roi(quadro)
        rastreador = inicializar_rastreador(
            args.rastreador, args, quadro, caixa
        )

        altura, largura = quadro.shape[:2]
        gravador = criar_gravador(saida, fps_fonte, largura, altura)

        # Grava o quadro usado na selecao com a ROI inicial.
        quadro_inicial = quadro.copy()
        desenhar_caixa(quadro_inicial, caixa)
        texto(
            quadro_inicial,
            f"Rastreador: {args.rastreador}",
            (15, 30),
            (0, 255, 0),
        )
        gravador.write(quadro_inicial)

        janela = "LAB 8 - Rastreamento"
        contador = 0
        inicio_medicao = time.perf_counter()

        while True:
            sucesso_leitura, quadro = captura.read()
            if not sucesso_leitura:
                break

            contador += 1
            sucesso_rastreio, caixa_atual = rastreador.update(quadro)

            if sucesso_rastreio:
                caixa = caixa_atual
                desenhar_caixa(quadro, caixa)
                texto(quadro, "RASTREAMENTO OK", (15, 30), (0, 255, 0))
            else:
                texto(
                    quadro,
                    "OBJETO PERDIDO - pressione R",
                    (15, 30),
                    (0, 0, 255),
                )

            tempo = max(time.perf_counter() - inicio_medicao, 1e-9)
            fps_processamento = contador / tempo
            texto(
                quadro,
                f"{args.rastreador} | FPS: {fps_processamento:.1f}",
                (15, 58),
                (0, 255, 255),
                0.55,
            )
            texto(
                quadro,
                "R: nova ROI | Q/ESC: sair",
                (15, altura - 18),
                (255, 255, 255),
                0.55,
            )

            gravador.write(quadro)
            cv2.imshow(janela, quadro)

            atraso = max(1, int(round(1000.0 / fps_fonte)))
            tecla = cv2.waitKey(atraso) & 0xFF

            if tecla in (ord("q"), ord("Q"), 27):
                break

            if tecla in (ord("r"), ord("R")):
                caixa = selecionar_roi(quadro)
                rastreador = inicializar_rastreador(
                    args.rastreador, args, quadro, caixa
                )

        print(f"Video salvo com sucesso em: {saida.resolve()}")

    finally:
        captura.release()
        if gravador is not None:
            gravador.release()
        cv2.destroyAllWindows()


def main():
    args = analisar_argumentos()
    try:
        executar(args)
    except KeyboardInterrupt:
        print("Operacao cancelada pelo usuario.")
        return 1
    except (FileNotFoundError, RuntimeError, cv2.error) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
