#!/usr/bin/env python3
"""
Projeto final de Visao Computacional - Grupo 2

Sistema estereoscopico para:
1) verificar duas cameras;
2) capturar pares de calibracao;
3) calibrar o par estereo;
4) validar visualmente a retificacao;
5) classificar materiais com ResNet-50 usando a camera esquerda;
6) gerar disparidade e estimar a distancia com o par estereo.

Exemplos:
    python executar_projeto.py --modo listar
    python executar_projeto.py --modo verificar --cam-esq 1 --cam-dir 0
    python executar_projeto.py --modo capturar --cam-esq 1 --cam-dir 0
    python executar_projeto.py --modo calibrar
    python executar_projeto.py --modo validar --cam-esq 1 --cam-dir 0
    python executar_projeto.py --modo executar --cam-esq 1 --cam-dir 0

No Linux, depois de validar a conversao ONNX:
    python3 executar_projeto.py --modo executar --modelo modelos/resnet50_waste.onnx --cam-esq 0 --cam-dir 2

Controles nas janelas:
    q ou ESC  - sair
    ESPACO    - salvar par de calibracao (modo capturar)
    s         - salvar imagens da execucao (modo executar)
    d         - alternar visualizacao da disparidade
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


NOME_JANELA = "Projeto Final - Grupo 2"
CORES_LIXEIRAS = {
    "cardboard": (255, 120, 40),  # azul em BGR
    "paper": (255, 120, 40),      # azul em BGR
    "plastic": (50, 50, 230),     # vermelho em BGR
    "glass": (50, 180, 70),       # verde em BGR
    "metal": (40, 220, 240),      # amarelo em BGR
    "trash": (150, 150, 150),     # cinza em BGR
}
LIXEIRAS = {
    "cardboard": "Lixeira azul (papel/papelao)",
    "paper": "Lixeira azul (papel/papelao)",
    "plastic": "Lixeira vermelha (plastico)",
    "glass": "Lixeira verde (vidro)",
    "metal": "Lixeira amarela (metal)",
    "trash": "Lixeira cinza (rejeitos)",
}
NOMES_PT = {
    "cardboard": "papelao",
    "glass": "vidro",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plastico",
    "trash": "rejeito",
}
CLASSES_ESPERADAS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def texto_sem_acento(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in normalizado if not unicodedata.combining(c)).lower().strip()


def categoria_material(nome_classe: str) -> str | None:
    """Converte variacoes de nomes para uma das seis classes do projeto."""
    nome = texto_sem_acento(nome_classe)
    mapa = {
        "cardboard": ("cardboard", "papelao"),
        "glass": ("vidro", "glass"),
        "metal": ("metal", "aluminio", "aluminum", "can", "lata"),
        "paper": ("papel", "paper"),
        "plastic": ("plastico", "plastic", "pet", "bottle"),
        "trash": ("trash", "rejeito", "lixo"),
    }
    for categoria, palavras in mapa.items():
        if any(palavra in nome for palavra in palavras):
            return categoria
    return None


def colocar_texto(
    imagem: np.ndarray,
    texto: str,
    posicao: tuple[int, int],
    cor: tuple[int, int, int] = (255, 255, 255),
    escala: float = 0.58,
    espessura: int = 1,
    fundo: tuple[int, int, int] | None = (20, 20, 20),
) -> None:
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    x, y = posicao
    (largura, altura), base = cv2.getTextSize(texto, fonte, escala, espessura)
    if fundo is not None:
        cv2.rectangle(
            imagem,
            (x - 4, y - altura - 5),
            (x + largura + 4, y + base + 4),
            fundo,
            -1,
        )
    cv2.putText(imagem, texto, (x, y), fonte, escala, cor, espessura, cv2.LINE_AA)


def criar_captura(indice: int, largura: int, altura: int, fps: int) -> cv2.VideoCapture:
    if sys.platform.startswith("linux"):
        captura = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if not captura.isOpened():
            captura.release()
            captura = cv2.VideoCapture(indice)
    else:
        captura = cv2.VideoCapture(indice)

    # MJPG costuma permitir duas webcams em 640 x 480 sem exceder a banda USB.
    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    captura.set(cv2.CAP_PROP_FPS, fps)
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return captura


def aplicar_ajustes_opcionais_camera(
    camera: cv2.VideoCapture,
    args: argparse.Namespace,
) -> None:
    """Aplica somente ajustes solicitados, pois cada webcam usa escalas diferentes."""
    if args.travar_foco:
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if args.foco is not None:
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        camera.set(cv2.CAP_PROP_FOCUS, float(args.foco))
    if args.exposicao is not None:
        # Em webcams V4L2, 0.25 normalmente representa exposicao manual.
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        camera.set(cv2.CAP_PROP_EXPOSURE, float(args.exposicao))


def abrir_par_cameras(args: argparse.Namespace) -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    if args.cam_esq == args.cam_dir:
        raise RuntimeError("Os indices das cameras esquerda e direita devem ser diferentes.")

    cam_esq = criar_captura(args.cam_esq, args.largura, args.altura, args.fps)
    cam_dir = criar_captura(args.cam_dir, args.largura, args.altura, args.fps)

    if not cam_esq.isOpened() or not cam_dir.isOpened():
        cam_esq.release()
        cam_dir.release()
        raise RuntimeError(
            "Nao foi possivel abrir as duas cameras. Rode primeiro '--modo listar' "
            "e ajuste --cam-esq e --cam-dir."
        )

    # Descarta quadros antigos que possam estar no buffer.
    for _ in range(8):
        cam_esq.grab()
        cam_dir.grab()

    aplicar_ajustes_opcionais_camera(cam_esq, args)
    aplicar_ajustes_opcionais_camera(cam_dir, args)

    return cam_esq, cam_dir


def ler_par(
    cam_esq: cv2.VideoCapture,
    cam_dir: cv2.VideoCapture,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    # grab nas duas cameras antes de retrieve reduz a diferenca temporal.
    ok_esq = cam_esq.grab()
    ok_dir = cam_dir.grab()
    if not ok_esq or not ok_dir:
        return False, None, None

    ok_esq, quadro_esq = cam_esq.retrieve()
    ok_dir, quadro_dir = cam_dir.retrieve()
    return bool(ok_esq and ok_dir), quadro_esq, quadro_dir


def listar_cameras(args: argparse.Namespace) -> None:
    print("\nProcurando cameras...\n")
    encontradas: list[int] = []
    for indice in range(args.max_indice + 1):
        captura = criar_captura(indice, args.largura, args.altura, args.fps)
        ok, quadro = captura.read()
        if captura.isOpened() and ok and quadro is not None:
            altura, largura = quadro.shape[:2]
            print(f"Camera encontrada no indice {indice}: {largura} x {altura}")
            encontradas.append(indice)
        captura.release()

    if encontradas:
        print("\nIndices encontrados:", encontradas)
        print("Teste o par, por exemplo:")
        if len(encontradas) >= 2:
            print(
                "python3 executar_projeto.py --modo verificar "
                f"--cam-esq {encontradas[0]} --cam-dir {encontradas[1]}"
            )
    else:
        print("Nenhuma camera foi encontrada. Confira cabos e permissoes do sistema.")


def verificar_cameras(args: argparse.Namespace) -> None:
    cam_esq, cam_dir = abrir_par_cameras(args)
    print("Cameras abertas. Pressione q ou ESC para sair.")

    try:
        while True:
            ok, quadro_esq, quadro_dir = ler_par(cam_esq, cam_dir)
            if not ok or quadro_esq is None or quadro_dir is None:
                print("Falha na captura do par.")
                break

            if quadro_esq.shape != quadro_dir.shape:
                quadro_dir = cv2.resize(quadro_dir, (quadro_esq.shape[1], quadro_esq.shape[0]))

            colocar_texto(quadro_esq, f"ESQUERDA - indice {args.cam_esq}", (12, 28))
            colocar_texto(quadro_dir, f"DIREITA - indice {args.cam_dir}", (12, 28))
            lado_a_lado = np.hstack((quadro_esq, quadro_dir))
            cv2.imshow(NOME_JANELA, lado_a_lado)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
    finally:
        cam_esq.release()
        cam_dir.release()
        cv2.destroyAllWindows()


def encontrar_cantos(
    imagem_cinza: np.ndarray,
    padrao: tuple[int, int],
    preciso: bool = False,
) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if not preciso:
        flags |= cv2.CALIB_CB_FAST_CHECK
    encontrou, cantos = cv2.findChessboardCorners(imagem_cinza, padrao, flags)
    if encontrou:
        criterio = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        cantos = cv2.cornerSubPix(imagem_cinza, cantos, (11, 11), (-1, -1), criterio)
        return True, cantos

    # A busca SB e mais robusta, mas tambem mais lenta. Por isso ela e usada
    # somente na validacao precisa, no momento de salvar ou calibrar.
    if preciso and hasattr(cv2, "findChessboardCornersSB"):
        flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags_sb |= cv2.CALIB_CB_ACCURACY
        encontrou, cantos = cv2.findChessboardCornersSB(
            imagem_cinza,
            padrao,
            flags=flags_sb,
        )
        return bool(encontrou), cantos

    return False, None


def proximo_numero_pares(pasta: Path) -> int:
    numeros: list[int] = []
    for arquivo in pasta.glob("esquerda_*.png"):
        achou = re.search(r"(\d+)$", arquivo.stem)
        if achou:
            numeros.append(int(achou.group(1)))
    return max(numeros, default=0) + 1


def capturar_calibracao(args: argparse.Namespace) -> None:
    pasta = Path(args.pasta_calibracao)
    pasta.mkdir(parents=True, exist_ok=True)
    numero = proximo_numero_pares(pasta)
    padrao = (args.colunas, args.linhas)
    cam_esq, cam_dir = abrir_par_cameras(args)

    print("\nCaptura da calibracao")
    print(f"Padrao configurado: {args.colunas} x {args.linhas} cantos internos")
    print("Mova e incline o tabuleiro. Pressione ESPACO quando os dois lados estiverem verdes.")
    print("Capture pelo menos 20 a 25 pares diferentes. Pressione q para sair.\n")

    contador_quadros = 0
    achou_esq = False
    achou_dir = False
    cantos_esq: np.ndarray | None = None
    cantos_dir: np.ndarray | None = None
    ultimo_sucesso_esq = -10.0
    ultimo_sucesso_dir = -10.0

    try:
        while True:
            ok, quadro_esq, quadro_dir = ler_par(cam_esq, cam_dir)
            if not ok or quadro_esq is None or quadro_dir is None:
                print("Falha ao capturar as cameras.")
                break

            cinza_esq = cv2.cvtColor(quadro_esq, cv2.COLOR_BGR2GRAY)
            cinza_dir = cv2.cvtColor(quadro_dir, cv2.COLOR_BGR2GRAY)
            contador_quadros += 1
            intervalo = max(1, args.intervalo_deteccao)
            deteccao_neste_quadro = contador_quadros % intervalo == 0
            if deteccao_neste_quadro:
                detectou_esq, cantos_esq = encontrar_cantos(cinza_esq, padrao, preciso=False)
                detectou_dir, cantos_dir = encontrar_cantos(cinza_dir, padrao, preciso=False)
                instante = time.monotonic()
                if detectou_esq:
                    ultimo_sucesso_esq = instante
                if detectou_dir:
                    ultimo_sucesso_dir = instante

            instante = time.monotonic()
            # Pequena histerese visual evita piscar entre verde e vermelho por
            # uma unica falha do detector rapido. A validacao ao salvar e atual.
            achou_esq = instante - ultimo_sucesso_esq < 0.7
            achou_dir = instante - ultimo_sucesso_dir < 0.7

            visual_esq = quadro_esq.copy()
            visual_dir = quadro_dir.copy()
            if deteccao_neste_quadro and detectou_esq and cantos_esq is not None:
                cv2.drawChessboardCorners(visual_esq, padrao, cantos_esq, True)
            if deteccao_neste_quadro and detectou_dir and cantos_dir is not None:
                cv2.drawChessboardCorners(visual_dir, padrao, cantos_dir, True)

            ambos = achou_esq and achou_dir
            cor = (30, 220, 70) if ambos else (30, 60, 230)
            status = "PRONTO PARA SALVAR" if ambos else "TABULEIRO NAO DETECTADO NOS DOIS LADOS"
            colocar_texto(visual_esq, f"ESQUERDA | {status}", (12, 28), cor=cor)
            colocar_texto(visual_dir, f"DIREITA | proximo par: {numero:03d}", (12, 28), cor=cor)

            cv2.imshow(NOME_JANELA, np.hstack((visual_esq, visual_dir)))
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), 27):
                break
            if tecla == 32:
                print("Validando o par com deteccao precisa...")
                preciso_esq, _ = encontrar_cantos(
                    cinza_esq, padrao, preciso=True
                )
                preciso_dir, _ = encontrar_cantos(
                    cinza_dir, padrao, preciso=True
                )
                if preciso_esq and preciso_dir:
                    nome_esq = pasta / f"esquerda_{numero:03d}.png"
                    nome_dir = pasta / f"direita_{numero:03d}.png"
                    cv2.imwrite(str(nome_esq), quadro_esq)
                    cv2.imwrite(str(nome_dir), quadro_dir)
                    print(f"Par {numero:03d} salvo.")
                    numero += 1
                    time.sleep(0.25)
                else:
                    lados = []
                    if not preciso_esq:
                        lados.append("esquerda")
                    if not preciso_dir:
                        lados.append("direita")
                    print(
                        "Par nao salvo. Falha na deteccao precisa: "
                        + " e ".join(lados)
                        + "."
                    )
    finally:
        cam_esq.release()
        cam_dir.release()
        cv2.destroyAllWindows()


def nomes_pares_calibracao(pasta: Path) -> list[tuple[Path, Path]]:
    esquerdas = {p.name.replace("esquerda_", ""): p for p in pasta.glob("esquerda_*.png")}
    direitas = {p.name.replace("direita_", ""): p for p in pasta.glob("direita_*.png")}
    chaves = sorted(set(esquerdas).intersection(direitas))
    return [(esquerdas[chave], direitas[chave]) for chave in chaves]


def erro_reprojecao(
    pontos_objeto: list[np.ndarray],
    pontos_imagem: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    matriz: np.ndarray,
    distorcao: np.ndarray,
) -> float:
    erros: list[float] = []
    for objeto, observado, rvec, tvec in zip(pontos_objeto, pontos_imagem, rvecs, tvecs):
        projetado, _ = cv2.projectPoints(objeto, rvec, tvec, matriz, distorcao)
        diferencas = observado.reshape(-1, 2) - projetado.reshape(-1, 2)
        erro = float(np.mean(np.linalg.norm(diferencas, axis=1)))
        erros.append(float(erro))
    return float(np.mean(erros))


def calibrar_estereo(args: argparse.Namespace) -> None:
    pasta = Path(args.pasta_calibracao)
    pares = nomes_pares_calibracao(pasta)
    if len(pares) < args.min_pares:
        raise RuntimeError(
            f"Foram encontrados apenas {len(pares)} pares. "
            f"Capture pelo menos {args.min_pares}."
        )

    padrao = (args.colunas, args.linhas)
    pontos_modelo = np.zeros((args.colunas * args.linhas, 3), np.float32)
    pontos_modelo[:, :2] = np.mgrid[0 : args.colunas, 0 : args.linhas].T.reshape(-1, 2)
    pontos_modelo *= args.quadrado

    pontos_objeto: list[np.ndarray] = []
    pontos_esq: list[np.ndarray] = []
    pontos_dir: list[np.ndarray] = []
    tamanho: tuple[int, int] | None = None

    print(f"\nAnalisando {len(pares)} pares de imagens...")
    for arquivo_esq, arquivo_dir in pares:
        imagem_esq = cv2.imread(str(arquivo_esq))
        imagem_dir = cv2.imread(str(arquivo_dir))
        if imagem_esq is None or imagem_dir is None:
            print(f"Ignorando par ilegivel: {arquivo_esq.name}")
            continue
        if imagem_esq.shape[:2] != imagem_dir.shape[:2]:
            print(f"Ignorando par com tamanhos diferentes: {arquivo_esq.name}")
            continue

        cinza_esq = cv2.cvtColor(imagem_esq, cv2.COLOR_BGR2GRAY)
        cinza_dir = cv2.cvtColor(imagem_dir, cv2.COLOR_BGR2GRAY)
        tamanho_atual = (cinza_esq.shape[1], cinza_esq.shape[0])
        if tamanho is not None and tamanho_atual != tamanho:
            print(f"Ignorando par com resolucao diferente: {arquivo_esq.name}")
            continue
        tamanho = tamanho_atual

        achou_esq, cantos_esq = encontrar_cantos(cinza_esq, padrao, preciso=True)
        achou_dir, cantos_dir = encontrar_cantos(cinza_dir, padrao, preciso=True)
        if achou_esq and achou_dir and cantos_esq is not None and cantos_dir is not None:
            pontos_objeto.append(pontos_modelo.copy())
            pontos_esq.append(cantos_esq.astype(np.float32))
            pontos_dir.append(cantos_dir.astype(np.float32))
            print(f"OK: {arquivo_esq.name}")
        else:
            print(f"Cantos nao encontrados: {arquivo_esq.name}")

    if tamanho is None or len(pontos_objeto) < args.min_pares:
        raise RuntimeError(
            f"Somente {len(pontos_objeto)} pares validos. "
            "Confira --colunas, --linhas e a qualidade das imagens."
        )

    criterio = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )

    rms_esq, matriz_esq, dist_esq, rvecs_esq, tvecs_esq = cv2.calibrateCamera(
        pontos_objeto, pontos_esq, tamanho, None, None
    )
    rms_dir, matriz_dir, dist_dir, rvecs_dir, tvecs_dir = cv2.calibrateCamera(
        pontos_objeto, pontos_dir, tamanho, None, None
    )

    flags = cv2.CALIB_FIX_INTRINSIC
    rms_estereo, matriz_esq, dist_esq, matriz_dir, dist_dir, R, T, E, F = cv2.stereoCalibrate(
        pontos_objeto,
        pontos_esq,
        pontos_dir,
        matriz_esq,
        dist_esq,
        matriz_dir,
        dist_dir,
        tamanho,
        criteria=criterio,
        flags=flags,
    )

    R1, R2, P1, P2, Q, roi_esq, roi_dir = cv2.stereoRectify(
        matriz_esq,
        dist_esq,
        matriz_dir,
        dist_dir,
        tamanho,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    erro_esq = erro_reprojecao(
        pontos_objeto, pontos_esq, rvecs_esq, tvecs_esq, matriz_esq, dist_esq
    )
    erro_dir = erro_reprojecao(
        pontos_objeto, pontos_dir, rvecs_dir, tvecs_dir, matriz_dir, dist_dir
    )

    erros_verticais: list[float] = []
    for cantos_esq, cantos_dir in zip(pontos_esq, pontos_dir):
        ret_esq = cv2.undistortPoints(cantos_esq, matriz_esq, dist_esq, R=R1, P=P1)
        ret_dir = cv2.undistortPoints(cantos_dir, matriz_dir, dist_dir, R=R2, P=P2)
        erros_verticais.extend(np.abs(ret_esq[:, 0, 1] - ret_dir[:, 0, 1]).tolist())
    erro_vertical = float(np.mean(erros_verticais))

    arquivo_saida = Path(args.arquivo_calibracao)
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arquivo_saida,
        matriz_esq=matriz_esq,
        dist_esq=dist_esq,
        matriz_dir=matriz_dir,
        dist_dir=dist_dir,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi_esq=np.asarray(roi_esq),
        roi_dir=np.asarray(roi_dir),
        tamanho=np.asarray(tamanho),
        quadrado=np.asarray(args.quadrado),
        rms_esq=np.asarray(rms_esq),
        rms_dir=np.asarray(rms_dir),
        rms_estereo=np.asarray(rms_estereo),
        erro_esq=np.asarray(erro_esq),
        erro_dir=np.asarray(erro_dir),
        erro_vertical=np.asarray(erro_vertical),
        pares_validos=np.asarray(len(pontos_objeto)),
    )

    base = float(np.linalg.norm(T))
    metricas = {
        "data_calibracao": datetime.now().isoformat(timespec="seconds"),
        "pares_validos": len(pontos_objeto),
        "resolucao": [tamanho[0], tamanho[1]],
        "cantos_internos": [args.colunas, args.linhas],
        "quadrado_m": args.quadrado,
        "rms_esquerda_px": float(rms_esq),
        "rms_direita_px": float(rms_dir),
        "rms_estereo_px": float(rms_estereo),
        "erro_reprojecao_esquerda_px": erro_esq,
        "erro_reprojecao_direita_px": erro_dir,
        "erro_epipolar_vertical_px": erro_vertical,
        "baseline_m": base,
    }
    arquivo_metricas = arquivo_saida.with_name("metricas_calibracao.json")
    with arquivo_metricas.open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)
    print("\nCALIBRACAO CONCLUIDA")
    print(f"Pares usados: {len(pontos_objeto)}")
    print(f"Resolucao: {tamanho[0]} x {tamanho[1]}")
    print(f"RMS camera esquerda: {rms_esq:.4f}")
    print(f"RMS camera direita:  {rms_dir:.4f}")
    print(f"RMS estereo:          {rms_estereo:.4f}")
    print(f"Erro medio esquerda: {erro_esq:.4f} pixel")
    print(f"Erro medio direita:  {erro_dir:.4f} pixel")
    print(f"Erro epipolar vertical medio: {erro_vertical:.4f} pixel")
    print(f"Base estimada: {base:.4f} m ({base * 1000:.1f} mm)")
    print(f"Parametros salvos em: {arquivo_saida.resolve()}")
    print(f"Metricas salvas em: {arquivo_metricas.resolve()}")

    if erro_esq > 1.0 or erro_dir > 1.0 or erro_vertical > 1.0:
        print("\nATENCAO: o erro ficou acima de 1 pixel. Capture pares mais variados e recalibre.")


def criar_stereo_sgbm(largura: int, num_disparidades: int, bloco: int) -> cv2.StereoSGBM:
    num_disparidades = max(16, (num_disparidades // 16) * 16)
    if num_disparidades >= largura:
        num_disparidades = max(16, ((largura // 2) // 16) * 16)
    bloco = max(3, bloco)
    if bloco % 2 == 0:
        bloco += 1

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparidades,
        blockSize=bloco,
        P1=8 * bloco * bloco,
        P2=32 * bloco * bloco,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def mediana_distancia(
    pontos_3d: np.ndarray,
    regiao: tuple[int, int, int, int],
    minimo: float,
    maximo: float,
) -> float | None:
    x1, y1, x2, y2 = regiao
    altura, largura = pontos_3d.shape[:2]
    x1 = int(np.clip(x1, 0, largura - 1))
    x2 = int(np.clip(x2, x1 + 1, largura))
    y1 = int(np.clip(y1, 0, altura - 1))
    y2 = int(np.clip(y2, y1 + 1, altura))

    # A orientacao do eixo Z pode mudar de sinal conforme a ordem das cameras.
    # Para a tarefa, interessa a distancia frontal, portanto usamos o modulo.
    profundidades = np.abs(pontos_3d[y1:y2, x1:x2, 2])
    validas = profundidades[
        np.isfinite(profundidades)
        & (profundidades >= minimo)
        & (profundidades <= maximo)
    ]
    if validas.size < 20:
        return None
    return float(np.median(validas))


def colorir_disparidade(disparidade: np.ndarray) -> np.ndarray:
    valida = np.isfinite(disparidade) & (disparidade > 0.5)
    normalizada = np.zeros(disparidade.shape, dtype=np.uint8)
    if np.any(valida):
        valores = disparidade[valida]
        minimo = float(np.percentile(valores, 5))
        maximo = float(np.percentile(valores, 95))
        if maximo > minimo:
            escala = np.clip((disparidade - minimo) * 255.0 / (maximo - minimo), 0, 255)
            normalizada[valida] = escala[valida].astype(np.uint8)
    colorida = cv2.applyColorMap(normalizada, cv2.COLORMAP_TURBO)
    colorida[~valida] = 0
    return colorida


@dataclass
class Calibracao:
    matriz_esq: np.ndarray
    dist_esq: np.ndarray
    matriz_dir: np.ndarray
    dist_dir: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    tamanho: tuple[int, int]
    escala_metros: float
    origem: str
    metricas: dict[str, Any]


def no_real(armazenamento: cv2.FileStorage, nome: str, padrao: float = 0.0) -> float:
    no = armazenamento.getNode(nome)
    return padrao if no.empty() else float(no.real())


def carregar_calibracao(caminho: str, fator_unidade_m: float) -> Calibracao:
    arquivo = Path(caminho)
    if not arquivo.exists():
        backup = arquivo.parent / "stereo_params_abc_backup.xml"
        if backup.exists():
            print(f"Calibracao principal nao encontrada. Usando o backup do Lab 6: {backup}")
            arquivo = backup
        else:
            raise RuntimeError(
                f"Calibracao nao encontrada: {arquivo}. "
                "Execute primeiro os modos capturar e calibrar."
            )

    if arquivo.suffix.lower() == ".npz":
        dados = np.load(str(arquivo))
        tamanho = tuple(int(v) for v in dados["tamanho"])
        metricas: dict[str, Any] = {}
        for chave in (
            "rms_esq",
            "rms_dir",
            "rms_estereo",
            "erro_esq",
            "erro_dir",
            "erro_vertical",
            "pares_validos",
        ):
            if chave in dados:
                metricas[chave] = float(np.asarray(dados[chave]).reshape(-1)[0])
        return Calibracao(
            matriz_esq=dados["matriz_esq"],
            dist_esq=dados["dist_esq"],
            matriz_dir=dados["matriz_dir"],
            dist_dir=dados["dist_dir"],
            R1=dados["R1"],
            R2=dados["R2"],
            P1=dados["P1"],
            P2=dados["P2"],
            Q=dados["Q"],
            tamanho=tamanho,
            escala_metros=1.0,
            origem=str(arquivo),
            metricas=metricas,
        )

    if arquivo.suffix.lower() in (".xml", ".yml", ".yaml"):
        fs = cv2.FileStorage(str(arquivo), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Nao foi possivel ler a calibracao: {arquivo}")
        try:
            largura = int(no_real(fs, "image_width", 640))
            altura = int(no_real(fs, "image_height", 480))
            metricas = {
                "rms_esq": no_real(fs, "rms_left"),
                "rms_dir": no_real(fs, "rms_right"),
                "rms_estereo": no_real(fs, "rms_stereo"),
                "pares_validos": no_real(fs, "num_valid_pairs"),
            }
            matrizes = {
                nome: fs.getNode(nome).mat()
                for nome in ("M1", "D1", "M2", "D2", "R1", "R2", "P1", "P2", "Q")
            }
        finally:
            fs.release()
        if any(valor is None for valor in matrizes.values()):
            raise RuntimeError("O XML nao contem todas as matrizes M1, D1, M2, D2, R1, R2, P1, P2 e Q.")
        return Calibracao(
            matriz_esq=matrizes["M1"],
            dist_esq=matrizes["D1"],
            matriz_dir=matrizes["M2"],
            dist_dir=matrizes["D2"],
            R1=matrizes["R1"],
            R2=matrizes["R2"],
            P1=matrizes["P1"],
            P2=matrizes["P2"],
            Q=matrizes["Q"],
            tamanho=(largura, altura),
            # O XML do Lab 6 foi salvo em unidades de quadrado. Cada unidade
            # equivale a 30 mm no tabuleiro fisico do grupo.
            escala_metros=fator_unidade_m,
            origem=str(arquivo),
            metricas=metricas,
        )

    raise RuntimeError("Formato de calibracao nao suportado. Use .npz, .xml, .yml ou .yaml.")


def criar_mapas(calibracao: Calibracao) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapa_esq_x, mapa_esq_y = cv2.initUndistortRectifyMap(
        calibracao.matriz_esq,
        calibracao.dist_esq,
        calibracao.R1,
        calibracao.P1,
        calibracao.tamanho,
        cv2.CV_32FC1,
    )
    mapa_dir_x, mapa_dir_y = cv2.initUndistortRectifyMap(
        calibracao.matriz_dir,
        calibracao.dist_dir,
        calibracao.R2,
        calibracao.P2,
        calibracao.tamanho,
        cv2.CV_32FC1,
    )
    return mapa_esq_x, mapa_esq_y, mapa_dir_x, mapa_dir_y


def criar_stereo_configurado(args: argparse.Namespace, largura: int) -> cv2.StereoSGBM:
    configuracao: dict[str, Any] = {}
    arquivo = Path(args.config_sgbm)
    if arquivo.exists():
        with arquivo.open("r", encoding="utf-8") as entrada:
            configuracao = json.load(entrada)

    num = int(configuracao.get("num_disparities", args.num_disparidades))
    num = max(16, (num // 16) * 16)
    if num >= largura:
        num = max(16, ((largura // 2) // 16) * 16)
    bloco = max(3, int(configuracao.get("block_size", args.bloco)))
    if bloco % 2 == 0:
        bloco += 1
    return cv2.StereoSGBM_create(
        minDisparity=int(configuracao.get("min_disparity", 0)),
        numDisparities=num,
        blockSize=bloco,
        P1=8 * bloco * bloco,
        P2=32 * bloco * bloco,
        disp12MaxDiff=int(configuracao.get("disp12_max_diff", 1)),
        uniquenessRatio=int(configuracao.get("uniqueness_ratio", 10)),
        speckleWindowSize=int(configuracao.get("speckle_window_size", 80)),
        speckleRange=int(configuracao.get("speckle_range", 2)),
        preFilterCap=int(configuracao.get("pre_filter_cap", 31)),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


class ClassificadorResNet:
    def __init__(self, caminho_modelo: str, caminho_classes: str) -> None:
        self.arquivo = Path(caminho_modelo)
        arquivo_classes = Path(caminho_classes)
        if not self.arquivo.exists():
            raise RuntimeError(f"Modelo nao encontrado: {self.arquivo}")
        if not arquivo_classes.exists():
            raise RuntimeError(f"Arquivo de classes nao encontrado: {arquivo_classes}")
        with arquivo_classes.open("r", encoding="utf-8") as entrada:
            self.classes = [str(nome) for nome in json.load(entrada)]
        if sorted(self.classes) != sorted(CLASSES_ESPERADAS):
            raise RuntimeError(
                "class_names.json deve conter exatamente: " + ", ".join(CLASSES_ESPERADAS)
            )

        self.tipo = self.arquivo.suffix.lower()
        if self.tipo == ".onnx":
            self.modelo = cv2.dnn.readNetFromONNX(str(self.arquivo))
        elif self.tipo in (".keras", ".h5"):
            try:
                from tensorflow.keras.models import load_model
            except ImportError as erro:
                raise RuntimeError(
                    "TensorFlow nao esta instalado. No Linux, use o modelo .onnx."
                ) from erro
            self.modelo = load_model(str(self.arquivo), compile=False)
        else:
            raise RuntimeError("Modelo deve estar em .onnx, .keras ou .h5.")
        print(f"Modelo carregado: {self.arquivo} ({self.tipo})")

    def prever(self, imagem_bgr: np.ndarray) -> tuple[str, float, np.ndarray, float]:
        inicio = time.perf_counter()
        if self.tipo == ".onnx":
            entrada = cv2.dnn.blobFromImage(
                imagem_bgr,
                scalefactor=1.0,
                size=(224, 224),
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
                crop=False,
            )
            self.modelo.setInput(entrada)
            probabilidades = self.modelo.forward().reshape(-1).astype(np.float32)
        else:
            imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
            entrada = cv2.resize(imagem_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            entrada = entrada.astype(np.float32)
            probabilidades = self.modelo.predict(entrada[None, ...], verbose=0)[0].astype(np.float32)

        if probabilidades.size != len(self.classes):
            raise RuntimeError(
                f"O modelo retornou {probabilidades.size} valores, mas ha {len(self.classes)} classes."
            )
        soma = float(np.sum(probabilidades))
        if np.any(probabilidades < 0.0) or not np.isclose(soma, 1.0, atol=1e-3):
            logits = probabilidades - float(np.max(probabilidades))
            probabilidades = np.exp(logits)
            probabilidades /= float(np.sum(probabilidades))
        indice = int(np.argmax(probabilidades))
        latencia_ms = (time.perf_counter() - inicio) * 1000.0
        return self.classes[indice], float(probabilidades[indice]), probabilidades, latencia_ms


def regiao_central(largura: int, altura: int, fracao: float) -> tuple[int, int, int, int]:
    fracao = float(np.clip(fracao, 0.20, 0.90))
    lado = int(min(largura, altura) * fracao)
    cx, cy = largura // 2, altura // 2
    return cx - lado // 2, cy - lado // 2, cx + lado // 2, cy + lado // 2


def validar_calibracao(args: argparse.Namespace) -> None:
    calibracao = carregar_calibracao(args.arquivo_calibracao, args.fator_unidade_m)
    args.largura, args.altura = calibracao.tamanho
    cam_esq, cam_dir = abrir_par_cameras(args)
    mapas = criar_mapas(calibracao)
    pasta = Path(args.pasta_saida) / "validacao_retificacao"
    pasta.mkdir(parents=True, exist_ok=True)
    print(f"Calibracao carregada: {calibracao.origem}")
    print("As linhas horizontais devem cruzar os mesmos pontos nas duas imagens.")
    print("q/ESC: sair | s: salvar validacao")
    try:
        while True:
            ok, esquerda, direita = ler_par(cam_esq, cam_dir)
            if not ok or esquerda is None or direita is None:
                break
            ret_esq = cv2.remap(esquerda, mapas[0], mapas[1], cv2.INTER_LINEAR)
            ret_dir = cv2.remap(direita, mapas[2], mapas[3], cv2.INTER_LINEAR)
            tela = np.hstack((ret_esq, ret_dir))
            for y in range(40, calibracao.tamanho[1], 40):
                cv2.line(tela, (0, y), (tela.shape[1] - 1, y), (0, 255, 255), 1)
            colocar_texto(tela, "ESQUERDA RETIFICADA", (12, 28), cor=(90, 255, 120))
            colocar_texto(
                tela,
                "DIREITA RETIFICADA",
                (calibracao.tamanho[0] + 12, 28),
                cor=(90, 255, 120),
            )
            cv2.imshow("Validacao da retificacao - Grupo 2", tela)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("s"):
                nome = pasta / f"retificacao_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(nome), tela)
                print(f"Validacao salva em: {nome.resolve()}")
    finally:
        cam_esq.release()
        cam_dir.release()
        cv2.destroyAllWindows()


def executar_sistema(args: argparse.Namespace) -> None:
    calibracao = carregar_calibracao(args.arquivo_calibracao, args.fator_unidade_m)
    largura_cal, altura_cal = calibracao.tamanho
    args.largura, args.altura = calibracao.tamanho
    cam_esq, cam_dir = abrir_par_cameras(args)
    mapas = criar_mapas(calibracao)
    stereo = criar_stereo_configurado(args, largura_cal)
    classificador = None if args.sem_modelo else ClassificadorResNet(args.modelo, args.classes)

    pasta_sessao = Path(args.pasta_saida) / f"sessao_{datetime.now():%Y%m%d_%H%M%S}"
    pasta_sessao.mkdir(parents=True, exist_ok=True)
    arquivo_log = pasta_sessao / "metricas_tempo_real.csv"
    log = arquivo_log.open("w", newline="", encoding="utf-8")
    colunas_log = [
        "data_hora",
        "classe",
        "confianca",
        "distancia_m",
        "profundidade_valida",
        "fps",
        "latencia_inferencia_ms",
    ]
    escritor = csv.DictWriter(log, fieldnames=colunas_log)
    escritor.writeheader()

    mostrar_disparidade = True
    contador_total = 0
    contador_fps = 0
    inicio_fps = time.perf_counter()
    fps_medido = 0.0
    historico_distancia: list[float] = []
    classe_atual = "Aguardando"
    confianca_atual = 0.0
    categoria_atual: str | None = None
    latencia_ms = 0.0
    probs_suaves: np.ndarray | None = None
    x1, y1, x2, y2 = regiao_central(largura_cal, altura_cal, args.fracao_roi)

    print("\nSistema iniciado.")
    print(f"Calibracao: {calibracao.origem}")
    print(f"Resultados: {pasta_sessao.resolve()}")
    print("q/ESC: sair | d: alternar disparidade | s: salvar imagens")

    try:
        while True:
            ok, quadro_esq, quadro_dir = ler_par(cam_esq, cam_dir)
            if not ok or quadro_esq is None or quadro_dir is None:
                print("Falha na captura. Encerrando de forma controlada.")
                break
            tamanho_recebido = (quadro_esq.shape[1], quadro_esq.shape[0])
            if tamanho_recebido != calibracao.tamanho:
                raise RuntimeError(
                    f"Resolucao esperada {calibracao.tamanho}, recebida {tamanho_recebido}."
                )

            ret_esq = cv2.remap(quadro_esq, mapas[0], mapas[1], cv2.INTER_LINEAR)
            ret_dir = cv2.remap(quadro_dir, mapas[2], mapas[3], cv2.INTER_LINEAR)
            cinza_esq = cv2.cvtColor(ret_esq, cv2.COLOR_BGR2GRAY)
            cinza_dir = cv2.cvtColor(ret_dir, cv2.COLOR_BGR2GRAY)
            disparidade = stereo.compute(cinza_esq, cinza_dir).astype(np.float32) / 16.0
            pontos_3d = cv2.reprojectImageTo3D(
                disparidade, calibracao.Q, handleMissingValues=False
            )
            pontos_3d *= calibracao.escala_metros

            # A profundidade usa a parte interna da mesma regiao classificada.
            margem_x = int((x2 - x1) * 0.23)
            margem_y = int((y2 - y1) * 0.23)
            regiao_distancia = (x1 + margem_x, y1 + margem_y, x2 - margem_x, y2 - margem_y)
            distancia = mediana_distancia(
                pontos_3d, regiao_distancia, args.dist_min, args.dist_max
            )
            if distancia is None:
                historico_distancia.clear()
                distancia_filtrada = None
            else:
                historico_distancia.append(distancia)
                historico_distancia = historico_distancia[-7:]
                distancia_filtrada = float(np.median(historico_distancia))

            contador_total += 1
            executar_inferencia = (
                classificador is not None
                and (contador_total == 1 or contador_total % args.processar_a_cada == 0)
            )
            if executar_inferencia:
                recorte = ret_esq[y1:y2, x1:x2]
                _, _, probabilidades, latencia_ms = classificador.prever(recorte)
                if probs_suaves is None:
                    probs_suaves = probabilidades
                else:
                    probs_suaves = 0.65 * probs_suaves + 0.35 * probabilidades
                indice = int(np.argmax(probs_suaves))
                classe_atual = classificador.classes[indice]
                confianca_atual = float(probs_suaves[indice])
                categoria_atual = categoria_material(classe_atual)
                escritor.writerow(
                    {
                        "data_hora": datetime.now().isoformat(timespec="milliseconds"),
                        "classe": classe_atual,
                        "confianca": f"{confianca_atual:.6f}",
                        "distancia_m": "" if distancia_filtrada is None else f"{distancia_filtrada:.4f}",
                        "profundidade_valida": distancia_filtrada is not None,
                        "fps": f"{fps_medido:.2f}",
                        "latencia_inferencia_ms": f"{latencia_ms:.2f}",
                    }
                )
                log.flush()

            contador_fps += 1
            agora = time.perf_counter()
            intervalo = agora - inicio_fps
            if intervalo >= 1.0:
                fps_medido = contador_fps / intervalo
                contador_fps = 0
                inicio_fps = agora

            visual = ret_esq.copy()
            reconhecido = categoria_atual is not None and confianca_atual >= args.confianca
            cor = CORES_LIXEIRAS.get(categoria_atual or "", (0, 165, 255))
            cv2.rectangle(visual, (x1, y1), (x2, y2), cor if reconhecido else (0, 165, 255), 2)
            colocar_texto(visual, "POSICIONE UM OBJETO DENTRO DO QUADRO", (12, 28))

            if classificador is None:
                texto_classe = "Classificacao desativada (--sem-modelo)"
                texto_lixeira = ""
            elif reconhecido and categoria_atual is not None:
                texto_classe = f"Material: {NOMES_PT[categoria_atual]} | Confianca: {confianca_atual:.1%}"
                texto_lixeira = LIXEIRAS[categoria_atual]
            else:
                texto_classe = f"Objeto nao reconhecido | Confianca: {confianca_atual:.1%}"
                texto_lixeira = "Aproxime o objeto e mostre apenas um material"

            if distancia_filtrada is None:
                texto_distancia = "Distancia: indisponivel"
                orientacao = "Use objeto com textura e boa iluminacao"
            else:
                texto_distancia = f"Distancia: {distancia_filtrada:.2f} m"
                if distancia_filtrada < args.faixa_min:
                    orientacao = "AFASTE O OBJETO"
                elif distancia_filtrada > args.faixa_max:
                    orientacao = "APROXIME O OBJETO"
                else:
                    orientacao = "Distancia adequada"

            colocar_texto(visual, texto_classe, (12, altura_cal - 82), cor=cor if reconhecido else (0, 165, 255))
            if texto_lixeira:
                colocar_texto(visual, texto_lixeira, (12, altura_cal - 58), cor=cor if reconhecido else (0, 165, 255))
            colocar_texto(visual, f"{texto_distancia} | {orientacao}", (12, altura_cal - 34))
            colocar_texto(
                visual,
                f"FPS: {fps_medido:.1f} | Inferencia: {latencia_ms:.0f} ms",
                (12, altura_cal - 10),
                cor=(90, 255, 120),
            )

            disparidade_colorida = colorir_disparidade(disparidade)
            cv2.rectangle(disparidade_colorida, (x1, y1), (x2, y2), (255, 255, 255), 2)
            colocar_texto(disparidade_colorida, "MAPA DE DISPARIDADE", (12, 28))
            exibicao = np.hstack((visual, disparidade_colorida)) if mostrar_disparidade else visual
            cv2.imshow(NOME_JANELA, exibicao)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("d"):
                mostrar_disparidade = not mostrar_disparidade
            if tecla == ord("s"):
                instante = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(pasta_sessao / f"esquerda_retificada_{instante}.png"), ret_esq)
                cv2.imwrite(str(pasta_sessao / f"direita_retificada_{instante}.png"), ret_dir)
                cv2.imwrite(str(pasta_sessao / f"resultado_{instante}.png"), visual)
                cv2.imwrite(
                    str(pasta_sessao / f"disparidade_{instante}.png"), disparidade_colorida
                )
                print(f"Imagens salvas em {pasta_sessao.resolve()}")
    finally:
        log.close()
        cam_esq.release()
        cam_dir.release()
        cv2.destroyAllWindows()


def construir_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projeto final: classificacao de reciclaveis com visao estereoscopica.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modo",
        required=True,
        choices=("listar", "verificar", "capturar", "calibrar", "validar", "executar"),
        help="Etapa que sera executada.",
    )
    parser.add_argument("--cam-esq", type=int, default=0, help="Indice da camera esquerda.")
    parser.add_argument("--cam-dir", type=int, default=2, help="Indice da camera direita.")
    parser.add_argument("--max-indice", type=int, default=6, help="Maior indice testado no modo listar.")
    parser.add_argument("--largura", type=int, default=640, help="Largura da captura.")
    parser.add_argument("--altura", type=int, default=480, help="Altura da captura.")
    parser.add_argument("--fps", type=int, default=30, help="FPS solicitado as cameras.")

    parser.add_argument(
        "--pasta-calibracao",
        default="calibracao/pares",
        help="Pasta dos pares do tabuleiro.",
    )
    parser.add_argument(
        "--arquivo-calibracao",
        default="calibracao/calibracao_estereo.npz",
        help="Arquivo de parametros .npz ou backup .xml.",
    )
    parser.add_argument("--colunas", type=int, default=6, help="Cantos internos horizontais do tabuleiro.")
    parser.add_argument("--linhas", type=int, default=8, help="Cantos internos verticais do tabuleiro.")
    parser.add_argument(
        "--quadrado",
        type=float,
        default=0.030,
        help="Lado real do quadrado em metros. Neste tabuleiro: 30 mm = 0.030 m.",
    )
    parser.add_argument("--min-pares", type=int, default=20, help="Quantidade minima de pares validos.")
    parser.add_argument(
        "--intervalo-deteccao",
        type=int,
        default=3,
        help="Executa a busca rapida do tabuleiro a cada N quadros.",
    )
    parser.add_argument(
        "--travar-foco",
        action="store_true",
        help="Desativa o autofoco depois que as cameras sao abertas.",
    )
    parser.add_argument(
        "--foco",
        type=float,
        default=None,
        help="Valor manual de foco, dependente do modelo da webcam.",
    )
    parser.add_argument(
        "--exposicao",
        type=float,
        default=None,
        help="Valor manual de exposicao, dependente do modelo da webcam.",
    )

    parser.add_argument("--num-disparidades", type=int, default=128, help="Faixa do StereoSGBM; multiplo de 16.")
    parser.add_argument("--bloco", type=int, default=5, help="Tamanho impar do bloco StereoSGBM.")
    parser.add_argument("--dist-min", type=float, default=0.20, help="Menor distancia aceita, em metros.")
    parser.add_argument("--dist-max", type=float, default=3.00, help="Maior distancia aceita, em metros.")
    parser.add_argument(
        "--config-sgbm",
        default="config_sgbm.json",
        help="Parametros do StereoSGBM em JSON.",
    )
    parser.add_argument(
        "--fator-unidade-m",
        type=float,
        default=0.030,
        help="Conversao para metros ao usar o XML antigo do Lab 6.",
    )

    parser.add_argument(
        "--modelo",
        default="modelos/resnet50_waste.keras",
        help="Modelo ResNet-50 .onnx, .keras ou .h5.",
    )
    parser.add_argument(
        "--classes",
        default="modelos/class_names.json",
        help="Arquivo JSON com a ordem das seis classes.",
    )
    parser.add_argument("--sem-modelo", action="store_true", help="Executa somente estereo e profundidade.")
    parser.add_argument("--confianca", type=float, default=0.60, help="Confianca minima da classificacao.")
    parser.add_argument(
        "--processar-a-cada",
        type=int,
        default=3,
        help="Executa a ResNet a cada N quadros.",
    )
    parser.add_argument(
        "--fracao-roi",
        type=float,
        default=0.58,
        help="Tamanho relativo do quadrado central classificado.",
    )
    parser.add_argument("--faixa-min", type=float, default=0.30, help="Inicio da faixa recomendada em metros.")
    parser.add_argument("--faixa-max", type=float, default=1.50, help="Fim da faixa recomendada em metros.")
    parser.add_argument("--pasta-saida", default="resultados", help="Pasta para imagens salvas.")
    return parser.parse_args(argumentos)


def main(argumentos: list[str] | None = None) -> None:
    args = construir_argumentos(argumentos)
    try:
        if args.modo == "listar":
            listar_cameras(args)
        elif args.modo == "verificar":
            verificar_cameras(args)
        elif args.modo == "capturar":
            capturar_calibracao(args)
        elif args.modo == "calibrar":
            calibrar_estereo(args)
        elif args.modo == "validar":
            validar_calibracao(args)
        elif args.modo == "executar":
            executar_sistema(args)
    except KeyboardInterrupt:
        print("\nExecucao interrompida pelo usuario.")
    except (RuntimeError, cv2.error, ValueError) as erro:
        print(f"\nERRO: {erro}")
        sys.exit(1)


if __name__ == "__main__":
    main()
