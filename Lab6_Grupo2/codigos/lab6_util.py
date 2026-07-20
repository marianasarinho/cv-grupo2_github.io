#!/usr/bin/env python3
"""Funcoes auxiliares do Lab 6 - disparidade, retificacao e profundidade."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CalibracaoStereo:
    caminho: Path
    image_size: Tuple[int, int]
    M1: np.ndarray
    D1: np.ndarray
    M2: np.ndarray
    D2: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    square_size: float = 1.0


def _ler_matriz(fs: cv2.FileStorage, nome: str, obrigatoria: bool = True) -> Optional[np.ndarray]:
    node = fs.getNode(nome)
    if node.empty():
        if obrigatoria:
            raise KeyError(f"Parametro ausente no arquivo de calibracao: {nome}")
        return None
    mat = node.mat()
    if mat is None and obrigatoria:
        raise ValueError(f"Nao foi possivel ler a matriz {nome}")
    return mat


def _ler_real(fs: cv2.FileStorage, nome: str, padrao: float) -> float:
    node = fs.getNode(nome)
    return float(node.real()) if not node.empty() else float(padrao)


def carregar_calibracao_xml(caminho: str | os.PathLike[str]) -> CalibracaoStereo:
    """Carrega o XML/YAML do Lab 5 e cria mapas de retificacao."""
    path = Path(caminho).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de calibracao nao encontrado: {path}\n"
            "Copie o stereo_params_abc.xml do Lab 5 para a pasta calibracao/."
        )

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise OSError(f"Nao foi possivel abrir o arquivo de calibracao: {path}")

    try:
        width = int(round(_ler_real(fs, "image_width", 640)))
        height = int(round(_ler_real(fs, "image_height", 480)))
        square_size = _ler_real(fs, "square_size", 1.0)
        M1 = _ler_matriz(fs, "M1")
        D1 = _ler_matriz(fs, "D1")
        M2 = _ler_matriz(fs, "M2")
        D2 = _ler_matriz(fs, "D2")
        R1 = _ler_matriz(fs, "R1")
        R2 = _ler_matriz(fs, "R2")
        P1 = _ler_matriz(fs, "P1")
        P2 = _ler_matriz(fs, "P2")
        Q = _ler_matriz(fs, "Q")
    finally:
        fs.release()

    size = (width, height)
    map1x, map1y = cv2.initUndistortRectifyMap(
        M1, D1, R1, P1, size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        M2, D2, R2, P2, size, cv2.CV_32FC1
    )

    return CalibracaoStereo(
        caminho=path,
        image_size=size,
        M1=M1,
        D1=D1,
        M2=M2,
        D2=D2,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        square_size=square_size,
    )


def retificar_par(
    esquerda: np.ndarray,
    direita: np.ndarray,
    calib: CalibracaoStereo,
) -> Tuple[np.ndarray, np.ndarray]:
    """Redimensiona para a resolucao calibrada e retifica o par."""
    size = calib.image_size
    if (esquerda.shape[1], esquerda.shape[0]) != size:
        esquerda = cv2.resize(esquerda, size, interpolation=cv2.INTER_AREA)
    if (direita.shape[1], direita.shape[0]) != size:
        direita = cv2.resize(direita, size, interpolation=cv2.INTER_AREA)

    rect_l = cv2.remap(esquerda, calib.map1x, calib.map1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(direita, calib.map2x, calib.map2y, cv2.INTER_LINEAR)
    return rect_l, rect_r


def parametros_padrao_sgbm() -> Dict[str, int]:
    return {
        "min_disparity": 0,
        "num_disparities": 128,
        "block_size": 5,
        "uniqueness_ratio": 10,
        "speckle_window_size": 100,
        "speckle_range": 2,
        "disp12_max_diff": 1,
        "pre_filter_cap": 31,
    }


def normalizar_parametros(params: Dict[str, Any]) -> Dict[str, int]:
    p = parametros_padrao_sgbm()
    p.update({k: int(v) for k, v in params.items() if k in p})

    p["num_disparities"] = max(16, int(math.ceil(p["num_disparities"] / 16.0)) * 16)
    p["block_size"] = max(3, p["block_size"])
    if p["block_size"] % 2 == 0:
        p["block_size"] += 1
    p["pre_filter_cap"] = int(np.clip(p["pre_filter_cap"], 1, 63))
    p["uniqueness_ratio"] = max(0, p["uniqueness_ratio"])
    p["speckle_window_size"] = max(0, p["speckle_window_size"])
    p["speckle_range"] = max(0, p["speckle_range"])
    p["disp12_max_diff"] = max(-1, p["disp12_max_diff"])
    return p


def criar_sgbm(params: Dict[str, Any], canais: int = 1) -> cv2.StereoSGBM:
    p = normalizar_parametros(params)
    bloco = p["block_size"]
    p1 = 8 * canais * bloco * bloco
    p2 = 32 * canais * bloco * bloco
    return cv2.StereoSGBM_create(
        minDisparity=p["min_disparity"],
        numDisparities=p["num_disparities"],
        blockSize=bloco,
        P1=p1,
        P2=p2,
        disp12MaxDiff=p["disp12_max_diff"],
        preFilterCap=p["pre_filter_cap"],
        uniquenessRatio=p["uniqueness_ratio"],
        speckleWindowSize=p["speckle_window_size"],
        speckleRange=p["speckle_range"],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def calcular_disparidade(
    esquerda_gray: np.ndarray,
    direita_gray: np.ndarray,
    params: Dict[str, Any],
    filtro: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula disparidade e aplica regularizacao para reduzir manchas.

    filtro:
      - "auto": usa WLS se cv2.ximgproc estiver disponível; caso contrário, mediana.
      - "wls": tenta WLS e cai para mediana se indisponível.
      - "mediana": filtro de mediana 5x5.
      - "nenhum": resultado bruto do StereoSGBM.
    """
    p = normalizar_parametros(params)
    matcher_esq = criar_sgbm(p, canais=1)
    disp16_esq = matcher_esq.compute(esquerda_gray, direita_gray)

    filtro = str(filtro).lower().strip()
    disp16_filtrada = disp16_esq

    usar_wls = filtro in ("auto", "wls")
    tem_ximgproc = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher")

    if usar_wls and tem_ximgproc:
        try:
            matcher_dir = cv2.ximgproc.createRightMatcher(matcher_esq)
            disp16_dir = matcher_dir.compute(direita_gray, esquerda_gray)

            wls = cv2.ximgproc.createDisparityWLSFilter(matcher_esq)
            wls.setLambda(8000.0)
            wls.setSigmaColor(1.5)
            disp16_filtrada = wls.filter(
                disp16_esq,
                esquerda_gray,
                disparity_map_right=disp16_dir,
            )
        except cv2.error:
            disp16_filtrada = disp16_esq

    if filtro == "mediana" or (usar_wls and not tem_ximgproc) or (
        usar_wls and np.shares_memory(disp16_filtrada, disp16_esq)
    ):
        # Suaviza pequenas oscilações mantendo as regiões originalmente inválidas.
        limite_invalido = (p["min_disparity"] - 1) * 16
        invalid = disp16_esq <= limite_invalido
        filtrada = cv2.medianBlur(disp16_esq, 5)
        filtrada[invalid] = disp16_esq[invalid]
        disp16_filtrada = filtrada

    if filtro == "nenhum":
        disp16_filtrada = disp16_esq

    disp = disp16_filtrada.astype(np.float32) / 16.0
    return disp, disp16_filtrada


def mascara_disparidade_valida(disparidade: np.ndarray, min_disparity: int = 0) -> np.ndarray:
    return np.isfinite(disparidade) & (disparidade > float(min_disparity))


def disparidade_visual(disparidade: np.ndarray, min_disparity: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna mapa em cinza e colorido com escala robusta por percentis."""
    valid = mascara_disparidade_valida(disparidade, min_disparity)
    gray = np.zeros(disparidade.shape, dtype=np.uint8)
    if np.count_nonzero(valid) > 50:
        vals = disparidade[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((disparidade - lo) * 255.0 / (hi - lo), 0, 255)
        gray[valid] = norm[valid].astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return gray, color


def reprojetar_3d(disparidade: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pontos = cv2.reprojectImageTo3D(disparidade, Q, handleMissingValues=False)
    valid = mascara_disparidade_valida(disparidade, 0)
    valid &= np.all(np.isfinite(pontos), axis=2)
    valid &= np.abs(pontos[:, :, 2]) < 1e6
    return pontos, valid


def mediana_profundidade(
    pontos3d: np.ndarray,
    valid: np.ndarray,
    x: int,
    y: int,
    raio: int = 3,
) -> Optional[np.ndarray]:
    h, w = valid.shape
    x0, x1 = max(0, x - raio), min(w, x + raio + 1)
    y0, y1 = max(0, y - raio), min(h, y + raio + 1)
    mask = valid[y0:y1, x0:x1]
    if np.count_nonzero(mask) < 3:
        return None
    pts = pontos3d[y0:y1, x0:x1][mask]
    return np.median(pts, axis=0)


def salvar_ply(
    caminho: str | os.PathLike[str],
    pontos3d: np.ndarray,
    cores_bgr: np.ndarray,
    valid: np.ndarray,
    max_pontos: int = 400_000,
) -> int:
    pts = pontos3d[valid]
    cores = cores_bgr[valid][:, ::-1]  # BGR -> RGB
    if len(pts) == 0:
        raise ValueError("Nao ha pontos 3D validos para salvar.")

    if len(pts) > max_pontos:
        passo = int(math.ceil(len(pts) / max_pontos))
        pts = pts[::passo]
        cores = cores[::passo]

    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as arq:
        arq.write("ply\n")
        arq.write("format ascii 1.0\n")
        arq.write(f"element vertex {len(pts)}\n")
        arq.write("property float x\nproperty float y\nproperty float z\n")
        arq.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        arq.write("end_header\n")
        for p, c in zip(pts, cores):
            arq.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
    return len(pts)


def carregar_params_json(caminho: str | os.PathLike[str]) -> Dict[str, int]:
    path = Path(caminho)
    if not path.exists():
        return parametros_padrao_sgbm()
    with path.open("r", encoding="utf-8") as arq:
        return normalizar_parametros(json.load(arq))


def salvar_params_json(caminho: str | os.PathLike[str], params: Dict[str, Any]) -> None:
    path = Path(caminho)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as arq:
        json.dump(normalizar_parametros(params), arq, indent=2, ensure_ascii=False)


def abrir_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Abre webcam no Linux, tentando V4L2 e depois o backend automatico."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera de indice {index}.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capturar_sincronizado(
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    ok_l = cap_l.grab()
    ok_r = cap_r.grab()
    if not (ok_l and ok_r):
        return False, None, None
    ok_l, frame_l = cap_l.retrieve()
    ok_r, frame_r = cap_r.retrieve()
    return bool(ok_l and ok_r), frame_l, frame_r


def desenhar_linhas_epipolares(imagem: np.ndarray, passo: int = 40) -> np.ndarray:
    out = imagem.copy()
    for y in range(passo, out.shape[0], passo):
        cv2.line(out, (0, y), (out.shape[1] - 1, y), (0, 255, 0), 1)
    return out


def colocar_texto(
    imagem: np.ndarray,
    texto: str,
    pos: Tuple[int, int] = (10, 25),
    escala: float = 0.6,
) -> None:
    cv2.putText(imagem, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(imagem, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1, cv2.LINE_AA)
