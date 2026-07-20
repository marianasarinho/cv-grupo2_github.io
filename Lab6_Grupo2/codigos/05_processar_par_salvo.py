#!/usr/bin/env python3
"""Retifica um par salvo, calcula disparidade, profundidade e nuvem de pontos."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from lab6_util import (
    calcular_disparidade,
    carregar_calibracao_xml,
    carregar_params_json,
    disparidade_visual,
    reprojetar_3d,
    retificar_par,
    salvar_ply,
)


def encontrar_ultimo_par(pasta: Path) -> Tuple[Path, Path]:
    esquerdas = sorted(pasta.glob("*_L.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for esq in esquerdas:
        dir_path = esq.with_name(esq.name.replace("_L.png", "_R.png"))
        if dir_path.exists():
            return esq, dir_path
    raise FileNotFoundError(
        f"Nenhum par *_L.png / *_R.png encontrado em {pasta}. "
        "Execute primeiro 03_capturar_par_estereo.py."
    )


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--esquerda", default="")
    parser.add_argument("--direita", default="")
    parser.add_argument(
        "--calibracao",
        default=str(base / "calibracao" / "stereo_params_abc.xml"),
    )
    parser.add_argument("--config", default=str(base / "config_sgbm.json"))
    parser.add_argument("--saida", default=str(base / "resultados" / "par_processado"))
    parser.add_argument(
        "--fator-unidade-cm",
        type=float,
        default=0.0,
        help="Ex.: 3.0 se 1 unidade da calibracao representa 3 cm. Zero mantem unidade original.",
    )
    parser.add_argument("--trocar", action="store_true")
    parser.add_argument("--sem-ply", action="store_true")
    args = parser.parse_args()

    if args.esquerda and args.direita:
        path_l, path_r = Path(args.esquerda), Path(args.direita)
    elif args.esquerda or args.direita:
        raise ValueError("Informe as duas imagens ou nenhuma delas.")
    else:
        path_l, path_r = encontrar_ultimo_par(base / "capturas")

    img_l = cv2.imread(str(path_l))
    img_r = cv2.imread(str(path_r))
    if img_l is None or img_r is None:
        raise FileNotFoundError("Nao foi possivel abrir o par informado.")
    if args.trocar:
        img_l, img_r = img_r, img_l
        path_l, path_r = path_r, path_l

    calib = carregar_calibracao_xml(args.calibracao)
    rect_l, rect_r = retificar_par(img_l, img_r, calib)
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    params = carregar_params_json(args.config)
    disparity, _ = calcular_disparidade(gray_l, gray_r, params)
    disp_gray, disp_color = disparidade_visual(disparity, params["min_disparity"])
    pontos3d, valid = reprojetar_3d(disparity, calib.Q)

    fator_cm = float(args.fator_unidade_cm)
    pontos_saida = pontos3d.copy()
    unidade = "unidade da calibracao"
    if fator_cm > 0:
        pontos_saida *= fator_cm
        unidade = "cm"

    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "01_esquerda_original.png"), img_l)
    cv2.imwrite(str(out_dir / "02_direita_original.png"), img_r)
    cv2.imwrite(str(out_dir / "03_esquerda_retificada.png"), rect_l)
    cv2.imwrite(str(out_dir / "04_direita_retificada.png"), rect_r)
    cv2.imwrite(str(out_dir / "05_disparidade_cinza.png"), disp_gray)
    cv2.imwrite(str(out_dir / "06_disparidade_colorida.png"), disp_color)
    np.save(str(out_dir / "disparidade_float.npy"), disparity)
    np.save(str(out_dir / "pontos3d.npy"), pontos_saida)
    np.save(str(out_dir / "mascara_valida.npy"), valid)

    qtd_ply = 0
    if not args.sem_ply:
        qtd_ply = salvar_ply(out_dir / "nuvem_pontos.ply", pontos_saida, rect_l, valid)

    z = np.abs(pontos_saida[:, :, 2][valid])
    resumo = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "imagem_esquerda": str(path_l.resolve()),
        "imagem_direita": str(path_r.resolve()),
        "calibracao": str(calib.caminho),
        "resolucao": list(calib.image_size),
        "unidade_profundidade": unidade,
        "fator_unidade_cm": fator_cm,
        "parametros_sgbm": params,
        "pixels_validos": int(np.count_nonzero(valid)),
        "pontos_ply": int(qtd_ply),
        "profundidade_abs_mediana": float(np.median(z)) if z.size else None,
        "profundidade_abs_percentil_10": float(np.percentile(z, 10)) if z.size else None,
        "profundidade_abs_percentil_90": float(np.percentile(z, 90)) if z.size else None,
    }
    (out_dir / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    painel_top = np.hstack([rect_l, rect_r])
    painel_bottom = np.hstack([cv2.cvtColor(disp_gray, cv2.COLOR_GRAY2BGR), disp_color])
    cv2.imwrite(str(out_dir / "07_painel_resultados.png"), np.vstack([painel_top, painel_bottom]))

    print("Processamento concluido.")
    print("Resultados em:", out_dir.resolve())
    print("Pixels 3D validos:", resumo["pixels_validos"])
    print("Unidade da profundidade:", unidade)
    if fator_cm <= 0:
        print("ATENCAO: para obter cm, informe --fator-unidade-cm com o tamanho real correspondente a 1 unidade do Lab 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
