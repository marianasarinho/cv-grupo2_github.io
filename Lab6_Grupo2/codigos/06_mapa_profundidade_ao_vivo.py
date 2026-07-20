#!/usr/bin/env python3
"""Mapa de disparidade/profundidade ao vivo com duas webcams calibradas."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from lab6_util import (
    abrir_camera,
    calcular_disparidade,
    carregar_calibracao_xml,
    carregar_params_json,
    capturar_sincronizado,
    disparidade_visual,
    mediana_profundidade,
    colocar_texto,
    reprojetar_3d,
    retificar_par,
    salvar_params_json,
)

JANELA_CTRL = "Controles ao vivo"
JANELA_DISP = "Mapa de profundidade"


def nada(_: int) -> None:
    pass


def criar_controles(params: Dict[str, int]) -> None:
    cv2.namedWindow(JANELA_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA_CTRL, 700, 420)
    cv2.createTrackbar("minDisp + 64", JANELA_CTRL, params["min_disparity"] + 64, 128, nada)
    cv2.createTrackbar("numDisp x16", JANELA_CTRL, max(1, params["num_disparities"] // 16), 20, nada)
    cv2.createTrackbar("block (2n+1)", JANELA_CTRL, max(1, (params["block_size"] - 1) // 2), 12, nada)
    cv2.createTrackbar("uniqueness", JANELA_CTRL, params["uniqueness_ratio"], 50, nada)
    cv2.createTrackbar("speckleWin", JANELA_CTRL, params["speckle_window_size"], 300, nada)
    cv2.createTrackbar("speckleRange", JANELA_CTRL, params["speckle_range"], 32, nada)


def ler_controles(params_base: Dict[str, int]) -> Dict[str, int]:
    p = dict(params_base)
    p.update(
        {
            "min_disparity": cv2.getTrackbarPos("minDisp + 64", JANELA_CTRL) - 64,
            "num_disparities": max(1, cv2.getTrackbarPos("numDisp x16", JANELA_CTRL)) * 16,
            "block_size": max(1, cv2.getTrackbarPos("block (2n+1)", JANELA_CTRL)) * 2 + 1,
            "uniqueness_ratio": cv2.getTrackbarPos("uniqueness", JANELA_CTRL),
            "speckle_window_size": cv2.getTrackbarPos("speckleWin", JANELA_CTRL),
            "speckle_range": cv2.getTrackbarPos("speckleRange", JANELA_CTRL),
        }
    )
    return p


def atualizar_painel_controles(params: Dict[str, int]) -> None:
    """Mostra os valores reais escolhidos em uma faixa legivel."""
    painel = np.zeros((210, 700, 3), dtype=np.uint8)
    linhas = [
        f"minDisparity       = {params['min_disparity']}",
        f"numDisparities    = {params['num_disparities']}  (multiplo de 16)",
        f"blockSize         = {params['block_size']}",
        f"uniquenessRatio   = {params['uniqueness_ratio']}",
        f"speckleWindowSize = {params['speckle_window_size']}",
        f"speckleRange      = {params['speckle_range']}",
    ]
    for i, texto in enumerate(linhas):
        cv2.putText(painel, texto, (18, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(painel, "Atalhos: 1=conservador  2=equilibrado  3=detalhado  4=plano  P=salvar",
                (18, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imshow(JANELA_CTRL, painel)


def aplicar_preset(indice: int) -> Dict[str, int]:
    presets = {
        1: dict(min_disparity=0, num_disparities=96, block_size=9, uniqueness_ratio=12, speckle_window_size=120, speckle_range=2),
        2: dict(min_disparity=0, num_disparities=128, block_size=7, uniqueness_ratio=8, speckle_window_size=100, speckle_range=2),
        3: dict(min_disparity=0, num_disparities=160, block_size=5, uniqueness_ratio=6, speckle_window_size=80, speckle_range=1),
        4: dict(min_disparity=0, num_disparities=96, block_size=11, uniqueness_ratio=18, speckle_window_size=180, speckle_range=2),
    }
    return presets[indice]


def escrever_controles(params: Dict[str, int]) -> None:
    cv2.setTrackbarPos("minDisp + 64", JANELA_CTRL, params["min_disparity"] + 64)
    cv2.setTrackbarPos("numDisp x16", JANELA_CTRL, max(1, params["num_disparities"] // 16))
    cv2.setTrackbarPos("block (2n+1)", JANELA_CTRL, max(1, (params["block_size"] - 1) // 2))
    cv2.setTrackbarPos("uniqueness", JANELA_CTRL, params["uniqueness_ratio"])
    cv2.setTrackbarPos("speckleWin", JANELA_CTRL, params["speckle_window_size"])
    cv2.setTrackbarPos("speckleRange", JANELA_CTRL, params["speckle_range"])


class CliqueProfundidade:
    def __init__(self, fator_cm: float, raio: int = 12) -> None:
        self.pontos3d: Optional[np.ndarray] = None
        self.valid: Optional[np.ndarray] = None
        self.xy: Optional[Tuple[int, int]] = None
        self.valor: Optional[np.ndarray] = None
        self.fator_cm = fator_cm
        self.raio = max(2, int(raio))

    def callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.pontos3d is None or self.valid is None:
            return
        valor = mediana_profundidade(self.pontos3d, self.valid, x, y, raio=self.raio)
        self.xy = (x, y)
        self.valor = valor
        if valor is None:
            print(f"Clique ({x}, {y}): sem profundidade valida.")
            return
        z = abs(float(valor[2]))
        if self.fator_cm > 0:
            print(f"Clique ({x}, {y}): X={valor[0]*self.fator_cm:.2f} cm, Y={valor[1]*self.fator_cm:.2f} cm, Z={z*self.fator_cm:.2f} cm")
        else:
            print(f"Clique ({x}, {y}): X={valor[0]:.3f}, Y={valor[1]:.3f}, |Z|={z:.3f} unidades da calibracao")


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-esq", type=int, default=0)
    parser.add_argument("--cam-dir", type=int, default=2)
    parser.add_argument("--calibracao", default=str(base / "calibracao" / "stereo_params_abc.xml"))
    parser.add_argument("--config", default=str(base / "config_sgbm.json"))
    parser.add_argument("--saida", default=str(base / "resultados" / "ao_vivo"))
    parser.add_argument("--trocar", action="store_true")
    parser.add_argument("--fator-unidade-cm", type=float, default=0.0)
    parser.add_argument("--sem-controles", action="store_true")
    parser.add_argument("--filtro", choices=["auto", "wls", "mediana", "nenhum"], default="auto")
    parser.add_argument("--raio-medicao", type=int, default=12,
                        help="Raio da janela usada na mediana da profundidade. 12 gera uma região 25x25.")
    parser.add_argument("--z-min-cm", type=float, default=20.0)
    parser.add_argument("--z-max-cm", type=float, default=200.0)
    args = parser.parse_args()

    calib = carregar_calibracao_xml(args.calibracao)
    width, height = calib.image_size
    cap_l = abrir_camera(args.cam_esq, width, height)
    cap_r = abrir_camera(args.cam_dir, width, height)
    params_base = carregar_params_json(args.config)
    if not args.sem_controles:
        criar_controles(params_base)

    clique = CliqueProfundidade(args.fator_unidade_cm, args.raio_medicao)
    cv2.namedWindow(JANELA_DISP, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(JANELA_DISP, clique.callback)

    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    params = params_base

    print("Teclas: S salva quadro | P salva parametros | 1/2/3/4 aplica presets | Q/ESC sai.")
    print("Clique no mapa colorido para medir a mediana da profundidade local.")
    print(f"Filtro de disparidade: {args.filtro}; janela de medição: {2*args.raio_medicao+1}x{2*args.raio_medicao+1}.")

    try:
        while True:
            ok, frame_l, frame_r = capturar_sincronizado(cap_l, cap_r)
            if not ok or frame_l is None or frame_r is None:
                print("Falha na captura das cameras.")
                break
            if args.trocar:
                frame_l, frame_r = frame_r, frame_l

            rect_l, rect_r = retificar_par(frame_l, frame_r, calib)
            gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
            if not args.sem_controles:
                params = ler_controles(params_base)
                atualizar_painel_controles(params)

            disparity, _ = calcular_disparidade(
                gray_l, gray_r, params, filtro=args.filtro
            )
            disp_gray, disp_color = disparidade_visual(disparity, params["min_disparity"])
            pontos3d, valid = reprojetar_3d(disparity, calib.Q)
            clique.pontos3d = pontos3d
            clique.valid = valid

            # Quando a unidade é conhecida, exibe profundidade em uma escala fixa.
            # Isso evita que pequenas oscilações sejam exageradas pela normalização
            # automática por percentis em cada quadro.
            if args.fator_unidade_cm > 0 and args.z_max_cm > args.z_min_cm:
                profundidade_cm = np.abs(pontos3d[:, :, 2]) * args.fator_unidade_cm
                valid_z = (
                    valid
                    & np.isfinite(profundidade_cm)
                    & (profundidade_cm >= args.z_min_cm)
                    & (profundidade_cm <= args.z_max_cm)
                )
                depth_gray = np.zeros(profundidade_cm.shape, dtype=np.uint8)
                escala = 255.0 / (args.z_max_cm - args.z_min_cm)
                norm = (args.z_max_cm - profundidade_cm) * escala
                depth_gray[valid_z] = np.clip(norm[valid_z], 0, 255).astype(np.uint8)
                vis_disp = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)
                vis_disp[~valid_z] = 0
            else:
                vis_disp = disp_color.copy()
            colocar_texto(vis_disp, f"num={params['num_disparities']} bloco={params['block_size']}")
            if clique.xy is not None:
                cv2.drawMarker(vis_disp, clique.xy, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                if clique.valor is not None:
                    z = abs(float(clique.valor[2]))
                    texto = f"Z={z * args.fator_unidade_cm:.1f} cm" if args.fator_unidade_cm > 0 else f"|Z|={z:.2f} u."
                    x_text = min(max(5, clique.xy[0] + 10), vis_disp.shape[1] - 180)
                    y_text = min(max(25, clique.xy[1] - 10), vis_disp.shape[0] - 10)
                    colocar_texto(vis_disp, texto, (x_text, y_text), 0.55)

            painel_rect = np.hstack([rect_l, rect_r])
            for y in range(40, painel_rect.shape[0], 40):
                cv2.line(painel_rect, (0, y), (painel_rect.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.imshow("Par retificado com linhas epipolares", painel_rect)
            cv2.imshow(JANELA_DISP, vis_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if not args.sem_controles and key in (ord("1"), ord("2"), ord("3"), ord("4")):
                params_base.update(aplicar_preset(int(chr(key))))
                escrever_controles(params_base)
                params = dict(params_base)
                print(f"Preset {chr(key)} aplicado: {params}")
            if key in (ord("p"), ord("P")):
                salvar_params_json(args.config, params)
                print("Parametros salvos em:", args.config)
            if key in (ord("s"), ord("S")):
                frame_count += 1
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = out_dir / f"captura_{frame_count:03d}_{stamp}"
                cv2.imwrite(str(prefix.with_name(prefix.name + "_L_rect.png")), rect_l)
                cv2.imwrite(str(prefix.with_name(prefix.name + "_R_rect.png")), rect_r)
                cv2.imwrite(str(prefix.with_name(prefix.name + "_disp.png")), disp_color)
                np.save(str(prefix.with_name(prefix.name + "_disparidade.npy")), disparity)
                np.save(str(prefix.with_name(prefix.name + "_pontos3d.npy")), pontos3d)
                salvar_params_json(args.config, params)
                print("Quadro salvo com prefixo:", prefix)
    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
