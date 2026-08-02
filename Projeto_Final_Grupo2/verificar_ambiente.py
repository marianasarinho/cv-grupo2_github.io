#!/usr/bin/env python3
"""Verifica o pacote antes dos testes no Windows ou da apresentacao no Linux."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
CLASSES_ESPERADAS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def status(ok: bool, mensagem: str) -> None:
    print(f"[{'OK' if ok else 'FALHA'}] {mensagem}")


def testar_cameras(indice_esq: int, indice_dir: int) -> bool:
    import cv2

    capturas = []
    resultado = True
    try:
        for indice in (indice_esq, indice_dir):
            backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
            captura = cv2.VideoCapture(indice, backend)
            captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capturas.append(captura)
            ok, frame = captura.read()
            valido = bool(captura.isOpened() and ok and frame is not None)
            status(valido, f"Camera {indice} capturou um quadro")
            resultado = resultado and valido
    finally:
        for captura in capturas:
            captura.release()
    return resultado


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testar-cameras", action="store_true")
    parser.add_argument("--cam-esq", type=int, default=0)
    parser.add_argument("--cam-dir", type=int, default=2)
    args = parser.parse_args()

    print("=" * 62)
    print("VERIFICACAO DO PROJETO FINAL - GRUPO 2")
    print("=" * 62)
    print(f"Sistema: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    tudo_ok = sys.version_info >= (3, 10)
    status(tudo_ok, "Python 3.10 ou superior")

    for pacote in ("numpy", "cv2"):
        existe = importlib.util.find_spec(pacote) is not None
        status(existe, f"Biblioteca {pacote} instalada")
        tudo_ok = tudo_ok and existe

    if importlib.util.find_spec("cv2") is not None:
        import cv2

        status(hasattr(cv2, "StereoSGBM_create"), f"OpenCV {cv2.__version__} com StereoSGBM")

    classes_path = RAIZ / "modelos" / "class_names.json"
    classes_ok = False
    if classes_path.exists():
        try:
            classes = json.loads(classes_path.read_text(encoding="utf-8"))
            classes_ok = sorted(classes) == sorted(CLASSES_ESPERADAS)
        except (OSError, json.JSONDecodeError):
            classes_ok = False
    status(classes_ok, "class_names.json possui as seis classes corretas")
    tudo_ok = tudo_ok and classes_ok

    onnx = RAIZ / "modelos" / "resnet50_waste.onnx"
    keras = RAIZ / "modelos" / "resnet50_waste.keras"
    # O modelo Keras foi o formato validado nos testes finais do grupo.
    # O ONNX continua aceito como alternativa para ambientes sem TensorFlow.
    modelo = keras if keras.exists() else onnx
    modelo_ok = modelo.exists()
    status(modelo_ok, f"Modelo encontrado: {modelo.name if modelo_ok else 'nenhum .onnx/.keras'}")
    tudo_ok = tudo_ok and modelo_ok
    if modelo_ok and modelo.suffix == ".keras":
        tensorflow_ok = importlib.util.find_spec("tensorflow") is not None
        status(tensorflow_ok, "TensorFlow instalado para abrir o modelo .keras")
        tudo_ok = tudo_ok and tensorflow_ok
    if modelo_ok and modelo.suffix == ".onnx" and importlib.util.find_spec("cv2") is not None:
        try:
            cv2.dnn.readNetFromONNX(str(modelo))
            onnx_ok = True
        except cv2.error:
            onnx_ok = False
        status(onnx_ok, "OpenCV conseguiu carregar o modelo ONNX")
        tudo_ok = tudo_ok and onnx_ok

    calibracao_nova = RAIZ / "calibracao" / "calibracao_estereo.npz"
    calibracao_backup = RAIZ / "calibracao" / "stereo_params_abc_backup.xml"
    calibracao_ok = calibracao_nova.exists() or calibracao_backup.exists()
    nome_calibracao = (
        calibracao_nova.name
        if calibracao_nova.exists()
        else calibracao_backup.name if calibracao_backup.exists() else "nenhuma"
    )
    status(calibracao_ok, f"Calibracao disponivel: {nome_calibracao}")
    tudo_ok = tudo_ok and calibracao_ok

    sgbm_ok = (RAIZ / "config_sgbm.json").exists()
    status(sgbm_ok, "Arquivo config_sgbm.json encontrado")
    tudo_ok = tudo_ok and sgbm_ok

    if args.testar_cameras and importlib.util.find_spec("cv2") is not None:
        cameras_ok = testar_cameras(args.cam_esq, args.cam_dir)
        tudo_ok = tudo_ok and cameras_ok

    print("-" * 62)
    if tudo_ok:
        print("AMBIENTE PRONTO PARA EXECUTAR O PROJETO.")
        return 0
    print("AMBIENTE AINDA INCOMPLETO. Corrija os itens marcados como FALHA.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
