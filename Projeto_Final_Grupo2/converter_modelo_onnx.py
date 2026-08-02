#!/usr/bin/env python3
"""Converte a ResNet-50 Keras para ONNX antes da apresentacao no Linux."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default="modelos/resnet50_waste.keras")
    parser.add_argument("--saida", default="modelos/resnet50_waste.onnx")
    parser.add_argument("--opset", type=int, default=15)
    args = parser.parse_args()

    try:
        import onnx
        import tensorflow as tf
        import tf2onnx
    except ImportError as erro:
        raise SystemExit(
            "Instale as dependencias de conversao: "
            "python -m pip install -r requirements-conversao.txt"
        ) from erro

    entrada = Path(args.entrada)
    saida = Path(args.saida)
    if not entrada.exists():
        raise SystemExit(f"Modelo nao encontrado: {entrada}")
    saida.parent.mkdir(parents=True, exist_ok=True)

    modelo = tf.keras.models.load_model(str(entrada), compile=False)
    assinatura = (
        tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_image"),
    )
    tf2onnx.convert.from_keras(
        modelo,
        input_signature=assinatura,
        opset=args.opset,
        inputs_as_nchw=["input_image:0"],
        output_path=str(saida),
    )
    onnx.checker.check_model(onnx.load(str(saida)))
    print(f"Modelo ONNX salvo em: {saida.resolve()}")
    print("Estrutura ONNX validada com sucesso.")
    print("Execute agora: python verificar_ambiente.py")


if __name__ == "__main__":
    main()
