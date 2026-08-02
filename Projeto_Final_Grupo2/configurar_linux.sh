#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Projeto Final - Grupo 2"
echo "Python: $(python3 --version)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "CV26" ]]; then
    echo "AVISO: o ambiente CV26 nao esta ativo."
    echo "Se ele existir, interrompa e execute: conda activate CV26"
fi

if python3 -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "OpenCV e NumPy ja estao instalados."
else
    echo "Instalando apenas as dependencias leves necessarias..."
    python3 -m pip install -r requirements-linux.txt
fi

python3 verificar_ambiente.py || true

echo
echo "Para testar as webcams:"
echo "python3 verificar_ambiente.py --testar-cameras --cam-esq 0 --cam-dir 2"
echo
echo "Para iniciar:"
echo "python3 executar_projeto.py --modo executar --cam-esq 0 --cam-dir 2"

