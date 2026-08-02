@echo off
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python 3.12 nao foi encontrado.
    echo Instale-o com: py install 3.12
    pause
    exit /b 1
)

if not exist .venv312 (
    py -3.12 -m venv .venv312
)

call .venv312\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
python verificar_ambiente.py

echo.
echo Para testar as cameras:
echo python verificar_ambiente.py --testar-cameras --cam-esq 1 --cam-dir 0
echo.
echo Para executar o sistema validado no Windows:
echo python executar_projeto.py --modo executar --cam-esq 1 --cam-dir 0
pause
