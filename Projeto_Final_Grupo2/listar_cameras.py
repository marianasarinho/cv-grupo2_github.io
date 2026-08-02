#!/usr/bin/env python3
import sys
from executar_projeto import main

main(["--modo", "listar", *sys.argv[1:]])

