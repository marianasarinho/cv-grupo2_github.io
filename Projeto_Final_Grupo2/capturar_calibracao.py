#!/usr/bin/env python3
import sys
from executar_projeto import main

main(["--modo", "capturar", *sys.argv[1:]])

