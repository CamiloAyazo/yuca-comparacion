"""Utilidades compartidas: configuración, semilla, dispositivo y registro (log)."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def cargar_config(ruta="configs/config.yaml"):
    """Lee el archivo de configuración y devuelve un diccionario."""
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Permite indicar dónde están los datos sin editar el archivo (útil en Colab).
    cfg["datos"]["raiz"] = os.environ.get("YUCA_DATOS", cfg["datos"]["raiz"])
    return cfg


def fijar_semilla(semilla):
    """Hace que los números 'aleatorios' sean los mismos en cada ejecución."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dispositivo():
    """Devuelve la GPU si existe; si no, la CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nombre_dispositivo():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


class Registro:
    """Escribe cada mensaje en pantalla y también en un archivo log.txt."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, mensaje):
        linea = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mensaje}"
        print(linea, flush=True)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(linea + "\n")
