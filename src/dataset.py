"""Carga de imágenes: particiones, transformaciones y lotes (batches)."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Media y desviación de ImageNet. MobileNetV2 y ResNet18 de torchvision (pesos
# IMAGENET1K_V1) fueron preentrenadas con esta misma normalización.
MEDIA = [0.485, 0.456, 0.406]
DESVIACION = [0.229, 0.224, 0.225]

RUTA_INVENTARIO = "resultados/inventario.csv"
RUTA_EXCLUIDOS = "resultados/excluidos.csv"


class Brillo:
    """Multiplica el brillo de la imagen. factor=1 la deja igual; <1 oscurece; >1 aclara."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, imagen):
        return ImageEnhance.Brightness(imagen).enhance(self.factor)


def transformaciones(tamano, aumento=False, brillo=None, recorte_central=True):
    """Pasos que se aplican a cada foto antes de entregarla al modelo.

    aumento=True solo se usa en ENTRENAMIENTO del experimento adicional.
    brillo=factor solo se usa sobre una copia del conjunto de PRUEBA.
    """
    pasos = []
    if brillo is not None:
        pasos.append(Brillo(brillo))
    if aumento:
        # Aumento moderado: no cambia el tono (el color es parte del síntoma).
        pasos += [
            transforms.RandomResizedCrop(tamano, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    elif recorte_central:
        pasos += [transforms.Resize(int(tamano * 256 / 224)), transforms.CenterCrop(tamano)]
    else:
        pasos += [transforms.Resize((tamano, tamano))]
    pasos += [transforms.ToTensor(), transforms.Normalize(MEDIA, DESVIACION)]
    return transforms.Compose(pasos)


class HojasYuca(Dataset):
    """Entrega (imagen, etiqueta, posición) a partir de las filas del inventario."""

    def __init__(self, tabla, raiz, clases, transformacion):
        self.tabla = tabla.reset_index(drop=True)
        self.raiz = Path(raiz)
        self.indice_clase = {c: i for i, c in enumerate(clases)}
        self.transformacion = transformacion

    def __len__(self):
        return len(self.tabla)

    def __getitem__(self, i):
        fila = self.tabla.iloc[i]
        imagen = Image.open(self.raiz / fila["ruta"]).convert("RGB")
        return self.transformacion(imagen), self.indice_clase[fila["clase"]], i


def cargar_particiones(limite=None, semilla=0):
    """Lee el inventario y devuelve {'train': tabla, 'validation': tabla, 'test': tabla}.

    Si existe resultados/excluidos.csv (control de duplicados), esas imágenes se quitan.
    'limite' sirve solo para pruebas rápidas del código: toma pocas imágenes por partición.
    """
    inventario = pd.read_csv(RUTA_INVENTARIO)
    if Path(RUTA_EXCLUIDOS).exists():
        excluidos = set(pd.read_csv(RUTA_EXCLUIDOS)["ruta"])
        inventario = inventario[~inventario["ruta"].isin(excluidos)]
    partes = {}
    for nombre in ["train", "validation", "test"]:
        tabla = inventario[inventario["particion"] == nombre].sort_values("ruta")
        if limite:
            tabla = tabla.sample(n=min(limite, len(tabla)), random_state=semilla).sort_values("ruta")
        partes[nombre] = tabla.reset_index(drop=True)
    return partes


def crear_cargador(tabla, raiz, clases, transformacion, lote, barajar, semilla=0, trabajadores=2):
    """Agrupa las imágenes en lotes. Solo se barajan las de entrenamiento."""
    generador = torch.Generator().manual_seed(semilla)
    return DataLoader(
        HojasYuca(tabla, raiz, clases, transformacion),
        batch_size=lote,
        shuffle=barajar,
        num_workers=trabajadores,
        generator=generador if barajar else None,
        pin_memory=torch.cuda.is_available(),
    )
