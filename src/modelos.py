"""Definición de los modelos: baseline (E0) y backbones preentrenados (E1–E3)."""

import torch
from torch import nn
from torchvision import models

PREENTRENADOS = ("mobilenet_v2", "resnet18")


class CNNPequena(nn.Module):
    """Red convolucional sencilla, entrenada desde cero. Es el baseline E0."""

    def __init__(self, n_clases):
        super().__init__()

        def bloque(entrada, salida):
            return nn.Sequential(
                nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
                nn.BatchNorm2d(salida),
                nn.ReLU(),
                nn.MaxPool2d(2),  # reduce el tamaño a la mitad
            )

        self.extractor = nn.Sequential(bloque(3, 16), bloque(16, 32), bloque(32, 64))
        self.clasificador = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_clases)
        )

    def forward(self, x):
        return self.clasificador(self.extractor(x))


class Lineal(nn.Module):
    """Clasificador lineal sobre los píxeles: la opción más simple posible para E0."""

    def __init__(self, n_clases, tamano):
        super().__init__()
        self.capa = nn.Linear(3 * tamano * tamano, n_clases)

    def forward(self, x):
        return self.capa(x.flatten(1))


def crear_modelo(nombre, n_clases, preentrenado=True, tamano=64):
    """Construye el modelo y le pone una cabeza nueva de n_clases salidas."""
    if nombre == "cnn_pequena":
        return CNNPequena(n_clases)
    if nombre == "lineal":
        return Lineal(n_clases, tamano)
    if nombre == "mobilenet_v2":
        pesos = models.MobileNet_V2_Weights.IMAGENET1K_V1 if preentrenado else None
        modelo = models.mobilenet_v2(weights=pesos)
        entradas = modelo.classifier[1].in_features  # 1280 características
        modelo.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(entradas, n_clases))
        return modelo
    if nombre == "resnet18":
        pesos = models.ResNet18_Weights.IMAGENET1K_V1 if preentrenado else None
        modelo = models.resnet18(weights=pesos)
        entradas = modelo.fc.in_features  # 512 características
        modelo.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(entradas, n_clases))
        return modelo
    raise ValueError(f"Modelo desconocido: {nombre}")


def cabeza(modelo, nombre):
    """Devuelve la parte final del modelo (la que clasifica en 5 clases)."""
    return modelo.classifier if nombre == "mobilenet_v2" else modelo.fc


def bloque_final(modelo, nombre):
    """Último bloque del backbone: el que se descongela en el ajuste fino (E2)."""
    if nombre == "mobilenet_v2":
        return modelo.features[17:]  # último bloque residual invertido + convolución final
    return modelo.layer4


def capa_gradcam(modelo, nombre):
    """Última capa convolucional: la que usa Grad-CAM para el mapa de calor."""
    return modelo.features[-1] if nombre == "mobilenet_v2" else modelo.layer4[-1]


def congelar_backbone(modelo, nombre):
    """Extracción de características: solo la cabeza puede aprender."""
    for parametro in modelo.parameters():
        parametro.requires_grad = False
    for parametro in cabeza(modelo, nombre).parameters():
        parametro.requires_grad = True


def descongelar_bloque_final(modelo, nombre):
    """Ajuste fino parcial: además de la cabeza, aprende el último bloque."""
    for parametro in bloque_final(modelo, nombre).parameters():
        parametro.requires_grad = True


def poner_en_modo_entrenamiento(modelo, nombre):
    """Activa el modo entrenamiento solo donde corresponde.

    En los modelos preentrenados el backbone se queda en modo evaluación para que
    las capas BatchNorm conserven las estadísticas aprendidas en ImageNet (también
    en E2: allí cambian los pesos del bloque descongelado, no esas estadísticas).
    """
    if nombre in PREENTRENADOS:
        modelo.eval()
        cabeza(modelo, nombre).train()
    else:
        modelo.train()


def tamano_modelo(modelo):
    """Número de parámetros (totales y entrenables) y tamaño aproximado en MB."""
    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    bytes_ = sum(p.numel() * p.element_size() for p in modelo.parameters())
    bytes_ += sum(b.numel() * b.element_size() for b in modelo.buffers())
    return {
        "parametros_totales": total,
        "parametros_entrenables": entrenables,
        "tamano_mb": round(bytes_ / 1e6, 2),
    }


def cargar_pesos(modelo, ruta, dispositivo):
    estado = torch.load(ruta, map_location=dispositivo)
    modelo.load_state_dict(estado["modelo"])
    return estado
