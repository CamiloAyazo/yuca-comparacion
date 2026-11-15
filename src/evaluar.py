"""Evaluación: predicciones, métricas, tiempo de inferencia y gráficas."""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # permite guardar gráficas sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn


@torch.no_grad()
def predecir(modelo, cargador, dispositivo, perdida=None):
    """Pasa todas las imágenes por el modelo. Devuelve etiquetas reales, probabilidades y pérdida."""
    modelo.eval()
    perdida = perdida or nn.CrossEntropyLoss()
    reales, probabilidades, suma_perdida, n = [], [], 0.0, 0
    for imagenes, etiquetas, _ in cargador:
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        salidas = modelo(imagenes)
        suma_perdida += perdida(salidas, etiquetas).item() * len(etiquetas)
        n += len(etiquetas)
        reales.append(etiquetas.cpu().numpy())
        probabilidades.append(torch.softmax(salidas, dim=1).cpu().numpy())
    return np.concatenate(reales), np.concatenate(probabilidades), suma_perdida / max(n, 1)


def calcular_metricas(reales, predichas, clases):
    """Accuracy, macro-F1 y precisión/recall/F1 de cada clase, más la matriz de confusión."""
    etiquetas = list(range(len(clases)))
    precision, recall, f1, soporte = precision_recall_fscore_support(
        reales, predichas, labels=etiquetas, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(reales, predichas)),
        "macro_f1": float(f1_score(reales, predichas, labels=etiquetas, average="macro", zero_division=0)),
        "por_clase": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "imagenes": int(soporte[i]),
            }
            for i, c in enumerate(clases)
        },
        "matriz_confusion": confusion_matrix(reales, predichas, labels=etiquetas).tolist(),
    }


def macro_f1_manual(matriz):
    """Macro-F1 calculado a mano desde la matriz de confusión (para explicarlo en la sustentación).

    Filas = clase real, columnas = clase predicha.
    """
    matriz = np.asarray(matriz, dtype=float)
    f1_clases = []
    for i in range(len(matriz)):
        verdaderos_positivos = matriz[i, i]
        predichos_como_i = matriz[:, i].sum()      # todo lo que el modelo llamó "clase i"
        reales_de_i = matriz[i, :].sum()            # todo lo que realmente era "clase i"
        precision = verdaderos_positivos / predichos_como_i if predichos_como_i else 0.0
        recall = verdaderos_positivos / reales_de_i if reales_de_i else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_clases.append(f1)
    return float(np.mean(f1_clases))  # promedio simple: cada clase pesa lo mismo


@torch.no_grad()
def medir_inferencia(modelo, tamano, dispositivo, lote=32, repeticiones=20):
    """Tiempo medio que tarda el modelo en procesar una imagen (ms), sin contar la lectura del disco."""
    modelo.eval()
    resultados = {}
    for nombre, n in (("ms_por_imagen_lote_1", 1), (f"ms_por_imagen_lote_{lote}", lote)):
        entrada = torch.randn(n, 3, tamano, tamano, device=dispositivo)
        for _ in range(3):  # calentamiento: las primeras pasadas siempre son más lentas
            modelo(entrada)
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        inicio = time.perf_counter()
        for _ in range(repeticiones):
            modelo(entrada)
        if dispositivo.type == "cuda":
            torch.cuda.synchronize()
        resultados[nombre] = round((time.perf_counter() - inicio) / (repeticiones * n) * 1000, 3)
    return resultados


def graficar_matriz(matriz, clases, titulo, ruta):
    matriz = np.asarray(matriz)
    fig, eje = plt.subplots(figsize=(5.5, 4.8))
    imagen = eje.imshow(matriz, cmap="Blues")
    eje.set_xticks(range(len(clases)), clases)
    eje.set_yticks(range(len(clases)), clases)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title(titulo)
    umbral = matriz.max() / 2 if matriz.max() else 0
    for i in range(len(clases)):
        for j in range(len(clases)):
            eje.text(j, i, int(matriz[i, j]), ha="center", va="center",
                     color="white" if matriz[i, j] > umbral else "black")
    fig.colorbar(imagen, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_curvas(historial, titulo, ruta):
    """Curvas de pérdida y macro-F1 en entrenamiento y validación, época por época."""
    historial = pd.DataFrame(historial)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for eje, columna, nombre in ((ejes[0], "perdida", "Pérdida"), (ejes[1], "macro_f1", "Macro-F1")):
        eje.plot(historial["epoca"], historial[f"train_{columna}"], marker="o", label="Entrenamiento")
        eje.plot(historial["epoca"], historial[f"val_{columna}"], marker="s", label="Validación")
        eje.set_xlabel("Época")
        eje.set_ylabel(nombre)
        eje.grid(alpha=0.3)
        eje.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_predicciones(tabla, reales, probabilidades, clases, ruta):
    """Guarda una fila por imagen de prueba: qué era, qué dijo el modelo y con qué confianza."""
    predichas = probabilidades.argmax(1)
    salida = pd.DataFrame(
        {
            "ruta": tabla["ruta"].values,
            "real": [clases[i] for i in reales],
            "predicha": [clases[i] for i in predichas],
            "confianza": probabilidades.max(1).round(4),
            "acierto": reales == predichas,
        }
    )
    for i, c in enumerate(clases):
        salida[f"p_{c}"] = probabilidades[:, i].round(4)
    salida.to_csv(ruta, index=False)
    return salida


def guardar_json(datos, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
