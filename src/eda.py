"""Análisis exploratorio (EDA) y control de duplicados."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.datos import CLASES, PARTICIONES

CARPETA = Path("resultados/eda")


def grafica_distribucion(inventario):
    """Barras con el número de imágenes por clase en cada partición."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    tabla = pd.crosstab(inventario["clase"], inventario["particion"]).reindex(index=CLASES, columns=PARTICIONES, fill_value=0)
    eje = tabla.plot(kind="bar", figsize=(7, 3.8), rot=0)
    eje.set_xlabel("Clase")
    eje.set_ylabel("Número de imágenes")
    eje.set_title("Distribución de clases por partición")
    eje.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CARPETA / "distribucion_clases.png", dpi=150)
    plt.close()
    return tabla


def resumen_resoluciones(inventario):
    """Estadísticas de ancho, alto y peso de los archivos."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    resumen = inventario[["ancho", "alto", "bytes"]].describe().round(1)
    resumen.to_csv(CARPETA / "resumen_resoluciones.csv")
    return resumen


def mosaico_ejemplos(inventario, raiz, por_clase=5, semilla=0):
    """Una fila de fotos de ejemplo por clase (entrenamiento), elegidas al azar con semilla fija."""
    CARPETA.mkdir(parents=True, exist_ok=True)
    train = inventario[inventario["particion"] == "train"]
    fig, ejes = plt.subplots(len(CLASES), por_clase, figsize=(2.2 * por_clase, 2.2 * len(CLASES)))
    for f, clase in enumerate(CLASES):
        muestra = train[train["clase"] == clase]
        muestra = muestra.sample(n=min(por_clase, len(muestra)), random_state=semilla)
        for c in range(por_clase):
            eje = ejes[f, c]
            eje.axis("off")
            if c < len(muestra):
                eje.imshow(Image.open(Path(raiz) / muestra.iloc[c]["ruta"]).convert("RGB"))
            if c == 0:
                eje.set_title(clase, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(CARPETA / "ejemplos_por_clase.png", dpi=110)
    plt.close(fig)
    return CARPETA / "ejemplos_por_clase.png"


def brillo_medio(inventario, raiz, por_clase=60, semilla=0):
    """Brillo promedio (0–255) de una muestra por clase: ayuda a pensar en atajos visuales."""
    filas = []
    train = inventario[inventario["particion"] == "train"]
    for clase in CLASES:
        muestra = train[train["clase"] == clase]
        muestra = muestra.sample(n=min(por_clase, len(muestra)), random_state=semilla)
        for ruta in muestra["ruta"]:
            gris = np.asarray(Image.open(Path(raiz) / ruta).convert("L").resize((64, 64)))
            filas.append({"clase": clase, "brillo": gris.mean()})
    tabla = pd.DataFrame(filas).groupby("clase")["brillo"].agg(["mean", "std", "count"]).reindex(CLASES).round(1)
    CARPETA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARPETA / "brillo_por_clase.csv")
    return tabla


# ------------------------------------------------------------------ duplicados

def duplicados_exactos(inventario):
    """Archivos idénticos byte a byte (misma huella MD5)."""
    repetidos = inventario[inventario.duplicated("md5", keep=False)].sort_values(["md5", "ruta"]).copy()
    if repetidos.empty:
        repetidos["entre_particiones"] = []
        return repetidos
    cruza = repetidos.groupby("md5")["particion"].nunique() > 1
    repetidos["entre_particiones"] = repetidos["md5"].map(cruza)
    return repetidos[["md5", "particion", "clase", "ruta", "entre_particiones"]]


def _huella_visual(ruta):
    """dHash: resume la imagen en 64 bits. Fotos casi iguales dan huellas casi iguales."""
    gris = np.asarray(Image.open(ruta).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (gris[:, 1:] > gris[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]


def casi_duplicados(inventario, raiz, distancia_max=2):
    """Pares de imágenes visualmente casi iguales (aunque el archivo no sea idéntico).

    Compara las huellas visuales de todas las imágenes; 'distancia_max' es cuántos de
    los 64 bits pueden diferir. Son CANDIDATOS: hay que mirarlos antes de decidir.
    """
    inv = inventario.reset_index(drop=True)
    huellas = np.array([_huella_visual(Path(raiz) / r) for r in inv["ruta"]], dtype=np.uint64)
    tabla_bits = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pares = []
    for inicio in range(0, len(huellas), 256):
        bloque = huellas[inicio:inicio + 256]
        distinto = bloque[:, None] ^ huellas[None, :]
        distancia = tabla_bits[distinto.view(np.uint8).reshape(len(bloque), len(huellas), 8)].sum(axis=2)
        for i, j in zip(*np.where(distancia <= distancia_max)):
            a, b = inicio + i, j
            if a < b and inv.at[a, "md5"] != inv.at[b, "md5"]:
                pares.append({
                    "ruta_a": inv.at[a, "ruta"], "ruta_b": inv.at[b, "ruta"],
                    "particion_a": inv.at[a, "particion"], "particion_b": inv.at[b, "particion"],
                    "clase_a": inv.at[a, "clase"], "clase_b": inv.at[b, "clase"],
                    "distancia": int(distancia[i, j]),
                })
    pares = pd.DataFrame(pares, columns=["ruta_a", "ruta_b", "particion_a", "particion_b", "clase_a", "clase_b", "distancia"])
    pares["entre_particiones"] = pares["particion_a"] != pares["particion_b"]
    return pares


def mostrar_pares(pares, raiz, n=6, ruta_salida=None):
    """Dibuja algunos pares candidatos para revisarlos a ojo."""
    pares = pares.head(n)
    if pares.empty:
        return None
    fig, ejes = plt.subplots(len(pares), 2, figsize=(6, 3 * len(pares)), squeeze=False)
    for f, (_, par) in enumerate(pares.iterrows()):
        for c, lado in enumerate(("a", "b")):
            ejes[f, c].imshow(Image.open(Path(raiz) / par[f"ruta_{lado}"]).convert("RGB"))
            ejes[f, c].set_title(par[f"ruta_{lado}"], fontsize=8)
            ejes[f, c].axis("off")
    fig.tight_layout()
    ruta_salida = ruta_salida or CARPETA / "pares_casi_duplicados.png"
    fig.savefig(ruta_salida, dpi=100)
    plt.close(fig)
    return ruta_salida


def generar_exclusiones(inventario, pares_casi=None, incluir_casi=False, retirar_conflictivas=False):
    """Lista de imágenes a retirar para evitar fugas. El conjunto de PRUEBA nunca se modifica.

    Regla: prioridad test > validation > train. Si una imagen de train (o validation) es
    idéntica a una de una partición con más prioridad, se retira la de menor prioridad.
    Dentro de una misma partición de train/validation se conserva una sola copia.
    Con incluir_casi=True se aplica lo mismo a los pares casi duplicados entre particiones.
    Con retirar_conflictivas=True, si una misma foto (mismo MD5) aparece con etiquetas
    distintas, se retiran TODAS sus copias de entrenamiento y validación (no se conserva ninguna).
    """
    prioridad = {"test": 0, "validation": 1, "train": 2}
    excluidos = {}
    for _, grupo in inventario[inventario.duplicated("md5", keep=False)].groupby("md5"):
        grupo = grupo.assign(p=grupo["particion"].map(prioridad)).sort_values(["p", "ruta"])
        if retirar_conflictivas and grupo["clase"].nunique() > 1:
            etiquetas = "/".join(sorted(grupo["clase"].unique()))
            for _, fila in grupo.iterrows():
                if fila["particion"] != "test":
                    excluidos[fila["ruta"]] = f"misma foto con etiquetas distintas ({etiquetas})"
            continue
        mejor = grupo.iloc[0]
        for _, fila in grupo.iloc[1:].iterrows():
            if fila["particion"] == "test":
                continue  # prueba no se toca
            excluidos[fila["ruta"]] = f"idéntica a {mejor['ruta']}"
    if incluir_casi and pares_casi is not None:
        for _, par in pares_casi[pares_casi["entre_particiones"]].iterrows():
            a, b = (par["ruta_a"], par["particion_a"]), (par["ruta_b"], par["particion_b"])
            peor, otro = (a, b) if prioridad[a[1]] > prioridad[b[1]] else (b, a)
            if peor[1] != "test":
                excluidos.setdefault(peor[0], f"casi idéntica a {otro[0]}")
    tabla = pd.DataFrame(sorted(excluidos.items()), columns=["ruta", "motivo"])
    return tabla
