"""Descarga e inventario del conjunto de datos Cassava (iCassava 2019).

Este módulo hace tres cosas:
  1. Descarga el archivo .zip oficial (el mismo que usa TensorFlow Datasets).
  2. Lo descomprime en una carpeta local.
  3. Recorre todas las imágenes y construye un "inventario": una tabla con una
     fila por imagen (partición, clase, nombre, tamaño, resolución y huella MD5).

Uso desde la terminal (o desde una celda de Colab con "!" al inicio):
    python -m src.datos --destino data

Uso desde un notebook:
    from src.datos import descargar_dataset, construir_inventario
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

# URL tomada del código fuente de TensorFlow Datasets (cassava.py, versión 0.1.0).
URL_DATASET = "https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip"

# Orden de etiquetas idéntico al de TensorFlow Datasets (0=cbb ... 4=healthy).
CLASES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]
NOMBRES_CLASES = {
    "cbb": "Tizón bacteriano (Cassava Bacterial Blight)",
    "cbsd": "Estriado marrón (Cassava Brown Streak Disease)",
    "cgm": "Ácaro verde (Cassava Green Mite)",
    "cmd": "Mosaico (Cassava Mosaic Disease)",
    "healthy": "Hoja sana",
}
PARTICIONES = ["train", "validation", "test"]

# Cantidades publicadas por TensorFlow Datasets y por el documento del profesor.
CONTEOS_ESPERADOS = {"train": 5656, "validation": 1889, "test": 1885}

EXTENSIONES = {".jpg", ".jpeg", ".png"}


def descargar_dataset(destino="data", url=URL_DATASET, carpeta_zip=None):
    """Descarga y descomprime el dataset. Si ya existe, no lo vuelve a bajar.

    destino:      carpeta donde quedan las imágenes descomprimidas.
    carpeta_zip:  carpeta donde se guarda el .zip (por defecto, la misma). En Colab se
                  guarda el .zip en Drive y se descomprime en el disco local, que es más rápido.
    Devuelve la ruta de la carpeta que contiene train/, validation/ y test/.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    carpeta_zip = Path(carpeta_zip) if carpeta_zip else destino
    carpeta_zip.mkdir(parents=True, exist_ok=True)
    raiz = destino / "cassavaleafdata"
    archivo_zip = carpeta_zip / "cassavaleafdata.zip"

    if all((raiz / p).is_dir() for p in PARTICIONES):
        print(f"[ok] El dataset ya está descomprimido en: {raiz}")
        return raiz

    if not archivo_zip.exists():
        print(f"[1/2] Descargando {url}")
        print("      (pesa alrededor de 1,3 GB; puede tardar varios minutos)")
        temporal = archivo_zip.with_suffix(".parcial")
        with urllib.request.urlopen(url) as respuesta, open(temporal, "wb") as salida:
            total = int(respuesta.headers.get("Content-Length") or 0)
            leido, siguiente_aviso = 0, 0
            while True:
                bloque = respuesta.read(1024 * 1024)
                if not bloque:
                    break
                salida.write(bloque)
                leido += len(bloque)
                if total and leido >= siguiente_aviso:
                    print(f"      {leido / 1e6:7.0f} MB de {total / 1e6:.0f} MB")
                    siguiente_aviso += total / 10
        # Solo se renombra al terminar: así una descarga cortada no parece completa.
        shutil.move(temporal, archivo_zip)
    else:
        print(f"[1/2] Ya existe el archivo {archivo_zip}; no se descarga de nuevo.")

    print("[2/2] Descomprimiendo...")
    with zipfile.ZipFile(archivo_zip) as z:
        z.extractall(destino)

    if not all((raiz / p).is_dir() for p in PARTICIONES):
        raise RuntimeError(
            f"Después de descomprimir no se encontraron las carpetas {PARTICIONES} "
            f"dentro de {raiz}. Revise el contenido de {destino}."
        )
    print(f"[ok] Dataset listo en: {raiz}")
    return raiz


def _md5(ruta):
    """Huella digital del archivo: dos archivos idénticos byte a byte dan el mismo MD5."""
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def construir_inventario(raiz):
    """Recorre train/validation/test y devuelve una tabla con una fila por imagen."""
    raiz = Path(raiz)
    filas, ilegibles = [], []

    otras = sorted(p.name for p in raiz.iterdir() if p.is_dir() and p.name not in PARTICIONES)
    if otras:
        print(f"[aviso] Carpetas adicionales que NO se incluyen en el inventario: {otras}")

    for particion in PARTICIONES:
        for carpeta_clase in sorted((raiz / particion).iterdir()):
            if not carpeta_clase.is_dir():
                continue
            if carpeta_clase.name not in CLASES:
                print(f"[aviso] Carpeta de clase inesperada: {carpeta_clase}")
            for ruta in sorted(carpeta_clase.iterdir()):
                if ruta.suffix.lower() not in EXTENSIONES:
                    continue
                try:
                    with Image.open(ruta) as img:
                        ancho, alto = img.size
                        modo = img.mode
                        img.verify()  # detecta archivos dañados
                except Exception as error:  # imagen corrupta o ilegible
                    ilegibles.append((str(ruta), repr(error)))
                    continue
                filas.append(
                    {
                        "particion": particion,
                        "clase": carpeta_clase.name,
                        "archivo": ruta.name,
                        "ruta": ruta.relative_to(raiz).as_posix(),
                        "ancho": ancho,
                        "alto": alto,
                        "modo": modo,
                        "bytes": ruta.stat().st_size,
                        "md5": _md5(ruta),
                    }
                )

    if ilegibles:
        print(f"[aviso] {len(ilegibles)} archivo(s) no se pudieron leer:")
        for ruta, error in ilegibles:
            print(f"        {ruta} -> {error}")
    return pd.DataFrame(filas)


def tabla_distribucion(inventario):
    """Tabla clase x partición con totales (cuántas imágenes hay de cada tipo)."""
    tabla = pd.crosstab(inventario["clase"], inventario["particion"])
    tabla = tabla.reindex(index=CLASES, columns=PARTICIONES, fill_value=0)
    tabla["total"] = tabla.sum(axis=1)
    tabla.loc["TOTAL"] = tabla.sum(axis=0)
    return tabla


def tabla_porcentajes(inventario):
    """Porcentaje de cada clase dentro de cada partición (para ver el desbalance)."""
    tabla = pd.crosstab(inventario["clase"], inventario["particion"], normalize="columns") * 100
    return tabla.reindex(index=CLASES, columns=PARTICIONES, fill_value=0).round(1)


def verificar_conteos(inventario, esperados=CONTEOS_ESPERADOS):
    """Compara lo descargado con las cantidades publicadas. Devuelve True si coinciden."""
    reales = inventario["particion"].value_counts().to_dict()
    todo_bien = True
    for particion, esperado in esperados.items():
        real = reales.get(particion, 0)
        estado = "OK" if real == esperado else "DIFERENTE"
        todo_bien = todo_bien and real == esperado
        print(f"  {particion:<11} esperado={esperado:<5} encontrado={real:<5} {estado}")
    return todo_bien


def main():
    parser = argparse.ArgumentParser(description="Descarga e inventario del dataset Cassava")
    parser.add_argument("--destino", default="data", help="carpeta donde se guardan los datos")
    parser.add_argument("--url", default=URL_DATASET)
    parser.add_argument("--salida", default="resultados/inventario.csv")
    args = parser.parse_args()

    raiz = descargar_dataset(args.destino, args.url)
    inventario = construir_inventario(raiz)

    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    inventario.to_csv(args.salida, index=False)

    print("\nImágenes por clase y partición:")
    print(tabla_distribucion(inventario).to_string())
    print("\nPorcentaje de cada clase dentro de cada partición:")
    print(tabla_porcentajes(inventario).to_string())
    print("\nComparación con las cantidades publicadas:")
    coincide = verificar_conteos(inventario)
    print(f"\nInventario guardado en: {args.salida} ({len(inventario)} filas)")
    return 0 if coincide else 1


if __name__ == "__main__":
    sys.exit(main())
