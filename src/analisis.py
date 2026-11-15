"""Análisis final: tabla consolidada, comparaciones, errores reales y Grad-CAM."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import modelos
from src.comun import cargar_config, dispositivo
from src.dataset import DESVIACION, MEDIA, transformaciones
from src.entrenar import definir_experimento

RESULTADOS = Path("resultados")


def leer_metricas(experimento):
    with open(RESULTADOS / experimento / "metricas.json", encoding="utf-8") as f:
        return json.load(f)


def tabla_consolidada(experimentos=("E0", "E1", "E2", "E3")):
    """Una fila por experimento con desempeño y costo. Solo incluye los que ya se ejecutaron."""
    filas = []
    for exp in experimentos:
        if not (RESULTADOS / exp / "metricas.json").exists():
            continue
        m = leer_metricas(exp)
        lote = [k for k in m["inferencia"] if k != "ms_por_imagen_lote_1"][0]
        filas.append({
            "Exp.": exp, "Modelo": m["modelo"],
            "Accuracy": round(m["accuracy"], 4), "Macro-F1": round(m["macro_f1"], 4),
            "Val macro-F1": round(m["val_macro_f1_mejor"], 4), "Mejor época": m["mejor_epoca"],
            "Entren. (s)": m["tiempo_entrenamiento_s"], "Entren. acumulado (s)": m["tiempo_entrenamiento_acumulado_s"],
            "Inferencia ms/img (lote 1)": m["inferencia"]["ms_por_imagen_lote_1"],
            "Inferencia ms/img (en lote)": m["inferencia"][lote],
            "Parámetros": m["parametros_totales"], "Entrenables": m["parametros_entrenables"],
            "Tamaño (MB)": m["tamano_mb"], "Dispositivo": m["dispositivo"],
        })
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla.to_csv(RESULTADOS / "tabla_consolidada.csv", index=False)
    return tabla


def tabla_por_clase(experimentos=("E0", "E1", "E2", "E3"), metrica="f1"):
    """F1 (o recall) de cada clase en cada experimento."""
    datos = {}
    for exp in experimentos:
        if (RESULTADOS / exp / "metricas.json").exists():
            datos[exp] = {c: round(v[metrica], 4) for c, v in leer_metricas(exp)["por_clase"].items()}
    tabla = pd.DataFrame(datos)
    if not tabla.empty:
        tabla.to_csv(RESULTADOS / f"{metrica}_por_clase.csv")
    return tabla


def graficas_comparacion(experimentos=("E0", "E1", "E2", "E3")):
    """Barras de F1 por clase y gráfico de desempeño frente a costo."""
    f1 = tabla_por_clase(experimentos, "f1")
    consolidada = tabla_consolidada(experimentos)
    if f1.empty:
        return None
    eje = f1.plot(kind="bar", figsize=(8, 3.8), rot=0)
    eje.set_xlabel("Clase")
    eje.set_ylabel("F1 en prueba")
    eje.set_title("F1 por clase en cada experimento")
    eje.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS / "comparacion_f1_por_clase.png", dpi=150)
    plt.close()

    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for eje, columna, nombre in ((ejes[0], "Inferencia ms/img (lote 1)", "Inferencia (ms por imagen)"),
                                 (ejes[1], "Entren. acumulado (s)", "Entrenamiento acumulado (s)")):
        eje.scatter(consolidada[columna], consolidada["Macro-F1"], s=60)
        for _, fila in consolidada.iterrows():
            eje.annotate(fila["Exp."], (fila[columna], fila["Macro-F1"]), textcoords="offset points", xytext=(6, 4))
        eje.set_xlabel(nombre)
        eje.set_ylabel("Macro-F1 en prueba")
        eje.grid(alpha=0.3)
    fig.suptitle("Desempeño frente a costo")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "comparacion_costo.png", dpi=150)
    plt.close(fig)
    return consolidada


def estabilidad(experimentos=("E1", "E3")):
    """Media y variación de macro-F1 entre ejecuciones con distintas semillas (E1, E1_s1, E1_s2...)."""
    filas = []
    for exp in experimentos:
        for carpeta in sorted(RESULTADOS.glob(f"{exp}*")):
            if carpeta.name == exp or carpeta.name.startswith(f"{exp}_s"):
                if (carpeta / "metricas.json").exists():
                    m = leer_metricas(carpeta.name)
                    filas.append({"experimento": exp, "ejecucion": carpeta.name,
                                  "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
    detalle = pd.DataFrame(filas)
    if detalle.empty:
        return detalle, detalle
    resumen = detalle.groupby("experimento")[["macro_f1", "accuracy"]].agg(["mean", "std", "min", "max", "count"]).round(4)
    detalle.to_csv(RESULTADOS / "estabilidad_detalle.csv", index=False)
    return detalle, resumen


def errores(experimento, n=8):
    """Los n errores en los que el modelo estaba MÁS seguro de su respuesta equivocada.

    Criterio declarado: se ordenan por confianza para no escoger a conveniencia.
    """
    pred = pd.read_csv(RESULTADOS / experimento / "predicciones_test.csv")
    return pred[~pred["acierto"]].sort_values("confianza", ascending=False).head(n).reset_index(drop=True)


def galeria_errores(experimento, n=8, cfg=None):
    """Dibuja los errores con su clase real, la predicha y la confianza."""
    cfg = cfg or cargar_config()
    tabla = errores(experimento, n)
    if tabla.empty:
        return tabla
    columnas = 4
    filas = int(np.ceil(len(tabla) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(3.2 * columnas, 3.5 * filas), squeeze=False)
    for eje in ejes.flat:
        eje.axis("off")
    for i, fila in tabla.iterrows():
        eje = ejes.flat[i]
        eje.imshow(Image.open(Path(cfg["datos"]["raiz"]) / fila["ruta"]).convert("RGB"))
        eje.set_title(f"#{i + 1} real: {fila['real']}\npredicha: {fila['predicha']} ({fila['confianza']:.0%})", fontsize=9)
    fig.suptitle(f"{experimento}: errores reales en prueba (los de mayor confianza)")
    fig.tight_layout()
    fig.savefig(RESULTADOS / experimento / "galeria_errores.png", dpi=110)
    plt.close(fig)
    tabla.to_csv(RESULTADOS / experimento / "errores_inspeccionados.csv", index=False)
    return tabla


def pares_confundidos(experimento, cfg=None, n=6):
    """Las confusiones más frecuentes: 'era X y el modelo dijo Y'."""
    cfg = cfg or cargar_config()
    clases = cfg["datos"]["clases"]
    matriz = np.array(leer_metricas(experimento)["matriz_confusion"])
    filas = [{"real": clases[i], "predicha": clases[j], "veces": int(matriz[i, j]),
              "% de la clase real": round(100 * matriz[i, j] / max(matriz[i].sum(), 1), 1)}
             for i in range(len(clases)) for j in range(len(clases)) if i != j and matriz[i, j] > 0]
    return pd.DataFrame(filas).sort_values("veces", ascending=False).head(n).reset_index(drop=True)


# ------------------------------------------------------------------ Grad-CAM

def _gradcam(modelo, capa, entrada, clase):
    """Mapa de calor: qué zonas de la última capa convolucional empujaron hacia 'clase'."""
    guardado = {}
    g1 = capa.register_forward_hook(lambda m, e, salida: guardado.update(activaciones=salida))
    g2 = capa.register_full_backward_hook(lambda m, ge, gs: guardado.update(gradientes=gs[0]))
    try:
        modelo.zero_grad()
        salida = modelo(entrada)
        salida[0, clase].backward()
    finally:
        g1.remove()
        g2.remove()
    pesos = guardado["gradientes"].mean(dim=(2, 3), keepdim=True)   # importancia de cada canal
    mapa = torch.relu((pesos * guardado["activaciones"]).sum(dim=1))[0]
    mapa = mapa / (mapa.max() + 1e-8)
    return mapa.detach().cpu().numpy()


def gradcam_galeria(experimento, n=4, solo_errores=False, cfg=None, prefijo=""):
    """Grad-CAM para n imágenes de prueba (mitad aciertos y mitad errores, o solo errores)."""
    cfg = cfg or cargar_config()
    clases = cfg["datos"]["clases"]
    definicion = definir_experimento(experimento, cfg)
    if definicion["modelo"] not in modelos.PREENTRENADOS:
        raise ValueError("Grad-CAM está preparado para E1, E2 y E3")
    disp = dispositivo()
    modelo = modelos.crear_modelo(definicion["modelo"], len(clases), preentrenado=False).to(disp)
    modelos.cargar_pesos(modelo, Path("checkpoints") / f"{prefijo}{experimento}.pt", disp)
    modelo.eval()
    for parametro in modelo.parameters():
        parametro.requires_grad = True  # Grad-CAM necesita gradientes aunque no se entrene

    pred = pd.read_csv(RESULTADOS / f"{prefijo}{experimento}" / "predicciones_test.csv")
    malas = pred[~pred["acierto"]].sort_values("confianza", ascending=False)
    buenas = pred[pred["acierto"]].sort_values("confianza", ascending=False)
    elegidas = malas.head(n) if solo_errores else pd.concat([buenas.head(n - n // 2), malas.head(n // 2)])
    elegidas = elegidas.reset_index(drop=True)

    tam = definicion["tamano"]
    transformacion = transformaciones(tam)
    capa = modelos.capa_gradcam(modelo, definicion["modelo"])
    media, desv = np.array(MEDIA), np.array(DESVIACION)
    fig, ejes = plt.subplots(2, len(elegidas), figsize=(3.2 * len(elegidas), 6.6), squeeze=False)
    for i, fila in elegidas.iterrows():
        imagen = Image.open(Path(cfg["datos"]["raiz"]) / fila["ruta"]).convert("RGB")
        entrada = transformacion(imagen).unsqueeze(0).to(disp)
        mapa = _gradcam(modelo, capa, entrada, clases.index(fila["predicha"]))
        visible = np.clip(entrada[0].detach().cpu().numpy().transpose(1, 2, 0) * desv + media, 0, 1)
        mapa_grande = np.asarray(Image.fromarray(np.uint8(mapa * 255)).resize((tam, tam), Image.BILINEAR)) / 255
        ejes[0, i].imshow(visible)
        ejes[0, i].set_title(f"real: {fila['real']}\npredicha: {fila['predicha']} ({fila['confianza']:.0%})", fontsize=9)
        ejes[1, i].imshow(visible)
        ejes[1, i].imshow(mapa_grande, cmap="jet", alpha=0.45)
        ejes[1, i].set_title("Grad-CAM de la clase predicha", fontsize=9)
        ejes[0, i].axis("off")
        ejes[1, i].axis("off")
    fig.suptitle(f"{experimento}: Grad-CAM (indica dónde miró el modelo, no demuestra causalidad)")
    fig.tight_layout()
    ruta = RESULTADOS / f"{prefijo}{experimento}" / "gradcam.png"
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    return ruta


def inferencia_una_imagen(experimento, ruta_imagen, cfg=None, prefijo=""):
    """Carga un modelo entrenado y clasifica UNA foto. Útil para la demostración en vivo."""
    cfg = cfg or cargar_config()
    clases = cfg["datos"]["clases"]
    definicion = definir_experimento(experimento, cfg)
    disp = dispositivo()
    modelo = modelos.crear_modelo(definicion["modelo"], len(clases), preentrenado=False,
                                  tamano=definicion["tamano"]).to(disp)
    modelos.cargar_pesos(modelo, Path("checkpoints") / f"{prefijo}{experimento}.pt", disp)
    modelo.eval()
    recorte = definicion["modelo"] in modelos.PREENTRENADOS
    entrada = transformaciones(definicion["tamano"], recorte_central=recorte)(Image.open(ruta_imagen).convert("RGB"))
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(entrada.unsqueeze(0).to(disp)), dim=1)[0].cpu().numpy()
    return pd.Series(probabilidades, index=clases).round(4).sort_values(ascending=False)
