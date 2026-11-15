"""Entrenamiento de los experimentos E0, E1, E2, E3 y la variante con augmentation.

Uso:
    python -m src.entrenar --exp E0
    python -m src.entrenar --exp E1
    python -m src.entrenar --exp E2      (necesita que E1 ya esté ejecutado)
    python -m src.entrenar --exp E3
    python -m src.entrenar --exp E1_aug  (experimento adicional: augmentation solo en entrenamiento)

Cada ejecución guarda en resultados/<nombre>/:
    config.json, log.txt, historial.csv, curvas.png, metricas.json,
    matriz_confusion.png y predicciones_test.csv
y el mejor modelo en checkpoints/<nombre>.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from src import evaluar, modelos
from src.comun import Registro, cargar_config, dispositivo, fijar_semilla, nombre_dispositivo
from src.dataset import cargar_particiones, crear_cargador, transformaciones


def definir_experimento(exp, cfg):
    """Traduce el nombre del experimento a sus condiciones concretas."""
    p, ext, ft, e0 = cfg["protocolo"], cfg["extraccion"], cfg["ajuste_fino"], cfg["e0"]
    base = {"tamano": p["tamano_imagen"], "aumento": False, "parte_de": None, "ajuste_fino": False}
    if exp == "E0":
        return {**base, "modelo": e0["tipo"], "tamano": e0["tamano_imagen"], "epocas": e0["epocas"],
                "tasa": e0["tasa_aprendizaje"], "descripcion": "Baseline sin preentrenamiento"}
    if exp in ("E1", "E1_aug"):
        return {**base, "modelo": ext["backbone_a"], "epocas": ext["epocas"], "tasa": ext["tasa_aprendizaje"],
                "aumento": exp.endswith("_aug"),
                "descripcion": "Backbone A congelado, solo cabeza" + (" + augmentation" if exp.endswith("_aug") else "")}
    if exp in ("E3", "E3_aug"):
        return {**base, "modelo": ext["backbone_b"], "epocas": ext["epocas"], "tasa": ext["tasa_aprendizaje"],
                "aumento": exp.endswith("_aug"),
                "descripcion": "Backbone B congelado, solo cabeza" + (" + augmentation" if exp.endswith("_aug") else "")}
    if exp == "E2":
        return {**base, "modelo": ext["backbone_a"], "epocas": ft["epocas"], "tasa": ft["tasa_aprendizaje"],
                "parte_de": "E1", "ajuste_fino": True,
                "descripcion": "Parte de E1; bloque final del backbone A descongelado"}
    raise ValueError(f"Experimento desconocido: {exp}")


def una_epoca(modelo, nombre_modelo, cargador, perdida, optimizador, disp):
    """Recorre una vez todas las imágenes de entrenamiento y ajusta los pesos."""
    modelos.poner_en_modo_entrenamiento(modelo, nombre_modelo)
    suma, n, reales, predichas = 0.0, 0, [], []
    for imagenes, etiquetas, _ in cargador:
        imagenes, etiquetas = imagenes.to(disp), etiquetas.to(disp)
        optimizador.zero_grad()            # 1. borrar los cálculos del lote anterior
        salidas = modelo(imagenes)         # 2. el modelo predice
        error = perdida(salidas, etiquetas)  # 3. se mide cuánto se equivocó
        error.backward()                   # 4. se calcula cómo corregir cada peso
        optimizador.step()                 # 5. se corrigen los pesos entrenables
        suma += error.item() * len(etiquetas)
        n += len(etiquetas)
        reales.append(etiquetas.cpu().numpy())
        predichas.append(salidas.argmax(1).cpu().numpy())
    reales, predichas = np.concatenate(reales), np.concatenate(predichas)
    return suma / n, f1_score(reales, predichas, average="macro", zero_division=0)


def ejecutar(exp, cfg=None, semilla=None, preentrenado=True, limite=None, epocas=None):
    """Ejecuta un experimento completo: entrena, elige la mejor época y evalúa en prueba.

    'limite', 'epocas' y 'preentrenado=False' existen solo para probar rápido que el código funciona.
    """
    cfg = cfg or cargar_config()
    p = cfg["protocolo"]
    clases = cfg["datos"]["clases"]
    semilla_base = p["semilla"]
    semilla = semilla_base if semilla is None else semilla
    sufijo = "" if semilla == semilla_base else f"_s{semilla}"
    # Las pruebas rápidas se guardan aparte para no confundirlas con resultados reales.
    prefijo = "PRUEBA_" if (limite or epocas or not preentrenado) else ""
    nombre = prefijo + exp + sufijo
    definicion = definir_experimento(exp, cfg)
    if epocas:
        definicion["epocas"] = epocas

    carpeta = Path("resultados") / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    ruta_pesos = Path("checkpoints") / f"{nombre}.pt"
    log = Registro(carpeta / "log.txt")

    fijar_semilla(semilla)
    disp = dispositivo()
    log(f"=== {nombre}: {definicion['descripcion']} ===")
    log(f"Dispositivo: {nombre_dispositivo()} | semilla: {semilla}")

    # ---------- Datos ----------
    partes = cargar_particiones(limite=limite, semilla=semilla)
    tam, lote, raiz = definicion["tamano"], p["tamano_lote"], cfg["datos"]["raiz"]
    recorte = definicion["modelo"] in modelos.PREENTRENADOS
    t_eval = transformaciones(tam, recorte_central=recorte)
    t_train = transformaciones(tam, aumento=definicion["aumento"], recorte_central=recorte)
    kw = {"lote": lote, "semilla": semilla, "trabajadores": p["trabajadores"]}
    c_train = crear_cargador(partes["train"], raiz, clases, t_train, barajar=True, **kw)
    c_val = crear_cargador(partes["validation"], raiz, clases, t_eval, barajar=False, **kw)
    c_test = crear_cargador(partes["test"], raiz, clases, t_eval, barajar=False, **kw)
    log(f"Imágenes: train={len(partes['train'])} val={len(partes['validation'])} test={len(partes['test'])}")

    # ---------- Modelo ----------
    modelo = modelos.crear_modelo(definicion["modelo"], len(clases), preentrenado, tam).to(disp)
    tiempo_previo = 0.0
    if definicion["modelo"] in modelos.PREENTRENADOS:
        modelos.congelar_backbone(modelo, definicion["modelo"])
    if definicion["parte_de"]:
        origen = Path("checkpoints") / f"{prefijo}{definicion['parte_de']}{sufijo}.pt"
        if not origen.exists():
            raise FileNotFoundError(f"{exp} parte de {definicion['parte_de']}: falta {origen}. Ejecute primero ese experimento.")
        estado = modelos.cargar_pesos(modelo, origen, disp)
        tiempo_previo = estado.get("tiempo_entrenamiento_s", 0.0)
        log(f"Pesos iniciales tomados de {origen} (mejor época de {definicion['parte_de']})")
    if definicion["ajuste_fino"]:
        modelos.descongelar_bloque_final(modelo, definicion["modelo"])
    tamano = modelos.tamano_modelo(modelo)
    log(f"Parámetros: {tamano['parametros_totales']:,} totales, {tamano['parametros_entrenables']:,} entrenables, {tamano['tamano_mb']} MB")

    # ---------- Pérdida y optimizador ----------
    pesos = None
    if p["pesos_por_clase"]:
        conteo = partes["train"]["clase"].value_counts().reindex(clases).fillna(1).values
        pesos = torch.tensor(conteo.sum() / (len(clases) * conteo), dtype=torch.float32, device=disp)
    perdida = nn.CrossEntropyLoss(weight=pesos)
    entrenables = [x for x in modelo.parameters() if x.requires_grad]
    optimizador = torch.optim.Adam(entrenables, lr=definicion["tasa"])

    evaluar.guardar_json(
        {"experimento": nombre, **definicion, "semilla": semilla, "tamano_lote": lote,
         "pesos_por_clase": p["pesos_por_clase"], "optimizador": "Adam", "preentrenado": preentrenado,
         "limite_prueba_rapida": limite, "dispositivo": nombre_dispositivo(),
         "torch": torch.__version__, **tamano},
        carpeta / "config.json",
    )

    # ---------- Entrenamiento ----------
    if disp.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    historial, mejor, mejor_epoca = [], -1.0, 0
    inicio = time.perf_counter()
    for epoca in range(1, definicion["epocas"] + 1):
        t0 = time.perf_counter()
        train_perdida, train_f1 = una_epoca(modelo, definicion["modelo"], c_train, perdida, optimizador, disp)
        reales, probs, val_perdida = evaluar.predecir(modelo, c_val, disp, perdida)
        val = evaluar.calcular_metricas(reales, probs.argmax(1), clases)
        historial.append({"epoca": epoca, "train_perdida": train_perdida, "train_macro_f1": train_f1,
                          "val_perdida": val_perdida, "val_macro_f1": val["macro_f1"],
                          "val_accuracy": val["accuracy"], "segundos": time.perf_counter() - t0})
        marca = ""
        if val["macro_f1"] > mejor:  # la mejor época se elige con VALIDACIÓN, nunca con prueba
            mejor, mejor_epoca, marca = val["macro_f1"], epoca, "  <- mejor hasta ahora (guardado)"
            torch.save({"modelo": modelo.state_dict(), "epoca": epoca, "val_macro_f1": mejor,
                        "tiempo_entrenamiento_s": tiempo_previo + time.perf_counter() - inicio}, ruta_pesos)
        log(f"Época {epoca:2d}/{definicion['epocas']} | train pérdida {train_perdida:.4f} F1 {train_f1:.4f} | "
            f"val pérdida {val_perdida:.4f} F1 {val['macro_f1']:.4f} acc {val['accuracy']:.4f}{marca}")
        pd.DataFrame(historial).to_csv(carpeta / "historial.csv", index=False)
    tiempo = time.perf_counter() - inicio
    evaluar.graficar_curvas(historial, f"{nombre}: curvas de entrenamiento y validación", carpeta / "curvas.png")

    # ---------- Evaluación final en PRUEBA (una sola vez, con la mejor época) ----------
    modelos.cargar_pesos(modelo, ruta_pesos, disp)
    t0 = time.perf_counter()
    reales, probs, _ = evaluar.predecir(modelo, c_test, disp, perdida)
    tiempo_test = time.perf_counter() - t0
    metricas = evaluar.calcular_metricas(reales, probs.argmax(1), clases)
    metricas.update({
        "experimento": nombre,
        "modelo": definicion["modelo"],
        "descripcion": definicion["descripcion"],
        "mejor_epoca": mejor_epoca,
        "val_macro_f1_mejor": mejor,
        "tiempo_entrenamiento_s": round(tiempo, 1),
        "tiempo_entrenamiento_acumulado_s": round(tiempo_previo + tiempo, 1),
        "tiempo_prueba_completa_s": round(tiempo_test, 1),
        "inferencia": evaluar.medir_inferencia(modelo, tam, disp, lote),
        "memoria_gpu_max_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if disp.type == "cuda" else None,
        "dispositivo": nombre_dispositivo(),
        **tamano,
    })
    evaluar.guardar_json(metricas, carpeta / "metricas.json")
    evaluar.graficar_matriz(metricas["matriz_confusion"], clases, f"{nombre}: matriz de confusión (prueba)",
                            carpeta / "matriz_confusion.png")
    evaluar.guardar_predicciones(partes["test"], reales, probs, clases, carpeta / "predicciones_test.csv")
    log(f"PRUEBA -> accuracy {metricas['accuracy']:.4f} | macro-F1 {metricas['macro_f1']:.4f} | "
        f"mejor época {mejor_epoca} | entrenamiento {tiempo:.0f} s")
    log(f"Resultados guardados en {carpeta}/")
    return metricas


def main():
    parser = argparse.ArgumentParser(description="Entrena y evalúa un experimento")
    parser.add_argument("--exp", required=True, choices=["E0", "E1", "E2", "E3", "E1_aug", "E3_aug"])
    parser.add_argument("--semilla", type=int, default=None, help="otra semilla (para medir estabilidad)")
    parser.add_argument("--limite", type=int, default=None, help="solo para probar: pocas imágenes")
    parser.add_argument("--epocas", type=int, default=None, help="solo para probar: pocas épocas")
    parser.add_argument("--sin-preentrenar", action="store_true", help="solo para probar sin internet")
    args = parser.parse_args()
    ejecutar(args.exp, semilla=args.semilla, preentrenado=not args.sin_preentrenar,
             limite=args.limite, epocas=args.epocas)


if __name__ == "__main__":
    main()
