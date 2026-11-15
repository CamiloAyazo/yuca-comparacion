"""Experimento adicional: robustez al brillo sobre una copia del conjunto de prueba.

No se entrena nada. Se toman los modelos ya entrenados (E1 y E3) y se vuelve a
evaluar el MISMO conjunto de prueba con las fotos más oscuras o más claras.
Todo lo demás queda igual, así que cualquier cambio en las métricas se debe al brillo.

Uso:  python -m src.adicional
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import evaluar, modelos
from src.comun import Registro, cargar_config, dispositivo, fijar_semilla
from src.dataset import cargar_particiones, crear_cargador, transformaciones
from src.entrenar import definir_experimento

CARPETA = Path("resultados/adicional_brillo")


def ejecutar(experimentos=("E1", "E3"), cfg=None, prefijo="", limite=None):
    cfg = cfg or cargar_config()
    p, clases, raiz = cfg["protocolo"], cfg["datos"]["clases"], cfg["datos"]["raiz"]
    carpeta = Path("resultados") / f"{prefijo}adicional_brillo"
    carpeta.mkdir(parents=True, exist_ok=True)
    log = Registro(carpeta / "log.txt")
    fijar_semilla(p["semilla"])
    disp = dispositivo()
    prueba = cargar_particiones(limite=limite, semilla=p["semilla"])["test"]
    filas = []
    for exp in experimentos:
        definicion = definir_experimento(exp, cfg)
        modelo = modelos.crear_modelo(definicion["modelo"], len(clases), preentrenado=False).to(disp)
        modelos.cargar_pesos(modelo, Path("checkpoints") / f"{prefijo}{exp}.pt", disp)
        for factor in cfg["adicional"]["factores_brillo"]:
            t = transformaciones(definicion["tamano"], brillo=None if factor == 1.0 else factor)
            cargador = crear_cargador(prueba, raiz, clases, t, p["tamano_lote"], barajar=False,
                                      trabajadores=p["trabajadores"])
            reales, probs, _ = evaluar.predecir(modelo, cargador, disp)
            m = evaluar.calcular_metricas(reales, probs.argmax(1), clases)
            fila = {"experimento": exp, "modelo": definicion["modelo"], "brillo": factor,
                    "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)}
            fila.update({f"recall_{c}": round(m["por_clase"][c]["recall"], 4) for c in clases})
            filas.append(fila)
            log(f"{exp} brillo x{factor}: accuracy {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "resultados_brillo.csv", index=False)

    fig, eje = plt.subplots(figsize=(6, 3.8))
    for exp, grupo in tabla.groupby("experimento"):
        eje.plot(grupo["brillo"], grupo["macro_f1"], marker="o", label=f"{exp} ({grupo['modelo'].iloc[0]})")
    eje.axvline(1.0, color="gray", linestyle=":", label="Imagen original")
    eje.set_xlabel("Factor de brillo aplicado a la prueba")
    eje.set_ylabel("Macro-F1 en prueba")
    eje.set_title("Robustez al cambio de brillo")
    eje.grid(alpha=0.3)
    eje.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "robustez_brillo.png", dpi=150)
    plt.close(fig)
    return tabla


if __name__ == "__main__":
    print(ejecutar().to_string(index=False))
