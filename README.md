# Comparación experimental de modelos preentrenados para reconocer enfermedades de la yuca

Proyecto de Aprendizaje Computacional – Ingeniería de Sistemas.

**Integrantes:** Camilo De Jesús Ayazo Argel · Camilo José Mangones Florez · Hernan Dario Herrera Sepúlveda

**Pregunta de investigación:** ¿Qué arquitectura de red neuronal preentrenada ofrece el mejor equilibrio entre
desempeño por clase, generalización y costo computacional para clasificar enfermedades de la yuca, bajo un
protocolo experimental común?

## Experimentos

| Exp. | Diseño | Pregunta |
|---|---|---|
| E0 | CNN pequeña entrenada desde cero | ¿Qué se consigue sin una arquitectura sofisticada? |
| E1 | Backbone A (MobileNetV2) congelado + cabeza | ¿Qué aporta el aprendizaje por transferencia? |
| E2 | Parte de E1; bloque final descongelado, tasa menor | ¿Mejora frente a E1 y a qué costo? |
| E3 | Backbone B (ResNet18) congelado + cabeza | ¿Qué arquitectura ofrece mejor compromiso? |
| Adicional | Brillo sobre copia de la prueba (o augmentation en E1) | Robustez |

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/` | Código: `datos`, `eda`, `dataset`, `modelos`, `entrenar`, `evaluar`, `adicional`, `analisis`, `comun` |
| `notebooks/` | 01 datos · 02 EDA y duplicados · 03 E0 y E1 · 04 E2 y E3 · 05 adicional y análisis |
| `configs/config.yaml` | Protocolo: semilla, tamaños, épocas, tasas |
| `resultados/` | Una carpeta por experimento: configuración, log, historial, métricas, gráficas, predicciones |
| `docs/` | Conceptos, cómo se hizo, ficha de datos, protocolo, hipótesis, guías de las dos entregas, declaración de IA |
| `bitacora/` | Bitácora grupal e individuales |
| `ESTADO.md` | Lista de actividades |

## Cómo ejecutar (Google Colab)

1. Copiar la carpeta `yuca-comparacion` a **Mi unidad** de Google Drive (debe quedar en `Mi unidad/yuca-comparacion`).
2. En Drive, abrir `notebooks/01_descarga_inventario.ipynb` con doble clic → *Abrir con Google Colaboratory*.
3. Menú **Entorno de ejecución → Ejecutar todo** y aceptar el permiso de Drive.
4. Seguir en orden: 02 → 03 → 04 → 05. Desde el 03 activar la GPU:
   **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**.

Los datos no están en el repositorio: el notebook descarga el `.zip` oficial (≈1,3 GB) una sola vez a `data/` en Drive.
Los modelos entrenados quedan en `checkpoints/` (Drive), tampoco se suben a GitHub.

Si se modifica un archivo de `src/` con el notebook abierto: **Entorno de ejecución → Reiniciar sesión** y volver a ejecutar.

## Ejecución por terminal (opcional)

```
pip install -r requirements.txt
python -m src.datos --destino data
python -m src.entrenar --exp E0
python -m src.entrenar --exp E1
python -m src.entrenar --exp E3
python -m src.entrenar --exp E2
python -m src.adicional
```

## Reproducibilidad

Semilla, particiones publicadas del dataset, lista de exclusiones (`resultados/excluidos.csv`) y configuración
exacta de cada ejecución (`resultados/<exp>/config.json`).
