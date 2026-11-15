# Protocolo experimental

> Este documento lo deciden **los integrantes**. Para cada decisión están las alternativas con sus ventajas y
> límites; el equipo escribe qué eligió, por qué y la fecha. Debe quedar cerrado **antes** de ejecutar E0.
>
> `configs/config.yaml` trae una **propuesta inicial** para que el código funcione:
> TFDS Cassava · PyTorch · A = MobileNetV2, B = ResNet18 · 224×224 · semilla 42 · lote 32 ·
> E1/E3: 10 épocas, Adam, tasa 0,001 · E2: 5 épocas, tasa 0,0001 · E0: CNN pequeña a 64×64, 10 épocas ·
> sin pesos por clase · adicional: brillo ×0,5–1,5 · métrica: macro-F1.
> Si el equipo la acepta, lo escribe abajo con su motivo; si cambia algo, edita el archivo y lo registra.

## D1 · Fuente de datos
| Alternativa | A favor | En contra |
|---|---|---|
| TFDS Cassava (9.430) | Tamaño manejable en Colab gratis; particiones ya publicadas (reproducible sin partir nosotros); es la primera opción del profesor | Menos imágenes; clases desbalanceadas |
| Kaggle 2020 (≈21.397) | Más datos | Requiere cuenta y API de Kaggle; hay que crear y documentar la partición propia; más del doble de tiempo por época |
| Mendeley (India, 3 clases) | Sirve para contrastar generalización | Solo 3 clases: no cumple por sí sola el alcance de 5; etiquetas por verificar |

**Decisión del equipo:** ______ · **Motivo:** ______ · **Fecha:** ______

## D2 · Biblioteca
| Alternativa | A favor | En contra |
|---|---|---|
| PyTorch + torchvision | Trae MobileNetV2 **y** ResNet18 preentrenadas (la pareja que recomienda el profesor) en la misma biblioteca | El ciclo de entrenamiento se escribe a mano (más líneas, pero se ve cada paso) |
| TensorFlow / Keras | Código de entrenamiento más corto (`model.fit`) | Keras no incluye ResNet18; habría que usar ResNet50 (mucho más grande) u otra equivalente y justificarlo |

**Decisión del equipo:** ______ · **Motivo:** ______ · **Fecha:** ______

## D3 · Cuál es el Backbone A y cuál el B
El A se usa en E1 y E2 (recibe fine-tuning); el B solo en E3.

**Decisión del equipo:** A = ______ · B = ______ · **Motivo:** ______

## D4 · Tamaño de imagen y normalización
- 224×224 es el tamaño con el que se preentrenaron ambas redes. Menor tamaño (p. ej. 160) entrena más rápido pero puede borrar lesiones pequeñas.
- Normalización: cada backbone debe recibir la que usó en su preentrenamiento (el profesor lo exige).

**Decisión del equipo:** ______

## D5 · Semilla y estabilidad
- Una sola semilla: más barato, pero no permite hablar de estabilidad.
- Varias semillas (p. ej. 3) en E1 y E3: permite reportar media y variación; triplica el tiempo de esos experimentos.

**Decisión del equipo:** semilla(s) = ______

## D6 · Presupuesto de entrenamiento
Debe ser el mismo para E1 y E3: épocas máximas, tamaño de lote, optimizador, tasa de aprendizaje y criterio de parada.
El modelo se elige mirando **validación**, nunca prueba.

**Decisión del equipo:** ______

## D7 · Diseño de E0 (baseline)
| Alternativa | A favor | En contra |
|---|---|---|
| Imagen reducida o histograma de color + clasificador lineal | Muy rápido, fácil de explicar | Probablemente muy débil |
| CNN pequeña entrenada desde cero | Más comparable con las redes grandes | Tarda más; más decisiones que justificar |

**Decisión del equipo:** ______

## D8 · Desbalance de clases
- No hacer nada y analizarlo con macro-F1 y métricas por clase.
- Pesos por clase en la función de pérdida.
Lo que se elija debe aplicarse igual en E1, E2 y E3.

**Decisión del equipo:** ______

## D9 · Experimento adicional
- Augmentation moderado solo en el entrenamiento de E1 o E3; o
- Perturbación de brillo sobre una copia del conjunto de prueba.

**Decisión del equipo:** ______ · **Objetivo declarado:** ______

## D10 · Métrica principal
El profesor recomienda macro-F1. **Decisión del equipo:** ______

## Reglas fijas (vienen del documento del profesor)
- El mismo conjunto de prueba para todos los experimentos.
- El conjunto de prueba no se consulta para elegir hiperparámetros.
- Sin augmentation en validación ni prueba.
- Comparación primaria E1 vs E3; secundaria E1 vs E2.

## D11 · Duplicados
Ver notebook 02, sección 7. **Decisión del equipo:** `INCLUIR_CASI` = ______ · **Motivo:** ______ · **Imágenes retiradas:** ______
