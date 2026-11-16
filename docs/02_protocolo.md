# Protocolo experimental

> Este documento lo deciden **los integrantes**. Para cada decisión están las alternativas con sus ventajas y
> límites; el equipo escribe qué eligió, por qué y la fecha. Debe quedar cerrado **antes** de ejecutar E0.
>
> `configs/config.yaml` trae una **propuesta inicial** para que el código funcione:
> TFDS Cassava · PyTorch · A = ResNet18, B = MobileNetV2 (decidido) · 224×224 · semilla 42 · lote 32 ·
> E1/E3: 10 épocas, Adam, tasa 0,001 · E2: 5 épocas, tasa 0,0001 · E0: CNN pequeña a 64×64, 10 épocas ·
> sin pesos por clase · adicional: brillo ×0,5–1,5 · métrica: macro-F1.
> Si el equipo la acepta, lo escribe abajo con su motivo; si cambia algo, edita el archivo y lo registra.

## D1 · Fuente de datos
| Alternativa | A favor | En contra |
|---|---|---|
| TFDS Cassava (9.430) | Tamaño manejable en Colab gratis; particiones ya publicadas (reproducible sin partir nosotros); es la primera opción del profesor | Menos imágenes; clases desbalanceadas |
| Kaggle 2020 (≈21.397) | Más datos | Requiere cuenta y API de Kaggle; hay que crear y documentar la partición propia; más del doble de tiempo por época |
| Mendeley (India, 3 clases) | Sirve para contrastar generalización | Solo 3 clases: no cumple por sí sola el alcance de 5; etiquetas por verificar |

**Decisión del equipo:** TFDS Cassava (iCassava 2019) · **Motivo:** tamaño manejable para 15 días y la GPU gratuita de Colab · **Fecha:** 2026-11-15 (Camilo Ayazo)

## D2 · Biblioteca
| Alternativa | A favor | En contra |
|---|---|---|
| PyTorch + torchvision | Trae MobileNetV2 **y** ResNet18 preentrenadas (la pareja que recomienda el profesor) en la misma biblioteca | El ciclo de entrenamiento se escribe a mano (más líneas, pero se ve cada paso) |
| TensorFlow / Keras | Código de entrenamiento más corto (`model.fit`) | Keras no incluye ResNet18; habría que usar ResNet50 (mucho más grande) u otra equivalente y justificarlo |

**Decisión del equipo:** PyTorch + torchvision · **Motivo:** trae MobileNetV2 y ResNet18 preentrenadas en la misma biblioteca, como recomienda el profesor · **Fecha:** 2026-11-15

## D3 · Cuál es el Backbone A y cuál el B
El A se usa en E1 y E2 (recibe fine-tuning); el B solo en E3.

**Decisión del equipo:** A = ResNet18 · B = MobileNetV2 · **Motivo:** el ajuste fino (E2) se hace sobre la red con más capacidad para adaptarse, aunque tarde más. Al descongelar `layer4` de ResNet18 pasan a entrenarse ≈8,4 M de 11,2 M parámetros · **Fecha:** 2026-11-15

## D4 · Tamaño de imagen y normalización
- 224×224 es el tamaño con el que se preentrenaron ambas redes. Menor tamaño (p. ej. 160) entrena más rápido pero puede borrar lesiones pequeñas.
- Normalización: cada backbone debe recibir la que usó en su preentrenamiento (el profesor lo exige).

**Decisión del equipo:** 224×224 para E1–E3 (tamaño de preentrenamiento) y normalización de ImageNet para ambos backbones · **Fecha:** 2026-11-15

## D5 · Semilla y estabilidad
- Una sola semilla: más barato, pero no permite hablar de estabilidad.
- Varias semillas (p. ej. 3) en E1 y E3: permite reportar media y variación; triplica el tiempo de esos experimentos.

**Decisión del equipo:** semillas 42, 1 y 2 en E1 y E3 (media y variación); E0 y E2 con semilla 42 · **Motivo:** saber si una diferencia entre arquitecturas es mayor que la variación por azar · **Fecha:** 2026-11-15

## D6 · Presupuesto de entrenamiento
Debe ser el mismo para E1 y E3: épocas máximas, tamaño de lote, optimizador, tasa de aprendizaje y criterio de parada.
El modelo se elige mirando **validación**, nunca prueba.

**Decisión del equipo:** lote 32, Adam. E1 y E3: 10 épocas, tasa 0,001. E2: 5 épocas, tasa 0,0001 (parte del mejor E1). La mejor época se elige por macro-F1 en validación · **Fecha:** 2026-11-15

## D7 · Diseño de E0 (baseline)
| Alternativa | A favor | En contra |
|---|---|---|
| Imagen reducida o histograma de color + clasificador lineal | Muy rápido, fácil de explicar | Probablemente muy débil |
| CNN pequeña entrenada desde cero | Más comparable con las redes grandes | Tarda más; más decisiones que justificar |

**Decisión del equipo:** CNN pequeña entrenada desde cero, imágenes de 64×64, 10 épocas, tasa 0,001 · **Fecha:** 2026-11-15

## D8 · Desbalance de clases
- No hacer nada y analizarlo con macro-F1 y métricas por clase.
- Pesos por clase en la función de pérdida.
Lo que se elija debe aplicarse igual en E1, E2 y E3.

**Decisión del equipo:** pesos por clase en la función de pérdida, iguales en E0–E3 · **Motivo:** que los errores en las clases escasas (cbb, healthy) pesen más · **Fecha:** 2026-11-15

## D9 · Experimento adicional
- Augmentation moderado solo en el entrenamiento de E1 o E3; o
- Perturbación de brillo sobre una copia del conjunto de prueba.

**Decisión del equipo:** ______ · **Objetivo declarado:** ______

## D10 · Métrica principal
El profesor recomienda macro-F1. **Decisión del equipo:** macro-F1 · **Fecha:** 2026-11-15

## Reglas fijas (vienen del documento del profesor)
- El mismo conjunto de prueba para todos los experimentos.
- El conjunto de prueba no se consulta para elegir hiperparámetros.
- Sin augmentation en validación ni prueba.
- Comparación primaria E1 vs E3; secundaria E1 vs E2.

## D11 · Duplicados
Ver notebook 02, sección 7. **Fecha:** 2026-11-15 (Camilo Ayazo).
- `INCLUIR_CASI = True`: se retiran de entrenamiento/validación los casi duplicados con copia en otra partición. **Motivo:** menos fuga de información; la prueba no se toca.
- `RETIRAR_CONFLICTIVAS = True` (opción agregada al código tras ver los datos): si una misma foto aparece con etiquetas distintas, se retiran todas sus copias de entrenamiento y validación. **Motivo:** que el modelo no aprenda de una etiqueta dudosa.
- **Prueba:** no se modifica; se usan las 1.885 fotos publicadas y se declara como limitación que unas 10 tienen etiqueta contradicha por una copia idéntica.
- **Imágenes retiradas:** 68 (56 de entrenamiento y 12 de validación). Particiones finales: **5.600 / 1.877 / 1.885**. Lista en `resultados/excluidos.csv`.
