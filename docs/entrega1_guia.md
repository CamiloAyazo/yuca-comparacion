# Entrega 1 · Guía para el control del día 7

## 1. Qué debe estar terminado

| Requisito del profesor | Dónde queda | Cómo se obtiene |
|---|---|---|
| Fuente elegida y descarga probada | notebook 01 ejecutado | Ejecutar notebook 01 |
| Inventario de clases | `resultados/inventario.csv`, `distribucion_clases.csv` | notebook 01 |
| Ficha de datos | `docs/01_ficha_datos.md` | La completa el equipo con las cifras reales |
| Protocolo | `docs/02_protocolo.md` + `configs/config.yaml` | Lo decide y firma el equipo |
| Hipótesis previas | `docs/03_hipotesis.md` | Las redacta el equipo antes de ejecutar |
| EDA y duplicados | `resultados/eda/`, `resultados/excluidos.csv` | notebook 02 |
| Pipeline reproducible | `src/` + `configs/config.yaml` | Ya está |
| E0 ejecutado | `resultados/E0/` | notebook 03 |
| E1 al menos parcial | `resultados/E1/` | notebook 03 |
| Métricas iniciales y registro de errores | `metricas.json`, `galeria_errores.png` | notebook 03 |
| Repositorio con commits de días distintos | GitHub | Subir cambios cada día de trabajo |
| Bitácora fechada, configuraciones, logs | `bitacora/`, `config.json`, `log.txt` | Bitácora: la escribe cada uno |

## 2. Orden de trabajo

1. **Día A:** subir el proyecto a Drive y a GitHub. Ejecutar notebook 01. Completar la ficha. Primer commit.
2. **Día B:** ejecutar notebook 02. Decidir exclusiones. Cerrar el protocolo. Escribir hipótesis de E0 y E1. Commit.
3. **Día C:** activar GPU, ejecutar notebook 03 (E0 y E1). Mirar curvas, matriz y errores; escribir en la bitácora. Commit.
4. **Día D:** ensayo de la demostración en vivo (sección final del notebook 03), cada integrante por separado.

## 3. Qué abrir y mostrar el día del control

1. El repositorio en GitHub: historial de commits y carpeta `bitacora/`.
2. `configs/config.yaml` y `docs/02_protocolo.md`: "estas son las condiciones fijas".
3. `docs/03_hipotesis.md`: "esto predijimos antes de ejecutar".
4. Notebook 02 ejecutado: distribución, ejemplos, duplicados.
5. Notebook 03 ejecutado: curvas y matriz de E0 y E1, tabla comparativa, galería de errores.
6. Sección **Demostración en vivo** del notebook 03, lista para ejecutar.

## 4. Demostración en vivo: qué puede pedir y cómo hacerlo

| Petición | Qué hacer |
|---|---|
| Mostrar el entorno | Celda de GPU del notebook 03; `!pip list \| grep -i torch`; mostrar la carpeta en Drive |
| Cargar datos en vivo | Celda A: imprime la forma del lote `(8, 3, 224, 224)` y las etiquetas |
| Hacer una inferencia | Celda B: cambiar `POSICION` y ejecutar |
| Recalcular macro-F1 | Celda C: cálculo a mano y con scikit-learn |
| Cambiar una transformación | Celda D: cambiar `FACTOR`. O en `src/dataset.py`, cambiar el tamaño o quitar el recorte central |
| Explicar un error | Abrir `resultados/E1/galeria_errores.png` y la bitácora donde anotaron su hipótesis |
| Cambiar la métrica | `f1_score(pred['real'], pred['predicha'], average='weighted')` y explicar la diferencia con `macro` |
| Recall de una clase | `recall_score(pred['real'], pred['predicha'], labels=['cbb'], average=None)` |

Si Colab perdió la sesión: ejecutar la celda de preparación (≈1–2 min para descomprimir). Los modelos entrenados están en `checkpoints/` en Drive; no hay que reentrenar para las celdas A–D. Para que la variable `m1` exista sin reentrenar:
```python
import json; m1 = json.load(open('resultados/E1/metricas.json'))
```

## 5. Reparto para la explicación individual

Cada uno explica su parte, pero todos deben poder responder sobre todo.

| Integrante | Explica | Debe haber ejecutado personalmente |
|---|---|---|
| Camilo Ayazo | Datos: fuente, inventario, EDA, duplicados y fugas | Notebooks 01 y 02 |
| Camilo Mangones | Modelos: baseline, transferencia, congelar el backbone, ciclo de entrenamiento | E0 y E1 (notebook 03) |
| Hernan Herrera | Evaluación: métricas, matriz de confusión, errores, repositorio y bitácora | Demostración en vivo y análisis de errores |

## 6. Guiones (para adaptar con sus propias palabras y cifras)

Los corchetes se llenan con sus resultados y decisiones reales. Si algo no lo hicieron así, cámbienlo: el profesor pregunta sobre lo que digan.

**Camilo Ayazo — datos (≈2 min)**
"Usamos el dataset Cassava de TensorFlow Datasets, que viene de la competencia iCassava 2019. Lo elegimos porque [motivo del equipo]. Tiene 9.430 fotos en cinco clases: cuatro enfermedades y hojas sanas. Lo descargamos del mismo archivo que usa TensorFlow Datasets y armamos un inventario con una fila por imagen; las cantidades [coincidieron / no coincidieron] con las publicadas. Lo primero que notamos es el desbalance: [clase] tiene [n] fotos y [clase] apenas [n]. Por eso no nos guiamos por accuracy sino por macro-F1. Buscamos duplicados de dos formas: archivos idénticos con MD5 y fotos casi iguales con una huella visual. Encontramos [n] y decidimos [decisión], sin tocar el conjunto de prueba. Una limitación es que no hay identificador de planta, así que no podemos descartar fotos de la misma planta en entrenamiento y prueba."

**Camilo Mangones — modelos (≈2 min)**
"El E0 es una red convolucional pequeña, de unos 24 mil parámetros, entrenada desde cero con imágenes de 64×64. Nos sirve de piso: obtuvo [macro-F1]. En E1 usamos ResNet18 preentrenada con ImageNet. Congelamos todo el backbone y solo entrenamos la última capa, que tiene unos 2.500 parámetros de los 11,2 millones. La idea es que la red ya sabe reconocer bordes y texturas, y solo le enseñamos a combinarlos para nuestras cinco clases. Entrenamos [n] épocas con Adam y tasa [valor]; en cada época medimos macro-F1 en validación y guardamos el mejor modelo. La prueba se usó una sola vez. E1 obtuvo [macro-F1] frente a [macro-F1] de E0. Nuestra hipótesis era [hipótesis] y [se cumplió / no se cumplió porque…]. La dificultad que tuve fue [dificultad real] y la resolví [cómo]."

**Hernan Herrera — evaluación (≈2 min)**
"La métrica principal es macro-F1: se calcula el F1 de cada clase y se promedian dándoles el mismo peso, así una clase pequeña cuenta igual que una grande. En la matriz de confusión de E1 se ve que [confusión principal]. La clase más difícil fue [clase], con recall de [valor]. Revisamos los ocho errores donde el modelo estaba más seguro; creemos que [hipótesis del equipo sobre los errores]. También registramos el costo: E1 tardó [s] en entrenar y [ms] por imagen. Todo queda guardado automáticamente: configuración, log, métricas y predicciones, y el repositorio tiene commits de [fechas]."

## 7. Preguntas probables y la idea de la respuesta

**Datos y protocolo**
1. *¿Por qué no usar accuracy?* — Con clases desbalanceadas premia acertar la clase grande; macro-F1 da el mismo peso a cada clase.
2. *¿Qué es una fuga de información?* — Que el modelo vea en entrenamiento algo que también está en prueba (la misma foto o una casi igual): el resultado sale mejor de lo real.
3. *¿Para qué sirve validación si ya hay prueba?* — Para decidir (mejor época, hiperparámetros) sin "gastar" la prueba. Si decidiéramos con la prueba, ya no sería una medida honesta.
4. *¿Por qué fijar la semilla?* — Para que el barajado y los pesos iniciales sean los mismos y otra persona obtenga lo mismo.
5. *¿Por qué normalizan con esos números?* — Son la media y desviación de ImageNet; la red fue preentrenada con imágenes así y espera la misma escala.
6. *¿Por qué 224×224?* — Es el tamaño con el que se preentrenaron ambos backbones.

**Modelos**
7. *¿Qué significa congelar?* — Que esos pesos no reciben correcciones (`requires_grad = False`); solo aprende la cabeza.
8. *¿Cómo sé que de verdad está congelado?* — El log muestra "2,565 entrenables" de 11,179,077 totales; el optimizador solo recibe los parámetros entrenables.
9. *¿Por qué el backbone queda en modo `eval` durante el entrenamiento?* — Para que BatchNorm no actualice sus estadísticas; si no, el backbone cambiaría aunque esté "congelado".
10. *¿Qué hace el Dropout de la cabeza?* — Apaga al azar el 20 % de las características en cada paso de entrenamiento para reducir sobreajuste; en evaluación no actúa.
11. *¿Por qué E0 usa 64×64 y E1 224×224?* — E0 es deliberadamente simple y barato; no es una comparación de arquitecturas sino un piso de referencia. Hay que decirlo como limitación.
12. *¿Qué es una época / un lote / la pérdida?* — Ver `docs/00_conceptos.md` §5.

**Resultados**
13. *¿Hay sobreajuste?* — Mirar `curvas.png`: comparar entrenamiento y validación. (La interpretación es de ustedes.)
14. *¿Cuál clase es más difícil y por qué creen?* — Recall/F1 por clase + lo que vieron en el EDA y en los errores.
15. *¿Su resultado serviría en Colombia?* — No se puede afirmar: el modelo solo se evaluó con fotos del mismo origen que las de entrenamiento.

**Modificaciones prácticas**
16. *Cambie el tamaño del lote.* — `configs/config.yaml` → `tamano_lote`. Explicar: afecta memoria y velocidad.
17. *Entrene E1 solo 2 épocas.* — `ejecutar('E1', epocas=2)`; se guarda como `PRUEBA_E1` para no pisar el resultado real.
18. *Use pesos por clase.* — `pesos_por_clase: true`; la pérdida castiga más los errores en clases escasas.
19. *Muestre la F1 de una sola clase.* — `pd.DataFrame(m1['por_clase']).T`.

## 8. Cómo demostrar que entienden

- Cada uno ejecuta la demostración completa solo, sin leer.
- Cada uno cambia algo pequeño (un factor, una posición, una métrica) y explica qué pasó.
- Cada uno tiene al menos dos entradas propias en su bitácora y commits propios.
- Decir con naturalidad qué apoyo de IA usaron y mostrar `docs/declaracion_uso_ia.md`.
