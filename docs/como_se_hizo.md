# Cómo se hizo el proyecto

Explicación de cada pieza, organizada por entrega. Para cada parte: qué hace, cómo está hecha, cómo se ejecuta y qué evidencia deja.

## Organización general

- `src/` contiene el código. Los notebooks solo **llaman** a ese código y muestran resultados. Así un cambio se hace en un solo lugar y todos los experimentos usan exactamente las mismas funciones (comparación justa).
- `configs/config.yaml` reúne todos los valores del protocolo (semilla, tamaño, épocas, tasas). Ningún número importante está escondido en el código.
- Cada experimento guarda sus resultados en `resultados/<nombre>/`. Nada se copia a mano.
- El proyecto vive en Google Drive (`MiUnidad/yuca-comparacion`) y se ejecuta en Google Colab. Lo que cambia se sube a GitHub.

---

# ENTREGA 1 (control del día 7)

## 1. Descarga e inventario — `src/datos.py`, notebook 01
- **Qué hace:** baja el `.zip` oficial (el mismo que usa TensorFlow Datasets), lo descomprime y crea una tabla con una fila por imagen.
- **Cómo:** `descargar_dataset()` descarga por bloques a un archivo `.parcial` y solo lo renombra al terminar (una descarga cortada no queda como buena). `construir_inventario()` recorre `train/validation/test`, abre cada foto con Pillow para leer ancho y alto, verifica que no esté dañada y calcula su huella MD5. `verificar_conteos()` compara con 5.656 / 1.889 / 1.885.
- **Evidencia:** `resultados/inventario.csv`, `resultados/distribucion_clases.csv`.
- **Por qué las particiones publicadas:** no hay que inventar una división; cualquiera puede reproducirla.

## 2. EDA y duplicados — `src/eda.py`, notebook 02
- **Qué hace:** gráfica de clases, tamaños, mosaico de ejemplos, brillo por clase, duplicados exactos y casi duplicados.
- **Cómo se detectan duplicados exactos:** misma huella MD5 = mismo archivo.
- **Cómo se detectan casi duplicados:** *dHash*. Cada foto se reduce a 9×8 píxeles en gris y se anota, para cada píxel, si es más claro que su vecino: salen 64 bits. Dos fotos casi iguales difieren en muy pocos bits (usamos máximo 2).
- **Cómo se evita la fuga:** `generar_exclusiones()` retira de entrenamiento/validación las imágenes repetidas en una partición de más prioridad (prueba > validación > entrenamiento). La prueba nunca se toca. La lista queda en `resultados/excluidos.csv` y `cargar_particiones()` la aplica siempre.
- **Límite que hay que declarar:** la versión descargada solo trae imagen, nombre de archivo y etiqueta (sin identificador de planta o finca), así que no podemos garantizar que fotos de la misma planta no estén en dos particiones.

## 3. Pipeline de datos — `src/dataset.py`
- `transformaciones()`: redimensionar → recorte central 224×224 → convertir a tensor → normalizar con media y desviación de ImageNet. Es igual en entrenamiento, validación y prueba (sin augmentation), por lo que cualquier diferencia entre modelos no viene de los datos.
- `HojasYuca`: entrega `(imagen, etiqueta, posición)` leyendo las rutas del inventario.
- `crear_cargador()`: arma lotes de 32; solo baraja el entrenamiento y lo hace con la semilla fija.

## 4. Reproducibilidad — `src/comun.py`
`fijar_semilla()` fija el azar de Python, NumPy y PyTorch. `Registro` escribe cada mensaje en pantalla y en `log.txt` con fecha y hora.

## 5. E0 baseline — `CNNPequena` en `src/modelos.py`
Tres bloques (convolución 3×3 → normalización → ReLU → reducción a la mitad) con 16, 32 y 64 filtros, promedio global y una capa lineal de 5 salidas. ≈24.000 parámetros, imágenes de 64×64, entrenada desde cero. Alternativa disponible en la configuración: `tipo: lineal` (clasificador lineal sobre píxeles).

## 6. E1 extracción de características — `src/modelos.py` y `src/entrenar.py`
1. `crear_modelo('mobilenet_v2')` carga los pesos de ImageNet y reemplaza la última capa (1.000 clases) por `Dropout + Linear(1280 → 5)`.
2. `congelar_backbone()` marca todos los pesos como no entrenables, menos la cabeza (6.405 parámetros entrenables de 2,2 millones).
3. `poner_en_modo_entrenamiento()` deja el backbone en modo evaluación para que sus capas BatchNorm no cambien sus estadísticas.
4. El ciclo `una_epoca()` hace los cinco pasos: borrar gradientes → predecir → medir error → calcular correcciones → aplicar correcciones.
5. Al final de cada época se evalúa en **validación**; si el macro-F1 mejora, se guarda ese modelo. La prueba se evalúa una sola vez, con el mejor modelo guardado.

## 7. Métricas — `src/evaluar.py`
`calcular_metricas()` usa scikit-learn (accuracy, macro-F1, precisión/recall/F1 por clase, matriz de confusión). `macro_f1_manual()` hace la misma cuenta a mano para poder explicarla. `medir_inferencia()` mide milisegundos por imagen con lote de 1 y de 32, después de un calentamiento y sin contar la lectura del disco.

## 8. Archivos que deja cada experimento
`config.json` (condiciones exactas), `log.txt`, `historial.csv` (una fila por época), `curvas.png`, `metricas.json`, `matriz_confusion.png`, `predicciones_test.csv` (una fila por foto de prueba).

---

# ENTREGA 2 (proyecto final)

## 9. E3 segunda arquitectura
Idéntico a E1, cambiando solo `mobilenet_v2` por `resnet18` (cabeza `Dropout + Linear(512 → 5)`). Épocas, tasa, lote, semilla, datos y transformaciones salen de las mismas líneas de `config.yaml` (sección `extraccion`), por lo que la única diferencia es la arquitectura.

## 10. E2 ajuste fino
`ejecutar('E2')` carga `checkpoints/E1.pt` (el mejor E1), llama `descongelar_bloque_final()` (en MobileNetV2: `features[17:]`, el último bloque residual y la convolución final; ≈890.000 parámetros entrenables) y entrena 5 épocas con tasa 0,0001. Se reporta el tiempo propio y el **acumulado** (E1 + E2), porque E2 no existe sin E1.

## 11. Estabilidad
`ejecutar('E1', semilla=1)` repite el experimento con otra semilla y lo guarda como `E1_s1`. `analisis.estabilidad()` calcula media y desviación entre semillas.

## 12. Experimento adicional — `src/adicional.py`
- **Brillo:** se cargan los modelos E1 y E3 ya entrenados y se evalúa el mismo conjunto de prueba multiplicando el brillo por 0,5 · 0,75 · 1,0 · 1,25 · 1,5. Con factor 1,0 el resultado debe coincidir con el original (comprobación incluida).
- **Augmentation (alternativa):** `ejecutar('E1_aug')` repite E1 con recorte aleatorio (70–100 %), volteo horizontal, rotación ±15° y variación leve de brillo y contraste, solo en entrenamiento.

## 13. Análisis — `src/analisis.py`
- `tabla_consolidada()`: lee los `metricas.json` y arma la tabla E0–E3 (desempeño y costo).
- `graficas_comparacion()`: F1 por clase y macro-F1 frente a tiempo.
- `galeria_errores()`: los 8 errores de mayor confianza (criterio fijo para no escoger a conveniencia).
- `pares_confundidos()`: confusiones más frecuentes.
- `gradcam_galeria()`: Grad-CAM implementado con dos "ganchos" en la última capa convolucional: uno guarda las activaciones y otro los gradientes; el mapa es la suma de activaciones ponderada por el promedio de sus gradientes.

## 14. Qué se verificó y qué no
- **Verificado por la IA:** todo el código y los cinco notebooks se ejecutaron completos con un conjunto **sintético** de la misma estructura (320 imágenes, sin pesos preentrenados, 1–2 épocas, CPU). Sirvió para comprobar que nada falla, que el macro-F1 manual coincide con scikit-learn y que brillo ×1,0 reproduce el resultado original.
- **No verificado:** la descarga real, los pesos de ImageNet y la GPU. Ningún resultado de ese ensayo es un resultado del proyecto.
