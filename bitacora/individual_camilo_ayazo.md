# Bitácora individual · Camilo De Jesús Ayazo Argel

Responsabilidad principal: datos (fuente, inventario, EDA, duplicados y fugas).

---

## 2026-11-14 (sábado)
- **Qué hice hoy:**
  - Leí el documento del profesor y revisé el plan de trabajo para las dos entregas.
  - Creé el repositorio `yuca-comparacion` en mi cuenta de GitHub.
  - Subí la estructura inicial del proyecto por la página web.
- **Qué archivo toqué / qué ejecuté:** todo el proyecto (`src/`, `notebooks/`, `configs/`, `docs/`, `bitacora/`). No ejecuté experimentos.
- **Qué decidí y por qué:** subir yo mismo los archivos al repositorio. Es la manera correcta de hacer las cosas: de nada me vale que quede hecho si al final no aprendo a hacerlo.
- **Qué dificultad tuve y cómo la resolví:** en el primer intento solo se cargaron los cuatro archivos sueltos y faltaban las carpetas. Lo resolví seleccionando todo el contenido en el Explorador de archivos y arrastrándolo a la página de carga; después comprobé en la página principal del repositorio que aparecieran las carpetas.

## 2026-11-15 (domingo, madrugada)
- **Qué hice hoy:** intenté ejecutar por primera vez el notebook 01 en Google Colab.
- **Qué archivo toqué / qué ejecuté:** `notebooks/01_descarga_inventario.ipynb`.
- **Qué decidí y por qué:** revisar el contenido de la carpeta de Drive desde Colab antes de mover o borrar nada, para saber la causa real del error.
- **Qué dificultad tuve y cómo la resolví:** la primera celda falló con `ModuleNotFoundError: No module named 'src'`. Al listar la carpeta vi que el proyecto había quedado en `yuca-comparacion/yuca-comparacion/`, es decir, una carpeta dentro de otra. Quedó diagnosticado y lo corregí en la sesión siguiente.
- **Qué aprendí:** a revisar la estructura de carpetas después de descomprimir y antes de subir, y que las rutas importan: el código busca los archivos en un lugar exacto.

## 2026-11-15 (domingo, noche)
- **Qué hice hoy:**
  - Corregí la carpeta duplicada en Drive moviendo el contenido un nivel hacia arriba.
  - Ejecuté completo el notebook 01: descarga e inventario del conjunto de datos.
  - Repasé el objetivo del proyecto y el orden de los experimentos (E0 a E3 y el adicional).
- **Qué archivo toqué / qué ejecuté:** `notebooks/01_descarga_inventario.ipynb`, que usa `src/datos.py`. Produjo `resultados/inventario.csv` y `resultados/distribucion_clases.csv`.
- **Qué obtuve:**
  - Descarga de 1.354 MB en 46 segundos.
  - 9.430 imágenes inventariadas.
  - Entrenamiento 5.656, validación 1.889 y prueba 1.885: coincide con lo publicado.
  - La clase con más fotos es cmd (4.431; 47,0 %) y la que menos tiene es healthy (527; 5,6 %).
- **Qué decidí y por qué:** trabajar con el conjunto Cassava de TensorFlow Datasets, por su tamaño manejable para el tiempo y los recursos que tenemos.
- **Qué dificultad tuve y cómo la resolví:** tuve que volver a conectar Google Drive porque la sesión de Colab anterior se había cerrado; lo hice desde una celda y acepté todos los permisos.
- **Qué me llamó la atención de los datos:** el desbalance. Casi la mitad de las fotos (47 %) son de una sola clase, cmd, mientras que healthy tiene apenas 5,6 %. Eso significa que el accuracy puede engañar: un modelo que respondiera siempre "cmd" acertaría cerca del 47 % sin haber aprendido nada. Por eso tiene sentido usar macro-F1 como métrica principal.
- **Qué no entiendo todavía:** qué significa exactamente congelar el backbone y en qué se diferencia del ajuste fino. Debo aclararlo antes de ejecutar E1.
- **Notebook 02 (EDA y duplicados):** lo ejecuté y revisé los duplicados. Encontré que algunas fotos están repetidas con etiquetas distintas (la misma foto como cbb, cbsd y cgm) y que hay fotos casi iguales entre entrenamiento, validación y prueba. Decidí retirar de entrenamiento y validación los casi duplicados y todas las copias de las fotos con etiquetas contradictorias, sin tocar la prueba. Quedaron 5.600 / 1.877 / 1.885 imágenes.
- **Protocolo e hipótesis:** antes de entrenar cerré el protocolo (Backbone A = ResNet18 y B = MobileNetV2, tres semillas en E1 y E3, pesos por clase, presupuesto de la propuesta) y escribí las hipótesis de E0 y E1 en `docs/03_hipotesis.md`.
- **E0 y E1:** se entrenaron en Colab con GPU T4. E0 obtuvo macro-F1 0,464 y E1 0,659. Mi hipótesis de E0 (> 0,55) no se cumplió y la de E1 (+0,20) quedó a 0,005 del umbral; la considero cumplida en esencia. La clase más difícil no fue cbsd, como esperaba, sino cbb. Al ver los errores, creo que se combinan síntomas parecidos entre cbb y cbsd, un posible atajo de "hoja amarilla = cbb" y las etiquetas dudosas que encontramos en los duplicados.
- **Por qué E0 quedó por debajo de lo que predije:** creo que al reducir las fotos a 64×64 se pierde el detalle y desaparecen las manchas pequeñas que distinguen las enfermedades.

---

## Plantilla para las siguientes entradas

## AAAA-MM-DD
- **Qué hice hoy:**
- **Qué archivo toqué / qué ejecuté:**
- **Qué decidí y por qué:**
- **Qué dificultad tuve y cómo la resolví:**
- **Qué no entiendo todavía:**
