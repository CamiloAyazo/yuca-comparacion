# Bitácora grupal

Proyecto: Comparación experimental de modelos preentrenados para reconocer enfermedades de la yuca.
Formato: **problema encontrado → alternativas → decisión → prueba → resultado → revisión de la decisión.**

> Las razones de cada decisión las dio el integrante que la tomó. "Revisión de la decisión" se llena días después, cuando ya se sabe si fue acertada.

---

## 2026-11-14 · Entorno de trabajo y estructura del proyecto
- **Quién:** Camilo Ayazo.
- **Problema encontrado:** el equipo no tenía definido dónde ejecutar los experimentos ni cómo organizar el código, y no tiene experiencia con terminales ni con instalación de bibliotecas.
- **Alternativas consideradas:** (a) Google Colab + Google Drive + GitHub por la página web; (b) instalación local con Visual Studio Code y terminal.
- **Decisión:** opción (a). Motivos: nuestros computadores no tienen GPU y Colab ofrece una gratuita; no tenemos experiencia instalando bibliotecas ni usando la terminal y Colab ya trae todo listo; y los tres podemos abrir y ejecutar lo mismo desde cualquier computador.
- **Prueba realizada:** se creó la estructura del proyecto (`src/`, `notebooks/`, `configs/`, `docs/`, `bitacora/`, `resultados/`).
- **Resultado:** proyecto listo para subir al repositorio.
- **Revisión de la decisión:** pendiente (se revisa después de ejecutar los experimentos).

## 2026-11-14 · Creación del repositorio y primera subida
- **Quién:** Camilo Ayazo.
- **Problema encontrado:** al subir el proyecto a GitHub por la página web, en la lista solo aparecieron los cuatro archivos sueltos (`.gitignore`, `ESTADO.md`, `README.md`, `requirements.txt`); las carpetas no se habían agregado.
- **Alternativas consideradas:** (a) repetir la carga arrastrando también las carpetas desde el Explorador de archivos; (b) usar el enlace "choose your files" de la página, que solo permite elegir archivos sueltos.
- **Decisión:** opción (a), porque es la única que conserva la estructura de carpetas, y hacerlo con la cuenta propia para aprender el procedimiento, que se repetirá en cada avance.
- **Prueba realizada:** se seleccionó todo el contenido de la carpeta y se arrastró a la página de carga de GitHub.
- **Resultado:** repositorio `CamiloAyazo/yuca-comparacion` con dos commits: "Initial commit" (14-nov, 11:40 p. m.) y "Estructura inicial del proyecto" (15-nov, 12:00 a. m.). Quedaron 36 archivos. No se subieron la carpeta `resultados` (estaba vacía y GitHub no sube carpetas vacías) ni `.gitignore`.
- **Revisión de la decisión:** pendiente (se revisa después de ejecutar los experimentos).

## 2026-11-15 · Error al ejecutar el notebook 01 en Colab
- **Quién:** Camilo Ayazo.
- **Problema encontrado:** la primera celda del notebook 01 falló con `ModuleNotFoundError: No module named 'src'`, aunque los permisos de Google Drive se habían aceptado.
- **Alternativas consideradas:** (a) que faltara algún permiso de Drive; (b) que la carpeta no estuviera en "Mi unidad"; (c) que hubiera una carpeta dentro de otra.
- **Decisión:** diagnosticar antes de cambiar nada, listando el contenido de la carpeta desde una celda de Colab.
- **Prueba realizada:** `os.listdir('/content/drive/MyDrive/yuca-comparacion')` mostró que adentro solo había otra carpeta `yuca-comparacion` con el proyecto. Era el caso (c).
- **Resultado:** se movió el contenido un nivel hacia arriba con una celda (`shutil.move`) y se eliminó la carpeta interna vacía. El proyecto quedó en `/content/drive/MyDrive/yuca-comparacion`.
- **Revisión de la decisión:** fue acertado diagnosticar antes de cambiar algo. Aprendimos dos cosas: (1) después de descomprimir un archivo hay que revisar la estructura de carpetas antes de subirla, porque puede quedar una carpeta dentro de otra; (2) las rutas importan: el código busca los archivos en una ubicación exacta y, si la carpeta queda en otro nivel, deja de funcionar aunque los archivos estén completos.

## 2026-11-15 · Fuente de datos, descarga e inventario
- **Quién:** Camilo Ayazo.
- **Problema encontrado:** había que elegir la fuente de datos entre las tres propuestas por el profesor y comprobar que la descarga funcionara.
- **Alternativas consideradas:** (a) TensorFlow Datasets – Cassava (9.430 imágenes, particiones publicadas); (b) Kaggle 2020 (≈21.397 imágenes, requiere partición propia); (c) Mendeley (India, 3 clases).
- **Decisión:** opción (a). Motivo: tamaño manejable. Con 15 días de plazo y la GPU gratuita de Colab, 9.430 imágenes se pueden entrenar varias veces en poco tiempo; Kaggle 2020 tiene más del doble.
- **Prueba realizada:** notebook `01_descarga_inventario.ipynb` en Google Colab. Descarga del archivo `cassavaleafdata.zip` desde la fuente que usa TensorFlow Datasets.
- **Resultado:**
  - Descarga completa: 1.354 MB en 46 segundos.
  - 9.430 imágenes inventariadas, sin avisos de archivos ilegibles.
  - Particiones encontradas: entrenamiento 5.656, validación 1.889, prueba 1.885. Coinciden con las publicadas.
  - Distribución total: cmd 4.431 (47,0 %), cbsd 2.406 (25,5 %), cgm 1.289 (13,7 %), cbb 777 (8,2 %), healthy 527 (5,6 %). La proporción es la misma en las tres particiones.
  - Las resoluciones no son uniformes (por ejemplo 666×500, 500×500 y 500×666).
- **Revisión de la decisión:** pendiente (se revisa después de ejecutar los experimentos).

## 2026-11-15 · EDA y control de duplicados (notebook 02)
- **Quién:** Camilo Ayazo.
- **Problema encontrado:** el control de duplicados mostró 26 grupos de archivos idénticos (68 archivos). En 18 de esos grupos la misma foto tiene etiquetas distintas, por ejemplo el mismo archivo guardado como `cbb`, `cbsd` y `cgm`. Además aparecieron 29 pares de fotos casi idénticas, 20 de ellos en particiones distintas; revisamos los 6 que dibuja el notebook y son la misma foto. La regla inicial de exclusión conservaba una copia con una de sus etiquetas, elegida por orden alfabético.
- **Alternativas consideradas:** (1) para los casi duplicados: retirarlos o dejarlos y declararlo como limitación; (2) para las fotos con etiquetas contradictorias: retirar todas sus copias de entrenamiento y validación, o mantener la regla inicial; (3) para la prueba: no tocarla, o retirar las fotos dudosas en todos los experimentos.
- **Decisión:** retirar los casi duplicados entre particiones, para reducir la fuga de información; retirar todas las copias de las fotos con etiquetas contradictorias, para que el modelo no aprenda de una etiqueta dudosa; y no tocar la prueba, declarando como limitación que unas 10 fotos tienen etiqueta dudosa.
- **Prueba realizada:** se agregó a `generar_exclusiones` (en `src/eda.py`) la opción `retirar_conflictivas` y se volvió a ejecutar el notebook 02 completo.
- **Resultado:** se retiran 68 imágenes (56 de entrenamiento y 12 de validación). Particiones finales: entrenamiento 5.600, validación 1.877, prueba 1.885. El brillo promedio por clase quedó entre 114,5 y 124,0 (escala 0–255).
- **Revisión de la decisión:** pendiente (se revisa después de ejecutar los experimentos).

## 2026-11-15 · Protocolo, hipótesis y primeros experimentos: E0 y E1 (notebook 03)
- **Quién:** Camilo Ayazo.
- **Problema encontrado:** antes de entrenar había que cerrar el protocolo y escribir las hipótesis, como exige el profesor. Además, la configuración guardada en Drive todavía tenía los valores de la propuesta inicial.
- **Alternativas consideradas:** Backbone A = MobileNetV2 o ResNet18; una semilla o tres; con o sin pesos por clase; presupuesto propuesto, E0 lineal o menos épocas.
- **Decisión:** A = ResNet18 y B = MobileNetV2; tres semillas en E1 y E3; pesos por clase; presupuesto propuesto. Se registraron las hipótesis de E0 y E1 antes de ejecutar (`docs/03_hipotesis.md`). Se actualizó `configs/config.yaml` en Drive.
- **Prueba realizada:** notebook `03_E0_E1.ipynb` en Colab con GPU Tesla T4, semilla 42.
- **Resultado (prueba, 1.885 imágenes):**
  - E0 (CNN pequeña, 64×64): macro-F1 0,464 · accuracy 0,597 · mejor época 8 · entrenamiento 348 s · 0,58 ms por imagen · 0,1 MB.
  - E1 (ResNet18 congelada): macro-F1 0,659 · accuracy 0,741 · mejor época 8 · entrenamiento 543 s · 3,72 ms por imagen · 44,8 MB.
  - F1 por clase en E1: cmd 0,870 · cbsd 0,692 · healthy 0,670 · cgm 0,648 · cbb 0,414.
  - 92 de 481 fotos de cbsd se clasificaron como cbb. En los 8 errores de mayor confianza de E1, 6 fueron predichos como cbb, casi todos en hojas muy amarillas o marchitas con puntos oscuros; dos de ellos (`test-cgm-136` y `test-cgm-102`) son fotos de prueba cuya copia idéntica tenía otra etiqueta.
  - La demostración en vivo funcionó: carga de un lote (8, 3, 224, 224), inferencia de una foto, macro-F1 a mano igual al de scikit-learn (0,659) y cambio de brillo.
- **Revisión de la decisión:** pendiente (se revisa con E2 y E3).

---

## Plantilla para las siguientes entradas

## AAAA-MM-DD · título corto
- **Quién:**
- **Problema encontrado:**
- **Alternativas consideradas:**
- **Decisión:**
- **Prueba realizada:**
- **Resultado:**
- **Revisión de la decisión:**
