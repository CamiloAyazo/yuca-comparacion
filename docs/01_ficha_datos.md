# Ficha del conjunto de datos

> Los campos marcados **[por completar]** se llenan con la salida real del notebook 01 o los verifica el equipo.
> No escribir cifras que no hayamos obtenido nosotros.

| Campo | Valor | Cómo se verificó |
|---|---|---|
| Nombre | Cassava (iCassava 2019) | Catálogo de TensorFlow Datasets |
| Página del catálogo | https://www.tensorflow.org/datasets/catalog/cassava | Documento del profesor |
| Archivo descargado | https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip | Código fuente de TFDS (`cassava.py`) |
| Versión | 0.1.0 (versión del constructor en TFDS) | Código fuente de TFDS |
| Origen | Competencia iCassava 2019 (página: https://www.kaggle.com/c/cassava-disease/overview) | Código fuente de TFDS |
| Referencia | Mwebaze et al. (2019), *iCassava 2019 Fine-Grained Visual Categorization Challenge*, arXiv:1908.02900 | Código fuente de TFDS |
| Licencia | **[por completar]** – el equipo debe consultarla en la página del catálogo y en las reglas de la competencia | |
| Procedencia de las fotos | **[por completar]** – leer el artículo (sección de datos) y citarlo | |
| Imágenes etiquetadas | 9.430 publicadas · encontradas: **[por completar]** | notebook 01, paso 6 |
| Particiones | train 5.656 · validación 1.889 · prueba 1.885 publicadas · encontradas: **[por completar]** | notebook 01, paso 6 |
| Clases | 5: `cbb`, `cbsd`, `cgm`, `cmd`, `healthy` | Código fuente de TFDS |
| Distribución por clase | **[por completar]** (pegar `resultados/distribucion_clases.csv`) | notebook 01, paso 5 |
| Resolución | **[por completar]** | notebook 01, paso 7 |
| Archivos ilegibles | **[por completar]** | notebook 01, paso 4 |
| Duplicados | **[por completar]** (días 3–5) | notebook 02 |
| Criterios de exclusión | **[por completar]** – decisión del equipo según lo que aparezca | |
| Identificadores de planta/finca | **[por completar]** – revisar si el dataset los trae; si no, declararlo como limitación | |

## Significado de las clases

| Sigla | Nombre |
|---|---|
| cbb | Tizón bacteriano de la yuca (Cassava Bacterial Blight) |
| cbsd | Enfermedad del estriado marrón (Cassava Brown Streak Disease) |
| cgm | Ácaro verde de la yuca (Cassava Green Mite) |
| cmd | Enfermedad del mosaico (Cassava Mosaic Disease) |
| healthy | Hoja sana |

Las etiquetas son categorías de referencia para un ejercicio académico, no una validación agronómica.
