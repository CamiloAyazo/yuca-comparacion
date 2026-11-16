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
| Imágenes etiquetadas | 9.430 publicadas · encontradas: 9.430 | notebook 01 (2026-11-15) |
| Particiones | train 5.656 · validación 1.889 · prueba 1.885 publicadas · encontradas: iguales | notebook 01 (2026-11-15) |
| Clases | 5: `cbb`, `cbsd`, `cgm`, `cmd`, `healthy` | Código fuente de TFDS |
| Distribución por clase | cmd 4.431 (47,0 %) · cbsd 2.406 (25,5 %) · cgm 1.289 (13,7 %) · cbb 777 (8,2 %) · healthy 527 (5,6 %); misma proporción en las tres particiones | notebook 01 |
| Resolución | Variable. Ancho 420–1.622 px (mediana 543), alto 347–1.240 px (mediana 500); las más comunes 500×500, 666×500 y 500×666 | notebook 02 |
| Archivos ilegibles | Ninguno | notebook 01 |
| Duplicados | 68 archivos en 26 grupos idénticos; en 18 grupos la misma foto tiene etiquetas distintas (sobre todo cbb, cbsd y cgm); 17 grupos cruzan particiones. Además, 29 pares casi idénticos, 20 entre particiones | notebook 02 |
| Criterios de exclusión | Ver protocolo D11: se retiran 68 imágenes de entrenamiento/validación; la prueba no se toca | notebook 02 |
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

## Hallazgo del control de duplicados
Hay fotos idénticas byte a byte guardadas con etiquetas distintas (por ejemplo, el mismo archivo como `cbb`, `cbsd` y `cgm`).
Esto indica errores de etiquetado en el conjunto publicado. En la prueba quedan unas 10 fotos así, que no se retiran
para conservar la partición publicada; se declara como limitación.
