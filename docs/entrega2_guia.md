# Entrega 2 · Guía para el proyecto final

## 1. Lista de entregables y dónde sale cada uno

| # | Entregable | Dónde queda | Quién lo produce |
|---|---|---|---|
| 1 | Informe (máx. 10 páginas) en Word y PDF | `informe/` | Se arma con sus resultados e interpretaciones |
| 2 | Código reproducible | `src/`, `notebooks/` | Listo |
| 3 | Archivo de dependencias | `requirements.txt` | Congelar versiones: `!pip freeze > requirements_colab.txt` |
| 4 | Instrucciones de ejecución | `README.md` | Listo |
| 5 | Semillas y particiones | `configs/config.yaml`, `resultados/inventario.csv`, `excluidos.csv` | notebooks 01–02 |
| 6 | Pesos / checkpoints | `checkpoints/*.pt` (Drive; compartir enlace) | notebooks 03–04 |
| 7 | Tabla consolidada E0–E3 | `resultados/tabla_consolidada.csv` | notebook 04 |
| 8 | Matrices de confusión | `resultados/E*/matriz_confusion.png` | automático |
| 9 | Métricas por clase | `resultados/f1_por_clase.csv`, `recall_por_clase.csv` | notebook 04 |
| 10 | Registro de recursos | columnas de tiempo, tamaño y memoria en la tabla consolidada | automático |
| 11 | Experimento adicional | `resultados/adicional_brillo/` (o `E1_aug/`) | notebook 05 |
| 12 | Bitácora cronológica | `bitacora/` | El equipo |
| 13 | Declaración de herramientas | `docs/declaracion_uso_ia.md` | El equipo (se actualiza en cada uso) |

## 2. Orden de trabajo

1. **Días 8–10:** escribir hipótesis de E2 y E3 → notebook 04 (E3, E2, tabla, estabilidad opcional).
2. **Días 11–13:** decidir experimento adicional y escribir su hipótesis → notebook 05 → llenar `docs/04_analisis_errores.md` → conclusiones provisionales.
3. **Días 14–15:** informe, revisión de reproducibilidad, ensayo de sustentación.

## 3. Estructura del informe (10 páginas)

| Sección | Páginas | Contenido |
|---|---|---|
| 1. Problema y pregunta | 0,5 | Pregunta de investigación y alcance |
| 2. Datos | 1 | Ficha, distribución, duplicados, exclusiones, límites |
| 3. Hipótesis previas | 0,5 | Tal como se escribieron, con fecha |
| 4. Protocolo | 1 | Condiciones fijas, métrica, presupuesto, qué se compara con qué |
| 5. Experimentos | 1,5 | E0–E3 y adicional: diseño de cada uno |
| 6. Resultados | 2,5 | Tabla consolidada, métricas por clase, matrices, curvas, costo |
| 7. Análisis de errores | 1,5 | 8 errores, confusiones, Grad-CAM, atajos posibles |
| 8. Límites y validez externa | 0,5 | Lo que no se puede concluir |
| 9. Conclusiones | 0,5 | Respuesta a la pregunta, contrastada con las hipótesis |
| Anexos | — | Bitácora, declaración de IA, figuras extra |

## 4. Cómo leer cada resultado (preguntas guía; las respuestas son del equipo)

- **E0 vs E1:** ¿cuánto sube macro-F1?, ¿en qué clases sube más?, ¿cuánto más cuesta?
- **E1 vs E3 (primaria):** ¿la diferencia de macro-F1 es mayor que la variación entre semillas?, ¿cuál es más rápido en inferencia?, ¿cuál pesa menos?, ¿ganan en las mismas clases?
- **E1 vs E2 (secundaria):** ¿mejoró?, ¿en qué clases?, ¿cuánto tiempo extra?, ¿las curvas muestran más distancia entre entrenamiento y validación?
- **Adicional:** ¿la caída con brillo bajo es igual que con brillo alto?, ¿un modelo resiste mejor?, ¿qué clase se pierde primero?
- **Cuidado:** no comparar E2 con E3 para hablar de arquitecturas: difieren en arquitectura **y** en entrenamiento.

## 5. Estructura de la sustentación final (≈12–15 min)

| Bloque | Quién | Tiempo |
|---|---|---|
| Problema, pregunta, objetivos | Camilo Ayazo | 1,5 min |
| Datos, duplicados, protocolo | Camilo Ayazo | 2 min |
| E0, E1, E2: qué es cada uno y qué dio | Camilo Mangones | 3 min |
| E3 y comparación de arquitecturas | Camilo Mangones | 2 min |
| Métricas por clase y matrices | Hernan Herrera | 2 min |
| Errores, Grad-CAM, experimento adicional | Hernan Herrera | 2 min |
| Costo, límites, conclusiones | Los tres | 2 min |

Los guiones con cifras se redactan cuando existan los resultados reales.

## 6. Preguntas difíciles y la idea de la respuesta

1. *¿Por qué no pueden decir que una arquitectura es "mejor" con una sola ejecución?* — Una diferencia pequeña puede ser azar de la semilla; por eso se repite con otras semillas y se compara con esa variación.
2. *¿Por qué E1 vs E3 es la comparación primaria y no E2 vs E3?* — Porque E1 y E3 solo difieren en la arquitectura. E2 además cambia qué capas se entrenan y la tasa.
3. *¿Qué descongelaron exactamente en E2?* — `features[17:]` de MobileNetV2: el último bloque residual invertido y la convolución final (≈890 mil parámetros).
4. *¿Por qué una tasa de aprendizaje menor en E2?* — Los pesos preentrenados ya son buenos; pasos grandes los dañarían.
5. *¿E2 partió de cero?* — No: de los pesos del mejor E1. Por eso su costo real es E1 + E2.
6. *¿Usaron el conjunto de prueba para decidir algo?* — No: la mejor época se elige con validación; la prueba se evalúa una vez por experimento.
7. *El experimento de brillo usa la prueba varias veces, ¿no es trampa?* — No se decide nada con esos resultados: no se elige modelo ni se ajusta nada; solo se mide robustez de modelos ya fijados.
8. *¿Por qué el augmentation no cambia el color?* — El color es parte del síntoma; alterarlo podría convertir una hoja enferma en una que parece sana.
9. *¿Qué muestra y qué no muestra Grad-CAM?* — Muestra zonas con mayor influencia en la salida de esa clase; no prueba que esa sea la razón ni que el modelo reconozca la enfermedad.
10. *¿Qué es un atajo visual?* — Una pista que coincide con la etiqueta sin ser el síntoma (fondo, brillo, encuadre). Indicios: Grad-CAM fuera de la hoja, caída fuerte al cambiar el brillo, diferencias de brillo por clase en el EDA.
11. *¿El tiempo de inferencia depende del computador?* — Sí; por eso se reporta el dispositivo y se comparan los modelos en la misma máquina.
12. *¿Por qué ResNet18 pesa cinco veces más que MobileNetV2?* — Tiene ≈11,2 M de parámetros frente a ≈2,2 M; MobileNetV2 usa convoluciones separables.
13. *¿Serviría en Colombia?* — No se puede afirmar; habría que evaluar con fotos locales (otras variedades, cámaras, condiciones).
14. *¿Qué harían con más tiempo?* — Respuesta del equipo (ideas: evaluar con el dataset de India, más semillas, fotos propias).
15. *¿Qué hizo la IA y qué hicieron ustedes?* — Responder con la declaración: exactamente lo que dice.

## 7. Simulacros individuales
Cuando tengan resultados, pidan en el chat: "simulacro para [nombre]". Se hace una pregunta a la vez, se espera la respuesta y se comenta qué estuvo bien y qué faltó.
