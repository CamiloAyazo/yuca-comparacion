# Análisis de errores reales

> Lo redacta el equipo mirando las galerías del notebook 05. Mínimo 8 errores.
> Criterio de selección usado: los errores de mayor confianza en el conjunto de prueba.

| # | Modelo | Archivo | Real | Predicha | Confianza | Qué se ve en la foto | Hipótesis del error (síntoma parecido, fondo, luz, calidad, lesión pequeña, etiqueta dudosa) |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |

## Patrones que observamos
**E1 (2026-11-15).** La confusión más frecuente es cbsd → cbb (92 de 481 fotos de cbsd). En los 8 errores de mayor confianza,
6 fueron predichos como cbb y casi todos son hojas muy amarillas o marchitas con puntos oscuros. Dos de ellos
(`test-cgm-136`, `test-cgm-102`) son fotos de prueba cuya copia idéntica tenía otra etiqueta.

Hipótesis del equipo (Camilo Ayazo):
- **Síntomas parecidos:** en etapas avanzadas, cbb y cbsd producen amarillamiento y necrosis similares.
- **Posible atajo visual:** el modelo pudo aprender que "hoja amarilla y seca" significa cbb, sin mirar el tipo de lesión.
- **Etiquetas dudosas:** hay fotos idénticas etiquetadas como cbb, cbsd y cgm; esa confusión de etiquetas pasa al modelo.

(Se comparará con E3 cuando esté ejecutado.)

## Grad-CAM (4 imágenes)
(¿Dónde está el calor: lesión, hoja completa, fondo? Recordar: no demuestra causalidad.)

## Posibles atajos visuales
