# Hoja de defensa · Camilo De Jesús Ayazo Argel · Control del día 7

Todo lo de esta hoja sale de lo que hicimos y de lo que yo decidí (bitácora, protocolo e hipótesis).
Leerla en voz alta dos o tres veces y luego decirlo con mis palabras, sin leer.

---

## 1. Mi explicación individual (unos 2 minutos)

**Qué hice**
- Creé el repositorio en GitHub y subí la estructura del proyecto con mi cuenta.
- Descargué el dataset Cassava de TensorFlow Datasets e hice el inventario: 9.430 fotos, cinco clases. Las particiones coinciden con lo publicado (5.656 / 1.889 / 1.885).
- Revisé los datos y los duplicados (notebook 02), cerré el protocolo, escribí las hipótesis y ejecuté E0 y E1 (notebook 03).

**Decisiones que tomé y por qué**
- **Colab:** no tenemos experiencia instalando bibliotecas, Colab trae todo listo y da una GPU que sí sirve para PyTorch. Mi portátil tiene gráficos integrados de Intel, no una GPU NVIDIA.
- **Dataset de TensorFlow Datasets:** por su tamaño manejable para 15 días.
- **Duplicados:** encontré la misma foto guardada con etiquetas distintas (por ejemplo, como cbb, cbsd y cgm a la vez). Decidí retirar todas esas copias de entrenamiento y validación, para que el modelo no aprenda de etiquetas dudosas. También retiré los casi duplicados entre particiones para evitar fuga de información. La prueba no la toqué. Quedaron 5.600 / 1.877 / 1.885.
- **ResNet18 como Backbone A:** el ajuste fino de E2 se hace sobre la red con más capacidad para adaptarse.
- **Pesos por clase:** para que los errores en las clases escasas (cbb, healthy) pesen más.
- **Macro-F1 como métrica principal:** el 47 % de las fotos son cmd; un modelo que diga siempre "cmd" tendría buen accuracy sin haber aprendido nada.

**Dificultad que resolví**
- El notebook falló con `No module named 'src'`. Listé la carpeta desde Colab y vi que el proyecto había quedado en una carpeta dentro de otra en Drive. Moví el contenido un nivel arriba y funcionó. Aprendí a revisar la estructura de carpetas y que las rutas importan.

**Resultados y lo que pienso de ellos**
- E0 (red pequeña desde cero, 64×64): macro-F1 **0,464**. Yo esperaba más de 0,55; creo que a 64×64 se pierden las manchas pequeñas.
- E1 (ResNet18 congelada): macro-F1 **0,659**, es decir, +0,195 sobre E0. Esperaba +0,20: se cumplió en esencia.
- La clase más difícil fue **cbb** (F1 0,414), no cbsd como pensaba. En los errores, el modelo dice "cbb" en hojas muy amarillas o marchitas. Creo que se combinan síntomas parecidos entre cbb y cbsd, un posible atajo de "hoja amarilla = cbb" y las etiquetas dudosas que encontramos.

---

## 2. Demostración en vivo (notebook 03, sección final)

Antes de entrar: abrir el notebook 03 en Colab con la GPU T4 y ejecutar la **primera celda** (aceptar Drive). Tarda 1–2 minutos.
Si piden algo que necesite `m1`, primero ejecutar:
```python
import json; m1 = json.load(open('resultados/E1/metricas.json'))
```

| Me piden | Qué hago | Qué digo |
|---|---|---|
| Cargar datos | Ejecutar la celda **A** | "Es un lote de 8 fotos, cada una con 3 canales de color y 224×224 píxeles." |
| Hacer una inferencia | Cambiar `POSICION` en la celda **B** y ejecutar | "Muestra la foto, su clase real y la probabilidad de cada clase." |
| Recalcular macro-F1 | Ejecutar la celda **C** | "Para cada clase: precisión = aciertos / todo lo que el modelo llamó así; recall = aciertos / todo lo que realmente era. F1 combina ambas, y macro-F1 es el promedio simple de las cinco. A mano da 0,659, igual que scikit-learn." |
| Cambiar una transformación | Cambiar `FACTOR` en la celda **D** (por ejemplo 0.5 o 1.5) | "Oscurezco o aclaro la misma foto y veo si el modelo cambia de opinión." |
| Explicar un error | Abrir `resultados/E1/galeria_errores.png` | Lo de cbb de arriba. |

---

## 3. Preguntas probables (respuesta corta)

- **¿Qué es congelar el backbone?** Sus pesos no se modifican (`requires_grad = False`); solo aprende la cabeza nueva de 5 salidas. En E1 son 2.565 parámetros entrenables de 11,2 millones.
- **¿Para qué validación si hay prueba?** Con validación elegimos la mejor época. La prueba se usa una sola vez al final; si decidiéramos con ella, el resultado saldría inflado.
- **¿Qué es fuga de información?** Que una foto de prueba (o casi igual) también esté en entrenamiento: el modelo la "conoce" y el resultado sale mejor de lo real.
- **¿Por qué fijar la semilla?** Para que el barajado y los pesos iniciales sean los mismos y otra persona obtenga el mismo resultado.
- **¿Por qué esos números en la normalización?** Son la media y la desviación de ImageNet, con las que se preentrenó la red.
- **¿Sirve en Colombia?** No se puede afirmar: las fotos son de Uganda; habría que probar con fotos de cultivos colombianos.
- **¿Qué hizo la IA?** Lo que dice `docs/declaracion_uso_ia.md`: escribió el código, ejecutó los notebooks y me presentó opciones; yo tomé las decisiones y elegí las interpretaciones.
