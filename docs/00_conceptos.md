# Conceptos explicados desde cero

Todos los ejemplos usan las fotos de hojas de yuca del proyecto. Después de cada concepto se indica **dónde aparece en nuestro código**.

## 1. Machine Learning (aprendizaje automático)
En lugar de escribir reglas a mano ("si la hoja tiene manchas amarillas en mosaico, es CMD"), se le muestran al computador miles de fotos ya etiquetadas y él ajusta sus propios números internos hasta acertar. Esos números se llaman **pesos** o **parámetros**.

## 2. Deep Learning (aprendizaje profundo)
Es Machine Learning con **redes neuronales de muchas capas**. Cada capa transforma un poco la información: las primeras detectan bordes y colores, las intermedias texturas y manchas, las últimas combinaciones complejas. "Profundo" solo significa "muchas capas".

## 3. CNN (red neuronal convolucional)
Tipo de red pensada para imágenes. Usa **filtros** pequeños (por ejemplo de 3×3 píxeles) que recorren la foto buscando un patrón; el mismo filtro sirve en cualquier parte de la imagen, así que una mancha se detecta igual en la esquina o en el centro.
*En el código:* `CNNPequena` en `src/modelos.py` (tres bloques de convolución).

## 4. Cómo procesa una imagen la red
1. La foto se redimensiona (224×224) y se convierte en números: 3 canales (rojo, verde, azul), cada píxel entre 0 y 1.
2. Se **normaliza**: se resta una media y se divide por una desviación, las mismas que se usaron cuando la red fue preentrenada.
3. Pasa por las capas convolucionales, que producen un vector de características (1.280 números en MobileNetV2, 512 en ResNet18).
4. La **cabeza clasificadora** convierte ese vector en 5 números, uno por clase; `softmax` los transforma en probabilidades que suman 1.
5. La clase con mayor probabilidad es la predicción.
*En el código:* `transformaciones()` en `src/dataset.py`; `inferencia_una_imagen()` en `src/analisis.py`.

## 5. Entrenamiento, validación y prueba
- **Entrenamiento (5.656 fotos):** las que el modelo ve para ajustar sus pesos.
- **Validación (1.889):** no se usan para ajustar pesos; sirven para decidir cosas (cuál época es la mejor).
- **Prueba (1.885):** se usan **una sola vez al final** para reportar el resultado. Si se usaran para decidir, el resultado saldría inflado.
Analogía: ejercicios de clase, simulacro y examen final.

**Época:** una pasada completa por todas las fotos de entrenamiento. **Lote (batch):** grupo de fotos (32) que se procesan juntas. **Pérdida (loss):** número que mide cuánto se equivocó el modelo; entrenar es reducirlo. **Tasa de aprendizaje:** tamaño del paso con que se corrigen los pesos.

## 6. Transfer Learning (aprendizaje por transferencia)
En vez de empezar de cero, se toma una red ya entrenada con 1,2 millones de fotos generales (ImageNet) y se reutiliza lo que aprendió (bordes, texturas, formas) para nuestro problema. Sirve porque tenemos pocas fotos de yuca para enseñar todo eso desde cero.

## 7. Feature Extraction (extracción de características)
El backbone preentrenado se **congela**: sus pesos no cambian. Solo se entrena la cabeza nueva de 5 salidas. Es rápido y barato. Son los experimentos **E1 y E3**.
*En el código:* `congelar_backbone()` pone `requires_grad = False` a todo menos la cabeza.

## 8. Fine-Tuning (ajuste fino)
Después de entrenar la cabeza, se **descongela** una parte final del backbone para que también se adapte a las hojas, con una tasa de aprendizaje más pequeña (para no destruir lo aprendido). Es el experimento **E2**. Entrena más parámetros, así que cuesta más y puede sobreajustar.
*En el código:* `descongelar_bloque_final()`.

## 9. MobileNetV2 y ResNet18
- **MobileNetV2:** diseñada para celulares; usa convoluciones "separables" que hacen el mismo trabajo con muchos menos cálculos. En nuestro proyecto tiene ≈2,2 millones de parámetros (≈9 MB).
- **ResNet18:** 18 capas con "conexiones residuales" (atajos que suman la entrada de un bloque a su salida, lo que facilita entrenar redes profundas). ≈11,2 millones de parámetros (≈45 MB).
Las dos fueron preentrenadas con ImageNet y están en la misma biblioteca (torchvision). **Backbone** = la red sin su capa final.

## 10. Baseline
Modelo sencillo que sirve de punto de referencia. Si una red preentrenada no supera con claridad al baseline, la complejidad no se justifica. Es el experimento **E0**.

## 11. Data Augmentation (aumento de datos)
Crear variaciones de las fotos de entrenamiento (recortes, giros, volteos) para que el modelo no memorice detalles irrelevantes. Solo se aplica al entrenamiento. Debe **preservar el síntoma**: no cambiamos el tono de color porque el amarillo de una mancha es parte de la enfermedad.

## 12. Accuracy (exactitud)
Porcentaje de fotos bien clasificadas. Engaña con clases desbalanceadas: si una clase tuviera casi la mitad de las fotos, un modelo que casi siempre respondiera esa clase tendría un accuracy aceptable sin reconocer las demás.

## 13. Precision (precisión)
De todas las fotos que el modelo llamó "CBB", ¿qué fracción lo era realmente? Baja precisión = muchas falsas alarmas.

## 14. Recall (sensibilidad)
De todas las fotos que realmente eran "CBB", ¿qué fracción encontró el modelo? Bajo recall = se le escapan muchos casos.

## 15. F1-Score
Combina precisión y recall en un solo número: F1 = 2 · P · R / (P + R). Solo es alto si los dos son altos.

## 16. Macro-F1
Promedio simple del F1 de las 5 clases. Cada clase pesa igual sin importar cuántas fotos tenga; por eso es la métrica principal: obliga a rendir bien también en las clases escasas.
*En el código:* `macro_f1_manual()` en `src/evaluar.py` lo calcula paso a paso.

## 17. Matriz de confusión
Tabla de 5×5: filas = clase real, columnas = clase predicha. La diagonal son aciertos; fuera de ella se ve qué clase se confunde con cuál.

## 18. Sobreajuste (overfitting)
El modelo memoriza las fotos de entrenamiento y falla con fotos nuevas. Se ve en las curvas: la pérdida de entrenamiento sigue bajando mientras la de validación se estanca o sube.

## 19. Generalización
Capacidad de acertar con fotos que nunca vio. La prueba mide generalización dentro del mismo dataset. **Validez externa:** estas fotos provienen de un origen concreto; no garantizan el mismo desempeño en cultivos colombianos (otras variedades, cámaras, luz, enfermedades).

## 20. Grad-CAM
Técnica que pinta un mapa de calor sobre la foto mostrando qué zonas influyeron más en la respuesta. Ayuda a sospechar atajos (por ejemplo, si el calor está en el suelo y no en la hoja). **No demuestra causalidad** ni que el modelo haya reconocido la enfermedad.
*En el código:* `gradcam_galeria()` en `src/analisis.py`.
