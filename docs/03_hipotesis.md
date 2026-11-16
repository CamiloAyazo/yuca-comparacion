# Hipótesis previas

> Las redactan los integrantes **antes** de ejecutar cada experimento, con fecha. No se editan después de ver
> los resultados: si una hipótesis falla, se comenta en el informe (eso también suma).

Formato sugerido: *"Creemos que ___ porque ___. Si es cierto, esperamos ver ___."*

| Exp. | Hipótesis | Predicción concreta (qué número o comportamiento esperamos) | Autor | Fecha |
|---|---|---|---|---|
| E0 | Aun entrenada desde cero, una red pequeña con 5.600 fotos alcanza a aprender los patrones más visibles de las enfermedades. | Macro-F1 en prueba mayor que 0,55. | Camilo Ayazo | 2026-11-15 |
| E1 | Lo aprendido por ResNet18 en ImageNet (bordes, texturas, manchas) sirve directamente para ver lesiones en hojas, aunque solo se entrene la cabeza. | Macro-F1 al menos 0,20 por encima de E0 (es decir, mayor que 0,75 si E0 cumple su predicción). La clase más difícil (F1 más bajo) será cbsd, porque sus manchas amarillas difusas se parecen al mosaico y a hojas sanas. | Camilo Ayazo | 2026-11-15 |
| Pesos por clase (E0–E3) | Dar más peso en la pérdida a las clases escasas mejora su detección. | El recall de cbb y healthy será mayor que sin pesos, aunque cmd pierda un poco. | Camilo Ayazo | 2026-11-15 |
| E2 | | | | |
| E3 | | | | |
| Adicional | | | | |

Preguntas para pensar (no son respuestas):
- ¿Qué clases creen que serán más difíciles y por qué? (miren la distribución del inventario y algunas fotos)
- ¿Cuánto esperan que mejore E1 sobre E0? ¿Por qué una red entrenada con fotos generales serviría para hojas?
- ¿Descongelar un bloque puede empeorar algo? ¿Qué costo tiene?
- ¿Cuál de las dos arquitecturas esperan que sea más rápida? ¿Y más precisa?

> Nota: la hipótesis sobre los pesos por clase solo se puede comprobar del todo con una ejecución sin pesos, que no está en el plan.
> Si no se hace, se analizará con el recall por clase de E1 y se declarará el límite.

## Contraste con los resultados (2026-11-15)

| Hipótesis | Resultado | ¿Se cumplió? | Lectura del equipo |
|---|---|---|---|
| E0: macro-F1 > 0,55 | 0,464 | No | **[por completar]** |
| E1: al menos +0,20 sobre E0 | +0,195 (0,659 frente a 0,464) | Por 0,005 no alcanzó el umbral | Se cumplió en esencia: la transferencia mejoró mucho y 0,005 no cambia la conclusión; se reporta el número tal cual. |
| E1: cbsd será la clase más difícil | La más difícil fue cbb (F1 0,414); cbsd quedó en 0,692 | No | Ver análisis de errores. |
| Pesos por clase suben el recall de cbb y healthy | Recall E1: cbb 0,484, healthy 0,705; precisión de cbb solo 0,362 | No comprobable sin ejecución sin pesos | Posible efecto adverso: los pesos pudieron empujar al modelo a predecir cbb de más. No se puede confirmar sin una ejecución sin pesos. |
