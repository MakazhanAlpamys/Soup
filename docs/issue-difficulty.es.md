# Dificultad de los issues

> 🇪🇸 Traducción al español de [`issue-difficulty.md`](issue-difficulty.md). Ante cualquier duda, manda el original en inglés.

Para ayudarte a elegir un issue que se ajuste a tu tiempo y tu experiencia, cada issue triado lleva una
etiqueta de `difficulty:1` a `difficulty:10`.

Estima cuánto trabajo y riesgo conlleva la corrección. **No** es una afirmación sobre lo
importante que es el bug.

Una etiqueta de dificultad también significa que un mantenedor revisó el issue y lo confirmó. Un issue
sin etiqueta de dificultad aún no está triado. Los duplicados, los trackers, las preguntas y los issues
cerrados como no planificados no llevan ninguna.

Los issues cerrados antes de que existiera este sistema se calificaron después, a partir del pull request que
los cerró, así que esas etiquetas antiguas son aproximadas. Una etiqueta es una estimación: cuando una etiqueta y
la tabla de abajo no coinciden, la tabla es la definición y la etiqueta se corrige.

## Qué significan los números

| Nivel | Cómo se ve | Ejemplo |
|---|---|---|
| 1 | Redacción, un error de tipeo o una corrección de una línea. Sin lógica nueva. | #1611, #1642 |
| 2 | Un archivo que no es de prueba, menos de unas 15 líneas cambiadas; el issue detalla la corrección; una prueba. | #1641 |
| 3 | Un archivo que no es de prueba, hasta unas 50 líneas cambiadas; las pruebas corren en CPU sin torch. | #1598 |
| 4 | De dos a cuatro archivos fuente en un subsistema, siguiendo un patrón que ya existe; las pruebas corren en CPU sin torch. | #1596 |
| 5 | Cinco o más archivos fuente, o un trainer, o el esquema de configuración; las pruebas necesitan torch. | #1345 |
| 6 | Varias tareas, trainers o backends a la vez, o cálculos numéricos no triviales; cada uno afectado necesita su propia prueba de regresión. | #805 |
| 7 | Un mecanismo nuevo o un cambio profundo en una ruta central (interiores del trainer, layer streaming, el pipeline de datos); demostrarlo exige una medición. | #1425 |
| 8 | La causa aún no se conoce o es difícil de reproducir, o el resultado debe ser bit-exact. | #342 |
| 9 | Un subsistema nuevo, o compatibilidad con varias familias de modelos a la vez. | #265 |
| 10 | Nivel de investigación: kernels escritos a mano o backward passes con una compuerta numérica. Raro; la mayoría del trabajo difícil se detiene en 8 o 9. | ninguno abierto ahora mismo |

## Hardware especial

Si la corrección no se puede verificar sin hardware especial o un servicio pago (una GPU grande, varias
GPU, una nube paga), suma 3 niveles a lo que obtendría el trabajo en sí, hasta 10. Un cambio de un módulo
que solo se puede comprobar en una H100 es nivel 6, no nivel 9. Dilo cuando reclames el issue, como
[CONTRIBUTING.md](../CONTRIBUTING.md) pide.

¿No estás de acuerdo con un número? Dilo en el hilo del issue y explica el motivo; lo revisaremos de nuevo.

## Abrir un issue

El formulario de issue tiene un campo opcional de "Dificultad sugerida". Tu conjetura ayuda, y un mantenedor
define la etiqueta final. Busca primero, por favor, y limita a un defecto por issue.
