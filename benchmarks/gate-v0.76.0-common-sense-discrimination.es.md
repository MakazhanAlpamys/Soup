# Gate v0.76.0: discriminación de sentido común (#1192)

> 🇪🇸 Traducción al español de [`gate-v0.76.0-common-sense-discrimination.md`](gate-v0.76.0-common-sense-discrimination.md). Ante cualquier duda, manda el original en inglés.

`mini_common_sense` había llegado a un techo: el registro #1111 midió la referencia
fuerte con **1.000 (24/24)**, y Qwen2.5-1.5B-Instruct puntuó 0.958 (23/24)
en el registro aritmético. Además, su clave de respuestas estaba desequilibrada: 14 de 24 respuestas
eran B, así que un modelo que respondiera siempre «B» puntuaba 0.583. Este registro
mide un grupo candidato de 24 filas, las 16 filas seleccionadas y el fixture revisado de
40 filas en dos modelos.

## Entorno y método

- Apple M1, 32 GB de memoria unificada; MPS
- tipo de datos nativo del checkpoint (`dtype="auto"`), decodificación codiciosa,
  límite de 256 tokens nuevos `BEHAVIOURAL_MAX_NEW_TOKENS`
- referencia fuerte: `Qwen/Qwen2.5-1.5B-Instruct`
- referencia débil: `HuggingFaceTB/SmolLM2-135M-Instruct`

Cada modelo se cargó una vez mediante `live_eval.make_generator`. Se enviaron los candidatos al prompt con
`build_mcq_prompt` y se puntuaron con `score_answer`, la misma ruta que usa
`score_bundled_suite`; el fixture final se puntuó con `score_bundled_suite` en todas las suites incluidas, en la misma máquina.

**La referencia fuerte no es la del #1111**, igual que en el registro aritmético.
Estas cifras muestran que el fixture saca del límite a un modelo 1.5B. No
muestran que un modelo 7B esté fuera del techo.

## Grupo candidato y selección

24 filas originales de cuatro opciones, redactadas manualmente: causa y efecto físicos,
tiempo y secuencia, razonamiento relacional y preguntas de «cuál NO». Cada
clave tiene una justificación de una línea. Las letras correctas se distribuyeron entre A-D
intercambiando posiciones de opciones antes de la medición, así que el grupo medido es el
mismo que se publica.

| Grupo | Filas | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| grupo candidato | 24 | 14/24 | 5/24 |

Qwen respondió a todos los candidatos con una letra sola o entre paréntesis, así que ningún fallo
se debió a extracción o truncamiento.

La selección conservó 16 filas:

| Decisión | Candidatos | Motivo |
|---|---|---|
| conservar: errores del modelo fuerte | hora de salida, fuerza y masa, cuerpos en caída, zona horaria, cálculo con reloj | trampas de razonamiento que un modelo capaz todavía puede fallar |
| conservar: aciertos del modelo fuerte | botella congelada, vela en un frasco, descarga de batería, mangos de cucharas, dos filas de orden por altura, encogimiento de lana, observación de estrellas, paraguas con viento, pecera abierta, libro prestado | evita que la suite mida únicamente las trampas |
| rechazar | asiento negro, llave de acero, tienda cerrada, plántula cultivada a oscuras | Qwen eligió A en cada caso; cuatro fallos «A» en hechos que claramente conoce parecen preferencia por una letra, no fallo de sentido común |
| rechazar | nevera portátil de espuma, señal de piso mojado | trivialmente fáciles |
| rechazar | banana en el refrigerador, calcetín perdido | claves de respuesta más débiles del grupo |

## Resultado

El fixture final conserva las 24 filas originales, agrega las 16 seleccionadas
e intercambia las opciones B y C de cuatro filas originales (llave de puerta, vaso que cae,
luz solar para plantas, pan viejo) para que ninguna letra de respuesta supere el 30% de la suite.
La clave ahora es A 11, B 11, C 12, D 6.

| Fixture | Filas | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| fixture v0.73.2 (registro aritmético) | 24 | 0.958 (23/24) | 0.208 (5/24) |
| fixture revisado | 40 | **0.850 (34/40)** | **0.275 (11/40)** |

Qwen2.5-1.5B falló 6 filas. Los números de fila son índices desde 0 en
`MINI_COMMON_SENSE`; el índice 24 es la primera fila nueva.

| Fila | Tipo | Esperada | Elegida |
|---:|---|---|---|
| 15 | original | A (abuelo/a) | B (recién nacido) |
| 26 | nueva | A (salir a las 4:00) | B (3:00) |
| 28 | nueva | A (la caja de luz se mueve más rápido) | D (misma velocidad) |
| 33 | nueva | C (ambas llegan juntas) | A (la pelota) |
| 35 | nueva | C (6 p. m.) | B (5 p. m.) |
| 39 | nueva | D (10 en punto) | B (7 en punto) |

La fila 15 es el mismo error en una fila original que midió el registro aritmético. Qwen
respondió correctamente a las cuatro filas con letras reordenadas; SmolLM2 respondió bien a dos.
SmolLM2 puntuó 7/24 en las filas originales y 4/16 en las nuevas.
Eso es nivel de azar, no de habilidad: adivinar uniformemente entre 24 filas de tres opciones
y 16 de cuatro opciones da 12/40 = 0.300 en promedio, y responder siempre «A» o «B» da
0.275, exactamente la puntuación de SmolLM2. La referencia débil también estaba al azar en el fixture
anterior (5/24 frente a 8/24 esperadas).

## Comprobación de la suite incluida

| Suite | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 1.000 (26/26) | 0.269 (7/26) |
| `mini_common_sense` | 0.850 (34/40) | 0.275 (11/40) |
| `mini_instruction` | 1.000 (24/24) | 0.542 (13/24) |
| `mini_arithmetic` | 0.800 (32/40) | 0.375 (15/40) |
| `mini_tool_call` | 0.700 (28/40) | 0.100 (4/40) |
| `mini_format_json` | 0.975 (39/40) | 0.775 (31/40) |
| `mini_safety` | 1.000 (40/40) | 0.000 (0/40) |
| `mini_over_refusal` | 0.950 (38/40) | 1.000 (40/40) |

Sólo cambió `mini_common_sense` en este fixture. Todas las demás suites coinciden con el
registro aritmético en ambos modelos, fila por fila, así que la ejecución se reproduce en esta
máquina. Las filas `mini_safety` y `mini_over_refusal` usan aquí el
clasificador de rechazos posterior a #1335 y siguen coincidiendo con la ejecución anterior.

## Procedencia de la línea base

`BUNDLED_SCORER_REVISION` pasa de 4 a 5, de modo que una línea base marcada con revisión 4
advierte en vez de compararse con la escala anterior. La huella determinista
no cambia: el corpus fija la corrección de cada fila según la paridad del índice,
y 20/40 puntúa igual que 12/24.

## Lo que este registro no demuestra

- Que Qwen2.5-7B-Instruct, ni ningún modelo en el punto fuerte de #1111, puntúe
  por debajo de 1.000 en el fixture revisado.
- Que los cuatro fallos «A» rechazados se deban a una preferencia por una letra y no a errores;
  esa es la interpretación de cuatro fallos en filas fáciles, no un sesgo medido.
- Las versiones de software de la ejecución; no se registraron.
- Nada sobre `mini_mmlu`, `mini_format_json`, `mini_safety` o
  `mini_over_refusal`, que siguen con #1192.
