# Gate v0.76.0: discriminación aritmética (#1192)

> 🇪🇸 Traducción al español de [`gate-v0.76.0-arithmetic-discrimination.md`](gate-v0.76.0-arithmetic-discrimination.md). Ante cualquier duda, manda el original en inglés.

`mini_arithmetic` había llegado a un techo: el registro #1111 midió la referencia fuerte
con **1.000 (36/36)** en el fixture publicado. Este registro mide el fixture revisado de 40 filas en dos modelos, ninguno de los cuales está en un límite.

## Entorno y método

- Apple M1, 32 GB de memoria unificada; macOS
- tipo de datos nativo del checkpoint (`dtype="auto"`), decodificación codiciosa,
  límite de 256 tokens nuevos `BEHAVIOURAL_MAX_NEW_TOKENS`
- referencia fuerte: `Qwen/Qwen2.5-1.5B-Instruct`
- referencia débil: `HuggingFaceTB/SmolLM2-135M-Instruct`

Cada modelo se cargó una vez mediante `live_eval.make_generator` y se puntuó con
`score_bundled_suite` en todas las suites incluidas, en la misma máquina.

**La referencia fuerte no es la del #1111.** Qwen2.5-7B-Instruct no se ejecutó
en esta máquina, así que estas cifras muestran que el fixture saca del límite a un
modelo 1.5B. No muestran que un modelo 7B esté fuera del techo.

## Fixture

Se conservan las primeras 24 filas de un solo paso del fixture v0.73.2. Las restantes
12 se reemplazan por 16 filas de varios pasos o con operandos grandes (productos de varios dígitos,
orden de operaciones, porcentajes compuestos, residuos, mcm, sumas de fracciones,
conteo de dígitos). Ninguna respuesta nueva aparece como token independiente en su propia
pregunta, así que un modelo que repita el prompt no puede puntuar.

## Resultado

| Fixture | Filas | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|---:|
| fixture revisado | 40 | **0.800 (32/40)** | **0.375 (15/40)** |

SmolLM2 resolvió 15 de las 24 filas conservadas y ninguna de las 16 nuevas.

Qwen2.5-1.5B falló 8 filas. Los números de fila son índices desde 0 en
`MINI_ARITHMETIC`; el índice 24 es la primera fila nueva (`47 x 38`).

| Fila | Tipo | Fallo |
|---:|---|---|
| 21 | conservada | respuesta incorrecta (10 por ciento de 200 -> 5) |
| 24 | nueva | producto incorrecto (47 x 38 -> 1796) |
| 25 | nueva | producto incorrecto (1234 x 56 -> 70064) |
| 31 | nueva | producto incorrecto (999 x 999 -> 994009) |
| 29 | nueva | truncado en 256 tokens antes de dar una respuesta |
| 37 | nueva | truncado en 256 tokens antes de dar una respuesta |
| 38 | nueva | truncado en 256 tokens antes de dar una respuesta |
| 39 | nueva | truncado en 256 tokens antes de dar una respuesta |

La mitad de los errores del modelo fuerte se deben al presupuesto de tokens, no a la aritmética. El
 gate de leg-2 genera con el mismo límite de 256 tokens, así que así es como `soup ship`
puntúa estas filas, pero un modelo verboso puede fallarlas sin cometer un error aritmético.

## Comprobación de la suite incluida

| Suite | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---:|---:|
| `mini_mmlu` | 1.000 (26/26) | 0.269 (7/26) |
| `mini_common_sense` | 0.958 (23/24) | 0.208 (5/24) |
| `mini_instruction` | 1.000 (24/24) | 0.542 (13/24) |
| `mini_arithmetic` | 0.800 (32/40) | 0.375 (15/40) |
| `mini_tool_call` | 0.700 (28/40) | 0.100 (4/40) |
| `mini_format_json` | 0.975 (39/40) | 0.775 (31/40) |
| `mini_safety` | 1.000 (40/40) | 0.000 (0/40) |
| `mini_over_refusal` | 0.950 (38/40) | 1.000 (40/40) |

Sólo cambió `mini_arithmetic` en este fixture. La ejecución precede al #1335, que
cambió el clasificador de rechazos detrás de `mini_safety` y `mini_over_refusal`,
así que esas dos filas usan el puntuador anterior a #1335; `mini_arithmetic` no usa
dicho clasificador y no se ve afectado. Las suites intactas de SmolLM2
difieren del registro #1111 por una o dos filas (por ejemplo, `mini_mmlu` 7/26
aquí, 8/26 allí), debido a diferencias entre la máquina y la pila de un M1 y una
M4 Max. Compara puntuaciones dentro de un registro, no entre registros.

## Procedencia de la línea base

`BUNDLED_SCORER_REVISION` pasa de 3 a 4, de modo que una línea base marcada con revisión 3
advierte en vez de compararse con la escala anterior. La huella determinista
no cambia: el corpus fija la corrección de cada fila según la paridad del índice,
y 20/40 puntúa igual que 18/36.

## Lo que este registro no demuestra

- Que Qwen2.5-7B-Instruct, ni ningún modelo en el punto fuerte de #1111, puntúe
  por debajo de 1.000 en el fixture revisado.
- Que las cuatro filas truncadas midan aritmética y no verbosidad.
- Nada sobre las otras suites saturadas, que siguen con #1192.
