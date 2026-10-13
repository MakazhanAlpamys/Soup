# Registro del gate — v0.73.2, leg 2 de `soup ship`

> 🇪🇸 Traducción al español de [`gate-v0.73.2-leg2-scoring.md`](gate-v0.73.2-leg2-scoring.md). Ante cualquier duda, manda el original en inglés.

**Equipo:** RTX 3050 Laptop 4 GB · Windows 11 · Python 3.10 · CPU para todas las ejecuciones de modelos
que siguen (un par 135M con 256 tokens nuevos; no se necesitó GPU ni se afirma haberla usado).
**Código de referencia:** v0.73.1 publicado (`bafbcdc`).

Este es el registro de trabajo, no un informe escrito después. Conserva las
mediciones retiradas, el hallazgo de revisión que se comprobó y rechazó parcialmente, y un susto
que resultó ser un artefacto de mis propias herramientas.

A diferencia de los registros v0.72.x, **esta etapa no necesitó GPU**: los cuatro defectos
están en un *scorer*, así que el instrumento fiel es un stub que emite las formas producidas por un
modelo real. Las dos ejecuciones con modelo real al final verifican el cableado integral, no
establecen los defectos.

---

## 0. Qué se afirmó y cómo se comprobó

| # | Afirmación del issue | Reproducido aquí como |
|---|---|---|
| #357 | Llama-3.1-8B puntúa 0.423 en `mini_mmlu`, por debajo de uno 0.5B | un stub que responde cada elemento **correctamente** al estilo `\boxed{C}` puntúa **0.000** |
| #346 | el 8B nombra la herramienta correcta 40/40 y puntúa 0.225 | un stub que nombra la herramienta correcta 40/40, pero deja sin una llave de cierre, puntúa **0.000** |
| #355 | `score_bundled_suite` devuelve 0.0 para un `gen` no invocable | devuelve **0.0** en 3 suites conductuales, **lanza una excepción** en las de opción múltiple: una asimetría que el issue no mencionó |
| #317 | leg 2 no tiene detector de rechazo excesivo | dos modelos, puntuaciones **idénticas byte a byte** en las 7 suites y el mismo veredicto SHIP, uno rechaza todas las solicitudes benignas |

Las cifras 0.423 / 0.225 son las del **registro H100**, en un 8B. No se volvieron
a medir aquí y no se afirma que sean cifras de este equipo. Lo que este equipo
establece es el *mecanismo*, en el caso extremo en que un modelo perfecto puntúa cero.

### Salida de reproducción (frente al v0.73.1 publicado)

```
#357 -- extract_mcq_letter does not know \boxed{C}
  extract_mcq_letter('...so the answer is \boxed{C}')  -> None
  mini_mmlu          perfect boxed-LETTER model scores 0.000  (want 1.000)
  mini_common_sense  perfect boxed-LETTER model scores 0.000  (want 1.000)
  mini_instruction   perfect boxed-LETTER model scores 1.000  (want 1.000)

#346 -- mini_tool_call ranks by brace hygiene
  expected : {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
  model out: {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}
  opens=3 closes=2
  a model naming the RIGHT tool 40/40 scores 0.000  (want 1.000)

#355 -- non-callable gen
  mini_tool_call    -> returned 0.0    <-- reads as 'failed every item'
  mini_format_json  -> returned 0.0
  mini_safety       -> returned 0.0
  mini_mmlu         -> raised TypeError  <-- ASYMMETRY

#317 -- leg 2 has no over-refusal detector
  leg-2 score maps identical: True
  helpful       -> SHIP, regressions=[]
  over-refusing -> SHIP, regressions=[]
  benign compliance: helpful=1.000  over-refusing=0.000
```

Que `mini_instruction` puntúe **1.000** en el caso #357 acotó la corrección: sus respuestas son texto libre, así que `score_answer` nunca llega al
extractor de letras de opciones y el valor entre cajas coincide por la ruta normal de tokens.
Por eso sólo dos de las cuatro suites MCQ/aritmética aparecen en
`SCORER_CHANGED_IN_V0_73_2` — comprobado, no supuesto:

```
mini_mmlu          items= 26  mcq-letter answers= 26  prompt changed= 26
mini_common_sense  items= 24  mcq-letter answers= 24  prompt changed= 24
mini_instruction   items= 24  mcq-letter answers=  0  prompt changed=  0
mini_arithmetic    items= 36  mcq-letter answers=  0  prompt changed=  0
```

---

## 1. Línea base RED

`tests/test_v07302.py` frente al v0.73.1 sin modificar: **53 fallaron / 39 pasaron**.

Las 39 que pasaron son los controles: comportamiento que ya era correcto y
debe seguir siéndolo. Si todo hubiera fallado, las pruebas estarían describiendo otro programa.

Final: **170 pasaron**.

---

## 2. La única decisión de diseño que surgió de una revisión, no de mí

El primer borrador colocó la **forma** con caja por encima de los niveles de pista y paréntesis. Una
revisión de código señaló que un modelo de razonamiento que encierra una respuesta provisional en cajas y luego
se corrige a sí mismo se interpretaría al revés. Se comprobó frente a las tres
implementaciones en lugar de discutirlo:

```
case                   want  A(pre)   B(tier)  C(ship)   fails for
boxed only             C     None     C        C         A
box AFTER echoed list  A     B        A        A         A
cue AFTER box          B     B        A        B         B
paren AFTER box        C     C        A        C         B
box AFTER cue          D     A        D        D         A
```

`A` = antes de la corrección, `B` = orden por niveles, `C` = publicado (decide la posición).

**Esta tabla también explica por qué se rechazó parcialmente un hallazgo de revisión.** La revisión TDD
calificó como vacías las filas 3 y 4, porque pasan con el código anterior a la corrección. Es cierto, pero son
las *únicas dos* que fallan en la variante ordenada por niveles; es decir, distinguen
otra respuesta incorrecta, no ninguna. Se **volvieron a etiquetar** como controles contra esa variante, en lugar de borrarlas, y las etiquetas ahora lo indican explícitamente para que nadie
las confunda después con evidencia obtenida primero al hacer fallar la prueba.

---

## 3. Retirado / corregido durante el trabajo

**Susto por dependencia del orden con «66 fallos» — retirado.** Una ejecución del nuevo archivo de pruebas
informó 66 fallos / 40 aprobadas, cuando la ejecución anterior había sido 96/0. Esa es
la señal del error de dependencia del orden del #389 y parecía una regresión real. Una
repetición limpia dio **1 fallo / 109 aprobadas**, y el único fallo era una aserción genuina sobre el panel. La ejecución de 66 fallos había leído un **archivo a medio escribir**. Se registra porque perseguirlo como un defecto real habría
sido una hora de trabajo desperdiciada por un fantasma.

**Un control que no demostraba nada, detectado por su propio fallo.** La primera versión
de la prueba de ampliación del piso usaba una alternancia en llamadas pares e impares para que dos repeticiones
no coincidieran. `mini_mmlu` tiene una cantidad **par** de elementos (26), así que la segunda pasada
empezaba con la misma paridad y reproducía exactamente la primera: piso 0.0, prueba
fallida. La lección es la regla propia del proyecto aplicada a sí misma: *un control sólo
cubre la variable que varía*, y este no variaba nada.

**Una prueba existente que pasaba por el motivo equivocado.**
`tests/test_v07125.py::_gold_answer` usaba `item["question"]` sin procesar como clave del modelo
falso. Tras la pista del prompt #357 esa búsqueda no encontraba ningún elemento MCQ
y devolvía `""`: un «modelo» que no responde nada. Así,
`test_live_dont_ship_on_regression` puntuaba base 0.0 = ajustado 0.0, no detectaba
regresión y habría salido con 0. Ahora pasa por `build_mcq_prompt`, el
mismo constructor que usa el detector; ese es el punto: dos copias de «¿qué prompt
enviamos?» son la causa de que divergieran.

---

## 4. Ejecuciones en vivo

Dos, ambas con `HuggingFaceTB/SmolLM2-135M-Instruct` (base) frente a
`HuggingFaceTB/SmolLM2-135M` (ajustado), CPU, generación real con 256 tokens nuevos.
Verifican el cableado, no los defectos.

### 4.1 El nuevo eje y el par

```
Leg 1 task win (metric): 0.3333 -> 0.3333  [no win]
mini_over_refusal   1.0000   0.9750   -0.0250   ok
mini_safety         0.0000   0.0000   +0.0000   ok
exit 2 (DON'T SHIP — leg 1 tied)
```

La propiedad de ejes emparejados, medida y no discutida: en los *mismos* modelos, el eje
benigno marca **1.000** mientras el dañino marca **0.000**. Un modelo de 135M no está
ajustado para seguridad, así que `mini_safety` en 0.000 es el resultado honesto esperado, y eso es
precisamente lo que da sentido al par: ninguno de los valores puede cambiar sin que el otro
quede visible.

Que aparezca `[no win]` es el segundo resultado. Consulta §5.

### 4.2 `--noise-floor 2`, `--task-mode metric`, `mini_mmlu`

```
noise floor: base repeat 1/2
noise floor: base repeat 2/2
noise floor: every axis repeated exactly — this instrument was deterministic
over these runs.

mini_mmlu   0.2692   0.1154   -0.1538   REGRESSED

Noise floor
  leg 1 task   0.0000
  mini_mmlu    0.0000
Measured over 2 base repeats.
exit 2
```

**El piso medido en este equipo es 0.0000 en ambos ejes.** La decodificación codiciosa en CPU
es determinista aquí. Ese es el resultado honesto y *no* contradice la dispersión 0.015 / 0.020 del
registro H100: se midió en GPU y este equipo no puede reproducirla. **No se afirma haber medido aquí ningún piso de GPU.**

De todos modos, la ejecución establece dos cosas:

- el eje `__task__` **sí** se mide en modo `metric` (aparece en la tabla),
  la rama que las pruebas unitarias no podían alcanzar sin una evaluación real;
- un piso 0.0 **no suprime nada**: `mini_mmlu` siguió teniendo una regresión de 0.1538 y
  siguió marcado. Que un piso ocultara silenciosamente una regresión real es el
  modo de fallo más probable de esta función, y la ejecución sirve de control contra ello.

### 4.3 El recorrido de ida y vuelta de la evidencia, con datos reales

La ejecución anterior produjo:

```json
{
  "task": {"mode": "metric", "base": 0.3333333333333333, "tuned": 0.3333333333333333},
  "benchmarks": {"mini_mmlu": {"base": 0.2692307692307692, "tuned": 0.11538461538461539}},
  "noise_floor": {"runs": 2, "floors": {"__task__": 0.0, "mini_mmlu": 0.0}}
}
```

Al volver a pasarla directamente por `soup ship --evidence`, sin cargar un modelo, reproduce el veredicto **idénticamente**: misma decisión, mismos deltas, mismo piso,
**exit 2 en ambas ocasiones**. Esta es la propiedad de #312 de que la salida sirve de entrada, extendida al
nuevo bloque y comprobada con datos medidos en vez de un fixture.

---

## 5. Dos defectos hallados al construir, ambos preexistentes

**El panel del veredicto nunca imprimía su propio indicador de leg 1.** `render_ship_panel`
construía el encabezado como `... [{won_str}]`, y `[no win]` sin más es marcado Rich válido
para una etiqueta desconocida, así que Rich lo eliminaba: en **todas las versiones hasta e incluido
v0.73.1**. Se delimitó con precisión ejecutando ambas superficies frente al código publicado:

```
a WIN    task_win.won=True   panel: Leg 1 task win (metric): 0.5000 -> 0.8000
         rubric: Leg 1 task win (metric): 0.5000 -> 0.8000  [won]
a LOSS   task_win.won=False  panel: Leg 1 task win (metric): 0.8000 -> 0.5000
         rubric: Leg 1 task win (metric): 0.8000 -> 0.5000  [no win]
```

`format_ship_rubric`, en texto simple y sin analizador de marcado, lo imprimió correctamente todo el tiempo. Esa asimetría explica por qué nadie lo notó.

**Nombres no confiables podían controlar la terminal.** Los nombres de benchmark y eje proceden
 de un archivo `--evidence`. `rich.markup.escape` neutraliza `[...]` y *nada
más*, así que un nombre de eje con secuencias OSC y CSI se mostraba con el marcado
inactivo, pero con los bytes de escape sin procesar. Ambas rutas de presentación ahora eliminan primero C0/DEL: el patrón
`_for_terminal` que ya existía en otros seis módulos de comandos y que `ship` nunca había adoptado.

---

## 6. Seguridad: el piso de evidencia

Un piso **ensancha** el gate, así que `"floors": {"mini_mmlu": 1.0}` en un archivo de evidencia
oculta cualquier posible caída en ese eje. Una revisión lo calificó de CRITICAL. Se
**rebajó la gravedad y se documenta el razonamiento**: no cruza una nueva frontera de confianza,
porque cualquiera que pueda editar ese archivo ya puede falsificar las puntuaciones sin procesar
y forzar SHIP directamente. Lo que sí hace es volver mucho más difícil detectar la manipulación en una
revisión, y `soup ci init` configura `ship --evidence` como gate de fusión de PR.

Por tanto: limitado a `[0, 1]`, máximo 50 ejes / nombres de 256 caracteres, **rechazado en vez de descartado**
si tiene formato incorrecto (un piso descartado se reproduce como un veredicto *diferente*,
que es exactamente el fallo que el recorrido de ida y vuelta pretende evitar), y cualquier piso
que exceda el umbral se anuncia por **ambos** lectores: la CLI en stderr, la herramienta MCP en el contenido que devuelve, porque stdout es el canal JSON-RPC.
Se rechazó el rechazo como solución: un recorrido de ida y vuelta legítimo tiene por diseño un piso > umbral, así que rechazarlo rompería el flujo correcto de la propia función.

Nueve casos de rechazo, cada uno con aserción del código de salida **y** una palabra clave que el operador
buscaría con grep:

```
[PASS] --noise-floor 1/0/-3/11             exit 3  "must be in [2, 10]"
[PASS] --noise-floor with --evidence       exit 3  "nothing to run under --evidence"
[PASS] evidence floor > 1.0                exit 1  "must be in [0.0, 1.0]"
[PASS] evidence noise_floor.runs = 1       exit 1  "runs must be in [2, 10]"
[PASS] evidence noise_floor 51 axes        exit 1  "too many axes (max 50)"
[PASS] evidence floors not an object       exit 1  "floors must be an object"
```

---

## 7. Lo que este registro NO demuestra

- **Aquí no se midió ningún piso de ruido de GPU.** La dispersión 0.015 / 0.020 corresponde al registro H100. Este equipo midió 0.0000 en CPU, donde la decodificación codiciosa es determinista.
- **`n=3`, un modelo, un dataset**, en el issue de origen. El piso **dimensiona**
  el efecto; no calibra un umbral, y aquí no se afirma qué valor de N es suficiente.
- **El piso de leg 1 no se midió en los modos con juez.** Una repetición con juez
  incorporaría el propio ruido de muestreo del juez en una cifra presentada como ruido de decodificación,
  así que la ejecución advierte y deja leg 1 en 0.0 en lugar de publicar una
  inferencia como si fuera un mecanismo.
- **La cifra 0.423 → 0.731 corresponde al registro H100, en un 8B.** Este equipo muestra el
  mecanismo en el extremo (un modelo perfecto con puntuación 0.000), no ese delta concreto.
- **Dos suites siguen en el techo o cerca de él** después de la reparación. Una suite
  fijada en 1.000 detecta una regresión tan mal como una fijada en 0.000,
  y esta versión no las redimensiona.

---

## 8. Efecto en el preprint: consultado y respondido, no implícito

El preprint de streaming por capas (DOI
[10.5281/zenodo.21918325](https://doi.org/10.5281/zenodo.21918325), v3) usa
`soup ship` como su **instrumento de medición** en la sección sobre calidad de convergencia,
así que esta versión debe contrastarse con él antes de etiquetarla.

**¿v0.73.2 cambia retroactivamente alguna cifra MEDIDA que indique el artículo?** No de manera retroactiva,
pero sí cambia el instrumento con que se obtuvieron esas cifras. El artículo
informa la reproducibilidad de leg-2 como un cambio de **0.375** en `mini_common_sense` y
**0.269** en `mini_mmlu` entre tres ejecuciones residentes idénticas, y una separación DON'T SHIP de 3 contra 1. Ambas suites tienen un **scorer diferente** a partir de esta
versión. Las cifras del artículo siguen siendo correctas *como mediciones* con el scorer v0.73.1;
**no** se vuelven a medir aquí y no hay fundamento para suponer
que resultarían iguales. La afirmación real del artículo en ese pasaje —que
la varianza se debe a la inicialización no sembrada del adaptador en la ruta residente, respaldada por
cinco ejecuciones en streaming con cambio de 0.0000 y un mismo hash de adaptador— es una propiedad de
las ejecuciones, no del scorer, y no se ve afectada.

**¿Cambia el ALCANCE descrito por el artículo?** No. El tema del artículo es
el streaming por capas; `soup ship` aparece sólo como instrumento. Esta versión no modifica el streaming.

**Decisión: alcance explícito, NO enmendado.** Un DOI publicado no se puede corregir a escondidas,
y aquí nada vuelve incorrecta una cifra indicada: sólo se señala que se midió con un scorer sustituido en esta versión. Quien vuelva a ejecutar esa sección con v0.73.2+ debería esperar cifras absolutas de leg-2 diferentes y
volver a medir, en lugar de comparar a través del límite entre versiones. Es el mismo
riesgo que advierte la versión para los archivos `--baseline`, por la misma razón.
