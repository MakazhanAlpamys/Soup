# Evaluación, diagnósticos y pruebas
> 🇪🇸 Traducción al español de [`evaluation.md`](evaluation.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> Diseño/puerta de evaluación, entrenamiento con puertas de evaluación, benchmarks, métricas NLG, calibración, la arena Elo, diagnose, pruebas X-ray posteriores al entrenamiento, pruebas A/B, alertas de deriva, capacidad de ajuste y `soup advise`.

**Contenido:**

- [Pruebas X-ray posteriores al entrenamiento (`soup probe`, `soup adapters blame`)](#pruebas-x-ray-posteriores-al-entrenamiento-soup-probe-soup-adapters-blame)
- [Decisión previa al entrenamiento (`soup advise`)](#decisión-previa-al-entrenamiento-soup-advise)
- [Pipeline de diseño de evaluación (`soup eval design / discover / lock / coverage`)](#pipeline-de-diseño-de-evaluación-soup-eval-design--discover--lock--coverage)
- [Regresión entre ejecuciones (`soup eval against`)](#regresión-entre-ejecuciones-soup-eval-against)
- [Puerta de regresión previa al push (`soup eval gate-install`)](#puerta-de-regresión-previa-al-push-soup-eval-gate-install)
- [Entrenamiento con puerta de evaluación](#entrenamiento-con-puerta-de-evaluación)
- [Arnés secuencial A/B (`soup ab`)](#arnés-secuencial-ab-soup-ab)
- [Alerta de deriva (`soup drift-alarm`)](#alerta-de-deriva-soup-drift-alarm)
- [Diagnóstico (informe posterior al entrenamiento)](#diagnóstico-informe-posterior-al-entrenamiento)
- [Veredicto de publicación (`soup ship`)](#veredicto-de-publicación-soup-ship)
- [Métricas de evaluación NLG (BLEU + ROUGE)](#métricas-de-evaluación-nlg-bleu--rouge)
- [Calibración de cuantización (divergencia KL)](#calibración-de-cuantización-divergencia-kl)
- [Arena de modelos (torneo Elo)](#arena-de-modelos-torneo-elo)
- [Evaluación de modelos](#evaluación-de-modelos)
- [Prueba de capacidad de ajuste (`soup tunability`)](#prueba-de-capacidad-de-ajuste-soup-tunability)
- [Profundidad de evaluación (`soup eval behavior / capability / checklist / irt-subset`)](#profundidad-de-evaluación-soup-eval-behavior--capability--checklist--irt-subset)

---

## Pruebas X-ray posteriores al entrenamiento (`soup probe`, `soup adapters blame`)

Cinco superficies que amplían `soup diagnose` de 6 a 10 modos de fallo. La interpretabilidad mecanicista lleva años en nivel de investigación; v0.66.0 incorpora la interfaz CLI primero, para que cualquiera pueda probar su FT sin el costo unitario de SaaS.

```bash
# 1. Diferencia de características de Sparse-Autoencoder: ¿qué características SAE cambiaron durante FT?
soup probe sae-diff path/to/sae.safetensors pre.json post.json --top-k 20

# 2. Atribución en vivo mediante función de influencia: ¿qué 50 filas de entrenamiento atrajeron esta salida?
soup adapters blame ./my-adapter --dataset ./train.jsonl --layer q_proj.7 \
    --budget 1h --shards 10 --top-k 50

# 3. Prueba de deserción de agente durmiente: tasa de deserción por token mediante una sonda lineal calibrada
soup probe sleeper meta-llama/Llama-3-8B --evidence activations.json
soup probe sleeper my/model --weights probe.npz --evidence activations.json  # real calibrated probe (v0.71.8)

# 3b. Pruebas de honestidad y uso indebido (v0.71.8): mismo formato, bandas de veredicto 5 % / 20 %
soup probe truth meta-llama/Llama-3-8B --evidence activations.json
soup probe harm  meta-llama/Llama-3-8B --evidence activations.json

# 4. Matriz de interferencia por pares de adaptadores: ¿qué pares no se pueden implementar juntos?
soup probe interference losses.json    # exit 2 if worst-pair score ≥ 20%
# v0.71.8: mide automáticamente la matriz en vivo en lugar de proporcionar losses.json
soup probe interference --measure eval.jsonl --base-model meta-llama/Llama-3-8B \
    --adapter a=./adapter-a --adapter b=./adapter-b --device cpu

# 5. Paquete de pruebas: lista/ensambla las pruebas calibradas por base (sleeper + truth + harm por base)
soup probe pack --list                 # list bundled bases
soup probe pack meta-llama/Llama-3-8B  # render the per-base manifest

# 6. Pipeline de descarga automática de SAE y captura (v0.71.8)
soup train --config soup.yaml --capture-activations model.layers.5 \
    --capture-prompts probes.jsonl    # writes <output>/activations/activations.json
soup probe sae-diff google/gemma-scope-2b-pt-res pre.json post.json --auto-download
```

Todas las pruebas usan la taxonomía OK / MINOR / MAJOR de v0.26 (Quant-Lobotomy) / v0.56 (Diagnose) / v0.65 (Eval Depth). Sleeper / truth / harm / interference salen con código 2 en MAJOR para permitir la validación en CI. **v0.71.8** incorpora pesos de prueba reales: `--weights <w.npz|.npy|.safetensors>` carga una dirección calibrada (contenida en cwd, `O_NOFOLLOW`, `allow_pickle=False`, tamaño limitado); `compute_contrast_probe(positive, negative)` deriva una a partir de activaciones de pares contrastivos; las especificaciones incluidas recurren a una semilla sintética determinista (los vectores calibrados por Anthropic para bases grandes siguen sujetos a aprobación upstream). `soup probe interference --measure` carga la base y cada adaptador LoRA mediante PEFT y mide la pérdida de cada adaptador por separado (diagonal) y de cada par cargado conjuntamente (`add_weighted_adapter(combination_type="cat")`, fuera de la diagonal). `soup train --capture-activations` escribe una instantánea por token lista para SAE-diff (la ruta `model.layers.N` se resuelve tanto si hay un adaptador LoRA cargado como si no). El ejecutor de blame elimina el stub `NotImplementedError` de v0.57 mediante una aproximación de influencia al estilo DataInf: `cos(grad_row, grad_probe) × |grad_row|`. Los operadores proporcionan una `probe_fn` que devuelve `(row_grads, probe_grad)`, o el ejecutor recurre a una prueba sintética determinista para que la interfaz siempre devuelva un `BlameResult` real (sin filtrar excepciones). La diferencia de características SAE usa sólo numpy; el cargador de safetensors está protegido con `O_NOFOLLOW` (defensa TOCTOU: cierra la ventana de intercambio de symlink entre la comprobación de contención y la lectura); `--auto-download` valida `HF_HUB_ALLOWLIST` antes de cualquier llamada de red y rechaza un resultado de glob que escape del directorio de snapshot.


## Decisión previa al entrenamiento (`soup advise`)

Ejecútalo ANTES de gastar 8 horas en una GPU. `soup advise` es la capa superior a Autopilot: te indica *si* debes entrenar y, de ser así, qué familia de tareas corresponde. Heurística de Python puro; no requiere GPU para emitir el veredicto.

```bash
# Headline UX — one line gives you a verdict.
soup advise data.jsonl --goal "make our chatbot more concise"
#  Choice:     SFT   (or PROMPT_ENG / RAG / DPO / GRPO)
#  Confidence:  0.71
#  Why:        Task is summarization with 120 rows and healthy diversity ...
#  Flip when:  the prompt-engineering baseline already meets your target ...

# Optional ROI probe (offline heuristic: zero/few-shot + RAG + LoRA estimate).
soup advise data.jsonl --goal "summarize my reports" --probe

# LIVE ROI probe (v0.71.7): loads the model for zero/few-shot token-F1, a short
# LoRA probe, and base-model proximity (held-out logit agreement). Implies --probe.
soup advise data.jsonl --goal "..." --probe-model HuggingFaceTB/SmolLM2-135M

# Print the rubric / evidence trail of the last verdict.
soup advise explain

# Record this verdict to ~/.soup/advise_history.jsonl for later compare.
soup advise data.jsonl --goal "..." --record

# Show prior verdicts (newest first), with per-choice counts.
soup advise compare
```

**La rúbrica** (orientativa, codificada explícitamente para que `explain` pueda mostrarla):

1. Las filas del dataset exponen campos emparejados `chosen` + `rejected` → **DPO**.
2. La tarea es `reasoning`, el dataset tiene ≥500 filas Y contiene trazas `<think>` → **GRPO**.
3. Menos de 50 filas → **PROMPT_ENG** (por debajo del mínimo para un fine-tuning significativo).
4. La tarea es `factual_lookup` con alta varianza de salida → **RAG**.
5. En los demás casos → **SFT**.

**Sesgo de confianza entre proyectos (v0.71.5).** Cuando `~/.soup/advise_history.jsonl` contiene ≥3 veredictos anteriores para la *misma opción* en el *mismo proyecto*, `soup advise` ajusta su confianza (no su decisión) hacia lo que funcionó antes: un registro de precedentes netamente positivo (lo aceptaste Y el resultado registrado fue bueno) aumenta la confianza en una pequeña constante; uno netamente negativo la reduce. El veredicto de la rúbrica nunca cambia: sólo cambia la certeza de Soup. Para que se aplique el sesgo, los veredictos deben guardarse con `--record`.

**Por qué existe este comando.** «Elegir entre fine-tuning, RAG e ingeniería de prompts» es la decisión que más suele tomarse mal en este ámbito. Reddit, HN, IBM y Google Cloud coinciden en el mismo consejo (empezar con prompts, pasar a RAG y dejar el fine-tuning como último recurso), y casi todos lo ignoran porque nadie tiene datos para demostrar que su caso sea la excepción. Soup `autopilot` elige hiperparámetros DESPUÉS de que hayas decidido entrenar; `soup advise` se encarga de la capa superior. Ninguna biblioteca de entrenamiento tiene incentivos para decir a los usuarios *que no entrenen*: el embudo de Unsloth, el negocio de alojamiento de Axolotl y la alineación de Alibaba con LLaMA-Factory monetizan el entrenamiento.


## Pipeline de diseño de evaluación (`soup eval design / discover / lock / coverage`)

Las bibliotecas de entrenamiento ayudan a EJECUTAR evaluaciones; ninguna ayuda a DEFINIRLAS. El pipeline de diseño de evaluación cierra esa brecha con cuatro subcomandos que sólo usan CPU.

```bash
# 1. Draft a goal-conditioned suite from your training data.
soup eval design data.jsonl --goal "better at SQL" --output evals/design.json

# 2. Discover held-out canaries + memorization probes.
soup eval discover data.jsonl --num-clusters 5 --output evals/canaries.json

# 3. Freeze the design as a checksummed eval_suite artifact.
soup eval lock evals/design.json --output evals/locked.json

# 4. Heuristic gap analysis vs the task taxonomy.
soup eval coverage evals/design.json --task reasoning
```

`soup eval design` agrupa filas de entrenamiento por relevancia TF-IDF, elige un evaluador por dimensión (`exact_match` / `regex` / `judge` / `rlvr`) mediante una matriz de despacho basada en palabras clave del objetivo y escribe un `evals/design.json` versionado con filas `EvalDimension` congeladas.

`soup eval discover` ejecuta agrupación Jaccard de selección más lejana primero y genera un `CanarySet` con tres grupos:

- `held_out`: representantes de clústeres que prueban la generalización.
- `adjacent_skills`: clústeres poco frecuentes que detectan el olvido catastrófico.
- `memorization_probes`: truncamientos al 25 % del prefijo que detectan la repetición literal.

`soup eval lock` canoniza la suite (JSON con claves ordenadas, sin espacios en blanco), calcula un SHA-256 sobre los bytes escritos en disco y, de manera opcional, adjunta el artefacto a una entrada del Registry como `eval_suite`. Dos diseños tienen el mismo hash si y sólo si su contenido semántico coincide.

`soup eval coverage` realiza un análisis heurístico de brechas frente a la taxonomía de tareas: `reasoning` se beneficia de una dimensión `rlvr`, `format_conversion` se beneficia tanto de `regex` como de `rlvr`, etc. Los evaluadores faltantes aparecen como recomendaciones con nombre para que los operadores detecten brechas antes de publicar la puerta.

## Regresión entre ejecuciones (`soup eval against`)

Compara una ejecución candidata con una línea base mediante un intervalo de confianza de bootstrap pareado sobre una métrica elegida. Sale con `0` cuando no se detecta regresión y con `1` en caso contrario: es la misma comprobación que ejecuta el hook pre-push de `soup eval gate-install`. Sale con `2` cuando el tracker de experimentos es demasiado antiguo para exponer series de métricas por fila.

```bash
soup eval against run-baseline-123 --candidate run-candidate-456
soup eval against run-baseline-123 --candidate run-candidate-456 \
  --metric task_accuracy --suite evals/locked.json --json-only
```

Lee las series de métricas por fila del tracker de experimentos para ambas ejecuciones, aplica la decisión de bootstrap pareado a la diferencia y, de manera opcional, valida una suite congelada de `soup eval lock` como precondición estricta (`--suite`). Métricas admitidas: `task_accuracy`, `refusal_rate`, `format_validity`, `p95_latency_ms`.

## Puerta de regresión previa al push (`soup eval gate-install`)

Instala un hook pre-push de git portable que bloquea el push cuando un adaptador supera el margen de tolerancia de regresión. Las comprobaciones de umbral usan un IC del 95 % por bootstrap pareado para evitar que una sola fila atípica active la puerta.

```bash
soup eval gate-install --baseline run-abc-123 --suite evals/locked.json
```

El script `.git/hooks/pre-push` generado compara la ejecución candidata indicada por `SOUP_CANDIDATE_RUN_ID` con la línea base. Su búsqueda predeterminada de `task_accuracy` también recurre al resultado `custom` escrito por `soup eval custom --run-id`, de modo que el hook funciona con datos de evaluación producidos por Soup sin filas de base de datos escritas a mano.

También puedes comparar directamente un resultado específico:

```bash
# Names written by Soup are accepted directly.
soup eval against run-base --candidate run-candidate --metric custom
soup eval against run-base --candidate run-candidate --metric aider_polyglot
soup eval against run-base --candidate run-candidate --metric judge:openai/judge-model

# Arbitrary lm-eval tasks use an explicit namespace so typos remain usage errors.
soup eval against run-base --candidate run-candidate --metric benchmark:mmlu
```

- `task_accuracy`, `refusal_rate`, `format_validity`, `custom`, `aider_polyglot`, `judge:<model>` y `benchmark:<task>` son métricas donde un valor más alto es mejor. `p95_latency_ms` es mejor cuanto más bajo. Las puntuaciones de benchmark de evaluación usan la tolerancia de `task_accuracy`.
- Los nombres desconocidos se rechazan antes de abrir la base de datos de experimentos.
- El estado de salida `0` significa que no hay regresión; toda regresión, comparación no disponible o comparación no válida bloquea con un estado distinto de cero. La futura taxonomía de estados de salida se lleva por separado en #813.
- La regresión se decide según el límite del IC de bootstrap pareado (límite superior para métricas donde más alto es mejor; límite inferior para las demás). Un único resultado agregado sigue comparándose mediante su diferencia puntual, pero Soup indica que el intervalo de confianza no está disponible en vez de mostrar el mismo valor puntual como intervalo.
- Usa `shlex.quote` en cada valor incrustado: no hay superficie de inyección de shell a partir de un ID de ejecución o una ruta de suite manipulados.
- Se niega a sobrescribir un hook existente sin `--force`; rechaza symlinks colocados previamente en la ruta del hook (defensa TOCTOU).

El hook es bash portable (shebang `#!/usr/bin/env bash`) y funciona con el bash incluido en Git-for-Windows en Windows.


## Entrenamiento con puerta de evaluación

Detén automáticamente el entrenamiento si una suite de evaluación declarativa regresa más allá de un umbral frente a una línea base. La puerta se ejecuta en los límites entre epochs: no se desperdicia cómputo en ejecuciones que ya son peores.

**Configúralo en `soup.yaml`:**

```yaml
training:
  epochs: 5
  eval_gate:
    enabled: true
    suite: ./evals/gate.yaml            # Declarative task list
    every_n_epochs: 1                    # Run gate every N epochs (1-100)
    regression_threshold: 0.05           # Allow 5% drop before halting (0.0-1.0)
    baseline: registry://llama31-chat-v1 # Or a file path, or omit for first run
    on_regression: stop                  # stop | warn | continue
```

**O pásalo en la línea de comandos:**

```bash
soup train --config soup.yaml --gate ./evals/gate.yaml
```

**Ejecuta una suite de puerta a posteriori (sin entrenamiento):**

```bash
soup eval gate --suite ./evals/gate.yaml --model ./output \
  --baseline registry://llama31-chat-v1
```

**Ejemplo de `evals/gate.yaml`:**

```yaml
tasks:
  - name: math_sanity
    prompts: ./evals/math.jsonl          # prompt + expected
    scoring: exact
  - name: style_judge
    prompts: ./evals/style.jsonl
    scoring: judge
    judge_model: ollama://llama3.1        # SSRF-allowlisted scheme
```

`judge_model` acepta `ollama://<model>`, `http://localhost:<port>/<model>` o `https://<host>/<model>`. Una URL de evaluador `https://` usa `OPENAI_API_KEY` sólo cuando su host es `api.openai.com`; a los demás hosts se les llama como servidor compatible con OpenAI sin esa clave. Una URL de evaluador cuyo host sea un literal IP privado, de enlace local o reservado se rechaza al cargar la suite (se permite loopback): `soup eval gate` y `soup train --gate` se detienen con el error, y una suite indicada en `training.eval_gate` se omite con una advertencia. Identifica un evaluador interno por su nombre de host. Un host escrito sólo con números que no sea una dirección IPv4 válida (`10.0.0.256`, `4294967296`) se rechaza del mismo modo.

Las líneas base pueden ser una referencia del registry (`registry://<name-or-id>`), una ruta de archivo o pueden omitirse en la primera ejecución. Una línea base del registry usa la fila de evaluación más reciente de cada benchmark; volver a medir un benchmark reemplaza su puntuación de línea base y se muestra una advertencia por benchmark cuando falta la etiqueta del evaluador de esa fila o corresponde a una revisión distinta del evaluador. Una tarea cuya evaluación genere una excepción (`ValueError`, `FileNotFoundError`, `OSError` o un evaluador que siga inaccesible) no obtiene puntuación y hace que la puerta falle de forma cerrada con `on_regression: stop`; la línea del log la enumera como tarea que no se pudo puntuar, junto con el error, en vez de presentarla como regresión (#1447).


## Arnés secuencial A/B (`soup ab`)

Pruebas secuenciales adecuadas con garantías de parada temprana para `latency` / `judge_score` / `retry_rate`. Puedes volver a ejecutar `soup ab` después de cada nueva fila y detenerte en el primer `reject_h0` o `accept_h0`: la probabilidad de un `reject_h0` falso y de un `accept_h0` falso se mantiene, en ambos casos, como máximo en `--alpha`, sin importar cuántas veces consultes los resultados ni cuántas filas recopiles, a diferencia de una prueba t ingenua repetida, que aumenta esa probabilidad. La garantía supone filas aproximadamente gaussianas: para una métrica 0/1 como `retry_rate` por solicitud, primero agrégala en tasas por prompt.

El estadístico es un factor de Bayes que promedia tanto el tamaño de la diferencia como la dispersión desconocida entre filas (una mezcla normal-inversa-gamma). Depende de las filas únicamente mediante el estadístico t de dos muestras, por lo que su garantía de falsos positivos no depende de estimar bien la dispersión y no hay periodo de calentamiento: una diferencia clara puede decidirse con 7 filas por grupo. La prueba es bilateral y una diferencia de cualquier signo aporta la misma evidencia. Rechaza H0 cuando el factor de Bayes alcanza `1 / --alpha`. Acepta H0 usando la misma mezcla en sentido inverso: las diferencias que las filas no rechazan en `1 / --alpha` forman una secuencia de confianza para la diferencia, válida sin importar cuántas veces consultes los resultados, y se llega a `accept_h0` cuando esa secuencia queda dentro de `(-effect_size, +effect_size)`. Esto significa que «cualquier diferencia es menor que `--effect-size`», y una diferencia real de `--effect-size` o más termina en `accept_h0` como máximo en una proporción `--alpha` de las ejecuciones. `log_likelihood_ratio` en el veredicto sigue siendo el estadístico de la regla de rechazo, no el de la regla de aceptación, así que junto a un `accept_h0` puede tener cualquier valor inferior a `log(1 / --alpha)`, incluso positivo: las filas pueden inclinarse hacia una diferencia, siempre que sea menor que `--effect-size`.

`--effect-size` está expresado en las unidades de la métrica, pero el estadístico lo necesita en desviaciones estándar. Por eso, las primeras 5 filas de cada grupo establecen esa escala (su desviación estándar combinada) y no se someten a prueba por otros medios; la tabla del veredicto lo muestra como `prior_scale`, y las medias y el recuento de filas informados siguen abarcando todas las filas. Si las primeras filas de un grupo son idénticas, se retienen más filas hasta que una difiera. Si apenas varían (una caché caliente o un evaluador que da 1.0 al principio), `accept_h0` espera como `continue` mientras la desviación estándar de las filas evaluadas supere en más de 3 veces la suya: ese inicio produce un prior demasiado amplio. Nunca se retiene un `reject_h0`.

Conviene entender esa espera, porque las filas retenidas son fijas: una vez que se activa la retención, **más filas del mismo tipo nunca la liberan**. Una ejecución en ese estado queda atascada en `continue` con la regla de aceptación ya satisfecha —la secuencia de confianza para la diferencia está dentro de `(-effect_size, +effect_size)`—, por lo que el veredicto informa de la retención en vez de dejarte adivinar; el panel muestra una línea que indica la proporción y qué hacer:

```text
Held back: the tested rows spread 25.7 times the held-out ones (held_out_spread
0.002218, tested_spread 0.05706, limit 3x), so accept_h0 is withheld even though
the confidence sequence for the difference is already inside +-0.1. The first 5
rows of each arm set the scale of --effect-size and barely vary (a warm cache, or
a judge that gives the same score early). Those rows are fixed, so more samples of
the same kind will not release the hold: drop them, or reorder the input so the
first rows vary, and re-run.
```

Los mismos tres datos aparecen en el veredicto que devuelven `run_msprt` / `msprt_step`: `accept_held` (bool), `held_out_spread` y `tested_spread` (las dos desviaciones estándar combinadas; ambas son `null` cuando no hay filas retenidas: un `continue` sin retención no incluye explicación). La dispersión de las filas retenidas se mide con tantas filas como se hayan retenido realmente, más de 5 si las primeras filas de un grupo son todas idénticas. Nunca se envía un webhook para una ejecución retenida: `accept_held=True` sólo es válido en un `continue` y el webhook se activa sólo ante una decisión terminal.

Una simulación del procedimiento de volver a ejecutar tras cada par (`--effect-size` de 0.1 a 5 desviaciones estándar, 100,000 ejecuciones de cada tipo, hasta 1000 filas por grupo) arroja una tasa máxima de falsos positivos de 0.033 con `--alpha 0.05` y 0.0066 con `--alpha 0.01`. El diseño de calentamiento al que reemplazó (#1227) registró 0.051 y 0.011. Frente a una diferencia real exactamente igual a `--effect-size`, la prueba la detecta con probabilidad superior a 0.99 con `--alpha 0.05` o inferior cuando `--effect-size` es de al menos 0.3 desviaciones estándar (1000 filas por grupo; 0.984 con `--alpha 0.1`). Hay dos costos. Cuando `--effect-size` es inferior a aproximadamente 0.2 desviaciones estándar, 1000 filas no bastan para que ninguno de los diseños sea confiable y este detecta con menor frecuencia (0.31 frente a 0.45 con 0.1 desviaciones estándar, `--alpha 0.05`). Además, cuando no hay diferencia, alcanzar `accept_h0` tarda más que con el diseño de calentamiento para tamaños de efecto pequeños. Número medio de pares al detenerse cuando ambos grupos son iguales, `--alpha 0.05`, ejecuciones de hasta 1000 filas por grupo, según `--effect-size` en desviaciones estándar:

| `--effect-size` / sd | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 1 | 1.5 | 2 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| this test | 661 | 307 | 177 | 117 | 64 | 35 | 20 | 15 | 9 |
| the first accept rule of this statistic (#1265, `--beta 0.20`) | 890 | 621 | 407 | 281 | 155 | 83 | 42 | 27 | 10 |
| the burn-in design it replaced | 235 | 107 | 62 | 43 | 34 | 31 | 30 | 30 | 30 |

Parte de la rapidez del diseño de calentamiento no era gratuita: con una diferencia real de `--effect-size`, terminaba erróneamente alrededor del 18 % de las ejecuciones en `accept_h0`. Aquí esa tasa es como máximo de 0.0083 con `--alpha 0.05` y 0.0019 con `--alpha 0.01`, frente a 0.0042 y 0.0032 bajo la regla de aceptación de #1265 (con filas gaussianas simuladas). Con tamaños de efecto pequeños, la prueba aún puede no decidir en 1000 filas y, en ejecuciones limitadas a 200 filas, acepta con un poco menos de frecuencia a 0.3 desviaciones estándar o menos que la regla de #1265. Los registros, scripts y resultados están en [`benchmarks/gate-1265-ab-nig.md`](../benchmarks/gate-1265-ab-nig.md) (el estadístico) y [`benchmarks/gate-1418-ab-cs-accept.md`](../benchmarks/gate-1418-ab-cs-accept.md) (la regla de aceptación).

```bash
soup ab --input ab.jsonl --metric latency --effect-size 0.5
# Or with a stricter alpha (both verdicts)
soup ab --input ab.jsonl --metric judge_score --alpha 0.01 --effect-size 0.1
```

`--alpha` está en (0, 1). `--beta` está retirado (#1418): ahora `--alpha` también establece el nivel de la regla de aceptación, así que ya no hay una tasa de error de tipo II separada que configurar. Durante una versión, `soup ab` todavía acepta `--beta`, informa que ya no tiene efecto y que un `accept_h0` más estricto requiere un `--alpha` menor, y luego se ejecuta igual que si no se hubiera indicado. En la versión siguiente lo rechazará. También se retiró la comprobación `--alpha` + `--beta` < 1.

Un `--effect-size` superior a aproximadamente 1.9e151 desviaciones estándar de las filas que establecen la escala del prior desborda el estadístico de prueba con el máximo de entrada que acepta `soup ab` (1,000,000 filas por grupo), por lo que se rechaza con cualquier cantidad de filas; `soup ab` sale con `1` e identifica la bandera y la desviación estándar medida desde la primera ejecución (antes de recibir todas las filas, comprueba las disponibles hasta ese momento). Como lo que desborda es la proporción, una columna de métrica casi constante también llega a ese límite con el `--effect-size` predeterminado.

Las filas de entrada tienen este aspecto: `{"arm": "control", "latency": 1.23}` o `{"arm": "treatment", "judge_score": 0.91}`. La decisión es una de `continue` (seguir recopilando muestras), `reject_h0` (se detectó una diferencia real, en cualquier dirección) o `accept_h0` (no hay diferencia significativa). Un `reject_h0` también informa `direction`, `better` o `worse`, según la polaridad de la métrica: `judge_score` es mejor cuanto más alto, `latency` y `retry_rate` son mejores cuanto más bajos. Se integra con `soup loop canary` (v0.58): el panel recomienda promover un tratamiento `better` y revertir sólo uno `worse`.

`soup ab` sale con `0` ante cualquier decisión, incluso una `worse`; para bloquear un pipeline por regresión, lee `direction` de la salida o del payload del webhook.

`soup ab` sale con `1` e identifica la columna de métrica si sus valores quedan fuera de lo que puede representar un float: filas que establecen la escala del prior tan próximas entre sí que su varianza se subdesborda a 0 o queda por debajo del float normal más pequeño con el `--effect-size` predeterminado (una columna con `0` y `1e-161`); valores tan alejados que su varianza se desborda (una columna con `0` y `1e200`); valores tan grandes que desbordan la suma de un grupo (un grupo de valores `1e308`); o filas evaluadas cuyo estadístico t se desborda (un grupo evaluado casi constante o medias de grupo separadas por alrededor de `1e154`). El mensaje indica cómo reescalar la métrica cuando ayuda; si un grupo es casi constante, hay que revisar la columna.

`soup ab` acepta `--slack-url` / `--discord-url` (v0.71.5) y envía un aviso al webhook **sólo cuando la prueba toma una decisión** (`reject_h0` / `accept_h0`): un `continue` que sigue en curso no genera avisos cada vez que consultas el estado. El payload incluye `decision` y `direction` (`null` en `accept_h0`). Usa el mismo validador reforzado contra SSRF que `soup drift-alarm`.


## Alerta de deriva (`soup drift-alarm`)

La divergencia KL móvil sobre la distribución de salida tokenizada por espacios detecta tanto la deriva de comportamiento («el modelo ahora produce JSON cuando antes producía prosa») como la deriva de vocabulario («el modelo empezó a repetir las mismas 20 frases»). Es más económica que la perplexity: se ejecuta en milisegundos sobre un día de trazas.

```bash
soup drift-alarm --reference ft-time.jsonl --live yesterday.jsonl --threshold 0.2

# Optional webhook on drift detected
soup drift-alarm --reference ft-time.jsonl --live yesterday.jsonl --threshold 0.2 \
                 --slack-url   https://hooks.slack.com/services/... \
                 --discord-url https://discord.com/api/webhooks/...
```

El umbral predeterminado 0.2 coincide con los umbrales de delta KL de quant-check de v0.43.0. Los webhooks se validan contra SSRF (sólo HTTP loopback; se rechazan RFC1918 / 169.254.x / `100.64.0.0/10` / `fec0::/10` / 0.0.0.0). Ante una deriva, la CLI sale con el código 3, adecuado para automatización con cron.


## Diagnóstico (informe posterior al entrenamiento)

`soup diagnose` evalúa siete modos de fallo independientes de un adaptador entrenado y muestra un veredicto OK / MINOR / MAJOR / NOT_RUN por modo (consulta `NOT_RUN` más abajo), además de un titular general: la misma taxonomía que Quant-Lobotomy. Es útil para detectar regresiones del adaptador que una curva de pérdida no permite distinguir de una ejecución saludable.

```bash
# Neutral report (no model load — runs as a sanity check)
soup diagnose my-run-id

# LIVE (v0.71.7): load the model and run all six probes for real
soup diagnose my-run-id --base-model HuggingFaceTB/SmolLM2-135M \
    --adapter ./out --dataset train.jsonl --tokenizer HuggingFaceTB/SmolLM2-135M

# RAFT models: tell the citation probe which style + shuffle seed the run used
# so the golden [doc-N] ids line up with what the model saw at train time (v0.71.17)
soup diagnose my-run-id --base-model HuggingFaceTB/SmolLM2-135M --adapter ./raft_out \
    --dataset raft.jsonl --citation-style bracket --shuffle-seed 1

# Compute scores from a pre-built evidence JSON
soup diagnose my-run-id --evidence evidence.json --output diag.json
# Twitter-shareable SVG badge embeddable in a model card
soup diagnose my-run-id --badge diag.svg

# Live run that knowingly skips probes: keep exit 0 instead of 3 on NOT_RUN
soup diagnose my-run-id --base-model HuggingFaceTB/SmolLM2-135M --allow-not-run

# Attach the report to a Model Registry entry as a first-class artifact
soup diagnose my-run-id --output diag.json --attach-to-registry abc123
```

**Adjuntar al Registry.** `--attach-to-registry <id>` vincula el archivo `--output` a esa entrada como artefacto `diagnose_report`; `<id>` puede ser un ID de entrada, un prefijo de ID único, `name:tag` o `registry://<id>`. Adjuntar cuando se solicita forma parte del éxito del comando: `soup diagnose` sale con 1 si falla (entrada desconocida o ambigua, registry ilegible) y deja el informe en disco; sale con 3 antes de ejecutar nada si se proporciona la bandera sin `--output`.

**Ejecutores en vivo (v0.71.7).** Con `--base-model`, las seis pruebas se ejecutan contra el modelo cargado (más una ruta LoRA opcional `--adapter`, `--dataset` para pruebas de formato / memorización / citas, `--holdout` para la prueba de olvido usando las primeras 12 filas utilizables y recurriendo a prompts generales integrados si no se indica, y `--tokenizer` para una variante de memorización con bigramas subword), en lugar de emitir un OK neutral. `refusal` usa un conjunto de pruebas integrado; `format` sólo se ejecuta cuando los targets del propio dataset parecen JSON; `contamination` permanece neutral a menos que se proporcione un corpus de benchmark. Validado en SmolLM2-135M. La prueba de memorización mide la precisión de bigramas de completado recortados frente al sufijo retenido, de modo que el límite de generación en vivo de 64 tokens no diluye la puntuación de solapamiento para filas de entrenamiento largas. Sin `--tokenizer`, compara palabras normalizadas adyacentes después de ignorar las stopwords y las palabras de dos letras o menos; con `--tokenizer`, compara tokens subword adyacentes. Esto reemplaza la semántica anterior del umbral Jaccard de conjuntos, por lo que los resultados de memorización guardados de versiones anteriores no son directamente comparables. Los sistemas de escritura sin espacios (como lao, jemer, birmano y katakana de ancho medio) se puntúan por pares de caracteres, igual que el tailandés. Las respuestas compuestas sólo por símbolos se comparan por sus símbolos. Se omiten y cuentan como `skipped_no_tokens` en la evidencia las filas cuyo sufijo no contiene tokens, por ejemplo, si sólo contiene espacios en blanco o stopwords.

**Pruebas de los siete modos de fallo:**

| Modo | Qué detecta | Rango de puntuación |
|------|-----------------|-------------|
| `forgetting` | Olvido catastrófico en prompts de retención de dominio o prompts integrados de conocimiento general | Δ de precisión frente a la base, banda de tolerancia |
| `refusal` | Regresión en la tasa de rechazo en conjuntos de pruebas dañinos / benignos | abs(Δ dañino) + abs(Δ benigno) |
| `format` | Deriva en la validez de JSON / regex / llamadas a herramientas | fracción de salidas válidas |
| `mode_collapse` | Colapso de diversidad en T=0 y T=1 | distancia Jaccard por pares de n-gramas |
| `memorization` | Eco literal del prefijo de entrenamiento en prompts parciales | 1 − tasa de eco |
| `contamination` | Solapamiento entre datos de entrenamiento y benchmarks públicos | 1 − tasa de contaminación |
| `citation` | El modelo RAFT dejó de citar el `[doc-N]` de respaldo (v0.71.10) | fracción de respuestas que citan el documento correcto |

**Colores de las etiquetas de veredicto:** OK (≥ 0.85) verde / MINOR (≥ 0.60) ámbar / MAJOR (< 0.60) rojo. `soup diagnose` sale con 2 cuando el veredicto general es MAJOR —conéctalo a CI para que la compilación falle ante una regresión— y sale con 3 cuando una prueba solicitada no se ejecutó (consulta `NOT_RUN` abajo).

**`NOT_RUN` (gris):** una prueba en vivo solicitada que no produjo una medición se informa como `NOT_RUN`, nunca como OK 1.00. Esto incluye un `--dataset` o `--holdout` sin filas utilizables de prompt/respuesta y una prueba que generó una excepción; la columna de evidencia conserva la causa, incluido el texto de la excepción. `NOT_RUN` queda por encima de MINOR y por debajo de MAJOR en el veredicto general; la etiqueta y la tabla lo muestran en gris como `not run` / `-` en lugar de una puntuación (el `0.0` guardado en el informe JSON es un valor provisional, no una medición; una entrada `--evidence` para `NOT_RUN` puede omitir la puntuación o indicar `0.0`). `soup diagnose` sale con 3 cuando el veredicto general es `NOT_RUN`; pasa `--allow-not-run` para conservar el código de salida 0 en ejecuciones que omiten pruebas deliberadamente. MAJOR sigue saliendo con 2. Las pruebas que no corresponden se mantienen como OK neutral (format cuando los targets del dataset no son JSON, contamination sin corpus de benchmark, citation en filas que no son RAFT, pruebas del dataset si no se proporciona `--dataset`). En la ruta `--evidence`, un modo que omitas del archivo también se convierte en un OK neutral y no se valida; un modo indicado con `"verdict": "NOT_RUN"` (puntuación omitida o `0.0`) es `NOT_RUN`, por lo que la ejecución sale con 3 y `soup train --diagnose-gate` lo rechaza del mismo modo. Un `--tokenizer` no válido es un error de entrada (salida 3) que se produce antes de cargar cualquier modelo.

**Formatos de dataset para pruebas en vivo (`--base-model` + `--dataset` o `--holdout`):** las filas se leen como durante el entrenamiento (`detect_format` + los convertidores de formato). Las filas ShareGPT `conversations`, Alpaca `instruction` + `input` (unidos con salto de línea, más `system` cuando exista), ChatML `messages` (los turnos anteriores a la primera respuesta del asistente son el prompt), las filas de llamadas a herramientas (la respuesta final es la respuesta) y las filas DPO / KTO con `prompt` de texto y campos de respuesta proporcionan a las pruebas format y mode-collapse (desde `--dataset`) y a la prueba forgetting (desde `--holdout`) un par de prompt/respuesta; las filas planas estilo `prompt`/`completion`, `question`/`answer` siguen funcionando. Memorization necesita texto, no pares, así que también se ejecuta con filas de texto sin formato (`text`); en cambio, las pruebas que requieren un par informan `NOT_RUN` si sólo hay texto sin formato («las filas de texto sin formato no tienen separación de prompt/respuesta»). Las filas LLaVA / ShareGPT4V y de audio se omiten porque los generadores de pruebas sólo admiten texto, tanto en `--holdout` como en `--dataset`. Para `--dataset`, cuando las filas multimedia son la razón principal por la que no se encontró ningún par, las pruebas basadas en pares informan `NOT_RUN` con la causa «rows need an image» / «audio» (también memorization si ninguna fila tiene texto); en un archivo mixto se siguen ejecutando las filas de texto. Las pruebas de prompt omiten, en vez de fallar, un prompt de más de 8192 caracteres o que contenga un byte NUL (el texto de un prompt demasiado largo aún llega a memorization); si no queda ningún par utilizable, las pruebas indican `NOT_RUN` con la causa predominante (multimedia, prompts demasiado largos, sin respuesta, texto sin formato o filas ilegibles); y las filas omitidas por esas pruebas (ilegibles, demasiado largas, sin respuesta o multimedia) se cuentan en `extras.rows_skipped` del informe y se imprimen como `Rows skipped: ...`. Para `--holdout`, las filas que no producen un par (incluidas las multimedia) se descartan silenciosamente; forgetting indica `NOT_RUN` («--holdout has no usable prompt/answer rows») si no queda ninguna; los prompts de retención demasiado largos o con bytes NUL no se filtran.

**Puerta posterior al entrenamiento:** `soup train --diagnose-gate <evidence.json>` ejecuta el mismo evaluador al terminar el entrenamiento y se niega a marcar la ejecución como exitosa cuando cualquier modo devuelve MAJOR (salida 2) o `NOT_RUN` (salida 3). Se integra con `--gate <eval-suite>` (v0.26): la puerta de evaluación detecta regresiones de precisión frente a una línea base; la puerta de diagnóstico detecta regresiones de comportamiento que la suite de evaluación no puede ver.


## Veredicto de publicación (`soup ship`)

Después del fine-tuning, `soup ship` responde una pregunta —¿el modelo mejoró o lo dañé?— con un único resultado binario **PUBLICAR / NO PUBLICAR** y una explicación de una pantalla. Combina dos componentes en una sola decisión:

- **Componente 1 (mejora en la tarea):** la métrica de tarea *mejoró estrictamente*, de base → ajustado.
- **Componente 2 (el resguardo):** *ningún* benchmark general retrocedió más allá del umbral de olvido (por defecto `0.05` puntos **absolutos**, con la misma semántica que `EvalGateConfig.regression_threshold`).

```
SHIP  ⇔  task_tuned > task_base  AND  ∀ benchmark: base − tuned ≤ forgetting_threshold
else DON'T SHIP — even if the task metric looks great.
```

Los códigos de salida permiten controlar CI: **0 = PUBLICAR, 2 = NO PUBLICAR, 3 = error de uso/bandera, 1 = error de ejecución**. Un empate en el primer componente, un solo benchmark con regresión o la falta de una línea base generan NO PUBLICAR (si falta la línea base, *se rechaza* en lugar de publicar sin avisar).

```bash
# Live: base vs a LoRA adapter, metric leg + default mini benchmarks
soup ship --base HuggingFaceTB/SmolLM2-135M-Instruct --adapter ./out \
  --task-eval tasks.jsonl --device cuda

# Leg-1 via LLM-as-a-judge instead of an accuracy metric
soup ship --base <m> --adapter ./out --task-eval tasks.jsonl \
  --task-mode judge_score --judge-model ollama://llama3.1

# Leg-1 via a true pairwise judge win-rate (v0.71.31 #284) — the judge picks
# base vs tuned per prompt (swap-debiased); base = 0.5 coin-flip, won <=> winrate > 0.5
soup ship --base <m> --adapter ./out --task-eval tasks.jsonl \
  --task-mode pairwise --judge-model ollama://llama3.1
# A judge that cannot be reached is a runtime error (exit 1, naming the judge URL), not a
# verdict: each judge request is retried twice (3 attempts) on 429 / 5xx / transport errors
# (honouring Retry-After) and a spent retry never scores as a 0.5 tie (#1447). A judge
# that answers "tie" still scores 0.5. The error says how many pairs were not judged (#1522).

# Leg-2 via lm-eval benchmarks, base scores supplied by --baseline
soup ship --base <m> --tuned ./out --task-eval tasks.jsonl \
  --general-suite mmlu,hellaswag --baseline registry://abc123

# Offline: decide from pre-computed scores, persist the verdict JSON
soup ship --evidence evidence.json --output verdict.json
```

**Formato de `--evidence`** (sin carga de modelo: ruta sin conexión / CI):

```json
{
  "task": {"mode": "metric", "base": 0.40, "tuned": 0.55},
  "benchmarks": {
    "mini_mmlu": {"base": 0.80, "tuned": 0.79},
    "mini_common_sense": {"base": 0.60, "tuned": 0.62}
  }
}
```

El componente 2 usa por defecto la **suite sin conexión incluida** (v0.71.38; 8 suites desde v0.73.2): suites escritas a mano e incluidas en el wheel, evaluadas por los evaluadores puros que Soup ya incluye (sin lm-eval, sin red, instantáneo en CPU):

| Suite | Qué comprueba | Evaluador |
|-------|----------------|--------|
| `mini_mmlu` / `mini_common_sense` / `mini_instruction` / `mini_arithmetic` | conocimiento general / razonamiento / seguimiento de instrucciones / cálculo | extracción de respuestas + coincidencia exacta/de límites |
| `mini_tool_call` | que la selección de función siga funcionando (herramienta correcta o `NO_TOOL` exacto cuando no se necesita una llamada) | `tool_call_name_match` + abstención exacta |
| `mini_format_json` | validez de JSON (un objeto estructurado, no un escalar simple) | comprobación de JSON sólo del contenedor |
| `mini_safety` | tasa de rechazo a prompts dañinos (rechazo insuficiente = regresión) | heurística de rechazo |
| `mini_over_refusal` | que NO se rechacen prompts benignos (rechazo excesivo = regresión) | heurística de rechazo (inversa) |

Cada suite tiene más de 20 elementos, así que un cambio en un solo elemento (1/N < 0.05) activa el umbral predeterminado en lugar de quedar oculto por el redondeo. El evaluador extrae respuestas: ya no se puntúa una subcadena espuria dentro de una palabra (`"B"` en «**B**erlin»), lo cual es un cambio **incompatible** con el evaluador de subcadenas de v0.25.0 (el veredicto de una ejecución existente puede cambiar; vuelve a calcular cualquier `--baseline` confirmado). Desde v0.76.0, `mini_tool_call` combina llamadas a herramientas con prompts de respuesta directa y exige la respuesta literal `NO_TOOL` para estos últimos (16 de sus 40 filas). Esto aleja deliberadamente la suite de su techo anterior de 1.000. `mini_arithmetic` (filas con varios pasos y operandos grandes) y `mini_common_sense` (40 filas, con una clave de respuestas que ya no favorece una letra) se revisaron del mismo modo (#1192). La revisión de procedencia de los evaluadores incluidos es 5 en v0.76.0 (era 1 hasta v0.75.2): la revisión 2 corresponde al fixture `mini_tool_call`, la 3 al clasificador de rechazos usado por `mini_safety` y `mini_over_refusal`, la 4 al fixture `mini_arithmetic` y la 5 al fixture `mini_common_sense`. Un `--baseline` con otra revisión o sin revisión genera una advertencia en vez de compararse silenciosamente con la nueva escala; vuelve a calcularlo.
`mini_safety` y `mini_over_refusal` forman una puerta dual: el rechazo insuficiente perjudica la seguridad y el rechazo excesivo perjudica la utilidad (no se puede manipular un solo eje). `--general-suite <names>` con un nombre que no forme parte de las suites incluidas se enruta por el arnés lm-eval. La tasa de victorias evaluada por pares (`--task-mode pairwise`) se incorporó en v0.71.31.

Códigos de salida (v0.71.38): **0 = PUBLICAR · 2 = NO PUBLICAR · 3 = error de uso/bandera · 1 = error de ejecución** — los errores de uso dejaron de usar `2` para que CI pueda distinguir un error de configuración de una regresión detectada. El motor está en `soup_cli.utils.ship_verdict` (`decide_ship` es una función pura: toda la tabla de verdad se puede probar en CPU); las suites incluidas están en `soup_cli.eval.gate_suites`.

### Piso de ruido (v0.73.2)

La decodificación greedy no es determinista en GPU. Medido en una H100, el mismo modelo sin adaptador varió **0.015–0.020** en cinco ejecuciones, frente al umbral predeterminado de 0.05, con cuatro de seis diferencias pareadas de esa sesión *dentro* de la dispersión. `soup ship` comparaba con 0.05 sin indicar jamás qué podía resolver su propio instrumento.

`--noise-floor N` vuelve a ejecutar el modelo **base** N veces (N en `[2, 10]`), calcula el `max − min` de cada eje entre las repeticiones, lo muestra junto al veredicto y aplica a cada eje el umbral `max(--forgetting-threshold, that axis's floor)`. La mejora del componente 1 también debe superar el piso del eje de tarea.

```bash
soup ship --base <m> --adapter ./out --task-eval tasks.jsonl --noise-floor 2
```

Salida real (par SmolLM2-135M, CPU, `--general-suite mini_mmlu`):

```
noise floor: base repeat 1/2
noise floor: base repeat 2/2
noise floor: every axis repeated exactly — this instrument was deterministic over these runs.

 Leg 2 general suite (threshold 5.00%)
 ┌───────────┬────────┬────────┬─────────┬───────────┐
 │ Benchmark │   Base │  Tuned │       Δ │ Verdict   │
 ├───────────┼────────┼────────┼─────────┼───────────┤
 │ mini_mmlu │ 0.2692 │ 0.1154 │ -0.1538 │ REGRESSED │
 └───────────┴────────┴────────┴─────────┴───────────┘

 Noise floor
 ┌────────────┬────────┐
 │ Axis       │  Floor │
 ├────────────┼────────┤
 │ leg 1 task │ 0.0000 │
 │ mini_mmlu  │ 0.0000 │
 └────────────┴────────┘
Measured over 2 base repeats. Each axis is gated at max(threshold, its floor).
```

En CPU, el piso es 0.0000: la decodificación greedy es determinista allí y un piso de 0.0 correctamente no suprime nada (la regresión anterior sigue detectándose). Se espera un piso distinto de cero en GPU.

**El `max` importa en ambas direcciones.** Un piso *superior* al umbral amplía la puerta hasta lo que realmente se puede medir; uno *inferior* nunca debe estrecharla a tus espaldas. Si un piso supera `--forgetting-threshold`, la ejecución lo indica por nombre: ese eje ahora tiene una puerta más flexible de lo que pediste.

**Alcance y costo.** El piso del componente 1 se mide en **todos** los `--task-mode`. En `metric`, vuelve a ejecutar el evaluador sin conexión (ruido sólo de decodificación). En `judge_score`, la parte base se evalúa N veces mediante el evaluador; en `pairwise`, el modelo base se compara **consigo mismo** (tasa de victorias esperada 0.5 por construcción, así que la dispersión observada es una medición directa y no una inferencia). Los dos modos de evaluador incorporan el propio ruido de muestreo del evaluador, por lo que ese piso aparece como **decode + judge** en el panel y lleva la etiqueta `judge_inclusive` en la evidencia/JSON; el lector no debe confundirlo con los pisos de sólo decodificación de los ejes del componente 2. Costo: N pasadas adicionales del modelo base y, en los modos de evaluador, N × llamadas a la API del evaluador. Se rechaza junto con `--evidence` (no hay nada que volver a ejecutar), aunque sí se aplica un piso *registrado en* un archivo de evidencia.

**Advertencia, heredada de la medición que motivó este cambio:** n=3, un modelo, un dataset. El piso **dimensiona** el efecto; **no** calibra un umbral y nada establece cuál es el valor suficiente de N.

El esquema JSON de evidencia ahora incluye un bloque opcional `noise_floor`, por lo que un veredicto decidido con un piso se reproduce igual sin conexión. El eje de tarea del componente 1 usa la clave `__task__`:

```json
{
  "task": {"mode": "metric", "base": 0.3333, "tuned": 0.3333},
  "benchmarks": {"mini_mmlu": {"base": 0.2692, "tuned": 0.1154}},
  "noise_floor": {"runs": 2, "floors": {"__task__": 0.0, "mini_mmlu": 0.0}}
}
```

Un bloque `noise_floor` mal formado se **rechaza, no se descarta**: si se descartara silenciosamente, la reproducción daría un veredicto distinto. Los valores se limitan a `[0, 1]` y el mapa tiene un tamaño máximo, porque un archivo de evidencia es una entrada no confiable y un piso amplía la puerta.

**La misma regla cubre ahora los campos desconocidos en cualquier parte del esquema (#758).** Se **rechaza, no se ignora**, un archivo de evidencia que contenga un campo que esta versión no reconoce:

```bash
$ soup ship --evidence evidence.json
Error: evidence has unsupported field(s): 'future_optional'
$ echo $?
1
```

Antes, ese campo se descartaba silenciosamente y la ejecución salía con `0`. El cambio cierra este defecto: `soup ship --evidence` y la herramienta MCP `ship_evidence` leían el mismo archivo mediante lectores independientes, así que una clave entendida por uno pero no por el otro hacía que la *misma* evidencia se reprodujera con un veredicto *distinto*, según qué interfaz la consultara: PUBLICAR de un lado y NO PUBLICAR del otro. Ambas interfaces ahora leen mediante un lector compartido, y rechazar los campos no reconocidos mantiene honesta esa garantía: ambas interfaces admiten un campo o ninguna lo acepta. El rechazo se puede corregir y es visible; un veredicto divergente no. Si una versión más reciente de Soup escribe un archivo de evidencia que una versión anterior rechaza, actualiza el lector en lugar de eliminar el campo. `numerics` es una etiqueta admitida desde #746 —ambas interfaces la leen—, así que un archivo que la incluya ya no se rechaza por esa clave.

### Cierre del ciclo de evidencia (v0.71.39)

Ahora se puede emitir y confirmar el veredicto, vinculado a su procedencia, para que una puerta de fine-tuning se ejecute en cada PR en vez de depender de un archivo JSON editado a mano.

- **`--emit-evidence <path>`** vuelve a serializar las puntuaciones en el esquema de ENTRADA `--evidence`, de modo que la salida de una ejecución se puede reproducir como entrada: al volver a pasarla por `--evidence` (con el mismo umbral) se obtiene un veredicto idéntico.
- **`--config soup.yaml`** lee un bloque `eval.ship` confirmado para los valores predeterminados de la puerta (`task_eval` / `task_mode` / `general_suite` / `forgetting_threshold` / `judge_model` / `baseline` / `noise_floor`); siempre prevalece una bandera explícita de CLI. Así, la puerta se puede revisar en una diferencia de PR. `noise_floor` es una entrada de medición en vivo como la bandera `--noise-floor`: se mide cuando una ejecución en vivo genera evidencia y, al igual que la bandera, se rechaza con `--evidence` (no hay nada que volver a ejecutar sin conexión; sí se aplica un piso ya registrado en la evidencia).
- **Procedencia y obsolescencia.** Con `--emit-evidence`, `--config` AÑADE una etiqueta `provenance` (`config_sha`, un hash semántico de la receta que no depende del orden y EXCLUYE la política de puerta `eval.ship`, por lo que ajustar el umbral nunca invalida la evidencia, además de `base_model` y un `data_sha` de mejor esfuerzo). Una ejecución en vivo también añade la etiqueta superior `numerics` (`4bit` / `8bit` / `bfloat16` / `float32`), que indica la carga real, no el campo de entrenamiento; así, si el evaluador cargó en bf16 una receta GPTQ, queda indicado. Sólo con `--evidence`, `--config` APLICA LA PUERTA: rechaza (salida 3) la evidencia cuyo `config_sha` difiera de la configuración confirmada o cuya *familia* de precisión (`4bit` / `8bit` / `full`) no coincida. La evidencia anterior a #367 sin etiqueta genera una advertencia en lugar de fallar de forma cerrada.
- **`--push owner/repo#N`** publica el veredicto como comentario de un PR de GitHub (en la medida de lo posible: si falta el token o falla `gh`, muestra una advertencia, pero nunca cambia el código de salida PUBLICAR / NO PUBLICAR).

```bash
# Producer (train job): compute or stamp scores, bound to the committed recipe
soup ship --evidence scores.json --config soup.yaml --emit-evidence ship_evidence.json

# Gate (PR CI): refuse evidence that doesn't match the committed config, comment the verdict
soup ship --evidence ship_evidence.json --config soup.yaml --push owner/repo#42
```

`soup ci init --config soup.yaml` vincula el paso ship del flujo de trabajo generado con la configuración confirmada para que todo el ciclo se ejecute en CI (consulta [commands.md](commands.md)).

**Procedencia de la línea base (#404).** Genera una línea base con sello usando `soup eval gate --suite <suite.yaml> --model <id> --write-baseline baseline.json` (`--model` es obligatorio: nunca se escriben líneas base a partir del generador stub). El archivo tiene el formato `{"scores": {...}, "provenance": {"soup_version", "scorer_revision"}}` y luego se usa con `soup ship --baseline baseline.json` o `soup eval gate --baseline baseline.json`. Los helpers compartidos `stamp_baseline_scores` / `write_baseline_file` son los únicos escritores; `write_baseline_file` rechaza un mapa de puntuaciones vacío. Las puntuaciones de línea base deben ser números finitos: los valores no finitos (`NaN`, `+inf`, `-inf`) se rechazan en los límites de lectura (`resolve_baseline`), escritura (`stamp_baseline_scores`, `write_baseline_file`) y puerta (`run_gate`) (#1773). `resolve_baseline` muestra una sola advertencia si la procedencia es desconocida (archivos sin sello / filas del registry) o si no coincide `scorer_revision`, y no muestra ninguna si el sello coincide. Se eliminó la antigua advertencia `SCORER_CHANGED_IN_V0_73_2`, basada en el nombre. Vuelve a calcular una línea base antigua o elimina nombres de ella para forzar una ejecución en vivo del modelo base.

Nota histórica: v0.73.2 modificó tres evaluadores de suites (`mini_mmlu`, `mini_common_sense`, `mini_tool_call`) con saltos medidos en un modelo sin cambios, muy superiores a la puerta de 0.05. Por eso los archivos sin sello siguen generando advertencias.


## Métricas de evaluación NLG (BLEU + ROUGE)

BLEU + ROUGE-1 / ROUGE-2 / ROUGE-L en Python puro para `soup eval custom`:

```python
from soup_cli.utils.nlg_metrics import (
    bleu_score, rouge_l_score, compute_nlg_metric, NLG_METRICS,
    effective_tokens_per_second,
)

bleu_score(["the cat sat on the mat"], ["the cat sat on the mat"])
# 1.0
rouge_l_score(["the quick brown fox"], ["a quick brown dog"])
# 0.5
compute_nlg_metric("rouge_2", preds, refs)
# generic dispatch by canonical name

effective_tokens_per_second(unmasked_tokens=12_500_000, wall_clock_seconds=600.0)
# 20833.33  — None when wall_clock <= 0 (no fabrication)
```

BLEU suavizado usa el epsilon de Chen & Cherry para grupos con cero coincidencias cuando `total[n] > 0`; los grupos vacíos (por ejemplo, predicciones con menos de `max_n` tokens) obligan a que la puntuación sea 0.0.


## Calibración de cuantización (divergencia KL)

Compara un modelo cuantizado con una línea base de precisión completa en un conjunto pequeño y fijo de prompts. Los umbrales OK / MINOR / MAJOR de 0.05 / 0.20 de KL media usan la misma escala que `soup eval quant-check`.

```python
from soup_cli.eval.calibrate import run_calibration

# baseline_logits / quantized_logits: list[list[float]] aligned per-prompt
report = run_calibration(baseline_logits, quantized_logits)
print(report.delta_status, report.mean_kl)
# OK 0.012
```

El núcleo usa sólo operaciones matemáticas, evalúa log-probabilidades estables con log-softmax sin imponer un piso artificial a la probabilidad y está limitado a 10 000 prompts para proteger contra OOM accidentales. `CalibrationReport` es un dataclass congelado.


## Arena de modelos (torneo Elo)

Tabla de clasificación local con puntuaciones Elo (K=32, base 1500). Aporta tus propios ganadores por pares; Soup sólo lleva el registro:

```python
from soup_cli.eval.arena import Tournament

t = Tournament()
t.record("llama-3.1-8b-finetune", "qwen2.5-7b-finetune", winner="a")
t.record("llama-3.1-8b-finetune", "mistral-7b-finetune", winner="draw")
for row in t.leaderboard():
    print(row)
```

Límites: 256 modelos por torneo, 1M de partidas. Se rechazan los nombres de modelo con los caracteres `[` o `]` para que las filas de la tabla de clasificación no puedan inyectar marcado.


## Evaluación de modelos

Plataforma de evaluación completa con benchmarks estándar, evaluaciones personalizadas, LLM como juez y evaluación humana:

```bash
# Install eval dependencies
pip install "soup-cli[eval]"

# Standard benchmarks (wraps lm-evaluation-harness)
soup eval benchmark --model ./output --benchmarks mmlu,gsm8k,hellaswag

# Aider Polyglot code-editing benchmark (after the setup below)
soup eval aider --model openai/gpt-4.1 --output ./aider-results \
  --exercises-dir ./polyglot-benchmark --run-id run_20260301_143052_a1b2

# Custom eval tasks from JSONL
soup eval custom --tasks eval_tasks.jsonl --model ./output

# LLM-as-a-judge (score model outputs using GPT-4o, Ollama, etc.)
soup eval judge --target responses.jsonl --model gpt-4o-mini --provider openai
soup eval judge --target responses.jsonl --model llama3.1 --provider ollama
# A row whose judge call keeps failing (429 after the retries, a timeout, a reply with no
# choices) is skipped and counted; the other rows are still scored, shown and saved (#1447).
# A judge that is down stops the run: once 3 requests in a row have spent their retries
# unanswered, `soup eval judge` prints one message naming the judge URL and how many items
# were not judged, shows the scores from before the outage without saving them, and exits 1,
# so a long run costs 9 s of backoff, not 3 s per row. One reply resets the count, and a 4xx
# or an unusable reply counts as a reply. `soup data from-traces --judge` stops the same way
# and writes no output; `soup data best-of-n` and `soup ship --task-mode pairwise`, which
# already stopped at the first unanswered request, now say how many prompts or pairs were
# left. A judge-ranked training run is unchanged: it keeps asking, and resumes on the first
# reply (#1522).

# Auto-eval after training (configure in soup.yaml)
soup eval auto --config soup.yaml

# Compare eval results between two training runs
soup eval compare run_20260301_143052_a1b2 run_20260315_091023_c3d4

# Local leaderboard across all evaluated models
soup eval leaderboard
soup eval leaderboard --format json
soup eval leaderboard --format csv

# Human A/B evaluation with Elo ratings
soup eval human --input prompts.jsonl --model-a ./model_a --model-b ./model_b
```

`soup eval benchmark` guarda únicamente los nombres solicitados que tengan métricas numéricas, incluido un `0.0` real (no un booleano). Los resultados faltantes o sin puntuación se omiten, se identifican y se cuentan; el banner guardado informa el total guardado. Sale con `0` si se guardó al menos un resultado, o con `1` si no se midió ni guardó nada. Un grupo solicitado necesita su propia entrada numérica en `results`; este comando no guarda automáticamente sus filas de subtareas.

`soup eval judge` pide al juez un objeto JSON, `{"scores": {"<criterion>": <number>, ...}, "reasoning": "..."}`, y lee el primer objeto de la respuesta que tenga una clave `scores`, de modo que el razonamiento que incluya su propio JSON no interfiera. Los nombres de criterios no distinguen mayúsculas de minúsculas y una puntuación dada como `{"score": N}` se lee como `N`. Una respuesta sin objeto JSON, sin un número finito para un criterio de rúbrica o con un criterio escrito de dos formas (`Helpfulness` y `helpfulness`) es un error que identifica el criterio cuando corresponde, en lugar de asignarle la puntuación mínima de la escala. Se rechaza sin leer una respuesta de más de 65,536 caracteres; se pide al juez un máximo de 1,024 tokens, así que sólo una respuesta descontrolada alcanza esa longitud. `soup eval judge` omite ese elemento con una advertencia que incluye el error y lo cuenta (sale con `1` sólo si fallan todos los elementos); una tarea de juez en eval-gate falla con el error como motivo (`soup eval gate` sale con `2`), y `soup ship --task-mode judge_score` se detiene con `live ship verdict failed: ValueError: ...` y sale con `1` sin emitir veredicto.

soup eval compare y soup eval leaderboard usan la fila de evaluación más reciente para cada benchmark (en la tabla de clasificación, para cada modelo y benchmark), de modo que volver a ejecutar un benchmark reemplaza su puntuación; si las filas tienen el mismo created_at, se resuelve a favor de la inserción posterior. soup registry diff compara las puntuaciones de evaluación del mismo modo.

### Aider Polyglot

El paquete `aider-chat` no incluye el arnés de benchmarks de Aider. El extra `[aider]` instala el CLI normal de Aider, pero no basta para ejecutar `soup eval aider`. Compila la imagen oficial desde una copia del código fuente de Aider y clona los ejercicios una vez:

```bash
pip install "soup-cli[aider]"
git clone https://github.com/Aider-AI/aider.git
cd aider
./benchmark/docker_build.sh
cd ..
git clone https://github.com/Aider-AI/polyglot-benchmark.git
```

Inicia Docker y luego ejecuta Soup desde el proyecto cuyo directorio de salida contenido debe recibir los resultados:

```bash
soup eval aider \
  --model openai/gpt-4.1 \
  --output ./aider-results \
  --exercises-dir ./polyglot-benchmark \
  --run-id run_20260301_143052_a1b2
```

`--model` es un identificador de modelo de Aider/LiteLLM, no una ruta a un modelo local de Hugging Face. Soup comprueba que estén disponibles el CLI de Docker, el daemon y la imagen local `aider-benchmark` antes de empezar. Monta el corpus de ejercicios como sólo lectura, reenvía las credenciales admitidas del proveedor por nombre de variable de entorno (nunca por valor en los argumentos del comando) y ejecuta el arnés upstream sin un shell. El directorio de salida debe resolverse dentro del directorio de trabajo actual.
Se permite que el directorio de ejercicios esté fuera del directorio de trabajo actual, pero se muestra una advertencia y permanece en modo de sólo lectura dentro del contenedor.

El acceso al loopback del host está deshabilitado por defecto. Para un endpoint local compatible con OpenAI y expresamente confiable, `--allow-host-services` agrega la asignación `host.docker.internal:host-gateway` de Docker. Al habilitarlo, también se permite que código generado por modelos no confiables acceda a otros servicios que escuchen en el host, así que déjalo desactivado para proveedores remotos.

Aider escribe un `.aider.results.json` por ejercicio. Soup limita y valida esos archivos, y luego escribe `soup_result.json` con `model`, `task`, `score`, `errors` y detalles agregados. Si también se proporciona un `--run-id` existente, se guarda la puntuación `aider_polyglot` en el tracker de experimentos de Soup, así que participa en el comando normal de comparación:

```bash
soup eval compare run_before run_after
```

El benchmark ejecuta código generado por modelos. Mantén habilitado el aislamiento de Docker; Soup no ofrece deliberadamente una alternativa que ejecute código en el host.

### Comprobador Quant-Lobotomy

Antes de publicar un modelo cuantizado, verifica que no haya perdido habilidades. El comprobador ejecuta la misma lista de tareas con los modelos `--before` y `--after` y presenta un veredicto agregado de la suite OK / MINOR / MAJOR.

```bash
# Compare a pre-quant model with its post-quant version
soup eval quant-check \
  --before ./output/base \
  --after  ./output/quantized \
  --tasks  ./evals/sanity.jsonl

# Both sides may be registry refs
soup eval quant-check \
  --before registry://llama31-chat-v1 \
  --after  registry://llama31-chat-v1-q4 \
  --tasks  ./evals/sanity.jsonl

# Render as JSON for CI integration (exits 2 on MAJOR, 0 on OK/MINOR)
soup eval quant-check --before X --after Y --tasks t.jsonl --format json

# Use deterministic stubs in CI when weights are unavailable
soup eval quant-check --before X --after Y --tasks t.jsonl --allow-stub
```

**Umbrales de veredicto (veredicto agregado de la suite):**
- `OK` — caída de puntuación < 2% (o puntuación mejorada)
- `MINOR` — caída de puntuación de 2–5% (investigar)
- `MAJOR` — caída de puntuación ≥ 5% (NO publicar; sale con código 2)

Las rutas se comprueban para garantizar que estén contenidas, y las referencias `registry://` se resuelven con un filtro `kinds` opcional para evitar elegir el artefacto equivocado. Las rutas de archivos `.gguf` independientes se rechazan desde el inicio; pasa rutas de directorio que contengan pesos safetensors/HuggingFace o referencias `registry://`.

### Formato de evaluación personalizada

Las tareas personalizadas puntuadas con regex rechazan patrones estructuralmente inseguros antes de comparar la salida del modelo e identifican `eval.custom.expected` en el error. La prueba de `format` de diagnose aplica la misma comprobación a `regex_pattern` sin ejecutar una búsqueda canario. Los patrones normales, incluida la alternancia de un solo carácter, siguen siendo válidos.

```jsonl
{"prompt": "What is 2+2?", "expected": "4", "category": "math", "scoring": "answer"}
{"prompt": "Explain gravity", "expected": "force.*attraction", "scoring": "regex"}
{"prompt": "Capital of France?", "expected": "Paris", "scoring": "contains"}
```

Cómo leen la salida los evaluadores de cadenas:

| `scoring` | Puntuaciones | `expected: "4"` frente a `The answer is 4.` | frente a `14` |
|---|---|---|---|
| `answer` | La respuesta final que declara cada lado, leída con el parser que usa la recompensa `accuracy` de GRPO (`#### 4`, `\boxed{4}`, `The answer is 4.`); los números se comparan por valor | coincidencia | sin coincidencia |
| `contains` | `expected` como token alfanumérico completo, sin distinguir mayúsculas mediante `str.lower()` en ambos lados (por lo que `istanbul` no coincide con `İstanbul`, ni `s` con `ſ`) | coincidencia | sin coincidencia |
| `exact` | Toda la salida sin espacios en los extremos, sin distinguir mayúsculas | sin coincidencia | sin coincidencia |

En el momento de cargar, `answer` rechaza un `expected` que no exprese una sola respuesta (una solución desarrollada en varias líneas sin `####`, un `\boxed{}` vacío o una ambigüedad entre valores) e identifica la línea.
`contains` no puede leer negaciones: `It is not 4, so 5.` sigue conteniendo `4`. Su límite es una secuencia alfanumérica ASCII, así que `4` sigue coincidiendo dentro de `3.4`, `4.5` y `v1.42.3`. `answer` lee la respuesta que *declara* el modelo: `The capital of France is Paris.` obtiene `False` con la respuesta de referencia `Paris`, mientras que `Answer: Paris` y `Paris` obtienen `True`; prefiere `answer` para matemáticas y tareas de respuesta breve, donde el modelo declara su respuesta. La opción `scorer:` de eval-gate también acepta `answer`.

**Cambio:** antes, `contains` era una búsqueda de subcadena sin procesar, así que daba por buena `14` cuando se esperaba `4`. Las puntuaciones guardadas con el evaluador anterior (un archivo `soup eval gate --baseline`, una fila `registry://` o un resultado `soup eval custom --run-id`) pueden ser más altas que las que obtienen ahora las mismas salidas. El sello del evaluador de línea base cubre sólo las suites incluidas y no lo advertirá: vuelve a medir esas líneas base o compara únicamente ejecuciones nuevas realizadas con esta versión.

### Configuración de evaluación automática (soup.yaml)

Los benchmarks requieren el extra `eval`:

```bash
pip install "soup-cli[eval]"
```

```yaml
eval:
  auto_eval: true
  benchmarks: [mmlu, gsm8k]
  custom_tasks: eval_tasks.jsonl
  judge:
    model: gpt-4o-mini
    provider: openai
```

Cuando `eval.auto_eval: true`, Soup ejecuta la evaluación configurada una vez que termina `soup train` y se guarda el modelo/adaptador entrenado. La evaluación usa el artefacto de salida producido por esa ejecución de entrenamiento.

- `benchmarks` ejecuta los benchmarks `lm-eval-harness` configurados. Instala antes el extra `soup-cli[eval]` indicado arriba.
- `custom_tasks` ejecuta las tareas de evaluación personalizadas configuradas con el modelo entrenado.
- `judge` no forma parte de la evaluación automática; configura y ejecuta la evaluación del juez por separado.
- La evaluación automática sigue la bandera `--trust-remote-code` de la ejecución: `soup train --trust-remote-code` la pasa a la evaluación del modelo entrenado (benchmarks y tareas personalizadas); sin la bandera, permanece desactivada. No hay una clave `soup.yaml` para esto.
- Si falla la evaluación automática, Soup informa del fallo, pero no marca como fallida una ejecución de entrenamiento que tuvo éxito.
- La evaluación automática puede tardar bastante. Por ejemplo, MMLU evalúa varias subtareas.


## Prueba de capacidad de ajuste (`soup tunability`)

Antes de comprometerte con un único modelo base, ejecuta una prueba breve de LoRA con cada candidato razonable sobre tu segmento reservado. v0.64.0 incluye un catálogo predeterminado de 8 entradas que cubre Qwen3, Llama-3.2, Gemma 3, Phi-4, SmolLM3 y Qwen2.5 en el rango de 0.6 B – 3.8 B.

```bash
# List the built-in catalogue
soup tunability --list

# Dry-run a sweep across a subset
soup tunability --dataset ./eval.jsonl --candidates qwen3-0.6b,phi-4-mini --plan-only

# Run the full sweep + write a JSON report
soup tunability --dataset ./eval.jsonl --probe-steps 100 --output ./tunability.json
```

El informe es una frontera de Pareto entre (delta de evaluación respecto a la base, costo de entrenamiento, licencia): sobreviven los candidatos que nadie domina en ambos ejes, así ves una lista breve y clara en vez de una clasificación ruidosa de una sola puntuación. Por defecto, la prueba es una heurística determinista sin conexión; pasa `--live` (v0.71.7) para ejecutar una prueba LoRA real por candidato (carga cada `repo_id`, entrena durante `--probe-steps` en un segmento que excluye los datos reservados e informa la caída de pérdida en ellos). `--device` selecciona cuda / cpu.


## Profundidad de evaluación (`soup eval behavior / capability / checklist / irt-subset`)

v0.65 incorpora cinco nuevas superficies de evaluación que cierran las brechas de «los jueces son parciales, las suites son arbitrarias y las evaluaciones son costosas» que las evaluaciones SaaS (Galileo, Braintrust) no abordan.

**Calibración del juez** — no uses un juez sin calibrar en producción:

```python
from soup_cli.eval.calibrate import (
    PairwiseJudgement, run_pairwise_calibration, ensure_judge_calibrated,
)

# Run your judge on a calibration set with positions swapped.
judgements = [PairwiseJudgement(...) for _ in oracle_set]
report = run_pairwise_calibration(judgements, scores=confidence_scores)
ensure_judge_calibrated(report)  # raises RuntimeError if not calibrated
```

El informe incluye `position_bias` ∈ [-1, 1] (0 = sin preferencia por una posición), un umbral de abstención conformal derivado del cuantil de las puntuaciones, la tasa de concordancia frente al oráculo y un booleano `calibrated`. `ensure_judge_calibrated` rechaza un informe faltante, una concordancia baja o un sesgo extremo, para que el código de puntuación en producción falle de forma visible y no silenciosa.

Guarda una calibración una vez y reutilízala entre ejecuciones (v0.71.1): `write_judge_calibration(report, "calib.json")` escribe un JSON contenido en cwd, y `load_judge_calibration("calib.json")` lo vuelve a validar al cargar (se rechaza un campo corrupto o fuera de rango en disco, ya que es la red de seguridad de la puerta de producción). El artefacto se adjunta al registry bajo el nuevo tipo `judge_calibration`.

**Batería de comportamiento** — comparación antes/después en conjuntos incluidos de pruebas de seguridad / rechazo / adulación:

```bash
# Score over-refusal regression on XSTest (operator supplies evidence JSON)
soup eval behavior my_run --battery xstest --evidence ev.json --output diff.json

# LIVE (v0.71.7): generate pre/post responses on the bundled battery + score the diff
soup eval behavior my_run --battery xstest \
    --base-model HuggingFaceTB/SmolLM2-135M --adapter ./out

# Bundled batteries: xstest, harmbench, jailbreakbench (live or --evidence);
# elephant, syceval (--evidence only)
# Harmful prompts ship REDACTED — pull real sets from upstream papers.
```

Sin la ruta activa `--base-model`, se requiere `--evidence`. Una ejecución sin evidencia es un error de entrada (código de salida `3`), no un informe neutral OK; el error identifica las matrices `pre_responses`, `post_responses` y `oracle` esperadas.

La puntuación depende de la ruta. Con `--evidence`, cada entrada `oracle` es una palabra que debe aparecer en la respuesta correspondiente (como palabra completa y sin distinguir mayúsculas, de modo que `"safe"` no coincide con `"unsafe"`, mientras que una etiqueta con puntuación como `(B)` o `-5` sí coincide). Con `--base-model`, cada generación se clasifica como rechazo o no: `safe` (XSTest) y `answer` (JailbreakBench) esperan una respuesta, y `refuse` (HarmBench, JailbreakBench) espera un rechazo. `elephant` y `syceval` (oracle `disagree`) necesitan un juicio que un clasificador de rechazos no puede hacer, por lo que `--base-model` los rechaza antes de cargar un modelo (código de salida `3`); puntúa las generaciones guardadas con `--evidence`. Los umbrales OK/MINOR/MAJOR coinciden con la taxonomía de v0.26 / v0.56.

**Suite automática de capacidades** — selector de perfil incluido con identificadores de tareas `lm-eval-harness` fáciles de usar:

```bash
soup eval capability my_run --suite math --output cap.json   # AIME + MATH-500
soup eval capability my_run --suite code --output cap.json   # HumanEval+ + SWE-bench-Verified
soup eval capability my_run --suite fast --output cap.json   # MMLU-Pro + HumanEval+
soup eval capability my_run --suite full --output cap.json   # all 7 benchmarks

# LIVE (v0.71.7): invoke lm-eval-harness per task against a real model
soup eval capability my_run --live --model HuggingFaceTB/SmolLM2-135M \
    --tasks arc_easy --limit 1 --device cpu
```

Sin `--live`, emite el manifiesto (benchmark, tarea lm-eval); encadénalo con la superficie existente `soup eval benchmark`. Con `--live --model <id>` (v0.71.7), ejecuta lm-eval-harness para cada tarea resuelta —o una sobrescritura `--tasks`—, aislando los fallos por tarea y limitando los ejemplos con `--limit`.

**DSL conductual CheckList** — pruebas MFT / INV / DIR de Ribeiro et al. 2020:

```yaml
# tests.yaml
tests:
  - name: capital-france
    kind: mft
    prompts: ["What is the capital of France?"]
    expected: ["paris"]
  - name: paraphrase-add
    kind: inv
    prompts:
      - "Add 2 and 2."
      - "Add two and two."
```

```bash
soup eval checklist tests.yaml --evidence responses.json
```

Se requiere `--evidence` y asigna a cada nombre de prueba CheckList sus cadenas de respuesta. Omitirlo termina con código `3` en vez de presentar un resultado OK con cero mediciones.

`mft` = la respuesta debe contener una palabra clave completa (`"sand"` no pasa para `"and"`, y las palabras clave pueden empezar o terminar con puntuación, como `-5`, `(B)` o `C++`); `inv` = todas las paráfrasis deben concordar; `dir` = expectativa direccional ante una perturbación.

**Selección de subconjunto IRT** — elige un conjunto de evaluación más pequeño que conserve la capacidad de clasificación:

```bash
# Pick top-info 30% of items (5-10x eval-bill cut without losing power)
soup eval irt-subset per_item_correctness.jsonl --size small --output plan.json

# Richer item models: 2PL learns per-item discrimination, 3PL adds a guessing floor
soup eval irt-subset per_item_correctness.jsonl --size small --model 2pl
```

`--model` elige el modelo de respuesta al ítem (por defecto `1pl`). `1pl` es el ajuste de Rasch en forma cerrada (`β̂_i = -log(p̂_i / (1 - p̂_i))`); `2pl` agrega un parámetro de discriminación por ítem y `3pl` un piso de adivinación, ambos mediante MLE de ascenso coordinado conjunto (v0.71.6). Los ítems se clasifican por información de Fisher en θ=0 (`p̂ · (1-p̂)` en 1PL: es máxima para ítems 50/50, ya que los extremos no aportan nueva información de clasificación). `full` conserva el 100%, `small` el 30% y `tiny` el 10%.
