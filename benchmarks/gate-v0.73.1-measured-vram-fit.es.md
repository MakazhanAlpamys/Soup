# Registro del gate — v0.73.1: medir el ajuste de VRAM en streaming en vez de predecirlo (#349)

> 🇪🇸 Traducción al español de [`gate-v0.73.1-measured-vram-fit.md`](gate-v0.73.1-measured-vram-fit.md). Ante cualquier duda, manda el original en inglés.

**Hardware.** RTX 3050 Laptop GPU, 4 GB (4.294 GB en total, 3.45 GB libres en reposo),
Windows 11, NVMe. torch 2.5.1+cu121, transformers 4.57.6, trl 0.19.1, peft 0.18.1,
Python 3.10. Todas las cifras siguientes se midieron con esa configuración. Nada aquí
se midió en otro lugar, y nada se extrapola a otro hardware.

**Qué se estaba probando.** La comprobación previa de streaming por capas predice la VRAM máxima a partir de
una fórmula ajustada a 10 ejecuciones reales (v0.72.3, `benchmarks/gate-v0.72.3-breadth.md`)
y rechaza una ejecución que predice que no cabrá. El contrato documentado es que
**nunca predice por debajo**, porque ese error no se anuncia por sí solo: un OOM en Linux,
un desbordamiento silencioso a la memoria del host en Windows/WDDM.
El issue #349 propuso sustituir la predicción por una medición. Este registro
intenta determinar si eso se justifica.

---

## Resultado principal

**La fórmula predice por debajo con secuencias largas, medido a través del `soup train`
real.** SmolLM2-135M transmitido en bf16, batch 1, `quantization: none`,
LoRA r=8, `stream_buffers: 2`. «Pico real» es `torch.cuda.max_memory_allocated()`
en todo el proceso de entrenamiento.

| seq | predicción | pico real | pred/real | veredicto |
|---|---|---|---|---|
| 4352 | 3.282 GB | 3.036 GB | 1.081x | predice por encima — seguro |
| 5120 | 3.844 GB | 4.118 GB | **0.934x** | **predice por debajo** |
| 6144 | 4.590 GB | 5.830 GB | **0.787x** | **21% por debajo** |

El cruce se sitúa entre seq 4352 y 5120. Se permitió continuar las ejecuciones con 5120 y 6144 mediante
`training.stream_vram_override` para observar el pico real; la predicción mostrada es la que imprimió
la comprobación previa publicada durante esa misma ejecución.

**La prueba existente no podría haberlo detectado.** Las diez filas de
`MEASURED_VRAM_GRID` están en seq 256 o 512: la cuadrícula varía *batch* (1..8) y
nunca la longitud de secuencia. Por tanto, `test_never_under_predicts` fijaba «nunca
predice por debajo al aumentar el batch» aunque se leía como una garantía global. Se ha
acotado para que lo diga, con una aserción que falla si se agrega alguna fila más larga
sin volver a establecer la afirmación.

---

## Tres interpretaciones erróneas, que se conservan porque costaron tiempo real

### 1. «La predicción se excede por 13.2x» — RETIRADO

Primera reproducción: se rechazó una configuración DPO con batch 2 x seq 2048 (predicción de
6.09 GB frente a 3.45 GB libres) y luego se completó en 7.3 s usando 0.462 GB al
forzarla. Una predicción excesiva de 13.2x.

Era un artefacto del fixture. Las filas materializaron **142 tokens frente a un
presupuesto de 2048**: el 6.9% de la forma para la que se presupuesta la fórmula, y 2048/142 = 14.4x,
que explica toda la brecha aparente. La predicción presupuesta la `max_length` *configurada*,
porque ese es el peor caso al que puede rellenarse un batch real. Al volver a ejecutar
con filas que realmente llenan 2048 tokens, la misma configuración predice 6.09 GB y
mide 5.304 GB: 1.15x, no 13.2x.

### 2. «Las pérdidas de preferencias se presupuestan ~8.8x por encima» — NO REPRODUCIDO

Las notas de v0.72.4 registran el costo de DPO por encima de lo residente como 51.76 MB, cuando un cargo de 14 B/elemento
para la misma forma es de ~458 MB. Esa proporción no aparece con la forma presupuestada: DPO predice 6.09 GB frente a
5.304 GB medidos (1.15x), junto a SFT con 3.095 frente a 3.051 (1.014x): el mismo margen,
no uno específico de preferencias. Las dos cifras miden cantidades diferentes (costo marginal por encima de lo residente frente a
pico total), así que no hay contradicción, pero la negativa práctica excesiva que implicaría
no está presente y no debería citarse como tal.

### 3. «Las ejecuciones fuera de presupuesto estaban desbordándose en silencio» — RETIRADO

`max_memory_reserved` llegó a 4.316 GB en una tarjeta de 4.294 GB, lo que parecía el
desbordamiento WDDM documentado. No es evidencia de uno: `num_alloc_retries` fue **0** en
todas las formas medidas, así que el asignador de caché nunca estuvo bajo presión. La memoria reservada
supera lo que debe caber porque conserva bloques liberados que devuelve bajo
presión. Por eso el gate publicado lee `max_memory_allocated`.

---

## Por qué el gate lee `allocated`, no `reserved`

La memoria reservada fue de 1.08x-1.41x la asignada en las formas medidas: no es una constante,
por lo que no se puede modelar. Más decisivamente, **usar la memoria reservada como criterio rechazaría la
configuración emblemática de esta misma función**: Llama-3.1-8B NF4 con batch 1 x seq 512
mide 3.4116 GB asignados frente a 3.6973 GB reservados, con 3.45 GB libres, y
se ejecuta. Un gate que rechaza la configuración principal no es más seguro.

---

## La sonda no es el paso de entrenamiento y es conservadora

`measure_step_peak_bytes` ejecuta un forward+backward causal-LM simple con IDs
de tokens sintéticos en la forma configurada. Frente al paso de entrenamiento real:

| seq | sonda | ejecución real | sonda/real |
|---|---|---|---|
| 4352 | 3.471 GB | 3.036 GB | +14.3% |
| 5120 | 4.633 GB | 4.118 GB | +12.5% |

Es **consistentemente conservadora**, la dirección correcta para un gate: la
sonda cruza el umbral de rechazo un poco antes de lo que lo haría el paso de entrenamiento. Por eso
el cruce de la propia sonda (entre seq 3072 y 4096) ocurre antes que el de la ruta de entrenamiento
(entre 4352 y 5120).

**La sonda no está validada para una pérdida de preferencias, y una afirmación hecha durante este
trabajo era errónea.** Un borrador anterior de este registro decía que la sonda medía por debajo un
paso DPO en un 42% y que, por lo tanto, no era segura. Eso comparaba dos *formas diferentes*:
la sonda con filas 4 x seq 1024 frente a DPO con filas 4 x seq 2048. Medida en la
misma forma, la sonda registra **6.021 GB frente a los 5.304 GB del paso DPO real,
es decir, +13.5%**: la misma dirección segura que muestra para SFT, no la opuesta.

De todos modos se mantiene la restricción `task: sft`, con una justificación distinta y más débil:
esa es **una forma, no una validación**. Una pérdida de preferencias reduce los logits
a log-probs por token (`selective_log_softmax`) en vez de mantener una conversión upcast de vocabulario completo,
por lo que la sonda mide un cálculo realmente distinto y esta coincidencia quizá no se mantenga entre formas. Si el signo cambia,
el fallo sería un gate que deja pasar ejecuciones fuera de presupuesto, peor que no tener sonda. Se rechaza al
cargar la configuración hasta medirla en una cuadrícula, no porque se sepa que es
insegura.

## Costo de la sonda

| modelo | forma | tiempo de sonda |
|---|---|---|
| SmolLM2-135M bf16 | 1 x 1024 | 1.02-1.15 s |
| SmolLM2-135M bf16 | 2 x 2048 | 5.09-5.13 s |
| Llama-3.1-8B NF4 | 1 x 512 | 5.33 s (+20.8 s de construcción del modelo) |
| SmolLM2-135M bf16 | 1 x 5120 (sobre presupuesto) | 51.72 s |

La última fila es el peor caso realista: una forma que no cabe tarda en medirse precisamente porque
está oscilando entre memoria. Pagarlo una vez para rechazar la ejecución es mejor que
pagarlo en cada paso.

## Mecanismo: NO establecido

No se ha identificado **qué hace divergir la fórmula por encima de ~4700 tokens**. `seq**2`
de la matriz de puntuaciones de atención es el candidato obvio; el exceso medido no
se ajusta claramente a eso, y no se confirmó ninguna alternativa. Se registra como desconocido
 en vez de adivinar. Este también es el argumento a favor de medir en lugar de agregar
otro coeficiente: una fórmula no puede modelar un término que nadie ha identificado.

---

## Verificación integral en vivo (a través del `soup train` real)

**Dirección de rechazo** — SmolLM2-135M, seq 5120, `stream_vram_override: 4000000000`:

- sonda desactivada: predicción de 3.85 GB frente a un presupuesto de 4.00 GB -> **aceptada**, la ejecución
  continúa, pico real **4.118 GB** (sobre presupuesto).
- sonda activada: misma configuración -> `a streaming step MEASURED 4.63 GB of VRAM at the
  configured shape but only 4.00 GB is free (the formula predicted 3.84 GB)`.

**Dirección de desbloqueo** — seq 3072, `stream_vram_override: 2330000000`:

- sonda desactivada: `a streaming step is predicted to need 2.35 GB of VRAM but only
  2.33 GB is free` -> **rechazada**.
- sonda activada: `Predicted over budget (2.35 GB vs 2.33 GB free), but
  training.stream_vram_probe is on: measuring the real peak before deciding` ->
  `measured peak 2.31 GB ... in 5.23 s` -> la ejecución continúa y se completa.

**Cuatro configuraciones rechazadas, cada una indicando su propia razón**: sonda con
`stream_layers: false`; sonda activada con `task: dpo`; streaming activado con `task: grpo`
(exclusión permanente de rollout); streaming con `batch_size: auto`.

Advertencia sobre la ruta de desbloqueo: la sonda se ejecuta antes del entrenamiento en el mismo proceso, así que
el pico de todo el proceso incluye la sonda y no confirma de manera independiente
el pico del paso de entrenamiento. Las comparaciones limpias entre sonda y ejecución real son las
ejecuciones con sonda desactivada de la tabla anterior.

---

## Revisiones

Cinco revisiones ECC secuenciales; se corrigieron todos los hallazgos.

- **python** — CRITICAL: `_zero_probe_grads` ocultaba excepciones en silencio,
anulando la protección contra corrupción que debía ofrecer (ahora registra una advertencia). HIGH:
`started` podía no estar ligado en la ruta OOM y convertir un rechazo limpio en un
`UnboundLocalError` (ahora se inicializa antes de `try`). MEDIUM: un fallo de sonda
que no fuera OOM descartaba su motivo.
- **code** — HIGH: cambiar `_stream_budget_lines` para devolver una tupla de 2 elementos rompió
`tests/test_issue348_calibrated_logits_wiring.py` y desactivó silenciosamente la
protección de regresión #348. El revisor lo detectó; la ejecución local había informado «exit 0»
porque el código de salida pertenecía a un `tail` canalizado, no a pytest.
- **security** — HIGH: con la sonda activada, un fallo predicho ya no se rechazaba
antes de tocar la GPU, así que un `soup.yaml` compartido podía provocar una
asignación real con una forma absurda (ahora limitada a 4x sobre presupuesto). LOW: `batch_size`
no tenía límite inferior: se aceptaba `-4`. Se corrigieron ambos.
- **tdd** — CRITICAL: el mensaje sin CUDA volvió a romper el stub #348 (corregido).
HIGH: nada demostraba que `setup()` realmente *llamara* a la sonda, y el límite de 4x
no tenía prueba. Ahora ambos están cubiertos; lo primero, mediante el arnés CUDA de extremo a extremo
existente con un control que falla si se ignora la respuesta.
- **verification-loop** — `soup --help` 0.85 s, exit 0; todos los símbolos públicos nuevos
se importan; no hay `torch` de nivel superior en ningún módulo cambiado; `test_cli_startup_is_light`
23/23.
