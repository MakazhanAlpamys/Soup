<!--
Registro de mediciones de trabajo, publicado íntegro. Las dos hipótesis
refutadas por su propia covariable, la forma que terminó mientras asignaba más
memoria que la capacidad de la tarjeta y las cuatro negativas se conservan en el orden en que
ocurrieron.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB según nvidia-smi,
driver 616.92, CUDA 13.4), Intel i9-14900HX, Windows 11 Pro 26200.
LA MÁQUINA ESTUVO CON BATERÍA en todas las ejecuciones siguientes (PowerOnline False, 79%,
descargando ~57 W): el límite de potencia aplicado a la GPU fue de 50.00 W frente a un
máximo de 115.00 W, así que ninguna cifra de rendimiento aquí describe lo que esta
tarjeta puede hacer conectada a la corriente.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
trl 0.29.1, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.0.1.
torchao NO está instalado. Todas las cifras anteriores en benchmarks/ hasta el
cambio de equipo del 2026-09-10 se obtuvieron con una RTX 3050 Laptop 4 GB y NO
son comparables con nada de aquí.
Soup: worktree separado en origin/main 71ecbbb3 (a6d52a7d = PR #1146, el
commit que evalúa este registro), C:\Users\user\projects\Soup-b836, ejecutado como
PYTHONPATH=<worktree>/src python -m soup_cli.cli, con
soup_cli.__file__ verificado bajo el worktree antes de la primera ejecución.
Arnés: ninguno. `soup bench train` ES el arnés; el fixture que lo dirige
se cita en la sección 2 y se confirmó bajo benchmarks/results/gate-836/
junto con cada informe JSON citado más abajo.
-->

# Registro del gate — #836: qué informa `soup bench train` en una tarjeta real
> 🇪🇸 Traducción al español de [`gate-836-bench-train-contract.md`](gate-836-bench-train-contract.md). Ante cualquier duda, manda el original en inglés.

**Estado: el contrato se cumple; la cifra que informa no se repite.**
2026-09-22. El punto de aceptación 7 de [#836](https://github.com/MakazhanAlpamys/Soup/issues/836)
pedía una ejecución de referencia publicada de `soup bench train`, incluida en
[#1146](https://github.com/MakazhanAlpamys/Soup/pull/1146). Aquí está, y la parte útil no es la cifra de referencia.

Se midieron todos los campos prometidos por el comando y todos se mantuvieron. En **trece ejecuciones con una configuración idéntica byte a byte**, el hash de configuración, los recuentos de tokens supervisados y totales, `max_memory_allocated`, `max_memory_reserved` y las tres comprobaciones fueron **idénticos byte a byte**; sin embargo, el rendimiento informado osciló entre **376.4 y 915.3 tokens supervisados/s, una dispersión de 2.43x** (coeficiente de variación 35.3%). La dispersión es **aditiva, no multiplicativa**: en cuatro formas cuyo tiempo por paso en modo rápido abarca 26x (0.197 s a 5.26 s), la diferencia entre la repetición más rápida y la más lenta de cada forma es aproximadamente constante: **0.17–0.28 s por paso**. Esa es la diferencia que tendrá que superar una comparación futura; en un paso de 0.2 s representa 1.9x y en uno de 5.3 s representa 4.6%.

**Esto no es una afirmación de rendimiento.** Una tarjeta, con batería, con límite de 50 W; dos modelos pequeños; una forma de secuencia por modelo; ninguna afirmación de calidad. Lo que se evalúa aquí es el *contrato*: que el comando informe lo que dice informar, se niegue cuando la medición no puede significar nada y no invente un valor ausente.

Convención de unidades: MB/GB decimales salvo que una cifra sea potencia de dos, en cuyo caso se escribe en MiB/GiB.

---

## 0. Para qué sirve este registro

`soup bench train` existe para que una cifra de rendimiento venga acompañada de evidencia de que el modelo entrenaba mientras se medía. Por tanto, aquí se plantean dos preguntas:

1. ¿Informa lo que promete, de extremo a extremo, en una tarjeta real?
2. ¿Cuál es el piso de ruido; es decir, qué magnitud de cambio podría resolver realmente una comparación futura de dos cifras de `soup bench train`?

La pregunta 2 terminó siendo el registro completo. La respuesta breve a la pregunta 1 es: sí, todos los campos, incluidas las cuatro negativas de la sección 9.

## 1. El equipo y por qué nada anterior es comparable

| | valor | cómo se obtuvo |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU | `nvidia-smi`; informe `provenance.card` |
| VRAM | 8151 MiB (8547.04 MB) | `nvidia-smi` |
| capacidad de cómputo | 12.0 (sm_120, Blackwell) | informe `provenance.compute_capability` |
| driver / runtime CUDA | 616.92 / 13.0 | informe `provenance.driver`, `provenance.cuda_runtime` |
| reloj SM máximo | 3090 MHz; observado hasta 2797 MHz bajo carga | `nvidia-smi --query-gpu=clocks.max.sm` |
| **potencia** | **límite actual y predeterminado 50.00 W, máximo 115.00 W** | `nvidia-smi -q -d POWER` |
| **CA** | **ausente — `PowerOnline: False`, batería 79%, descarga de 57.2 W** | `root\WMI` `BatteryStatus` |
| sistema operativo / Python | Windows-11-10.0.26200-SP0 / 3.12.10 | informe `provenance` |
| tipo de datos / optimizador | `torch.bfloat16` / `OptimizerNames.ADAMW_TORCH` | informe `provenance` |
| seed / data_seed | 1234 / 1234 | se configura y se repite en `provenance` |

La línea de batería no es decorativa. Una GPU de laptop limitada a 50 W de 115 W posibles funciona al 43% de su presupuesto de potencia, y la forma B de abajo alcanzó un pico de 47.7 W; es decir, llegó al límite. Repetir cualquiera de estas mediciones conectada a corriente establece una **nueva línea base**, no una comparación.

## 2. El fixture y las cuatro formas

Dos modelos públicos totalmente almacenados en caché, un fixture sintético generado una vez y con hash, LoRA en todos los brazos, `quantization: none`, backend `transformers`, `task: sft`.

| archivo | sha256 | contenido |
|---|---|---|
| `bench_fixture.jsonl` | `5fcb5e40f8fd16f8ad6ca63e4da169ec5be1b482a98e7fd3196652a4dd9265fd` | 64 filas alpaca, cortas (~85 tokens cada una) |
| `bench_fixture_long.jsonl` | `c9d7362c41807ab1023434b17fcc48d400dfa761acf51ba443a6c1b85a7b14cc` | 64 filas alpaca, ~482 tokens de salida cada una |
| `bench.yaml` | `6f7c3edbbf4436a6c5c0cfc156b98da2445669987aafbefe6d98fcc878a7d510` | forma A |
| `bench_b.yaml` | `7275d80c1e0d73c1be5fcdb6c39055579ef613abddf5b28653d49af37db79bfc` | formas B |
| `bench_c.yaml` | `3528b1aee16348daa076595f7cee4db0e191967fbef6da83e9f1f5aeb2f5e678` | formas C y D |
| `bench_frozen.yaml` | (sección 9) | negativa por cero parámetros entrenables |
| `bench_dpo.yaml` | (sección 9) | negativa por tarea fuera de alcance |

Los siete archivos anteriores y los 25 informes JSON están confirmados bajo
[`results/gate-836/`](results/gate-836/).

| forma | modelo | entrenables | batch x tokens | pasos/calentamiento |
|---|---|---|---:|---|
| **A** | `HuggingFaceTB/SmolLM2-135M-Instruct` | 921,600 / 135,436,608 (0.68%), 120 tensores | 4 x ~85 | 12/4 y 60/20 |
| **B** | `Qwen/Qwen2.5-0.5B-Instruct` | 1,081,344 / 495,114,112 (0.22%), 96 tensores | 8 x 512 | 12/4 |
| **C** | `Qwen/Qwen2.5-0.5B-Instruct` | igual que B | 2 x 512 | 12/4 |
| **D** | `Qwen/Qwen2.5-0.5B-Instruct` | igual que B | 2 x 512 | 200/40 |

LoRA usa `r: 16, alpha: 32, dropout: 0.0` en `q_proj`/`v_proj` en todos los casos.
`data.val_split` es el valor predeterminado del esquema, por lo que el cargador divide las 64 filas en
57 de entrenamiento / 7 de evaluación; `bench train` fija `eval_strategy = "no"`, así que las filas de evaluación se cargan pero nunca se usan.

## 3. La ejecución de referencia — forma A, `--steps 12 --warmup 4`

El comando, íntegro:

```
cd <scratch>
PYTHONPATH=C:/Users/user/projects/Soup-b836/src \
  python -m soup_cli.cli bench train --config bench.yaml --steps 12 --warmup 4 -o run1.json
```

y sus últimas cuatro líneas, íntegras:

```
{'train_runtime': '21.45', 'train_samples_per_second': '2.238', 'train_steps_per_second': '0.559', 'train_loss': '4.004', 'epoch': '0.8'}
12 steps (4 warm-up discarded): median 0.3053s, p95 0.3654s, 1448 supervised of
2724 tokens, 594.7 supervised tok/s
Report: run1.json
Valid: the model trained while it was measured.
```

Código de salida 0. Observa las dos cifras de rendimiento en líneas consecutivas y su diferencia: el `train_steps_per_second` propio de HF Trainer es 0.559 (1.79 s/paso) e incluye la construcción del cargador de datos y todo el primer paso; la mediana de `bench train` es 0.3053 s y no los incluye. Esa diferencia —5.9x— explica por qué se descarta el calentamiento, y se ve en la salida publicada sin instrumentar nada.

El informe (`results/gate-836/shapeA_run1.json`):

| campo | valor |
|---|---|
| `valid` / `failures` | `true` / `[]` |
| `checks.trainable_parameters` | 120 |
| `checks.grad_norm` | `"all reported steps finite and non-zero"` |
| `checks.parameters_changed` | `true` |
| `timing` | mediana 0.3053 s, p95 0.3654 s, total 2.435 s, contabilizados 8, calentamiento descartado 4 |
| `tokens` | útiles 1448, total 2724, utilización 0.5316 |
| `throughput` | 594.70 tokens útiles/s, 1118.75 tokens totales/s |
| `memory` | 796,314,624 B asignados (796.31 MB) / 1,140,850,688 B reservados (1140.85 MB) |
| `steps_requested` / `steps_measured` | 12 / 12 |
| `config_hash` | `268ca2462c1a7b852e4f77c8d98ed4fb89d89e481b202ded21035ab641326cbc` |

La forma A cabe con margen: 796.31 MB asignados de 8547.04 MB de
VRAM. `provenance.sm_clock_mhz_after_run` indicó **285**; véase la sección 10.3 para
saber por qué esa cifra no sirve.

## 4. Trece repeticiones con una configuración idéntica byte a byte

`--steps 12 --warmup 4`, mismo archivo de configuración, mismo fixture, misma sesión, mismo
equipo. Las ejecuciones 1–3 no se muestrearon; 7–10 se ejecutaron con `nvidia-smi -lms 100` a la vez;
e1–e6 se ejecutaron con un muestreador de proceso/frecuencia de CPU a la vez (sección 7).

| ejecución | mediana (s) | p95 (s) | total (s) | tokens útiles/s | muestreador |
|---|---:|---:|---:|---:|---|
| run1 | 0.3053 | 0.3654 | 2.435 | 594.7 | ninguno |
| run2 | 0.2252 | 0.3157 | 1.877 | 771.5 | ninguno |
| run3 | 0.4378 | 0.4678 | 3.452 | 419.5 | ninguno |
| run7 | 0.4346 | 0.4694 | 3.473 | 417.0 | nvidia-smi |
| run8 | 0.4355 | 0.4754 | 3.539 | 409.2 | nvidia-smi |
| run9 | 0.4746 | 0.5976 | 3.847 | 376.4 | nvidia-smi |
| run10 | 0.4289 | 0.4566 | 3.443 | 420.6 | nvidia-smi |
| e1 | 0.4706 | 0.5152 | 3.748 | 386.4 | proceso + CPU |
| e2 | 0.1972 | 0.2659 | 1.655 | 874.7 | proceso + CPU |
| e3 | 0.1982 | 0.2049 | 1.582 | 915.3 | proceso + CPU |
| e4 | 0.4330 | 0.4617 | 3.464 | 418.0 | proceso + CPU |
| e5 | 0.3431 | 0.3714 | 2.763 | 524.1 | proceso + CPU |
| e6 | 0.3559 | 0.3770 | 2.852 | 507.7 | proceso + CPU |

- tiempo mediano por paso: **0.1972 – 0.4746 s**, razón **2.41x**, diferencia **0.2774 s**, mediana de medianas 0.4289 s
- tokens útiles/s: **376.4 – 915.3**, razón **2.43x**, mediana 420.6, media 541.2, desviación estándar 190.8, **CV 35.3%**

La distribución no es una bimodalidad limpia —e5 y e6 quedan entre los grupos—,
pero los extremos rápido y lento están bien separados y ambos incluyen
repeticiones sin muestreo (run2 rápida, run3 lenta), por lo que los muestreadores no son la causa.

**Si este registro publicara una sola cifra, tendría un error de hasta 2.4x.** Ese
es el hallazgo.

## 5. Todo lo que no es tiempo es idéntico byte a byte

En las mismas trece ejecuciones:

| campo | valores distintos |
|---|---|
| `config_hash` | 1 (`268ca246…`) |
| `tokens.useful` / `tokens.total` | 1 (`1448` / `2724`) |
| `memory.max_memory_allocated_bytes` | 1 (`796314624`) |
| `memory.max_memory_reserved_bytes` | 1 (`1140850688`) |
| `checks` | 1 (`{trainable_parameters: 120, grad_norm: "all reported steps finite and non-zero", parameters_changed: true}`) |

Así que la parte determinista del contrato sí es determinista en este equipo, y una regresión en tokens, memoria o comprobaciones de entrenamiento sería visible con n=1. Solo la parte de tiempos necesita repeticiones.

Vale la pena conservar una excepción: `max_memory_reserved` **no** es invariable entre formas con la misma asignación. La forma C indicó 4,982,833,152 B en dos ejecuciones y 4,980,736,000 B en la tercera; la forma D —la misma configuración con distintos `--steps`— indicó 5,073,010,688 B, mientras que `max_memory_allocated` fue `4052621824` en las seis. Eso es el asignador en caché y precisamente por eso el contrato informa ambos por separado en vez de escoger uno.

## 6. La dispersión es aditiva, no multiplicativa

El mismo comando para cuatro formas; repetición más rápida y más lenta de cada una:

| forma | pasos/calentamiento | mediana más rápida | mediana más lenta | razón | **diferencia** |
|---|---|---:|---:|---:|---:|
| A (n=13) | 12/4 | 0.1972 | 0.4746 | 2.41x | **0.2774 s** |
| A (n=3) | 60/20 | 0.1965 | 0.4401 | 2.24x | **0.2436 s** |
| B (n=3) | 12/4 | 5.2596 | 5.5039 | 1.046x | **0.2443 s** |
| C (n=3) | 12/4 | 0.2045 | 0.3750 | 1.83x | **0.1705 s** |
| D (n=3) | 200/40 | 0.2036 | 0.3854 | 1.89x | **0.1818 s** |

La columna de razón va de 1.05x a 2.41x. La columna de diferencia va de 0.17 s a
0.28 s entre formas cuyo tiempo por paso en modo rápido difiere 26x. Lee la
columna de diferencia, no la de razón: **algo añade ocasionalmente cerca de una quinta de
segundo a un paso, y que parezca catastrófico o invisible depende por completo de cuánto dure el paso.**

Dos consecuencias prácticas:

- La concordancia de 4.6% de la forma B **no** demuestra que la herramienta sea precisa. Es el mismo ruido de 0.24 s dividido entre un paso de 5.3 s. Si el registro solo hubiera ejecutado la forma B, habría publicado «las repeticiones concuerdan dentro del 5%» y habría sido incorrecto.
- Ejecutar más tiempo no lo resuelve. La forma D contabilizó 160 pasos en una ventana de 33–63 s y aun así tuvo una dispersión de 1.89x; la forma A con 40 pasos contabilizados tuvo una dispersión de 1.93x. El costo aditivo no es un transitorio inicial que una ventana más larga promedie y elimine.

El mecanismo **no está establecido**. Se probaron dos posibles explicaciones y ambas fallaron (sección 7). Lo que sí está establecido es la magnitud.

## 7. Dos hipótesis, cada una refutada por su propia covariable

**Hipótesis 1 — otra sesión en el equipo.** El equipo de desarrollo se comparte; mientras se ejecutaba la tercera ejecución de la forma D, `Win32_Process` mostró que arrancaban dos procesos `pytest` de otra sesión 33 s y 41 s después del inicio (23:06:25 y 23:06:33 frente a 23:05:52 de d3). Esa ejecución es la única lectura realmente contaminada de este registro y se informa como tal: **mediana d3 de 0.2092 s (modo rápido), p95 0.3726 s (modo lento), total 40.890 s**, rendimiento 3676.6 tokens/s entre ambos modos; una ejecución que pasó de un modo a otro mientras se medía, y que además muestra con claridad cómo la pareja mediana-y-p95 del informe discrepa abiertamente sobre el modo. Buen comportamiento de la herramienta.

Pero la hipótesis no sobrevive a la serie e, que muestreó el recuento de procesos pares durante toda la vida de cada ejecución (el bucle pedía 250 ms; `Get-CimInstance Win32_Process` es lo bastante lento como para obtener 18–29 muestras por ejecución, aproximadamente una por segundo):

| ejecución | tokens útiles/s | `python.exe` par (mediana/máximo/muestras) |
|---|---:|---|
| e1 | 386.4 (lento) | 0.0 / 2 / 29 |
| e2 | 874.7 (rápido) | 0.0 / 1 / 24 |
| e3 | 915.3 (rápido) | 0.5 / 1 / 18 |
| e4 | 418.0 (lento) | 0.0 / 1 / 26 |
| e5 | 524.1 (medio) | 0.0 / 1 / 27 |
| e6 | 507.7 (medio) | 0.0 / 0 / 23 |

La mediana fue 0 en las seis y el máximo nunca superó 2. La ejecución más rápida y la más lenta tuvieron el mismo recuento de pares. **Refutada.**

**Hipótesis 2 — escalado de frecuencia de CPU con batería.** La utilización de GPU de la forma A se muestreó al 11–15% *como máximo* durante toda la vida de cada ejecución (media 0.8–1.5%, media cuando no era cero 4.0–5.7%), así que la forma A está limitada por el host y la frecuencia de CPU era la sospechosa obvia. Se muestreó a la vez `\Processor Information(_Total)\% Processor
Performance`:

| ejecución | tokens útiles/s | % rendimiento del procesador (mediana/mín./máx.) |
|---|---:|---|
| e1 | 386.4 (lento) | 114 / 92 / 123 |
| e2 | 874.7 (rápido) | 117 / 102 / 126 |
| e3 | 915.3 (rápido) | 114 / 97 / 121 |
| e4 | 418.0 (lento) | 112 / 94 / 157 |
| e5 | 524.1 (medio) | 116 / 63 / 153 |
| e6 | 507.7 (medio) | **150** / 137 / 155 |

Las tres ejecuciones más lentas y las dos más rápidas difieren como máximo en 5 puntos, y la ejecución con frecuencia de CPU sostenida mucho *más alta* (e6, 150%) está en la mitad lenta. **Refutada.**

Ambas se publican como resultados negativos. Las cifras del lado GPU recogidas al probarlas se sostienen por sí solas y vale la pena conservarlas: durante la forma A, el reloj SM de la tarjeta *mientras estaba ocupada* tuvo una mediana por ejecución de 2370–2557 MHz y alcanzó 2610–2782 MHz; la potencia tuvo una mediana por ejecución de 13.8–21.3 W y un pico de 28.3 W frente a un límite de 50 W. Por tanto, la forma A no tenía déficit de reloj ni de potencia. Simplemente no era trabajo de GPU.

## 8. La forma B se completó mientras asignaba 1.51x la VRAM de la tarjeta

La forma B (batch 8 x 512 en un vocabulario de 151,936 filas) es el único brazo que saturó la GPU: 100% de utilización cuando estaba ocupada, 71–78% de las muestras ocupadas, reloj mediano de 2782 MHz cuando estaba ocupada (pico 2797), mediana de potencia de 32.7–33.3 W y pico de 47.7 W frente al límite de 50 W.

| ejecución | mediana (s) | p95 (s) | total (s) | tokens útiles/s | asignados | reservados |
|---|---:|---:|---:|---:|---:|---:|
| b1 | 5.3080 | 5.4830 | 39.131 | 695.2 | 12,879.61 MB | 16,944.99 MB |
| b2 | 5.5039 | 5.5611 | 39.973 | 680.6 | 12,879.61 MB | 16,944.99 MB |
| b3 | 5.2596 | 5.3270 | 38.268 | 710.9 | 12,879.61 MB | 16,944.99 MB |

**12,879.61 MB asignados en una tarjeta con 8547.04 MB.** Las tres ejecuciones terminaron, código de salida 0, `valid: true`. Este es el comportamiento WDDM del que ya advierte el README de esta carpeta: Windows desborda a memoria compartida del host en vez de generar `CUDA out of memory`; y se manifiesta en `bench train` de una forma concreta: los campos `memory` provienen de `torch.cuda.max_memory_allocated` / `_reserved`, son contadores del asignador y están **correctos**; simplemente no afirman que la configuración cupiera en VRAM. En esta plataforma, una cifra `memory` superior a la capacidad de la tarjeta indica que no cupo.

El costo de no caber se mide frente a la forma C, mismo modelo y longitud de secuencia con batch 2 en vez de 8:

| | forma B (batch 8) | forma C (batch 2) |
|---|---:|---:|
| asignado | 12,879.61 MB (1.51x VRAM) | 4052.62 MB (0.47x VRAM) |
| paso mediano | 5.2596–5.5039 s | 0.2045–0.3750 s |
| tokens útiles/s | 680.6–710.9 | 2553.5–4666.6 |

El batch mayor es **3.8x a 6.6x más lento por token supervisado**, según qué repetición de cada forma se compare; vuelve a aparecer el ruido de la sección 6. «Terminó» no significa «cupó», y el informe indica cuál de las dos cosas ocurrió si consultas `memory` frente a la tarjeta y no de forma aislada.

## 9. Cuatro negativas, comprobadas en la práctica

Las cuatro, textuales; todas con código de salida 1.

**(a) No queda nada que medir.**

```
$ python -m soup_cli.cli bench train --config bench.yaml --steps 2 --warmup 2 -o refusal_a.json
--steps 2 leaves nothing after --warmup 2: raise --steps or lower --warmup.
EXIT=1
```
No se cargó ningún modelo y **no se escribió ningún archivo de informe** (`ls refusal_a.json` →
no existe ese archivo).

**(b) Cero parámetros entrenables.** `bench_frozen.yaml` elimina LoRA y fija
`training.unfrozen_parameters: ["^no_module_is_named_this$"]`, un patrón Spectrum
que no coincide con nada:

```
$ python -m soup_cli.cli bench train --config bench_frozen.yaml --steps 6 --warmup 2 -o refusal_b.json
apply_unfrozen_parameters: no parameter matched any of 1 pattern(s) — the model has NO trainable parameters. Check your training.unfrozen_parameters against the model's parameter names (run `soup spectrum scan` to regenerate).
Spectrum targeted FT: 0 parameter tensor(s) unfrozen (LoRA off)
Spectrum targeted FT: 0 trainable / 134,515,008 total (0.00%)
...
0 trainable parameter tensors with real storage.
EXIT=1
```
Se cargaron el modelo y el dataset, se construyó el trainer y se activó el preflight
`check_trainable` antes de `trainer.train()`; así el Trainer no llegó a fallar dentro de
autograd, que es precisamente la razón de verificarlo temprano. De nuevo, **no se escribió ningún informe**.

**(c) Tarea fuera de alcance.** Inmediato, antes de cargar nada:

```
$ python -m soup_cli.cli bench train --config bench_dpo.yaml --steps 12 --warmup 4 -o refusal_c.json
Cannot bench this config: task: dpo -- bench train measures the SFT trainer only
EXIT=1
```

**(d) Falta el archivo de configuración.**

```
$ python -m soup_cli.cli bench train --config nope.yaml -o refusal_d.json
Config not found: nope.yaml
EXIT=1
```

Cada una identifica el motivo real. No hubo suposiciones.

También se comprobaron otras dos promesas y se cumplieron: ningún run creó el directorio `output:` de la propia configuración (`./output_bench836/`) en las 25 ejecuciones, y se limpiaron **todos** los directorios temporales `soup-bench-*` de `%TEMP%`; no quedó ninguno al terminar la sesión.

## 10. Cuatro observaciones sobre el propio contrato. Ninguna es un defecto.

**10.1 `config_hash` incluye las sobrescrituras propias de la herramienta, por lo que cambia con `--steps`.** La forma C con `--steps 12` tiene el hash `7ec538f1f605…`; el archivo idéntico con `--steps 200` tiene `c2cd23cc9807…`. Al comparar los bloques `resolved_config`, solo hay una diferencia: `training.save_steps` 13 frente a 61 —el valor `steps + 1` que establece `run_bench_train` para que no se guarde nada—. Posiblemente sea el comportamiento correcto (el hash describe la ejecución realizada, no el archivo), pero quien compare dos informes del mismo `soup.yaml` con distinto número de pasos verá hashes diferentes y no debería concluir que la configuración cambió. Lo mismo sucede con la forma A: `268ca246…` con 12 pasos, `70e143f2…` con 60.

**10.2 «El informe todavía se escribe» se refiere a las comprobaciones posteriores a la ejecución, no a las dos negativas de preflight.** `docs/serving-and-export.md` dice que el comando «sale con código 1 si alguna comprobación falla. El informe se sigue escribiendo, con los fallos incluidos», y luego enumera `trainable_parameters` como una de esas comprobaciones. Medición: cuando se activa `trainable_parameters` —que en el uso normal siempre es la copia de preflight, pues de otro modo el Trainer fallaría en autograd—, el comando genera un error antes de que exista un informe y **no escribe nada** (sección 9b). La oración y la tabla son ciertas, pero describen rutas de código diferentes; quien planee analizar el informe después de un fallo debe saber cuál aplica.

**10.3 `provenance.sm_clock_mhz_after_run` es una lectura en reposo en esta tarjeta y no sirve como la covariable para la que se añadió.** Su propio docstring la llama «el reloj al que terminaron los pasos medidos». Comparación con un muestreador de 100 ms: la mediana del reloj de las muestras ocupadas de la tarjeta fue 2370–2557 MHz *durante* los pasos, pero el campo registró, en las 25 ejecuciones de este registro, **180 MHz catorce veces,
1417 cuatro veces, 225 tres veces y 172 / 232 / 277 / 285 una vez cada una**. Las lecturas 1417 capturan el descenso de frecuencia; el resto son de reposo. De manera concluyente, las tres ejecuciones de la forma C —4645.6, 4666.6 y 2553.5 tokens/s— registraron 225, 232 y 225 MHz, así que el campo no distingue una diferencia de 1.83x en lo que pretende explicar. Muestrear durante la ventana contabilizada lo resolvería; es una mejora futura, no un impedimento para publicar, y el campo describe honestamente una única lectura posterior a la ejecución.

**10.4 El informe no contiene una distribución por paso.** `timing` es
`median / p95 / counted_steps / warmup_steps_discarded / total_seconds`. Basta para *detectar* una ejecución que cambió de modo durante el trabajo —la mediana de d3 de 0.2092 s frente a p95 de 0.3726 s y total de 40.890 s para 160 pasos (media 0.2556 s) es exactamente esa señal—, pero no para localizar el cambio. A la luz de la sección 6, una futura revisión que emitiera tiempos por paso (o algunos cuantiles más) permitiría diagnosticar este tipo de ruido solo con el informe.

## 11. Lo que estas cifras NO significan

- **No es una cifra de capacidad de la tarjeta.** RTX 5070 Laptop con **batería**, GPU limitada a 50 W de 115 W. Conectada a corriente sería otra línea base.
- **No es una afirmación entre modelos.** Dos modelos, 135M y 0.5B, ambos con LoRA y `quantization: none`. Nada de esto se extrapola a 7B, a NF4 ni a la ruta de streaming, que esta versión no toca en absoluto.
- **No es una afirmación entre formas.** Cuatro formas, cada una con una sola longitud de secuencia y tamaño de batch. La sección 8 muestra cuánto difieren dos formas del *mismo modelo*.
- **No es una afirmación de calidad.** `parameters_changed: true` demuestra que algo cambió. Es un piso, no un gate. La pérdida de la forma D bajó a `0.07103` en 200 pasos sobre 57 filas; es memorización de un fixture sintético y no significa nada.
- **No es una línea base de comparación con n=1.** Cualquier uso futuro de `soup bench train` para mostrar un aumento de velocidad requiere repeticiones y debe indicar su dispersión. En este equipo no es posible resolver con ejecuciones individuales un cambio que ahorre menos de ~0.2 s por paso.
- **No es comparable con ningún registro `gate-v0.72.*`**, todos obtenidos en una RTX 3050 Laptop 4 GB.

## 12. Reproducción

No se añade un script de arnés: el comando enviado es el arnés y los dos
fixtures se generan de forma determinista y quedan confirmados.

```bash
pip install "soup-cli[train]"          # or an editable checkout at >= the #1146 merge
# fixtures + configs: benchmarks/results/gate-836/
soup bench train --config bench.yaml   --steps 12  --warmup 4  -o run1.json    # shape A
soup bench train --config bench_b.yaml --steps 12  --warmup 4  -o b1.json      # shape B
soup bench train --config bench_c.yaml --steps 12  --warmup 4  -o c1.json      # shape C
soup bench train --config bench_c.yaml --steps 200 --warmup 40 -o d1.json      # shape D
```

Repite cada caso al menos tres veces y consulta la dispersión antes de leer la cifra.
Las negativas de la sección 9 se reproducen en segundos y (c) y (d) no requieren GPU.

Los 25 informes JSON de esta sesión están confirmados íntegros bajo
[`results/gate-836/`](results/gate-836/), incluido el contaminado `d3.json`.

## Adenda, 2026-09-23 — el instrumento cambió después de este registro (#1166)

Todo lo anterior queda sin cambios y describe el comando tal como se midió. Desde entonces se atendieron tres
observaciones de la sección 10:

- **10.3:** se eliminó `provenance.sm_clock_mhz_after_run`. Lo reemplaza
  `provenance.sm_clock_mhz_busy`, que da `min` / `median` / `max` /
  `sample_count` de las muestras de `nvidia-smi` conservadas solo durante los pasos contabilizados
  (posteriores al calentamiento), además de `unavailable_reason` cuando no hay cifras.
  Las consultas empiezan según un horario de 100 ms y una consulta lenta omite intervalos. Cada
  muestra lleva marca de tiempo en el punto medio de su consulta. Los valores
  `sm_clock_mhz_after_run` de este registro (y del JSON confirmado en
  `results/gate-836/`) son lecturas en reposo posteriores a la ejecución. No los compares
  con el `sm_clock_mhz_busy` de un informe posterior.
- **10.4:** `timing.step_seconds` ahora enumera todos los pasos contabilizados en el orden de ejecución.
- **10.2:** `docs/serving-and-export.md` ahora distingue las negativas de preflight, que
  no escriben informe, de los fallos en comprobaciones posteriores, que escriben un
  informe `valid: false`. No cambió el código porque el comportamiento medido en la sección 9 ya era correcto.

El cronometraje y el recuento de tokens no cambiaron, pero el muestreador agrega carga nueva: durante los
pasos cronometrados, un hilo inicia `nvidia-smi` aproximadamente cada 100 ms. Sigue sin resolverse si cambia alguna cifra de rendimiento anterior. **Seguimiento:** vuelve a ejecutar
los comandos de la sección 12 en el equipo de la sección 1 dos veces, una con el muestreador y otra
sin él (por ejemplo, con un `clock_sampler` sin operación). Compara ambas con
este registro y recoge los relojes en ventana ocupada para las formas A–D. No había GPU disponible
cuando se hizo este cambio.
