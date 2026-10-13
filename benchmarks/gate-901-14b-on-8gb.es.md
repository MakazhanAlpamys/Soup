<!--
Registro de mediciones de trabajo, publicado íntegro. El desvío equivocado (una
hipótesis de memoria que refutó la primera ejecución de sondeo), las reducciones en
puro torch y las cifras medidas que después resultaron provenir del instrumento
incorrecto se conservan en el orden en que ocurrieron.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB por nvidia-smi,
8.518 GB por torch, driver 616.92), Intel i9-14900HX, 31.7 GB DDR5-5600,
Windows 11 Pro 26100. Fixture en D: (NVMe), caché de fragmentos NF4 en C: (NVMe).
Stack: Python 3.12.10 · torch 2.14.0+cu130 · transformers 5.17.0 · peft
0.20.0 · trl 0.29.x · bitsandbytes 0.50.2.
Soup: branch fix/stream-901 (v0.75.0 + los cuatro commits que evalúa este registro),
worktree C:\Users\user\projects\Soup-stream, venv propio, soup_cli.__file__
verificado en Soup-stream\src antes de cada ejecución.
Arnés: benchmarks/harness/synth_checkpoint.py (fixture 14B),
benchmarks/harness/issue901_stream_step_probe.py (sondeo de paso), además de tres
scripts temporales de puro torch citados en la sección 3. Los JSON sin procesar de las dos
ejecuciones de sondeo que sobrevivieron y el registro de extremo a extremo están en
benchmarks/results/probe-rtx5070/issue901_*.
-->

# Registro del gate — #901: «OOM de 14B dentro de SFTTrainer.__init__ en una tarjeta de 8 GB»
> 🇪🇸 Traducción al español de [`gate-901-14b-on-8gb.md`](gate-901-14b-on-8gb.md). Ante cualquier duda, manda el original en inglés.

**Estado: REPRODUCIDO, EXPLICADO, CORREGIDO, 2026-09-15.** El problema nunca fue
la VRAM. Un fallo al fijar en páginas el almacén de 9.93 GB dejó un error CUDA obsoleto en
el runtime; la alternativa enviada, que pasa a un almacén paginable, no lo limpió, y
el primer lanzamiento de kernel de la ejecución informó ese error obsoleto como `CUDA error: out of
memory` con 7.3 GB libres. El fijado en páginas falló porque el asignador de host en caché de torch redondea cada solicitud fijada a la siguiente potencia de dos; fijar el almacén tensor por tensor costaba 1.73x su tamaño. Tras las correcciones, la misma configuración termina de extremo a extremo en este equipo de 8 GB / 32 GB y el almacén queda fijado en páginas: pico medido de **2.944 GB asignados / 3.244 GB reservados** frente a la predicción de **3.39 GB** del preflight, que siempre había sido correcta.

**No se afirma nada sobre rendimiento ni calidad.** El fixture es una
*forma* sintética de Qwen2.5-14B con pesos aleatorios: el pico de memoria y el costo de memoria fijada dependen del tamaño de los tensores, no de sus valores, y eso es todo lo que mide este registro.

Convención de unidades: GB decimales, salvo que una cifra sea potencia de dos, en cuyo
caso se escribe en GiB.

---

## 0. El informe y el equipo que lo reprodujo

| | informante (#901) | este equipo |
|---|---|---|
| GPU | RTX 3070 8 GB, Windows 11 | RTX 5070 Laptop 8 GB, Windows 11 |
| RAM | 32 GB | 31.7 GB |
| VRAM libre en preflight | 7.41 GB | 7.35 GB |
| modelo | Qwen/Qwen2.5-14B-Instruct | forma Qwen2.5-14B sintética (`--arch qwen2`) |
| almacén | 9.93 GB NF4, 48 capas, «could not page-lock … PAGEABLE» | 9.932 GB NF4, 48 capas, mismo mensaje |
| predicción del preflight | 3.39 GB en batch 1 x seq 384 (logits 0.82 GB) | 3.39 GB, línea idéntica |
| fallo | `AcceleratorError: CUDA error: out of memory` en `SFTTrainer.__init__` (o en el forward del sondeo) | `AcceleratorError: CUDA error: out of memory` en la primera operación CUDA después de la construcción |

El control del propio informante —gemma-2-2b transmitido hasta completarse en el mismo equipo— fue lo que indicó «algo a escala de 14B», y resultó ser «algo con un almacén de 9.93 GB»: el almacén de 2B se fija en páginas, el de 14B no, y solo el fallo al fijarlo contamina la ejecución.

## 1. El fixture

`benchmarks/harness/synth_checkpoint.py` incorporó `--shape qwen2.5-14b --arch
qwen2` (48 capas, hidden 5120, intermediate 13824, 40/8 cabezas, cabeza untied de 152064 filas,
sesgos q/k/v): 14.77 B parámetros, 29.54 GB bf16 en D:. El
generador tardó 2178 s debido a una pausa de 2124 s antes de la capa 28 (marcas de tiempo
11:51:24 → 12:27 en los archivos de capa); la escritura por capa tarda 0.9 s antes y
después, y no se explica la pausa: había otras dos sesiones ejecutando
suites en el equipo. Solo importa para la higiene de tiempos; nada de aquí depende del generador.

Fragmentación a NF4 (doble cuantización, cuantizado en GPU): 56.2 s, 9.932 GB en
48 fragmentos de capa + 2 fragmentos grandes + extras.

## 2. Reproducción y la hipótesis de memoria que refutó

Antes de la ejecución: RAM física libre 15.48 GB, carga de confirmación 23.54 GB de
47.35 GB, otros dos procesos Python de ~0.1 GB cada uno; GPU con 0 MiB usados.

`issue901_stream_step_probe.py --seq 384 --batch 1 --lora-r 16 --lora-targets
q_proj,v_proj` (la forma del informante; `q_proj,v_proj` es lo que PEFT selecciona con
`target_modules: auto` para Qwen2, y reproduce los 12.58M entrenables informados):

```
layer streaming could not page-lock the base (AcceleratorError); falling back to a PAGEABLE RAM store. ...
shards        ... (56.2 s); build 40.0 s
store         9.932 GB pageable tier ram; buffers 2 x 142 MB + large slot 1557 MB
after build   allocated 1.895 GB  reserved 1.919 GB  free 5.393 GB
...
    ids = torch.randint(0, int(config.vocab_size), (args.batch, args.seq), device=device)
torch.AcceleratorError: CUDA error: out of memory
```

**Hallazgo 2: falla la primera operación CUDA tras la construcción, y es una asignación de 3 KB con 5.39 GB libres.** Sea lo que sea, no es un problema aritmético: `torch.randint` para 384 int64 no puede necesitar 5 GB. La hipótesis que traje a la ejecución —que el embedding de 1.557 GB, `lm_head` de 1.557 GB y el slot de capa grande de 1.557 GB estaban residentes y que la fórmula omitía uno— murió aquí, antes de tocar una sola capa del decoder. El comentario del mantenedor en el issue proponía dos opciones, «la fórmula subestima» y «fragmentación WDDM del slot de 1.5 GB»; ninguna sobrevive al fallo de una asignación de 3 KB.

Lo único que había hecho la ejecución fallida al runtime CUDA antes de esa línea era
fallar al fijar páginas.

## 3. Reducción a puro torch, sin Soup

Tres scripts temporales (conservados en el bloc de trabajo de la sesión y citados aquí; la
forma permanente es el archivo de prueba).

**(a) Fijar N GB y luego ejecutar operaciones CUDA.** `torch.empty(int(gb*1e9), uint8,
pin_memory=True)`, seguido de `randint` + multiplicación de matrices 2048² + asignación de 1.5 GB en dispositivo:

| solicitud de fijado | fijado | siguientes operaciones CUDA | después de liberar |
|---|---|---|---|
| 9.93 GB (el almacén) | **FALLÓ** `AcceleratorError: CUDA error: out of memory`, 0.1 s | **FALLARON**, mismo error, VRAM libre 7.322 GB | cuda ok |
| 4 GB | ok, 0.9 s | ok | ok |
| 7 GB | ok, 1.8 s | ok | ok |
| 8 GB | ok, 1.7 s | ok | ok |
| 9 GB | **FALLÓ**, 0.0 s | **FALLARON** | — |
| 9.93 GB en fragmentos de 5 x 1.99 GB | ok, 2.1 s en total | ok | ok |

Dos hechos a la vez: el fallo depende de cada solicitud (9 GB en un bloque falla; 9.93 GB
por fragmentos funciona), y después de un fallo al fijar, la *siguiente* operación CUDA falla mientras la
posterior funciona («después de liberar… cuda ok»; no quedó nada retenido).

**(b) Qué operación se activa y cuál lo limpia.** Después de fallar al fijar 9.93 GB:

| secuencia | resultado |
|---|---|
| malloc, sync, memcpy, kernel, kernel | malloc ok · sync ok · memcpy ok · **kernel FALLÓ** · kernel ok |
| pinsmall, kernel, kernel | pinsmall ok · **kernel FALLÓ** · kernel ok |
| sync, sync, kernel, kernel | sync ok · sync ok · **kernel FALLÓ** · kernel ok |
| lasterr, kernel, kernel | no hay enlace de `cudaGetLastError` en `torch.cuda.cudart()` (solo `cudaError`, `cudaGetErrorString`) · **kernel FALLÓ** · kernel ok |

**Hallazgo 3: un `cuMemHostAlloc` fallido deja activado el último error del runtime por hilo; `cudaMalloc`, `cudaDeviceSynchronize` y una copia de memoria no lo consumen, pero la verificación del error al lanzar un kernel sí lo hace y lo limpia.** Así, la ruta alternativa entrega la ejecución con un contexto cuyo primer lanzamiento de kernel está condenado: en el informe ese lanzamiento era `param.data.to(torch.bfloat16)` dentro de `SFTTrainer.__init__`; en el sondeo era el forward, que se registró como «el contexto CUDA quizá ya no sea utilizable»; aquí fue `randint`. Reducir `max_length` nunca habría cambiado nada.

**(c) Por qué falló el fijado de 9.93 GB con 15 GB de RAM libres.** `RamSource` fija
una llamada `torch.empty` por tensor (30 por capa). Al replicarlo contra la caché real de fragmentos
14B, con bytes privados de `psutil` junto a la solicitud:

| patrón | solicitado | delta de confirmación privada | razón |
|---|---|---|---|
| por tensor (enviado) | 6.82 GB (48 capas decoder) | **+11.83 GB** | 1.73x |
| un búfer plano por capa | 6.82 GB | +12.64 GB | 1.85x |
| 100 x 35.39 MB (el NF4 empaquetado de un gate_proj) | 3.54 GB | +6.72 GB | **1.90x** |
| 100 x 64 MiB justo después de liberar lo anterior | 6.71 GB | +0.00 GB | reutilización de caché |
| 100 x 65 MiB después | 6.82 GB | `CUDA_ERROR_OUT_OF_MEMORY from cuMemHostAlloc` | — |

Y los propios contadores de torch lo muestran sin heurística:
`torch.cuda.host_memory_stats()['active_bytes.current']` aumenta **131 072**
para una solicitud de 100 000 bytes, 131 072 para una de 131 072 bytes y
**67 108 864** para una de 35 389 440 bytes; al liberar, `allocated_bytes.current` se mantiene en 67 371 008 (en caché) con `active_bytes` en 0, y
`torch._C._host_emptyCache()` lo devuelve a 0.

**Hallazgo 3c: el asignador de host en caché de torch redondea cada solicitud fijada a la siguiente potencia de dos.** Por eso, el almacén de 9.93 GB (6.82 GB de tensores decoder +
3.11 GB de fragmentos del vocabulario) pidió al driver aproximadamente 17 GB de memoria
fijada en páginas frente a ~15.5 GB libres, lo que significaba «no se pudo fijar la base». También explica la tabla de (a): 9 GB se redondean a 16 GiB y se rechazan; 8 GB son 2^33 y se conceden; cinco bloques de 1.99 GB se redondean a 5 x 2 GiB y caben. Y explica el «límite de memoria fijada en páginas de 7.12 / 7.65 GB» del equipo de referencia en registros anteriores: un almacén bf16 de 5.55 GB, de 3B, fijado tensor por tensor pedía ~10 GB de 16.9 GB.

## 4. Las correcciones, una por una, verificadas en la reproducción

**Corrección A: vaciar el error obsoleto antes de construir el almacén paginable**
(`eb923a3f`). `drain_stale_cuda_error` lanza un kernel trivial para que la comprobación consuma el error y un segundo para confirmar que el contexto está sano; un segundo fallo o cualquier otro error se propaga. `release_cached_pinned_memory`
devuelve los bloques que dejó en la caché de host el intento abandonado.
`recover_from_failed_page_lock` ejecuta ambas acciones entre el constructor fijado fallido
y el paginable, en el nivel RAM y en el almacenamiento temporal del nivel de disco.

Mismo sondeo y misma forma; el almacén sigue siendo paginable:

```
store         9.932 GB pageable tier ram; ...; build 34.6 s
after build   allocated 1.896 GB  reserved 1.898 GB  free 5.414 GB
  [before step             ] alloc 1.896 GB  peak 1.896 GB  reserved 1.898 GB
  [embed fwd out           ] alloc 1.900 GB  peak 1.900 GB
  [L0 fwd out              ] alloc 1.938 GB  peak 2.105 GB
  [L47 fwd out             ] alloc 2.122 GB  peak 2.290 GB
  [lm_head fwd out         ] alloc 2.247 GB  peak 2.290 GB
  [after forward (loss)    ] alloc 2.477 GB  peak 2.714 GB  reserved 2.816 GB
  [lm_head bwd start       ] alloc 2.360 GB  peak 2.944 GB  reserved 3.051 GB
  [L0 bwd start            ] alloc 2.137 GB  peak 2.944 GB  reserved 3.242 GB
  [after backward          ] alloc 2.130 GB  peak 2.944 GB  reserved 3.244 GB
RESULT ok: peak allocated 2.944 GB  peak reserved 3.244 GB  alloc retries 0  ooms 0
```
JSON: `issue901_14b_seq384_pageable_after_fixA.json`.

**Hallazgo 4a: el paso cabe con 4.4 GB de sobra y el preflight sobrestimó un 15% con sus 3.39 GB, en la dirección para la que está ajustado.** El pico aparece al inicio del backward a través de `lm_head` (el gradiente de logits fp32 junto a los logits bf16), justo donde indica el término de logits de la fórmula.

**Corrección B: fijar el almacén en arenas de potencias de dos** (`6a3cfd1e`).
`plan_pinned_arenas` empaqueta los tensores en arenas de 256 MiB (ampliadas para que quepa el tensor más grande, cada una ajustada a la potencia de dos superior a lo que contiene), y
`RamSource` extrae cada tensor como una vista alineada a 256 bytes. Mismo sondeo:

```
store         9.932 GB pinned tier ram; ...; build 10.2 s
RESULT ok: peak allocated 2.944 GB  peak reserved 3.244 GB  alloc retries 0  ooms 0
stats.pinned_bytes = 10 737 418 240
```
JSON: `issue901_14b_seq384_pinned_arenas.json`.

**Hallazgo 4b: el mismo almacén ahora se fija exactamente como 10 GiB (1.081x sus bytes), mientras que el fijado tensor por tensor solicitaba ~17 GB, así que ya no se activa la alternativa; la construcción baja de 34.6 s (fijado fallido, recuperación y copia paginable) a 10.2 s.** El pico de VRAM no cambia ni un byte, como debe ser: el cambio ocurre en el host. La prueba `tests/test_issue901_pinned_arenas.py` fija el límite aritmético del plan para este almacén exacto en <= 1.10x.

**Revisado el 2026-09-15 13:47, tras tres revisiones que cambiaron la regla del empaquetador.**
La ejecución anterior usaba una arena fija de 256 MiB con una excepción para tensores grandes. La revisión encontró que esa regla empaquetaba un almacén bf16 14B a 1.46x (dos proyecciones de 141.6 MB no caben juntas en una arena de 256 MiB), y una primera sustitución —capacidad determinada por el tensor que abre cada arena— falló con el mismo almacén por el otro lado (una arena abierta por una proyección de 52 MB seguía midiendo 256 MiB). La regla enviada es la potencia de dos superior a cuatro veces el tensor más grande del almacén, con mínimo de 256 MiB y máximo de 2 GiB. En ESTE almacén, el tensor más grande es la matriz de vocabulario de 1.557 GB, así que cada arena mide 2 GiB: 5 arenas, 10 GiB, por cálculo; misma cifra, alcanzada de otra forma. Nueva medición con el código final (`issue901_14b_seq384_pinned_arenas_final.json`, RAM física libre inicial 19.04 GB, 5 procesos Python): almacén de 9.932 GB fijado, **10.737 GB fijados en páginas**, pico de 2.944 GB asignados / 3.244 GB reservados, 0 reintentos; la fila anterior no cambia ni un byte, construcción en 15.4 s (antes 10.2 s; el equipo no estaba inactivo). El caso bf16 es solo aritmético, fijado en <= 1.10x por el archivo de prueba; aquí no se construyó ningún almacén bf16 14B.

**Corrección C: el sondeo interpreta un `AcceleratorError` de falta de memoria como OOM**
(`b602bb15`). No se ejercitó en la reproducción después de la Corrección A (ya no se produce ningún error); es la forma #649 que mostraba la salida del sondeo del informante, y solo cambia cuál de dos negativas ve el operador.

## 5. De extremo a extremo, mediante el `soup train` real

El `soup.yaml` del informante, idéntico salvo por la ruta base, 80 filas de chat
(72/8 después de `val_split: 0.1`), `stream_vram_probe: true` como en su segunda
ejecución, el tokenizador Qwen2.5 copiado junto a los pesos sintéticos, ejecución 13:03-13:06:

```
│   base store   9.93 GB across 48 layers (pinned)
│   VRAM buffers 2 x 142 MB + 1 x 1557 MB large-layer slot = 1841 MB
│   peak VRAM    ~3.39 GB at batch 1 x seq 384 (logits 0.82 GB)
│   free VRAM    7.32 GB
measured peak 2.94 GB (3.24 GB reserved) in 4.61 s at batch 1 x seq 384; predicted 3.39 GB
Layer streaming ready: 48 layers, 9.93 GB pinned RAM store (10.74 GB page-locked),
  2 x 142 MB decoder buffers + 1 x 1557 MB large-layer slot
LoRA applied: 12,582,912 trainable / 14,770,033,664 total (0.09%)
Training started!
100%|██████████| 18/18 [01:56<00:00,  6.49s/it]
{'train_runtime': '116.8', 'train_samples_per_second': '0.616', 'train_steps_per_second': '0.154', ...}
│ GPU peak: 2.7/7.9 GB
```

exit 0, adaptador escrito en `output_qwen14b_streamed`. Registro:
`issue901_e2e_soup_train.log`.

**Hallazgo 5: la configuración del issue termina el entrenamiento en una tarjeta de 8 GB con 32 GB de RAM, pasando por `SFTTrainer.__init__` y 18 pasos del optimizador, con el almacén fijado en páginas.** La pérdida (~13, pesos aleatorios) no significa nada y no se cita como resultado. El primer paso tardó 28.5 s y el estado estable ~4.5 s por paso del optimizador de cuatro filas; es una ejecución funcional sobre una forma sintética, no una cifra de rendimiento, y no se afirma ninguna.

En el camino salió a la luz un defecto del fixture: el generador escribió
`"transformers_version": "synthetic"` en `config.json`, que `AutoConfig`
acepta y `AutoTokenizer` rechaza (`InvalidVersion`). Ahora el generador escribe la versión instalada; se conserva la marca `_soup_synthetic`.

## 6. Pruebas y gates

- `tests/test_issue901_failed_page_lock_recovery.py` (15 pruebas, 7 en
  hardware real, incluida la caracterización del comportamiento upstream que fallará el día que torch limpie el error por sí mismo) y
  `tests/test_issue901_pinned_arenas.py` (14, 16 con los
  casos parametrizados): aprobadas aquí, incluidas las pruebas CUDA.
- Conjunto de regresión de streaming (`test_v07200/02/03/04`, `test_issue623`,
  `test_issue971_*`, `test_issue349`, `test_issue385`): **755 aprobadas, 10
  omitidas**, y los cuatro ids de `TestFloat16StreamingIsBitExact` —`[none]` y
  `[nf4]`, fp16 y bf16— pasan explícitamente con el almacén de arenas. Una prueba directa con `torch.equal` también confirma que el almacén fijado es idéntico bit a bit a los fragmentos.

## 7. Lo que NO se midió

- **Un Qwen2.5-14B-Instruct real.** Solo formas; sin calidad, curva de pérdida ni
  rendimiento. Los tiempos por paso de la 5070 no son comparables con nada.
  *El informante lo midió después con un checkpoint real; véase la sección 8.*
- **La tarjeta del informante.** Una RTX 3070 es Ampere; el mecanismo del error obsoleto
  es una propiedad del runtime CUDA y el redondeo del asignador de host de torch, no específico de ninguna tarjeta, pero el informante se ofreció a probar la corrección y el registro lo indica en vez de darlo por hecho.
  *El informante ya lo ejecutó en la RTX 3070; véase la sección 8.*
- **Linux.** Todas las cifras son de Windows/WDDM. La prueba de caracterización del archivo de pruebas indicará si el error obsoleto existe allí.
- **El almacenamiento temporal del nivel de disco**, que todavía fija tensor por tensor y cuesta lo mismo, 1.7-1.9x; `staging_bytes_for` del planificador calcula los bytes solicitados, así que su umbral de rechazo queda corto por ese factor. Se deja para el trabajo del nivel de disco.
- **El límite de fijado en función de la RAM.** El fijado por bloques alcanzó 9.93 GB y el fijado tensor por tensor alcanzó 14.1 GB de confirmación privada antes de que el equipo dijera que no; no se buscó el límite real.
- **No se tomaron registros de carga de confirmación inicial para las ejecuciones 2, 3 ni la ejecución de extremo a extremo**; se muestra el registro de la ejecución 1, y los experimentos de fijado comenzaron con 21.0-21.7 GB de RAM física libre. Se menciona, no se oculta.

## 8. Adenda (2026-09-25): ejecución del informante con un Qwen2.5-14B-Instruct real

Medición de @hasheng del 2026-09-15, reportada en #901 (14:42Z); es una medición de terceros en su tarjeta, no se volvió a ejecutar aquí. Qwen2.5-14B-Instruct, el `soup_14b.yaml` original,
RTX 3070 8 GB (Ampere), Windows, `main` en `614b7f64`. El almacén base es de 9.93 GB en 48 capas,
fijado (10.74 GB fijados en páginas) sin alternativa paginable. El sondeo de VRAM midió un pico de 2.92 GB
(3.27 GB reservados) con batch 1 x seq 384 frente a la predicción de 3.39 GB; se ejecutaron 18 de 18 pasos en unos
3 minutos, la GPU alcanzó un pico de 2.7 de 8.0 GB y se escribió un adaptador real. Las secciones 0-7 quedan
como estaban.
