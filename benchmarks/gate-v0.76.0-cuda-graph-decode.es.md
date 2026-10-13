<!--
Registro de medición de trabajo, publicado tal como se escribió. Los criterios se fijaron en un
commit antes de cualquier ejecución; el lanzamiento se desvió del plan (en segundo plano, no en
la ventana en primer plano de la persona responsable) y así se indica, sin disimularlo.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB según nvidia-smi /
8.518 GB según torch, driver 616.92), Intel i9-14900HX (núcleos P/E híbridos, 32
lógicos), 32 GB, Windows 11 Pro 26200. Pila: Python 3.12.10 · torch
2.14.0+cu130 · transformers 5.17.0 · peft 0.20.0.
-->

# Gate v0.76.0: decodificación con CUDA graphs opcional para `soup infer` / `soup bench infer`

> 🇪🇸 Traducción al español de [`gate-v0.76.0-cuda-graph-decode.md`](gate-v0.76.0-cuda-graph-decode.md). Ante cualquier duda, manda el original en inglés.

`soup infer --cuda-graphs` y `soup bench infer --cuda-graphs` decodifican un modelo Qwen2/Llama residente (opcionalmente bajo un adaptador LoRA sin fusionar) mediante CUDA graphs.
Este registro es el gate declarado de antemano que decidió si se publica la bandera. **Se aprobaron todos
los criterios.** Es una nueva línea base en el equipo RTX 5070 y no es comparable
con ninguna cifra de la era RTX 3050 de esta carpeta.

El trabajo adapta el experimento local del mantenedor, que informó 32.16 -> 62.57
tok/s (+94.55%) con el proceso **fijado a un solo núcleo de CPU**. El motivo de este gate
es la advertencia de ese registro: sin fijar, la línea base del experimento era de 14.48
tok/s frente a 31.85 fijada, y la ruta graph nunca se había medido sin fijar,
que es como la usan los usuarios.

## 0. Qué evalúa y los criterios fijados antes de cualquier ejecución

Criterios, registrados en `8b54b472` (`bench(infer): the pre-declared --cuda-graphs
gate, committed before any run`) antes de la primera medición:

- **(a)** IDs de tokens idénticos en cada solicitud y condiciones coincidentes (misma
  carga de trabajo, versiones, huellas de configuración, GPU, dtype, mapa de dispositivos, afinidad)
  en todas las ejecuciones de un escenario;
- **(b)** rendimiento en caliente SIN FIJAR: la ejecución graph MÁS LENTA es >= 1.20x la ejecución de línea base MÁS RÁPIDA (límite por ejecución, no una media agrupada);
- **(c)** VRAM reservada máxima adicional, incluida la carga y la primera captura, <= 256
  MiB por encima de la línea base;
- **(d)** de extremo a extremo: `soup infer` con 24 prompts de longitudes variadas y sin ordenar
  (21-1339 caracteres), incluida carga y compilación, sin fijar, es más rápido con la
  bandera en AMBAS ejecuciones graph que en AMBAS de línea base, con respuestas idénticas.

Árboles en prueba (`results/cuda-graph-decode/cg-trees.txt`): A = `8736def5`, la
base de fusión de la rama con `origin/main`, que no tiene la bandera; B = `fb90ecf7`,
la cabeza revisada de `feat/cuda-graph-decode` después de la ronda de correcciones. Ambos se prepararon
con `git archive`, para que ninguno cambiara durante la ejecución.

Modelo: `Qwen/Qwen2.5-1.5B-Instruct` en la revisión `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
FP16, sin cuantizar, totalmente residente (`_load_model` de `commands/infer.py`, el
cargador publicado). Carga de trabajo por solicitud del arnés: el prompt de chat fijo de 82 tokens
del experimento, 128 tokens nuevos codiciosos, batch 1. Tres calentamientos y siete
solicitudes cronometradas por proceso nuevo; CUDA sincronizado en ambos límites del cronometraje; el
intervalo cronometrado incluye tokenización, transferencias, generación y detokenización.

## 1. Condiciones, incluida la desviación del plan

- **Lanzamiento: proceso en segundo plano de la sesión del agente, no la ventana en primer plano
  de la persona responsable.** El plan pedía iniciar el gate desde una ventana de PowerShell enfocada
  para que Windows no redujera la prioridad del proceso. La persona responsable indicó
  a la sesión que lo ejecutara. Ambos brazos de cada escenario se ejecutaron con el mismo
  lanzamiento, en orden A-B-B-A-B-A, así que las razones (b, d) comparan condiciones equivalentes; las
  cifras absolutas sin fijar pueden ser bajas frente a un lanzamiento en primer plano y no deben
  citarse como «lo que obtiene un usuario» sin esa aclaración.
- Sin fijar (`cpu_affinity.effective` = los 32 procesadores lógicos) en las doce
  ejecuciones de escenario y los cuatro brazos de CLI; `--cpu-affinity 0` sólo para las dos ejecuciones puente.
- Equipo en reposo, comprobado antes del inicio (`results/cuda-graph-decode/quiet-box.txt`):
  GPU 0% / 0 MiB / 50 C, plan de energía Balanced, corriente alterna, batería 100%, 17 GB libres
  de memoria virtual, carga de CPU del equipo 0-15%. Se pidió a dos sesiones de agente pares que
  pausaran y confirmaron estar inactivas; el único proceso externo restante era un `pytest -q`
  de otro proyecto, con 141 MB y 48 s de CPU total, considerado insignificante.
- `env.ps1` (archivo de entorno del experimento, cargado por el gate) establece sólo
  ubicaciones de caché (HF, torch, inductor, pip), `HF_HUB_DISABLE_*`, `TEMP`,
  `SOUP_HOME`, `PYTHONIOENCODING=utf-8` y `TOKENIZERS_PARALLELISM=false`. El gate borra su
  `PYTHONPATH`; cada brazo importa desde su propio árbol preparado y verifica `soup_cli.__file__` antes de ejecutarse.
- **El arnés no puede ejecutarse desde `benchmarks/harness/` tal como se publicó**: su contención de rutas
  se basa en el abuelo del directorio (el espacio de trabajo del experimento), e
  importa el checkout Soup indicado por `--source`. Por eso el script del gate
  copia `cuda_graph_decode_bench.py` y `cuda_graph_decode_compare.py` al espacio de trabajo y ejecuta esas
  copias. Bytes publicados: `bench` es el `bench_decode.py` del experimento con una línea vacía añadida por
  `ruff` I001 (sha256 `fb3cc16d…` -> `fc7a9ccc…`); `compare` es idéntico byte a byte a
  `compare_results.py` (`6a23c28e…`).
- Los diagnósticos de fase prefill estaban desactivados (`--prefill-phase-repeats 0`): ningún criterio
  los usa y importan un archivo del experimento que esta rama no publica.
- Hora de ejecución: 08:06Z -> 08:35Z el 2026-09-25, 18 procesos nuevos. No hubo evento de suspensión
  en ese intervalo (la laptop había estado suspendida de 01:02 a 11:22 hora local esa noche,
  lo que arruinó un bloque de la sonda de nueva grabación de la sección 4; el propio gate
  se inició después de reanudar).

## 2. Resultados

### Escenario «full» (el modelo base)

| ejecución | tok/s (agrupado) | tok/s por solicitud | reservado MiB | asignado MiB | uso GPU % | potencia W | SM MHz | CPU proceso % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full-a1` | 20.71 | 12.8 - 33.4 | 3002 | 2984 | 26.1 | 39.6 | 2755.8 | 94 |
| `full-b1` | 62.70 | 59.0 - 65.8 | 3044 | 2995 | 74.4 | 69.4 | 2605.5 | 94 |
| `full-b2` | 60.10 | 58.1 - 61.8 | 3044 | 2995 | 72.4 | 68.1 | 2578.6 | 94 |
| `full-a2` | 13.51 | 12.5 - 18.0 | 3002 | 2984 | 16.6 | 34.9 | 2751.1 | 93 |
| `full-b3` | 61.85 | 60.7 - 62.8 | 3044 | 2995 | 69.3 | 64.5 | 2610.9 | 97 |
| `full-a3` | 16.01 | 12.5 - 33.7 | 3002 | 2984 | 20.4 | 37.3 | 2745.6 | 94 |

- (a) PASS: IDs idénticos en las 42 solicitudes; todas las comprobaciones de condiciones coincidentes son verdaderas.
- (b) PASS: B más lenta 60.10 / A más rápida 20.71 = **2.902x**.
- (c) PASS: **+42 MiB** reservados (3044 frente a 3002), +11 MiB asignados.
- Agrupado: +279.1% (intervalo percentil bootstrap de 95% +225 a +330%, condicionado
  a las ejecuciones observadas, no un intervalo de ensayos emparejados).

### Escenario «lora» (el mismo modelo base bajo LoRA sintético de rango 16 en q/k/v/o, sin fusionar)

| ejecución | tok/s (agrupado) | tok/s por solicitud | reservado MiB | asignado MiB | uso GPU % | potencia W | SM MHz | CPU proceso % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora-a1` | 17.80 | 12.3 - 20.0 | 3020 | 3000 | 28.9 | 38.0 | 2749.1 | 98 |
| `lora-b1` | 49.19 | 47.4 - 51.9 | 3060 | 3012 | 68.5 | 55.7 | 2353.9 | 91 |
| `lora-b2` | 60.30 | 59.7 - 61.0 | 3060 | 3012 | 79.1 | 69.2 | 2614.1 | 97 |
| `lora-a2` | 15.17 | 9.2 - 20.6 | 3020 | 3000 | 24.1 | 36.3 | 2764.6 | 94 |
| `lora-b3` | 51.10 | 47.6 - 55.6 | 3060 | 3012 | 65.2 | 58.7 | 2402.4 | 95 |
| `lora-a3` | 8.10 | 7.3 - 8.6 | 3020 | 3000 | 12.3 | 31.2 | 2757.5 | 96 |

- (a) PASS: IDs idénticos en las 42 solicitudes. El adaptador no es una operación vacía: la
  comprobación de identidad de Task-5 registró que sus respuestas difieren de las del modelo base.
- (b) PASS: B más lenta 49.19 / A más rápida 17.80 = **2.763x**.
- (c) PASS: **+40 MiB** reservados (3060 frente a 3020).
- Agrupado: +334.6% (bootstrap +261 a +413%, misma salvedad).

### Puente fijado a las cifras publicadas del experimento

| ejecución | afinidad | tok/s | experimento (2026-09-23, mismo equipo) |
|---|---|---:|---:|
| `pinned-a` (línea base) | CPU 0 | 31.29 | 32.16 (A1-A3: 31.29-33.10) |
| `pinned-b` (graphs) | CPU 0 | 65.64 | 62.57 (B1-B3: 54.11-68.50) |

Ambas cifras están dentro del rango de variación entre ejecuciones del propio experimento, con IDs idénticos en
`pinned-b` frente a `pinned-a`. Así que este equipo reproduce el experimento al
medirlo de la misma forma; los escenarios sin fijar anteriores representan otra condición,
no otra máquina.

### (d) `soup infer` de extremo a extremo, 24 prompts variados, 128 tokens cada uno, A-B-B-A

| brazo | árbol | tiempo total (s, carga + calentamiento + 24 filas) | Duration del comando (s) | Throughput del comando |
|---|---|---:|---:|---:|
| `a1` | main | 282.2 | 265.6 | 11.6 tok/s |
| `b1` | candidato, `--cuda-graphs` | 101.5 | 57.3 | 53.6 tok/s |
| `b2` | candidato, `--cuda-graphs` | 96.3 | 55.1 | 55.8 tok/s |
| `a2` | main | 131.4 | 115.5 | 26.6 tok/s |

- PASS: respuestas idénticas en las 24 filas de los cuatro brazos, y ambos tiempos totales graph
  (96-102 s) por debajo de ambos tiempos de línea base (131-282 s), incluida la compilación.
- La Duration del comando excluye el calentamiento único (el reloj del resumen
  empieza después), por eso el tiempo total de `b1` supera su Duration en ~44 s,
  mientras que el de `a1` lo hace en ~17 s: la diferencia es la carga del modelo más la
  captura.

## 3. Qué indican las cifras además de los veredictos

- **Sin fijar, la línea base es el brazo inestable.** Su tasa por ejecución varía de 13.5 a
  20.7 tok/s (full) y de 8.1 a 17.8 (lora); dentro de una ejecución, las solicitudes varían de 12.5
  a 33.7. El brazo graph varía entre 60.1 y 62.7 (full), 4% entre ejecuciones. Fijar el experimento
  estabilizó la línea base en 31-33; la bandera la estabiliza
  sin fijar, porque el host tiene que programar ~1,230 lanzamientos de kernel menos por token.
- **Uso de GPU** (en todo el dispositivo, muestras de `nvidia-smi` cada 1 segundo dentro de los
  intervalos cronometrados): 12-29% sin la bandera, 65-79% con ella. La CPU del proceso se mantuvo en
  91-98% en ambos brazos: la línea base consume un núcleo al lanzar kernels; el brazo graph
  lo consume en el bucle de muestreo.
- Potencia: 31-40 W en línea base, 56-74 W con graphs. La bandera cambia inactividad por trabajo; no
  hace que la tarjeta cueste menos por token que a plena utilización.

## 4. La cuestión de volver a grabar, medida antes del gate

Hallazgo de revisión: transformers 5.17 construye un `StaticCache` NUEVO en cada llamada a `generate()`
(`_prepare_static_cache`), y los árboles cudagraph vuelven a grabarse cuando cambia la dirección de una entrada
estática, así que quizá cada fila vuelva a grabarse y, después de
`cudagraph_unexpected_rerecord_limit` (128), la decodificación pase silenciosamente a ejecución eager.
Se midió con `benchmarks/harness/cuda_graph_decode_rerecord_probe.py` (un
proceso, greedy, 32 tokens, mismo generador de prompts variados que (d)):

| filas | grabaciones al final | regrabaciones contadas | `cudagraph_skips` | reservada | tok/s en bloques de 128 filas |
|---:|---:|---:|---:|---:|---|
| 256 | 146 | 0 | 0 | plana 3044-3070 MiB | 42.0 -> 48.1 (primeras frente a últimas 32) |
| 1024 | 341 | 0 | 0 | plana, máx. 3074 MiB | 42.0 / 46.4 / 47.9 / 48.0 / 47.4 / 47.5 / [sleep] / 53.2 |

Aproximadamente una grabación cada tres filas, y ninguna cuenta para el límite:
en torch 2.14 un `StaticInputIdxMismatch` provoca una regrabación, pero está exento de
`cudagraph_unexpected_rerecord_limit` (`cudagraph_trees.py`, el bloque que
incrementa `num_rerecord`). El rendimiento no se degrada hasta las 1024 filas y
la memoria no crece. El bloque `[sleep]` corresponde a la laptop suspendida dentro de la fila 893;
su mediana por fila fue de 0.613 s, acorde con las filas vecinas. Un
`StaticCache` persistente (con `.reset()` por fila y direcciones estables por construcción)
eliminaría las regrabaciones; es un seguimiento posterior, no una condición de publicación.

## 5. Lo que este registro NO muestra

- **Un lanzamiento en primer plano.** Consulta la sección 1. Las razones se mantienen; las tasas
  absolutas sin fijar son un límite inferior de lo que mostraría una ventana enfocada.
- **`soup chat`** (fuera del alcance: un historial creciente recompila la caché estática en cada turno)
  ni **`soup serve`**.
- **Arquitecturas distintas de Qwen2**; Llama pasa el validador y está cubierto por la prueba GPU
  de modelo diminuto, no por esta medición.
- **Linux o macOS**, modelos cuantizados o descargados, varias GPU, rendimiento con `--temperature
  > 0` (cada solicitud cronometrada aquí es greedy; la prueba CLI confirma que las filas muestreadas
  usan la temperatura solicitada, y la identidad greedy es la evidencia del cambio de caché, no una
  medición de muestreo).
- **`--max-tokens` largo.** Todo aquí usa 128 tokens (32 en la sonda).
  Con `keep_inference_input_mutations=False`, cada actualización K/V es una salida graph
  más una copia de vuelta, aproximadamente el tráfico de una caché KV adicional por paso, que crece
  con prompt + max_tokens. No medido.
- **Tamaño de batch > 1.** La CLI usa batch 1.

## 6. Archivos sin procesar

`results/cuda-graph-decode/`: los catorce informes del arnés
(`cg-gate-{full,lora}-{a1,b1,b2,a2,b3,a3}.json`, `cg-gate-pinned-{a,b}.json`,
cada uno con muestras por solicitud, IDs de corrección y telemetría de 1 segundo), las dos
comparaciones, los tres veredictos, `cli-wall-seconds.json`, `cg-trees.txt`, las dos
sondas de regrabación y `quiet-box.txt`. Arnés: `harness/cuda_graph_decode_gate.ps1`
(el ejecutor), `_bench.py`, `_compare.py`, `_verdict.py` (los criterios, con
sus propias pruebas en `tests/test_cuda_graph_decode_verdict.py`), `_prompts.py`,
`_synthetic_lora.py`, `_rerecord_probe.py`.
