<!--
Registro de mediciones de trabajo, publicado íntegro. La regla de decisión de §2 y el
protocolo de §3 se escribieron en el issue #1702 el 2026-10-07, antes de cualquier ejecución de la
serie de velocidad, y se copian aquí sin cambios: una regla escrita después de conocer las cifras
no es una regla.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB, driver 616.92),
Intel i9-14900HX, 31.7 GB DDR5-5600, Samsung PM9B1 NVMe (C:). Windows 11 Pro
10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2, accelerate 1.15.0.
Soup: branch perf/pinned-arena-packing-1702 en main 7016df2b; soup_cli.__file__
se registra en cada resultado de ejecución.
Arnés: benchmarks/harness/issue1702_arena_pack_gate.py, un proceso por ejecución,
a través de stream_probe.build (la ruta enviada shard_checkpoint -> build_streamed_model).
Su programación y regla están en tests/test_issue1702_arena_pack_gate.py.
-->

# Registro del gate — empaquetado de arenas de mayor a menor (#1702)
> 🇪🇸 Traducción al español de [`gate-1702-arena-packing.md`](gate-1702-arena-packing.md). Ante cualquier duda, manda el original en inglés.

**Estado (2026-10-07): ahorro de memoria confirmado, control estricto aprobado, serie de velocidad ejecutada:
NO HAY DIFERENCIA medible a +/-15.1%.**
En `main` más el cambio, el nivel RAM de Qwen3-8B NF4 fija en páginas 6,442,450,944 bytes,
mientras que el plan en orden fija 8,589,934,592, y tres pasos deterministas del optimizador
son idénticos byte a byte entre ambos planes (§4). La serie de velocidad (§5) completó 40
de 40 ejecuciones: la diferencia media del nivel RAM es +1.96%, dentro de la banda de +/-15.13%
del control; por la regla escrita de antemano, no se afirma ni aceleración ni ralentización.
La banda es amplia; la serie no puede descartar un cambio menor que eso.

---

## 0. La pregunta

`plan_pinned_arenas` ahora también planifica una estrategia estable de first-fit decreasing y la devuelve cuando fija en páginas estrictamente menos bytes que el recorrido en orden. En el nivel RAM eso cambia dónde vive cada tensor del almacén: los tensores de una capa ya no son vecinos en una misma arena. Las copias de host a dispositivo son por tensor y cada tensor sigue siendo una vista contigua fijada en páginas, así que no se espera ningún efecto en el tiempo del paso. Esperarlo no equivale a medirlo. ¿El diseño empaquetado cambia el tiempo del paso de entrenamiento?

## 1. Lo establecido sin medir tiempos

| | plan en orden | plan empaquetado |
|---|---:|---:|
| Nivel RAM Qwen3-8B NF4, arenas | 4 x 2 GiB | 3 x 2 GiB |
| bytes fijados en páginas | 8,589,934,592 | 6,442,450,944 |
| bytes del almacén (1,154 tensores) | 6,073,035,760 | 6,073,035,760 |
| RSS máximo del proceso, ejecución de 3 pasos | 10.85 GB | 8.71 GB |
| VRAM máxima asignada / reservada | 2.807 / 3.047 GB | 2.807 / 3.047 GB |
| Nivel de disco Qwen3-8B NF4, almacenamiento temporal fijado en páginas | 4,294,967,296 | 4,294,967,296 (mismo plan) |

Las tres primeras filas también son cálculos aritméticos y se prueban sin GPU
(`tests/test_issue1702_pinned_arena_packing.py`). Las demás filas provienen de las cuatro
ejecuciones de §4.

## 2. La regla de decisión (de #1702, escrita antes de cualquier ejecución)

1. Nunca se descartan ejecuciones fallidas. Cada una se enumera con su error y cuenta contra
   su brazo. Un bloque de semilla/modo con una ejecución fallida no tiene `d`. Un modo con menos de 4 de
   5 `d` es INCONCLUSO.
2. Banda de ruido `b` = el mayor entre 3% y el máximo `|d|` observado en el control.
3. Si la media de `d` del control queda fuera de +/-3%, la serie es INCONCLUSA: el equipo no estaba
   estable o el orden no compensó. No se afirma nada sobre velocidad. Se permite una repetición en
   otra ventana, y se publican ambas series.
4. En caso contrario, con `m` = media de `d` en el nivel RAM:
   - `m < -b`: más lento. La cifra se incluye en el PR y el cambio se decide como intercambio de
     velocidad por 2 GiB de memoria fijada en páginas.
   - `m > +b`: más rápido, se informa con la cifra.
   - cualquier otro valor: no hay diferencia medible a +/-`b`; el PR afirma únicamente el ahorro de memoria.
5. No se ajusta nada después de ver los resultados: ni se descarta una semilla, ni se agregan
   ejecuciones aparte de la única repetición declarada.

`d = mean(packed) / mean(in-order) - 1` por semilla y modo, sobre las dos ejecuciones de
cada brazo en ese bloque, en tokens de entrada por segundo a lo largo de los 10 pasos medidos (5,120 posiciones
divididas entre la suma de los 10 tiempos de paso sincronizados con CUDA).

## 3. El protocolo (de #1702, escrito antes de cualquier ejecución)

- **Receta.** Qwen/Qwen3-8B (revisión `b968826d9c46dd6066d109eabc6255188de91218`), NF4
  con doble cuantización, cálculo bf16, LoRA r16 (alpha 32, dropout 0) en
  `q_proj`/`v_proj`, `PagedAdamW8bit` a 2e-4, batch 1, bloques de 512 tokens, 5 pasos de calentamiento y
  10 pasos de optimización medidos, dos búferes de capas GPU, lectura anticipada 2, límite de asignador
  de 4,000,000,000 bytes por proceso. Semillas 3 / 17 / 29 / 43 / 59.
- **Datos.** 32 bloques de 512 tokens de WikiText-2 raw v1, tokenizados sin tokens especiales;
  una ejecución toma los primeros 15. El archivo no está en el repositorio (sha256
  `663f28bf9bf8307a9018c3bf4281d58d053fa55752de01f160101bc83cb0a8ee`). Sin
  `--batches`, el arnés genera ids de tokens aleatorios con semilla y lo indica en el
  resultado.
- **Brazos.** `in-order`: dentro del proceso, el recorrido largest-first del planificador apunta al recorrido
  en orden, por lo que la opción candidata nunca puede ganar y el plan es el anterior a #1702.
  `packed`: el planificador enviado. No existe un campo de configuración para esto.
- **Modos.** Nivel RAM: el objeto de estudio. Nivel de disco con E/S directa: control negativo.
  Su plan es igual en ambos brazos, así que cualquier diferencia entre sus brazos es ruido u orden de ejecución.
- **Orden.** Una serie, cada ejecución en un proceso nuevo. Por semilla y modo, las cuatro ejecuciones siguen
  el patrón A B B A; la siguiente semilla empieza con el otro brazo y el otro modo.
  5 semillas x 2 modos x 4 ejecuciones = 40 ejecuciones
  (`python benchmarks/harness/issue1702_arena_pack_gate.py` imprime la lista).
- **Validez.** Una ejecución RAM que no muestre 8,589,934,592 bytes fijados en páginas en el
  brazo en orden o 6,442,450,944 en el empaquetado, o un control cuyos brazos fijen
  cantidades distintas, indica un fallo del arnés e invalida la serie. Son ejecuciones fallidas las que no fijan
  en páginas el almacenamiento temporal, las del control que no leen mediante E/S directa y las que no dan
  resultado después de 15 minutos (se detienen).
- **Alimentación** (añadida el 2026-10-07, antes de cualquier ejecución de la serie). Un ensayo del driver
  en Qwen2.5-1.5B encontró la laptop con batería: un paso tardó 0.83 s allí frente a 0.31 s
  conectada a corriente. El driver no inicia una serie con batería y detiene la serie si el
  equipo pasa a batería; una serie con cualquier ejecución registrada con batería, antes o después, es
  INCONCLUSA y cuenta como la única repetición permitida.
- **El equipo.** No se ejecuta nada más durante toda la serie: ni suites de pruebas ni otro trabajo en GPU.
  Se registran la RAM en uso, la carga de confirmación y la alimentación de CA antes y después de cada ejecución.

```bash
python benchmarks/harness/issue1702_arena_pack_gate.py --mode series \
    --weights $MODEL --shards $SHARDS --batches $BATCHES --out $OUT
```

## 4. Control estricto: idéntico byte a byte (2026-10-07)

Tres pasos del optimizador, una semilla (3), `torch.use_deterministic_algorithms(True)` y
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, un proceso nuevo por brazo y modo. Después de cada paso,
la ejecución calcula el SHA-256 de los logits completos, la pérdida, todos los 144 gradientes LoRA y los
144 tensores de adaptador actualizados: 290 resúmenes por paso, 870 por ejecución.

| modo | en orden | empaquetado | resúmenes iguales |
|---|---:|---:|---|
| Nivel RAM, bytes fijados en páginas | 8,589,934,592 | 6,442,450,944 | 870 de 870 |
| Nivel de disco (E/S directa), bytes fijados en páginas | 4,294,967,296 | 4,294,967,296 | 870 de 870 |

Las cuatro ejecuciones dan el mismo resumen de resúmenes,
`1b127f4880e4f35d07609a1c503728e17ac22e2f0ea0f69f81ba62f023e04140`: ambos planes coinciden
entre sí, y el nivel RAM coincide con el nivel de disco. Pérdidas en los tres
pasos: 2.699491, 3.197587, 3.101483 en todas las ejecuciones.
Archivo de resumen: `benchmarks/results/probe-rtx5070/issue1702/strict.json`.

```bash
python benchmarks/harness/issue1702_arena_pack_gate.py --mode strict \
    --weights $MODEL --shards $SHARDS --batches $BATCHES --seed 3 --out $OUT
```

Esto afirma algo sobre tres pasos bajo ejecución determinista. Muestra que los dos
formatos entregan los mismos bytes al mismo cálculo. No afirma nada sobre la ejecución por defecto
(no determinista) ni sobre una ejecución larga.

## 5. Serie de velocidad: NO HAY DIFERENCIA medible a +/-15.1% (2026-10-07)

Una serie, 16:40Z-17:16Z (35.6 minutos), en `4e692d31`, en una ventana declarada sin
suite de pruebas ni otro trabajo en GPU en el equipo. Se completaron 40 de 40 ejecuciones; no hubo fallos ni
fallos del arnés. Todos los registros muestran alimentación de CA; antes de las ejecuciones había 13.3-16.5 GB de RAM
libre y la carga de confirmación era 52.6-56.8%. El servicio Windows Search estaba en ejecución.
Archivos: `benchmarks/results/probe-rtx5070/issue1702/series.jsonl` (una línea por ejecución, en
orden) y `verdict.json`.

| semilla | Nivel RAM `d` | control `d` |
|---:|---:|---:|
| 3 | -2.71% | +3.03% |
| 17 | +4.75% | -0.39% |
| 29 | -1.41% | +9.69% |
| 43 | +3.58% | -0.34% |
| 59 | +5.61% | -15.13% |
| media (SD muestral) | **+1.96%** (3.77%) | -0.63% (9.09%) |

Según §2: la media del control es -0.63%, dentro de +/-3%, así que la serie es válida (regla 3). La
banda de ruido `b` es 15.13%, el mayor `|d|` del control (regla 2). La media `d` del nivel RAM,
+1.96%, está dentro de +/-15.13% (regla 4): **no hay diferencia medible a +/-15.13%.
El PR afirma únicamente el ahorro de memoria.** No corresponde repetir: la regla lo permite solo después de una
serie inconclusa.

| modo, brazo (10 ejecuciones cada uno) | tokens/s, media (SD) | paso mediano | construcción del almacén | RSS máximo | bytes fijados en páginas |
|---|---:|---:|---:|---:|---:|
| RAM, en orden | 515.8 (30.7) | 0.972 s | 11.5 s | 10.84 GB | 8,589,934,592 |
| RAM, empaquetado | 525.5 (31.5) | 0.963 s | 9.0 s | 8.69 GB | 6,442,450,944 |
| disco, en orden | 273.1 (24.2) | 1.866 s | 1.7 s | 6.13 GB | 4,294,967,296 |
| disco, empaquetado | 270.6 (22.2) | 1.882 s | 1.5 s | 6.12 GB | 4,294,967,296 |

La VRAM máxima fue de 2.807 GB asignados y 3.047 GB reservados en todas las ejecuciones.

**Qué significa esta resolución.** La banda es amplia porque el control se ejecutó en dos
regímenes. 15 de sus 20 ejecuciones tienen un paso mediano de 1.86-2.01 s y 5 de 1.62-1.64 s:
las primeras tres ejecuciones de la semilla 29 (una en orden, dos empaquetadas) y la primera y la última
 ejecución de la semilla 59 (ambas en orden). Un bloque dividido entre ambos regímenes da +9.69% y
-15.13%; los otros tres bloques se mantienen dentro de aproximadamente 3%. El plan es igual en ambos brazos
en el nivel de disco, así que esto se debe al equipo y no al cambio, que es precisamente la función del control.
El nivel RAM no muestra esa división (sus pasos medianos abarcan 0.875-1.102 s y sus cinco `d` van
de -2.71% a +5.61%), pero la regla toma la banda del control y se escribió primero. Por tanto, esta
serie no puede descartar un cambio menor al 15% en ninguna dirección, y no muestra ninguno.

## 6. Lo que este registro no demuestra

- Una tarjeta física de 4 GB: la tarjeta tiene 8 GB y las ejecuciones usan un límite de asignador
  de software, que no limita el contexto CUDA, el driver ni la memoria fijada en páginas.
- Ningún modelo aparte de Qwen3-8B. Las pruebas del planificador cubren dos geometrías Qwen2.5-14B solo
  como cálculo aritmético (aproximadamente 7% menos memoria fijada en páginas en cada una).
- Que una negativa a fijar páginas sea menos probable. El plan empaquetado solicita 2 GiB
  menos a Windows, y eso es todo lo que se sabe.
