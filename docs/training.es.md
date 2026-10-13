# Tareas y métodos de entrenamiento

> 🇪🇸 Traducción al español de [`training.md`](training.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-entrenamiento, destilación, clasificación, visión/audio/TTS, desaprendizaje, RAFT/RA-DIT y los detectores de blindaje del loop.

> **¿Vas a entrenar un modelo más grande que tu GPU?** `training.stream_layers: true` hace streaming de la base congelada desde la RAM de la CPU (con desborde a disco NVMe), una capa decodificadora a la vez, así que la VRAM máxima queda acotada por una capa en lugar de por el modelo completo. Agrega `quantization: 4bit` y una base de 8B cabe en una tarjeta de 4 GB con batch 1 y 512 tokens por step; en un modelo con vocabulario de 128k, cada 512 tokens adicionales cuestan alrededor de 1 GB más (ver [Dimensionar una corrida con streaming](performance-and-quantization.md#sizing-a-streaming-run-v0723)). Funciona para `sft` y, desde la v0.72.4, para `dpo` / `orpo` / `simpo` / `kto` — el modelo de referencia de DPO es la misma base en streaming con sus adaptadores desactivados, así que no necesita una segunda copia del modelo (en un checkpoint con pesos desatados, `dpo` y `kto` todavía mantienen una copia de la cabeza de salida por step) — ver [Layer Streaming](performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724).

**Contenido:**

- [Qué tareas aplican `training.quantization`](#qué-tareas-aplican-trainingquantization)
- [Ensayo de aprendizaje continuo (`--replay`)](#ensayo-de-aprendizaje-continuo---replay)
- [Blindaje del loop](#blindaje-del-loop)
- [Desaprendizaje (`task='unlearn'`, NPO / SimNPO / RMU)](#desaprendizaje-taskunlearn-npo--simnpo--rmu)
- [Pre-entrenamiento continuado](#pre-entrenamiento-continuado)
- [Destilación de conocimiento](#destilación-de-conocimiento)
- [Clasificación de secuencias](#clasificación-de-secuencias)
- [Reasoning effort + control de EOT](#reasoning-effort--control-de-eot)
- [Variantes de loss EBFT / GDPO](#variantes-de-loss-ebft--gdpo)
- [Variantes de objetivo GRPO](#variantes-de-objetivo-grpo)
- [Modelo de recompensa de proceso (PRM)](#modelo-de-recompensa-de-proceso-prm)
- [GRPO guiado por PRM (RL supervisado por proceso)](#grpo-guiado-por-prm-rl-supervisado-por-proceso)
- [Loss de preferencia multiobjetivo ponderado](#loss-de-preferencia-multiobjetivo-ponderado)
- [Compatibilidad con modelos MoE](#compatibilidad-con-modelos-moe)
- [Fine-tuning de visión / multimodal](#fine-tuning-de-visión--multimodal)
- [Fine-tuning de audio / voz](#fine-tuning-de-audio--voz)
- [GRPO Plus — variantes de objetivo, RL de contexto largo, agentes multiturno](#grpo-plus--variantes-de-objetivo-rl-de-contexto-largo-agentes-multiturno)
- [Entrenamiento DPO](#entrenamiento-dpo)
- [Variedad de preferencias — BCO + despachador unificado + variantes de KL](#variedad-de-preferencias--bco--despachador-unificado--variantes-de-kl)
- [Entrenamiento GRPO (razonamiento)](#entrenamiento-grpo-razonamiento)
- [Fine-tuning de tool-calling](#fine-tuning-de-tool-calling)
- [PPO / pipeline completo de RLHF](#ppo--pipeline-completo-de-rlhf)
- [Entrenamiento KTO (preferencias no emparejadas)](#entrenamiento-kto-preferencias-no-emparejadas)
- [Entrenamiento ORPO (sin modelo de referencia)](#entrenamiento-orpo-sin-modelo-de-referencia)
- [Entrenamiento SimPO (preferencia simple)](#entrenamiento-simpo-preferencia-simple)
- [Entrenamiento IPO (preferencia regularizada)](#entrenamiento-ipo-preferencia-regularizada)
- [RAFT — fine-tuning aumentado por recuperación](#raft--fine-tuning-aumentado-por-recuperación)
- [RA-DIT — ajuste de instrucciones dual aumentado por recuperación](#ra-dit--ajuste-de-instrucciones-dual-aumentado-por-recuperación)
- [Entrenamiento con currículo (BETA)](#entrenamiento-con-currículo-beta)
- [Fine-tuning de TTS (BETA, v0.52.0)](#fine-tuning-de-tts-tasktts-beta-activo-en-v07120)
- [Entrenamiento de clasificador / reranker / cross-encoder (BETA, v0.52.0)](#entrenamiento-de-clasificador--reranker--cross-encoder-beta-v0520)
- [Destilación de conocimiento (BETA, v0.52.0)](#destilación-de-conocimiento-beta-v0520)
- [EBFT + GDPO (BETA, v0.52.0)](#ebft--gdpo-beta-v0520)
- [`reasoning_effort` y `train_on_eot` de gpt-oss (v0.52.0)](#reasoning_effort-y-train_on_eot-de-gpt-oss-v0520)
- [División de validación y evaluación (`training.eval_steps`)](#división-de-validación-y-evaluación-trainingeval_steps)
- [Semillas y reproducibilidad (`training.seed`)](#semillas-y-reproducibilidad-trainingseed)
- [Rewind — qué fila disparó la loss (`soup rewind`)](#rewind--qué-fila-disparó-el-loss-soup-rewind)
- [Fine-tuning completo (`lora.r: 0`)](#fine-tuning-completo-lorar-0)

---

## División de validación y evaluación (`training.eval_steps`)

`data.val_split` (por defecto `0.1`) reserva esa proporción de las filas fuera del entrenamiento, y la corrida las evalúa:

```yaml
data:
  val_split: 0.1    # el valor por defecto
training:
  eval_steps: 50    # opcional; sin definir = evalúa al final de cada epoch
```

- **Por defecto: cada epoch.** Al final de cada epoch el entrenador hace una pasada sobre la división de validación y registra `eval_loss`. Aparece como la fila **Val loss** del panel en vivo, se guarda como `val_loss` en las métricas de la corrida dentro del experiment tracker y se transmite como `TrainEvent.val_loss` a la Web UI. La pasada es un forward sobre las filas reservadas, unos pocos por ciento de la corrida con la división por defecto.
- **`training.eval_steps: N`** evalúa cada N steps del optimizador en su lugar, contados después de la acumulación de gradientes igual que `save_steps`. El último step también se evalúa cuando no es múltiplo de N, así que un valor mayor que la corrida igual evalúa una vez.
- **Batch size.** La evaluación corre con el batch de entrenamiento por dispositivo ya resuelto (después de `batch_size: auto`), no con el 8 por defecto de Hugging Face, así que una tarjeta dimensionada para entrenar también admite la evaluación. Las corridas con streaming de capas (`stream_layers: true`) evalúan igual, haciendo streaming de las capas congeladas como lo hace un step de entrenamiento.
- **`val_split: 0`** entrena con todas las filas y no evalúa nada, salvo que los datos traigan su propia división de validación: la división `validation` de un dataset del HF hub, o la división `val` / `validation` de un cache pre-tokenizado.

**Tareas basadas en generación (`grpo`, `ppo`, `online_dpo`).** Evaluarlas implica generar completions, lo que cuesta por fila más o menos lo mismo que entrenar con ella. Por eso no evalúan por defecto, y tampoco reservan filas: se ignora `data.val_split`, entrenan todas las filas y la corrida imprime una nota de una línea diciéndolo. En `grpo`, define `training.eval_steps` para reservar la división y evaluarla; TRL genera completions para los prompts reservados y registra `eval_loss` junto con las recompensas de evaluación. En `grpo` el número de validación de la corrida es la recompensa reservada, `eval_reward` (la recompensa media sobre las completions reservadas, más alto es mejor): el panel en vivo lo muestra como `Val reward`, el tracker lo guarda en la columna `val_reward` de la tabla de métricas y el flujo de eventos de entrenamiento lo lleva como `val_reward` (#1389). El `eval_loss` de TRL en `grpo` es su objetivo de política con razón de importancia 1, no una verosimilitud: las ventajas se normalizan dentro de cada grupo, así que se queda cerca de cero, puede ser negativo y no mide la calidad sobre las filas reservadas. No se registra ni se muestra como loss: la serie `val_loss` de una corrida `grpo` está vacía y `Val loss` no aparece. La elección por tarea vive en `VALIDATION_METRICS` dentro de `utils/eval_schedule.py`; cualquier otra tarea registra `eval_loss` como `val_loss`. TRL necesita grupos completos de `num_generations` completions en un batch de evaluación, así que `grpo` evalúa con el mayor múltiplo de `num_generations` que quepa en el batch de entrenamiento, y lo dice cuando eso difiere del batch de entrenamiento. Las recompensas de la evaluación nunca llegan al detector de reward hacking, al controlador de mitigación ni al detector de echo trap: esos solo leen las generaciones de entrenamiento. `ppo` y `online_dpo` rechazan `training.eval_steps`: el loop de PPO de TRL nunca llama a `evaluate()`, y el `evaluate()` de online DPO falla con filas que solo tienen prompt. Una evaluación basada en generación para esas dos es una función aparte.

**Rechazado al cargar la config.** `training.eval_steps` con `backend: mlx` (mlx-lm evalúa la división con su propia cadencia, #739), con `task: unlearn` (entrena con `forget_set` / `retain_set` y no tiene división de validación) y con `val_split: 0` en archivos locales o URIs remotas, donde no habría nada que evaluar.

**Se evalúan pero no se rastrean: `prm` y `moe_lora_routing`.** No adjuntan ningún callback de entrenamiento en vivo (#802), así que su `eval_loss` llega al historial de logs del entrenador y no al experiment tracker.

Hasta #1223 ningún entrenador del backend transformers programaba una evaluación: la división por defecto se reservaba fuera del entrenamiento y nunca se usaba, y `val_loss` quedaba vacía.

---

## Semillas y reproducibilidad (`training.seed`)

Dos perillas, ambas sin definir por defecto:

```yaml
training:
  seed: 1234        # inicialización de pesos de los params nuevos, orden de los datos, dropout
  data_seed: 99     # opcional — SOLO el orden de los datos, así la inicialización queda fija
```

`seed` llega a `TrainingArguments.seed` (que `Trainer.__init__` pasa a `transformers.set_seed`, y cubre `random`, `numpy` y `torch`) y al sampler FFD de multipack. `data_seed` llega a `TrainingArguments.data_seed`; defínelo junto con `seed` para variar solo el orden en que se ven las filas manteniendo fija la inicialización.

**Por qué lo quieres.** Sin una perilla de seed, cada corrida de una config tomaba el mismo valor por defecto, así que "correrla otra vez con otra seed" era imposible: las réplicas de un mismo brazo solo diferían por la permutación de filas y por el no determinismo de la GPU. Eso subestima la dispersión entre corridas, y la dispersión es el listón que tiene que superar una diferencia real entre brazos. Tres réplicas con tres seeds son la barra de error honesta más barata que puedes ponerle a un cambio de entrenamiento. Crea una config por réplica para que la seed y su directorio de salida queden juntos en la receta registrada:

```yaml
# soup-seed-1.yaml (repite para las seeds 2 y 3)
training:
  seed: 1
output: runs/seed-1
```

```bash
for s in 1 2 3; do
  soup train --config "soup-seed-$s.yaml" && echo "seed $s done"
done
```

**"Sin definir" no significa lo mismo para los dos campos.** Un `seed` sin definir se resuelve a 42, el propio valor por defecto de `TrainingArguments` de HuggingFace. Un `data_seed` sin definir se queda en `None`, que HF lee como "sigue a `seed`" y no como una seed propia, así que omitirlo no es el mismo tipo de valor por defecto que omitir `seed`. El sampler de multipack igual recibe `0`, el valor que tiene desde la v0.37.0. Los campos son `Optional[int]` en lugar de tener 42 por defecto justamente para que el entrenador pueda distinguir "sin definir" de "explícitamente 42" y mantener intactos esos valores históricos distintos.

**Qué cambia para una corrida que no define ninguno de los dos campos.** Los valores con los que entrena son los mismos que antes: seed 42, `data_seed` en `None`. Lo nuevo es *cuándo* llega la seed. Antes de #353 nada llamaba a `set_seed` antes de `get_peft_model`, así que `lora_A` y cualquier cabeza de clasificación recién inicializada se sacaban del generador por defecto de torch, que se siembra desde la entropía una vez por proceso. La inicialización de una corrida sin seed variaba entonces de un proceso a otro, y ahora es determinista en 42. Si obtenías dispersión entre réplicas de corridas que no definían seed, de ahí venía: esas corridas ahora son idénticas entre sí, y variar una réplica implica definir `training.seed` a propósito.

Límites: `[0, 2**32 - 1]` (`set_seed` alimenta `numpy.random.seed`, que rechaza cualquier valor fuera de ese rango), `0` es una seed legítima y un `true` de YAML se rechaza en lugar de convertirse en la seed 1 sin avisar. `data_seed` se reenvía a la configuración del dataloader de Accelerate y requiere `accelerate >= 1.1.0`; por debajo de eso, transformers advierte y lo ignora (`seed` no se ve afectado).

**Alcance.** Cada envoltorio de tarea pasa ambos campos a la subclase de `TrainingArguments` que construye y aplica `seed` antes de cargar el modelo, así que el adaptador LoRA y cualquier cabeza recién inicializada se sacan con la seed configurada y no con la que el proceso tuviera por casualidad (#353). `unlearn` no construye ningún `Trainer`, y su vector de control RMU también sigue a `training.seed`, quedándose en 0 cuando la seed no está definida. Las rutas de `pretrain` y de streaming de capas también toman `seed` para sus samplers.

Hasta la v0.73.0 esto solo llegaba al entrenador de **SFT**, así que `training.seed: 7` en una corrida de DPO o GRPO se aceptaba y entrenaba en silencio con 42.

La única ruta que todavía ignora ambos campos es el **backend MLX** (`backend: mlx`), cuyos entrenadores no siembran nada en absoluto — MLX tiene su propio RNG (`mx.random`). Definir cualquiera de los dos campos ahí ahora imprime una advertencia que lo nombra (`MLX backend ignores: training.seed ...`) en lugar de aceptarlo en silencio, así que una corrida de MLX no puede parecer sembrada sin estarlo.

**No es una garantía de determinismo.** Una seed fija hace reproducible el RNG del *software*. No hace que los kernels de CUDA sean reproducibles bit a bit — las operaciones atómicas no deterministas, los algoritmos con autotuning y una GPU o una versión de librería distintas todavía pueden mover los últimos dígitos. Para repeticiones bit a bit también necesitas `torch.use_deterministic_algorithms(True)`, que Soup no configura por ti.

---

## Rewind — qué fila disparó la loss (`soup rewind`)

Mientras una corrida de SFT entrena, Soup escribe `<output>/rewind.jsonl`: una línea por micro-batch con las filas del dataset que contiene, la loss media de tokens supervisados de cada fila y su conteo de tokens supervisados. Cuando la loss da un salto, `soup rewind` lee ese archivo y nombra las filas que cargaron el step. No necesita modelo ni GPU, y corre en segundos.

```yaml
training:
  rewind_log: true   # por defecto; define false para no escribir nada
```

```bash
soup rewind                      # corrida más reciente: lista los picos y detalla el peor
soup rewind <run_id> --step 340  # ordena las filas de un step
soup rewind <run_id> --top 25    # cuántas filas listar (por defecto 10)
soup rewind <run_id> --json r.json --no-preview
```

Un pico es un step cuya loss no es finita, o que supera en más de 2x la mediana de los 20 steps medidos anteriores. Las filas se ordenan por su parte de la loss de tokens sumada del step, así que una fila larga y mal formada resalta incluso cuando su media por token es normal. Ejemplo de salida (números ilustrativos):

```text
                Step 340 rows
 Row    Loss   Tokens   Share  Preview
  91  6.8120    3,900   94.1%  iVBORw0KGgoAAAANSUhEUgAAAyAAAAJYCAYAAAC…
  12  1.1034      212    0.8%  Sure — here is a summary of the article…
row 91: 94% of step 340's loss, 3,900 tokens
Inspect or drop this row, then retrain.
```

Alcance: tarea `sft` en los backends transformers y MLX, un solo proceso, en la ruta de texto plano. El registrador se desactiva solo, con una advertencia que nombra el motivo, para corridas reanudadas, `packing` / `padding_free`, multipack, visión, audio, un dataset pre-tokenizado y cualquier otra cosa que no pueda atribuir fila por fila. Las previsualizaciones solo se muestran cuando el dataset en disco todavía coincide con el que usó la corrida.

**Qué cuesta.** Escribir una línea por micro-batch midió menos de 0.1 ms por micro-batch, lo que es despreciable frente a cualquier step real, y la memoria máxima de una corrida no cambió con el registrador activado ni desactivado. El archivo crece sin tope: alrededor de 14 MB por cada 50,000 micro-batches con batch 8, o unos 100 MB para un millón de filas a lo largo de tres epochs. Bórralo, o define `rewind_log: false`, si eso te importa.

**Qué contiene.** Índices de fila enteros, una loss y un conteo de tokens por fila — nunca texto de tu dataset. Las previsualizaciones de `soup rewind` se reconstruyen en el momento de la lectura a partir de tu propio archivo de datos, y solo mientras su huella digital siga coincidiendo. Los índices y los conteos de tokens siguen siendo metadatos débiles sobre un dataset privado, así que trata el archivo como tratarías el directorio `output` donde vive.

---

## Fine-tuning completo (`lora.r: 0`)

Entrena todos los parámetros, sin adaptador:

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: sft
training:
  quantization: none    # el FT completo entrena pesos en punto flotante
  lora:
    r: 0                # <- sin adaptador; la base misma entrena
```

`lora.r: 0` es la forma admitida para el fine-tuning completo a secas. Es la que el entrenador de clasificadores de `soup` lee como "sin adaptador" desde la v0.71.12, y la que `soup card` ya resuelve como un modelo denso en lugar de un adaptador — así la ficha del modelo, la entrada del registro y el entrenador coinciden sin banderas extra. Con un rank de 0 el entrenador de SFT omite `get_peft_model` por completo: `soup train` imprime `Full fine-tuning: N parameter tensor(s) trainable (lora.r=0, no adapter)` en lugar de `LoRA applied`, y el directorio de salida contiene un modelo completo, no un adaptador por fusionar.

Requisitos, cada uno rechazado al cargar la config nombrando el motivo: `task: sft`, `backend: transformers`, `modality: text` y `quantization: none` (los pesos cuantizados no se pueden entrenar directamente — usa LoRA encima de ellos, es decir QLoRA). Es mutuamente excluyente con cualquier función de LoRA (`use_dora` / `use_vera` / `use_olora` / `use_rslora` / `rank_pattern` / `alpha_pattern` / `init_strategy` / `moe_lora` / `use_longlora` / `relora_steps` / `loraplus_lr_ratio`) — una perilla de LoRA junto a `r: 0` es una contradicción, no algo que se ignore en silencio — y con los otros dos modos de "LoRA apagado", [Spectrum](#spectrum--entrenamiento-dirigido-según-el-snr-de-capas-soup-spectrum-scan-v07123) (`unfrozen_parameters`) y LISA (`lisa_enabled`), porque cada uno decide por su cuenta qué entrena. No se puede combinar con streaming de capas: el streaming mantiene el decodificador en el meta device y entrena solo el adaptador, así que no habría nada que ajustar por completo.

`freeze_layers` / `freeze_ratio` **sí** siguen siendo válidos con `r: 0` — "entrena todo lo que esté por encima de la capa N" es una técnica real — y el entrenador respeta lo que hayan congelado en lugar de descongelarlo en silencio. Si no dejan nada entrenable, la corrida se rechaza en lugar de quemar horas de GPU en un no-op.

Elige entre los tres modos de "LoRA apagado" según cuánto quieras entrenar:

| Forma | Entrena | Úsalo cuando |
| --- | --- | --- |
| `lora.r: 0` | todo | tienes la VRAM y quieres la línea base más fuerte |
| `unfrozen_parameters` ([Spectrum](#spectrum--entrenamiento-dirigido-según-el-snr-de-capas-soup-spectrum-scan-v07123)) | un conjunto elegido a mano / ordenado por SNR | quieres calidad de FT completo en las capas que importan |
| `lisa_enabled` (LISA) | un subconjunto aleatorio rotativo | quieres calidad de FT completo con memoria tipo LoRA |

> El fine-tuning completo necesita mucha más memoria que LoRA: solo el estado del optimizador son ~8 bytes por parámetro con AdamW. `batch_size: "auto"` estima la memoria a partir de un modelo con forma de LoRA, así que en una corrida de FT completo tiende a ser optimista — define `batch_size` explícitamente, o empieza bajo y súbelo. (Esto no es nuevo de `r: 0`; lo mismo vale para las rutas de FT completo con Spectrum y LISA.)

**dtype de carga** (#339, #471, #492): una base congelada — LoRA o QLoRA — se carga con el dtype propio del checkpoint (`torch_dtype="auto"`) en lugar de convertirla siempre a fp32, ya que la base nunca recibe un step del optimizador. Medido en una H100, Llama-3.1-8B, LoRA: 48,241 MiB de pico -> 18,658 MiB, un ahorro de 28.9 GB / 2.59x. En una tarjeta CUDA anterior a Ampere (T4 / P100 / V100 / GTX 16xx / RTX 20xx), `"auto"` daría almacenamiento bf16 mientras que el cómputo del entrenamiento se queda correctamente en fp16 — la misma pregunta sobre la tarjeta que ya hace `_resolve_mixed_precision` — así que ahí la base congelada carga explícitamente `torch.float16`. Los tres modos de "LoRA apagado" de arriba (`lora.r: 0`, `unfrozen_parameters`, `lisa_enabled`) son el caso de base entrenable y en su lugar cargan explícitamente pesos maestros `torch.float32` — una elección deliberada de precisión para los parámetros que el optimizador realmente actualiza, no una conversión accidental, y no les afecta la comprobación de tarjeta de arriba.

---

## Qué tareas aplican `training.quantization`

`quantization` es `4bit` por defecto, pero no todos los entrenadores lo leen. Estos ocho cargan la base (y, en el caso de `distill`, al teacher) sin cuantizar diga lo que diga el campo; `prm` la carga como pesos maestros fp32 (ver [Modelo de recompensa de proceso](#modelo-de-recompensa-de-proceso-prm)):

| tarea | quantization | notas |
|---|---|---|
| `distill` | solo `none` | tanto el student como el teacher congelado cargan sin cuantizar |
| `classifier`, `reranker`, `cross_encoder` | solo `none` | fine-tuning completo salvo que `classifier_lora: true` |
| `prm` | solo `none` | siempre un fine-tuning completo; un bloque `lora` se rechaza (`lora.r: 0` sí se admite) |
| `moe_lora_routing` | solo `none` | base congelada, solo entrena el router |
| `unlearn` | solo `none` | la política y la copia de referencia cargan sin cuantizar |
| `asr` | solo `none` | fine-tuning completo salvo que `asr_lora: true` |

Para estas tareas un `quantization` sin definir se resuelve a `none`, así que la config guardada y el chequeo previo de VRAM describen la corrida que realmente ocurre (#795).

Un `4bit` u `8bit` explícito (o `load_in_8bit: true`) **carga con una advertencia y se resuelve a `none`**, porque toda config que Soup volcó mientras `4bit` era el valor por defecto lo lleva literalmente. La advertencia nombra la tarea y la versión que lo rechazará; define `quantization: none` para silenciarla. Un valor del Quant Menu (`gptq`, `awq`, ...) o un ajuste exclusivo de 4 bits como `bnb_4bit_quant_storage` se rechaza al cargar la config, nombrando la tarea. Cualquier otra tarea aplica el campo como se documenta en [Rendimiento y cuantización](performance-and-quantization.md).

---

## Ensayo de aprendizaje continuo (`--replay`)

El fine-tuning en una tarea nueva puede borrar la anterior. El ensayo es la defensa estándar: mezcla otra vez una porción de los datos viejos.

```bash
soup train --config new_task.yaml --replay old_task.jsonl --replay-ratio 0.1
```

o en `soup.yaml`:

```yaml
data:
  train: new_task.jsonl
  replay: old_task.jsonl
  replay_ratio: 0.1      # fracción del conjunto mezclado FINAL
  replay_seed: 0
```

**La proporción es una parte del conjunto final**, no de los datos nuevos: `n_replay = round(r/(1-r) · n_new)`. Con `0.1` sobre 1000 filas nuevas, eso son 111 filas de replay → 1111 en total → exactamente 10%. La consola informa lo que hizo:

```
Replay: +26 old rows interleaved (30.2% of 86)
```

Tres garantías que vale la pena conocer:

- **Intercaladas, nunca al final.** Un bloque final de filas viejas significaría que el modelo las ve todas en los últimos steps — un segundo mini-finetune, que es justo el fallo que el ensayo existe para prevenir.
- **Solo `train`; la validación se queda pura de la tarea nueva**, así que tu evaluación sigue midiendo la tarea que estás aprendiendo. Mide aparte la retención de la tarea vieja con `soup eval custom` / `soup ship`.
- **Un pool demasiado pequeño reporta un déficit en lugar de repetir filas** — una fila vista dos veces por epoch es un experimento distinto.

El archivo de replay tiene su propia detección de formato, así que el conjunto viejo puede ser alpaca mientras el nuevo es sharegpt.

**Alcance (v1):** solo `sft` y `pretrain`, e incompatible con `packing`/`multipack` — esos concatenan filas en bloques fijos, así que la proporción deja de tener sentido en los límites de bloque. Ambos se rechazan con un error claro en lugar de mezclar mal en silencio.

**Resultado honesto.** Validado a escala de prueba de mecanismo (SmolLM2-135M + LoRA): al entrenar la tarea B partiendo de un modelo que conocía la tarea A, el replay retuvo A **7% mejor que un control sin replay** (loss 0.565 → 0.526) con un costo de ~5% para la tarea B — el intercambio esperado. Pero el olvido sin replay era solo de **+4%**, es decir leve: LoRA en un modelo de 135M apenas se desvía. La dirección y el mecanismo están probados; el tamaño del efecto con fine-tuning completo o en 7B+ no está probado en una máquina de 4 GB.

## Blindaje del loop

Seis superficies protegen el loop de entrenamiento de los modos de fallo que cuestan una hora de GPU de verdad. El esquema y los kernels de matemáticas se lanzaron en la v0.70.0; el cableado en vivo de los callbacks del entrenador se lanzó en la **v0.71.11**, validado de extremo a extremo con SmolLM2-135M.

```bash
# Detector de reward hacking — auto-detención cuando la política empieza a engañar al RM
# (índice de separación de clústeres InfoRM, Wang et al. 2024 arXiv:2402.09345)
soup train --config soup.yaml \
    --reward-hack-detector info_rm --reward-hack-halt   # detente ante un veredicto HACK

# MITIGACIÓN de reward hacking en lazo cerrado (v0.71.26) — detectar Y autocorregir
soup train --config grpo.yaml --reward-hack-mitigation kl_control   # sube KL, se recupera
soup train --config grpo.yaml --reward-hack-mitigation log_only     # solo observar, sin acción

# Destilación por distancia de Wasserstein entre dos modelos que COMPARTEN un
# tokenizador, por ejemplo dos tamaños de la misma familia (#681: wasserstein / topk_align
# reenvían los token ids del student al teacher sin cambios, así que el par debe
# tokenizar de forma idéntica; para un tokenizador realmente distinto ver
# wasserstein_aligned más abajo).
# (Universal Logit Distillation, Boizard et al. 2024 arXiv:2402.12030)
#   training:
#     uld_strategy: wasserstein
soup train --config soup.yaml

# Destilación entre tokenizadores DISTINTOS, por ejemplo Llama -> Mistral,
# sin necesidad de vocabulario compartido (v0.71.18). Alinea las secuencias de tokens de student/teacher
# sobre tramos de caracteres decodificados, así que puedes destilar un student BPE de GPT-2 desde un
# teacher SentencePiece de Llama. Un token del student sin contraparte en el teacher
# (una pieza de byte-fallback, una pieza vacía, un token especial solo del student) no
# tiene objetivo y se queda fuera de la loss.
#   training:
#     uld_strategy: wasserstein_aligned

# Destilación reverse-KL de MiniLLM (Gu et al. 2024 arXiv:2306.08543).
# Solo config: no hay ninguna bandera --minillm-* más allá de --minillm-on-policy de abajo
# (#979). Mezcla offline: la proporción de mezcla es el peso del teacher en el objetivo
# reverse-KL y debe ser > 0 (la proporción 0 es KL(student || stopgrad(student)) y
# se rechaza). On-policy: la mezcla 0 es válida — muestreo solo del student, la loss sigue
# siendo KL(student || teacher).
#   training:
#     minillm_enabled: true
#     minillm_teacher_mix_ratio: 0.3
#     minillm_pretrain_anchor_weight: 0.1
#     minillm_pretrain_anchor_path: ./pretrain.jsonl
soup train --config soup.yaml

# Rollout on-policy REAL de MiniLLM (v0.71.18, Gu et al. §3.1) — muestrea un
# rollout autorregresivo nuevo de la mezcla teacher/student por token en cada step,
# luego reverse-KL normalizada por longitud. training.minillm_rollout_length ajusta el
# rollout (auto min(max_length, 32)). Aquí la mezcla 0 significa muestreo solo del student.
# --minillm-on-policy es la única bandera real aquí; minillm_enabled: true igual
# tiene que definirse en la config (#979).
soup train --config soup.yaml --minillm-on-policy

# Checkpoint a mitad de epoch para PPO/GRPO — TorchTune lo esquiva; Soup lo incluye
#   training:
#     rl_checkpoint_save_every_steps: 500
#     rl_checkpoint_keep_last: 3
#     rl_checkpoint_include_optimizer: true
soup train --config grpo.yaml

# Driver del loop iterativo de DPO — muestrear -> puntuar con el RM -> recomponer pares -> reentrenar
# (quita --plan-only para correr el loop; --plan-only solo renderiza el plan de cada ronda)
soup iterative-dpo \
    --base-model meta-llama/Llama-3.1-8B \
    --reward-model ./output_rm \
    --prompts ./prompts.jsonl \
    --output-dir ./iterative_dpo_out \
    --rounds 5 \
    --pairs-per-round 1000

# Detector de echo trap de RAGEN — auto-detención cuando las trayectorias colapsan a autorrepetición
# (Zhu et al. 2025 arXiv:2504.14437)
#   training:
#     echo_trap_enabled: true
#     echo_trap_threshold: 0.6
#     echo_trap_halt: true
#     echo_trap_tokenizer_aware: true
soup train --config grpo.yaml
```

`training.echo_trap_tokenizer_aware` (también disponible como la opción real `--echo-trap-tokenizer-aware`) cambia los n-gramas del echo trap de tokens separados por espacios a los ids enteros del tokenizador activo. Esto captura la repetición de subpalabras que el texto decodificado con mucha puntuación puede ocultar, pero la puntuación pasa a ser específica del tokenizador en lugar de agnóstica al vocabulario.

El detector `info_rm` se queda en silencio en cualquier step cuya división por la mediana deje una de las mitades constante — una recompensa de dos valores (por ejemplo, el 0/1 de `reward_fn: accuracy`) siempre lo hace, y también una recompensa continua en la que la mitad del batch cae en el mismo valor (por ejemplo, recompensas recortadas en 0). El índice de separación ahí mediría la tasa de éxito y no la salud del modelo de recompensa, así que el step no emite voto: no se registra ninguna línea base y nada se detiene ni muta β (se registra una advertencia por corrida). La única excepción: una vez que existe una línea base, un step en el que *todas* las recompensas son idénticas igual vota 0.0 — una corrida con 0/1 nunca registra una línea base y se queda en silencio, mientras que una recompensa continua realmente colapsada se lee como HACK igual que antes. Para recompensas de dos valores usa `reward_hack_detector: rm_ensemble` o las señales de longitud/repetición.

**Cambió:** antes de esta versión, una división con la mitad constante podía registrar una separación inflada (~3×10⁴ en un primer step al 50%) como línea base y clasificar como HACK steps posteriores más sanos — los logs del detector y los archivos `mitigation_log.jsonl` de corridas anteriores a esta versión no son comparables.

### Automitigación de reward hacking en lazo cerrado (v0.71.26)

Los detectores de arriba *detienen*; `training.reward_hack_mitigation` (o la bandera `--reward-hack-mitigation`) hace que el entrenador *se autocorrija* a mitad de corrida. Requiere `reward_hack_detector` en una corrida `grpo` de transformers, y tiene cuatro modos:

- **`log_only`** — solo observar. Agrega un `mitigation_log.jsonl` por step bajo el directorio de salida de la corrida (la caída de InfoRM/ensemble, el veredicto OK/WARN/HACK, la media/desviación de la recompensa, la tendencia de longitud de las completions, la repetición) y demuestra que nunca muta β. Corre esto primero para *ver* el hacking antes de dejar que un controlador actúe sobre él.
- **`kl_control`** — un controlador reversible **bang-bang + histéresis**. Cuando un voto multiseñal (`reward_hack_signals`: la caída del detector + `length_trend` + `repetition`) se mantiene por encima de `reward_hack_trip_band` durante `reward_hack_dwell_steps`, multiplica β por `reward_hack_kl_gain` (acotado a `[reward_hack_beta_floor > 0, reward_hack_beta_ceil]`, sin cruzar nunca 0); después de `reward_hack_release_patience` steps por debajo de la banda, relaja β de vuelta hacia el piso. El dwell y la paciencia de liberación evitan que oscile. β se escribe en **ambos**, `trainer.beta` y `trainer.args.beta`, para que surta efecto en el GRPO estándar *y* en las variantes de GRPO de Soup (y en `trainer.args.kl_coef` en PPO).
- **`pid_lagrangian`** — un controlador **PID-Lagrangian** (Stooke et al. 2020) que mantiene la señal de hacking en `reward_hack_signal_target` (Kp/Ki/Kd con anti-windup integral vía `reward_hack_integral_clamp`), más una **escalera de escalado**: subir β → tras `reward_hack_rollback_patience` steps con HACK persistente, retroceder al último checkpoint de RL bueno (requiere `rl_checkpoint_save_every_steps`) → tras `reward_hack_max_recovery_attempts` retrocesos, detener temprano con una explicación de rendición en lenguaje llano.
- **Blindaje antimanipulación** (cualquier modo de control): `reward_hack_signal_smoothing` (`ema`/`median` sobre `reward_hack_smoothing_window`), `reward_hack_conservative_on_disagreement` (cuando los detectores no concuerdan, mantener KL alto + proteger contra el colapso de una distribución de recompensa bimodal) y `reward_hack_reward_shaping` (restar una penalización acotada por `reward_hack_shaping_strength` sobre el proxy manipulado — `length`/`repetition`/`sentinel` — en la costura de la reward fn).

**Alcance:** solo prueba de mecanismo. Validado en SmolLM2-135M + una tarea sintética de length hacking en una sola RTX 3050 (los cuatro modos en vivo, incluido un retroceso real a mitad de corrida). La prueba en GPU es solo para GRPO. **Con `task: ppo` las banderas de reward hacking y echo trap se rechazan** (`reward_hack_detector`, `reward_hack_mitigation`, `echo_trap_enabled`): el entrenador PPO de trl que admite esta compilación no acepta funciones de recompensa, así que la señal de recompensa/completion que leen esos callbacks nunca se captura, y una corrida que las lleve se anunciaría y luego quedaría inerte. Si el loop suprime el hacking sin colapsar la recompensa real en 7B+ con un modelo de recompensa real es una pregunta abierta que la comunidad puede validar. `reward_hack_mitigation ∈ {kl_control, pid_lagrangian}` es mutuamente excluyente con `ref_model_ema_alpha` (ambos manejan la dinámica de KL/ref).

Cada detector se compone con el `soup why` de la v0.34 (explicador de anomalías), la recuperación de picos de la v0.32 y el `GRPOStabilityCallback` de la v0.53.11 #127, así que una sola corrida de GRPO puede tener InfoRM + echo trap + recuperación de picos + EMA del modelo ref en el lugar, todo activo a la vez y sin duplicar la recolección de trayectorias/estado. Los callbacks de reward hacking y echo trap leen las recompensas por step a través de un búfer de captura compartido y thread-safe (Soup envuelve tus funciones de recompensa para no tener que hacer monkeypatch a TRL); `rm_ensemble` necesita ≥2 funciones de recompensa para calcular una divergencia. La mezcla de teacher de MiniLLM es un análogo offline de mezcla de distribuciones del *muestreo* con mezcla de teacher on-policy del paper, y ULD compara las distribuciones después de recortar los ids del teacher al vocabulario del teacher (correcto para pares de la misma familia / vocabulario extendido; una tokenización realmente distinta necesita un paso de alineación de secuencias). La EMA del modelo de referencia (`--ref-model-ema-alpha`) se actualiza en el lugar — sin un round-trip completo de `state_dict` — así que es barata a escala 70B+.


## Desaprendizaje (`task='unlearn'`, NPO / SimNPO / RMU)

El derecho al olvido del RGPD + la respuesta a filtraciones de CSAM/PII, convertidos en producto. Tres backends de método:

- **NPO** — Negative Preference Optimization (loss solo negativa con forma de DPO; necesita un modelo de referencia).
- **SimNPO** — NPO normalizado por longitud sin modelo ref (más rápido y más estable en secuencias largas).
- **RMU** — Representation Misdirection Unlearning (ruido en el residual stream sobre las entradas de olvido).

```yaml
# unlearn.yaml
base: HuggingFaceTB/SmolLM2-135M
task: unlearn
data:
  train: traces.jsonl
  forget_set: gdpr_deletion_set.jsonl   # filas que se desaprenden (messages / prompt+completion / text)
  retain_set: capability_anchors.jsonl  # opcional — ancla la capacidad general
training:
  unlearn_method: npo           # o simnpo / rmu
  unlearn_alpha: 0.5            # peso del conjunto de retención [0.0, 10.0]
```

```bash
# Corre el loop de desaprendizaje (validado en SmolLM2-135M — NPO/SimNPO bajan la loss de olvido).
soup train --config unlearn.yaml --yes

# Puntúa la corrida en TOFU / MUSE / WMDP (veredicto OK / MINOR / MAJOR).
soup eval unlearning <run-id> --benchmark tofu --evidence evidence.json --output report.json
```

`task: unlearn` está activo (v0.71.9): carga una política envuelta en LoRA, una copia de referencia congelada (NPO / RMU) y los conjuntos JSONL de olvido / retención, y luego optimiza la loss de cada método — la de NPO, `(2/β)·mean(-logσ(-β·(π_logp − ref_logp)))`, lleva el log-prob del conjunto de olvido de la política por debajo del de la referencia (= olvido), mientras que el conjunto de retención ancla la capacidad. Corre NPO/SimNPO **con** un `retain_set` — sin uno, la política no tiene ancla de utilidad y Soup advierte con fuerza.

El desaprendizaje respeta `training.optimizer`, `scheduler`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `batch_size` y `gradient_accumulation_steps`. El `batch_size: auto` por defecto se resuelve a 1 para el desaprendizaje en lugar de a la estimación de memoria que usan los otros entrenadores, así que una config hace el mismo número de actualizaciones en cualquier tarjeta. Este loop limitado por memoria procesa un ejemplo a la vez y acumula gradientes para `batch_size * gradient_accumulation_steps` ejemplos por actualización del optimizador; un grupo parcial final se promedia por su tamaño real. `initial_loss` y `final_loss` siguen siendo losses por ejemplo sin escalar, y `total_steps` y el presupuesto de 2,000 steps siguen contando ejemplos en lugar de actualizaciones del optimizador.

`data.max_length` ahora controla la tokenización de olvido y retención en lugar del antiguo tope de 256 tokens. Su valor por defecto es 2,048, así que las configuraciones existentes pueden usar más memoria y tardar más. Define `data.max_length: 256` para conservar el antiguo límite de longitud de secuencia, o elige explícitamente un valor que quepa en el modelo y el dispositivo.

Tres ejes ortogonales: **Calidad del olvido** (delta de la loss de olvido antes/después), **Utilidad del modelo** (exactitud de retención preservada), **PrivLeak** (distancia de la AUC de inferencia de pertenencia respecto de 0.5). Los mini-fixtures incluidos para los tres benchmarks vienen en la caja (la v0.71.1 agregó MUSE + WMDP junto al conjunto TOFU existente), así que `--benchmark muse|wmdp` corre sin aportar evidencia. Las sondas del conjunto de olvido de WMDP vienen **redactadas** (prompts de relleno + respuestas `REFUSED`) — Soup nunca incluye contenido literal de conocimiento peligroso.


## Pre-entrenamiento continuado

Sigue entrenando un modelo con texto crudo para adaptación de dominio:

```yaml
base: meta-llama/Llama-3.1-8B
task: pretrain

data:
  train: ./data/corpus.jsonl   # {"text": "..."} o archivos .txt planos
  format: plaintext
  max_length: 4096

training:
  epochs: 1
  lr: 1e-5
  quantization: 4bit
```

```bash
soup init --template pretrain
soup train
```


## Destilación de conocimiento

Entrena un modelo student pequeño para que iguale la distribución de salida de un teacher más grande. Cada fila de train/val debe conservar al menos un objetivo de pérdida causal después del truncado en `data.max_length`, y una fila sin ninguno se rechaza en la configuración indicando la división y el número de fila, como hace SFT.

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: distill
modality: text
backend: transformers

data:
  train: ./data/chat.jsonl
  max_length: 2048
  chat_template: chatml

training:
  teacher_model: meta-llama/Llama-3.1-8B
  distill_divergence: forward_kl   # kl | forward_kl | reverse_kl | js
  distill_temperature: 2.0
  distill_chunk_size: 256          # token chunk size for divergence evaluation
  distill_checkpoint: true         # non-reentrant activation checkpointing
  epochs: 3
  lr: 5e-5
```

La loss = CE del student + (T**2) × KL(teacher_logits / T  ||  student_logits / T).
El teacher se carga una vez, se congela mediante `requires_grad_(False)` + `.eval()`, y sus
entradas/logits se conectan automáticamente entre dispositivos CPU/CUDA.

La destilación solo se ejecuta con `backend: transformers`; la carga de configuración rechaza
`backend: mlx` y `backend: unsloth`.
La acumulación de gradientes usa la cantidad de objetivos de entrenamiento desplazados y no enmascarados de toda la ventana del optimizador. Por lo tanto, dividir las mismas filas en microbatches de longitudes desiguales conserva la media de tokens del lote completo, en lugar de ponderar todos los microbatches por igual.

El kernel de divergencia de tokens calcula las tres divergencias en FP32 y devuelve deliberadamente
un escalar FP32, incluso con logits FP16/BF16. Por eso, los valores de forward-KL también pueden
diferir ligeramente del cálculo anterior de baja precisión. Las fórmulas en espacio logarítmico
para reverse-KL y Jensen-Shannon evitan losses finitas con gradientes no finitos; la conversión
a FP32 también conserva losses pequeñas que de otro modo se reducirían a cero por subdesbordamiento.
Los logits no finitos se propagan a la loss/los gradientes, lo que permite que el GradScaler de AMP
omita los steps con desbordamiento en vez de abortar el entrenamiento con una excepción de validación.

Los intermedios FP32 cuestan tiempo y memoria. La [medición del PR #736 del mantenedor](https://github.com/MakazhanAlpamys/Soup/pull/736)
en una RTX 5070 (BF16, B=1/S=512/V=32000, pasada hacia delante y hacia atrás) encontró que el kernel
enviado originalmente tardaba 1.71 veces más y usaba 1.24 veces la memoria máxima que main.
Esa medición incluía comprobaciones de valores finitos que se eliminaron aquí: sin ellas, tardaba
9.40 ms frente a 6.39 ms, con un pico idéntico de 376.5 MiB frente a 303.6 MiB. Estas mediciones
dependen del hardware y no son un benchmark de esta implementación revisada, que también evita
crear tensores de probabilidad que no se usan. Solo dos copias de logits FP32 ocupan alrededor
de 7.8 GiB con B=4/S=2048/V=128256; reserva memoria también para los demás intermedios.

### Fragmentación y checkpointing de activaciones

La destilación con vocabularios grandes (p. ej., 150k para Qwen 2.5) y secuencias largas
puede agotar la memoria de GPU debido a los enormes tensores intermedios de logits y probabilidades.
Soup ofrece dos controles para acotar la memoria de activaciones:

- `distill_chunk_size`: evalúa la divergencia en fragmentos de tokens activos de respuesta
  (`labels != -100`). El prefiltrado descarta de inmediato los tokens enmascarados del prompt y
  del padding de la memoria autograd retenida, mientras que la fragmentación limita los tensores
  de activación transitorios máximos durante la evaluación de la divergencia. Los tamaños de
  fragmento moderados (p. ej., de 64 a 256 tokens) equilibran la reducción de memoria máxima
  con el costo adicional de lanzar kernels.
- `distill_checkpoint`: envuelve la evaluación de fragmentos en checkpointing de activaciones
  no reentrante (`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`). Descarta los
  tensores intermedios de `log_softmax` y probabilidades durante la pasada hacia delante y los
  vuelve a calcular durante la pasada hacia atrás, lo que reduce sustancialmente los bytes de
  tensores autograd retenidos a costa de tiempo de recálculo durante la pasada hacia atrás. Ten
  en cuenta que activar `distill_checkpoint: true` sin definir `distill_chunk_size` procesa todos
  los tokens activos en un único fragmento; esto reduce los bytes retenidos mediante checkpointing,
  pero deja sin acotar el pico transitorio de activaciones.


Define `distill_mode: sequence` (el valor por defecto es `token`) para entrenar con las **continuaciones
generadas** por el teacher en vez de hacer coincidencia de logits por token: una destilación KD con
etiquetas duras, compatible con tokenizadores distintos, que funciona aunque student y teacher no
compartan vocabulario. El modo `sequence` es mutuamente excluyente con la ruta de logits
`uld_strategy` entre tokenizadores (son objetivos distintos para la misma tarea; el entrenador
rechaza la combinación durante la configuración). Se omiten las filas sin un turno de prompt,
el recuento y las primeras posiciones se muestran durante la configuración, se rechaza antes de
entrenar una división que quede sin filas utilizables, y el rechazo indica la posición de la fila
en tus datos. (v0.71.12)


## Clasificación de secuencias

Entrena una cabeza clasificadora sobre cualquier modelo base: admite clasificación
monoclase, multietiqueta y reranking con cross-encoder.

```yaml
base: BAAI/bge-base-en-v1.5
task: classifier              # or `reranker`, `cross_encoder`
modality: text
backend: transformers

data:
  train: ./data/labelled.jsonl   # rows: {"text": "...", "label": "spam"} or {"text": "...", "label": [0, 2]}
  max_length: 256

training:
  num_labels: 3
  classifier_kind: single_label   # or `multi_label`
  label_names: [ham, spam, promo] # required when labels are strings
  epochs: 5
  lr: 2e-5
  batch_size: 32
```

**Formatos de filas según la tarea:**
- `task: classifier`: secuencias de una sola entrada mediante `{"text": "sample", "label": 1}` o `{"text": "sample", "label": "spam"}` (también acepta filas ChatML con `label`). Los datasets de multietiqueta pasan listas de índices o nombres de etiquetas activas: `{"text": "sample", "label": [0, 2]}` o `{"text": "sample", "label": ["ham", "promo"]}`.
- `task: reranker`: texto de consulta y documento mediante `{"text": "query doc", "label": "relevant"}` (o un índice entero de clase).
- `task: cross_encoder`: entradas emparejadas mediante `{"text_a": "query", "text_b": "doc", "label": 1}` o `{"question": "query", "answer": "doc", "label": 1}`.

Enruta `classifier` / `reranker` / `cross_encoder` a través de
`AutoModelForSequenceClassification`. Las cabezas multietiqueta admiten hasta 1024 entradas por
fila, eliminan duplicados mediante conversión a set y rechazan bytes nulos en cadenas de etiquetas.

Agrega una sección `lora:` para entrenar un clasificador con **encoder congelado + adaptador LoRA**
en lugar del modelo completo: se entrenan el adaptador pequeño y la cabeza clasificadora
(inicializada desde cero), mientras la red troncal del encoder permanece congelada:

```yaml
training:
  num_labels: 3
  lora:
    r: 16
    alpha: 32
```

(v0.71.12)


## Control de esfuerzo de razonamiento + EOT

Control del esfuerzo de razonamiento al estilo gpt-oss para el ajuste por instrucciones.

```yaml
training:
  reasoning_effort: high      # low | medium | high
  train_on_eot: true          # do NOT mask the EOT/EOS token in the loss
```

`reasoning_effort` inserta `<|reasoning_effort|>high<|/reasoning_effort|>` en
el turno del sistema (y crea uno si no existe). `train_on_eot=True` hace que el modelo
aprenda cuándo dejar de generar, entrenándolo con el token EOS final en vez de
ocultarlo de la loss. Ambos parámetros se limitan a las tareas de la familia SFT.


## Variantes de loss EBFT / GDPO

**`training.gdpo_variant` se rechaza al cargar la configuración**
([#1309](https://github.com/MakazhanAlpamys/Soup/issues/1309)). Esto rompe las configuraciones
que definen `standard`, `length_normalized` o `margin`: antes se cargaban, pero entrenaban
silenciosamente con la loss por defecto, porque las versiones compatibles de TRL (0.29 y
posteriores) no exponen el hook `DPOTrainer.dpo_loss`. Se rechaza todo valor no nulo; si no se
define o es null, la configuración se carga. Ninguna receta, plantilla ni ejemplo distribuido
establece este campo.

Elimina `gdpo_variant` para entrenar `task: dpo` normal con `standard`. Para
`length_normalized`, `task: simpo` es el objetivo más parecido, **pero no es un reemplazo
idéntico**. `margin` no tiene equivalente. DPO normal, sin configuración de GDPO:

```yaml
task: dpo
training:
  dpo_beta: 0.1
```

EBFT (`ebft_variant: structured | strided`) se rechaza al cargar la configuración
([#1230](https://github.com/MakazhanAlpamys/Soup/issues/1230)): todavía no es un objetivo
distinto. El término que agregaba puntuaba los logits de cada posición contra el token de
entrada de esa misma posición, sin desplazamiento causal; por tanto, premiaba copiar la entrada
en vez de predecir el token siguiente. Si se desplaza al token siguiente, equivale a la propia
entropía cruzada del modelo y la loss contaría dos veces la entropía cruzada. Una configuración
que lo usara nunca entrenaba correctamente. Elimina `ebft_variant` y `ebft_temperature`; el
rechazo seguirá vigente hasta que el objetivo EBFT previsto se implemente a partir de su referencia.


## Variantes del objetivo GRPO

Soup incluye kernels matemáticos activos para 6 variantes del objetivo GRPO además del valor
por defecto. Define `grpo_variant` en `training` y el entrenador crea automáticamente una
subclase de `trl.GRPOTrainer` para dirigir `compute_loss` al kernel correspondiente:

```yaml
task: grpo
training:
  reward_fn: accuracy
  num_generations: 4
  grpo_variant: gspo         # sequence-level length-normalized ratio
  # or: dapo / dr_grpo / bnpo / rft / two_sided
  # grpo_delta: 0.2          # required when grpo_variant=two_sided; optional for gspo
```

Variantes:

- **standard**: referencia de base al estilo de DeepSeek-R1 (delegada a `compute_loss` de TRL).
- **gspo**: Group Sequence Policy Optimization (ratio normalizado por longitud a nivel de secuencia, con clipping).
- **dapo**: clipping asimétrico desacoplado (`eps_lo=0.2, eps_hi=0.28`).
- **dr_grpo**: suma de tokens sin normalización por longitud de muestra.
- **bnpo**: surrogate PPO normalizado por longitud.
- **two_sided**: clipping simétrico con `grpo_delta` proporcionado por el operador.
- **rft**: ajuste fino por muestreo de rechazo (solo contribuyen los tokens con ventaja positiva).

Todas las variantes aplican `grpo_beta` (por defecto, `0.1`) como penalización KL respecto de la
política de referencia, en el punto donde la aplica la loss GRPO propia de trl: el estimador por
token de trl `exp(ref - logp) - (ref - logp) - 1`, ponderado por `grpo_beta` y reducido del mismo
modo que el término de política de esa variante. `rft` la aplica solo a las completions aceptadas
con las que entrena. La β que establece durante la ejecución el controlador de reward hacking
(`kl_control` / `pid_lagrangian`) llega a todas las variantes de la misma manera.
Define `grpo_beta: 0` para una corrida sin KL (sin penalización KL respecto de la política de
referencia y sin ejecutar la pasada hacia delante de referencia), como en las recetas publicadas
DAPO y Dr. GRPO ([#1247](https://github.com/MakazhanAlpamys/Soup/issues/1247)).

Con un `grpo_beta` distinto de cero, una corrida de variantes registra la misma métrica `kl` que
la loss estándar de trl: la media del lote de ese estimador por token sobre los tokens de completion
(solo los tokens aceptados en `rft`), como `kl` durante el entrenamiento y `eval_kl` durante la evaluación.

El callback de estabilidad (actualización del modelo ref EMA, búfer de replay, contador de alertas TIS)
se adjunta automáticamente cuando se define cualquiera de `ref_model_ema_alpha` / `replay_buffer_size`
/ `tis_threshold` / etc.


## Modelo de recompensa de proceso (PRM)

Entrena una cabeza de recompensa escalar sobre cadenas de razonamiento con supervisión por paso. El
formato de datos es la estructura `prm` de v0.42.0: una fila por `{prompt, completions: [step1,
step2, ...], labels: [r1, r2, ...]}`:

```yaml
task: prm
data:
  format: prm
  train: ./prm_train.jsonl
  max_length: 2048
training:
  epochs: 1
  lr: 1.0e-5
```

El entrenador carga `AutoModelForCausalLM`, adjunta una cabeza de recompensa `nn.Linear(hidden, 1)`
y calcula el MSE entre los escalares predichos en los tokens de límite de paso y las etiquetas
correspondientes a cada paso. La cabeza de recompensa se guarda dentro del checkpoint del modelo
(`reward_head.*` en `model.safetensors`) y el tokenizador se guarda junto a ella, por lo que el
directorio resultante se puede cargar de forma independiente.

PRM entrena todos los parámetros base junto con la cabeza, así que todos se cargan como pesos
maestros fp32 en cada dispositivo. En CUDA, autocast ejecuta la pasada hacia delante en bf16 (fp16
en tarjetas anteriores a Ampere); en MPS usa bf16 cuando el runtime lo admite; CPU entrena en fp32.
Con el optimizador por defecto (AdamW), reserva alrededor de 16 bytes por parámetro antes de las
activaciones (pesos fp32, gradientes fp32 y dos momentos AdamW fp32), que es lo que predice la
comprobación previa de VRAM para `task: prm` cuando `batch_size` es un entero (con el valor por
defecto `batch_size: auto`, no se ejecuta la comprobación previa). El checkpoint guardado es fp32,
y sin DeepSpeed cargar la base fp32 requiere alrededor del doble de RAM del host que una carga bf16.
Con DeepSpeed, cada rank carga la base en fp32 en el host antes de que exista el motor (4 bytes por
parámetro de RAM del host por rank); luego el motor la convierte a su tipo bf16/fp16, mantiene su
propia copia maestra fp32 y guarda un checkpoint de 16 bits. Una base bf16 sin pesos maestros
redondearía la mayoría de las actualizaciones con estas tasas de aprendizaje (#1235).


## GRPO guiado por PRM (RL con supervisión de procesos)

Usa un PRM entrenado como **recompensa por paso** dentro de GRPO: la señal de supervisión de procesos
de la era o1. Define `training.prm_reward` con un directorio PRM (un checkpoint `task=prm`) o un id
de Hub, `org/name` o `org/name@revision`; el id de Hub se descarga (pesos, tokenizador y archivos de
código personalizado) cuando se crea la recompensa, antes del entrenamiento, y allí se rechaza un
repositorio que no sea un PRM entrenado con Soup (sin cabeza de recompensa), indicando el valor
configurado (#1466). El PRM divide cada completion generada en pasos de razonamiento (heurística de
saltos de línea), puntúa cada paso con su cabeza de recompensa y combina las puntuaciones por paso
en una recompensa escalar que GRPO optimiza. **Reemplaza** `reward_fn` y usa el punto de integración
existente de ajuste de reward + mitigación de reward hacking, por lo que el controlador de v0.71.26
sigue observándola (TRL la registra como `rewards/prm_reward`).

```yaml
task: grpo
backend: transformers        # required (the PRM reward runs a transformers forward)
modality: text               # required
data:
  format: chatml
  train: ./grpo_prompts.jsonl
training:
  prm_reward: ./my-prm       # a `soup train task=prm` checkpoint dir (or HF id)
  prm_aggregate: min         # weakest-link (default) | prod | last
  num_generations: 4
  grpo_beta: 0.04
```

`prm_aggregate='min'` (eslabón más débil, la agregación PRM estándar) es el valor por defecto
seguro; `prod` presupone puntuaciones de paso calibradas en `[0,1]` (la cabeza PRM de Soup se
entrena con MSE sin restricciones, por lo que `prod` puede dispararse con etiquetas no calibradas).

**Entornos de rollout incluidos.** Tres entornos de juguete deterministas en Python puro
preparan la ruta de rollout openenv desde el inicio; combina cualquiera de ellos con
`rollout_backend=openenv`:

```yaml
training:
  rollout_backend: openenv
  rollout_func: soup_cli.envs.calculator:rollout   # or retrieval_qa / guess_number
  reward_fn: verifiable
  verifiable_domain: math
```

Recetas listas para usar: `grpo-env-calculator`, `grpo-env-retrieval-qa`,
`grpo-env-guess-number`. Los entornos son *seeders* deterministas de un solo intercambio
prompt/respuesta (el contrato openenv en vivo solo pasa los prompts semilla, no el modelo); no son
episodios interactivos de varios turnos.

**Alcance:** solo prueba de mecanismo; validado en SmolLM2-135M con un PRM sintético pequeño (la
recompensa PRM puntúa completions buenas por encima de las malas e impulsa las ventajas de GRPO).
No es una afirmación sobre modelos de recompensa de producción; se buscan colaboradores para validar
la escala (#286).


## DPO en línea (`task='online_dpo'`): juez en el loop (v0.71.31)

A diferencia del DPO offline (filas estáticas `prompt/chosen/rejected`), **DPO en línea** genera
dos completions por prompt **on-policy** en cada step y le pregunta a un *juez* —o a un *modelo de
recompensa*— cuál es mejor; la ganadora pasa a ser `chosen` y la otra, `rejected`. El juez cierra
el loop. Envuelve `OnlineDPOTrainer` de TRL; los datos contienen solo prompts (como GRPO). Solo
Transformers + texto.

```yaml
base: HuggingFaceTB/SmolLM2-135M-Instruct
task: online_dpo
data:
  train: ./data/prompts.jsonl      # prompt-only (or any format — prompts are extracted)
training:
  online_dpo_judge: "ollama://llama3.1"   # a pairwise judge (ollama://|https://|http://localhost)
  # OR: reward_model: ./my-reward-model   # exactly one of judge / reward_model
  online_dpo_loss_type: sigmoid           # sigmoid | ipo
  online_dpo_max_new_tokens: 64
  dpo_beta: 0.1
  lora: { r: 8, alpha: 16, target_modules: auto }
```

El juez es el `JudgeEvaluator` compatible con OpenAI de Soup, adaptado a `BasePairwiseJudge` de
TRL (con corrección de sesgo por intercambio: solo registra una ganadora cuando coinciden los
órdenes A,B y B,A). Receta: `online-dpo-smollm2-135m`. La prueba de mecanismo se validó en
SmolLM2-135M con un juez sintético (no es una afirmación de RLHF de producción; #286).
Una URL de juez `https://` usa `OPENAI_API_KEY` solo si su host es `api.openai.com`; otros hosts
se invocan como servidores compatibles con OpenAI sin esa clave. Al cargar soup.yaml, se rechaza
un `online_dpo_judge` cuyo host sea una IP literal privada, de enlace local o reservada (se permite
loopback); indica el juez interno mediante su nombre de host. También se rechaza un host escrito
solo con números que no sea una dirección IPv4 válida (`10.0.0.256`, `4294967296`).

Se excluyen de la loss los pares que el juez no pueda ordenar: un empate, una llamada fallida o
ilegible al juez, o un veredicto que cambia al invertir el orden de las dos completions. Ese par no
aporta gradiente y la loss es la media de los pares ordenados de cada lote. La proporción de pares
sin ordenar se registra como `judge/invalid_rate`, con una WARNING la primera vez que ocurre y en
cada step en que no se ordena ningún par. Las métricas `loss` y `train_loss` registradas promedian
solo los lotes
y que ordenaron un par; si durante una ventana de registro no se ordenó ninguno, no se registra `loss`.
Un step en que no se ordenó ninguno no se omite: el optimizador igualmente avanza, por lo que el
momentum y el decaimiento de pesos de AdamW siguen modificando los pesos y el scheduler de tasa de
aprendizaje avanza. Esa deriva tiene un límite: si el juez no ordena ninguno de 32 pares seguidos
(por ejemplo, porque su servidor está caído), el entrenamiento se detiene con un error que indica
el juez; una corrida más corta en la que no haya ordenado ningún par falla del mismo modo en vez de
guardar un adaptador. Antes de esto, TRL entrenaba cada uno de esos pares como si hubiera ganado la
segunda completion (#1225).


## Loss de preferencias multiobjetivo ponderada

Mezcla términos DPO / SimPO / ORPO / IPO en una sola corrida de entrenamiento mediante
`preference_loss_weights` (deben sumar 1.0):

```yaml
task: preference
training:
  preference_loss_weights:
    simpo: 0.6
    orpo: 0.4
```

El wrapper de combinación calcula una suma ponderada mediante los kernels del árbol
`compute_dpo_term` / `compute_simpo_term` / `compute_orpo_term` /
`compute_ipo_term`. La loss con el mayor peso es la *principal*:
se crea su entrenador de trl y, en trl 0.29, se extraen de la pasada hacia delante de ese mismo
entrenador los log-probs por secuencia que necesitan los kernels. Por tanto,
qué mezclas se entrenan depende del entrenador que se cree:

| Mezcla | Resultado en trl 0.29 |
| --- | --- |
| `simpo` + `orpo` | Entrena. Los entrenadores CPO (SimPO) y ORPO devuelven sus log-probs por secuencia (`concatenated_forward`). |
| `dpo` o `ipo` tiene el mayor peso, junto con cualquiera de `dpo`, `ipo`, `simpo`, `orpo` | Entrena. Ambos crean el `DPOTrainer` de trl: los log-probs de la política se obtienen de los logits de su propia pasada hacia delante, y los log-probs de referencia de `compute_ref_log_probs`. |
| `simpo` u `orpo` tiene el mayor peso y la mezcla incluye `dpo` o `ipo` | Se detiene en el primer step e indica los términos. `dpo` e `ipo` comparan la política con una referencia congelada, y los entrenadores CPO / ORPO no la ejecutan. Asigna el mayor peso a `dpo` o `ipo`. |
| `bco` mezclado con cualquier otro término | Se rechaza al cargar la configuración (formato de datos incompatible). |

**Qué es un término.** En una mezcla, `simpo` y `orpo` son solo sus *términos de preferencia* —el objetivo de margen / razón de momios—, sin el término de verosimilitud (NLL) que incluyen las corridas independientes con `preference_loss: simpo` / `preference_loss: orpo`. Por lo tanto, una mezcla `{simpo: 1.0}` no da el mismo valor que una corrida `preference_loss: simpo`, y `training.cpo_alpha` no la afecta. Usa `training.preference_loss` si quieres la loss completa independiente. Cada término sí respeta su propio campo de configuración: `training.simpo_gamma` para SimPO y `training.orpo_beta` para ORPO, tenga o no el mayor peso esa loss.

Los términos `dpo` e `ipo` son las losses que entrenan `preference_loss: dpo` y
`preference_loss: ipo`: la loss sigmoide de DPO con `training.dpo_beta` y la loss cuadrática de
IPO con `training.ipo_tau`, donde la razón logarítmica de cada lado se divide por la longitud de su
respuesta, como hace trl. Leen esos dos campos sin importar cuál de ellos sea el principal, así que
`{dpo: 0.5, ipo: 0.5}` y `{ipo: 0.5, dpo: 0.5}` representan la misma loss. `dpo_beta_schedule`
no está disponible con una mezcla.

**Costo de usar `dpo` / `ipo` como término principal.** La política se ejecuta una vez, como en
una corrida de una sola loss. La referencia se ejecuta dos veces por step: trl calcula sus propios
log-probs de referencia en línea y conserva solo sus medias, así que la mezcla vuelve a solicitarlos
al entrenador (`compute_ref_log_probs`, sin gradientes). Con logits de 16 bits (autocast bf16 o
fp16), la mezcla también conserva para la pasada hacia atrás un tensor adicional del tamaño de los
logits, además del que conserva la propia loss de trl: 2 bytes × filas × tokens × vocabulario,
donde un lote de 4 pares tiene 8 filas. Son alrededor de 250 MiB para 4 pares de 512 tokens y un
vocabulario de 32,000 tokens. Con logits float32 no conserva ninguno.

**Evaluación.** Con `val_split` definido y un término principal `simpo` / `orpo`,
`prediction_step` de trl llama directamente a `get_batch_loss_metrics`, así que la loss de evaluación
es la loss propia de trl para el término principal, no el valor de la mezcla. Por lo tanto, el
entrenamiento y la evaluación reportan números distintos para el mismo lote. Con un término
principal `dpo` / `ipo`, `prediction_step` de trl pasa por `compute_loss`, así que la loss de
evaluación sí es el valor de la mezcla.


## Compatibilidad con modelos MoE

Ajusta modelos Mixture of Experts (Mixtral, Qwen3-30B-A3B, DeepSeek V3) con ScatterMoE LoRA:
aplica LoRA tanto a las capas de atención como a las capas FFN de expertos:

```yaml
base: Qwen/Qwen3-30B-A3B
task: sft

training:
  moe_lora: true              # target expert + attention layers
  moe_aux_loss_coeff: 0.01    # router load-balancing loss
  quantization: 4bit
```

Soup detecta automáticamente las arquitecturas MoE. No todas las tareas leen `moe_lora`: la tabla
por tarea en [Rendimiento y cuantización](performance-and-quantization.md#moe-expert-quantization--router-only-training-live-in-v07120)
indica cuáles lo aplican y cuáles lo rechazan al cargar la configuración.

```bash
soup init --template moe
soup train
```


## Ajuste fino de visión / multimodal

Ajusta modelos de visión y lenguaje (LLaMA-3.2-Vision, Qwen2-VL, Pixtral) con datos de imagen + texto:

```bash
# Instalar compatibilidad con visión
pip install "soup-cli[vision]"

# Crear una configuración de visión
soup init --template vision

# Entrenar
soup train --config soup.yaml
```

```yaml
base: meta-llama/Llama-3.2-11B-Vision-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

**Formatos de datos de visión admitidos:**

**LLaVA:**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this image."}, {"from": "gpt", "value": "A cat on a mat."}]}
```

**ShareGPT4V:**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nWhat does this show?"}, {"from": "gpt", "value": "Quarterly revenue."}]}
```

Una ruta relativa `image` se resuelve respecto de `data.image_dir` o, si no está definido, respecto
del directorio del archivo de datos. Un dataset de Hub o un URI remoto no tiene directorio de
archivo de datos, así que define `data.image_dir` cuando sus filas contengan rutas de imágenes;
consulta [Data Pipeline Pro](data.md#data-pipeline-pro).
También se admiten imágenes PIL decodificadas en memoria (como las de una columna `Image()` de
Hugging Face Hub) y estructuras de imagen Parquet (`{"bytes": ..., "path": ...}`) escritas por
`datasets`; el campo `path` de una estructura sigue las mismas reglas de resolución de directorios
y contención de rutas que una cadena de ruta simple.

`soup data inspect` resuelve las rutas de imágenes del mismo modo o usa `--image-dir` cuando se
proporciona, y muestra automáticamente estadísticas de imágenes (cantidad, formatos, archivos
faltantes) para datasets de visión.


## Ajuste fino de audio / voz

Ajusta modelos de audio y lenguaje (Qwen2-Audio, Whisper) con datos de audio + texto:

```bash
# Instalar compatibilidad con audio
pip install "soup-cli[audio]"

# Crear una configuración de audio
soup init --template audio

# Entrenar
soup train --config soup.yaml
```

```yaml
base: Qwen/Qwen2-Audio-7B-Instruct
task: sft
modality: audio

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

**Formato de datos de audio:**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe this audio."}, {"role": "assistant", "content": "Hello world."}]}
```

### Ajuste fino ASR (`task='asr'`, Whisper): v0.71.32

Ajusta Whisper a tu acento o dominio. whisper-tiny (39M) y base (74M)
se entrenan con una GPU de 4 GB. Las filas son `{"audio": <path>, "text": <transcript>}` bajo
`data.format='asr'`; el cargador reforzado decodifica el audio a mono de 16 kHz.

```yaml
base: openai/whisper-tiny
task: asr
data:
  train: ./data/train.jsonl
  format: asr                 # rows: {"audio": "clip.wav", "text": "hello world"}
  audio_dir: ./data/audio     # audio paths resolve here (containment-checked)
training:
  epochs: 3
  lr: 1e-4
  batch_size: 2
  asr_language: en            # optional; sets + persists the decoder prefix
  asr_task: transcribe        # transcribe | translate
  asr_lora: true              # optional LoRA on q/v; default = full fine-tune
  quantization: none
output: ./out
```

```json
{"audio": "clip0.wav", "text": "hello world"}
```

Transcribe y puntúa después del entrenamiento:

```bash
soup infer --task asr --model ./out --input eval.jsonl --output preds.jsonl --audio-dir ./data/audio
# -> preds carry {"transcription", "wer", "cer"} per row + a corpus WER summary
```

Notas: `task='asr'` requiere `backend='transformers'` y una base Whisper (una base que no sea
Whisper se rechaza antes de descargar). `asr_language`/`asr_task` se guardan en un archivo lateral
`asr_generation.json` para que `soup infer --task asr` los restaure (sobrescríbelos con
`--asr-language`/`--asr-task`). WER/CER usan un normalizador ligero: son útiles para comparar
cambios antes/después, pero no para valores absolutos comparables con tablas de clasificación.
`whisper-large-v3-asr` solo incluye análisis de configuración (requiere una GPU más grande).


## GRPO Plus: variantes del objetivo, RL de contexto largo, agentes multi-turno

Soup incluye siete variantes del objetivo GRPO, vLLM en espera entre rollouts, cuatro backends de
rollout de agentes, siete controles de estabilidad/eficiencia y modelos de recompensa de proceso.

```yaml
# soup.yaml — DAPO con búfer de replay y enmascaramiento de truncado TIS
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo
data:
  train: ./prompts.jsonl
  format: chatml
training:
  reward_fn: accuracy
  num_generations: 8
  # New: GRPO objective variants
  grpo_variant: dapo                  # one of: gspo / dapo / dr_grpo / bnpo / two_sided / rft / standard
  # grpo_delta: 0.2                   # required when grpo_variant: two_sided (optional for gspo)
  grpo_fp16: true                     # FP16 RL (unsloth parity)
  # Long-context + memory-efficient RL
  # long_context_grpo: true           # staged for future Tiled MLP; refused as of v0.77 — #808
  vllm_sleep_mode: true               # between-rollouts vLLM standby — LIVE (vLLM >= 0.7)
  # Multi-turn agent rollout — openenv is LIVE: your function's rows replace the prompt dataset
  rollout_backend: openenv            # one of: art / ruler / nemo_gym / openenv
  rollout_func: my_module:my_rollout  # module:function resolver (openenv; trusted operator code)
  # Stability / efficiency knobs
  ref_model_ema_alpha: 0.99           # EMA sync policy → reference
  replay_buffer_size: 2048
  async_grpo_prefetch: true           # overlap rollout + train
  tis_threshold: 2.0                  # truncated importance sampling
  mask_truncated_completions: true    # paired with tis_threshold
  defer_rerolling: true
  skip_zero_advantage: true
  off_policy_mask_threshold: 0.5
```

Modelos de recompensa de proceso (con supervisión por paso):

```yaml
# soup.yaml
base: meta-llama/Llama-3.1-8B
task: prm                              # New: Process Reward Model
data:
  train: ./prm_dataset.jsonl
  format: prm                          # stepwise-supervised data shape
training:
  epochs: 3
  lr: 1e-5
```

RL de visión en Qwen2-VL / Pixtral / InternVL (en preparación):

```yaml
# soup.yaml
base: Qwen/Qwen2-VL-7B-Instruct
task: grpo
modality: vision
data:
  train: ./vlm_prompts.jsonl
  format: llava
training:
  reward_fn: accuracy
  # vision_grpo: true                  # staged for VLM-RL; refused as of v0.77 — #808
```

Todas las banderas se distribuyeron como validaciones de esquema en v0.50.0. `vllm_sleep_mode`,
el rollout `openenv` y el entrenamiento PRM (`task: prm`) están activos. Otros backends de rollout
(`art`, `ruler`, `nemo_gym`) generan el error "not yet validated", mientras que las banderas en
preparación sin consumo (p. ej., `long_context_grpo`, `vision_grpo`) muestran advertencias en v0.76
y se rechazan desde v0.77 (#808).


## Entrenamiento DPO

Entrena con datos de preferencias usando Direct Preference Optimization:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: dpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  dpo_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

Una base publicada como wrapper de visión y lenguaje (`model_type: minimax_m3_vl` de MiniMax-M3)
no está registrada en `AutoModelForCausalLM`, así que una configuración DPO simple no puede cargarla.
Define `modality: vision`; DPO la carga igual que la ruta de visión de SFT, mediante
`AutoModelForImageTextToText` con el processor del modelo, y entrena la torre de lenguaje; los
objetivos LoRA se resuelven en esa torre como antes. Las filas de preferencias no llevan imágenes,
y `soup recipes verify` construye la misma clase que carga el entrenador. `backend: unsloth` no
lee `modality` para DPO y se rechaza al cargar la configuración (#1393).


## Variedad de preferencias: BCO + despachador unificado + variantes KL

Cinco losses de preferencias están disponibles detrás de una sola opción de configuración. Elige
una loss sin renombrar la tarea, programa el descenso de β durante el entrenamiento y actualiza
periódicamente la referencia congelada.

### BCO (Binary Classifier Optimization)

El mismo formato de entrada que DPO; las filas se dividen internamente al esquema no emparejado de
BCO de TRL (`{prompt, completion, label}`).

```yaml
task: bco
data:
  train: ./data/preferences.jsonl
  format: dpo
training:
  bco_beta: 0.1
```
### Despachador unificado de preferencias

Usa `task: preference` + `training.preference_loss` para cambiar de loss sin modificar
`task`. Así, hacer barridos de hiperparámetros sobre el tipo de loss resulta sencillo.

```yaml
task: preference
data:
  train: ./data/preferences.jsonl
  format: dpo
training:
  preference_loss: dpo   # or simpo, orpo, ipo, bco
```

Los valores heredados `task: dpo` / `task: simpo` / etc. siguen siendo opciones de primera clase;
la superficie unificada es aditiva.

### Variantes DPO controladas por KL

Programa el descenso de β durante el entrenamiento con `dpo_beta_schedule`:

```yaml
task: dpo   # or task: preference + preference_loss: dpo, or task: ipo
training:
  dpo_beta: 0.1
  dpo_beta_schedule: linear   # linear | cosine | exponential
  dpo_beta_end: 0.01
```

Restringido a tareas de la familia DPO (`dpo`, `ipo` o `preference` con `preference_loss in {dpo, ipo}`); solo backend transformers.

Sincroniza periódicamente el modelo de referencia congelado con el student activo mediante
`dpo_ref_regen_epochs`:

```yaml
training:
  epochs: 4
  dpo_beta: 0.1
  dpo_ref_regen_epochs: 2  # copies .default. adapter weights into .ref. every 2 epochs
```

Con LoRA, TRL crea en el modelo un adaptador `.ref.` congelado. En cada N.º de epoch (indexado
 desde 1 y omitiendo el epoch 0), los parámetros del adaptador `.default.` activo se copian al
adaptador `.ref.` congelado bajo `torch.no_grad()`. Requiere LoRA (`lora.r >= 1`); el ajuste fino
completo (`lora.r: 0`) se rechaza al cargar la configuración (#1345).

### Loss de preferencias multiobjetivo

```yaml
task: preference
training:
  preference_loss_weights: {simpo: 0.7, orpo: 0.3}
```

El esquema valida de 2 a 5 entradas que sumen 1 y rechaza `bco` mezclado con una loss emparejada
al cargar la configuración. En trl 0.29 se entrena una mezcla `simpo` + `orpo`, y también una
mezcla cuyo peso mayor sea `dpo` o `ipo`. Un término `dpo` o `ipo` bajo un término principal
`simpo` / `orpo` se detiene en el primer step e indica los términos que no pudo calcular, porque
esos entrenadores no ejecutan una referencia congelada (consulta [Loss de preferencias
multiobjetivo ponderada](#loss-de-preferencias-multiobjetivo-ponderada)). Define
`training.preference_loss` para usar una sola loss.


## Entrenamiento GRPO (razonamiento)

Entrena modelos de razonamiento con Group Relative Policy Optimization (al estilo DeepSeek-R1):

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: sharegpt
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  grpo_beta: 0.1  # KL penalty; 0 = KL-free (DAPO / Dr. GRPO)
  num_generations: 4
  reward_fn: accuracy   # or 'format', or path to custom .py
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

```bash
# Crear una configuración de razonamiento
soup init --template reasoning

# Entrenar
soup train --config soup.yaml
```

**Vigilante de gradientes (#342).** Si aparecen gradientes no finitos (NaN o Inf) durante una
corrida GRPO, se omite el step del optimizador: los pesos y el estado del optimizador no cambian,
y el step sigue contando para el total de steps y el scheduler de tasa de aprendizaje. El recuento
de steps omitidos se registra al final de la corrida (como advertencia en consola si la proporción
supera el 5 %) y se guarda en `trainer_state.json` para que `soup adapters audit` pueda detectarlo.

**Funciones de recompensa integradas:**
- `accuracy`: 1.0 cuando la respuesta final de la completion coincide con la respuesta correcta; de lo contrario, 0.0 (sin crédito parcial)
- `format`: comprueba que haya bloques de razonamiento estructurado `<think>...</think>`

`accuracy` y el dominio verificable `math` leen la completion y la respuesta correcta con el mismo
parser. La respuesta final se determina con este orden de prioridad:

1. el resto de la línea después del último `####` (si `#### Final Answer` es un encabezado, se usa la línea siguiente);
2. el contenido del último `\boxed{}` (se permite un espacio antes de la llave y llaves anidadas como `\boxed{\frac{1}{2}}`);
3. lo que sigue al último `The answer is` o `Answer:` (también `**Answer**:`; la respuesta puede estar en la línea siguiente), hasta el final de su cláusula: un `. `, `, ` o `; ` fuera de corchetes, o las conjunciones `because` o `since` (sin distinguir mayúsculas, y con espacios a ambos lados; no `as`). Una coma o punto y coma seguido de un número continúa una lista (`41, 42 or 43`), y un inciso entre paréntesis forma parte de la cláusula. Así, `The answer is Washington, D.C.` se interpreta como `Washington`, `The answer is 42 (six times seven).` como `42`, y `The answer is (3, 4).` como `(3, 4)`.

Una caja tiene prioridad sobre una frase, y un `\boxed{}` o una frase posterior a una línea `####`
tiene prioridad sobre esta (la línea era un encabezado markdown o una respuesta que luego se corrigió).
Esto incluye texto irrelevante: `#### Paris` seguido de `I hope this answer is helpful!` se interpreta
como `helpful`; por eso, termina una completion en su línea `####`. Si una completion no contiene
ninguno de esos elementos, se usa su última línea y su último número, por lo que `Six times seven
is 42.` e incluso `Not 42.` se interpretan como 42; solo cuenta el número final, así que enumerar
candidatos no da crédito.

El número de una frase de respuesta se lee en su propia cláusula: `The answer is 41 apples, not 42.`
se interpreta como 41, porque la cláusula termina en la coma. Una cláusula que indique **más de un
valor distinto** es una respuesta ambigua y no da una respuesta: `The answer is either 41 or 42.`,
`Answer: 41 or 42`, `The answer is 42 (or 43).` y `the answer is 41, 42 or 43` obtienen 0.0 frente
a cualquier respuesta correcta, y se rechaza una respuesta correcta escrita así. Un `because` /
`since` dentro de la frase o después de puntuación termina la cláusula (`The answer is 42 because
6*7=42.` y `The answer is 42, because 6*7=42.` se interpretan como 42), mientras que las
justificaciones entre paréntesis forman parte de la cláusula y la vuelven ambigua (`The answer is
42 (6*7=42).`). Repetir el mismo valor no es ambigüedad (`42 (i.e. 42.0)`), y los dígitos de una
respuesta entre corchetes o en LaTeX (`(3, 4)`, `\begin{pmatrix} 3 \\ 4 \end{pmatrix}`, `\frac{14}{3}`, `2^{10}`) o de una hora o razón (`3:45`, `1:1,000`) no son valores separados: esa respuesta se compara como texto. Las respuestas `\boxed{}` y `####` se comparan completas, así que una lista allí cuenta como una única respuesta y solo coincide con una respuesta correcta que tenga la misma lista.

Una respuesta correcta numérica se compara por valor, así que `#### 1,000`, `\boxed{1000}` y
`The answer is $1000.` coinciden con una respuesta correcta `1000`. Cualquier otra respuesta
correcta (`\frac{14}{3}`, `p - q`, `Paris`) se compara como texto, sin distinguir mayúsculas ni
espacios en blanco, `$`, `\(...\)` y `\[...\]`, `\left` / `\right`, ni `\dfrac` / `\tfrac` frente
a `\frac`; así, `\boxed{\dfrac{14}{3}}` coincide con `\frac{14}{3}`. Ambos lados también eliminan
la puntuación final `. , ; : !`, separadores de millares de LaTeX como `1{,}000`, los comandos de
espaciado de LaTeX `\,` `\!` `\;` `\:` y `\ `, y el signo menos Unicode. Un salto de fila `\\` se
conserva íntegro, así que una matriz coincide sin importar el espaciado entre filas. Se desenvuelve
un wrapper `\text{}` / `\textbf{}` / `\mathrm{}` / `\mbox{}` y se conserva su contenido; se eliminan
`^\circ` / `^{\circ}` / `°`; se ponen entre llaves los argumentos compactos de `\frac` para que
coincidan tanto si son un carácter sin llaves como un grupo ya delimitado, con o sin espacio antes
(`\frac12`, `\frac1{2}`, `\frac{1}2`, `\frac 34` y `\frac9{19}` se leen como `\frac{N}{D}`); y un
prefijo de variable de una letra se interpreta como su lado derecho (`x = 7` se interpreta como
`7`, en cualquiera de los dos lados). Sin embargo, si ambos lados nombran una variable y los nombres
difieren (`x = 3` frente a `y = 3`), el par obtiene 0.0, porque la variable de una directriz o una
asíntota forma parte de la respuesta. Las unidades no se eliminan (`42 apples` frente a `42`) y no
se evalúa nada (`\frac{1}{2}` no equivale a `0.5`).

Para GRPO, Soup conserva las columnas del dataset de origen y TRL se las pasa a las funciones de
recompensa como argumentos con nombre. También se expone como `answer` un `output` de Alpaca o el
turno final del assistant en ShareGPT/ChatML; una columna explícita `answer` tiene prioridad. Las
funciones integradas que dependen de una respuesta correcta validan sus entradas antes de generar:

| Recompensa | Metadatos de origen requeridos |
|---|---|
| `accuracy` o `math` verificable | `answer`, o una respuesta de referencia del assistant, que incluya una respuesta final |
| `code` verificable | `expected` o `answer` |
| `json_schema` verificable | `schema` |

Una respuesta correcta debe indicar la respuesta final con `####`, `\boxed{}`, `The answer is` o
`Answer:`, o ser la respuesta simple en una línea (`42`, `Paris`, `\frac{14}{3}`). Se rechaza antes
de generar una fila cuya respuesta correcta no indique ninguno de estos elementos (por ejemplo,
una solución de referencia de varias líneas sin respuesta marcada, o una bajo el encabezado
`#### Solution`: una línea `####` que no sea un número y tenga más texto puede ser un encabezado
markdown, así que la respuesta correcta no puede depender de ella), o cuya frase de respuesta sea
ambigua (`The answer is either 41 or 42.`). El rechazo indica la división, número de fila, campo y
cantidad de otras filas con el mismo problema, porque una respuesta así daría 0.0 a toda completion
y dejaría a GRPO sin señal. Un dataset que mezcle respuestas correctas numéricas y LaTeX, como
MATH-500, también se carga con `math`; la validación indica cuántas no son numéricas y, por tanto,
se compararán como texto normalizado.

**Funciones de recompensa personalizadas**: indica un archivo Python:
```python
# my_reward.py
def reward_fn(completions, **kwargs):
    """Score each completion. Return list of floats."""
    return [1.0 if "correct" in c[-1]["content"] else 0.0 for c in completions]
```
```yaml
training:
  reward_fn: ./my_reward.py
```

Las recompensas personalizadas pueden leer cualquier columna de origen preservada mediante
`kwargs`. Deben devolver exactamente una puntuación numérica finita por completion; Soup comprueba
este contrato e informa el nombre de la recompensa y el recuento antes de que TRL intente crear un
tensor de recompensa.

**Ensembles de recompensas**: enumera varias recompensas separadas por comas y se combinan (solo
GRPO). Esto también habilita el detector de reward hacking `rm_ensemble`, que requiere ≥ 2 recompensas:
```yaml
training:
  reward_fn: "accuracy,format"   # both are scored every step
```

### Sintetiza un verificador a partir de tus datos (`soup reward synth`)

No escribas un verificador a mano: genéralo a partir de respuestas de referencia (gold). Soup
infiere un verificador *determinista* (numérico / JSON Schema / regex / llamadas a herramientas),
genera un archivo `.py` legible y editable y **se niega a generarlo** si no puede distinguir tus
referencias de respuestas incorrectas generadas automáticamente (el informe de calibración es
obligatorio). El archivo generado se usa normalmente con `reward_fn: reward.py`.

```bash
# infer + calibrate + emit (exit 0 kept, 2 refused, 1 error)
soup reward synth references.jsonl -o reward.py --output-report calib.json

# preview the induced spec without writing anything
soup reward synth references.jsonl --plan-only

# force a family instead of auto-detecting
soup reward synth answers.jsonl -o reward.py --kind numeric --tolerance 1e-6
```

Las referencias son un JSONL donde la respuesta correcta de cada fila está en un campo `answer`
(cámbialo con `--field`) o en el último turno del assistant de una lista `messages`.
`--min-discrimination` define cuán claramente debe separar el verificador las referencias de los
negativos perturbados antes de generarlo. La versión 1 solo admite familias deterministas: un
marcador `\boxed{}`/`####` ayuda al verificador numérico, y se pide a las completions que marquen
su respuesta (práctica estándar de RLVR). El verificador numérico lee tanto las respuestas correctas
como las completions con el parser que usan las recompensas integradas `math` / `accuracy`
(`soup_cli.utils.final_answer`), así que las respuestas correctas `#### 1,000`, `\boxed {42}` y
`6*7=42\n#### 42` se puntúan igual en ambos; `1,000` se reconoce como numérico. El informe de
calibración enumera cada respuesta correcta que el verificador generado rechace
(`rejected_references` en `--output-report`).

**Cambio:** un `reward.py` sintetizado antes de esta versión extraía el último número con una regex
local; los que se sintetizan ahora usan el parser de respuestas compartido en ambos lados, por lo
que respuestas correctas del tipo `1,000` y completions `The answer is …` se puntúan cuando antes
obtenían 0; regenera los verificadores antiguos antes de comparar corridas. La regeneración también
deja de premiar algunas completions que la regex anterior premiaba: ahora obtienen 0 las respuestas
marcadas `####` o `\boxed{}` que llevan unidades, `%`, un símbolo de moneda que no sea dólar o
código (`#### 42 apples`, `\boxed{42\%}`, `#### €42`, `#### 42 USD`), una frase de respuesta
ambigua o justificada (`The answer is either 41 or 42`, `The answer is 42 because 6*7=42`) y una
completion cuyo último `\boxed{}` sea incorrecto aunque el primero sea correcto; ahora coinciden
con las funciones integradas. `42%` sin marcar o `The answer is 42%.` sigue dando crédito, como
con las funciones integradas; solo las respuestas delimitadas por marcadores rechazan sufijos. El
archivo generado importa `soup_cli`, por lo que funciona donde esté instalado Soup y no es
completamente autónomo.

### Prueba de resistencia de un verificador frente a la explotación (`soup reward stress`)

Aunque un verificador supere la calibración, podría dar crédito por contenido basura.
`soup reward stress` proporciona al verificador completions degeneradas deterministas —vacías,
rellenas para aumentar longitud, repetidas y con spam de sentinelas— puntuadas frente a tus
respuestas correctas reales, y señala las que **acepte**. Es la contraparte adversarial de `synth`:
la calibración comprueba que el verificador distingue las referencias de respuestas incorrectas
*amigables*; `stress` evalúa si un modelo con reward hacking podría explotarlo.

```bash
# probe a synthesized verifier (or any reward .py) — exit 0 robust, 2 gameable, 1 error
soup reward stress reward.py --references golds.jsonl --output-report stress.json

# probe a builtin verifier instead of a .py file
soup reward stress verifiable --verifiable-domain math --references golds.jsonl

# JSON-schema references may be stored as objects in a `schema` field
soup reward stress verifiable --verifiable-domain json_schema \
    --references schemas.jsonl --field schema

# tune the attack set / accept threshold / gameability tolerance
soup reward stress reward.py --references golds.jsonl \
    --attacks empty,length,repetition,sentinel,wrapped_junk,answer_spray \
    --sentinel GOLD --threshold 0.5 --max-gameable 0.0
```

El informe muestra la tasa de aceptación por ataque y un veredicto general. Si un verificador que
requiere respuestas correctas se prueba **sin** `--references`, se produce un error definitivo
(no se puede medir), nunca un falso resultado de «robusto». Como cada familia de ataques evalúa
varias variantes distintas frente a referencias de muestra (hasta 23 invocaciones por lotes al
verificador, o 4,600 completions puntuadas con el límite de 200 respuestas correctas), los
verificadores lentos o basados en modelos tardarán proporcionalmente más que las comprobaciones
simples de cadenas. Probar un `.py` ejecuta el código de su módulo, como cualquier recompensa
personalizada: prueba solo archivos de confianza.
Para el dominio integrado `json_schema`, las referencias se pasan como metadatos `schema=`; los
objetos JSON seleccionados mediante `--field` se decodifican antes de que el verificador los puntúe.

Limitación actual: las familias de ataques integradas producen texto plano que no es JSON válido,
así que `json_schema` las rechaza durante el análisis antes de evaluar las restricciones específicas
del esquema. Por eso, el resultado aún no distingue entre un esquema estricto y uno permisivo. El
valor `reference_accept` tampoco es un control significativo de autoaceptación para este dominio,
porque puntúa el documento de esquema como si fuera una instancia del propio esquema (y normalmente
es `0%`).

### Recompensas verificables (RLVR)

Usa `reward_fn: verifiable` con un `verifiable_domain` para obtener recompensas deterministas y
comprobables matemáticamente, sin modelo juez ni heurísticas. Es ideal para GRPO en tareas de
matemáticas, código o salida estructurada.

```yaml
training:
  reward_fn: verifiable
  verifiable_domain: math          # or: code, json_schema
  num_generations: 4
```

Tres dominios integrados:

| Dominio | Qué comprueba |
|---|---|
| `math` | Lee la respuesta final de la completion y de la respuesta correcta con el parser compartido descrito en «Funciones de recompensa integradas» (`####`, `\boxed{}`, `The answer is` / `Answer:`, o, si no, el último número de la completion). Una respuesta correcta numérica obtiene 1.0 si está dentro de 1e-4 y 0.6 si está dentro de 1e-2 (comparación como decimales exactos); cualquier otra obtiene 1.0 si coincide con el texto normalizado. No se evalúa nada: no se usa `eval()` sobre la salida del usuario |
| `code` | Ejecuta el Python generado con un timeout de 5 s, RLIMIT de 512 MB en POSIX, `python -I -S`, parche de socket y directorio de trabajo efímero. La salida se limita a 10 KB. Panel de advertencia en el primer uso |
| `json_schema` | Valida la salida con un JSON Schema proporcionado por ejemplo en el dataset |

> **Nota:** el dominio `code` ejecuta generaciones no confiables. Soup aplica aislamiento agresivo, pero no confíes en él para aislamiento de nivel de producción; usa una VM o un contenedor con datos públicos.


## Ajuste fino para llamadas a herramientas

Entrena modelos para emitir llamadas a funciones estructuradas (`tool_calls` al estilo de OpenAI,
con argumentos JSON).

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/tool_calls.jsonl
  format: tool-calling

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
```

**Formato de datos de llamadas a herramientas:**
```json
{"messages": [
  {"role": "user", "content": "What's the weather in Paris?"},
  {"role": "assistant", "tool_calls": [
    {"id": "c1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
  ]}
]}
```

Agrega una lista de nivel superior `tools` (esquemas de funciones OpenAI) para presentar los
esquemas al modelo como turno del sistema; con `format: auto`, se detecta como llamada a herramientas
una fila que tenga `messages` y `tools`. Las trayectorias de varios turnos conservan su orden: cada
turno del assistant mantiene sus `tool_calls` y cada turno `tool`, su `tool_call_id`. La estructura
anterior, con las llamadas en una lista `tool_calls` de nivel superior, sigue cargándose: esas
llamadas se convierten en un turno del assistant después de `messages`.

Los argumentos se analizan solo como JSON: nunca se usa `eval()`. `soup eval custom` puede evaluar
la precisión de las llamadas a herramientas (nombre de función + igualdad del JSON de argumentos).

```bash
soup init --template tool-calling
```


## PPO / pipeline RLHF completo

Entrena modelos con el pipeline RLHF completo: calentamiento SFT → modelo de recompensa → alineación PPO.

```bash
# Crear una configuración RLHF
soup init --template rlhf
```

**Paso 1: calentamiento SFT**: ajusta un modelo base con tus datos:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: alpaca
output: ./output_sft
```

**Paso 2: entrena el modelo de recompensa**: aprende preferencias a partir de comentarios humanos:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: reward_model
data:
  train: ./data/preferences.jsonl
  format: dpo
output: ./output_rm
```

**Paso 3: alineación PPO**: optimiza la política con el modelo de recompensa:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: ppo
data:
  train: ./data/prompts.jsonl
  format: chatml
training:
  reward_model: ./output_rm
  epochs: 1
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_penalty: 0.05
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
output: ./output_ppo
```

`epochs` controla las pasadas completas por el dataset de entrenamiento. `ppo_epochs`
controla las pasadas de optimización dentro de cada actualización PPO. Soup pasa ambos valores,
además de `ppo_kl_penalty`, a los nombres correspondientes de la `PPOConfig` activa de TRL y
muestra el programa efectivo durante la configuración.

Un lote de rollout PPO contiene `batch_size` x `gradient_accumulation_steps` prompts en cada
proceso, y TRL descarta los lotes parciales; por tanto, el conjunto de entrenamiento debe tener al
menos `batch_size` x `gradient_accumulation_steps` x cantidad de filas igual a la cantidad de
procesos. Con menos filas nunca se llegaría a un step, así que `soup train` lo rechaza antes de
cargar modelos e indica el número de filas y ambos valores de configuración.

PPO admite dos fuentes de recompensa:
- **Modelo de recompensa** (`reward_model`): modelo de recompensa preentrenado (del paso 2)
- **Función de recompensa** (`reward_fn`): función invocable (igual que en GRPO: `accuracy`, `format` o `.py` personalizado)


## Entrenamiento KTO (preferencias no emparejadas)

Entrena con datos de preferencias no emparejadas: no necesitas pares chosen+rejected:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: kto

data:
  train: ./data/kto_train.jsonl
  format: kto

training:
  epochs: 3
  kto_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

**Tamaño del lote.** KTO necesita un `batch_size` por dispositivo de al menos 2: el término KL
de TRL es degenerado con batch 1, por lo que `batch_size: 1` se rechaza al cargar la configuración.
`batch_size: auto` nunca se resuelve por debajo de 2, y `soup local-rl train --train-method kto`
establece `batch_size: 2`. Aumenta `gradient_accumulation_steps` para tener un lote efectivo mayor.

**Formato de datos KTO:**
```json
{"prompt": "What is 2+2?", "completion": "4", "label": true}
{"prompt": "What is 2+2?", "completion": "Fish", "label": false}
```


## Entrenamiento ORPO (sin modelo de referencia)

ORPO combina SFT y alineación en un solo paso; no necesita un modelo de referencia:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: orpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  orpo_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```


## Entrenamiento SimPO (preferencia simple)

SimPO usa probabilidades logarítmicas normalizadas por longitud como recompensas implícitas; no
requiere referencia:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: simpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  simpo_gamma: 0.5
  cpo_alpha: 1.0
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

**Se rechazan las filas cuya completion trl trunca por completo.** El entrenador CPO de trl
trunca cada respuesta a `data.max_length` menos la longitud de la respuesta **más larga**, por lo
que si un `chosen` largo va junto a un `rejected` corto, el lado corto queda sin tokens entrenables.
La probabilidad logarítmica normalizada por longitud de SimPO resulta entonces 0/0, todos los
parámetros del adaptador se entrenan a NaN y el filtro nan-inf de transformers informa una loss
`0.0`. `soup train` se detiene antes del primer step e indica cuántas filas y cuáles tienen el
problema, junto con el `data.max_length` implicado. Aumenta `data.max_length`, equilibra el par o
elimina la fila.


## Entrenamiento IPO (preferencias regularizadas)

IPO es una variante de DPO con fundamentos teóricos y una regularización más fuerte:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: ipo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  ipo_tau: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```


## RAFT: ajuste fino con aumento de recuperación

Cuando necesitas que un modelo *cite* el documento que está leyendo en vez de alucinar, RAFT
(Stanford 2024) es la receta de referencia. Cada fila de entrenamiento incluye una consulta, un
documento de referencia, una lista de documentos distractores y la respuesta; el modelo aprende a
prestar atención al documento pertinente e ignorar el ruido.

```yaml
# soup.yaml
data:
  train: ./data/raft.jsonl
  format: raft

training:
  citation_faithful: true        # enable citation precision/recall scoring
  citation_style: bracket        # cite as [doc-1] inline
```

`training.citation_recall_threshold` se valida, pero nada condiciona el guardado o la ejecución en
función de ese valor. Definirlo genera una advertencia en v0.76 y se rechaza desde v0.77 (#761).

```jsonl
# RAFT JSONL row shape
{"query": "When was Python released?", "golden_doc": "Python was released in 1991 by Guido van Rossum.", "distractor_docs": ["Ruby was released in 1995.", "Java was released in 1995."], "answer": "1991 [doc-1]"}
```

```bash
# Receta lista para usar con Llama 8B
soup recipes show raft-llama3-8b
soup recipes use raft-llama3-8b
```

La puntuación de citas está disponible como kernel puro para la compuerta de evaluación:

```python
from soup_cli.utils.citation_faithful import score_citations

score = score_citations(
    predicted="The answer is 1991 [doc-1].",
    expected_ids=("doc-1",),
)
# CitationScore(precision=1.0, recall=1.0, f1=1.0, predicted_count=1, expected_count=1)
```

El ajuste fino fiel a las citas se limita a `task in {sft, pretrain}` + `data.format='raft'`;
las corridas mal configuradas fallan al cargar la configuración e indican el campo en cuestión.

Por dentro, una corrida `format: raft` entrena **solo la respuesta**: cada fila se convierte en un
prompt (documento de referencia + distractores, barajados de forma determinista según
`data.raft_shuffle_seed` y etiquetados como `[doc-N]`), seguido de la respuesta; el tramo del
prompt se excluye de la loss y, con `citation_faithful: true`, los tramos `[doc-id]` entre corchetes
de la respuesta reciben un peso de loss por token mayor. Las filas cuyo prompt ocupe todo
`max_length` (respuesta completamente truncada) se descartan con una advertencia, en lugar de
reducir el dataset sin avisar.

Por defecto, el orden de los documentos permanece fijo durante toda la corrida. Define
`data.raft_epoch_shuffle: true` para **volver a permutar** los documentos de referencia y
distractores *en cada epoch* (se incorpora una sal por epoch a la semilla de barajado), para que el
modelo no se acostumbre a un puesto fijo de cita; es útil en corridas con varios epochs.
`epoch=0` reproduce exactamente el orden de la permutación heredada, así que activar esta opción
nunca modifica el primer epoch. (v0.71.17)

Puntúa las citas de un modelo entrenado desde la CLI:

```bash
# filas {predicted, expected_ids}, O filas RAFT evaluadas contra su propio [doc-N] de referencia
soup eval citation preds.jsonl --style bracket
# filas RAFT: pasa la semilla de barajado usada en el entrenamiento para alinear el id del documento de referencia
soup eval citation raft.jsonl --shuffle-seed 0 --output citation.json
```

`soup diagnose` también incorpora un modo de fallo `citation` que señala si un modelo dejó de
citar el documento de respaldo.


## RA-DIT: ajuste dual por instrucciones con aumento de recuperación

RA-DIT (Meta 2023) es la versión en dos etapas de RAFT: primero entrena un recuperador
sentence-transformer (contrastivo) y después ajusta el generador con filas al estilo RAFT. Se
incluyen dos recetas emparejadas:

```bash
# Etapa 1: entrena el recuperador (usa el entrenador de embeddings v0.16 de Soup)
soup recipes use ra-dit-retriever
soup train

# Etapa 2: entrena el generador con datos RAFT y señala el recuperador
soup recipes use ra-dit-llama3-8b
soup train
```

El esquema exige que coincidan las etapas y tareas: `ra_dit_stage: retriever` requiere `task: embedding`;
`ra_dit_stage: generator` requiere `task: sft`. Una receta mal configurada falla al cargar la
configuración e indica el campo correspondiente.

Ejecuta ambas etapas con un solo comando, `soup ra-dit`:

```bash
soup ra-dit --retriever-config retriever.yaml --generator-config generator.yaml
# vista previa del plan + enlace resuelto al recuperador, sin entrenar:
soup ra-dit -r retriever.yaml -g generator.yaml --plan-only
```

Entrena el recuperador y luego **registra** ese recuperador entrenado como pareja del generador
(escribe su directorio de salida en `training.ra_dit_retriever_model` de la configuración del
generador) y entrena el generador al estilo RAFT. El recuperador registrado es el que se usa al
implementar o servir el modelo: la etapa 2 no fusiona los pesos del recuperador. Un `soup train`
normal con una configuración de etapa de generador sin modelo de recuperador **enlaza
automáticamente** la corrida más reciente de recuperador RA-DIT del Registry; pasa
`--retriever-model <m>` para anularlo.


## Entrenamiento con currículo (BETA)

Añade reponderación dinámica por capas sobre el agrupador estático `curriculum`. Cada N steps, el
entrenador agrega la loss y la norma de gradiente por muestra en una señal de incertidumbre por
grupo, la procesa con softmax (temperatura configurable) y un piso (water-filling para que ningún
grupo baje de `curriculum_dynamic_floor`), y repondera el sampler. Los grupos vacíos usan la mediana
de los grupos poblados; ante entradas degeneradas se usa una distribución uniforme.

```yaml
training:
  curriculum: true                          # static bucketer (v0.23.0)
  curriculum_buckets: 4
  curriculum_metric: perplexity             # length (default) | loss | perplexity
  curriculum_dynamic: true                  # NEW — dynamic re-weighting
  curriculum_dynamic_recompute_steps: 50    # refresh every 50 global steps
  curriculum_dynamic_floor: 0.05            # min weight per bucket
  curriculum_dynamic_temperature: 1.0       # softmax temp on uncertainty
```

**Agrupamiento por percentil de dificultad (v0.71.5).** Cuando `curriculum_metric` es `loss` o
`perplexity`, el callback dinámico asigna cada muestra de un step a un grupo según su *rango* en una
ventana móvil de 512 steps de la señal de dificultad (perplejidad = `exp(min(loss, 50))`), en vez
de usar la alternativa round-robin que se usa con `length`. Esto mantiene los grupos calibrados
según la distribución de loss activa, en lugar de ordenarlos estáticamente por longitud. `length`
(el valor por defecto) conserva la asignación round-robin.

Visualiza la evolución registrada del peso de los grupos con `soup runs curriculum-curve <run_id>`.

Seguridad para DDP / acumulación de gradientes: los lanzamientos multi-rank deben conectar un hook
`all_reduce` en las estadísticas por grupo (un validador cruzado rechaza de antemano las corridas
multi-rank sin coordinación). La ampliación a varios entrenadores aparte de `sft` / `pretrain` está
registrada para v0.48.1.


## Ajuste fino TTS (`task='tts'`, BETA, activo en v0.71.20)

Activo desde v0.71.20 (se quitó del stub de esquema v0.52.0). Las familias de cadenas de códec
(`orpheus`, `llasa`, `spark`, `oute`) entrenan con **entropía cruzada del siguiente token sobre
secuencias de chat intercaladas `[text][audio-codec-token]`**, por lo que `TTSTrainerWrapper`
reutiliza la maquinaria de modelo/tokenizador/LoRA/CE de SFT y añade plantillas de emoción por
familia y registro de tokens de códec. `sesame_csm` es diferente: el entrenamiento actual de CSM usa
texto y 32 codebooks Mimi como tramas multimodales paralelas. Por eso, Soup rechaza CSM en esta
ruta de cadenas de códec con SFT de texto en vez de entrenar silenciosamente con el objetivo
incorrecto; todavía se necesita un entrenador CSM dedicado.

Hay dos flujos de trabajo:

**Chat pre codificado (activo para las familias de cadenas de códec).** Ejecuta **offline** el
códec de audio de la familia para que el turno del assistant ya contenga la cadena discreta de
tokens de códec; luego entrena con `data.format: chatml`. Es entropía cruzada normal y funciona en
cualquier GPU (validado de extremo a extremo en SmolLM2-135M-Instruct). Este flujo no convierte
los codebooks Mimi paralelos de Sesame CSM en un ejemplo de entrenamiento CSM válido.

```yaml
base: HuggingFaceTB/SmolLM2-135M-Instruct   # or canopylabs/orpheus-3b-0.1-ft
task: tts
modality: audio_out
data:
  train: ./data/tts_pre_encoded.jsonl   # assistant turns carry codec tokens
  format: chatml
  new_special_tokens: ["<|codec_0|>", "<|codec_1|>"]   # your codec vocab
training:
  tts_family: orpheus
  tts_emotion: neutral   # Orpheus + Oute only
  lora:
    r: 16
    alpha: 32
```

Los `data.new_special_tokens` proporcionados por el operador se registran (sin duplicados, solo
los tokens que aún no estén en el vocabulario) y se cambia el tamaño de la matriz de embeddings a
través del modelo (posiblemente envuelto por PEFT), de modo que haya filas para los ids de tokens
de códec. Esto solo sucede con `backend: transformers`: la configuración de unsloth no tiene un
paso que agregue tokens, así que `backend: unsloth` junto con `new_special_tokens` se rechaza al
cargar (#1358). Orpheus + Oute admiten condicionamiento por emoción mediante `training.tts_emotion`
con una lista permitida por familia (Orpheus: neutral / happy / sad / angry / excited / calm /
whisper / laugh; Oute: neutral / happy / sad / angry / calm / excited); el wrapper antepone la
cadena de control de emoción de la familia al primer turno del usuario.

**Códec en vivo (condicionado por hardware/dependencias).** Definir `data.format: audio` pide al
entrenador que codifique audio sin procesar **durante el entrenamiento**. Orpheus y Llasa están
activos en la ruta de cadenas de códec: Orpheus usa `pip install snac` a 24 kHz; Llasa usa el extra
`[audio]` de Soup (torchaudio + soundfile). Torchaudio debe coincidir con la versión instalada de
Torch (los metadatos recientes de torchaudio quizá no hagan que pip imponga esta compatibilidad);
luego Soup remuestrea a 16 kHz, llama al códec nativo de Transformers `HKUSTAudio/xcodec2-hf` y
muestra los ids resultantes como `<|s_ID|>` entre los tokens delimitadores de generación de voz de
Llasa. El audio mantiene límites de duración/bytes y se lee mediante un descriptor `O_NOFOLLOW`.
La codificación en vivo de audio sin procesar de Spark y Oute ahora falla de forma segura:
Spark-TTS no tiene un paquete `sparktts` instalable y su entorno oficial fija Torch/Transformers por
debajo de la pila admitida por Soup; `outetts` actual fija Transformers 4.52.3 y la preparación de
Oute también requiere alineación de transcripción/palabras. Para esas dos familias, codifica antes
en el entorno upstream y entrena el chat resultante con tokens de códec mediante
`data.format: chatml`. Sesame CSM falla antes con un mensaje específico de arquitectura porque sus
32 codebooks Mimi paralelos requieren un entrenador multimodal nativo, no un adaptador de cadena
de códec.

Se incluyen tres recetas listas para cadenas de códec: `orpheus-tts-sft`, `llasa-tts`,
`oute-tts`; cópialas con `soup recipes use <name>`. Los validadores cruzados rechazan el backend
`mlx`, `modality != audio_out` y etiquetas de emoción que no estén en la lista permitida por familia.


## Entrenamiento de clasificadores / rerankers / cross-encoders (BETA, v0.52.0)

Tres nuevos tipos de tarea se basan en el entrenador de embeddings existente: `task: classifier`
(clasificación de secuencias monoclase o multietiqueta), `task: reranker` (puntuación puntual de
recuperación), `task: cross_encoder` (puntuación de entradas emparejadas). Solo esquema; wrapper
del entrenador activo en v0.52.1.

```yaml
base: BAAI/bge-base-en-v1.5
task: classifier
data:
  train: ./data/classification.jsonl
training:
  num_labels: 3
  classifier_kind: single_label
  label_names: [negative, neutral, positive]
```

`num_labels` está limitado a `[1, 1024]` y rechaza explícitamente los booleanos antes de comprobar
si el valor es entero; `label_names` (opcional) debe ser único, con ≤128 caracteres por etiqueta,
y tener la misma longitud que `num_labels` cuando se define.


## Destilación de conocimiento (BETA, v0.52.0)

Nueva `task: distill` con `training.teacher_model` (id de HF o ruta local),
`training.distill_divergence` (`kl` / `forward_kl` / `reverse_kl` / `js`; `kl` se normaliza a
`forward_kl`) y `training.distill_temperature` (limitada a `[0.05, 100.0]`, solo valores finitos).
Solo esquema; loop activo en v0.52.1.

```yaml
base: meta-llama/Llama-3.2-1B
task: distill
data:
  train: ./data/distill.jsonl
training:
  teacher_model: meta-llama/Llama-3.1-8B
  distill_divergence: forward_kl
  distill_temperature: 2.0
```

El validador cruzado rechaza `task='distill'` sin `teacher_model`, y rechaza los campos
`teacher_model` / `distill_*` si `task` no es `distill`. También rechaza `task='distill'` con
`backend: mlx` o `backend: unsloth` al cargar la configuración; la destilación solo se ejecuta
con `backend: transformers`.


## EBFT + GDPO (BETA, v0.52.0)

Ambos campos se rechazan al cargar la configuración. `training.gdpo_variant` no hacía nada
silenciosamente con versiones compatibles de TRL porque el hook de loss DPO ya no existe
([#1309](https://github.com/MakazhanAlpamys/Soup/issues/1309)); elimínalo y usa DPO normal para
`standard` o SimPO como el objetivo más cercano (no idéntico) para `length_normalized`. `margin`
no tiene equivalente. Energy-Based Fine-Tuning (`training.ebft_variant` +
`training.ebft_temperature`) sigue rechazándose ([#1230](https://github.com/MakazhanAlpamys/Soup/issues/1230)):
su término no tenía desplazamiento causal, por lo que premiaba copiar la entrada y, si se desplazaba,
duplicaría la entropía cruzada. Consulta [Variantes de loss EBFT / GDPO](#variantes-de-loss-ebft--gdpo).


## `reasoning_effort` + `train_on_eot` de gpt-oss (v0.52.0)

`training.reasoning_effort: low | medium | high` inserta un token de prefijo del sistema durante
el entrenamiento para modelos gpt-oss; `training.train_on_eot: true` incluye tokens de control
EOT/EOS explícitos en la loss SFT (axolotl `train_on_eot`). Ambos se limitan al conjunto de tareas
de la familia SFT (`sft` / `pretrain` / `distill` / `classifier` / `reranker` / `cross_encoder`):
definirlos en DPO / GRPO / PPO / etc. produce un error al cargar la configuración. Conexión activa
del formateador en v0.52.1.

## MoLE: enrutamiento de adaptadores por token (`task='moe_lora_routing'`)

Entrena una pequeña **red de gating** que enruta cada token a una combinación ponderada de N LoRA
de tareas congelados (Mixture of LoRA Experts, Wu et al. 2024). El modelo base y todos los
adaptadores de tarea permanecen congelados: solo aprende el router a elegir qué adaptador o
adaptadores debe usar cada token.

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: moe_lora_routing
modality: text
backend: transformers

data:
  train: ./data/chat.jsonl
  max_length: 512

training:
  mole_task_adapters:        # 2-64 LoRA adapter paths (HF ids or local dirs)
    - ./adapters/math
    - ./adapters/code
    - ./adapters/chat
  mole_top_k: 2              # 1 <= top_k <= len(mole_task_adapters)
  mole_temperature: 1.0      # [1e-6, 100.0]
  epochs: 1
```

La compuerta es el único parámetro entrenable; se guarda como `mole_gate.pt` junto a la corrida,
dentro de cada `checkpoint-N`, y se restaura con `--resume`.
Se entrena como peso maestro fp32 en todos los dispositivos, así que su gradiente y los momentos
de AdamW también son fp32, incluso cuando la base congelada se carga en bf16 (en CUDA), y
`mole_gate.pt` se guarda en fp32: una `Linear(hidden, N)` ocupa `4 x hidden x N` bytes, unos 7 KB
para el ejemplo anterior y 2 MiB con hidden 8192 y 64 adaptadores. Una compuerta bf16 sin copia
fp32 redondearía la mayoría de los steps de AdamW con el `lr` por defecto (#1266).
`compute_loss` ejecuta N+1 pasadas hacia delante por step (base + cada adaptador bajo
`torch.no_grad()`, combinados con los pesos de gating por token), por lo que el tiempo por step
crece con la cantidad de adaptadores de tarea. Para servir el router entrenado, consulta
`soup serve --mole` en [Servicio y exportación](serving-and-export.md).
(v0.71.12)


## Controles de arquitectura: Mixture-of-Depths, LLaMA Pro, LongLoRA

Dos transformaciones de arquitectura que solo estaban en el esquema ya están activas para SFT /
Pretrain en Llama / Qwen / Mistral. Ambas se aplican al configurar el entrenador:

```yaml
training:
  # Mixture-of-Depths (arXiv 2404.02258): route only the top-k tokens through each
  # block. capacity_factor is the fraction of tokens that get the residual update.
  use_mod: true
  mod_capacity_factor: 0.125
```

```yaml
training:
  # LLaMA Pro: append zero-initialised identity decoder blocks and train only the new
  # ones (freeze_trainable_layers must equal expand_layers and freezes the
  # originals). Needs quantization: none and lora.r: 0 (task: sft).
  expand_layers: 4
  freeze_trainable_layers: 4
  quantization: none
  lora:
    r: 0
```

`use_mod` se adjunta DESPUÉS de `get_peft_model`, así que los routers nuevos son entrenables.
`expand_layers` se ejecuta ANTES; como `get_peft_model` volvería a congelar los bloques agregados,
se rechaza usar rango LoRA con `expand_layers` y los bloques se entrenan con `lora.r: 0`
([#1326](https://github.com/MakazhanAlpamys/Soup/issues/1326)). Las arquitecturas no compatibles
muestran una advertencia y se omiten. Elige MoD o LLaMA Pro para cada corrida, no ambos. (v0.71.12)

LongLoRA S² (`use_longlora: true`) se rechaza al cargar la configuración
([#1240](https://github.com/MakazhanAlpamys/Soup/issues/1240)). Su sobrescritura rotaba las
proyecciones query/key de la mitad de las cabezas con wrap-around bajo atención causal completa,
por lo que las posiciones anteriores podían ver los últimos tokens de la secuencia, y no aplicaba
atención agrupada. Para ampliar el contexto, usa `rope_scaling_type` con LoRA normal; consulta
[Contexto largo](peft-and-efficiency.md#long-context--yarn-llama-31-ntk-longlora).


## Spectrum: entrenamiento dirigido según SNR por capa (`soup spectrum scan`, v0.71.23)

Spectrum (arXiv:2406.06623) ajusta únicamente las capas con más señal. `soup spectrum scan`
recorre los shards `.safetensors` de un modelo **tensor por tensor**: no carga el modelo, así que
se ejecuta en una máquina con CPU incluso para modelos muy grandes, y calcula la relación
señal-ruido de valores singulares por matriz de pesos con un umbral de ruido de Marchenko-Pastur.
Clasifica las capas dentro de cada grupo de tipo de módulo e imprime el `--top-percent` superior
como bloque de configuración listo para pegar:

```bash
soup spectrum scan --model HuggingFaceTB/SmolLM2-135M --top-percent 25 --modules mlp,attn -o patch.yaml
```

```yaml
# patch.yaml — paste into your soup.yaml
training:
  unfrozen_parameters:
  - model.layers.0.mlp.down_proj
  - model.layers.29.self_attn.v_proj
  # ...
```

Luego entrena con el parche: el entrenador SFT congela **todos** los parámetros y descongela solo
el conjunto coincidente (ajuste fino completo, LoRA desactivado):

```yaml
base: HuggingFaceTB/SmolLM2-135M
task: sft
training:
  quantization: none        # Spectrum trains float weights — quantization off
  unfrozen_parameters:
  - model.layers.0.mlp.down_proj
  - model.layers.29.self_attn.v_proj
```

Las entradas de `unfrozen_parameters` son patrones regex que se comparan con los nombres de
parámetros. Requiere `task: sft`, `backend: transformers`, `modality: text` y `quantization: none`,
y es mutuamente excluyente con las funciones LoRA (`use_dora` / `use_vera` / `moe_lora` /
`relora_steps` / …) y los demás controles de congelación (`freeze_layers` / `freeze_ratio` /
`train_router_only` / `expand_layers`): las combinaciones incompatibles se rechazan de forma
explícita al cargar la configuración. Los escaneos se guardan en caché bajo `~/.soup/spectrum/`
(cámbialo con `SOUP_SPECTRUM_CACHE_DIR`); `--no-cache` la omite. Se recomienda
`--modules mlp,attn` (en lugar de `all`, el valor por defecto) para modelos muy grandes: omite las
enormes matrices de embedding/lm_head. El kernel SNR es numpy puro e invariante a la transposición,
por lo que los pesos `Conv1D` de GPT-2 obtienen la misma puntuación que los pesos Linear.
(v0.71.23)
