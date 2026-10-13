# Referencia de configuración

> 🇪🇸 Traducción al español de [`config-reference.md`](config-reference.md). Ante cualquier duda, manda el original en inglés.

[← Volver al índice de la documentación](README.md)

> **Archivo generado — no lo edites a mano.**
> Se renderiza desde `SoupConfig` en `src/soup_cli/config/schema.py`. Tras añadir o
> quitar un campo, o cambiar su tipo, valor por defecto o límites, ejecuta
> `python scripts/gen_reference_docs.py` y confirma el resultado; la suite de tests falla
> hasta que la página coincida. Las descripciones se actualizan cuando se publica una
> versión, así que entre versiones el texto de aquí puede ir por detrás del código.

Todas las claves que acepta `soup.yaml`. Una clave que no aparezca en esta página **rechaza la carga** — consulta
[Claves de configuración desconocidas](backends-and-ops.md#unknown-config-keys) para ver el error y la
sugerencia del nombre más cercano. Las guías enlazadas desde el [índice de la documentación](README.md) explican qué hacen
los ajustes en conjunto; esta página es la tabla de consulta.

Cómo leer una tabla:

- **Clave** es la ruta YAML completa: `training.lora.r` es `r:` bajo `lora:` bajo
  `training:`.
- **Tipo** usa `str`, `int`, `float`, `bool`, `list[...]` y `dict[...]`; `null` significa
  que la clave puede omitirse o ponerse a `null`; `section` es un bloque anidado con una tabla
  propia. Un valor entre comillas (`"auto"`) es el único literal aceptado junto al tipo.
- **Por defecto** se escribe tal como lo lee YAML (`null`, `true`, `"auto"`, `["a"]`).
  **requerido** significa que no hay ninguno. *(definido en tiempo de ejecución)* marca un valor por defecto calculado
  en la máquina, que esta página no imprime.
- **Restricciones** son las reglas sobre el propio valor: límites, longitudes, las opciones
  permitidas. Las reglas que abarcan varias claves (un ajuste válido para una sola tarea o backend) las
  aplican los validadores del esquema y se nombran en la descripción.

Secciones:

- [Claves de nivel superior](#claves-de-nivel-superior)
- [`data`](#data)
- [`training`](#training)
- [`training.lora`](#traininglora)
- [`training.eval_gate`](#trainingeval_gate)
- [`eval`](#eval)
- [`eval.ship`](#evalship)
- [`advise`](#advise)

## Claves de nivel superior

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `base` | `str` | **requerido** |  | Nombre o ruta del modelo base (ID de modelo de HF) |
| `task` | `str` | `"sft"` | uno de `"sft"`, `"dpo"`, `"grpo"`, `"ppo"`, `"reward_model"`, `"kto"`, `"orpo"`, `"simpo"`, `"ipo"`, `"bco"`, `"preference"`, `"pretrain"`, `"embedding"`, `"prm"`, `"tts"`, `"classifier"`, `"reranker"`, `"cross_encoder"`, `"distill"`, `"unlearn"`, `"moe_lora_routing"`, `"online_dpo"`, `"asr"` | Tipo de tarea de entrenamiento. v0.50.0 añadió 'prm'; v0.52.0 añade 'tts' (ajuste fino de TTS), 'classifier' / 'reranker' / 'cross_encoder' (cabezas de clasificación) y 'distill' (destilación de conocimiento). v0.61.0 añade 'unlearn' (NPO / SimNPO / RMU). v0.67.0 añade 'moe_lora_routing' (gating por token sobre N LoRA de tareas). |
| `modality` | `str` | `"text"` | uno de `"text"`, `"vision"`, `"audio"`, `"audio_out"` | Modalidad de entrenamiento: text (por defecto), vision (multimodal), audio (entrada de audio) o audio_out (salida de audio — en pares con task='tts', v0.52.0). |
| `backend` | `str` | `"transformers"` | uno de `"transformers"`, `"unsloth"`, `"mlx"` | Backend de entrenamiento: transformers (por defecto), unsloth (2-5x más rápido en CUDA) o mlx (Apple Silicon M1-M4) |
| `data` | `section` | **requerido** |  | Consulta [`data`](#data). |
| `training` | `section` | valores por defecto de la sección |  | Consulta [`training`](#training). |
| `output` | `str` | `"./output"` |  | Directorio de salida para el modelo entrenado |
| `experiment_name` | `str \| null` | `null` |  | Nombre del experimento para el seguimiento |
| `eval` | `section \| null` | `null` |  | Configuración de evaluación para la auto-evaluación después del entrenamiento. Consulta [`eval`](#eval). |
| `advise` | `section \| null` | `null` |  | Ajustes de decisión previa al vuelo que consume `soup advise`. Consulta [`advise`](#advise). |

## `data`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `data.train` | `str \| list[str]` | **requerido** |  | Ruta a los datos de entrenamiento o nombre de dataset de HF, o una lista de >= 2 rutas de archivos locales para combinar vía data.interleave. (#443) |
| `data.format` | `str` | `"auto"` | uno de `"alpaca"`, `"sharegpt"`, `"chatml"`, `"dpo"`, `"kto"`, `"llava"`, `"sharegpt4v"`, `"plaintext"`, `"embedding"`, `"audio"`, `"tool-calling"`, `"auto"`, `"prm"`, `"pre_tokenized"`, `"input_output"`, `"video"`, `"multimodal"`, `"raft"`, `"asr"`, `"cross_encoder"` | Formato de datos |
| `data.val_split` | `float` | `0.1` | `>= 0.0`, `<= 0.5` | Proporción de división de validación |
| `data.max_length` | `int` | `2048` | `>= 64`, `<= 1048576` | Longitud máxima de secuencia en tokens |
| `data.image_dir` | `str \| null` | `null` |  | Directorio base para resolver rutas relativas de imágenes en datasets de visión |
| `data.audio_dir` | `str \| null` | `null` |  | Directorio base para resolver rutas relativas de audio en datasets de audio |
| `data.train_on_responses_only` | `bool` | `true` |  | Enmascara con IGNORE_INDEX (-100) los tokens que no son del asistente. Cuando es True, solo el contenido del asistente contribuye a la loss de SFT. Replica el valor por defecto de LlamaFactory + Axolotl — reemplaza la heurística de TRL. (v0.36.0) |
| `data.train_on_messages_with_train_field` | `bool` | `false` |  | Máscara de entrenamiento por mensaje vía messages\[i].train: bool. Mutuamente excluyente con train_on_responses_only. (v0.36.0) |
| `data.chat_template` | `str \| null` | `null` |  | Reemplaza la plantilla de chat del tokenizador. Acepta un nombre registrado (chatml, llama3, qwen2.5, mistral, gemma3, phi4, deepseek-r1) o una cadena Jinja en bruto. None = usar la plantilla incluida del tokenizador (falla de forma explícita si no existe). (v0.36.0) |
| `data.raft_shuffle_seed` | `int \| null` | `null` | `>= 0`, `<= 2147483647` | Semilla para el shuffle de documentos golden/distractor de RAFT (data.format='raft'). Los documentos siempre se mezclan por robustez frente a distractores; este ajuste fija qué permutación reproducible se usa. None = semilla 0. (v0.71.10 #199) |
| `data.raft_epoch_shuffle` | `bool` | `false` |  | Re-permuta los documentos golden/distractor de RAFT en CADA época de entrenamiento (data.format='raft'). Cuando es False (por defecto), el orden de los documentos se fija una vez en el momento de la tokenización y permanece igual entre épocas; cuando es True, el trainer recompone + re-tokeniza las filas por época con una sal de época para que el modelo no pueda memorizar una posición fija de documento golden. (v0.71.17 #253) |
| `data.replay` | `str \| null` | `null` |  | Ruta a un dataset ANTIGUO para intercalarlo en el entrenamiento como repaso de aprendizaje continuo, de modo que el ajuste fino en una tarea nueva no borre la anterior. Las filas se mezclan SOLO en train (nunca en val, que permanece puro de la tarea nueva). Solo sft / pretrain; incompatible con packing / multipack. (v0.71.36) |
| `data.replay_ratio` | `float` | `0.1` | `> 0.0`, `<= 0.5` | Fracción del conjunto de train mezclado FINAL que son filas de replay: n_replay = round(r/(1-r) \* n_new). Con 0.1 sobre 1000 filas nuevas son 111 filas de replay -> 1111 en total -> 10.0%. (v0.71.36) |
| `data.replay_seed` | `int \| null` | `null` | `>= 0`, `<= 2147483647` | Semilla para el muestreo de replay + interleave. None = semilla 0. (v0.71.36) |
| `data.video_dir` | `str \| null` | `null` |  | Directorio base para resolver rutas relativas de video en datasets de video. Replica image_dir / audio_dir. (v0.42.0) |
| `data.tokenized_path` | `str \| null` | `null` |  | Ruta a una caché pre-tokenizada producida por `soup data preprocess`. Cuando se define, el trainer omite la etapa de tokenización y lee los tensores directamente. Replica tokenized_path de LF / el tipo `empty` de Axolotl. (v0.42.0) |
| `data.streaming` | `bool` | `false` |  | Paso directo a `streaming=True` de los datasets de HF. Úsalo para datasets que no caben en disco. Se combina con `buffer_size`. (v0.42.0) |
| `data.buffer_size` | `int \| null` | `null` |  | Tamaño del buffer de shuffle para datasets en streaming. None = valor por defecto de HF. Límites \[1, 1_000_000\]. (v0.42.0) |
| `data.shards` | `int \| null` | `null` |  | Número de shards para las divisiones del dataset de HF (axolotl `shards`). Límites \[1, 1024\]. (v0.42.0) |
| `data.interleave` | `str \| dict \| null` | `null` |  | Estrategia de interleave para varios datasets: 'concat' / 'under' / 'over' / {strategy: 'probs', probs: \[...\]}. (v0.42.0) |
| `data.mask_history` | `bool` | `false` |  | mask_history de LF — enmascara todos los turnos del asistente salvo el último durante el cálculo de la loss. (v0.42.0) |
| `data.train_on_prompt` | `bool` | `false` |  | train_on_prompt de LF — incluye los tokens del prompt en la loss. Inverso de train_on_responses_only. (v0.42.0) |
| `data.eval_on_each_dataset` | `bool` | `false` |  | eval_on_each_dataset de LF — al intercalar, ejecuta la evaluación en cada dataset constituyente por separado. (v0.42.0) |
| `data.split_thinking` | `bool` | `false` |  | split_thinking de Axolotl — separa los bloques de razonamiento `<think>` de la respuesta final para un enmascarado más fino. Estilo Qwen3. (v0.42.0) |
| `data.image_min_pixels` | `int \| null` | `null` |  | Cantidad mínima de píxeles por imagen para datos de visión. (v0.42.0) |
| `data.image_max_pixels` | `int \| null` | `null` |  | Cantidad máxima de píxeles por imagen para datos de visión. (v0.42.0) |
| `data.image_resize_algorithm` | `str \| null` | `null` | uno de `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos"` | Algoritmo de redimensionado de Pillow para el preprocesamiento de imágenes. (v0.42.0) |
| `data.video_fps` | `float \| null` | `null` |  | Fotogramas por segundo objetivo para el preprocesamiento de video. (v0.42.0) |
| `data.video_maxlen` | `int \| null` | `null` |  | Número máximo de fotogramas por clip de video. Límites (0, 4096]. (v0.42.0) |
| `data.add_new_tokens` | `list[str] \| null` | `null` |  | Añade estos tokens al vocabulario del tokenizador + redimensiona los embeddings. Límite de 10_000 entradas; por token &lt;= 256 caracteres; sin duplicados. (v0.42.0) |
| `data.new_special_tokens` | `list[str] \| null` | `null` |  | Como add_new_tokens pero registrados como additional_special_tokens para que el tokenizador no los divida. (v0.42.0) |
| `data.resize_vocab` | `bool` | `false` |  | Redimensiona la matriz de embeddings de entrada/salida del modelo cuando add_new_tokens / new_special_tokens ampliaron el vocabulario. (v0.42.0) |
| `data.extend_conversation` | `bool` | `false` |  | Extensión de conversación al estilo Unsloth — extiende el último turno del asistente con N tokens más para prompts de 'continuar'. (v0.42.0) |
| `data.skip_prepare_dataset` | `bool` | `false` |  | skip_prepare_dataset de Axolotl — vía de escape cuando la entrada ya está en el esquema que espera el trainer. (v0.42.0) |
| `data.remove_unused_columns` | `bool` | `false` |  | remove_unused_columns de HF Trainer. Ningún trainer lee este campo; los trainers que lo definen pasan False para que un collator personalizado aún pueda ver las columnas extra. Un `true` explícito carga con una advertencia y se ignora, y una versión posterior lo rechaza. (v0.42.0; #759) |
| `data.prompt_strategy` | `str \| null` | `null` |  | Transformación Python 'module.path:function_name' al estilo Axolotl, resuelta de forma perezosa y aplicada a cada fila durante el formateo de SFT. |
| `data.forget_set` | `str \| null` | `null` |  | Ruta o nombre de dataset de HF para el conjunto de olvido (filas a desaprender). Requerido cuando task='unlearn'. Se rechazan los bytes nulos, limitado a 4096 caracteres. La comprobación de contención se delega al cargador del lado del trainer para que los IDs de dataset de HF (p. ej. ``locuslab/TOFU``) aún pasen el esquema. (v0.61.0) |
| `data.retain_set` | `str \| null` | `null` |  | Ruta o nombre de dataset de HF para el conjunto de retención (filas cuyo desempeño debe preservarse). Opcional pero recomendado — NPO/SimNPO/RMU se degradan sin uno. Misma validación que forget_set. (v0.61.0) |

## `training`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `training.epochs` | `int` | `3` | `>= 1` | Número de épocas de entrenamiento |
| `training.lr` | `float` | `2e-05` | `> 0` | Tasa de aprendizaje |
| `training.batch_size` | `int \| "auto"` | `"auto"` |  | Tamaño de batch. 'auto' = encuentra el máximo que quepa en memoria. |
| `training.auto_batch_size_strategy` | `str` | `"auto"` | uno de `"auto"`, `"static"`, `"probe"` | Cómo elegir el tamaño de batch automático: 'static' (fórmula rápida), 'probe' (bucle real de intento/reducción a la mitad ante OOM), 'auto' (probe en CUDA, static en CPU). Por defecto 'auto' (v0.36.0). |
| `training.gradient_accumulation_steps` | `int` | `4` | `>= 1` |  |
| `training.seed` | `int \| null` | `null` | `>= 0`, `<= 4294967295` | Semilla de entrenamiento (inicialización de pesos de los parámetros nuevos, orden de los datos, dropout). Llega a TrainingArguments.seed y al muestreador de multipack. Sin definir = valor por defecto de HF de 42. |
| `training.data_seed` | `int \| null` | `null` | `>= 0`, `<= 4294967295` | Semilla separada solo para el muestreo de datos (TrainingArguments.data_seed). Defínela junto a `seed` para variar el orden de los datos manteniendo fija la inicialización. Sin definir = sigue a `seed`. |
| `training.warmup_ratio` | `float` | `0.03` | `>= 0.0`, `<= 0.5` |  |
| `training.weight_decay` | `float` | `0.01` | `>= 0.0` |  |
| `training.max_grad_norm` | `float` | `1.0` | `> 0` |  |
| `training.lora` | `section` | valores por defecto de la sección |  | Consulta [`training.lora`](#traininglora). |
| `training.quantization` | `str` | `"4bit"` | uno de `"4bit"`, `"8bit"`, `"none"`, `"gptq"`, `"awq"`, `"hqq:1bit"`, `"hqq:2bit"`, `"hqq:3bit"`, `"hqq:4bit"`, `"hqq:5bit"`, `"hqq:6bit"`, `"hqq:8bit"`, `"aqlm"`, `"eetq"`, `"mxfp4"`, `"fp8"`, `"bitnet_1.58"` | Cuantización (v0.38.0 — Quant Menu): 4bit (BNB QLoRA), 8bit (BNB), none, gptq / awq (carga un checkpoint pre-cuantizado y entrena LoRA encima), hqq:Nbit (HQQ 1-8 bit, N en {1..6, 8}), aqlm (2-bit extremo), eetq (8-bit rápido), mxfp4 (base pre-cuantizada MXFP4, descuantizada al cargar), fp8 (carga un checkpoint FP8 con descuantización al cargar), bitnet_1.58 (reservado hasta que el enrutamiento del trainer BitNet esté disponible). |
| `training.gptq_disable_exllama` | `bool` | `true` |  | v0.38.0 — desactiva el backend exllama para GPTQ. PEFT requiere el backend triton; exllama rompe silenciosamente el entrenamiento de adaptadores. |
| `training.bnb_4bit_quant_storage` | `str \| null` | `null` | uno de `"uint8"`, `"float16"`, `"bfloat16"`, `"float32"` | v0.38.0 — dtype de almacenamiento de BNB 4-bit. FSDP+QLoRA lo resuelve automáticamente al dtype de cómputo flotante efectivo; fuera de FSDP, None mantiene el uint8 heredado por defecto de BNB. |
| `training.quantization_aware` | `bool \| "fp8" \| "quest"` | `false` |  | Entrenamiento con conocimiento de cuantización (QAT). False=desactivado, True=rechazado al cargar (el QAT int8 no está implementado, #1222), 'fp8'=entrenamiento FP8 en H100/B100 (v0.28.0), 'quest'=ruta experimental mixta W4/A4+A16 QuEST (#674). |
| `training.fp8_recipe` | `str` | `"tensorwise"` | uno de `"tensorwise"`, `"rowwise"`, `"rowwise_with_gw_hp"` | Receta de escalado FP8 (solo se usa cuando quantization_aware='fp8'). 'tensorwise' (la más rápida, por defecto), 'rowwise' (más precisa, CUTLASS rowwise), 'rowwise_with_gw_hp' (la más precisa, grad_weight en alta precisión). (v0.28.1). |
| `training.optimizer` | `str` | `"adamw_torch"` |  | Nombre del optimizador (v0.41.0 amplió la lista permitida: HF-native, bnb-8bit, lomo/adalomo, apollo_adamw, grokadamw, schedule_free). Nota: diez nombres que transformers 5.x rechaza en TrainingArguments (p. ej. badam, adam_mini, muon, dion, came_pytorch, ao_adamw\_{fp8,4bit,8bit}) se rechazan al cargar la configuración (#1269); adamw_hf y muon siguen funcionando en backend: mlx. Consulta soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS para la lista completa. |
| `training.lr_groups` | `list[dict[str, str \| float]] \| null` | `null` |  | Preparado, no aplicado: una anulación de LR por módulo (lista de entradas {pattern, lr} o un dict {pattern: lr}, limitada a 32) que ningún optimizador lee; definirla avisa al cargar y luego se rechaza (#761). (v0.41.0) |
| `training.expand_layers` | `int \| null` | `null` | `>= 1`, `<= 64` | LLaMA Pro: añade N bloques transformer inicializados a cero y congela los originales antes de SFT o pre-entrenamiento. |
| `training.freeze_trainable_layers` | `int \| null` | `null` |  | LLaMA Pro: solo se aplica junto con expand_layers, debe ser igual a él y se rechaza en caso contrario. Congela todos los parámetros del modelo original para que solo entrenen los bloques añadidos; no selecciona las top-N ni las bottom-N capas. Magnitud limitada a 1000. (v0.41.0) |
| `training.use_mod` | `bool` | `false` |  | Activa el enrutamiento selectivo de tokens Mixture-of-Depths (arXiv:2404.02258). Operativo en v0.71.12 #84 para SFT + Pretrain en Llama / Qwen / Mistral; cada capa decodificadora recibe un router que pasa por el bloque solo los top-k tokens (k = floor(seq_len \* mod_capacity_factor)). |
| `training.mod_capacity_factor` | `float` | `0.125` | `> 0.0`, `<= 1.0` | Fracción de tokens enrutados a través de cada bloque cuando use_mod=True. Acotada a (0, 1]; por defecto 0.125 (según el artículo de MoD). (v0.71.12 #84) |
| `training.load_in_8bit` | `bool \| null` | `null` |  | Alias amigable para quantization='8bit' / 'none'. Cuando es True, reescribe quantization a '8bit' si actualmente es 'none'/'4bit'. Entra en conflicto con load_in_16bit. (v0.41.0) |
| `training.load_in_16bit` | `bool \| null` | `null` |  | Alias amigable: cuando es True, define quantization='none' (LoRA completo en bf16/fp16). Entra en conflicto con load_in_8bit. (v0.41.0) |
| `training.scheduler` | `str` | `"cosine"` |  | Tipo de scheduler de LR |
| `training.save_steps` | `int` | `100` |  | Guarda un checkpoint cada N steps |
| `training.logging_steps` | `int` | `10` |  | Registra métricas cada N steps |
| `training.eval_steps` | `int \| null` | `null` | `>= 1` | Evalúa la división de validación cada N steps del optimizador (contados después de la acumulación de gradientes, como save_steps). Sin definir = evalúa al final de cada época. grpo solo evalúa cuando esto está definido, porque una evaluación allí genera completions. (#1223) |
| `training.dpo_beta` | `float` | `0.1` | `> 0` | beta de DPO — coeficiente de penalización KL |
| `training.kto_beta` | `float` | `0.1` | `> 0` | beta de KTO — coeficiente de penalización KL |
| `training.orpo_beta` | `float` | `0.1` | `> 0` | beta de ORPO — peso del odds ratio |
| `training.simpo_gamma` | `float` | `0.5` | `>= 0` | gamma de SimPO — término de margen de recompensa |
| `training.cpo_alpha` | `float` | `1.0` | `> 0` | alpha de CPO/SimPO — peso de la loss NLL |
| `training.ipo_tau` | `float` | `0.1` | `> 0` | tau de IPO — fuerza de regularización |
| `training.bco_beta` | `float` | `0.1` | `> 0` | beta de BCO — coeficiente de penalización KL |
| `training.preference_loss` | `str \| null` | `null` | uno de `"dpo"`, `"simpo"`, `"orpo"`, `"ipo"`, `"bco"` | Loss de preferencia para task='preference'. Uno de: dpo, simpo, orpo, ipo, bco. Mutuamente excluyente con task en {dpo, simpo, orpo, ipo, bco}. |
| `training.dpo_beta_schedule` | `str \| null` | `null` | uno de `"linear"`, `"cosine"`, `"exponential"` | Recocido del β de DPO durante el entrenamiento. None = β constante (por defecto). Requiere dpo_beta_end. Solo tareas de la familia DPO (dpo, ipo, preference+dpo). |
| `training.dpo_beta_end` | `float \| null` | `null` | `> 0` | β objetivo al final del entrenamiento cuando dpo_beta_schedule está definido. Debe ser > 0. El β inicial es dpo_beta. |
| `training.dpo_ref_regen_epochs` | `int \| null` | `null` | `>= 1`, `<= 1000` | Reemplaza el adaptador ref congelado por el adaptador activo actual cada N épocas. Copia los pesos del adaptador .default. en .ref. bajo torch.no_grad(). Requiere LoRA (lora.r >= 1); el ajuste fino completo no es compatible. |
| `training.preference_loss_weights` | `dict[str, float] \| null` | `null` |  | Mezcla ponderada de losses de preferencia, p. ej. {'dpo': 0.7, 'bco': 0.3}. Cada peso ∈ (0, 1]; los pesos deben sumar 1.0 (±1e-6). Todas las claves deben ser miembros de {dpo, simpo, orpo, ipo, bco}. Requiere task='preference'; mutuamente excluyente con preference_loss (la forma escalar). Limitado a 5 componentes (el conjunto admitido). |
| `training.grpo_beta` | `float` | `0.1` | `>= 0.0`; finite | beta de GRPO - coeficiente de penalización KL (por defecto 0.1). Ponlo a 0 para desactivar por completo la penalización KL (recetas sin KL como DAPO y Dr. GRPO, #1247). |
| `training.num_generations` | `int` | `4` | `>= 2` | Número de generaciones por prompt para GRPO |
| `training.reward_fn` | `str \| null` | `"accuracy"` |  | Función de recompensa: 'accuracy', 'format', 'verifiable', una ruta a un archivo .py personalizado, o un ensemble separado por comas de las anteriores (p. ej. 'accuracy,format') — la forma con comas es solo para GRPO (v0.71.40). |
| `training.verifiable_domain` | `str \| null` | `null` | uno de `"math"`, `"code"`, `"json_schema"` | Dominio de recompensa verificable de RLVR: math \| code \| json_schema. Requerido cuando reward_fn='verifiable'. |
| `training.prm_reward` | `str \| null` | `null` |  | Ruta (o id de HF) a un PRM entrenado con Soup (task='prm') usado como recompensa por step de GRPO (v0.71.30). Cuando se define, el PRM reemplaza a reward_fn. Requiere task='grpo', backend='transformers', modality='text'. |
| `training.prm_aggregate` | `str` | `"min"` | uno de `"min"`, `"prod"`, `"last"` | Cómo se pliegan las puntuaciones por step del PRM en una sola recompensa: min (eslabón más débil, por defecto) \| prod \| last. Solo tiene sentido cuando prm_reward está definido. NOTA: 'prod' asume que las puntuaciones por step están acotadas en \~\[0,1] (una probabilidad de corrección del step). La cabeza PRM de Soup se entrena con regresión MSE sin restricciones, así que 'prod' puede dispararse / cambiar de signo con etiquetas no acotadas — prefiere el 'min' por defecto a menos que tus etiquetas de PRM estén calibradas a \[0,1]. |
| `training.online_dpo_judge` | `str \| null` | `null` |  | URL del juez (ollama://model \| https://... \| http://localhost) para task='online_dpo'. Mutuamente excluyente con reward_model. |
| `training.online_dpo_loss_type` | `str` | `"sigmoid"` | uno de `"sigmoid"`, `"ipo"` | Tipo de loss de Online DPO (se asigna a OnlineDPOConfig.loss_type). |
| `training.online_dpo_max_new_tokens` | `int` | `64` | `>= 1`, `<= 4096` | Máximo de tokens nuevos generados por step de online-DPO. |
| `training.asr_language` | `str \| null` | `null` |  | Idioma objetivo para task='asr' (Whisper), p. ej. 'en' o 'spanish'. Define el prompt forzado del decodificador; None usa el valor por defecto del modelo / la detección de idioma. |
| `training.asr_task` | `str` | `"transcribe"` | uno de `"transcribe"`, `"translate"` | Objetivo de decodificación de Whisper para task='asr': 'transcribe' (mismo idioma) o 'translate' (a inglés). |
| `training.asr_lora` | `bool` | `false` |  | LoRA opcional para task='asr' (adapta las proyecciones de atención q/v). Por defecto False = ajuste fino completo (el Whisper pequeño cabe en una GPU de 4 GB). Replica classifier_lora — el bloque estándar training.lora sigue aportando r/alpha/dropout cuando se activa. |
| `training.grpo_variant` | `str \| null` | `null` | uno de `"standard"`, `"gspo"`, `"dapo"`, `"dr_grpo"`, `"bnpo"`, `"two_sided"`, `"rft"` | Variante del objetivo GRPO: standard \| gspo \| dapo \| dr_grpo \| bnpo \| two_sided \| rft. Por defecto es None (GRPO heredado). Requiere task='grpo'; el trainer de GRPO instala la loss seleccionada. |
| `training.grpo_delta` | `float \| null` | `null` | `> 0.0`, `<= 1.0` | Radio de recorte simétrico para grpo_variant='two_sided' (requerido) o grpo_variant='gspo' (radio de recorte de secuencia opcional, por defecto 0.2). Se rechaza para todas las demás variantes. |
| `training.grpo_fp16` | `bool` | `false` |  | Fuerza precisión mixta FP16 para GRPO/RL (paridad con unsloth). En CUDA, el trainer de GRPO reenvía fp16=True y bf16=False; en dispositivos que no son CUDA este flag no habilita FP16. |
| `training.long_context_grpo` | `bool` | `false` |  | Activa GRPO de contexto largo (unsloth: 380K B200 / 110K H100). Solo validación de compatibilidad: aún ningún trainer consume este flag. Requiere task='grpo' en un backend que no sea mlx y use_ring_attention=False. |
| `training.vllm_sleep_mode` | `bool` | `false` |  | Activa el modo de suspensión/en espera de vLLM entre rollouts (ahorro de memoria durante el step de optimización). Requiere backend en {transformers, unsloth}; el trainer de GRPO lo reenvía a los runtimes compatibles de TRL y vLLM. |
| `training.rollout_backend` | `str \| null` | `null` | uno de `"art"`, `"ruler"`, `"nemo_gym"`, `"openenv"` | Backend de rollout de agente multiturno (paridad con unsloth / axolotl): art (OpenPipe ART) / ruler / nemo_gym / openenv. Requiere task='grpo'. openenv se ejecuta en vivo (v0.71.21 #125) vía training.rollout_func; art/ruler/nemo_gym están condicionados a una importación perezosa. |
| `training.rollout_func` | `str \| null` | `null` |  | Función de rollout de OpenEnv como 'module.path:function_name' (v0.71.21 #125). Requiere rollout_backend='openenv'. El llamable recibe la lista de prompts semilla y devuelve filas de rollout ({'prompt': str\|messages, 'answer'?: str}) que reemplazan el dataset de prompts de GRPO. Política de entrada confiable: nombra código controlado por el operador (replica data.prompt_strategy). |
| `training.ref_model_ema_alpha` | `float \| null` | `null` | `> 0.0`, `<= 1.0` | Coeficiente de media móvil exponencial para la sincronización del modelo ref (policy → reference). Debe estar en (0, 1]. None = desactivado. Paridad con Axolotl. |
| `training.replay_buffer_size` | `int \| null` | `null` | `>= 1`, `<= 1000000` | Tamaño acotado del buffer de replay para rollouts de GRPO. None = desactivado. Paridad con Axolotl. |
| `training.async_grpo_prefetch` | `bool` | `false` |  | Solapa rollout + entrenamiento vía prefetch asíncrono (axolotl). Requiere backend en {transformers, unsloth}. |
| `training.tis_threshold` | `float \| null` | `null` | `> 0.0`, `<= 100.0` | Umbral de muestreo de importancia truncado (unsloth, axolotl). Debe estar en (0, 100]. None = desactivado. |
| `training.mask_truncated_completions` | `bool` | `false` |  | Enmascara los completions truncados al calcular el gradiente de la policy (en pares con tis_threshold). Paridad con Unsloth + axolotl. |
| `training.defer_rerolling` | `bool` | `false` |  | Aplaza el re-lanzamiento (re-roll) de prompts idénticos entre steps de optimización (axolotl). Ahorra rollouts en prompts repetidos. |
| `training.skip_zero_advantage` | `bool` | `false` |  | Omite el backward pass en muestras cuyo advantage es exactamente cero (axolotl). Evita cómputo desperdiciado en muestras sin señal. |
| `training.off_policy_mask_threshold` | `float \| null` | `null` | `>= 0.0`, `<= 1.0` | Umbral de máscara off-policy para el gating de token/secuencia (axolotl). Debe estar en \[0, 1]. None = desactivado. |
| `training.vision_grpo` | `bool` | `false` |  | Activa Vision RL / VLM RL — extiende GRPO/PPO a la modalidad de visión (Qwen2-VL / Pixtral / InternVL). Requiere modality='vision', task en {grpo, ppo}, backend en {transformers, unsloth}. La validación de compatibilidad está disponible, pero aún ningún trainer consume este flag. |
| `training.hub` | `str` | `"hf"` | uno de `"hf"`, `"modelscope"`, `"modelers"` | Hub de descarga de modelos durante el entrenamiento. 'hf' (por defecto), 'modelscope' (alojado en China; refleja la mayoría de Llama/Qwen/etc.), 'modelers' (hub de Openmind). `soup push --hub` selecciona su destino de subida de forma independiente a este campo. |
| `training.tts_family` | `str \| null` | `null` | uno de `"orpheus"`, `"sesame_csm"`, `"llasa"`, `"spark"`, `"oute"` | Familia de modelo TTS — requerida cuando task='tts'. Las opciones ejecutables son orpheus, llasa, spark y oute. sesame_csm se conserva solo para que las configuraciones heredadas reciban un rechazo explícito hasta que Soup tenga un trainer multimodal CSM nativo. |
| `training.tts_emotion` | `str \| null` | `null` |  | Etiqueta de emoción opcional para las familias condicionadas por emoción (Orpheus / Oute). Permitida por familia. (v0.52.0) |
| `training.num_labels` | `int \| null` | `null` | `>= 1`, `<= 1024` | Número de etiquetas de salida para task en (classifier, reranker, cross_encoder). Requerido cuando task es una de esas. (v0.52.0) |
| `training.classifier_kind` | `str \| null` | `null` | uno de `"single_label"`, `"multi_label"` | Tipo de cabeza de clasificación de secuencias: single_label (por defecto para task='classifier') o multi_label. (v0.52.0) |
| `training.label_names` | `list[str] \| null` | `null` |  | Nombres de etiqueta legibles por humanos opcionales. La longitud debe coincidir con num_labels. Limitado a 1024 entradas. (v0.52.0) |
| `training.mole_task_adapters` | `list[str] \| null` | `null` |  | Rutas (ids de HF o directorios locales) a los N adaptadores LoRA de tareas pre-entrenados sobre los que enruta la puerta MoLE. Requerido cuando task='moe_lora_routing'; 2-64 entradas, sin duplicados. El modelo base + cada adaptador de tarea permanecen congelados — solo entrena la puerta. (v0.71.12 #222) |
| `training.mole_top_k` | `int \| null` | `null` |  | Número de adaptadores de tarea a los que se enruta cada token (despacho top-k disperso). Por defecto es num_task_adapters (denso) cuando no se define. 1 &lt;= top_k &lt;= len(mole_task_adapters). (v0.71.12 #222) |
| `training.mole_temperature` | `float \| null` | `null` |  | Temperatura del softmax para el router MoLE (por defecto 1.0). (1e-6, 100.0]. (v0.71.12 #222) |
| `training.teacher_model` | `str \| null` | `null` |  | Id de HF o ruta local del modelo teacher — requerido cuando task='distill'. El trainer de destilación lo carga congelado. Se rechazan los bytes nulos, limitado a 512 caracteres. |
| `training.distill_divergence` | `str \| null` | `null` | uno de `"forward_kl"`, `"reverse_kl"`, `"js"` | Divergencia usada para la loss de destilación. 'kl' es un alias de 'forward_kl' (forma canónica). (v0.52.0) |
| `training.distill_temperature` | `float \| null` | `null` |  | Temperatura del softmax aplicada a los logits del teacher y del student antes de la divergencia. Acotada a \[0.05, 100.0\]. (v0.52.0) |
| `training.distill_mode` | `str` | `"token"` | uno de `"token"`, `"sequence"` | Modo de destilación. 'token' (por defecto, v0.53.2) = KL de logits alineada por columnas — requiere un tokenizador compartido (o define uld_strategy para la ruta de logits entre tokenizadores distintos). 'sequence' (v0.71.12) = KD a nivel de secuencia: el teacher GENERA un completion por prompt y el student hace CE simple sobre la salida re-tokenizada, lo que funciona con CUALQUIER par de tokenizadores (p. ej. student Llama-3 / teacher Qwen-2). |
| `training.distill_chunk_size` | `int \| null` | `null` |  | Tamaño de chunk de tokens para evaluar el kernel de divergencia de destilación. Divide en chunks los tokens supervisados activos para reducir la memoria máxima retenida. Sin definir (None), procesa todos los tokens en un solo chunk. (#722) |
| `training.distill_checkpoint` | `bool` | `false` |  | Activa el checkpointing de activaciones no reentrante por chunk de tokens durante el cálculo de la divergencia de destilación para minimizar la memoria retenida por autograd. (#722) |
| `training.classifier_lora` | `bool` | `false` |  | Envuelve la cabeza de clasificación de secuencias con LoRA (task_type='SEQ_CLS') en lugar del ajuste fino completo. Opcional (False por defecto preserva el comportamiento de ajuste fino completo de v0.53.2). Reutiliza el bloque training.lora (r / alpha / dropout / target_modules). Solo se admite para task en (classifier, reranker, cross_encoder). (v0.71.12 #146) |
| `training.ebft_variant` | `str \| null` | `null` | uno de `"structured"`, `"strided"` | Rechazado al cargar la configuración (#1230): EBFT aún no es un objetivo distinto. El término añadido no tenía un desplazamiento causal, así que premiaba copiar el token de entrada, y desplazado duplica la cross-entropy. |
| `training.ebft_temperature` | `float \| null` | `null` |  | Temperatura de muestreo para el proxy de energía de EBFT. Acotada a \[1e-4, 100.0\]. Solo se aplica con ebft_variant, que se rechaza al cargar la configuración (#1230). (v0.52.0) |
| `training.gdpo_variant` | `str \| null` | `null` | uno de `"standard"`, `"length_normalized"`, `"margin"` | Rechazado al cargar la configuración (#1309): las versiones de TRL admitidas carecen del hook de loss de DPO. Elimina este campo; usa task: dpo simple para el estándar, o task: simpo como el objetivo normalizado por longitud más cercano (no idéntico). |
| `training.moe_expert_quant` | `str \| null` | `null` | uno de `"nf4"`, `"int8_rowwise"` | Cuantización por experto para los bloques Linear de MoE fusionado. Requiere moe_lora=true. (v0.52.0) |
| `training.train_router_only` | `bool` | `false` |  | Congela todos los expertos + entrena solo el router de gating (receta MoE de unsloth). Requiere moe_lora=true. (v0.52.0) |
| `training.reasoning_effort` | `str \| null` | `null` | uno de `"low"`, `"medium"`, `"high"` | Nivel de esfuerzo de razonamiento en tiempo de entrenamiento de gpt-oss. El formateador de datos de la familia SFT lo inyecta como prefijo del prompt. |
| `training.train_on_eot` | `bool` | `false` |  | Incluye los tokens de control EOT / EOS explícitos en la loss de SFT (axolotl ``train_on_eot``). False por defecto coincide con la convención de HF Trainer. (v0.52.0) |
| `training.kv_cache_type` | `str \| null` | `null` | uno de `"q8_0"`, `"bf16"`, `"f16"`, `"fp8"` | Tipo de elemento del KV-cache para inferencia (q8_0 / bf16 / f16 / fp8). Lo aplica soup serve en el backend transformers; la compatibilidad de backend y hardware se valida antes de cargar el modelo. |
| `training.fp8_attention` | `bool` | `false` |  | Extiende el menú FP8 de v0.28.0 a la atención FP8 (flag con paridad en axolotl). Requiere quantization_aware='fp8'. Los wrappers de la familia DPO, GRPO/PPO, pre-entrenamiento, modelo de recompensa y embedding lo enrutan a través de la configuración de precisión avanzada compartida; la ruta SFT por defecto no lo consume. |
| `training.nvfp4` | `bool` | `false` |  | Entrenamiento NVFP4 solo para Blackwell (unsloth + axolotl). Los wrappers de la familia DPO, GRPO/PPO, pre-entrenamiento, modelo de recompensa y embedding lo enrutan a través de la conversión de torchao; la ruta SFT por defecto no lo consume. |
| `training.unsloth_bnb_4bit` | `bool` | `false` |  | Promueve Unsloth Dynamic 4-bit de 'inferable' a un flag nativo. Requiere backend='unsloth' y quantization='4bit'. (v0.53.0) |
| `training.bnb_4bit_use_double_quant` | `bool \| null` | `null` |  | Aplica la doble cuantización de BNB 4-bit (paridad con LF / Axolotl). Sin definir significa el valor por defecto incluido (activado, para coincidir con todas las rutas de carga de 4-bit); ponlo a false para desactivarla. Solo tiene sentido cuando quantization='4bit'. (v0.53.0) |
| `training.llm_int8` | `bool` | `false` |  | Alias explícito de LLM.int8 de 8-bit para quantization='8bit'. Cuando es True, requiere quantization='8bit'. (v0.53.0) |
| `training.quantize_ref_model` | `bool` | `false` |  | Aplica la misma configuración del Quant Menu al modelo de referencia (modelo ref de DPO/IPO/SimPO/ORPO/BCO) — amplía v0.40.5. (v0.53.0) |
| `training.quantize_reward_model` | `bool` | `false` |  | Aplica la misma configuración del Quant Menu al modelo de recompensa (tarea PPO/reward_model) — amplía v0.40.5. (v0.53.0) |
| `training.echo_trap_enabled` | `bool` | `false` |  | Activa la detección de echo-trap al estilo RAGEN durante RL de agentes multiturno. Requiere task en {'grpo', 'ppo'} en un backend que no sea mlx; instala el callback de señal RL compartido. |
| `training.echo_trap_threshold` | `float` | `0.6` | `>= 0.0`, `<= 1.0` | Umbral sobre la señal de echo agregada. Por encima de esto = TRAP. Acotado a \[0.0, 1.0\]. (v0.70.0) |
| `training.echo_trap_halt` | `bool` | `false` |  | Detiene el entrenamiento automáticamente ante un veredicto TRAP. Requiere echo_trap_enabled=True. (v0.70.0) |
| `training.echo_trap_tokenizer_aware` | `bool` | `false` |  | Usa n-gramas de ids de tokenizador para la puntuación de echo-trap en lugar de tokens separados por espacios. Más sensible a la repetición de subpalabras pero ligado al vocabulario del tokenizador activo. Requiere echo_trap_enabled=True. (v0.70.x) |
| `training.rl_checkpoint_save_every_steps` | `int \| null` | `null` | `>= 1`, `<= 10000000` | Guarda un checkpoint a mitad de época compatible con RL cada N steps. None = usar solo el checkpoint por época de HF Trainer. Requiere task en {'grpo', 'ppo'}; el callback persiste el adaptador de policy, el estado opcional del optimizador y un manifiesto. |
| `training.rl_checkpoint_keep_last` | `int` | `3` | `>= 1`, `<= 100` | Número de checkpoints RL recientes que se conservan. Los más antiguos se podan en el momento de escribir. (v0.70.0) |
| `training.rl_checkpoint_include_optimizer` | `bool` | `true` |  | Incluye el estado del optimizador AdamW / Lion en el checkpoint RL de mitad de época. (v0.70.0) |
| `training.rl_checkpoint_include_ref_model` | `bool` | `false` |  | Incluye el estado del modelo de referencia congelado en el checkpoint RL. False por defecto (el modelo ref se puede reconstruir desde cfg.base). (v0.70.0) |
| `training.rl_checkpoint_include_rollout_buffer` | `bool` | `false` |  | Incluye el búfer de rollout / replay en el checkpoint RL para que las ejecuciones reanudadas no pierdan la experiencia recopilada. (v0.70.0) |
| `training.minillm_enabled` | `bool` | `false` |  | Activa la destilación on-policy al estilo MiniLLM (Gu et al. 2024). Requiere task='distill' en un backend distinto de mlx; instala la ruta de muestreo mixto con teacher y loss reverse-KL. |
| `training.minillm_teacher_mix_ratio` | `float` | `0.0` | `>= 0.0`, `<= 1.0` | Mezcla de teacher usada por MiniLLM. Offline (minillm_on_policy=false): peso de la distribución del teacher en el objetivo reverse-KL (ratio \* teacher + (1 - ratio) \* stopgrad(student)). Se rechaza 0.0 cuando MiniLLM está activado porque entonces el teacher tiene peso cero. On-policy: probabilidad de muestrear del teacher durante rollout; 0.0 = muestreo solo del student, y la loss sigue siendo KL(student \|\| teacher). Rango típico 0.2-0.5. (v0.70.0; cero offline rechazado en #692) |
| `training.minillm_length_normalize` | `bool` | `true` |  | Normaliza por longitud la log-probabilidad del rollout antes del término reverse-KL. Evita que las completions largas dominen el gradiente. (v0.70.0) |
| `training.minillm_pretrain_anchor_weight` | `float` | `0.0` | `>= 0.0`, `<= 1.0` | Peso del término ancla de loss de preentrenamiento (SFT sobre un corpus pequeño de preentrenamiento). Evita desviarse del lenguaje coherente. Requiere minillm_pretrain_anchor_path cuando es > 0. (v0.70.0) |
| `training.minillm_pretrain_anchor_path` | `str \| null` | `null` |  | Ruta al JSONL de preentrenamiento usado por el término ancla. Obligatoria cuando minillm_pretrain_anchor_weight > 0. Se rechazan bytes nulos; máximo 4096 caracteres. (v0.70.0) |
| `training.minillm_on_policy` | `bool` | `false` |  | Usa el rollout MiniLLM verdaderamente on-policy con mezcla de teacher (Gu et al. 2024 §3.1): muestrea un rollout autoregresivo nuevo por step (mezcla teacher/student por token) y después calcula sobre él el reverse-KL normalizado por longitud. Desactivado por defecto = mezcla de distribuciones offline económica. Requiere minillm_enabled=True. (v0.71.18 #257) |
| `training.minillm_rollout_length` | `int \| null` | `null` | `>= 1`, `<= 512` | Longitud del rollout on-policy (número de tokens generados por step). Si es None, el trainer de distil deriva automáticamente min(max_length, 32) para el presupuesto de GPU de consumo. El bucle autoregresivo vuelve a procesar el prefijo creciente completo en cada step (\~O(L^2) graph), así que mantenlo pequeño. Requiere minillm_on_policy=True. (v0.71.18 #257) |
| `training.uld_strategy` | `str \| null` | `null` | uno de `"wasserstein"`, `"topk_align"`, `"wasserstein_aligned"` | Estrategia de destilación entre tokenizadores: 'wasserstein' (no requiere alineación), 'topk_align' (requiere uld_top_k) o 'wasserstein_aligned' (alineación de secuencias de tokens para tokenizadores completamente disjuntos, v0.71.18 #258). Requiere task='distill' en un backend distinto de mlx. |
| `training.uld_top_k` | `int \| null` | `null` | `>= 1`, `<= 262144` | Logits Top-K del teacher que se alinearán (solo uld_strategy='topk_align'). Acotado a \[1, 262144] para limitar vocabularios patológicos. (v0.70.0) |
| `training.reward_hack_detector` | `str \| null` | `null` | uno de `"info_rm"`, `"rm_ensemble"` | Detector de reward hacking para GRPO/PPO. 'info_rm' sigue la separación de clústeres InfoRM durante el entrenamiento; 'rm_ensemble' sigue la varianza por pares de un ensemble de RM. Requiere task en {'grpo', 'ppo'} en un backend distinto de mlx; instala el callback compartido de señales de recompensa. |
| `training.reward_hack_halt` | `bool` | `false` |  | Detiene automáticamente el entrenamiento ante un veredicto HACK (drop_pct >= 30% en la separación de clústeres). Requiere definir reward_hack_detector. (v0.70.0) |
| `training.reward_hack_mitigation` | `str` | `"off"` | uno de `"off"`, `"log_only"`, `"kl_control"`, `"pid_lagrangian"` | Modo de mitigación de reward hacking de circuito cerrado (v0.71.26). 'off' = sin mitigación; 'log_only' = instrumenta y añade un registro de mitigación sin modificar el entrenamiento; 'kl_control' = controlador KL/β bang-bang reversible; 'pid_lagrangian' = controlador PID-Lagrangiano + escalera de escalamiento con rollback. Cualquier modo distinto de 'off' requiere reward_hack_detector (la fuente de señal) y task en {'grpo', 'ppo'} en un backend distinto de mlx. |
| `training.reward_hack_beta_floor` | `float` | `0.02` | `> 0.0` | v0.71.26 — límite inferior de β/kl_coef para el controlador de mitigación. Debe ser > 0 (β=0 desactiva la ruta ref-log-prob durante la generación). |
| `training.reward_hack_beta_ceil` | `float` | `1.0` | `> 0.0`, `<= 1000.0` | v0.71.26 — límite superior de β/kl_coef para el controlador de mitigación. Debe ser > reward_hack_beta_floor. |
| `training.reward_hack_trip_band` | `float` | `0.3` | `>= 0.0`, `<= 1.0` | v0.71.26 — drop_pct de hacking a partir del cual el controlador quiere SUBIR β. Debe ser > reward_hack_release_band. |
| `training.reward_hack_release_band` | `float` | `0.1` | `>= 0.0`, `<= 1.0` | v0.71.26 — drop_pct de hacking hasta el cual el controlador quiere RELAJAR β. Debe ser &lt; reward_hack_trip_band. |
| `training.reward_hack_dwell_steps` | `int` | `2` | `>= 1`, `<= 100000` | v0.71.26 — cantidad de steps consecutivos en la banda de activación necesarios antes de que el controlador suba β (histéresis). |
| `training.reward_hack_release_patience` | `int` | `3` | `>= 1`, `<= 100000` | v0.71.26 — cantidad de steps consecutivos en la banda de liberación necesarios antes de que el controlador relaje β. |
| `training.reward_hack_kl_gain` | `float` | `1.5` | `> 1.0`, `<= 100.0` | v0.71.26 — incremento multiplicativo de β por step de activación (>1). β se multiplica al activarse / se divide al liberarse y se limita a \[floor, ceil]. |
| `training.reward_hack_signals` | `list[str]` | `["info_rm"]` | max length 4 | v0.71.26 — señales combinadas en el voto multiseñal del controlador. Lista permitida (máx. 4): info_rm, rm_ensemble, length_trend, repetition. |
| `training.reward_hack_pid_kp` | `float` | `0.5` | `>= 0.0`, `<= 1000.0` | v0.71.26 — ganancia proporcional PID (modo pid_lagrangian). |
| `training.reward_hack_pid_ki` | `float` | `0.1` | `>= 0.0`, `<= 1000.0` | v0.71.26 — ganancia integral PID (modo pid_lagrangian). |
| `training.reward_hack_pid_kd` | `float` | `0.05` | `>= 0.0`, `<= 1000.0` | v0.71.26 — ganancia derivativa PID (modo pid_lagrangian). |
| `training.reward_hack_signal_target` | `float` | `0.15` | `>= 0.0`, `< 1.0` | v0.71.26 — drop_pct de hacking objetivo que mantiene el controlador PID (modo pid_lagrangian). |
| `training.reward_hack_integral_clamp` | `float` | `1.0` | `> 0.0`, `<= 1000.0` | v0.71.26 — límite anti-windup PID del acumulador integral. Independiente de beta_ceil. |
| `training.reward_hack_rollback` | `bool` | `false` |  | v0.71.26 — activa el rollback al último checkpoint RL correcto dentro de la escalera de escalamiento. Requiere rl_checkpoint_save_every_steps definido. |
| `training.reward_hack_rollback_patience` | `int` | `3` | `>= 1`, `<= 100000` | v0.71.26 — cantidad de steps HACK consecutivos antes de activar un rollback. |
| `training.reward_hack_max_recovery_attempts` | `int` | `2` | `>= 0`, `<= 1000` | v0.71.26 — cantidad máxima de rollbacks antes de que el controlador detenga el entrenamiento anticipadamente (último peldaño de la escalera de escalamiento). |
| `training.reward_hack_signal_smoothing` | `str` | `"none"` | uno de `"none"`, `"ema"`, `"median"` | v0.71.26 — suavizado por señal antes del voto del controlador: 'none', 'ema' (0.5·prev+0.5·new) o 'median' sobre una ventana. |
| `training.reward_hack_smoothing_window` | `int` | `8` | `>= 2`, `<= 256` | v0.71.26 — longitud de la ventana para suavizar señales. |
| `training.reward_hack_conservative_on_disagreement` | `bool` | `false` |  | v0.71.26 — si los detectores discrepan, mantiene alto KL (usa la señal MAX) en vez de relajarlo. |
| `training.reward_hack_reward_shaping` | `bool` | `false` |  | v0.71.26 — aplica una penalización acotada al proxy manipulado (longitud/repetición/sentinel) mediante un adaptador de shaping sobre la función de recompensa. Requiere reward_hack_shaping_strength > 0 y un modo de control. |
| `training.reward_hack_shaping_kind` | `str` | `"length"` | uno de `"length"`, `"repetition"`, `"sentinel"` | v0.71.26 — indica qué proxy manipulado penaliza el adaptador de reward-shaping. |
| `training.reward_hack_shaping_strength` | `float` | `0.0` | `>= 0.0`, `<= 1.0` | v0.71.26 — magnitud de la penalización acotada de reward-shaping \[0, 1]. |
| `training.ppo_epochs` | `int` | `4` | `>= 1` | Número de épocas de optimización PPO por batch |
| `training.ppo_clip_ratio` | `float` | `0.2` | `> 0`, `<= 1.0` | Rango de clipping PPO para el ratio de policy |
| `training.ppo_kl_penalty` | `float` | `0.05` | `>= 0` | Coeficiente de penalización por divergencia KL para PPO |
| `training.reward_model` | `str \| null` | `null` |  | Ruta o ID de HF de un modelo de recompensa entrenado para PPO |
| `training.loraplus_lr_ratio` | `float \| null` | `null` | `> 0` | Ratio lr de LoRA+: lr_B = lr × ratio. None = desactivado (LoRA estándar). |
| `training.use_lorafa` | `bool` | `false` |  | Activa el optimizador LoRA-FA (Frozen-A LoRA): congela las matrices LoRA A para reducir la memoria de activaciones |
| `training.use_galore` | `bool` | `false` |  | Activa GaLore (Gradient Low-Rank Projection) para entrenamiento eficiente en memoria |
| `training.galore_rank` | `int` | `128` | `>= 1` | Rango de proyección de GaLore |
| `training.galore_update_proj_gap` | `int` | `200` | `>= 1` | Intervalo de actualización de la proyección GaLore (steps) |
| `training.galore_scale` | `float` | `0.25` | `> 0` | Factor de escala del gradiente GaLore |
| `training.moe_lora` | `bool` | `false` |  | Activa LoRA con reconocimiento de MoE (ScatterMoE): aplica LoRA a las capas FFN de los expertos |
| `training.moe_aux_loss_coeff` | `float` | `0.01` | `>= 0` | Coeficiente de loss auxiliar de balanceo de carga para modelos MoE |
| `training.use_liger` | `bool` | `false` |  | Activa operaciones fusionadas de Liger Kernel (ahorro de memoria del 20-60%, aceleración del 20-40%) |
| `training.use_flash_attn` | `bool` | `false` |  | Activa FlashAttention (detecta automáticamente v2/v3/v4 para acelerar la atención) |
| `training.use_ring_attention` | `bool` | `false` |  | Activa Ring FlashAttention para paralelismo de secuencias entre GPUs |
| `training.rope_scaling_type` | `str \| null` | `null` | uno de `"linear"`, `"dynamic"`, `"yarn"`, `"llama3"` | Método de escalado RoPE para contexto largo: linear, dynamic, yarn o llama3 (v0.49.0). 'longrope' se rechaza al cargar la configuración (#1239). Lo aplican las tareas sft, tts y pretrain en backend transformers (sft: la ruta de texto sin layer streaming); se rechaza al cargar la configuración en cualquier otro caso (#1697). |
| `training.yarn_factor` | `float \| null` | `null` | `> 1.0`, `<= 1024.0` | Factor de escalado YaRN (s). Opcional: si se omite, el runtime recurre a ``target_length / original_length`` (comportamiento por defecto de HF). |
| `training.yarn_attn_factor` | `float \| null` | `null` | `>= 0.0`, `<= 10.0` | Multiplicador de temperatura de atención YaRN (por defecto 1.0 en HF). |
| `training.yarn_beta_fast` | `int \| null` | `null` | `>= 1`, `<= 1024` | Límite beta_fast de YaRN (por defecto de HF: 32). |
| `training.yarn_beta_slow` | `int \| null` | `null` | `>= 1`, `<= 1024` | Límite beta_slow de YaRN (por defecto de HF: 1). |
| `training.use_longlora` | `bool` | `false` |  | Rechazado al cargar la configuración (#1240): la anulación de LongLoRA rotaba las proyecciones Q/K de la mitad de las cabezas con wrap-around bajo atención causal completa, filtrando tokens futuros, y no aplicaba atención agrupada S². Usa rope_scaling_type con LoRA simple para ampliar el contexto. |
| `training.gradient_checkpointing` | `bool \| "selective" \| "medium" \| "full" \| "auto"` | `false` |  | Checkpointing de gradientes para ahorrar memoria en secuencias largas. False/True (bool heredado) o nivel: 'selective' (solo atención), 'medium' (bloques alternos), 'full' (todos los bloques), 'auto' (elige según la VRAM disponible). (v0.28.0). |
| `training.use_cut_ce` | `bool` | `false` |  | Activa Cut Cross-Entropy (CCE) para modelos con vocabulario grande. Ahorra 8-24GB de VRAM en vocabulario Llama 3.1 de 128k. Requiere cut_cross_entropy. Incompatible con backends Unsloth/MLX. |
| `training.kernel_auto_compose` | `bool` | `false` |  | Campo de compatibilidad no admitido. Define use_liger o use_flash_attn explícitamente, o ambos. |
| `training.packing_cross_doc_attn_mask` | `bool` | `false` |  | Nunca funcionó: Soup asignaba esto a packing_strategy='attention_free', que no está en la lista permitida de TRL (bfd / bfd-requeue / wrapped) en ninguna versión publicada de trl. En su lugar, usa packing: true con FlashAttention. (v0.28.0 / #691 / #709). |
| `training.activation_offloading` | `str \| null` | `null` | uno de `"cpu"`, `"disk"` | Descarga activaciones a CPU o disco durante el paso backward. None=desactivado, 'cpu'=descarga a RAM, 'disk'=descarga a un archivo temporal. (v0.28.0). |
| `training.embedding_loss` | `str` | `"contrastive"` | uno de `"contrastive"`, `"triplet"`, `"cosine"` | Función de loss para entrenamiento de embeddings: contrastive, triplet o cosine |
| `training.embedding_margin` | `float` | `0.5` | `> 0` | Margen para loss contrastive/triplet (un valor mayor implica una separación más estricta) |
| `training.embedding_pooling` | `str` | `"mean"` | uno de `"mean"`, `"cls"`, `"last"` | Estrategia de pooling para embeddings de oraciones: mean, cls o último token |
| `training.embedding_temperature` | `float` | `0.05` | `> 0` | Temperatura para loss contrastive (InfoNCE): un valor menor implica una similitud más estricta |
| `training.curriculum` | `bool` | `false` |  | Activa el aprendizaje curricular (ordena el dataset por dificultad, de fácil a difícil) |
| `training.curriculum_metric` | `str` | `"length"` | uno de `"length"`, `"perplexity"`, `"loss"` | Métrica de dificultad del currículo: longitud, perplexity o loss |
| `training.curriculum_buckets` | `int` | `4` | `>= 1`, `<= 20` | Número de etapas de dificultad del aprendizaje curricular |
| `training.curriculum_dynamic` | `bool` | `false` |  | BETA: repondera dinámicamente los buckets del currículo cada N steps mediante estimación online de incertidumbre (loss por muestra + norma del gradiente). Requiere curriculum=true. Los lanzamientos multirrango deben conectar un hook all_reduce con las estadísticas por bucket (consulta utils.curriculum_dynamic.validate_distributed_curriculum). |
| `training.curriculum_dynamic_recompute_steps` | `int` | `50` | `>= 1`, `<= 100000` | Recalcula los pesos del sampler de buckets del currículo cada N steps globales de entrenamiento. |
| `training.curriculum_dynamic_floor` | `float` | `0.05` | `> 0.0`, `<= 0.5` | Peso normalizado mínimo por bucket después del softmax. Debe estar en (0.0, 1/curriculum_buckets]; el validador cruzado lo ajusta al techo por configuración. Evita que los buckets queden sin muestras. |
| `training.curriculum_dynamic_temperature` | `float` | `1.0` | `> 0.0`, `<= 100.0` | Temperatura del softmax sobre la señal de incertidumbre. Un valor mayor produce una distribución más plana; uno menor concentra el peso en los buckets más difíciles. |
| `training.loss_watchdog` | `bool` | `false` |  | Activa la detección de picos de loss (detención automática si la loss supera el umbral) |
| `training.loss_watchdog_threshold` | `float` | `3.0` | `> 0`, `<= 100.0` | Detiene el entrenamiento si la loss supera este umbral |
| `training.loss_watchdog_patience` | `int` | `5` | `>= 1`, `<= 1000` | Número de steps consecutivos con loss alta antes de detenerse |
| `training.rewind_log` | `bool` | `true` |  | Escribe &lt;output>/rewind.jsonl: las filas del dataset en cada microbatch de entrenamiento con la loss de cada fila; `soup rewind` lo lee para identificar las filas detrás de un pico de loss. Solo SFT (transformers y MLX, proceso único); las ejecuciones reanudadas y el packing lo desactivan con una advertencia. |
| `training.loss_spike_recovery` | `bool` | `false` |  | Ante la activación del watchdog: escribe &lt;output>/spike_recovery.json con LR reducida y el contador de intentos para relanzar (en lugar de detenerse sin más). Requiere loss_watchdog=true. |
| `training.loss_spike_recovery_max_attempts` | `int` | `3` | `>= 1`, `<= 10` | Número máximo de intentos de recuperación de picos antes de abandonar |
| `training.loss_spike_recovery_lr_decay` | `float` | `0.5` | `> 0.0`, `< 1.0` | Multiplica la LR por este factor en cada recuperación de pico (0.5 = reduce a la mitad) |
| `training.relora_steps` | `int \| null` | `null` | `>= 1`, `<= 10000000` | Ejecuta un reinicio ReLoRA fiel al artículo cada N steps globales: fusiona la actualización LoRA en el peso base, reinicializa lora_A/lora_B y, opcionalmente, limpia el estado del optimizador. None lo desactiva. Requiere LoRA de precisión completa (training.quantization por defecto es '4bit' y debe definirse como 'none'); incompatible con VeRA, DoRA, layer streaming y compilación FSDP2. |
| `training.relora_warmup_ratio` | `float` | `0.1` | `>= 0.0`, `<= 1.0` | Omite las activaciones de ReLoRA durante la primera fracción warmup_ratio del entrenamiento |
| `training.relora_reset_optimizer` | `bool` | `true` |  | Limpia el estado del optimizador para los parámetros lora_A/lora_B después de cada reinicio de fusión+reinicialización ReLoRA |
| `training.relora_prune_ratio` | `float` | `0.9` | `> 0.0`, `< 1.0` | Obsoleto: se conserva para que sigan cargando los YAML antiguos. Los reinicios ya no podan por magnitud los pesos del adaptador; relora_reset_optimizer controla la limpieza del optimizador. |
| `training.convergence_detection` | `bool` | `false` |  | Busca mesetas u oscilaciones de loss y muestra recomendaciones (continue / early_stop / lower_lr) al final del entrenamiento. |
| `training.convergence_window` | `int` | `50` | `>= 5`, `<= 10000` | Preparado, pero no se aplica durante el entrenamiento: número de losses recientes que se examinan para detectar meseta u oscilación |
| `training.convergence_rel_tol` | `float` | `0.005` | `> 0.0`, `<= 1.0` | Preparado, pero no se aplica durante el entrenamiento: umbral de rango relativo por debajo del cual la ventana se considera una meseta |
| `training.warmup_auto` | `bool` | `false` |  | Elige automáticamente warmup_steps a partir de dataset_size × epochs × warmup_ratio. Anula cualquier warmup_steps manual en el trainer. |
| `training.auto_mixed_precision` | `bool` | `false` |  | Elige bf16/fp16 según el modelo y la capacidad de cómputo de la GPU. Se aplica a task='sft' (y 'tts', que entrena mediante el trainer SFT); se rechaza en todas las demás tareas, cuyos trainers eligen su propia precisión. |
| `training.grad_accum_auto_tune` | `bool` | `false` |  | Supervisa la VRAM en cada step; advierte (y recomienda nuevos valores de batch/accum) cuando la presión de memoria es alta. Solo informativo: no reconstruye el DataLoader ni cambia la acumulación durante una ejecución. |
| `training.grad_accum_pressure_threshold` | `float` | `0.92` | `> 0.05`, `< 0.99` | Fracción de utilización de VRAM que activa una recomendación |
| `training.freeze_layers` | `int \| null` | `null` | `>= 1`, `<= 1000` | Congela las primeras N capas (desde abajo). Entrena solo las capas restantes. Se aplica a las tareas sft y tts en la ruta de texto con backend transformers y sin layer streaming; se rechaza en los demás casos. |
| `training.freeze_ratio` | `float \| null` | `null` | `> 0.0`, `< 1.0` | Congela esta fracción de capas (0.75 = congela el 75% desde abajo). Se aplica a las tareas sft y tts en la ruta de texto con backend transformers y sin layer streaming; se rechaza en los demás casos. |
| `training.unfrozen_parameters` | `list[str] \| null` | `null` |  | Entrenamiento dirigido Spectrum (#266): patrones regex de nombres de parámetros que deben seguir entrenables; todos los demás se congelan. Ajuste fino completo, LoRA desactivado. Genera los patrones con `soup spectrum scan`. Incompatible con funciones LoRA / freeze_layers / freeze_ratio / train_router_only; solo sft + backend transformers. |
| `training.lisa_enabled` | `bool` | `false` |  | Activa el muestreo de importancia por capas LISA (#267): cada lisa_interval_steps, congela todas las capas decoder salvo un conjunto aleatorio de lisa_num_layers (embeddings + head siempre entrenables). Ajuste fino completo, LoRA desactivado. Solo sft o pretrain + transformers + text + quantization=none; incompatible con funciones LoRA / freeze_layers / freeze_ratio / unfrozen_parameters. |
| `training.lisa_num_layers` | `int` | `2` | `>= 1`, `<= 64` | LISA: número de capas decoder que permanecen entrenables por intervalo (limitado al número de capas del modelo). Un valor pequeño usa memoria similar a LoRA. |
| `training.lisa_interval_steps` | `int` | `20` | `>= 1`, `<= 1000000` | LISA: vuelve a muestrear las capas decoder activas cada N steps globales. |
| `training.lisa_reset_optimizer` | `bool` | `true` |  | Limpia el estado del optimizador de las capas decoder que LISA vuelve a congelar en cada intervalo (evita momentos Adam obsoletos). Es equivalente a relora_reset_optimizer. |
| `training.lisa_train_embeddings` | `bool` | `true` |  | Mantiene entrenable en cada intervalo LISA el grupo siempre activo (embeddings de entrada + LM head + norma final). El valor true por defecto corresponde a LISA según la publicación (#267). Define false para congelar ese grupo y entrenar solo las lisa_num_layers capas decoder muestreadas: el grupo siempre activo representa la mayor parte de la memoria de LISA, así que este ajuste es el que permite ahorrar memoria (#377). Es un intercambio real: el grupo siempre activo puede ser decisivo para la calidad, así que mide ambas opciones. |
| `training.stream_layers` | `bool` | `false` |  | BETA: transmite la capa base congelada desde la RAM de CPU capa por capa para poder ajustar un modelo mayor que la VRAM. Solo sft + transformers + text + quantization=none; batch_size 1, sin acumulación de gradientes. Es más lento que el entrenamiento residente, pero estos modelos ni siquiera podían ejecutarse en la tarjeta. |
| `training.stream_source` | `str` | `"auto"` | uno de `"auto"`, `"ram"`, `"disk"` | Ubicación de la base transmitida. 'ram' (el único nivel implementado en v0.72.0) mantiene la base en la RAM de CPU; 'disk' es el nivel de desbordamiento de v0.72.3; 'auto' elige RAM solo cuando el almacén cabe en la memoria RAM libre y el almacén más los extras residentes caben en el límite físico de RAM. |
| `training.stream_ngram_source` | `str` | `"auto"` | uno de `"auto"`, `"ram"`, `"disk"` | Ubicación del embedding N-gram PLE congelado de Qwen4-Exp durante el layer streaming. 'disk' reúne solo las filas solicitadas desde los safetensors originales mediante mmap de solo lectura; 'ram' mantiene la tabla en RAM de CPU; 'auto' usa RAM solo cuando la tabla y el nivel de base seleccionado más los extras residentes caben en la RAM libre y el límite físico de RAM. Las tablas PLE afines oMLX/oQ requieren 'disk' (o 'auto') para que solo se descuantifiquen las filas seleccionadas. Un valor distinto del predeterminado genera una advertencia si el checkpoint no tiene una tabla PLE externa. |
| `training.stream_buffers` | `int` | `2` | `>= 2`, `<= 8` | Búferes de capa preasignados en VRAM. 2 = doble búfer, que permite cargar la siguiente capa mientras se calcula la actual. Un solo búfer no puede superponer carga y cálculo. |
| `training.stream_read_ahead` | `int` | `2` | `>= 1`, `<= 8` | Capas que el nivel de disco asíncrono prepara por adelantado en su hilo de fondo. Siempre hay una ranura de preparación ocupada por la capa que se está usando, así que un ajuste N prepara N-1 capas por adelantado, no N: 1 = la lectura simplemente sale del hilo de cálculo (no se prepara ninguna capa adicional), 2 = una capa en vuelo mientras se usa otra, hasta 7 preparadas por adelantado con el ajuste máximo de 8. Cada nivel cuesta la memoria host fijada de una capa, que son 441 MB en un modelo 70B. |
| `training.stream_pin` | `bool \| null` | `null` |  | Fuerza a activar o desactivar la memoria host bloqueada en páginas (fijada): el almacén base del nivel RAM o la preparación host del lector asíncrono del nivel disco (#971). None (por defecto) deja que el equipo decida: intenta bloquear esa memoria en páginas y recurre a memoria paginable —anunciando el costo— si el host no puede bloquearla. 'false' fuerza el almacén paginable, la única salida conocida cuando se sospecha de una ruta de fijación (fue la única mitigación mientras #331 estuvo activo). 'true' fuerza la fijación y RECHAZA la ejecución en vez de degradar silenciosamente si el equipo no puede bloquearla: en el nivel RAM indica el tamaño del almacén; en el nivel disco indica training.stream_read_ahead como profundidad que determina cuánta preparación se bloquea. La fijación de páginas ofrece hasta 6.56x el rendimiento medido, así que una alternativa silenciosa consume todo el margen que la función pretende aportar. Los destinos distintos de CUDA (CPU o MPS) son el único caso en que la fijación no es APLICABLE: allí se anuncia 'true' y la ejecución continúa con una fuente CPU paginable, de modo que la clave se puede conservar en una configuración compartida entre equipos CUDA y no CUDA. |
| `training.stream_vram_override` | `int \| null` | `null` | `>= 0` | Bytes que se supondrán libres en lugar de medir torch.cuda.mem_get_info(). mem_get_info() es una consulta al controlador a nivel de dispositivo, así que no detecta un límite por proceso (set_per_process_memory_fraction, una partición MIG o una tarjeta que también usa otro proceso); en ese hardware, la comprobación previa ve toda la VRAM libre del dispositivo, independientemente de lo que el proceso pueda usar realmente. Define este valor para reemplazar la cifra medida en cualquier dirección: súbelo para permitir una sobreestimación que se sabe segura o bájalo para que la comprobación previa aplique un límite que el controlador no puede informar. |
| `training.stream_vram_probe` | `bool` | `false` |  | Decide la comprobación previa de VRAM de layer streaming mediante una MEDICIÓN en lugar de la fórmula ajustada. Tras construir el modelo transmitido, ejecuta un forward+backward real con la forma configurada y su pico determina si la ejecución continúa; la predicción se imprime al lado para que cualquier divergencia sea visible. La fórmula subestima a partir de seq 4352 (mide 0.934x del pico real en seq 5120 y 0.787x en 6144), que es la dirección que no avisa: OOM en Linux y desbordamiento silencioso a memoria host en Windows. Frente a la prueba, la misma fórmula marca 0.992x en seq 4096 y 0.830x en 5120, porque la prueba queda un 12.5-14.3% por encima del step de entrenamiento real, la dirección que permite usarla como control seguro. Desactivada por defecto: cuesta un step (medido en 1-5 s) y puede rechazar una ejecución que la fórmula aceptaría. Solo task='sft': la prueba ejecuta forward+backward de un modelo causal de lenguaje simple, que SÍ es el step de SFT, pero no una loss de preferencias. En la única forma de preferencias medida marca un 13.5% por encima (prueba 6.021 GB frente a un step DPO real de 5.304 GB), la misma dirección segura que muestra en SFT; pero una sola forma no es una validación, así que la restricción se mantiene hasta medirla con varias formas. |
| `training.stream_disk_kind` | `str \| null` | `null` | uno de `"nvme"`, `"ssd"`, `"hdd"` | Anula el tipo de medio de disco detectado automáticamente para el nivel de desbordamiento de streaming a disco. La detección clasifica el dispositivo mediante una lectura secuencial medida cuando la marca de rotación del kernel no es confiable: virtio y otros discos paravirtuales indican que giran sin información del medio, por lo que de otro modo se rechaza un disco rápido de nube (#365). Define esto si incluso esa detección es incorrecta: 'nvme' fuerza la activación del nivel de disco; 'ssd'/'hdd' lo desactivan. El valor resuelto se imprime junto al detectado. |
| `training.packing` | `bool` | `false` |  | Empaqueta varias muestras cortas en una secuencia para acelerar el entrenamiento |
| `training.multipack` | `bool` | `false` |  | Usa un sampler FFD de empaquetado de bins para maximizar los tokens por batch en datos de longitudes desiguales. Incompatible con packing. Solo compatible con tareas sft / pretrain (backend transformers). (v0.37.0). |
| `training.neftune_alpha` | `float \| null` | `null` | `>= 0.0`, `<= 50.0` | Alpha de ruido NEFTune (0-50). Añade ruido a los embeddings para mejorar la calidad del chat. |
| `training.forgetting_detection` | `bool` | `false` |  | Preparado, pero no se aplica durante el entrenamiento: activa una evaluación periódica de conocimiento general para detectar olvido catastrófico |
| `training.forgetting_eval_steps` | `int` | `100` | `>= 10`, `<= 10000` | Preparado, pero no se aplica durante el entrenamiento: ejecuta una evaluación de olvido cada N steps |
| `training.forgetting_threshold` | `float` | `0.1` | `>= 0.01`, `<= 0.5` | Preparado, pero no se aplica durante el entrenamiento: advierte si la precisión cae más que este umbral (0.01-0.50) |
| `training.forgetting_benchmark` | `str` | `"mini_mmlu"` | uno de `"mini_mmlu"`, `"mini_common_sense"`, `"mini_instruction"` | Preparado, pero no se aplica durante el entrenamiento: mini benchmark integrado para detectar olvido |
| `training.forgetting_stop` | `bool` | `false` |  | Preparado, pero no se aplica durante el entrenamiento: detiene la ejecución ante un olvido grave |
| `training.checkpoint_intelligence` | `bool` | `false` |  | Preparado, pero no se aplica durante el entrenamiento: registra los mejores checkpoints según su calidad |
| `training.checkpoint_eval_steps` | `int` | `200` | `>= 50`, `<= 10000` | Preparado, pero no se aplica durante el entrenamiento: evalúa la calidad del checkpoint |
| `training.checkpoint_eval_metric` | `str` | `"composite"` | uno de `"judge"`, `"mmlu"`, `"custom"`, `"composite"` | Preparado, pero no se aplica durante el entrenamiento: métrica de calidad del checkpoint |
| `training.checkpoint_eval_tasks` | `str \| null` | `null` |  | Preparado, pero no se aplica durante el entrenamiento: tareas JSONL para puntuar checkpoints |
| `training.checkpoint_keep_top` | `int` | `3` | `>= 1`, `<= 20` | Preparado, pero no se aplica durante el entrenamiento: conserva los N mejores checkpoints de calidad |
| `training.early_stop_on_regression` | `bool` | `false` |  | Preparado, pero no se aplica durante el entrenamiento: se detiene tras regresiones consecutivas |
| `training.early_stop_patience` | `int` | `2` | `>= 1`, `<= 10` | Preparado, pero no se aplica durante el entrenamiento: regresiones antes de detenerse (1-10) |
| `training.eval_gate` | `section \| null` | `null` |  | Configuración opcional EvalGateConfig: bloquea el entrenamiento ante regresiones. Consulta [`training.eval_gate`](#trainingeval_gate). |
| `training.use_fsdp2_compile` | `bool` | `false` |  | Activa torch.compile sobre FSDP2 para acelerar el entrenamiento un +20-30%. Requiere --fsdp, CUDA y backend=transformers. |
| `training.parallelism` | `str` | `"data"` | uno de `"data"`, `"pipeline"` | Estrategia distribuida: 'data' (DDP/FSDP/DeepSpeed) o 'pipeline' (paralelismo de pipeline, solo conexión de v0.27.0). |
| `training.pipeline_stages` | `int` | `1` | `>= 1`, `<= 16` | Número de etapas de paralelismo de pipeline. Se ignora cuando parallelism='data'. |
| `training.unlearn_method` | `str \| null` | `null` | uno de `"npo"`, `"simnpo"`, `"rmu"` | Backend del método de desaprendizaje; obligatorio cuando task='unlearn'. npo (Negative Preference Optimization, loss negativa solo de preferencias con forma DPO); simnpo (NPO normalizado por longitud sin modelo de referencia); rmu (Representation Misdirection Unlearning, ruido en el flujo residual). |
| `training.unlearn_alpha` | `float \| null` | `null` |  | Ponderación del conjunto retenido en la loss de desaprendizaje (coeficiente de mezcla entre olvidar y retener). 0.0 = solo loss de olvido; los valores mayores favorecen progresivamente el conjunto retenido. Acotado a \[0.0, 10.0]. (v0.61.0) |
| `training.ra_dit_stage` | `str \| null` | `null` | uno de `"retriever"`, `"generator"` | Etapa del pipeline RA-DIT. 'retriever' entrena el sentence-transformer mediante el trainer de embedding v0.16; 'generator' ejecuta SFT al estilo RAFT con `data.format='raft'`. Se combina con la receta RAFT de v0.62.0. (v0.62.0) |
| `training.ra_dit_retriever_model` | `str \| null` | `null` |  | ID opcional del modelo retriever (p. ej. `sentence-transformers/all-mpnet-base-v2`) que la etapa generator usa para codificar previamente los documentos distractores. (v0.62.0) |
| `training.citation_faithful` | `bool` | `false` |  | Activa la puntuación de precisión / recall de citas y una regla de máscara de loss que enfatiza los fragmentos de cita. Requiere `data.format='raft'`; los trainers SFT y RAFT aplican la máscara ponderada de fragmentos de cita. |
| `training.citation_style` | `str \| null` | `null` | uno de `"bracket"`, `"inline"`, `"footnote"` | Estilo de representación de citas. 'bracket' = etiqueta en línea `[doc-1]` (valor canónico por defecto de RAFT), 'inline' = `(doc-1)` y 'footnote' = `[^doc-1]`. Se usa para puntuar citas y enmascarar fragmentos. |
| `training.citation_recall_threshold` | `float \| null` | `null` |  | Preparado, no aplicado: umbral de recall de citas en \[0.0, 1.0] que no controla el guardado ni la ejecución; al definirlo se muestra una advertencia al cargar y luego se rechaza (#761). (v0.62.0) |
| `training.grace_codebook` | `bool` | `false` |  | Reserva la configuración del codebook GRACE para entrenamiento. Solo se valida la compatibilidad: ningún trainer ni comando de edición de conocimiento consume todavía este campo. |
| `training.grace_codebook_size` | `int \| null` | `null` |  | Cantidad reservada de entradas del codebook GRACE. Obligatoria cuando grace_codebook=True y acotada a \[1, 100_000], pero aún no se consume en runtime. |
| `training.grace_codebook_dim` | `int \| null` | `null` |  | Dimensión reservada de las entradas del codebook GRACE. Obligatoria cuando grace_codebook=True y acotada a \[1, 16_384], pero aún no se consume en runtime. |

## `training.lora`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `training.lora.r` | `int` | `64` | `>= 0` | Rango LoRA. 0 = ajuste fino completo: sin adaptador, se entrena cada parámetro base (solo sft / embedding + transformers + text + quantization='none'). |
| `training.lora.alpha` | `int` | `16` |  | Alpha de LoRA |
| `training.lora.dropout` | `float` | `0.05` |  | Dropout de LoRA |
| `training.lora.target_modules` | `str \| list[str]` | `"auto"` |  | Módulos de destino para LoRA. 'auto' = dejar que peft decida. |
| `training.lora.target_parameters` | `"auto" \| list[str] \| null` | `null` |  | Tensores nn.Parameter 2-D/3-D sin procesar que se adaptarán con PEFT LoRA. Usa 'auto' para seleccionar parámetros de destino específicos de la arquitectura (actualmente expertos enrutados de Qwen4-Exp), una lista de sufijos de nombres de parámetros para control explícito u omítelo para desactivar. Requiere dropout=0 y está conectado para SFT/pretrain de transformers. |
| `training.lora.use_dora` | `bool` | `false` |  | Activa DoRA (Weight-Decomposed Low-Rank Adaptation) |
| `training.lora.use_rslora` | `bool` | `false` |  | Activa el escalado LoRA estabilizado por rango (mejor para rangos altos) |
| `training.lora.use_vera` | `bool` | `false` |  | Activa VeRA (Vector-based Random Matrix Adaptation). Las matrices aleatorias compartidas consumen mucha menos memoria que LoRA. Incompatible con use_dora y use_olora. |
| `training.lora.use_olora` | `bool` | `false` |  | Activa OLoRA (inicialización LoRA ortogonal mediante descomposición QR). Pasa init_lora_weights='olora' a peft. Incompatible con use_dora y use_vera. Equivale a init_strategy='olora'. |
| `training.lora.rank_pattern` | `dict[str, int] \| null` | `null` |  | Anulación del rango LoRA por patrón de módulo de destino. Asigna patrones de nombres de módulos (p. ej. 'q_proj', 'experts.\*.w1') a valores enteros de rango. Útil en configuraciones MoE donde las FFN de expertos necesitan menor rango que attn. Incompatible con use_vera (VeRA comparte un rango entre módulos). |
| `training.lora.alpha_pattern` | `dict[str, int] \| null` | `null` |  | Anulación de alpha LoRA por patrón de módulo de destino. Asigna patrones de nombres de módulos a valores enteros de alpha. Se combina con rank_pattern. Incompatible con use_vera. |
| `training.lora.init_strategy` | `str` | `"random"` | uno de `"random"`, `"pissa"`, `"olora"`, `"loftq"` | Estrategia de inicialización LoRA. 'random' (por defecto) usa la inicialización Kaiming estándar. 'pissa' (PiSSA) inicializa A/B a partir del SVD del peso base: converge más rápido al principio, pero añade un paso SVD en la primera época. 'olora' equivale a use_olora=True (inicialización QR ortogonal). 'loftq' (v0.41.0) inicializa A/B y una base de pocos bits conjuntamente; útil con QLoRA. No se puede combinar con use_dora ni use_vera. |
| `training.lora.loftq_iter` | `int` | `1` | `>= 1`, `<= 10` | Cantidad de iteraciones LoftQ (1-10). Un valor mayor mejora la inicialización consciente de la cuantización a costa de latencia de preparación única. Solo se usa cuando init_strategy='loftq'. |
| `training.lora.loftq_bits` | `int` | `4` | uno de `2`, `4`, `8` | Ancho de bits objetivo de LoftQ: debe ser uno de {2, 4, 8}. Solo se usa cuando init_strategy='loftq'. |

## `training.eval_gate`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `training.eval_gate.enabled` | `bool` | `false` |  | Activa la puerta de evaluación |
| `training.eval_gate.suite` | `str \| null` | `null` |  | Ruta al YAML del conjunto de evaluación (evals/gate.yaml) |
| `training.eval_gate.every_n_epochs` | `int` | `1` | `>= 1`, `<= 100` | Ejecuta la puerta cada N épocas (1-100) |
| `training.eval_gate.regression_threshold` | `float` | `0.05` | `>= 0.0`, `<= 1.0` | Caída absoluta máxima frente a la línea base antes de declarar regresión |
| `training.eval_gate.baseline` | `str \| null` | `null` |  | registry://&lt;id> \| 'previous' \| ruta de archivo: puntuaciones con las que comparar |
| `training.eval_gate.on_regression` | `str` | `"stop"` | uno de `"stop"`, `"warn"`, `"continue"` | Acción ante una regresión: detener el entrenamiento \| solo advertir \| continuar |

## `eval`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `eval.auto_eval` | `bool` | `false` |  | Ejecuta la evaluación automáticamente cuando termina el entrenamiento |
| `eval.benchmarks` | `list[str] \| null` | `null` |  | Nombres de benchmarks de lm-evaluation-harness que se ejecutarán |
| `eval.custom_tasks` | `str \| null` | `null` |  | Ruta al archivo JSONL de evaluación personalizada |
| `eval.judge` | `dict \| null` | `null` |  | Configuración de LLM-as-a-judge: modelo, rúbrica, proveedor |
| `eval.ship` | `section \| null` | `null` |  | Valores por defecto del veredicto de soup ship (task_eval / task_mode / general_suite / forgetting_threshold / judge_model / baseline / noise_floor). Consulta [`eval.ship`](#evalship). |

## `eval.ship`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `eval.ship.task_eval` | `str \| null` | `null` |  | JSONL de tareas de evaluación de éxito de la etapa 1 |
| `eval.ship.task_mode` | `str` | `"metric"` | uno de `"metric"`, `"judge_score"`, `"pairwise"` | Modo de etapa 1: metric \| judge_score \| pairwise (tasa de victorias del juez) |
| `eval.ship.general_suite` | `str \| null` | `null` |  | Lista separada por comas de benchmarks de etapa 2 (por defecto: conjunto offline incluido) |
| `eval.ship.forgetting_threshold` | `float` | `0.05` | `>= 0.0`, `<= 1.0` | Caída máxima permitida en puntos absolutos en la etapa 2 antes de NO PUBLICAR |
| `eval.ship.judge_model` | `str \| null` | `null` |  | URL del modelo juez para task_mode judge_score / pairwise |
| `eval.ship.baseline` | `str \| null` | `null` |  | registry://&lt;id> \| JSON de línea base con sello ({scores, provenance.scorer_revision}) de las puntuaciones base de etapa 2 |
| `eval.ship.noise_floor` | `int \| null` | `null` | `>= 2`, `<= 10` | Repeticiones de la ejecución BASE usadas para medir el piso de ruido de la etapa 2; una caída dentro de ese piso se considera ruido, no regresión. Acotado a \[2, 10]. Entrada en vivo: se mide al generar evidencia y se rechaza con --evidence. |

## `advise`

| Clave | Tipo | Por defecto | Restricciones | Descripción |
|---|---|---|---|---|
| `advise.goal` | `str \| null` | `null` | max length 4096 | Cadena de objetivo por defecto para `soup advise`. Si se define, mejora la clasificación de tareas. |
| `advise.probe` | `bool` | `false` |  | Ejecuta por defecto la prueba ROI de 10 minutos cuando se invoca `soup advise` mediante esta configuración. Stubs heurísticos en v0.54.0. |
| `advise.record` | `bool` | `false` |  | Añade cada veredicto de esta configuración a \~/.soup/advise_history.jsonl con accepted=True. |
