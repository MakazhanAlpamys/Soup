# PEFT, contexto largo y eficiencia de entrenamiento

> 🇪🇸 Traducción al español de [`peft-and-efficiency.md`](peft-and-efficiency.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> DoRA/LoRA+/rsLoRA/VeRA/OLoRA/NEFTune, PiSSA/ReLoRA, el zoo de optimizadores y PEFT, LLaMA Pro, GaLore, contexto largo con YaRN/LongLoRA, empaquetado, currículo, congelamiento, watchdog de loss y ajuste automático.

**Contenido:**

- [Anulación del forward de LongLoRA](#anulación-del-forward-de-longlora)
- [Multipack — muestreador de bin-packing FFD](#multipack--muestreador-de-bin-packing-ffd)
- [Contexto largo — YaRN, Llama 3.1 NTK, LongLoRA](#contexto-largo--yarn-llama-31-ntk-longlora)
- [Expansión de bloques de LLaMA Pro](#expansión-de-bloques-de-llama-pro)
- [Zoo de optimizadores y PEFT](#zoo-de-optimizadores-y-peft)
- [Calidad de LoRA — PiSSA, ReLoRA, rango por patrón, parches quirúrgicos](#calidad-de-lora--pissa-relora-rango-por-patrón-parches-quirúrgicos)
- [DoRA (LoRA con descomposición de pesos)](#dora-lora-con-descomposición-de-pesos)
- [LoRA+ (tasas de aprendizaje diferenciadas)](#lora-tasas-de-aprendizaje-diferenciadas)
- [rsLoRA (escalado estabilizado por rango)](#rslora-escalado-estabilizado-por-rango)
- [VeRA y OLoRA (PEFT de menor consumo)](#vera-y-olora-peft-de-menor-consumo)
- [NEFTune (fine-tuning con embeddings ruidosos)](#neftune-fine-tuning-con-embeddings-ruidosos)
- [Empaquetado de muestras](#empaquetado-de-muestras)
- [Aprendizaje por currículo](#aprendizaje-por-currículo)
- [Congelamiento de capas](#congelamiento-de-capas)
- [Watchdog de loss](#watchdog-de-loss)
- [Estabilidad del entrenamiento y ajuste automático](#estabilidad-del-entrenamiento-y-ajuste-automático)
- [Inteligencia de entrenamiento (olvido + calidad del checkpoint)](#inteligencia-de-entrenamiento-olvido--calidad-del-checkpoint)
- [GaLore (entrenamiento de parámetros completos eficiente en memoria)](#galore-entrenamiento-de-parámetros-completos-eficiente-en-memoria)
- [Poda de profundidad + Distill-Heal (`soup shrink`)](#poda-de-profundidad--distill-heal-soup-shrink)

---

## Poda de profundidad + Distill-Heal (`soup shrink`)

`soup shrink` reduce un modelo descartando su **bloque contiguo de capas decoder** menos importante y, de forma opcional, *sana* el loss con destilación de conocimiento. Implementa "The Unreasonable Ineffectiveness of the Deeper Layers" (Gromov et al., arXiv:2403.17887), con un saneamiento por destilación estilo Minitron en lugar del simple fine-tune con LoRA del artículo.

**Cómo clasifica las capas.** Para cada bloque candidato `[L, L+n)`, un forward pass por prompt de calibración (`output_hidden_states=True`) mide la **distancia angular** del flujo residual al entrar (`hidden_states[L]`) y al salir (`hidden_states[L+n]`) del bloque, promediada sobre cada token que no sea de pad en el conjunto de calibración. El bloque con la distancia *más baja* transforma menos el flujo residual, así que es el más seguro de descartar. La primera y la última capa decoder siempre están protegidas (son las que más transforman).

```bash
# Ver la tabla de importancia + el bloque elegido sin escribir nada:
soup shrink --model HuggingFaceTB/SmolLM2-135M-Instruct --drop-ratio 0.25 \
    --calib calib.jsonl --device cpu --plan-only

# Podar 25% + sanar (destilar el original en el student podado, fusionar en un
# modelo denso) + obtener un veredicto de perplejidad SHIP/DON'T-SHIP:
soup shrink --model HuggingFaceTB/SmolLM2-135M-Instruct --drop-ratio 0.25 \
    --calib calib.jsonl --heal heal.jsonl --heal-steps 200 \
    --tolerance 0.10 -o shrunk --device cpu --attach-to-registry <id>
```

- **`--drop-ratio F` / `--drop-layers N`** (exactamente uno) — cuántas capas contiguas descartar; la *posición* la elige automáticamente el escaneo de importancia.
- **`--calib <jsonl>`** — prompts de calibración (`{"text": ...}` / `{"prompt": ...}` / `messages` de chat). Debe permanecer bajo el cwd.
- **`--heal <jsonl> --heal-steps N`** — destila el original a profundidad completa en el student podado (LoRA logit-KD) como un subproceso `soup train` aislado y luego fusiona el adaptador de vuelta en la base podada, de modo que la salida es un único modelo denso. El saneamiento mantiene al teacher residente (~2× la memoria del modelo); con `--device cpu` el saneamiento corre en CPU (validado ≤ 3 B en una tarjeta de 4 GB).
- **`--tolerance F`** — se entrega si la regresión de perplejidad se mantiene dentro de `F` (por defecto `0.10` = 10 %). Código de salida `0 = SHIP`, `2 = DON'T SHIP`, `1 = error`.
- **`-o <dir>`** — el modelo reducido queda en `<dir>/model`; el veredicto, en `<dir>/shrink_report.json`.

**Arquitecturas admitidas (v1):** Llama / Qwen / SmolLM. Las demás se rechazan con un mensaje amable. Las configuraciones MoE que ubican sus capas MoE por número de capa (un `mlp_only_layers` o `decoder_sparse_step` no predeterminado en modelos Qwen MoE, `moe_layers` o `interleave_moe_layer_step` en Llama4-text) se rechazan antes de cargar el modelo, porque una poda no puede renumerarlas. Las listas por capa, como `layer_types` y `no_rope_layers`, se recortan junto con las capas. La pasada de importancia carga el modelo, así que está validada en vivo hasta ≤ 3 B; los modelos más grandes funcionan, pero no están validados en el hardware de referencia. La perplejidad es una media no ponderada de las perplejidades por ejemplo: sirve para la *razón* antes/después que usa el veredicto, pero no es directamente comparable con los números absolutos de `soup eval`.

---

## Anulación del forward de LongLoRA

`training.use_longlora: true` **se rechaza al cargar la configuración** ([#1240](https://github.com/MakazhanAlpamys/Soup/issues/1240)). La anulación que instalaba no era atención dispersa con desplazamiento S². Hacía rodar las salidas de proyección de query/key de la mitad de las cabezas a lo largo de la secuencia con `torch.roll`, lo que envuelve los últimos `group_size // 2` tokens al frente, mientras la atención seguía siendo totalmente causal. Las posiciones anteriores podían atender claves calculadas a partir de los últimos tokens de la secuencia, así que los tokens futuros se filtraban al loss de entrenamiento, y no corría ninguna atención agrupada, por lo que tampoco ahorraba memoria ni cómputo. Una configuración que lo activaba nunca entrenaba correctamente.

Sigue rechazada hasta que exista atención S² real: primero RoPE, luego desplazar q, k y v de la mitad de las cabezas, atender de forma causal dentro de cada grupo y revertir la salida. Toda grafía que se lea como verdadero (`yes`, `on`, `1` ...) se rechaza; elimina la clave o ponla en `false`. Para extender el contexto, define `training.rope_scaling_type` (consulta [Contexto largo](#contexto-largo--yarn-llama-31-ntk-longlora)) y entrena LoRA simple.


## Multipack — muestreador de bin-packing FFD

La mayor ganancia individual de throughput de Soup en el fine-tuning de chat con datos de longitud desigual. En lugar de rellenar (padding) cada muestra hasta `max_length`, Multipack usa **bin packing First-Fit-Decreasing** para agrupar muestras de longitud variable en bins que se acercan a `batch_size × max_seq_length`, eliminando el desperdicio de padding.

```yaml
training:
  multipack: true
  packing: false   # mutuamente excluyente con multipack
```

**Cómo se compone:**
- **Multipack** elige QUÉ muestras van juntas (bin packing FFD).
- El aislamiento de documentos empaquetados es la estrategia `bfd` por defecto de TRL cuando FlashAttention es el `attn_implementation`. `packing_cross_doc_attn_mask` se rechaza al cargar la configuración (nunca se mapeó a una `packing_strategy` válida de TRL).

**Lista de arquitecturas permitidas** — 18 admitidas (Llama 3.x, Qwen 2/3, Mistral, Gemma 2/3, Phi 3/4, DeepSeek V2/V3, Mixtral, Falcon, StableLM, SmolLM2). Las arquitecturas desconocidas **fallan ruidosamente al cargar la configuración** en vez de quedarse en silencio sin hacer nada (corrección crítica frente a la trampa del fallo silencioso de Axolotl).

**Cableado en vivo** — ya está. Los wrappers de los trainers SFT y Pretrain realmente instancian la subclase multipack cuando `multipack: true` está definido. La anulación de `get_train_dataloader` que hace la factory instala `MultipackBatchSampler(real_batches=False)` (produce una `list[int]` plana por secuencia empaquetada, compatible con DataLoader) como `batch_sampler=` del DataLoader, reenviando `dataloader_drop_last`/`num_workers`/`pin_memory` desde `TrainingArguments`. La anulación de `_get_train_sampler` se mantiene como respaldo defensivo que no hace nada y siempre delega en super, así que cualquier bucle de evaluación/predicción de HF que evite `get_train_dataloader` igual obtiene la forma correcta de `Sampler[int]` (sin desajuste de forma por listas anidadas). Multipack es **solo para sft / pretrain** en el backend `transformers`; los trainers de preferencias / RLHF y el backend MLX reciben mensajes de error distintos que nombran la razón real. Los datasets deben exponer `input_ids` (preferido) o `length` por fila; el texto sin procesar dispara una advertencia de todo ceros.

**Sharding multi-GPU (v0.71.19).** Bajo FSDP / DeepSpeed ZeRO / DDP (`num_processes > 1`), la anulación de `get_train_dataloader` enruta el DataLoader de multipack a través de `accelerator.prepare`, de modo que el `BatchSamplerShard` de accelerate reparte en round-robin bins FFD completos a cada rank (preservando el empaquetado; la semilla del bin es idéntica entre ranks, así que todos coinciden en el orden global antes del sharding). La ruta de una sola GPU devuelve el DataLoader sin cambios. La corrección multi-GPU se prueba con mocks: una validación real con 2+ GPU está en la lista de QA pendiente.

**Endurecimiento contra DoS** — el empaquetador FFD tiene un tope de 1M de elementos (un límite a la memoria retenida; la colocación en sí es O(N log N) desde #726); el constructor de máscaras 4D limita las asignaciones a 2³¹ celdas; el analizador Jinja de plantillas de chat limita a 128KB. Toda entrada numérica rechaza `bool` de forma explícita (coincide con la política del proyecto desde v0.30.0).

El `JinjaTemplateAnalyzer` (también v0.37.0) recorre los AST de las plantillas de chat para descubrir referencias no estándar `message.<field>` (`tool_calls`, `name`, `weight`, `train`), que usa la ruta `train_on_messages_with_train_field` de v0.36.0 para que las máscaras de entrenamiento por mensaje reconozcan campos más allá de `role` / `content`. El analizador parsea las plantillas sin renderizarlas, así que un `soup.yaml` manipulado no puede provocar SSRF.


## Contexto largo — YaRN, Llama 3.1 NTK, LongLoRA

Soup incluye cuatro estrategias de escalado de RoPE (`linear`, `dynamic`, `yarn`, `llama3`); LongLoRA se rechaza, mira más abajo:

```yaml
# soup.yaml
base: meta-llama/Meta-Llama-3-8B  # 8k, sin escalado de RoPE nativo
task: sft
data:
  train: ./data.jsonl
  max_length: 32768  # extender de 8k → 32k
training:
  rope_scaling_type: yarn      # linear | dynamic | yarn | llama3
  yarn_factor: 4.0             # extensión de 4x
  yarn_beta_fast: 32
  yarn_beta_slow: 1
  yarn_attn_factor: 1.0
  gradient_checkpointing: true  # obligatorio por encima de 64k
```

**YaRN.** La mejor calidad para extensiones de 4-8x. Los parámetros ajustables (`yarn_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`) solo se aplican cuando `rope_scaling_type=yarn`; el esquema los rechaza en caso contrario. Los kernels matemáticos en Python puro se exponen en `soup_cli.utils.long_context.yarn_*` para referencia / emisión de configuración. La rotación de RoPE en sí corre dentro de HF Transformers.

**Llama 3.1 NTK-aware.** Usa `rope_scaling_type: llama3` para el escalado de bandas de frecuencia estilo Llama 3.1. En un checkpoint sin escalado de RoPE emite `factor = data.max_length / max_position_embeddings` sobre `original_max_position_embeddings = max_position_embeddings`, con `low_freq_factor` 1 y `high_freq_factor` 4, de modo que un checkpoint de 8k extendido a 64k obtiene el factor 8 propio de Llama 3.1 sobre 8192. `detect_llama3_rope_in_config` puede identificar el bloque en un dict de configuración de modelo de HF, pero `soup train` solo cambia RoPE cuando `rope_scaling_type` es explícito; si lo omites, se conserva la configuración de RoPE nativa del checkpoint. En un checkpoint que ya trae un bloque `llama3` (Llama 3.1, 3.2 y 3.3 lo hacen), `rope_scaling_type: llama3` se compone con él en lugar de reemplazarlo: se mantienen el `original_max_position_embeddings`, `low_freq_factor` y `high_freq_factor` del checkpoint, y su `factor` se multiplica por `data.max_length / max_position_embeddings`. Llama-3.1-8B extendido de 131072 a 262144 tokens entrena con factor 16 sobre 8192, así que ningún par de frecuencias rota más rápido de lo que lo hacía en el preentrenamiento.

El escalado de RoPE se aplica antes de construir el modelo en `task: sft` (la ruta de texto con `backend: transformers` y sin streaming de capas) y en `task: tts` y `task: pretrain` con `backend: transformers`. En todos los demás casos, `rope_scaling_type` se rechaza al cargar la configuración ([#1697](https://github.com/MakazhanAlpamys/Soup/issues/1697)): en las otras 20 tareas, en `sft` con `backend: mlx`, `modality: vision`, `modality: audio` o `training.stream_layers: true`, y en cualquiera de las tres con `backend: unsloth`. Esas configuraciones construyen el modelo con el RoPE propio del checkpoint; el mensaje nombra cada ajuste que hay que cambiar y las claves `yarn_*` definidas junto a él. Los parámetros del modelo que ya existen y son independientes del tipo, como `rope_theta`, se conservan; los ajustables que pertenecen a un algoritmo de RoPE anterior se eliminan cuando cambia el tipo. Un checkpoint cuyo bloque de RoPE ya está escalado (cualquier `rope_type` distinto de `default`, por ejemplo `yarn`, `longrope` o `llama3`) nunca se reemplaza: salvo `llama3` sobre un bloque `llama3`, extenderlo se rechaza antes de construir el modelo, y el error nombra el `rope_type` y el `factor` del checkpoint. Un `data.max_length` igual o menor que el `max_position_embeddings` del checkpoint no extiende nada, así que ninguno de estos rechazos aplica. Los modelos como Gemma 3 que usan secciones anidadas de RoPE por capa se rechazan en lugar de modificarse parcialmente. `rope_scaling_type: longrope` se rechaza al cargar la configuración, sea cual sea `data.max_length`. Sus vectores `short_factor` y `long_factor` por dimensión solo existen en checkpoints ya escalados con LongRoPE, y extender esos se rechaza, de modo que no puede extender ningún checkpoint. Para extender un checkpoint sin escalado de RoPE, usa `linear`, `dynamic`, `yarn` o `llama3`. Para ajustar un checkpoint LongRoPE como Phi-3-mini-128k en su longitud nativa, deja `rope_scaling_type` sin definir: se usa el bloque de RoPE propio del checkpoint tal como viene.

**LongLoRA S².** `training.use_longlora: true` se rechaza al cargar la configuración ([#1240](https://github.com/MakazhanAlpamys/Soup/issues/1240)): la anulación que instalaba filtraba tokens futuros a posiciones anteriores y no aplicaba agrupación S² (consulta [Anulación del forward de LongLoRA](#anulación-del-forward-de-longlora)). Usa en su lugar una de las estrategias de escalado de RoPE de arriba con LoRA simple.

```yaml
# Llama 3.1 trae escalado llama3 hasta 128k; esto se compone con él hasta 256k
base: meta-llama/Llama-3.1-8B
training:
  rope_scaling_type: llama3
  gradient_checkpointing: full
data:
  max_length: 262144
```


## Expansión de bloques de LLaMA Pro

Agrega `N` bloques transformer inicializados en cero a un modelo base y entrena **solo los bloques nuevos**: mantiene intacto el comportamiento original mientras añade capacidad para un dominio nuevo (según el artículo de LLaMA Pro, `arxiv.org/abs/2401.02415`).

```yaml
# soup.yaml — entrenamiento continuado de LLaMA Pro sobre una base Llama-3.1
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: ./domain.jsonl
training:
  quantization: none            # la expansión de bloques necesita una base sin cuantizar
  expand_layers: 4              # añade 4 bloques decoder con init en cero
  freeze_trainable_layers: 4    # debe ser igual a expand_layers: congela el modelo original, entrena solo los bloques añadidos
  lora:
    r: 0                        # entrena los bloques añadidos directamente; un rango LoRA se rechaza junto con expand_layers
  lr: 5e-5
  epochs: 1
```

**Qué ocurre al iniciar el trainer.** Soup copia en profundidad los últimos `expand_layers` bloques decoder, inicializa en cero las proyecciones residuales de cada clon (`mlp.down_proj` + `self_attn.o_proj`) para que el bloque añadido actúe inicialmente como identidad, los añade a `model.model.layers` y actualiza `config.num_hidden_layers`. `freeze_trainable_layers` debe ser igual a `expand_layers`: congela todos los parámetros salvo los bloques añadidos, la receta canónica de LLaMA Pro de "entrenar solo los bloques nuevos". No selecciona las capas top-N ni bottom-N, y cualquier otro valor, incluidos `0` o uno negativo, se rechaza al cargar la configuración.

**Alcance.** Funciona en `task: sft` con `backend: transformers`, `modality: text`, `quantization: none` y `lora.r: 0`. Un rango LoRA se rechaza junto con `expand_layers` ([#1326](https://github.com/MakazhanAlpamys/Soup/issues/1326)): la expansión corre antes de `get_peft_model`, que congela de nuevo todos los parámetros que no sean adaptadores, incluidos los bloques añadidos, así que solo entrenarían los adaptadores de las capas originales. `task: pretrain` no tiene ruta con `lora.r: 0`: un rango se rechaza al cargar la configuración, y `lora.r: 0` carga pero luego falla dentro de peft al iniciar el trainer ([#1813](https://github.com/MakazhanAlpamys/Soup/issues/1813)), así que la receta necesita `task: sft`. Otras tareas, backends y modalidades, y una base cuantizada, se rechazan al cargar la configuración. Ningún otro trainer aplica la expansión, y los bloques añadidos solo se admiten sobre una base sin cuantizar. Límites: `expand_layers ∈ [1, 64]`. La sobreexpansión (más bloques nuevos que capas tiene la base) se recorta en silencio al número de capas de la base. Las arquitecturas que no tienen la forma de Llama (p. ej. `dense_4h_to_h` de Falcon) emiten un `warnings.warn` porque la heurística de init en cero de las residuales solo coincide con los nombres estándar `down_proj` / `o_proj`: los bloques añadidos se siguen añadiendo y son entrenables, pero pierden la garantía de init como identidad.


## Zoo de optimizadores y PEFT

Elige entre un catálogo más amplio de optimizadores y usa la inicialización de LoRA consciente de la cuantización:

```yaml
training:
  # nativos de HF, bnb-8bit y añadidos de v0.41.0 (lomo, apollo_adamw,
  # grokadamw, schedule_free). Diez nombres retirados (badam, adam_mini, muon,
  # dion, came_pytorch, ao_adamw_{fp8,4bit,8bit}, ...) se rechazan al
  # cargar la configuración porque transformers 5.x los rechaza (#1269);
  # muon y adamw_hf todavía funcionan en backend: mlx.
  optimizer: adafactor

  # Alias amigables para usuarios que vienen de LlamaFactory / Axolotl
  # load_in_8bit: true      # equivalente a quantization: 8bit
  # load_in_16bit: true     # equivalente a quantization: none

  quantization: none        # obligatorio: PEFT LoftQ cuantiza la base misma

  lora:
    init_strategy: loftq    # init de LoRA consciente de la cuantización (también: pissa / olora / random)
    loftq_iter: 1
    loftq_bits: 4

  # la expansión de bloques de LLaMA Pro necesita lora.r: 0, así que no puede compartir una corrida con una
  # init de LoRA como LoftQ; consulta Expansión de bloques de LLaMA Pro arriba.
```

Errores amigables generales: los errores de tipeo en `optimizer:` se rechazan al cargar la configuración, con los añadidos de v0.41.0 listados en el mensaje; `load_in_8bit` mezclado con `load_in_16bit` lanza una excepción en vez de elegir uno en silencio.

**`lr_groups` no se aplica.** La tasa de aprendizaje por módulo que describe se parsea y se valida (los patrones deben ser regex compilables), pero ningún optimizador la lee, así que todos los parámetros entrenan con `lr`. Definirlo genera una advertencia en v0.76 y se rechaza a partir de v0.77 (#761).

PiSSA, OLoRA, LoftQ y VeRA se aplican a través del constructor PEFT compartido en el backend Transformers. Soup rechaza estas variantes en MLX y Unsloth en lugar de sustituirlas en silencio por LoRA común. PiSSA y LoftQ además requieren `quantization: none`: PiSSA necesita pesos base en punto flotante para su SVD, mientras que LoftQ hace la conversión a bits bajos por su cuenta, así que una base ya cuantizada no es válida para ninguno de los dos inicializadores.

En el backend Transformers, `target_modules: auto` se resuelve en este orden, y una lista explícita de objetivos siempre gana sin cambios:

1. **Las arquitecturas que PEFT mapea por sí mismo** (`llama`, `mistral`, `qwen2`, …) quedan a lo que PEFT define por defecto.
2. **Las arquitecturas que Soup mapea** se resuelven desde `utils/peft_wiring.py`. La familia Qwen3.5 apunta a `q_proj` y `v_proj` en las capas de atención completa, más `in_proj_qkv` y `out_proj` en las capas fusionadas de atención lineal, porque PEFT no mapea `qwen3_5_text`. Las arquitecturas MoE para las que Soup incluye recetas (`qwen3_moe`, `deepseek_v3`, `deepseek_v4`, `glm4_moe`, `glm_moe_dsa`, `granitemoehybrid`, `kimi_k2`/`kimi_k25`, `gpt_oss`, `minimax_m2`, `minimax_m3_vl`, `mistral3`) apuntan a sus proyecciones de atención; PEFT no mapea ninguna de ellas (#1070). MiniMax-M3 usa una regex acotada a su torre de lenguaje, así que un fine-tune de texto no adapta el codificador de visión. `mistral3` (el wrapper de visión-lenguaje detrás de `mistral-medium-3-5-sft`, que por eso define `modality: vision` para que el SFT cargue la clase de imagen-texto) usa el mismo tipo de regex de torre de lenguaje: `q_proj`, `k_proj`, `v_proj` y `o_proj` solo bajo `language_model...self_attn`, así que la torre de visión Pixtral, el proyector multimodal y las proyecciones MLP quedan sin adaptar (#1395). `glm4_moe` es GLM-4.6 y *no* es `glm_moe_dsa` (GLM-5 / GLM-5.1): los dos tienen formas de atención distintas.
3. **Todo lo demás falla de forma cerrada.** `auto` sobre una arquitectura que ni PEFT ni Soup mapean se rechaza en la configuración inicial, nombrando el `model_type`, en lugar de llegar al `No target_modules passed` de PEFT. No es un comportamiento nuevo —PEFT también las rechazaba—, solo un mensaje más claro. Abarca modelos densos y también MoE: al momento de escribir esto, `phi3`, `smollm3`, `lfm2` y varias arquitecturas de visión/audio del catálogo caen aquí. Da una lista explícita de `target_modules` o, para un modelo MoE, define `training.moe_lora: true`, que aporta los objetivos de expertos y se comprueba *antes* del rechazo. `training.lora.target_parameters` por sí solo también basta.

**`granitemoehybrid` solo se adapta en parte, y lo dice en la configuración inicial.** Granite 4.0 es un híbrido: en `ibm-granite/granite-4.0-h-tiny-base`, solo 4 de las 40 capas decoder tienen un `self_attn` (`config.layer_types` es 36 `linear_attention` + 4 `full_attention`), así que la entrada de proyecciones de atención de arriba alcanza a un décimo del decoder. Las otras 36 capas son bloques Mamba-2 (`mamba.in_proj`, `mamba.out_proj`), y las proyecciones de expertos compartidos de cada capa (`shared_mlp.input_linear`, `shared_mlp.output_linear`) y los expertos enrutados fusionados también se dejan intactos, de forma coherente con todas las demás filas de esa tabla, donde la política es solo proyecciones de atención. El entrenamiento imprime una línea amarilla `Partial LoRA coverage:` que nombra el tipo de modelo, la fracción contada y lo que se omitió, para que el adaptador pequeño no sea una sorpresa al fusionar. Si quieres alcanzar las proyecciones de espacio de estados o de expertos compartidos, nómbralas explícitamente:

```yaml
training:
  lora:
    target_modules: [q_proj, k_proj, v_proj, o_proj, in_proj, out_proj]
```

Esa lista es correcta como nombres de módulos en esta arquitectura —es lo que reporta `named_modules()`—, pero Soup no ha medido si adaptar una proyección de espacio de estados entrena bien, y por eso no es lo predeterminado. Tómalo como la forma de alcanzar esas capas, no como una recomendación de hacerlo.

El backend MLX conserva su propio valor predeterminado de clave completa (`self_attn.q_proj`, `self_attn.v_proj`).

Los expertos enrutados de Qwen4-Exp son parámetros 3-D sin procesar en lugar de módulos `nn.Linear`, así que `target_modules: auto` / `all-linear` deliberadamente no los incluye. Opta por la ruta de objetivos por parámetro de PEFT para un adaptador residente de SFT o preentrenamiento continuado de mayor capacidad:

```yaml
training:
  lora:
    r: 16
    alpha: 32
    dropout: 0                 # requerido por PEFT ParamWrapper
    target_modules: auto       # toda la familia lineal de Qwen4-Exp
    target_parameters: auto    # tensores gate_up_proj + down_proj enrutados
    rank_pattern:
      experts.gate_up_proj: 2
      experts.down_proj: 2
```

`target_parameters: auto` falla de forma cerrada cuando una arquitectura no tiene un mapeo registrado; también se acepta una lista explícita de sufijos de nombres de parámetros. Actualmente está limitado a `sft` / `pretrain` de texto residente en el backend Transformers y a LoRA/rsLoRA simples con inicialización aleatoria. PEFT requiere dropout cero para parámetros sin procesar y advierte que `torch.compile` puede recompilar o romper el grafo alrededor de los wrappers de parámetros. Los adaptadores MoE con objetivos por parámetro además materializan una contribución para cada experto durante la inferencia; fusiona el adaptador en la base para el despliegue cuando no necesites el hot-swapping.

Consulta `soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS` para ver la lista completa de optimizadores admitidos.


## Calidad de LoRA — PiSSA, ReLoRA, rango por patrón, parches quirúrgicos

Cinco mejoras en la superficie de PEFT que LlamaFactory y Axolotl mantienen:

```yaml
training:
  quantization: none            # obligatorio: el valor por defecto del esquema es 4bit; ReLoRA/PiSSA necesitan float
  lora:
    init_strategy: pissa          # 'random' (por defecto), 'pissa', 'olora'
    rank_pattern:                 # anulación de rango por módulo objetivo
      q_proj: 8
      v_proj: 16
    alpha_pattern:                # anulación de alpha por módulo objetivo
      q_proj: 16
  relora_steps: 500               # reinicio con merge+reinit cada 500 steps
  relora_warmup_ratio: 0.1        # omite el primer 10% del entrenamiento
  relora_prune_ratio: 0.9         # obsoleto; se conserva para YAML antiguos (se ignora)
  relora_reset_optimizer: true    # limpia el estado del optimizador después de cada reinicio
```

**PiSSA** inicializa el par LoRA a partir de la SVD del peso base, lo que da una convergencia temprana más rápida que la init aleatoria a costa de una pasada extra de SVD en la primera epoch. También se acepta `init_strategy: olora`; definir el heredado `use_olora: true` se alinea automáticamente por compatibilidad hacia atrás.

**ReLoRA es un cambio de comportamiento.** El `training.quantization` de Soup **es `4bit` por defecto**. Los reinicios de ReLoRA fusionan la actualización de LoRA en el peso base, así que `relora_steps` ahora **exige un `quantization: none` explícito** en soup.yaml. Una configuración que antes solo definía `relora_steps` se rechaza al parsear. Ninguna receta, plantilla o ejemplo incluido define `relora_steps`.

ReLoRA se dispara cada N steps globales, fusiona la actualización de LoRA (B @ A, con el escalado de PEFT) en el peso base congelado, reinicializa `lora_A` (Kaiming) y `lora_B` (ceros), opcionalmente limpia el estado del optimizador de esos parámetros del adaptador y ejecuta un breve recalentamiento de la tasa de aprendizaje. El primer step tras el reinicio usa `1/(W+1)` de la LR objetivo y luego avanza con ese mismo incremento hasta la LR completa. Los reinicios solo se acumulan fielmente sobre una base fp32; las bases bf16/fp16 pierden parte del delta de fusión por redondeo. Los adaptadores de embeddings (`lora_A` vacío, normalmente `embed_tokens`) se omiten; el LoRA de capas Linear se fusiona aunque peft ponga dicts vacíos `lora_embedding_A`/`lora_embedding_B` en cada `LoraLayer`. Útil para corridas de entrenamiento muy largas donde la capacidad del adaptador se satura. `relora_prune_ratio` se conserva para YAML compatibles hacia atrás, pero ya no poda los pesos del adaptador.

Después del entrenamiento, Soup fusiona el adaptador activo en la base en memoria ya acumulada y guarda la salida final como un modelo denso independiente. Carga o exporta esa salida directamente; no ejecutes `soup merge` sobre ella. Los directorios de checkpoints intermedios siguen siendo artefactos del trainer, y `--resume` / `--hf-resume` se rechazan cuando `relora_steps` está configurado, porque esos checkpoints no codifican el estado de reinicios acumulado.

**Rango/alpha por patrón** mapean patrones de nombres de módulos a rangos enteros. Útil en configuraciones MoE donde los FFN de expertos necesitan un rango menor que la atención. Límites: 256 claves × valor 1024.

**Parches quirúrgicos** (el reemplazo de `ClippableLinear` en Gemma 4, la eliminación de `lora_dropout` en expertos 3-D de MoE fusionados) se activan solos cuando coinciden el nombre del modelo y la arquitectura. Ambos están condicionados y son silenciosos en modelos no relacionados.

**Registro de plantillas** — las 21 plantillas integradas ahora viven como `src/soup_cli/templates/*.yaml` con un índice `manifest.json`. `soup init --template <name>` lee el YAML; las copias en línea en `schema.py` se mantienen como respaldo por compatibilidad hacia atrás, obsoletas en favor del registro YAML.

**Alcance multi-trainer** — ReLoRA y los parches quirúrgicos están cableados en todos los trainers del backend transformers: `sft`, `dpo`, `grpo`, `kto`, `orpo`, `simpo`, `ipo`, `ppo`, `reward_model`, `pretrain`, `embedding`, `bco`, más el despachador unificado `task: preference`. El validador cruzado del esquema rechaza el backend MLX, una cuantización distinta de `none`, `stream_layers`, `lora.use_vera`, `lora.use_dora` y `use_fsdp2_compile` (el callback es específico de HF Trainer y el reinicio requiere una base float escribible).


## DoRA (LoRA con descomposición de pesos)

Activa DoRA para mejorar la calidad de LoRA con descomposición de magnitud:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_dora: true  # activa DoRA
```

Funciona con todas las tareas de entrenamiento en el backend transformers (mira la nota sobre bases cuantizadas más abajo). En `backend: unsloth` y `backend: mlx`, `use_dora: true` se rechaza al cargar la configuración: ninguno de los dos backends construye un adaptador DoRA, así que entrenaría LoRA simple en silencio.

> **No en bases GPTQ / AWQ / AQLM / EETQ.** peft no tiene variante DoRA para esas capas y lanza una excepción al enganchar el adaptador, así que `use_dora: true` con `quantization: gptq`, `awq`, `aqlm` o `eetq` se rechaza al cargar la configuración. Usa LoRA simple en esas bases, o una base `4bit` / `8bit` / `hqq:Nbit` / sin cuantizar para conservar DoRA.


## LoRA+ (tasas de aprendizaje diferenciadas)

Usa tasas de aprendizaje distintas para las matrices A y B de LoRA:

```yaml
training:
  lr: 2e-5
  loraplus_lr_ratio: 16.0  # lr_B = lr × 16
  lora:
    r: 64
    alpha: 16
```

- **Compatibilidad:** Cableado en todas las tareas basadas en Trainer que entrenan un adaptador LoRA, incluidas las tareas de preferencias y RL (`dpo`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `grpo`, `online_dpo`, `ppo`, `reward_model`, `distill`). La mayoría engancha el optimizador LoRA+ una vez construido el trainer; `ppo` se lo pasa al constructor del trainer, porque el trainer PPO de trl construye su optimizador y su scheduler de LR de forma anticipada y le añade el modelo de valor con la tasa de aprendizaje base, como hace el optimizador por defecto de trl. `classifier`, `reranker`, `cross_encoder` y `asr` hacen fine-tune completo por defecto, así que LoRA+ solo aplica ahí con `classifier_lora: true` / `asr_lora: true` y `lora.r > 0`. Se rechaza al parsear la configuración en tareas que no tienen una matriz LoRA $B$ entrenable: `prm` (fine-tune completo), `moe_lora_routing` (solo entrena la compuerta de enrutamiento) y `classifier` / `reranker` / `cross_encoder` / `asr` sin su flag de LoRA. También se rechaza en `unlearn`, que corre su propio bucle de optimizador en vez de un Trainer. Se rechaza con `lora.use_vera` (VeRA entrena vectores de escalado, no matrices $A$/$B$, así que cada tensor entrenable correría con `lr * ratio`). También se rechaza en el backend `mlx` (#1324): el optimizador MLX que construye Soup (`trainer/mlx_optim.py`) corre cada tensor LoRA con una sola tasa de aprendizaje (sin grupos de parámetros), así que la razón se ignoraría en silencio; quítala o usa `backend: transformers`. `backend: unsloth` recibe el mismo optimizador LoRA+ que `transformers`. Mutuamente excluyente con `use_lorafa`.
- **Optimizadores:** LoRA+ construye su optimizador sin el modelo, así que se rechaza al cargar la configuración con `optimizer: apollo_adamw`, `lomo` o `adalomo` (transformers solo puede construirlos a partir del modelo) y con `use_galore`.


## LoRA-FA (LoRA con A congelada)

Congela las matrices de proyección aleatorias en LoRA $A$ y actualiza solo las matrices LoRA $B$ usando `create_lorafa_optimizer` de PEFT ([arXiv:2308.03303](https://arxiv.org/abs/2308.03303)):

```yaml
training:
  lr: 2e-4
  use_lorafa: true
  lora:
    r: 64
    alpha: 16
```

### Punto de operación y advertencias
- **Punto de operación medido del adaptador:** Entrena exactamente 50.0% menos parámetros por proyección adaptada (entrena $B$, congela $A$), lo que reduce a la mitad los estados del optimizador AdamW (`exp_avg_B`, `exp_avg_sq_B`) en proyecciones cuadradas.
- **Retención analítica de activaciones:** Congelar $A$ evita almacenar las activaciones de entrada $x \in \mathbb{R}^{B \times L \times d_{in}}$ para la retropropagación del adaptador a través de $A$. Solo se retiene $u = A x \in \mathbb{R}^{B \times L \times r}$, lo que da una razón analítica de activaciones del adaptador de $r / d_{in}$ (~64× de reducción para rango 64 en dimensión oculta 4096; la razón exacta escala con el rango que elijas).
- **Alcance y limitaciones:** Estos valores representan un punto de operación de micro-benchmark y una razón analítica de tensores guardados para las proyecciones del adaptador: **no son ahorros totales ni de VRAM máxima del LLM, ni un resultado de throughput de extremo a extremo, ni una afirmación de calidad.** La VRAM máxima de entrenamiento en un fine-tune completo de un LLM está dominada por las activaciones del modelo base, las KV caches y los pesos; los ahorros totales de VRAM de extremo a extremo son bastante menores. La calidad en la tarea final y el throughput de extremo a extremo frente a LoRA estándar siguen sin medirse. Consulta [`benchmarks/gate-725-lorafa-operating-point.md`](../benchmarks/gate-725-lorafa-operating-point.md) para ver las cifras medidas.
- **Compatibilidad:** Se admite en el backend `transformers` para las tareas `sft`, `pretrain` y `embedding`. Mutuamente excluyente con `loraplus_lr_ratio` (que diferencia las tasas de $A$ y $B$), `use_galore`, `lora.use_vera` (VeRA entrena vectores de escalado, así que `create_lorafa_optimizer` no encuentra matrices $B$), optimizadores que no sean AdamW y el backend `mlx`. Requiere `lora.r` y `lora.alpha` explícitos. LoRA-FA no se ha validado con `stream_layers: true` (streaming de capas); no se recomienda combinarlos.


## rsLoRA (escalado estabilizado por rango)

Usa el escalado de LoRA estabilizado por rango para obtener mejor rendimiento con rangos altos:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_rslora: true  # activa el escalado estabilizado por rango
```

Funciona con todas las tareas de entrenamiento y todos los backends. Recomendado para rango de LoRA ≥ 32.


## VeRA y OLoRA (PEFT de menor consumo)

Dos variantes más de LoRA para presupuestos de memoria más ajustados:

**VeRA** (Vector-based Random Adaptation) — comparte matrices de proyección aleatorias congeladas entre todas las capas y entrena solo vectores de escalado pequeños. Archivo de adaptador mucho más pequeño.

```yaml
training:
  lora:
    r: 256           # VeRA normalmente necesita un rango mayor (128-512)
    alpha: 1
    use_vera: true
```

**OLoRA** (Orthonormal LoRA) — inicializa los pesos de LoRA a partir de pesos base descompuestos por QR; converge más rápido.

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_olora: true
```

> **Mutuamente excluyentes:** `use_dora`, `use_vera` y `use_olora` no se pueden combinar en una misma configuración. Soup lo valida al cargar.


## NEFTune (fine-tuning con embeddings ruidosos)

Agrega ruido a los embeddings durante el entrenamiento para mejorar la calidad del modelo de chat:

```yaml
training:
  neftune_alpha: 5.0  # intensidad del ruido (0-50, normalmente 5-15)
```

Funciona con las tareas SFT, DPO, KTO, ORPO, SimPO e IPO.


## Empaquetado de muestras

Empaqueta varias muestras cortas en una sola secuencia para entrenar más rápido:

```yaml
training:
  packing: true  # empaqueta muestras cortas juntas (entrenamiento más rápido)
```

Funciona con las tareas SFT y Pretrain. El SFT empaquetado conserva la máscara de loss solo para el asistente (`train_on_responses_only`, `train_on_messages_with_train_field`, `mask_history` o una columna `labels` ya tokenizada). Se emite una advertencia si `max_length < 256`.


## Aprendizaje por currículo

Ordena el dataset por dificultad (fácil → difícil) para lograr mejor convergencia:

```yaml
training:
  curriculum: true             # activa el aprendizaje por currículo
  curriculum_metric: length    # ordena por: length, perplexity o loss
  curriculum_buckets: 4        # número de etapas de dificultad
```


## Congelamiento de capas

Congela las capas inferiores del modelo y entrena solo las superiores (como el `finetuning_type: freeze` de LLaMA-Factory). Solo `task: sft` (y `tts`, que entrena a través del trainer de SFT) aplica estos dos campos; en cualquier otra tarea la configuración se rechaza al cargar, porque ese trainer entrenaría todas las capas (#1497). Dentro de `sft` y `tts` solo se aplican en la ruta de texto con `backend: transformers`: `backend: mlx`, `backend: unsloth`, `modality: vision`, `modality: audio` y `stream_layers: true` también se rechazan al cargar: esas configuraciones nunca leen los campos y siempre enganchan un adaptador, así que la corrida entrenaría todas las capas (#1581).

```yaml
training:
  freeze_layers: 24    # congela las primeras 24 capas, entrena el resto
  # O
  freeze_ratio: 0.75   # congela el 75% de las capas desde abajo
```

Funciona con y sin LoRA. Cuando se usa con LoRA, LoRA se aplica solo a las capas no congeladas.

## LISA — muestreo de importancia por capas (v0.71.34)

LISA (Layerwise Importance Sampled AdamW, [arXiv:2403.17919](https://arxiv.org/abs/2403.17919)) apunta a la calidad de un fine-tune completo con una memoria similar a la de LoRA. **Medido a 7B+, cumple la primera mitad y no la segunda**: mira [lo que cuesta en realidad](#lo-que-lisa-cuesta-en-realidad-medido-a-3b-y-8b) más abajo antes de elegirlo por encima de LoRA. En lugar de elegir capas una sola vez (eso es el `unfrozen_parameters` estático de Spectrum), LISA vuelve a muestrear un pequeño conjunto aleatorio de capas decoder **cada N steps** y congela el resto; los embeddings de entrada, la LM head y la norm final permanecen entrenables todo el tiempo por defecto (define `lisa_train_embeddings: false` para congelar también ese grupo; mira [el intercambio de memoria](#recuperar-el-costo-siempre-activo-lisa_train_embeddings) más abajo).

```yaml
task: sft                 # o `pretrain` — el preentrenamiento continuado usa el mismo
                          # mecanismo de FT completo de las capas activas (#307)
backend: transformers
modality: text
training:
  quantization: none      # LISA es FT completo de las capas activas
  lisa_enabled: true
  lisa_num_layers: 2       # capas decoder activas por intervalo (limitadas a la profundidad del modelo)
  lisa_interval_steps: 20  # cadencia de remuestreo, en steps globales
  lisa_train_embeddings: true  # por defecto; false congela embeddings + head + norm final
```

Como en cada momento solo entrena un puñado de capas (y su estado del optimizador se limpia cuando se vuelven a congelar), la memoria máxima del optimizador es aproximadamente `embeddings + head + lisa_num_layers`, muy por debajo de un fine-tune completo, mientras que todas las capas se siguen actualizando a lo largo del entrenamiento. LISA es solo `sft` o `pretrain` + `transformers` + `text` + `quantization: none`, y es mutuamente excluyente con las funciones de LoRA, `freeze_layers`/`freeze_ratio` y el `unfrozen_parameters` de Spectrum (cada uno decide de forma independiente qué se entrena).

### Lo que LISA cuesta en realidad, medido a 3B y 8B

Medido en una H100 80GB, con Alpaca, 200 steps, 3 repeticiones intercaladas por brazo y el reloj SM fijado en 1980 MHz durante todo el proceso. La activación de LISA se verificó en lugar de darse por supuesta: el conjunto de capas entrenables rota exactamente en el intervalo, el recuento de parámetros entrenables coincide con la aritmética hasta la unidad, y `lisa_num_layers` definido en *todas* las capas reproduce el brazo de fine-tune completo hasta tres decimales.

**Llama-3.1-8B-Instruct:**

| brazo | VRAM máxima | loss en held-out |
|---|---|---|
| fine-tune completo | **no cabe** (OOM a 73.94 GB, también con `batch_size 1`) | – |
| LISA (2 capas / 20 steps) | 52.14 GB | 1.294 |
| LoRA r=16 | **34.56 GB** | **1.275** |

**Qwen2.5-3B-Instruct, cada brazo con su propia mejor tasa de aprendizaje:**

| brazo | VRAM máxima | loss en held-out |
|---|---|---|
| fine-tune completo | 57.60 GB | 1.2905 |
| LISA | 19.37 GB | **1.2463** |
| LoRA r=16 | **15.93 GB** | **1.2420** |

**La afirmación sobre la calidad se sostiene: LISA superó al fine-tune completo en ambas tasas de aprendizaje.** La afirmación sobre la memoria no: LISA usa **1.22x lo de LoRA a 3B y 1.51x a 8B**, y la brecha *se amplía* con la escala.

La razón es estructural, no un fallo de ajuste. Los embeddings de entrada, la LM head y la norm final permanecen entrenables **en todos** los intervalos, y a 8B eso es el **70.7%** de todo lo que entrena LISA (66.9% a 3B). Así que `lisa_num_layers` solo controla cerca del 30% del costo, y el otro 70% crece con el vocabulario x la dimensión oculta: un costo que LoRA nunca paga.

### Recuperar el costo siempre activo: `lisa_train_embeddings`

Define `lisa_train_embeddings: false` para congelar el grupo siempre activo (embeddings de entrada, LM head, norm final) de modo que solo entrenen las `lisa_num_layers` capas decoder muestreadas. Como ese grupo es la mayoría de lo que entrena LISA, esta es la perilla que realmente acerca la memoria de LISA al objetivo similar a LoRA que promete el artículo.

Es un **intercambio real, no una victoria gratis**: el conjunto siempre activo probablemente sostiene el resultado de calidad de LISA, así que congelarlo puede mover el loss en held-out. El valor por defecto sigue siendo `true` (LISA exactamente como se publicó) justamente porque esta debería ser una decisión medida, no un cambio silencioso: mide ambas formas en tu modelo antes de decidirte.

> **Advertencia previa al vuelo.** El pre-flight analítico de VRAM sigue clasificando LISA como fine-tune completo sin importar `lisa_train_embeddings`, así que **todavía no** acredita el ahorro de congelar el grupo siempre activo: una corrida con embeddings congelados que sí cabría puede seguir rechazándose antes de lanzarse. Esto es deliberado: predecir de más es el fallo seguro (predecir de menos es un desborde silencioso en Windows), y acreditar el ahorro necesita una constante medida en hardware GPU. Usa `--allow-oom-attempt` para lanzar una corrida que el pre-flight rechaza de forma conservadora.

**Elige LISA cuando necesites actualizaciones de rango completo en un modelo demasiado grande para un fine-tune completo**: su verdadera ventaja es que 8B entrena en una sola tarjeta de 80 GB, donde un fine-tune completo necesita unos 120 GB. En los demás casos, prefiere LoRA: en estos tamaños igualó o superó a LISA en loss de held-out usando un tercio menos de memoria.

Los valores por defecto están bien elegidos y no necesitan cambios a 7B+. Aumentar `lisa_num_layers` degrada la memoria, la velocidad **y** la calidad de forma monótona (held-out a 3B: 1.2504 con 2, 1.2673 con 8, 1.2950 con 16), y por encima de 8 provoca OOM en una tarjeta de 80 GB a 8B. `lisa_interval_steps` está en un óptimo amplio y plano: de 1 a 50 son indistinguibles, y solo 200 (una sola muestra durante la corrida) degrada.

Una advertencia sobre los números: la calidad en held-out aquí es loss en distribución y exactitud de tokens sobre una partición de validación de Alpaca, no un benchmark final, así que no puede ver regresiones de capacidad que sí vería un benchmark de tarea. Registro completo de las mediciones: [`benchmarks/gate-h100-validation.md`](../benchmarks/gate-h100-validation.md).

Nota de implementación: el modelo se deja totalmente entrenable en el momento de configurar el trainer, para que el optimizador de HF (construido antes de que se dispare el primer callback) contenga todos los parámetros del decoder; luego el callback de LISA alterna `requires_grad` en cada intervalo: los parámetros congelados no producen gradiente y el optimizador los omite.


## Watchdog de loss

Detiene el entrenamiento automáticamente cuando el loss supera un umbral (como el `loss_watchdog_threshold` de Axolotl):

```yaml
training:
  loss_watchdog: true           # activa la detección de picos de loss
  loss_watchdog_threshold: 3.0  # detiene si el loss supera este valor
  loss_watchdog_patience: 5     # steps consecutivos por encima del umbral antes de detenerse
```

> **Nota sobre backends:** Definir `loss_watchdog: true` se rechaza en `backend: mlx` al validar la configuración (Soup no implementa el watchdog en el callback de MLX, que no tiene control de detención).

## Estabilidad del entrenamiento y ajuste automático

Ajuste previo al vuelo + redes de seguridad durante el entrenamiento. Todos los flags son opcionales.

### Buscador de rango de LR

Ejecuta un barrido geométrico de LR estilo fast.ai antes de la corrida real de entrenamiento. Soup escribe un reporte JSON con la LR recomendada, la curva de loss y el punto de divergencia para que elijas la LR con confianza.

```bash
soup train --config soup.yaml \
  --find-lr \
  --find-lr-start 1e-7 \
  --find-lr-end 1e-1 \
  --find-lr-steps 100 \
  --find-lr-output ./lr_finder.json
```

El reporte contiene los `lrs[]` geométricos, los `losses[]` sin procesar y suavizados con EMA, la LR recomendada (el gradiente negativo más pronunciado antes de la divergencia), la LR con el loss mínimo y el punto de divergencia, si lo hay.

El barrido toma una fila de entrenamiento por step. Con menos filas que `--find-lr-steps`, ejecuta un step por fila sobre el mismo rango `--find-lr-start` → `--find-lr-end` e imprime una línea que lo indica. La recomendación necesita al menos 4 puntos, así que `--find-lr-steps` debe ser al menos 4 y un conjunto de entrenamiento con menos de 4 filas se rechaza antes de cargar el modelo. Si el loss deja de ser finito a mitad de camino, el reporte cubre los steps anteriores; si eso deja menos de 4, el comando dice dónde divergió en lugar de escribir un reporte. Si el barrido no puede ejecutarse en absoluto (la configuración no carga, el modelo o el dataset no se pueden cargar, o `torch` no está instalado), el comando nombra la causa, sale con 1 y no escribe ningún reporte.

### Programación de warmup automática

```yaml
training:
  warmup_auto: true       # elige warmup_steps a partir de dataset_size × epochs × warmup_ratio
  warmup_ratio: 0.03      # 3% del total de steps de actualización (por defecto)
```

Se limita a `[10, 1000]` para que los datasets diminutos tengan algo de warmup y los enormes no quemen medio millón de steps desperdiciados.

### Precisión mixta automática

```yaml
training:
  auto_mixed_precision: true
```

Elige `bf16` en Ampere+, `fp16` en Turing o en modelos conocidos por ser estables en fp16 (Qwen2 / Qwen2.5 / Phi-3 / Phi-3.5) y `no` en anteriores a Pascal. Los pares multi-versión (`qwen2.5` vs `qwen2`, `phi-3.5` vs `phi-3`) coinciden con la subcadena más larga de forma determinista.

Solo `task: sft` lee el campo (y `task: tts`, que entrena a través del trainer de SFT); desde #1618, definirlo en cualquier otra tarea se rechaza al cargar la configuración, porque esos trainers eligen bf16/fp16 por su cuenta y entrenarían con su precisión por defecto sin importar esta clave.

La ruta experimental QuEST (`quantization_aware: quest`) rechaza este flag al cargar la configuración porque su evidencia cubre BF16, no FP16; consulta el [límite de evidencia de QuEST](performance-and-quantization.md#evidence-boundary).

### Recuperación automática de picos de loss

Extiende el watchdog: en lugar de detenerse ante un pico, escribe `<output>/spike_recovery.json` con la LR decaída y el número de intentos para relanzar. Limitado a 3 intentos por defecto.

```yaml
training:
  loss_watchdog: true                   # obligatorio
  loss_spike_recovery: true             # opta por la recuperación
  loss_spike_recovery_max_attempts: 3
  loss_spike_recovery_lr_decay: 0.5     # reduce la LR a la mitad en cada recuperación
```

> **Nota sobre backends:** Definir `loss_spike_recovery: true` se rechaza en `backend: mlx` al validar la configuración (la recuperación de picos la impulsa el watchdog y el watchdog no puede dispararse en MLX).

### Detector de convergencia

```yaml
training:
  convergence_detection: true
  convergence_window: 50      # steps a inspeccionar en busca de meseta / oscilación
  convergence_rel_tol: 0.005  # rango relativo por debajo de esto == meseta
```

Calcula consejos de `continue` / `early_stop` / `lower_lr` a partir de la curva de loss para los llamadores que invocan el detector. `convergence_detection` se reporta como no aplicado por `soup train`; definir `convergence_window` o `convergence_rel_tol` con valores distintos de sus predeterminados emite una advertencia al cargar en v0.76 y se rechazará a partir de v0.77 (#808).

### Aviso de presión de VRAM

```yaml
training:
  grad_accum_auto_tune: true
  grad_accum_pressure_threshold: 0.92
```

Registra la memoria máxima en cada step. La presión es la memoria máxima asignada dividida por la memoria total del dispositivo CUDA que usa la corrida (#1620), así que el umbral significa lo mismo en una tarjeta de laptop de 4 GB que en un acelerador de 80 GB. Cuando la presión cruza el umbral, recomienda un nuevo par `(batch, accum)` que preserva el batch efectivo (limitado a `accum=1024`), junto con el máximo y el total medidos.

> **Nota sobre backends:** Definir `grad_accum_auto_tune: true` se rechaza en `backend: mlx` al validar la configuración (no hay un total de VRAM con el que medir presión en memoria unificada).

> **v0.33.0:** `--find-lr` ahora ejecuta un bucle de entrenamiento con barrido de LR en proceso (reemplaza la curva de prueba de v0.32.0), la recuperación de picos escribe una pista `spike_recovery.json` con la LR decaída para relanzar, y el aviso de grad-accum imprime un par `(batch, accum)` recomendado cuando la presión de VRAM cruza el umbral. El rebobinado en vivo del estado del optimizador y la reconstrucción en vivo del DataLoader siguen pendientes (restricciones de HF Trainer / TRL upstream).


## Inteligencia de entrenamiento (olvido + calidad del checkpoint)

Los ajustes `forgetting_detection`, `checkpoint_intelligence`, `early_stop_on_regression`, `convergence_detection` y `forgetting_threshold` están reservados para callbacks planificados durante el entrenamiento. El esquema los acepta, pero no se aplican durante el entrenamiento en esta compilación. `soup train` imprime una nota de aviso cuando alguno se define con un valor distinto del predeterminado, y dirige a los usuarios a `--gate <suite.yaml>`. Sus perillas de ajuste (`forgetting_eval_steps`, `forgetting_benchmark`, `forgetting_stop`, `checkpoint_eval_steps`, `checkpoint_eval_metric`, `checkpoint_eval_tasks`, `checkpoint_keep_top`, `convergence_window` y `convergence_rel_tol`) emiten una advertencia al cargar en v0.76 y se rechazan a partir de v0.77 según #808, junto con `early_stop_patience` (#761), así que cada una se reporta una vez con la fecha de rechazo.

Hoy puedes usar la puerta de evaluación en vivo para la detección de regresiones y la detención automática:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/chat.jsonl
training:
  epochs: 5
  eval_gate:
    enabled: true
    suite: ./evals/gate.yaml
    every_n_epochs: 1
    regression_threshold: 0.05
    baseline: registry://llama31-chat-v1
    on_regression: stop
```

La puerta se ejecuta en los límites de epoch. Consulta [Entrenamiento con puerta de evaluación](evaluation.md#eval-gated-training) para el formato de la suite y la invocación posterior al entrenamiento.


## GaLore (entrenamiento de parámetros completos eficiente en memoria)

Entrena sin LoRA usando proyección de rango bajo de gradientes: ahorra memoria del optimizador:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: none      # obligatorio: GaLore es incompatible con la cuantización
  use_galore: true
  galore_rank: 128
  galore_update_proj_gap: 200
  galore_scale: 0.25
```

> **Nota:** GaLore requiere `quantization: none` y `backend: transformers` (no unsloth).
