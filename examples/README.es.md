# Ejemplos de Soup

> 🇪🇸 Traducción al español de [`README.md`](README.md). Ante cualquier duda, manda el original en inglés.

Ejemplos de configuración y datasets de muestra.

Aquí viven dos tipos de archivos, y cada uno promete algo distinto:

- **Ejemplos ejecutables** — se parsean y se ejecutan tal cual contra los fixtures incluidos en
  [`data/`](data/). Existen para demostrar que tu configuración funciona de extremo a extremo. Los
  fixtures son de 5–10 filas, así que estas ejecuciones terminan rápido y no producen **nada útil**:
  son una prueba de humo, no entrenamiento. Para eso, apunta `data.train` a datos reales.
- **Plantillas** — configs válidas donde tú aportas los datos. No se ejecutarán hasta
  que lo hagas.

Consulta [data/README.md](data/README.md) para saber qué son los fixtures y dónde conseguir datos reales.

---

## Ejemplos ejecutables

| Config | Tarea | Modelo base | Fixture | Requiere |
|--------|------|------------|---------|-------|
| [`configs/sft_basic.yaml`](configs/sft_basic.yaml) | sft | TinyLlama-1.1B-Chat | `alpaca_tiny.jsonl` (10 filas) | 4 GB |
| [`configs/grpo_reasoning.yaml`](configs/grpo_reasoning.yaml) | grpo | TinyLlama-1.1B-Chat | `reasoning_math.jsonl` (5 filas) | 4 GB |
| [`configs/rlhf_step1_sft.yaml`](configs/rlhf_step1_sft.yaml) | sft | TinyLlama-1.1B-Chat | `alpaca_tiny.jsonl` (10 filas) | 4 GB |
| [`configs/rlhf_step2_reward.yaml`](configs/rlhf_step2_reward.yaml) | reward_model | TinyLlama-1.1B-Chat | `chat_preferences.jsonl` (5 filas) | 4 GB |
| [`configs/rlhf_step3_ppo.yaml`](configs/rlhf_step3_ppo.yaml) | ppo | TinyLlama-1.1B-Chat | `alpaca_tiny.jsonl` (10 filas) | 4 GB |
| [`configs/dpo_example.yaml`](configs/dpo_example.yaml) | dpo | Llama-3.1-8B-Instruct | `dpo_sample.jsonl` (8 filas) | ~13 GB |
| [`configs/dpo_chat.yaml`](configs/dpo_chat.yaml) | dpo | Llama-2-7b-chat | `chat_preferences.jsonl` (5 filas) | ~22 GB |

Los cinco configs de TinyLlama están verificados para pasar la comprobación previa en una tarjeta de 4 GB. Los dos
configs de base más grande no pueden — sus pesos por sí solos son 4.0 GB y 7.0 GB — así que la columna "Requiere"
es lo que `soup train` predice para ellos, margen incluido.

Ejecuta cualquiera de ellos:

```bash
soup train --config examples/configs/sft_basic.yaml
```

Valida una config sin entrenar:

```bash
soup train --config examples/configs/sft_basic.yaml --dry-run
```

Los modelos restringidos (Llama-2, Llama-3.1) necesitan acceso a Hugging Face y un login primero:
`huggingface-cli login`. Los configs de TinyLlama no necesitan ninguno.

`soup train` estima la VRAM máxima antes de cargar nada y se niega a iniciar una ejecución que
predice que no cabrá, imprimiendo el desglose y qué cambiar. Los configs de TinyLlama están
dimensionados para una tarjeta pequeña (`batch_size: 4`, `max_length: 512`) — en una más grande puedes
subir ambos.

### SFT básico

`sft_basic.yaml` — ajuste por instrucciones con LoRA (r=16) sobre filas en formato alpaca. El
config completo más pequeño del repo; una buena base para copiar.

### Alineación de preferencias (DPO)

`dpo_chat.yaml` y `dpo_example.yaml` entrenan ambos con pares `prompt` / `chosen` / `rejected`.
`dpo_example.yaml` añade QLoRA (`quantization: 4bit`) para un entrenamiento eficiente en memoria
sobre una base más grande, y `dpo_beta: 0.1` define con qué fuerza tira la penalización KL
de vuelta hacia el modelo de referencia.

### Razonamiento (GRPO)

`grpo_reasoning.yaml` genera `num_generations: 4` completions por prompt y las puntúa
con la recompensa integrada `accuracy`. Consulta
[docs/training.md](../docs/training.md) para funciones de recompensa personalizadas.

### Pipeline RLHF completo

Tres configs, ejecutados en orden — el paso 3 lee el modelo de recompensa que escribe el paso 2:

```bash
soup train --config examples/configs/rlhf_step1_sft.yaml
soup train --config examples/configs/rlhf_step2_reward.yaml
soup train --config examples/configs/rlhf_step3_ppo.yaml
```

---

## Plantillas

Configs válidas que necesitan tus datos antes de ejecutarse.

### Visión

[`configs/vision_llama.yaml`](configs/vision_llama.yaml) — LLaMA-3.2-11B-Vision sobre
pares de imagen + conversación en formato LLaVA. No hay fixture de visión en este repo
porque no hacemos commit de archivos de imagen, así que `data.train` y `data.image_dir` son
marcadores de posición. Reemplaza ambos, luego:

```bash
soup train --config examples/configs/vision_llama.yaml
```

### Plantillas integradas

`soup init` escribe un `soup.yaml` inicial que completas con tus propios datos:

```bash
soup init --template kto     # unpaired preference (thumbs up/down labels)
soup init --template orpo    # reference-free alignment
soup init --template simpo   # length-normalized preference
soup init --template ipo     # regularized preference (squared hinge)
soup init --template pretrain    # continued pre-training on raw text
soup init --template moe         # Mixture-of-Experts (Qwen3, Mixtral, DeepSeek V3)
soup init --template longcontext # RoPE scaling for 128k+ context
soup init --template embedding   # sentence embeddings (BGE, E5, GTE)
soup init --template vision      # vision-language
soup init --template audio       # audio/speech — needs pip install "soup-cli[audio]"
soup train
```

Lista completa: `soup init --help`. Las plantillas disponibles son `audio`, `bco`, `chat`, `code`,
`embedding`, `eu-ai-act`, `hipaa`, `ipo`, `kto`, `longcontext`, `medical`, `moe`,
`orpo`, `pretrain`, `reasoning`, `rlhf`, `simpo`, `soc2`, `sr-11-7`, `tool-calling`,
`vision`.

---

## Flujo de trabajo con datos sintéticos

Genera datos de entrenamiento a partir de un LLM local, fíltralos y puntúalos, y luego entrena con el resultado.
Guía paso a paso en [synthetic_workflow.md](synthetic_workflow.md); la config con la que entrena
es [synthetic_workflow.yaml](synthetic_workflow.yaml) (una plantilla — lee
`./synth_clean.jsonl`, que el flujo de trabajo produce).

## Demo de mitigación de reward-hacking

[`reward_hacking/rewards.py`](reward_hacking/rewards.py) ofrece funciones de recompensa
sintéticas para la función de mitigación en bucle cerrado (`soup train
--reward-hack-mitigation`): un proxy `length_hack_reward` / `sentinel_reward` manipulable
desacoplado de un `true_score` reservado. Apunta el `training.reward_fn` de una config GRPO a un
`.py` que reexporte uno como `reward_fn`, habilita `reward_hack_detector: info_rm` +
`reward_hack_mitigation: kl_control`, y observa `mitigation_log.jsonl` bajo el directorio de salida
de la ejecución. Consulta
[docs/training.md](../docs/training.md#closed-loop-reward-hacking-auto-mitigation-v07126).

---

## Formatos de dataset

Soup autodetecta y normaliza:

- **Alpaca**: `instruction`, `input`, `output`
- **ShareGPT**: `conversations` con `from`/`value`
- **ChatML**: `messages` al estilo OpenAI con `role`/`content`
- **DPO/ORPO/SimPO/IPO**: `prompt` + `chosen` + `rejected`
- **KTO**: `prompt` + `completion` + `label`
- **LLaVA / ShareGPT4V**: `image` + `conversations`
- **Texto plano**: `.txt` sin procesar o JSONL con un campo `text` (pre-entrenamiento)
- **Audio**: ruta `audio` + `messages`

### Inspeccionar un dataset

```bash
soup data inspect examples/data/alpaca_tiny.jsonl
```

```
                   Dataset Stats
┌────────────────────┬────────────────────────────┐
│ Metric             │ Value                      │
├────────────────────┼────────────────────────────┤
│ Total samples      │ 10                         │
│ Columns            │ instruction, input, output │
│ Avg length (chars) │ 180                        │
│ Min length         │ 60                         │
│ Max length         │ 368                        │
│ Empty fields       │ 0                          │
│ Duplicates         │ 0                          │
└────────────────────┴────────────────────────────┘
```

### Convertir entre formatos

El formato de origen se autodetecta; tú nombras el destino:

```bash
soup data convert examples/data/alpaca_tiny.jsonl \
  --to chatml \
  --output alpaca_as_chatml.jsonl
```

## Estructura de directorios

```
examples/
  configs/                    # YAML configs
    sft_basic.yaml            # runnable
    dpo_chat.yaml             # runnable
    dpo_example.yaml          # runnable
    grpo_reasoning.yaml       # runnable
    rlhf_step1_sft.yaml       # runnable
    rlhf_step2_reward.yaml    # runnable
    rlhf_step3_ppo.yaml       # runnable
    vision_llama.yaml         # template — bring your own images
  data/                       # format fixtures (see data/README.md)
    alpaca_tiny.jsonl
    chat_preferences.jsonl
    dpo_sample.jsonl
    reasoning_math.jsonl
  reward_hacking/             # synthetic reward fns for the mitigation demo
  synthetic_workflow.md       # synthetic-data walkthrough
  synthetic_workflow.yaml     # template config for that walkthrough
```

## Usar tus propios datos

1. Pon tus datos en uno de los formatos compatibles de arriba.
2. Apunta la config a ellos:

```yaml
data:
  train: /path/to/your/data.jsonl
  format: alpaca   # or sharegpt, chatml, dpo, llava, ... — omit for auto-detect
```

3. Entrena:

```bash
soup train --config your_config.yaml
```

## Estructura de la config

Las configs están anidadas: `base`, `task`, `data`, `training`, `output` en el nivel superior.
[`config/schema.py`](../src/soup_cli/config/schema.py) es la única fuente de verdad.

### Mínima

```yaml
base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
task: sft

data:
  train: ./your_data.jsonl
  format: alpaca
  max_length: 512

training:
  epochs: 3
  lr: 5e-4
  batch_size: 16
  lora:
    r: 16
    alpha: 32

output: ./output/
```

### Más opciones

```yaml
base: meta-llama/Llama-2-7b-hf
task: dpo
backend: transformers        # or unsloth, mlx

data:
  train: ./dataset.jsonl
  format: dpo
  max_length: 2048
  val_split: 0.1

training:
  epochs: 2
  lr: 1e-4
  dpo_beta: 0.1
  batch_size: 16             # or "auto" to probe for the largest that fits
  gradient_accumulation_steps: 4
  quantization: 4bit         # 4bit, 8bit, none, gptq, awq, fp8, ...
  scheduler: cosine
  warmup_ratio: 0.1
  gradient_checkpointing: true
  lora:
    r: 64
    alpha: 128
    dropout: 0.05
    target_modules: auto
    # use_dora: true         # Weight-Decomposed LoRA
    # use_rslora: true       # rank-stabilized scaling

output: ./output_advanced/
```

## Después del entrenamiento

### Inferencia por lotes

```bash
soup infer --model ./output_sft_basic/ --input prompts.jsonl --output results.jsonl
```

### Fusionar el adaptador LoRA en un modelo completo

```bash
soup merge --adapter ./output_sft_basic/ --output ./merged_model/
```

### Exportar a GGUF

```bash
soup export --model ./output_sft_basic/ --format gguf --quant q8_0 --output model.gguf
```

Requiere un llama.cpp compilado — consulta [docs/serving-and-export.md](../docs/serving-and-export.md).
`soup export --model ... --deploy ollama` exporta y registra con Ollama en un solo paso.

### Monitorear con Weights & Biases

```bash
pip install wandb
soup train --config your_config.yaml --wandb
```

## Problemas comunes

### "CUDA out of memory"

- Baja `training.batch_size`, o configúralo en `"auto"` para probar el tamaño más grande que quepa
- Añade `training.quantization: 4bit`
- Añade `training.gradient_checkpointing: true`
- Usa un modelo base más pequeño
- O transmite la base capa por capa: `training.stream_layers: true`
  (consulta [docs/training.md](../docs/training.md))

### "Dataset not found"

- Las rutas en una config se resuelven desde el directorio donde ejecutas `soup`, no desde la
  ubicación de la config. Ejecuta desde la raíz del repo, o usa una ruta absoluta.
- Verifica que el archivo se parsee: `soup data validate your_data.jsonl`

### "Model not found on Hugging Face"

- Verifica la ortografía del id del modelo
- Los modelos restringidos (Llama, Gemma) necesitan `huggingface-cli login` y aceptar los términos

### Errores de validación de la config

`soup train` valida antes de hacer cualquier trabajo. `base: Field required` o
`data -> train: Field required` significa que la config usa un diseño plano antiguo — consulta
[Estructura de la config](#estructura-de-la-config) arriba para el anidamiento actual.

## Más información

- [README principal](../README.md)
- [Documentación completa](../docs/README.md)
- [Ingeniería de datos](../docs/data.md)
- [Entrenamiento](../docs/training.md)
- [CONTRIBUTING](../CONTRIBUTING.md)

## ¿Preguntas?

- [GitHub Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)
- [Issues](https://github.com/MakazhanAlpamys/Soup/issues)
- [SECURITY.md](../SECURITY.md)
