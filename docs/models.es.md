# Modelos compatibles y extras opcionales

> 🇪🇸 Traducción al español de [`models.md`](models.md). Ante cualquier duda, manda el original en inglés.

[← Volver al README de Soup](../README.md)

> Familias de modelos recomendadas, la guía de tamaño según VRAM y la matriz de extras de pip.

> Los tamaños de modelo en las recetas representan el recuento total de parámetros (todos los parámetros residentes en los modelos MoE, que responden a los requisitos de VRAM de la tarjeta).

## Modelos compatibles

Soup funciona con **cualquiera** de los **340,000+** modelos de generación de texto de [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). Si un modelo admite `AutoModelForCausalLM`, funciona con Soup — sin necesidad de cambiar nada de configuración.

### Modelos recomendados

| Familia de modelos | Modelos | Tamaños | Ideal para |
|---|---|---|---|
| **Llama 4** | Llama-4-Scout-17B, Llama-4-Maverick-17B | 109B-400B | General, multilingüe |
| **Llama 3.x** | Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct | 1B–70B | Chat, seguimiento de instrucciones |
| **Llama 3.2 Vision** | Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision | 11B–90B | Comprensión de imágenes |
| **Gemma 3** | Gemma-3-4B-IT, Gemma-3-9B-IT, Gemma-3-27B-IT | 4B–27B | Eficiente, multilingüe |
| **Qwen 3.5 / 3.6 / 3.8** | Qwen3.5-0.8B…397B-A17B, Qwen3.6-27B/35B-A3B, Qwen3.8-27B | 0.8B–397B | Contexto de 262K, visión nativa, MoE |
| **Qwen 3** | Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B | 0.6B–235B | Razonamiento, código, MoE |
| **Qwen 2.5** | Qwen2.5-7B-Instruct, Qwen2.5-Coder-32B-Instruct, Qwen2.5-Math-7B-Instruct | 0.5B–72B | Código, matemáticas |
| **DeepSeek** | DeepSeek-R1-Distill-Llama-8B, DeepSeek-V3-0324, DeepSeek-V4-Flash/Pro | 1.5B–1.6T | Razonamiento (GRPO), código, MoE |
| **GLM** | GLM-4.6, GLM-5, GLM-5.1 | 357B-754B | Chino + inglés, MoE |
| **Kimi** | Kimi-K2, Kimi-K2.5, Kimi-K2.6 | ~1T (MoE) | Agéntico de contexto largo, MoE |
| **MiniMax** | MiniMax-M2, MiniMax-M3 | 230B–428B | Agéntico, MoE (licencia comunitaria) |
| **Phi-4** | Phi-4-14B, Phi-4-mini-reasoning | 3.8B–14B | Razonamiento compacto |
| **Mistral** | Mistral-7B-Instruct-v0.3, Mistral-Small-24B | 7B–24B | Rápido, eficiente |
| **Mixtral** | Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B | 47B–141B | Arquitectura MoE |
| **CodeLlama** | CodeLlama-7b-Instruct-hf, CodeLlama-34b-Instruct | 7B–34B | Generación de código |
| **StarCoder 2** | StarCoder2-15B, StarCoder2-7B | 3B–15B | Autocompletado de código |
| **Yi** | Yi-1.5-34B-Chat, Yi-1.5-9B-Chat | 6B–34B | Chat multilingüe |
| **InternLM 3** | InternLM3-8B-Instruct | 8B | Chino + inglés |
| **Falcon** | Falcon-11B, Falcon-40B-Instruct | 7B–180B | Pesos abiertos |

**Mistral-Large-3 no se distribuye como receta.** `mistralai` lo publica en el diseño nativo de Mistral (`params.json` + `consolidated-*.safetensors`), sin `config.json`, que `AutoConfig` no puede leer, y ninguna bandera cambia eso. Existe una conversión en formato Hugging Face en el Hub como `FriendliAI/Mistral-Large-3-675B-Instruct-2512-BF16`; es una resubida de terceros, así que Soup no apunta ninguna receta a ella (#1145).

Los checkpoints Qwen3.5, Qwen3.6 y Qwen3.8 anuncian en el Hub una arquitectura multimodal de generación
condicional, pero las recetas del catálogo de Soup son deliberadamente solo de texto.
Su `modality: text` explícito selecciona `AutoModelForCausalLM`, que instancia el
decodificador de lenguaje sin la torre visual. Esto es apropiado para SFT solo de texto,
pre-entrenamiento y datos GRPO; no es un ajuste fino multimodal completo. El backend de Transformers
requiere Transformers 5.16.1 o posterior, que además aporta el
decodificador de texto `qwen4_exp` que usa Qwen3.8-Flash-Next (#571). En runtimes de Torch
cuyo operador `scatter` todavía requiere índices `int64`, Soup amplía solo los
tensores de índice locales del indexador QSA antes de la inyección de LoRA; Torch y la clase
Transformers upstream permanecen sin cambios.

Con `training.stream_layers: true`, el decodificador de texto Qwen4-Exp se admite por
su propia ruta con pruebas de paridad. Su enorme tabla PLE de N-gramas congelada se excluye
del fragmento (shard) de la capa del decodificador y se puede servir desde los safetensors originales
con `training.stream_ngram_source: disk`. Las lecturas son dispersas y de solo lectura; Soup
no crea una segunda copia densa de PLE en su caché de fragmentos.

Los checkpoints Transformers densos y los paquetes Qwen4 afines de oMLX/oQ se aceptan por
esta ruta SFT con layer streaming. Para oQ, Soup des-cuantiza cada capa de decodificador congelada
hacia la caché de flujo reutilizable y mapea los tensores expertos Switch-MLP fusionados al
decodificador de texto de Transformers. La tabla PLE empaquetada permanece en el checkpoint
original y solo se des-cuantizan las filas seleccionadas, así que oQ requiere
`stream_ngram_source: disk` (o `auto`). Los pesos de la torre de visión y de MTP no forman
parte del CausalLM solo de texto y se ignoran. `training.quantization` debe
permanecer en `none`; la paridad de NF4 en streaming es aparte y no se valida para Qwen4.

El oráculo de paridad Qwen4 residente frente a streaming cubre actualmente CPU float32.
La paridad BF16 en CUDA no se ha medido, así que la preparación para producción con BF16 en CUDA
sigue pendiente aunque el runtime selecciona BF16 ahí. El checkpoint oQ de producción
de 176.9B de parámetros completó la construcción de caché y la configuración de entrenamiento en una
M4 Max, pero no un paso completo del optimizador; no está validado como entrenable en un
Mac de 128 GiB. Consulta el [registro de compuerta de M4 Max](../benchmarks/gate-qwen4-ple-m4-max.md).

### Modelos de visión (con `modality: vision`)

| Modelo | Tamaño | Formatos compatibles |
|---|---|---|
| LLaMA-3.2-11B-Vision-Instruct | 11B | LLaVA, ShareGPT4V |
| Qwen2-VL-7B-Instruct | 7B | LLaVA, ShareGPT4V |
| Pixtral-12B-2409 | 12B | LLaVA, ShareGPT4V |

### Modelos ASR (`task: asr`, Whisper — v0.71.32)

| Receta | Base | Tamaño | Estado |
|---|---|---|---|
| `whisper-tiny-asr` | openai/whisper-tiny | 39M | Funciona en 4 GB |
| `whisper-base-asr` | openai/whisper-base | 74M | Funciona en 4 GB |
| `whisper-large-v3-asr` | openai/whisper-large-v3 | 1.5B | Solo parseo (GPU más grande) |

Las filas son `{"audio": <path>, "text": <transcript>}` con `data.format: asr`. Consulta
[Entrenamiento → ajuste fino de ASR](training.md).

### Guía rápida de tamaños

| VRAM | Modelo máximo (QLoRA 4-bit) | Ejemplo |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (completo) o MoE | Mixtral-8x22B, DeepSeek-V3 |

> **Nota:** Soup detecta tu GPU automáticamente y estima el tamaño de batch óptimo. Usa `soup doctor` para revisar tu configuración.

## Extras opcionales

> **Python 3.10, 3.11 o 3.12.** Desde la v0.73.0 el paquete declara
> `requires-python = ">=3.10,<3.13"`. Esas son exactamente las versiones que prueba CI. Sin el
> límite superior, pip en 3.13+ resolvía ruedas de PyTorch que nadie había validado, y el fallo
> no era un mensaje de error de Soup — era un crash del cargador dentro de `c10.dll` / `libc10.so` antes de que
> corriera cualquier código de Soup. Si estás en 3.13+, crea un entorno 3.12; la compatibilidad se amplía cuando CI lo haga.

### Cómo entrecomillar el extra

**Usa comillas dobles.** `pip install "soup-cli[train]"` es la única forma que funciona en todos
los shells — `cmd.exe`, PowerShell, bash y zsh. Todos los comandos de la tabla de abajo la usan.

Los tutoriales y videos más antiguos (incluidos algunos nuestros) muestran la forma con comillas
simples `pip install 'soup-cli[train]'`. Esa es sintaxis de bash / zsh / PowerShell, y falla en
`cmd.exe` de Windows, que no tiene entrecomillado con comillas simples y pasa las comillas directamente a pip:

```
ERROR: Invalid requirement: "'soup-cli[train]'": Expected package name at the start of dependency specifier
```

Si te topas con eso, cambia el `'` por `"` — pip está rechazando un carácter de comilla literal, no hay
nada malo con el paquete. (Quitar las comillas por completo también funciona en Windows, pero entonces zsh lee
`[train]` como un glob y falla.)

### La tabla de extras

El `pip install soup-cli` básico es una instalación ligera — la CLI, el sistema de configuración y las herramientas de datos, sin
PyTorch. Agrega `[train]` para ajustar, o instala otros extras solo cuando los necesites:

Cada fila de abajo está escrita con `pip` y funciona tal cual dentro de un virtualenv, un notebook de Colab
o una imagen de Docker. Fuera de uno, en Debian 12 / Ubuntu 23.04 o posterior, `pip` se niega con
`error: externally-managed-environment`; sustituye `pip` por `pipx` o `uv tool` y la forma de los extras
no cambia. Consulta [la sección de instalación del README](../README.md#1-install).

| Extra | Instalación | Qué agrega |
|---|---|---|
| `train` | `pip install "soup-cli[train]"` | Stack de entrenamiento: torch, transformers, peft, trl, datasets, bitsandbytes, accelerate |
| `all` | `pip install "soup-cli[all]"` | `train` + `serve` + `ui` + `data` de una sola vez |
| `fast` | `pip install "soup-cli[fast]"` | Backend Unsloth (2-5x más rápido, menos VRAM) |
| `vision` | `pip install "soup-cli[vision]"` | Ajuste fino de visión / multimodal (Pillow) |
| `audio` | `pip install "soup-cli[audio]"` | Ajuste fino de audio / voz (librosa, soundfile) |
| `mlx` | `pip install "soup-cli[mlx]"` | Backend SFT autónomo para Apple Silicon con datos locales; `[train]` no es obligatorio |
| `qat` | `pip install "soup-cli[qat]"` | torchao: entrenamiento FP8 (`quantization_aware: fp8`), atención FP8, NVFP4, los optimizadores `ao_adamw_*` y `soup export --format torchao` |
| `serve` | `pip install "soup-cli[serve]"` | Servidor de inferencia (FastAPI + uvicorn) |
| `serve-fast` | `pip install "soup-cli[serve-fast]"` | Backend de inferencia vLLM (2-4x de rendimiento) |
| `sglang` | `pip install "soup-cli[sglang]"` | Backend de inferencia SGLang |
| `ui` | `pip install "soup-cli[ui]"` | Interfaz web + servidor de inferencia |
| `tui` | `pip install "soup-cli[tui]"` | Panel a pantalla completa de Textual (`soup tui`) |
| `eval` | `pip install "soup-cli[eval]"` | Evaluación de benchmarks (lm-evaluation-harness) |
| `aider` | `pip install "soup-cli[aider]"` | CLI de Aider; la evaluación Polyglot también necesita la imagen Docker de Aider compilada desde el código fuente |
| `data` | `pip install "soup-cli[data]"` | Deduplicación (MinHash mediante datasketch) |
| `data-pro` | `pip install "soup-cli[data-pro]"` | Detección de idioma + PII (langdetect, presidio) |
| `deepspeed` | `pip install "soup-cli[deepspeed]"` | Entrenamiento multi-GPU (DeepSpeed ZeRO) |
| `liger` | `pip install "soup-cli[liger]"` | Operaciones fusionadas de Liger Kernel |
| `ring-attn` | `pip install "soup-cli[ring-attn]"` | Ring FlashAttention (paralelismo de secuencia) |
| `onnx` / `tensorrt` | `pip install "soup-cli[onnx]"` | Exportación a ONNX / TensorRT-LLM |
| `awq` / `gptq` | `pip install "soup-cli[awq]"` | Exportación cuantizada AWQ / GPTQ; no se puede instalar junto a `[train]` (ver abajo) |
| `trackers` | `pip install "soup-cli[trackers]"` | Registro con MLflow / SwanLab / Trackio |
| `remote` | `pip install "soup-cli[remote]"` | Datasets remotos (s3 / gs / az / oci) |
| `dev` | `pip install "soup-cli[dev]"` | Pruebas + lint + types (pytest, ruff, mypy, pre-commit) |

**`[awq]` y `[gptq]` no se pueden instalar junto a `[train]`.** Se midió que la exportación AWQ solo funciona con `transformers` 4.52.4 o anterior, mientras que `[train]` requiere `transformers>=5.16.1`; `auto-gptq` no publica ninguna rueda para Python 3.12; y ambos proyectos upstream (AutoAWQ y AutoGPTQ) están archivados. Ejecuta la exportación AWQ o GPTQ desde un entorno aparte. Ambos formatos de exportación están obsoletos y se eliminarán en Soup v0.77 ([#338](https://github.com/MakazhanAlpamys/Soup/issues/338)); hasta entonces `soup export --format awq` y `--format gptq` arrancan con un aviso que nombra los formatos a los que migrar (`gguf`, `onnx`, `tensorrt`, `bitnet`, `tq1_0`).

La lista completa y autoritativa de extras está en [`pyproject.toml`](../pyproject.toml).
