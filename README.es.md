<!-- synced-from: README.md sha256:d3268701c46f0e357132864b930f9532339996c1274ee9424c8a6b26ec3bbb37 -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <strong>Español</strong></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Ajusta y post-entrena LLMs con un solo comando. Sin SSH, sin infierno de configuración.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Sitio web</a> &middot;
  <a href="#inicio-rápido">Inicio rápido</a> &middot;
  <a href="#interfaz-web">Interfaz web</a> &middot;
  <a href="#configuración">Configuración</a> &middot;
  <a href="#documentación">Documentación</a> &middot;
  <a href="docs/commands.md">Comandos</a> &middot;
  <a href="docs/models.md">Modelos</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a> &middot;
  <a href="https://t.me/souptasters">Telegram</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Tests"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Website"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - Fine-tune an 8B LLM on a 4 GB laptop GPU | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup convierte el dolor del fine-tuning de LLMs en un flujo de trabajo simple. Una configuración, un comando y listo.

```bash
pip install "soup-cli[train]"   # agrega [train] para hacer fine-tuning; `soup-cli` a secas es la CLI ligera
soup init --template chat
soup train
```

**Ajusta un modelo de 8B en la GPU de una laptop de 4 GB.** Layer streaming mantiene la base congelada
fuera de VRAM y se la entrega a la GPU capa de decodificador por capa. Medido en una RTX 3050 Laptop de 4 GB:
Llama-3.1-8B-Instruct + NF4 a **119.6 tok/s, 3.32 GB de pico** — bit a bit idéntico a una ejecución
residente normal, y reproducido de forma independiente en una H100 a 113.00 tok/s en los mismos 3.32 GB.
(Ambas cifras se midieron en la v0.72.2, antes de la reparación de corrección de la v0.73.0 que costó
−4.8% en 32B; ninguna se ha vuelto a medir en una tarjeta de 4 GB desde entonces — remedición pendiente en el
issue [#361](https://github.com/MakazhanAlpamys/Soup/issues/361).) Es opcional (`stream_layers: true`)
y sigue en BETA —
[cómo funciona](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[todas las mediciones](benchmarks/) · [paper](https://doi.org/10.5281/zenodo.21771064) ·
**[compruébalo tú mismo en una T4 de Colab gratis](notebooks/proof-4gb.ipynb)** (limita el proceso a
4 GB y luego verifica que un modelo en streaming es bit a bit idéntico a uno normal)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="soup train pre-flight for Llama-3.1-8B on a 4 GB card: a 3.60 GB base store pinned in RAM across 32 layers and two 113 MB VRAM buffers, then a measured peak of 3.32 GB at 119.6 tok/s, stopping short of the 4 GB line (measured on v0.72.2, before the #331 repair; re-measurement pending in issue #361)"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, batch 1, seq 512 en una RTX 3050 Laptop de 4 GB — <b>3.32 GB de pico, 119.6 tok/s</b> (medido en la v0.72.2, antes de la reparación del #331; remedición pendiente en el issue #361). <a href="https://youtu.be/T1LCErE943E">Video completo (90s)</a></sub>
</p>

## ¿Por qué Soup?

Entrenar LLMs sigue siendo doloroso. Incluso los equipos con experiencia dedican 30-50% de su tiempo a
pelear con la infraestructura en lugar de mejorar los modelos. Soup resuelve eso.

- **Cero SSH.** Nunca vuelvas a entrar por SSH a una máquina GPU rota.
- **Una configuración.** Un simple archivo YAML es todo lo que necesitas.
- **Todo automático.** Tamaño de batch, detección de GPU, cuantización — resuelto.
- **Funciona en local.** Entrena en tu propia GPU con QLoRA. Sin nube.

## Novedades

**v0.76.0 — las opciones que Soup aceptaba y nunca aplicaba ahora detienen la carga.** Las opciones que
se validaban y se documentaban, y que nadie leía en la ruta que decían configurar, ahora se rechazan al
cargar, nombrando la opción y qué cambiar; el resto avisa y nombra la versión que las va a rechazar. Una
configuración que antes cargaba puede dejar de cargar: ejecuta
`soup doctor --config soup.yaml` primero.

- **Incompatible: una opción que nunca surtía efecto se rechaza al cargar.** Entre ellas
  `training.quantization_aware: true` (nunca ejecutó entrenamiento consciente de cuantización),
  `training.use_longlora`, `training.ebft_variant`, `training.gdpo_variant`,
  `training.rope_scaling_type` donde ninguna configuración lo lee, `freeze_layers` / `freeze_ratio`
  fuera de la ruta de texto `sft` / `tts` en el backend transformers, `auto_mixed_precision`
  fuera de `sft` / `tts`, y `backend: unsloth` en otras siete tareas que no tienen
  configuración de unsloth. Cada mensaje nombra el campo y qué cambiar. `lr_groups`, `early_stop_patience`,
  `forgetting_eval_steps` y otras perillas que nadie lee ahora avisan al cargar y se rechazarán desde la v0.77.
- **La pérdida de validación ahora se evalúa.** `data.val_split` reservaba filas fuera del entrenamiento en cada
  ejecución, y ningún trainer del backend transformers las evaluaba. Cada tarea que recibe una
  partición ahora la evalúa al final de cada época, o cada N pasos con `training.eval_steps`,
  y registra la pérdida de validación; las ejecuciones tardan unos puntos porcentuales más. **Incompatible para `grpo`, `ppo` y
  `online_dpo`:** por defecto ya no dejan filas fuera (`data.val_split` se ignora con
  una nota), así que todas las filas entrenan.
- **Incompatible: cada `grpo_variant` no estándar aplica la penalización KL `grpo_beta`.** `gspo`,
  `dapo`, `dr_grpo`, `bnpo`, `two_sided` y `rft` validaban `grpo_beta` y la descartaban, así que las
  configuraciones de variantes existentes no reproducirán ejecuciones anteriores. `grpo_beta: 0` ahora corre sin KL y
  omite el forward de referencia.
- **Layer streaming lee su nivel de disco en un hilo de fondo, mediante I/O directo, y puede
  repartirlo en franjas entre discos NVMe.** En un store NF4 de 36 GB con forma de 70B, más grande que la RAM (una máquina RTX 5070
  Laptop de 8 GB), un paso en frío pasó de 110.5 s con la lectura síncrona a 16.2-17.3 s, y
  de 17.4 s a 9.97 s entre dos discos con `SOUP_LAYER_STREAM_STRIPE_DIRS` (una prueba de carga
  sostenida: dos rondas, 1.28x y 1.41x, sin veredicto formal;
  [registro](benchmarks/probe-rtx5070-two-drive-sustained.md)). Un store que cabe en RAM sigue siendo
  más rápido en el nivel de RAM, que es lo que elige `stream_source: auto`. Es opcional (`stream_layers: true`) y
  sigue en BETA; registros: [I/O directo](benchmarks/gate-974-disk-tier-direct-io.md),
  [franjas](benchmarks/gate-two-drive-striping.md).
- **Comandos y opciones nuevas.** `soup adapters check` escanea un adaptador LoRA en busca de fallos silenciosos;
  `soup bench train` entrena un número fijo de pasos y sale con código 1 cuando el modelo no estaba entrenando
  (cubre `task: sft` en el backend transformers; `soup bench` ahora es un grupo y
  `soup bench infer` es el antiguo benchmark de inferencia);
  `soup data clean` sanea y repara un dataset sin tocar el archivo de entrada;
  `soup recipes verify` comprueba en el meta device, sin descargar pesos, que cada receta
  publicada puede adjuntar un adaptador LoRA; `soup rewind` nombra las filas del dataset detrás de un pico de pérdida.
  `soup infer` gana `--batch-size` opcional y un `--cuda-graphs` experimental (PyTorch 2.14 o
  posterior; en una RTX 5070 Laptop, sin fijar, Qwen2.5-1.5B en FP16, la ejecución más lenta con él decodificó
  2.9x más rápido que la más rápida sin él). `soup ab` ejecuta una prueba de dos lados robusta a la varianza, y
  `soup adapters sign --backend sigstore` y `soup attest emit --sign sigstore` firman con
  OIDC sin llaves de Sigstore (`pip install "soup-cli[sigstore]"`).
- **Obsoleto:** `soup export --format awq` y `--format gptq` se eliminarán en la v0.77; ambos
  proyectos de los que dependen están archivados y ninguno de los dos extras se instala junto a `[train]`.

> Sólo Python **3.10–3.12**. En 3.13+, pip solía resolver ruedas de PyTorch no probadas que
> fallan en la extensión nativa antes de que Soup llegue a ejecutarse.

Los destacados más antiguos están en la página de [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

## Inicio rápido

### 1. Instalar

Soup es una aplicación de línea de comandos, así que la instalación más limpia le da su propio
entorno y coloca `soup` en tu `PATH`:

```bash
# Núcleo ligero: CLI + configuración + herramientas de datos, sin PyTorch
pipx install soup-cli
uv tool install soup-cli          # la misma idea, si ya usas uv

# Agrega el stack de entrenamiento (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Todo (train + serve + ui + data) de una vez
pipx install "soup-cli[all]"

# O desde GitHub (último desarrollo)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

¿Ya estás dentro de un virtualenv, un notebook de Colab o una imagen de Docker? Usa `pip`
directamente, con los mismos nombres y extras:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

Usa `pip` en lugar de `pipx` si además quieres `import soup_cli` desde tu propio
código, ya que pipx aísla deliberadamente la aplicación de todo lo demás.

La tabla completa de extras (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) está en
[`docs/models.md`](docs/models.md#optional-extras).

> **¿`error: externally-managed-environment`?** Eso es
> [PEP 668](https://peps.python.org/pep-0668/), no un problema de Soup. Debian 12,
> Ubuntu 23.04 y posteriores impiden que `pip` escriba en el Python del sistema, porque
> `apt` también gestiona esos archivos. `pipx` y `uv tool` lo esquivan dándole a Soup
> su propio entorno, y por eso aparecen primero arriba. `python3 -m venv
> .venv && source .venv/bin/activate` y después `pip` a secas funciona igual de bien.

> **Comillas dobles, no simples.** `"soup-cli[train]"` es la única grafía que funciona en todos los
> shells — `cmd.exe`, PowerShell, bash y zsh. Si copiaste `'soup-cli[train]'` de un tutorial
> antiguo y pip lo rechazó, esa es la razón:
> [por qué, y el error exacto](docs/models.md#quoting-the-extra).

`soup init`, `soup data …` y los demás comandos de datos e inspección funcionan con la instalación ligera.
El fine-tuning (`soup train`) necesita el extra `[train]`.

### 2. Crea una configuración

```bash
soup init                       # asistente interactivo
soup init --template chat       # o parte de una plantilla
```

Plantillas: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Entrena, prueba, publica

```bash
soup train --config soup.yaml                 # LoRA, cuantización, batching — todo resuelto
soup chat  --model ./output                    # habla con tu modelo
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # fusiona el LoRA en la base
soup export --model ./output --format gguf --quant q4_k_m   # GGUF para Ollama / llama.cpp
```

Hay más destinos de exportación (ONNX, TensorRT, AWQ, GPTQ, BitNet) y opciones de despliegue en
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Interfaz web

¿Prefieres un navegador? `soup ui` sirve un panel local para experimentos,
configuración de entrenamiento, métricas en vivo, exploración de datasets y chat con modelos.

```bash
pip install "soup-cli[ui]"
soup ui
# Abre http://127.0.0.1:7860/?token=<token> e imprime la misma URL
```

![Soup Web UI — New Training](docs/assets/web-ui-new-training.png)

[Documentación de la interfaz web](docs/serving-and-export.md#web-ui)

## Configuración

Un `soup.yaml` completo:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # de 2 a 5x más rápido, pip install "soup-cli[fast]"

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
  quantization: 4bit

output: ./output
```

`config/schema.py` es la única fuente de verdad para cada campo. Las opciones avanzadas de datos,
entrenamiento y PEFT se documentan en [Documentación](#documentación).

> **Las claves de configuración desconocidas se rechazan desde la v0.75.** Una clave que ningún modelo declara — un error
> de tipeo como `quantizaton`, o un campo que sólo existe en un Soup más nuevo — antes se validaba
> sin queja y se descartaba, así que la ejecución continuaba simplemente sin aplicar la opción.
> La v0.74 lo reportaba al cargar con el campo que probablemente querías decir; desde la **v0.75** esa misma
> configuración no carga, así que corrige o elimina la clave en vez de confiar en que se
> ignore. Consulta [Claves de configuración desconocidas](docs/backends-and-ops.md#unknown-config-keys).

## Documentación

La referencia completa de funciones vive en [`docs/`](docs/). Empieza aquí:

| Guía | Cubre |
|---|---|
| [Tareas y métodos de entrenamiento](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-entrenamiento, destilación, clasificación, visión/audio/TTS, desaprendizaje, RAFT/RA-DIT, detectores de endurecimiento de bucles |
| [PEFT, contexto largo y eficiencia](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, zoológico de optimizadores y PEFT, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, currículo, auto-tuning |
| [Rendimiento y cuantización](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, formatos de guardado, Cut Cross-Entropy, gradient checkpointing, kernels, offloading de activaciones, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Ingeniería de datos](docs/data.md) | Formatos, el pipeline con paridad Axolotl/LF, herramientas de datos, generación sintética y forge, tarjetas de calidad, herramientas de trazas, datasets remotos, mezcla, DAGs de recetas |
| [Evaluación y sondas](docs/evaluation.md) | Diseño y compuerta de evaluación, entrenamiento con compuerta de evaluación, benchmarks, métricas NLG, calibración, arena Elo, diagnose, sondas de rayos X post-entrenamiento, A/B, drift, tuneabilidad, `soup advise` |
| [Servicio y exportación](docs/serving-and-export.md) | Servidor compatible con OpenAI, inferencia por lotes, benchmarking, merge/export, endpoint de Anthropic Messages, decodificación especulativa (entrena y mide tu propio draft), autopiloto de despliegue, interfaz web, Agent Forge |
| [Adaptadores, registro y gobernanza](docs/adapters-and-governance.md) | Ciclo de vida y gestión de adaptadores, registro de modelos, Soup Cans, el volante de datos (`soup loop`), edición de conocimiento, steering, controles de cadena de suministro |
| [Inicio rápido de cumplimiento y gobernanza](docs/compliance.md) | Plantillas de `init` para HIPAA/SOC2/EU-AI-Act/SR-11-7, procedencia (BOM/attest/repro-receipt), registro de auditoría, air-gap, autogeneración de model card (`soup card`), compuerta de CI (`soup ci init`) |
| [Backends, plataforma y operación](docs/backends-and-ops.md) | Backends MLX/Unsloth, entrenamiento en GPU en la nube con Modal, hubs alternativos, integración con HF Hub, autopiloto, seguimiento de experimentos, plan/apply, lockfiles de entorno, ajuste a hardware, completions, plugins, comandos de utilidad |
| [Referencia de comandos](docs/commands.md) | La lista completa de comandos de `soup` |
| [Modelos compatibles y extras](docs/models.md) | Familias de modelos recomendadas, la guía de tamaño según VRAM, la matriz de extras de pip |

## Formatos de datos

Alpaca, ShareGPT, ChatML, pares de preferencia (DPO / ORPO / SimPO / IPO / KTO), visión, audio,
ASR, texto plano, embedding, RAFT y más — todos detectados automáticamente desde JSONL, JSON, CSV, Parquet o
TXT, así que en la mayoría de los casos apuntas `data.train` a un archivo y nada más cambia. Los esquemas con un
ejemplo resuelto por formato, más el pipeline de datos (URI remotos, streaming, sharding,
intercalado, expansión de vocabulario, ingesta de documentos), están en
[`docs/data.md`](docs/data.md#data-formats).

## Comandos habituales

```bash
soup train  --config soup.yaml        # entrena (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl --output results.jsonl   # inferencia por lotes
soup chat   --model ./output          # chat interactivo
soup serve  --model ./output          # servidor de API compatible con OpenAI
soup ui                               # panel local en el navegador
soup merge  --adapter ./output        # fusiona el LoRA en el modelo base
soup export --model ./output --format gguf           # exporta para desplegar
soup eval   benchmark --model ./output               # evalúa
soup data   inspect ./data/train.jsonl               # estadísticas del dataset
soup recipes list                     # más de 100 recetas de modelos listas para usar
soup autopilot --model <id> --data d.jsonl --goal chat  # sin configuración
soup doctor                           # comprueba GPU / dependencias / entorno
```

La lista completa de comandos está en [`docs/commands.md`](docs/commands.md).

## Modelos compatibles

Soup funciona con **cualquier** modelo de generación de texto del
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) — si carga con
`AutoModelForCausalLM`, funciona, sin cambiar la configuración. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4 y más de 100 otros vienen como recetas listas para usar (`soup recipes list`).

| VRAM | Modelo máximo (QLoRA 4-bit) | Ejemplo |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (completo) o MoE | Mixtral-8x22B, DeepSeek-V3 |

Las tablas completas de modelos y de visión, y la matriz de extras opcionales, están en [`docs/models.md`](docs/models.md).

## Docker

Ejecuta Soup sin instalar CUDA ni PyTorch en local (la imagen se publica en GHCR en cada versión):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # o compila en local
```

## Requisitos

- Python 3.10, 3.11 o 3.12 (esas son las versiones que prueba CI; 3.13+ todavía no es compatible
  porque el stack de PyTorch no se ha validado ahí)
- GPU con CUDA (recomendado), Apple Silicon (MPS) o CPU (experimental — muy lento)
- 8 GB+ de VRAM para modelos de 7B con QLoRA

Todas las tareas de entrenamiento corren en CPU para pruebas (cuantización desactivada automáticamente). Los extras
opcionales (`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) se listan en
[`docs/models.md`](docs/models.md#optional-extras).

## Solución de problemas

```bash
soup doctor    # GPU, recursos del sistema, dependencias y versión en un solo lugar
```

Ruedas de CUDA, discrepancias de versión: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Desarrollo

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # pruebas unitarias (rápidas, sin GPU)
pytest tests/ -m smoke -v          # pruebas smoke (descarga un modelo diminuto, entrena)
pytest tests/ -m gpu --no-cov -v   # pruebas de GPU (necesitan una tarjeta CUDA; reporta resultados, ver CONTRIBUTING.md)

pre-commit install                 # opcional: lint+format de ruff al commitear
```

Mira [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo completo y [SECURITY.md](SECURITY.md) para
reportar una vulnerabilidad. La telemetría es estrictamente opcional (`SOUP_TELEMETRY=1`, desactivada por defecto; consulta la [Política de privacidad](docs/backends-and-ops.md#privacy-policy)).

## Apoya a Soup

Soup es Apache-2.0 y gratis — y seguirá siéndolo. Se construye y se mantiene en abierto desde una
única laptop de 4 GB, y por eso cada número de rendimiento de esta documentación está medido, no
afirmado.

Si Soup te ahorró una ejecución de entrenamiento, [darle una estrella al repositorio](https://github.com/MakazhanAlpamys/Soup)
es lo que más ayuda, y no cuesta nada. Si quieres financiar el trabajo directamente:

**[❤️ Donar](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — pago único, del monto que quieras (usa
*Change amount* en la página de pago). Los pagos los procesa Stripe bajo el negocio registrado del
mantenedor, **MePlay, Inc.** — ese nombre, no "Soup", es el que aparece en la página de pago
y en el estado de cuenta de tu tarjeta.

Las donaciones compran tiempo de GPU para el trabajo que depende de hardware — multi-GPU, validación de 8B+,
Apple Silicon — que una sola laptop de 4 GB no puede alcanzar.

La otra forma de mover exactamente esos puntos es el **hardware mismo**. Van detrás de compuertas honestas
de "requiere \<hardware\>" en lugar de afirmaciones no verificadas, así que si tienes acceso a una máquina
más grande — o créditos de GPU sin usar — correr uno de los
issues [`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
y publicar los números ayuda tanto como financiar el tiempo de GPU. Esos issues dicen
exactamente qué está bloqueado hoy por falta de hardware.

## Colaboradores

Construido por la comunidad ❤️ — gracias a todos los que han contribuido. Ver
[CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Contributors](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## Contacto

Los errores y las peticiones de funciones van al
[rastreador de issues](https://github.com/MakazhanAlpamys/Soup/issues), las preguntas a
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions) — ambas se responden más rápido
y ayudan a la siguiente persona con el mismo problema.

Para chat en vivo, ayuda de configuración y todo lo que se lee mejor como conversación, únete al
[Discord](https://discord.gg/dgd2pJcjwP) o a la [comunidad de Telegram](https://t.me/souptasters).
Todo lo que debería seguir siendo encontrable en seis meses
pertenece a Issues o Discussions — una respuesta en Discord ayuda a una persona, un issue ayuda a todos
los que choquen con lo mismo. El [Código de Conducta](CODE_OF_CONDUCT.md) también aplica ahí.

Para todo lo que no encaja en público — reportes de seguridad (ver [SECURITY.md](SECURITY.md)),
temas del Código de Conducta o prensa — escribe a **team@trysoup.dev**. Esa es la dirección del proyecto
y la correcta para cualquier cosa relacionada con Soup. **makazanalpamys@gmail.com** es la dirección personal del
mantenedor; llega a la misma persona y es un buen respaldo.

## Cómo citar Soup

Layer streaming — entrenar un modelo de 8B en la GPU de una laptop de 4 GB haciendo streaming de la base congelada
desde la RAM del host, una capa de decodificador a la vez — se describe en un preprint, junto con el protocolo
de corrección que verifica una ejecución en streaming contra una residente (forward y backward declarados
por separado, porque son dos afirmaciones y no una).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**La versión 3 (13 de agosto de 2026) es la actual.** El título y la afirmación no cambian — 8B en 4 GB —
y ningún número medido ha cambiado desde la v1. Lo que hace la v3 es **retirar una explicación que habíamos
publicado**, que es también la forma más corta de describir para qué sirve el paper:

- **Retractado en la v3: "layer streaming está limitado por la transferencia host-a-dispositivo, no por la GPU."**
  Eso era una *inferencia* a partir de la réplica en H100 de abajo, y nunca se había medido. Lo
  medimos el 11 de agosto y es falso en la configuración publicada: borrar todos los
  bytes host-a-dispositivo compra **1.4%**, el stream de cómputo espera por una copia el **0.20%** del
  paso, y el paso corre al **71.3%** del techo de GEMM de esa tarjeta en la misma sesión. El mayor
  costo específico del streaming es la des-cuantización NF4 por capa, con 9.8%
  ([el registro](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). Todas las mediciones se sostienen;
  la réplica sobrevive en una forma más débil — la restricción es común a ambas máquinas y no es el
  cómputo de la GPU.
- **Réplica en hardware que no se parece en nada al original** (agregada en la v2): 119.6 tok/s en la RTX
  3050 contra una mediana de 113.00 en una H100, con el mismo pico de 3.32 GB. Ambas son anteriores a la
  reparación del #331; la remedición en 4 GB está pendiente en el issue #361.
- **Un defecto silencioso de gradiente incorrecto, encontrado y reparado.** En NF4 por encima de ~165 MiB por capa el
  forward seguía siendo bit a bit exacto y la curva de pérdida se veía sana mientras los gradientes estaban mal. La
  causa está nombrada en la librería de la que depende y reportada allí; la reparación está validada contra controles
  en 32B y 72B reales.
- **Exactitud bit a bit en tamaños de modelo reales** en lugar de juguetes de tres capas: forward de 0.5B a 72B,
  backward en 8B y 14B.
- **Calidad del modelo entrenado, medida por primera vez**, e indistinguible de una ejecución residente.
- **Una comparación contra DeepSpeed** — incluido el resultado que no nos favorece: ocho tarjetas
  de ZeRO-3 son más lentas que una tarjeta entrenando en residente.
- **La sección de limitaciones reescrita**: de los diez puntos de la v1, uno se cerró y cuatro más se acotaron,
  y se agregaron siete nuevos.

Cita la versión que usaste. `10.5281/zenodo.21771064` es el DOI de concepto y siempre resuelve a
la última versión (hoy la v3); la v1 y la v2 siguen siendo citables con sus propios DOI de versión y no se
editan — la retractación de arriba es una versión nueva precisamente para que el registro de lo que afirmamos,
y cuándo, siga intacto.

Los registros de medición detrás de cada número del paper están en [`benchmarks/`](benchmarks/), publicados
tal como se escribieron — incluidos los fallos, las suposiciones que resultaron equivocadas y los números que
se midieron y luego se descartaron.

```bibtex
@misc{makazhan2026exact,
  title        = {Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU},
  author       = {Makazhan, Alpamys},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3},
  doi          = {10.5281/zenodo.21918325},
  url          = {https://doi.org/10.5281/zenodo.21918325}
}
```

## Licencia

[Apache-2.0](LICENSE). Copyright © los colaboradores de Soup.
