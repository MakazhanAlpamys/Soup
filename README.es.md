<!-- synced-from: README.md sha256:5b33b108c15916db92b052ddf457ca8000e2647ea124924cb08287eac083f19f -->
<p align="center">🌍 <a href="README.md">English</a> | <a href="README.tr.md">Türkçe</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <strong>Español</strong> | <a href="README.pt.md">Português</a></p>

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Haga fine-tuning y postentrenamiento de LLM con un solo comando. Sin SSH, sin el infierno de la configuración.</strong>
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
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Descargas"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Licencia Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Pruebas"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Sitio web"></a>
  <a href="https://discord.gg/dgd2pJcjwP"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://t.me/souptasters"><img src="https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white" alt="Telegram"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - Fine-tuning de un LLM de 8B en la GPU de una laptop de 4 GB | Product Hunt" width="250" height="54">
    </picture>
  </a>
  <a href="https://trendshift.io/repositories/98395?utm_source=repository-badge&amp;utm_medium=badge&amp;utm_campaign=badge-repository-98395" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/98395" alt="MakazhanAlpamys/Soup | Trendshift" width="250" height="55">
  </a>
</p>

---

Soup convierte la complicación del fine-tuning de LLM en un flujo de trabajo sencillo. Una configuración, un comando, listo.

```bash
pip install "soup-cli[train]"   # add [train] to fine-tune; bare `soup-cli` is the light CLI
soup init --template chat
soup train
```

**Haga fine-tuning de un modelo de 8B en la GPU de una laptop de 4 GB.** El layer streaming mantiene la base congelada fuera de
la VRAM y se la entrega a la GPU una capa del decodificador a la vez. Medido en una RTX 3050 Laptop de 4 GB:
Llama-3.1-8B-Instruct + NF4 a **119.6 tok/s, con un pico de 3.32 GB**, idéntico bit a bit a una ejecución
normal con todo residente en memoria, y reproducido de forma independiente en una H100 a 113.00 tok/s con los mismos 3.32 GB.
(Ambas cifras se midieron en la v0.72.2, antes de la corrección de exactitud de la v0.73.0, que supuso una
pérdida de 4.8 % de rendimiento en 32B; ninguna se ha vuelto a medir en una tarjeta de 4 GB desde entonces: la nueva medición está pendiente en
el issue [#361](https://github.com/MakazhanAlpamys/Soup/issues/361).) Es opcional (`stream_layers: true`)
y sigue en BETA:
[cómo funciona](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[todas las mediciones](benchmarks/) · [artículo](https://doi.org/10.5281/zenodo.21771064) ·
**[compruébelo usted mismo en una Colab T4 gratuita](notebooks/proof-4gb.ipynb)** (limita el proceso a
4 GB y luego verifica que un modelo en streaming es idéntico bit a bit a uno normal)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="Verificación previa de soup train para Llama-3.1-8B en una tarjeta de 4 GB: un almacén base de 3.60 GB fijado en RAM a lo largo de 32 capas y dos búferes de VRAM de 113 MB, y luego un pico medido de 3.32 GB a 119.6 tok/s, sin llegar a la línea de los 4 GB (medido en la v0.72.2, antes de la corrección #331; nueva medición pendiente en el issue #361)"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, batch 1, secuencia de 512 en una RTX 3050 Laptop de 4 GB — <b>pico de 3.32 GB, 119.6 tok/s</b> (medido en la v0.72.2, antes de la corrección #331; nueva medición pendiente en el issue #361). <a href="https://youtu.be/T1LCErE943E">Video completo (90 s)</a></sub>
</p>

## ¿Por qué Soup?

Entrenar LLM sigue siendo un dolor de cabeza. Incluso los equipos con experiencia dedican entre el 30 y el 50 % de su tiempo a pelear
con la infraestructura en lugar de mejorar los modelos. Soup resuelve eso.

- **Cero SSH.** Nunca más tendrá que conectarse por SSH a un equipo con la GPU averiada.
- **Una sola configuración.** Basta con un sencillo archivo YAML.
- **Todo automático.** Tamaño de batch, detección de GPU, cuantización: todo resuelto.
- **Funciona en local.** Entrene en su propia GPU con QLoRA. No se necesita la nube.

## Novedades

**v0.75.0: el mismo `soup.yaml` entrenaba en MLX una receta distinta de la de transformers,
sin avisar.** Seis opciones de entrenamiento se validaban, se documentaban y se aceptaban, pero en
ese backend nada las leía. **Los 60 pull requests de esta versión vinieron de personas distintas del
mantenedor**: 22 colaboradores en total.

- **Cambio incompatible: una clave de configuración desconocida ahora rechaza la carga.** La v0.74
  advertía y fijaba esta versión como plazo. Un error tipográfico como `quantizaton`, o una clave que solo existe en una
  versión más reciente de Soup, antes se descartaba y la ejecución seguía sin aplicar ese ajuste; ahora
  falla en la CLI (código de salida 1) y en la API (`ValueError`), e indica el campo que probablemente
  quería escribir. El detector aplica el remapeo de `lora:` desde la raíz que el esquema admite
  desde la v0.40.1, de modo que esa forma de escribirlo se acepta, no se rechaza; los dos archivos de
  `soup fetch examples` que la usaban pasaron al formato canónico `training.lora`. Todas las recetas y plantillas se cargan
  sin problemas, a los nombres de clave se les aplica escape antes de llegar a la terminal y el análisis está acotado.
- **MLX respeta la configuración que aceptó.** `train_on_responses_only`, `warmup_ratio` /
  `scheduler` / `weight_decay` / `optimizer`, `max_grad_norm`, `gradient_accumulation_steps`
  y `gradient_checkpointing` se validaban y luego se descartaban con `backend: mlx`. Solo
  8 de los 32 nombres de optimizador tienen equivalente en MLX; los otros 24 se rechazan por su nombre
  en lugar de convertirse en silencio en AdamW. MLX ahora también alimenta el panel en vivo, el rastreador y
  `soup ui`, y `soup doctor --config` enumera los ajustes que un backend no lee.
- **La pérdida de validación no se registraba en ningún lado.** Se calculaba en todos los backends y se descartaba:
  ni columna de métricas, ni campo en los eventos, ni nada en el panel. Ahora se registra, se transmite
  y se muestra.
- **Cambio incompatible: `grpo_variant: gspo` es el objetivo a nivel de secuencia publicado**
  (arXiv:2507.18071), que reemplaza una heurística de centrado por columnas en la que un token de relleno
  también desplazaba el gradiente de todas las filas que compartían su columna. Las configuraciones gspo existentes no
  reproducirán ejecuciones anteriores.
- **Los endpoints de lectura de la interfaz web y SSE exigen autenticación**, con tickets de corta duración y de un solo uso
  en lugar de un token en la cadena de consulta; `--public` ya no sirve `/docs` ni
  `/openapi.json` a la red local; y un subproceso de entrenamiento ya no se cuelga cuando nadie
  lee su salida.
- **`torch>=2.6.0`** cierra la limitación conocida de la v0.74.0: con 2.5.1, `trl>=0.29` no podía
  importarse y todos los entrenadores de preferencias quedaban inutilizados. También se corrigió que `training.loraplus_lr_ratio`
  hacía fallar todas las ejecuciones que lo definían, y que `packing: true` generaba un error con TRL 0.29.

> Solo Python **3.10–3.12**. En 3.13+, pip solía resolver wheels de PyTorch sin probar, que
> se bloquean en la extensión nativa incluso antes de que Soup llegue a ejecutarse.

Los aspectos destacados de versiones anteriores están en la página de [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

## Inicio rápido

### 1. Instalación

Soup es una aplicación de línea de comandos, así que la instalación más limpia le da su propio
entorno y deja `soup` en su `PATH`:

```bash
# Light core: CLI + config + data tools, no PyTorch
pipx install soup-cli
uv tool install soup-cli          # same idea, if you already use uv

# Add the training stack (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Everything (train + serve + ui + data) in one shot
pipx install "soup-cli[all]"

# Or from GitHub (latest dev)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

¿Ya está dentro de un virtualenv, un notebook de Colab o una imagen de Docker? Use `pip`
directamente, con los mismos nombres y extras:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

Use `pip` en lugar de `pipx` si además quiere hacer `import soup_cli` desde su propio
código, ya que pipx aísla deliberadamente la aplicación de todo lo demás.

La tabla completa de extras (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) está en
[`docs/models.md`](docs/models.md#optional-extras).

> **¿Aparece `error: externally-managed-environment`?** Eso es
> [PEP 668](https://peps.python.org/pep-0668/), no un problema de Soup. Debian 12,
> Ubuntu 23.04 y versiones posteriores impiden que `pip` escriba en el Python del sistema, porque
> `apt` también administra esos archivos. `pipx` y `uv tool` lo evitan dándole a Soup
> su propio entorno, y por eso aparecen primero arriba. `python3 -m venv
> .venv && source .venv/bin/activate` seguido de `pip` a secas funciona igual de bien.

> **Comillas dobles, no simples.** `"soup-cli[train]"` es la única forma de escribirlo que funciona en todos los
> shells: `cmd.exe`, PowerShell, bash y zsh. Si copió `'soup-cli[train]'` de un tutorial
> antiguo y pip lo rechazó, esa es la razón:
> [por qué, y el error exacto](docs/models.md#quoting-the-extra).

`soup init`, `soup data …` y los demás comandos de datos e inspección funcionan con la instalación ligera.
El fine-tuning (`soup train`) requiere el extra `[train]`.

### 2. Cree una configuración

```bash
soup init                       # interactive wizard
soup init --template chat       # or start from a template
```

Plantillas: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Entrene, pruebe y publique

```bash
soup train --config soup.yaml                 # LoRA, quantization, batching — all handled
soup chat  --model ./output                    # talk to your model
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # merge LoRA into the base
soup export --model ./output --format gguf --quant q4_k_m   # GGUF for Ollama / llama.cpp
```

Más destinos de exportación (ONNX, TensorRT, AWQ, GPTQ, BitNet) y opciones de despliegue están en
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Interfaz web

¿Prefiere un navegador? `soup ui` sirve un panel local para experimentos,
configuración de entrenamientos, métricas en vivo, exploración de datasets y chat con el modelo.

```bash
pip install "soup-cli[ui]"
soup ui
# Opens http://127.0.0.1:7860
```

![Soup Web UI — New Training](docs/assets/web-ui-new-training.png)

[Documentación de la interfaz web](docs/serving-and-export.md#web-ui)

## Configuración

Un `soup.yaml` completo:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install "soup-cli[fast]"

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

`config/schema.py` es la única fuente de verdad de cada campo. Las opciones avanzadas de datos, entrenamiento
y PEFT están documentadas en [Documentación](#documentación).

> **Las claves de configuración desconocidas se rechazan desde la v0.75.** Una clave que ningún modelo declara
> (un error tipográfico como `quantizaton`, o un campo que solo existe en una versión más reciente de Soup) antes pasaba
> la validación y se descartaba, de modo que la ejecución continuaba sin aplicar ese ajuste.
> La v0.74 lo advertía al cargar la configuración, junto con el campo que probablemente quería escribir; desde la **v0.75** esa misma
> configuración no se carga, así que corrija o elimine la clave en lugar de confiar en que se
> ignore. Consulte [Claves de configuración desconocidas](docs/backends-and-ops.md#unknown-config-keys).

## Documentación

La referencia completa de funciones está en [`docs/`](docs/). Comience aquí:

| Guía | Contenido |
|---|---|
| [Tareas y métodos de entrenamiento](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, preentrenamiento, destilación, clasificación, visión/audio/TTS, unlearning, RAFT/RA-DIT, detectores de robustecimiento de bucles |
| [PEFT, contexto largo y eficiencia](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, catálogo de optimizadores y métodos PEFT, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, curriculum, ajuste automático |
| [Rendimiento y cuantización](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, formatos de guardado, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Ingeniería de datos](docs/data.md) | Formatos, el pipeline con paridad Axolotl/LF, herramientas de datos, generación sintética y forge, scorecards de calidad, herramientas de trazas, datasets remotos, mezcla, DAG de recetas |
| [Evaluación y sondas](docs/evaluation.md) | Diseño y gate de evaluación, entrenamiento controlado por evaluación, benchmarks, métricas NLG, calibración, arena Elo, diagnóstico, sondas X-ray posteriores al entrenamiento, A/B, deriva, ajustabilidad, `soup advise` |
| [Servicio y exportación](docs/serving-and-export.md) | Servidor compatible con OpenAI, inferencia por lotes, benchmarking, merge/exportación, endpoint Anthropic Messages, decodificación especulativa (entrene y mida su propio modelo borrador), autopilot de despliegue, interfaz web, Agent Forge |
| [Adaptadores, registro y gobernanza](docs/adapters-and-governance.md) | Ciclo de vida y gestión de adaptadores, registro de modelos, Soup Cans, el data flywheel (`soup loop`), edición de conocimiento, steering, controles de la cadena de suministro (scan/sign/BOM/attest/audit/airgap) |
| [Inicio rápido de cumplimiento y gobernanza](docs/compliance.md) | Plantillas `init` de HIPAA/SOC2/EU-AI-Act/SR-11-7, procedencia (BOM/attest/repro-receipt), registro de auditoría, air-gap, generación automática de model cards (`soup card`), gate de CI (`soup ci init`) |
| [Backends, plataforma y operaciones](docs/backends-and-ops.md) | Backends MLX/Unsloth, hubs alternativos, integración con HF Hub, autopilot, seguimiento de experimentos, plan/apply, lockfiles de entorno, ajuste al hardware, autocompletado, plugins, comandos de utilidad |
| [Referencia de comandos](docs/commands.md) | La lista completa de comandos de `soup` |
| [Modelos compatibles y extras](docs/models.md) | Familias de modelos recomendadas, la guía de tamaños según VRAM, la matriz de extras de pip |

## Formatos de datos

Alpaca, ShareGPT, ChatML, pares de preferencias (DPO / ORPO / SimPO / IPO / KTO), visión, audio,
ASR, texto plano, embeddings, RAFT y más: todo se detecta automáticamente a partir de JSONL, JSON, CSV, Parquet o
TXT, de modo que en la mayoría de los casos basta con apuntar `data.train` a un archivo y nada más cambia. Los esquemas con un
ejemplo práctico por formato, junto con el pipeline de datos (URI remotos, streaming, sharding,
interleaving, ampliación de vocabulario, ingesta de documentos), están en
[`docs/data.md`](docs/data.md#data-formats).

## Comandos comunes

```bash
soup train  --config soup.yaml        # train (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl   # batch inference
soup chat   --model ./output          # interactive chat
soup serve  --model ./output          # OpenAI-compatible API server
soup ui                               # local browser dashboard
soup merge  --adapter ./output        # merge LoRA into the base model
soup export --model ./output --format gguf           # export for deployment
soup eval   benchmark --model ./output               # evaluate
soup data   inspect ./data/train.jsonl               # dataset stats
soup recipes list                     # 100+ ready-made model recipes
soup autopilot --model <id> --data d.jsonl --goal chat  # zero-config
soup doctor                           # check GPU / deps / environment
```

La lista completa de comandos está en [`docs/commands.md`](docs/commands.md).

## Modelos compatibles

Soup funciona con **cualquier** modelo de generación de texto del
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation): si se carga con
`AutoModelForCausalLM`, funciona, sin cambiar nada en la configuración. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4 y más de 100 otros vienen como recetas listas para usar (`soup recipes list`).

| VRAM | Modelo máximo (QLoRA 4-bit) | Ejemplo |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (completo) o MoE | Mixtral-8x22B, DeepSeek-V3 |

Las tablas completas de modelos y de visión, junto con la matriz de extras opcionales, están en [`docs/models.md`](docs/models.md).

## Docker

Ejecute Soup sin instalar CUDA ni PyTorch en local (la imagen se publica en GHCR con cada versión):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # or build locally
```

## Requisitos

- Python 3.10, 3.11 o 3.12 (son las versiones que prueba la CI; 3.13+ todavía no es compatible
  porque el stack de PyTorch no se ha validado allí)
- GPU con CUDA (recomendado), Apple Silicon (MPS) o CPU (experimental, muy lenta)
- 8 GB+ de VRAM para modelos de 7B con QLoRA

Todas las tareas de entrenamiento pueden ejecutarse en CPU con fines de prueba (la cuantización se desactiva automáticamente). Los extras
opcionales (`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) se enumeran en
[`docs/models.md`](docs/models.md#optional-extras).

## Solución de problemas

```bash
soup doctor    # GPU, system resources, dependencies, and version in one place
```

Wheels de CUDA y versiones incompatibles: [`docs/backends-and-ops.md`](docs/backends-and-ops.md#troubleshooting).

## Desarrollo

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # unit tests (fast, no GPU)
pytest tests/ -m smoke -v          # smoke tests (downloads a tiny model, trains)
pytest tests/ -m gpu --no-cov -v   # GPU tests (need a CUDA card; report results, see CONTRIBUTING.md)

pre-commit install                 # optional: ruff lint+format on commit
```

Consulte [CONTRIBUTING.md](CONTRIBUTING.md) para ver el flujo de trabajo completo y [SECURITY.md](SECURITY.md) para
reportar una vulnerabilidad. La telemetría es estrictamente opcional (`SOUP_TELEMETRY=1`, desactivada por defecto; consulte la [Política de privacidad](docs/backends-and-ops.md#privacy-policy)).

## Apoye a Soup

Soup es Apache-2.0 y gratuito, y seguirá siéndolo. Se desarrolla y mantiene de forma abierta en
una única laptop de 4 GB, y por eso cada cifra de rendimiento de esta documentación está medida en lugar de
simplemente afirmada.

Si Soup le ahorró una ejecución de entrenamiento, [darle una estrella al repositorio](https://github.com/MakazhanAlpamys/Soup)
es lo que más ayuda, y no cuesta nada. Si desea financiar el trabajo directamente:

**[❤️ Donar](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)**: un pago único, por el monto que quiera (use
*Change amount* en la página de pago). Stripe procesa los pagos a nombre de la empresa registrada del
mantenedor, **MePlay, Inc.**; ese nombre, y no "Soup", es el que aparece en la página de pago
y en el estado de cuenta de su tarjeta.

Las donaciones sirven para comprar tiempo de GPU para el trabajo condicionado por el hardware (multi-GPU, validación de 8B+, Apple Silicon)
y que una única laptop de 4 GB no puede abordar.

La otra forma de avanzar justamente en esos puntos es el **propio hardware**. Esos puntos se publican con una
advertencia explícita del tipo «requires \<hardware\>», en lugar de afirmaciones sin verificar, así que, si tiene acceso a una
máquina más grande, o créditos de GPU sin usar, ejecutar lo que pide uno de los issues
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
y publicar los números ayuda tanto como financiar el tiempo de GPU. Esos issues indican
exactamente qué está bloqueado hoy por falta de hardware.

## Colaboradores

Construido por la comunidad ❤️: gracias a todas las personas que han contribuido. Consulte
[CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Colaboradores](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## Contacto

Los errores y las solicitudes de funciones van en el
[gestor de issues](https://github.com/MakazhanAlpamys/Soup/issues); las preguntas, en
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions). En ambos casos se responde más rápido
y se ayuda a la siguiente persona que tenga el mismo problema.

Para el chat en vivo, la ayuda con la instalación y todo lo que se lee mejor como una conversación, únase al
[Discord](https://discord.gg/dgd2pJcjwP) o a la [comunidad de Telegram](https://t.me/souptasters).
Todo lo que deba poder encontrarse dentro de seis meses
va en Issues o Discussions: una respuesta en Discord ayuda a una persona, un issue ayuda a todas
las que se topen con lo mismo. El [Código de conducta](CODE_OF_CONDUCT.md) también se aplica allí.

Para todo lo que no encaje en un espacio público (reportes de seguridad —véase [SECURITY.md](SECURITY.md)—,
asuntos del Código de conducta o prensa), escriba a **team@trysoup.dev**. Es la dirección del proyecto
y la adecuada para cualquier tema relacionado con Soup. **makazanalpamys@gmail.com** es la dirección
personal del mantenedor; llega a la misma persona y sirve como alternativa.

## Cómo citar Soup

El layer streaming —entrenar un modelo de 8B en la GPU de una laptop de 4 GB transmitiendo la base congelada desde
la RAM del host, una capa del decodificador a la vez— se describe en un preprint, junto con el protocolo
de exactitud que verifica una ejecución en streaming frente a una residente (el paso hacia adelante y el hacia atrás se presentan
por separado, porque son dos afirmaciones y no una).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**La versión 3 (13 de agosto de 2026) es la vigente.** El título y la afirmación no cambian (8B en 4 GB),
y ninguna cifra medida ha cambiado desde la v1. Lo que hace la v3 es **retirar una explicación que habíamos
publicado**, que es también la manera más breve de resumir el propósito del artículo:

- **Retractado en la v3: "el layer streaming está limitado por la transferencia de host a dispositivo, no por la GPU."**
  Era una *inferencia* a partir de la réplica en H100 que figura más abajo, y nunca se había medido. La
  medimos el 11 de agosto y resulta falsa en la configuración publicada: eliminar todos los bytes de host a dispositivo
  solo gana **1.4 %**, el stream de cómputo espera una copia durante el **0.20 %** del
  paso, y el paso se ejecuta al **71.3 %** del techo de GEMM de esa tarjeta en la misma sesión.
  El mayor costo específico del streaming es la decuantización NF4 por capa, con 9.8 %
  ([el informe de medición](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). Todas las mediciones se mantienen;
  la réplica sobrevive en una forma más débil: la restricción es común a ambas máquinas y no
  es el cómputo de la GPU.
- **Réplica en un hardware completamente distinto del original** (añadida en la v2): 119.6 tok/s en la RTX
  3050 frente a una mediana de 113.00 en una H100, con el mismo pico de 3.32 GB. Ambas son anteriores a la
  corrección #331; la nueva medición en 4 GB está pendiente en el issue #361.
- **Un defecto silencioso de gradientes erróneos, encontrado y corregido.** En NF4 por encima de ~165 MiB por capa, el
  paso hacia adelante seguía siendo idéntico bit a bit y la curva de pérdida parecía sana, mientras los gradientes eran erróneos. La
  causa está identificada en la biblioteca upstream y se reportó allí; la corrección se verifica contra controles
  en 32B y 72B reales.
- **Exactitud bit a bit en tamaños de modelo reales** en lugar de ejemplos de juguete de tres capas: paso hacia adelante de 0.5B a 72B,
  paso hacia atrás en 8B y 14B.
- **Calidad del modelo entrenado, medida por primera vez**, e indistinguible de una ejecución residente.
- **Una comparación con DeepSpeed**, incluido el resultado que no nos favorece: ocho tarjetas
  con ZeRO-3 son más lentas que una sola tarjeta entrenando con todo residente.
- **La sección de limitaciones, reescrita**: de los diez puntos de la v1, uno se cerró y cuatro más se acotaron,
  y se añadieron siete nuevos.

Cite la versión que utilizó. `10.5281/zenodo.21771064` es el DOI de concepto y siempre redirige a
la última versión (hoy la v3); la v1 y la v2 siguen siendo citables con sus propios DOI de versión y no se
modifican: la retractación anterior es una versión nueva precisamente para que el registro de lo que afirmamos,
y de cuándo, permanezca intacto.

Los registros de medición detrás de cada cifra del artículo están en [`benchmarks/`](benchmarks/), publicados
tal como se escribieron, incluidos los fracasos, las suposiciones que resultaron erróneas y las cifras que
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
