<<<<<<< Updated upstream
<p align="center">**🌍 [Türkçe](README.md) | [English](README.en.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)**
=======
<p align="center"><strong>🌍 <a href="README.md">Türkçe</a> | <a href="README.en.md">English</a> | <a href="README.ar.md">العربية</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.ru.md">Русский</a> | <a href="README.es.md">Español</a></strong></p>

<!-- synced-from: README.md sha256:73b53c056cd7db0f63c60a2e04e194606a6aa34cbae9b69ef29b732d85b336d3 -->
>>>>>>> Stashed changes

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
<<<<<<< Updated upstream
  <strong>Ajusta y posentrena LLMs con un solo comando. Sin SSH, sin infierno de configuración.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Sitio web</a> &middot;
  <a href="#inicio-rápido">Inicio rápido</a> &middot;
  <a href="#interfaz-web">Interfaz web</a> &middot;
  <a href="#configuración">Configuración</a> &middot;
  <a href="docs/commands.md">Comandos</a> &middot;
  <a href="docs/models.md">Modelos</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Descargas"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Licencia Apache-2.0">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Pruebas"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
=======
  <strong>Ajusta LLMs con un comando. Sin SSH, sin infierno de configuraciones.</strong>
>>>>>>> Stashed changes
</p>

---

<<<<<<< Updated upstream
Soup convierte el dolor del ajuste fino de LLMs en un flujo de trabajo simple. Un config, un comando, listo.

```bash
pip install "soup-cli[train]"   # añade [train] para ajuste fino; solo soup-cli es el CLI ligero
=======
Soup convierte el dolor del ajuste fino de LLMs en un flujo de trabajo simple.

```bash
pip install "soup-cli[train]"
>>>>>>> Stashed changes
soup init --template chat
soup train
```

<<<<<<< Updated upstream
**Ajusta un modelo de 8B en una GPU portátil de 4 GB.** La transmisión de capas mantiene la base congelada fuera de VRAM y la alimenta a la GPU una capa decodificadora a la vez. Medido en RTX 3050 Laptop 4 GB: Llama-3.1-8B-Instruct + NF4 a **119.6 tok/s, pico de 3.32 GB** — bit exacto contra una ejecución normal residente, y reproducido independientemente en una H100 a 113.00 tok/s con los mismos 3.32 GB. Opcional (`stream_layers: true`) y aún BETA —
[cómo funciona](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[todas las mediciones](benchmarks/) ·
[artículo](https://doi.org/10.5281/zenodo.21771064) ·
**[compruébalo tú mismo en una Colab T4 gratis](notebooks/proof-4gb.ipynb)**

---

## 🎯 ¿Por qué Soup?

Entrenar LLMs sigue siendo doloroso. Incluso los equipos experimentados pasan 30-50% de su tiempo luchando con infraestructura en lugar de mejorar modelos. Soup lo arregla.

- **Cero SSH.** Nunca más SSH a una caja de GPU rota.
- **Un config.** Solo necesitas un archivo YAML simple.
- **Todo automático.** Tamaño de lote, detección de GPU, cuantización — gestionado.
- **Funciona localmente.** Entrena en tu propia GPU con QLoRA. No se necesita nube.

### Novedades — v0.74.0

**v0.74.0 — la base congelada se estaba cargando en fp32 todo el tiempo.** Arreglar solo eso reduce el VRAM máximo 2.59x en un config sin cambios. **116 de las 120 solicitudes de extracción fusionadas en esta versión vinieron de fuera del mantenedor**, por 25 personas.

- Cada carga SFT elevaba silenciosamente la base congelada a fp32. Medido en H100 con Llama-3.1-8B + LoRA: **48,241 MiB → 18,658 MiB pico — 2.59x, 28.9 GB**.
- **Transformers 5.x, TRL 0.29, PEFT 0.20.**
- Cuatro evasiones SSRF de la misma forma (IPv4 abreviado, decimal, hexadecimal y octal) corregidas.
- **`soup serve` ahora sale con 2** cuando se vincula a un host no loopback sin `--tool-auth-token`.
- **`soup train --cloud lambda`** — solo plan por defecto, `--cloud-submit` envía en vivo.

---

## 🏗️ Arquitectura y módulos

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70+ comandos CLI (train, eval, data, serve, agent, …)
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — fuente única de verdad para todos los campos de configuración
│   ├── autodistill/     # Auto-destilación (captura de datos, publicación, worker MLX)
│   ├── autopilot/       # Selección de tarea/cuantización/LR sin configuración
│   ├── cans/            # Soup Cans: empaquetado de modelos, publicación, verificación
│   ├── cloud/           # Controladores de nube Lambda Labs / Modal
│   └── utils/           # Ayudantes compartidos (350+ módulos)
├── graft/               # Configuración de Graft MCP (cableado a nivel de usuario)
├── docs/                # Guías temáticas + referencia completa de comandos
├── templates/           # Plantillas iniciales chat.yaml, code.yaml, medical.yaml
├── benchmarks/          # Registros de medición (datos crudos para cada número)
├── tests/               # 460 archivos de prueba
└── pyproject.toml       # Dependencias, extras, sistema de construcción
```

**Dependencias principales (instalación ligera — sin PyTorch):**
`typer`, `rich`, `pydantic>=2`, `pyyaml`, `huggingface-hub`, `plotext`, `packaging`

**Extras `[train]`:** `torch>=2.6.0`, `transformers>=5.16.1,<6`, `peft>=0.20,<1`, `trl>=0.29`, `datasets`, `bitsandbytes`, `accelerate`

**Nota:** Este repositorio contiene código fuente Python junto con scripts de configuración, flujos de trabajo CI y herramientas de empaquetado; no afirmamos "%100 Python puro".

---

## 🚀 Instalación

### Requisitos

- Python 3.10, 3.11 o 3.12 (estas son las versiones que prueba CI; 3.13+ aún no soportado)
- GPU con CUDA (recomendado), Apple Silicon (MPS) o CPU (experimental — muy lento)
- 8 GB+ VRAM para modelos 7B con QLoRA

### Pasos de instalación

```bash
# Núcleo ligero: CLI + configuración + herramientas de datos, sin PyTorch
pipx install soup-cli
uv tool install soup-cli          # misma idea, si ya usas uv

# Añadir la pila de entrenamiento (torch, transformers, peft, trl, datasets, …)
pipx install "soup-cli[train]"

# Todo (train + serve + ui + data) de una vez
pipx install "soup-cli[all]"

# O desde GitHub (último desarrollo)
pipx install "git+https://github.com/MakazhanAlpamys/Soup.git"
```

¿Ya estás dentro de un virtualenv, un notebook Colab o una imagen Docker? Usa `pip` directamente:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **¿`error: externally-managed-environment`?** Esto es [PEP 668](https://peps.python.org/pep-0668/), no un problema de Soup. Usa `pipx` o `uv tool`.

> **Comillas dobles, no simples.** `"soup-cli[train]"` es la única ortografía que funciona en todos los shells.

Tabla completa de extras (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, `liger`, `onnx`, `tensorrt`, …): [`docs/models.md`](docs/models.md#optional-extras)

### Instalación desde el fork

Este repositorio es el fork `Ercaner1988/Soup`. Upstream: `MakazhanAlpamys/Soup`.

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

---

## 📖 Uso

### Inicio rápido

```bash
# 1. Crear configuración
soup init                       # asistente interactivo
soup init --template chat       # o comenzar desde una plantilla

# 2. Entrenar, probar, enviar
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo user/my-model

# 3. Fusionar y exportar
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**Plantillas:** `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`

### Comandos comunes

```bash
soup train  --config soup.yaml                         # entrenar (SFT/DPO/GRPO/PPO/KTO/ORPO/…)
soup infer  --model ./output --input prompts.jsonl    # inferencia por lotes
soup chat   --model ./output                           # chat interactivo
soup serve  --model ./output                           # servidor API compatible con OpenAI
soup ui                                                # panel de navegador local
soup merge  --adapter ./output                         # fusionar LoRA en el modelo base
soup export --model ./output --format gguf             # exportar para despliegue
soup eval   benchmark --model ./output                 # evaluar
soup data   inspect ./data/train.jsonl                 # estadísticas del conjunto de datos
soup recipes list                                      # 100+ recetas de modelo listas para usar
soup autopilot --model <id> --data d.jsonl --goal chat # sin configuración
soup doctor                                            # verificar GPU / dependencias / entorno
```

### Configuración

Un `soup.yaml` completo:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x más rápido, pip install "soup-cli[fast]"

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

`config/schema.py` es la fuente única de verdad para cada campo.

> **Las claves de configuración desconocidas advierten hoy y serán rechazadas en v0.75.**

### Formatos de datos

Alpaca, ShareGPT, ChatML, pares de preferencia (DPO / ORPO / SimPO / IPO / KTO), visión, audio, ASR, texto plano, incrustación, RAFT y más — detección automática desde JSONL, JSON, CSV, Parquet o TXT. Ver [`docs/data.md`](docs/data.md).

### Modelos soportados

Soup funciona con **cualquier** modelo de generación de texto en el [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral, Mixtral, DeepSeek R1/V3, Phi-4 y 100+ otros vienen como recetas listas para usar (`soup recipes list`).

| VRAM | Modelo máximo (QLoRA 4-bit) | Ejemplo |
|------|------------------------------|---------|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (completo) o MoE | Mixtral-8x22B, DeepSeek-V3 |

### Interfaz web

```bash
pip install "soup-cli[ui]"
soup ui
# Abre http://127.0.0.1:7860
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ Pruebas y puertas de calidad

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — debe estar limpio antes de cualquier commit
pytest tests/ -v --tb=short        # pruebas unitarias (rápidas, sin GPU; 460 archivos de prueba)
pytest tests/ -m smoke -v          # pruebas de humo (descarga un modelo pequeño, entrena)
pre-commit install                 # opcional: ruff lint+formateo al commit
```

**Matriz CI:** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**Seguridad:** `soup doctor` verifica GPU, recursos del sistema, dependencias y versión en un solo lugar. La telemetría es estrictamente opt-in (`SOUP_TELEMETRY=1`, desactivada por defecto).

---

## 📚 Documentación

| Guía | Cubre |
|------|-------|
| [Tareas y métodos de entrenamiento](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, llamada a herramientas, PRM, pre-entrenamiento, destilación |
| [PEFT y eficiencia](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA |
| [Rendimiento y cuantización](docs/performance-and-quantization.md) | QAT, FP8, caché KV, transmisión de capas, multi-GPU / DeepSpeed / FSDP |
| [Ingeniería de datos](docs/data.md) | Formatos, herramientas de datos, generación sintética, DAG de recetas |
| [Evaluación y sondas](docs/evaluation.md) | Diseño de evaluación, benchmarks, métricas NLG, A/B |
| [Servicio y exportación](docs/serving-and-export.md) | Servidor compatible con OpenAI, fusión/exportación, Interfaz web |
| [Adaptadores y gobernanza](docs/adapters-and-governance.md) | Ciclo de vida del adaptador, registro de modelos, Soup Cans |
| [Cumplimiento](docs/compliance.md) | Plantillas HIPAA/SOC2/EU-AI-Act/SR-11-7, registro de auditoría |
| [Backends y operaciones](docs/backends-and-ops.md) | Backends MLX/Unsloth, HF Hub, piloto automático |
| [Referencia de comandos](docs/commands.md) | Lista completa de comandos `soup` |
| [Modelos y extras](docs/models.md) | Familias de modelos, guía de VRAM, matriz de extras de pip |

---

## 👥 Contribuyentes

El repositorio upstream (`MakazhanAlpamys/Soup`) está construido por la comunidad. Este fork (`Ercaner1988/Soup`) está personalizado para integración con Graft MCP.

| Contribuyente | Commits | Rol |
|--------------|---------|-----|
| Alpamys (Alpamys Makazhan) | 912 | Propietario del proyecto y desarrollador principal |
| AmixDigital | 37 | Sección de puerta frontal de la Interfaz web |
| Srinivasan R | 24 | Máscara de solo respuesta MLX, pérdidas de validación, guardia de recipe-id |
| Amir Fathi | 24 | Contribuciones |
| Ben Younes | 15 | Contribuciones |
| Salil Mhatre | 11 | Contribuciones |
| Darsh | 9 | Contribuciones |
| Nurkhan Esenbek | 9 | Contribuciones |
| Harshit Sharma | 7 | Contribuciones |
| UmranPros | 6 | Contribuciones |
| (otros 24 contribuyentes) | 5–1 | Ver [CONTRIBUTORS.md](CONTRIBUTORS.md) |
| **Ercan Er** | **2** | **Configuración de Graft MCP (cableado a nivel de usuario)** |

> Métricas del resultado `git shortlog -sn HEAD` (rama ercan1, 2026-09-09).

---

## 📄 Licencia

[Apache-2.0](LICENSE). Copyright © los contribuyentes de Soup.

Este fork (`Ercaner1988/Soup`) se basa en el proyecto original: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

### Cita

```bibtex
@misc{makazhan2026exact,
  title     = {Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU},
  author    = {Makazhan, Alpamys},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v3},
  doi       = {10.5281/zenodo.21918325},
  url       = {https://doi.org/10.5281/zenodo.21918325}
}
```
=======
## ¿Por qué Soup?

- **Cero SSH**
- **Una configuración**
- **Todo automático**
- **Funciona localmente**

## Inicio rápido

```bash
pip install "soup-cli[train]"
soup init --template chat
soup train
```

## Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup-cli:latest
```

## Requisitos

- Python 3.10-3.12
- NVIDIA GPU o Apple Silicon

## Licencia

Apache-2.0
>>>>>>> Stashed changes
