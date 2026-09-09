**🌍 [Türkçe](README.md) | [English](README.en.md) | [العربية](README.ar.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Русский](README.ru.md) | [Español](README.es.md)**

<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Fine-tune and post-train LLMs in one command. No SSH, no config hell.</strong>
</p>

<p align="center">
  <a href="https://trysoup.dev">Website</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#web-ui">Web UI</a> &middot;
  <a href="#configuration">Config</a> &middot;
  <a href="docs/commands.md">Commands</a> &middot;
  <a href="docs/models.md">Models</a> &middot;
  <a href="https://discord.gg/dgd2pJcjwP">Discord</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Tests"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI"></a>
</p>

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.

```bash
pip install "soup-cli[train]"   # add [train] to fine-tune; bare `soup-cli` is the light CLI
soup init --template chat
soup train
```

**Fine-tune an 8B model on a 4 GB laptop GPU.** Layer streaming keeps the frozen base out of VRAM and feeds it to the GPU one decoder layer at a time. Measured on an RTX 3050 Laptop 4 GB: Llama-3.1-8B-Instruct + NF4 at **119.6 tok/s, 3.32 GB peak** — bit-exact against a normal resident run, and reproduced independently on an H100 at 113.00 tok/s in the same 3.32 GB. Opt-in (`stream_layers: true`) and still BETA — [how it works](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) · [all measurements](benchmarks/) · [paper](https://doi.org/10.5281/zenodo.21771064) · **[check it yourself on a free Colab T4](notebooks/proof-4gb.ipynb)**

---

## 🎯 Why Soup?

Training LLMs is still painful. Even experienced teams spend 30-50% of their time fighting infrastructure instead of improving models. Soup fixes that.

- **Zero SSH.** Never SSH into a broken GPU box again.
- **One config.** A simple YAML file is all you need.
- **Auto everything.** Batch size, GPU detection, quantization — handled.
- **Works locally.** Train on your own GPU with QLoRA. No cloud required.

### What's New — v0.74.0

**v0.74.0 — the frozen base was being loaded in fp32 the whole time.** Fixing that alone cuts peak VRAM 2.59x on an unchanged config. **116 of the 120 merged pull requests in this release came from outside the maintainer**, by 25 people.

- Every SFT load silently upcast the frozen base to fp32. Measured on an H100 with Llama-3.1-8B + LoRA: **48,241 MiB → 18,658 MiB peak — 2.59x, 28.9 GB**.
- **Transformers 5.x, TRL 0.29, PEFT 0.20.**
- Four SSRF bypasses of the same shape (abbreviated, decimal, hex and octal IPv4) fixed.
- **`soup serve` now exits 2** when bound to a non-loopback host without `--tool-auth-token`.
- **`soup train --cloud lambda`** — plan-only by default, `--cloud-submit` submits live.

---

## 🏗️ Architecture and Modules

```
Soup/
├── src/soup_cli/
│   ├── commands/        # 70+ CLI commands (train, eval, data, serve, agent, …)
│   ├── config/
│   │   └── schema.py    # Pydantic v2 — single source of truth for all config fields
│   ├── autodistill/     # Auto-distillation (data capture, publish, MLX worker)
│   ├── autopilot/       # Zero-config task/quantization/LR selection
│   ├── cans/            # Soup Cans: model packaging, publishing, verification
│   ├── cloud/           # Lambda Labs / Modal cloud controllers
│   └── utils/           # Shared helpers (350+ modules)
├── graft/               # Graft MCP configuration (user-level wiring)
├── docs/                # Topic guides + full command reference
├── templates/           # chat.yaml, code.yaml, medical.yaml start templates
├── benchmarks/          # Measurement records (raw data for every number)
├── tests/               # 460 test files
└── pyproject.toml       # Dependencies, extras, build system
```

**Core dependencies (light install — no PyTorch):**
`typer`, `rich`, `pydantic>=2`, `pyyaml`, `huggingface-hub`, `plotext`, `packaging`

**`[train]` extras:** `torch>=2.6.0`, `transformers>=5.16.1,<6`, `peft>=0.20,<1`, `trl>=0.29`, `datasets`, `bitsandbytes`, `accelerate`

**Note:** The repository contains Python source code alongside setup scripts, CI workflows, and packaging tools; we do not claim "%100 pure Python".

---

## 🚀 Installation

### Requirements

- Python 3.10, 3.11 or 3.12 (these are the versions CI tests; 3.13+ is not supported yet)
- GPU with CUDA (recommended), Apple Silicon (MPS), or CPU (experimental — very slow)
- 8 GB+ VRAM for 7B models with QLoRA

### Installation Steps

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

Already inside a virtualenv, a Colab notebook, or a Docker image? Use `pip` directly:

```bash
pip install soup-cli
pip install "soup-cli[train]"
pip install "soup-cli[all]"
```

> **`error: externally-managed-environment`?** This is [PEP 668](https://peps.python.org/pep-0668/), not a Soup problem. Use `pipx` or `uv tool`.

> **Double quotes, not single.** `"soup-cli[train]"` is the only spelling that works in every shell.

Full extras table (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, `liger`, `onnx`, `tensorrt`, …): [`docs/models.md`](docs/models.md#optional-extras)

### Install From Fork

This repository is the `Ercaner1988/Soup` fork. Upstream: `MakazhanAlpamys/Soup`.

```bash
git clone https://github.com/Ercaner1988/Soup.git
cd Soup
pip install -e ".[dev]"
```

---

## 📖 Usage

### Quick Start

```bash
# 1. Create a config
soup init                       # interactive wizard
soup init --template chat       # or start from a template

# 2. Train, test, ship
soup train --config soup.yaml
soup chat  --model ./output
soup push  --model ./output --repo user/my-model

# 3. Merge and export
soup merge  --adapter ./output
soup export --model ./output --format gguf --quant q4_k_m
```

**Templates:** `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`, `simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`

### Common Commands

```bash
soup train  --config soup.yaml                         # train (SFT/DPO/GRPO/PPO/KTO/ORPO/…)
soup infer  --model ./output --input prompts.jsonl    # batch inference
soup chat   --model ./output                           # interactive chat
soup serve  --model ./output                           # OpenAI-compatible API server
soup ui                                                # local browser dashboard
soup merge  --adapter ./output                         # merge LoRA into base model
soup export --model ./output --format gguf             # export for deployment
soup eval   benchmark --model ./output                 # evaluate
soup data   inspect ./data/train.jsonl                 # dataset stats
soup recipes list                                      # 100+ ready-made model recipes
soup autopilot --model <id> --data d.jsonl --goal chat # zero-config
soup doctor                                            # GPU / deps / environment check
```

### Configuration

A complete `soup.yaml`:

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

`config/schema.py` is the single source of truth for every field.

> **Unknown config keys warn today and will be rejected in v0.75.**

### Data Formats

Alpaca, ShareGPT, ChatML, preference pairs (DPO / ORPO / SimPO / IPO / KTO), vision, audio, ASR, plaintext, embedding, RAFT and more — auto-detected from JSONL, JSON, CSV, Parquet or TXT. See [`docs/data.md`](docs/data.md).

### Supported Models

Soup works with **any** text-generation model on the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral, Mixtral, DeepSeek R1/V3, Phi-4 and 100+ others ship as ready-made recipes (`soup recipes list`).

| VRAM | Max model (QLoRA 4-bit) | Example |
|------|-------------------------|---------|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

### Web UI

```bash
pip install "soup-cli[ui]"
soup ui
# Opens http://127.0.0.1:7860
```

### Docker

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
```

---

## 🛡️ Testing and Quality Gates

```bash
pip install -e ".[dev]"
ruff check src/soup_cli/ tests/    # lint — must be clean before any commit
pytest tests/ -v --tb=short        # unit tests (fast, no GPU; 460 test files)
pytest tests/ -m smoke -v          # smoke tests (downloads tiny model, trains)
pre-commit install                 # optional: ruff lint+format on commit
```

**CI matrix:** Python 3.10 / 3.11 / 3.12 × Ubuntu / Windows / macOS

**Security:** `soup doctor` checks GPU, system resources, dependencies, and version in one place. Telemetry is strictly opt-in (`SOUP_TELEMETRY=1`, default off).

---

## 📚 Documentation

| Guide | Covers |
|-------|--------|
| [Training tasks & methods](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-training, distillation |
| [PEFT & efficiency](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA |
| [Performance & quantization](docs/performance-and-quantization.md) | QAT, FP8, KV-cache, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Data engineering](docs/data.md) | Formats, data tools, synthetic generation, recipe DAGs |
| [Evaluation & probes](docs/evaluation.md) | Eval design, benchmarks, NLG metrics, A/B |
| [Serving & export](docs/serving-and-export.md) | OpenAI-compatible server, merge/export, Web UI |
| [Adapters & governance](docs/adapters-and-governance.md) | Adapter lifecycle, model registry, Soup Cans |
| [Compliance](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 templates, audit log |
| [Backends & ops](docs/backends-and-ops.md) | MLX/Unsloth backends, HF Hub, autopilot |
| [Command reference](docs/commands.md) | Full `soup` command list |
| [Models & extras](docs/models.md) | Model families, VRAM guide, pip extras matrix |

---

## 👥 Contributors

The upstream repository (`MakazhanAlpamys/Soup`) is built by the community. This fork (`Ercaner1988/Soup`) is customized for Graft MCP integration.

| Contributor | Commits | Role |
|-------------|---------|------|
| Alpamys (Alpamys Makazhan) | 912 | Project owner and lead developer |
| AmixDigital | 37 | Web UI front-door section |
| Srinivasan R | 24 | MLX response-only mask, validation losses, recipe-id guard |
| Amir Fathi | 24 | Contributions |
| Ben Younes | 15 | Contributions |
| Salil Mhatre | 11 | Contributions |
| Darsh | 9 | Contributions |
| Nurkhan Esenbek | 9 | Contributions |
| Harshit Sharma | 7 | Contributions |
| UmranPros | 6 | Contributions |
| (24 other contributors) | 5–1 | See [CONTRIBUTORS.md](CONTRIBUTORS.md) |
| **Ercan Er** | **2** | **Graft MCP configuration (user-level wiring)** |

> Metrics from `git shortlog -sn HEAD` output (ercan1 branch, 2026-09-09).

---

## 📄 License

[Apache-2.0](LICENSE). Copyright © Soup contributors.

This fork (`Ercaner1988/Soup`) is based on the original project: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)

### Citation

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