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
  <a href="#configuration">Config</a> &middot;
  <a href="#documentation">Docs</a> &middot;
  <a href="docs/commands.md">Commands</a> &middot;
  <a href="docs/models.md">Models</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Tests"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Website"></a>
</p>

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.

```bash
pip install "soup-cli[train]"   # add [train] to fine-tune; bare `soup-cli` is the light CLI
soup init --template chat
soup train
```

## Why Soup?

Training LLMs is still painful. Even experienced teams spend 30-50% of their time fighting
infrastructure instead of improving models. Soup fixes that.

- **Zero SSH.** Never SSH into a broken GPU box again.
- **One config.** A simple YAML file is all you need.
- **Auto everything.** Batch size, GPU detection, quantization — handled.
- **Works locally.** Train on your own GPU with QLoRA. No cloud required.

## What's New

**v0.71.36 — Data Moat II.** A semantic layer over your training data, plus two tools for what a fine-tune forgets and what it leaks.

- **`soup data dedup --semantic`** — dedup by meaning, not by shared tokens. Catches
  *reworded* duplicates MinHash's shingling scores as distinct. Zero new dependencies.
- **`soup data topics <data>`** — cluster your data and get a coverage table
  (`82% code · 6% math · 1% safety`) with a warning for thin topics, so you can see what
  you are actually training on.
- **`soup data canary insert|check`** — insert unique secrets, then prove whether a model
  memorized them: each secret's loss is ranked against never-inserted controls. Exit 2 on a
  leak, so CI can gate. On SmolLM2-135M a memorized set lands at percentile 0.0 against a
  clean model's 4%–93% spread.
- **`soup train --replay old.jsonl`** — continual-learning rehearsal: interleave your old
  data so the new task doesn't erase it.
- **Honest result:** semantic dedup is **not** a paraphrase detector. Measured, paraphrase
  cosines (0.49–0.76) *overlap* with genuinely-distinct rows (0.54–0.76) — "Add two numbers"
  vs "Multiply two numbers" scores **higher** than a real paraphrase — so no threshold
  separates them. The default is deliberately conservative; see the CHANGELOG.
- **Two blocking bugs fixed:** the hardware-fit gate **refused to train any model you merged
  yourself** (a local checkpoint's name has no size marker, so it guessed 7B and predicted
  16 GB), and every `pip install "soup-cli[extra]"` hint printed **without the extra** —
  17 sites where the suggested command succeeds and leaves the feature still broken.

```bash
soup data topics train.jsonl                    # what am I actually training on?
soup data dedup train.jsonl --semantic -o clean.jsonl

soup data canary insert train.jsonl -o canaried.jsonl --manifest secrets.json
soup data canary check --manifest secrets.json --base ./my-model  # exit 2 = leak

soup train --config sft.yaml --replay old_task.jsonl --replay-ratio 0.1
```

<details>
<summary>Previous release — v0.71.33, <code>soup draft</code> (measure speculative decoding)</summary>

`soup draft measure` reports a draft model's **acceptance rate** + real plain-vs-assisted tok/s
(exit 0/2/1 for CI); `soup draft distill` distils your target into a dense tiny draft, auto-wired
into `soup serve --auto-spec`. The honest result on a small same-family pair: distillation didn't
move acceptance (69.3% → 69.3%) and assisted decoding was a net slowdown — which is exactly the
number you want *before* shipping speculative decoding.

```bash
soup draft measure --target ./my-tuned-model --draft HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts prod-prompts.jsonl        # -> acceptance %, real tok/s, ship-or-not
```

</details>

Full history: [CHANGELOG.md](CHANGELOG.md) &middot; [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases).

## Quick Start

### 1. Install

```bash
# Light core: CLI + config + data tools, no PyTorch
pip install soup-cli

# Add the training stack (torch, transformers, peft, trl, datasets, …)
pip install "soup-cli[train]"

# Everything (train + serve + ui + data) in one shot
pip install "soup-cli[all]"

# Or from GitHub (latest dev)
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

The full extras table (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) lives in
[`docs/models.md`](docs/models.md#optional-extras).

> **Use double quotes around the extra.** They are the only spelling that works in
> every shell — `cmd.exe`, PowerShell, bash, and zsh.
>
> Older tutorials and videos (including some of ours) show the single-quoted
> `pip install 'soup-cli[train]'`. That is bash / zsh / PowerShell syntax, and it
> fails on Windows `cmd.exe`, which has no single-quote quoting and hands the
> quotes straight to pip:
>
> ```
> ERROR: Invalid requirement: "'soup-cli[train]'": Expected package name at the start of dependency specifier
> ```
>
> If you hit that, swap the `'` for `"` — pip is rejecting a literal quote
> character, nothing is wrong with the package. (Dropping the quotes entirely
> works on Windows too, but zsh then reads `[train]` as a glob and fails.)

`soup init`, `soup data …`, and the other data/inspection commands work on the light install.
Fine-tuning (`soup train`) needs the `[train]` extra.

### 2. Create a config

```bash
soup init                       # interactive wizard
soup init --template chat       # or start from a template
```

Templates: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Train, test, ship

```bash
soup train --config soup.yaml                 # LoRA, quantization, batching — all handled
soup chat  --model ./output                    # talk to your model
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # merge LoRA into the base
soup export --model ./output --format gguf --quant q4_k_m   # GGUF for Ollama / llama.cpp
```

More export targets (ONNX, TensorRT, AWQ, GPTQ, BitNet) and deployment options live in
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Configuration

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

`config/schema.py` is the single source of truth for every field. Advanced data, training,
and PEFT options are documented under [Documentation](#documentation).

## Documentation

The full feature reference lives in [`docs/`](docs/). Start here:

| Guide | Covers |
|---|---|
| [Training tasks & methods](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-training, distillation, classification, vision/audio/TTS, unlearning, RAFT/RA-DIT, loop-hardening detectors |
| [PEFT, long context & efficiency](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, optimizer & PEFT zoo, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, curriculum, auto-tuning |
| [Performance & quantization](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, multi-GPU / DeepSpeed / FSDP |
| [Data engineering](docs/data.md) | Formats, the Axolotl/LF-parity pipeline, data tools, synthetic generation & forge, quality scorecards, trace tooling, remote datasets, mixing, recipe DAGs |
| [Evaluation & probes](docs/evaluation.md) | Eval design/gate, eval-gated training, benchmarks, NLG metrics, calibration, Elo arena, diagnose, post-train X-ray probes, A/B, drift, tunability, `soup advise` |
| [Serving & export](docs/serving-and-export.md) | OpenAI-compatible server, batch inference, benchmarking, merge/export, Anthropic Messages endpoint, speculative decoding (train + measure your own draft), deploy autopilot, Web UI, Agent Forge |
| [Adapters, registry & governance](docs/adapters-and-governance.md) | Adapter lifecycle/management, model registry, Soup Cans, the data flywheel (`soup loop`), knowledge editing, steering, supply-chain controls (scan/sign/BOM/attest/audit/airgap) |
| [Compliance & governance quickstart](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` templates, provenance (BOM/attest/repro-receipt), audit log, air-gap, model-card autogen (`soup card`), CI gate (`soup ci init`) |
| [Backends, platform & ops](docs/backends-and-ops.md) | MLX/Unsloth backends, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, completions, plugins, utility commands |
| [Command reference](docs/commands.md) | The full `soup` command list |
| [Supported models & extras](docs/models.md) | Recommended model families, the VRAM size guide, the pip extras matrix |

## Data Formats

All formats are auto-detected from JSONL, JSON, CSV, Parquet, or TXT:

- **alpaca** — `{"instruction": ..., "input": ..., "output": ...}`
- **sharegpt** — `{"conversations": [{"from": "human", "value": ...}, ...]}`
- **chatml** — `{"messages": [{"role": "user", "content": ...}, ...]}`
- **dpo / orpo / simpo / ipo** — `{"prompt": ..., "chosen": ..., "rejected": ...}`
- **kto** — `{"prompt": ..., "completion": ..., "label": true}`
- **llava / sharegpt4v** (vision), **audio**, **plaintext** (pre-training), **embedding**,
  **prm**, **pre_tokenized**, **video**, **multimodal**

Full schemas and the Axolotl/LlamaFactory-parity data pipeline (remote URIs, streaming,
sharding, interleaving, vocab expansion, document ingestion) are in
[`docs/data.md`](docs/data.md).

## Common Commands

```bash
soup train  --config soup.yaml        # train (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl   # batch inference
soup chat   --model ./output          # interactive chat
soup serve  --model ./output          # OpenAI-compatible API server
soup merge  --adapter ./output        # merge LoRA into the base model
soup export --model ./output --format gguf           # export for deployment
soup eval   benchmark --model ./output               # evaluate
soup data   inspect ./data/train.jsonl               # dataset stats
soup recipes list                     # 100+ ready-made model recipes
soup autopilot --model <id> --data d.jsonl --goal chat  # zero-config
soup doctor                           # check GPU / deps / environment
```

The complete command list is in [`docs/commands.md`](docs/commands.md).

## Supported Models

Soup works with **any** text-generation model on the
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) — if it loads with
`AutoModelForCausalLM`, it works, zero config changes. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4, and 100+ others ship as ready-made recipes (`soup recipes list`).

| VRAM | Max model (QLoRA 4-bit) | Example |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

Full model + vision tables and the optional-extras matrix are in [`docs/models.md`](docs/models.md).

## Docker

Run Soup without installing CUDA or PyTorch locally (image published to GHCR on every release):

```bash
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml
docker compose up   # or build locally
```

## Requirements

- Python 3.10+
- GPU with CUDA (recommended), Apple Silicon (MPS), or CPU (experimental — very slow)
- 8 GB+ VRAM for 7B models with QLoRA

All training tasks run on CPU for testing (quantization auto-disabled). Optional extras
(`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) are listed in
[`docs/models.md`](docs/models.md#optional-extras).

## Troubleshooting

```bash
soup doctor    # GPU, system resources, dependencies, and version in one place
```

- **`ImportError: DLL load failed while importing _C` (Windows)** — reinstall PyTorch for your
  CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- **`soup version` ≠ `pip show soup-cli`** — multiple Python installs; use a virtualenv.

## Development

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # unit tests (fast, no GPU)
pytest tests/ -m smoke -v          # smoke tests (downloads a tiny model, trains)

pre-commit install                 # optional: ruff lint+format on commit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and [SECURITY.md](SECURITY.md) to
report a vulnerability.

## Contributors

Built by the community ❤️ — thank you to everyone who has contributed. See
[CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Contributors](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## License

[Apache-2.0](LICENSE). Copyright © the Soup contributors.
