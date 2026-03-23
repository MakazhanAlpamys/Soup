<p align="center">
  <img src="soup.png" alt="Soup" width="200">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Fine-tune LLMs in one command. No SSH, no config hell.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#data-tools">Data Tools</a> &middot;
  <a href="#experiment-tracking">Tracking</a> &middot;
  <a href="#model-evaluation">Eval</a> &middot;
  <a href="#all-commands">Commands</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Tests"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.

```bash
pip install soup-cli
soup init --template chat
soup train
```

## Why Soup?

Training LLMs is still painful. Even experienced teams spend 30-50% of their time fighting infrastructure instead of improving models. Soup fixes that.

- **Zero SSH.** Never SSH into a broken GPU box again.
- **One config.** A simple YAML file is all you need.
- **Auto everything.** Batch size, GPU detection, quantization — handled.
- **Works locally.** Train on your own GPU with QLoRA. No cloud required.

## Quick Start

### 1. Install

```bash
# From PyPI (recommended):
pip install soup-cli

# Or from GitHub (latest dev):
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

### 2. Create config

```bash
# Interactive wizard
soup init

# Or use a template
soup init --template chat    # conversational fine-tune
soup init --template code    # code generation
soup init --template medical # domain expert
```

### 3. Train

```bash
soup train --config soup.yaml
```

That's it. Soup handles LoRA setup, quantization, batch size, monitoring, and checkpoints.

### 4. Test your model

```bash
soup chat --model ./output
```

### 5. Push to HuggingFace

```bash
soup push --model ./output --repo your-username/my-model
```

### 6. Merge & Export

```bash
# Merge LoRA adapter with base model
soup merge --adapter ./output

# Export to GGUF for Ollama / llama.cpp
soup export --model ./output --format gguf --quant q4_k_m
```

## Config Example

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

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

## DPO Training

Train with preference data using Direct Preference Optimization:

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

## Chat with your model

```bash
# Chat with a LoRA adapter (auto-detects base model)
soup chat --model ./output

# Specify base model explicitly
soup chat --model ./output --base meta-llama/Llama-3.1-8B-Instruct

# Adjust generation
soup chat --model ./output --temperature 0.3 --max-tokens 256
```

## Push to HuggingFace

```bash
# Upload model to HF Hub
soup push --model ./output --repo your-username/my-model

# Make it private
soup push --model ./output --repo your-username/my-model --private
```

## Merge LoRA Adapter

Merge a LoRA adapter with its base model into a standalone model:

```bash
# Auto-detect base model from adapter_config.json
soup merge --adapter ./output --output ./merged

# Specify base model and dtype
soup merge --adapter ./output --base meta-llama/Llama-3.1-8B --dtype bfloat16
```

## Export to GGUF

Export models to GGUF format for use with [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
# Export LoRA adapter (auto-merges with base, then converts)
soup export --model ./output --format gguf --quant q4_k_m

# Export with different quantizations
soup export --model ./output --format gguf --quant q8_0
soup export --model ./output --format gguf --quant f16

# Export a full (already merged) model
soup export --model ./merged --format gguf

# Specify llama.cpp path manually
soup export --model ./output --format gguf --llama-cpp /path/to/llama.cpp
```

Supported quantizations: `q4_0`, `q4_k_m`, `q5_k_m`, `q8_0`, `f16`, `f32`

After export, use with Ollama:
```bash
echo 'FROM ./my-model.q4_k_m.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model
```

## Resume Training

Resume a training run from a checkpoint:

```bash
# Auto-detect latest checkpoint in output directory
soup train --config soup.yaml --resume auto

# Resume from a specific checkpoint
soup train --config soup.yaml --resume ./output/checkpoint-500
```

## Weights & Biases Integration

Send training metrics to [W&B](https://wandb.ai/) for cloud-based experiment tracking:

```bash
# Enable W&B logging (requires: pip install wandb)
soup train --config soup.yaml --wandb
```

Make sure `WANDB_API_KEY` is set or run `wandb login` first.

## Inference Server

Start a local OpenAI-compatible inference server:

```bash
# Install server dependencies
pip install 'soup-cli[serve]'

# Start server
soup serve --model ./output --port 8000

# With custom settings
soup serve --model ./output --port 8080 --host 127.0.0.1 --max-tokens 1024
```

Endpoints:
- `POST /v1/chat/completions` — chat completions (streaming supported)
- `GET /v1/models` — list available models
- `GET /health` — health check

Compatible with OpenAI SDK:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="output",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Synthetic Data Generation

Generate training data using LLMs:

```bash
# Generate using OpenAI API
soup data generate --prompt "Create math word problems" --count 100 --format alpaca

# Use a different model
soup data generate --prompt "Medical Q&A pairs" --model gpt-4o --count 500

# Deduplicate against existing data
soup data generate --prompt "..." --count 200 --dedup-with existing.jsonl

# Use seed examples to guide style
soup data generate --prompt "..." --seed examples.jsonl --count 100
```

## Hyperparameter Sweep

Search for the best hyperparameters:

```bash
# Grid search over learning rate and LoRA rank
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --param lora_r=8,16,32

# Random search with max runs
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --strategy random --max-runs 5

# Preview without running
soup sweep --config soup.yaml --param lr=1e-5,2e-5 --param epochs=2,3 --dry-run

# Early stopping: skip remaining runs if loss exceeds 1.5x best
soup sweep --config soup.yaml --param lr=1e-5,2e-5,5e-5 --early-stop 1.5
```

## Model Comparison

Compare outputs of two models side-by-side:

```bash
# Compare with inline prompts
soup diff --model-a ./model_v1 --model-b ./model_v2 --prompt "Explain gravity"

# Compare with a prompts file
soup diff --model-a ./base --model-b ./finetuned --prompts test_prompts.jsonl

# Save results
soup diff --model-a ./a --model-b ./b --prompts prompts.txt --output results.jsonl
```

## Multi-GPU / DeepSpeed

Train on multiple GPUs with DeepSpeed:

```bash
# ZeRO Stage 2 (recommended for most cases)
soup train --config soup.yaml --deepspeed zero2

# ZeRO Stage 3 (for very large models)
soup train --config soup.yaml --deepspeed zero3

# ZeRO Stage 2 with CPU offload (memory-constrained)
soup train --config soup.yaml --deepspeed zero2_offload

# Custom DeepSpeed config
soup train --config soup.yaml --deepspeed ./my_ds_config.json
```

## Quickstart Demo

Run a complete demo in one command — creates sample data, config, and trains a tiny model:

```bash
# Full demo (creates data + config + trains TinyLlama)
soup quickstart

# Just create files without training
soup quickstart --dry-run

# Skip confirmation
soup quickstart --yes
```

## Health Check

Check your environment for compatibility issues:

```bash
soup doctor
```

Shows: Python version, GPU availability, all dependency versions, and fix suggestions.

## Version Info

```bash
# Basic version
soup version

# Full system info (useful for bug reports)
soup version --full
# → soup v0.4.0 | Python 3.11.5 | CUDA 12.1 | extras: serve, data
```

## Error Handling

Soup shows friendly error messages by default (2-3 lines with a fix suggestion). For full tracebacks:

```bash
# Global flag goes BEFORE the command
soup --verbose train --config soup.yaml

# Works with any command
soup --verbose eval --model ./output --benchmarks mmlu
```

> **Note:** `--verbose` is a global flag — it must go **before** the command name, not after.

## Data Formats

Soup supports these formats (auto-detected). Files can be JSONL, JSON, CSV, or Parquet.

**Alpaca:**
```json
{"instruction": "Explain gravity", "input": "", "output": "Gravity is..."}
```

**ShareGPT:**
```json
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
```

**ChatML:**
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

**DPO (preference pairs):**
```json
{"prompt": "Explain gravity", "chosen": "Gravity is a force...", "rejected": "I don't know"}
```

## Data Tools

```bash
# Inspect a dataset
soup data inspect ./data/train.jsonl

# Validate format
soup data validate ./data/train.jsonl --format alpaca

# Convert between formats
soup data convert ./data/train.jsonl --to sharegpt --output converted.jsonl

# Merge multiple datasets
soup data merge data1.jsonl data2.jsonl --output merged.jsonl --shuffle

# Remove near-duplicates (requires: pip install 'soup-cli[data]')
soup data dedup ./data/train.jsonl --threshold 0.8

# Extended statistics (length distribution, token counts, languages)
soup data stats ./data/train.jsonl
```

## Experiment Tracking

Every `soup train` run is automatically tracked in a local SQLite database (`~/.soup/experiments.db`).

```bash
# List all training runs
soup runs

# Show detailed info + loss curve for a run
soup runs show run_20260223_143052_a1b2

# Compare two runs side by side
soup runs compare run_1 run_2

# Delete a run
soup runs delete run_1
```

## Model Evaluation

Evaluate models on standard benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
# Install eval dependencies
pip install 'soup-cli[eval]'

# Evaluate on benchmarks
soup eval --model ./output --benchmarks mmlu,gsm8k,hellaswag

# Link results to a training run
soup eval --model ./output --benchmarks mmlu --run-id run_20260223_143052_a1b2
```

## Features

| Feature | Status |
|---|---|
| LoRA / QLoRA fine-tuning | ✅ |
| SFT (Supervised Fine-Tune) | ✅ |
| DPO (Direct Preference Optimization) | ✅ |
| Auto batch size | ✅ |
| Auto GPU detection (CUDA/MPS/CPU) | ✅ |
| Live terminal dashboard | ✅ |
| Alpaca / ShareGPT / ChatML / DPO formats | ✅ |
| HuggingFace datasets support | ✅ |
| Interactive model chat | ✅ |
| Push to HuggingFace Hub | ✅ |
| LoRA merge (adapter + base → full model) | ✅ |
| Export to GGUF (Ollama / llama.cpp) | ✅ |
| Resume training from checkpoint | ✅ |
| Weights & Biases integration | ✅ |
| Experiment tracking (SQLite) | ✅ |
| Data tools (convert, merge, dedup, stats) | ✅ |
| Model evaluation (lm-eval) | ✅ |
| Inference server (OpenAI-compatible) | ✅ |
| Synthetic data generation | ✅ |
| Hyperparameter sweep (grid/random + early stopping) | ✅ |
| Model comparison (diff) | ✅ |
| Multi-GPU / DeepSpeed | ✅ |
| Friendly error messages | ✅ |
| Health check (soup doctor) | ✅ |
| Rich download progress bars | ✅ |
| Quickstart demo | ✅ |
| Confirmation prompts | ✅ |
| Web dashboard | 🔜 |
| Cloud mode (BYOG) | 🔜 |

## All Commands

```
soup init [--template chat|code|medical]      Create config
soup train --config soup.yaml                 Start training
soup chat --model ./output                    Interactive chat
soup push --model ./output --repo user/name   Upload to HuggingFace
soup merge --adapter ./output                 Merge LoRA with base model
soup export --model ./output --format gguf    Export to GGUF (Ollama)
soup eval --model ./output --benchmarks mmlu  Evaluate on benchmarks
soup serve --model ./output --port 8000       OpenAI-compatible API server
soup sweep --config soup.yaml --param lr=...  Hyperparameter search
soup diff --model-a ./a --model-b ./b         Compare two models
soup data inspect <path>                      View dataset stats
soup data validate <path> --format alpaca     Check format
soup data convert <path> --to chatml          Convert between formats
soup data merge data1.jsonl data2.jsonl       Combine datasets
soup data dedup <path> --threshold 0.8        Remove duplicates (MinHash)
soup data stats <path>                        Extended statistics
soup data generate --prompt "..." --count 100 Generate synthetic data
soup runs                                     List training runs
soup runs show <run_id>                       Run details + loss graph
soup runs compare <run_1> <run_2>             Compare two runs
soup doctor                                   Check environment
soup quickstart [--dry-run]                   Full demo
soup version [--full]                         Show version (--full: system info)
soup --verbose <command>                      Full traceback on errors
```

## Requirements

- Python 3.9+
- GPU with CUDA (recommended) or Apple Silicon (MPS) or CPU (slow)
- 8 GB+ VRAM for 7B models with QLoRA

### Optional Extras

| Extra | Install | What it adds |
|---|---|---|
| `serve` | `pip install 'soup-cli[serve]'` | Inference server (FastAPI + uvicorn) |
| `data` | `pip install 'soup-cli[data]'` | Deduplication (MinHash via datasketch) |
| `eval` | `pip install 'soup-cli[eval]'` | Benchmark evaluation (lm-evaluation-harness) |
| `deepspeed` | `pip install 'soup-cli[deepspeed]'` | Multi-GPU training (DeepSpeed ZeRO) |
| `dev` | `pip install 'soup-cli[dev]'` | Tests + linting (pytest, ruff) |

## Development

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

# Lint
ruff check soup_cli/ tests/

# Run unit tests (fast, no GPU needed)
pytest tests/ -v

# Run smoke tests (downloads tiny model, runs real training)
pytest tests/ -m smoke -v
```

## Changelog

See [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) for version history.

## License

MIT
