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
  <a href="#model-evaluation">Eval</a>
</p>

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.

```bash
pip install git+https://github.com/MakazhanAlpamys/Soup.git
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
# From GitHub (recommended for now):
pip install git+https://github.com/MakazhanAlpamys/Soup.git

# From PyPI (coming soon):
# pip install soup-cli
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

## Data Formats

Soup supports these formats (auto-detected):

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
| Experiment tracking (SQLite) | ✅ |
| Data tools (convert, merge, dedup, stats) | ✅ |
| Model evaluation (lm-eval) | ✅ |
| Web dashboard | 🔜 |
| Cloud mode (BYOG) | 🔜 |

## Requirements

- Python 3.9+
- GPU with CUDA (recommended) or Apple Silicon (MPS) or CPU (slow)
- 8 GB+ VRAM for 7B models with QLoRA

## Development

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git
cd Soup
pip install -e ".[dev]"

# Run unit tests (fast, no GPU needed)
pytest tests/ -v

# Run smoke tests (downloads tiny model, runs real training)
pytest tests/ -m smoke -v
```

## License

MIT
