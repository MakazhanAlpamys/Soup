# 🍜 Soup

**Fine-tune LLMs in one command. No SSH, no config hell.**

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
pip install soup-cli
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

## Data Tools

```bash
# Inspect a dataset
soup data inspect ./data/train.jsonl

# Validate format
soup data validate ./data/train.jsonl --format alpaca
```

## Features

| Feature | Status |
|---|---|
| LoRA / QLoRA fine-tuning | ✅ |
| SFT (Supervised Fine-Tune) | ✅ |
| DPO (Direct Preference Optimization) | 🔜 |
| Auto batch size | ✅ |
| Auto GPU detection (CUDA/MPS/CPU) | ✅ |
| Live terminal dashboard | ✅ |
| Alpaca / ShareGPT / ChatML formats | ✅ |
| HuggingFace datasets support | ✅ |
| Experiment tracking | 🔜 |
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
pytest tests/ -v
```

## License

MIT
