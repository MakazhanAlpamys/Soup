<p align="center">
  <img src="soup.png" alt="Soup" width="280">
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
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
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

### New in v0.25.0 — "Beyond the Wrapper"

- **`soup autopilot`** — give it a model, a dataset, and a goal; Soup picks the task, quantization, LoRA rank, LR, epochs, and perf flags automatically
- **Apple Silicon MLX backend** — train on M1-M4 chips via `pip install 'soup-cli[mlx]'`
- **Tool-calling / agentic fine-tuning** — new `tool-calling` data format + eval scoring
- **RLVR verifiable rewards** — `reward_fn: verifiable` with `verifiable_domain: math | code | json_schema` for deterministic GRPO rewards
- **VeRA + OLoRA PEFT** — `lora.use_vera` / `lora.use_olora` for smaller-footprint adaptation
- **Data augmentation** — `soup data augment --strategy rephrase|translate|style`
- **Training intelligence** — optional `forgetting_detection` + `checkpoint_intelligence` tracks catastrophic forgetting and picks the best checkpoint by quality, not just loss
- **9 new recipes** — Llama 4 Scout, Qwen 3 14B/32B, Gemma 3 12B/27B, DeepSeek V3, plus 5 MLX recipes (45 total)

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
soup init --template chat       # conversational fine-tune
soup init --template code       # code generation
soup init --template medical    # domain expert
soup init --template reasoning  # GRPO reasoning training
soup init --template vision     # vision/multimodal fine-tune
soup init --template kto        # KTO unpaired preference alignment
soup init --template orpo       # ORPO (no reference model needed)
soup init --template simpo      # SimPO length-normalized preference
soup init --template ipo        # IPO regularized preference
soup init --template rlhf       # full RLHF pipeline (SFT→RM→PPO)
soup init --template pretrain   # continued pre-training on raw text
soup init --template moe        # MoE fine-tuning (ScatterMoE LoRA)
soup init --template longcontext # 128k+ context fine-tuning
soup init --template embedding  # sentence embedding fine-tuning
soup init --template audio      # audio/speech model fine-tuning
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

# Export to ONNX (pip install 'soup-cli[onnx]')
soup export --model ./output --format onnx

# Export to TensorRT-LLM (pip install 'soup-cli[tensorrt]')
soup export --model ./output --format tensorrt

# Export to AWQ quantized model (pip install 'soup-cli[awq]')
soup export --model ./output --format awq --bits 4 --group-size 128

# Export to GPTQ quantized model (pip install 'soup-cli[gptq]')
soup export --model ./output --format gptq --bits 4 --group-size 128
```

## Config Example

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

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

## Unsloth Backend (2-5x Faster Training)

Use the [Unsloth](https://github.com/unslothai/unsloth) backend for significantly faster training and up to 80% less VRAM:

```bash
# Install unsloth support
pip install 'soup-cli[fast]'
```

Then add one line to your config:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
backend: unsloth  # 2-5x faster, -80% VRAM

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

Works with all training tasks: SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, and Pretrain. If unsloth is installed but not enabled, Soup will suggest it automatically.

> **Tip:** Soup auto-detects unsloth. When installed, you'll see a hint during `soup train` if you haven't enabled it yet.

## Continued Pre-training

Continue training a model on raw text for domain adaptation:

```yaml
base: meta-llama/Llama-3.1-8B
task: pretrain

data:
  train: ./data/corpus.jsonl   # {"text": "..."} or plain .txt files
  format: plaintext
  max_length: 4096

training:
  epochs: 1
  lr: 1e-5
  quantization: 4bit
```

```bash
soup init --template pretrain
soup train
```

## MoE Model Support

Fine-tune Mixture of Experts models (Mixtral, Qwen3-30B-A3B, DeepSeek V3) with ScatterMoE LoRA — applies LoRA to both attention layers and expert FFN layers:

```yaml
base: Qwen/Qwen3-30B-A3B
task: sft

training:
  moe_lora: true              # target expert + attention layers
  moe_aux_loss_coeff: 0.01    # router load-balancing loss
  quantization: 4bit
```

Soup auto-detects MoE architectures. Works with all training tasks.

```bash
soup init --template moe
soup train
```

## Vision / Multimodal Fine-tuning

Fine-tune vision-language models (LLaMA-3.2-Vision, Qwen2-VL, Pixtral) on image+text data:

```bash
# Install vision support
pip install 'soup-cli[vision]'

# Create a vision config
soup init --template vision

# Train
soup train --config soup.yaml
```

```yaml
base: meta-llama/Llama-3.2-11B-Vision-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

**Supported vision data formats:**

**LLaVA:**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this image."}, {"from": "gpt", "value": "A cat on a mat."}]}
```

**ShareGPT4V:**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nWhat does this show?"}, {"from": "gpt", "value": "Quarterly revenue."}]}
```

`soup data inspect` automatically shows image statistics (count, formats, missing files) for vision datasets.

## Audio / Speech Fine-tuning

Fine-tune audio-language models (Qwen2-Audio, Whisper) on audio+text data:

```bash
# Install audio support
pip install 'soup-cli[audio]'

# Create an audio config
soup init --template audio

# Train
soup train --config soup.yaml
```

```yaml
base: Qwen/Qwen2-Audio-7B-Instruct
task: sft
modality: audio

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  val_split: 0.1

training:
  epochs: 3
  lr: 1e-5
  quantization: 4bit
  lora:
    r: 64
    alpha: 16
```

**Audio data format:**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe this audio."}, {"role": "assistant", "content": "Hello world."}]}
```

## Quantization-Aware Training (QAT)

Train with simulated quantization for significantly better post-quantization quality compared to standard QLoRA:

```bash
# Install QAT support
pip install 'soup-cli[qat]'
```

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: 4bit
  quantization_aware: true  # Enable QAT
  lora:
    r: 64
    alpha: 16

output: ./output
```

**When to use QAT vs post-training quantization:**
- **QAT** (`quantization_aware: true`): Better quality when you plan to deploy with aggressive quantization (int8/int4). ~5-10% slower training, but the model learns to compensate for quantization noise.
- **Post-training quantization** (default): Faster training, good enough for most use cases. Quantize after training with `soup export --quant q4_k_m`.

QAT works with all training tasks (SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, Pretrain) and vision modality. Not compatible with the unsloth backend. After QAT training, export to GGUF normally with `soup export`.

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

## GRPO Training (Reasoning)

Train reasoning models with Group Relative Policy Optimization (DeepSeek-R1 style):

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: sharegpt
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy   # or 'format', or path to custom .py
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

```bash
# Create a reasoning config
soup init --template reasoning

# Train
soup train --config soup.yaml
```

**Built-in reward functions:**
- `accuracy` — checks if the final answer matches expected (supports `####` and `\boxed{}` formats)
- `format` — checks for structured `<think>...</think>` reasoning blocks

**Custom reward functions** — point to a Python file:
```python
# my_reward.py
def reward_fn(completions, **kwargs):
    """Score each completion. Return list of floats."""
    return [1.0 if "correct" in c[-1]["content"] else 0.0 for c in completions]
```
```yaml
training:
  reward_fn: ./my_reward.py
```

## PPO / Full RLHF Pipeline

Train models with the full RLHF pipeline: SFT warmup → Reward Model → PPO alignment.

```bash
# Create an RLHF config
soup init --template rlhf
```

**Step 1: SFT warmup** — fine-tune a base model on your data:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: alpaca
output: ./output_sft
```

**Step 2: Train reward model** — learn preferences from human feedback:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: reward_model
data:
  train: ./data/preferences.jsonl
  format: dpo
output: ./output_rm
```

**Step 3: PPO alignment** — optimize the policy using the reward model:
```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: ppo
data:
  train: ./data/prompts.jsonl
  format: chatml
training:
  reward_model: ./output_rm
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_penalty: 0.05
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
output: ./output_ppo
```

PPO supports two reward sources:
- **Reward model** (`reward_model`): pre-trained reward model (from step 2)
- **Reward function** (`reward_fn`): callable function (same as GRPO — `accuracy`, `format`, or custom `.py`)

## KTO Training (Unpaired Preferences)

Train with unpaired preference data — no need for chosen+rejected pairs:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: kto

data:
  train: ./data/kto_train.jsonl
  format: kto

training:
  epochs: 3
  kto_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

**KTO data format:**
```json
{"prompt": "What is 2+2?", "completion": "4", "label": true}
{"prompt": "What is 2+2?", "completion": "Fish", "label": false}
```

## ORPO Training (No Reference Model)

ORPO combines SFT and alignment in one step — no reference model needed:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: orpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  orpo_beta: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

## SimPO Training (Simple Preference)

SimPO uses length-normalized log probabilities as implicit rewards — reference-free:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: simpo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  simpo_gamma: 0.5
  cpo_alpha: 1.0
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

## IPO Training (Regularized Preference)

IPO is a theoretically grounded DPO variant with stronger regularization:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: ipo

data:
  train: ./data/preferences.jsonl
  format: dpo

training:
  epochs: 3
  ipo_tau: 0.1
  lora:
    r: 64
    alpha: 16
  quantization: 4bit
```

## DoRA (Weight-Decomposed LoRA)

Enable DoRA for improved LoRA quality with magnitude decomposition:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_dora: true  # Enable DoRA
```

Works with all training tasks and backends.

## LoRA+ (Differentiated Learning Rates)

Use different learning rates for LoRA A and B matrices:

```yaml
training:
  lr: 2e-5
  loraplus_lr_ratio: 16.0  # lr_B = lr × 16
  lora:
    r: 64
    alpha: 16
```

## rsLoRA (Rank-Stabilized Scaling)

Use rank-stabilized LoRA scaling for better performance at high ranks:

```yaml
training:
  lora:
    r: 64
    alpha: 16
    use_rslora: true  # Enable rank-stabilized scaling
```

Works with all training tasks and backends. Recommended for LoRA rank ≥ 32.

## NEFTune (Noisy Embeddings Fine-Tuning)

Add noise to embeddings during training for better chat model quality:

```yaml
training:
  neftune_alpha: 5.0  # Noise intensity (0-50, typically 5-15)
```

Works with SFT, DPO, KTO, ORPO, SimPO, and IPO tasks.

## Sample Packing

Pack multiple short samples into one sequence for faster training:

```yaml
training:
  packing: true  # Pack short samples together (faster training)
```

Works with SFT and Pretrain tasks. Warning emitted if `max_length < 256`.

## Curriculum Learning

Sort dataset by difficulty (easy → hard) for better convergence:

```yaml
training:
  curriculum: true             # Enable curriculum learning
  curriculum_metric: length    # Sort by: length, perplexity, or loss
  curriculum_buckets: 4        # Number of difficulty stages
```

## Freeze Training

Freeze bottom layers of the model — train only the top layers (like LLaMA-Factory's `finetuning_type: freeze`):

```yaml
training:
  freeze_layers: 24    # Freeze first 24 layers, train the rest
  # OR
  freeze_ratio: 0.75   # Freeze 75% of layers from the bottom
```

Works with and without LoRA. When used with LoRA, LoRA is applied only to unfrozen layers.

## Loss Watchdog

Auto-stop training when loss spikes above a threshold (like Axolotl's `loss_watchdog_threshold`):

```yaml
training:
  loss_watchdog: true           # Enable loss spike detection
  loss_watchdog_threshold: 3.0  # Stop if loss exceeds this value
  loss_watchdog_patience: 5     # Consecutive steps above threshold before stopping
```

## GaLore (Memory-Efficient Full-Parameter Training)

Train without LoRA using gradient low-rank projection — saves optimizer memory:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  quantization: none      # Required: GaLore is incompatible with quantization
  use_galore: true
  galore_rank: 128
  galore_update_proj_gap: 200
  galore_scale: 0.25
```

> **Note:** GaLore requires `quantization: none` and `backend: transformers` (not unsloth).

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

### ONNX Export

Export models to ONNX format for use with [ONNX Runtime](https://onnxruntime.ai/):

```bash
pip install 'soup-cli[onnx]'
soup export --model ./output --format onnx
soup export --model ./output --format onnx --output ./model_onnx
```

### TensorRT-LLM Export

Export models to TensorRT-LLM format for high-throughput GPU inference:

```bash
pip install 'soup-cli[tensorrt]'
soup export --model ./output --format tensorrt
soup export --model ./output --format tensorrt --output ./model_trt
```

After export, use with Ollama manually or auto-deploy:
```bash
# Manual (3-step)
echo 'FROM ./my-model.q4_k_m.gguf' > Modelfile
ollama create my-model -f Modelfile
ollama run my-model

# Auto-deploy (1-step)
soup export --model ./output --format gguf --deploy ollama --deploy-name my-model
```

### Deploy to Ollama

Deploy a GGUF model directly to your local [Ollama](https://ollama.com/) instance:

```bash
# Deploy a GGUF model
soup deploy ollama --model ./output/model.q4_k_m.gguf --name soup-my-model

# Deploy with system prompt and parameters
soup deploy ollama --model ./model.gguf --name soup-chat \
  --system "You are a helpful assistant." \
  --template chatml \
  --parameter temperature=0.7 \
  --parameter top_p=0.9

# Export + deploy in one command
soup export --model ./output --format gguf --deploy ollama

# List Soup-deployed models
soup deploy ollama --list

# Remove a model
soup deploy ollama --remove soup-my-model
```

Auto-detected chat templates: `chatml`, `llama`, `mistral`, `vicuna`, `zephyr` (or `auto` to infer from soup.yaml).

## Resume Training

Resume a training run from a checkpoint:

```bash
# Auto-detect latest checkpoint in output directory
soup train --config soup.yaml --resume auto

# Resume from a specific checkpoint
soup train --config soup.yaml --resume ./output/checkpoint-500
```

## Run Management & Cleanup

LLM training generates massive checkpoint files. Soup automatically manages an SQLite database of your training loss and metrics, empowering you to safely reclaim disk space once training is complete.

```bash
# List all historical training runs
soup runs list

# Compare two differing experiments side-by-side
soup runs compare run_202611... run_202612...

# Intelligently clean up redundant checkpoints
# (Preserves the final model and the checkpoint with the lowest loss)
soup runs clean run_202611...

# Preview space that would be reclaimed across ALL experiments
soup runs clean --all --dry-run
```

By default, the `clean` command operates in "surgical mode" (`--keep-weights`), deleting huge optimizer state files (`optimizer.pt`) from lesser checkpoints to save gigabytes, but keeping their lightweight evaluation weights just in case you want to load them later.


## Batch Inference

Run a model on a list of prompts and save results:

```bash
# JSONL input (each line: {"prompt": "..."})
soup infer --model ./output --input prompts.jsonl --output results.jsonl

# Plain text input (one prompt per line)
soup infer --model ./output --input prompts.txt --output results.jsonl

# Custom generation settings
soup infer --model ./output --input prompts.jsonl --output results.jsonl \
  --max-tokens 512 --temperature 0.3
```

Output is JSONL with `prompt`, `response`, and `tokens_generated` fields. Shows a progress bar and throughput summary.

## Inference Benchmarking

Quickly measure your model's generation speed and memory footprint before deployment:

```bash
# Benchmark local speed and VRAM usage on 3 automatically generated prompts
soup bench ./output

# Customizing benchmarking parameters
soup bench ./output --num-prompts 5 --max-tokens 256

# Use custom prompts from a text file (one per line) or JSONL
soup bench ./output --prompts-file my_prompts.txt
soup bench ./output --prompts-file bench_suite.jsonl
```

This acts as a built-in "speedometer," outputting Tokens-Per-Second (TPS), Total Latency, and Peak VRAM allocations into a clean status table.

## TensorBoard Integration

Log training metrics to TensorBoard for local visualization:

```bash
# Enable TensorBoard logging (requires: pip install tensorboard)
soup train --config soup.yaml --tensorboard

# View logs
tensorboard --logdir ./output/runs/
```

> **Note:** `--tensorboard` and `--wandb` cannot be used together. Pick one.

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

### vLLM Backend (2-4x Faster Inference)

Use [vLLM](https://github.com/vllm-project/vllm) for significantly better throughput in production:

```bash
# Install vLLM support
pip install 'soup-cli[serve-fast]'

# Start with vLLM backend
soup serve --model ./output --backend vllm

# Multi-GPU with tensor parallelism
soup serve --model ./output --backend vllm --tensor-parallel 2

# Control GPU memory usage
soup serve --model ./output --backend vllm --gpu-memory 0.8
```

> **Tip:** Soup auto-detects vLLM. When installed, you'll see a hint during `soup serve` if you haven't enabled it yet.

### SGLang Backend (v0.17.0+)

Use [SGLang](https://github.com/sgl-project/sglang) as an alternative high-throughput backend:

```bash
# Install SGLang support
pip install 'soup-cli[sglang]'

# Start with SGLang backend
soup serve --model ./output --backend sglang

# Multi-GPU with tensor parallelism
soup serve --model ./output --backend sglang --tensor-parallel 2
```

### Speculative Decoding (v0.16.0+)

Use a smaller draft model to speed up generation (2-3x faster):

```bash
# Transformers backend — uses HF assisted generation
soup serve --model ./output --speculative-decoding small-draft-model --spec-tokens 5

# vLLM backend — uses vLLM native speculative decoding
soup serve --model ./output --backend vllm --speculative-decoding small-draft-model
```

> **Note (v0.10.10+):** `max_tokens` is capped at 16,384 per request. Error details are never exposed in HTTP responses.

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

# Use a local OpenAI-compatible server (soup serve, Ollama, etc.)
soup data generate --prompt "..." --provider server --api-base http://localhost:11434/v1
```

### Multi-Provider Support (v0.20.0+)

```bash
# Generate via local Ollama instance
soup data generate --prompt "..." --provider ollama --model llama3.1
soup data generate --prompt "..." --ollama-model llama3.1  # shorthand

# Generate via Anthropic Claude API (set ANTHROPIC_API_KEY env var)
soup data generate --prompt "..." --provider anthropic --model claude-3-haiku-20240307

# Generate via local vLLM server
soup data generate --prompt "..." --provider vllm --model meta-llama/Llama-3.1-8B-Instruct
```

### Domain Templates (v0.20.0+)

```bash
# Code instruction pairs (Python, JS, Go, Rust, Java)
soup data generate --prompt "..." --template code --language Python --task-type function

# Multi-turn conversations
soup data generate --prompt "..." --template conversation --turns 6 --topic "science"

# QA from context document
soup data generate --prompt "..." --template qa --context document.txt

# Preference data (DPO/KTO/ORPO)
soup data generate --prompt "..." --template preference --pref-task dpo

# Chain-of-thought reasoning (GRPO)
soup data generate --prompt "..." --template reasoning --domain math
```

### Quality Pipeline (v0.20.0+)

```bash
# Auto-validate after generation (remove malformed entries)
soup data generate --prompt "..." --validate

# Auto-filter by quality (coherence scoring)
soup data generate --prompt "..." --filter

# Auto-dedup (MinHash, requires: pip install 'soup-cli[data]')
soup data generate --prompt "..." --dedup

# Full quality pipeline: validate + filter + dedup
soup data generate --prompt "..." --quality-pipeline
```

## Config Migration

Switch from other tools with one command:

```bash
# Import from LLaMA-Factory
soup migrate --from llamafactory llama3_lora_sft.yaml

# Import from Axolotl
soup migrate --from axolotl axolotl_config.yml

# Import from Unsloth notebook
soup migrate --from unsloth finetune.ipynb

# Preview without writing
soup migrate --from llamafactory config.yaml --dry-run
```

Automatically maps model, LoRA, training params, quantization, and task type. Warns about unsupported features.

## Ready-Made Recipes

29 pre-built configs for popular models — no guessing hyperparameters:

```bash
# List all recipes
soup recipes list

# Preview a recipe
soup recipes show llama3.1-8b-sft

# Use a recipe (writes soup.yaml)
soup recipes use llama3.1-8b-sft

# Search by task or keyword
soup recipes search --task grpo
soup recipes search "reasoning"
soup recipes search --size 7b
```

Recipes cover Llama 3.1/3.2, Qwen 2.5/3, Mistral, Gemma 3, Phi-4, DeepSeek R1 across SFT, DPO, GRPO, KTO, ORPO, SimPO, IPO, PPO, embedding, pretrain, and vision tasks.

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

## Multi-GPU / DeepSpeed / FSDP

Train on multiple GPUs with DeepSpeed or PyTorch FSDP2:

```bash
# DeepSpeed ZeRO Stage 2 (recommended for most cases)
soup train --config soup.yaml --deepspeed zero2

# DeepSpeed ZeRO Stage 3 (for very large models)
soup train --config soup.yaml --deepspeed zero3

# DeepSpeed ZeRO Stage 2 with CPU offload (memory-constrained)
soup train --config soup.yaml --deepspeed zero2_offload

# FSDP2 Full Shard (native PyTorch, like ZeRO-3)
soup train --config soup.yaml --fsdp full_shard

# FSDP2 Shard Grad Op (like ZeRO-2)
soup train --config soup.yaml --fsdp shard_grad

# FSDP2 Full Shard with CPU offload
soup train --config soup.yaml --fsdp full_offload
```

## Performance + Long-Context

Optimize training throughput and extend context windows:

```yaml
# soup.yaml — performance options
training:
  use_liger: true            # Liger Kernel fused ops (20-60% memory savings)
  use_flash_attn: true       # FlashAttention v2/v3 auto-detection
  gradient_checkpointing: true  # Required for long sequences

  # Long-context (128k+ tokens)
  rope_scaling_type: dynamic  # RoPE scaling: linear, dynamic, yarn, longrope
  # use_ring_attention: true  # Sequence parallelism across GPUs

data:
  max_length: 131072          # Up to 1M tokens supported
```

Install optional performance packages:

```bash
pip install 'soup-cli[liger]'     # Liger Kernel fused operations
pip install flash-attn --no-build-isolation  # FlashAttention
pip install 'soup-cli[ring-attn]' # Ring FlashAttention (sequence parallelism)
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

Shows: Python version, GPU availability, system resources (RAM/Disk), all dependency versions, and fix suggestions.

## Version Info

```bash
# Basic version
soup version

# Machine-readable output
soup version --json
# -> {"version": "0.24.0", "python": "3.11.5", "platform": "linux"}

# Full system info (useful for bug reports)
soup version --full
# -> soup v0.17.3 | Python 3.11.5 | CUDA 12.1 | extras: serve, data

# Full system info in JSON
soup version --full --json
# -> {"version": "0.24.0", "python": "3.11.5", "platform": "linux", "torch": "2.2.0", ...}
```

## Web UI

Launch a local web interface to manage experiments, start training, explore data, and chat with models — all from your browser.

```bash
pip install 'soup-cli[ui]'
soup ui
# -> opens http://127.0.0.1:7860 in your browser
# -> prints auth token to console
```

**Pages:**
- **Dashboard** — view all experiment runs, loss charts, system info, multi-run comparison
- **New Training** — create configs from templates or 29 ready-made recipes, validate, start training with live SSE log streaming and progress bar
- **Data Explorer** — browse and inspect datasets (JSONL, JSON, CSV, Parquet)
- **Model Chat** — chat with streaming responses, configurable temperature/top_p/max_tokens, system prompt, adapter selection, markdown rendering, chat export

**v0.24.2 Enhancements:**
- **Training Live Monitor** — real-time SSE log streaming, live metrics, progress bar with ETA
- **Enhanced Metrics** — 2x2 chart grid (loss, LR, grad_norm, throughput) + GPU memory chart, eval results table
- **Multi-Run Compare** — overlay loss curves from up to 5 runs side-by-side
- **Chat Upgrade** — SSE streaming via proxy, typing indicator, cancel button, markdown renderer (bold, italic, code blocks), chat export as JSON
- **Config Builder** — recipe dropdown (29 configs), config schema API for dynamic form generation

**Security (v0.10.10+):** The Web UI generates a random auth token at startup (printed to console). All mutating endpoints (start/stop training, delete runs, inspect data, validate config) require `Authorization: Bearer <token>` header. CORS is restricted to the served origin. Data inspection is sandboxed to the working directory.

```bash
# Custom port, don't auto-open browser
soup ui --port 8080 --no-browser
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

Soup supports these formats (auto-detected). Files can be JSONL, JSON, CSV, Parquet, or TXT.

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

**DPO / ORPO / SimPO / IPO (preference pairs):**
```json
{"prompt": "Explain gravity", "chosen": "Gravity is a force...", "rejected": "I don't know"}
```

**KTO (unpaired preferences):**
```json
{"prompt": "Explain gravity", "completion": "Gravity is a force...", "label": true}
```

**LLaVA (vision):**
```json
{"image": "photo.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this."}, {"from": "gpt", "value": "A cat."}]}
```

**ShareGPT4V (vision):**
```json
{"image": "chart.png", "conversations": [{"from": "human", "value": "<image>\nExplain this chart."}, {"from": "gpt", "value": "Revenue growth."}]}
```

**Plaintext (pre-training):**
```json
{"text": "Raw text document for continued pre-training..."}
```
Or use `.txt` files directly (one document per line).

**Embedding (sentence embedding pairs/triplets):**
```json
{"anchor": "What is Python?", "positive": "Python is a programming language."}
{"anchor": "What is Python?", "positive": "A programming language.", "negative": "A type of snake."}
```

**Audio (speech + conversation):**
```json
{"audio": "recording.wav", "messages": [{"role": "user", "content": "Transcribe."}, {"role": "assistant", "content": "Hello world."}]}
```

## Data Tools

```bash
# Inspect a dataset
soup data inspect ./data/train.jsonl

# Validate format (auto-detects if --format not specified)
soup data validate ./data/train.jsonl
soup data validate ./data/train.jsonl --format alpaca

# Convert between formats
soup data convert ./data/train.jsonl --to sharegpt --output converted.jsonl

# Merge multiple datasets
soup data merge data1.jsonl data2.jsonl --output merged.jsonl --shuffle

# Remove near-duplicates (requires: pip install 'soup-cli[data]')
soup data dedup ./data/train.jsonl --threshold 0.8

# Extended statistics (length distribution, token counts, languages)
soup data stats ./data/train.jsonl

# Filter by quality (perplexity + coherence scoring)
soup data filter ./data/train.jsonl --coherence 0.3
soup data filter ./data/train.jsonl --perplexity 500 --coherence 0.3
soup data filter ./data/train.jsonl --score-only  # add scores without filtering
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

Full-featured evaluation platform with standard benchmarks, custom evals, LLM-as-a-judge, and human evaluation:

```bash
# Install eval dependencies
pip install 'soup-cli[eval]'

# Standard benchmarks (wraps lm-evaluation-harness)
soup eval benchmark --model ./output --benchmarks mmlu,gsm8k,hellaswag

# Custom eval tasks from JSONL
soup eval custom --tasks eval_tasks.jsonl --model ./output

# LLM-as-a-judge (score model outputs using GPT-4o, Ollama, etc.)
soup eval judge --target responses.jsonl --model gpt-4o-mini --provider openai
soup eval judge --target responses.jsonl --model llama3.1 --provider ollama

# Auto-eval after training (configure in soup.yaml)
soup eval auto --config soup.yaml

# Compare eval results between two training runs
soup eval compare run_20260301_143052_a1b2 run_20260315_091023_c3d4

# Local leaderboard across all evaluated models
soup eval leaderboard
soup eval leaderboard --format json
soup eval leaderboard --format csv

# Human A/B evaluation with Elo ratings
soup eval human --input prompts.jsonl --model-a ./model_a --model-b ./model_b
```

### Custom Eval Format

```jsonl
{"prompt": "What is 2+2?", "expected": "4", "category": "math", "scoring": "exact"}
{"prompt": "Explain gravity", "expected": "force.*attraction", "scoring": "regex"}
{"prompt": "Capital of France?", "expected": "Paris", "scoring": "contains"}
```

### Auto-Eval Config (soup.yaml)

```yaml
eval:
  auto_eval: true
  benchmarks: [mmlu, gsm8k]
  custom_tasks: eval_tasks.jsonl
  judge:
    model: gpt-4o-mini
    provider: openai
```

## All Commands

```
soup init [--template chat|code|...|audio]       Create config
soup train --config soup.yaml                 Start training
soup train --config soup.yaml --tensorboard   Train with TensorBoard logging
soup train --config soup.yaml --fsdp full_shard  Train with FSDP2
soup infer --model ./output --input p.jsonl   Batch inference
soup chat --model ./output                    Interactive chat
soup push --model ./output --repo user/name   Upload to HuggingFace
soup merge --adapter ./output                 Merge LoRA with base model
soup export --model ./output --format gguf    Export to GGUF (Ollama)
soup export --model ./output --deploy ollama  Export GGUF + auto-deploy to Ollama
soup export --model ./output --format onnx    Export to ONNX
soup export --model ./output --format tensorrt Export to TensorRT-LLM
soup export --model ./output --format awq     Export to AWQ (4-bit)
soup export --model ./output --format gptq    Export to GPTQ (4-bit)
soup deploy ollama --model m.gguf --name x    Deploy GGUF to Ollama
soup deploy ollama --list                     List Soup-deployed models
soup deploy ollama --remove <name>            Remove model from Ollama
soup eval benchmark --model ./output          Evaluate on standard benchmarks
soup eval custom --tasks eval.jsonl           Custom eval tasks from JSONL
soup eval judge --target resp.jsonl           LLM-as-a-judge evaluation
soup eval auto --config soup.yaml             Auto-eval from config
soup eval compare <run1> <run2>               Compare eval results
soup eval leaderboard                         Local model leaderboard
soup eval human --input p.jsonl               Human A/B evaluation
soup serve --model ./output --port 8000       OpenAI-compatible API server
soup serve --model ./output --backend vllm    vLLM backend (2-4x throughput)
soup serve --model ./output --backend sglang  SGLang backend
soup serve --model ./output --speculative-decoding draft-model  Speculative decoding
soup sweep --config soup.yaml --param lr=...  Hyperparameter search
soup diff --model-a ./a --model-b ./b         Compare two models
soup data inspect <path>                      View dataset stats
soup data validate <path>                     Check format (auto-detect)
soup data convert <path> --to chatml          Convert between formats
soup data merge data1.jsonl data2.jsonl       Combine datasets
soup data dedup <path> --threshold 0.8        Remove duplicates (MinHash)
soup data stats <path>                        Extended statistics
soup data generate --prompt "..." --count 100 Generate synthetic data
soup data generate ... --provider ollama      Use local Ollama instance
soup data generate ... --provider anthropic   Use Claude API
soup data generate ... --provider vllm        Use local vLLM server
soup data generate ... --template code        Domain templates (code/conversation/qa/preference/reasoning)
soup data generate ... --quality-pipeline     Auto validate + filter + dedup
soup data filter <path> --coherence 0.3       Quality filter (perplexity/coherence)
soup data sample <path> --n 1000             Random sample subset
soup data sample <path> --n 1000 --strategy diverse  Cluster-based diverse sampling
soup data sample <path> --n 1000 --strategy hard     Sample hardest examples
soup data sample <path> --pct 10             Sample by percentage
soup data split <path> --val 10 --test 10    Split into train/val/test
soup data split <path> --val 500 --absolute  Split with absolute counts
soup data split <path> --val 10 --stratify category  Stratified by field
soup data search "code instructions"         Search HuggingFace Hub for datasets
soup data search --sort likes --limit 10     Sort and paginate search results
soup data preview teknium/OpenHermes-2.5     Preview remote dataset metadata
soup data download user/dataset -o data.jsonl  Download HF dataset as JSONL
soup data download user/ds --samples 1000    Stream first 1000 samples
soup data register --name my-ds --path d.jsonl --format alpaca  Register dataset
soup data unregister --name my-ds            Remove from registry
soup data registry                           List all registered datasets
soup profile --config soup.yaml              Estimate memory/speed before training
soup profile --config soup.yaml --gpu a100   Estimate for specific GPU
soup profile --config soup.yaml --json       Machine-readable output
soup adapters list ./output/                 Scan for LoRA adapters
soup adapters info ./output/checkpoint-500/  Show adapter metadata
soup adapters compare adapter1/ adapter2/    Compare two adapters
soup serve --model m --adapters chat=./c code=./d  Multi-adapter serving
soup migrate --from llamafactory config.yaml  Import config from LLaMA-Factory
soup migrate --from axolotl config.yml        Import config from Axolotl
soup migrate --from unsloth notebook.ipynb    Import config from Unsloth notebook
soup migrate --from llamafactory c.yaml --dry-run  Preview without writing
soup recipes list                             List all 29 ready-made recipes
soup recipes show llama3.1-8b-sft            Print recipe YAML
soup recipes use llama3.1-8b-sft             Copy recipe to soup.yaml
soup recipes search "reasoning"              Search by keyword/task/size
soup runs                                     List training runs
soup runs show <run_id>                       Run details + loss graph
soup runs compare <run_1> <run_2>             Compare two runs
soup ui [--port 7860]                         Web UI (experiments, training, data)
soup doctor                                   Check environment
soup quickstart [--dry-run]                   Full demo
soup version [--full] [--json]                Show version (--full: system info, --json: JSON output)
soup --verbose <command>                      Full traceback on errors
```

## Supported Models

Soup works with **any** of the **340,000+** text-generation models on [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation). If a model supports `AutoModelForCausalLM`, it works with Soup — zero config changes needed.

### Recommended Models

| Model Family | Models | Sizes | Best For |
|---|---|---|---|
| **Llama 4** | Llama-4-Scout-17B, Llama-4-Maverick-17B | 17B | General, multilingual |
| **Llama 3.x** | Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct | 1B–70B | Chat, instruction following |
| **Llama 3.2 Vision** | Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision | 11B–90B | Image understanding |
| **Gemma 3** | Gemma-3-4B-IT, Gemma-3-9B-IT, Gemma-3-27B-IT | 4B–27B | Efficient, multilingual |
| **Qwen 3** | Qwen3-8B, Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B | 0.6B–235B | Reasoning, code, MoE |
| **Qwen 2.5** | Qwen2.5-7B-Instruct, Qwen2.5-Coder-32B-Instruct | 0.5B–72B | Code, math |
| **DeepSeek** | DeepSeek-R1-Distill-Llama-8B, DeepSeek-V3-0324 | 1.5B–671B | Reasoning (GRPO), code |
| **Phi-4** | Phi-4-14B, Phi-4-mini-reasoning | 3.8B–14B | Compact reasoning |
| **Mistral** | Mistral-7B-Instruct-v0.3, Mistral-Small-24B-Instruct | 7B–24B | Fast, efficient |
| **Mixtral** | Mixtral-8x7B-Instruct-v0.1, Mixtral-8x22B | 47B–141B | MoE architecture |
| **CodeLlama** | CodeLlama-7b-Instruct-hf, CodeLlama-34b-Instruct | 7B–34B | Code generation |
| **StarCoder 2** | StarCoder2-15B, StarCoder2-7B | 3B–15B | Code completion |
| **Yi** | Yi-1.5-34B-Chat, Yi-1.5-9B-Chat | 6B–34B | Multilingual chat |
| **InternLM 3** | InternLM3-8B-Instruct | 8B | Chinese + English |
| **Falcon** | Falcon-11B, Falcon-40B-Instruct | 7B–180B | Open-weight |

### Vision Models (with `modality: vision`)

| Model | Size | Supported Formats |
|---|---|---|
| LLaMA-3.2-11B-Vision-Instruct | 11B | LLaVA, ShareGPT4V |
| Qwen2-VL-7B-Instruct | 7B | LLaVA, ShareGPT4V |
| Pixtral-12B-2409 | 12B | LLaVA, ShareGPT4V |

### Quick Size Guide

| VRAM | Max Model (QLoRA 4-bit) | Example |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

> **Note:** Soup auto-detects your GPU and estimates the optimal batch size. Use `soup doctor` to check your setup.

## Docker

Run Soup without installing CUDA or PyTorch locally using the official Docker image (published to GitHub Container Registry on every release). This is the fastest way to get started and avoid dependency hell.

```bash
# Pull and run
docker pull ghcr.io/makazhanalpamys/soup:latest
docker run --gpus all -v $(pwd):/workspace ghcr.io/makazhanalpamys/soup train --config soup.yaml

# Or with compose (builds locally if image not pulled)
docker compose up
```

## Requirements

- Python 3.9+
- GPU with CUDA (recommended) or Apple Silicon (MPS) or CPU (experimental)
- 8 GB+ VRAM for 7B models with QLoRA

> **CPU note:** All training tasks (SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, Pretrain) work on CPU but will be very slow. Quantization (`4bit`/`8bit`) is auto-disabled on CPU. GRPO on CPU uses `min_new_tokens=1` to prevent empty generation errors. A default chat template is set automatically if the tokenizer lacks one. PPO datasets are tokenized before training to ensure compatibility with trl's experimental API.

### Optional Extras

| Extra | Install | What it adds |
|---|---|---|
| `vision` | `pip install 'soup-cli[vision]'` | Vision/multimodal fine-tuning (Pillow) |
| `qat` | `pip install 'soup-cli[qat]'` | Quantization-Aware Training (torchao) |
| `fast` | `pip install 'soup-cli[fast]'` | Unsloth backend (2-5x faster, -80% VRAM) |
| `ui` | `pip install 'soup-cli[ui]'` | Web UI + inference server (FastAPI + uvicorn) |
| `serve` | `pip install 'soup-cli[serve]'` | Inference server (FastAPI + uvicorn) |
| `serve-fast` | `pip install 'soup-cli[serve-fast]'` | vLLM inference backend (2-4x throughput) |
| `data` | `pip install 'soup-cli[data]'` | Deduplication (MinHash via datasketch) |
| `eval` | `pip install 'soup-cli[eval]'` | Benchmark evaluation (lm-evaluation-harness) |
| `deepspeed` | `pip install 'soup-cli[deepspeed]'` | Multi-GPU training (DeepSpeed ZeRO) |
| `liger` | `pip install 'soup-cli[liger]'` | Liger Kernel fused ops (20-60% memory savings) |
| `ring-attn` | `pip install 'soup-cli[ring-attn]'` | Ring FlashAttention (sequence parallelism) |
| `onnx` | `pip install 'soup-cli[onnx]'` | ONNX export (optimum + onnxruntime) |
| `tensorrt` | `pip install 'soup-cli[tensorrt]'` | TensorRT-LLM export (high-throughput GPU inference) |
| `dev` | `pip install 'soup-cli[dev]'` | Tests + linting (pytest, ruff) |

## Troubleshooting

### `ImportError: DLL load failed while importing _C` (Windows)

PyTorch's C extension fails to load. Common causes:

```bash
# Fix: reinstall PyTorch with the correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Multiple Python versions conflict

If `pip show soup-cli` shows a different version than `soup version`, you have multiple Python installations with separate packages.

```bash
# Check which Python is active
python --version
which python    # Linux/macOS
where python    # Windows

# Fix: use a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
pip install soup-cli
```

### Quick environment check

```bash
soup doctor    # Shows GPU, system resources, dependencies, and version info
```

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
