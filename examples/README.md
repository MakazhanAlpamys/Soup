# Soup Examples

Real-world configuration examples and sample datasets to get you running quickly.

## Quick Start with Examples

### 1. Basic SFT (Supervised Fine-Tuning)

Fine-tune TinyLlama on a small instruction-following dataset:

```bash
soup train --config examples/configs/sft_basic.yaml
```

**What it does:**
- Trains TinyLlama-1.1B for 1 epoch
- Uses LoRA for efficient memory usage
- Outputs to `./output_sft_basic/`
- Takes ~2-3 minutes on a consumer GPU

### 2. Chat Assistant (DPO)

Train a chat model with preference learning:

```bash
soup train --config examples/configs/dpo_chat.yaml
```

**What it does:**
- Uses Llama 2 7B base model
- Trains with DPO (Direct Preference Optimization) on chat preferences
- Better alignment than SFT alone
- Outputs to `./output_dpo_chat/`

### 3. Reasoning Model (GRPO)

Fine-tune a reasoning model with step-by-step answer verification:

```bash
soup train --config examples/configs/grpo_reasoning.yaml
```

**What it does:**
- Trains on reasoning tasks (math, logic)
- Uses GRPO (Group Relative Policy Optimization) to optimize for correctness
- Generates multiple outputs per prompt and selects the best
- Outputs to `./output_reasoning/`

### 4. Vision Model

Fine-tune LLaMA-Vision on image-caption pairs:

```bash
soup train --config examples/configs/vision_llama.yaml
```

**What it does:**
- Trains LLaMA-3.2-Vision-90B on image-text data
- Uses LLaVA format for images + text
- Outputs to `./output_vision/`

### 5. Alignment Methods (KTO / ORPO / SimPO / IPO)

Train with alternative preference optimization:

```bash
# KTO — unpaired preference (only needs thumbs up/down labels)
soup init --template kto
soup train

# ORPO — reference-free alignment (no reference model needed)
soup init --template orpo
soup train

# SimPO — length-normalized preference optimization
soup init --template simpo
soup train

# IPO — regularized preference (squared hinge loss)
soup init --template ipo
soup train
```

### 6. Continued Pre-training

Continue training on raw text corpora:

```bash
soup init --template pretrain
soup train
```

**What it does:**
- Trains on plain text (`.txt` files or JSONL with `text` field)
- No instruction format needed — just raw text
- Useful for domain adaptation (legal, medical, code)

### 7. MoE Models

Fine-tune Mixture-of-Experts models (Qwen3, Mixtral, DeepSeek V3):

```bash
soup init --template moe
soup train
```

**What it does:**
- Auto-detects MoE architecture (ScatterMoE / SwitchTransformers)
- `moe_lora: true` targets expert-specific LoRA modules
- Optional `moe_aux_loss_coeff` for load balancing

### 8. Long-Context Fine-Tuning (128k+)

Extend context windows for long-document understanding:

```bash
soup init --template longcontext
soup train
```

**What it does:**
- Uses RoPE scaling (dynamic) to extend context to 128k tokens
- Enables gradient checkpointing and FlashAttention for memory efficiency
- Supports `linear`, `dynamic`, `yarn`, `longrope` scaling types
- Optional Liger Kernel for fused ops: `pip install 'soup-cli[liger]'`

### 9. Embedding Model Fine-Tuning

Fine-tune sentence embedding models (BGE, E5, GTE) with contrastive or triplet loss:

```bash
soup init --template embedding
soup train
```

**What it does:**
- Supports contrastive, triplet, and cosine loss functions
- Configurable pooling: mean, CLS, or last token
- Works with pair data (`anchor` + `positive`) or triplets (`+ negative`)
- Compatible with BGE, E5, GTE, INSTRUCTOR, and any HuggingFace model

### 10. Audio / Speech Model

Fine-tune audio-language models (Qwen2-Audio, Whisper):

```bash
pip install 'soup-cli[audio]'
soup init --template audio
soup train
```

**What it does:**
- Trains on audio+text pairs (WAV/MP3 files + conversation)
- Supported models: Qwen2-Audio, Whisper (via transformers)
- Uses `modality: audio` with `format: audio` data

### 11. Batch Inference

Run inference on a batch of prompts:

```bash
soup infer --model ./output_sft_basic/ --input prompts.jsonl --output results.jsonl
```

### 12. Full RLHF Pipeline

Complete reinforcement learning from human feedback:

```bash
# Step 1: Pre-train with SFT
soup train --config examples/configs/rlhf_step1_sft.yaml

# Step 2: Train a reward model
soup train --config examples/configs/rlhf_step2_reward.yaml

# Step 3: PPO with reward model
soup train --config examples/configs/rlhf_step3_ppo.yaml
```

## Dataset Formats

Datasets are included in JSONL format. Soup auto-detects and normalizes:

- **Alpaca**: `instruction`, `input`, `output` fields
- **ShareGPT**: `conversations` with `from`/`value` fields
- **ChatML**: OpenAI-style `messages` with `role`/`content`
- **DPO/ORPO/SimPO/IPO**: `prompt` + `chosen` + `rejected` fields
- **KTO**: `prompt` + `completion` + `label` fields
- **LLaVA / ShareGPT4V**: Vision format with `image` + `conversations`
- **Plaintext**: Raw `.txt` files or JSONL with `text` field (for pre-training)
- **Audio**: `audio` path + `messages` (for audio/speech models)

### Example: Inspect a Dataset

```bash
soup data inspect examples/data/alpaca_tiny.jsonl
```

Output:
```
📊 Dataset Statistics

Format detected: alpaca
Total entries: 50
Sample 1:
  instruction: "Identify the odd one out"
  input: "twitter, instagram, skype"
  output: "skype"
```

### Example: Convert Between Formats

```bash
# Convert Alpaca to ChatML
soup data convert examples/data/alpaca_tiny.jsonl \
  --from alpaca --to chatml \
  --output alpaca_as_chatml.jsonl
```

## Directory Structure

```
examples/
  configs/              # YAML configuration files
    sft_basic.yaml
    dpo_chat.yaml
    grpo_reasoning.yaml
    vision_llama.yaml
    rlhf_step1_sft.yaml
    rlhf_step2_reward.yaml
    rlhf_step3_ppo.yaml
  
  data/                 # Sample datasets (JSONL)
    alpaca_tiny.jsonl
    chat_preferences.jsonl
    reasoning_math.jsonl
```

## Using Your Own Data

1. **Prepare data** in one of the supported formats
2. **Update the config** with your data path:

```yaml
data:
  path: /path/to/your/data.jsonl
  format: alpaca  # or sharegpt, chatml, llava
```

3. **Run training**:

```bash
soup train --config your_config.yaml
```

## Tips & Tricks

### Save Space: Use Quantization

Add quantization to reduce model size:

```yaml
quantization: int8  # Reduces memory by 4x
```

### Speed Up Training: Use Unsloth Backend

Unsloth is 2-5x faster training:

```bash
pip install 'soup-cli[fast]'
```

Then in your config:

```yaml
backend: unsloth
```

### Monitor Training: Use Weights & Biases

Enable W&B logging:

```bash
pip install wandb
soup train --config your_config.yaml --wandb
```

### Export for Inference: Convert to GGUF

After training, convert for Ollama/llama.cpp:

```bash
soup export output_sft_basic/ --output model.gguf --quant q8_0
```

Then use with Ollama:

```bash
ollama create my-model -f Ollama.modelfile
```

### Merge LoRA Adapter

Merge your LoRA adapter into a standalone model:

```bash
soup merge output_sft_basic/ --output merged_model/
```

## Common Issues

### "CUDA out of memory"

- Reduce `batch_size` in config
- Enable quantization: `quantization: int8`
- Use smaller model: Mistral-7B instead of Llama-70B

### "Dataset not found"

- Check file path in config (use absolute path if unsure)
- Verify format is correct: `soup data inspect your_data.jsonl`

### "Model not found on Hugging Face"

- Check model ID spelling
- Ensure you have HuggingFace token: `huggingface-cli login`
- Or use a different model that's publicly available

## Creating Your Own Configs

### Minimal Config Template

```yaml
model: tinyllama-1.1b
data:
  path: ./your_data.jsonl
  format: alpaca
task: sft
lora_r: 16
lora_alpha: 32
batch_size: 32
num_epochs: 3
learning_rate: 5e-4
output_dir: ./output/
```

### Advanced Config Template

```yaml
model: llama-2-7b
data:
  path: ./dataset.jsonl
  format: sharegpt
task: dpo
backend: unsloth
quantization: int8
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
batch_size: 16
gradient_accumulation_steps: 4
num_epochs: 2
learning_rate: 1e-4
warmup_ratio: 0.1
max_seq_length: 2048
output_dir: ./output_advanced/
```

See [config schema documentation](../CLAUDE.md#config-system) for all available options.

## Learn More

- **README**: [Main documentation](../README.md)
- **CONTRIBUTING**: [How to contribute](../CONTRIBUTING.md)
- **CLAUDE.md**: [Architecture and detailed docs](../CLAUDE.md)

## Questions?

- Check the [GitHub Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)
- Open an [Issue](https://github.com/MakazhanAlpamys/Soup/issues)
- Read [SECURITY.md](../SECURITY.md) for security questions

Happy training! 🍲
