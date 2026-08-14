# Soup Examples

Configuration examples and sample datasets.

Two kinds of files live here, and they promise different things:

- **Runnable examples** — parse and run as-is against the bundled fixtures in
  [`data/`](data/). They exist to prove your setup works end-to-end. The fixtures are
  5–10 rows, so these runs finish quickly and produce **nothing useful** — they are a
  smoke test, not training. Point `data.train` at real data for that.
- **Templates** — valid configs where you supply the data. They will not run until
  you do.

See [data/README.md](data/README.md) for what the fixtures are and where to get real data.

---

## Runnable examples

| Config | Task | Base model | Fixture | Needs |
|--------|------|------------|---------|-------|
| [`configs/sft_basic.yaml`](configs/sft_basic.yaml) | sft | TinyLlama-1.1B-Chat | `alpaca_tiny.jsonl` (10 rows) | 4 GB |
| [`configs/grpo_reasoning.yaml`](configs/grpo_reasoning.yaml) | grpo | TinyLlama-1.1B-Chat | `reasoning_math.jsonl` (5 rows) | 4 GB |
| [`configs/rlhf_step1_sft.yaml`](configs/rlhf_step1_sft.yaml) | sft | TinyLlama-1.1B-Chat | `alpaca_tiny.jsonl` (10 rows) | 4 GB |
| [`configs/rlhf_step2_reward.yaml`](configs/rlhf_step2_reward.yaml) | reward_model | TinyLlama-1.1B-Chat | `chat_preferences.jsonl` (5 rows) | 4 GB |
| [`configs/rlhf_step3_ppo.yaml`](configs/rlhf_step3_ppo.yaml) | ppo | TinyLlama-1.1B-Chat | `alpaca_tiny.jsonl` (10 rows) | 4 GB |
| [`configs/dpo_example.yaml`](configs/dpo_example.yaml) | dpo | Llama-3.1-8B-Instruct | `dpo_sample.jsonl` (8 rows) | ~13 GB |
| [`configs/dpo_chat.yaml`](configs/dpo_chat.yaml) | dpo | Llama-2-7b-chat | `chat_preferences.jsonl` (5 rows) | ~22 GB |

The five TinyLlama configs are verified to clear the pre-flight on a 4 GB card. The two
larger-base configs cannot — their weights alone are 4.0 GB and 7.0 GB — so the "Needs"
column is what `soup train` predicts for them, margin included.

Run any of them:

```bash
soup train --config examples/configs/sft_basic.yaml
```

Validate a config without training:

```bash
soup train --config examples/configs/sft_basic.yaml --dry-run
```

Gated models (Llama-2, Llama-3.1) need Hugging Face access and a login first:
`huggingface-cli login`. The TinyLlama configs need neither.

`soup train` estimates peak VRAM before it loads anything and refuses to start a run it
predicts won't fit, printing the breakdown and what to change. The TinyLlama configs are
sized for a small card (`batch_size: 4`, `max_length: 512`) — on a bigger one you can
raise both.

### Basic SFT

`sft_basic.yaml` — instruction tuning with LoRA (r=16) on alpaca-format rows. The
smallest complete config in the repo; a good base to copy.

### Preference alignment (DPO)

`dpo_chat.yaml` and `dpo_example.yaml` both train on `prompt` / `chosen` / `rejected`
pairs. `dpo_example.yaml` adds QLoRA (`quantization: 4bit`) for memory-efficient
training on a larger base, and `dpo_beta: 0.1` sets how hard the KL penalty pulls
back toward the reference model.

### Reasoning (GRPO)

`grpo_reasoning.yaml` generates `num_generations: 4` completions per prompt and scores
them with the built-in `accuracy` reward. See
[docs/training.md](../docs/training.md) for custom reward functions.

### Full RLHF pipeline

Three configs, run in order — step 3 reads the reward model that step 2 writes:

```bash
soup train --config examples/configs/rlhf_step1_sft.yaml
soup train --config examples/configs/rlhf_step2_reward.yaml
soup train --config examples/configs/rlhf_step3_ppo.yaml
```

---

## Templates

Valid configs that need your data before they run.

### Vision

[`configs/vision_llama.yaml`](configs/vision_llama.yaml) — LLaMA-3.2-11B-Vision on
image + conversation pairs in LLaVA format. There is no vision fixture in this repo
because we do not commit image files, so `data.train` and `data.image_dir` are
placeholders. Replace both, then:

```bash
soup train --config examples/configs/vision_llama.yaml
```

### Built-in templates

`soup init` writes a starter `soup.yaml` you fill in with your own data:

```bash
soup init --template kto     # unpaired preference (thumbs up/down labels)
soup init --template orpo    # reference-free alignment
soup init --template simpo   # length-normalized preference
soup init --template ipo     # regularized preference (squared hinge)
soup init --template pretrain    # continued pre-training on raw text
soup init --template moe         # Mixture-of-Experts (Qwen3, Mixtral, DeepSeek V3)
soup init --template longcontext # RoPE scaling for 128k+ context
soup init --template embedding   # sentence embeddings (BGE, E5, GTE)
soup init --template vision      # vision-language
soup init --template audio       # audio/speech — needs pip install "soup-cli[audio]"
soup train
```

Full list: `soup init --help`. Available templates are `audio`, `bco`, `chat`, `code`,
`embedding`, `eu-ai-act`, `hipaa`, `ipo`, `kto`, `longcontext`, `medical`, `moe`,
`orpo`, `pretrain`, `reasoning`, `rlhf`, `simpo`, `soc2`, `sr-11-7`, `tool-calling`,
`vision`.

---

## Synthetic-data workflow

Generate training data from a local LLM, filter and score it, then train on the result.
Walkthrough in [synthetic_workflow.md](synthetic_workflow.md); the config it trains with
is [synthetic_workflow.yaml](synthetic_workflow.yaml) (a template — it reads
`./synth_clean.jsonl`, which the workflow produces).

## Reward-hacking mitigation demo

[`reward_hacking/rewards.py`](reward_hacking/rewards.py) provides synthetic reward
functions for the closed-loop mitigation feature (`soup train
--reward-hack-mitigation`): a gameable `length_hack_reward` / `sentinel_reward` proxy
decoupled from a held-out `true_score`. Point a GRPO config's `training.reward_fn` at a
`.py` that re-exports one as `reward_fn`, enable `reward_hack_detector: info_rm` +
`reward_hack_mitigation: kl_control`, and watch `mitigation_log.jsonl` under the run's
output dir. See
[docs/training.md](../docs/training.md#closed-loop-reward-hacking-auto-mitigation-v07126).

---

## Dataset formats

Soup auto-detects and normalizes:

- **Alpaca**: `instruction`, `input`, `output`
- **ShareGPT**: `conversations` with `from`/`value`
- **ChatML**: OpenAI-style `messages` with `role`/`content`
- **DPO/ORPO/SimPO/IPO**: `prompt` + `chosen` + `rejected`
- **KTO**: `prompt` + `completion` + `label`
- **LLaVA / ShareGPT4V**: `image` + `conversations`
- **Plaintext**: raw `.txt` or JSONL with a `text` field (pre-training)
- **Audio**: `audio` path + `messages`

### Inspect a dataset

```bash
soup data inspect examples/data/alpaca_tiny.jsonl
```

```
                   Dataset Stats
┌────────────────────┬────────────────────────────┐
│ Metric             │ Value                      │
├────────────────────┼────────────────────────────┤
│ Total samples      │ 10                         │
│ Columns            │ instruction, input, output │
│ Avg length (chars) │ 180                        │
│ Min length         │ 60                         │
│ Max length         │ 368                        │
│ Empty fields       │ 0                          │
│ Duplicates         │ 0                          │
└────────────────────┴────────────────────────────┘
```

### Convert between formats

The source format is auto-detected; you name the target:

```bash
soup data convert examples/data/alpaca_tiny.jsonl \
  --to chatml \
  --output alpaca_as_chatml.jsonl
```

## Directory structure

```
examples/
  configs/                    # YAML configs
    sft_basic.yaml            # runnable
    dpo_chat.yaml             # runnable
    dpo_example.yaml          # runnable
    grpo_reasoning.yaml       # runnable
    rlhf_step1_sft.yaml       # runnable
    rlhf_step2_reward.yaml    # runnable
    rlhf_step3_ppo.yaml       # runnable
    vision_llama.yaml         # template — bring your own images
  data/                       # format fixtures (see data/README.md)
    alpaca_tiny.jsonl
    chat_preferences.jsonl
    dpo_sample.jsonl
    reasoning_math.jsonl
  reward_hacking/             # synthetic reward fns for the mitigation demo
  synthetic_workflow.md       # synthetic-data walkthrough
  synthetic_workflow.yaml     # template config for that walkthrough
```

## Using your own data

1. Put your data in one of the supported formats above.
2. Point the config at it:

```yaml
data:
  train: /path/to/your/data.jsonl
  format: alpaca   # or sharegpt, chatml, dpo, llava, ... — omit for auto-detect
```

3. Train:

```bash
soup train --config your_config.yaml
```

## Config shape

Configs are nested: `base`, `task`, `data`, `training`, `output` at the top level.
[`config/schema.py`](../src/soup_cli/config/schema.py) is the single source of truth.

### Minimal

```yaml
base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
task: sft

data:
  train: ./your_data.jsonl
  format: alpaca
  max_length: 512

training:
  epochs: 3
  lr: 5e-4
  batch_size: 16
  lora:
    r: 16
    alpha: 32

output: ./output/
```

### More options

```yaml
base: meta-llama/Llama-2-7b-hf
task: dpo
backend: transformers        # or unsloth, mlx

data:
  train: ./dataset.jsonl
  format: dpo
  max_length: 2048
  val_split: 0.1

training:
  epochs: 2
  lr: 1e-4
  dpo_beta: 0.1
  batch_size: 16             # or "auto" to probe for the largest that fits
  gradient_accumulation_steps: 4
  quantization: 4bit         # 4bit, 8bit, none, gptq, awq, fp8, ...
  scheduler: cosine
  warmup_ratio: 0.1
  gradient_checkpointing: true
  lora:
    r: 64
    alpha: 128
    dropout: 0.05
    target_modules: auto
    # use_dora: true         # Weight-Decomposed LoRA
    # use_rslora: true       # rank-stabilized scaling

output: ./output_advanced/
```

## After training

### Batch inference

```bash
soup infer --model ./output_sft_basic/ --input prompts.jsonl --output results.jsonl
```

### Merge the LoRA adapter into a full model

```bash
soup merge --adapter ./output_sft_basic/ --output ./merged_model/
```

### Export to GGUF

```bash
soup export --model ./output_sft_basic/ --format gguf --quant q8_0 --output model.gguf
```

Requires a built llama.cpp — see [docs/serving-and-export.md](../docs/serving-and-export.md).
`soup export --model ... --deploy ollama` exports and registers with Ollama in one step.

### Monitor with Weights & Biases

```bash
pip install wandb
soup train --config your_config.yaml --wandb
```

## Common issues

### "CUDA out of memory"

- Lower `training.batch_size`, or set it to `"auto"` to probe for a size that fits
- Add `training.quantization: 4bit`
- Add `training.gradient_checkpointing: true`
- Use a smaller base model
- Or stream the base layer-by-layer: `training.stream_layers: true`
  (see [docs/training.md](../docs/training.md))

### "Dataset not found"

- Paths in a config resolve from the directory you run `soup` in, not from the config's
  location. Run from the repo root, or use an absolute path.
- Check the file parses: `soup data validate your_data.jsonl`

### "Model not found on Hugging Face"

- Check the model id spelling
- Gated models (Llama, Gemma) need `huggingface-cli login` and accepted terms

### Config validation errors

`soup train` validates before doing any work. `base: Field required` or
`data -> train: Field required` means the config uses an old flat layout — see
[Config shape](#config-shape) above for the current nesting.

## Learn more

- [Main README](../README.md)
- [Full docs](../docs/README.md)
- [Data engineering](../docs/data.md)
- [Training](../docs/training.md)
- [CONTRIBUTING](../CONTRIBUTING.md)

## Questions?

- [GitHub Discussions](https://github.com/MakazhanAlpamys/Soup/discussions)
- [Issues](https://github.com/MakazhanAlpamys/Soup/issues)
- [SECURITY.md](../SECURITY.md)
