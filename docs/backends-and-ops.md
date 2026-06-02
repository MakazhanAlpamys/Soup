# Backends, Platform & Ops

[← Back to the Soup README](../README.md)

> MLX/Unsloth backends, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, shell completions, the plugin system, and the standalone utility commands.

**Contents:**

- [Autopilot (Zero-Config)](#autopilot-zero-config)
- [Apple Silicon (MLX Backend)](#apple-silicon-mlx-backend)
- [Unsloth Backend (2-5x Faster Training)](#unsloth-backend-2-5x-faster-training)
- [Chat with your model](#chat-with-your-model)
- [Push to HuggingFace](#push-to-huggingface)
- [HuggingFace Hub Deep Integration](#huggingface-hub-deep-integration)
- [Resume Training](#resume-training)
- [Run Management & Cleanup](#run-management--cleanup)
- [Alternative Model Hubs](#alternative-model-hubs)
- [TensorBoard Integration](#tensorboard-integration)
- [Weights & Biases Integration](#weights--biases-integration)
- [Ready-Made Recipes](#ready-made-recipes)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [Model Comparison](#model-comparison)
- [Quickstart Demo](#quickstart-demo)
- [Health Check](#health-check)
- [Version Info](#version-info)
- [Error Handling](#error-handling)
- [Experiment Tracking](#experiment-tracking)
- [Profiling Extras](#profiling-extras)
- [VS Code Setup (`.vscode/launch.json`)](#vs-code-setup-vscodelaunchjson)
- [Observability & Dev UX](#observability--dev-ux)
- [GPU Live Monitor](#gpu-live-monitor)
- [Soup Fetch — Bundled Examples](#soup-fetch--bundled-examples)
- [Llama 4 Delinearizer](#llama-4-delinearizer)
- [Ctrl+C Graceful Save](#ctrlc-graceful-save)
- [Checkpoint-Now Trigger File](#checkpoint-now-trigger-file)
- [Onboarding Wizard Helper](#onboarding-wizard-helper)
- [Standalone Sweep Config](#standalone-sweep-config)
- [Alternative Model Hubs (ModelScope / Modelers)](#alternative-model-hubs-modelscope--modelers)
- [Experiment Trackers (MLflow / SwanLab / Trackio)](#experiment-trackers-mlflow--swanlab--trackio)
- [Telemetry (opt-IN, hardware-info-only)](#telemetry-opt-in-hardware-info-only)
- [Plugin System](#plugin-system)
- [External Integrations Catalog](#external-integrations-catalog)
- [Advanced Trainer Plugins](#advanced-trainer-plugins)
- [Soup Plugin Callbacks](#soup-plugin-callbacks)
- [Terraform-Style Plan & Apply (`soup plan` / `soup apply`)](#terraform-style-plan--apply-soup-plan--soup-apply)
- [Hermetic Env Lockfile (`soup env`)](#hermetic-env-lockfile-soup-env)
- [Hardware-Fit Calculator](#hardware-fit-calculator)
- [Shell Completions (`soup completions`)](#shell-completions-soup-completions)
- [License Advisor (`soup license-advisor`)](#license-advisor-soup-license-advisor)

---

## Autopilot (Zero-Config)

Skip the YAML entirely. Give Autopilot a base model, a dataset, and a goal — it analyzes your data, model, and hardware, then picks the task, quantization, LoRA rank, learning rate, epochs, and performance flags for you.

```bash
# Zero-config: pick everything automatically
soup autopilot --model meta-llama/Llama-3.1-8B-Instruct \
               --data ./data/train.jsonl \
               --goal chat

# Other goals: chat | code | reasoning | instruct | vision
soup autopilot --model Qwen/Qwen2.5-7B --data ./data/math.jsonl --goal reasoning

# Constrain to a GPU budget (1GB to 1TB)
soup autopilot --model <id> --data d.jsonl --goal chat --gpu-budget 24GB

# Preview the generated config without running
soup autopilot --model <id> --data d.jsonl --goal chat --dry-run
```

Autopilot writes a ready-to-run `soup.yaml`. Edit it by hand if needed, then `soup train`.


## Apple Silicon (MLX Backend)

Fine-tune on M1-M4 Macs via Apple's [MLX](https://github.com/ml-explore/mlx) framework — no CUDA, no emulation.

```bash
# Install MLX support
pip install 'soup-cli[mlx]'
```

```yaml
base: mlx-community/Llama-3.2-3B-Instruct-4bit
task: sft
backend: mlx  # Apple Silicon only

data:
  train: ./data/train.jsonl
  format: alpaca

training:
  epochs: 3
  lr: 2e-5
  lora:
    r: 16
    alpha: 32
```

MLX backend supports SFT, DPO, and GRPO. Use `soup recipes search --tag mlx` for ready-made Apple Silicon configs.


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

# Group into a Collection
soup push --model ./output --repo your-username/my-model \
    --collection your-username/my-collection-abc123
```


## HuggingFace Hub Deep Integration

Soup treats HF Hub as a first-class artifact backend. One env var, one flag,
no token flags to plumb — all operations respect `huggingface-cli login`
credentials by default.

```bash
# Self-hosted Hub: set once, every command routes there.
export HF_ENDPOINT=https://hf.internal.example.com

# Auto-push each save_steps checkpoint to HF as a 'checkpoint-<N>' branch.
soup train -c soup.yaml --push-as your-username/my-model

# Resume from the latest branch pushed above.
soup train -c soup.yaml --push-as your-username/my-model --hf-resume

# Upload a local JSONL file as an HF dataset repo.
soup data push --input train.jsonl --hf-dataset your-username/my-dataset

# Wrap your fine-tuned model in a Gradio chat Space in one command.
soup deploy hf-space \
    --model your-username/my-model \
    --space your-username/my-chat-space \
    --template gradio-chat

# Or a Streamlit app:
soup deploy hf-space \
    --model your-username/my-model \
    --space your-username/my-chat-space \
    --template streamlit-chat
```

**Auto-resume workflow:** if training crashes, the next `soup train ... --push-as
... --hf-resume` call picks up the latest `checkpoint-<N>` branch from your HF
repo and downloads it back to `output_dir`, then resumes — no manual copy /
paste of checkpoint paths. Cwd containment and `local_dir_use_symlinks=False`
prevent filesystem escape from a crafted repo.

**Auth** follows standard HF conventions: `HF_TOKEN` env var > `HUGGINGFACE_HUB_TOKEN`
> `~/.cache/huggingface/token` (set by `huggingface-cli login`) > `~/.huggingface/token`.
No custom token flags. The deprecated `--token` on `soup push` still works but emits
a warning.

**Model card v2** is auto-generated on first push: it reads sidecar
`training_config.yaml` / `soup.yaml` to surface `task` / `base` / `lr` /
`optimizer`, and accepts an optional eval scorecard (markdown table).
Markdown-active chars in task names and scores are neutralised for safe
rendering on HF Hub.


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


## Alternative Model Hubs

Set `training.hub` in your `soup.yaml` to download from / push to a non-HuggingFace hub. Useful in regions where HF Hub is unreachable or blocked.

```yaml
training:
  hub: modelscope   # or 'modelers' (Openmind), default 'hf'
```

Override the endpoint via env var:

```bash
export MODELSCOPE_ENDPOINT=https://my-mirror.example.com
export MODELERS_ENDPOINT=https://corp-modelers.internal   # HTTPS only for non-loopback
soup train --config soup.yaml
```

The endpoint validator follows the same SSRF rules as `HF_ENDPOINT`: only `http`/`https` schemes; plain HTTP allowed only for `localhost` / `127.0.0.1` / `::1`; private and link-local IPs (RFC1918, 169.254/16, etc.) rejected on plain HTTP. `backend: mlx` is incompatible with non-HF hubs (`mlx-lm` only downloads from HF Hub).

The hub adapter is schema-only in this release; the live downloader and uploader land in v0.51.1.


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


## Ready-Made Recipes

80 pre-built configs for popular models — no guessing hyperparameters:

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
soup recipes search "medical"
soup recipes search "vision"
```

**What's covered:**

| Category | Models |
|---|---|
| **General SFT / DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO / Embedding / Pretrain** | Llama 3.1 / 3.2 / 4, Qwen 2.5 / 3, Mistral, Gemma 3, Phi-4, DeepSeek R1 / V3 |
| **Vision (multimodal)** | Llama-3.2-Vision (11B + 90B), Pixtral-12B, Qwen2-VL (7B + 72B), InternVL 2.5, MiniCPM-V 2.6 |
| **Audio (speech)** | Qwen2-Audio, SeamlessM4T v2 (translation), Whisper-large-v3 (ASR) |
| **Reasoning** | All 6 DeepSeek-R1-Distill sizes (Qwen 1.5B / 7B / 14B / 32B + Llama 8B / 70B), Qwen3-Coder 30B, Qwen3-30B-A3B reasoning, Phi-4 reasoning |
| **Small / edge / mobile** | SmolLM2 (135M / 360M / 1.7B), Qwen2.5 (0.5B / 1.5B / 3B), Gemma 2 2B, Phi-3.5-mini, Llama-3.2 (1B / 3B) |
| **Domain specialists** | BioMistral 7B, Meditron 7B (medical) — CodeLlama (13B / 70B), Magicoder 6.7B (code) — Mathstral 7B (math) — Llama-2-13b-finance (FinGPT-style starter) — Nemotron-4 340B |
| **Multimodal reasoning** | Llama-3.2-Vision GRPO, Pixtral DPO |
| **Multi-GPU** | llama3-70b-fsdp2, qwen3-32b-zeropp, deepseek-v3-pipeline |
| **Apple Silicon (MLX)** | llama3.1-8b / qwen3-8b / gemma3-9b SFT-MLX |
| **Tool-calling / agentic** | qwen3-8b-tools, llama4-scout-tools |


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
soup doctor [--nccl]
```

Shows: Python version, GPU availability, system resources (RAM/Disk), all dependency versions, and fix suggestions. Use `--nccl` to measure and check multi-GPU communication bandwidth against expected hardware ceilings.


## Version Info

```bash
# Basic version
soup version

# Machine-readable output
soup version --json
# -> {"version": "0.26.0", "python": "3.11.5", "platform": "linux"}

# Full system info (useful for bug reports)
soup version --full
# -> soup v0.26.0 | Python 3.11.5 | CUDA 12.1 | extras: serve, data

# Full system info in JSON
soup version --full --json
# -> {"version": "0.26.0", "python": "3.11.5", "platform": "linux", "torch": "2.2.0", ...}
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

# Replay an old run's summary + loss curve from history
soup runs replay run_1
```

Every completed run also stores an estimated cost (`$` per run) computed from the
captured GPU device name and duration. `soup runs show` renders `—` for CPU /
MPS / unknown GPUs (no fabricated zeros).

As of v0.71.5, the metric-series lookup that powers replay (`ExperimentTracker.get_metric_series`)
transparently falls back to the `eval_results` table when a metric has no per-step
rows — so you can plot a benchmark-score curve (e.g. `mmlu`, `gsm8k`) the same way
you plot `loss`, without caring which table holds the series.

### Tracker integrations (--tracker mlflow / swanlab / trackio)

```bash
# Stream metrics to MLflow (set MLFLOW_TRACKING_URI to your server URL)
soup train --config soup.yaml --tracker mlflow

# Or SwanLab (cloud or local)
soup train --config soup.yaml --tracker swanlab

# Or Trackio (offline-friendly batched upload)
soup train --config soup.yaml --tracker trackio
```

`--tracker` is mutually exclusive with `--wandb` and `--tensorboard`. Soup
validates the tracker name against a closed allowlist (`mlflow` / `swanlab` /
`trackio` / `wandb` / `tensorboard` / `none`); the upstream package itself is
loaded by HF Trainer at run time, so install the one you need separately:

```bash
pip install mlflow      # or: swanlab / trackio
```

### Telemetry (opt-in)

Soup ships a hardware-info-only telemetry payload (Soup version + command +
Python major.minor + OS + arch + duration). It is **off by default** and never
sends model names, dataset paths, or config contents. Enable explicitly:

```bash
SOUP_TELEMETRY=1 soup train --config soup.yaml
```

The PostHog network upload itself is deferred to v0.43.1; v0.43.0 ships the
payload schema only so you can audit it before opting in.


## Profiling Extras

CUDA memory snapshots, anomaly tracing, and an NCCL bandwidth reference table:

```python
from soup_cli.utils.profiling_v0_43 import (
    memory_snapshot_context, detect_anomaly_context, nccl_bandwidth_check,
)

with memory_snapshot_context("run-123") as path:
    train_step()
    # On CUDA, dumps profiles/run-123.snapshot.pickle on exit.

with detect_anomaly_context():
    train_step()
    # torch.autograd.set_detect_anomaly(True)

result = nccl_bandwidth_check(
    gpu="h100", link="nvlink", measured_gb_per_sec=400.0,
)
# {'expected_gb_per_sec': 450.0, 'measured_gb_per_sec': 400.0,
#  'ratio': 0.8889, 'status': 'OK'}
```


## VS Code Setup (`.vscode/launch.json`)

One-shot writer for a sane debugger config:

```python
from soup_cli.utils.vscode_setup import write_vscode_launch
write_vscode_launch(config_path="soup.yaml")
# Writes ./.vscode/launch.json with `soup train` + pytest entries.
```

Symlink-rejected at the target path regardless of `force=True` to defend
against pre-placed symlinks redirecting the write outside cwd.


## Observability & Dev UX

Tools that explain *why* a run misbehaved instead of dumping a stack trace.

### `soup why`

Heuristic explainer — reads the most recent (or named) run and surfaces
plain-English diagnoses with concrete next steps.

```bash
soup why                 # most recent run
soup why run_2026_abc    # specific run id (or prefix)
```

Detects: NaN/Inf loss, plateau (≥30 steps with <0.5% change), divergence
(loss > 3× initial), persistent high gradient norm, learning rate outside the
typical `[1e-6, 5e-3]` band. Pure rule-based — no model calls.

### `soup tui`

Full-screen Textual dashboard. Two-pane: run list (left) + selected-run detail
(right). `r` refreshes, `q` quits.

```bash
pip install 'soup-cli[tui]'
soup tui --refresh 1.0 --limit 50
```

### Auto-profiling — `soup train --profile`

Records a `torch.profiler` Chrome-trace over an early-steps window (default
`wait=1, warmup=1, active=5, repeat=1`). Output: `<output>/profiles/<run_id>.trace.json`.
Open in `chrome://tracing` or Perfetto.

### Crash bundles — `.crash` files

When training fails, Soup auto-writes a self-contained `.crash` JSON to
`./.soup-crashes/crash_<utc>_<hex>.crash` containing: redacted error trace,
classified failure kind (`oom` / `nan` / `cuda` / `dataloader` / `nccl` /
`other`), GPU state at crash time, env summary, last-50 metric rows, and the
config (recursively redacted of `hf_*` / `sk-*` / `Bearer …` tokens). The
output_dir is reduced to `os.path.basename` so `$HOME` doesn't leak.

### `--log-level quiet|normal|verbose|debug`

Global flag on the root `soup` command. Wires a Rich-formatted logger on the
`soup` namespace; `debug` enables timestamps + module paths.

```bash
soup --log-level verbose train --config soup.yaml
soup --log-level debug runs show <id>
```


## GPU Live Monitor

```bash
soup monitor                # 2s refresh, Util / Mem / VRAM / Temp / Power per GPU
soup monitor --refresh 0.5  # faster polling
soup monitor --once         # single snapshot, no Live panel
```

Calls `nvidia-smi` via list-args subprocess (no shell), 5s timeout, list of `GpuSample` rows rendered into a Rich table. Apple Silicon prints a yellow advisory pointing at Activity Monitor / `powermetrics`; native Apple Silicon support lands in v0.44.1.


## Soup Fetch — Bundled Examples

```bash
soup fetch examples                          # list bundled entries
soup fetch examples llama-3.1-8b-lora        # write to ./llama-3.1-8b-lora.yaml
soup fetch examples qwen2.5-7b-dpo -o ./my-config.yaml --force
soup fetch deepspeed_configs zero3-cpu-offload
```

Closed catalog (`MappingProxyType`) of ready-to-edit YAML / JSON. Output path cwd-contained, bundled-source `os.path.commonpath` check (defends against catalog escape), `os.lstat + S_ISLNK` symlink-reject at the write target.


## Llama 4 Delinearizer

```bash
soup delinearize-llama4 ./llama4-checkpoint --target ./out-delinearized --yes
```

Plans Llama 4 expert-weight reshape for export. v0.44.0 ships the planner; live runtime in v0.44.1. `is_llama4_model` uses a word-boundary regex matching the `is_gemma4_model` pattern — `ungemma-llama-4ish` is rejected.


## Ctrl+C Graceful Save

First SIGINT → trainer writes a checkpoint and continues. Second SIGINT → trainer stops cleanly after the next save. No-state fallback raises `KeyboardInterrupt` so the user never gets stuck. `GracefulSaveHandler.install()` is idempotent and swallows `signal.signal` failures on non-main threads.


## Checkpoint-Now Trigger File

```bash
touch ./out/.checkpoint_now    # trainer saves on the next step, then deletes the trigger
```

Path containment via `is_under_cwd`; `os.lstat + S_ISLNK` rejection at the trigger target so a pre-placed symlink can't redirect the write.


## Onboarding Wizard Helper

```python
from soup_cli.utils.onboarding import render_onboarding_yaml

text = render_onboarding_yaml({
    "base": "meta-llama/Llama-3.2-1B",
    "dataset": "./train.jsonl",
    "task": "sft",
    "quantization": "4bit",
    "epochs": 3,
})
```

Five-question wizard input → fully-validated `soup.yaml`. Literal allowlists on `task` (`sft` / `dpo` / `kto` / `orpo` / `simpo` / `ipo` / `bco` / `preference`) and `quantization` (`4bit` / `8bit` / `none`); `epochs ∈ [1, 10]`; `output` cwd-contained; null-byte rejection on every string.


## Standalone Sweep Config

```bash
soup sweep --config sweep.yaml
```

```yaml
# sweep.yaml
strategy: random
n_runs: 20
seed: 42
params:
  lr: [0.0001, 0.0005, 0.001]
  epochs: [1, 3, 5]
```

Strict scalar allowlist on values (`str` / `int` / `float` / `bool`); `_MAX_FILE_BYTES=256KB`, `_MAX_PARAM_KEYS=32`, `_MAX_VALUES_PER_KEY=64`; `SweepSpec.params` is `MappingProxyType[str, Tuple[Any, ...]]` for genuine immutability.


## Alternative Model Hubs (ModelScope / Modelers)

Set `training.hub` to fetch the base model from a non-HF Hub:

```yaml
base: baichuan-inc/Baichuan2-7B
task: sft
training:
  hub: modelscope     # or "modelers"
```

`soup train` pre-fetches the model into `./.soup_hub_cache/<sanitized-slug>/` via the matching SDK (`modelscope.snapshot_download` / `openmind_hub.snapshot_download`) and rewrites `cfg.base` to the local snapshot. Re-runs reuse the cached snapshot. Both `huggingface-hub`, `modelscope`, and `openmind-hub` are lazy-imported — install only what you need.

Programmatic API:

```python
from soup_cli.utils.hubs import download_repo, upload_repo

local_path = download_repo("modelscope", "baichuan-inc/Baichuan2-7B", local_dir="./snap")
upload_repo("modelers", "my-org/my-model", folder_path="./output", commit_message="Soup v0.53.8")
```

The dispatcher enforces shape validation on every input (bool / null-byte / leading-slash / `..` segments / control characters / oversize all rejected) and runs cwd-containment on `local_dir` / `folder_path`.


## Experiment Trackers (MLflow / SwanLab / Trackio)

Pick a tracker on the CLI; Soup threads it into HF Trainer's `report_to`:

```bash
soup train --tracker mlflow
soup train --tracker swanlab
soup train --tracker trackio
```

If the package is not installed, Soup now surfaces a friendly advisory before training starts instead of a mid-run ImportError:

```
--tracker mlflow requires the 'mlflow' package. Install with: pip install soup-cli[trackers] (or pip install mlflow)
```

```bash
pip install soup-cli[trackers]   # mlflow + swanlab + trackio
```


## Telemetry (opt-IN, hardware-info-only)

Soup ships an opt-IN telemetry sender that POSTs hardware-info-only payloads (`soup_version` / `command` / `python` major.minor / `os` / `arch` / optional `duration_seconds`) — no dataset paths, model names, or config contents. Enable per-shell:

```bash
SOUP_TELEMETRY=1 soup train --config soup.yaml
```

The sender uses a 1-second hard timeout, HTTPS-only with private-IP / link-local rejection (same SSRF policy as hub endpoints), and swallows every exception silently — telemetry can never crash training. Disabled by default until a public privacy policy ships.


## Plugin System

Drop a Python module under `src/soup_cli/plugins/` (or any package importable by Soup) and register at import time:

```python
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        ...
    def post_train(self, ctx):
        ...

register_plugin(
    name="my-plugin",
    version="1.0.0",
    plugin=MyPlugin(),
    description="Hooks into pre/post-train",
    templates=["my-template"],         # optional
    model_groups=["my-arch-family"],   # optional
)
```

```bash
soup plugins              # list registered plugins
soup plugins enable foo
soup plugins disable foo
```

Plugin names are kebab-case (`^[a-z0-9][a-z0-9-]{0,39}$`); versions are semver-ish (`MAJOR.MINOR.PATCH`); registry caps `_MAX_PLUGINS=64`, `_MAX_TEMPLATES_PER_PLUGIN=32`, `_MAX_MODEL_GROUPS_PER_PLUGIN=32`. Re-registering the same `(name, version, plugin, templates, model_groups, description)` is idempotent; any field mismatch is rejected with a clear error. Trainer-callback wiring of `pre_train` / `post_train` / `pre_step` / `post_step` lands in v0.45.1.


## External Integrations Catalog

```python
from soup_cli.utils.integrations import list_integrations, get_integration

list_integrations()                       # 15 entries
get_integration("lm-studio").target_artifacts   # ("gguf",)
```

15 ecosystem targets covered: `lm-studio`, `comfyui`, `stable-diffusion-cpp`, `open-webui`, `ollama`, `tei`, `pgvector`, `faiss`, `weaviate`, `sentence-transformers`, `claude-code`, `cursor`, `continue`, `cline`, `sillytavern`. Auto-detect + launch wiring lands with v0.46.0 Deploy Autopilot.


## Advanced Trainer Plugins

```python
from soup_cli.utils.trainer_plugins import validate_trainer_plugin_list

validate_trainer_plugin_list(["grokfast", "spectrum"])
# returns ("grokfast", "spectrum") — canonical lowercase, dedup, ≤ 8 entries
```

6-entry allowlist (`cce_plugin`, `grokfast`, `spectrum`, `llmcompressor`, `sonicmoe`, `math_verify`) so a future `training.trainer_plugins: [...]` schema field has a stable surface. Live callbacks in v0.45.1.


## Soup Plugin Callbacks

Register a plugin once via the v0.45.0 registry API; v0.53.6 wires it into every
transformer-backend trainer as a real HF `TrainerCallback`:

```python
# src/soup_cli/plugins/my_plugin.py — auto-discovered at `soup` startup
from soup_cli.plugins import register_plugin

class MyPlugin:
    def pre_train(self, ctx):
        print("training about to start, args =", ctx["args"])

    def post_step(self, ctx):
        if ctx["state"].global_step % 100 == 0:
            print(f"step {ctx['state'].global_step}")

register_plugin(name="my-plugin", version="0.1.0", plugin=MyPlugin())
```

A misbehaving plugin hook is swallowed at WARNING — one bad plugin must never crash
a multi-hour training run. The hook snapshot is taken at callback-construction time,
so a plugin registered MID-run does not retroactively receive events.


## Terraform-Style Plan & Apply (`soup plan` / `soup apply`)

A training run is a one-shot infrastructure-shaped operation: spot price, expected cost, base SHA, dataset SHA, peak VRAM. v0.64 borrows Terraform's plan-apply split so you can review the numbers before committing money.

```bash
# Render a pre-flight summary + write soup.tfstate
soup plan --config soup.yaml

# Apply — refuses on drift (exit 3) if the YAML changed since `plan`
soup apply --config soup.yaml

# Validate without actually running anything
soup apply --config soup.yaml --dry-run
```

The state file is a thin JSON envelope; the actual training is still driven by `soup train`. The gate prevents the "wait, why did I spend another $0.50 on the wrong config" surprise.


## Hermetic Env Lockfile (`soup env`)

The "CUDA hell" problem: a fine-tune that worked on Friday breaks on Monday because PyPI silently upgraded `transformers` past the trainer's compat band. v0.34 `soup doctor` surfaces some of this; v0.64 makes it lockable.

```bash
# Snapshot the current env into soup-env.lock
soup env lock

# Print the locked env summary
soup env status

# Compare current env to the lock — exit 3 on ABI-sensitive drift
soup env check
```

`soup-env.lock` captures Python + platform + CUDA + 15 ABI-sensitive packages (torch / transformers / peft / trl / accelerate / bitsandbytes / flash-attn / xformers / deepspeed / unsloth / vllm / sentencepiece / tokenizers / datasets / huggingface-hub). Wire `soup env check` into your CI to refuse silent ABI breakage.


## Hardware-Fit Calculator

Given (params, seq_len, batch_size, optimizer, quant, peft, gradient_checkpointing), the analytical predictor returns a 5-bucket peak-VRAM breakdown (weights / optimizer / gradients / activations / overhead) and an OK/OOM verdict with a 10% safety margin.

```python
from soup_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit

inp = HardwareFitInput(
    params_b=7.0, seq_len=2048, batch_size=4,
    optimizer="adamw_torch", quant="4bit", peft="lora",
    gradient_checkpointing=True,
)
report = decide_hardware_fit(inp, available_vram_gb=24.0)
print(report.ok, report.reason)
# True | 'fits: peak 7.76 GB + 10% margin <= 24.00 GB available'
```

When it doesn't fit, the report names actionable knobs: `--batch-size halve`, `--quantization 4bit`, `--gradient-checkpointing auto`. Composes with v0.40.3 live CUDA OOM probe (`make_cuda_probe_fn`) which still runs when `auto_batch_size_strategy: probe`.


## Shell Completions (`soup completions`)

Tab-completion for `soup` + every subcommand. The generated script is Click/Typer-backed so new commands are picked up automatically.

```bash
# Bash
eval "$(soup completions bash)"        # current shell
soup completions bash >> ~/.bashrc     # permanent

# Zsh
soup completions zsh > "${fpath[1]}/_soup"

# Fish
soup completions fish > ~/.config/fish/completions/soup.fish
```

Recipe names auto-complete from the 115+ catalogue; `--target-modules` falls back to canonical Llama-shape defaults (`q_proj` / `k_proj` / `v_proj` / etc.). Live HF-config introspection per `base` lands in v0.64.1.


## License Advisor (`soup license-advisor`)

Picking a license-clean base for a specific deployment target is a recurring legal-review pain point. v0.64 captures the three most common deploy contexts as a closed allowlist and surfaces the per-license downstream risk.

```bash
# What licenses are safe for a B2C consumer product?
soup license-advisor --target b2c

# Defense — restricted-use community licenses forbidden
soup license-advisor --target defense

# Per-license check: Llama community license + 800M MAU = block (exit 3)
soup license-advisor --target b2c --license llama-3 --monthly-active-users 800000000
```

The Llama-family allowlist is tight (no `.startswith` over-match), so a hypothetical future `llama-permissive-2030` won't false-trigger the 700M-MAU gate. Composes with v0.60 `soup adapters merge --license <id>` for the merge-time conflict gate.


