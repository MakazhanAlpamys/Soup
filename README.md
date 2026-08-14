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
  <a href="docs/models.md">Models</a> &middot;
  <a href="https://discord.gg/8RgVbFA6Zq">Discord</a> &middot;
  <a href="https://www.producthunt.com/products/soup-cli">Product Hunt</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/soup-cli/"><img src="https://img.shields.io/pypi/v/soup-cli?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/project/soup-cli"><img src="https://img.shields.io/pepy/dt/soup-cli?color=blue" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License">
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MakazhanAlpamys/65fdc943f85f3b2c46ecddb415c2b779/raw/soup_tests.json" alt="Tests"></a>
  <a href="https://github.com/MakazhanAlpamys/Soup/actions"><img src="https://github.com/MakazhanAlpamys/Soup/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://trysoup.dev"><img src="https://img.shields.io/badge/website-trysoup.dev-blue" alt="Website"></a>
  <a href="https://discord.gg/8RgVbFA6Zq"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://doi.org/10.5281/zenodo.21771064"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21771064-blue?logo=zenodo&logoColor=white" alt="DOI: 10.5281/zenodo.21771064"></a>
</p>

<p align="center">
  <a href="https://www.producthunt.com/products/soup-cli?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-soup-cli">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=dark">
      <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1217869&amp;theme=light" alt="Soup CLI - Fine-tune an 8B LLM on a 4 GB laptop GPU | Product Hunt" width="250" height="54">
    </picture>
  </a>
</p>

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.

```bash
pip install "soup-cli[train]"   # add [train] to fine-tune; bare `soup-cli` is the light CLI
soup init --template chat
soup train
```

**Fine-tune an 8B model on a 4 GB laptop GPU.** Layer streaming keeps the frozen base out of
VRAM and feeds it to the GPU one decoder layer at a time. Measured on an RTX 3050 Laptop 4 GB:
Llama-3.1-8B-Instruct + NF4 at **119.6 tok/s, 3.32 GB peak** — bit-exact against a normal
resident run, and reproduced independently on an H100 at 113.00 tok/s in the same 3.32 GB.
(The tok/s figure was measured on v0.72.2, before the v0.73.0 correctness repair that cost
−4.8% at 32B; it has not been re-run on a 4 GB card since.) Opt-in (`stream_layers: true`)
and still BETA —
[how it works](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[all measurements](benchmarks/) · [paper](https://doi.org/10.5281/zenodo.21771064) ·
**[check it yourself on a free Colab T4](notebooks/proof-4gb.ipynb)** (caps the process to
4 GB, then asserts a streamed model is bit-identical to a normal one)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="soup train pre-flight for Llama-3.1-8B on a 4 GB card: a 3.60 GB base store pinned in RAM across 32 layers and two 113 MB VRAM buffers, then a measured peak of 3.32 GB at 119.6 tok/s, stopping short of the 4 GB line"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, batch 1, seq 512 on an RTX 3050 Laptop 4 GB — <b>3.32 GB peak, 119.6 tok/s</b>. <a href="https://youtu.be/T1LCErE943E">Full video (90s)</a></sub>
</p>

## Why Soup?

Training LLMs is still painful. Even experienced teams spend 30-50% of their time fighting
infrastructure instead of improving models. Soup fixes that.

- **Zero SSH.** Never SSH into a broken GPU box again.
- **One config.** A simple YAML file is all you need.
- **Auto everything.** Batch size, GPU detection, quantization — handled.
- **Works locally.** Train on your own GPU with QLoRA. No cloud required.

## What's New

**v0.73.1 — the free GPU tier, and a formula caught under-predicting.** A patch release
carrying everything that landed since v0.73.0, headed by a fix that made Soup unusable on
the cards most people actually have.

- **bf16 was assumed on every CUDA card, in fourteen places.** Anything pre-Ampere — T4,
  P100, V100, GTX 16xx, i.e. the entire free tier on Colab and Kaggle — failed on **every**
  task, not just streaming. The trap worth knowing: `torch.cuda.is_bf16_supported()`
  defaults to `including_emulation=True`, so a T4 answers **True** and the first attempt at
  this fix was a no-op on the exact hardware it was written for.
- **New: `training.stream_vram_probe`** decides the layer-streaming VRAM check by
  *measuring* one real step instead of predicting it. Measured on a 4 GB RTX 3050, the
  prediction formula **under-predicts at long sequence** — 0.934x the real peak at seq 5120
  and **0.787x at seq 6144** — which is the direction that does not announce itself (an OOM
  on Linux, a silent spill to host memory on Windows). The grid it was fitted on only ever
  varied *batch size*, at seq 256 and 512, so it had no evidence there at all.
- **Under `use_fsdp2_compile`, every `checkpoint-*` still loaded as a dead adapter.** The
  final save was repaired in v0.73.0; the periodic checkpoints were not, so `--resume` and
  `load_best_model_at_end` silently continued from a re-zeroed `lora_B`. Measured at 70B:
  320 canonical keys in the output root, 320 prefixed ones in `checkpoint-100`.
- **`backend: mlx` never dispatched to the MLX trainer**, and `training.seed` reached the
  SFT wrapper and nothing else — so a seeded `task: grpo` run trained at 42 and replicates
  that differed only in their seed were the same run.
- **Three published claims retracted** where the evidence did not support them, including
  "bound by host-to-device transfer". No measured number changed.

The measurement record for the VRAM work, published as written — including the **three
readings withdrawn during it**, two of which briefly looked like the headline result — is
[`benchmarks/gate-v0.73.1-measured-vram-fit.md`](benchmarks/gate-v0.73.1-measured-vram-fit.md).

```yaml
# soup.yaml — then just `soup train --config soup.yaml`
training:
  stream_layers: true      # base streams out of VRAM; only the adapter trains
  quantization: 4bit       # NF4 — ~4x smaller store, so 8B fits a 4 GB card
  batch_size: 4            # bigger batches amortise the weight read
  stream_source: auto      # RAM when it fits, NVMe disk when it does not
  seed: 1234               # new in v0.73.0
```

> Python **3.10–3.12** only. v0.73.0 adds the upper bound that was missing: on 3.13+, pip
> used to resolve untested PyTorch wheels that crash in the native extension before Soup
> runs at all.

<details>
<summary>Previous release — v0.72.4, align on a laptop (DPO / ORPO / SimPO / KTO over layer streaming)</summary>

Layer streaming used to support supervised fine-tuning only; v0.72.4 opened it to the
preference losses. The risk was one thing: DPO needs a reference model, and a second copy
would double memory and defeat the point. Soup uses *the same streamed base with its
adapters switched off* — measured at **0.914×** the SFT peak, where forcing a real second
instance cost **+730 MB, exactly one copy of the weights**. Bit-exact against a normal
non-streamed run for all four. Honest cost: free in *memory*, not in *time* — DPO reads the
layer stack **1.52×** as often per step. `grpo` / `ppo` stay excluded on purpose.

> **Trained with `stream_layers: true` on v0.72.0?** That adapter is inert — its tensors were
> saved under keys with an extra `.inner.` segment, so every loader returned the untuned base.
> Fixed in v0.72.1; re-run or re-save. Check with:
> `python -c "from safetensors.torch import load_file; print([k for k in load_file('adapter_model.safetensors') if '.inner.' in k][:3])"`

</details>

<details>
<summary>Previous release — v0.71.40, soup reward synth (generate a reward verifier from your data)</summary>

Point `soup reward synth` at a JSONL of reference outputs and it infers a deterministic verifier,
writes a readable / committable `.py` reward function, and — the part nobody else does — *refuses* to
emit one that can't tell your references from bad answers (four families: `numeric` / `json_schema` /
`regex` / `tool_call`; a mandatory calibration report is the moat). Reward ensembles
(`reward_fn: "accuracy,format"`) also train now. (#311)

```bash
soup reward synth references.jsonl -o reward.py --output-report calib.json
```

</details>

<details>
<summary>Previous release — v0.71.39, CI for weights not prompts (emit + provenance-bind the ship verdict)</summary>

`soup ship`'s verdict became emittable, committable, and provenance-bound: `--emit-evidence` makes a
run replay into an identical verdict, `eval.ship` in `soup.yaml` + `--config` makes the gate policy
reviewable, and `--config` binds evidence to the exact recipe that produced it (stale evidence → exit 3).
`soup ship --push owner/repo#N` posts the SHIP / DON'T-SHIP card on the PR.

</details>

<details>
<summary>Previous release — v0.71.38, The gate grows teeth (real leg-2 regression gate)</summary>

`soup ship`'s regression leg became real: a fixed, extraction-based scorer over seven bundled,
offline suites (MCQ · arithmetic · tool-calling · JSON validity · safety/refusal). A tune that
wins your task but quietly breaks tool-calling now gets a **DON'T SHIP**. Zero new deps.

```bash
soup ship --base ./base --adapter ./my-lora --task-eval my_task.jsonl
#   exit 0 = SHIP · 2 = DON'T SHIP · 3 = bad flags · 1 = runtime error
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

> **Double quotes, not single.** `"soup-cli[train]"` is the only spelling that works in every
> shell — `cmd.exe`, PowerShell, bash and zsh. If you copied `'soup-cli[train]'` from an older
> tutorial and pip rejected it, that is the reason:
> [why, and the exact error](docs/models.md#quoting-the-extra).

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
| [Performance & quantization](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Data engineering](docs/data.md) | Formats, the Axolotl/LF-parity pipeline, data tools, synthetic generation & forge, quality scorecards, trace tooling, remote datasets, mixing, recipe DAGs |
| [Evaluation & probes](docs/evaluation.md) | Eval design/gate, eval-gated training, benchmarks, NLG metrics, calibration, Elo arena, diagnose, post-train X-ray probes, A/B, drift, tunability, `soup advise` |
| [Serving & export](docs/serving-and-export.md) | OpenAI-compatible server, batch inference, benchmarking, merge/export, Anthropic Messages endpoint, speculative decoding (train + measure your own draft), deploy autopilot, Web UI, Agent Forge |
| [Adapters, registry & governance](docs/adapters-and-governance.md) | Adapter lifecycle/management, model registry, Soup Cans, the data flywheel (`soup loop`), knowledge editing, steering, supply-chain controls (scan/sign/BOM/attest/audit/airgap) |
| [Compliance & governance quickstart](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` templates, provenance (BOM/attest/repro-receipt), audit log, air-gap, model-card autogen (`soup card`), CI gate (`soup ci init`) |
| [Backends, platform & ops](docs/backends-and-ops.md) | MLX/Unsloth backends, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, completions, plugins, utility commands |
| [Command reference](docs/commands.md) | The full `soup` command list |
| [Supported models & extras](docs/models.md) | Recommended model families, the VRAM size guide, the pip extras matrix |

## Data Formats

Alpaca, ShareGPT, ChatML, preference pairs (DPO / ORPO / SimPO / IPO / KTO), vision, audio,
ASR, plaintext, embedding, RAFT and more — all auto-detected from JSONL, JSON, CSV, Parquet or
TXT, so in most cases you point `data.train` at a file and nothing else changes. Schemas with a
worked example per format, plus the data pipeline (remote URIs, streaming, sharding,
interleaving, vocab expansion, document ingestion), are in
[`docs/data.md`](docs/data.md#data-formats).

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

- Python 3.10, 3.11 or 3.12 (those are the versions CI tests; 3.13+ is not supported yet
  because the PyTorch stack has not been validated there)
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

## Support Soup

Soup is Apache-2.0 and free — and stays that way. It is built and maintained in the open on a
single 4 GB laptop, which is why every performance number in these docs is measured rather than
claimed.

If Soup saved you a training run, [starring the repo](https://github.com/MakazhanAlpamys/Soup)
helps most, and it costs nothing. If you would like to fund the work directly:

**[❤️ Donate](https://buy.stripe.com/4gMcN441k3pha3T19ye7m04)** — one-off, any amount (use
*Change amount* on the checkout page). Payments are processed by Stripe under the maintainer's
registered business, **MePlay, Inc.** — that name, not "Soup", is what appears on the checkout
page and on your card statement.

Donations buy GPU time for the hardware-gated work — multi-GPU, 8B+ validation, Apple Silicon —
that a single 4 GB laptop cannot reach.

The other way to move exactly those items is **hardware itself**. They ship behind honest
"requires \<hardware\>" gates rather than unverified claims, so if you have access to a bigger
box — or GPU credits going unused — running one of the
[`help wanted`](https://github.com/MakazhanAlpamys/Soup/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
issues and posting the numbers helps as much as funding the GPU time would. Those issues say
exactly what is blocked on hardware today.

## Contributors

Built by the community ❤️ — thank you to everyone who has contributed. See
[CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Contributors](https://contrib.rocks/image?repo=MakazhanAlpamys/Soup)](https://github.com/MakazhanAlpamys/Soup/graphs/contributors)

## Contact

Bugs and feature requests belong in the
[issue tracker](https://github.com/MakazhanAlpamys/Soup/issues), questions in
[Discussions](https://github.com/MakazhanAlpamys/Soup/discussions) — both get answered faster
and help the next person with the same problem.

For live chat, setup help, and everything that reads better as a conversation, join the
[Discord](https://discord.gg/8RgVbFA6Zq). Anything that should still be findable in six months
belongs in Issues or Discussions — a Discord answer helps one person, an issue helps everyone
who hits the same thing. The [Code of Conduct](CODE_OF_CONDUCT.md) applies there too.

For anything that does not fit in public — security reports (see [SECURITY.md](SECURITY.md)),
Code of Conduct matters, or press — email **team@trysoup.dev**. That is the project address
and the right one for anything Soup-related. **makazanalpamys@gmail.com** is the maintainer's
personal address; it reaches the same person and is a fine fallback.

## Citing Soup

Layer streaming — training an 8B model on a 4 GB laptop GPU by streaming the frozen base from
host RAM one decoder layer at a time — is described in a preprint, together with the correctness
protocol that verifies a streamed run against a resident one (forward and backward stated
separately, because they are two claims and not one).

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop
> GPU* (v3). Zenodo. https://doi.org/10.5281/zenodo.21918325

**Version 3 (13 August 2026) is current.** The title and the claim are unchanged — 8B on 4 GB —
and no measured number has changed since v1. What v3 does is **withdraw an explanation we had
published**, which is also the shortest way to describe what the paper is for:

- **Retracted in v3: "layer streaming is bound by host-to-device transfer, not by the GPU."**
  That was an *inference* from the H100 replication below, and it had never been measured. We
  measured it on 11 August and it is false at the published configuration: deleting every
  host-to-device byte buys **1.4%**, the compute stream waits on a copy for **0.20%** of the
  step, and the step runs at **71.3%** of that card's same-session GEMM ceiling. The largest
  streaming-specific cost is the per-layer NF4 dequantisation, at 9.8%
  ([the record](benchmarks/probe-v0.73.0-what-bounds-streaming.md)). Every measurement stands;
  the replication survives in a weaker form — the constraint is common to both machines and is
  not the GPU's compute.
- **Replication on hardware nothing like the original** (added in v2): 119.6 tok/s on the RTX
  3050 against a median 113.00 on an H100, at the same 3.32 GB peak.
- **A silent wrong-gradient defect, found and repaired.** On NF4 above ~165 MiB per layer the
  forward stayed bit-exact and the loss curve looked healthy while the gradients were wrong. The
  cause is named in the upstream library and reported there; the repair is gated against controls
  on real 32B and 72B.
- **Bit-exactness at real model sizes** instead of three-layer toys: forward from 0.5B to 72B,
  backward at 8B and 14B.
- **Trained-model quality, measured for the first time**, and indistinguishable from a resident run.
- **A comparison against DeepSpeed** — including the result that does not flatter us: eight cards
  of ZeRO-3 are slower than one card training resident.
- **The limitations section rewritten**: of v1's ten items, one closed and four more narrowed,
  and seven new ones added.

Cite the version you used. `10.5281/zenodo.21771064` is the concept DOI and always resolves to
the latest version (v3 today); v1 and v2 remain citable at their own version DOIs and are not
edited — the retraction above is a new version precisely so that the record of what we claimed,
and when, stays intact.

The measurement records behind every number in it are in [`benchmarks/`](benchmarks/), published
as written — including the failures, the assumptions that turned out wrong, and the numbers that
were measured and then discarded.

```bibtex
@misc{makazhan2026exact,
  title        = {Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU},
  author       = {Makazhan, Alpamys},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v3},
  doi          = {10.5281/zenodo.21918325},
  url          = {https://doi.org/10.5281/zenodo.21918325}
}
```

## License

[Apache-2.0](LICENSE). Copyright © the Soup contributors.
