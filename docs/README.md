# Soup Documentation

[← Back to the main README](../README.md)

The main [README](../README.md) is the 5-minute front door. This directory holds the full
feature reference — every `soup` capability, grouped by area.

| Guide | Covers |
|---|---|
| [Training tasks & methods](training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-training, distillation, classification, vision/audio/TTS, unlearning, RAFT/RA-DIT, loop-hardening detectors |
| [PEFT, long context & efficiency](peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, optimizer & PEFT zoo, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, curriculum, auto-tuning, depth pruning + distill-heal (`soup shrink`) |
| [Performance & quantization](performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, multi-GPU / DeepSpeed / FSDP |
| [Data engineering](data.md) | Formats, the Axolotl/LF-parity pipeline, data tools, synthetic generation & forge, quality scorecards, trace tooling, remote datasets, mixing, recipe DAGs |
| [Evaluation & probes](evaluation.md) | Eval design/gate, eval-gated training, benchmarks, NLG metrics, calibration, Elo arena, diagnose, `soup ship` verdict, post-train X-ray probes, A/B, drift, tunability, `soup advise` |
| [Serving & export](serving-and-export.md) | OpenAI-compatible server, batch inference, benchmarking, merge/export, Anthropic Messages endpoint, speculative decoding (train + measure your own draft), deploy autopilot, Web UI, Agent Forge |
| [Adapters, registry & governance](adapters-and-governance.md) | Adapter lifecycle/management, model registry, Soup Cans, the data flywheel (`soup loop`), knowledge editing, steering, supply-chain controls |
| [Compliance & governance quickstart](compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` templates, provenance (BOM/attest/repro-receipt), audit log, air-gap, model-card autogen (`soup card`), CI gate (`soup ci init`) |
| [Backends, platform & ops](backends-and-ops.md) | MLX/Unsloth backends, Modal cloud GPU training, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, completions, plugins, utility commands |
| [Command reference](commands.md) | The full `soup` command list |
| [Supported models & extras](models.md) | Recommended model families, the VRAM size guide, the pip extras matrix |

> Per-release notes live on the [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases)
> page; see also the repo-root [CHANGELOG.md](../CHANGELOG.md).
