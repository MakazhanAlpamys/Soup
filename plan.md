# Soup — Roadmap

**Repo:** https://github.com/MakazhanAlpamys/Soup
**PyPI:** https://pypi.org/project/soup-cli/ (`pip install soup-cli`)
**Version:** v0.10.4 | 639 tests | CI green

### How to publish

```bash
# 1. Bump version in pyproject.toml + soup_cli/__init__.py
# 2. Tag and push
git tag v0.X.0
git push --tags
# GitHub Actions auto-publishes to PyPI
```

---

## Completed (v0.0.1 – v0.10.4)

- **CLI:** init, train, chat, push, merge, export, eval, serve, sweep, diff, doctor, quickstart, ui, version
- **Data:** inspect, validate, convert, merge, dedup, stats, generate
- **Training:** SFT, DPO, GRPO, PPO/RLHF, Reward Model, LoRA/QLoRA, QAT, auto batch size, resume, DeepSpeed, W&B, Unsloth backend
- **Multimodal:** `modality: vision`, LLaVA/ShareGPT4V, LLaMA-Vision/Qwen2-VL/Pixtral
- **Serving:** OpenAI-compatible API, transformers + vLLM backends, SSE streaming, tensor parallelism
- **Tracking:** SQLite experiment tracker, runs list/show/compare/delete
- **Export:** GGUF (Ollama/llama.cpp), LoRA merge
- **Web UI:** Dashboard, New Training, Data Explorer, Model Chat (FastAPI + SPA)
- **UX:** friendly errors, --verbose, Rich progress bars, confirmation prompts
- **Community:** CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, examples/, FUNDING.yml
- **Tests:** 624 tests, 41 files, ruff lint, CI on Python 3.9/3.11/3.12
- **v0.10.1 bugfixes:** Windows UnicodeEncodeError, PPO trl compat, compute dtype for CPU, diff torch_dtype, wandb version pin
- **v0.10.2 bugfixes:** ASCII progress bar, plotext fallback, auto-disable 4bit on CPU, friendly CPU error messages, torchvision compat check in doctor
- **v0.10.3-v0.10.4 bugfixes:** PPO use_cpu for CPU training, PPO trl >=0.28 API compat (args= vs config=, builtin .train()), GRPO CPU warning + use_cpu, use_cpu error message

---

## Roadmap

### v0.11.0 — Alignment methods

- [ ] KTO (`task: kto`) — unpaired preference data, wraps trl.KTOTrainer
- [ ] ORPO (`task: orpo`) — no reference model needed, wraps trl.ORPOTrainer
- [ ] SimPO (`task: simpo`) — simple preference optimization
- [ ] IPO (`task: ipo`) — identity preference optimization
- [ ] Templates: `soup init --template kto`, `--template orpo`

### v0.12.0 — Advanced PEFT

- [ ] DoRA (`use_dora: true`) — magnitude decomposition, already in PEFT
- [ ] LoRA+ (`loraplus_lr_ratio`) — different lr for A and B matrices
- [ ] GaLore — memory-efficient full-parameter training on consumer GPUs

### v0.13.0 — Batch inference + TensorBoard

- [ ] `soup infer --input prompts.jsonl --output results.jsonl` — batch inference (reuse diff.py code)
- [ ] `--tensorboard` flag in train — `report_to="tensorboard"`, HF Trainer handles it
- [ ] Supported models page in README (Llama 4, Gemma 3, Qwen 2.5/3, Phi-4, DeepSeek R1/V3)

### v0.14.0 — Pre-training + MoE

- [ ] `task: pretrain` — continued pre-training on raw text
- [ ] Plain text / tokenized datasets support
- [ ] MoE model support (Qwen3 30B-A3B, Mixtral, DeepSeek V3)
- [ ] ScatterMoE LoRA

### v0.15.0 — Performance + Long-context

- [ ] Liger Kernel integration (fused operations)
- [ ] FlashAttention-3/4 auto-detection
- [ ] FSDP2 support alongside DeepSpeed
- [ ] Sequence parallelism via Ring FlashAttention
- [ ] 128k+ context fine-tuning

### v0.16.0 — Embedding models + Export

- [ ] `task: embedding` — sentence transformers, BGE, E5
- [ ] Contrastive loss, triplet loss
- [ ] ONNX export (`soup export --format onnx`)
- [ ] TensorRT-LLM export
- [ ] Speculative decoding (`soup serve --speculative-decoding`)

### v0.17.0 — Data + Audio

- [ ] Local model as data generation provider
- [ ] Quality filters (perplexity, coherence scoring)
- [ ] `modality: audio` — Qwen2-Audio, Whisper fine-tuning
- [ ] SGLang backend for serving

### v0.18.0 — Cloud GPU (monetization)

Last step — all features are built, product is ready to sell.

- [ ] `soup login` — user registration, Stripe card linking
- [ ] `soup cloud run --config soup.yaml --gpu a100` — cloud training
- [ ] Backend API (FastAPI on VPS) — pod management, billing
- [ ] RunPod API integration (our account, 20-30% markup)
- [ ] `soup cloud status` — usage, balance, history
- [ ] Landing page soup.cloud
- [ ] Cost estimator and budget auto-stop
- [ ] Vast.ai, Lambda Labs, Modal — additional providers

---

## Principles

1. **CLI-first** — everything works from the terminal; UI is a bonus
2. **Zero config by default** — `soup train` with a minimal config just works
3. **Fail fast, fail loud** — bad data or missing GPU = immediate, clear error
4. **Open source core** — CLI is always free; monetize via managed service
5. **Test-driven** — every feature has tests, written alongside the code

---

## Competitive positioning

| Feature | Soup | LLaMA-Factory | Axolotl | Unsloth |
|---|---|---|---|---|
| One-command training | **Yes** | Partial | No | Notebook |
| Web UI | **Yes** | Yes | No | No |
| Experiment tracking | **Built-in SQLite** | W&B only | W&B only | No |
| Hyperparam sweep | **Yes + early-stop** | No | No | No |
| Data toolkit (7 tools) | **Yes** | Basic | No | No |
| Model diff | **Yes** | No | No | No |
| GGUF export | **Yes** | Yes | No | Yes |
| GRPO + custom rewards | **Yes** | Yes | Yes | Yes |
| Full RLHF pipeline | **Yes** | Yes | Yes | No |
| Cloud GPU | No | SageMaker | RunPod templates | No |
| MoE training | No | Yes | Yes | **12x faster** |
| DoRA/GaLore | No | Yes | Yes | Partial |
| KTO/ORPO/SimPO | No | Yes | Yes | No |
| Pre-training | No | Yes | Yes | Yes |
| 100+ model day-0 | No | **Yes** | Partial | Yes |

**Soup's edge:** best UX, most integrated toolkit, lowest barrier to entry.
**Gap to close:** cloud, advanced PEFT, more alignment methods, model breadth.
