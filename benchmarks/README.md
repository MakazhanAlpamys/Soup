# Measurement records

Raw gate records for Soup's layer-streaming feature and its release gate,
published as written.

These are not a report assembled after the fact. They are the working records
kept while each item was built and verified, so they contain the failures, the
assumptions that turned out wrong, and the numbers that were measured and then
discarded — in the order those things happened.

They are the evidence behind the preprint:

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model
> on a 4 GB Laptop GPU.* Zenodo.
> [10.5281/zenodo.21771064](https://doi.org/10.5281/zenodo.21771064)

| File | What it gates | Headline |
|---|---|---|
| [`gate-v0.72.0-layer-streaming.md`](gate-v0.72.0-layer-streaming.md) | The streaming path itself | Bit-exactness vs a resident reference; 3B bf16 trained on a 4 GB card |
| [`gate-v0.72.2-nf4.md`](gate-v0.72.2-nf4.md) | NF4 quantised streaming | Llama-3.1-8B at 119.6 tok/s in a 3.32 GB peak |
| [`gate-v0.72.3-breadth.md`](gate-v0.72.3-breadth.md) | Nine architectures, batching, accumulation, resume, disk tier | Peak-VRAM predictor at 0.85% worst-case error; accumulation is per-token I/O-neutral |
| [`gate-v0.72.4-preference-losses.md`](gate-v0.72.4-preference-losses.md) | DPO / ORPO / SimPO / KTO over the streaming engine | DPO's reference model costs no extra weights — 0.914x the SFT peak, against +730.44 MB for a real second instance |
| [`probe-v0.73.0-what-bounds-streaming.md`](probe-v0.73.0-what-bounds-streaming.md) | What the streamed step is actually bound by, and Cut Cross-Entropy on top of it | **Not** transfer-bound: 71.3% of the card's same-session GEMM ceiling, and deleting every host-to-device byte buys 1.4%. CCE triples the usable microbatch for +9.6% |
| [`run-t4-colab-free-tier.md`](run-t4-colab-free-tier.md) | Not a gate — one completed run on hardware the maintainer does not own | An 8B NF4 streamed run finishes on a free-tier Colab **T4 (sm_75, Turing)** inside a 4.00 GB process cap, peak **2.91 GB** against a predicted 3.02 (the estimator over-predicts by 3.8%, the safe direction). **No throughput is quoted** — a capped card is not a benchmark — and gradient exactness on Turing is *not* shown |
| [`gate-v0.73.1-measured-vram-fit.md`](gate-v0.73.1-measured-vram-fit.md) | Measuring the streaming VRAM fit instead of predicting it | The peak-VRAM formula **under-predicts at long sequence** — 0.934x the real peak at seq 5120 and 0.787x at 6144, measured through the real `soup train`, a direction the v0.72.3 grid could not see because all ten of its rows sit at seq 256 or 512. Carries **three readings that were withdrawn** during the work, including two that looked like the headline result |
| [`gate-395-second-stack-vram.md`](gate-395-second-stack-vram.md) | The VRAM fit on a second GPU/stack — the verification `gate-v0.73.1` could not do on one card | The under-prediction **does not reproduce**: flat 1.16x over-prediction across seq 2048-6144 on two models, against 0.934x / 0.787x on the RTX 3050. The gap is a single stack-specific term, not drift with sequence. The SDPA math-path hypothesis is published as a **negative result** with a positive control, and explicitly *not* extended to the 3050. Carries a **discarded first sweep** whose numbers were an artifact of `max_length` truncating rather than padding |
| [`gate-v0.73.2-leg2-scoring.md`](gate-v0.73.2-leg2-scoring.md) | `soup ship`'s leg-2 scorers — the release gate itself, not streaming | A suite scored **0.000** for a stub answering every item *correctly*, twice over: `\boxed{C}` was unknown to the MCQ extractor, and a tool call one closing brace short fell through to the inner object. Two models with **byte-identical** scores on all seven suites, one refusing every benign request, were indistinguishable to the gate. The measured noise floor on this box is **0.0000** — CPU greedy decode is deterministic, and the H100's 0.015/0.020 is explicitly *not* re-claimed here. Carries a **withdrawn** order-dependence scare, a control that varied nothing, and a review finding that was checked against three implementations and **partly rejected** |
| [`run-m1-8gb-mlx-sft.md`](run-m1-8gb-mlx-sft.md) | Not a gate — the MLX backend's training loop on hardware CI does not have | `MLXSFTTrainerWrapper` runs end to end on an **8 GB M1**, and the shipped `llama3.1-8b-sft-mlx` recipe **fits**: peak **5.154 GB**, 48 iterations in 71 s. A model too large does **not** fail here, it pages — free memory sat at 5-6% while the machine stayed responsive, so an allocation-failure pre-flight would not fire (the #649 trap from the other platform). Carries a **failed fixture** — the model id #23 itself recommends ships legacy `weights.NN.safetensors` and cannot be loaded — and two **corrections** I made mid-measurement |
| [`gate-qwen4-ple-m4-max.md`](gate-qwen4-ple-m4-max.md) | Qwen4-Exp/oQ decoder mapping and read-only PLE on Apple Silicon | **Partial pass only:** 1,167/1,167 cached tensors and tiny CPU/MPS parity passed; the production optimizer step was stopped and is explicitly **not validated** |
| [`gate-h100-validation.md`](gate-h100-validation.md) |  The method on someone else's hardware: bit-exactness at real sizes, convergence quality, DeepSpeed, variance | **Forward** bit-exact to 72B; **backward** bit-exact to 14B NF4 pre-repair, re-gated after the STEP 14 fix at 32B (256/256) **and at 72B (320/320, the size where the defect was worst)**; 2.93x DeepSpeed ZeRO-3 offload in 9.7x less VRAM; and the silent wrong-gradient defect that fix repairs. **Carries three dated 2026-08-13 corrections**: it explains the H100 replication as host-to-device transfer, which the probe record above later measured and refuted. The original lines are left standing with the correction beside them |

## Harnesses

[`harness/`](harness/) holds the measurement scripts that can be run against a
released Soup, so a claim in a record above can be re-measured rather than taken
on trust. It starts small on purpose — most of this session's ~20 harnesses live
only in a scratchpad on a machine that is gone, which is
[#379](https://github.com/MakazhanAlpamys/Soup/issues/379).

| Script | STEP / record | Question or claim | Cost / requirements |
|---|---|---|---|
| [`issue331_qlora_scope.py`](harness/issue331_qlora_scope.py) | #331 | Does the wrong-gradient defect reach **ordinary QLoRA**? Three arms in one process, with a positive control. Answer: no — 0.0 against a control that diverges by 3.77e-01 | ~15 s, 4 GB card, no downloads |
| [`bnb_repro.py`](harness/bnb_repro.py) | Upstream bitsandbytes #2034 / NF4 defect | Minimal standalone reproducer for the NF4 stale-gradient mechanism: recycled buffers expose `MatMul4Bit`'s `ctx` lifetime issue; private buffers are the reference and bf16 is the control | CUDA GPU required; no downloads; historical environment: torch 2.13.0+cu130, bitsandbytes 0.50.0; ~1 min |
| [`bitexact.py`](harness/bitexact.py) | v0.72.0 / STEP 1; gate-h100-validation.md Reproducing | Shard -> stream -> compare logits, LoRA gradients, and loss curve against a resident reference using matching numerics; NF4 uses a resident NF4 reference | CUDA GPU required; model/checkpoint weights required; historical run: Qwen2.5-0.5B-Instruct NF4, seq 64; CUDA validation: torch 2.5.1+cu121, transformers 4.57.6, trl 0.19.1, peft 0.18.1 |
| [`fsdp_sharding_probe.py`](harness/fsdp_sharding_probe.py) | STEP 20 / [#373](https://github.com/MakazhanAlpamys/Soup/issues/373) | Is the base actually **sharded**, or was the flag merely accepted? Per-rank `local == total / world_size` asserted, with the single-GPU control expecting the *full* count so a probe that always divides fails instead of passing everything. Repairs the STEP 20 probe, which raised `TypeError('FullyShardedDataParallel does not support len()')` on all eight ranks at 70B — a `torch.compile` wrapper, not FSDP | Accounting is pure-Python and CPU-tested; a real reading needs the multi-GPU box under test |
| [`mlx_sft_smoke.py`](harness/mlx_sft_smoke.py) | [#23](https://github.com/MakazhanAlpamys/Soup/issues/23) / [`run-m1-8gb-mlx-sft.md`](run-m1-8gb-mlx-sft.md) | Does `MLXSFTTrainerWrapper.train()` actually run against a real MLX runtime, and what does it peak at? Asserts MLX **dispatch before measuring**, so a transformers-path number cannot be published as an MLX one (#363). Generates its own fixture; prints host memory and swap either side | Apple Silicon + `pip install -e ".[mlx]"`; downloads the chosen model; 20 s at 0.5B, ~6 min at 8B including the Hub fetch |
| [`issue395_second_stack_vram.py`](harness/issue395_second_stack_vram.py) | #395 / [`gate-395-second-stack-vram.md`](gate-395-second-stack-vram.md) | Does the long-sequence under-prediction reproduce on a second stack, and is the gap the logits multiplier? Answer: no and no — the loss term measures 12.000000 (spread 0), and `14 x E` alone exceeds the whole real peak on every row, so the `+2` retained copy is absent. Three modes: sweep, `--measure`, `--sdpa` | 24 GB card (largest arm peaks 11.85 GB); downloads SmolLM2-135M and Qwen2.5-0.5B; needs `jinja2>=3.1.0`; a few minutes |

## Hardware

Every number in the four `gate-v0.72.*` records was measured on one machine:

- **GPU** — RTX 3050 Laptop, 4 GB (4.29 GB usable)
- **Host** — 16.9 GB RAM, NVMe
- **OS** — Windows 11

`gate-h100-validation.md` is the exception and the reason it exists: 8x H100
80 GB, 503 GB RAM, Ubuntu 24.04, on a much newer torch/bitsandbytes/trl/peft
stack. It is the first record from hardware other than the laptop, and the first
able to hold a *resident* reference for an 8B–72B model — which is what turns
"bit-exact on a 3-layer toy" into "bit-exact on real models".

`gate-395-second-stack-vram.md` is the third box: an **NVIDIA A10G 23 GB**
(AWS `g5.xlarge`), Ubuntu 22.04, torch 2.13.0+cu130 / transformers 5.16.1 /
trl 0.29.1 / peft 0.20.0. It exists to answer a question one card cannot —
whether the long-sequence under-prediction is a property of the formula or of
the stack it was fitted on. Note the direction of the dependency floor:
`pyproject.toml [train]` now requires `transformers>=5.16.1`, `trl>=0.29.0`,
`peft>=0.20.0`, so this stack **is** that floor while the RTX 3050 stack below
is beneath it.

`run-t4-colab-free-tier.md` is the second exception and a much weaker one: a free
Colab **Tesla T4 (sm_75, Turing)**, one run, no repeats, no captured
correctness comparison, on a session that cannot be returned to. It is filed here
because it is the only evidence that the streaming path executes at all on a
pre-Ampere card, not because it gates anything.

Windows/WDDM matters for reading these: it spills into shared host memory rather
than raising `CUDA out of memory`, so a run completing is not evidence that its
configuration fits. That is why peak VRAM is reported alongside every throughput
figure, and why the fit decision refuses rather than warns.

## Reading the numbers

- **Throughput is quoted with the SM clock it was taken at.** This card's boost
  clock varies about 13% between sessions, so a fraction-of-ceiling stated
  without its clock is not meaningful. Where a GEMM ceiling is compared against,
  it was measured in the same session.
- **The correctness reference always matches the numerics under test** — a
  streamed NF4 run is compared against a *resident NF4* run, never against
  resident bf16, which would hide a real defect inside quantisation error.
- **"Bit-exact" is always two claims, never one.** The **forward** (logits,
  `torch.equal`) and the **backward** (every LoRA gradient tensor) are measured
  independently and do not always agree: in `gate-h100-validation.md` the forward
  is exact at every size up to 72B while the backward, pre-repair, was wrong above
  ~165 MiB per NF4 layer. So "bit-exact at 72B" on its own is not a statement this
  record makes — check which half, at which quantisation, at which MiB/layer. That
  file opens with a per-model ledger giving all four for every row, and marks
  anything unmeasured "not tested" rather than leaving it blank.
- **Derived figures are labelled as arithmetic.** Where a line says "1M tokens =
  2.3 h", that is division, not a measured wall-clock run.

## Reproducing

The implementation ships in Soup under Apache-2.0. Reproduction commands are in
Appendix A of the paper; the correctness protocol runs as part of the project's
test suite, so a regression in bit-exactness fails CI rather than reaching a
user.

```bash
pip install "soup-cli[train]"
```
