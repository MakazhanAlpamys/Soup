# Measurement records

Raw gate records for Soup's layer-streaming feature, published as written.

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
| [`gate-h100-validation.md`](gate-h100-validation.md) |  The method on someone else's hardware: bit-exactness at real sizes, convergence quality, DeepSpeed, variance | **Forward** bit-exact to 72B; **backward** bit-exact to 14B NF4 pre-repair, re-gated after the STEP 14 fix at 32B (256/256) **and at 72B (320/320, the size where the defect was worst)**; 2.93x DeepSpeed ZeRO-3 offload in 9.7x less VRAM; and the silent wrong-gradient defect that fix repairs. **Carries three dated 2026-08-13 corrections**: it explains the H100 replication as host-to-device transfer, which the probe record above later measured and refuted. The original lines are left standing with the correction beside them |

## Harnesses

[`harness/`](harness/) holds the measurement scripts that can be run against a
released Soup, so a claim in a record above can be re-measured rather than taken
on trust. It starts small on purpose — most of this session's ~20 harnesses live
only in a scratchpad on a machine that is gone, which is
[#379](https://github.com/MakazhanAlpamys/Soup/issues/379).

| Script | Question it answers | Cost |
|---|---|---|
| [`issue331_qlora_scope.py`](harness/issue331_qlora_scope.py) | Does the #331 wrong-gradient defect reach **ordinary QLoRA**? Three arms in one process, with the positive control that makes an exact result mean something. Answer: no — 0.0 against a control that diverges by 3.77e-01 | ~15 s, 4 GB card, no downloads |

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
