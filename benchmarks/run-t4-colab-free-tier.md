<!--
Working record, published as written.

This is a single completed run, not a gate. It was captured from the notebook's
printed output on a machine that no longer exists (a free Colab session), so
everything below is what was on screen — no post-hoc re-measurement, and the
gaps are marked as gaps.

Hardware: Tesla T4 (sm_75, Turing), 15.6 GB, free-tier Google Colab.
Driver/library versions were not recorded.
-->

# An 8B streamed run on a free-tier Colab T4

**Status: RUN COMPLETED. Correctness NOT gated — see "What this does not
establish", which is longer than the results section on purpose.**

Every layer-streaming number this project has published came from one of two
machines: the maintainer's RTX 3050 Laptop (4 GB, Ampere, Windows) or a borrowed
8×H100 (Ampere-successor, Ubuntu). Both are Ampere or newer. The pre-Ampere fix
in #385/#387 was verified *using* fp16 **on an Ampere card**, which establishes
the plumbing and says nothing about Turing kernels — bitsandbytes NF4 on sm_75
in particular.

This run closes exactly that gap and nothing wider: **the code path runs to
completion on a Turing card.** It is the first streaming result from hardware the
maintainer does not own.

Unit convention: **decimal GB**, matching the gate records, except where a figure
was printed in GiB and is marked as such.

---

## Configuration

| | |
|---|---|
| GPU | Tesla T4, sm_75 (Turing), 15.6 GB total |
| Process cap | **4.00 GB** via `torch.cuda.set_per_process_memory_fraction` (fraction 0.256) |
| Model | `NousResearch/Meta-Llama-3.1-8B-Instruct` |
| Quantization | `4bit` (NF4) |
| Streaming | `stream_layers: true`, `stream_buffers: 2` |
| Batch / seq | `batch_size: 1`, `max_length: 256` |
| Adapter | LoRA r=8, α=16 |
| Precision | **fp16** — a T4 has no bf16 hardware |
| Notebook | [`notebooks/proof-4gb.ipynb`](../notebooks/proof-4gb.ipynb) |

**The cap was shown to bite rather than assumed.** A T4 has 15.6 GB, so a run
that merely completed on one would prove nothing about a 4 GB budget. The
process was constrained to 4.00 GB and the constraint was tested directly: a
deliberate allocation of **4.29 GiB was refused**. That is the only reason the
run means anything.

---

## What the pre-flight panel printed

```
base store    3.60 GB across 32 layers (pinned)
VRAM buffers  2 x 113 MB = 225 MB
resident      2101 MB (embeddings + adapters)
peak VRAM     ~3.02 GB at batch 1 x seq 256 (logits 0.46 GB)
free VRAM     15.10 GB
forecast      31-46 tok/s (from 2.20 TFLOPS measured on this card @ 1185 MHz)
```

LoRA applied: **3,407,872 trainable / 8,030,261,248 total (0.04%)**.

---

## What was measured

| Quantity | Value |
|---|---|
| **Peak VRAM** (`torch.cuda.max_memory_allocated`, training process) | **2.91 GB** |
| Predicted peak (pre-flight, same config) | ~3.02 GB |
| Prediction error | **+3.8%, over-predicting** |
| Steps / epochs | 7 steps, 1 epoch |
| Adapter written | **128 tensors, 128 non-zero** |

The prediction error is the one result here worth carrying forward. v0.72.3
fitted the peak-VRAM estimator to ten real runs with the explicit property that
it **never under-predicts**, because the estimate is allowed to stop a run and
under-predicting on Windows/WDDM is a silent spill rather than an exception. On a
different vendor's silicon, a different CUDA stack, a different quantisation
kernel generation and a model outside the fitted grid, it still erred by 3.8% in
the safe direction. One point is not a validation of the estimator; it is one
point that did not contradict it.

### The loss, quoted because it is what printed

```
4.5266, 4.2388, 3.7485, 4.6930, 4.1940, 4.1127, 4.0674
```

Seven steps, non-monotonic, 4.5266 → 4.0674. **This is not evidence of
learning** and is recorded only so the run's output is complete. A seven-step
curve on one seed at batch 1 is consistent with almost anything.

---

## An observation the run made by accident

The pre-flight panel reported **free VRAM 15.10 GB** — it read the *device*, not
the per-process cap the run was actually held to. So the fit decision was taken
against a budget **3.8× larger** than the one in force.

Nothing was harmed: the run genuinely fit (2.91 GB measured against a 4.00 GB
cap), so it did not survive on luck. But on capped hardware — Colab, Kaggle, a
MIG slice, anything behind `set_per_process_memory_fraction` — **the pre-flight
is not what enforces the real budget**, and a config that overshot would have
been waved through to a CUDA OOM instead of refused. That is precisely the case
`training.stream_vram_override` (#347) exists for, and this run is the first time
the gap has been seen rather than reasoned about.

---

## What this does **not** establish

- **No throughput claim. None.** The card was under an artificial memory cap;
  that is not a benchmark, and the notebook deliberately quotes no tok/s either.
  The panel's `31-46 tok/s` is a *compute bound derived from a GEMM probe*, not a
  measurement of what this run achieved — the two are routinely confused, which
  is why the forecast is printed with the clock it was taken at.
- **Backward / gradient exactness at 8B on Turing is not shown.** An adapter with
  128 of 128 tensors non-zero proves gradients *flowed*; it does not prove they
  were *correct*. The distinction is not academic here — #331 was a defect that
  produced a healthy loss curve, a bit-exact forward and silently wrong gradients
  on every layer but the last `stream_buffers`, and it was NF4-only and
  size-dependent. Nothing in this run would have caught it.
- **The notebook's section 4 — streamed-vs-resident bit-exactness on the T4 —
  produced no captured output.** It is recorded here as **unrun**, not as a pass.
  A T4 with 15.6 GB can in principle hold a resident NF4 8B reference, so this is
  the obvious next measurement and it has not been taken.
- **Forward exactness on sm_75 is not shown either.** The fp16 exactness results
  in the v0.73.0 notes were measured on an Ampere card using fp16.
- **One run, one seed, one configuration, no repeats**, on a session that cannot
  be returned to. Nothing about variance is claimed.
- Library versions (torch / bitsandbytes / transformers / peft) were **not
  recorded**, so this run is not exactly reproducible even on identical hardware.

## What it does establish

That the layer-streaming path — NF4 sharding, the pinned host store, the
two-buffer pool, fp16 compute selection after #385/#387 — **executes end to end
on a pre-Ampere card and writes a live adapter**, at an 8B model size, inside a
4 GB process budget. Before this, that had never been run.
