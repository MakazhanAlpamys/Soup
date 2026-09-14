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

**STALE, #714: the `31-46 tok/s` line above is superseded, not current.** #648
changed the compute probe to measure the GEMM ceiling in the card's resolved
stream dtype instead of bf16 through Turing's software emulation, and this
run's session predates that fix. `notebooks/proof-4gb.ipynb` (the run behind
#910, pinned to v0.75.0, which carries #648) re-ran the same probe in a later
session and its panel printed, verbatim:

```
forecast      288-423 tok/s — a compute-bound bound, not a promise
              (from 20.39 TFLOPS measured on this card now using float16 @ 525 MHz)
```

**This is not a controlled dtype-isolated comparison, and the two sessions ran
different plans** — #910's own record shows base store 5.70 GB (was 3.60 GB),
an extra 1051 MB large-layer slot, and 0 MB resident beyond adapters (was
2101 MB). The clock figures are not a control either: `sm_clock_mhz()`
(`layer_stream_runtime.py:1220`) takes a single post-hoc `nvidia-smi` sample,
and its own docstring records this box's boost clock moving 442-952 MHz
*inside one measurement run*, and 3.5 and 6.75 TFLOPS from two sessions at the
same reported clock. 525 MHz measured against 20.39 TFLOPS would be 94.8% of
that clock's scaled peak — implausibly high for a real GEMM — so the printed
clock does not describe the rate that was actually sustained; treat it as
decoration, per the helper's own docstring, not as an experimental control.

What does hold up despite that: the ~9.3x jump is consistent across all three
published quantities (`20.39 / 2.20` = 9.27, `288 / 31` = 9.29, `423 / 46` =
9.20), which is the right order of magnitude for a probe that stopped running
in bf16-through-emulation and started running in the card's native fp16 —
consistent with, not proof of, the dtype being the whole cause.

**The new bracket remains unvalidated against a measured tok/s from this run**
— #714 asked for the bracket to be checked against a measured tok/s, which
this run does not have: the trainer's own metrics render in the notebook as
`<IPython.core.display.HTML object>` rather than a captured
`train_samples_per_second`, and the notebook deliberately quotes no tok/s
either (see below). This record fixes a wrong forecast; it does not close the
validation #714 also asked for. That half should be tracked as its own issue.

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
  The panel's `31-46 tok/s` is **stale — see "What the pre-flight panel
  printed" above (#714)**, superseded by `288-423 tok/s` from a corrected
  probe, and still unvalidated against a measured tok/s either way. Whichever
  bracket is current, it is a *compute bound derived from a GEMM probe*, not a
  measurement of what this run achieved — the two are routinely confused, which
  is why the forecast is printed with the clock it was taken at.
- **Backward / gradient exactness at 8B on Turing is not shown.** An adapter with
  128 of 128 tensors non-zero proves gradients *flowed*; it does not prove they
  were *correct*. The distinction is not academic here — #331 was a defect that
  produced a healthy loss curve, a bit-exact forward and silently wrong gradients
  on every layer but the last `stream_buffers`, and it was NF4-only and
  size-dependent. Nothing in this run would have caught it.
- **Section 4 — streamed-vs-resident forward bit-exactness on the T4 — has since
  been run and recorded.** `notebooks/proof-4gb.ipynb` (#844) was executed end to
  end on a free Colab T4 (Tesla T4, driver-reported 15.6 GB). The date does not
  appear anywhere in the notebook's own output, so it is provenance rather than
  a printed result: the commit that carries the run (`e06864c`) was made at
  **2026-09-12T19:50:14Z** (UTC; the author's local +05:30 clock reads
  2026-09-13, which is why an earlier version of this line said that instead).
  Run against **soup-cli 0.75.0**, `torch 2.11.0+cu128`, `transformers 5.16.1`,
  `peft 0.20.0`, `bitsandbytes 0.50.2` (printed by the notebook's own version
  cell — the library-version gap the rest of this document notes does not apply
  to this second run): `max |streamed - resident| = 0.0`, `torch.equal = True`.
  That is `HuggingFaceTB/SmolLM2-135M-Instruct` in fp16, unquantized
  (`shard_checkpoint` called without `quant`, defaulting to `QUANT_NONE`), not the
  8B NF4 configuration this document measures — a resident NF4 8B reference on
  Turing is still the obvious next measurement and has not been taken.

  **The same notebook run also re-measured section 5** (the 8B NF4 run this
  document's "What was measured" table above reports) and got substantially
  different numbers on the same card: base store 5.70 GB (was 3.60 GB), an
  extra 1051 MB large-layer VRAM slot alongside the original 2 × 113 MB buffers,
  0 MB resident beyond adapters (was 2101 MB), and a **measured peak of 1.83 GB**
  (was 2.91 GB). This is self-consistent, not a contradiction: the embeddings and
  head moved out of the resident allocation into a streamed large-layer slot
  between the two runs, which explains both the larger pinned store and the
  smaller peak. The original run's numbers above are left as originally
  captured, per this document's own working-record convention; this second
  measurement is recorded here rather than silently overwriting them.
- **Forward exactness on sm_75 in fp16 is now shown** (see above); the v0.73.0
  notes' fp16 exactness results were measured on an Ampere card, not Turing.
- **One run, one seed, one configuration, no repeats**, on a session that cannot
  be returned to. Nothing about variance is claimed.
- Library versions (torch / bitsandbytes / transformers / peft) were **not
  recorded**, so this run is not exactly reproducible even on identical hardware.

## What it does establish

That the layer-streaming path — NF4 sharding, the pinned host store, the
two-buffer pool, fp16 compute selection after #385/#387 — **executes end to end
on a pre-Ampere card and writes a live adapter**, at an 8B model size, inside a
4 GB process budget. Before this, that had never been run.
