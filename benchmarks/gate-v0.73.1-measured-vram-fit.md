# Gate record — v0.73.1: measuring the streaming VRAM fit instead of predicting it (#349)

**Hardware.** RTX 3050 Laptop GPU, 4 GB (4.294 GB total, 3.45 GB free at rest),
Windows 11, NVMe. torch 2.5.1+cu121, transformers 4.57.6, trl 0.19.1, peft 0.18.1,
Python 3.10. Every number below was measured in that configuration. Nothing here
was measured anywhere else, and nothing is extrapolated to other hardware.

**What was being tested.** The layer-streaming pre-flight predicts peak VRAM from
a formula fitted to 10 real runs (v0.72.3, `benchmarks/gate-v0.72.3-breadth.md`)
and refuses a run it predicts will not fit. Its documented contract is that it
**never under-predicts**, because under-prediction is the failure that does not
announce itself — an OOM on Linux, a silent spill to host memory on Windows/WDDM.
Issue #349 proposed replacing the prediction with a measurement. This record is
the attempt to establish whether that is justified.

---

## The headline result

**The formula under-predicts at long sequence, measured through the real
`soup train`.** SmolLM2-135M streamed in bf16, batch 1, `quantization: none`,
LoRA r=8, `stream_buffers: 2`. "Real peak" is `torch.cuda.max_memory_allocated()`
over the whole training process.

| seq | predicted | real peak | pred/real | verdict |
|---|---|---|---|---|
| 4352 | 3.282 GB | 3.036 GB | 1.081x | over-predicts — safe |
| 5120 | 3.844 GB | 4.118 GB | **0.934x** | **under-predicts** |
| 6144 | 4.590 GB | 5.830 GB | **0.787x** | **under by 21%** |

The crossover sits between seq 4352 and 5120. Runs at 5120 and 6144 were let
through with `training.stream_vram_override` so the real peak could be observed;
the prediction shown is the one the shipped pre-flight printed in that same run.

**The existing test could not have caught this.** All ten rows of
`MEASURED_VRAM_GRID` are at seq 256 or 512 — the grid varies *batch* (1..8) and
never sequence length. `test_never_under_predicts` therefore pinned "never
under-predicts as batch grows" while reading as a global guarantee. It has been
narrowed to say so, with an assertion that fails if a longer row is ever added
without re-establishing the claim.

---

## Three readings that were wrong, kept because they cost real time

### 1. "The prediction over-predicts by 13.2x" — WITHDRAWN

First reproduction: a DPO config at batch 2 x seq 2048 was refused (predicted
6.09 GB against 3.45 GB free) and then completed in 7.3 s using 0.462 GB when
forced through. A 13.2x over-prediction.

It was an artifact of the fixture. The rows realised **142 tokens against a
budgeted 2048** — 6.9% of the shape the formula budgets for, and 2048/142 = 14.4x,
which is the whole of the apparent gap. The prediction budgets the *configured*
`max_length` because that is the worst case a real batch can pad up to. Re-run
with rows that genuinely fill 2048 tokens, the same config predicts 6.09 GB and
measures 5.304 GB — 1.15x, not 13.2x.

### 2. "Preference losses are over-budgeted ~8.8x" — NOT REPRODUCED

The v0.72.4 notes record DPO's above-resident cost as 51.76 MB where a 14 B/element
charge for the same shape is ~458 MB. At the budgeted shape that ratio does not
appear: DPO predicted 6.09 GB against a measured 5.304 GB (1.15x) sits alongside
SFT's 3.095 vs 3.051 (1.014x) — the same margin, not a preference-specific one.
The two figures measure different quantities (marginal above-resident cost vs
total peak) so this is not a contradiction, but the *practical* over-refusal it
implies is not present and should not be cited as one.

### 3. "The over-budget runs were silently spilling" — WITHDRAWN

`max_memory_reserved` reached 4.316 GB on a 4.294 GB card, which looked like the
documented WDDM spill. It is not evidence of one: `num_alloc_retries` was **0** on
every shape measured, so the caching allocator never came under pressure. Reserved
overshoots what must fit because it holds freed blocks that are returned under
pressure. This is the reason the shipped gate reads `max_memory_allocated`.

---

## Why the gate reads `allocated`, not `reserved`

Reserved ran 1.08x-1.41x allocated across the shapes measured — not a constant,
so it cannot be modelled. More decisively, **gating on reserved would refuse this
feature's own flagship configuration**: Llama-3.1-8B NF4 at batch 1 x seq 512
measures 3.4116 GB allocated against 3.6973 GB reserved, with 3.45 GB free, and it
runs. A gate that refuses the headline config is not a safer gate.

---

## The probe is not the training step, and is conservative

`measure_step_peak_bytes` runs a plain causal-LM forward+backward on synthetic
token ids at the configured shape. Against the real training step:

| seq | probe | real run | probe/real |
|---|---|---|---|
| 4352 | 3.471 GB | 3.036 GB | +14.3% |
| 5120 | 4.633 GB | 4.118 GB | +12.5% |

Consistently **conservative**, which is the correct direction for a gate: the
probe crosses into refusal slightly before the training step would. It is why the
probe's own crossover (between seq 3072 and 4096) is earlier than the training
path's (between 4352 and 5120).

**For a preference loss the probe is not validated, and one claim made during this
work was wrong.** An earlier draft of this record said the probe under-measures a
DPO step by 42% and was therefore unsafe. That compared two *different shapes* —
the probe at rows 4 x seq 1024 against DPO at rows 4 x seq 2048. Measured at the
matching shape, the probe reads **6.021 GB against the real DPO step's 5.304 GB,
i.e. +13.5%** — the same safe direction it shows for SFT, not the opposite one.

The `task: sft` restriction stays anyway, on a different and weaker justification:
that is **one shape, which is not a validation**. A preference loss reduces logits
to per-token log-probs (`selective_log_softmax`) rather than holding a full-vocab
upcast, so the probe is measuring a genuinely different computation and its
agreement here may not hold across shapes. If the sign ever flips, the failure is
a gate that waves through over-budget runs — worse than no probe. It is refused at
config load until it is measured across a grid, not because it is known to be
unsafe.

## Probe cost

| model | shape | probe wall-clock |
|---|---|---|
| SmolLM2-135M bf16 | 1 x 1024 | 1.02-1.15 s |
| SmolLM2-135M bf16 | 2 x 2048 | 5.09-5.13 s |
| Llama-3.1-8B NF4 | 1 x 512 | 5.33 s (+20.8 s model build) |
| SmolLM2-135M bf16 | 1 x 5120 (over budget) | 51.72 s |

The last row is the honest worst case: a shape that does not fit is slow to
measure precisely because it is thrashing. Paying it once to refuse the run beats
paying it every step.

## Mechanism: NOT established

What makes the formula diverge past ~4700 tokens is **not identified**. `seq**2`
from the attention score matrix is the obvious candidate; the measured excess does
not cleanly fit it, and no alternative was confirmed. Recorded as unknown rather
than guessed at. This is also the argument for measuring rather than adding
another coefficient: a formula cannot model a term nobody has identified.

---

## Live end-to-end verification (through the real `soup train`)

**Refuse direction** — SmolLM2-135M, seq 5120, `stream_vram_override: 4000000000`:

- probe off: predicted 3.85 GB against a 4.00 GB budget -> **accepted**, run
  proceeds, real peak **4.118 GB** (over budget).
- probe on: same config -> `a streaming step MEASURED 4.63 GB of VRAM at the
  configured shape but only 4.00 GB is free (the formula predicted 3.84 GB)`.

**Unblock direction** — seq 3072, `stream_vram_override: 2330000000`:

- probe off: `a streaming step is predicted to need 2.35 GB of VRAM but only
  2.33 GB is free` -> **refused**.
- probe on: `Predicted over budget (2.35 GB vs 2.33 GB free), but
  training.stream_vram_probe is on: measuring the real peak before deciding` ->
  `measured peak 2.31 GB ... in 5.23 s` -> run proceeds and completes.

**Four configs each refused naming its own reason**: probe with
`stream_layers: false`; probe on `task: dpo`; streaming on `task: grpo`
(permanent rollout exclusion); streaming with `batch_size: auto`.

Caveat on the unblock arm: the probe runs before training in the same process, so
that run's process-wide peak includes the probe itself and does not independently
confirm the training step's peak. The clean probe-vs-real comparisons are the
probe-off runs in the table above.

---

## Reviews

Five sequential ECC reviews; every finding fixed.

- **python** — CRITICAL: `_zero_probe_grads` swallowed exceptions silently,
  defeating the corruption guard it exists to be (now logs a warning). HIGH:
  `started` could be unbound on the OOM path, turning a clean refusal into an
  `UnboundLocalError` (now bound before the `try`). MEDIUM: a non-OOM probe
  failure discarded its reason.
- **code** — HIGH: changing `_stream_budget_lines` to return a 2-tuple broke
  `tests/test_issue348_calibrated_logits_wiring.py`, silently disabling the #348
  regression guard. Caught by the reviewer; the local run had reported "exit 0"
  because the exit code belonged to a piped `tail`, not to pytest.
- **security** — HIGH: with the probe on, a predicted miss no longer refused
  before the GPU was touched, so a shared `soup.yaml` could drive a real
  allocation at an absurd shape (now capped at 4x over budget). LOW: `batch_size`
  had no lower bound at all — `-4` parsed. Both fixed.
- **tdd** — CRITICAL: the off-CUDA message re-broke the #348 stub (fixed).
  HIGH: nothing proved `setup()` actually *called* the probe, and the 4x ceiling
  had no test. Both now covered, the former through the existing CUDA end-to-end
  harness with a control that fails if the answer is ignored.
- **verification-loop** — `soup --help` 0.85 s, exit 0; every new public symbol
  imports; no top-level `torch` in any changed module; `test_cli_startup_is_light`
  23/23.
