<!--
Measurement record for Soup layer streaming, published verbatim.

These are the working gate records written while the feature was built, not a
report assembled afterwards: they contain the failures, the corrected
assumptions and the numbers that were discarded, in the order they happened.
They are the evidence behind the paper "Exact Layer Streaming: LoRA Fine-Tuning
of an 8B Model on a 4 GB Laptop GPU".

Hardware for every number below: RTX 3050 Laptop (4 GB, 4.29 GB usable),
16.9 GB host RAM, NVMe, Windows 11, unless a line states otherwise.
-->

# v0.72.0 Layer Streaming — Gate results

**Status: both gates PASS. v0.72.0 shipped 2026-07-26 (tag `v0.72.0`, PyPI live).
Numbers below are final and were reproduced verbatim in the release notes, CHANGELOG and
the accompanying report. §"After the gates" records what implementation
and review then found — the gates were necessary but not sufficient.**

Hardware (all numbers on this box, no extrapolation):
**Windows 11 · RTX 3050 Laptop 4 GB (CC 8.6, driver 591.44) · 16.9 GB RAM · NVMe · PCIe**
torch 2.5.1+cu121 · transformers (llama modeling) · peft 0.18.1 · accelerate 1.12.0 · Python 3.10

Unit convention: **decimal GB throughout** (16.9 GB host RAM, 7.65 GB page-locked ceiling).
Earlier drafts mixed GiB and GB for the same machine — 15.7 GiB and 16.9 GB are the same RAM,
7.12 GiB and 7.65 GB the same pinned ceiling.

---

## GATE 1 — correctness — **PASS** (2026-07-26)

Throwaway spike, nothing under `src/`. Model **HuggingFaceTB/SmolLM2-135M**
(ungated; 30 layers, d=576, ffn=1536, vocab=49152, tied embeddings), bf16,
LoRA r=16 on `q_proj,v_proj` (921,600 trainable), seq 512, batch 1,
**double buffering (n=2) + dedicated prefetch stream + events from the first
prototype** (plan §2.4).

Construction actually under test:
- checkpoint sharded to `layer_NNN.safetensors` + `extras.safetensors`
- `RamSource` — whole base pinned in CPU RAM (212.4 MB), Tier 1
- `LayerBufferPool(n=2)` pre-allocated VRAM buffers + `torch.cuda.Event`
- skeleton built with `accelerate.init_empty_weights` — **270 decoder-layer
  weight tensors (30 layers x 9) never leave `meta`**; only embed / final norm /
  LoRA are materialised. The resident load never happens (plan P14).
- `torch.func.functional_call` feeds pooled buffers into the unmodified HF
  layer (plan §5.3 / P13)
- `torch.utils.checkpoint(..., use_reentrant=False)`; base weights carry
  `requires_grad=False` — no `detach()`, no `no_grad()` (plan P2)

| # | Check | Threshold | Measured | Verdict |
|---|---|---|---|---|
| 1 | streamed vs resident logits, max abs diff | < 1e-3 | **0.0 (bit-exact)** | PASS |
| 2 | LoRA grad on **layer 0** | != 0 | **9.50e-01**; 30/30 layers non-zero | PASS |
| 3 | 100 steps streamed vs resident, loss curve | within noise | **max rel 0.0**; 0.61296 -> 0.22499 on both | PASS |
| 4 | same seed twice -> identical loss | identical | **max diff 0.0** | PASS |
| 5 | + pinned-CPU boundary offload (`saved_tensors_hooks`) vs resident | < 2% rel | 4.83e-3 rel | PASS |
| 6 | buffer-count invariance n=2 vs n=3 | identical | **max diff 0.0** | PASS |

Peak VRAM during the run: 2.09 GB — but that figure includes the **resident
reference model held simultaneously** for comparison, so it is not the
streaming footprint. 5788 layer loads served from the pool.

Notes:
- Checks 1/3/4/6 are **bit-exact**, which is the expected result: streaming
  substitutes the identical weight bytes into the identical kernels. Anything
  other than 0.0 would have meant a real numerical difference.
- Check 5 is the only non-zero: 0.48% relative over 25 steps. Verified **not a
  race** — the offload path is bit-reproducible across two same-seed runs
  (`offload_determinism_max_diff = 0.0`), so it is a different-but-valid bf16
  rounding path amplified by 25 optimizer steps. Async pinned activation
  offload is **out of v0.72.0 scope** regardless; recorded for v0.72.2.
- The `owner[slot]` early-overwrite assert (plan P1 tripwire) never fired.
- Two real bugs the gate caught, both of which would have silently broken a
  naive implementation:
  1. PEFT's `LoraModel` calls `self.model.forward(...)` **directly**, bypassing
     `__call__` — so a prefetch pre-hook on the CausalLM wrapper never fires.
     The hook must go on the module that owns `.layers`.
  2. PEFT initialises adapters on the base layer's device, which is `meta`
     here; adapter weights must be explicitly re-materialised (A: kaiming, B: 0).

---

## GATE 2 — the number — **> 50 tok/s row of the §8 table** (2026-07-26)

Common config: bf16 · `task=sft`-shaped loop · LoRA r=16 on `q_proj,v_proj` ·
batch 1 · gradient checkpointing ON (so C=6) · `bitsandbytes.PagedAdamW8bit` ·
**double buffering n=2 + dedicated prefetch stream** · 50 measured steps after
10 warm-up · GPU util from `nvidia-smi dmon -s u` (`sm` column).

| Model | RAM store | S | tok/s | GPU util (mean/max) | Peak VRAM | RAM store size | TFLOPS eff |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | pinned | 512 | **978.6** | 91.4% / 100% | 1.47 GB | 0.72 GB | 2.90 |
| Qwen2.5-1.5B | pinned | 512 | **525.0** | 96.8% / 100% | 1.82 GB | 2.62 GB | 4.86 |
| Qwen2.5-1.5B | pinned | 1024 | **487.6** | 96.7% / 100% | 2.96 GB | 2.62 GB | 4.52 |
| Qwen2.5-3B | **pageable** | 512 | **143.1** | 79.3% / 100% | 2.15 GB | 5.55 GB | 2.65 |

Baselines / premise:
- **Qwen2.5-3B resident training OOMs on this card** — `CUDA out of memory`
  on a plain forward+backward at S=512. This is the premise of the feature and
  it is confirmed, not assumed. Streaming runs the same model in **2.15 GB**.
- **Qwen2.5-0.5B resident is a VALID baseline** (peak 3.23 GB, genuinely inside
  4 GB): resident 1398.0 tok/s vs streamed 978.6 -> **streaming costs 1.43x**.
  This is the only honest streaming-vs-resident overhead figure here.
- **Qwen2.5-1.5B resident is NOT a valid baseline** and its number is discarded:
  it reported a 6.15 GB "peak" on a 4 GB card, i.e. it spilled into WDDM shared
  host memory and crawled at 91.3 tok/s. Quoting "streaming is 5.7x faster than
  resident" from that would be a garbage claim.

TFLOPS back-check (plan §3): **the plan's assumed 9 TFLOPS peak for this card is
wrong**, so the naive back-check mislabels a good number as "IMPLAUSIBLE".
Measured on this box: achievable dense bf16 GEMM = **6.76 TFLOPS** (fp32/TF32 =
3.38, the expected 2x Ampere ratio); GA107 theoretical dense bf16 tensor peak
~24.5 TFLOPS. The 1.5B run's 4.86 TFLOPS is 20% of theoretical — plausible, and
corroborated by 96.8% GPU utilisation.

> **RE-MEASURED 2026-07-27 (v0.72.1 session) — 6.76 is CORRECT; the ceiling tracks GPU CLOCK.**
> Re-measuring the same square shapes gave 7.08/7.52/7.66 in one session and 6.23/6.63/6.75 in
> another. Repeating 4096^3 six times *within* a session is stable to <1% (6.93-6.94) at a pinned
> 862 MHz / 63 C. So the ~13% spread is between-session boost-clock state, not measurement error,
> and **6.76 reproduces almost exactly in a low-clock session**. Rule: a ceiling is only comparable
> to a throughput measured in the SAME session, and any fraction-of-ceiling must state the SM clock.
> Shape-matched weighted ceilings at 862 MHz: 8B 7.58 (high-clock session) / 3B 6.67 / 1.5B 6.39 /
> 0.5B 5.90. Numerator convention: decoder C=6 (NOT 8 — the frozen base has no dL/dW),
> embed_tokens 0 FLOPs (lookup), lm_head C=4 (outside the checkpointed layers, frozen).
> See `gate-v0.72.2-nf4.md`.
 Every other run is under 32% even of the
plan's understated 9.

**Reading of the §8 decision table: the ">50 tok/s" row — "Mechanism is healthy.
Ship v0.72.0 BETA, go straight to v0.72.1 (NF4)."** 143 tok/s at 3B, 525 at
1.5B, 979 at 0.5B; GPU utilisation 79–97% everywhere, nowhere near the <30%
scheduler-broken or <50% hardware-bound pathologies.

**Corrected 2026-07-27 — this paragraph originally read "IO is fully hidden".**
That is true of the 1.5B row and NOT of the 3B headline row, and the step-level
arithmetic (from these same numbers) says so: at 143.1 tok/s and a 5.55 GB store,
a step is 3.578 s and moves 2 x 5.55 = 11.10 GB, i.e. an implied 3.1 GB/s, while
compute at the measured 6.76 TFLOPS ceiling accounts for only ~1.26 s — about 35%
of the step. The 3B run is TRANSFER-bound; its pageable store makes
`copy_(non_blocking=True)` synchronous, which is the same fact as the 79.3% util.
Per-row compute fraction: 0.5B 31% / 1.5B 61% / 3B 35%. Only the 1.5B row (pinned,
5.4 GB/s implied, 96.8% util) is compute-bound as plan §8 predicted.

`nvidia-smi dmon -s u` reports SM occupancy — "a kernel was resident" — which is
necessary but NOT sufficient for the transfer to be hidden. Reading it as proof of
overlap is what produced the original wrong sentence.

### Honesty caveats on these numbers
1. **The 3B run used a PAGEABLE store, not a pinned one, so 143 tok/s is a
   LOWER BOUND.** This box cannot page-lock 5.55 GB alongside an IDE + agent
   session: measured maximum pinned host allocation is **7.65 GB** (with 9.1 GB
   "available"), and the 3B attempt failed with a host-side
   `CUDA error: out of memory` at 9.3 GB available even after the loader was
   rewritten to remove its transient copies. Pageable memory makes
   `copy_(non_blocking=True)` synchronous, which is visible as the utilisation
   drop from 96.8% (pinned, 1.5B) to 79.3% (pageable, 3B). This is exactly the
   RAM ceiling plan P16 predicted.
2. Numbers are Windows/WDDM (plan P15) and therefore systematically pessimistic
   versus Linux.
3. The plan's §4.3 ceiling table for this card is built on the wrong peak
   figure; do not quote its per-model tok/s.
4. Nothing here was measured above 3B. **No 8B/14B claim is supported.**

Derived, clearly labelled as arithmetic and not measurement:
1M training tokens at the measured rates = **1.9 h at 3B**, 0.53 h at 1.5B.

### Implementation findings the gates produced (feed into Phase C)
- The prefetch pre-hook must be attached to the module that **owns `.layers`**,
  not the CausalLM wrapper: PEFT's `LoraModel.forward` calls
  `self.model.forward(...)` directly and bypasses `__call__` hooks.
- The streamed layer wrapper must be **attribute-transparent** (`__getattr__`
  delegating to the wrapped layer): this `transformers` reads
  `decoder_layer.attention_type` straight off the layer object.
- PEFT initialises adapter weights on the base layer's device, which is `meta`
  in a streaming build; adapters must be explicitly re-materialised.
- The RAM store must be **allocated once and filled by `copy_`**, not
  `load_file -> .to() -> .pin_memory()`; the latter's 3 transient copies per
  layer are what push a large base over the page-locked ceiling.
- `expandable_segments:True` (plan P7) is **not supported on Windows** — torch
  warns and ignores it. Do not rely on it here.

---

## After the gates — what a passing gate did NOT catch (2026-07-26/27)

Recorded because the honest lesson of this release is that **6/6 correctness checks and a
clean throughput table still left a bug that would have killed every real run.** The gates
tested the streaming mechanism in isolation; they did not test it inside a real trainer.

1. **CRITICAL — every streaming run would have died at trainer construction.**
   `transformers.Trainer.__init__` calls `_move_model_to_device` -> `model.to()`, and `.to()`
   on a module holding `meta` parameters raises `NotImplementedError: Cannot copy out of meta
   tensor`. The failure lands immediately after the pre-flight prints "Layer streaming ready",
   so the run looks healthy right up to the crash. 145 green tests missed it because **no test
   built a real `Trainer`/`SFTTrainer` from the streamed model** — the spike drove its own loop.
   Fixed with `hf_device_map` + a `StreamedDecoderLayer._apply` override that passes meta
   tensors through while still moving/casting the real LoRA params. A *second* meta-move inside
   `accelerate.prepare_model` was then found by the new end-to-end test, not by review.
2. **HIGH — the shard cache could silently stream the wrong weights.** It was keyed by model
   slug with only a dtype check, so a local checkpoint retrained in place (or two ids colliding
   onto one slug) would train against stale weights with no error. Now keyed by a
   `source_fingerprint` (per-shard name + size + mtime digest) that invalidates like dtype does.
3. **HIGH — DoRA / VeRA / PiSSA / OLoRA read the real base weight at `get_peft_model()` time**
   and crash opaquely on `meta`. Now refused by name at config-parse.
4. **The pre-flight hardware-fit gate refused the runs this feature exists to enable.** It
   models a RESIDENT run, so it rejected a 3B streaming config outright. Found by the step-6
   smoke, not by any test. Now skipped for streaming runs, with a passing control test proving
   resident configs are still gated.

Shipped scope is narrower than the gates alone would justify: RAM tier · bf16 · `task=sft` ·
Llama/Qwen · batch 1 · no gradient accumulation · no `--resume`. Every refusal names the
release that lifts it (NF4 -> v0.72.1; disk tier, larger batches, accumulation, resume ->
v0.72.2).

Final test count 16576 -> **16735** (+159 in `tests/test_v07200.py`), 324 test files.

---

## Citation check on the write-up (2026-07-27)

The gate numbers above are ours and were never in doubt. The *third-party* numbers drafted
around them for the accompanying report were checked against source and
**three were wrong**: MegaTrain listed as a 24 GB/3090 system (it is a single H200 with 1.5 TB
host RAM, up to 120B); a "26% GPU utilisation on a 4090" figure attributed to ZeRO-Infinity
(which predates the 4090 — not their measurement); and LSP-Offload's overhead given as 1.45×
with 1.31× separately credited to gradient checkpointing (the real published figure is a 31%
slowdown for LSP-Offload itself — one wrong number, re-used under a second wrong label).

Consequence for the claim: "every published system floors at 24 GB" was **false** —
LSP-Offload (arXiv:2406.10181) already fine-tunes 1.3B on a 4 GB laptop GPU. The defensible
headline is **1.3B -> 3B on a 4 GB card**, with bit-exactness verified, not "nobody has done
this."
