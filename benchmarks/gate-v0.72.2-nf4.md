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

# NF4 Layer Streaming — Gate results (now the **v0.72.2** slot)

> **Filename kept as `v0721-*` deliberately.** These gates were run while NF4 was
> v0.72.1. They surfaced a shipped v0.72.0 correctness defect (adapters saved with an
> `.inner.` key prefix reload as zero tensors), which was split out to ship alone and
> first as **v0.72.1**; NF4 moved to **v0.72.2**. **These results stand as-is and are not
> to be re-run** — see plan §7.1. The adapter-key defect is written up below because
> this gate is what found it.

**Status: GATE 1 PASS (6/6, bit-exact vs resident NF4). GATE 2 complete, incl. the 8B headline.**
Nothing under `src/` was written before these ran (plan §7.1 rule).

Hardware (all numbers on this box, no extrapolation):
**Windows 11 · RTX 3050 Laptop 4 GB (CC 8.6) · 16.9 GB RAM · NVMe**
torch 2.5.1+cu121 · bitsandbytes 0.49.2 · transformers 4.57.6 · peft 0.18.1 ·
trl 0.19.1 · accelerate 1.12.0 · Python 3.10.8

Unit convention: **decimal GB** throughout (matches the v0.72.0 gate doc).

The reference is **resident NF4**, per plan §7.1 — not resident bf16, which would
differ by quantisation error and hide a real bug inside it. Streamed NF4 and
resident NF4 hold the *same quantised bytes* and run the *same bitsandbytes
kernels*, so the standard is bit-exactness.

---

## GATE 1 — correctness — **PASS** (2026-07-27)

Throwaway spike (`scratchpad/v0721_gate1.py`), nothing under `src/`.
Model **HuggingFaceTB/SmolLM2-135M** (llama arch, 30 layers, tied embeddings),
NF4 + double quant + bf16 compute — i.e. the repo's own `quant_menu` defaults —
LoRA r=16 on `q_proj,v_proj`, seq 512, batch 1, double buffering (n=2) with a
dedicated prefetch stream.

Construction under test (the v0.72.0 engine, plus the P3 work):
- checkpoint **pre-quantised offline**, one tensor at a time on the GPU, into
  per-layer shards of packed `uint8` + `absmax` (+ nested `absmax`/`offset`)
- `Params4bit` **views rebuilt over the pooled buffers on every call**, with a
  `QuantState` reassembled from the streamed tensors + two shared constant code
  tables
- skeleton on `meta` via `init_empty_weights` + `replace_with_bnb_linear`;
  **270 decoder weight tensors never leave `meta`**
- `functional_call` feeds the rebuilt `Params4bit` into the unmodified HF layer
- `checkpoint(use_reentrant=False)`; base `requires_grad=False`, no `detach()`

| # | Check | Threshold | Measured | Verdict |
|---|---|---|---|---|
| 0 | offline shard bytes vs the resident model's `Params4bit` | identical | **IDENTICAL** (7 weights on layer 0: packed + absmax + nested absmax) | PASS |
| 1 | streamed vs resident NF4 logits, max abs diff | < 1e-3 | **0.0 (bit-exact)** | PASS |
| 2 | LoRA grad on **layer 0** | != 0 | **4.921e-01**; 30/30 layers non-zero | PASS |
| 3 | 25 steps streamed vs resident, loss curve | within noise | **max rel 0.0**; 12.99325 -> 12.09359 on both | PASS |
| 4 | same seed twice -> identical loss | identical | **0.0** | PASS |
| 5 | buffer-count invariance n=2 vs n=3 | identical | **0.0** | PASS |

RAM store 0.055 GB pinned; pool 3.7 MB x 2.

### P3 is solved, and the probe that settles it

`Params4bit` carries a `quant_state` and quantises on transfer to CUDA, so NF4
weights cannot be byte-copied into a plain buffer (plan P3). Measured answers
(`scratchpad/probe_nf4_api.py`):

- `quantize_4bit` is **deterministic and byte-identical** across repeats and
  across a CPU round-trip — so offline sharding reproduces exactly what a
  resident load would have produced. Check 0 confirms this end-to-end.
- A `Params4bit` **rebuilt over an existing packed buffer** with an explicit
  `quant_state` (`bnb_quantized=True`) produces a **bit-identical** forward.
- `torch.func.functional_call` accepts it, **including over a `meta`
  placeholder** of a different shape — so the v0.72.0 substitution mechanism
  carries over unchanged.
- Gradient flows to the layer input through `matmul_4bit` with the weight at
  `requires_grad=False` — plan P2 holds.

Per-weight streamed bytes (double quant): packed `N/2` + absmax `N/64` +
nested absmax `N/4096` + a 4-byte offset ≈ **0.516 bytes/param**. The NF4 code
table (16 fp32) and the nested code table (256 fp32) were verified **constant
across every weight**, so one shared resident copy is safe — the sharder
asserts this rather than assuming it.

### What the gate caught that a green run would not have

**1. PEFT silently used a different LoRA math path (found by check 1).**
With byte-identical adapters and byte-identical weights, streamed vs resident
logits differed by **9.375e-01**. Cause: `transformers.from_pretrained` stamps
`is_loaded_in_4bit` on the model, and PEFT reads it to dispatch
`lora.bnb.Linear4bit`. A `meta` skeleton has no such marker, so PEFT dispatched
the generic `lora.layer.Linear`, which still *runs* against a `Linear4bit` base
— it just casts and accumulates differently. No crash, no warning; the loss
curve looked healthy. Fixed by stamping the markers `from_pretrained` sets.
This is the single most important finding of the gate.

**2. Two ways the gate itself was nearly vacuous** (harness bugs, fixed before
the numbers above were taken):
- PEFT initialises `lora_B = 0`, so at step 0 the adapter contributes nothing
  and `dL/dA` is structurally zero for *any* correct implementation. The first
  run "passed" check 1 bit-exactly while the adapter path was doing nothing at
  all, and "failed" check 2 for a reason that had no bearing on streaming. The
  gate now randomises `lora_B` first, which makes the adapter path load-bearing
  in checks 1 and 3.
- The adapter sync between the two models silently copied **0 tensors** because
  of the key-prefix bug below, leaving the models with different adapters. It
  now refuses to run rather than compare two differently-initialised models.

---

## Blocking rider found in shipped v0.72.0 — adapters are saved unloadable

Reproduced through the **shipped** `build_streamed_model`
(`scratchpad/check_inner_keys.py`, CPU, tiny-random-Llama):

`install_streaming` replaces `layers[i]` with a wrapper that holds the real
layer as a child named `inner`. Every adapter parameter therefore gains an
`.inner.` segment in its state-dict path, and that name is what
`get_peft_model_state_dict` -> `save_pretrained` writes to disk:

```
base_model.model.model.layers.0.inner.self_attn.q_proj.lora_A.weight
                               ^^^^^^
```

Loading that adapter back into a normal model drops **8 of 8** tensors. PEFT
emits a `UserWarning` about missing keys and returns a model that is byte-for-byte
the untuned base:

```
reload into a plain model: 0/4 lora_B tensors non-zero
VERDICT: ADAPTER SILENTLY LOST ON RELOAD
```

Consequence: **every adapter produced by `soup train --stream-layers` in
v0.72.0 is inert outside the streaming path** — `soup merge`, `soup serve`,
`soup chat` and `PeftModel.from_pretrained` all load it as a no-op. The
training run itself is correct (the gates prove that); only the artifact it
writes is unusable. Nothing in the 159 v0.72.0 tests saves an adapter and loads
it back.

This must be fixed in v0.72.1 — it is in the same component the NF4 work
modifies, and shipping NF4 on top of an unloadable artifact would compound it.

**Both candidate fixes were spiked and both work** (`scratchpad/fix_inner_keys_spike.py`,
CPU, tiny-random-Llama; control = shipped code):

| Variant | saved keys with `.inner.` | reload into a plain model |
|---|---|---|
| shipped v0.72.0 (control) | 8 / 8 | **0/4 adapters — LOST** |
| A: transparent `state_dict()` on the wrapper | 0 | 4/4 — round-trips |
| B: wrapper shares the inner layer's `_parameters`/`_modules` | 0 | 4/4 — round-trips |

A is surgical (serialisation only); B removes the nesting level altogether, so
`named_parameters()` is canonical too — which also unblocks loading *into* a
streamed model later (`--resume`, a v0.72.2 item). Both keep the forward
working. Choice to be made at implementation, TDD-first, with the reload
round-trip as the spec.

---

## GATE 2 — the number

Protocol unchanged from v0.72.0 so the rows are comparable: tok/s over 50 steps
after 10 warm-up · `torch.cuda.max_memory_allocated()` · SM occupancy from
`nvidia-smi dmon -s u` · `PagedAdamW8bit` · batch 1 · double buffering · TFLOPS
back-check at C=6 against this box's **measured** 6.76 TFLOPS dense bf16 GEMM
ceiling (theoretical GA107 peak ~24.5; the plan's "9" is wrong for this card).

Parameter counts are read from the safetensors headers, so a tied `lm_head` is
not double-counted (counting `model.parameters()` on the meta skeleton reported
Qwen2.5-3B as 3.40 B instead of 3.09 B and inflated TFLOPS accordingly).

| Model | Quant | Store | S | tok/s | GPU util | Peak VRAM | TFLOPS eff |
|---|---|---|---|---|---|---|---|
| **Llama-3.1-8B** | **NF4** | **3.60 GB pinned** | 512 | **122.5** | **98.9%** / 100% | **3.45 GB** | 5.90 |
| Qwen2.5-3B | **NF4** | 1.43 GB **pinned** | 512 | **244.3** | 94.5% / 100% | 1.91 GB | 4.53 |
| Qwen2.5-3B | bf16 *(v0.72.0)* | 5.55 GB pageable | 512 | 143.1 | 79.3% / 100% | 2.15 GB | 2.65 |

### The 8B headline

**Llama-3.1-8B fine-tunes on a 4 GB card at 122.5 tok/s**, with the base
page-locked (3.60 GB store, under the box's measured 7.65 GB pinned ceiling) and
peak VRAM 3.45 GB — genuinely inside the card, not a WDDM spill. Untied
`embed_tokens` + `lm_head` stay resident and bf16 (2.10 GB of that 3.45 GB), which
is why 8B is close to this card's ceiling; treating them as streamed large layers
is a v0.72.3 item. NF4 shard set on disk 5.70 GB, sharded once and cached.

Derived, labelled as arithmetic: 1M training tokens = **2.3 h at 8B**, 1.1 h at 3B.

### The TFLOPS back-check, resolved — the DENOMINATOR was wrong

The first pass read 5.90 TFLOPS = **87% of the v0.72.0 gate's 6.76 TFLOPS**
"achievable dense bf16 GEMM" figure, which is not credible for an eager loop
that also dequantises NF4. It was held back rather than published. Both
candidate explanations were then tested:

**(b) gradient checkpointing off, so C=6 is the wrong constant — DISCONFIRMED.**
The streamed layer wraps its body in `checkpoint(use_reentrant=False)` whenever
grad is enabled, and the measured loop is a forward+backward. C=6 is correct.

**(a) the 6.76 ceiling was measured at shapes that do not reach peak — PARTLY,
but the dominant effect turned out to be something else: GPU CLOCK STATE.**

An first re-measure suggested 6.76 was simply an under-measurement (square
shapes came back 7.08 / 7.52 / 7.66). A second re-measure in a later session
gave **6.23 / 6.63 / 6.75** for the *same* shapes — i.e. 6.76 reproduced almost
exactly. Repeating 4096^3 six times inside one session then showed the
measurement is stable to **<1%** (6.93, 6.94, 6.93, 6.94, 6.93, 6.94) with the
SM clock pinned at **862 MHz / 63 C**.

So the ~13% spread is *between* sessions and tracks the boost clock, not
measurement noise. **6.76 was never wrong; it was measured in a lower-clock
session.** The methodological rule that follows is stronger than the original
correction: **a ceiling is only comparable to a throughput measured in the same
session at the same clock**, and any fraction-of-ceiling should state the clock
it was taken at. Shape still matters, but less than clock.

Shape-matched ceilings, all measured in ONE session at 862 MHz, bf16, B=1/S=512:

| op | M x K x N | TFLOPS | FLOP share |
|---|---|---|---|
| q_proj / o_proj | 512 x 4096 x 4096 | 7.71 | 15.4% |
| k_proj / v_proj | 512 x 4096 x 1024 | 7.57 | 3.8% |
| gate/up_proj | 512 x 4096 x 14336 | 7.49 | 53.8% |
| down_proj | 512 x 14336 x 4096 | 7.69 | 26.9% |
| **FLOP-weighted** | | **7.58** | |

| model shapes | weighted ceiling @862 MHz |
|---|---|
| Llama-3.1-8B | 7.58 *(measured in the earlier high-clock session)* |
| Qwen2.5-3B | 6.67 |
| Qwen2.5-1.5B | 6.39 |
| Qwen2.5-0.5B | 5.90 |

Smaller models sit lower, as expected — their GEMMs are too small to saturate.

**The numerator convention, fixed and stated so a reader can reproduce it:**
decoder params at **C=6** (2 fwd + 2 recompute + 2 dL/dx). **Not C=8** — the
base is frozen, so no dL/dW is computed for it, and that missing term is exactly
what C=8 would add. `embed_tokens` contributes **0 FLOPs** (a lookup).
`lm_head` runs at **C=4** (2 fwd + 2 dL/dx): it sits outside the checkpointed
decoder layers so its forward is not recomputed, and it is frozen.

| run | GFLOP/token | effective | % of its shape-matched ceiling |
|---|---|---|---|
| Qwen2.5-0.5B @ 978.6 tok/s | 2.69 | 2.63 | 45% |
| Qwen2.5-1.5B @ 525.0 tok/s | 8.79 | 4.62 | **72%** |
| Qwen2.5-3B @ 143.1 tok/s | 17.89 | 2.56 | 38% |
| Llama-3.1-8B NF4 @ 122.5 tok/s | 43.98 | 5.39 | 71% |

The 1.5B row being the highest is exactly right: it is the one run measured at
96.8% SM occupancy, i.e. the compute-bound one. The 3B bf16 row at 38% is the
transfer-bound pageable-store run. The picture is now coherent across all four.

**Caveat that must travel with these fractions:** the throughputs come from
earlier sessions and the ceilings from this one, so each ratio inherits the
cross-session clock uncertainty (~13%). Before any of these fractions is
published as load-bearing, re-measure throughput and ceiling **back to back in
one session** and report the SM clock alongside.

**Publish tok/s and peak VRAM as the measurements — those are clock-sensitive
too but they are what was actually observed. The TFLOPS column is derived; quote
it only with the constant, the parameter accounting and the clock stated.**

**NF4 makes the 3B row 1.71x faster than bf16 streaming, and the reason is not
the arithmetic — it is that a 1.43 GB store fits under the page-locked ceiling
where a 5.55 GB one did not.** Pinning restores async `copy_`, which is visible
as utilisation 79.3% -> 94.5%. This is the v0.72.0 honesty caveat #1 ("the 3B
number is a LOWER BOUND because the store went pageable") being lifted by NF4.

### The resident-NF4 3B baseline was measured and then DISCARDED

First attempt: 34.1 tok/s, **peak VRAM 6.07 GB on a 4 GB card**, 100% util,
0.63 TFLOPS. A 6.07 GB peak on a 4 GB card is a WDDM shared-host-memory spill —
the identical failure mode that made v0.72.0 discard its Qwen2.5-1.5B resident
row. Publishing it would have produced a **"streaming is 7.2x faster than
resident"** headline out of a number that measures Windows paging, not
training. It is discarded, not reported.

It was also not a fair comparison: gradient checkpointing was off on the
resident side, so it ran at C=4 while the streamed side runs at C=6 *and* it
held every activation, which is what pushed it over the card. The harness now
enables checkpointing for resident throughput runs; a re-measurement is queued
behind the 8B headline.

**Consequence to state in the release:** as of this gate there is still **no
valid resident baseline at 3B on this box** — resident bf16 OOMs (v0.72.0) and
resident NF4 spills. The honest claim remains the one v0.72.0 made at 0.5B, not
a 3B speed-up ratio.

Remaining row pending: the 8B headline.

---

## v0.72.2 step-6 reproduction — through the SHIPPED code (2026-07-28)

The Gate-2 rows above came from the throwaway spike. These were re-measured with
`shard_checkpoint` / `build_streamed_model` as released, same protocol (50 steps
after 10 warm-up, batch 1, S=512, `PagedAdamW8bit`, double buffering), and the
GEMM ceiling taken **in the same session at the stated clock**, per the rule the
gate itself established.

| Model | Quant | Store | tok/s | GPU util | Peak VRAM | SM clock | eff TFLOPS | % of same-session ceiling |
|---|---|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | NF4 | **3.60 GB pinned** | **119.6** | 100% | **3.32 GB** | 952 MHz / 70 C | 5.26 | 68% |
| Qwen2.5-3B | NF4 | 1.43 GB pinned | 264.2 | 100% | 1.76 GB | 960 MHz / 72 C | 4.73 | 61% |

Agreement with the spike is close (8B: 119.6 vs 122.5 tok/s, 3.32 vs 3.45 GB
peak, identical 3.60 GB pinned store), i.e. the released implementation
reproduces the gate rather than merely resembling it. The residual is inside the
~13% between-session clock spread the gate documented.

Shard cache behaved: the second 8B run reported `shards ready in 0.0s`.

**The 3B NF4-vs-bf16 comparison, and its caveat.** 264.2 tok/s (NF4) against
v0.72.0's 143.1 tok/s (bf16) is 1.85x, but the cause is *not* arithmetic — it is
that a 1.43 GB store page-locks where a 5.55 GB one did not, which restores
async `copy_` and moves utilisation 79.3% -> 100%. The bf16 row was measured in
an earlier session at an unrecorded clock, so the ratio inherits that
uncertainty; the mechanism (pinning) is the load-bearing claim, not the factor.

Derived, labelled as arithmetic: 1M training tokens = **2.3 h at 8B**, 1.05 h at 3B.

**Also found by step 6, and fixed:** a streamed NF4 model reported
878,154,048 parameters for SmolLM2-135M (true 134,515,008) while the resident
NF4 path reported 134,975,808. PEFT computes `Params4bit` totals as
`numel * 2 * quant_storage.itemsize`, correct for a resident tensor whose numel
is the packed count but not for a `meta` placeholder carrying the logical shape.
Display-only; the loss curve was byte-identical before and after. At 8B it would
have printed ~52 B.
