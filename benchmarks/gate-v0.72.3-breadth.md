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

# v0.72.3 "Breadth" — per-item gate results

Inherited standard — **a streamed run must be
bit-exact against the resident run of the same numerics**; what changes per item is
the *reference*, not the standard. Gradient accumulation is the one item whose gate
is a **measured I/O cost**, not an equality (the brief says so explicitly).

Box: Windows 11 · RTX 3050 Laptop 4.29 GB · 16.9 GB RAM · NVMe · Python 3.10 ·
torch 2.5.1+cu121 · transformers 4.57.6 · peft 0.18.1 · trl 0.19.1 · bitsandbytes 0.49.2.

Gates are throwaway scripts in the session scratchpad; nothing under `src/` was
written before a gate passed.

---

## GATE 1 — multi-arch (Mistral / Gemma / Phi) — **PASS 14/14** (2026-07-28)

The brief: *"wrong layer detection streams weights into the wrong module and trains
something plausible-but-wrong. Gate = bit-exactness on a small model of each family
added, not just the allowlist entry."*

**Method.** Tiny (3-layer, hidden 32, vocab 64) from-config checkpoints of each
family, written to disk as real `config.json` + `model.safetensors`, then:
shard → `build_streamed_model` → compare logits against **the same checkpoint loaded
resident with the same adapter weights**. CPU, float32. For `quant=nf4` the reference
is a **resident NF4 load**, never a resident bf16 one — a bf16 reference differs by
quantisation error and would hide a real bug inside it.

From-config rather than downloaded checkpoints because the module tree comes from the
same `modeling_*.py`, so layer detection, name mapping and `replace_with_bnb_linear`
conversion are exercised identically — and it stays runnable on a 4 GB box. The risk
the brief names is a *naming* property, not a size property.

**Vacuity defence.** PEFT initialises `lora_B = 0`, so a completely detached adapter
is byte-identical to a fresh one and every parity assertion passes for the wrong
reason. Every run randomises `lora_B` first and asserts a non-zero number of tensors
were copied to the reference.

Checks per family: (1) logits bit-exact `torch.equal`; (2) layer-0 LoRA gradient
non-zero (plan P2 — a severed graph still lowers loss, so nothing else catches it);
(3) decoder parameters still on `meta` (or it is not streaming at all).

| family | quant | max abs logit diff | bit-exact | layer-0 LoRA grad | meta decoder params |
|---|---|---|---|---|---|
| llama *(control)* | none | 0.000e+00 | yes | 3.2391e+00 | 27 |
| llama *(control)* | nf4  | 0.000e+00 | yes | 3.3290e+00 | 27 |
| **mistral** | none | 0.000e+00 | yes | 3.2391e+00 | 27 |
| **mistral** | nf4  | 0.000e+00 | yes | 3.3290e+00 | 27 |
| **gemma** | none | 0.000e+00 | yes | 5.1453e-01 | 27 |
| **gemma** | nf4  | 0.000e+00 | yes | 5.2169e-01 | 27 |
| **gemma2** | none | 0.000e+00 | yes | 3.8365e+00 | 33 |
| **gemma2** | nf4  | 0.000e+00 | yes | 4.0558e+00 | 33 |
| **gemma3_text** | none | 0.000e+00 | yes | 4.3802e+00 | 39 |
| **gemma3_text** | nf4  | 0.000e+00 | yes | 4.5446e+00 | 39 |
| **phi** | none | 0.000e+00 | yes | 2.9908e+00 | 42 |
| **phi** | nf4  | 0.000e+00 | yes | 3.0868e+00 | 42 |
| **phi3** | none | 0.000e+00 | yes | 3.2312e+00 | 18 |
| **phi3** | nf4  | 0.000e+00 | yes | 2.9316e+00 | 18 |

**Notes.**

- **`phi3` is the interesting row.** Phi-3 *fuses* Q/K/V into a single `qkv_proj` and
  gate/up into `gate_up_proj`, so it has no `q_proj`/`v_proj` at all — the LoRA target
  set had to be `["qkv_proj"]`. It is bit-exact anyway, which is the strongest
  evidence that layer detection and the shard name map are driven by the real module
  tree rather than by Llama-shaped assumptions.
- **`gemma3` (multimodal) is deliberately NOT added** — only `gemma3_text`. A real
  `google/gemma-3-*-it` reports `model_type='gemma3'` for the vision-capable wrapper;
  refusing it is correct, because streaming a multimodal wrapper as if it were a
  causal LM is exactly the silent-mis-train the allowlist exists to prevent.
- Two initial failures were **the gate harness's own fault, not the feature's**:
  `Phi3Config` defaults `pad_token_id=32000`, out of range for a 64-token fixture
  vocab. Fixed in the fixture (`pad_token_id=0`) and re-run; recorded here rather than
  quietly dropped.
- `meta decoder params` differs per family because the families have different
  per-layer parameter counts (gemma2/gemma3 carry extra norms; phi3 fuses, so fewer
  and larger tensors). All are `> 0`, which is the property being asserted.

**Verdict: the allowlist may be extended to `mistral`, `gemma`, `gemma2`,
`gemma3_text`, `phi`, `phi3`.**

---

## GATE 2 — pre-flight VRAM budget (batch- and vocab-aware) — **PASS** (2026-07-28)

The brief: *"Untied embed + lm_head — 8B has them untied, so two large matrices go
resident. Budget them in `estimate_stream_vram` or the 4 GB card OOMs on a model the
planner said would fit."* Operator addition: **batch scales the intra-layer transient
AND the logits tensor** (`batch × seq × vocab` plus its fp32 upcast), and on a
large-vocab model that term is larger than both layer buffers combined.

**Method — measure first, fit second.** Real streamed forward+backward+step on real
cached checkpoints over a (model, batch, seq) grid, recording
`torch.cuda.max_memory_allocated()`. The two models were chosen for a **3.1× vocab
contrast** (49 152 vs 151 936) so the logits term could be separated from everything
else rather than assumed. Nothing was implemented before this ran.

Card RTX 3050 Laptop 4.29 GB · SM clock 952 MHz (one row throttled to 442 MHz, noted
below) · free RAM 0.30–1.21 GB across the grid · pageable store (pinning is a *speed*
property and does not move VRAM).

Solving the two models simultaneously for the per-token cost yields:

```
peak = pool + extras + adapter_params·16 + 13.5 MB
     + batch · seq · ( 14·vocab  +  2·n_layers·hidden  +  4·(hidden + intermediate) )
```

- `14·vocab` — bf16 logits (2) + fp32 upcast (4) + fp32 log-softmax (4) + fp32 grad
  (4), i.e. exactly what `transformers` `ForCausalLMLoss` holds live. **The shipped
  `estimate_logits_bytes` used 6** (bf16 + upcast only), a first-principles guess that
  under-predicts this term by 2.33×.
- `2·n_layers·hidden` — the `checkpoint(use_reentrant=False)` boundary save, one bf16
  copy per layer.
- `4·(hidden + intermediate)` — the live transient inside the ONE layer being
  recomputed (independent of `n_layers`, which is the whole point of streaming).

| model | B | S | measured GB | predicted GB | err |
|---|---|---|---|---|---|
| SmolLM2-135M | 1 | 256 | 0.285 | 0.286 | +0.58% |
| SmolLM2-135M | 1 | 512 | 0.471 | 0.473 | +0.53% |
| SmolLM2-135M | 2 | 512 | 0.843 | 0.848 | +0.59% |
| SmolLM2-135M | 4 | 512 | 1.584 | 1.596 | +0.75% |
| SmolLM2-135M | 8 | 512 | 3.069 | 3.094 | +0.82% |
| Qwen2.5-0.5B | 1 | 256 | 0.920 | 0.924 | +0.47% |
| Qwen2.5-0.5B | 1 | 512 | 1.476 | 1.486 | +0.62% |
| Qwen2.5-0.5B | 2 | 512 | 2.592 | 2.609 | +0.63% |
| Qwen2.5-0.5B | 4 | 512 | 4.819 | 4.854 | +0.73% |
| Qwen2.5-0.5B | 8 | 512 | 9.267 | 9.346 | +0.85% |

**Worst absolute error 0.85%. Worst under-prediction: none — every prediction is ≥ the
measurement**, which is the only safe direction for a gate that must refuse configs.

**Independent check, nothing fitted to it:** the published v0.72.2 Llama-3.1-8B NF4
row — different model, different quantisation, different session, untied embeddings —
predicts **3.57 GB against a measured 3.32 GB (+7.5%, over)**. The formula was fitted
on two models 16–60× smaller and still brackets an 8B NF4 run on the safe side.

### Two findings that change the implementation

1. **The logits term dominates, exactly as called.** At Qwen2.5-0.5B B=8 S=512 it is
   **8.71 GB of the 9.35 GB predicted — 146× the entire buffer pool (0.060 GB)**. A
   pre-flight that budgeted only weights and buffers would green-light this config.
   Batch budgeting is therefore not a refinement of the estimator; it is the estimator.

2. **Windows/WDDM does not OOM — it spills, silently.** The B=8 row allocated
   **9.27 GB on a 4.29 GB card and raised nothing**, finishing all three steps. So on
   this platform *"it did not crash"* is **not** evidence that a config fits, and the
   fit/no-fit direction of the estimator **cannot be validated by observing an OOM
   here**. What is validated is the *demand* prediction (±0.85%); refusing when demand
   exceeds the card is then policy applied to a measured quantity, and is documented as
   such rather than claimed to be empirically OOM-verified on this box. This is the
   same WDDM shared-memory spill that invalidated the v0.72.0 1.5B "resident baseline".

One row (Qwen B=4) was taken while the SM clock had dropped to 442 MHz from 952 MHz.
Clock does not affect an allocation measurement, so the row stands; it is recorded
because the throughput items must not mix clock states.

### Reserve for the fit decision — also measured

The plan's `DEFAULT_WORKSPACE_BYTES` of 1 GB is **not** usable as the reserve: charged on
top of a 3.57 GB prediction it would refuse the 8B NF4 run that this feature exists to
enable and that v0.72.2 actually measured. Measured on this box instead:

```
before CUDA init : 3.460 GB free of 4.294 GB
after  CUDA init : 3.447 GB free   -> context + driver + display = 0.847 GB
1.5 GB tensor    : allocator 1.500 GB, driver-visible 2.349 GB
                   -> overhead beyond `allocated` = 0.849 GB (stable)
```

So the fit budget is read from `torch.cuda.mem_get_info()` **at pre-flight time** rather
than hardcoded — the offset includes desktop/display usage and is therefore a property of
the machine, not of the card model. Against this box's 3.445 GB of allocator-visible VRAM
the decision is consistent with every measured row: 8B NF4 (3.32 GB) fits, Qwen B=4
(4.82 GB) does not.

### Item 2 — implemented and verified end-to-end

Real `soup train --stream-layers` on SmolLM2-135M, **batch 2** (the first batch > 1 ever
run under streaming): completed exit 0, 18 steps, adapter written, and the saved adapter
carries **0 `.inner.` keys with 60 non-zero `lora_B` tensors** — v0.72.1's canonical-key
property survives batch > 1.

Pre-flight panel from that run:

```
peak VRAM    ~0.48 GB at batch 2 x seq 256 (logits 0.35 GB)
free VRAM    3.46 GB
forecast     5685-8361 tok/s — a compute-bound bound, not a promise
             (from 6.75 TFLOPS measured on this card now @ 862 MHz)
```

The 6.75 TFLOPS @ 862 MHz agrees with the box's independently recorded ~5.9–6.7 TFLOPS at
a pinned 862 MHz, i.e. the runtime probe reproduces the known ceiling rather than
inventing one.

The refusal direction, same model at batch 64:

```
a streaming step is predicted to need 12.08 GB of VRAM but only 3.46 GB is free.
Streaming bounds the WEIGHTS, not the activations or the logits — lower
training.batch_size or data.max_length, both of which scale this linearly.
```

`batch_size: "auto"` remains refused, for a reason that does not expire: it resolves by
OOM-probing a **resident** model, which a streaming run never loads.

---

## GATE 3 — gradient accumulation — **PASS** (2026-07-28)

The brief: *"I/O multiplies linearly: every micro-batch re-reads the entire model (plan
P9). This is the one place where the 'batch is nearly free' property breaks. **Measure**
tok/s at accum 1 / 2 / 4 and publish it; do not assume it is cheap."*

### Part A — correctness (CPU, float32, tiny Llama)

The inherited standard. Accumulated **adapter gradients** from a streamed run must be
bit-exact against a resident run accumulating the same micro-batches — the prefetcher
re-primes on every forward, and if it mis-tracked direction across a micro-batch boundary
a stale buffer would produce quietly wrong gradients rather than a crash.

| accum | grad tensors | max abs grad diff | bit-exact | layer-0 non-zero | prefetch primes |
|---|---|---|---|---|---|
| 2 | 12 | **0.0** | yes | 4 | 2 |
| 4 | 12 | **0.0** | yes | 4 | 4 |

Per-micro-batch losses are identical to 6 dp on both sides. Primes == accum, i.e. exactly
one prime per micro-batch forward, which is the intended scheduling.

### Part B — the measured I/O cost (CUDA)

Qwen2.5-0.5B bf16 · S=256 · 50 steps after 10 warm-up · **store 0.72 GB PINNED in every
row** (the confound that mattered — pinned vs pageable — is held constant throughout) ·
one session, back-to-back.

| batch | accum | eff. batch | **tok/s** | s / opt-step | layer loads / 1k tok | peak VRAM | free RAM | SM clock |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 556.6 | 0.460 | 175.78 | 0.842 GB | 2.56 GB | 862 MHz |
| 1 | 2 | 2 | 543.4 | 0.942 | 175.78 | 0.846 GB | 3.33 GB | 952 MHz |
| 1 | 4 | 4 | 540.1 | 1.896 | 175.78 | 0.846 GB | 3.16 GB | 952 MHz |
| 2 | 1 | 2 | 1069.1 | 0.479 | 87.89 | 1.320 GB | 3.20 GB | 952 MHz |
| 4 | 1 | 4 | **1378.0** | 0.743 | 43.95 | 2.280 GB | 2.66 GB | 960 MHz |
| 2 | 2 | 4 | 1094.5 | 0.936 | 87.89 | 1.325 GB | 2.31 GB | 960 MHz |

**Confirmation pass.** The first row was taken at 862 MHz and the rest at 952–960, a ~10%
spread, so the three rows the headline rests on were re-measured **interleaved**
(A/B/C/A/B/C, so monotonic drift cannot favour one arm):

| config | repeat 1 | repeat 2 | spread |
|---|---|---|---|
| batch 1 / accum 1 | 552.8 | 550.9 | 0.4% |
| batch 1 / accum 4 | 551.4 | 553.2 | 0.3% |
| batch 4 / accum 1 | 1393.4 | 1393.4 | 0.0% |

### What the numbers actually say

1. **Accumulation is per-token I/O-neutral, not linear-cost.** `layer loads / 1k tokens`
   is **constant at 175.78 across accum 1, 2 and 4**, and tok/s is flat within 3%
   (556.6 → 543.4 → 540.1; 552.8 vs 551.4 on the interleaved repeat). accum=N does N
   micro-batches, N model reads and N times the tokens — the ratio does not move. The
   plan's P9 wording ("multiplies IO linearly") is right per *optimizer step* and
   misleading per *token*, which is the unit that decides wall-clock.

2. **The real cost is opportunity cost against raising batch.** At the *same effective
   batch of 4*: batch 4 / accum 1 delivers **1393.4 tok/s** against batch 1 / accum 4 at
   **553.2 — a measured 2.52×**, because one weight read is amortised over four times the
   tokens (43.95 vs 175.78 loads per 1k tokens). Publishing only the accum column would
   have read as "accumulation is free"; it is free *per token* and expensive *per unit of
   effective batch*.

3. **Accumulation's actual value under streaming is that it buys effective batch at
   constant VRAM**: peak moved 0.842 → 0.846 GB across accum 1→4, while reaching the same
   effective batch by raising batch cost 0.842 → 2.28 GB. So the guidance the pre-flight
   should give is: **raise `batch_size` until the VRAM pre-flight refuses, then use
   accumulation for the rest** — which is exactly the pairing this release now makes
   possible, since item 2 is what tells the user where that ceiling is.

4. Batch scaling is real but sub-linear: 556.6 → 1069.1 (1.92×) → 1378.0 (2.48×) for
   batch 1 → 2 → 4, as the run crosses from I/O-bound into compute-bound.

### A defect in item 2's own GEMM probe, found by item 3's smoke

The accumulation smoke printed **3.54 TFLOPS @ 862 MHz** where an earlier run had printed
**6.75 TFLOPS @ 862 MHz** — a 2× swing at the same *reported* clock, which would have made
the forecast bracket meaningless. Diagnosed rather than accepted:

| probe size | five repeats (TFLOPS) | spread |
|---|---|---|
| 2048³ | 3.19, 3.50, 3.89, 3.85, 4.41 | **38%** |
| 4096³ | 3.92, 3.97, 3.99, 3.81, 4.16 | 9% |

At 2048 the sample is too short for the boost clock to engage and the repeats ramp
**monotonically upward**, so whichever repeat happens to be first sets the answer. Fixed
by moving to 4096³ and taking the **best of 3** repeats. Best-of-N is not cherry-picking
here: a *ceiling* has one-sided noise, since contention, a cold clock and thermal
throttling can only ever make an achievable rate look slower than it truly is.

This is also the concrete justification for refusing to compile a per-card TFLOPS constant
into the source — the same card, at the same reported clock, differed 2× between sessions.

### Verified end-to-end

Real `soup train --stream-layers` on SmolLM2-135M at **batch 2 × accum 4** (effective
batch 8): completed exit 0, 5 optimizer steps, adapter written with **0 `.inner.` keys**.
The pre-flight printed the measured advisory:

```
! accumulating 4x at batch 2: the base is re-read once per micro-batch. Per token
  that is free, but reaching effective batch 8 by raising training.batch_size
  instead measured ~2.5x faster. Accumulation holds peak VRAM flat, so raise
  batch_size while the budget above allows, then accumulate for the rest.
```

---

## GATE 4 — checkpoint / resume — **PASS** (2026-07-28)

The brief: *"silent failure is resuming against a stale shard cache or a mismatched
optimizer state. Gate = save mid-run, resume, assert the loss curve CONTINUES rather than
restarts, and assert the shard fingerprint is re-verified on resume."*

### The failure, established before anything was changed

v0.72.1 fixed the **save** direction only, deliberately — `state_dict()` delegates at the
wrapper's own prefix while `named_parameters()` still carries `.inner.`. The load
direction was therefore still broken, and measurably so:

| check | before |
|---|---|
| saved tensors / of which `.inner.` | 12 / **0** (save side correct) |
| adapter tensors in model / **landed** | 12 / **0** |
| non-zero `lora_B` in the checkpoint | 6 (so the comparison is not vacuous) |
| **resumed losses vs from-scratch losses** | **byte-identical** |
| shard fingerprint detects a changed source | already true (v0.72.0) |

The mechanism: `nn.Module.load_state_dict` narrows the dict **by child name** as it
descends. The wrapper's only child is `inner`, so a canonical
`...layers.0.self_attn.q_proj.lora_A.weight` matches no child prefix and is dropped.
PEFT reported the keys as missing via a `UserWarning` and continued.

### The fix

A `_register_load_state_dict_pre_hook` on `StreamedDecoderLayer` that injects
`.inner.`-prefixed copies into the (already prefix-narrowed) state dict. The pre-hook runs
at the start of the wrapper's own `_load_from_state_dict`, and torch's child loop reads
that same dict object afterwards — so the redirected keys are visible when it descends
into `inner`. Load-side only: it redirects keys rather than re-parenting the module tree,
so the forward path is untouched and **v0.72.0's bit-exactness gates stay valid without
being re-run** — the same reasoning that justified the v0.72.1 approach.

### After, on the production CUDA path

Driven through the real `PeftModel.load_adapter`, which is what
`Trainer._load_from_checkpoint` calls for a PEFT model:

| check | after |
|---|---|
| tensors landed | **12 / 12** |
| `hf_device_map` preserved | **true** (`{'': 0}` unchanged) |
| decoder params still on `meta` | 27 (still streaming, not materialised) |
| resumed vs from-scratch loss curve | **differ** — the checkpoint contributes |

`hf_device_map` is checked explicitly because `install_streaming` sets it so
`Trainer._move_model_to_device` skips `.to()` on meta weights; a loader that rewrites it
would reintroduce v0.72.0's CRITICAL.

### Two honest limitations found while gating

1. **PEFT re-dispatches a CPU-built streamed model.** On a CPU build, `load_adapter` moved
   every parameter to `cuda:0` and rewrote `hf_device_map` from `{'': 'cpu'}` to
   `{'base_model': 0}`, breaking the model. It fires only when the device map mentions
   `"cpu"`, which the production path never does — streaming exists to bound VRAM, so a
   CPU-built streamed model is a test convenience, not a configuration. The CPU tests
   therefore exercise the redirection mechanism through `set_peft_model_state_dict` (what
   `load_adapter` calls internally to place weights), and the full `load_adapter` path is
   covered by a CUDA-gated test.

2. **End-to-end `soup train --resume` cannot be demonstrated on this box, for a reason
   that is not streaming's.** `transformers.check_torch_load_is_safe()` raises unless
   torch ≥ 2.6 (CVE-2025-32434); this box has torch 2.5.1+cu121, so *every* resume fails
   here. Proven with a **control**: the identical config with `stream_layers: false`
   produces the identical error. What is verified is the streaming-specific half — the
   adapter round-trip and loss continuity — at the library level on CUDA, plus a
   behavioural CLI test that both `--resume` and `--hf-resume` are no longer refused.

---

## GATE 5 — disk-kind detection (the `soup doctor` rider) — **PASS** (2026-07-28)

v0.72.0 shipped a `choose_tier` that refuses anything but NVMe — wired to a **hardcoded
`disk_kind="nvme"`** in the trainer. A guard connected to a constant can never fire, so
the refusal existed only on paper.

Probed on this box: `Get-PhysicalDisk` returns
`{"MediaType":"SSD","BusType":"NVMe","Size":512110190592}` — correctly identifying the
512 GB NVMe. Two things fell out of the measurement:

1. **`BusType` must beat `MediaType`.** An NVMe drive reports `MediaType: SSD`; keying on
   media type alone would classify every NVMe disk as SATA SSD and refuse the disk tier
   universally.
2. **The probe costs 9.04 s cold, ~2.4 s warm** (PowerShell + CIM startup; Windows caches
   it after the first launch). Too slow to run unconditionally, and it shapes the API:
   `choose_tier` now takes a **callable**, so the probe runs only when the base does *not*
   fit in RAM — the case where the answer matters. The result is cached per volume per
   process. Linux reads `/sys/block/*/queue/rotational` instantly; unknown platforms
   return `"unknown"`, which `choose_tier` refuses (the safe direction — believing a
   spinning disk is NVMe costs hours of thrashing, plan P11).

`soup doctor --disk` reports it:
`Disk type │ NVMe — layer streaming can use the disk overflow tier`.

**Opt-in, following the command's own `--nccl` convention.** Measured: `soup doctor`
17.56 s by default, 19.97 s with `--disk`. A first reading of this over-claimed — 9 s was
subtracted from the total to infer "roughly doubled", which the direct A/B does not
support; the true warm cost is ~2.4 s on a command that already takes ~17.5 s (it imports
torch, transformers and friends to build the dependency table). The flag is still the right
call, because the cost is paid by every user while only streaming users benefit, and a
streaming run probes lazily on its own regardless.

---

## GATE 6 — the disk overflow tier — **PASS** (2026-07-28)

**The limitation, up front rather than at the end: this box cannot measure the RAM-vs-disk
performance gap, and this release does not claim one.** Two independent reasons —
safetensors memory-maps the shards, so the OS page cache keeps them resident between steps
on any machine with spare RAM; and at ~5 effective TFLOPS the NVMe read hides under compute
anyway (plan §2.3). What is gated here is correctness, which *is* demonstrable.

The brief asks for "the same four checks with `DiskSource` substituted". The strongest
available reference is the **RAM tier**, not the resident model: both stream through the
same buffer pool, the same prefetcher and the same layer wrapper, differing only in where
`get(idx, name)` reads from — so any difference is attributable to the source. The
resident comparison is kept as well, anchoring the tier to ground truth rather than merely
to its sibling.

| check | result |
|---|---|
| disk vs **RAM tier** logits | **0.0 — bit-exact** |
| disk vs **resident** logits | **0.0 — bit-exact** |
| layer-0 LoRA gradient | 2.6260 (non-zero, plan P2) |
| determinism (rebuild + re-run) | bit-exact |
| `store_bytes` (held resident) | **0** — the point of the tier |
| `disk_bytes` vs RAM tier `store_bytes` | 148 480 == 148 480 (accounting agrees) |

**Behaviour change this creates.** `stream_source: auto` (the default) now falls back to
disk instead of refusing when the base will not fit in RAM. That is what the plan's tier
order specifies, but a *silent* fallback to a slower path is the exact failure mode this
project criticises elsewhere, so the pre-flight note says what happened, that nothing is
held resident, **that the slowdown is unmeasured**, and that `stream_source: 'ram'` is how
an operator asks to be refused instead. A non-NVMe disk is still refused outright.

Consequence for the v0.72.2 tests: "does not fit in RAM" used to mean an exception and now
means a tier decision. Those controls were **restored intact under `stream_source: 'ram'`**
— the regime where the early size probe is the thing under test — rather than weakened, and
the new tier behaviour is pinned separately in `tests/test_v07203.py` through the real
trainer.
