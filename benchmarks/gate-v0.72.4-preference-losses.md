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

# v0.72.4 "Preference losses" — gate results

Box: Windows 11 · RTX 3050 Laptop 4.29 GB · 16.9 GB RAM · NVMe · Python 3.10.8 ·
torch 2.5.1+cu121 · transformers 4.57.6 · peft 0.18.1 · **trl 0.19.1**.

Gates are throwaway scripts in the session scratchpad; nothing under `src/` was
written before a gate passed.

> **Version caveat, stated up front.** CI resolves trl **0.29.1** / peft 0.20.0 /
> torch 2.13.0. Everything below is measured against trl **0.19.1**. The gate
> therefore also records *which TRL internals the property depends on*, so the
> shipped tests assert the property rather than the internals.
>
> **A dependency break found during this release, and two wrong diagnoses before
> the right one.** Six trainers (`bco`, `dpo`, `ipo`, `kto`, `orpo`, `simpo`) pass
> `max_prompt_length` to their trl config, and trl removed it in STAGES — which is
> why each spot-check gave a different answer. Read per config off the published
> wheels:
>
> | version | dpo | kto | orpo | cpo | bco |
> |---|---|---|---|---|---|
> | 0.24.0 | yes | yes | yes | yes | yes |
> | 0.25.1 | yes | yes | yes | yes | **NO** |
> | 0.26.0 | yes | **NO** | **NO** | **NO** | NO |
> | 0.29.0 | **NO** | NO | *gone* | *gone* | *gone* |
>
> First diagnosis: "trl 1.x removed them" — taken from an existing note in this
> repo claiming CI runs trl 1.9.2. Wrong; CI's install log shows **0.29.1**, so a
> `<1` cap excluded nothing. Second: "the break is 0.29.0" — right for `dpo`, but
> checked only `dpo_config.py` and extrapolated. The dependency is capped
> **`<0.25`**, the last release on which all six work.
>
> This is a pre-existing bug, not one this release introduced: the trl imports
> live inside `setup()`, which no test had ever called on those wrappers, so CI
> stayed green while `soup train --task orpo` was broken for anyone installing
> fresh. v0.72.4's end-to-end preference tests are what surfaced it.

> ### Correction, post-release: the table above is wrong, and so was the cap
>
> The paragraph above records three diagnoses. It should record four — the third
> was also wrong, and it is the one that shipped. Left in place above rather than
> edited, because the failure mode is the point.
>
> The method ("read per config off the published wheels") was right; the execution
> was not. At 0.25 `BCOConfig` **moved** to `trl/experimental/bco/`, and at 0.26
> `kto`/`orpo`/`cpo` followed. All of them stayed publicly re-exported from `trl`
> with `max_prompt_length` intact. Seeing a config file vanish from `trl/trainer/`
> was read as the field being removed. A relocation was scored as a deletion.
>
> Re-derived by parsing each `*Config` class's own annotated fields across every
> wheel from 0.24.0 to 0.29.1 (all five inherit from `TrainingArguments`, so there
> is no inherited-field escape hatch):
>
> | version | dpo | kto | orpo | cpo | bco |
> |---|---|---|---|---|---|
> | 0.24.0 – 0.26.2 | yes | yes | yes | yes | yes |
> | 0.27.0 – 0.27.2 | yes | **NO** | yes | yes | yes |
> | 0.28.0 | yes | NO | **NO** | **NO** | **NO** |
> | 0.29.0 – 0.29.1 | **NO** | NO | NO | NO | NO |
>
> Then settled the way a static read cannot settle it — by **constructing** all six
> configs with the exact keyword arguments the wrappers pass:
>
> ```
> trl 0.26.2   OK dpo · OK ipo · OK kto · OK orpo · OK simpo · OK bco
> trl 0.27.0   FAIL kto: KTOConfig.__init__() got an unexpected keyword
>                        argument 'max_prompt_length'
>              OK dpo · OK orpo          <- the control: the boundary is KTO's alone
> ```
>
> So the cap is **`<0.27`**, not `<0.25`; 0.25.0, 0.25.1, 0.26.0, 0.26.1 and 0.26.2
> were excluded for no reason. Two further corrections fall out of the same pass:
>
> - **`ORPOConfig` / `CPOConfig` are not "gone" at 0.29** in the sense the table
>   implies — the modules still exist under `trl/experimental/`. What changed is
>   that they, `BCOConfig` and their trainers are no longer exported from the `trl`
>   namespace, which is what Soup imports. The 0.29 break is an `ImportError` at
>   `from trl import ORPOConfig`, i.e. harder than a rejected keyword argument.
> - **The floor `>=0.7.0` was impossible** and nobody checked it while carefully
>   fixing the ceiling. `setup()` imports `GRPOTrainer` unconditionally, and trl
>   first exports it at **0.14.0** (`OnlineDPOTrainer` / `KTOTrainer` / `BCOTrainer`
>   / `BasePairwiseJudge` at 0.11.0; 0.7.0 exports none of them). Now `>=0.14.0`.
>
> The transferable lesson is narrower than "check your bounds": **a version bound
> derived by reading source is a hypothesis, and the experiment that tests it is
> constructing the object.** Both wrong answers here were produced by reading, and
> both would have been caught in one minute by building the six configs.

**The inherited standard** — a streamed run must be bit-exact against the
resident run of the same numerics; what changes per slot is the *reference*, not
the standard. This slot adds a second, independent standard, because bit-exactness
cannot see the failure that matters here:

> DPO needs a reference model. Implemented naively as a second model instance it
> doubles memory and defeats the entire feature. It must be **the same streamed
> base with adapters disabled** — one set of weights, one stream, no second pass.
> Gate = assert peak VRAM under DPO is within the activation delta of peak VRAM
> under SFT, **not ~2x**. A passing loss curve does not detect this; only the
> memory assertion does.

---

## Correction to the brief, established before gating

The brief says "ORPO / SimPO / KTO are reference-free and ride the engine
unchanged." **That is true for ORPO and SimPO and false for KTO.**

| loss | TRL trainer | reference model? | evidence |
|---|---|---|---|
| DPO | `DPOTrainer` | **yes** — implicit | `dpo_trainer.py:321-327`, `null_ref_context` at `:865-877` |
| KTO | `KTOTrainer` | **yes** — implicit | `kto_trainer.py:466-476`, `null_ref_context` at `:810`, used at `:926` |
| ORPO | `ORPOTrainer` | no | zero occurrences of `ref_model` in the file |
| SimPO | `CPOTrainer` | no | zero occurrences of `ref_model` in the file |

`KTOTrainer.__init__` has byte-for-byte the same three-branch shape as DPO:

```python
if ref_model:            self.ref_model = ref_model
elif self.is_peft_model or args.precompute_ref_log_probs:
                         self.ref_model = None          # <- the free path
else:                    self.ref_model = create_reference_model(model)  # <- 2x
```

So **KTO inherits the whole trap** and was gated separately rather than waved
through as reference-free.

The free path turns on exactly one predicate: `isinstance(model, PeftModel)`.
Soup's wrappers apply PEFT manually (`get_peft_model`) before constructing the
TRL trainer, so it holds — but it is a *predicate*, not a guarantee, which is why
check A1 asserts it directly instead of assuming it.

---

## GATE 1 — DPO. **PASS 14/14**

Two scripts, because the first one's memory arm was not evidence.

### 1a — correctness battery, CPU float32, 4-layer Llama

float32 on CPU so "bit-exact" means exactly `0.0`, not "within bf16 noise".

| check | result |
|---|---|
| **A1** one model / one stream | PASS — `ref_model=None`, `is_peft_model=True`, `RamSource` constructed **exactly 1** time |
| **A3** reference ≠ policy | PASS — `max\|policy−ref\|` chosen **8.219452e-01**, rejected **5.105972e-01** |
| **A3-control** zero adapter ⇒ ref == policy | PASS — exactly **0.000e+00** |
| **A4** bit-exact vs **resident DPO** | PASS — `max\|loss_streamed − loss_resident\| = 0.000e+00` over 16 synced adapter tensors (streamed 0.67242944, resident 0.67242944) |
| **A5** layer-0 adapter gradient ≠ 0 | PASS — 1.276013e-01; all 4 layers have non-zero adapter grad |
| **A6** determinism, same seed twice | PASS — 0.000e+00 |

**A3 is the check a loss curve cannot make.** The streamed layer substitutes base
weights through `functional_call` rather than the module's own `forward`, so
`disable_adapter()` being a no-op through that path is entirely plausible. If it
were, the reference would *be* the policy, every log-ratio would be 0, and the DPO
loss would sit at `−logsigmoid(0) = 0.6931` forever — which reads as "training
slowly", not as a bug. The zero-adapter control is what proves the 8.2e-01 gap
comes from the adapter and not from some other difference between the two forwards.

### 1b — the memory assertion at a size where WEIGHTS dominate, CUDA bf16

1a's memory arm passed on a **0.30 MB** model, where a second copy of the base is
1.6% of the peak. **That is not evidence** — it would pass for an implementation
that *did* keep a second copy. Re-run at 365.2M params = **730.4 MB bf16**,
24 layers, vocab 260, seq 64, batch 1, so a second instance would dominate:

```
streamed SFT peak VRAM     :     89.53 MB
streamed DPO peak VRAM     :     81.87 MB   (-7.66 MB, 0.914x)
CONTROL: DPO + 2nd model   :    812.32 MB   (+730.44 MB, 9.922x)
the base's own weights     :    730.44 MB
RAM store, SFT vs DPO      : 729.91 MB vs 729.91 MB
VRAM buffer pool, SFT/DPO  :  60.83 MB vs  60.83 MB
```

| check | result |
|---|---|
| **B1** DPO−SFT delta < the base's weight bytes | PASS — −7.66 MB vs 730.44 MB, i.e. **−1.0%** of one copy |
| **B2** DPO peak is not ~2x SFT | PASS — **0.914x** (a second instance would be **9.16x**) |
| **B3** CONTROL: an explicit second model DOES cost ~the weights | PASS — **+730.44 MB**, 100.0% of the weight bytes |
| **B4** one store, one pool | PASS — 729.91 MB / 60.83 MB, byte-identical to SFT |

**B3 is what gives B1/B2 teeth.** "DPO is not 2x SFT" means nothing unless the same
harness can show what 2x looks like; forcing `ref_model=` to a real second instance
moves the peak by **exactly one copy of the weights**. That is the cost the implicit
reference avoids, measured rather than argued.

DPO's peak is *below* SFT's (0.914x), which is not an anomaly: this SFT arm computes
a loss over all 64 positions while the DPO arm splits the same 64 into a 32-token
prompt and a 32-token completion, so its logits tensor is smaller. The claim being
gated is "no second copy of the weights", and a **negative** delta is the strongest
form of it.

### The I/O cost — published, not buried

`layer loads/step: SFT=46 DPO=70 over 24 layers -> 1.52x (1.92 vs 2.92 per layer)`.

Exactly the predicted ratio: SFT traverses the stack twice per step (forward +
checkpoint recompute); DPO traverses it three times (policy forward + reference
forward + recompute), so **3/2 = 1.5x**. Streaming makes the reference free in
*memory*, not in *time*. The 4-layer toy in 1a reported 2.00x, which is
boundary-dominated and is not quoted.

---

## GATE 2 (ORPO / SimPO) + GATE 3 (KTO) — **PASS 10/11**

Run against the **shipped** code, not a spike: `TrainerWrapper.setup()` → the real
TRL trainer → `train()`, for all five tasks. Model 24 layers × hidden 1024,
453.08 MB store, vocab 64, seq 64.

```
task   peak (resident)   store      pool    batch x rows   predicted
sft     61.81 (47.79)   453.08 MB  37.76 MB   1 x 1         60.75 MB
dpo     61.82 (56.31)   453.08 MB  37.76 MB   1 x 2         64.74 MB
orpo    61.82 (56.31)   453.08 MB  37.76 MB   1 x 2         64.74 MB
simpo   61.82 (56.31)   453.08 MB  37.76 MB   1 x 2         64.74 MB
kto     61.82 (56.31)   453.08 MB  37.76 MB   2 x 1         64.74 MB
```

| check | result |
|---|---|
| **C1** ORPO / SimPO have no reference model at all | PASS — the `ref_model` attribute is **absent** on both trainers |
| **C2** ORPO / SimPO store+pool identical to SFT | PASS — 453.08 MB / 37.76 MB; peak **1.000x** SFT |
| **C3** KTO takes the implicit reference | PASS — `ref_model=None`, store identical to SFT's |
| **C4** CONTROL: forcing a second model DOES cost the weights | PASS — 61.82 → 515.04 MB, **+453.22 MB vs 453.08 MB** of weights |
| **C5** pre-flight never under-predicts (dpo/orpo/simpo/kto) | PASS — **+4.7%** each |
| **C5** pre-flight never under-predicts (sft) | **−1.7%** — see below |

**KTO also required `batch_size >= 2`**, refused outright by TRL ("Actual (not
effective) batch size must be > 1. KTO will not work properly because the KL term
will be equivalent to the implied reward"). So KTO is streamable *only because
v0.72.3 lifted the batch-1 restriction* — under v0.72.0–.2 it could not have
shipped at all.

### The C5 SFT miss (−1.7%) is a fixture artifact, with evidence

It is on the **SFT** path, which this slot does not touch: the only change to the
budget multiplies `batch` by a per-task row count, which is **1** for SFT, so the
arithmetic is identical. The authoritative check is v0.72.3's peak-VRAM grid test —
**10 real measured runs**, within 1%, never under-predicting — which passes
unchanged. The −1.7% is 1.06 MB on a 61.81 MB peak at a fixture point far outside
the fitted envelope (vocab **64** vs the 49 152 / 151 936 it was fitted on), where
the constant terms dominate and the logits term is ~0.

---

## Three measurement attempts that were INVALID, recorded so they are not repeated

1. **Peak measured across `setup()`** charged the pre-flight's own GEMM ceiling
   probe — three 4096³ bf16 matrices, ~100 MB transient — to the training step, and
   reported a 44% "under-prediction" that did not exist. Reset the peak counter
   *after* setup.
2. **Cross-run retention inflated every peak by one buffer pool.** The pool is held
   by reference cycles in the module tree: measured `close()` alone retains
   **+47.65 MB**, `close()` + `gc.collect()` retains **+0.00 MB**. Not a leak and
   not a defect — but back-to-back streamed runs in one process hold the previous
   pool until a cycle-collection pass.
3. **Two attempts to fit a preference-specific logits constant.** With short rows
   `max_length` never bound the effective sequence — DPO measured an identical
   63.43 MB at seq 128, **256 and 512**; an independent variable that does not move
   measures nothing. With long rows the batch shapes showed TRL was not truncating
   to `max_length` either (`(1, 2401)` at every setting). **No constant was
   published rather than fabricating one.**

## The row multiplier is real, and had to be re-measured to prove it

DPO, ORPO and SimPO build their forward through `concatenated_inputs` +
`torch.cat`, so **2 × batch_size** rows reach the model in ONE tensor; KTO runs its
KL batch as a **separate** forward, so its count is 1x.

At vocab 64 the logits term is ~0, so 1x and 2x predict almost the same number:
**that check passed for both answers and was therefore not evidence.** Re-run at
vocab 32 000 the two are **+71.3% apart** (83.04 MB vs 142.22 MB), and all five
tasks still never under-predict.

## The estimator is a sound UPPER bound for preference losses, and it is loose

Charging `2 × batch` rows at `ForCausalLMLoss`'s measured 14 bytes/element is a
genuine upper bound — the concatenated forward *is* an SFT-shaped forward at twice
the rows, and TRL's preference losses reduce logits to per-token log-probs
(`selective_log_softmax`) instead of holding a full-vocab fp32 upcast, so their true
per-element cost is strictly lower. Measured: DPO's whole above-resident cost was
**51.76 MB** where the 14 B/elt charge for the same shape is ~458 MB.

Practical boundary, computed exactly (Llama-3.2-1B, vocab 128 256, NF4, 3.32 GB
free — this card):

```
seq  512  SFT 1.65 GB allowed  |  DPO 2.63 GB allowed
seq  768  SFT 2.14 GB allowed  |  DPO 3.60 GB REFUSED
seq 1024  SFT 2.63 GB allowed  |  DPO 4.58 GB REFUSED
```

The conservatism does **not** block the realistic seq-512 configuration; the exact
point where it starts refusing runs that would probably fit is **seq 768 at batch 1**.
Shipped as-is because under-predicting is the strictly worse failure — on Windows it
is not an exception but a silent WDDM spill — and documented rather than papered over.

---

## End-to-end, through the released CLI

SmolLM2-135M, NF4, streamed (30 layers, 0.05 GB pinned RAM store, 2 × 2 MB VRAM
buffers), real `soup train`:

| task | loss | note |
|---|---|---|
| dpo | 0.6931 → 0.6695 | starts at `−logsigmoid(0)` because `lora_B = 0` makes the reference equal the policy, then moves — the A3 property visible in a real run |
| orpo | 5.3816 → 5.0559 | |
| simpo | 5.2033 → 4.8749 | |
| kto | 0.5000 → 0.4904 (6 epochs) | its single-step run is flat at 0.5 for the same initialisation reason; checked over more steps rather than assumed |

All four saved adapters: **120 tensors, 0 keys carrying the streaming wrapper
segment, 60/60 non-zero `lora_B`** — i.e. ordinary LoRA adapters that load into any
non-streaming model.

### One CPU-only limitation, found by CI rather than locally

A full KTO **training step** over a streamed model runs on CUDA (verified here and
on the dev box) but fails on a CPU-only runner under newer torch/TRL with
`Tensor on device cpu is not on the expected device meta!`. Streaming exists to
bound VRAM, so a streamed model on CPU is a test convenience rather than a real
configuration — the same stance v0.72.3 took on PEFT's re-dispatch — and that test
is therefore gated to the production device. Everything else about KTO (the schema
gate, `setup()`, the reference behaviour, the layer-read accounting) is still
exercised on CPU.

Worth stating plainly: this was invisible on the dev box, which has CUDA. The
locally-green suite is a weaker signal than it looks whenever a code path forks on
device.

> ### Correction, post-release: the gate above blamed the wrong variable
>
> "A streamed model on CPU is a test convenience" is a reasonable-sounding
> generalisation applied to a specific, undiagnosed failure — and it is wrong here.
> Three pieces of evidence:
>
> 1. **The same CI run contradicts it.** In the run that motivated the skip,
>    exactly one test failed. `test_v07200.py::test_one_training_step_actually_runs`
>    — the identical streamed full-trainer `train()` test for SFT — passed on that
>    same CPU runner, as did the NF4 variants and all four preference
>    bit-exactness tests, KTO included. CPU streaming demonstrably works there.
> 2. **The test passes on CPU here.** Executing that exact test body on the dev box
>    with CUDA masked (`CUDA_VISIBLE_DEVICES=-1`, torch 2.5.1 / trl 0.19.1 /
>    transformers 4.57.6): **PASSED**, one step, loss 0.5. The variable is the
>    newer stack CI resolves, not the device.
> 3. **The error is a streaming property, not a device property.**
>    `Tensor on device cpu is not on the expected device meta!` comes from
>    `check_same_device` in `torch/_prims_common/__init__.py` — an operation
>    received a `meta` placeholder alongside a real tensor, via torch's
>    decomposition path. A meta placeholder is reachable by a real op in KTO's
>    second (KL) forward; newer torch decomposes more operations, which is why it
>    surfaces there and not here.
>
> The consequence is worse than a mislabelled skip. CI has no GPU runners
> (ubuntu / windows / macos only), so `skipif(not cuda)` made the test **dead
> everywhere in CI** — it ran only on the dev box, under the old torch where it
> passes, while `torch` carries no upper bound and users get the stack that fails.
> The skip now gates on the actual variable and the meta leak is tracked as its
> own issue rather than absorbed into a generalisation.

Ten rejected configurations each named its own reason (rollout tasks, unsupported
task, KTO at batch 1, unsloth backend, 8-bit, no adapter, DoRA, `batch_size: auto`,
packing).
