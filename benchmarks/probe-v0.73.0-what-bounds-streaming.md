<!--
Working measurement record, published verbatim.

Like the gate records beside it, this is the log kept while the work happened,
not a report assembled afterwards: the failed instrumentation, the corrected
premise and the numbers that were discarded stay in, in the order they occurred.

Hardware: RTX 3050 Laptop (4 GB, 4.29 GB usable), 16.9 GB host RAM, NVMe,
Windows 11. torch 2.5.1+cu121 · bitsandbytes 0.49.2 · transformers 4.57.6 ·
peft 0.18.1 · Python 3.10.8. Soup v0.73.0 (working tree at 57c18e5).
-->

# What actually bounds layer streaming — and can Cut Cross-Entropy lift it?

**Status: TASK 1 ANSWERED — the run is compute-bound, not transfer-bound.
TASK 2 ANSWERED for the kernel, BLOCKED for the shipped flag — Cut Cross-Entropy
triples the usable microbatch (1 -> 3) for +9.6% throughput, but Soup's own
`training.use_cut_ce` cannot engage on any pip-installed stack (three
independent blockers, all measured in §7).**

Two questions, asked in this order because the second only makes sense after the
first:

1. Soup says publicly that layer streaming is **bound by host-to-device
   transfer**. That claim had never been measured. Is it true?
2. Cut Cross-Entropy has shipped since v0.28.0 and layer streaming since
   v0.72.0. Nobody has ever switched both on. What happens?

Unit convention: **decimal GB**, matching the gate records.

---

## 0. The premise was wrong before the first measurement, and in the safe direction

The brief that started this work computed the achieved transfer rate as
**~6.7 GB/s** — "about half the Gen4 x8 ceiling" — and reasoned that if half the
bus is already in use, the bus is plausibly the constraint.

That figure reproduces exactly if the step is assumed to be **128 tokens**:

```
7.2 GB / (128 tok / 119.6 tok/s) = 6.73 GB/s
```

The published 119.6 tok/s row is at **S=512**, not 128. At the real sequence
length the same arithmetic gives:

```
6.864 GB per step / 4.281 s = 1.60 GB/s
```

A second, smaller correction: the naive count is 64 layer loads per step
(32 forward + 32 backward-recompute). The measured count is **61.0** — the
prefetcher skips a load when the slot already owns the layer it is about to
need. So the moved volume is 61 x 112.529 MB = **6.864 GB**, not 7.2 GB.

This does not weaken the brief's conclusion, it strengthens it: the run is using
**~20% of what this box can actually do**, not ~50%. Recorded because a premise
that survives into the answer is how a wrong answer gets published.

---

## 1. What this box can actually do (T1a)

`scratchpad/t1_pcie.py`. Pinned host buffers copied to device on a side stream,
timed with CUDA events, best-of-30. The "real pattern" arm allocates the exact
30 tensors of one Llama-3.1-8B NF4 layer shard, so per-call overhead is
separable from bandwidth.

| # | What | GB/s | Note |
|---|---|---|---|
| A | the real 30-tensor layer pattern, **pinned** | **7.77** | best; median 7.50 |
| B | one contiguous block of the same 112.529 MB, pinned | 7.42 | |
| C | the same 30-tensor pattern, **pageable** | 4.91 | the fallback path |
| D | 512 MB block, pinned | 6.86 | |

Negotiated link, checked idle and again **under load**:
`PCIe Generation Current: 4`, `Link Width Current: 8x` (max 16x) — i.e. Gen4 x8,
~15.75 GB/s theoretical.

**Finding 1a — splitting a layer into 30 tensors costs nothing.** A is not
slower than B; it is marginally faster, inside noise. There is no per-copy launch
overhead worth removing, so "batch the shard into one blob" is not an
optimisation. This was worth checking before assuming it.

**Finding 1b — the achievable rate is ~7.8 GB/s idle, and the real run does
better.** During training the copy stream was measured at **9.38 GB/s** (§3),
above the idle probe. The idle probe under-reads because the link sits in a lower
power state between synchronised bursts. So "achievable" here is 7.8–9.4 GB/s
depending on link state, against 15.75 theoretical.

---

## 2. The control: does the harness reproduce the published row? (T1b)

`scratchpad/t1_step.py --mode baseline`. Model built through the **shipped**
path (`read_shard_index` + `build_streamed_model`), NF4 + double quant, LoRA r=8
on q/k/v/o, batch 1, S=512, `PagedAdamW8bit`, 2 buffers, RAM tier, pinned.

| | published v0.72.2 step-6 | measured here |
|---|---|---|
| tok/s | 119.6 | **122.1 – 122.5** |
| peak VRAM | 3.32 GB | **3.448 GB** |
| pinned store | 3.60 GB | **3.601 GB** |
| SM clock | 952 MHz | 952–960 MHz |

The store matches to three decimals. Throughput is 2% high and peak VRAM 0.13 GB
high, both explainable by the LoRA configuration not being recorded in the
published row (this harness uses r=8 on four projections). **The control holds**;
everything below is measured on a run that reproduces the published one.

---

## 3. Where the step actually goes (T1c)

CUDA-event instrumentation, `--mode events`. Two numbers matter and they are not
the same number:

- **copy time** — how long the transfers take on their own stream. Can overlap.
- **stall** — how long the *compute* stream sits blocked on a layer's copy
  event. This is the only part of the transfer that is **not** hidden.

Measured by recording an event on the compute stream immediately before and
after `wait_event`, so the gap between them is exactly the blocked interval.

| quantity | per step | share of the 4.190 s step |
|---|---|---|
| step time | 4.190 s | 100% |
| bytes moved | 6.864 GB | |
| copy time on the prefetch stream | 0.732 s | 17.5% |
| **compute stream stalled on a copy** | **0.0084 s** | **0.20%** |
| copy-stream rate | 9.38 GB/s | |
| average rate over the step | 1.638 GB/s | |

tok/s with the instrumentation: **122.20**, against 122.14 uninstrumented — the
measurement is free.

**Finding 3 — transfers are essentially perfectly hidden.** The compute stream
waits 8.4 ms out of 4190 ms. Double buffering is doing its job.

---

## 4. The ablations, interleaved (T1d)

A profile shows where time is *attributed*. An ablation shows what removing a
term actually *buys*, which is the stronger claim. Four arms, switched by flag
inside **one process on one model**, and **interleaved** A/B/C/D per round so
monotonic clock drift cannot favour one arm — the method the v0.72.3
accumulation gate used for the same reason.

Both ablation arms produce **garbage mathematics** and are timing-only:
`nocopy` leaves stale bytes in the buffers, `nodequant` multiplies by a cached
zero weight.

Llama-3.1-8B NF4, batch 1, S=512, 8 steps x 2 rounds, all at **960 MHz**:

| arm | step | tok/s | vs baseline |
|---|---|---|---|
| A baseline | 4.211 s | 121.6 | — |
| B **no H2D transfers at all** | 4.150 s | 123.4 | **−1.44%** |
| C no NF4 dequantisation | 3.799 s | 134.9 | −9.80% |
| D neither | 3.743 s | 136.8 | −11.3% |

Round-to-round spread: 1.4% (A), 0.35% (B), 0.31% (C).

**Finding 4 — deleting every byte of host-to-device traffic makes the step 1.4%
faster.** 6.864 GB per step, gone, for a 1.4% gain. Layer streaming at this
configuration is not transfer-bound, and no amount of PCIe bandwidth would
change the headline number.

**Finding 4b — with both the transfer and the dequantisation removed, 88.7% of
the step remains.** Whatever bounds this run is neither of the two things
streaming adds.

---

## 5. So what is it? (T1e)

### 5a. The step splits cleanly into a fixed cost and a per-token cost

Sequence sweep at batch 1, one build, 8 steps per point:

| tokens | step | tok/s | peak VRAM |
|---|---|---|---|
| 16 | 0.983 s | 16.3 | 2.860 GB |
| 32 | 1.029 s | 31.1 | 2.869 GB |
| 64 | 1.043 s | 61.4 | 2.888 GB |
| 128 | 1.417 s | 90.3 | 2.931 GB |
| 256 | 2.269 s | 112.8 | 3.040 GB |
| 384 | 3.186 s | 120.5 | 3.183 GB |
| 512 | 4.179 s | 122.5 | 3.448 GB |

Least squares over 128–512:

```
step(S) = 0.462 s + 7.190 ms/token
```

At S=512 that is **11.1% fixed, 88.1% proportional to tokens**. The fixed part is
what the transfers and the launch overhead live in; the proportional part is the
model's own arithmetic.

**The fit is deliberately restricted to 128–512, and the three points below it
say why.** Extrapolated to S=16 it predicts 0.577 s where 0.983 s was measured,
and S=16/32/64 are nearly flat at ~1.0 s — so the "fixed" term is not one
constant across the whole range. That is not a bad fit, it is the physics of
§5c: the per-layer launch and Python overhead is **CPU-side**, so at long
sequences most of it is issued while the GPU is still busy and disappears from
the wall clock, while at short ones the CPU cannot stay ahead and the same
overhead is fully exposed. 0.462 s is therefore the part that remains
unavoidably serial at S=512, and ~1.0 s at S=16 is the same overhead with
nothing left to hide behind. Both numbers are real; they are the hidden and
exposed views of one cost, which is why §5a and §5c do not disagree.

### 5b. The proportional part is real GEMM work, at 71% of this card's ceiling

Measured **in the same session at the same clock**, because the v0.72.2 gate
established that a fraction-of-ceiling quoted across sessions is meaningless on
this card (13% boost-clock spread). `scratchpad/t1_ceiling.py` runs the streamed
step first, then the ceiling, and reports the clock for both.

| shape | M x K x N | TFLOPS |
|---|---|---|
| q_proj / o_proj | 512 x 4096 x 4096 | 7.67 |
| k_proj / v_proj | 512 x 4096 x 1024 | 7.51 |
| gate / up_proj | 512 x 4096 x 14336 | 7.47 |
| down_proj | 512 x 14336 x 4096 | 7.67 |
| **FLOP-weighted** | | **7.55** @ 952 MHz |

Streamed step in the same session: 4.182 s -> 122.4 tok/s @ 952 MHz, i.e.
22.52 TFLOP/step at the gate's C=6 accounting -> **5.38 TFLOPS effective =
71.3% of the same-session, shape-matched ceiling.**

With the two streaming-specific terms ablated away (arm D), the same arithmetic
gives 6.02 TFLOPS = **79.7% of ceiling**.

### 5c. There *is* a per-layer overhead, and it is visible only when compute is small

The same four arms at **S=16**, where the arithmetic is negligible:

| arm | step |
|---|---|
| A baseline | 0.974 – 1.187 s |
| B no transfers | 0.914 – 0.999 s |
| C no dequant | 0.814 – 0.868 s |
| D neither | **0.626 – 0.716 s** |

At 16 tokens the GEMM work is ~0.09 s, yet arm D still costs ~0.67 s. That
residue — roughly 10 ms per layer visit across 64 visits — is per-layer
**launch and Python** overhead: 30 pooled tensors, a rebuilt `Params4bit` view
per weight, a `functional_call`, and ~50 kernel launches, all done 64 times.

At S=512 this is hidden behind compute and costs nothing. It is why the sweep's
tok/s curve is steep below ~128 tokens: at short sequences the run **is**
fixed-cost-bound, and there the transfers do matter.

### 5d. "@100% util" was never in tension with anything

Sampled under load: `utilization.gpu` **100%**, memory-controller utilisation
44–51%, clock pinned at 952 MHz, link steady at Gen4 x8. `utilization.gpu` is the
fraction of time at least one kernel is resident — it is not occupancy and not
efficiency. 100% util with 71% of the GEMM ceiling is the ordinary signature of a
compute-bound eager loop, and it is consistent with, not contradicted by, the
transfers being hidden.

---

## 6. Task 1, answered in one sentence

**We are compute-bound, not transfer-bound: the streamed step runs at 71.3% of
this card's same-session bf16 GEMM ceiling, deleting every host-to-device byte
buys 1.4%, and the compute stream spends 0.20% of the step waiting on a copy —
the largest streaming-specific overhead is not the bus but the per-layer NF4
dequantisation, at 9.8%, followed by ~10 ms/layer-visit of kernel-launch and
Python overhead that only becomes the binding constraint below ~128 tokens per
step.**

### What this means for the public claim

"Bound by host-to-device transfer" is **wrong at the published configuration**
and should not be repeated. What the H100 result actually shows is the weaker but
still interesting statement the brief reasoned towards: the bottleneck is
**common to both machines and is not the bus**. On the H100 that is consistent
with the same finding here — an eager, per-layer, dequantise-then-GEMM loop is
not fed by the interconnect.

Two honest qualifications:

- The transfer claim **is** true at short sequences / small token counts per
  step, where the fixed 6.864 GB dominates. The sweep in §5a is where the
  crossover lives.
- Nothing here says the H100 run was compute-bound *for the same reason*. That
  box is gone and this record makes no claim about it.

### What would actually make it faster, ranked by measured headroom

1. **The 28.7% gap to the GEMM ceiling** (arm D at 79.7%) — attention,
   elementwise and launch overhead. Largest single pool.
2. **The NF4 dequantisation, 9.8%.** Every streamed weight is dequantised into a
   dense bf16 tensor on every visit — 448 calls per step, ~437 MB of VRAM writes
   per layer visit. This is the cost of the #331 repair, which deliberately took
   the weight out of `MatMul4Bit` to make gradient checkpointing see it. A fused
   dequantise-and-multiply that is still checkpoint-visible is the obvious
   target, and it is the *only* item on this list that is specific to streaming.
3. **Transfers: 1.4%.** Not worth touching.

---

# TASK 2 — Cut Cross-Entropy on top of layer streaming

## 7. Soup's own `use_cut_ce` flag cannot engage. Three blockers, all measured

The schema permits the combination — this parses, and no cross-validator objects:

```yaml
training:
  stream_layers: true
  quantization: 4bit
  use_cut_ce: true
```

`training.use_cut_ce` routes to `utils/cut_ce.py::apply_cut_ce`, which does
`from cut_cross_entropy.transformers import cce_patch`. Every attempt to make
that import succeed on this stack failed, for a *different* reason each time:

| # | What was tried | Result |
|---|---|---|
| 1 | `pip install cut-cross-entropy` (what the module docstring instructs) | wheel **25.1.1 ships no `transformers` submodule at all** — 31 files, none matching `transformer`. The import can never succeed. |
| 2 | `pip install cut-cross-entropy[transformers]` | the extra only adds a `transformers>=4.44.2` *dependency*; it adds no code. Same 31 files. |
| 3 | `pip install git+https://github.com/apple/ml-cross-entropy` (25.9.3, which **does** ship the submodule) | `ImportError: cannot import name '_CONFIG_FOR_DOC' from transformers.models.gemma2.modeling_gemma2`. Shimming that symbol surfaced the next one, `GEMMA2_INPUTS_DOCSTRING`. |
| 4 | (on 25.9.3) calling the kernel directly | `PackageNotFoundError: No package metadata was found for triton` — `is_triton_3_2()` looks up dist `triton`, but Windows installs `triton_windows`, which provides the *module* `triton` 3.1.0 under a different *distribution* name. |

Blocker 3 is the load-bearing one and it is **not Windows-specific**: CCE's
integration is written against a transformers generation that no longer exports
those private symbols, and `patch.py` imports the gemma2 patcher eagerly, so a
**llama** model trips over a **gemma2** incompatibility. Soup pins
`transformers<5.0.0`, so 4.57.6 is squarely inside the supported range.

**How this presents to a user.** `apply_cut_ce` catches
`(ImportError, AttributeError, NotImplementedError)` and returns `False`. The
trainer does check the return value and prints a yellow line — so this is **not**
a silent no-op, unlike the Liger defect (#78) where the flag was set but never
reached `TrainingArguments`. But the message names neither cause:

> `Cut Cross-Entropy: no matching architecture found or cut_cross_entropy not installed`

Both halves are false here. The package *is* installed and llama *is* a matching
architecture; what failed is an upstream import. A user reading that line would
go and re-install the package that is already there.

**Consequence: `training.use_cut_ce` has been in the tree since v0.28.0 and, on
any pip-installed stack, has never been able to do anything.** No measurement in
this record uses it.

### What was measured instead, and how to read it

CCE is wired **by hand** here, the way CCE's own README shows:

```python
hidden = decoder(input_ids=ids).last_hidden_state
loss = linear_cross_entropy(hidden[:, :-1], lm_head.weight, labels[:, 1:])
```

on **cut-cross-entropy 25.1.1** + **triton-windows 3.1.0.post17**. Both arms
share the identical decoder forward and differ only in the loss. So the numbers
below measure **the kernel**, not Soup's flag.

---

## 8. Correctness — and it forces a change to what "correct" can mean here

The reference must carry the **same loss kernel**: CCE is not bit-exact against
torch CE, so streamed+CCE against resident+plain-CE would diverge for a reason
that has nothing to do with streaming. Measured on the same streamed model,
CCE against `ForCausalLMLoss`: **7.901e-03**. The caveat is real, not theoretical.

Model **SmolLM2-135M NF4** — the size GATE 1 of the v0.72.2 record used, and the
only class where a resident NF4 reference actually fits on this card (that record
measured resident NF4 **3B** spilling to 6.07 GB of host memory on a 4 GB card,
so a resident 8B reference is not available here at any batch). Batch 1, S=256,
240 adapter tensors synced streamed -> resident and **verified identical before
the comparison** rather than assumed.

| arm | loss, streamed vs resident | max grad abs diff, 240 tensors |
|---|---|---|
| **control** — ForCausalLMLoss | **0.000e+00** | **0.000e+00** |
| **test** — Cut Cross-Entropy | **0.000e+00** | 9.328e-03 / 1.178e-02 (two runs) |

The control is what makes the test row readable: with the ordinary loss this
harness sees streaming as bit-exact in both halves, so it *can* detect equality.

### The control that decides how to read the CCE row

Running the **same loss twice on the same model** — no streaming involved in the
comparison at all:

| repeat, same model, same input | max grad abs diff |
|---|---|
| streamed, ForCausalLMLoss | **0.000e+00** |
| resident, ForCausalLMLoss | **0.000e+00** |
| streamed, **CCE** | 8.858e-03 |
| resident, **CCE** | 9.461e-03 |

**Finding 8 — the streamed-vs-resident gradient gap under CCE (0.93–1.18e-02)
sits inside CCE's own run-to-run noise on either model (0.89–0.95e-02).** It is
CCE's non-determinism — its backward accumulates the weight gradient with
atomics — not a streaming defect. The forward is unaffected: the loss is
bit-exact streamed vs resident, and the HF path repeats at exactly 0.0 on both
models, so the noise is specific to the CCE backward.

**Finding 8b — this is the part with a cost.** Soup's central claim, and the
criterion its CI enforces, is bit-exactness against a resident reference. Turning
CCE on **destroys that criterion**, and not because of streaming: a resident run
is not reproducible against *itself*. Any future release that ships
`use_cut_ce` with streaming has to replace "bit-exact" with "within the loss
kernel's measured noise floor", and has to measure that floor per model and per
shape. That is a strictly weaker guarantee than the one the project currently
makes, and it should be a deliberate decision rather than a side effect of
setting a flag.

---

## 9. Peak VRAM and the microbatch sweep

Llama-3.1-8B NF4, streamed, S=512, LoRA r=8 on q/k/v/o, `PagedAdamW8bit`,
2 buffers, 4 timed steps after 2 warm-up. `hf` = `ForCausalLMLoss`, `cce` =
hand-wired `linear_cross_entropy`. Card total **4.294 GB**.

| arm | microbatch | tokens/step | tok/s | peak VRAM | verdict |
|---|---|---|---|---|---|
| hf | 1 | 512 | 121.52 | **3.448 GB** | fits |
| hf | 2 | 1024 | 125.14 | 4.512 GB | **spilled** (above the card) |
| hf | 3 | 1536 | 40.95 | 5.573 GB | **spilled**, 3.0x collapse |
| hf | 4 | 2048 | — | — | hard `CUDA error: out of memory` |
| cce | 1 | 512 | 120.52 | **3.144 GB** | fits |
| cce | 2 | 1024 | 129.01 | 3.463 GB | fits |
| cce | 3 | 1536 | **132.03** | 3.782 GB | fits |
| cce | 4 | 2048 | 19.20 | 4.100 GB | **spilled**, 6.9x collapse |

`cce b=3` was measured twice, in two different processes: 131.69 and 132.03
tok/s, peak **3.782 GB both times**.

### 9a. The headline: memory, not speed

- **At equal batch, CCE saves 0.304 GB (−8.8%) at b=1.** That is the whole of
  the answer to "measure peak VRAM in both cases" — and on its own it is not very
  interesting.
- **The slope is the interesting part, and it is dead linear.** Peak grows by
  **1.063 GB per extra microbatch** without CCE (1.064, 1.061) and by
  **0.319 GB** with it (0.319, 0.319) — a **3.33x** reduction in the marginal
  cost of a microbatch.
- The difference, **0.744 GB per 512 tokens**, is the cross-entropy pipeline:
  **11.32 bytes per logit element**. Soup's own peak-VRAM pre-flight charges
  `LOGITS_BYTES_PER_ELEMENT = 14`. This is an independent measurement of the same
  quantity from a different direction, and it lands **19% below** the shipped
  constant — a data point for the open #327 ("the pre-flight over-predicts"),
  measured here as a marginal cost rather than fitted inside a total.
- **Usable microbatch goes 1 -> 3.** Without CCE this configuration fits only at
  batch 1 on this card; with CCE it fits at 3.

### 9b. Where the linear growth stops — and it is not where the memory does

tok/s against microbatch, CCE arm: **120.52 -> 129.01 -> 132.03**, i.e. +7.0%
then +2.3%. Task 1's sequence sweep independently fitted
`step(S) = 0.462 s + 7.190 ms/token`, whose asymptote is **139.1 tok/s**. Against
that:

| microbatch | tok/s | % of the 139.1 asymptote |
|---|---|---|
| 1 | 120.52 | 86.6% |
| 2 | 129.01 | 92.7% |
| 3 | 132.03 | 94.9% |

**The curve flattens between microbatch 2 and 3 — before the memory runs out,
not because of it.** This is Task 1's finding arriving from the other side: the
run is already compute-bound at batch 1, so raising the batch cannot buy
throughput, it can only amortise the fixed 0.462 s/step over more tokens. That
fixed cost is 11% of the step at b=1 and ~4% at b=3, and once it is amortised
there is nothing else for batching to recover.

So the honest summary of Task 2 is: **CCE buys memory headroom (3.3x cheaper
marginal batch, 1 -> 3 usable microbatch), and the throughput that headroom
converts into is +9.6% and then stops.** Anyone hoping CCE unlocks a large
speed-up on this configuration should read §6 first.

### 9c. Three things went wrong measuring this, all recorded

1. **The first sweep ran ~20 minutes and left nothing.** The process died (exit
   4) with an empty log; results were only written at the end. Fixed by
   persisting after every point — which is the only reason the data above
   survived the second crash.
2. **A hard `CUDA error: out of memory` is not the allocator's
   `torch.cuda.OutOfMemoryError`.** It poisons the CUDA context, so every later
   call in that process fails; the `hf b=4` row killed the run and the remaining
   points had to be re-measured in a fresh process. A sweep that walks a batch
   size upward past the refusal point must expect to be restarted, not to
   continue.
3. **My own spill tripwire was wrong, twice.** `peak <= card total (4.294 GB)`
   would have called `cce b=4` (4.100 GB) a fit, and it collapsed 6.9x.
   `peak <= free-at-idle (3.460 GB)` would have called `cce b=2` and `b=3`
   spills, and they ran at 129 and 132 tok/s with no collapse at all. Neither
   threshold predicts this box.

   What the data actually supports: **a peak above the card's total is proof of a
   spill** (`hf` b=2 and b=3; `cce` never exceeds it), **a peak below it is not
   proof of a fit**, and the only reliable signature is behavioural — a
   throughput collapse against the neighbouring point. Note that `hf b=2` spilled
   by 0.2 GB and got *faster* (125.14 vs 121.52 tok/s), so the absence of a
   collapse does not prove a fit either. This is the same Windows/WDDM property
   the v0.72.3 gate recorded when it measured 9.27 GB allocated on this 4.29 GB
   card with nothing raised.

---

## 10. What was NOT measured

Stated rather than implied, because several of these look like they should have
been in scope:

- **Soup's `training.use_cut_ce` end to end through `soup train`.** It cannot
  engage (§7). Every CCE number here comes from a hand-wired kernel call, and
  none of them is evidence that the shipped flag works.
- **A resident 8B reference of any kind.** It does not fit on this card, so the
  §8 correctness gate runs at 135M. Nothing here re-verifies streaming
  correctness at 8B; the H100 record already does that, and this record does not
  extend it.
- **Nsight Systems.** `torch.profiler` was used instead. Its CUPTI layer
  **silently dropped every device event** over a 28-second, 5-step window — an
  82 MB trace containing 340,425 events, of which zero were kernels. It works
  over a single step. The profiler also inflates the step by ~31% (5.48 s vs
  4.19 s), so §3–§5 rely on CUDA events and ablations, and the profiler is used
  only for the relative op breakdown.
- **CUDA's own `bandwidthTest` binary.** Not installed on this box; §1 is a torch
  equivalent (pinned host buffers, CUDA-event timed). The two are not guaranteed
  to agree.
- **Whether the H100 run was compute-bound for the same reason.** That box is
  gone. This record makes no claim about it.
- **bf16 (non-NF4) streaming** under the same decomposition, and **sequences
  beyond 512** or the disk tier.
- **Whether a fused dequantise-and-multiply would actually recover the 9.8%.**
  §4 measures what removing the dequantisation is worth, not what a real fused
  kernel would cost.
- **CCE at other sequence lengths or vocabularies.** The 11.32 bytes/element in
  §9a is one shape (S=512, vocab 128256), from two slope segments.
- **cce b=5 and b=6.** Stopped deliberately: b=4 had already spilled, so those
  points would have measured Windows paging for ~30 minutes.

---

## 11. Follow-ups this record earns

1. **`training.use_cut_ce` is dead on arrival** — three blockers in §7, and the
   warning text names none of them. Either pin a working integration, vendor the
   ~40 lines of patch, or refuse the flag loudly at config-parse time instead of
   printing a yellow line mid-run that sends the user to re-install a package
   they already have.
2. **The public "bound by host-to-device transfer" claim is wrong** at the
   published configuration and needs correcting wherever it appears (§6).
3. **The NF4 dequantisation is worth 9.8%** and is the only streaming-specific
   term with real headroom (§4).
4. **#327 gains an independent measurement**: the CE pipeline's marginal cost
   here is 11.32 B/element against the shipped constant of 14 (§9a).
5. **If CCE is ever shipped with streaming, the bit-exactness gate has to
   change** — CCE is not reproducible against itself, resident or streamed
   (§8b).
