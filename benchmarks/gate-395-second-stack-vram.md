# Gate record — #395: the streaming VRAM fit on a second GPU/stack

**Hardware.** NVIDIA A10G, 23.0 GB (AWS `g5.xlarge`), Ubuntu 22.04, driver
580.126.09. torch 2.13.0+cu130, transformers 5.16.1, trl 0.29.1, peft 0.20.0,
Python 3.10. Every number below was measured in that configuration. Nothing here
was measured anywhere else, and nothing is extrapolated to other hardware.

This is the "second GPU/stack" #395 asks for. The record it answers,
[`gate-v0.73.1-measured-vram-fit.md`](gate-v0.73.1-measured-vram-fit.md), was
taken on an RTX 3050 Laptop 4 GB / Windows 11 / torch 2.5.1 / **transformers
4.57.6**. Both the GPU and the stack differ here, and the transformers gap is a
major version.

---

## The headline result

**The under-prediction does not reproduce.** Same protocol — real `soup train`
setup + one step, bf16, batch 1, `quantization: none`, LoRA r=8,
`stream_buffers: 2`, "real peak" is `torch.cuda.max_memory_allocated()`.

SmolLM2-135M:

| seq | predicted | real peak | pred/real | verdict |
|---|---|---|---|---|
| 2048 | 1.596 GB | 1.365 GB | 1.169x | over-predicts — safe |
| 3072 | 2.345 GB | 2.017 GB | 1.163x | over-predicts — safe |
| 4096 | 3.094 GB | 2.658 GB | 1.164x | over-predicts — safe |
| 4352 | 3.281 GB | 2.818 GB | 1.164x | over-predicts — safe |
| 5120 | 3.842 GB | 3.303 GB | 1.163x | over-predicts — safe |
| 6144 | 4.591 GB | 3.943 GB | 1.164x | over-predicts — safe |

Qwen2.5-0.5B (3.1x the vocabulary, where the logits term dominates hardest):

| seq | predicted | real peak | pred/real | verdict |
|---|---|---|---|---|
| 2048 | 4.854 GB | 4.177 GB | 1.162x | over-predicts — safe |
| 4096 | 9.346 GB | 8.012 GB | 1.166x | over-predicts — safe |
| 5120 | 11.592 GB | 9.929 GB | 1.167x | over-predicts — safe |
| 6144 | 13.837 GB | 11.848 GB | 1.168x | over-predicts — safe |

Against the RTX 3050 at the same shapes: 1.081x at 4352, **0.934x** at 5120,
**0.787x** at 6144. Here the ratio is flat at 1.16x through 6144. There is no
crossover on this stack, and the direction property holds everywhere measured.

---

## What the gap actually is — measured, not back-solved

An earlier revision of this record led with **11.83 B/element**, back-solved by
subtracting the formula's own modelled non-logits terms from the real peak. That
number is withdrawn. It was one of two readings the data could not distinguish,
and the instrument to settle it (`measure_logits_loss_bytes_per_element`) was
already in the tree and was not run. It has now been run, on the box that could.

### 1. The loss term is the documented 12, on this stack too

```
stack: torch 2.13.0+cu130 | NVIDIA A10G | trl 0.29.1 | transformers 5.16.1

measure_logits_loss_bytes_per_element(), 3 repeats at the default shape:
  12.0
  12.0
  12.0
  spread: 0.00e+00

across vocabularies:
  vocab   8192: 12.0
  vocab  32768: 12.0
  vocab  49152: 12.0
  vocab 151936: 12.01010951979781   (allocator rounding)
```

`LOGITS_LOSS_BYTES_PER_ELEMENT`'s invariance claim — *"byte-identical under torch
2.5.1 and 2.13.0"* — **holds here**. This is a third stack agreeing, not a
counter-example. The 11.83 was never a measurement of this term.

### 2. The `+2` retained bf16 copy is ABSENT on this stack

This is the finding. The shipped 14 is `12` (loss) + `2` (one further bf16
logits-shaped tensor, charged unconditionally, retained by something #327 has
not identified). If that copy were retained here, the whole real peak would have
to exceed the logits term at 14 alone. It does not — on any row:

| model | seq | 14 x E | real peak |
|---|---|---|---|
| SmolLM2-135M | 2048 | 1.409 GB | **1.365 GB** |
| SmolLM2-135M | 4096 | 2.819 GB | **2.658 GB** |
| SmolLM2-135M | 5120 | 3.523 GB | **3.303 GB** |
| SmolLM2-135M | 6144 | 4.228 GB | **3.943 GB** |
| Qwen2.5-0.5B | 2048 | 4.356 GB | **4.177 GB** |
| Qwen2.5-0.5B | 4096 | 8.713 GB | **8.012 GB** |
| Qwen2.5-0.5B | 5120 | 10.891 GB | **9.929 GB** |
| Qwen2.5-0.5B | 6144 | 13.069 GB | **11.848 GB** |

`14 x E` alone exceeds the entire measured peak, leaving no room for the
non-logits terms let alone the retained copy. Ten of ten rows, two models, a
3.1x vocabulary contrast. The retention #327 cannot explain **does not happen on
this stack**, which is a constraint on any explanation of it.

### 3. The non-logits terms are over-estimated by ~15%

Holding the measured 12 fixed and asking what the rest of the model does:

| model | seq | non-logits modelled | implied true | over-estimate |
|---|---|---|---|---|
| SmolLM2-135M | 2048 | 0.187 GB | 0.157 GB | 1.190x |
| SmolLM2-135M | 4096 | 0.275 GB | 0.242 GB | 1.138x |
| SmolLM2-135M | 6144 | 0.363 GB | 0.319 GB | 1.138x |
| Qwen2.5-0.5B | 2048 | 0.498 GB | 0.443 GB | 1.124x |
| Qwen2.5-0.5B | 6144 | 0.769 GB | 0.646 GB | 1.189x |

**mean 1.152x, population stdev 0.0249** over all ten rows.

So the 1.16x over-prediction decomposes as: the absent `+2` copy (the larger
share, and vocab-scaled) plus a ~15% over-estimate of the non-logits terms. Not
"the logits multiplier is different here", which is what the withdrawn 11.83
implied.

## What this says about the mechanism #395 could not identify

Stated carefully, because only one half of this is measured here.

Measured: on this stack the term is **flat** in sequence length. Not measured
here, but implied by the 3050 record's own numbers: 14 / 0.934 = 15.0 B/element
at seq 5120 and 14 / 0.787 = 17.8 at 6144. On that stack the multiplier **grew
with sequence length**; on this one it does not move.

If that reading is right, the reason no coefficient ever fitted is structural. A
term that varies both per-stack and per-sequence-length is not a constant, so no
constant can carry the never-under-predict guarantee across stacks — which is
the argument `training.stream_vram_probe` already makes. This record is evidence
for that decision rather than against it.

### The leading hypothesis is not supported here

#395 names it: "a switch from a memory-efficient path to the math path would
materialise the full attention matrix and is the leading hypothesis worth
killing or confirming first."

Forcing each SDPA backend at the shapes in question (1 head, 64 dim, bf16,
causal), extra allocation over baseline:

| seq | flash | mem-efficient | math |
|---|---|---|---|
| 2048 | 2.4 MB | 2.4 MB | 383.9 MB |
| 3072 | 3.7 MB | 3.5 MB | 831.5 MB |
| 4096 | 4.9 MB | 4.7 MB | 1463.8 MB |
| 5120 | 6.1 MB | 5.9 MB | 2279.7 MB |
| 6144 | 7.3 MB | 7.1 MB | 3267.4 MB |

The math path is exactly as expensive as the hypothesis requires — 3.27 GB at
seq 6144 would sink a 4 GB card on its own. **But SDPA does not select it here.**
With no explicit backend context the default allocates 2.4-7.3 MB across the same
range, matching flash and mem-efficient and nowhere near the one-head `seq^2`
figure (8.4 MB at 2048 rising to 75.5 MB at 6144).

So on this stack the attention matrix is never materialised, and the hypothesis
is not the explanation for anything measured here. It is **not** refuted for the
RTX 3050: that box ran Windows/WDDM on torch 2.5.1, where flash-attention shape
coverage was narrower and a math fallback is plausible. Confirming or killing it
there needs that hardware.

---

## Determinism

Near, but not bit-identical as reported on the 3050. Repeating two shapes in a
fresh process:

| seq | run 1 | run 2 | delta |
|---|---|---|---|
| 4096 | 2.6578 GB | 2.6493 GB | 0.32% |
| 6144 | 3.9430 GB | 3.9439 GB | 0.02% |

Immaterial against a 16% gap, but recorded rather than rounded away.

---

## What this record does not establish

- **Which** of the four fp32/bf16 buffers stopped being live. Inferred from the
  arithmetic, not observed in an allocator snapshot.
- Anything about the RTX 3050. The 15.0 / 17.8 B/element figures above are
  arithmetic on that record's published ratios, not new measurements.
- **Why** the `+2` copy is retained on the RTX 3050 and not here. This record
  narrows #327 by one stack; it does not explain the retention. torch, trl and
  transformers were each already swapped on the H100 box without reproducing it,
  so the cause is none of those three.
- Whether the ~15% non-logits over-estimate is one term or several. It is flat
  across a 3.1x vocabulary contrast and a 3x sequence range, which argues for a
  fixed-fraction effect rather than drift, but this record does not decompose it.
- Batch scaling at long sequence. Every row here is batch 1.

---

## Reproducing this

`benchmarks/harness/issue395_second_stack_vram.py`, against a released Soup:

```
python benchmarks/harness/issue395_second_stack_vram.py            # the sweep
python benchmarks/harness/issue395_second_stack_vram.py --measure  # section 1
python benchmarks/harness/issue395_second_stack_vram.py --sdpa     # section 4
```

Soup 0.73.3, A10G 23 GB (AWS `g5.xlarge`), Ubuntu 22.04, torch 2.13.0+cu130,
transformers 5.16.1, trl 0.29.1, peft 0.20.0, Python 3.10. Needs
`jinja2>=3.1.0` — the `[dev]` extra may resolve lower and the chat-template
path raises without it. A 24 GB card suffices; the largest arm peaks at
11.85 GB.

## Method note

The first attempt at this sweep was wrong and is worth recording, because the
failure is silent. `data.max_length` **truncates**; it does not pad. Rows that
tokenise short produce a run at their own length regardless of the configured
maximum, so the "real peak" stops responding to `seq` while the prediction keeps
climbing — which reads as a large, clean over-prediction that grows with
sequence, and is entirely an artifact of the harness.

The corrected harness overshoots the row content 3x so truncation lands on the
target, and reads the realised length off the collated batch
(`input_ids.shape[-1]`) into its own column. Every row above was checked that
way: requested and realised sequence length are equal in all ten.
