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

## The gap is one term, and it can be named

The ratio's flatness is the finding. Decomposing each row into the logits term
(`estimate_logits_bytes`, linear in `seq x vocab`) and everything else, then
solving for the bytes-per-element the hardware actually used:

| model | seq | logits @ 14 B | non-logits | implied B/element |
|---|---|---|---|---|
| SmolLM2-135M | 2048 | 1.409 GB | 0.187 GB | 11.70 |
| SmolLM2-135M | 3072 | 2.114 GB | 0.231 GB | 11.83 |
| SmolLM2-135M | 4096 | 2.819 GB | 0.275 GB | 11.83 |
| SmolLM2-135M | 4352 | 2.995 GB | 0.286 GB | 11.84 |
| SmolLM2-135M | 5120 | 3.523 GB | 0.319 GB | 11.86 |
| SmolLM2-135M | 6144 | 4.228 GB | 0.363 GB | 11.85 |
| Qwen2.5-0.5B | 2048 | 4.356 GB | 0.498 GB | 11.82 |
| Qwen2.5-0.5B | 4096 | 8.713 GB | 0.633 GB | 11.86 |
| Qwen2.5-0.5B | 5120 | 10.891 GB | 0.701 GB | 11.86 |
| Qwen2.5-0.5B | 6144 | 13.069 GB | 0.769 GB | 11.87 |

**mean 11.832, population stdev 0.046**, against a shipped
`LOGITS_BYTES_PER_ELEMENT` of 14.

Ten measurements, two models, a 3.1x vocabulary contrast and a 3x sequence
range, and the implied constant moves by less than half a percent. The
non-logits terms are 0.19-0.77 GB and cannot account for a 0.65-2.0 GB gap. So
the 16% is not drift across the range and not an unmodelled term that grows with
sequence: it is the logits multiplier alone, and it is **stack-specific**.

The v0.72.0 estimate charged 6 B/element; GATE 2 measured 14 on transformers
4.57.x, attributed to `ForCausalLMLoss` holding the bf16 logits, the fp32
upcast, `log_softmax`'s fp32 output and the fp32 gradient at once. On
transformers 5.16.1 the same protocol measures 11.83. That is consistent with
one of those four buffers no longer being live simultaneously, though this
record does not establish which — that would need an allocator snapshot inside
the loss, and is not claimed here.

---

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
- That 11.83 is stable across transformers versions. It is one point on one
  stack; that is the whole reason the grid is per-stack.
- Batch scaling at long sequence. Every row here is batch 1.

---

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
