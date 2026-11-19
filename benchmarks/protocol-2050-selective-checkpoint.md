# #2050 measurement protocol v1

Written before the first Mistral-7B timing arm. The executable/file and model hashes
are recorded in `freeze.json` before any full-size correctness gate or timing arm.
No performance selection is made from a pilot. Tiny-model harness tests are preparation.

## Fixed scope

- Base Soup: `e05b6f6db42f9cc804fdece2b6eb812432d6aabd`.
- Public model: `unsloth/mistral-7b-instruct-v0.3`, revision
  `60b99dcbc7f853b7cd18472a607e8d045b96f03e`; original safetensors SHA256 recorded.
- Windows, RTX 3090, GPU UUID recorded in the freeze; unchanged 350 W limit.
- Pinned RAM NF4 double-quant store, two buffers, batch 1, seq512 then seq2048.
- Existing #361 recipe: seed3, input seed17, q/v LoRA r16/alpha32/dropout0,
  PagedAdamW8bit lr2e-4, 10 warm-ups and 50 measured synchronized steps.
- Deterministic algorithms, CUBLAS workspace `:4096:8`, TF32 disabled,
  SDPA math for every arm. This is a qualified backend comparison; results do
  not establish flash-attention performance or a portable default.
- Modes: full unchanged streamed checkpoint; qkv and qkvo save only frozen
  base projection GEMM outputs with PyTorch selective checkpoint contexts.
  Adapters, MLP, attention, and dequantized weights are recomputed. Existing
  slot ownership checks and non-reentrant checkpoint remain in force.
- No production config/predictor change, packed kernel change, or #674 dataset use.

## Correctness and execution gates

CPU and actual CUDA NF4 tiny Mistral: GQA, 4 layers recycling 2 buffers,
nonzero LoRA B, full/full repeat, all logits/losses/gradients and two AdamW updates
must be bit-exact. Synthetic 3-layer/1-slot fixture also compares input gradients.
Mutating a cached tensor must fail and removing the installer must restore forwards.

On Mistral-7B, before timing **at either sequence**, run fresh full, full-repeat,
qkv, qkvo processes, two updates each, B filled with 0.01 in every gate arm,
native PagedAdamW8bit. Hash every logit/loss/LoRA gradient/updated parameter.
Require complete equality against full, including the full/full repeat.
These audit runs are not throughput evidence. Then run three separate untimed
operator profiles with the same hash gate; report executed forward/backward
GEMM and NF4 dequantization counts. Profiling must also preserve the audit hashes.
Failure stops the campaign; no approximate equality or silently dropped candidate.

Check actual retained tensors before first backward in every process:
96 QKV or 128 QKVO outputs, exact unique storage bytes:

| policy | seq512 bytes | seq2048 bytes |
|---|---:|---:|
| qkv | 201326592 | 805306368 |
| qkvo | 335544320 | 1342177280 |

No expanded weights may be cached. `cache_bytes` counts unique storage, not views.

## Timing and decision

For each sequence, qkv then qkvo vs a fresh full control, **five AB/BA pairs**
through `stream_ab_pairs.py --harness issue2050_selective_checkpoint.py`.
Pair order: AB, BA, AB, BA, AB. Each arm is a fresh process. No magnitude-based
exclusions or reruns; an operational failure stops and preserves the attempt.

Primary metric: sum of 50 measured step seconds. Paired percent reduction is
`100*(A-B)/A`. Resample whole pairs 10000 times, seed42; statistic is median.
Report ordinary 95% percentile interval using the existing pair harness and
a 98.75% interval for the four planned comparisons (Bonferroni family95%).
The conservative final positive verdict requires median >=2%, lower bound of
**both** intervals >0, correctness passed, and every paired allocated-VRAM
increase <= exact activation bytes +67108864 bytes (64 MiB fixed allocator
tolerance). Reserved VRAM is descriptive. Negative 95% upper gives NO GAIN;
other failures of a positive rule are INCONCLUSIVE/INELIGIBLE as appropriate.
All four comparisons remain in the family, including negative results.

Before every arm: AC, >=10 GiB free RAM, no foreign pytest, target GPU <=65 C,
<1024 MiB used, original 350 W limit. During measurement, require valid target-GPU
samples and <=75 C. No power/service changes. Process timeout30min; whole
campaign budget6h. A refused admission or nonzero exit stops without retries.

Report all pairs, raw rows, exactness receipts, saved bytes, peak allocated and
reserved VRAM, target-GPU temperature/clocks, dequantization behavior, file hashes,
failed starts, and limitations. Positive evidence only motivates a separately
reviewed opt-in config plus predictor follow-up; it does not enable a default.
