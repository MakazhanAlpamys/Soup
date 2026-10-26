# #2050: selective attention checkpointing on RTX 3090

Four paired comparisons on one RTX 3090, with bit-exact correctness gates.

## Verdict

| sequence | saved outputs | paired median reduction | 95% interval | family-adjusted 98.75% interval | memory gate | verdict |
|---:|---|---:|---|---|---|---|
| 512 | QKV | +2.202% | [-34.197%, +2.915%] | [-34.197%, +2.915%] | pass | INCONCLUSIVE |
| 512 | QKVO | -22.925% | [-32.563%, -5.109%] | [-32.563%, -5.109%] | pass | NO GAIN |
| 2048 | QKV | +1.916% | [+1.501%, +2.056%] | [+1.501%, +2.056%] | pass | INCONCLUSIVE |
| 2048 | QKVO | +3.141% | [+3.027%, +3.432%] | [+3.027%, +3.432%] | pass | CONFIRMED |

The predeclared rule passed for QKVO/seq2048. A separate opt-in/predictor follow-up can evaluate these qualified cases.

Positive reduction means less measured-window wall time. This is a speed measurement, not a model-quality score. Each comparison has five complete fresh-process AB/BA pairs; none were excluded or repeated.

With five pairs, the bootstrap median equals the smallest pair in about 5.8% of resamples, above both lower-tail cutoffs (2.5% and 0.625%). Both intervals therefore span the observed minimum and maximum here; the family correction does not change them. QKVO/seq2048 passes with all five pairs above +3.0% and a median reduction of 3.141%.

QKV/seq512 includes pair 4 with a -34.197% reduction; that pair remains in the result. This campaign did not isolate the cause of the short-sequence timing variation. QKV/seq2048 has a 1.916% median against the fixed 2% threshold.
The slow QKV/seq512 saved arm stayed at 0.798-0.871 s per step with 89% sampled GPU utilization, versus 0.586-0.590 s and 99% in the fast saved arm. This was a sustained slower state across all 50 steps, rather than a few long steps.

QKVO/seq2048 retained 1.342177280 GB of projection outputs. Across its five pairs, the largest allocated peaks were 4.990 GB versus 3.690 GB for full recompute. The per-pair memory gate passed. This measurement adds no public Soup option.

## Correctness and scope

All 14 full-size gate/profile processes passed: seq512/2048, full/full repeat, QKV, QKVO and a separate profile of each mode. Every logit, loss, all 128 LoRA gradient tensors and all 128 updated parameter tensors matched exactly after each of two PagedAdamW8bit updates. B was deliberately nonzero in these gates. The ordinary timing recipe retained its initial zero B and its original seeds.

The tiny tests additionally cover four real Mistral GQA layers cycling through two buffers, a three-layer/one-slot synthetic input-gradient check, cache mutation refusal and removal of all installed forwards.

The installer saves only frozen base-projection outputs. It does not save LoRA products or expanded NF4 weights. The native non-reentrant layer checkpoint, pool refill/prefetch and backward slot-ownership checks remain active.

Every process verified actual unique retained storage and tensor count:

| policy | tensors | seq512 bytes | seq2048 bytes |
|---|---:|---:|---:|
| QKV | 96 | 201326592 | 805306368 |
| QKVO | 128 | 335544320 | 1342177280 |

## Setup and qualifications

- Soup base: `e05b6f6db42f9cc804fdece2b6eb812432d6aabd`; frozen source SHA256s are in [freeze.json](results/issue2050-rtx3090-v1/freeze.json).
- GPU: `GPU-7b0178c9-5f01-96cd-6911-25d9e5fa4126`, RTX 3090, Windows, unchanged 350 W limit.
- Mistral-7B-Instruct-v0.3, unsloth public mirror, pinned revision `60b99dcbc7f853b7cd18472a607e8d045b96f03e`; all source-file hashes recorded.
- NF4 with double quantization, pinned RAM tier, two buffers, batch 1. LoRA q/v, r=16, alpha=32, dropout=0; PagedAdamW8bit lr=2e-4; model seed 3/input seed 17.
- Every timing arm: 10 warm-ups plus 50 synchronized measured steps. The metric is the sum of the existing runner's per-step seconds (stored at 0.0001 s).
- All arms explicitly used deterministic SDPA math, TF32 disabled and CUBLAS workspace `:4096:8`. These results do not establish default flash-attention performance, performance on other GPUs, or a safe production default.
- Operator tracing and tensor hashing run only in separate untimed processes. Their hashes also matched the unprofiled correctness gates. Timing hooks check cache storage once during warm-up, then remove themselves.
- The wrapper fixes telemetry to the selected physical UUID. The reused #361 runner's default sampler otherwise reads the first physical GPU.
- Temperatures, admission records and clocks are in each row. No services, power limits or other GPUs were changed for this campaign.
- The other three RTX 3090s were inventoried, but their activity throughout the campaign was not checked. Only the selected GPU was sampled during the timing arms. Shared host CPU or RAM-bandwidth contention cannot be ruled out; the seq512 slowdown's cause remains unisolated.

## All pairs

Allocated/reserved peaks are decimal GB and include warm-up, following #361. The memory gate uses allocated increase <= exact saved bytes + 64 MiB in every pair. Because original peaks are rounded to 0.001 GB, the audit uses a conservative 1 MB upper rounding allowance when checking this inequality.

| seq/policy | pair/order | full seconds | saved seconds | reduction | full alloc/reserved GB | saved alloc/reserved GB |
|---|---|---:|---:|---:|---|---|
| 512/qkv | 0/AB | 30.1161 | 29.4236 | +2.299% | 1.374/1.546 | 1.568/1.625 |
| 512/qkv | 1/BA | 30.0843 | 29.4447 | +2.126% | 1.374/1.546 | 1.568/1.625 |
| 512/qkv | 2/AB | 30.1100 | 29.4470 | +2.202% | 1.374/1.546 | 1.568/1.625 |
| 512/qkv | 3/BA | 30.3786 | 29.4931 | +2.915% | 1.374/1.546 | 1.568/1.625 |
| 512/qkv | 4/AB | 31.2147 | 41.8891 | -34.197% | 1.374/1.546 | 1.568/1.625 |
| 512/qkvo | 0/AB | 30.9013 | 39.1166 | -26.586% | 1.374/1.546 | 1.699/1.743 |
| 512/qkvo | 1/BA | 30.7736 | 36.3380 | -18.082% | 1.374/1.546 | 1.699/1.743 |
| 512/qkvo | 2/AB | 30.3100 | 37.2586 | -22.925% | 1.374/1.546 | 1.699/1.743 |
| 512/qkvo | 3/BA | 30.3308 | 31.8804 | -5.109% | 1.374/1.546 | 1.699/1.743 |
| 512/qkvo | 4/AB | 30.2189 | 40.0592 | -32.563% | 1.374/1.546 | 1.699/1.743 |
| 2048/qkv | 0/AB | 128.8632 | 126.3122 | +1.980% | 3.69/4.238 | 4.47/5.077 |
| 2048/qkv | 1/BA | 128.5733 | 126.2602 | +1.799% | 3.69/4.238 | 4.47/5.077 |
| 2048/qkv | 2/AB | 128.6287 | 125.9847 | +2.056% | 3.69/4.238 | 4.47/5.077 |
| 2048/qkv | 3/BA | 128.5043 | 126.0425 | +1.916% | 3.69/4.238 | 4.47/5.077 |
| 2048/qkv | 4/AB | 128.5977 | 126.6677 | +1.501% | 3.69/4.238 | 4.47/5.077 |
| 2048/qkvo | 0/AB | 128.7374 | 124.6924 | +3.142% | 3.69/4.238 | 4.99/5.614 |
| 2048/qkvo | 1/BA | 129.0499 | 124.9966 | +3.141% | 3.69/4.238 | 4.99/5.614 |
| 2048/qkvo | 2/AB | 128.7891 | 124.8414 | +3.065% | 3.69/4.238 | 4.99/5.614 |
| 2048/qkvo | 3/BA | 128.9365 | 125.0337 | +3.027% | 3.69/4.238 | 4.99/5.614 |
| 2048/qkvo | 4/AB | 128.9876 | 124.5604 | +3.432% | 3.69/4.238 | 4.99/5.614 |

## Executed operations in the separate profiles

Counts below cover two complete updates, separated by original forward and backward (which includes checkpoint recomputation). They count dispatched operations, rather than Python projection calls or inferred FLOPs. Aggregated counts are retained in the compact process evidence. An unchanged dequantization count means that saving GEMM outputs did not remove dequantization for backward.

| sequence/mode | phase:operator | two-update count |
|---|---|---:|
| s512-full | `backward:aten.bmm.default` | 382 |
| s512-full | `backward:aten.mm.default` | 1592 |
| s512-full | `backward:bitsandbytes.dequantize_4bit.default` | 448 |
| s512-full | `backward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s512-full | `forward:aten.bmm.default` | 128 |
| s512-full | `forward:aten.mm.default` | 706 |
| s512-full | `forward:bitsandbytes.dequantize_4bit.default` | 448 |
| s512-full | `forward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s512-qkv | `backward:aten.bmm.default` | 382 |
| s512-qkv | `backward:aten.mm.default` | 1400 |
| s512-qkv | `backward:bitsandbytes.dequantize_4bit.default` | 448 |
| s512-qkv | `backward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s512-qkv | `forward:aten.bmm.default` | 128 |
| s512-qkv | `forward:aten.mm.default` | 706 |
| s512-qkv | `forward:bitsandbytes.dequantize_4bit.default` | 448 |
| s512-qkv | `forward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s512-qkvo | `backward:aten.bmm.default` | 382 |
| s512-qkvo | `backward:aten.mm.default` | 1336 |
| s512-qkvo | `backward:bitsandbytes.dequantize_4bit.default` | 448 |
| s512-qkvo | `backward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s512-qkvo | `forward:aten.bmm.default` | 128 |
| s512-qkvo | `forward:aten.mm.default` | 706 |
| s512-qkvo | `forward:bitsandbytes.dequantize_4bit.default` | 448 |
| s512-qkvo | `forward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s2048-full | `backward:aten.bmm.default` | 382 |
| s2048-full | `backward:aten.mm.default` | 1592 |
| s2048-full | `backward:bitsandbytes.dequantize_4bit.default` | 448 |
| s2048-full | `backward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s2048-full | `forward:aten.bmm.default` | 128 |
| s2048-full | `forward:aten.mm.default` | 706 |
| s2048-full | `forward:bitsandbytes.dequantize_4bit.default` | 448 |
| s2048-full | `forward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s2048-qkv | `backward:aten.bmm.default` | 382 |
| s2048-qkv | `backward:aten.mm.default` | 1400 |
| s2048-qkv | `backward:bitsandbytes.dequantize_4bit.default` | 448 |
| s2048-qkv | `backward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s2048-qkv | `forward:aten.bmm.default` | 128 |
| s2048-qkv | `forward:aten.mm.default` | 706 |
| s2048-qkv | `forward:bitsandbytes.dequantize_4bit.default` | 448 |
| s2048-qkv | `forward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s2048-qkvo | `backward:aten.bmm.default` | 382 |
| s2048-qkvo | `backward:aten.mm.default` | 1336 |
| s2048-qkvo | `backward:bitsandbytes.dequantize_4bit.default` | 448 |
| s2048-qkvo | `backward:bitsandbytes.dequantize_blockwise.default` | 448 |
| s2048-qkvo | `forward:aten.bmm.default` | 128 |
| s2048-qkvo | `forward:aten.mm.default` | 706 |
| s2048-qkvo | `forward:bitsandbytes.dequantize_4bit.default` | 448 |
| s2048-qkvo | `forward:bitsandbytes.dequantize_blockwise.default` | 448 |

seq512 QKV: 96 fewer backward/recompute matrix multiplications per update; dequantization counts unchanged. seq512 QKVO: 128 fewer backward/recompute matrix multiplications per update; dequantization counts unchanged. seq2048 QKV: 96 fewer backward/recompute matrix multiplications per update; dequantization counts unchanged. seq2048 QKVO: 128 fewer backward/recompute matrix multiplications per update; dequantization counts unchanged.

## Decision and reproducibility

The frozen rule requires median reduction >= 2%, ordinary 95% and family 98.75% whole-pair bootstrap lower bounds above 0, exactness and the memory gate. 10000 resamples, seed 42. Five pairs give a small, discrete bootstrap sample; the bounds describe this experiment and do not guarantee population coverage. All four planned comparisons stay in the correction family.

See the [pre-timing protocol](protocol-2050-selective-checkpoint.md), [independent audit](results/issue2050-rtx3090-v1/independent-audit.json), [compact process evidence](results/issue2050-rtx3090-v1/process-evidence.json) and all four raw pair directories beside it. The campaign froze hashes before its first full-size gate and timing arm, then rechecked them before every subprocess.

The 14 original gate/profile receipts were validated before folding them into one file. Each process records SHA256 over its complete two-update tensor-hash record, per-update summaries, tensor counts and the full-reference comparison. The digest uses UTF-8 JSON with sorted keys, compact separators and ASCII escapes. All four comparisons, operator counts and the original independent-audit result reproduce exactly. The compact data compares digests; the individual tensor hashes and shape-level profiles remain in the [original submitted commit](https://github.com/MakazhanAlpamys/Soup/tree/95a08a21544359ed81ffd0581fc80ce642cbd9e8/benchmarks/results/issue2050-rtx3090-v1).

[source-blobs.json](results/issue2050-rtx3090-v1/source-blobs.json) pins all 11 frozen files by Git blob ID and records their original LF/CRLF form. Retrieve a blob with `git cat-file blob <id>` from the recorded commit; for an exact campaign replay, restore its recorded line endings and overlay the files onto the recorded base. The audit's optional `--tree` check verifies both the Git blob identity and the original frozen SHA256 after reconstructing those line endings. Retained JSON and text have logical digests that survive checkout newline conversion, so no binary source archive or special `.gitattributes` is needed.

```powershell
# From this PR's checkout; audits recorded evidence without a model load
python benchmarks/harness/issue2050_audit.py `
  benchmarks/results/issue2050-rtx3090-v1 `
  --output independent-audit.json
```

To verify source too, add `--tree <original-source-directory>` with the files from commit `95a08a21544359ed81ffd0581fc80ce642cbd9e8`. The current checkout includes later tests and a benchmark installer guard that refuses any streamed layer with `recompute=False` before patching forwards. Those edits were not part of the measured campaign, so `--tree .` correctly rejects them. The frozen sources, evidence and decision rules remain unchanged.

No #674 checkpoints or datasets were accessed, and no model weights are included in the evidence. AI-assisted implementation and measurement with OpenAI Codex.
