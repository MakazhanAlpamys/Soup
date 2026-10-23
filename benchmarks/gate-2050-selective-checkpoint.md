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

QKV/seq512 includes pair 4 with a -34.197% reduction; that pair remains in the result. This campaign did not isolate the cause of the short-sequence timing variation. QKV/seq2048 has a 1.916% median against the fixed 2% threshold.

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

Counts below cover two complete updates, separated by original forward and backward (which includes checkpoint recomputation). They count dispatched operations, rather than Python projection calls or inferred FLOPs. Full shape-level counts remain in each profile receipt. An unchanged dequantization count means that saving GEMM outputs did not remove dequantization for backward.

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

See the [pre-timing protocol](protocol-2050-selective-checkpoint.md), [independent audit](results/issue2050-rtx3090-v1/independent-audit.json), [preparation record](results/issue2050-rtx3090-v1/preparation-notes.md) and raw receipts/rows beside it. The campaign froze hashes before its first full-size gate and timing arm, then rechecked them before every subprocess. The independent local audit also checks those frozen source bytes.

[source-snapshot.zip](results/issue2050-rtx3090-v1/source-snapshot.zip) preserves the exact 11 frozen source files and their original line endings; [source preservation receipt](results/issue2050-rtx3090-v1/source-preservation.json) verifies every entry against the freeze. The evidence directory disables Git text conversion so its byte hashes survive checkout. For an exact source replay, overlay the snapshot onto a fresh checkout of the recorded base before running the campaign. The optional audit --tree check expects these exact source bytes.

```powershell
# Dedicated root containing source/, model/, model-manifest.json
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python source/benchmarks/harness/issue2050_campaign.py --root D:/experiment --gpu GPU-...
python source/benchmarks/harness/issue2050_audit.py D:/experiment/evidence-v1 `
  --tree source --output independent-audit.json
```

No #674 checkpoints or datasets were accessed, and no model weights are included in the evidence. AI-assisted implementation and measurement with OpenAI Codex.

## Submission test validation

These checks ran after the scientific timing campaign. Return code 0 means success; the nonzero full-suite attempts stopped at five errors and are retained. The requested empty CUDA mask did not prevent four-device reports in simulated-CUDA tests. Later attempts exposed Windows long-path and temporary-directory admission failures. No production code or scientific result was changed for these attempts. The final broad run reached 38300 passed and 184 skipped before its five-failure stop. The same five cases also failed in isolation on an unchanged checkout of the recorded base and on the candidate; see [baseline verification](results/issue2050-rtx3090-v1/baseline-verification.json). The full suite is not green, and tests beyond the early stop remain unrun.

| test run | return code | pytest summary | evidence |
|---|---:|---|---|
| Targeted CPU/CUDA NF4 subset | 0 | 138 passed in 35.55s | [receipt](results/issue2050-rtx3090-v1/tests-lab.json) |
| Full suite, requested empty CUDA visibility | 2 | 5 failed, 9768 passed, 15 skipped, 5 warnings in 93.97s (0:01:33) | [receipt](results/issue2050-rtx3090-v1/tests-full-cpu.json) |
| Full suite, one GPU, default TEMP | 2 | 14638 passed, 30 skipped, 5 warnings, 5 errors in 106.04s (0:01:46) | [receipt](results/issue2050-rtx3090-v1/tests-full-single-gpu.json) |
| Full suite, one GPU, short basetemp only | 2 | 5 failed, 7776 passed, 11 skipped, 4 warnings in 51.63s | [receipt](results/issue2050-rtx3090-v1/tests-full-single-short.json) |
| Full suite, one GPU, short TEMP and basetemp | 2 | 5 failed, 38300 passed, 184 skipped, 805 warnings in 635.18s (0:10:35) | [receipt](results/issue2050-rtx3090-v1/tests-full-single-temp.json) |
| Five failures isolated on clean base | 1 | 5 failed, 1 warning in 19.41s | [receipt](results/issue2050-rtx3090-v1/tests-failure-comparison-baseline.json) |
| Same five isolated on candidate | 1 | 5 failed, 1 warning in 17.47s | [receipt](results/issue2050-rtx3090-v1/tests-failure-comparison-source.json) |
