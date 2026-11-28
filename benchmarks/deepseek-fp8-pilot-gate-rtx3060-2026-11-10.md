# DeepSeek FP8 activation pilot gate on RTX 3060

Measured by @Batyrkhantzy111 on 2026-11-10.

**Question.** Does a native BF16 run with FP8 block-64 E4M3 activation storage and replayed forward routes pass the tiny DeepSeek numerical qualification for a full-model pilot?

## Setup and commands

One RTX 3060 12 GB (SM 8.6), Linux, CUDA 13.0, PyTorch 2.11.0+cu130, Transformers 5.19.0, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, safetensors 0.8.0 and TRL 0.29.1. The gate receipt does not give host RAM, an OS release or the NVIDIA driver. The measured checkpoint was a three-layer synthetic DeepSeek fixture with BF16 compute and a per-expert NF4 cache with double quantization disabled. The planned target was `deepseek-ai/DeepSeek-V3` at `e815299b0bcbac849fa540c768ef21845365c9eb`.

The gate, test harness and runbook live at the evidence commit. The test receipt records the full selected pytest invocation and its absolute output paths.

```text
benchmarks/harness/p3_deepseek_quality_gate.py
tests/test_p3_deepseek_quality_gate.py
benchmarks/p3-deepseek-quality-pilot-runbook.md
```

## Method and results

One fixed 8 × 64-token window used the same inputs, labels and initial adapter for the BF16-storage control and the FP8-storage candidate. The candidate replayed its forward expert routes during backward. The gate used deterministic algorithms, disabled TF32 and used strict efficient SDPA for both arms. A separate MATH SDPA pass established a reference noise floor. The gate made no optimizer update, and included a low-precision negative control.

| Gradient criterion | Observed | Frozen threshold | Result |
|---|---:|---:|---|
| Global relative L2 | 0.05028628354277727 | ≤ 0.05 | Fail |
| Minimum tensor cosine | 0.9947999141202121 | ≥ 0.99 | Pass |

**Verdict — blocked.** Native execution completed, but the frozen gradient gate refused admission to the full-model study.

## Limitations and superseded runs

The fixture establishes execution and a numerical refusal on a small input. Full-model memory, throughput and quality remain unmeasured in this pilot. Later package and endpoint-test fixes superseded earlier execution checks and preserved the same failed numerical receipt. The scheduled pilot and its training and quality stages remained proposals.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/p3-deepseek-quality
