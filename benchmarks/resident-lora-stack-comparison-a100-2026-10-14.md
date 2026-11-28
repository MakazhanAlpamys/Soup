# Resident TinyLlama LoRA stack comparison

**Verdict (2026-10-14): mixed.** Soup reproduced the retained PEFT tensors exactly. The external stacks failed strict numerical admission on both cards. A separate default-Boost experiment retained descriptive timings.

Measured by @kok-o.

## Question and setup

Do full-step resident LoRA stacks match a same-dtype PEFT reference, and what step times and memory do the tested stacks exhibit under the recorded GPU policy?

- Strict cohort on an RTX 3090 24 GB (sm86), driver 595.71.05. Separately declared timing cohort on an A100 PCIe 40 GB (sm80), driver 595.91.07, stock 250 W. Both use Linux, Python 3.12.3, torch 2.11.0+cu126, Triton 3.6.0, Transformers 5.17.0, PEFT 0.20.0, Liger 0.8.4 and Unsloth/Unsloth Zoo 2026.10.4. The strict cohort also records xformers 0.0.35 and bitsandbytes 0.50.1. The result reports do not give host RAM or a distro release.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. Dense frozen FP16 or BF16 weights with FP32 rank-16 LoRA adapters on seven projection types. Batch 1, sequence 512, fixed synthetic tokens and AdamW. Dropout, KV cache and checkpointing disabled.

## Method and results

Both cohorts compared each stack with the same-dtype PEFT reference at 50 updates. The strict gate requires all losses to match to three decimals and every retained tensor difference to have relative L2 at most 1%. The separate A100 timings include the full optimizer step with three cycles per cell at observed default GPU Boost. The stack implementations include their own kernels, so time differences cannot be assigned to a single operator.

| Case | RTX 3090 losses matched | RTX 3090 tensor failures / 1,232 | A100 losses matched | A100 tensor failures / 1,232 |
|---|---:|---:|---:|---:|
| Soup FP16 | 50/50 | 0 | 50/50 | 0 |
| Liger FP16 | 1/50 | 1,232 | 1/50 | 831 |
| Unsloth FP16 | 1/50 | 1,232 | 1/50 | 860 |
| Soup BF16 | 50/50 | 0 | 50/50 | 0 |
| Liger BF16 | 1/50 | 1,226 | 1/50 | 1223 |
| Unsloth BF16 | 0/50 | 1,225 | 0/50 | 1224 |

The RTX 3090 protocol admitted 0 of 72 planned timing arms.

| A100 dtype | Candidate | Soup mean ms | Candidate mean ms | Median candidate/Soup ratio | Soup / candidate peak allocated GiB |
|---|---|---:|---:|---:|---:|
| FP16 | Liger | 219.617 | 197.830 | 0.898394 | 3.054 / 3.489 |
| FP16 | Unsloth | 219.830 | 193.966 | 0.888043 | 3.054 / 2.906 |
| BF16 | Liger | 216.344 | 192.961 | 0.885300 | 3.054 / 3.489 |
| BF16 | Unsloth | 215.809 | 190.970 | 0.886838 | 3.054 / 2.906 |

## Commands

The harnesses, Linux lock and fixture manifests live at the evidence commit. Restore the recorded inputs before collecting native operator witnesses and timing.

```text
benchmarks/harness/d2_operator_probes.py
benchmarks/harness/d2_training_baselines.py
benchmarks/harness/d2_audit_training.py
benchmarks/harness/d2_audit_stack_timings.py
```

## Limits and superseded runs

The initial RTX 3090 default-mode comparison failed numerical admission. Its deterministic repeat retains the failures for Liger and Unsloth. The A100 default-Boost campaign superseded a fixed-clock plan that could not complete its control on the available hardware. Its independent descriptive comparison leaves the RTX 3090 gate intact.

A100 GPU clocks varied by arm under the observed policy. These are resident results for a fixed short synthetic fixture. Equivalent-quality speed, longer-run convergence, streaming performance and task accuracy remain open.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/b2923981b7cb938852999459b19c968e5fbeb37f/benchmarks
