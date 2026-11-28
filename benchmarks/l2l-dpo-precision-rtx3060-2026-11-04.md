# Measurement record - precision controls for L2L DPO divergence

**Verdict (2026-11-04) - Mixed.** FP32 matched gradient accumulation with pinned and default SDPA. BF16 exceeded the second-backend noise floor after step zero. The kernel probe could not uniquely identify the default BF16 kernel.

Measured by @M1kado0.

## 1. Setup

- RTX 3060 12288 MiB, driver 580.173.02, Xeon E5-2696 v3, 263801688 kB host RAM, Ubuntu 24.04 CUDA 13.0 development container. Both sessions used the same host.
- Python 3.12.3, torch 2.14.1+cu130, transformers 5.19.0, trl 0.29.1, peft 0.21.2.
- `HuggingFaceTB/SmolLM2-135M`, synthetic DPO rows, one microbatch per step, six measured windows. BF16 used BF16 streamed weights and activations with FP32 LoRA gradients. The harness forced FP32 and disabled TF32 to test a precision control. The pending product combination is `training.l2l` with `task: dpo`.

## 2. Method and results

The first session ran duplicate gradient accumulation (GA) and L2L with pinned SDPA MATH in FP32. The BF16 floor came from GA on EFFICIENT_ATTENTION versus GA on MATH. The second session repeated those arms and added unpinned FP32 and replay of the observed SDPA inputs against each available kernel. The BF16 rule checked each gradient tensor and each loss against 1.5 times its own floor.

| Control | Result |
|---|---|
| FP32, pinned MATH | Maximum gradient and loss differences 0 at steps 0–5, both GA-repeat and L2L-versus-GA comparisons |
| FP32, unpinned | Maximum gradient and loss differences 0 at steps 0–5 for both comparisons |
| BF16, step 0 | Gradient difference 1.648e-03 against a floor of 5.371e-03. Loss difference 2.980e-07 against a floor of 5.248e-03. Per-tensor and loss checks passed |
| BF16, steps 1–5 | 238 recorded checks exceeded the tolerance. The largest exceeded the limit by 12.67x |
| SDPA replay | Multiple BF16 backends matched default outputs. Forced MATH differed by 1.953e-03 at the recorded sample points |

## 3. Commands

The harness and launcher live at the evidence commit. The two measured source revisions used the same driver command.

```bash
python benchmarks/harness/l2l_pref_diagnose.py run --device cuda --out /root/results/a8 \
  --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf
```

## 4. Limits and earlier runs

The second session reproduced the first and extended it with default-SDPA controls. The first session retains its results. The fixture supports a precision-sensitive discrepancy, with later steps reflecting already divergent adapter updates. Equal outputs from multiple kernels leave their identities unresolved. The experiment did not identify the first divergent layer or qualify production FP32, default BF16 tolerance, full-run quality or other preference losses.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results
