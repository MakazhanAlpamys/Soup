# Gate record - L2L qualification for four preference losses

**Verdict (2026-11-04) - Mixed.** DPO passed the narrower readiness rule at every measured window size. ORPO, SimPO and KTO each failed at least one cell. All four methods retain their negative verdict under the initial bit-equality rule.

Measured by @M1kado0.

## 1. Setup

- Two Linux hosts with RTX 3060 12288 MiB. The November 2 host used Xeon E5-2690 v4, 65791040 kB RAM and driver 595.84. The November 4 host used EPYC 7452, 131744572 kB RAM and driver 580.173.02.
- Ubuntu 24.04 CUDA 13.0 development container, Python 3.12.3, torch 2.14.1+cu130, transformers 5.19.0, trl 0.29.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0.
- Real `HuggingFaceTB/SmolLM2-135M` weights and 12 synthetic preference rows per method. The initial run used the trainer's native dtype without recording the effective precision. The follow-up recorded unpinned FP32 and pinned BF16 with FP32 LoRA gradients and TF32 disabled.
- The pending product combination is `training.l2l` with `task: dpo`. ORPO, SimPO and KTO were experimental arms and remain outside the qualified combination.

## 2. Method and results

The initial rule compared ordinary streaming gradient accumulation (GA) twice, L2L and an L2L SFT load-count reference. GA repeated bit for bit in every cell, so the initial rule required exact equality in losses and gradients. All four methods matched the SFT layer-load counts. Foreign-read telemetry was unknown and informational for this compute check.

| Method | Initial maximum absolute gradient/loss difference at window 1 | Window 2 | Window 4 |
|---|---:|---:|---:|
| DPO | 9.033e-03 / 1.900e-02 | 3.418e-03 / 5.804e-03 | 2.563e-03 / 6.409e-03 |
| ORPO | 1.495e-02 / 2.000e-02 | 1.199e-02 / 1.625e-02 | 1.150e-02 / 6.524e-03 |
| SimPO | 1.652e-02 / 1.590e-02 | 1.050e-02 / 5.645e-03 | 6.470e-03 / 3.965e-04 |
| KTO | 2.441e-03 / 3.099e-03 | 9.422e-04 / 2.419e-03 | 5.264e-04 / 7.039e-04 |

The rerun required bit-equal FP32 losses and gradients throughout, and native BF16 step-0 differences within 1.5 times the MATH-versus-EFFICIENT_ATTENTION noise floor for each checked tensor and loss. It also required matching SFT layer-load counts. Later BF16 steps were logged for context.

| Method | Readiness at windows 1 / 2 / 4 | Deciding result |
|---|---|---|
| DPO | Pass / Pass / Pass | FP32 gradient error 0 across all windows. BF16 step-0 checks passed |
| ORPO | Fail / Fail / Fail | BF16 gradient examples versus tensor limits: 1.850e-04 / 1.602e-04, 1.106e-04 / 8.869e-05, 5.188e-04 / 4.807e-04 |
| SimPO | Pass / Fail / Pass | At window 2, BF16 gradient difference 2.232e-04 versus 1.831e-04 limit |
| KTO | Pass / Pass / Fail | At window 4, FP32 gradient difference 0 and FP32 loss difference 2.980e-08 |

## 3. Commands

The launchers and per-arm commands live at the evidence commit. The first driver ran all four methods once per window size. The readiness driver tested both precision modes.

```bash
python benchmarks/harness/l2l_pref_check.py run --device cuda --out /root/results/a6 \
  --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf
python benchmarks/harness/l2l_pref_diagnose.py ready --device cuda --out /root/results/ready \
  --label "RTX 3060 12GB, synthetic data, SmolLM2-135M" --model-path /root/hf
```

## 4. Limits and earlier runs

The narrower readiness gate adds evidence and preserves the original all-window exactness failures. It qualifies DPO on this fixture with the stated comparators. The BF16 tolerance applies to the first step with pinned SDPA. Later BF16 updates, default-kernel tolerance, other datasets and models, throughput and training quality remain outside the qualification.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
