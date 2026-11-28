# Moonlight FP8 activation storage and paired evaluation on RTX 4090

Measured by @Batyrkhantzy111 on 2026-11-12 and 2026-11-13.

**Question.** Does storing layer boundaries in block-64 E4M3 with replayed forward routes reduce persistent host memory while meeting a predeclared evaluation margin against BF16 storage?

## Setup and commands

RTX 4090 on Linux for training and final scoring, 24,564 MiB VRAM, driver 595.99.02, and approximately 60.2 GiB cgroup memory on the training host. The training runbook gives Python 3.12.3, PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 5.19.0, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0 and safetensors 0.8.0. Physical host RAM and the kernel release are unreported in the bound training receipt. Moonlight-16B-A3B revision `476b36a473d4467f94469414bef6cee75c9c8172` used a per-expert NF4 cache without double quantization, BF16 compute and the same initial LoRA adapter in both arms.

The runbook, training script and final evaluation launcher live at the evidence commit.

```text
benchmarks/p3-moonlight-quality-runbook.md
benchmarks/harness/p3_deepseek_quality_gate.py
benchmarks/harness/p3_deepseek_quality_train.py
benchmarks/results/p3-moonlight-quality/unattended-eval-v1/code/remote/launch_driver.sh
```

## Method

The control kept layer-boundary inputs in BF16. The candidate stored block-64 E4M3 with FP32 scales and replayed forward routing in backward. The independent fine-tunes ran for 64 steps per arm and length with seed 23 and 4,096 tokens per optimizer step. The final resident-NF4 evaluator used BF16 compute, natural routing and paired accuracy on ARC, BoolQ, HellaSwag and LAMBADA at sequence lengths 1,024 and 4,096. It used one-sided pointwise Clopper–Pearson bounds on the harmful and helpful discordant pairs with a union bound and a one-percentage-point degradation margin.

## Results

| Measure | BF16 storage | FP8 storage with route replay |
|---|---:|---:|
| Persistent layer-boundary store per step (bytes) | 452,984,832 | 240,648,192 |
| Global relative L2 gradient difference versus BF16 in the numerical gate | Reference | 0.04576395133753395 (limit 0.05) |
| Minimum gradient tensor cosine versus BF16 in the numerical gate | Reference | 0.9921371271366706 (floor 0.99) |

The committed final scoring summary records a non-inferiority decision. The paired item-level data needed to independently recompute it were kept separately. This record reports the numerical gate and memory results backed by the retained receipts.

**Verdict — mixed.** The numerical gate passed and the persistent boundary store shrank. Independent verification of the reported quality decision requires the paired scoring data.

## Limitations and superseded runs

Persistent boundary storage excludes other allocations, including the separate route replay buffer. The BF16 resource block at sequence length 1,024 was void. End-to-end performance, total activation memory and uncertainty across training seeds remain unestablished. The final piecewise evaluator superseded an interrupted evaluation and an attempt stopped before scoring. Its source ledger distinguishes reused scored pieces from the pieces completed in the final run.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/p3-moonlight-quality
