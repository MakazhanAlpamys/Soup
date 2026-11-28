# FSDP2 LoRA scaling and adapter reload on RTX 3090 and RTX 4090

**Verdict (2026-10-27): mixed.** The accepted FSDP2 cohorts passed the training checks and reloaded the saved adapters exactly. Two GPUs improved throughput in the conservative math BF16 cohort only.

Measured by @kok-o.

## Question and setup

Does FSDP2 actually shard a frozen base during LoRA training, preserve updates and fresh-process adapter reload, and increase throughput from one GPU to two at fixed global batch?

- FP32 cohort on a Linux host with two RTX 3090s (24 GB each), AMD EPYC 7742, 64 GB allocated RAM, driver 580.167.08 and PCIe NODE topology without NVLink. The host reported 270,074,556,416 bytes of total RAM.
- Both BF16 cohorts on a Linux host with two RTX 4090s (24 GB each), AMD EPYC 9354, about 51.5 GB allocated RAM, driver 580.178.04 and no NVLink. The host reported 270,093,586,432 bytes of total RAM.
- Measured stack for all three cohorts included Python 3.12.3, torch 2.13.0+cu130, CUDA 13.0, NCCL 2.29.7, Transformers 5.17.0, Accelerate 1.14.0, PEFT 0.20.0 and TRL 0.29.1.
- `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`. LoRA r=8/alpha=16 on q/v, dropout 0. Each optimizer step uses 8 synthetic sequences of 512 tokens, microbatch 1 per rank. Accumulation is 8 on one GPU and 4 on two. The three precision policies were IEEE FP32, conservative math BF16 and deterministic native Flash BF16.

## Method and results

Each campaign ran two matched fresh-process pairs, reversing the arm order. Each arm used two correctness steps, five warmup steps, ten timed steps and one final inspection step. Throughput uses the actual non-padding tokens over the sum of synchronized step windows from the slowest rank. The gate checked finite loss and gradients, frozen base weights, matching rank data and updates, named local DTensor shards and root/checkpoint adapter reload.

| Cohort | 1 GPU tok/s | 2 GPUs tok/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| RTX 3090, IEEE FP32 | 4216.4633 | 2075.1697 | 0.492159 | 0.246079 |
| RTX 4090, conservative math BF16 | 6932.1733 | 7584.7686 | 1.094140 | 0.547070 |
| RTX 4090, deterministic native Flash BF16 | 8178.1305 | 8102.4617 | 0.990747 | 0.495374 |

The two-rank audit recorded 247,286,720 local elements and 27 FSDP2 modules per rank, with 494,573,440 unique elements total. All 24 accepted fresh-process root/checkpoint reloads matched the 96 adapter tensors exactly, with zero observed eval-logit error.

For the first pair, rank 0 measured the following CUDA peaks.

| Cohort | One GPU allocated / reserved MiB | Two GPUs allocated / reserved MiB |
|---|---:|---:|
| IEEE FP32 | 4123.03 / 4350.00 | 3679.88 / 4586.00 |
| Conservative math BF16 | 2861.83 / 3080.00 | 2647.28 / 3122.00 |
| Deterministic native Flash BF16 | 2437.11 / 2654.00 | 2211.49 / 2682.00 |

## Commands

The harness and original Linux launcher are pinned at the evidence commit. The campaign writes per-arm command receipts for FP32, conservative BF16 and native Flash BF16. The saved evidence includes rank JSON, package freezes, shard and memory inventories and adapter manifests.

```text
benchmarks/h3-fsdp-linux.sh
benchmarks/harness/h3_fsdp_run.py
benchmarks/harness/h3_fsdp_campaign.py
```

## Limits and superseded runs

The October 26 short pilot did not qualify for scaling. The native Flash default failed the cross-world first-step gate and its timings were excluded. The accepted deterministic run changes the execution policy explicitly. The campaigns used joint FSDP2 sharding of the frozen tied embedding and head under `--fsdp full_shard`, a pending change at the CLI publication base. A later CPU regression corrected the `cpu_ram_efficient_loading` variant. The GPU campaigns establish no RAM-efficient-load training result.

All ratios are within their own cohorts. Setup, heavy probes and reload are outside the timed window. Four or eight GPUs, multi-node behavior, 70B, model quality and optimizer/scheduler/RNG resume remain unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
