# Streamed and resident Mistral-7B LoRA batch capacity

**Verdict (2026-10-29): mixed.** Resident QLoRA was faster in the measured runs and used more VRAM. Its equal-batch loss check failed. The result for layer-major versus ordinary streamed accumulation varied by host.

Measured by @M1kado0.

## Question and setup

How do streamed and resident LoRA compare at the same effective batch, and what is the largest tested resident microbatch that fits?

- October 28: RTX 3070 8 GB with Core i7-8700K, driver 580.159.03 and cgroup ceiling 45,718,962,176 bytes. Two RTX 3090 24 GB hosts used EPYC 7532 and EPYC 7452, drivers 580.178.04 and 615.71.10 and cgroup ceilings of 91,732,574,208 and 105,856,892,928 bytes, respectively.
- October 29: RTX 5070 Ti 16 GB with AMD EPYC 7V13, driver 595.104.02, host RAM 540,792,193,024 bytes visible in the session inventory. No cgroup limit was captured there. Ubuntu 24.04 CUDA 13.0 images and Python 3.12 were used. Package inventories record torch 2.14.1+cu130, Transformers 5.19.0, TRL 0.29.1, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2 and safetensors 0.8.0.
- `mistralai/Mistral-7B-v0.1`, NF4 base, BF16 compute, rank-8 LoRA, 64 Alpaca rows, cap 256, MATH SDPA and seed 42. All matched-batch arms used four rows per optimizer step. The largest resident-batch arm uses a different batch and step ordering. The layer-major arm uses `training.l2l`, pending in the CLI at the publication base.

## Method and results

Compare ordinary streamed gradient accumulation, layer-major streaming and resident accumulation. Run a resident allocation sweep and time the largest passing tested batch as a fourth arm where possible. Rotate arms forward and backward within blocks. Throughput counts non-padding tokens. The resident equal-batch arm also checks loss against the ordinary streamed reference.

| Host | Streamed accumulation tok/s | Layer-major tok/s | Resident accumulation tok/s | Largest tested resident batch that fit | Largest-batch tok/s |
|---|---:|---:|---:|---:|---:|
| RTX 3070 | 70.0 | 181.4 | 279.6 | 1 | — |
| RTX 3090, EPYC 7532 | 132.8 | 104.2 | 297.2 | 8 | 546.4 |
| RTX 3090, EPYC 7452 | 184.3 | 158.2 | 324.4 | 8 | 654.4 |
| RTX 5070 Ti | 199.0 | 213.1 | 366.6 | 4 | 893.1 |

| Host | Streamed accumulation peak allocated GB | Layer-major peak allocated GB | Resident accumulation peak allocated GB | First failing batch tested |
|---|---:|---:|---:|---:|
| RTX 3070 | 1.15 | 1.13 | 7.10 | 2 |
| RTX 3090, EPYC 7532 | 1.15 | 1.13 | 7.10 | 16 |
| RTX 3090, EPYC 7452 | 1.15 | 1.13 | 7.10 | 16 |
| RTX 5070 Ti | 0.92 | 0.89 | 7.08 | 8 |

Resident equal-batch loss exceeded its allowed band in 10 of 16 steps on the RTX 3070 and 9 of 16 steps on each RTX 3090. It also failed on the RTX 5070 Ti. The RTX 5070 Ti layer-major throughput difference was +14.2 tok/s with a 95% interval of [−6.2, +34.6].

## Commands

The following captured launchers live at the evidence commit. They set the environment, prepare the dataset and invoke the same campaign harness. The evidence contains block orders, individual run records and allocation failures.

```text
benchmarks/harness/l2l_ticket11/remote.sh
benchmarks/harness/l2l_ticket06_valid/remote.sh
benchmarks/harness/l2l_ticket11_campaign.py
```

## Limits and superseded runs

The October 28 overlay hosts could not establish foreign-read validity. GPU clocks also varied across runs. The follow-up on a verified volume completed 56 admitted runs in seven blocks, two blocks short of the repeat rule. It provides a separate observation on the RTX 5070 Ti, and the October 28 timings remain descriptive. The changed batch ordering in the largest-batch arm prevented a matched-loss comparison. The study does not establish quality, spill performance or the batch capacity of larger checkpoints.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks
