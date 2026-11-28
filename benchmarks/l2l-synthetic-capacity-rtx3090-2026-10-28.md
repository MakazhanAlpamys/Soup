# Measurement record - streamed L2L on synthetic 70B-shaped weights

**Verdict (2026-10-28) - Mixed.** The synthetic shape completed the measured steps within the recorded VRAM footprint. Unknown foreign-read telemetry limits the repeated timing to a descriptive result.

Measured by @M1kado0.

## 1. Setup

- RTX 3090 24 GB, driver 580.173.02 with CUDA 13.0 reported by the GPU tool, and 105021665280 bytes of host RAM reported by `free -b`. The October 27 single run used RTX 3090 Ti, 24564 MiB, driver 595.80 with CUDA 13.2 reported, and 134890684416 bytes of host RAM reported by `free -b`.
- Linux Ubuntu 24.04 containers, torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 and trl 0.29.1.
- Random Llama-3.1-70B-shaped weights, 80 layers, hidden size 8192 and a 32768-row output head, using a Mistral-7B-v0.1 tokenizer and Alpaca rows. The source fixture is BF16 and streamed in NF4 with BF16 compute. The SFT shape used batch 1, four microbatches, a 256-token sequence cap and LoRA rank 8. `training.stream_source: disk` selected the weight tier. `training.l2l` is pending.

## 2. Method and results

Both timing profiles excluded the first of four optimizer steps. The RTX 3090 campaign used fresh A-B-B-A processes with identical L2L configurations in both arms to estimate repeatability. Ordinary gradient accumulation did not fit the 24 GB configuration used for this comparison. Strict deterministic algorithms and the SDPA MATH backend were in effect.

| Measure | RTX 3090 Ti single run, October 27 | RTX 3090 repeatability campaign, October 28 |
|---|---:|---:|
| Unpadded tokens/s | 5.10 | 29.54 (29.27, 29.73) |
| Step time, s | 61.94 | 10.69 (10.62, 10.78) |
| Allocated / reserved VRAM, GiB | 3.33 / 3.54 | 3.33 / 3.54 |
| Process peak RSS, GiB | 4.51 | 4.79 to 5.05 |
| Pinned activation store, GiB | 0.61 | 0.61 |
| Layer loads per step | 450.3333333333333 | 160.67 |
| Timed process reads | 554.19 GiB across 3 steps | 64.13 GiB per step |
| Validity | Checks absent | 20 runs in 5 blocks. Suspend and power OK. Foreign reads unknown in all 20 |

Within the repeatability campaign, the difference between identical arms was +0.04 tok/s, with a 95% interval of [-0.17, +0.26]. The hosts, code and cache conditions differed between dates, and the study did not isolate the cause of the different readings.

## 3. Commands

The launch scripts and training configurations live at the evidence commit. They prepare the data, build the synthetic checkpoint, install the tokenizer and invoke the saved benchmark driver.

```bash
benchmarks/harness/l2l_synth70b/remote.sh
benchmarks/harness/l2l_synth70b/remote_campaign.sh
```

## 4. Limits and earlier runs

The repeated campaign added a repeatability estimate and validity sensors. The earlier single run remains a separate descriptive measurement. Process I/O counters do not independently verify physical drive traffic or an uncontended drive. The study recorded no trainer spill, left cache-drop success unverified and measured neither a real 70B nor model quality.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results
