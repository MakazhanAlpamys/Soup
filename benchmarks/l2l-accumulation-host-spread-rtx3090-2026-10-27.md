# Gate record - streamed accumulation throughput across three hosts

**Verdict (2026-10-27) — Mixed.** The measured throughput ranking depended on the host. Transfer bandwidth tracked the ordinary accumulation wait, and the repeat and causal checks remain incomplete.

Measured by @M1kado0.

## 1. Setup

How much of the spread in ordinary streamed gradient accumulation (GA) can be traced to layer-transfer waits, and does the comparison with layer-major microbatching hold across hosts? `training.l2l` is pending.

All three hosts used an Ubuntu 24.04 CUDA 13.0 development image and Python 3.12 environment with torch 2.14.1+cu130, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2, TRL 0.29.1 and Triton 3.8.0. The model was RAM-streamed `mistralai/Mistral-7B-v0.1` in NF4 with BF16 compute. Both arms used the same 64 Alpaca rows, batch size 1, four microbatches per update, 16 updates, MATH SDPA and deterministic execution.

| GPU and CPU | Visible host RAM (bytes) | Cgroup RAM limit (bytes) | Driver / reported CUDA | Pinned host-to-GPU copy (GB/s) |
|---|---:|---:|---|---:|
| RTX 3090 24 GB, AMD EPYC 7532 | 270,109,917,184 | 129,652,228,096 | 580.173.02 / 13.0 | 19.7–23.6 |
| RTX 3090 24 GB, Intel Core i7-12700KF | 134,920,081,408 | Unverified | 580.126.09 / 13.0 | 10.3–12.6 |
| RTX 3070 Ti 8 GB, Intel Core i7-8700 | 67,188,449,280 | 64,500,006,912 | 595.71.05 / 13.2 | 13.1 |

## 2. Method and results

The throughput runs used four ABBA blocks per host. Separate profile runs placed CUDA events around layer waits, activation transfers and optimizer work, and treated the leftover step time as compute. The host probe measured pinned copies independently. Rates are unprofiled, while step components are means over 15 warm profiled updates.

| Host CPU | GA tok/s | Layer-major tok/s | GA minus layer-major tok/s (95% interval) |
|---|---:|---:|---:|
| AMD EPYC 7532 | 182.3 | 146.4 | +35.9 [+33.9, +38.0] |
| Intel Core i7-12700KF | 130.6 | 299.0 | -168.4 [-171.2, -165.5] |
| Intel Core i7-8700 | 138.6 | 244.1 | -105.4 [-109.1, -101.8] |

| Host CPU and arm | Step (s) | Layer wait (s) | Compute residual (s) | Activation put/get (s) | Optimizer (s) | Layer loads/update |
|---|---:|---:|---:|---:|---:|---:|
| AMD EPYC 7532, GA | 1.80 | 0.38 | 1.41 | 0 | 0.005 | 244 |
| AMD EPYC 7532, layer-major | 2.22 | 0.01 | 2.18 | 0.024 | 0.005 | 64.3 |
| Intel Core i7-12700KF, GA | 2.50 | 1.61 | 0.89 | 0 | 0.002 | 244 |
| Intel Core i7-12700KF, layer-major | 1.10 | 0.04 | 0.98 | 0.074 | 0.002 | 64.3 |
| Intel Core i7-8700, GA | 2.37 | 1.13 | 1.24 | 0 | 0.004 | 244 |
| Intel Core i7-8700, layer-major | 1.35 | 0.015 | 1.28 | 0.052 | 0.003 | 64.3 |

The GA wait per load was 6.6, 4.6 and 1.6 ms at 12.6, 13.1 and 23.4 GB/s of pinned copy bandwidth, respectively. On the EPYC host the layer-major arm's compute residual was 0.77 s higher, more than the measured wait saved. Its cause remains unidentified.

## 3. Commands

The three runners and the analysis script live at the evidence commit. Set `CARD` to `24gb` for the two RTX 3090 hosts or `8gb` for the RTX 3070 Ti. The kernel profiler ran only on the EPYC host.

```bash
bash benchmarks/harness/l2l_ticket10/remote.sh
bash benchmarks/harness/l2l_ticket10/remote_profile.sh
bash benchmarks/harness/l2l_ticket10/remote_kernel.sh
```

## 4. Superseded runs and limits

The CUDA-event pass superseded the first synchronized profile, which missed the GPU stream wait. The kernel profiler perturbed the step and its summed kernel durations included overlapping work. Its trace could not establish a CPU bottleneck. The corrected sensor analysis discarded idle model-load samples that the first analyzer had misclassified as throttling.

The repeat rule requested a fifth block on every host. The four observed blocks per host are descriptive, with unknown power and stall validity checks. CPU, GPU and transfer bandwidth varied together, and the earlier 72.3 tok/s GA result was not reproduced. Loss was outside the qualification criteria. Disk, spill and quality were unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/l2l-ticket10
