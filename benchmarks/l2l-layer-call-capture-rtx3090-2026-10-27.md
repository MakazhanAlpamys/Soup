# Gate record - layer-call capture across sequence lengths

**Verdict (2026-10-27) — Mixed.** Early capture cut redundant layer loads and kept loss parity. The throughput comparison with streamed gradient accumulation changed sign between the two RTX 3090 campaigns.

Measured by @M1kado0.

## 1. Setup

Can the layer-major trainer capture the first decoder layer's call for each new sequence length without loading the whole model? `training.l2l` is pending.

- NVIDIA GeForce RTX 3070 8 GB and RTX 3090 24 GB. The cgroup RAM limits were 64,762,150,912 and 91,674,902,528 bytes, respectively, in both campaigns. Drivers 580.159.03 and 580.178.04 reported CUDA 13.0.
- Ubuntu 24.04 CUDA 13.0 development image, Python 3.12 environment, torch 2.14.1+cu130, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2, TRL 0.29.1 and Triton 3.8.0.
- `mistralai/Mistral-7B-v0.1` in NF4 with BF16 compute. The RAM-streamed recipe uses the first 64 Alpaca rows, length cap 256, batch size 1, four microbatches per update, 16 optimizer updates, deterministic MATH SDPA and zero LoRA dropout.

## 2. Method and results

The harness counted capture calls, new shapes and layer loads in each window. It timed capture with CUDA synchronization around each call. The first-layer arguments include position tensors sized by sequence length, so the cache still keys on shape. For zero dropout, the new capture stops in the pre-hook of layer 0 once it has the arguments. Dropout continues through full capture to record RNG draws. The other candidate retained full capture and padded rows to a multiple of 64.

Before the change, both cards averaged 3.27 new shapes per update, 103.5 layer loads inside capture and 62 outside it. The first campaign used four ABBA blocks per pair. The two early-capture rates below come from the pair against GA and the separate pair against full capture, in that order.

| Arm | RTX 3070 tok/s | RTX 3090 tok/s | Loads/update | GPU peak allocated | Activation store |
|---|---:|---:|---:|---:|---:|
| Ordinary streamed GA | 97.8 | 184.8 | 244 | 1.15 GB | 0 |
| Layer-major, full capture | 65.7 | 126.3 | 165.5 | 1.13 GB | 191 MB |
| Layer-major, early capture | 83.1 / 82.3 | 154.8 / 158.7 | 64.3 | 1.13 GB | 191 MB |
| Layer-major, full capture with bucket padding | 83.1 | 159.5 | 63.1 | 1.15 GB | 235 MB |

Capture time per update, in RTX 3070 / RTX 3090 order, fell from 1.08 / 0.52 s to 0.031 / 0.016 s. With early capture the largest relative loss differences against GA were 1.6e-7 and 1.7e-7. Bucket padding changed the losses relative to the unpadded recipe by up to 5.4e-3 and 2.8e-3, respectively. A separate synthetic forward/backward probe passed its tolerance on both cards.

| Five-block repeat | GA tok/s | Early capture tok/s | GA minus early capture tok/s (95% interval) |
|---|---:|---:|---:|
| RTX 3070 | 111.1 | 80.4 | +30.7 [+29.3, +32.2] |
| RTX 3090 | 72.3 | 139.7 | -67.4 [-74.6, -60.3] |

## 3. Commands

These runners live at the evidence commit. Set `CARD` to `8gb` or `24gb`. The runners use the baseline checkout at `0e845e25` and the corrected checkout at `87555475`.

```bash
bash benchmarks/harness/l2l_ticket09/remote.sh
bash benchmarks/harness/l2l_ticket09/remote_f1.sh
```

## 4. Superseded runs and limits

The five-block repeat supersedes the original GA versus early-capture timing for the repeat rule. The other timing pairs stopped at four blocks and fell short of that rule. The two campaigns use separate host observations, and the original GA speed ranking applies only to its campaign. The runs had unknown power and stall checks. An image startup failure produced no measurements. The corrected analyzer supersedes the summary's post-change consistency flag, which assumed that each capture still loaded all 32 layers.

The effect applies to zero-dropout capture. Bucket padding was dropped after the loss and activation-store changes. Its noise floor was unmeasured. The recipe leaves task quality, disk streaming and spill unmeasured.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/results/l2l-capture-ticket09
