<!--
Measurement record, published as measured. The decision rule below was written to
RULE.md by the driver before the first arm ran (2026-10-14T19:12:17+0500 for the
scale-cache experiment, 2026-10-14T19:26:12+0500 for the train-threads experiment)
and is copied here unchanged.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8 GB) on AC power, 34.1 GB host RAM.
Stack (from the `versions` field of every result JSON): Python 3.12.10, soup-cli 0.76.0,
torch 2.14.0+cu130, bitsandbytes 0.50.2, transformers 5.17.0, peft 0.20.0, trl 0.29.1,
accelerate 1.15.0. Soup tree at commit 701f8b84802595c33f67f7395e62fbb0e6d15d7c.
Harness: benchmarks/harness/issue361_nf4_throughput.py, driven by
benchmarks/harness/stream_ab_pairs.py (new in this record).
Raw data: D:/soup-bench/ab-1892-scale-cache/ and D:/soup-bench/ab-train-threads/ (RULE.md,
ledger.json, p<k>-A/B.json and .log, summary.json).
-->

# Gate record - end-to-end A/B of `training.stream_nf4_scale_cache` (#1892)

**Status (2026-10-14): CONFIRMED on one laptop.** Median paired reduction of the
measured window 3.387 %, 95 % bootstrap interval [2.389, 6.936], against a rule of
median >= 2.0 % with a lower bound > 0. The companion experiment (`--train-threads 1`
on top of the cache) shows NO GAIN and that idea is dropped.

---

## 1. Setup

- Model: `unsloth/mistral-7b-instruct-v0.3`, revision `60b99dcbc7f853b7cd18472a607e8d045b96f03e`,
  NF4 with double quantisation, bfloat16 streaming dtype, layer streaming from a pinned RAM
  store (4.138 GB pinned, 2 buffers).
- Training shape: LoRA r=16 on `q_proj,v_proj`, batch 1 x sequence 512, 10 warm-up steps
  (not counted) and 50 measured steps per arm.
- Arms: A = control, B = candidate, one fresh process per arm. Experiment 1: A = default,
  B = `--nf4-scale-cache`. Experiment 2: A = `--nf4-scale-cache`,
  B = `--nf4-scale-cache --train-threads 1`.
- Order: five balanced pairs, AB on even pairs and BA on odd pairs, after one uncounted
  priming A arm per experiment (75.7 s and 73.5 s wall).
- Guards, checked before every arm, no waiting (a failed guard stops the run): free RAM
  >= 10 GiB (12.46-13.25 GiB before each counted arm of experiment 1), on AC power, no
  pytest process running.

## 2. Pre-registered rule

```
metric   = measured-window seconds of an arm (sum of the harness's measured step times)
per pair = reduction % = (A - B) / A * 100, A = control arm, B = candidate arm
verdict  = CONFIRMED if median >= 2.0 and the 95% whole-pair bootstrap lower bound > 0
           (10,000 draws, seed 42); NO GAIN if the upper bound < 0; else INCONCLUSIVE.
No exclusion on timing magnitude; only an operational failure (non-zero exit) voids a pair.
```

The bootstrap resamples the five pair reductions with replacement and takes the median of
each draw; the interval is the 2.5th and 97.5th percentile of the 10,000 medians.

## 3. Experiment 1 - scale cache off (A) vs on (B)

| pair | order | A seconds | B seconds | reduction | A tok/s | B tok/s |
|---|---|---:|---:|---:|---:|---:|
| 0 | AB | 38.918 | 37.955 | 2.474 % | 657.79 | 674.48 |
| 1 | BA | 40.524 | 37.713 | 6.936 % | 631.73 | 678.82 |
| 2 | AB | 38.604 | 37.296 | 3.387 % | 663.15 | 686.40 |
| 3 | BA | 38.536 | 37.615 | 2.389 % | 664.31 | 680.57 |
| 4 | AB | 39.237 | 37.636 | 4.081 % | 652.44 | 680.20 |

Median 3.387 %, 95 % bootstrap [2.389, 6.936]. **Verdict: CONFIRMED.**

| quantity | A (off) | B (on) |
|---|---:|---:|
| peak VRAM allocated | 1.326 GB | 1.772 GB |
| peak VRAM reserved | 1.411 GB | 1.839 GB |
| scale-cache bytes | 0 | 436,207,616 |
| GPU utilisation | 98-99 % | 99 % |
| SM clock during run | 2542-2572 MHz | 2520-2542 MHz |
| GPU temperature (max) | 80-81 C | 80-81 C |

The VRAM cost of the cache is +0.446 GB allocated (1.772 - 1.326), within about 10 MB of
the 436,207,616 bytes (0.436 GB) the cache holds.

## 4. Experiment 2 - train-threads 1 on top of the cache

A = `--nf4-scale-cache`, B = `--nf4-scale-cache --train-threads 1`.

| pair | order | A seconds | B seconds | reduction |
|---|---|---:|---:|---:|
| 0 | AB | 37.295 | 37.502 | -0.555 % |
| 1 | BA | 37.193 | 37.201 | -0.022 % |
| 2 | AB | 38.036 | 38.350 | -0.826 % |
| 3 | BA | 37.626 | 37.635 | -0.024 % |
| 4 | AB | 37.394 | 38.959 | -4.185 % |

Median -0.555 %, 95 % bootstrap [-4.185, -0.022]. **Verdict: NO GAIN** (upper bound < 0).
Peak VRAM, cache bytes and GPU utilisation (99 %) are the same as experiment 1's cache
arm; the maximum GPU temperature (81-82 C) is 1 C higher than in experiment 1's cache arm
(80-81 C). Limiting torch to one CPU thread during training does not help, so that idea
is dropped.

## 5. Commands

```
python benchmarks/harness/stream_ab_pairs.py --tree <worktree at 701f8b848> \
    --weights <hf cache>/models--unsloth--mistral-7b-instruct-v0.3/snapshots/60b99dcbc7f853b7cd18472a607e8d045b96f03e \
    --shards <nf4 shard dir> --out <dir> --label scale-cache \
    --b-args=--nf4-scale-cache --prime
python benchmarks/harness/stream_ab_pairs.py <same paths> --out <dir> --pairs 5 \
    --label scale-cache --b-args=--nf4-scale-cache
python benchmarks/harness/stream_ab_pairs.py <same paths> --out <dir> --label train-threads \
    --a-args=--nf4-scale-cache --b-args="--nf4-scale-cache --train-threads 1" --prime
python benchmarks/harness/stream_ab_pairs.py <same paths> --out <dir> --pairs 5 \
    --label train-threads --a-args=--nf4-scale-cache \
    --b-args="--nf4-scale-cache --train-threads 1"
```

The priming run writes `RULE.md` (with the arm arguments) before the first arm starts, so
its arguments are the ones recorded in each `RULE.md`. Both runs use the driver's default
common arguments `--warmup 10 --steps 50 --no-ceiling`.
`python benchmarks/harness/stream_ab_pairs.py report <dir>` recomputes `summary.json`
from the per-arm JSON files.

## 6. Caveats

- **Exactness is not shown here.** The losses differ from run to run, also between
  repeats of the same arm (for example first-step loss 9.7298-9.7483, last-step loss
  0.1261-0.1359 across the ten counted arms of experiment 1; the minimum and maximum of
  `loss_first` and `loss_last` in its `p<k>-A/B.json` files), because this harness runs
  with the default non-deterministic attention backward. Bit-identical results of the
  cache rest on the CPU and CUDA tests in PR #1892, not on this record.
- **One laptop, one model, one shape.** A single RTX 5070 Laptop on AC, one 7B NF4
  model, batch 1 x 512. Other cards, models and sequence lengths are not measured.
- **Two failed starts preceded the valid run.** Two earlier attempts to start the
  experiment failed before producing data and are not part of any number above; no
  pair was dropped after seeing a result, and every counted pair is in the tables.
- Five pairs bound the interval coarsely; the lower bound 2.389 % is the smallest of the
  five reductions by construction of the resampling.
