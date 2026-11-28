# Partial RAM residency for dense NVMe-streamed training

Measured by @nurseit14 on 2026-10-31, with synthetic pilots on 2026-10-26 and 2026-10-28.

## 1. Question and setup

Does keeping some dense decoder weights in pinned host RAM preserve training
exactness and reduce physical NVMe reads? Does the evidence meet the speed rule?

The dense runs used Linux x86_64, RTX 4090 GPUs reporting 24564 MiB, a 64 GB-class
host, physical NVMe with XFS and direct I/O. The frozen protocol required at least
64 GB assigned RAM and 24 GiB available at preflight. The dense pilot and confirmation
drivers were 595.112.07 and 595.84, respectively, both reporting CUDA 13.2.
The frozen installation used CPython 3.11.16, Torch 2.14.1, Transformers 5.17.0,
PEFT 0.21.2, Accelerate 1.15.0, TRL 0.29.1 and bitsandbytes 0.50.2. The preflight
required glibc 2.39 or newer. The Linux distribution release is unspecified here.

The dense model was `unsloth/Meta-Llama-3.1-8B` at
`e9a141a2091ea561b96483212645a2a05e6f99fc`, BF16 storage and compute.
The control and candidate used LoRA q/v, rank 4, alpha 8, dropout 0, AdamW at
0.001, model seed 7 and adapter seed 3. Six fixed 256-token batches have
255 prediction targets per update. Earlier pilots used a synthetic FP32
eight-layer Llama and 128-token batches.

## 2. Method

Both dense arms used pinned buffers, read-ahead 1 and direct I/O. Partial RAM
used a 16 GiB accounted allocation envelope, with a 6 GiB fixed reserve and
4 GiB resident allocation cap. Each fresh ABBA arm ran six updates, with two
warmups. Exact loss, gradient hashes, final adapter hashes and real adapter
updates were mandatory. Physical reads required direct-read calibration and a
deliberate foreign-reader control. Suspend, power, foreign reads, clocks and
exclusive GPU ownership gated timing. The instrumented window includes
numerical checks and hashes so its read and time boundaries agree.

## 3. Results

| Run | Completed work | Numerical result | Read or timing outcome |
|---|---|---|---|
| Synthetic application-read pilot | 5 ABBA blocks, 20 arms, 80 measured steps | Exact | Application bytes/step 729917472 → 547438104. Timing descriptive only |
| Synthetic calibrated pilot | 5 ABBA blocks, 20 arms, 80 measured steps | Exact | Mean process physical bytes/step 729956352 → 547467264. No speed verdict |
| Dense 8B pilot | 5 ABBA blocks, 20 arms, 80 measured steps | Exact, 8/32 layers resident | Physical process bytes/step 26,173,685,760 → 19,630,264,320. Instrumented 42.53 → 57.49 tokens/s, descriptive only |
| Dense 8B confirmation | 22/80 arms, 88 measured steps | Exact on completed arms, 6 CUDA controls passed | Arms 14 and 19 void on foreign reads. Arm 22 interrupted and excluded |

In the dense pilot the measured read reduction was 25%, with unchanged VRAM
peaks and complete calibration, clock and ownership checks. The frozen rule
required 20 blocks. The separate confirmation stopped early after two
foreign-read failures and supplies no usable confirmation estimate.

**Verdict — no verdict.** Dense numerical parity and lower physical reads were observed, with insufficient valid repetitions for a speed verdict.

## 4. Commands

The measured launch scripts, frozen comparison gate and replay live at the
evidence commit. The launch scripts supply the recorded absolute deadlines.
The last two commands recompute the evidence gates from the retained packets.
The synthetic pilots used the earlier comparison harnesses in that same tree.

```bash
benchmarks/c3/compare.py
benchmarks/c3/isolated_compare.py
/data/c3/env/bin/python -m benchmarks.c3.isolated_compare --mode cuda-representative --prepared /data/c3/repo/.c3_prepared/prepared.json --output /data/c3/evidence/comparison --deadline-unix-s 1793427067.966808
/data/c3/env/bin/python -m benchmarks.c3.isolated_compare --mode cuda-representative --abba-blocks 20 --prepared /data/c3/repo/.c3_prepared/prepared.json --output /data/c3/evidence/comparison --deadline-unix-s 1793445153.6265
python benchmarks/c3/replay_evidence.py benchmarks/c3/evidence/20261031_attempt017 --summary
python benchmarks/c3/replay_evidence.py benchmarks/c3/evidence/20261031_attempt019 --summary
```

## 5. Limits and run history

Earlier startup, setup, staging-alignment, containment and supervisor failures
remain recorded. The calibrated synthetic pilot advanced the instrumentation.
The dense pilot was followed by a separate confirmation and keeps its original
insufficient-repetitions outcome. Process-lifetime RSS includes setup and cache
fill. The dense timing
uses ordinary streamed backward. The confirmation's tiny CUDA Trainer control
covers the RAM-only layer-to-layer integration and cleanup, with no 8B
layer-to-layer throughput evidence. `training.stream_partial_ram` and
`training.l2l` are pending.

Evidence: https://github.com/SoupTeam/soup-rnd/tree/96df4325d422397384d21bb21d492843c67fb84e/benchmarks/c3
