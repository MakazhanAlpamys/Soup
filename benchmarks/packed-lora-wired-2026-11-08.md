# Packed-LoRA kernels through `soup train`: end-to-end runs on an RTX 5070 Laptop, 2026-11-08

What [`packed-lora-gpu-tests-2026-11-08.md`](packed-lora-gpu-tests-2026-11-08.md) could not show:
the kernels inside a whole training run, started the way a user starts it. `training.packed_lora`
is switched on in `soup.yaml`, `soup train` checks the run, loads the model, trains, evaluates and
saves, and the result is set beside plain Soup on the same rows with the same seed.

The numbers come from `benchmarks/harness/packed_lora_compare.py` (one invocation per row, the
command lines are below) and are kept in full in
[`evidence/packed-lora-wired-2026-11-08.json`](evidence/packed-lora-wired-2026-11-08.json): the loss
and gradient norm of every optimizer step, the seconds of every step, torch's peak. Nothing here is
a result on any other card, and nothing is a result of more than one run per row.

## Setup

WSL2 Ubuntu on an RTX 5070 Laptop (8 GB, sm_120, driver 616.92), Python 3.13, torch 2.11.0+cu128,
triton 3.6.0, transformers 5.17.0, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2. The GPU was idle
apart from the run (nvidia-smi read 0 MiB before each). The recipe is the harness's: chatml rows
(840 of them, sha256 `eba8a5bc6074eced5adb255f3f8872d942779978171421e13f9c670494479c8`), `max_length`
512, LoRA r 16 / alpha 32 / dropout 0 on all seven projections, lr 2e-4 constant, `adamw_torch`,
clipping 1.0, 4-bit NF4, `gradient_checkpointing: true`, seed 3407. `--batch B --accum A` reads the
same B x A rows per optimizer step in the same order. Code: `soup_cli` of branch
#2650 on top of `3af840ce4` (main with #2623): the runs of the tables below were
made with `src/` as it is in #2650, except that the checks of `soup train` did not yet refuse
`--fsdp` / `--deepspeed` (they run before anything is loaded and touch no training code); the GPU
tests in `gpu-test-runs.md` ran on `src/` as it is in #2650.

```
python benchmarks/harness/packed_lora_compare.py --out plain  --model Qwen/Qwen2.5-1.5B-Instruct --data rows.jsonl --batch 2 --accum 4 --steps 105
python benchmarks/harness/packed_lora_compare.py --out packed --model Qwen/Qwen2.5-1.5B-Instruct --data rows.jsonl --batch 8 --accum 1 --steps 105 --packed '{enabled: true}'
python benchmarks/harness/packed_lora_compare.py --out packed7b --model Qwen/Qwen2.5-7B-Instruct --data rows.jsonl --batch 4 --accum 2 --steps 105 --stop-after 20 --packed '{enabled: true, gpu_budget_mib: 7000}'
```

Tokens per second are the non-pad tokens TRL counts, over the optimizer steps after step 5, divided
by their wall time (CUDA synchronised at both ends of each step).

## Qwen2.5-1.5B-Instruct, 105 steps

Plain Soup runs 2 x 4 rows (a plain batch of 8 does not fit the card); the packed run reads the same
8 rows as one micro-batch.

| arm | tok/s (steps 6-105) | median step | torch peak allocated | step 1 loss |
|---|---:|---:|---:|---:|
| plain Soup, 2 x 4 | 1,569.0 | 2.464 s | 4,107.8 MiB | 1.4442 |
| `packed_lora: {enabled: true}`, 8 x 1 | 5,183.6 | 0.786 s | 5,703.2 MiB | 1.4442 |

3.30 times the tokens per second, 1.39 times the peak memory (no budget: whatever fits stays
resident, and the measurement of the host link read 1.7 GB/s, so the activations stayed on the GPU).
The run printed `[packed_lora] steps: 105 packed, 0 stock`.

Loss, plain against packed, the mean over the steps named:

| steps | plain | packed | packed against plain |
|---|---:|---:|---:|
| 1-50 | 1.2639 | 1.2646 | +0.060 % |
| 51-105 | 1.2278 | 1.2279 | +0.012 % |
| 1-105 | 1.2450 | 1.2454 | +0.035 % |
| last 16 | 1.2651 | 1.2652 | +0.010 % |

Step by step the largest relative difference is 0.65 % (step 2) and the mean is 0.059 %.

### Step 2 is where the two runs differ most

Step 1 is the same loss (the forward is the same to bf16 rounding) and the same gradient norm to
three digits. Step 2 differs by +0.65 %, step 3 by +0.57 %, step 4 by +0.21 %, and the difference
then falls. Plain Soup run twice with the 8 rows split differently (2 x 4 and 1 x 8, the first four
steps) differs from itself by at most 0.10 % at step 1 and by 0.03 % at step 2, so the step-2 offset
is a property of the packed gradients, not of how a batch is split. Those gradients agree with the
stock step to a cosine of 0.9992 to 0.9994 (exact mode) and 0.9941 to 0.9949 (default) on this model
(`test_packed_lora_smoke_gpu.py`), and the first Adam step moves every B entry by the learning rate
whatever its gradient, so a small difference in direction shows at step 2. This record does not show
more than that.

## Qwen2.5-7B-Instruct, a budget that fits the 8 GB card

Plain `soup train` does not take this run. Its hardware-fit gate predicts 11.0 GB of peak VRAM
against 8.5 GB available and exits; with `--allow-oom-attempt` it runs, with torch's peak allocated
memory at 9,844 MiB on a card of 8,151 MiB (the rest was in system memory) and 375 to 379 seconds
per optimizer step (8 x 1 rows, gradient checkpointing on), so only two steps were taken.

With `training.packed_lora` and `gpu_budget_mib: 7000` the plan was made before the model was
loaded: `plan: weights resident, lm_head+embedding on host, activations 16/28 resident (int8 gate/up
stash), est peak 6756 MiB of budget 7000 MiB (2048 tokens), host 3911 MiB pinned`.

| arm | steps | tok/s | median step | torch peak allocated (reserved) | loss, steps 1 / 2 / 3 |
|---|---:|---:|---:|---:|---|
| plain Soup, 1 x 8, `--allow-oom-attempt` | 2 | n/a | 375-379 s | 9,844 MiB (10,312) | 1.5063 / 1.4431 / - |
| packed, 4 x 2, budget 7000 (default numerics) | 20 | 1,399.8 (steps 6-20) | 2.926 s | 6,649.5 MiB (7,088) | 1.5070 / 1.4782 / 1.4887 |
| packed, 4 x 2, budget 7000, `int8_backward: off` | 3 | 1,087.8 (steps 2-3) | 3.770 s | 6,510.7 MiB (6,828) | 1.5070 / 1.4774 / 1.4889 |
| packed, 1 x 8, budget 7000 | 3 | 794.5 (steps 2-3) | 5.263 s | 6,276.9 MiB (7,088) | 1.5056 / 1.4773 / 1.4884 |

The plan's estimate (6,756 MiB) is 1.6 % above torch's measured peak (6,649.5 MiB). The 20-step run
printed `steps: 40 packed, 0 stock` (two micro-batches a step) and its loss fell from 1.5070 at
step 1 to 1.0533 at step 20. Step 1 matches plain Soup's to 0.05 %; at step 2 all three packed
runs sit at 1.477-1.478 and the plain run at 1.443 (+2.4 %): the step-2 offset of the 1.5B table,
larger. Exact mode and the default agree with each other to 0.05 % there, and so do the 4 x 2 and
the 1 x 8 split, so it is not the int8 path and not how the rows are split. It cannot be followed
beyond step 2 here: plain Soup needs six minutes a step on this card.

The gradient of the first step against the stock step on the real 7B checkpoint
(`SOUP_PACKED_LORA_SMOKE_MODEL=Qwen/Qwen2.5-7B-Instruct pytest -m "smoke and gpu" -k
real_checkpoint_matches`, 4 rows of up to 96 tokens, the stock forward fed the same embedding as the
hook feeds it), two runs of the test (the LoRA A matrices are not seeded, so the figures move from
run to run): exact mode loss difference 2.15e-3 and 1.94e-4, gradient cosine 0.99945 and 0.99930;
exact mode with the embedding and lm_head parked in pinned host memory 1.94e-4 and 0.99924;
default numerics (int8 backward, stochastic rounding) 2.61e-3 / 0.99191 and 3.40e-4 / 0.99258. The
same test on the 1.5B (two runs): exact 0.9992 to 0.9994, parked 0.9994, default 0.9941 to 0.9949,
loss difference 0.045 % to 0.25 %. The test's bars are a loss within 0.5 % and a cosine above
0.998 (exact) and 0.99 (default).

## Tiny-model end-to-end tests

`tests/test_packed_lora_e2e_gpu.py` (11 tests, `pytest -m gpu`) trains a random 4-bit Qwen2
(hidden 256, 2 layers) for 20 steps through the real `SFTTrainerWrapper`, as plain Soup and with
`training.packed_lora`: the largest relative difference of the loss over the 20 steps is 2e-4 (mean
5e-5), the adapters each run saved score 6.3015 (plain) and 6.3025 (packed) on a fixed batch when
loaded onto a fresh 4-bit base, two packed runs in one process agree and leave the GPU memory where
it was (19 MiB before, 22 MiB after both), and a left-padded batch runs the stock forward and is
counted (`steps_stock`).

## What this does not show

- One run per row; the 1.5B plain-against-packed comparison is one seed. The 105-step mean loss
  difference (+0.035 %) is a single-run figure, not an interval.
- 7B beyond step 3 against plain Soup: the plain run cannot be taken on this card in a useful time.
- Speed on any other card. The 5070 Laptop's link to the host read 1.7 GB/s in the 1.5B run, which
  is below the planner's threshold for moving activations, so that run kept them on the GPU.
- Qwen3, fp16 on a T4, several GPUs: out of scope (see the docs section).
