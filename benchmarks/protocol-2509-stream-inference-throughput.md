# Protocol: forward-only throughput of the streamed inference mode (#2509)

This rule is written and committed before the first timed run. It decides whether step 1 of #2509 gets built. Step 1 wires the forward-only mode into evaluation and judge scoring, and into batch forward passes on models larger than VRAM. The results go to their own record. This file does not change after the first timed run; a correction is added under a dated heading.

## Question

Is the forward-only mode of #2510 a usable scoring engine on this laptop for a model that does not fit in VRAM? And does it add overhead of its own compared with the training forward?

## Box and code

- RTX 5070 Laptop GPU 8 GB (sm_120), 32 GB RAM, Windows 11, Python 3.12.10, torch 2.14.0+cu130. These are new numbers for this box and are not compared with any RTX 3050 record.
- The code is PR #2510 at `f6997ec62`. sha256 before the first run:
  - `benchmarks/harness/stream_infer_throughput.py`: `50f30b2ab6062e6f5f50cc30f68349ef231de80ffa18b105d6969ef53bf2a1be`
  - `src/soup_cli/utils/stream_inference.py`: `81fe702773a2fbba64ab60ca49c97e467341eadd43b074d684cc4a6c263f4674`
  - `src/soup_cli/utils/layer_stream_runtime.py`: `e318e5554abd41e2810df8b1027d5ea513695531fd0de182a0522324837df93b`

## Arms

There is one process per arm, run in the order M7 then Q14. Each grid cell gets `--warmup 2 --repeats 5`, and the median is reported. Input ids are uniform random from a fixed generator, so these are timing figures only, with no quality claim. Logit correctness is covered by the CPU bit-exact tests in `tests/test_layer_stream_inference_mode.py`, not by this record.

| arm | shard cache | tier | buffers | grid (batch x seq_len) |
|---|---|---|---|---|
| M7 | `unsloth/mistral-7b-instruct-v0.3`, NF4, 32 layers, 3.9 GB | ram (pinned) | 2 | 1,2,4 x 128,512,2048 |
| Q14 | `Qwen/Qwen2.5-14B-Instruct`, NF4, 48 layers, 9.3 GB | disk (direct I/O) | 2 | 1,2,4 x 128,512,2048 |

Q14 runs on the disk tier for two reasons. A 9.3 GB store would need about 10 GiB of pinned RAM on the RAM tier (`gate-901-14b-on-8gb.md`), and the disk tier is how a model larger than the card and the free RAM runs on a laptop.

## Reference

The reference is `probe-rtx5070-what-bounds-streaming.md` §3, the training step of Mistral-7B NF4 on the pinned RAM tier at batch 1 x 512 tokens: **1.053 s** (8 timed steps, spread 1.041 to 1.069 s), **486.0 tok/s**. That step moves 7.40 GB through 61 layer loads and 2 large loads, because the backward reloads each layer, at 16.7 GB/s on the copy stream.

## Gates

- **G1, overhead of the inference path.**
  - Rule: M7 at batch 1 x 512 reaches at least **972 tok/s**, which is 2 x 486.0.
  - Why 2x: a training step at that shape does at least three forward-equivalents of compute (forward, checkpoint recompute, backward) and two weight passes. A forward-only pass slower than half that step means the inference path adds overhead of its own.
  - FAIL: find that overhead before anything else is built.
- **G2, usefulness on a model larger than VRAM.**
  - Rule: the best Q14 grid cell with peak VRAM at or below 7.5 GB reaches at least **500 tok/s**.
  - Why 500 tok/s: 1,000 sequences of 512 tokens (512k tokens) then score in at most 17.1 minutes, an evaluation or judge pass that fits a laptop session.
  - PASS: build step 1.
  - FAIL: record whether the arm is read-bound or compute-bound, from the batch 1 x 128 time against the batch 4 x 2048 time, and stop.
- **Informational, no verdict:**
  - the whole scaling curve;
  - peak VRAM per cell and any cell that runs out of memory (recorded, not retried);
  - Q14 at batch 1 x 128 as the read floor of one forward.

## Validity

- A hold is announced to every live session before the first run and lifted after the last one.
- A run is void if any of the following was true at the start:
  - another process held GPU memory (`nvidia-smi --query-compute-apps` not empty, or more than 600 MiB used before the arm starts);
  - free RAM was below 6 GB;
  - the box went to sleep during the run.
- A void arm is re-run whole, and the void run is listed in the record.
- One laptop, one run per arm. A gate that misses by less than 5% is reported as a miss, with that margin stated; the threshold is not moved.
