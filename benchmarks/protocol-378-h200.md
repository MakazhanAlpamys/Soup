# #378: H200 sequence/checkpointing sweep protocol

This fixes the next measurement protocol before the timed matrix. It is a
second Hopper/current-stack measurement, not a claim to reproduce the older
H100 values. A results record and the documentation curve require the complete
matrix; the short activation probes are not timing evidence.

## Fixed workload

- `meta-llama/Llama-3.1-8B`, local snapshot
  `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (all four original safetensors shards).
- Real `SFTTrainerWrapper.setup` and its `Trainer.train`, resident BF16 base,
  FP32 trainable LoRA, batch 4, accumulation 1, explicit llama3 chat template,
  response-only causal labels, no padding/packing or CPU offload.
- LoRA r=16, alpha=32, dropout=0 on q/k/v/o/gate/up/down; AdamW, LR=2e-5,
  constant scheduler, seed/data_seed=378. The telescope text fixture overshoots
  max_length; the collated [4,S] input is asserted, and the actual input IDs and
  labels are hashed. TRL's equivalent Liger label-storage conversion is handled
  by checking the collated inputs, not comparing dataset column names.
- Independently sweep S={1024,2048,4096,8192} and model checkpointing off/on.
  Checkpointing uses non-reentrant PyTorch recomputation when enabled.
- Four arms: default SDPA, FA2, Liger, FA2+Liger. One fresh process per run.
  Five repetitions per feasible cell, interleaved with a fixed cyclic rotation
  of the arm order. No per-arm block sizes, optimizer, sequence or batch tuning.
- Five warm-up updates followed by **80 measured updates**, optimizer included.
  Synchronized `on_step_begin`/`on_step_end` time excludes model loading,
  the preceding data-loader fetch, between-step logging and final saving.
  This is training-step throughput, not complete CLI wall time. Total Trainer
  seconds and all individual step times are also retained.
- Process CUDA allocated peak includes resident weights, optimizer state and
  warm-up. Steady peak and reserved peak are separate raw fields. No process
  memory fraction or allocator setting is changed. Sampling reports SM clock,
  power, temperature and whole-device utilization, without attributing another
  process's allocations to this run.

## Activation before measurement

Separate eight-step counter probes verify real SDPA/FA2 calls and Liger module
execution for every configuration. Timing processes have those wrappers and
the profiler disabled. Representative S=1024/checkpoint-off probes also record
the top CPU/CUDA profiler events, so the baseline's actual attention backend
can be read rather than inferred from the SDPA flag.

A failed same-shape eight-step activation probe is retained. If it OOMs, its
five timing slots are explicitly marked `not_run_probe_oom`; no throughput or
ratio is invented. Runtime/activation errors stop the campaign for investigation.
Incomplete/mismatched records, changed token masks, changed stacks and
instrumented timings are rejected by the summarizer. A grid may be fully
accounted for while containing OOM cells; completed repetitions and failures
are reported separately.

## Card and dependency provenance

Physical GPU 2: H200, UUID `<redacted>`,
143771 MiB total. An unrelated idle process holds 49850 MiB. Available headroom
is therefore about 90 GiB. OOM means this shared session did not have sufficient
headroom, not that every H200 would fail. No other process, clocks or power
limits are changed.

Python 3.12, torch 2.14.1+cu130, Transformers 5.19.0, TRL 0.29.1, PEFT 0.21.2,
Triton 3.8.0, Liger 0.8.5, FlashAttention 2.8.3.post2. Dependencies live in an
isolated benchmark environment; the existing contributor environment is unchanged.

FA2's published source setup passes `-std=c++17`; torch 2.14's headers require
`std::strong_ordering` from C++20. Building stock failed. Only the four compiler
standard arguments in setup.py were changed to C++20; no kernel/dispatch source
was changed. CUDA 13 nvcc, `FLASH_ATTN_CUDA_ARCHS=90`, `NVCC_THREADS=2`; the
successful resumed build used `MAX_JOBS=32`. Source archive SHA256:
`d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b`.

The compiled FA2 passed explicit FP64 causal GQA forward/backward checks in FP16
and BF16 (B=2/S=128/Hq=4/Hkv=1/D=128). Max relative L2 error in BF16 was 0.306%,
against a declared 1% norm limit; FP16 was 0.038%, against 0.2%. An initial
elementwise comparison to BF16 SDPA math failed one of 32768 K/V-gradient
elements (difference 0.0144 against atol 0.01). This is retained as a limitation,
not called bit-exact parity: the FP64 check reports every max-absolute and norm
error for both implementations. It is a build sanity check, not training-quality
evidence. No throughput run is accepted before the numeric and activation checks.

## Commands

From the source checkout, with the pinned training stack and compiled libraries:

```bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python benchmarks/harness/issue378_flash_math.py --output /tmp/fa2-fp64.json
python benchmarks/harness/issue378_sweep.py \
  --model /path/to/pinned/llama-snapshot --output /tmp/soup-378-h200 \
  --math-evidence /tmp/fa2-fp64.json
```

`--phase probes` stops after activation checks. `--resume --phase timings`
continues a frozen campaign; changed source/harness/model/step settings are
refused. `progress.json`, per-case logs, raw results and the summary remain in
the output directory. Nothing is pushed or published by the campaign script.
