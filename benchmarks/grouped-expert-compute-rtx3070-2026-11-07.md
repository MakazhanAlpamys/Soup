# Measurement record - grouped expert compute on an RTX 3070

Measured by @M1kado0 on 2026-11-07. This layer diagnostic informed the
bounded grouped compute feature in PR #3428.

## 1. Question and setup

At what routed-row count does grouping expert weight restores or GEMMs
reduce work, and does that show up in wall time?

- Two Linux rentals of the same host with a GeForce RTX 3070 8 GB
  (8,196,653,056 bytes, compute capability 8.6) and Ryzen 5 5600X. The
  base image was `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, upgraded to
  PyTorch 2.11.0+cu130 with torchvision and torchaudio removed. Host RAM
  and distro release were unavailable in the captured environment.
- Driver 610.43.02, PyTorch CUDA 13.0, cuBLAS 13.1.0.3, Python 3.11.13,
  Transformers 5.20.2, safetensors 0.8.0 and psutil 7.0.0. The experiment
  used cuBLASLt, with BF16 reduced-precision reduction and split-K disabled.
- One synthetic BF16 MoE layer used the hidden size 2048 and expert width
  1408 of `ByteDance-Seed/academic-ds-9B` at
  `2a205727f616995ad4aba57806f8e2d0d2f28743`. Eight random-weight experts
  received equal row counts with top-1 routing and route weight 0.5. A
  resident copy of the same layer supplied the numerical reference.

## 2. Method

Each arm ran one forward and one backward on the same input. Sequential
execution restored each expert individually. Grouped restore used a group
width of 4 with native GEMMs. The third arm also grouped GEMMs through
cuBLAS, with native calls through 16 rows. All three arms used per-expert
native backward recomputation.

After a warm-up, one profiled iteration per arm and bucket attributed CUDA
kernels, copies and device time to phases. A separate unprofiled series took
the median of 20 repeats in rotating arm order with forward and backward
synchronizations. The harness compared output, input gradient and
route-weight gradient with the resident layer.

## 3. Commands

The harness and captured launcher scripts live at the evidence commit
(the harness is under `benchmarks/harness/`). The first rental ran harness
revision `c103d6d1`. The second used `7aa1c4ac`, which sized the
activation budget from the largest bucket. Run the following after unpacking
the respective source snapshot and installing the recorded stack.

```bash
pip install -q "transformers==5.20.2" safetensors psutil rich pydantic pyyaml typer
pip install -q torch==2.11.0 && pip uninstall -y -q torchvision torchaudio
PYTHONPATH=/root/soup/src B14G_OUT=/root/out B14G_WORK=/root B14G_TAG=vast-rtx3070 \
  python3 /root/soup/benchmarks/harness/b14_group_diag.py
```

The second rental used this launcher, which selects its three buckets and
records the effective dependency versions.

```bash
bash benchmarks/results/b14-group-diagnostic-20261107/vast-rtx3070-256-512/remote-group2.sh
```

## 4. Results

Selected rows from the first rental include the native cutoff and every bucket
in which grouped GEMM was active. The second rental is shown in full.

| Rental | Rows per expert | Arm | Kernels | Forward / backward kernels | Copies | Forward device µs | Backward device µs | Median wall ms | Bitwise equal |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| First | 16 | Sequential | 385 | 119 / 262 | 54 | 722 | 1243 | 50.19 | Yes |
| First | 16 | Grouped restore | 385 | 119 / 262 | 47 | 716 | 1233 | 49.15 | Yes |
| First | 16 | Grouped GEMM | 385 | 119 / 262 | 47 | 717 | 1235 | 50.31 | Yes |
| First | 32 | Sequential | 385 | 119 / 262 | 54 | 599 | 1232 | 49.99 | Yes |
| First | 32 | Grouped restore | 385 | 119 / 262 | 47 | 588 | 1210 | 49.35 | Yes |
| First | 32 | Grouped GEMM | 373 | 107 / 262 | 63 | 580 | 1215 | 50.45 | No |
| First | 64 | Sequential | 385 | 119 / 262 | 54 | 637 | 1329 | 50.05 | Yes |
| First | 64 | Grouped restore | 385 | 119 / 262 | 47 | 633 | 1323 | 49.45 | Yes |
| First | 64 | Grouped GEMM | 373 | 107 / 262 | 63 | 617 | 1320 | 50.68 | No |
| First | 128 | Sequential | 385 | 119 / 262 | 54 | 797 | 1787 | 50.56 | Yes |
| First | 128 | Grouped restore | 385 | 119 / 262 | 47 | 812 | 1785 | 49.40 | Yes |
| First | 128 | Grouped GEMM | 373 | 107 / 262 | 63 | 804 | 1787 | 51.00 | Yes |
| Second | 128 | Sequential | 385 | 119 / 262 | 54 | 914 | 1968 | 50.42 | Yes |
| Second | 128 | Grouped restore | 385 | 119 / 262 | 47 | 893 | 1974 | 49.58 | Yes |
| Second | 128 | Grouped GEMM | 373 | 107 / 262 | 63 | 860 | 1975 | 50.68 | Yes |
| Second | 256 | Sequential | 385 | 119 / 262 | 54 | 2005 | 3839 | 52.13 | Yes |
| Second | 256 | Grouped restore | 385 | 119 / 262 | 47 | 2017 | 3853 | 51.26 | Yes |
| Second | 256 | Grouped GEMM | 373 | 107 / 262 | 63 | 1448 | 3852 | 52.07 | Yes |
| Second | 512 | Sequential | 385 | 119 / 262 | 54 | 2731 | 6437 | 54.40 | Yes |
| Second | 512 | Grouped restore | 385 | 119 / 262 | 47 | 2745 | 6401 | 53.91 | Yes |
| Second | 512 | Grouped GEMM | 373 | 107 / 262 | 63 | 2763 | 6404 | 55.13 | Yes |

Counts were identical across the smaller native buckets within each arm.
At 32 and 64 rows, grouped GEMM failed bitwise equality for output and
route-weight gradient. The input gradient matched. Those timings remain
diagnostic. At 256 rows, grouped GEMM reduced forward device time by 27.8%
against sequential execution, with bitwise equality and medians of
52.07 ms versus 52.13 ms for the complete layer.

**Verdict — mixed. The grouped path removed 12 forward launches at the tested buckets from 32 rows upward, with no clear wall-time gain and two bitwise mismatches.**

## 5. Limitations

- The harness temporarily admitted this geometry to the grouped GEMM shape
  list. Random weights and balanced top-1 routing leave real checkpoint
  behavior, uneven routing and full-model training unverified.
- The warm shard stayed in the file cache. The diagnostic did not measure
  packed NF4, grouped backward, attention or optimizer updates. The wall
  medians do not provide a paired uncertainty interval.
- Grouped GEMM synchronizes during launch. The experiment did not isolate
  that overhead. Earlier RTX 3070 attempts stopped on PyTorch policy flags
  or a torchvision mismatch. The working PyTorch 2.11.0 runs replaced those
  attempts, and the second valid run extended the first.

Evidence: benchmarks/evidence/grouped-expert-compute-rtx3070-2026-11-07/ (source commit 8bd47cfbe94b)
