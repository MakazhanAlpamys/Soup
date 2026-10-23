<!--
Working measurement record, published as written. The criteria were fixed in a
commit before any run; the launch deviated from the plan (background, not the
owner's foreground window) and that is stated, not smoothed over.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB by nvidia-smi /
8.518 GB by torch, driver 616.92), Intel i9-14900HX (hybrid P/E cores, 32
logical), 32 GB, Windows 11 Pro 26200. Stack: Python 3.12.10 · torch
2.14.0+cu130 · transformers 5.17.0 · peft 0.20.0.
-->

# v0.76.0 gate: opt-in CUDA graph decoding for `soup infer` / `soup bench infer`

`soup infer --cuda-graphs` and `soup bench infer --cuda-graphs` decode a resident
Qwen2/Llama model (optionally under one unmerged LoRA adapter) through CUDA graphs.
This record is the pre-declared gate that decided whether the flag ships. **Every
criterion passed.** It is a NEW baseline on the RTX 5070 box and is not comparable
to any RTX 3050-era number in this folder.

The work ports the maintainer's local experiment, which reported 32.16 -> 62.57
tok/s (+94.55%) with the process **pinned to one CPU core**. The reason this gate
exists is the caveat in that record: unpinned, the experiment's baseline was 14.48
tok/s against 31.85 pinned, and the graph path had never been measured unpinned,
which is how users run it.

## 0. What it gates, and the criteria fixed before any run

Criteria, committed at `8b54b472` (`bench(infer): the pre-declared --cuda-graphs
gate, committed before any run`) before the first measurement:

- **(a)** identical token IDs on every request, and matched conditions (same
  workload, versions, config fingerprints, GPU, dtype, device map, affinity)
  across every run of a scenario;
- **(b)** UNPINNED warm throughput: the SLOWEST graph run is >= 1.20x the FASTEST
  baseline run (a per-run bound, not a pooled mean);
- **(c)** extra peak reserved VRAM, including load and the first capture, <= 256
  MiB over the baseline;
- **(d)** end to end: `soup infer` over 24 prompts of varied, unsorted length
  (21-1339 characters), load and compile included, unpinned, is faster with the
  flag in BOTH graph runs than in BOTH baseline runs, with identical responses.

Trees under test (`results/cuda-graph-decode/cg-trees.txt`): A = `8736def5`, the
branch's merge base with `origin/main`, which has no flag; B = `fb90ecf7`, the
reviewed head of `feat/cuda-graph-decode` after the fix pass. Both were staged
with `git archive`, so neither tree could move under the run.

Model: `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
FP16, unquantized, fully resident (`_load_model` from `commands/infer.py`, the
shipped loader). Workload per harness request: the experiment's fixed 82-token
chat prompt, 128 greedy new tokens, batch 1. Three warm-ups and seven timed
requests per fresh process; CUDA synchronized at both timing boundaries; the
timed span includes tokenization, transfers, generation and detokenization.

## 1. Conditions, including the one that deviated from the plan

- **Launch: a background process of the agent session, not the owner's foreground
  window.** The plan asked the owner to start the gate from a focused PowerShell
  window so Windows would not lower the process priority. The owner instructed
  the session to run it instead. Both arms of every scenario ran under the same
  launch, in A-B-B-A-B-A order, so the ratios (b, d) compare like with like; the
  absolute unpinned numbers may read low against a foreground launch and must
  not be quoted as "what a user gets" without that label.
- Unpinned (`cpu_affinity.effective` = all 32 logical CPUs) for the twelve
  scenario runs and the four CLI arms; `--cpu-affinity 0` for the two bridge runs
  only.
- Quiet box, checked before the start (`results/cuda-graph-decode/quiet-box.txt`):
  GPU 0% / 0 MiB / 50 C, power scheme Balanced, AC, battery 100%, 17 GB free
  virtual memory, box CPU load 0-15%. Two peer agent sessions were asked to
  pause and confirmed idle; the one remaining foreign process was a `pytest -q`
  from another project at 141 MB and 48 s of total CPU, judged negligible.
- `env.ps1` (the experiment's environment file, sourced by the gate) sets only
  cache locations (HF, torch, inductor, pip), `HF_HUB_DISABLE_*`, `TEMP`,
  `SOUP_HOME`, `PYTHONIOENCODING=utf-8` and `TOKENIZERS_PARALLELISM=false`. Its
  `PYTHONPATH` is cleared by the gate; each arm imports from its own staged tree
  and asserts `soup_cli.__file__` before running.
- **The harness cannot run from `benchmarks/harness/` as published**: its path
  containment is rooted at its own grandparent (the experiment workspace), and
  it imports the Soup checkout named by `--source`. The gate script therefore
  copies `cuda_graph_decode_bench.py` and `cuda_graph_decode_compare.py` into the
  workspace and runs those copies. Published bytes: `bench` is the experiment's
  `bench_decode.py` with one blank line added by `ruff` I001 (sha256
  `fb3cc16d…` -> `fc7a9ccc…`); `compare` is byte-identical to
  `compare_results.py` (`6a23c28e…`).
- Prefill-phase diagnostics were off (`--prefill-phase-repeats 0`): no criterion
  reads them, and they import an experiment file this branch does not publish.
- Run time: 08:06Z -> 08:35Z on 2026-09-25, 18 fresh processes. No sleep event
  in that window (the laptop had slept 01:02-11:22 local earlier that night,
  which spoiled one block of the re-record probe in section 4; the gate itself
  was started after wake).

## 2. Results

### Scenario "full" (the base model)

| run | tok/s (pooled) | per-request tok/s | reserved MiB | allocated MiB | GPU util % | power W | SM MHz | proc CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full-a1` | 20.71 | 12.8 - 33.4 | 3002 | 2984 | 26.1 | 39.6 | 2755.8 | 94 |
| `full-b1` | 62.70 | 59.0 - 65.8 | 3044 | 2995 | 74.4 | 69.4 | 2605.5 | 94 |
| `full-b2` | 60.10 | 58.1 - 61.8 | 3044 | 2995 | 72.4 | 68.1 | 2578.6 | 94 |
| `full-a2` | 13.51 | 12.5 - 18.0 | 3002 | 2984 | 16.6 | 34.9 | 2751.1 | 93 |
| `full-b3` | 61.85 | 60.7 - 62.8 | 3044 | 2995 | 69.3 | 64.5 | 2610.9 | 97 |
| `full-a3` | 16.01 | 12.5 - 33.7 | 3002 | 2984 | 20.4 | 37.3 | 2745.6 | 94 |

- (a) PASS: identical IDs on all 42 requests; every matched-condition check true.
- (b) PASS: slowest B 60.10 / fastest A 20.71 = **2.902x**.
- (c) PASS: **+42 MiB** reserved (3044 vs 3002), +11 MiB allocated.
- Pooled: +279.1% (bootstrap 95% percentile interval +225 to +330%, conditional
  on the observed runs, not a paired-trial interval).

### Scenario "lora" (the same base under a synthetic rank-16 LoRA on q/k/v/o, unmerged)

| run | tok/s (pooled) | per-request tok/s | reserved MiB | allocated MiB | GPU util % | power W | SM MHz | proc CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lora-a1` | 17.80 | 12.3 - 20.0 | 3020 | 3000 | 28.9 | 38.0 | 2749.1 | 98 |
| `lora-b1` | 49.19 | 47.4 - 51.9 | 3060 | 3012 | 68.5 | 55.7 | 2353.9 | 91 |
| `lora-b2` | 60.30 | 59.7 - 61.0 | 3060 | 3012 | 79.1 | 69.2 | 2614.1 | 97 |
| `lora-a2` | 15.17 | 9.2 - 20.6 | 3020 | 3000 | 24.1 | 36.3 | 2764.6 | 94 |
| `lora-b3` | 51.10 | 47.6 - 55.6 | 3060 | 3012 | 65.2 | 58.7 | 2402.4 | 95 |
| `lora-a3` | 8.10 | 7.3 - 8.6 | 3020 | 3000 | 12.3 | 31.2 | 2757.5 | 96 |

- (a) PASS: identical IDs on all 42 requests. The adapter is not a no-op: the
  Task-5 identity check recorded that its responses differ from the base
  model's.
- (b) PASS: slowest B 49.19 / fastest A 17.80 = **2.763x**.
- (c) PASS: **+40 MiB** reserved (3060 vs 3020).
- Pooled: +334.6% (bootstrap +261 to +413%, same caveat).

### The pinned bridge to the experiment's published numbers

| run | affinity | tok/s | experiment (2026-09-23, same box) |
|---|---|---:|---:|
| `pinned-a` (baseline) | CPU 0 | 31.29 | 32.16 (A1-A3: 31.29-33.10) |
| `pinned-b` (graphs) | CPU 0 | 65.64 | 62.57 (B1-B3: 54.11-68.50) |

Both inside the experiment's own run-to-run range, with identical IDs in
`pinned-b` against `pinned-a`. So this box reproduces the experiment when
measured its way, and the unpinned scenarios above are a different condition,
not a different machine.

### (d) `soup infer` end to end, 24 varied prompts, 128 tokens each, A-B-B-A

| arm | tree | wall (s, load + warm-up + 24 rows) | in-command Duration (s) | in-command Throughput |
|---|---|---:|---:|---:|
| `a1` | main | 282.2 | 265.6 | 11.6 tok/s |
| `b1` | candidate, `--cuda-graphs` | 101.5 | 57.3 | 53.6 tok/s |
| `b2` | candidate, `--cuda-graphs` | 96.3 | 55.1 | 55.8 tok/s |
| `a2` | main | 131.4 | 115.5 | 26.6 tok/s |

- PASS: identical responses in all 24 rows of all four arms, and both graph
  walls (96-102 s) below both baseline walls (131-282 s), compile included.
- The in-command Duration excludes the one-time warm-up (the summary's clock
  starts after it), which is why `b1`'s wall exceeds its Duration by ~44 s
  where `a1`'s does by ~17 s: that difference is the model load plus the
  capture.

## 3. What the numbers say beyond the verdicts

- **Unpinned, the baseline is the unstable arm.** Its per-run rate spans 13.5 to
  20.7 tok/s (full) and 8.1 to 17.8 (lora); within one run, requests span 12.5
  to 33.7. The graph arm spans 60.1 to 62.7 (full), 4% run to run. The
  experiment's pinning stabilised the baseline at 31-33; the flag stabilises it
  without pinning, because the host has ~1,230 fewer kernel launches per token
  to schedule.
- **GPU utilisation** (device-wide, 1-second `nvidia-smi` samples inside the
  timed spans): 12-29% without the flag, 65-79% with it. Process CPU stayed at
  91-98% in both arms: the baseline burns a core launching kernels; the graph
  arm burns it in the sampling loop.
- Power: 31-40 W baseline, 56-74 W graphs. The flag trades idle for work; it
  does not make the card cheaper per token than it is at full utilisation.

## 4. The re-record question, measured before the gate

A review finding: transformers 5.17 builds a NEW `StaticCache` per `generate()`
call (`_prepare_static_cache`), and cudagraph trees re-record when a static
input's address changes, so every row might re-record, and after
`cudagraph_unexpected_rerecord_limit` (128) decode would silently run eager.
Measured with `benchmarks/harness/cuda_graph_decode_rerecord_probe.py` (one
process, greedy, 32 tokens, the same varied prompt generator as (d)):

| rows | recordings at end | counted re-records | `cudagraph_skips` | reserved | 128-row block tok/s |
|---:|---:|---:|---:|---:|---|
| 256 | 146 | 0 | 0 | flat 3044-3070 MiB | 42.0 -> 48.1 (first vs last 32) |
| 1024 | 341 | 0 | 0 | flat, max 3074 MiB | 42.0 / 46.4 / 47.9 / 48.0 / 47.4 / 47.5 / [sleep] / 53.2 |

About one recording per three rows, and none of them counts toward the limit:
in torch 2.14 a `StaticInputIdxMismatch` re-records but is exempt from
`cudagraph_unexpected_rerecord_limit` (`cudagraph_trees.py`, the block that
increments `num_rerecord`). Throughput does not degrade through 1024 rows and
memory does not grow. The `[sleep]` block is the laptop sleeping inside row 893;
its median row was 0.613 s, in line with its neighbours. A persistent
`StaticCache` (`.reset()` per row, addresses stable by construction) would
remove the re-records entirely; it is a follow-up, not a shipping condition.

## 5. What this record does NOT show

- **A foreground launch.** See section 1. The ratios stand; the absolute
  unpinned rates are a lower bound on what a focused window would show.
- **`soup chat`** (excluded from scope: a growing history recompiles the static
  cache every turn) and **`soup serve`**.
- **Any architecture other than Qwen2**; Llama is accepted by the validator and
  covered by the tiny-model GPU test, not by this measurement.
- **Linux or macOS**, quantized or offloaded models, multi-GPU, `--temperature
  > 0` throughput (every timed request here is greedy; the CLI test pins that
  sampled rows use the requested temperature, and greedy identity is the
  evidence for the cache change, not a sampling measurement).
- **Long `--max-tokens`.** Everything here is 128 tokens (32 in the probe).
  With `keep_inference_input_mutations=False`, each K/V update is a graph output
  plus a copy back, about one extra KV-cache worth of traffic per step, growing
  with prompt + max_tokens. Unmeasured.
- **Batch size > 1.** The CLI is batch 1.

## 6. Raw files

`results/cuda-graph-decode/`: the fourteen harness reports
(`cg-gate-{full,lora}-{a1,b1,b2,a2,b3,a3}.json`, `cg-gate-pinned-{a,b}.json`,
each with per-request samples, correctness IDs and 1-second telemetry), the two
comparisons, the three verdicts, `cli-wall-seconds.json`, `cg-trees.txt`, the two
re-record probes and `quiet-box.txt`. Harness: `harness/cuda_graph_decode_gate.ps1`
(the runner), `_bench.py`, `_compare.py`, `_verdict.py` (the criteria, with
their own tests in `tests/test_cuda_graph_decode_verdict.py`), `_prompts.py`,
`_synthetic_lora.py`, `_rerecord_probe.py`.
