# Probe: two reads of a streamed routed expert per step instead of three

Design and scope: #2170. Runtime change: #2158 (`two_read_experts=True`).

**Status: three pre-registered comparisons PASS under strict determinism** on RTX 5090 VMs
with an **emulated** 8,151 MiB envelope:

- Moonlight-16B-A3B on real weights;
- `k2-reduced` and the full Kimi K2 shape on random weights with uniform routing. These two
  rows are capacity and scale checks, not quality or speed claims for a real K2.

In every step of every run, the loss and the LoRA gradient digest are the same in all four
runs, and two reads read exactly 2/3 of the expert bytes. This is not evidence of operation
on a physical 8 GB card, and the K2 rows say nothing about a real K2 checkpoint or real
routing.

## 1. What is compared

`moe_stream_step.py` (one process per run) builds the model with `build_streamed_model`. Each
comparison runs four processes on one host, in this order:

| run | path |
|---|---|
| A1 | three reads (default) |
| B1 | two reads (`--two-read-experts`) |
| B2 | two reads |
| A2 | three reads |

All four share:

- the store, the inputs and the initial adapters;
- `--deterministic strict`: deterministic algorithms, cuDNN deterministic, TF32 off,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

`benchmarks/harness/two_read_compare.py` (on `main`) judges the four records. A shape passes when all of these hold:

- every step's loss (`float.hex`) and LoRA gradient digest are equal in all four runs;
- per step, the expert fetches and expert bytes of B are exactly 2/3 of A's;
- every run's memory verdict is `ok` against the envelope, under the cgroup limit, with swap 0.

Timing is reported per matched pair (B1/A1, B2/A2) on the steps after step 0. Step 0 runs
after a guest page-cache drop; the VM host's cache is not controlled.

| setting | value |
|---|---|
| sequence, batch, LoRA | 512 tokens, 1, r = 8 (attention and shared expert; routed experts frozen) |
| buffers, seed | 2 layer buffers, seed 3 |
| steps | Moonlight 5; `k2-reduced` and K2 2 (one cold, one later) |
| routing | Moonlight: its own router; K2 shapes: `--route uniform` |
| store | NF4 (double quant), `expert_layout="per_expert"` |
| envelope | `--memory-mode emulated`: allocator fraction = (8,151 − context overhead − 256 MiB margin) / card total; NVML process memory sampled every 25 ms (a lower bound) |
| host memory | each run in a cgroup with `memory.max` 32 GiB, `memory.swap.max` 0 |
| stack | torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, Python 3.12.15 |

Moonlight weights: `moonshotai/Moonlight-16B-A3B` at revision
`476b36a473d4467f94469414bef6cee75c9c8172`, with random token IDs.
`k2-reduced` is K2 with 1 dense and 8 MoE layers and 96 experts at top-2, each expert at K2's
geometry. Under uniform routing every expert gets the same tokens per step as in K2
(512 × 2 / 96 = 512 × 8 / 384).

## 2. Results

| Shape | Weights / routing | Host | Steps | Later-step median A1 / A2, s | B1 / B2, s | B/A per pair | Expert fetches per step A → B | Expert bytes per step A → B | NVML peak, MiB | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Moonlight-16B-A3B | real, own router | host A | 5 | 11.25 / 11.14 | 8.02 / 7.88 | 0.70-0.73 | 4,248 → 2,832 | 19.0 → 12.6 GB | 2,822-2,846 | PASS |
| `k2-reduced` | random, uniform | host A | 2 | 12.36 / 12.68 | 8.89 / 8.81 | 0.69-0.72 | 2,304 → 1,536 | 52.3 → 34.9 GB | 5,140-5,154 | PASS |
| full K2 shape (1.03 T params) | random, uniform | host B | 2 | 928.85 / 970.22 | 651.81 / 635.86 | 0.70 / 0.66 | 69,120 → 46,080 | 1.570 → 1.047 TB | 5,776 | PASS |

Per run (seconds per step; step 0 first; all values from the step JSONs):

| Shape | Run | Steps (s) | cgroup peak, MiB | Code |
|---|---|---|---|---|
| Moonlight | A1 | 12.07, 11.59, 11.33, 11.17, 10.61 | 5,040 | `21c2d217` |
| Moonlight | B1 | 8.91, 8.39, 8.13, 7.91, 7.78 | 4,691 | `21c2d217` |
| Moonlight | B2 | 9.02, 8.24, 7.92, 7.84, 7.75 | 4,553 | `21c2d217` |
| Moonlight | A2 | 11.97, 11.39, 11.11, 11.16, 10.68 | 5,040 | `21c2d217` |
| `k2-reduced` | A1 | 13.94, 12.36 | 6,805 | `21c2d217` |
| `k2-reduced` | B1 | 10.36, 8.89 | 6,879 | `21c2d217` |
| `k2-reduced` | B2 | 10.30, 8.81 | 6,804 | `21c2d217` |
| `k2-reduced` | A2 | 14.16, 12.68 | 6,805 | `21c2d217` |
| K2 | A1 | 945.12, 928.85 | 25,771 | `a0b69e6e` |
| K2 | B1 | 646.69, 651.81 | 27,228 | `a0b69e6e` |
| K2 | B2 | 650.72, 635.86 | 27,268 | `a0b69e6e` |
| K2 | A2 | 970.72, 970.22 | 27,260 | `a0b69e6e` |

- **Code column:** the commit each run recorded (`environment.soup_commit`, with
  `soup_dirty: false`). The exact code is in §3.
- **cgroup peak:** includes page cache charged to the run.
- **Hosts:** the rows ran on two different VMs. Compare two reads with three reads within a
  row, not across rows.

## 3. Code that was measured

The runs used commits of the contributor's working branch. Those commits were later
rewritten into #2158 and are **not on any public branch**: GitHub returns "No commit found"
for all three, on the fork and on upstream. They are pinned by the git blob ids of the files
that ran (`git hash-object FILE` reproduces each one):

| commit | used by | `expert_stream.py` | `layer_stream_runtime.py` | `moe_stream_step.py` | `two_read_compare.py` |
|---|---|---|---|---|---|
| `e154ec9a1f43e91a764fe640599d3df32d2ee879` | run A2 CUDA gates (§7) | `6ce097d1a01a` | `f7a0ce785eb7` | `b6772c4b6496` | `1eadd4af887f` |
| `21c2d2179974e60a69c42634c1b1fb5ac6779e29` | Moonlight, `k2-reduced` | `6ce097d1a01a` | `f7a0ce785eb7` | `36c8bb923070` | `6e8bd11de001` |
| `a0b69e6ef1a6cd9c7fe37663346c5717f454a30c` | full K2 | `6ce097d1a01a` | `f7a0ce785eb7` | `bb7f50f5674e` | `6e8bd11de001` |

All three have the same `src/` tree (`f222486de585`): upstream `e79341ef` plus the two-read
patch to `expert_stream.py` and `layer_stream_runtime.py` (+260 / −16). Between those runs,
only the harness changed. `21c2d217` → `a0b69e6e` adds NVML version fields to
`moe_stream_step.py`'s environment record. `two_read_compare.py` (`6e8bd11de001`) is the
version that wrote the three saved `*-compare.json`.

**Against #2158, merged as `c819b3c0` (head `1d9fd46f`).** The squash commit adds the same
lines to the two source files as the head did. Between `1d9fd46f` and `c819b3c0` those files
differ only in one comment in `layer_stream_runtime.py`, from #2205, outside the two-read
path. The two-read path is the same as the one these runs took:

- `layer_stream_runtime.py` adds and removes the same lines (+34 / −6), apart from one
  docstring word.
- `expert_stream.py` differs in four ways, none on the path these runs took:
  - the forward-hook guard: the checkpoint scopes take a guard and call it, and it only
    raises. These runs registered no forward hooks;
  - the contract docstrings;
  - the removal of `OlmoeDecoderLayer` from the admitted decoders: every step run used
    `DeepseekV3DecoderLayer`;
  - the dispatch mode's second parameter renamed from `types` to `_types`, which nothing
    reads.
- #2158 sits on a newer `main` that includes `training.stream_recompute` (#2152). These runs
  predate #2152. Its default keeps the checkpoint recompute, which is what they used.

No step run was repeated on the code of #2158.

## 4. What a reader can rerun

**Level 1: re-analysis of the saved files. Possible for every row.** This directory holds
each comparison's eight inputs. The run records name their step JSON by its path on the
rented host (`/workspace/out/...`), so point the comparator at the directory:

```bash
cd benchmarks/results/probe-expert-stream-two-reads/series-b2-2026-10-22
python ../../../harness/two_read_compare.py \
  --runs moonlight-A1.run.json moonlight-B1.run.json moonlight-B2.run.json moonlight-A2.run.json \
  --out /tmp/moonlight-compare.json --memory-max 34359738368 --swap-max 0 --step-dir .
```

The same command with `k2-reduced-*` in this directory, and with `k2-*` in
`k2-session-2026-10-22/`, gives the other two rows. Each step JSON must hash to its record's
`expect_json.sha256` and carry its `run_id`, so an edited file is refused. Rerun this way,
all three give PASS. `failures`, `timing` and `per_run` equal the saved `*-compare.json`.
So do the inputs, except the `record` and `step_json` paths, which name where the files were
read; `run_id` and `step_sha256` are equal. Only `finished_utc` differs: it is the time of the
rerun. The files are stored without line-ending
conversion (`.gitattributes`), so the hashes hold on any checkout.

This level checks the rule against the recorded numbers. It does not re-measure anything.

**Level 2: a new GPU run. Not possible from the repository today.**

- The step harness options these runs used (`--two-read-experts` and the read accounting,
  `--deterministic strict`, `--memory-mode emulated`) are not on `main`. A
  `--two-read-experts` flag alone would not be enough: the comparator also checks fields that
  `moe_stream_step.py` does not write, such as the strict-determinism setting, the
  environment and numerics records, the initial and gradient digests, the memory envelope
  and verdicts, and the supervisor run id.
- The supervisor that wrote the `*.run.json` (`cgroup_run.py`) and `--drop-caches` are not
  in the repository. Like `--two-read-experts` and its read accounting in `moe_stream_step.py`,
  they are later pieces of #2170, one per pull request. The host setup is described in §5,
  not scripted.
- The measured commits are not public (§3).
- A new run would also need the hardware: a card with more than 8,151 MiB for the emulated
  envelope (these runs used an RTX 5090 32 GB), 32 GiB host RAM, and about 600 GB of disk for
  the K2 shape.

Until those land, a GPU rerun is a new experiment on the code of #2158. Its numbers would not
be checked against this table; they would get their own record.

| Row | Level 1 (saved files) | Level 2 (new GPU run) |
|---|---|---|
| Moonlight-16B-A3B | yes | no: step harness options not upstream |
| `k2-reduced` | yes | no: step harness options and `synth_moe_store.py --shape k2-reduced` not upstream |
| full K2 shape | yes | no: as `k2-reduced`, plus a 533 GB store |
| CUDA gates (§7) | no: their JSONs are not kept, only hashed | no: the checkpoint gate harness is not upstream; the block gate's `--two-reads` is not either |

## 5. Hosts

Both hosts were rented vast.ai KVM VMs (`docker.io/vastai/kvm:cuda-12.9.1-auto`, Ubuntu
22.04) and were destroyed after the session.

| | host A (Moonlight, `k2-reduced`) | host B (full K2) |
|---|---|---|
| GPU | GeForce RTX 5090 32 GB (sm_120) | GeForce RTX 5090 32 GB (sm_120) |
| driver | 580.95.05 | 580.95.05 (open kernel module) |
| guest | 30 vCPU, 73 GB RAM | 46 vCPU, 98 GB RAM |
| disk | 700 GB virtual disk on host NVMe | 700 GB virtual disk on host NVMe |

Host setup before the runs, done as root on the disposable VM:

- **cgroup:** a cgroup v2 series parent with `memory.max` 32 GiB and `memory.swap.max` 0.
  Each run gets a leaf under it. Both limits were read back before the first run.
- **Page cache:** `echo 3 > /proc/sys/vm/drop_caches` before every run's step 0.
- **Admission probe:** a run sized past the limit had to end `oom-killed`, and a run under
  it had to end `ok`.
- **Stack freeze, host B only:** apt update timers and services masked, unattended
  upgrades off, NVIDIA/CUDA packages held. The stack was read back unchanged at the end.

## 6. Earlier sessions (not counted)

- **Series B, 2026-10-21 (RTX 5060 8 GB): Moonlight comparison FAIL.** Losses were equal at
  step 0, but the LoRA gradient digests differed between all four runs, including A1 vs A2
  in the same arm. The cause is the SDPA (memory-efficient) attention backward, which is
  nondeterministic on CUDA without `--deterministic strict`. It was diagnosed on a separate
  host, and every later run used strict mode.
- **Series B second session, 2026-10-22 (host A): full K2 interrupted.** K2 A1 and B1
  ran. B2 refused at start with `NVML/RM version mismatch`; the NVML user-space library no
  longer matched the kernel module, most likely after an unattended driver upgrade during
  the session. Those two K2 runs are not combined with the later K2 session, which froze the
  stack first.
- **A K2-only attempt, 2026-10-22, stopped before any run:** the guest's sshd never accepted
  the key.

## 7. CUDA correctness gates (run A2, 2026-10-21; historical)

Run A2 ran on an RTX PRO 6000 Blackwell Max-Q, at `e154ec9a`. It ran two harnesses, all
bit for bit, and everything passed.

**Block regression** (`expert_stream_gate.py`, blob `3df08cd7ebc1`, `--two-reads`):

- the routed-experts block alone, with no decoder layer and no checkpoint, so there is no
  recompute to skip: with `--two-reads` the block runs through the two-read expert autograd
  node, which reads once in the forward and once in the backward;
- `StreamedExperts` against the resident OLMoE experts block, in three cells: `cpu-fp32`,
  `cuda-bf16`, `cuda-bf16-nf4`;
- the experts of OLMoE layers 0, 7 and 15, 512 tokens;
- output, input gradient and routing-weight gradient equal in every cell;
- in every cell, 128 fetches for the 64 experts hit, two per expert.

**Checkpoint gates** (`expert_checkpoint_gate.py`, blob `a491d9671bb4`):

- three arms: resident, three reads and two reads, under the real checkpoint wrapper;
- four cells: `cuda-fp32`, `cuda-bf16`, `cuda-bf16-autocast`, `cuda-bf16-nf4`;
- two repeats each.

It ran twice:

- three `OlmoeDecoderLayer` layers (0, 7, 15), 512 tokens;
- the full Moonlight-16B-A3B model (27 layers, 26 MoE), built with `build_streamed_model`,
  256 tokens.

The OLMoE checkpoint is `allenai/OLMoE-1B-7B-0924` at revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`.

**The OLMoE decoder gate is historical.** At `e154ec9a` the two-read install admitted
`OlmoeDecoderLayer`. #2158 removed it (#2170): only `DeepseekV3DecoderLayer` is admitted, and
a test expects the refusal for the real OLMoE layer. That result is kept as it was recorded.
It is not evidence for any current code path, and it does not argue for admitting OLMoE
again. The block regression uses OLMoE experts but no OLMoE decoder layer.

The gates' JSONs (`block-two.json`, `gate-olmoe.json`, `gate-moonlight.json`) are listed in
`SHA256SUMS.txt` and are not kept here.

## 8. What is kept here

This directory keeps the 27 files `two_read_compare.py` reads for the three passing
comparisons: the four step JSONs, four `*.run.json` and the `*-compare.json` of each shape,
0.51 MB. `SHA256SUMS.txt` lists every file the series produced, kept or omitted, by sha256
and size: 483 files, 27 kept and 456 omitted (5.5 MB).

The omitted files are:

- host logs: `host/`, apt and dpkg history, three `.gz`;
- copies of `env.log` and `pip-freeze.txt`;
- per-step gradient dumps;
- SSH and VM-log output;
- rental and instance metadata.

Before publication, account and access details were removed from the rental and instance
files, and the hashes listed are of the redacted files.
