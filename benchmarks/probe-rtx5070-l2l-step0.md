<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any run on either fixture, which is the point of
it: a rule written after the numbers are in is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (C: on a CPU root port, D: behind
the chipset). Windows 11 Pro 10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2.
Soup: branch probe/l2l-step0, cut from origin/main at 6879f323, worktree
C:\Users\user\projects\Soup-l2l. src/ is byte-identical to 6879f323; only
benchmarks/ and one test file change. soup_cli.__file__ is stamped in every
block's JSON.
Harness: benchmarks/harness/l2l_probe.py (CLI), l2l_schedule.py (the
layer-major step and its GA reference), l2l_activations.py (pinned host store +
unbuffered spill), l2l_box.py (box stamps, suspend watch), l2l_rule.py (the
verdict, recomputed from the committed JSON alone), l2l_session.sh (one process
per block). It reuses stream_probe.py's build(), Instruments, gpu_facts and Sink.
-->

# Probe — does layer-major micro-batching (L2L) work on the shipped streamed runtime? (L2L step 0)

**Status: rule committed 2026-10-02 (`7a2127c1`), BEFORE the run. Amended once before any
F1/F2 run (this commit): V2 compares the loads and bytes per step of L2L at k = 1 with the
shipped step instead of their times, and I6 reports the time ratio. Reason: a smoke run on
`HuggingFaceTB/SmolLM2-135M` (not a fixture) showed L2L at k = 1 10-20% FASTER than the
shipped step with identical loads (57/step), identical bytes and bit-identical gradients:
the ±5% time band would have voided G2-G4 for a difference that is not a harness error.
Run 2026-10-07 11:28-13:22Z (§5): G1 and G2 PASS, G3 and G4 NO VERDICT (V4 on L2L arm A
at k = 1); §6.**

---

## 0. The question

Layer-major micro-batching means: load and dequantise each decoder layer ONCE per step, run all
k micro-batches through it, park each micro-batch's boundary activation in host RAM, and walk
the backward in reverse the same way (plan.md § "Giant MoE on this laptop", step 0;
Pudipeddi et al., arXiv 2002.05645, is the prior art). Today's streamed step re-reads the whole
model per micro-batch under gradient accumulation. This probe asks four things:

1. Does L2L give the SAME LoRA gradients as gradient accumulation over the same k micro-batches?
2. Does the read hide under the compute?
3. Does the step pay for itself?
4. Does VRAM stay flat in k?

It measures the mechanism with zero changes to `src/`: the harness drives the shipped
`StreamedDecoderLayer`s, buffer pool, prefetcher and `AsyncDiskSource` in layer-major order.

What it does not claim, stated before the numbers:

- On this 70B shape, L2L is expected to be no faster than the largest plain batch that fits.
  R2 measures that batch (`probe-rtx5070-batch-control.md`).
- L2L's value is that VRAM stays flat in k, which the 671B/1T MoE models need. No number for
  those models is claimed here.

## 1. Fixtures, arms and order

**F2 — timing and correctness.**
- Weights: `D:/synth/llama-70b-shape-v32k-f4`, the synthetic Llama-3.1-70B shape: random values,
  32k untied head, NF4.
- Store: the default one-root store
  `C:\Users\user\.soup\layer-stream\D___synth__llama-70b-shape-v32k-f4` (NOT the `.refit` copy
  beside it).
- Streaming: disk tier, direct I/O, `read_ahead 2`, 2 layer buffers, seq 512 per micro-batch,
  one drive.
- LoRA: r 8 on q/k/v/o, dropout 0.0, adapter seed 3. Input seed 17. Optimizer `PagedAdamW8bit`,
  lr 1e-4.

**F1 — correctness only.** `unsloth/mistral-7b-instruct-v0.3`, NF4, RAM tier, the same LoRA.

**One L2L step:**
1. Call the shipped prime hook (the embed load + `prefetcher.prime()`).
2. Embed all k micro-batches.
3. Forward: layer 0..L-1, each over all k micro-batches under `no_grad`, storing each layer's
   input.
4. Head: per micro-batch, `norm` + `lm_head` + the model's own `loss_function`, the summed loss
   divided by `num_items_in_batch` = k x 511, then its backward.
5. Backward: layer L-1..0, each over all k micro-batches, one forward+backward per (layer,
   micro-batch) with the wrapper's checkpoint off, so there is no third forward.
6. One `optimizer.step`.

The sequence of layer loads is the same as today's step. A repeated call on a loaded layer
loads nothing (`StreamPrefetcher.advance`).

**Arms.**

| arm | what it removes | judged? |
|---|---|---|
| A | nothing: the shipped reads | yes |
| B | the source read and the H2D copy (`Instruments.nocopy`) | timing-only: computes garbage |
| C | the NF4 dequantisation (`Instruments.nodequant`) | timing-only: computes garbage |
| D | both | timing-only: computes garbage |

- Before every arm the adapters are restored from one snapshot and a fresh optimizer is built,
  so no arm inherits another's garbage.
- Each arm: 1 warm-up + 3 timed steps. A step = forward + backward + `optimizer.step` +
  `zero_grad` + `synchronize`.

**Blocks** (one process each, in this order):

| # | file | content |
|---|---|---|
| 1 | `correctness_f1.json` | F1, k = 4: GA, GA again, L2L |
| 2 | `correctness_f2.json` | F2, k = 2: the same |
| 3 | `timing.json` | F2: `plain` A/B at batch 1 (today's step via `stream_probe.run_steps`); `l2l` A/B at k = 1, 2, 3, 4, 8, 16; `l2l` C/D at k = 4 |
| 4 | `spill.json` | F2, k = 16, arm A: `pinned` (every activation in pinned RAM) against `spill` (layers 0-39 through `D:\soup-l2l-spill`, unbuffered; layers 40-79 pinned, i.e. a k = 8 budget) |

If a correctness block's GA-vs-GA comparison is not exact, it re-runs once in a fresh process
with `--deterministic`, writing `correctness_f<n>_det.json` (§2, G1).

**Order inside blocks 3 and 4:** two rounds.
- In round 0: plain A, plain B; then L2L at k ascending, A before B; then C, D.
- In round 1: plain B, plain A; then L2L at k descending, B before A; then D, C.
- Block 4: round 0 = pinned, spill; round 1 = spill, pinned.

**Control cited, not re-run.** R2's SINGLE rows (`probe-rtx5070-batch-control.md`, rule commit
`5a4df9be`, run by soup-6d) give tok/s at batch 1/2/3 and the largest batch that fits. Its code
is the head of PR #1536 (`2c6ed087`), whose one-drive reader is the PR's per-drive dispatcher
with one drive, so it is not byte-for-byte this branch's reader. This record's own plain
batch-1 arms are the same-session control.

## 2. The decision rule, written before the run

**Values.**
- Arm time = mean of its 3 timed steps.
- The value used = mean of the two rounds.
- T(k) = 512 k / value: tok/s.
- P1 / P1B = the plain batch-1 step, arms A / B.

**Gates — build step 1 (L2L in the streamed trainer, SFT first) only if ALL hold:**

- **G1 (correctness).**
  - On F1 at k = 4 and on F2 at k = 2, every LoRA gradient tensor and every per-micro-batch loss
    of the L2L step is `torch.equal` to the GA reference over the same k micro-batches.
  - The GA reference is k calls of the shipped `model(input_ids, labels, num_items_in_batch=N)`
    with N = k x 511, then `backward`.
  - Valid only if a second GA reference in the same process is also `torch.equal` to the first
    (A-vs-A).
  - LoRA B is made non-zero before the comparison (seed 23, scale 0.02, `bitexact.py`'s
    convention), and every compared gradient must contain a non-zero element.
  - **Fallback:** if A-vs-A is not exact, the block re-runs in a fresh process with
    `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. If A-vs-A is
    still not exact there (or deterministic mode raises), G1 has NO VERDICT on that fixture, and
    the record gives max |Δ| per tensor.
- **G2 (the read hides).** T_A(16) ≥ 0.8 x T_B(16).
- **G3 (it pays).** T_A(16) ≥ 1.3 x T_A(1).
- **G4 (VRAM flat in k).** Peak allocated, arm A at k = 16, max over rounds, ≤ the same at
  k = 1 + 0.5 GB.

**Validity — when a row fails, the gates that depend on it have NO VERDICT; reported as measured:**

- **V1.** `runtime.source.direct_io` is True in blocks 2-4.
- **V2.** L2L at k = 1 moves exactly the work of the shipped batch-1 step, in arm A and in
  arm B: layer + large loads per step within 0.5 and bytes moved per step within 1 MB of P1 /
  P1B. This shows the harness measures the shipped step's reads; G1 shows its arithmetic.
  G2-G4 depend on it. *(Amended before any F1/F2 run — see the status line.)*
- **V3.** At each block's start:
  - free physical memory ≥ the block's largest pinned activation bytes + 4 GB;
  - before the model is built, `nvidia-smi` reports ≤ 500 MiB in use, and its compute-apps list
    holds no other PID.
  - An arm whose pinned store cannot be allocated, or whose k does not fit the free memory, is
    SKIPPED and recorded. Without k = 16, G2-G4 have no verdict. The harness never silently
    lowers k.
- **V4.** Per (block, arm, k): |round 0 − round 1| / mean ≤ 5%. A gate that uses an arm failing
  V4 has no verdict.
- **V5.**
  - The box is on AC for every 2-s sample of every arm.
  - There is no gap > 30 s between samples (wall clock).
  - Otherwise the arm is void, and a gate that uses it has no verdict.
- **V6.**
  - No other process reads 1 GB or more during an arm (per-process read bytes; the
    foreign-reader convention of R2's rule — R2's attempt 1 lost an arm to SearchIndexer.exe
    reading 4.94 GB).
  - Otherwise the arm is void, and a gate that uses it has no verdict.

No row compares an absolute time with an earlier record: R1's validity row fired three times on
C:'s own 3.5-4.7 GB/s spread. Every gate here is relative, within one session.

**Informational (no gate):**

- **I1.** T_A(2) and T_A(3) against R2's SINGLE batch 2/3 tok/s and its largest batch that fits.
  Cited, with the code caveat of §1.
- **I2.** Dequant share inside L2L at k = 4: (B_4 − D_4) / B_4; and C_4 against A_4.
- **I3.** L2L's own overhead: B_16 / (16 x P1B).
- **I4.** Spill: T_A(spill, 16) / T_A(pinned, 16) in block 4. ≥ 0.95 reads "spill is free at
  this ratio". Stamps: unbuffered open, bytes written and read per step = 40 x 16 x 8 MiB.
- **I5.** Per k: step time, tok/s, peak allocated/reserved, loads/step, bytes/step, implied
  GB/s, activation bytes (pinned/spilled), time blocked on the store.
- **I6.** L2L at k = 1 against the shipped step, time ratio per arm (A, B). L2L's backward
  runs each layer once without the checkpoint wrapper, so a ratio below 1 is expected.

**Predictions, written before the run** (§21 of `probe-rtx5070-what-bounds-streaming.md`:
B(S) = 1.24 s + 0.0101 s x S; read ≈ 17.1 s per step):

- A_k ≈ max(17.1, k x 6.43) s, i.e. ≈ 30 / 57 / 76 / 78 / 79 / 79 tok/s at k = 1/2/3/4/8/16.
- B_k ≈ k x 6.43 s (+≤3%).
- Peak allocated ≈ 4.4 GB at k = 1, +≤0.3 GB at k = 16.
- Dequant ≈ 23% of B_4.
- Spill ratio ≥ 0.97.
- G1: exact on both fixtures.
- R2's batch 3, if it fits, ≈ 88 tok/s — faster than L2L at k = 3, because a batch pays the
  per-visit dequantisation once and L2L pays it per micro-batch.

**Outcomes:**

| result | meaning | next |
|---|---|---|
| G1-G4 pass | the mechanism works on the shipped runtime, bit-exact, VRAM flat | step 1, SFT first. I2 decides whether step 1 caches the dense layer per visit (+1.7 GB VRAM on this shape); I4 whether the spill design is viable |
| G1 fails, A-vs-A exact | L2L changes the arithmetic | step 1 blocked until the cause is found |
| G1 no verdict | instrument nondeterminism | step 1's gate must run in deterministic mode; G2-G4 still read |
| G2 fails | reads not overlapped with compute | investigate the reader/prefetch interplay first |
| G3 fails | contradicts the arithmetic | investigate before anything |
| G4 fails | the harness keeps activations on the GPU | harness bug: fix, re-run; no verdict on the mechanism |

## 3. What this does NOT measure

- **Any giant-MoE number.** The 70B fixture has 80 dense layers; the boundary-activation
  arithmetic for DeepSeek-V3 / Kimi K2 is in plan.md, not measured here.
- **Training quality.** F2 is random-valued; G1 is arithmetic identity, not convergence.
- **A dense-weight cache per visit.** I2 estimates its worth; nothing here builds it.
- **Two drives.** PR #1536 is not on this branch.
- **Dropout > 0.** A real config's `lora.dropout` 0.05 needs a per-micro-batch RNG state under
  L2L. That is step 1's problem.

## 4. Box state and the commands

Before the run:
- The laptop is on AC with the lid open.
- Every live Claude session has been asked by message for a hold (no CUDA, no test suite, no
  large copy on C: or D:), and the hold is listed in `.claude/SESSIONS.md`.
- R2 and soup-56's sustained run have finished.

```bash
cd /c/Users/user/projects/Soup-l2l
bash benchmarks/harness/l2l_session.sh          # blocks 1-4, one process each, ~2 h
python benchmarks/harness/l2l_rule.py benchmarks/results/probe-rtx5070/l2l   # §5/§6 tables
```

A timing or spill block whose V1 or V3 row already fails at its start exits 3 and the
session stops there: its gates could only be "no verdict", and running it anyway is the
owner's call (`--allow-invalid`). A failed page-lock releases the blocks it left pinned
(the shipped #901 recovery) before the arm is recorded as skipped.

## 5. Results

### 5.1 The run — 2026-10-07, 11:28:29-13:21:58Z

Run from `23d567c6` (clean tree; `src/` byte-identical to `6879f323`), with the base interpreter,
`PYTHONPATH=Soup-l2l/src` and `HF_HUB_OFFLINE=1`; `soup_cli.__file__` resolved under
`Soup-l2l\src` (`logs/soup_cli_file.txt`, and every block's `meta`). Session measure-28, for
soup-28, inside the owner's machine window, right after R2 attempt 2 (10:15-10:46Z). The four blocks
ran one process each, as separate commands in `l2l_session.sh`'s order, including its
deterministic re-run rule. Start and end times are in `logs/session.txt`:

| block | what | UTC |
|---|---|---|
| 1 | correctness F1, k = 4 | 11:28:29-11:28:56 |
| 1 det | the same, `--deterministic` (A-vs-A not exact) | 11:29:20-11:29:49 |
| 2 | correctness F2, k = 2 | 11:30:01-11:31:51 |
| 2 det | the same, `--deterministic` (A-vs-A not exact) | 11:31:58-11:34:03 |
| 3 | timing F2 | 11:34:24-12:50:58 |
| 4 | spill F2 | 12:51:13-13:21:58 |

**Box.** AC online for every sample of every arm; no arm is V5-void. Before the run the owner
stopped Windows Search (Stopped at 11:22Z) and closed ChatGPT, whose on-device-model process had
sat in `nvidia-smi`'s compute-apps list, which V3 refuses. `wsl --shutdown` ran at 11:27Z. Every
block started with 0 MiB in use and no other compute PID. An optional timing `--dry-run` at 11:28Z
(output in a scratch folder, no arm) met V3 with 14.913 GB free after the build against 14.737 GB
needed, and allocated the k = 16 store. Each block's own V3 then read 13.15-15.73 GB free; the
timing and spill blocks read 15.649 and 15.726 GB against 14.737 needed.

**Windows Search came back during block 3.** `SearchIndexer.exe` was restarted at 11:40:34Z. The
service is trigger-started, so a `Stop-Service` lasts only until the next trigger, and no elevated
shell was available to stop it again. By 12:14Z it had read 8.88 GB (8.27 GiB), and nothing more
up to 13:19Z. 8.52 GB of that fell inside three round-0 timing arms, which V6 voids (V6 lists
readers of 1 GB or more in an arm):

| arm | SearchIndexer.exe read in the arm |
|---|---|
| l2l A k = 3, round 0 | 1.85 GB |
| l2l B k = 3, round 0 | 5.56 GB |
| l2l A k = 4, round 0 | 1.12 GB |

No other process read 1 GB or more in any arm of blocks 1-4. No gate reads k = 3 or k = 4; I1 and
I2 do, and the rule's output below says how they lose those rounds.

### 5.2 Correctness (blocks 1-2)

| run | fixture, k | A-vs-A (GA twice) | L2L against GA | per-micro-batch losses |
|---|---|---|---|---|
| block 1, default | F1 Mistral-7B NF4, RAM tier, k = 4 | **not exact**: 250 of 256 tensors differ, max abs diff 0.2133 | 250 of 256 differ, max abs diff 0.2639 | equal in both comparisons |
| block 1, deterministic | the same | exact, 256 of 256 | **exact**, 256 of 256, max abs diff 0 | equal |
| block 2, default | F2 70B shape NF4, disk tier, k = 2 | **not exact**: 634 of 640 differ, max abs diff 6.292e-03 | 634 of 640 differ, max abs diff 6.616e-03 | equal in both comparisons |
| block 2, deterministic | the same | exact, 640 of 640 | **exact**, 640 of 640, max abs diff 0 | equal |

Losses, identical for GA and L2L in all four runs: F1 3.257613 / 3.105495 / 3.219895 / 3.134643;
F2 6.035345 / 5.996969. In the default mode the GA reference does not reproduce itself, and L2L
differs from it by the same order of magnitude. Under `torch.use_deterministic_algorithms(True)`
with `CUBLAS_WORKSPACE_CONFIG=:4096:8` all three agree bit for bit, which is G1's pre-registered
fallback.

### 5.3 Timing (block 3)

Each arm: 1 warm-up and 3 timed steps; the value is the mean of the two rounds. Every arm loaded
157 + 2 layers and moved 70,378,258,732 bytes per step.

| arm | k | round 0 (s) | round 1 (s) | value (s) | tok/s | round spread | peak alloc (GB) | validity |
|---|---|---|---|---|---|---|---|---|
| plain A | 1 | 17.373 | 17.419 | 17.396 | 29.4 | 0.3% | 4.509 | ok |
| plain B | 1 | 6.397 | 6.436 | 6.417 | 79.8 | 0.6% | 4.509 | ok |
| l2l A | 1 | 17.166 | 19.765 | 18.465 | 27.7 | **14.1%** | 4.018 | **V4 fails** |
| l2l B | 1 | 6.949 | 7.002 | 6.975 | 73.4 | 0.8% | 4.018 | ok |
| l2l A | 2 | 17.907 | 18.114 | 18.010 | 56.9 | 1.1% | 4.028 | ok |
| l2l B | 2 | 13.870 | 13.930 | 13.900 | 73.7 | 0.4% | 4.028 | ok |
| l2l A | 3 | 22.688 | 22.559 | 22.624 | 67.9 | 0.6% | 4.036 | **V6, round 0** |
| l2l B | 3 | 20.895 | 20.906 | 20.901 | 73.5 | 0.1% | 4.036 | **V6, round 0** |
| l2l A | 4 | 28.051 | 28.078 | 28.064 | 73.0 | 0.1% | 4.044 | **V6, round 0** |
| l2l B | 4 | 27.748 | 27.836 | 27.792 | 73.7 | 0.3% | 4.044 | ok |
| l2l A | 8 | 55.819 | 55.966 | 55.893 | 73.3 | 0.3% | 4.078 | ok |
| l2l B | 8 | 55.604 | 55.696 | 55.650 | 73.6 | 0.2% | 4.078 | ok |
| l2l A | 16 | 111.752 | 111.839 | 111.796 | 73.3 | 0.1% | 4.145 | ok |
| l2l B | 16 | 111.654 | 111.434 | 111.544 | 73.4 | 0.2% | 4.145 | ok |
| l2l C | 4 | 23.881 | 23.848 | 23.864 | 85.8 | 0.1% | 3.443 | ok |
| l2l D | 4 | 22.064 | 21.846 | 21.955 | 93.3 | 1.0% | 3.443 | ok |

**The arm that fails V4.** l2l A at k = 1 read 17.128 / 17.196 / 17.173 s in round 0 and
17.157 / 17.150 / **24.989** s in round 1: two steps as in round 0, and one 7.8 s longer. Its
implied host-to-device rate is 4.10 GB/s in round 0 and 3.56 GB/s in round 1. It is the last A arm
of round 1, after about 70 minutes of reading C:. §1's arm order puts the k = 1 arms there so that
a drive throttle would show up as a V4 "no verdict" rather than a bias, and that is the row that
fired. The files do not establish the cause.

### 5.4 Spill (block 4)

| arm | round 0 (s) | round 1 (s) | value (s) | tok/s | peak alloc (GB) |
|---|---|---|---|---|---|
| pinned, k = 16 | 111.466 | 112.245 | 111.856 | 73.2 | 4.145 |
| spill (layers 0-39 through `D:\soup-l2l-spill`, unbuffered), k = 16 | 111.932 | 112.150 | 112.041 | 73.1 | 4.145 |

### 5.5 The rule's output

`python benchmarks/harness/l2l_rule.py benchmarks/results/probe-rtx5070/l2l`, committed as
`verdict.md` with the raw files, and recomputed identically from the committed files:

| gate | status | detail |
|---|---|---|
| G1 | pass | F1: 256 tensors + losses equal (deterministic); F2: 640 tensors + losses equal (deterministic) |
| G2 | pass | T_A(16) 73.3 vs 0.8 x T_B(16) 58.8 tok/s |
| G3 | no verdict | ('l2l', 'A', 1): V4: round spread 14.1% > 5% |
| G4 | no verdict | ('l2l', 'A', 1): V4: round spread 14.1% > 5% |

| row | ok | detail |
|---|---|---|
| V1 | True | direct_io=True |
| V2 | True | A: loads 159.0 vs 159.0, bytes 70.378 vs 70.378 GB; B: loads 159.0 vs 159.0, bytes 70.378 vs 70.378 GB |
| V3 | True | free 15.649 GB, commit available 26.480 GB, need 10.737 GB + 4.0 GB margin; 0 MiB before the build; no foreign GPU PID |

| arm | valid | step s | tok/s | peak GB | why |
|---|---|---|---|---|---|
| plain/A/k1 | True | 17.396 | 29.4 | 4.509 |  |
| plain/B/k1 | True | 6.417 | 79.8 | 4.509 |  |
| l2l/A/k1 | False | 18.465 | 27.7 | 4.018 | V4: round spread 14.1% > 5% |
| l2l/B/k1 | True | 6.975 | 73.4 | 4.018 |  |
| l2l/A/k16 | True | 111.796 | 73.3 | 4.145 |  |
| l2l/B/k16 | True | 111.544 | 73.4 | 4.145 |  |
| l2l/A/k4 | False | 28.064 | 73.0 | 4.044 | V6: round 0 foreign reader SearchIndexer.exe 1.12 GB |
| l2l/B/k4 | True | 27.792 | 73.7 | 4.044 |  |
| l2l/C/k4 | True | 23.864 | 85.8 | 3.443 |  |
| l2l/D/k4 | True | 21.955 | 93.3 | 3.443 |  |
| l2l/A/k2 | True | 18.010 | 56.9 | 4.028 |  |
| l2l/A/k3 | False | 22.624 | 67.9 | 4.036 | V6: round 0 foreign reader SearchIndexer.exe 1.85 GB |

I2: dequant_share 0.210, c4_over_a4 None (A_4 is V6-void). I3: overhead 1.086. I4: ratio 0.998,
free True, direct True. I6 (B): 1.087.

**Verdict line: NO VERDICT: G3, G4.**

(The V3 row is the rule's dictionary, written out; its values are unchanged.)

## 6. Verdict

By the rule of §2 as committed (`7a2127c1`, amended before any run in `ec95e48`), applied by
`l2l_rule.py` to the committed JSON:

- **G1 (correctness): PASS**, through its pre-registered fallback. In the default mode the GA
  reference was not exact against itself on either fixture (§5.2), so each block re-ran in a fresh
  process with deterministic algorithms. There A-vs-A is exact, and the L2L step's LoRA gradients
  (256 tensors on F1 at k = 4, 640 on F2 at k = 2) and per-micro-batch losses are `torch.equal`
  to GA.
- **G2 (the read hides): PASS.** T_A(16) = 73.3 tok/s against 0.8 x T_B(16) = 58.8 tok/s.
- **G3 (it pays): NO VERDICT.** It reads T_A(1), and l2l A at k = 1 fails V4 (round spread 14.1%,
  §5.3).
- **G4 (VRAM flat in k): NO VERDICT**, by the same row on the same arm.
- **Validity:** V1, V2 and V3 hold. V4 fails on l2l A k = 1 only. V5 holds for every arm. V6 voids
  round 0 of l2l A k = 3, B k = 3 and A k = 4, arms no gate reads.

The outcome table of §2 has no row for G1 and G2 passing with G3 and G4 undecided, so this record
does not choose a next step; that is the owner's call. The numbers G3 and G4 would have read are
recorded above and are **not** judged here: T_A(16) / T_A(1) = 73.3 / 27.7 tok/s, where round 0
alone gives 29.8; peak allocated 4.145 GB at k = 16 against 4.018 GB at k = 1.

**I1 (the R2 control), cited from R2's own §6** (`probe-rtx5070-batch-control.md` on
`probe/batch-control`, attempt 2, commit `91b070c4`; R2 ran the PR #1536 head `2c6ed087` with its
per-drive dispatcher on one drive, which is not byte-for-byte this branch's reader). SINGLE
batch 2 is **NEAR-FREE** (`ē` 1.027, no label; 57.13 / 57.60 tok/s). SINGLE batch 3 is **PARTIAL**
(`ē` 0.899, no label, DISAGREES AT b3; 75.21 / 75.54 tok/s). The largest batch that fits is 3
(6.312 GB allocated). Against them: T_A(2) = 56.9 tok/s, and T_A(3) = 67.9 tok/s, whose round 0 is
V6-void (round 1 alone: 68.1).

Nothing in §2 was edited after the run. The records are local, on `probe/l2l-step0`, and not
pushed.
