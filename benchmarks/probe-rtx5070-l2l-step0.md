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

**Status: rule committed 2026-10-01, BEFORE the run. Results pending.**

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
- **V2.** L2L at k = 1 is within ±5% of P1 (arm A) and of P1B (arm B). This shows the harness
  measures the shipped step. G2-G4 depend on it.
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

## 5. Results

Pending.

## 6. Verdict

Pending.
