# Probe: do more tokens per step raise the tokens/s of the synthetic 1T MoE step? RTX 5070 Laptop 8 GB

Status: **PRE-REGISTERED (d83e57721); RUN 2026-10-23: NO VERDICT, gate V5 failed in 3 of 4 runs
(section 12).** This rule, its thresholds and the harness change it needs (section 3) were
committed together, before any run. Results are in section 12. Nothing in sections 1-11 may be edited after the first run except to
correct a typo, and any such correction is listed in section 12.

**Every number this probe produces is "random weights, shape only; routing forced uniform".**
It comes from a measurement harness (`benchmarks/harness/moe_stream_step.py`), not from
`soup train`. It runs one LoRA step per process on the synthetic store of
`probe-rtx5070-giant-moe-step.md` (1,026,408,232,448 parameters, K2 shape). This probe claims
nothing about "1T on 8 GB" beyond what that record shows: one full step completes, on random
weights, in a measurement harness. Prior art for streaming layers through a small device
memory, and for amortising the per-layer transfer over more samples per step, is L2L
(Pudipeddi et al. 2020, "Training Large Neural Networks with Constant Memory using a New
Execution Algorithm"). This probe tests the simplest form of that amortisation, a longer
sequence in one step. It does not implement L2L's layer-major micro-batching.

## 1. Question

The base record (`probe-rtx5070-giant-moe-step.md`, run 2) measured one step at seq 512 at
867.95 s (step 0) and 1,108.77 s (step 1). Each step read 1,570,336,174,080 bytes of NF4
experts, about 0.5 tokens/s. If the step's time is the expert fetch, and the fetch does not
grow with the tokens, then 4x the tokens per step should cost little extra time, and tokens/s
should rise nearly 4x. Does it, on this laptop?

## 2. What the code says (read, not run)

### 2.1 Knobs and what is timed

`moe_stream_step.py` takes `--seq`, `--batch` (already present, default 1), `--steps`,
`--route model|uniform`, `--tier ram|disk`, `--no-pin`, `--buffers`, `--lora-r` and, since
this commit, `--loss-tokens`. The timed region runs from one `cuda.synchronize()` to the next:
forward, backward and `AdamW.step()`. The `tokens_per_second` it reports is
`batch * seq / seconds`.

### 2.2 What a step reads, and why it does not grow with seq

`StreamedExperts` (`src/soup_cli/utils/expert_stream.py`) fetches every routed-to expert
three times: in the forward, in the decoder layer's checkpoint recompute, and in its own
backward, where it recomputes the expert MLP locally to get the input gradient. Each fetch is
synchronous: an unbuffered read, a host-to-device copy, an NF4 dequantisation, then the
compute. The dense part of each layer (attention, norms, router, shared expert) and the two
vocabulary matrices stream through the disk tier, which also reads with direct I/O (#974, PR
#991). So no byte of the step goes through the page cache.

`--route uniform` sends flattened row t to experts 8t .. 8t+7 (mod 384). Every expert of
every MoE layer is hit once `batch * seq >= 48`. A step then makes 60 x 384 x 3 = 69,120 expert
fetches, 1,570,336,174,080 bytes, **at every seq in this probe**.

| seq (batch 1) | token-expert pairs per expert per pass | experts hit per layer, uniform | untouched fraction, uniformly random top-8 |
|---|---|---|---|
| 512 | 10.7 | 384 / 384 | 2.1e-5 |
| 1024 | 21.3 | 384 / 384 | 4.3e-10 |
| 2048 | 42.7 | 384 / 384 | 1.9e-19 |
| 4096 | 85.3 | 384 / 384 | 3.5e-38 |

The last column is `(376/384)^T`. Even with random routing, coverage does not grow past 512.
With random weights, the model's own routing collapsed to about 9.7 experts per layer and pass
(base record section 4.1), so it is not an arm here.

### 2.3 What grows with the tokens

| work per step | seq 512 | seq 2048 |
|---|---|---|
| expert fetches / bytes | 69,120 / 1.570 TB | 69,120 / 1.570 TB |
| token-proportional compute (TFLOP) | ~116 | ~483 |

The compute row counts four kinds of work:
- MLA projections, the shared expert and layer 0's dense MLP: forward, recompute and input-gradient.
- The routed-expert GEMMs: forward, recompute, local recompute and input-gradient.
- Causal attention scores.
- The vocabulary projection and loss on 512 positions in both arms (identical, so it cancels).

The extra 367 TFLOP at an achieved 10-30 TFLOPS is +12 to +37 s. The routed GEMMs have 10.7
and 42.7 rows per expert, so they stay bound by reading the dense weight from VRAM, which is
fixed per fetch. That should keep the real increment near the low end. Allowing for elementwise,
gather and scatter work, **the prediction is t(2048) = t(512) + 10 to 60 s**. This is an
estimate, not a threshold.

### 2.4 VRAM per seq: the vocabulary is what stops a long step

Calibration: the base record's step-0 peak (allocated) was 4,342 MiB at seq 512. Three terms
add to a fixed base:
- **Checkpoint boundaries.** The 61 decoder inputs plus the final norm and lm_head inputs stay on the card from the forward to their backward: 63 x 7168 x 2 B = 0.861 MiB per token.
- **The loss at the vocabulary of 163,840.** transformers upcasts the logits to fp32, `cross_entropy` saves its log-softmax, and its backward holds two more fp32 tensors: 12 x 163,840 B = 1.875 MiB per loss position.
- **One decoder layer's recompute plus backward working set**, estimated from the saved tensors of MLA attention and the expert loop: 0.32-0.52 MiB per token. Its peak falls at the top layer, while all boundaries are still live.

Without the per-token terms, the base comes out at 2,941 MiB, which closes against the
bottom-up 2,900 MiB (2,783 MiB stream buffers from the run's JSON plus 116 MiB of fp32
adapters).

Attention: transformers' default `sdpa` path gets no mask (`is_causal=True`). The query/key
head dim is 192 and the value head dim 128. Torch's SDPA dispatch is expected to refuse its
flash kernel, which needs equal head dims, and to take the memory-efficient one, which
accepts a different value dim. That keeps attention memory linear in T. This was not checked
on the GPU at registration. If the math kernel were chosen instead, the fp32 scores alone
would add 1 GiB at seq 2048. Gate V3 would catch that, and the run would be reported as not
fitting.

| seq (batch 1) | peak alloc, full loss (harness as on main) | peak alloc, `--loss-tokens 512` | under the 7,500 MiB gate? |
|---|---|---|---|
| 512 | 4,342 (measured) | 4,342 (same computation) | yes |
| 1024 | 5,743 | 4,783 | yes either way (not an arm) |
| 2048 | **8,545**, over the 8,151 MiB card | **5,665-5,770** | only with `--loss-tokens` |
| 4096 | 14,149 | 7,780-8,599 | **no** |

- Seq 4096 does not fit even with a short loss window. The boundaries alone are 3,528 MiB on top of the 2,941 MiB base. Keeping them off the card (host-resident boundary activations) is a larger change than this probe.
- `--batch` is not a way round this. The uniform router flattens rows, and every term above scales with `batch * seq` (the loss term with `batch * loss_tokens`).
- So the arms are **seq 512 and seq 2048, batch 1, `--loss-tokens 512` in both**.
- Peak reserved ran 447-710 MiB above allocated in the base record, so the 2048 arm is predicted at **6,120-6,480 MiB reserved**.

### 2.5 Host RAM per seq

Nothing on the host scales with the tokens: activations stay on the card, nothing is
offloaded, and the ids are 8 bytes a token. The base record's peak working set was 7.75 GB
(step 0) and 7.89 GB (step 1). **The prediction is 7.6-8.2 GB at both seq values.** Each run
starts with at least 10 GB free, so at least ~2 GB stays free. The watchdog floor (section 8)
is 1.5 GB.

## 3. The harness change (same commit as this rule)

- **`--loss-tokens N`** (default 0 = all positions, unchanged behaviour). For 0 < N < seq, the step passes `logits_to_keep=N` and `labels=ids[:, -N:]`. The vocabulary projection and the cross entropy then cover the last N positions. The loss's own shift keeps next-token targets: position T-N+j predicts id T-N+j+1.
  - Every position still runs forward and backward through all 61 layers, and the gradient reaches positions outside the window: `tests/test_moe_stream_step.py` checks both on a tiny CPU DeepSeek-V3.
  - With N = seq, the step is the base record's computation exactly (`labels=ids`, no `logits_to_keep`).
- **`loss_positions` and `os_io`** in every step entry. `os_io` comes from psutil over the timed region: the bytes this process read, and the bytes each physical drive read. Drive read times are not recorded: psutil's Windows `read_time` was not credible on this box (7.4 s for 7.1 TB).
- Tests: `tests/test_moe_stream_step.py`.
  - 15 CPU tests. Eight mutations of the change were each killed (label slicing, the full-window guard, an off-by-one window caught only by the real-model test, the labels passed to the model, the negative-value refusal, the drive delta, the platform-refusal catch, the process delta).
  - One `gpu` test runs the windowed step end to end on the tiny store. It was not run at registration (CPU-only session), so it is pre-flight P2.

## 4. Arms, order, cache state

| run | arm | command differs only in |
|---|---|---|
| a1 | A | `--seq 512` |
| b1 | B | `--seq 2048` |
| b2 | B | `--seq 2048` |
| a2 | A | `--seq 512` |

- **Order ABBA**, one fresh process per run, `--steps 1`. Every measured step is the first step of its process, as the base record's 867.95 s was.
- The pairs are (a1, b1) and (b2, a2). A drift that slows later runs works against the claim in the first pair and for it in the second, and cancels in the pooled ratio.
- The base record's two consecutive steps differed by 1.277x for no known reason, so drift is a real risk on this box.
- **Page cache:** not a variable. Every expert read and every dense-layer read is direct I/O (section 2.2), so each run reads its 1.57 TB from the drive whatever ran before. Nothing is evicted or warmed. Gates V7 and V8 check that the bytes came from the drive.
- The next run starts as soon as the previous run's block returns. There is no cool-down, and the gaps are recorded in `box.jsonl`.

## 5. Primary metric and decision rule (thresholds fixed now)

`t(run)` is `steps[0].seconds`. Tokens/s is `seq / t`, so the tokens/s ratio of a pair is
`4 x t(A) / t(B)`.

- R1 = 4 x t(a1) / t(b1)
- R2 = 4 x t(a2) / t(b2)
- Rpool = 4 x (t(a1) + t(a2)) / (t(b1) + t(b2))

Outcomes, valid only if every gate in section 6 passes in all four runs:

- **SHIP** iff min(R1, R2, Rpool) >= 3.0. Then the record may state the claim below, with the measured Rpool. The words "nearly in proportion" are allowed only if also min(R1, R2) >= 3.4 and Rpool >= 3.6.
- **REFUTED** iff Rpool < 2.0: 4x the tokens cost at least 2x the time. At seq 2048 the step is no longer dominated by a token-independent cost.
- **NOT SHIPPED** otherwise: the record states the measured R1, R2 and Rpool, and makes no scaling claim.
- **NO VERDICT** if any gate fails in any run, or if the series is cut by the time cap (section 9).

Why 3.0: the base record's 1.277x step-to-step change, landing wholly against the claim,
gives 4 / 1.277 = 3.13. So 3.0 tolerates drift of that size plus a compute increment of up
to ~4%. Anything worse fails, which is the conservative direction.

The claim, fixed now: *"On an RTX 5070 Laptop (8 GB), one LoRA step of a synthetic
1.026 T-parameter Kimi K2-shaped store (random weights, routing forced uniform; a
measurement harness, not `soup train`) ran at >= 3x the tokens per second at seq 2048 as at
seq 512 (loss over the last 512 positions in both), while the step's expert reads stayed at
1.57 TB."*

## 6. Validity gates (every run)

- **V1 completes.** Exit code 0. `step.json` has exactly one step with `loss_finite`, `grads_finite`, `grads_present == 610` and `loss_positions == 512`.
- **V2 same work.** `store_bytes == 533,242,299,230`. `expert_fetches == 69,120` and `expert_bytes_read == 1,570,336,174,080` in every run (the base record's per-step values). `layer_loads` is equal in all four runs (120 expected, the base record's step 0).
- **V3 VRAM.** `peak_vram_mib <= 7,500` (the base record's gate; the card has 8,151 MiB) and `peak_reserved_mib <= 7,800`, so Windows never spills to shared memory, which would slow the step silently.
- **V4 host RAM.** `AvailableBytes >= 10e9` in the run's `before` line of `box.jsonl` (the owner's floor). `peak_host_bytes_so_far < 20e9`. The watchdog did not fire.
- **V5 no swap in the step.** `os_io.disk_read_bytes["PhysicalDrive0"] <= 2,147,483,648`. C: is disk 0 and holds `C:\pagefile.sys`, so paging in shows up there. The allowance covers DLL pages a first CUDA step maps in; P2 runs just before a1, so those pages should already be cached.
- **V6 awake, on AC.** In `watchdog.json`, `gaps == []` and `off_ac_samples == 0`. `ac == 2` in both box lines. No Kernel-Power event 41 or 42 in the System log between a1's `before` line and a2's `after` line.
- **V7 the bytes came from the drive, and only for this process.** With D = `os_io.disk_read_bytes["PhysicalDrive1"]` (D: is disk 1; P3 confirms) and P = `os_io.process_read_bytes`: D >= `expert_bytes_read` and |D - P| <= 0.02 x P. `watchdog.json` has `foreign_compute_pids == []`. A measurement hold stands on `.claude/SESSIONS.md` for the whole window.
- **V8 direct I/O.** `stats.direct_io_drives >= 1`.
- **V9 code.** P0-P2 pass (section 8).

A run that fails V4-V7 is a disturbed box, not a result. The whole ABBA series may then be
repeated once, in a new window, and runs from two series are never mixed. A failure of V1-V3
or V8-V9 is itself the finding, and the verdict is NO VERDICT.

## 7. Showing where the time goes (reported; it fixes the wording, not the verdict)

Per run: E = `expert_bytes_read`, P, D (section 6), and the drive rate D / t. From the
pooled arms, fit t(T) = F + c x T:
- c = (mean t(B) - mean t(A)) / 1536 seconds per token;
- F = mean t(A) - 512 c;
- F / mean t(A) is the token-independent share of the seq-512 step.

Wording rule:
- The record may call the step **disk-bound** only if D / t >= 3.27 GB/s in every run. That is 80% of the 4.09 GB/s (spread 4.09-4.14) that the async direct-I/O disk tier sustained over a step in `probe-rtx5070-what-bounds-streaming.md` section 21. That was measured on C:, the same Samsung MZAL81T0HFLB model as D:.
- Otherwise, if F / mean t(A) >= 0.9, it calls the step **fetch-bound**: the time is the per-expert fetch chain (synchronous read, host-to-device copy, dequantisation), which scales with bytes, not tokens, while the drive ran below its ceiling.
- Otherwise it calls it neither.

Prediction: the base record's expert-read rate was 1.81 and 1.42 GB/s, so expect
"fetch-bound, not disk-bound". This probe cannot split the read from the copy and the
dequantisation inside that chain.

## 8. Procedure (PowerShell, in this order)

All commands run from the worktree at this commit, with the base interpreter (not a venv).

**P0, import path.** The output must be a path under the worktree's `src` and `True`:

    $Tree = "C:\Users\user\projects\soup-wt\moe-seq-scaling"; $env:PYTHONPATH = "$Tree\src"; Set-Location $Tree
    python -c "import sys, soup_cli; print(soup_cli.__file__, sys.prefix == sys.base_prefix)"

**P1, code.** `git log -1 --format=%H` is this commit, or a later one that touches only
`benchmarks/results/` and section 12.

**P2, wiring on the GPU** (seconds, tiny store). One test must pass:

    python -m pytest tests/test_moe_stream_step.py -m gpu --no-cov -q

**P3, box.** Before anything else, check all of these and write a measurement hold on the board:
- `Get-Partition -DriveLetter D | Select-Object DiskNumber` returns 1. If not, use the number shown in V7.
- `D:/soup-bench/moe-k2-synth/index.json` exists. The store is on the owner's post-demo cleanup list, so check it first.
- `nvidia-smi --query-compute-apps=pid --format=csv,noheader` prints nothing.
- At least 10 GB of RAM is free, and the box is on AC with the lid open.

**Arms.** Run this block four times, in order, with `$Name, $Seq` = `a1, 512`, `b1, 2048`,
`b2, 2048`, `a2, 512`. Each block takes 15-20 minutes, which is longer than a 10-minute
foreground tool timeout. Launch each one as its own background call and wait for it to
return before starting the next.

    $Name = "a1"; $Seq = 512
    $Tree = "C:\Users\user\projects\soup-wt\moe-seq-scaling"; $env:PYTHONPATH = "$Tree\src"; Set-Location $Tree
    $Dir = "$Tree\benchmarks\results\probe-rtx5070\giant-moe\seq-scaling\$Name"
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $Box = { param($At) [ordered]@{ at = $At; utc = (Get-Date).ToUniversalTime().ToString("o")
        mem = Get-CimInstance Win32_PerfRawData_PerfOS_Memory | Select-Object AvailableBytes, CommittedBytes, PagesInputPersec
        ac = (Get-CimInstance Win32_Battery).BatteryStatus
        gpu = (nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader) } |
        ConvertTo-Json -Compress | Add-Content "$Dir\box.jsonl" }
    & $Box "before"
    $Step = Start-Process -FilePath python -PassThru -NoNewWindow `
        -RedirectStandardOutput "$Dir\stdout.log" -RedirectStandardError "$Dir\stderr.log" -ArgumentList @(
        "benchmarks/harness/moe_stream_step.py", "--store", "D:/soup-bench/moe-k2-synth",
        "--tier", "disk", "--no-pin", "--route", "uniform", "--batch", "1", "--steps", "1",
        "--loss-tokens", "512", "--seq", "$Seq", "--out", "$Dir\step.json")
    $null = $Step.Handle
    python benchmarks/harness/fp8_loss_check_watchdog.py --pid $Step.Id --out $Dir --min-free-gb 1.5 --disk D:/
    $Step.WaitForExit(); "exit $($Step.ExitCode)" | Set-Content "$Dir\exit.txt"
    & $Box "after"

The watchdog follows the step's process to its exit, sampling every 5 s: available RAM, the
process tree's RSS, `nvidia-smi`, AC, and gaps (sleep). It stops the step below 1.5 GB free,
which makes V4 fail. Its `--match` mode needs a standalone `train` argument, so the step is
named by `--pid`.

**After a2:**

    Get-WinEvent -FilterHashtable @{ LogName = "System"; ProviderName = "Microsoft-Windows-Kernel-Power"; Id = 41, 42; StartTime = (Get-Item "$Tree\benchmarks\results\probe-rtx5070\giant-moe\seq-scaling\a1\box.jsonl").CreationTime } -ErrorAction SilentlyContinue | Select-Object TimeCreated, Id

It must print nothing (V6). Then lift the hold, compute section 5 and section 7 from the four
`step.json` files, and write section 12.

## 9. Wall-clock budget

| part | time |
|---|---|
| P0-P3 | ~5 min |
| each run | 15-20.5 min (step 14-19.5 min, predicted t(512) 850-1,150 s and t(2048) 860-1,210 s, plus ~1 min of start, build and exit) |
| four runs | 60-82 min |
| **total** | **65-87 min, ~80 min expected** |

**Cap:** no run starts later than 95 minutes after a1's `before` line. A series the cap cuts
short is NO VERDICT.

## 10. Predictions (written now; not thresholds)

| | seq 512 | seq 2048 |
|---|---|---|
| step seconds | 850-1,150 | t(512) + 10 to 60 |
| tokens/s | 0.45-0.60 | 1.7-2.4 |
| R1, R2, Rpool | | 3.74-3.96 |
| peak alloc VRAM | 4,342 MiB | 5,665-5,770 MiB |
| peak reserved | about 5,050 MiB (base: 5,052) | 6,120-6,480 MiB |
| peak working set | 7.6-8.2 GB | 7.6-8.2 GB |
| expert bytes per step | 1,570,336,174,080 | 1,570,336,174,080 |
| drive rate D / t | 1.4-1.9 GB/s | 1.3-1.9 GB/s |
| wording (section 7) | fetch-bound, not disk-bound | |

## 11. What this probe cannot show

- Quality or convergence: the weights are random.
- A full-position loss at seq 2048: predicted at 8,545 MiB, it does not fit. The 2048 arm trains the loss on 512 of its 2,048 positions. Every position runs forward and backward, but a real fine-tune at that length would need a chunked loss.
- Seq 4096 or beyond: the boundary activations do not fit on the card (section 2.4).
- Model routing: random weights collapse it (section 2.2).
- How the fetch chain splits between the drive, the copy and the dequantisation.
- More than one step per process, or whether the base record's step-1 slowdown recurs.

## 12. Results

Series run on 2026-10-23, from a1's `before` line (14:05:38Z) to a2's `after` line (15:20:45Z), about 75 minutes. Commands exactly as section 8, at commit `d83e57721`, with the base interpreter, one fresh process per run, `--steps 1`. Raw files are in `benchmarks/results/probe-rtx5070/giant-moe/seq-scaling/{a1,b1,b2,a2}/` (`box.jsonl`, `watchdog.jsonl`, `watchdog.json`, `step.json`, `exit.txt`, `stdout.log`, `stderr.log`).

Every number below is "random weights, shape only; routing forced uniform", from a measurement harness (`benchmarks/harness/moe_stream_step.py`), not `soup train`, one step per process. Nothing here shows more than that.

A first attempt at a1 started at 13:54:46Z the same day and died at about 14:01Z when the session that launched it closed (no `exit.txt`, process gone). Its files were moved out of the repository and are not used here. The series below is a full run from scratch.

### 12.1 Verdict

**NO VERDICT.** Gate V5 (no swap in the step) failed in three of the four runs (a1, b1, b2). Section 5 makes an outcome valid only if every gate passes in all four runs, and section 6 calls a run that fails V4-V7 "a disturbed box, not a result". So the section-5 claim is not made, and none of SHIP, REFUTED or NOT SHIPPED is declared. Section 6 allows the whole ABBA series to be repeated once, in a new window, with the two series never mixed. That was not done in this window.

The measured ratios are listed in 12.3 as numbers, not as a result. For the record, they sit above the 3.0 SHIP floor and above the 3.4 / 3.6 "nearly in proportion" floors, and the rule gives them no standing because V5 failed.

### 12.2 Arms

| run | seq | step seconds | tokens/s | peak alloc (MiB) | peak reserved (MiB) | peak working set (GB) | RAM available before (GB) | bytes read from D: (GB) | drive rate D / t (GB/s) |
|---|---|---|---|---|---|---|---|---|---|
| a1 | 512 | 1036.40 | 0.494 | 4,342 | 5,052 | 7.620 | 13.72 | 1,586.79 | 1.531 |
| b1 | 2048 | 1117.61 | 1.832 | 6,091 | 7,176 | 7.608 | 10.47 | 1,586.79 | 1.420 |
| b2 | 2048 | 1120.52 | 1.828 | 6,091 | 7,176 | 7.603 | 15.14 | 1,586.79 | 1.416 |
| a2 | 512 | 1036.30 | 0.494 | 4,342 | 5,052 | 7.729 | 15.61 | 1,586.79 | 1.531 |

- Peak working set is the watchdog's process-tree peak. The harness's own `peak_host_bytes_so_far` is 7.77, 7.76, 7.76, 7.76 GB.
- The loss repeats exactly inside each pair: 13.4771 at seq 512 (a1, a2) and 13.4916 at seq 2048 (b1, b2).
- Gaps between runs, `after` to the next `before`: 37 s, 27 s, 28 s. The latest start was a2, 57 min 19 s after a1's `before` line (cap: 95 min).

### 12.3 Ratios (section 5)

- R1 = 4 x 1036.40 / 1117.61 = **3.709**
- R2 = 4 x 1036.30 / 1120.52 = **3.699**
- Rpool = 4 x (1036.40 + 1036.30) / (1117.61 + 1120.52) = **3.704**
- Seq 2048 cost 81.21 s more than seq 512 in the first pair, 84.22 s in the second, 82.72 s pooled. The two runs of the same arm differed by 0.10 s (a1 vs a2) and 2.91 s (b1 vs b2).

### 12.4 Validity gates (section 6)

| gate | a1 | b1 | b2 | a2 |
|---|---|---|---|---|
| V1 completes | pass | pass | pass | pass |
| V2 same work | pass | pass | pass | pass |
| V3 VRAM | pass | pass | pass | pass |
| V4 host RAM | pass | pass | pass | pass |
| V5 no swap in the step | **FAIL** | **FAIL** | **FAIL** | pass |
| V6 awake, on AC | pass | pass | pass | pass |
| V7 bytes from the drive | pass | pass | pass | pass |
| V8 direct I/O | pass | pass | pass | pass |
| V9 code | pass | pass | pass | pass |

- **V1:** exit 0 in all four, one step each, `loss_finite` and `grads_finite` true, `grads_present` 610, `loss_positions` 512.
- **V2:** `store_bytes` 533,242,299,230, `expert_fetches` 69,120, `expert_bytes_read` 1,570,336,174,080 and `layer_loads` 120 in all four runs.
- **V3:** peak alloc 4,342 / 6,091 / 6,091 / 4,342 MiB against 7,500; peak reserved 5,052 / 7,176 / 7,176 / 5,052 MiB against 7,800. The seq-2048 margins are 1,409 MiB (alloc) and 624 MiB (reserved).
- **V4:** available RAM in the `before` lines was 13.72 / 10.47 / 15.14 / 15.61 GB (floor 10). `peak_host_bytes_so_far` was under 7.8 GB in every run (limit 20 GB). The watchdog did not fire. The lowest free RAM it sampled inside a run was 2.87 / 3.47 / 7.31 / 7.72 GB (its floor is 1.5 GB).
- **V5:** `os_io.disk_read_bytes["PhysicalDrive0"]` was 2,312,677,888 / 2,457,071,616 / 8,110,742,528 / 1,722,067,456 bytes against an allowance of 2,147,483,648. That is 1.08x, 1.14x and 3.78x the allowance in a1, b1 and b2, and 0.80x in a2.
- **V6:** `gaps == []`, `off_ac_samples == 0` and `ac == 2` in both box lines of all four runs. The Kernel-Power query of section 8 (events 41 and 42, from a1's `box.jsonl` creation time) printed nothing. Lid state cannot be read by software. No sleep gap was seen in the 5 s samples.
- **V7:** D (`PhysicalDrive1`) was 1,586,785,970,176 bytes in a1, a2 and b1 and 1,586,785,978,368 in b2, which is 1.0105x `expert_bytes_read`. |D - P| / P was 0.045%, 0.054%, 0.055%, 0.055% (limit 2%). `foreign_compute_pids == []` in all four. A measurement hold covering the window was declared on `.claude/SESSIONS.md` before the series (by the session that launched this one) and was read there, read-only, after it.
- **V8:** `direct_io_drives == 1` in all four.
- **V9:** P0: the import path was under the worktree's `src` and the prefix check printed `True`. P1: `git log -1` gave `d83e5772187ff1b218cfe11797964e0e22ee12ee`, with a clean tree. P2: `1 passed, 15 deselected in 25.73s`.

What the V5 failure looked like. The `PagesInputPersec` counter in the box lines rose by 597,063 / 639,377 / 1,898,033 / 488,689 pages between each run's `before` and `after` line. At 4,096 B per page that is 2.45 / 2.62 / 7.77 / 2.00 GB, against PhysicalDrive0 reads of 2.31 / 2.46 / 8.11 / 1.72 GB, in the same rank order. That is consistent with page-ins (hard faults) being the source of the C: reads. Whether they came from the pagefile or from mapped library files was not separated, and nothing here tests it. Step time did not follow the C: reads: b1 and b2 differ by 0.26% in step time with 2.46 GB and 8.11 GB read from C:.

### 12.5 Where the time goes (section 7)

- mean t(A) = 1036.35 s, mean t(B) = 1119.065 s.
- c = (1119.065 - 1036.35) / 1536 = 0.05385 s per token.
- F = 1036.35 - 512 x 0.05385 = 1008.78 s. F / mean t(A) = **0.973**.
- Drive rate D / t: 1.531, 1.420, 1.416, 1.531 GB/s. None reaches the 3.27 GB/s line, so the step is not called disk-bound.
- F / mean t(A) is at least 0.9, so by the wording rule the step is called **fetch-bound**: the time is the per-expert fetch chain, which scales with bytes, not tokens, while the drive ran below its ceiling. This probe cannot split that chain into read, copy and dequantisation. The section-7 wording does not depend on the verdict, but it comes from a series that failed V5.

### 12.6 Predictions (section 10) against measured

| | predicted | measured | |
|---|---|---|---|
| step seconds, seq 512 | 850-1,150 | 1036.4, 1036.3 | inside |
| step seconds, seq 2048 | t(512) + 10 to 60 | 1117.61, 1120.52 (t(512) + 81.2, + 84.2) | outside, 21-24 s above the top of the range |
| tokens/s, seq 512 | 0.45-0.60 | 0.494 | inside |
| tokens/s, seq 2048 | 1.7-2.4 | 1.832, 1.828 | inside |
| R1, R2, Rpool | 3.74-3.96 | 3.709, 3.699, 3.704 | below the range by 0.03-0.04 |
| peak alloc, seq 512 | 4,342 MiB | 4,342 | equal |
| peak alloc, seq 2048 | 5,665-5,770 MiB | 6,091 | 321 MiB above the top of the range |
| peak reserved, seq 512 | about 5,050 MiB | 5,052 | inside |
| peak reserved, seq 2048 | 6,120-6,480 MiB | 7,176 | 696 MiB above the top of the range |
| peak working set | 7.6-8.2 GB | 7.60-7.73 GB | inside |
| expert bytes per step | 1,570,336,174,080 | 1,570,336,174,080 | equal |
| drive rate, seq 512 | 1.4-1.9 GB/s | 1.531 | inside |
| drive rate, seq 2048 | 1.3-1.9 GB/s | 1.420, 1.416 | inside |
| wording | fetch-bound, not disk-bound | fetch-bound, not disk-bound | as predicted |

The VRAM estimate of section 2.4 was low at seq 2048 (both allocated and reserved), and the compute increment of section 2.3 was larger than estimated. The gates were not affected by the VRAM miss.

### 12.7 Procedure notes

- The P3 box checks ran before P2 (the rule puts P3 "before anything else"). P2 ran directly as a plain pytest call, not through the lock script. P3 values: `Get-Partition` gave disk 1 for D:, `index.json` existed, `nvidia-smi` listed no compute apps, 13.19 GB RAM was available, `BatteryStatus` was 2.
- A sleep guard ran for the whole series: a wrapper process that sets Windows' execution-state request (`SetThreadExecutionState`) while it lives, around a 7,500 s sleep, stopped after a2. No power setting was changed.
- Available RAM was at least 10 GB at every launch, so no arm waited for it.
- The hold on the shared board was not edited or lifted by the session that ran the arms.
- After the series, the Status line at the top of this file (above section 1) was changed from "PRE-REGISTERED, NOT RUN." to name the run and its outcome. No text of sections 1-11 was edited, and no typo correction was made.
