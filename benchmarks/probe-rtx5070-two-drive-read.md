<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE the harness read a single byte, which is the point
of it: a rule written after the numbers are in is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (MZAL81T0HFLB-00BL2, 954 GB each),
Windows 11 Pro 26100. Stack: Python 3.12.10, torch 2.14.0+cu130.
Soup: branch probe/two-drive-read cut from origin/main d7a4f7a7, worktree
C:\Users\user\projects\Soup-read.
Harness: benchmarks/harness/two_drive_read.py (committed with this file). It
imports the read primitive from benchmarks/harness/issue974_unbuffered.py, so
the bytes are read exactly the way gate-974 read them.
-->

# Probe — does the second NVMe add read bandwidth? (disk-tier read rate, R1)

**Status: decision rule committed 2026-09-28, BEFORE the run. Results pending.**

---

## 0. The question, and why it decides a feature

The cold 70B-shaped step on this box is bound by the read outright: 17.0-17.2 s
at seq 512 AND at seq 256, while the compute floor with the read removed halves
(6.4 -> 3.8 s). The step moves 70.38 GB at 4.09-4.14 GB/s averaged
(`probe-rtx5070-what-bounds-streaming.md` §21, Finding 21). One drive's
unbuffered cold ceiling is 5.65 GB/s at K = 2-4 parallel ranges (gate-974 §4).
So on one drive the step cannot go below ~12.5 s whatever the reader does.

The box has TWO drives, and the disk tier reads from one: the NF4 shard caches
live on C:, the bf16 fixtures on D:. If the two deliver ~2x together, a store
striped across both could take the 70B step to its compute floor (~6.5 s at
1x512), and every giant-MoE row in the roadmap's L2L table falls by the same
factor, because those rows are `read seconds x compute ceiling` tokens per step.
If the drives share a bottleneck, striping is work for nothing. The roadmap has
said "a second NVMe halves it" since 2026-09-24; nobody measured it. This does,
before any sharder or reader code changes.

## 1. The two drives and where they attach

| | C: | D: |
|---|---|---|
| physical disk | 0 | 1 |
| model | Samsung MZAL81T0HFLB-00BL2 (PM9B1), 954 GB | same |
| filled | 730 GB used, 224 GB free | 156 GB used, 798 GB free |
| controller location | `PCIROOT(0)#PCI(0600)#PCI(0000)` | `PCIROOT(0)#PCI(1D04)#PCI(0000)` |
| root port | `PEG0` = 00:06.0, Intel `8086:A74D` — a **CPU** root port | `RP13` = 00:1D.4, Intel `8086:7A34` — a **chipset (PCH)** root port |
| path to memory | CPU lanes directly | the DMI 4.0 x8 link (~15.75 GB/s), shared with the chipset's other devices |
| what lives there | the NF4 shard caches (two 70B stores, 36.4 GB each) | the bf16 fixtures (138 GB 70B-shaped, 27.5 GB 14B-shaped) |

The CPU is an i9-14900HX (Raptor Lake HX). One drive on CPU lanes and one
behind the chipset is the usual layout on this class of laptop, and it is the
reason the answer is not obvious either way: DMI 4.0 x8 carries ~15.75 GB/s,
enough for one drive with room to spare, but it is not the drive's own link.

## 2. The decision rule, written before the run

Inputs, per round, at K = 4 ranges per span (the reader's default
`read_ranges`): `alone_A`, `alone_B` — each drive reading alone; `rate_A`,
`rate_B` — each drive's rate in the both-at-once arm, over the window in which
BOTH were reading. Then

```
r_sum = (rate_A + rate_B) / max(alone_A, alone_B)      # weighted striping's ceiling
r_alt = 2 x min(rate_A, rate_B) / max(alone_A, alone_B) # plain alternate-layer striping
```

The rule reads the **median over three rounds**. A = C:, B = D:.

| measured (medians) | verdict | what follows |
|---|---|---|
| **`alone_A` outside 4.5-6.2 GB/s** | the instrument does not reproduce gate-974's 5.1-5.65 GB/s at K = 2-4 on this drive | **no verdict.** Record the run, find out why, re-run. A comparison of two drives on a session that does not reproduce the known one is not evidence |
| **`r_sum >= 1.6` AND `r_alt >= 1.6`** | the drives are independent | build **plain alternate-layer striping** (layer i on drive i mod 2) in the sharder and the reader, gated bit-exact against the single-drive disk tier. Predicted 70B step: `max(70.38 GB / (r_alt x 4.1 GB/s), 6.4 s)` — a prediction until the step measures it |
| **`r_sum >= 1.6 > r_alt`** | independent but unequal | build **weighted** striping (bytes per drive in proportion to its rate — Colibri's 9 + 3 GB/s scheme); plain alternation would leave the difference on the table |
| **`1.2 <= r_sum < 1.6`** | partly shared | **do not build yet.** Measure where the reader loses 4.1 vs 5.65 GB/s on one drive first (R3): that gap is worth 1.37x with no layout change, and the second drive is worth `r_sum` at most |
| **`r_sum < 1.2`** | a shared bottleneck | **do not build.** The second drive is capacity, not bandwidth; say so in the roadmap and stop |

**Why 1.6.** 2.0 is perfect. At 1.6 the 70B step's read goes from ~17 s to
~10.7 s at today's reader efficiency — a 1.6x step for a sharder-layout change
and a reader change, which is worth building. Below it the gain is smaller than
what the single-drive reader gap (R3, 1.37x) and pinned RAM residency (~1.5x
at ~12 GB resident, an estimate) offer without touching the layout, so it does
not go first.

**What the rule does NOT decide**, stated before the numbers so none of it is
read into them afterwards: whether the reader reaches this rate inside the real
step (R3 and the striping gate measure that); the host-to-device side (at
2 x 5.65 GB/s the copy needs ~11 GB/s, against ~16.7 GB/s pinned H2D); sustained
reads past a few seconds (thermal throttling of a DRAM-less drive is not
exercised by ~2 s arms).

## 3. Instrument

- `two_drive_read.py`, the #974 primitive: `CreateFileW(FILE_FLAG_NO_BUFFERING
  | FILE_FLAG_SEQUENTIAL_SCAN)`, one handle per range, one `ReadFile` per range,
  into a sector-aligned **pinned** arena (one per drive). Unbuffered reads never
  touch the page cache, so there is nothing to evict and every read is cold.
- A span = 400 MiB, sector-aligned, wholly inside one file — the size of one
  70B NF4 layer file (those are ~430 MB; the span is their first 400 MiB). On
  C: the spans are the layer files of the two 70B NF4 stores in file order; on
  D: consecutive 400 MiB windows of the bf16 fixtures. Each span is read as
  K = 4 parallel ranges of 100 MiB, and the next span starts when all four have
  landed — the reader's shape. **No span is read twice in a run.**
- Per drive: one untimed warm-up span. Then three rounds; each round runs
  A alone, B alone and both at once, 24 spans (10.07 GB) per drive per arm, in
  an order that rotates so each arm runs first once.
- In the both-at-once arm the two drives start from a barrier. Each drive's
  rate is its bytes inside the window where both were reading, divided by the
  window; a span that straddles the window's end is pro-rated. Aggregate over
  the whole arm and each drive's own whole-arm rate are recorded beside it.
- Byte check after every arm: the first and last MiB of the last span read into
  each arena against a buffered read of the same offsets. A mismatch makes the
  run exit non-zero.

## 4. Box state

This runs beside the contributor loop's own sessions (pytest suites and
auto-merge loops) and a separate local project's llama-server. Each round
stamps available physical memory, commit, the Python process count and the
GPU's memory/utilisation/temperature into the JSON. The validity row in §2 is
what catches a session contaminated by someone else's I/O.

## 5. Results

### Run 1 — 2026-09-28 10:28, NO VERDICT: the validity row fired

JSON `benchmarks/results/probe-rtx5070/two-drive/r1_k4.json`; `soup_cli`
stamped from this worktree (`Soup-read\src`); every byte check OK. GB/s:

| round (order) | A = C: alone | B = D: alone | both: A | both: B | both: aggregate | window (s) | r_sum | r_alt |
|---|---|---|---|---|---|---|---|---|
| 0 (a, b, ab) | 3.62 | 4.27 | 3.64 | 4.95 | 7.65 | 2.03 | 2.014 | 1.707 |
| 1 (b, ab, a) | 3.94 | 3.94 | 4.04 | 5.27 | 7.90 | 1.91 | 2.362 | 2.049 |
| 2 (ab, a, b) | 3.85 | 5.83 | 4.46 | 5.75 | 8.69 | 1.75 | 1.751 | 1.529 |
| **median** | **3.855** | 4.266 | 4.035 | 5.270 | — | — | 2.014 | 1.707 |

**Verdict under §2: none.** `alone_A` = 3.855 GB/s is outside 4.5-6.2. The
r medians would sit in the "independent drives" row, and they are **not read as
a verdict**: the rule says a comparison made on a session that does not
reproduce the known drive is not evidence, and it was written before this run
for exactly this case.

**What the run's own stamps show about why.** At the start the box had
**3.03 GB of available physical memory and commit 56.65 GB against a limit of
56.69** — at the limit, with the pagefile expanded (the rounds then read
3.18 / 3.89 / 4.89 GB available, commit 56.6 / 53.8 / 51.0). The pagefile is
`C:\pagefile.sys` (peak use since boot 9.7 GB). gate-974's unbuffered rows were
measured at 19.34 GB free and commit 21.6 of 47.35 GB. **This run logged no disk
counters, so paging I/O on C: during the arms is a hypothesis, not a
measurement.** Two patterns fit it and neither proves it: C: was the slower
drive in two of the three alone pairs and tied in the third, while D: — which
nothing else touches — reached 6.5 GB/s on single spans and a 6.05 GB/s median
per span in round 2, above anything gate-974 recorded on C:; and **each drive
read faster in the both-at-once arm than alone in five of six comparisons**,
which a shared bottleneck cannot produce and which points at the host (CPU
power state, background load) rather than at the drives. That last pattern is
not explained here.

**A defect in the rule's validity row, found while diagnosing — disclosed, NOT
corrected.** The 4.5-6.2 GB/s bound was derived from a summary of gate-974
("5.1-5.65 GB/s at K = 2-4") that had kept its best readings. gate-974 §4's own
K = 4, one-request-per-range readings are **5.12 GB/s (run A, layers 12-17) and
4.18 (run B, layers 54-59)**, its K = 2-8 one-request readings span 4.02-5.30,
and it calls the difference a 20-30% drive/position effect. The committed bound
is therefore tighter than the reference it names: a quiet-box C: reading of
~4.2 GB/s would fail it while matching gate-974's own run B. The row stays as
committed — changing a validity bound after seeing the numbers it rules on is
what pre-registration exists to prevent. If a later run lands between 4.0 and
4.5 GB/s on a quiet box, the record says so and the call goes to the owner.

**Instrument change before run 2** (the rule is unchanged): the harness now
samples Windows performance counters through PDH every 0.25 s on a background
thread — `LogicalDisk` read and write bytes per volume, `Memory\Pages/sec` and
`Committed Bytes`, `Processor\% Processor Time` and `Processor Information\%
Processor Performance` (the actual clock as a share of nominal). The harness
only reads, so **a write on either volume during an arm is someone else's
I/O**; the read counter cross-checks the harness's own rate. Also fixed: the
box stamp's Python-process count was lost to a decoding error (`tasklist`
prints in the OEM code page, and the no-break space in its memory column is
0xFF in cp866), so run 1's `python_processes` is `null`.

## 6. Verdict

*Pending.*

## 7. What this does NOT measure

- **The streamed step.** This is the read primitive on two drives, not the
  reader and not a training step. A striped reader has to earn its own number.
- **The copy to the GPU.** Bytes land in pinned host memory and stop there.
- **Sustained load.** Arms are ~2 s; a DRAM-less drive's thermal throttle under
  minutes of reads is not exercised.
- **Any drive other than these two.** Two identical PM9B1 on one CPU and one
  PCH root port; a desktop with both drives on CPU lanes, or two different
  drives, is a different measurement.
