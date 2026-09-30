<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any arm ran — before the striped cache existed —
which is the point of it: a rule written after the numbers are in is not a
rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (MZAL81T0HFLB-00BL2, 954 GB each):
C: on a CPU root port (PEG0, 00:06.0), D: behind the chipset (RP13, 00:1D.4,
over the DMI 4.0 x8 link). Windows 11 Pro 10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2.
Soup: branch feat/two-drive-striping at f798c46e (the commit this gate starts
from), worktree C:\Users\user\projects\Soup-stripe; soup_cli.__file__ is
stamped under Soup-stripe\src before every arm.
Harness: benchmarks/harness/stream_probe.py (the cold 70B step of gate-971 and
gate-974; --stripe-dir and the direct-I/O / placement stamps added in
f798c46e), one process per arm.
-->

# Gate record — two-drive striping of the disk-tier cache (R4)

**Status: NO VERDICT (2026-09-30, two sequences).** Sequence 2 (`gate2_*`), a
new complete sequence, stopped after four of its six arms on §2's first row.
Round 1's SINGLE arm measured 22.107 s, outside 15.0-21.5 s (one of its three
steps took 31.581 s), so the session did not reproduce the known step. The
same round's STRIPED arm measured 18.052 s; that is recorded, but the
no-verdict row decides first (§6.1). Attempt 3 (sequence 1, `gate_*`) had
stopped earlier on a loss of AC power. It is kept as measured and does not
count toward this verdict. The rule was committed before either run, and §2
and §4 are unchanged.

*Amended 2026-09-30, still before any arm ran:* §4 gains a third void
condition, a suspended box. The first build attempt timed out because the
laptop slept with its lid closed (2026-09-28/29), and a step that spans a
suspend reads as a slow step. §2 is unchanged.

---

## 0. The question

The cold 70B-shaped NF4 step on this box is bound by the read (17.0-17.2 s at
seq 512 and 256, `probe-rtx5070-what-bounds-streaming.md` §21). Two drives read
7.65-9.15 GB/s together against ~4 for one (`probe-rtx5070-two-drive-read.md`,
no formal verdict — its validity row fired three times). This branch writes
decoder layer i to drive i mod 2 and keeps one read in flight per drive. Does
the training step get faster, and by how much?

What the rule's two thresholds mean, as arithmetic and not as a prediction. The
step moves 70.38 GB (157 decoder-layer loads of 441.4 MB plus the embedding and
the head, 536.9 MB each; gate-974 §8). **12.0 s** is 5.87 GB/s averaged over
the step, above the best rate the unbuffered read primitive ever reached on C:
(5.65 GB/s, gate-974 §4), the drive the single-root cache lives on: a step at
or under 12.0 s is not reachable from C: alone at any rate this box has
recorded. **14.0 s** is 5.03 GB/s, 1.2x the 4.09-4.14 GB/s the single-drive
step averages. The floor with the read removed is 6.43 s at seq 512 (§21, arm
B), so no layout takes the step below that.

## 1. The fixture and the two arms

Weights `D:/synth/llama-70b-shape-v32k-f4` — the synthetic Llama-3.1-70B-shaped
bf16 source with a 32k head, random values (`harness/synth_checkpoint.py`);
untied, so the embedding and the head stream through the large-layer slot. NF4,
disk tier, seq 512, batch 1.

- **Arm SINGLE:** the existing one-root cache under
  `C:\Users\user\.soup\layer-stream\` (default `--shards`), `--read-ahead 2`
  (the shipped default).
- **Arm STRIPED:** `--shards
  C:/Users/user/.soup/layer-stream-striped/D___synth__llama-70b-shape-v32k-f4
  --stripe-dir D:/soup-stripe --read-ahead 3` (the feature's default for two
  drives).
- Both arms pinned (the harness default).

The command per arm, with the arm's flags from above in place of `<arm flags>`:

```
PYTHONPATH="C:/Users/user/projects/Soup-stripe/src" python benchmarks/harness/stream_probe.py --weights "D:/synth/llama-70b-shape-v32k-f4" --quant nf4 --tier disk --seq 512 --batch 1 --steps 3 --warmup 1 --step <arm flags> --label "R4 gate round <r> arm <ARM>" --out benchmarks/results/probe-rtx5070/two-drive/gate_r<r>_<arm>.json
```

Everything else is the harness's default and the same in both arms: LoRA r 8 on
q/k/v/o, adapter seed 3, input seed 17, `PagedAdamW8bit`, 2 layer buffers,
`read_ranges` 4. `--read-ahead 3` is passed explicitly because the harness
builds the model through `shard_checkpoint` -> `build_streamed_model` directly;
the N + 1 default for N roots lives in the setup path, which the harness does
not go through. Without the flag the gate would measure a striped cache at
depth 2.

The striped layout: decoder layer i goes to root i mod 2. Root 0 is the
`--shards` folder on C:, which also keeps `extras.safetensors`, the two
`large_*.safetensors` files (embedding, head) and `index.json`; root 1 is
`D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4`, a per-model folder the
sharder creates inside the empty `D:\soup-stripe` made for this gate (the
owner approved it). The 80 decoder layers split 40 / 40 and C: carries the two
large files besides — from the file sizes, roughly 35.7 GB of each step's
reads land on C: and 34.7 GB on D: (arithmetic, not measured).

**Before the rounds, not timed.** The striped cache does not exist yet. It is
built by one STRIPED run with `--steps 1 --warmup 0 --step --skip-events`, which
reads the bf16 source on D: and writes the even layers to the C: folder above
and the odd ones to `D:\soup-stripe\...`; its sharding time and its live
placement line (`direct_io`, layer roots 0,1,0,1,...) are recorded. The SINGLE
cache was checked on the CPU with the sharder's own cache check
(`inspect_shard_cache`, with the arguments the harness passes: bfloat16, nf4,
double quant, device kind cuda, no stripe roots) at 22:57 on 2026-09-28: **hit,
"cache is reusable"** — index format 5, 80 layers, written 2026-09-12 — so it is
reused as is and gets no build run. Build runs are published and are not rounds.

## 2. The decision rule, written before the run

Copied verbatim from the task brief (§2 of `task-8-gate-brief.md`). Nothing in
this section is edited once a number exists.

| measured, per arm: `step_s_mean` of the `--step` block's uninstrumented steps (3 timed after 1 warm-up) | verdict |
|---|---|
| any SINGLE arm outside **15.0-21.5 s** — gate-974 §8's own cold rows are 16.175 s (15.77-17.06) and 17.311 s (16.28-20.42), widened ~5% each side | **no verdict**: the session does not reproduce the known step; record and stop |
| either arm with `direct_io` false or `pinned` false | **no verdict**: the reader did not run the path under test |
| STRIPED <= **12.0 s** in every round | **SHIP**; quote the median and the speed-up over the same round's SINGLE |
| STRIPED <= **14.0 s** in every round, the row above not met | **SHIP, stating the measured number** and that the 12 s target was missed |
| anything else | **DO NOT SHIP**; record why |

Also recorded, not part of the verdict: per-step losses of the two arms in a round (same seeds, same bytes, so equal losses are expected — a difference is reported and investigated), `implied_h2d_gb_per_s`, `stall_share`, peak VRAM, SM clock.

Rounds: three, interleaved, first arm alternating — round 0: SINGLE, STRIPED; round 1: STRIPED, SINGLE; round 2: SINGLE, STRIPED. One process per arm.

**The inputs, as the JSON names them** (fixed with the rule, before the run):

- `step_s_mean` is the `kind: "step"`, `label: "step_plain"` record of the
  arm's JSON. `--step` then runs the same steps again with the CUDA-event
  brackets on (`step_events`); that point is recorded and not judged.
- `direct_io` and `pinned` are `meta.direct_io` and `meta.pinned`.
- The speed-up is the round's SINGLE `step_s_mean` divided by its STRIPED
  `step_s_mean`; "the median" is the median of the three STRIPED means.
- `stall_share` is read from `step_events`: the plain point does not bracket
  the compute stream's waits, so its `stall_share` is 0 by construction.
  `implied_h2d_gb_per_s`, `peak_alloc_gb` and the two SM clock samples are
  printed from both points.
- Numbers are quoted from the JSON at three decimals; none is rounded in a way
  that moves a row.

**The losses, and what a difference would mean.** Equal losses are the
expectation the rule states, and the record already holds evidence that they
will NOT be bit-identical across processes even for one arm: gate-974 §8's NEW
arm at two positions (same code, same cache, same seeds) reads 11.835622 and
11.836516 on its first timed step. The backward kernels on this card are not
deterministic, and every timed step here comes after at least one optimizer
update. So a SINGLE-vs-STRIPED difference within a round is read against the
same-arm difference across rounds (SINGLE against SINGLE, STRIPED against
STRIPED), and "same bytes" is checked directly: after the six arms, never
between them, the striped cache is compared tensor by tensor with the
single-root cache.

## 3. What this does NOT measure

- **A real 70B checkpoint.** Shape only, random values — no quality claim.
- **Warm (page-cached) reads.** Both arms read through direct I/O (the rule's
  second row gives no verdict otherwise), which never touches the page cache,
  so every read is cold whatever ran before it.
- **More than two drives**, or any other pair: one PM9B1 on CPU lanes and one
  behind the chipset is this box's layout, not a general one.
- **Batch > 1**, and any seq other than 512.
- **The RAM tier.**
- **The setup path.** `soup train` reaches striping through the stream setup,
  which applies the N + 1 read-ahead default and the distinct-volume and NVMe
  checks; the harness calls the sharder and the model builder directly and
  passes the depth itself, so none of that wiring is exercised here.
- **Sustained load.** An arm is eight steps (the plain point and the
  instrumented one); a DRAM-less drive's thermal behaviour over a long run is
  not exercised.

## 4. Box state, per arm

Other Claude sessions run test suites on this box. Before the build run they
are asked, by message, to avoid CUDA runs until the gate reports done. Right
before each arm, and again right after it, one stamp is appended to
`benchmarks/results/probe-rtx5070/two-drive/gate_box_state.log`:

- AC status and battery % (`GetSystemPowerStatus`);
- available physical RAM, and commit used / limit (`GlobalMemoryStatusEx`,
  whose page-file fields are the counters behind `Win32_OperatingSystem`'s
  `TotalVirtualMemorySize` / `FreeVirtualMemory`; checked equal on this box
  before the run);
- the Python process count (`tasklist`, the driver itself included);
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory` and the GPU's
  total memory in use (on WDDM the per-process column prints `[N/A]`, so the
  total is what bounds any one process);
- the active scheme's AC ASPM index (`powercfg /q SCHEME_CURRENT SUB_PCIEXPRESS
  ASPM`, read only — this gate changes no setting);
- `soup_cli.__file__` as the arm's own environment resolves it.

**Before an arm starts:** AC on, GPU memory in use <= 1024 MiB, commit headroom
(limit - used) >= 8 GiB. The driver waits for those, polling every 30 s, for at
most 15 minutes; if they do not hold by then the sequence stops and the arms
not run are listed as not run. The first round starts at least 60 s after the
build run ends: the sharder does not flush its writes, and the lazy writer
should not be draining them onto either drive while an arm reads.

**While an arm runs**, the driver samples AC status and commit every 2 s and,
through PDH (English counter names), `\LogicalDisk(C:)` and `\LogicalDisk(D:)`
read and write bytes/s, `\Memory\Pages/sec`, `\Memory\Committed Bytes`,
`\Processor(_Total)\% Processor Time` and
`\Processor Information(_Total)\% Processor Performance` every 1 s. The harness
only reads the two drives, so a write on either one during an arm is someone
else's I/O. These are recorded, never used to set a verdict.

**A void arm.** An arm during which any AC sample reads battery, during which
the box was suspended (two consecutive 2-s power samples more than 30 s apart:
a closed lid sleeps this laptop, and the sampler freezes with it), or that exits
without writing its `step_plain` point, is void: its JSON and log are kept and
published under a `_void` name, and it is re-run once, in the same position,
when the pre-arm checks hold again. A second void in the same position ends the
gate with no verdict.

**Stopping early.** After each arm the driver applies the two rows a single arm
can decide — a SINGLE arm outside 15.0-21.5 s, and `direct_io` or `pinned` not
true in any arm — and stops if either fires, because no later arm can turn a
no-verdict outcome into a verdict. Nothing else stops the sequence.

The driver is `benchmarks/harness/two_drive_stripe_gate.py`, committed on its
own after this file and before the build run. It runs the §1 command, one arm
per process and never two at once, writes each arm's stdout and stderr to its
own `.log` beside its JSON, and keeps every stamp and sample in
`gate_driver.json`.

## 5. Results

Two sequences ran on 2026-09-30. **Attempt 3** (sequence 1, files `gate_*`,
§5.1-§5.7) built the striped cache and stopped after four arms on a loss of AC
power. **Sequence 2** (files `gate2_*`, §5.8-§5.13) is a new, complete
sequence. It reused the striped cache and none of attempt 3's arms. Attempt 3's
subsections below are as written when it stopped.

Attempt 3 ran from worktree HEAD `a0ff99b5`. Between the header's `f798c46e`
and that commit, `src/` and `benchmarks/harness/stream_probe.py` are
byte-identical (`git diff f798c46e a0ff99b5 -- src
benchmarks/harness/stream_probe.py` is empty). Only this record, the driver
and its log changed. Every number below is quoted from the committed JSON in
`results/probe-rtx5070/two-drive/`, at three decimals unless the column says
otherwise.

### 5.1 Attempt 3: the build run (not a round)

`gate_build_striped` ran from 14:56:36 to 15:01:14 local (UTC+5), 275.9 s wall.

- Sharding: **231.812 s** (`meta.shard_seconds`). The log prints
  `Re-sharding layer cache: cache index is missing or unreadable.`, which is
  expected for a cache that did not exist. Model build took 5.974 s.
- Placement: 80 `layer_roots`, exactly `i mod 2` (40 on root 0, 40 on root 1),
  `stripe_roots ['D:\\soup-stripe']`, `direct_io True`, `pinned True`,
  `read_ahead 3`, `AsyncDiskSource`, staging 2398 MB pinned.
- Its one step, untimed (`--warmup 0`, the first step of a fresh process), read
  21.538 s with 158 + 3 loads. It is not a round and is not judged.
- The SINGLE cache got no build run. Its first arm's `shard_seconds` is 0.024 s,
  a cache hit, as §1 recorded on 2026-09-28.

During the build, PDH saw C: written at up to 774.3 MB/s and D: at up to
390.1 MB/s (the sharder's writes). The first round started after the driver's
60 s settle.

### 5.2 Attempt 3: the timed arms, in the order they ran

| run | round | arm | ran (local) | `step_s_mean` (plain) | min-max | tok/s | implied H2D GB/s | `direct_io` / `pinned` | peak alloc GB | SM MHz start->end |
|---|---|---|---|---|---|---|---|---|---|---|
| `gate_r0_single` | 0 | SINGLE | 15:02:32-15:05:16 | **17.414** | 17.277-17.527 | 29.402 | 4.042 | True / True | 4.378 | 180->1432 |
| `gate_r0_striped` | 0 | STRIPED | 15:05:17-15:07:00 | **10.320** | 9.966-10.563 | 49.613 | 6.820 | True / True | 4.378 | 180->1470 |
| `gate_r1_striped` | 1 | STRIPED | 15:07:00-15:08:45 | **10.340** | 10.211-10.407 | 49.516 | 6.806 | True / True | 4.378 | 180->1560 |
| `gate_r1_single_void` | 1 | SINGLE, **void** | 15:08:46-15:11:34 | 16.979 (not used) | 16.829-17.198 | 30.155 | 4.145 | True / True | 4.378 | 1072->1282 |
| re-run of `gate_r1_single` | 1 | SINGLE | **not run**: pre-arm wait 15:11:35-15:26:58, on battery at every poll | | | | | | | |
| `gate_r2_single` | 2 | SINGLE | **not run** | | | | | | | |
| `gate_r2_striped` | 2 | STRIPED | **not run** | | | | | | | |

Every arm that ran moved the same 70,378,258,732 bytes per step (157 layer
loads plus 2 large loads). Peak reserved was 4.798 GB in every arm.

The per-round form the rule reads:

| round | SINGLE `step_s_mean` | STRIPED `step_s_mean` | speed-up (SINGLE / STRIPED) |
|---|---|---|---|
| 0 | 17.414 | 10.320 | 1.687x |
| 1 | void, re-run not run | 10.340 | none |
| 2 | not run | not run | none |

The rule's "median" is the median of three STRIPED means, and it does not
exist here, because only two STRIPED arms ran (10.320 and 10.340 s). The only
in-round speed-up is round 0's.

### 5.3 Attempt 3: the instrumented point (`step_events`), recorded, not judged

`stall_share` is read here, as §2 fixes. `copy_s/step` is the harness's
copy-stream bracket time. It is recorded and not interpreted in this record.

| run | `step_s_mean` | tok/s | implied H2D GB/s | `stall_share` | stall s/step | copy s/step | peak alloc GB | SM MHz start->end |
|---|---|---|---|---|---|---|---|---|
| `gate_r0_single` | 17.226 | 29.722 | 4.085 | 0.000406 | 0.0070 | 11.722 | 4.378 | 1432->1267 |
| `gate_r0_striped` | 10.023 | 51.084 | 7.022 | 0.000054 | 0.0005 | 3.810 | 4.378 | 1470->1545 |
| `gate_r1_striped` | 10.146 | 50.461 | 6.936 | 0.000050 | 0.0005 | 3.855 | 4.378 | 1560->1717 |
| `gate_r1_single_void` | 19.489 (void) | 26.272 | 3.611 | 0.000026 | 0.0005 | 13.466 | 4.378 | 180->1267 |

### 5.4 Attempt 3: losses

The three timed steps of each point. The plain point's steps are process steps
2-4, after one warm-up update. The instrumented point's are steps 6-8.

| run | plain | instrumented |
|---|---|---|
| `gate_r0_single` | 11.935977, 11.829816, 11.611748 | 10.765285, 10.185145, 9.532068 |
| `gate_r0_striped` | 11.950090, 11.839045, 11.606027 | 10.755948, 10.174613, 9.537658 |
| `gate_r1_striped` | 11.943791, 11.809216, 11.616005 | 10.768602, 10.172529, 9.543611 |
| `gate_r1_single_void` | 11.939507, 11.833977, 11.600917 | 10.766873, 10.181394, 9.545099 |

As §2 expected, no two runs are bit-identical. The one valid within-round
difference, round 0's SINGLE minus STRIPED on the plain point, is -0.014113,
-0.009229 and +0.005721. The same-arm difference across rounds, STRIPED r0
minus r1, is +0.006299, +0.029829 and -0.009978. So the cross-arm difference
sits inside the same-arm one. The void arm's losses fall in the same spread;
they are listed and not used. On the first timed step, the step gate-974 §8
compared, the two differences are 0.014113 and 0.006299. That is 7-16x the
0.000894 gate-974 recorded between two processes of one arm. It is reported,
not investigated here. The data half of "same
bytes" is settled directly in §5.5.

### 5.5 Attempt 3: same bytes, the two caches compared tensor by tensor

After the sequence ended, and with no arm running,
`harness/compare_shard_caches.py` matched every safetensors file of the
single-root cache with its striped counterpart (decoder layer i at the root
`layer_roots[i]` names). It compared each tensor's name, dtype, shape and a
SHA-256 of its data bytes, reading plain buffered 64 MiB chunks with no torch
and no mapping. Result (`gate_cache_compare.json`): **all equal**. That is 83
files (80 layers, `extras`, 2 `large_*`), 43 of them on C: and 40 on D:, with
2,405 tensors and 36.388 GB compared in 132 s. No safetensors file exists in
the striped root 0 that the single-root cache lacks. `index.json` is not
compared, because it differs by the stripe fields by design.

### 5.6 Attempt 3: box state and the during-arm samples

From `gate_box_state.log` (a stamp before and after every arm) and
`gate_driver.json` (every 2-s power sample and every 1-s PDH sample):

- **AC.** Every 2-s sample of the three valid timed arms and of the build run
  read AC; battery rose from 89% to 94%. The box lost AC at **15:09:13**
  (System log: Kernel-Power event 105, "Power source change"), 27-28 s into
  `gate_r1_single`. What removed it is not recorded. 69 of that arm's 83 power samples read
  battery, and its after stamp reads AC 0 at 91%. No reconnect event followed.
  At 15:27:51 the box was still discharging, at 84%.
- **No suspend.** The largest gap between consecutive power samples was
  2.016-2.029 s in every arm.
- **Commit headroom** was at least 21.299 GiB at every sample of every timed
  arm (14.503 GiB during the build), and 32.44-34.99 GiB at the before stamps.
  The bar is 8 GiB.
- **GPU.** 0 MiB in use and no compute apps at every stamp.
- **Other processes.** Four `python.exe` at every timed arm's stamps, and six
  at the build's before stamp; the driver is one of them. The command lines
  listed at 14:56 (before the build) and 15:27 (after the stop) were a `gh`
  polling loop, two editor language servers and a short session-audit script
  (present at 14:56, gone by 15:27). No test suite or large copy
  is visible: the count held at four, and neither drive saw more than
  14.61 MB/s of writes during a timed arm.
- **ASPM** AC index 2 at every stamp (read only; nothing was changed).
- `soup_cli.__file__` resolved under `Soup-stripe\src` at every stamp.
- **PDH.** These are means over each arm's whole wall time (process start,
  model build and both step blocks), not over the step.

  | run | C: read GB/s mean / max | D: read GB/s mean / max | C: write MB/s mean / max | D: write MB/s | pages/s mean |
  |---|---|---|---|---|---|
  | `gate_r0_single` | 3.400 / 4.466 | 0.000 / 0.000 | 0.52 / 8.97 | 0.00 | 795 |
  | `gate_r0_striped` | 2.704 / 4.767 | 2.687 / 4.777 | 0.59 / 6.99 | 0.00 | 422 |
  | `gate_r1_striped` | 2.684 / 4.719 | 2.668 / 4.655 | 0.82 / 14.61 | 0.00 | 389 |
  | `gate_r1_single_void` | 3.340 / 4.721 | 0.000 / 0.000 | 0.37 / 13.89 | 0.00 | 262 |

  The harness only reads, so the small writes on C: are someone else's I/O. D:
  saw no writes during any timed arm.

### 5.7 Attempt 3: anomalies, as measured

1. **The driver's `not_run` list is incomplete.** In `gate_driver.json` the
   invocation's `not_run` names only `gate_r2_single` and `gate_r2_striped`.
   The re-run of `gate_r1_single` did not run either, but the list counts an
   arm as run when any attempt has an exit code, and the void attempt has one.
   The `outcome` string names the correct stop position ("pre-arm checks not
   met within 15 min before gate_r1_single"). The driver was not edited after
   the run.
2. **The void arm's clocks.** It sampled 1072 MHz at the start of its plain
   block, where every other arm shows the idle 180 MHz. Its instrumented block,
   run entirely on battery, took 19.489 s against round 0's 17.226 s. It is void
   and used for nothing.
3. **Loss differences.** They are larger than the one cross-process difference
   on record (§5.4).

### 5.8 Sequence 2: how it ran

The driver was invoked at 15:57:47 from worktree HEAD `7599360c` with
`--plan rounds --run-prefix gate2 --settle-s 60`. `7599360c` adds only
`--run-prefix` to the driver, which renames the round files so that attempt
3's committed files are not overwritten. `src/` and
`benchmarks/harness/stream_probe.py` are byte-identical to `a0ff99b5` and
`f798c46e`.

- No build run. The striped cache built in attempt 3 (§5.1), byte-identical to
  the single-root cache (§5.5), was reused. Every arm's `shard_seconds` is
  0.005-0.008 s, a cache hit.
- The box read AC at 15:57:36, before the launch.
- The first arm started at 15:58:50, after the 60 s settle.
- No arm was void.
- After round 1's SINGLE arm, the driver applied §2's first row and stopped:
  `stopped: no verdict: SINGLE arm gate2_r1_single at 22.106659533334703 s is
  outside 15.0-21.5 s`. `gate2_r2_single` and `gate2_r2_striped` were not run.

### 5.9 Sequence 2: the timed arms, in the order they ran

| run | round | arm | ran (local) | `step_s_mean` (plain) | min-max | tok/s | implied H2D GB/s | `direct_io` / `pinned` | peak alloc GB | SM MHz start->end |
|---|---|---|---|---|---|---|---|---|---|---|
| `gate2_r0_single` | 0 | SINGLE | 15:58:50-16:01:41 | **18.133** | 17.917-18.275 | 28.236 | 3.881 | True / True | 4.378 | 180->1312 |
| `gate2_r0_striped` | 0 | STRIPED | 16:01:41-16:03:31 | **11.367** | 11.036-11.914 | 45.041 | 6.191 | True / True | 4.378 | 330->1530 |
| `gate2_r1_striped` | 1 | STRIPED | 16:03:32-16:06:12 | **18.052** | 17.894-18.204 | 28.362 | 3.899 | True / True | 4.378 | 457->1320 |
| `gate2_r1_single` | 1 | SINGLE | 16:06:13-16:09:31 | **22.107** | 16.974-31.581 | 23.160 | 3.184 | True / True | 4.378 | 420->1492 |
| `gate2_r2_single` | 2 | SINGLE | **not run** (the driver stopped on §2's first row) | | | | | | | |
| `gate2_r2_striped` | 2 | STRIPED | **not run** | | | | | | | |

The three plain steps of `gate2_r1_single` were 31.581, 17.765 and 16.974 s,
so one step makes its mean. Those of `gate2_r1_striped` were 17.894, 18.204
and 18.059 s, uniformly slow. Every arm moved 70,378,258,732 bytes per step
(157 + 2 loads), and peak reserved was 4.798 GB in every arm.

| round | SINGLE `step_s_mean` | STRIPED `step_s_mean` | speed-up (SINGLE / STRIPED) |
|---|---|---|---|
| 0 | 18.133 | 11.367 | 1.595x |
| 1 | 22.107 (outside 15.0-21.5) | 18.052 | 1.225x |
| 2 | not run | not run | none |

The rule's median of three STRIPED means does not exist: round 2 was not run.

### 5.10 Sequence 2: the instrumented point (`step_events`), recorded, not judged

| run | `step_s_mean` | tok/s | implied H2D GB/s | `stall_share` | stall s/step | copy s/step | peak alloc GB | SM MHz start->end |
|---|---|---|---|---|---|---|---|---|
| `gate2_r0_single` | 18.653 | 27.448 | 3.773 | 0.000687 | 0.0128 | 14.150 | 4.378 | 1312->1380 |
| `gate2_r0_striped` | 10.179 | 50.300 | 6.914 | 0.000056 | 0.0006 | 3.857 | 4.378 | 1530->1567 |
| `gate2_r1_striped` | 17.961 | 28.506 | 3.918 | 0.000021 | 0.0004 | 12.710 | 4.378 | 1320->1605 |
| `gate2_r1_single` | 20.176 | 25.376 | 3.488 | 0.000017 | 0.0003 | 14.680 | 4.378 | 1492->1440 |

### 5.11 Sequence 2: losses

| run | plain | instrumented |
|---|---|---|
| `gate2_r0_single` | 11.941118, 11.828694, 11.597034 | 10.762340, 10.179412, 9.541021 |
| `gate2_r0_striped` | 11.955779, 11.843419, 11.606232 | 10.769289, 10.173697, 9.538158 |
| `gate2_r1_striped` | 11.943206, 11.842164, 11.607185 | 10.766464, 10.176335, 9.522297 |
| `gate2_r1_single` | 11.941950, 11.832151, 11.606636 | 10.758423, 10.193248, 9.533957 |

On the plain point:

- **Cross-arm, same round** (SINGLE minus STRIPED): round 0 -0.014661,
  -0.014725, -0.009197; round 1 -0.001256, -0.010013, -0.000549.
- **Same arm, across rounds** (r0 minus r1): SINGLE -0.000832, -0.003457,
  -0.009602; STRIPED +0.012573, +0.001255, -0.000954.
- Within this sequence, the largest cross-arm difference (0.014725) is
  slightly larger than the largest same-arm one (0.012573). Attempt 3's
  same-arm spread reached 0.029829 (§5.4).

**Pooled over both sequences, which is observed after the fact, not a
finding.** On the first timed step, all four SINGLE losses on record fall
below all four STRIPED ones. The SINGLE ones are 11.935977 to 11.941950 (the
void arm included), and the STRIPED ones are 11.943206 to 11.955779. On the
second and third steps the two sets overlap. The two caches hold the same
bytes (§5.5), so this is not a data difference. It is noticed across eight
numbers and not investigated here.

### 5.12 Sequence 2: box state and the during-arm samples

- **AC** at every 2-s sample of every arm. The battery was **charging, 68% to
  84%**, over the sequence; in attempt 3's valid arms it read 89-94%.
- **No suspend.** The largest gap between power samples was 2.016-2.036 s.
- **Commit headroom** was at least 20.429 GiB at every sample, and 32.25-33.83
  GiB at the before stamps.
- **GPU:** 0 MiB and no compute apps at every stamp. **ASPM** AC index 2 at
  every stamp. `soup_cli.__file__` resolved under `Soup-stripe\src` at every
  stamp.
- **Python processes:** 4 at every stamp except the after stamp of
  `gate2_r1_single` (16:09:31), which read 5. At 16:09:53 only the `gh` polling
  loop and the two editor language servers remained, so the fifth was
  transient and its command line was not captured.
- **System log, 15:58-16:10:** one Service Control Manager event 7040 at
  15:59:43 (the Background Intelligent Transfer Service's start type changed),
  and one Kernel-Power 566 session-state event at 16:09:49, after the last arm.
  No disk, storage, thermal or power-source event.

| run | C: read GB/s mean / max | D: read GB/s mean / max | C: write MB/s mean / max | D: write MB/s max | pages/s mean |
|---|---|---|---|---|---|
| `gate2_r0_single` | 3.266 / 4.388 | 0.000 / 0.000 | 0.90 / 15.60 | 0.02 | 1699 |
| `gate2_r0_striped` | 2.536 / 4.740 | 2.518 / 4.727 | 0.57 / 9.15 | 0.06 | 860 |
| `gate2_r1_striped` | 1.717 / 4.670 | 1.707 / 4.399 | 0.82 / 25.15 | 0.00 | 257 |
| `gate2_r1_single` | 2.804 / 4.580 | 0.000 / 0.000 | 0.23 / 8.61 | 0.00 | 49 |

The 10-s bins of the same PDH samples (`gate_driver.json`, seconds after
each arm started) show where round 1's time went:

- `gate2_r0_striped`: each drive read 2.7-3.4 GB/s in every bin from 20 s to
  100 s.
- `gate2_r1_striped`: each drive read 3.0-3.1 GB/s in the 20-s bin, then
  1.75-1.97 GB/s in every bin from 30 s to 140 s, both drives together.
- `gate2_r1_single`: C: read 2.07-2.23 GB/s in every bin from 20 s to 70 s,
  then 2.50-4.07 GB/s from 80 s on. Its 31.581 s step is its first timed step.

On the wall clock, those two slow stretches meet, about 16:04:02 to 16:07:33.
Every 10-s bin with reads in that interval reads 2.23 GB/s per drive or less:
1.75-2.23, and 1.40-1.45 in the last, partial bin of `gate2_r1_striped`, where
that arm was ending. The same arms read 2.5-4.1 GB/s outside it.

### 5.13 Sequence 2: anomalies, as measured

1. **Round 1 is slow in both arms, on both drives at once** (§5.12). The
   slowdown spans the end of one arm and the start of the next, a different
   process each. What caused it is not established. Two things are recorded
   and not tested: the battery was charging throughout sequence 2 and not
   during attempt 3's valid arms, and a service start-type event came at
   15:59:43. The harness only reads, and D: saw no writes.
2. **SINGLE round 0 read 18.133 s**, against attempt 3's 17.414 s. Both are
   inside the band.
3. **SM clock at the start of the plain block** was 330, 457 and 420 MHz in
   three of the four arms, not the idle 180 MHz of attempt 3's valid arms.
4. **The first-step loss separation** pooled over both sequences (§5.11).

## 6. Verdict

### 6.1 Sequence 2 (`gate2_*`): **NO VERDICT**

Attempt 3's arms (§5.1-§5.7, §6.2) stay in this record as measured and do not
count toward this verdict.

| row | inputs (sequence 2) | result |
|---|---|---|
| any SINGLE arm outside 15.0-21.5 s | round 0: 18.133 s (inside). Round 1: **22.107 s (outside)**. Round 2: not run | **fires: no verdict**, "the session does not reproduce the known step" |
| either arm with `direct_io` or `pinned` false | True / True on all four arms | does not fire |
| STRIPED <= 12.0 s in every round | 11.367 (round 0), 18.052 (round 1), round 2 not run | not reached: the first row has decided |
| STRIPED <= 14.0 s in every round | the same | not reached |
| anything else: DO NOT SHIP | none | not reached |

The rule's rows are applied in order. The first row is a validity condition
whose outcome is "no verdict ... record and stop", and "anything else" means
what no earlier row covers. The driver applied that row after round 1's SINGLE
arm and stopped, as §4 says, because no later arm can turn a no-verdict
outcome into a verdict.

What sequence 2 measured, and no more:

- Round 0: SINGLE 18.133 s, STRIPED 11.367 s (1.595x).
- Round 1: STRIPED 18.052 s, above both SHIP thresholds, and in the same round
  SINGLE 22.107 s, outside its validity band. By the rule, the session did not
  reproduce the known step in that round, so the 18.052 s is recorded and
  sets nothing.
- Round 1's two arms ran through one stretch in which every drive read well
  below its rate outside that stretch (§5.12).

Across both sequences, as a count and not a verdict: three of the four
STRIPED arms on record are under 12.0 s (10.320, 10.340, 11.367). The fourth,
18.052 s, is in the round whose SINGLE arm fell outside the band.

### 6.2 Attempt 3 (sequence 1, `gate_*`), as written when it stopped

**NO VERDICT.** The §2 rule applied row by row to the arms that exist:

| row | inputs | result |
|---|---|---|
| any SINGLE arm outside 15.0-21.5 s | round 0: 17.414 s. Round 1: void, re-run not run. Round 2: not run | does not fire on the one valid SINGLE arm; rounds 1 and 2 have no SINGLE input |
| either arm with `direct_io` or `pinned` false | True / True on all three valid arms (and on the void arm and the build run) | does not fire |
| STRIPED <= 12.0 s in every round | 10.320 (round 0), 10.340 (round 1), round 2 not run | **no input for round 2**: "every round" cannot be established |
| STRIPED <= 14.0 s in every round | the same | **no input for round 2** |
| anything else: DO NOT SHIP | none | not applied (below) |

Why this is "no verdict" and not "anything else: DO NOT SHIP". §4 ends the
sequence when the pre-arm checks do not hold within 15 minutes, and lists the
remaining arms as not run. That happened before the void arm's re-run, the
fourth of six positions. The rule's rows take three rounds as input, and two
SINGLE arms and one STRIPED arm of those three rounds do not exist. The last
row is read here as covering a complete set of rounds that meets neither SHIP
row, not a sequence the protocol stopped. The one stop §4 names an outcome for,
a second void in the same position, ends with no verdict. A void whose re-run
could never start is read the same way. The reading is stated so that it can
be overruled. The measured numbers do not depend on it.

What the arms that ran show, and no more: two STRIPED steps at 10.320 s and
10.340 s, both under the 12.0 s line, and one valid SINGLE step at 17.414 s
in the same round (1.687x). The caches hold the same bytes (§5.5). These are
measurements, not a verdict, because the rule asks for every round and a third
STRIPED arm never ran.

Not decided here: whether a new, complete sequence runs. The driver has no
resume, so a verdict needs all six arms again, under the same §1 commands and
the same §2 and §4. It would be published beside this attempt, which stays as
measured.
