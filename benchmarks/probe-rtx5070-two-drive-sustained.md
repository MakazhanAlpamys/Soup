<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any arm ran, and before the striped cache it reads
was rebuilt, which is the point of it: a rule written after the numbers are in
is not a rule.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (MZAL81T0HFLB-00BL2, 954 GB each):
C: on a CPU root port (PEG0, 00:06.0), D: behind the chipset (RP13, 00:1D.4,
over the DMI 4.0 x8 link). Windows 11 Pro 10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2 (read from the installed packages on 2026-10-01).
Soup: branch probe/two-drive-sustained, cut at 2c6ed087 (the head of PR #1536,
feat/two-drive-striping), worktree C:\Users\user\projects\Soup-sustain.
src/ is byte-identical to 2c6ed087; only benchmarks/ changed.
soup_cli.__file__ resolves under Soup-sustain\src (checked on 2026-10-01, and
stamped before and after every arm).
Harness: benchmarks/harness/stream_probe.py, the gate's cold 70B step, with one
change in 6ee60ce0: each timed step records its wall-clock start
(started_unix), read outside the timed region. The driver is
benchmarks/harness/two_drive_sustained.py (6ee60ce0), which imports the gate's
driver two_drive_stripe_gate.py unchanged. One process per arm.
-->

# Probe — does the two-drive speed-up hold under sustained reading? (R4)

**Status: NO VERDICT (§2 row 2), 2026-10-01.** The rule (§2) was committed in `119991f8`,
before the run. The sequence rebuilt the striped cache and ran round 0 and round 1's STRIPED arm.
It then stopped inside round 1's SINGLE arm, when the laptop lost AC and hibernated at 20:49 local.
That arm is void under §4. It was not re-run, and round 2 never ran, so the rule's three rounds do
not exist (§5.1, §6). Round 0 alone reads u_late 2.065, which is **not a verdict**. Both STRIPED
arms had recurring stretches of 13.5-s steps, with both drives at about 2.5 GB/s; this is
recorded, not judged (§5.7).

---

## 0. The question

The two-drive gate ([`gate-two-drive-striping.md`](gate-two-drive-striping.md), SHIP)
timed three-step bursts, one process per arm, on the cold 70B-shaped NF4 step at seq 512,
batch 1. Its decided sequence read STRIPED at 10.050, 9.971 and 9.929 s and SINGLE at
17.552, 17.431 and 17.438 s, a speed-up of 1.746-1.756x (median 1.748x, gate §5.18). The
judged block of each arm was three timed steps, about 30 s (STRIPED) or 52 s (SINGLE) of
reading, and a whole arm read for about 80 s or 140 s.

Training does not read for 50 s. It reads for hours, and a striped step keeps both drives
reading at about 3.5 GB/s each the whole time. PR #1536 says so under what it does not cover:
"The gate times three-step bursts. A sustained multi-minute striped run is not measured; one
aborted sequence saw both drives at about half rate for 3.5 minutes, cause not found." That
sequence is the gate's sequence 2 (§5.12): from about 16:04:02 to 16:07:33, five minutes after
its first arm started, every 10-s bin read 2.23 GB/s per drive or less, against 2.5-4.1 GB/s
outside that stretch, in two arms and two processes. Sequence 4 then ran its six arms back
to back for 12 min 40 s with no dip (§5.21). Nothing on record says which of the two a long
run looks like.

So: **when each arm reads for about six minutes without a break, and the sequence reads for
about forty, does the striped speed-up hold, or do the drives (or the box) slow down?** And
does the SINGLE arm drift the same way? The number that matters to a user is the ratio of
the two under sustained load, not either arm alone.

The arithmetic of the load, not a prediction. Each step reads 70.38 GB (157 decoder-layer
loads of 441.4 MB plus the embedding and the head, 536.9 MB each; gate §0). A STRIPED arm of
37 steps reads about 2.60 TB, about 1.32 TB of it from C: and 1.28 TB from D: (35.7 GB and
34.7 GB per step, gate §1's arithmetic from the file sizes). A SINGLE arm of 22 steps reads
about 1.55 TB, all from C:.

## 1. The fixture and the two arms

The gate's fixture, unchanged except for the step counts: weights
`D:/synth/llama-70b-shape-v32k-f4` (synthetic Llama-3.1-70B shape, bf16, 32k head, random
values, untied); NF4, disk tier, seq 512, batch 1; LoRA r 8 on q/k/v/o, adapter seed 3, input
seed 17, `PagedAdamW8bit`, 2 layer buffers, `read_ranges` 4; both arms pinned.

- **Arm SINGLE:** the one-root cache under `C:\Users\user\.soup\layer-stream\` (the default
  `--shards`), `--read-ahead 2`, `--steps 21 --warmup 1 --step --skip-events`.
- **Arm STRIPED:** `--shards
  C:/Users/user/.soup/layer-stream-striped/D___synth__llama-70b-shape-v32k-f4 --stripe-dir
  D:/soup-stripe --read-ahead 3`, `--steps 36 --warmup 1 --step --skip-events`.

The command per arm, as the driver builds it (`<arm flags>` and `<steps>` from above):

```
python benchmarks/harness/stream_probe.py --weights D:/synth/llama-70b-shape-v32k-f4 --quant nf4 --tier disk --seq 512 --batch 1 <steps> --step --skip-events <arm flags> --label "R4 sustained round <r> arm <ARM>" --out benchmarks/results/probe-rtx5070/two-drive/sustain_r<r>_<arm>.json
```

with `PYTHONPATH=C:/Users/user/projects/Soup-sustain/src` set by the driver.

**Why 21 and 36 steps.** The question is about time under load, so both arms read for the
same time, not the same number of steps. At the gate's sequence-4 medians, 21 SINGLE steps
take 21 x 17.44 = 366 s and 36 STRIPED steps 36 x 9.97 = 359 s: about six minutes of timed
steps each, after one warm-up step. A common step count would give the faster arm less time
(24 STRIPED steps are about 4 minutes, 24 SINGLE steps about 7), so the faster arm's late
window would sit earlier in its run than the slower arm's. Six minutes per arm runs each late
window to the end of the sixth minute of reading inside one process, past the point at which
sequence 2's slowdown began (about five minutes after that sequence's first arm started,
counted across arms). `--skip-events` keeps each
arm one uninterrupted block of uninstrumented steps. The gate's instrumented block would split
the timed steps into two blocks with an untimed warm-up between them, and nothing in §2
reads it.

**The two windows, fixed now.** Timed step k (k = 1..N) is `records[k-1]` of the arm's
`step_plain` point; process step 1 is the warm-up and is not recorded.

| window | SINGLE (N = 21) | STRIPED (N = 36) | where it sits, at the gate's step times |
|---|---|---|---|
| early | timed steps 1-5 (process steps 2-6) | timed steps 1-5 | the first ~87 s (SINGLE) / ~50 s (STRIPED) of the timed block: the gate's burst, two steps longer |
| late | timed steps 12-21 (the last 10) | timed steps 27-36 (the last 10) | from ~192 s / ~259 s to the end at ~366 s / ~359 s |

**The striped cache has to be rebuilt first.** The PR's final code names stripe root 1's
folder `<slug>-<12 hex of a SHA-256 of the primary cache's realpath>`
(`stripe_roots.stripe_folder_name`, the one resolver `layer_shard.stripe_dirs` uses), and the
folder's `stripe.json` names its primary cache. The gate's striped cache was
built at f798c46e, before that change: its odd layers are in
`D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4`, with a marker that has no
`primary_cache` field. For the STRIPED `--shards` above, the code under test names root 1's
folder **`D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80`** (computed with
`stripe_folder_name` on 2026-10-01; the driver's dry run prints the same). The sharder's own
cache check, run on the CPU on 2026-10-01 with the arguments the harness passes (bfloat16,
nf4, double quant, device kind cuda; metadata reads only, no torch import):

- SINGLE, no stripe roots: **hit**, "cache is reusable". It is reused as is.
- STRIPED, stripe root `D:\soup-stripe`: **miss**, "stripe marker
  D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80\stripe.json is missing or
  unreadable". The first STRIPED process re-shards.

The STRIPED arm keeps the gate's `--shards` path. The default `--shards` is the SINGLE cache:
pointed at it with `--stripe-dir`, a STRIPED arm would find the stripe roots changed,
re-shard the SINGLE arm's cache into the striped layout, and the next SINGLE arm would
re-shard it back. The gate's separate primary keeps the SINGLE cache untouched, and the
rebuild rewrites the same 44 files on C: instead of adding a new 18.7 GB set (C: had 116 GB
free on 2026-10-01, `df`).

**Before the rounds, not timed: one build run** (`--plan build`). It is one STRIPED arm with
`--steps 1 --warmup 0 --step --skip-events`. It reads the bf16 source on D:, writes the even
layers, `extras.safetensors`, the two `large_*.safetensors` files and `index.json` to the C:
folder above (about 18.7 GB), and writes the 40 odd layers (40 x 441,433,020 bytes =
17.66 GB) and the marker to the new D: folder. The gate's build of the same cache took
275.9 s wall, 231.812 s of it sharding (gate §5.1). Its sharding time, its live placement
line and its one untimed step are recorded. It is not a round. The old folder
`D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4` is read by nothing at this commit and is
left where it is; deleting it is the owner's call.

## 2. The decision rule, written before the run

Nothing in this section is edited once a number exists.

**Inputs, per arm and per round:**

- `E` and `L`: the median `step_s` of the arm's early and late windows (§1). The median of
  10 values is the mean of the 5th and 6th.
- `d = L / E`, the arm's drift.
- `u_early(r) = E_SINGLE / E_STRIPED` and `u_late(r) = L_SINGLE / L_STRIPED`, in round r.
- A drive's window rate: the mean of the 1-s PDH samples of
  `\LogicalDisk(X:)\Disk Read Bytes/sec` stamped inside the window, from the first step's
  `started_unix` to the last step's `started_unix + step_s`.

| row | measured | verdict |
|---|---|---|
| 1 | any SINGLE arm whose `E` is outside **15.0-21.5 s** | **NO VERDICT**: the session does not reproduce the known step; record and stop |
| 2 | any arm off the path under test: `direct_io` or `pinned` not true; a source other than `AsyncDiskSource`; read-ahead not 2 (SINGLE) or 3 (STRIPED); a STRIPED arm whose stripe roots are not `D:\soup-stripe` or whose 80 layers are not placed i mod 2, or a SINGLE arm with any stripe root; a timed step that did not load 157 + 2; a round arm with `shard_seconds` above **30 s** (it re-sharded). Or the protocol stopped: a position void twice, or a pre-arm wait expired (§4), so some round lacks an arm | **NO VERDICT**: the arms did not measure what the rule reads |
| 3 | `u_late` >= **1.595** in every round | **SUSTAINED**. Quote `u_late` per round and its median, `u_early` per round, and `d` per arm |
| 4 | `u_late` >= **1.20** in every round, row 3 not met | **DEGRADES**: the speed-up shrinks under sustained reading and stays at or above the gate's 1.2x line. Quote as row 3, and name every arm with `d` above **1.080** and every drive the arm reads (C: for SINGLE, both for STRIPED) whose early-window rate exceeds its late-window rate by more than **1.10x** |
| 5 | anything else: some round's `u_late` below 1.20 | **DEGRADES BELOW THE GATE'S FLOOR**. Quote and name as row 4 |

The rows are applied in order; the first that fires decides. Rows 1 and 2 are validity
conditions. "Every round" means the three rounds of §4's sequence.

**Where every threshold comes from.** Each is read from the committed gate record; the
source line is quoted.

| threshold | value | source in `gate-two-drive-striping.md` |
|---|---|---|
| SINGLE band (row 1) | 15.0-21.5 s | §2, row 1: "any SINGLE arm outside **15.0-21.5 s** — gate-974 §8's own cold rows are 16.175 s (15.77-17.06) and 17.311 s (16.28-20.42), widened ~5% each side" |
| burst speed-up (reference) | 1.748x (1.746-1.756) | §5.18: "The SINGLE means span 17.431-17.552 s, and the speed-ups 1.746-1.756x (median 1.748x)." |
| SUSTAINED floor (row 3) | 1.595 | §5.9, the round table: "\| 0 \| 18.133 \| 11.367 \| 1.595x \|" — the lowest of the five in-round speed-ups the gate's rule did not reject. The other four are §5.2's "\| 0 \| 17.414 \| 10.320 \| 1.687x \|" and §5.18's 1.746, 1.748 and 1.756x. Sequence 2's round 1 (1.225x) is excluded: its SINGLE arm fired row 1. So row 3 reads "the late window is inside the spread the burst figure itself had across the gate's valid rounds", 8.8% under the 1.748x median |
| DEGRADES floor (rows 4, 5) | 1.20 | §0: "**14.0 s** is 5.03 GB/s, 1.2x the 4.09-4.14 GB/s the single-drive step averages." The gate's lower SHIP line, stated by the gate as a ratio |
| re-shard (row 2) | 30 s | §5.17: "Every arm's `shard_seconds` is 0.003-0.158 s, a cache hit."; §5.1: the build's "Sharding: **231.812 s**". 30 s is far from both |
| arm drift flag (rows 3-5, a label) | 1.080 | §5.9, the arm table: "\| `gate2_r0_striped` \| 0 \| STRIPED \| 16:01:41-16:03:31 \| **11.367** \| 11.036-11.914 \| ..." — 11.914 / 11.036 = 1.080, the largest max/min step ratio in any non-void gate arm the rule did not reject. The other eleven read 1.005-1.060 (computed from the committed `gate*_r*.json`) |
| drive drift flag (rows 4, 5, a label) | 1.10 | §5.21: "every SINGLE arm read C: at 3.68-4.06 GB/s, and every STRIPED arm read 3.15-3.47 GB/s from each drive" — 4.06 / 3.68 = 1.103 and 3.47 / 3.15 = 1.102, the 10-s bin spread of one drive in a sequence with no dip |
| foreign reader (a void, §4) | 1 GB per arm | §5.21: "the largest foreign reader in any arm was `MoUsoCoreWorker.exe` (Windows Update's orchestrator, started during the arm), which read 58.2 MB during `gate4_r0_single`." 1 GB is 17x that, and about 6.5x after scaling the 153-s gate arm to a ~400-s arm |

**What the rule deliberately does, stated now so none of it is read in afterwards:**

1. **The late window is judged against the gate's burst figure, not against this sequence's
   early windows.** Only the first arm starts rested. C: is read in every arm, back to back,
   for the whole sequence of about 40 minutes. D: is read in every STRIPED arm, and the
   order puts round 0's and round 1's STRIPED arms back to back (about 13 minutes). So every
   later arm's early window already sits under sustained load, and `u_early` is not a fresh
   burst. It is recorded per round, so a reader can see whether a lower `u_late` came within
   arms (`u_early` high, `u_late` low) or across them (both low).
2. **Row 1 can fire on a slowdown it was not written to catch.** If C: slows under the load
   that came before a round-1 or round-2 SINGLE arm, that arm's `E` can leave the band. Row 1
   cannot tell that from a disturbed session (sequence 2's round 1 is the precedent), so it
   gives no verdict. The record then says, from the per-drive window rates and the 10-s bins,
   whether C: alone slowed or both drives did.
3. **Medians, not the gate's mean of three.** One slow step cannot move a median of 5 or 10
   (`gate2_r1_single`'s first timed step read 31.581 s, and its mean of three left the band
   while its median, 17.765 s, did not). A slowdown that lasts is what the windows exist to
   see. The band check here is therefore a little more lenient than the gate's.
4. **The two drift flags are labels, not rows.** They name which arm and which drive slowed
   in a DEGRADES verdict. They set no verdict, and under SUSTAINED they are quoted, not
   applied.

**The inputs, as the JSON names them** (fixed with the rule):

- `step_s`, `started_unix`, `layer_loads`, `large_loads`: each entry of `records` in the
  arm JSON's `kind: "step"`, `label: "step_plain"` record.
- `direct_io`, `pinned`, `source_class`, `read_ahead`, `stripe_roots`, `layer_roots`,
  `shard_seconds`: the arm JSON's `meta`.
- The PDH samples: `runs[*].during_samples.pdh` in `sustain_driver.json`, as
  `[unix time, {"read_c": ..., "read_d": ..., ...}]`, bytes per second.
- The foreign readers: `runs[*].after.top_readers[*].read_delta` for every pid other than the
  driver's.
- `python benchmarks/harness/two_drive_sustained.py --summarize` recomputes every input and
  applies the rows from those committed files alone. The comparisons are made on unrounded
  values against the thresholds as written; numbers are quoted at three decimals, and none is
  rounded in a way that moves a row.

**Also recorded, not part of the verdict:** each arm's `E` across rounds (the sequence-level
drift of each layout); the per-step read rate of each drive; 10-s bins of each drive's read
and write rates; per-step losses (the first 21 timed steps of the two arms are the same
optimizer trajectory, so they compare step by step, read against the same-arm spread as the
gate did); every 10 s, the GPU's SM clock, temperature, power and throttle reasons
(`nvidia-smi`); every 1 s, `\Memory\Pages/sec`, committed bytes, `% Processor Time` and
`% Processor Performance`; AC, battery percentage and whether it is charging; the SM clock
at the start and end of each block; `implied_h2d_gb_per_s` and `peak_alloc_gb`.

## 3. What this does NOT measure

- **Hours.** Each arm reads for about six minutes, and the sequence for about forty. C: reads
  nearly without a break for the whole sequence. D: reads for at most about 13 minutes at a
  stretch (round 0's and round 1's STRIPED arms, back to back) and rests while SINGLE arms
  run: about 6.6 minutes before round 0's STRIPED arm, and about 13 before round 2's. A
  training run reads both drives for hours. A SUSTAINED verdict says the speed-up held for
  this long, not longer.
- **Drive temperature.** It is not read: `Get-PhysicalDisk | Get-StorageReliabilityCounter`
  refused without elevation on this box ("Access to a CIM resource was not available to the
  client", checked 2026-10-01), and this probe changes no privilege. A slowdown is placed on a
  drive or on the host only through the per-drive read rates and the GPU and CPU samples,
  never through a temperature. The GPU's temperature is sampled.
- **Why anything slowed.** A DEGRADES verdict names which arm and which drive slowed, not the
  cause (thermal, firmware, host, power state). If sequence 2's slowdown recurs, it is
  recorded, not explained.
- **Ambient conditions.** Room temperature, the fan profile and the laptop's surface are not
  recorded or controlled. The battery's charging state is recorded, not controlled.
- What the gate's §3 already excludes: a real 70B checkpoint, warm (page-cached) reads, more
  than two drives or any other pair or box, batch > 1, seq other than 512, the RAM tier, and
  the setup path that `soup train` goes through. The read-ahead difference between the arms
  (2 against 3, gate §7) is unchanged.

## 4. Box state, per arm

The gate's §4 protocol, run by `two_drive_sustained.py`, which imports the gate driver's
functions rather than copying them. The same stamp is taken right before and right after
every arm: AC and battery, available RAM, commit used and limit, the Python process count,
GPU memory and compute apps, the AC ASPM index (read only), `soup_cli.__file__`, and every
process's cumulative read and write bytes, with the eight processes whose reads grew most
named in the after stamp. While an arm runs, the gate's sampler takes AC and commit every 2 s
and the same PDH counters every 1 s. A third thread runs `nvidia-smi` every 10 s.

What differs from the gate's §4:

- **Files.** The stamps go to `sustain_box_state.log` and every sample to
  `sustain_driver.json`, both in `results/probe-rtx5070/two-drive/`, so the gate's committed
  logs are never appended to. Each arm writes `sustain_r<round>_<arm>.json` and `.log`; the
  build writes `sustain_build_striped.json` and `.log`.
- **A fourth void condition.** Besides battery power at any sample, a suspend (two power
  samples more than 30 s apart) and no `step_plain` point, an arm is void when a process
  other than the driver read at least 1 GB during it (§2's threshold table). A void arm is
  kept under a `_void` name and re-run once in the same position when the pre-arm checks hold
  again. A second void in the same position stops the sequence, and row 2 gives no verdict.
  If the per-process capture fails in either stamp, the check has no input for that arm. That
  is recorded and does not void the arm.
- **Stopping early.** After each round arm the driver applies rows 1 and 2, the two a single
  arm can settle, and stops if either fires, because no later arm can turn a no-verdict
  outcome into a verdict. Nothing else stops the sequence.
- **The `not_run` list** counts a position as run only when a non-void attempt exists, which
  repairs the gap gate §5.7 item 1 recorded.
- **Settle: 300 s** between the end of the build and the first arm (the gate used 60). The
  build writes about 36 GB across both drives over about four minutes (the gate's build wrote
  C: at up to 774.3 MB/s and D: at up to 390.1 MB/s, §5.1). Only the first arm is meant to
  start rested (§2, note 1), and it should not start on drives still busy or warm from
  writing. Five minutes gives whatever the writes set off at
  least as long to subside as the writes took. That is a judgement, not a measurement: the
  drives' temperature cannot be read (§3).

Unchanged from the gate: before every arm, AC on, GPU memory in use at most 1024 MiB, and
commit headroom at least 8 GiB, polled every 30 s for at most 15 minutes, after which the
sequence stops and the arms not run are listed. Rounds and order: three rounds, the first arm
alternating, as in the gate: round 0 SINGLE, STRIPED; round 1 STRIPED, SINGLE; round 2
SINGLE, STRIPED. One process per arm, never two at once. Before the build, the other Claude
sessions on this box are asked by message to avoid CUDA runs, test suites and large copies
on C: or D: until this probe reports done. The laptop stays plugged in with the lid open (a
closed lid sleeps it).

**Estimated wall time**, from the gate's own figures, without voids or pre-arm waits:

| part | estimate | from |
|---|---|---|
| build run | 4.6 min | the gate's build, 275.9 s wall (§5.1) |
| settle | 5.0 min | §4 above |
| 3 SINGLE arms | 3 x 6.6 min | 22 steps x 17.44 s + ~15 s of process start (gate arms' wall time minus their steps: 13-15 s) |
| 3 STRIPED arms | 3 x 6.4 min | 37 steps x 9.97 s + ~15 s |
| cache comparison (after the rounds, not judged) | 2.2 min | the gate's comparison took 132 s (§5.5) |
| **total** | **about 51 min** | each void re-run adds about 6.5 min, each pre-arm wait up to 15 |

**The commands, in order** (Git Bash, from the worktree, with the base interpreter, not a
venv; the driver sets each arm's `PYTHONPATH` to `Soup-sustain/src` itself):

```
cd C:/Users/user/projects/Soup-sustain
PYTHONPATH="C:/Users/user/projects/Soup-sustain/src" python -c "import soup_cli; print(soup_cli.__file__)"
python benchmarks/harness/two_drive_sustained.py --plan build && python benchmarks/harness/two_drive_sustained.py --plan rounds --settle-s 300
PYTHONPATH="C:/Users/user/projects/Soup-sustain/src" python benchmarks/harness/compare_shard_caches.py --single C:/Users/user/.soup/layer-stream/D___synth__llama-70b-shape-v32k-f4 --striped C:/Users/user/.soup/layer-stream-striped/D___synth__llama-70b-shape-v32k-f4 --out benchmarks/results/probe-rtx5070/two-drive/sustain_cache_compare.json
python benchmarks/harness/two_drive_sustained.py --summarize
```

The `python -c` line must print a path under `Soup-sustain\src`. The line after it runs the
build and, only if the build completes, the rounds after the 300 s settle; at the end the
driver prints `--summarize`'s table itself. The comparison runs after the rounds and never between them,
because it reads both caches in full through the page cache. The last line reprints §2's
inputs and rows from the committed files.

## 5. Results

One sequence ran on 2026-10-01, from worktree HEAD `119991f8`. The driver's `worktree_state`
found `src/` and `benchmarks/harness` clean at both invocations. The sequence built the striped
cache, ran three arms, and stopped inside the fourth. Every number below is quoted from the
committed `sustain_*` files in `results/probe-rtx5070/two-drive/`, at three decimals unless a
column says otherwise. Times are local (UTC+5).

### 5.1 How it ran, and how it stopped

- **Pre-flight, 20:08:54-20:09:16.** `soup_cli.__file__` resolved to
  `Soup-sustain\src\soup_cli\__init__.py`. A `--dry-run` of the build plan, written into a
  scratch folder (not `results/`), found the pre-arm check met: AC on, charging at 66%, 26.40 GiB
  of commit headroom, GPU at 0 MiB with no compute apps. The other `python.exe` processes were a
  merge-polling loop (gone by 20:09:41) and two editor language servers.
- **The build.** `--plan build` was invoked at 20:09:26 and completed (§5.2). `--plan rounds
  --settle-s 300` was invoked at 20:14:23 and settled until 20:19:23.
- **Three arms.** Round 0's SINGLE and STRIPED arms and round 1's STRIPED arm ran back to back,
  20:19:26-20:38:31, with no void and no pre-arm wait (§5.3).
- **Round 1's SINGLE arm.** It started at 20:38:36. Its log ends at the last line an arm prints
  before its timed block (`nf4 linears 560 patched for the ablation`). Its JSON holds only `meta`
  (`started` 20:38:52) and an empty `records`, which is what an arm's JSON holds until its last
  step.
- **20:49:08-20:49:12, from the System log (Kernel-Power).**
  - Event 566 at 20:49:08: session 143 to 145, `PowerStateAc=true`.
  - Event 105 at 20:49:10: "Power source change", `AcOnline=false`.
  - Event 42 at 20:49:12: "The system is entering sleep", `TargetState=5` and
    `EffectiveState=5` (hibernate), `Reason=0`.
  - The box resumed at 22:57:54-22:57:58. Kernel-General 1 records the system time moving from
    20:49:16 to 22:57:54; Kernel-Boot 18, 25, 27, 30 and 32 record a resume through the boot
    manager; Power-Troubleshooter 1 records the return from a low-power state.
  - At 22:58 the box read AC 0, battery 93%.
- **The driver did not survive the resume.**
  - The driver ran inside a background shell of the agent running this probe. That shell had a
    2-hour wall-clock limit counted from 20:09:26, which includes the 2 h 9 min of hibernation.
    The limit expired on resume, and the shell and its processes were stopped.
  - At 22:58 no `python.exe` of the driver or of the arm existed, and the GPU held 0 MiB.
  - So the driver never took round 1 SINGLE's after stamp, never applied §4's void rule to it,
    and never saved its samples. `sustain_driver.json` has no entry for `sustain_r1_single`, and
    the rounds invocation has no `ended`, `outcome` or `not_run`.
  - That arm's 1-s PDH, 2-s power and 10-s GPU samples were held in the driver's memory and are
    lost.
- **Round 1's SINGLE arm is void under §4, on all three of its conditions.** AC read battery
  (20:49:10), the box was suspended (2 h 9 min), and the arm wrote no `step_plain` point. The
  driver could not rename its files, so its JSON and log were renamed by hand to §4's name for a
  void arm, `sustain_r1_single_void.json` / `.log`. Their content is as the arm left it.
- **Not re-run.** §4 re-runs a void arm in its position once the pre-arm checks hold again. That
  did not happen, for four reasons:
  - the driver that would have done it was gone;
  - the driver cannot resume a sequence part way: its rounds plan always starts at round 0;
  - at 22:58 the box was on battery, which fails the pre-arm check;
  - the peer sessions' hold on CUDA, test suites and heavy disk had ended at 21:20.

  No new arm was started, on the controller's instruction. So round 1 lacks its SINGLE arm, and
  `sustain_r2_single` and `sustain_r2_striped` never ran.
- **After the stop.** The cache comparison ran on battery, with no arm running (§5.6).
  `--summarize` was then run by hand, because the driver prints it only at the end of a rounds
  invocation and it had been stopped. Its output is `sustain_summarize.log`.

### 5.2 The build run (not a round)

`sustain_build_striped` ran from 20:09:28 to 20:14:21, 292.6 s wall.

- **Sharding: 237.617 s** (`meta.shard_seconds`). The log first prints the miss §1 predicted:
  `Re-sharding layer cache: stripe marker
  D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80\stripe.json is missing or
  unreadable.` Model build took 5.800 s.
- **Placement.** 80 `layer_roots`, exactly `i mod 2` (40 per root), and `stripe_roots
  ['D:\\soup-stripe']`. Both roots were opened `open_direct`. `direct_io True`, `pinned True`,
  `read_ahead 3`, `AsyncDiskSource`, staging 2398 MB pinned.
- **The new D: folder.** `D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80` holds
  41 entries: 40 odd layers and `stripe.json`. The marker reads `primary_cache`
  `c:\users\user\.soup\layer-stream-striped\d___synth__llama-70b-shape-v32k-f4`, `position` 1,
  `n_roots` 2. The old folder `D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4` was not touched.
- **Its one untimed step read 29.296 s**, with 158 + 3 loads. The gate's build step read 21.538 s.
  This step is not a round and is not judged.
- **During the build**, PDH saw C: written at up to 872.8 MB/s and D: at up to 394.3 MB/s. Commit
  headroom fell to 8.977 GiB at its lowest sample (26.28 GiB at the before stamp, 26.61 GiB at the
  after stamp).

### 5.3 The timed arms, in the order they ran

| run | round | arm | ran (local) | `E` (early median) | `L` (late median) | `d` = L/E | `step_s_mean` | min-max | tok/s | implied H2D GB/s | `direct_io` / `pinned` | peak alloc GB | SM MHz start->end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sustain_r0_single` | 0 | SINGLE | 20:19:26-20:26:19 | **17.519** | **17.398** | 0.993 | 17.405 | 16.912-17.926 | 29.416 | 4.044 | True / True | 4.378 | 180->1297 |
| `sustain_r0_striped` | 0 | STRIPED | 20:26:26-20:32:10 | **8.376** | **8.426** | 1.006 | 8.665 | 8.173-13.470 | 59.086 | 8.122 | True / True | 4.378 | 1417->2370 |
| `sustain_r1_striped` | 1 | STRIPED | 20:32:15-20:38:31 | **9.295** | **8.983** | 0.966 | 9.460 | 8.221-13.546 | 54.125 | 7.440 | True / True | 4.378 | 1410->2482 |
| `sustain_r1_single_void` | 1 | SINGLE, **void** | 20:38:36 to the hibernation at 20:49:12; stopped at the resume (§5.1) | none | none | none | none | none | none | none | True / True (`meta`) | none | none |
| `sustain_r2_single` | 2 | SINGLE | **not run** | | | | | | | | | | |
| `sustain_r2_striped` | 2 | STRIPED | **not run** | | | | | | | | | | |

**What the three completed arms had in common:**

- Every timed step moved 70,378,258,732 bytes (157 + 2 loads). Peak reserved was 4.798 GB.
- `shard_seconds` was 0.009, 0.017 and 0.014 s, a cache hit each time. The source was
  `AsyncDiskSource` throughout.
- The STRIPED arms kept `layer_roots` i mod 2 over `stripe_roots ['D:\\soup-stripe']` at
  read-ahead 3. The SINGLE arm had no stripe roots, at read-ahead 2.
- `--summarize` finds no row-2 problem in any of the three.

**The void arm.** Its `meta` reads `direct_io True`, `pinned True`, `AsyncDiskSource`,
read-ahead 2, no stripe roots, `shard_seconds` 0.007 s. Its `started` stamp is 20:38:52. Compare
round 0's SINGLE arm: its first timed step began 23 s after `started`, and its 21 timed steps took
365.5 s. In the void arm, at 20:49:08 (616 s after `started`) the JSON still held no step record.
So its 22 process steps (one warm-up and 21 timed) had either averaged more than 28.0 s each,
against 17.405 s in round 0, or the process had stalled. Which one is not on record, because its
samples were lost (§5.1).

The per-round form the rule reads:

| round | `E` SINGLE | `E` STRIPED | `u_early` | `L` SINGLE | `L` STRIPED | `u_late` |
|---|---|---|---|---|---|---|
| 0 | 17.519 | 8.376 | 2.092 | 17.398 | 8.426 | 2.065 |
| 1 | void, not re-run | 9.295 | none | void, not re-run | 8.983 | none |
| 2 | not run | not run | none | not run | not run | none |

### 5.4 Every timed step, and the per-drive window rates

`step_s` of every timed step, in order. The early window is steps 1-5 and the late window is the
last 10 (§1). Each list below is early | middle | late; bold marks the slow steps of §5.7.

- `sustain_r0_single` (21): 17.519, 17.632, 17.320, 17.375, 17.713 | 17.394, 17.346, 17.227,
  17.279, 17.470, 17.162 | 17.675, 17.314, 16.912, 17.534, 17.393, 17.199, 17.298, 17.403,
  17.926, 17.418
- `sustain_r0_striped` (36): 8.293, 8.672, 8.298, 8.376, 8.482 | 8.278, 8.173, 8.193, 8.295,
  8.360, 8.343, 8.632, 8.256, 8.338, 8.248, 8.447, 8.292, 8.315, 8.275, 8.426, 8.335, 8.731,
  8.389, 8.381, 8.304, 8.351 | 8.498, 8.282, 8.354, 8.189, 8.309, 8.723, 8.337, **10.886,
  13.470, 11.421**
- `sustain_r1_striped` (36): **9.791, 10.292**, 9.295, 8.845, 8.421 | 8.406, 8.511, 8.844, 8.278,
  8.290, 8.434, 8.657, 8.359, 8.318, 8.261, 8.358, 8.334, 8.314, 9.521, **13.489, 12.874**, 8.365,
  8.221, 8.359, 8.254, **13.463** | **13.471**, 9.267, 8.699, 8.357, 8.398, 8.688, 9.589,
  **13.546, 13.536**, 8.442

Each drive's mean read rate over the two windows (§2), and its early/late ratio:

| run | C: early GB/s | C: late GB/s | D: early GB/s | D: late GB/s | C: early/late | D: early/late |
|---|---|---|---|---|---|---|
| `sustain_r0_single` | 3.933 | 3.958 | 0.000 | 0.000 | 0.994 | none (not read) |
| `sustain_r0_striped` | 4.029 | 3.591 | 4.040 | 3.593 | 1.122 | 1.124 |
| `sustain_r1_striped` | 3.650 | 3.331 | 3.655 | 3.331 | 1.096 | 1.097 |

### 5.5 Losses

The first 21 timed steps of the two arms follow the same optimizer trajectory (§2). The losses
of every step are in the committed JSONs.

- **Cross-arm, round 0** (SINGLE minus STRIPED, steps 1-21): the absolute difference has median
  0.006932 and maximum 0.038596 (step 3). On the first timed step it is +0.006480.
- **Same arm, across rounds** (STRIPED round 0 minus round 1): over steps 1-21, median 0.009902
  and maximum 0.023760 (step 20); over all 36 steps, median 0.005075. On the first timed step it
  is +0.004231.
- **The largest cross-arm difference exceeds the largest same-arm one** (0.038596 against
  0.023760). In the gate it was the other way round (§5.20 there: 0.017111 inside 0.017391).
  This record has only one cross-arm pair and one same-arm pair. It is recorded, not judged.
- The data half of "same bytes" is settled in §5.6.

### 5.6 Same bytes: the two caches compared tensor by tensor

The comparison ran after the stop, 22:59:59-23:02:23, on battery and with no arm running. The
command was §4's; the script needs no torch and no GPU. Result (`sustain_cache_compare.json`):
**all equal**.

- 83 files compared: 80 layers, `extras` and the two `large_*` files.
- 43 of them are on C: and 40 on D:, and the D: files were read from the new
  `-cba3360a9c80` folder.
- 2,405 tensors and 36.388 GB, in 144 s (the gate took 132 s).
- No safetensors file exists in striped root 0 that the single-root cache lacks. `index.json` is
  not compared.

### 5.7 Box state, the during-arm samples and the per-process capture

**Power, suspend and commit.**

- **AC** at every 2-s sample of the build and of the three completed arms. The battery was
  **charging** at every stamp: battery flag 8 at the build's before stamp and 9 at every other,
  rising from 66% before the build to 95% after round 1's STRIPED arm.
- **No suspend in those arms.** The largest gap between power samples was 2.018-2.030 s.
- **Commit headroom** was at least 12.083 GiB at every sample of a round arm (8.977 GiB during
  the build), and 26.85-26.97 GiB at the round arms' before stamps.

**The stamps.** GPU at 0 MiB with no compute apps, ASPM AC index 2, and `soup_cli.__file__` under
`Soup-sustain\src`, at every stamp. The Python process count read 4 at every stamp except two:
5 at the build's before stamp and at round 0 SINGLE's after stamp. Which process the fifth was is
not captured.

**PDH over each whole arm**, process start and teardown included:

| run | C: read GB/s mean / max | D: read GB/s mean / max | C: write MB/s mean / max | D: write MB/s max | pages/s mean |
|---|---|---|---|---|---|
| `sustain_r0_single` | 3.674 / 4.614 | 0.000 / 0.000 | 0.53 / 23.40 | 0.00 | 685 |
| `sustain_r0_striped` | 3.676 / 4.845 | 3.670 / 4.845 | 0.92 / 27.07 | 0.05 | 355 |
| `sustain_r1_striped` | 3.355 / 4.836 | 3.347 / 4.780 | 3.38 / 45.49 | 0.09 | 825 |

**10-s bins**, aligned to each arm's first timed step, full bins only:

- `sustain_r0_single`: C: read 3.714-4.080 GB/s in all 36 bins (median 3.976). No dip.
- `sustain_r0_striped`: C: read 3.89-4.17 GB/s and D: 3.93-4.18 GB/s in the first 28 bins. The
  last three bins (20:31:35-20:32:05) read 2.54, 2.57 and 2.66 GB/s on C:, and 2.48, 2.58 and
  2.70 GB/s on D:.
- `sustain_r1_striped`: 3.32-3.60 GB/s per drive in its first three bins. Three dips, on both
  drives at once and each reaching 2.49 GB/s on both, are listed here with the bin after each:
  - 20:35:29-20:36:09: C: 2.97, 2.51, 2.52, 3.96 and D: 2.93, 2.49, 2.55, 3.98;
  - 20:36:29-20:37:09: C: 3.41, 2.49, 2.58, 3.32 and D: 3.37, 2.49, 2.57, 3.36;
  - 20:37:49-20:38:29: C: 2.84, 2.49, 2.55, 3.62 and D: 2.80, 2.49, 2.55, 3.66.

  Every other bin read 3.84-4.16 GB/s per drive.
- No bin of any arm fell to sequence 2's level (2.23 GB/s or less, gate §5.12). The lowest was
  2.48 GB/s (D:, `sustain_r0_striped`).

**The slow steps, step by step.** Each entry gives the step numbers, when they started, `step_s`,
and C: / D: GB/s over each step:

- `sustain_r0_striped` steps 34-36, from 20:31:31: 10.886, 13.470, 11.421 s. C: 3.10, 2.55, 2.90
  and D: 3.05, 2.53, 2.95.
- `sustain_r1_striped` steps 19-21, from 20:35:25: 9.521, 13.489, 12.874 s. C: 3.58, 2.53, 2.63
  and D: 3.56, 2.53, 2.65.
- `sustain_r1_striped` steps 26-28, from 20:36:34: 13.463, 13.471, 9.267 s. C: 2.55, 2.52, 3.59
  and D: 2.54, 2.52, 3.62.
- `sustain_r1_striped` steps 33-35, from 20:37:44: 9.589, 13.546, 13.536 s. C: 3.52, 2.53, 2.50
  and D: 3.49, 2.53, 2.52.
- `sustain_r1_striped` steps 1-4, from 20:32:49: 9.791, 10.292, 9.295, 8.845 s. C: 3.43, 3.32,
  3.60, 3.89 and D: 3.47, 3.32, 3.54, 3.92. Unlike the other slow steps, these came with high
  `% Processor Time`: 56.7-69.5% in steps 1-4, against 22.4-33.7% in steps 5 and 9-11 and
  4.4-12.4% in the rest of the arm. C: was written at 8.7-16.2 MB/s in those four steps.

What those entries share, as measured:

- Every step near 13.5 s (13.463-13.546 s) read 2.50-2.55 GB/s from each drive. That is
  70.38 GB in 13.46-13.55 s, 5.20-5.23 GB/s for the two drives together.
- Every STRIPED step not listed above read 3.78-4.21 GB/s per drive.
- The first slow stretch began 4 min 36 s after round 0 STRIPED's first timed step. In round 1
  the stretches began 69 and 70 s apart (20:35:25, 20:36:34, 20:37:44).
- In those stretches (not counting round 1's steps 1-4), `% Processor Time` read 4.4-7.8% and C:
  writes 0.2-3.9 MB/s, in line with the neighbouring steps.
- The SINGLE arm, which read C: alone for 6 min 5 s before round 0's STRIPED arm, had no slow
  step: 16.912-17.926 s, and 3.83-4.06 GB/s per step.
- **Stated as reasoning, not measured:** in a STRIPED step the reader takes layers from the two
  drives alternately, so one slow drive would hold back both drives' rates. These counters
  therefore cannot say which drive, if either, slowed first.

**GPU (10-s `nvidia-smi`).**

- `sustain_r0_single`: SM 180-1702 MHz, 47-71 °C, 3.9-91.3 W.
- `sustain_r0_striped`: SM 255-2542 MHz, 55-84 °C, up to 124.3 W.
- `sustain_r1_striped`: SM 270-2527 MHz, 56-85 °C, up to 121.3 W.
- The throttle-reason field read `0x0000000000000004` or `0x0000000000000400` in every sample of
  every arm, the SINGLE arm included.
- The 10 samples taken during the steps of 12.8 s or more read SM 922-1875 MHz, 52.2-87.1 W and
  67-75 °C. Recorded, not interpreted.

**Top readers, from the after stamps.** No capture failed. No process other than the driver read
1 GB or more in any arm, so the foreign-reader void never fired. The largest reader per arm:

- build: `chrome.exe` (17648), 834.7 MB, below the threshold;
- `sustain_r0_single`: `svchost.exe` (16716), 150.9 MB, then `msedgewebview2.exe`, 125.2 MB;
- `sustain_r0_striped`: `codex.exe`, 133.1 MB;
- `sustain_r1_striped`: `codex.exe`, 76.9 MB. That arm's list also holds three processes new
  during it: `MoUsoCoreWorker.exe` (59.9 MB), `svchost.exe` (22280, 46.4 MB) and `audiodg.exe`.

The full lists are in `sustain_box_state.log`. The capture names survivors only: a process that
started and exited inside an arm is in neither stamp.

**System log, 20:09-20:49.** Before the 20:49:08 sequence (§5.1), the System log holds only:

- Kernel-Power 566 at 20:22:34 (session 142 to 143), during round 0's SINGLE arm.
- Windows Update events during round 1's STRIPED steps 26-31:
  - downloads started at 20:36:39, inside step 26, which is in the second slow stretch;
  - two Store app updates (OpenAI Codex, WhatsApp) each started and failed with 0x80073D02, at
    20:36:41-20:36:42 and 20:37:30-20:37:31.

  The other two slow stretches have no event.
- Kernel-General 16 at 20:37:27 (a registry hive's access history).
- No disk, storage, NTFS, thermal or power-source event before 20:49:10.

### 5.8 Anomalies, as measured

1. **STRIPED read faster than in the gate.**
   - This sequence: early medians 8.376 and 9.295 s, means 8.665 and 9.460 s, against the gate's
     sequence-4 means of 9.929-10.050 s. Per drive it read about 3.8-4.2 GB/s in the steps that
     did not slow, against the gate's 3.15-3.47 GB/s bins.
   - The SINGLE arm read as in the gate: 17.405 s, against 17.431-17.552 s. So round 0's ratios
     (`u_early` 2.092, `u_late` 2.065) sit above the gate's 1.746-1.756x.
   - The code under test is not the gate's. Between the gate's `f798c46e` and `2c6ed087`,
     `git diff --stat` shows changes to `utils/async_disk_source.py`, `utils/layer_shard.py`,
     `utils/layer_stream.py`, `utils/layer_stream_runtime.py`, `utils/stripe_roots.py` and
     `trainer/stream_setup.py`. The striped cache was also rebuilt into a new D: folder.
     `benchmarks/harness/stream_probe.py` is unchanged between the two commits; `6ee60ce0` then
     added only the per-step `started_unix`.
   - Which of these, if any, explains the faster STRIPED steps is not established here.
2. **Recurring slow steps in both STRIPED arms** (§5.7). They began 4 min 36 s into round 0's
   STRIPED steps and recurred about every 70 s in round 1, at 13.46-13.55 s a step with both
   drives at 2.50-2.55 GB/s. The cause is not established. The drives' temperature cannot be read
   on this box (§3).
3. **Round 1 STRIPED's first four steps were slow, with high CPU** (56.7-69.5%) and C: writes
   (§5.7). The new processes in that arm's capture were `MoUsoCoreWorker.exe`, `svchost.exe`
   (22280) and `audiodg.exe`.
4. **Round 1's SINGLE arm never finished.** At 20:49:08 it was 616 s past its `started` stamp
   with no step record (§5.3). Either it averaged more than 28.0 s per process step, or it had
   stalled. It is void, and its samples are lost.
5. **The box lost AC and hibernated** at 20:49:10-20:49:12 (Kernel-Power 105 `AcOnline=false`,
   then 42 `Reason=0`), against §4's "The laptop stays plugged in with the lid open". The driver
   was then stopped by the agent's 2-hour shell limit, which counted the hibernation (§5.1). That
   second failure is in this probe's method, not in the box: the sequence could not continue on
   resume even if AC had returned.
6. **Activity by another agent of the same session, disclosed by the controller.** That agent was
   writing a different rule in another worktree (`Soup-r2`). It ran two driver dry-runs and
   sampler tests between about 20:13 and 20:17, and one `import bitsandbytes` at about 20:31
   (possibly a brief CUDA context). All of it was read-only, with no GPU work and no large disk
   reads.
   - 20:13-20:17 falls in the build's last minute (the build ended at 20:14:21) and in the
     settle (to 20:19:23), not in a timed arm.
   - About 20:31 falls in round 0's STRIPED arm. That arm's step 34 began at 20:31:31, and its
     last three steps (20:31:31-20:32:07) are its only slow ones.
   - What that arm's records show: its stamps read 4 Python processes before and after. Its
     per-process capture names nobody above 133.1 MB, and it cannot see a process that started
     and exited inside the arm. Its CPU samples in those steps read 7.0-7.7%, like its other
     steps.
   - The same slow-step shape recurs three times in round 1's STRIPED arm, for which no activity
     was disclosed.
   - Nothing was voided for it. §4 voids an arm on battery power, a suspend, no `step_plain`
     point or a foreign reader of 1 GB or more, and none of these fired in that arm.
7. **The build's untimed step read 29.296 s**, against the gate build's 21.538 s. It is not
   judged.
8. **The battery charged** from 66% to 95% over the build and the three arms, as it did in the
   gate's sequence 4 (gate §7).

## 6. Verdict

**NO VERDICT (row 2).** The rule reads three complete rounds, and only round 0 is complete.

| row | inputs | result |
|---|---|---|
| 1: any SINGLE arm's `E` outside 15.0-21.5 s | round 0: 17.519 s. Rounds 1 and 2 have no valid SINGLE arm | does not fire |
| 2: an arm off the path under test, or the protocol stopped | The three completed arms show no problem: `direct_io` / `pinned` True, `AsyncDiskSource`, read-ahead 2 / 3, i mod 2 over `D:\soup-stripe` / no stripe roots, 157 + 2 loads in every timed step, `shard_seconds` 0.009-0.017 s. The protocol stopped: round 1's SINGLE arm is void (AC lost, suspended, no `step_plain` point) and was not re-run, and round 2 did not run. `--summarize`: "no valid arm: ['r1 SINGLE', 'r2 SINGLE', 'r2 STRIPED']; protocol: not finished" | **fires: NO VERDICT** |
| 3: `u_late` >= 1.595 in every round | none | not reached |
| 4: `u_late` >= 1.20 in every round | none | not reached |
| 5: anything else | none | not reached |

**Why row 2 applies to this stop.** Row 2 names two ways the protocol can stop: a position void
twice, or an expired pre-arm wait. This stop is neither. It is a void that was never re-run,
because the driver was gone. Its effect is the one the row names, "some round lacks an arm", and
rows 3-5 read every round. `--summarize` applies it the same way: any outcome other than
`complete` fires row 2.

**What round 0 alone reads. This is NOT a verdict and cannot stand in for one.**

- `u_early(0)` = 17.519 / 8.376 = 2.092, and `u_late(0)` = 17.398 / 8.426 = 2.065. Taken alone,
  that is above row 3's 1.595.
- `d` is 0.993 for SINGLE and 1.006 for STRIPED; neither is above the 1.080 label.
- Round 0 STRIPED's window rates fell from 4.029 to 3.591 GB/s on C: and from 4.040 to
  3.593 GB/s on D:. Those early/late ratios, 1.122 and 1.124, are above the 1.10 label, which
  names drives only in a DEGRADES verdict.
- Round 0 STRIPED's late median includes its three slow steps (10.886, 13.470, 11.421 s), three
  of the ten.
- Round 1's STRIPED arm (`E` 9.295 s, `L` 8.983 s) has no SINGLE partner, so it gives no ratio.
- The arm that would have tested the rule's second note — a SINGLE arm whose early window starts
  after 12 minutes of striped reading — is the one that did not finish.

A verdict needs a complete new sequence: a new `--run-prefix`, on AC with the lid open, with the
driver in a process that no tool time limit can stop part way. Whether and when to run it is the
owner's call. The cache this sequence built is reusable as is (§5.2, §5.6).
