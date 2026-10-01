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

**Status: rule committed 2026-10-01, BEFORE the run. Results pending.** No arm has run and
the striped cache §1 needs has not been rebuilt. The only executions so far are the driver's
`--dry-run` into a scratch folder and a synthetic check of its rule code on fake arm files.
Neither wrote anything under `results/`.

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

Pending.

## 6. Verdict

Pending.
