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

**Status: NO VERDICT (§2 row 2), in both sequences.** The rule (§2) was committed in
`119991f8`, before either run.

**Sequence 2 (2026-10-02, prefix `sustain2`)** completed rounds 0 and 1 and round 2's SINGLE
arm (§5.9, §6). Round 2's STRIPED arm was void twice: Windows Search's indexer read 1.27 GB and
then 1.94 GB during it, over §4's 1 GB foreign-reader bar, and the driver stopped.

- Rounds 0 and 1 alone read `u_late` 1.279 and 1.412. That is **not a verdict**.
- The 2.5 GB/s per-drive plateau set in about two minutes into round 0's STRIPED arm and then held
  to the arm's end.
- Round 1's STRIPED arm sat on the plateau from its third step.
- C: alone dipped about once a minute in the later SINGLE arms (§5.14, §5.15).

**Attempt 1 (2026-10-01, prefix `sustain`)** stopped inside round 1's SINGLE arm, when the laptop
lost AC and hibernated. Only its round 0 is complete: `u_late` 2.065, not a verdict. Its STRIPED
arms had recurring 25-30 s stretches of 13.5-s steps, with both drives at about 2.5 GB/s
(§5.1-§5.8).

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

Two sequences have run. **Attempt 1** (2026-10-01, prefix `sustain`) is §5.1-§5.8, as first
committed in `f46a39de`. **Sequence 2** (2026-10-02, prefix `sustain2`) is §5.9-§5.15.

Attempt 1 ran on 2026-10-01, from worktree HEAD `119991f8`. The driver's `worktree_state`
found `src/` and `benchmarks/harness` clean at both invocations. The sequence built the striped
cache, ran three arms, and stopped inside the fourth. Every number below is quoted from the
committed `sustain_*` (attempt 1) and `sustain2_*` (sequence 2) files in `results/probe-rtx5070/two-drive/`, at three decimals unless a
column says otherwise. Times are local (UTC+5).

### 5.1 Attempt 1: how it ran, and how it stopped

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

### 5.2 Attempt 1: the build run (not a round)

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

### 5.3 Attempt 1: the timed arms, in the order they ran

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

### 5.4 Attempt 1: every timed step, and the per-drive window rates

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

### 5.5 Attempt 1: losses

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

### 5.6 Attempt 1: same bytes: the two caches compared tensor by tensor

The comparison ran after the stop, 22:59:59-23:02:23, on battery and with no arm running. The
command was §4's; the script needs no torch and no GPU. Result (`sustain_cache_compare.json`):
**all equal**.

- 83 files compared: 80 layers, `extras` and the two `large_*` files.
- 43 of them are on C: and 40 on D:, and the D: files were read from the new
  `-cba3360a9c80` folder.
- 2,405 tensors and 36.388 GB, in 144 s (the gate took 132 s).
- No safetensors file exists in striped root 0 that the single-root cache lacks. `index.json` is
  not compared.

### 5.7 Attempt 1: box state, the during-arm samples and the per-process capture

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

### 5.8 Attempt 1: anomalies, as measured

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

### 5.9 Sequence 2 (2026-10-02, prefix `sustain2`): how it ran, and how it stopped

The second complete sequence was started under a new `--run-prefix`, so attempt 1's arm files are
untouched. As the driver is written, its stamps and samples are appended to the same
`sustain_box_state.log` and `sustain_driver.json`; attempt 1's entries in them are unchanged, and
`collect` reads only the entries of the prefix it is given. The worktree was at HEAD `3a324993` for both
invocations, and the driver's
`worktree_state` found `src/` and `benchmarks/harness` clean at both invocations. Times are local
(UTC+5).

- **What came just before.** The per-drive throttle probe
  ([`probe-rtx5070-drive-throttle.md`](probe-rtx5070-drive-throttle.md)) read C: alone, unbuffered,
  21:12:40-21:22:48 (two void attempts of its first arm), and did not read D:. So this sequence did
  not start on a C: rested as long as attempt 1's.
- **Pre-flight.** `soup_cli.__file__` resolved to `Soup-sustain\src\soup_cli\__init__.py`. The
  striped cache built by attempt 1 was in place (`D:\soup-stripe\D___synth__llama-70b-shape-v32k-f4-cba3360a9c80`).
- **The build plan, run as §4's command line has it.** `--plan build --run-prefix sustain2` was
  invoked at 21:25:14. `sustain2_build_striped` ran 21:25:17-21:25:57 (40.6 s wall) and was a cache
  hit: `shard_seconds` 0.029 s, 80 `layer_roots` i mod 2 over `['D:\\soup-stripe']`, `direct_io`
  and `pinned` True, `AsyncDiskSource`, read-ahead 3. Its one untimed step read 19.692 s. It wrote
  nothing to either cache. Not a round, not void, and not judged.
- **The rounds.** `--plan rounds --settle-s 300 --run-prefix sustain2` was invoked at 21:26:00 and
  settled until about 21:31:00. The two invocations ran as one background job with a 2-hour limit.
- **Five positions ran without a void or a pre-arm wait**, back to back, 21:31:05-22:09:14.
- **Round 2's STRIPED arm was void twice**, both times on the foreign-reader condition:
  - attempt 1, 22:09:17-22:16:36: `foreign reader: SearchIndexer.exe(26528) 1.27 GB`;
  - attempt 2, 22:16:42-22:27:52: `foreign reader: SearchIndexer.exe(26528) 1.94 GB`.

  The driver renamed their files `sustain2_r2_striped_void.*` and `sustain2_r2_striped_void2.*`,
  recorded `stopped: sustain2_r2_striped void twice; no verdict` with `not_run
  ['sustain2_r2_striped']`, printed `--summarize` and exited with code 4 at 22:27:55.
- **After the stop.** The cache comparison ran 22:28:24-22:30:20, with no arm running (§5.13), and
  `--summarize --run-prefix sustain2` was run by hand (`sustain2_summarize.log`).
- **No suspend, no AC loss.** The laptop stayed on AC with the lid open throughout (§5.14).

### 5.10 Sequence 2: the timed arms, in the order they ran

| run | round | arm | ran (local) | `E` (early median) | `L` (late median) | `d` = L/E | `step_s_mean` | min-max | tok/s | implied H2D GB/s | `direct_io` / `pinned` | peak alloc GB | SM MHz start->end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sustain2_r0_single` | 0 | SINGLE | 21:31:05-21:37:49 | **17.388** | **17.298** | 0.995 | 17.354 | 17.072-17.760 | 29.504 | 4.056 | True / True | 4.378 | 1417->1447 |
| `sustain2_r0_striped` | 0 | STRIPED | 21:37:54-21:45:13 | **8.353** | **13.525** | 1.619 | 11.354 | 8.219-13.941 | 45.093 | 6.198 | True / True | 4.378 | 1417->1305 |
| `sustain2_r1_striped` | 1 | STRIPED | 21:45:19-21:54:27 | **13.580** | **13.590** | 1.001 | 14.382 | 8.311-28.068 | 35.599 | 4.893 | True / True | 4.378 | 1417->2265 |
| `sustain2_r1_single` | 1 | SINGLE | 21:54:32-22:01:43 | **17.303** | **19.185** | 1.109 | 18.708 | 17.185-22.301 | 27.368 | 3.762 | True / True | 4.378 | 1792->787 |
| `sustain2_r2_single` | 2 | SINGLE | 22:01:47-22:09:14 | **17.711** | **19.779** | 1.117 | 19.490 | 17.454-25.097 | 26.270 | 3.611 | True / True | 4.378 | 1815->1200 |
| `sustain2_r2_striped_void` | 2 | STRIPED, **void** | 22:09:17-22:16:36 | (8.544) | (13.520) | (1.582) | 11.299 | 8.230-13.763 | 45.313 | 6.229 | True / True | 4.378 | 1417->2182 |
| `sustain2_r2_striped_void2` | 2 | STRIPED, **void** | 22:16:42-22:27:52 | (17.454) | (17.544) | (1.005) | 17.508 | 17.281-18.076 | 29.244 | 4.020 | True / True | 4.378 | 1432->1702 |

The void attempts' numbers are in brackets: they are recorded, and no row reads them.

**What every arm had in common, the two void attempts included:**

- Every timed step moved 70,378,258,732 bytes (157 + 2 loads). Peak reserved was 4.798 GB.
- `shard_seconds` was 0.007-0.022 s, a cache hit each time. The source was `AsyncDiskSource`
  throughout.
- The STRIPED arms kept `layer_roots` i mod 2 over `['D:\\soup-stripe']` at read-ahead 3. The
  SINGLE arms had no stripe roots, at read-ahead 2.
- `--summarize` finds no row-2 problem in any of the five valid arms. Its only row-2 inputs are
  the missing round-2 STRIPED arm and the stopped protocol.

The per-round form the rule reads:

| round | `E` SINGLE | `E` STRIPED | `u_early` | `L` SINGLE | `L` STRIPED | `u_late` |
|---|---|---|---|---|---|---|
| 0 | 17.388 | 8.353 | 2.082 | 17.298 | 13.525 | 1.279 |
| 1 | 17.303 | 13.580 | 1.274 | 19.185 | 13.590 | 1.412 |
| 2 | 17.711 | void twice | none | 19.779 | void twice | none |

### 5.11 Sequence 2: every timed step, and the per-drive window rates

`step_s` of every timed step, in order: early | middle | late (§1). Bold marks the steps of
12.8 s or more in a STRIPED arm and of 19.0 s or more in a SINGLE arm.

- `sustain2_r0_single` (21): 17.383, 17.468, 17.388, 17.088, 17.568 | 17.302, 17.125, 17.434,
  17.459, 17.760, 17.250 | 17.633, 17.455, 17.516, 17.124, 17.198, 17.261, 17.368, 17.238,
  17.072, 17.335
- `sustain2_r0_striped` (36): 8.259, 8.850, 8.346, 8.353, 8.424 | 8.473, 8.835, 8.472, 8.306,
  8.219, 8.269, 8.571, 8.317, 8.461, 8.299, 11.303, **13.537, 13.582, 13.488, 13.485, 13.357,
  13.941, 13.538, 13.506, 13.459, 13.491** | **13.565, 13.490, 13.531, 13.520, 13.505, 13.491,
  13.823, 13.642, 13.537, 13.510**
- `sustain2_r1_striped` (36): 8.311, 8.630, **13.580, 13.614, 13.675** | **13.591, 13.574,
  13.538, 13.533, 13.578, 13.609, 13.795, 13.570, 13.590, 13.530, 13.496, 13.548, 13.571, 13.566,
  13.599, 22.993, 24.050, 13.537, 13.487, 13.563, 13.567** | **13.510, 13.579, 13.492, 13.606,
  13.595, 13.815, 13.572, 13.584, 18.250, 28.068**
- `sustain2_r1_single` (21): 17.218, 17.706, 17.270, 17.303, 17.371 | 17.259, **19.508, 19.169**,
  17.387, **20.824**, 18.110 | 17.487, **22.149**, 17.519, 18.388, **20.757**, 17.185, **22.301**,
  17.278, **19.982, 20.702**
- `sustain2_r2_single` (21): 17.668, 17.711, 17.619, **19.931, 19.019** | 17.454, **20.450**,
  18.818, 17.825, **22.187**, 17.541 | 18.288, **22.885**, 18.269, **25.097**, 18.794, **21.348,
  20.013, 19.545, 20.817**, 18.010
- void `sustain2_r2_striped_void` (36): steps 1-16 at 8.230-8.917 s, then steps 17-36 at
  13.441-13.763 s.
- void `sustain2_r2_striped_void2` (36): every step 17.281-18.076 s.

Each drive's mean read rate over the two windows (§2), and its early/late ratio:

| run | C: early GB/s | C: late GB/s | D: early GB/s | D: late GB/s | C: early/late | D: early/late |
|---|---|---|---|---|---|---|
| `sustain2_r0_single` | 3.953 | 3.977 | 0.000 | 0.000 | 0.994 | none (not read) |
| `sustain2_r0_striped` | 4.015 | 2.506 | 4.021 | 2.505 | 1.602 | 1.605 |
| `sustain2_r1_striped` | 2.932 | 2.191 | 2.934 | 2.191 | 1.339 | 1.339 |
| `sustain2_r1_single` | 3.960 | 3.557 | 0.000 | 0.000 | 1.113 | none (not read) |
| `sustain2_r2_single` | 3.741 | 3.409 | 0.000 | 0.000 | 1.097 | none (not read) |
| void `_void` | 3.972 | 2.515 | 3.969 | 2.514 | 1.579 | 1.579 |
| void `_void2` | 1.943 | 1.936 | 1.947 | 1.938 | 1.003 | 1.005 |

### 5.12 Sequence 2: losses

As in §5.5 (recorded, not judged):

- **Cross-arm, SINGLE minus STRIPED, steps 1-21.** Round 0: median absolute difference 0.006381,
  maximum 0.031302 (step 8), first timed step +0.006381. Round 1: median 0.007954, maximum
  0.023487 (step 14), first step -0.003266.
- **Same arm across rounds.** STRIPED round 0 minus round 1: steps 1-21 median 0.006589, maximum
  0.030277 (step 17); all 36 steps median 0.004003. SINGLE round 0 minus round 1: median 0.009480,
  maximum 0.024006 (step 14); round 0 minus round 2: median 0.006919, maximum 0.019055; round 1
  minus round 2: median 0.008882, maximum 0.036052 (step 13).
- **Against attempt 1's round 0.** SINGLE: median 0.009949, maximum 0.023378. STRIPED (36 steps):
  median 0.004498, maximum 0.026865.
- The largest cross-arm difference (0.031302) sits inside the largest same-arm one (0.036052), as
  in the gate and unlike attempt 1's single pair. The final losses: SINGLE 1.425866-1.440552,
  STRIPED 0.019509-0.020058 (the void attempts included).

### 5.13 Sequence 2: same bytes, the caches compared again

22:28:24-22:30:20, after the stop, with no arm running, §4's command with the output named for
this prefix (`sustain2_cache_compare.json` / `.log`; attempt 1's `sustain_cache_compare.*` is
untouched). Result: **ALL EQUAL**: 83 files, 2,405 tensors, 36.388 GB, in 117 s.

### 5.14 Sequence 2: box state, the during-arm samples and the per-process capture

**Power, suspend and commit.**

- **AC** at every 2-s sample of the build and of every arm, the two void attempts included.
- The battery **charged** from 65% (build) to 100% (by the end of round 2's SINGLE arm). Its flag read 8
  at the build, 9 at every round stamp, and 1 at the very last after stamp, at 100%.
- **No suspend.** The largest gap between power samples was 2.015-2.055 s.
- **Commit headroom** was at least 11.724 GiB at every sample of a round arm (its low, in round 2
  SINGLE) and 15.715 GiB in the build.

**The stamps.** Every stamp read GPU 0 MiB with no compute apps, ASPM AC index 2, and
`soup_cli.__file__` under `Soup-sustain\src`. The Python process count read 5-7.

**PDH over each whole arm**, process start and teardown included:

| run | C: read GB/s mean / max | D: read GB/s mean / max | C: write MB/s mean / max | D: write MB/s max | CPU % mean | pages/s mean |
|---|---|---|---|---|---|---|
| build | 0.925 / 3.268 | 0.851 / 3.484 | 22.95 / 149.07 | 0.06 | 12.9 | 8610 |
| `sustain2_r0_single` | 3.752 / 4.622 | 0.000 / 0.000 | 0.33 / 13.43 | 0.00 | 5.9 | 144 |
| `sustain2_r0_striped` | 2.874 / 4.686 | 2.869 / 4.821 | 0.77 / 36.52 | 0.06 | 6.6 | 465 |
| `sustain2_r1_striped` | 2.303 / 4.724 | 2.301 / 4.724 | 0.20 / 14.41 | 0.09 | 5.5 | 18 |
| `sustain2_r1_single` | 3.517 / 4.520 | 0.000 / 0.000 | 0.47 / 16.92 | 0.00 | 5.7 | 508 |
| `sustain2_r2_single` | 3.404 / 4.583 | 0.000 / 0.000 | 5.14 / 126.79 | 0.00 | 7.2 | 2471 |
| void `_void` | 2.877 / 4.488 | 2.873 / 4.853 | 1.21 / 35.60 | 0.09 | 6.9 | 94 |
| void `_void2` | 1.879 / 2.526 | 1.877 / 2.640 | 0.54 / 20.13 | 0.09 | 5.5 | 57 |

**10-s bins**, aligned to each arm's first timed step, full bins only:

- `sustain2_r0_single` (36 bins): C: 3.85-4.07 GB/s (median 3.98). No bin below 3.0.
- `sustain2_r0_striped` (40 bins): bins 0-12 (21:38:22-21:40:32) read 3.75-4.15 GB/s on C:. From
  bin 13 (21:40:32) to the last, all 27 bins read **2.36-2.59 GB/s on each drive** (C: and D:
  within 0.05 of each other in every bin).
- `sustain2_r1_striped` (51 bins): bins 0-1 read 3.87 / 3.58 GB/s (C:). Bins 2-50 (from 21:46:07)
  read 2.42-2.58 GB/s on each drive, except two deeper stretches on both drives at once:
  - 21:50:17-21:50:47: C: 1.08, 0.61, 0.92 and D: 1.05, 0.61, 0.95;
  - 21:53:47-21:54:17: C: 1.31, 0.62, 0.79 and D: 1.27, 0.62, 0.82.
- `sustain2_r1_single` (39 bins): C: 2.51-4.05 GB/s (median 3.92). Six bins below 3.0 GB/s, at
  21:57:08 (2.58), 21:57:58 (2.81), 21:58:58 (2.73), 21:59:38 (2.80), 22:00:28 (2.53) and
  22:01:18 (2.51): 50-60 s apart, each a single 10-s bin.
- `sustain2_r2_single` (40 bins): C: 2.60-4.05 GB/s (median 3.79). Ten bins below 3.0 GB/s, from
  22:03:33 to 22:08:33, at 2.60-2.99, in single bins or pairs about 50 s apart (22:03:33,
  22:04:23, 22:05:13, 22:06:03-22:06:23, 22:06:53-22:07:13, 22:07:43-22:08:03, 22:08:33).
- void `_void` (40 bins): bins 0-13 read 3.49-4.12 GB/s on C:. From bin 14 (22:12:06) to the last,
  all 26 bins read 2.43-2.58 GB/s on each drive.
- void `_void2` (63 bins): every bin read **1.86-2.04 GB/s** on each drive (median 1.95 / 1.94).

**The slow stretches, step by step** (C: / D: GB/s over each step):

- `sustain2_r0_striped`: steps 1-15 read 3.88-4.20 GB/s per drive. Step 16, from 21:40:28, read
  11.303 s at C: 3.09 / D: 3.05. Steps 17-36, from 21:40:40 to the end of the arm, read
  13.357-13.941 s at 2.43-2.56 GB/s per drive. The plateau began **2 min 6 s** after the arm's
  first timed step (21:38:22), 2 min 34 s after its process started.
- `sustain2_r1_striped`: steps 1-2 read 8.311 / 8.630 s (4.00 / 3.90 GB/s on C:). Every later
  step read 13.487-13.815 s at 2.45-2.53 GB/s per drive, except steps 21-22 (from 21:50:08:
  22.993, 24.050 s; C: 1.54, 1.36 and D: 1.53, 1.38) and steps 35-36 (from 21:53:38: 18.250,
  28.068 s; C: 1.94, 1.16 and D: 1.93, 1.17). The plateau began at step 3, 21:46:04, 17 s after
  the arm's first timed step; the arm started 6 s after round 0's STRIPED arm ended.
- `sustain2_r1_single`: steps 7, 8, 10, 13, 16, 18, 20 and 21 read 19.169-22.301 s, at C:
  3.08-3.57 GB/s, against 3.88-4.00 GB/s in steps 1-6.
- `sustain2_r2_single`: steps 4, 5, 7, 10, 13, 15, 17, 18, 19 and 20 read 19.019-25.097 s, at C:
  2.80-3.61 GB/s (step 15: 25.097 s, 2.80 GB/s).
- void `_void`: steps 1-16 read 3.81-4.20 GB/s per drive. Steps 17-36, from 22:12:02 to the end,
  read 2.46-2.57 GB/s. That plateau began 2 min 16 s after the arm's first timed step (22:09:46).
- void `_void2`: every step read 1.87-1.98 GB/s per drive, from the first (22:17:19) to the last.

What those entries share, as measured:

- **Three levels per drive in the STRIPED arms**: about 4.0 GB/s, a plateau at 2.43-2.59 GB/s
  (10-s bins) and a level at 1.86-2.04 GB/s that held for a whole arm (`_void2`), plus two short
  excursions to 0.61-1.31 GB/s in `sustain2_r1_striped`. Both drives sat at the same level in every
  bin (within 0.09 GB/s), as in attempt 1.
- **The 2.5 GB/s plateau no longer recurred and recovered as in attempt 1; once it began, it held
  to the end of the arm**, in all three STRIPED arms that reached it. It began after 15 and 16
  quiet steps (about 2 min 6 s and 2 min 16 s) in the two STRIPED arms that followed a SINGLE arm
  (D: idle about 12 and 15 minutes before them), and within 17 s in the arm that followed a
  STRIPED arm.
- **C: alone also dipped**: from round 1 on, both SINGLE arms had one-bin dips to 2.51-2.99 GB/s
  about every 50-60 s. Round 0's SINGLE arm, the first of the sequence, had none.
- In the plateau steps the GPU read SM 967-1800 MHz, 64-74 °C and 37.6-131.3 W
  (`sustain2_r0_striped`), against SM 1717-2557 MHz, 71-81 °C and 69.3-126.7 W in its steps under
  9.5 s.
- CPU: the median 10-s bin read 5.1-6.5% in every arm, its highest 9.1-15.0%. C: writes stayed
  under 10 MB/s per 10-s bin except in round 2's SINGLE arm (up to 32.1 MB/s, from about 22:06).
- **Stated as reasoning, not measured**, as in §5.7: in a STRIPED step the reader takes layers
  from the two drives alternately, so the counters of a STRIPED step cannot say which drive slowed
  first. The SINGLE arms' dips are C: alone.

**GPU (10-s `nvidia-smi`).**

| run | SM MHz | °C | W |
|---|---|---|---|
| build | 1327-2527 | 48-66 | 11.4-69.2 |
| `sustain2_r0_single` | 0-2190 | 41-70 | 11.0-74.3 |
| `sustain2_r0_striped` | 0-2557 | 54-81 | 10.3-131.3 |
| `sustain2_r1_striped` | 180-2430 | 54-78 | 11.6-134.4 |
| `sustain2_r1_single` | 517-2205 | 53-71 | 11.5-67.3 |
| `sustain2_r2_single` | 0-2047 | 53-70 | 8.3-92.8 |
| void `_void` | 0-2512 | 54-79 | 7.7-118.1 |
| void `_void2` | 0-2355 | 53-73 | 8.1-111.5 |

- The throttle-reason field read `0x0000000000000004` or `0x0000000000000400` in every sample.
- Two samples read an SM clock no card has: 22290 MHz at 21:45:19 (`sustain2_r1_striped`'s first
  sample, before its first step) and 7237 MHz at 22:27:52 (`_void2`'s last, after its last step).
  They are kept in the JSON and left out of the table above.

**Top readers, from the after stamps.** No capture failed. The largest reader per arm (read bytes):

- build: `svchost.exe` (21332), 261.6 MB;
- `sustain2_r0_single`: `svchost.exe` (16716), 114.6 MB;
- `sustain2_r0_striped`: `msedgewebview2.exe` (6952), 343.1 MB; then `NVDisplay.Container.exe`
  53.0 MB, `codex.exe` 47.4 MB, `audiodg.exe` (new) 24.4 MB;
- `sustain2_r1_striped`: `svchost.exe` (16716), 60.5 MB; then `msedgewebview2.exe` 59.4 MB,
  `NVDisplay.Container.exe` 53.0 MB;
- `sustain2_r1_single`: `codex.exe` (9972), 164.6 MB;
- `sustain2_r2_single`: `SearchIndexer.exe` (26528), **599.4 MB**, below the bar; then
  `codex.exe` 309.9 MB, `svchost.exe` (16716) 193.9 MB;
- void `_void`: `SearchIndexer.exe` (26528), **1,269.2 MB**; then `codex.exe` 81.3 MB;
- void `_void2`: `SearchIndexer.exe` (26528), **1,942.7 MB**; then `msedgewebview2.exe` 150.7 MB,
  `svchost.exe` (16716) 136.3 MB.

So the plateau began in `sustain2_r0_striped` and held through `sustain2_r1_striped` with no
reader above 343.1 MB in either arm. The full lists are in `sustain_box_state.log`. The capture
names survivors only.

**System log, 21:25-22:31.** No disk, storage, NTFS, thermal, power-source or sleep event.

- Windows Update tried a Store app update (OpenAI Codex) at 21:25:26, during the build. It failed
  with 0x80073D02.
- The time service logged its "VMICTimeProvider ... not supported" information event at 21:40:38,
  21:57:42 and 22:14:46. The first falls 10 s after step 16 of `sustain2_r0_striped` began
  (21:40:28), the step in which that arm's plateau set in. The other two fall inside
  `sustain2_r1_single` and `_void`, at no step boundary of note.

**Activity of this probe's operator during the sequence, disclosed.** None of it is in any capture.

- During the arms: a monitor read the driver's stdout log every 20-30 s, and one `ls`/`tail`
  (21:37:43) and one `tasklist` (22:19:43) ran.
- Three short read-only Python scripts ran in the settle (21:26-21:31). They read the throttle
  probe's committed JSON and attempt 1's 7-KB `sustain_r0_single.json`. No arm was running.

Also running, from other sessions: an idle watcher script, a merge-polling loop and short
GitHub-posting scripts (network only), and two editor language servers.

### 5.15 Sequence 2: anomalies, as measured

1. **The slowdown changed shape.** In attempt 1 the 13.5-s steps came in 25-30 s stretches about
   70 s apart, and the arm recovered between them. In sequence 2, once the 2.5 GB/s plateau began
   in a STRIPED arm it held to the end of the arm. That happened in all three STRIPED arms that
   reached it (rounds 0 and 1, and round 2's void attempt 1). Round 1's STRIPED arm was on the
   plateau from its third step.
2. **A lower level, for a whole arm:** round 2's second STRIPED attempt (void) read 1.86-2.04 GB/s
   per drive in every bin, 17.28-18.08 s a step: a STRIPED step as slow as a SINGLE one.
3. **Two short excursions to 0.61-1.31 GB/s on both drives** in round 1's STRIPED arm (steps
   21-22 and 35-36, 18.3-28.1 s steps).
4. **C: alone dipped in the SINGLE arms from round 1 on**: one-bin dips to 2.51-2.99 GB/s about
   every 50-60 s. Round 0's SINGLE arm had none. This is the same shape the throttle probe saw on
   C: alone, unbuffered and with no GPU work, in its second (void) attempt: 11-12 s at about
   2.6 GB/s, about 60 s apart (throttle record §5.4).
5. **Windows Search's indexer read 1.27 GB and 1.94 GB in the two round-2 STRIPED attempts** (and
   599.4 MB in round 2's SINGLE arm), and voided the position. It read nothing above 1 GB in
   rounds 0 and 1, where the plateau began and held.
6. **Round 0's STRIPED early window was as fast as attempt 1's**: `E` 8.353 s against 8.376 s.
7. **Two impossible SM-clock samples** (22290 and 7237 MHz), each at an arm's edge, outside any
   step.

## 6. Verdict

### Sequence 2 (2026-10-02, prefix `sustain2`): NO VERDICT (row 2)

The rule reads three complete rounds. Rounds 0 and 1 are complete. Round 2 has its SINGLE arm, but
its STRIPED arm was void twice.

| row | inputs | result |
|---|---|---|
| 1: any SINGLE arm's `E` outside 15.0-21.5 s | 17.388, 17.303 and 17.711 s (rounds 0, 1, 2) | does not fire |
| 2: an arm off the path under test, or the protocol stopped | The five valid arms show no problem: `direct_io` / `pinned` True, `AsyncDiskSource`, read-ahead 2 / 3, i mod 2 over `D:\soup-stripe` / no stripe roots, 157 + 2 loads in every timed step, `shard_seconds` 0.007-0.022 s. The protocol stopped: round 2's STRIPED position was void twice, on the foreign-reader condition (`SearchIndexer.exe` 1.27 GB, then 1.94 GB). `--summarize`: "no valid arm: ['r2 STRIPED']; protocol: stopped: sustain2_r2_striped void twice; no verdict" | **fires: NO VERDICT** |
| 3: `u_late` >= 1.595 in every round | none | not reached |
| 4: `u_late` >= 1.20 in every round | none | not reached |
| 5: anything else | none | not reached |

This stop is one row 2 lists by name ("a position void twice"). Attempt 1's was not.

**What rounds 0 and 1 alone read. This is NOT a verdict and cannot stand in for one.**

- **`u_late`.**
  - Round 0: 17.298 / 13.525 = 1.279.
  - Round 1: 19.185 / 13.590 = 1.412.
  - The median of the two is 1.345.
  - Both are below row 3's 1.595 and at or above row 4's 1.20. A third round is missing, so this
    places the two rounds in row 4's band and decides nothing.
- **`u_early`.** 2.082 (round 0) and 1.274 (round 1). Round 1's STRIPED arm was on the 2.5 GB/s
  plateau from its third step, so its early window was already slow. Round 1's drop came across
  arms, not within one (§2 note 1).
- **`d`.**
  - SINGLE: 0.995, 1.109 and 1.117.
  - STRIPED: 1.619 and 1.001.
  - Above the 1.080 label: round 0 STRIPED (1.619), round 1 SINGLE (1.109) and round 2 SINGLE
    (1.117).
- **Drive early/late ratios above the 1.10 label.**
  - Round 0 STRIPED: C: 1.602 and D: 1.605.
  - Round 1 STRIPED: C: 1.339 and D: 1.339.
  - Round 1 SINGLE: C: 1.113.
  - Round 2 SINGLE's C: is 1.097, under the label.
- **The case note 2 was written for.** The later SINGLE arms slowed in their late windows (`L`
  19.185 and 19.779 s), with C: alone dipping about once a minute (§5.14). Their `E` stayed inside
  the band, so row 1 did not fire.
- **The void attempts, for completeness.** Had either round-2 STRIPED attempt counted, round 2's
  `u_late` would read 1.463 (attempt 1) or 1.127 (attempt 2, below 1.20). Neither counts.

A verdict needs a sequence with three complete rounds. In two windows the box has not given one:
first a hibernation, then Windows Search's indexer. Whether to pause the indexer, or exclude the
cache folders from it, for a third attempt is a change to the box, not to the rule. That, and
whether to run again at all, is the owner's call. The cache is unchanged and verified equal again
(§5.13).

### Attempt 1 (2026-10-01, prefix `sustain`): NO VERDICT (row 2)

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
