<!--
Working measurement record, published verbatim. The decision rule in §2 was
written and committed BEFORE any arm ran, which is the point of it: a rule
written after the numbers are in is not a rule. Its harness was committed
before the rule text (dcc72ebb), and nothing in the harness has read a cache
file as a measurement: the only metadata listing made so far was the folders'
file and size count in §1.

Hardware: RTX 5070 Laptop GPU (Blackwell sm_120, 8151 MiB), Intel i9-14900HX,
31.7 GB DDR5-5600, two Samsung PM9B1 NVMe (MZAL81T0HFLB-00BL2, 954 GB each):
C: on a CPU root port (PEG0, 00:06.0), D: behind the chipset (RP13, 00:1D.4,
over the DMI 4.0 x8 link). Windows 11 Pro 10.0.26200.
Stack: Python 3.12.10, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0,
bitsandbytes 0.50.2 (read from the installed packages on 2026-10-01).
Soup: branch probe/two-drive-sustained, worktree C:\Users\user\projects\Soup-sustain,
cut at 2c6ed087 (the head of PR #1536, feat/two-drive-striping). src/ is
byte-identical to 2c6ed087 (git diff 2c6ed087 HEAD -- src is empty, checked on
2026-10-02); only benchmarks/ changed.
Harness: benchmarks/harness/two_drive_throttle.py (driver),
two_drive_throttle_read.py (one arm's reader) and two_drive_throttle_rule.py
(the rule of §2 as code), all in dcc72ebb. The reader imports the R1 read
primitive from two_drive_read.py; the driver imports the stamps, pre-arm
checks, sampler and void rule from two_drive_stripe_gate.py and the GPU sampler
and foreign-reader check from two_drive_sustained.py, unchanged.
-->

# Probe - does either drive slow down by itself under sustained reading? (follows R4)

**Status: NO VERDICT (§2 row 1), 2026-10-02.** The rule (§2) was committed in `3a324993`,
before the run. The sequence stopped in its first position. C-ALONE round 0 was void twice: Windows
Search's indexer (`SearchIndexer.exe`) read 2.37 GB and then 1.82 GB during the arm, over §4's 1 GB
foreign-reader bar. No D-ALONE or BOTH arm ran (§5.1, §6).

Descriptive only, from the two void attempts (§5.4):

- read from rest for 300 s, C: had no dip;
- read again straight after, C: fell to about 2.6 GB/s for 11-12 s, three times, about once a
  minute. Each drop is too short to be an episode.

Before the run, the reader and driver had been exercised only on synthetic arms (`--selftest`) and
in two dry runs that read no cache file. On 2026-10-01 the pre-arm check was not met (another
session held 4746 MiB of GPU memory). On 2026-10-02 at 14:30 it was not met either (the laptop was
on battery at 74%).

---

## 0. The question

The sustained striping probe ([`probe-rtx5070-two-drive-sustained.md`](probe-rtx5070-two-drive-sustained.md),
no verdict) saw recurring slow stretches in both of its STRIPED arms and none in its SINGLE arm
(§5.7 there):

- steps of 13.463-13.546 s instead of about 8.4 s, with **both drives at 2.50-2.55 GB/s** each,
  against 3.78-4.21 GB/s per drive in every other STRIPED step;
- the first stretch began 4 min 36 s after round 0 STRIPED's first timed step (about 5 min 5 s
  after the arm's process started); in round 1 three stretches began 69 and 70 s apart (20:35:25,
  20:36:34, 20:37:44);
- no foreign reader of 1 GB or more in any arm (the largest in the dipping arms: 133.1 MB);
- the SINGLE arm, C: alone for 6 min 5 s, had no slow step (C: 3.71-4.08 GB/s in every 10-s bin).

Recomputed here at 5-s resolution from the same 1-s PDH samples (`two_drive_throttle_rule.py
--calibrate`, §2): each stretch is a **flat plateau** of 5-6 bins (25-30 s), on both drives at the
same level (42 of its 44 dip bins sit between 2.40 and 2.62 GB/s). It is a step between two levels,
not a ramp.

A STRIPED step reads layers from the two drives in turn with one layer in flight per drive, so a
slow drive holds back both: the counters of a striped step cannot say which drive slowed. The
hypothesis behind this probe is that **one drive thermally throttles**, most likely D: (behind the
chipset), and the other follows. It is not established, and the drives' temperatures cannot be read
here (§3). So:

**Under sustained unbuffered reading, does each drive alone show periodic slowdowns, and does
reading both at once change that?** Read as four outcomes, none predicted:

| the dips come from | what this probe would show |
|---|---|
| one drive, by itself (thermal, firmware, power state) | that drive dips in its ALONE arms |
| a path the two drives share (DMI, CPU package, memory, power budget) | neither dips alone, one or both dip in BOTH |
| the training step (reader pipeline, host-to-device copy, GPU load or heat) | neither drive dips anywhere |
| something not reproduced in 5 minutes per arm | dips in one round only, or nowhere |

## 1. The fixture and the arms

**What is read.** The caches the sustained probe built and verified (tensor by tensor equal,
sustained §5.6); they are only read, never written:

| drive | folder | listed 2026-10-02 (names and sizes only) |
|---|---|---|
| C: | `C:/Users/user/.soup/layer-stream/D___synth__llama-70b-shape-v32k-f4` (the one-root 70B NF4 cache) | 84 files, **82 spans** of 400 MiB (34.4 GB) |
| D: | `D:/soup-stripe/D___synth__llama-70b-shape-v32k-f4-cba3360a9c80` (the striped cache's 40 odd layers) | 41 files, **40 spans** (16.8 GB) |

C: reads the one-root cache in every arm, including BOTH, so C: is compared with itself. (A
striped step reads C:'s half from the separate striped primary, which this probe does not touch.)

**How it is read.** The R1 primitive ([`probe-rtx5070-two-drive-read.md`](probe-rtx5070-two-drive-read.md)
§3), imported not re-typed: `FILE_FLAG_NO_BUFFERING`, one handle per range, one `ReadFile` per
range, into a sector-aligned pinned arena, one arena per drive. A span is the first 400 MiB of a
file (a layer file is 441.4 MB), read as K = 4 parallel ranges of 100 MiB; the next span starts
when all four have landed, so **one span is in flight per drive**, as in the striped reader. It
differs from R1 in two ways, both on purpose:

- **Time-boxed and cyclic.** An arm reads for 300 s, wrapping round the drive's span list in file
  order. The same bytes are read again and again, as a training run re-reads them every step (a sustained arm of 37 steps read each file about 70 times: two loads per layer per step); R1 read no span twice. An unbuffered read never
  touches the page cache, so a re-read is as cold as the first.
- **Every span is logged**, start and duration to 0.1 ms, so the arm can be binned at any width
  afterwards. A 5-s bin holds about 30-65 spans.

In a BOTH arm each drive has its own thread and its own arena and they share only a start barrier.
**Unlike the striped training reader they are not locked to one layer at a time**: a slow drive
slows only itself, and each drive's rate stays separable. That is what lets BOTH say which drive
dipped. The reader creates a CUDA context to pin its arenas and runs no compute beyond allocating one element.

**The arms** (one process each, never two at once). Round 1 is round 0 reversed:

| position | round | arm | drives read |
|---|---|---|---|
| 0 | 0 | C-ALONE | C: |
| 1 | 0 | D-ALONE | D: |
| 2 | 0 | BOTH | C: and D: |
| 3 | 1 | BOTH | C: and D: |
| 4 | 1 | D-ALONE | D: |
| 5 | 1 | C-ALONE | C: |

**Why 300 s.** The earliest dip on record came about 276 s into the reading (4 min 36 s into
round 0 STRIPED's timed steps), 5 min 5 s after its process started. An arm shorter than that
cannot show a dip that starts at the same age, so a 300 s arm reads just past it, with a 285 s
coverage floor in §2. Six such arms are 30 min of reading by themselves.

**Rest: 30 s** between arms, plus about 20 s of stamps (§4). That is process turnaround, not a
cool-down, and it is stated as such: about 75 s pass between one arm's last read and the next
arm's first. What cools a drive is the other drive's ALONE arm (about 450 s idle for the drive
not read), and BOTH to BOTH is deliberately close to back to back, as the sustained probe's two
STRIPED arms were (a few seconds apart there). The drives' temperatures are unknown (§3), so "rested" below
means only "not read for that long".

**Where each drive stands at each arm's start, at the planned times** (reader start-up 25 s, an
arm 300 s, 75 s between arms; the actual times are recorded). *Idle* is the time since the drive was
last read; *run* is how long the drive had been read with gaps under 120 s when the arm starts.

| position | arm | C: idle / run | D: idle / run |
|---|---|---|---|
| 0 | C-ALONE r0 | before the sequence: unknown / 0 | - |
| 1 | D-ALONE r0 | - | at least 400 s / 0 |
| 2 | BOTH r0 | 450 s / 0 | 75 s / 375 s |
| 3 | BOTH r1 | 75 s / 375 s | 75 s / 750 s |
| 4 | D-ALONE r1 | - | 75 s / 1125 s |
| 5 | C-ALONE r1 | 450 s / 0 | - |

A dash is a drive the arm does not read. So D: is read from rest in D-ALONE r0 and after nearly 19
minutes of reading in D-ALONE r1; C: is read from rest in both its ALONE arms (the first after an
unknown rest, the second after 450 s) and is warm only in BOTH r1. The rule (§2, note 1) says what
that asymmetry does to the verdicts.

## 2. The decision rule, written before the run

Nothing in this section is edited once a number exists. It is implemented in
`benchmarks/harness/two_drive_throttle_rule.py`, which `two_drive_throttle.py --summarize` applies
to the committed files alone.

**Definitions.**

- **Bin.** 5 s, counted from the arm's first read. Two rates per drive per bin: the reader's own
  (the bytes of the logged spans, a span's bytes spread over its duration) and the volume's PDH
  `\LogicalDisk(X:)\Disk Read Bytes/sec` (the mean of the 1-s samples stamped inside the bin; a
  bin with fewer than 3 samples has none). The 1-s versions of both are recorded.
- **Dip bin (per drive).** The reader's own rate **and** the PDH rate are both **below 3.0 GB/s**
  (strictly). Two instruments, so a dip is neither a stall of the reader's thread nor a quirk of
  the counter.
- **Episode.** At least **3 consecutive** dip bins (15 s) on one drive in one arm.
- **A drive dips in an arm** when it has at least one episode in that arm.

**The rows**, applied in order. Row 1 judges the whole sequence; rows 2-5 are applied to each drive
separately, C: and D:, and the first that fires decides that drive. The verdict is the pair.

| row | measured | verdict |
|---|---|---|
| 1 | any validity condition below fails, or a position has no valid arm, or the driver did not finish | **NO VERDICT**: the arms did not measure what the rule reads |
| 2 | drive X dips in its own ALONE arm in **both** rounds | **X THROTTLES**: read alone, the drive slows |
| 3 | drive X dips in **neither** ALONE arm, and in at least one BOTH arm | **X ONLY WHEN BOTH**: the slowdown needs the other drive reading too |
| 4 | drive X does not dip in any of its four arms | **X STEADY** |
| 5 | anything else: X dips in exactly one of its two ALONE arms | **X INCONSISTENT**: reported as measured |

Rows 2-5 cover every case: with `(a0, a1)` for X's ALONE arms and `b` for "X dips in at least one
BOTH arm", row 2 is `a0 and a1`, row 3 is `not a0, not a1, b`, row 4 is `not a0, not a1, not b`, and
what remains is exactly one of `a0`, `a1`.

**Validity (row 1).** In two groups, handled differently by the driver (§4).

- *Environment, per arm; a failure voids the arm, which is re-run once in its position:* AC on at
  every 2-s power sample and after the arm; no gap of more than 30 s between power samples (a
  suspend); the reader wrote its JSON; no process other than the driver read **1 GB or more** during
  the arm (per-process capture, before and after); commit headroom at least **8 GiB** at every 2-s
  sample.
- *Instrument, per arm; a failure stops the sequence, since no later arm can repair it:* the arm
  read `pinned` arenas; exactly the arm's drives were read; the byte check (first and last MiB of
  the last span against a buffered read) is true for each; the files cycled over hold at least
  **16 GB** per drive; the arm read for at least **285 s**; each drive read in the arm reached a 5-s
  bin of at least **3.5 GB/s** (the instrument reproduces the quiet rate at least once); no more than
  **10%** of an arm's bins lack a PDH value.

**Where each threshold comes from.** Every value is read from a committed record, and the source is
quoted. `--calibrate` reprints the first three rows' numbers from `sustain_driver.json`.

| threshold | value | source |
|---|---|---|
| dip level | 3.0 GB/s per drive, strict | sustained §5.7: "Every step near 13.5 s (13.463-13.546 s) read 2.50-2.55 GB/s from each drive" and "Every STRIPED step not listed above read 3.78-4.21 GB/s per drive". At 5-s resolution in the same PDH samples (`--calibrate`): 44 bins below 3.0 GB/s, 42 of them between 2.40 and 2.62 (the other two, 2.88 and 2.97, are transition bins); 275 bins not next to a dip bin read 3.23-4.31 GB/s (median 4.01), the lowest being round 1's first 20 s under CPU contention (sustained §5.7). 3.0 sits 0.4 above the highest plateau bin and 0.2 below the lowest quiet one |
| bin | 5 s | A dip lasts 25-30 s (5-6 bins) in all four stretches on record, on both drives; sustained §5.7's own bins are 10 s. 5 s holds 30-60 spans of the reader |
| episode length | 3 bins (15 s) | The shortest stretch on record is 5 bins (25 s): round 0 STRIPED's is 6 and round 1's are 5, 6 and 5, on both drives (`--calibrate`). 15 s is 60% of the shortest |
| arm length, coverage | 300 s, 285 s | sustained §5.7: "The first slow stretch began 4 min 36 s after round 0 STRIPED's first timed step"; sustained §0 and gate §5.12: dips "five minutes after that sequence's first arm started" |
| quiet level | 3.5 GB/s | R1, "What the day's record adds up to": "C: alone read 3.52-4.68 GB/s across twenty-four single-drive arms"; D: alone read 3.94-6.02 in R1's rounds |
| foreign reader | 1 GB per arm | sustained §2's threshold table (the gate's largest was 58.2 MB). Note the limit: in the dipping arms the largest foreign reader was 133.1 MB (sustained §5.7), so the bar does not exclude a smaller reader as a cause; each episode records the volume's write rate and the CPU load (below) |
| commit headroom | 8 GiB | the gate's pre-arm check (`HEADROOM_MIN`), here also held at every sample |
| files cycled | 16 GB per drive | the listed pools, 34.4 GB (C:) and 16.8 GB (D:) (§1). A sustained STRIPED step reads the 40 layers of a drive's half (17.7 GB on D:, 18.7 GB on C: with the embedding and head) about twice, and a SINGLE step the 36.4 GB of C: |
| PDH coverage | 90% of bins | a bin needs 3 of its 5 samples; 10% is the share the rule can lose without a dip being hidden |

**What the rule deliberately does, stated now so none of it is read in afterwards.**

1. **THROTTLES needs the dip in both rounds, and the two rounds do not start alike (§1's table).**
   A drive whose first dip needs more than 300 s of reading from rest can dip in its warm ALONE arm
   and not in its rested one. That reads as row 5, INCONSISTENT, with the episodes in one round
   only. It is not called THROTTLES. Each episode records `continuous_read_s`, how long its drive
   had been read, with gaps under 120 s, when it began, so a reader of the record can tell "dips only
   after long reading" from "dips at random". This is the cost of asking for a dip in both rounds,
   as the question was put.
2. **A dip in a BOTH arm is attributed to the drive that shows it.** The two drives are not locked
   together here, so one drive's dip does not slow the other except through a shared path. Each
   BOTH episode records `other_drive_dip_share`, the share of its bins in which the other drive was
   also a dip bin. 1.0 on both drives at the same bins, as in the sustained arms, would say the
   dips are simultaneous even without the reader's lockstep; the rule does not set a row on it.
3. **The dip level is absolute.** A slowdown that stays above 3.0 GB/s (D: from 5.5 to 3.6, say)
   is not an episode. It is not hidden: the per-arm best and median bins and all the 1-s and 5-s
   bins are recorded. It sets no row.
4. **STEADY says "no dip of the sustained probe's kind in this reading".** It does not say the
   drive never throttles: only 300 s per arm, one drive pair, and no GPU load were read (§3).
5. **Period, depth and duration are descriptive.** Each episode records its start, length, minimum
   and mean rate, and the spacing of starts within an arm is reported (median). The rule does not
   test for the 70-s period.
6. **Medians and means do not enter.** An episode is a run of bins, so one bad bin cannot make
   one, and a single dip is enough for "dips in an arm" (the first dip in the sustained record
   came after 4.5 minutes, so one episode per arm is all a 5-minute arm can be asked for).

**What follows from each verdict** (planned next steps, not a result):

| verdict | what follows |
|---|---|
| X THROTTLES | the drive is the lever. Its temperature (needs elevation) and a cooling or pacing run on that drive are next. The cause (thermal, firmware, power state) is not distinguished here. For D: a striped reader that waits on the slower drive is the consequence the sustained probe saw |
| X ONLY WHEN BOTH | the slowdown needs both drives reading. The shared path is next: the DMI link, the CPU package, memory bandwidth, the laptop's power budget |
| X STEADY (for both drives) | raw reading does not reproduce the dips in these arms. The step itself is next: the reader's pipeline, the host-to-device copy, the GPU's load and heat. This probe cannot say which |
| X INCONSISTENT | reported with the round and the `continuous_read_s` of each episode. A re-run under a new `--run-prefix` decides; no cause is named |

**The rule's own checks, run before this commit.** `two_drive_throttle.py --selftest` (CPU only, no
disk beyond a temporary folder) builds synthetic arms and checks the binning, the strict threshold,
the two-instrument requirement, the 3-bin minimum, every validity condition and every row, then
writes the files of a whole sequence and reads them back through `collect` and `summarize`. It
was mutation-checked on temporary copies: of 20 one-line mutants of the rule, 19 are killed (seven
only after a test was added for a survivor of the first pass) and 1 is equivalent. `two_drive_throttle_rule.py
--calibrate` runs the detector (PDH only: that record has no reader spans) on the sustained arms:

```
sustain_r0_single C: 0 episode(s)
sustain_r0_striped C: 1 episode(s) (start s, length s, min GB/s) [(280.0, 30.0, 2.44)]
sustain_r0_striped D: 1 episode(s) (start s, length s, min GB/s) [(280.0, 30.0, 2.4)]
sustain_r1_striped C: 3 episode(s) [(165.0, 25.0, 2.43), (225.0, 30.0, 2.44), (305.0, 25.0, 2.44)]
sustain_r1_striped D: 3 episode(s) [(165.0, 25.0, 2.41), (225.0, 30.0, 2.44), (305.0, 25.0, 2.44)]
```

That is the sustained record's own list (one stretch in round 0, three in round 1, none in the
SINGLE arm), found by a detector whose constants were chosen from those same samples. It shows the
detector finds what was seen and nothing in the quiet arm; it is not evidence about this probe's
arms.

**The inputs, as the files name them** (fixed with the rule):

- the reader's JSON, `throttle_r<round>_<arm>.json`: `t0_unix`, `read_s_planned`, `pinned`, and per
  drive `spans.start_s`, `spans.dur_s`, `spans.nbytes`, `check_ok`, `pool_bytes`, `pool_spans`;
- the PDH samples: `runs[*].during_samples.pdh` in `throttle_driver.json`, as `[unix time,
  {"read_c": ..., "read_d": ..., "write_c": ..., "cpu_pct": ...}]`, bytes per second;
- the void inputs: `runs[*].during`, `after.ac_line_status`, `after.top_readers`.

`python benchmarks/harness/two_drive_throttle.py --summarize` recomputes every input and applies
the rows from the committed files alone. The comparisons are made on unrounded values against the
thresholds as written.

**Also recorded, not part of the verdict:** the 1-s own and PDH rates per drive; each drive's write
rate and the CPU load inside every episode; GPU SM clock, temperature, power and throttle reasons
every 10 s (`nvidia-smi`); AC, battery and commit every 2 s; `% Processor Time`, paging and
`% Processor Performance` every 1 s; the box stamps before and after every arm; the per-pass count
of each drive.

## 3. What this does NOT measure

- **Temperature.** `Get-PhysicalDisk | Get-StorageReliabilityCounter` refused without elevation on
  this box (sustained §3, checked 2026-10-01), and this probe changes no privilege. A dip is
  placed on a drive only through its read rate. THROTTLES names a drive that slows when read alone;
  it does not say the cause is heat.
- **The training step.** No GPU compute, no host-to-device copy, no layer decode. The GPU is idle
  but for a CUDA context. A dip that needs the GPU's heat or power draw (the sustained probe's GPU
  ran at 55-85 degrees C and up to 124 W) cannot appear here, and "STEADY everywhere" would say
  exactly that and no more.
- **The striped reader.** The drives are decoupled in BOTH, there is no read-ahead and no
  per-layer barrier across drives. R3' left depth 1 against depth 2 ambiguous by its rule (1.10x in two rounds of three).
- **Unique bytes.** The same 34.4 GB (C:) and 16.8 GB (D:) are cycled, as a step does; a drive that
  serves a re-read from an internal cache would hide in this. R1 treats the PM9B1 as DRAM-less (its
  §2 and §5 run 2) and nothing here tests that.
- **A long onset.** 300 s per arm, and at most about 24 minutes of reading in D:'s longest run
  (§1's table). A dip that needs an hour is outside it. Rested starts are only as rested as the idle times in §1.
- **The cause of anything.** No row names heat, firmware, the power state or the host.
- **Foreign activity below the bar.** A reader under 1 GB, a write burst or a Windows Update download
  can coincide with a dip; the episodes record write rate and CPU, nothing excludes them.
- **Other drives, other boxes, other days.** One pair of identical PM9B1, one laptop, one
  sequence. The rule needs two rounds; two rounds are not a distribution.
- What the earlier records already exclude: a real 70B checkpoint, warm (page-cached) reads, batch
  and sequence length (the step is not run at all), and the RAM tier.

## 4. Box state, per arm

The gate's and the sustained probe's protocol, run by `two_drive_throttle.py`, which imports their
functions rather than copying them. The same stamp is taken right before and right after every arm:
AC and battery, available RAM, commit used and limit, the Python process count, GPU memory and
compute apps, the AC ASPM index (read only), `soup_cli.__file__`, and every process's cumulative
read and write bytes, with the eight processes whose reads grew most named in the after stamp. While
an arm runs, the gate's sampler takes AC and commit every 2 s and the PDH counters every 1 s, and a
third thread runs `nvidia-smi` every 10 s.

What differs from the sustained probe:

- **Files.** `throttle_box_state.log`, `throttle_driver.json` and, per arm,
  `throttle_r<round>_<arm>.json` / `.log` (`c`, `d`, `both`), all in
  `benchmarks/results/probe-rtx5070/two-drive/`. The driver refuses to start when files of the
  same `--run-prefix` already exist, so a re-run takes a new prefix (`throttle2`).
- **Plan and rest.** Six arms in §1's order, 30 s of rest before every arm but the first.
- **Void.** Battery power at any sample, a suspend, no reader JSON, a foreign reader of 1 GB or
  more, or commit headroom under 8 GiB at any sample. A void arm is kept under a `_void` name and
  re-run once in the same position when the pre-arm checks hold again; a second void in a position
  stops the sequence.
- **Stopping early.** After each arm the driver applies row 1's instrument conditions and stops if
  one fails: no later arm can turn that into a verdict.
- **The listing.** Before the first arm the driver lists both folders (names and sizes only; no
  file is opened) and stops if either holds under 16 GB. `--check-files` does only this.
- **No build and no settle.** The caches exist and are not written. The first arm starts when the
  pre-arm check holds; how rested the drives are at that moment is not verified beyond it (§3).
  `--settle-s` exists and defaults to 0.

Unchanged: before every arm, AC on, GPU memory in use at most 1024 MiB, and commit headroom at least
8 GiB, polled every 30 s for at most 15 minutes, after which the sequence stops. The laptop stays
plugged in with the **lid open** (a closed lid sleeps it). Before the run, the other Claude
sessions on this box are asked by message to avoid CUDA runs, test suites and large copies on C: or
D: until it reports done. **The driver must outlive its launcher**: the sustained run's driver was
stopped when the shell that held it reached a 2-hour wall-clock limit that kept counting through a
hibernation. Run it as one background job whose time limit is well above the estimate below.

**Estimated wall time,** from the plan, without voids or pre-arm waits. The reader start-up is an
assumption (torch, the CUDA context, listing two folders and one untimed span), not a measurement.

| part | estimate | from |
|---|---|---|
| 6 reader start-ups | 6 x 25 s = 2.5 min | assumed |
| 6 arms of reading | 6 x 300 s = 30 min | §1 |
| 5 rests and 5 pairs of stamps | 5 x (30 + 20) s = 4.2 min | §1 |
| **total** | **about 37 min** | `--dry-run` prints 36.7 min |

Each void re-run adds about 6 minutes and each pre-arm wait up to 15.

**The commands, in order** (Git Bash, from the worktree, with the base interpreter, not a venv; the
driver sets each arm's `PYTHONPATH` to `Soup-sustain/src` itself):

```
cd C:/Users/user/projects/Soup-sustain
PYTHONPATH="C:/Users/user/projects/Soup-sustain/src" python -c "import soup_cli; print(soup_cli.__file__)"
python benchmarks/harness/two_drive_throttle.py --selftest
python benchmarks/harness/two_drive_throttle.py --check-files
python benchmarks/harness/two_drive_throttle.py --dry-run
python benchmarks/harness/two_drive_throttle.py --run-prefix throttle
python benchmarks/harness/two_drive_throttle.py --summarize --run-prefix throttle
```

The `python -c` line must print a path under `Soup-sustain\src`. `--dry-run` writes to a scratch
folder and lists no cache folder; it stamps once and prints the six reader commands. The
`--run-prefix throttle` line is the run: launch it as one background job with a timeout of at least
50 minutes (3,000,000 ms in the Bash tool; the tool's 30-minute default would stop it inside arm
5), with AC on, the lid open and the other sessions told. At the end the driver prints
`--summarize` itself; the last line reprints §2's inputs and rows from the committed files.

## 5. Results

One sequence ran on 2026-10-02, from worktree HEAD `3a324993`. The driver's `worktree_state` found
`src/` and `benchmarks/harness` clean. It stopped in its first position. Every number below is
quoted from the committed `throttle_*` files in `results/probe-rtx5070/two-drive/`. Times are
local (UTC+5).

### 5.1 How it ran, and how it stopped

- **Pre-flight, 21:11-21:12.**
  - `soup_cli.__file__` resolved to `Soup-sustain\src\soup_cli\__init__.py`.
  - `--selftest`: "all assertions passed".
  - `--check-files`: C: 84 files, 82 spans of 400 MiB (34.4 GB); D: 41 files, 40 spans (16.8 GB);
    both "ok".
  - `--dry-run` at 21:12:05, into a scratch folder: pre-arm check met. AC on, battery 45% and
    charging, commit headroom 26.94 GiB, GPU 0 MiB with no compute apps, 5 Python processes.
- **The run.** `--run-prefix throttle` was invoked at 21:12:32 as one background job with a
  2-hour limit (7,200,000 ms). Its stdout is `throttle_driver_stdout.log`.
- **Position 0, C-ALONE round 0, attempt 1.** The process ran 21:12:37-21:17:40 and read for
  300.0 s from 21:12:40. **Void**: `foreign reader: SearchIndexer.exe(26528) 2.37 GB`. The driver
  renamed its files `throttle_r0_c_void.json` / `.log`.
- **Attempt 2, the re-run in the same position.** The pre-arm check held at once, and the process
  ran 21:17:43-21:22:48, reading for 300.1 s from 21:17:46. No rest came before it; §1's 30 s rest
  comes before every *position* but the first. **Void**: `foreign reader: SearchIndexer.exe(26528)
  1.82 GB`. Its files are `throttle_r0_c_void2.json` / `.log`.
- **The stop.** A second void in one position stops the sequence (§4). The driver recorded
  `stopped: throttle_r0_c void twice; no verdict`, with all six positions not run, and exited with
  code 4 at 21:22:49. No D-ALONE or BOTH arm ran.
- **No second sequence.** A re-run under a new `--run-prefix` was not started. The brief for this
  window said to stop and record when no valid sequence came out of §4's own handling.
  `--summarize --run-prefix throttle` was run at 21:24 (`throttle_summarize.log`).

### 5.2 The two attempts (both void)

Both attempts read the one-root cache on C: (`pool_spans` 82, `pool_bytes` 34,393,292,800).
They passed every instrument condition of row 1:

- `pinned` True;
- only C: read;
- byte check OK;
- pool at least 16 GB;
- at least 285 s of reading;
- a best 5-s bin of at least 3.5 GB/s;
- PDH missing in 0 of 60 bins.

They failed only the environment condition on foreign readers.

| attempt | read from | read s | spans | passes | bytes | own mean GB/s | PDH mean GB/s | own 5-s bins min-max (median) | PDH 5-s bins min-max | dip bins | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (`_void`) | 21:12:40 | 300.0 | 2,786 | 33.98 | 1,168.5 GB | 3.895 | 3.894 | 3.645-4.042 (3.896) | 3.518-4.058 | 0 | 0 |
| 2 (`_void2`) | 21:17:46 | 300.1 | 2,781 | 33.91 | 1,166.4 GB | 3.888 | 3.877 | 2.559-4.272 (4.039) | 2.570-4.222 | 6 | 0 |

At 1 s:

- attempt 1: own 3.28-4.41 GB/s (median 3.88), PDH 3.31-4.56.
- attempt 2: own 2.16-4.61 (median 4.00), PDH 0.42-4.75.
- The two lowest PDH seconds are each attempt's first sample (3.31 and 0.42 at second 0, against
  3.67 and 3.81 by the reader's own clock).

### 5.3 Every 5-s bin, C: (GB/s, from the arm's first read)

Attempt 1, the reader's own rate:
3.645, 3.868, 3.849, 3.894, 3.929, 3.788, 4.002, 3.850, 4.020, 3.909, 3.935, 3.883, 3.864, 4.019,
3.809, 4.015, 3.802, 3.945, 3.895, 3.845, 3.928, 3.803, 3.997, 3.794, 3.939, 3.926, 3.874, 3.882,
3.830, 3.952, 3.809, 3.955, 3.932, 3.728, 3.913, 3.798, 3.918, 3.854, 3.964, 3.898, 3.910, 3.928,
3.863, 3.929, 3.807, 3.988, 3.885, 3.960, 3.839, 3.823, 3.969, 3.787, 4.008, 3.826, 3.991, 3.892,
3.916, 4.042, 3.888, 3.984

Attempt 1, PDH:
3.518, 3.902, 3.729, 3.983, 3.909, 3.859, 3.986, 3.823, 4.058, 3.824, 3.990, 3.898, 3.844, 4.036,
3.819, 4.010, 3.794, 3.957, 3.902, 3.863, 3.917, 3.814, 3.975, 3.807, 3.895, 3.980, 3.896, 3.813,
3.914, 3.897, 3.843, 3.948, 3.931, 3.698, 3.952, 3.818, 3.895, 3.862, 3.940, 3.853, 3.994, 3.925,
3.844, 3.942, 3.824, 3.918, 3.888, 3.997, 3.856, 3.812, 3.958, 3.840, 3.953, 3.834, 4.027, 3.828,
3.925, 4.035, 3.927, 3.963

Attempt 2, the reader's own rate (the bins below 3.0 in bold):
3.868, 3.982, 4.002, 3.982, 4.184, 4.073, 4.155, 4.125, 3.776, 4.053, 4.017, 4.130, 4.062, 4.077,
4.118, 3.963, 4.164, 3.968, 4.056, 4.135, 3.898, 4.116, 3.975, 4.123, 4.125, 3.941, 4.081, 3.911,
3.773, **2.559, 2.657**, 3.937, 4.187, 4.033, 4.081, 4.272, 3.959, 4.091, 3.988, 4.191, 4.070,
**2.648, 2.623**, 3.516, 3.914, 4.209, 4.144, 4.012, 4.079, 4.050, 4.019, 4.048, 3.441, **2.623,
2.638**, 4.040, 4.225, 4.037, 3.955, 4.149

Attempt 2, PDH:
3.186, 3.973, 3.938, 4.017, 4.195, 3.994, 4.212, 4.135, 3.891, 3.908, 4.050, 4.121, 4.074, 4.067,
4.134, 3.957, 4.133, 3.931, 4.117, 4.129, 3.861, 4.172, 3.952, 4.159, 4.102, 3.906, 4.073, 3.901,
3.880, **2.570, 2.658**, 3.895, 4.171, 4.004, 4.162, 4.143, 4.081, 4.030, 3.974, 4.222, 4.110,
**2.762, 2.592**, 3.423, 3.949, 4.156, 4.110, 4.071, 4.060, 4.029, 4.114, 3.912, 3.747, **2.637,
2.604**, 3.916, 4.203, 4.113, 3.892, 4.139

The 1-s values of both instruments are in `throttle_driver.json` (`runs[*].analysis.drives.c`).

### 5.4 Attempt 2's three short drops (descriptive; they set no row)

Attempt 2 has six dip bins, in three pairs: 145-155 s, 205-215 s and 265-275 s. Each pair is two
bins (10 s). An episode needs three (15 s), so the rule finds none.

| drop | 1-s seconds below 3.0 GB/s (own) | local | own 1-s min / mean | C: read continuously when it began | CPU in its 5-s bins | C: writes in its 5-s bins |
|---|---|---|---|---|---|---|
| 1 | 144-154 s (11 s) | 21:20:10-21:20:21 | 2.16 / 2.609 | 450.8 s | 3.2-19.1% | 0.0-11.9 MB/s |
| 2 | 205-215 s (11 s) | 21:21:11-21:21:22 | 2.56 / 2.638 | 511.8 s | 0.0-14.6% | 0.2-6.4 MB/s |
| 3 | 263-274 s (12 s) | 21:22:09-21:22:21 | 2.53 / 2.634 | 569.8 s | 2.6-18.0% | 0.0-9.2 MB/s |

- **Spacing.** The starts are 61 and 58 s apart.
- **Continuous reading.** C: was read from 21:12:40 (attempt 1's first read) with one gap of
  6.8 s, between attempt 1's last read and attempt 2's first. That gap is under the rule's 120 s,
  so the reading counts as continuous.
- **Other seconds.** A single second below 3.0 GB/s also fell at 44 s (2.93, 21:18:30), and
  attempt 1's lowest second was 3.28 GB/s at 166 s. Attempt 1, read from rest, had no bin below
  3.645 GB/s.
- **Against the sustained record, as numbers only.** There each stretch was 25-30 s long, 69-70 s
  apart, at 2.40-2.62 GB/s, on both drives at once, in STRIPED steps. Here, on C: read alone with no
  GPU work, each drop is 11-12 s long, about 60 s apart, at 2.53-2.70 GB/s (one second at 2.16).

### 5.5 Box state, the during-arm samples and the per-process capture

**Power, suspend and commit.**

- **AC** at every 2-s sample of both attempts.
- **The battery charged** from 46% to 54% (attempt 1) and from 54% to 61% (attempt 2).
- **No suspend.** The largest gap between power samples was 2.019 s in both attempts.
- **Commit headroom** was at least 22.473 GiB (attempt 1) and 24.426 GiB (attempt 2) at every
  sample.

**The stamps.** Every stamp read:

- GPU at 0 MiB with no compute apps;
- ASPM AC index 2;
- `soup_cli.__file__` under `Soup-sustain\src`.

The Python process count read 5 before attempt 1, 7 after it and before attempt 2, and 6 after
attempt 2. Which processes those were is not captured.

**PDH over each whole attempt.**

| attempt | C: read GB/s mean / max | D: read GB/s mean / max | C: write MB/s mean / max | CPU % mean / max | pages/s mean |
|---|---|---|---|---|---|
| 1 | 3.851 / 4.563 | 0.000 / 0.000 | 2.15 / 32.80 | 10.1 / 26.8 | 181 |
| 2 | 3.842 / 4.752 | 0.000 / 0.000 | 1.92 / 26.55 | 8.6 / 30.6 | 100 |

**GPU (10-s `nvidia-smi`).** This probe runs no compute; the reader holds only a CUDA context.

- attempt 1: SM 180-1417 MHz, 44-48 °C, 3.5-12.1 W;
- attempt 2: SM 0-2557 MHz, 46-49 °C, 3.6-12.8 W;
- throttle reasons `0x0000000000000004` or `0x0000000000000400` in every sample.

**Top readers, from the after stamps.** No capture failed. Each list is read / write bytes:

- **Attempt 1:**
  - `SearchIndexer.exe` (26528), 2,374.6 / 202.4 MB;
  - `codex.exe` (9972), 237.9 / 50.3 MB;
  - `Code.exe` (2748), 68.9 / 4.2 MB;
  - `svchost.exe` (16716), 66.3 / 0.0 MB;
  - `ChatGPT.exe` (25148), 49.5 / 0.7 MB;
  - `audiodg.exe` (36876, new), 36.6 / 0.0 MB;
  - `NVDisplay.Container.exe` (3872), 26.4 / 26.4 MB;
  - `svchost.exe` (3984), 13.3 / 1.5 MB.
- **Attempt 2:**
  - `SearchIndexer.exe` (26528), 1,821.5 / 155.3 MB;
  - `NVDisplay.Container.exe` (3872), 246.7 / 286.6 MB;
  - `codex.exe` (9972), 156.7 / 30.5 MB;
  - `svchost.exe` (16716), 122.9 / 0.0 MB;
  - `svchost.exe` (21332), 63.0 / 27.4 MB;
  - `MoUsoCoreWorker.exe` (23788, new), 47.5 / 2.6 MB;
  - `Code.exe` (2748), 15.3 / 0.8 MB;
  - `svchost.exe` (3984), 11.5 / 3.3 MB.

The full lists are in `throttle_box_state.log`. The capture names survivors only.

**The indexer, beyond the capture.** These are read-only observations, not part of any condition.

- `SearchIndexer.exe` (26528) had been running since 2026-10-01 18:11:55. At 21:18:08 its
  cumulative read count was 41.9 GB.
- Over 21:18:08-21:18:18, inside attempt 2, it read 0.1 MB, so its reads came in bursts.
- Its bytes are not visible in the volume counters. In attempt 1, C: PDH mean minus the reader's
  own mean is -0.001 GB/s, where 2.37 GB over 300 s would add 0.008 GB/s. In attempt 2 it is
  -0.011 GB/s. D: read 0.000 throughout.
- So what the per-process counter charged to the indexer did not appear as extra disk reads on C:
  or D: at this precision. It may have been served from the cache, or not been disk I/O at all;
  that is not established. The bar counts process read bytes, and it fired as written.

**System log, 21:10-21:23.** It holds no disk, storage, NTFS, thermal, power-source or sleep
event.

- One Kernel-Power 566 (a session unlock) at 21:18:20, inside attempt 2.
- One DCOM permission warning at 21:13:53, inside attempt 1.
- Windows Update activity during and just after attempt 2: the IsolationSession service's start
  type changed at 21:20:27 and 21:20:57, update downloads started at 21:21:21 (twice) and 21:21:55,
  a Store app update (WhatsApp) failed with 0x80073D02 at 21:21:35-21:21:44, and the Photos app
  update installed at 21:23:06-21:23:07, after attempt 2 ended.
- None of these falls at the start of a drop: they began at 21:20:10, 21:21:11 and 21:22:09.

**Activity of this probe's operator during the attempts, disclosed.** None of it is in either
capture.

- During attempt 1, a read-only analysis script (one short Python process) read about 0.9 MB of
  the sustained record's committed JSON files.
- Over 21:18:08-21:18:18, during attempt 2, one PowerShell query read the indexer's counters.
- A monitor read the driver's stdout log every 30 s.

Also running, from other sessions: an idle watcher script, a merge-polling loop and a short
GitHub-posting script (network only), and two editor language servers.

### 5.6 Anomalies, as measured

1. **Both voids came from Windows Search's indexer**, which read 2.37 GB and 1.82 GB in bursts
   during the two attempts. Those reads are not visible as disk reads on C: or D: (§5.5). No such
   reader appeared in the sustained record's arms, whose largest was 834.7 MB (`chrome.exe`, in
   the build).
2. **Attempt 2's three 11-12 s drops on C: alone**, at 2.53-2.70 GB/s and about 60 s apart. They
   began after 7.5 minutes of continuous reading. The rested attempt 1 had none. They are too short
   to be episodes, and both arms are void (§5.4).
3. **Outside its drops, attempt 2 read faster than attempt 1**: median 5-s bin 4.039 against
   3.896 GB/s.

## 6. Verdict

**NO VERDICT, C: and D: (row 1).** The first position was void twice, and the sequence stopped
there.

| row | inputs | result |
|---|---|---|
| 1: a validity condition fails, a position has no valid arm, or the driver did not finish | C-ALONE round 0 was void twice: the foreign-reader condition, `SearchIndexer.exe` (26528) at 2.37 GB and 1.82 GB against the 1 GB bar. The driver stopped (`throttle_r0_c void twice; no verdict`). `--summarize`: "no valid arm: ['r0 c', 'r0 d', 'r0 both', 'r1 both', 'r1 d', 'r1 c']; protocol: stopped: throttle_r0_c void twice; no verdict" | **fires: NO VERDICT** |
| 2-5 | none | not reached |

**What the void attempts show. This is NOT a verdict and cannot stand in for one.**

- Read from rest for 300 s, C: had no dip.
- Read again straight after, C: dropped to about 2.6 GB/s for 11-12 s, three times, about once a
  minute. The drops began 7.5 minutes into the continuous reading.
- Under the rule those drops are not episodes (two bins, not three).
- D: was never read alone or with C:.

A verdict needs a complete sequence: a new `--run-prefix` (`throttle2`), on AC with the lid open,
and the foreign-reader condition met in every arm. Whether that means pausing Windows Search or
excluding the cache folders from it during the run is a change to the box, not to the rule. It is
the owner's call.
