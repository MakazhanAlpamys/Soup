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

**Status: rule committed before the run; results pending.** No arm has run. The reader and driver
have been exercised only on synthetic arms (`--selftest`) and in two dry runs that read no cache
file: 2026-10-01 (pre-arm check not met, another session held 4746 MiB of GPU memory) and
2026-10-02 (not met, the laptop was on battery at 74%). The only look at the cache folders was the
listing of §1 (names and sizes).

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

Pending. No arm has run.

## 6. Verdict

Pending.
