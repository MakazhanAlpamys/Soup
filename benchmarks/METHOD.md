# Benchmark method: validity checks and repeats

This page is for anyone who writes a benchmark rule or runs its arms. It says
which checks watch each arm, what their outcomes mean, how a rule commits its
validity rows, how many blocks a comparison needs, and how to check the checks
on your own laptop.

The code is in `harness/arm_validity.py`. `harness/f1_pilot.py` runs a pilot
through it. Both are benchmark harness code and are not in the shipped wheel.

## Before the first arm

1. Run the manual check below once on the machine, at least the clean arm. If a
   check comes back unknown there, it will come back unknown in every arm.
2. Write the decision rule with its validity rows and required checks, and
   commit it.
3. Plug in the charger and close programs that read the disk under test.
4. Run the first 5 blocks as the pilot, then follow `blocks_to_add` until it
   returns 0.
5. Keep every arm record, void ones included.

## Words

These terms have one meaning each in `benchmarks/`.

| Term | Meaning |
|---|---|
| arm | One variant under comparison, identified by everything that distinguishes it from the other variants. A and B below stand for the two arms of a comparison. |
| run | One execution of one arm in a fresh process. Its number is the median of its timed steps. |
| round | One pass over every arm in a fixed order. Consecutive rounds reverse the order. |
| block | Four runs in the order A, B, B, A, so two rounds. It yields one difference between the means of A and B. |
| pilot | The first 5 blocks of a comparison. They estimate the run-to-run spread and count toward the final test. |
| box stamp | A snapshot of the machine taken before and after an arm: time, free memory, commit charge, power source, GPU peers and GPU memory in use. GPU peers are the processes on the GPU. |
| foreign reads | Bytes read from the disk under test during an arm by anything other than the benchmark's own process and its children. |
| check outcome | ok, void or unknown, for one check of one arm. |
| void arm | An arm with a void check. Its number is not used. Void means unknown, not slow. |
| validity row | A condition, committed with the rule before the first arm, that checks whether the measurement measured what the rule reads. |
| no verdict | The outcome of a gate that depends on a void arm, or on an arm whose required check is unknown. It is distinct from fail. |

## The three checks

`ArmWatch` runs three checks on every arm. Each one can void the arm.

| Check | Void when | Linux sensor | Windows sensor | macOS |
|---|---|---|---|---|
| `suspend` | the machine slept during the arm | boot-time minus monotonic time grows by more than 1 s between the start and end of the arm, or the kernel's suspend counter rises | the wall clock, sampled every 2 s, jumps more than 30 s between two samples | unknown |
| `power` | one power sample shows the machine off mains | `/sys/class/power_supply`, sampled every 2 s | `GetSystemPowerStatus`, sampled every 2 s | unknown |
| `foreign_reads` | foreign reads reach 1 GB, which is 1,000,000,000 bytes | the disk-wide read counter in `/proc/diskstats` minus the reads of this process and its children | per-process read counters through psutil; one other process reading 1 GB or more | unknown |

### Sleep

On Linux, `CLOCK_BOOTTIME` keeps counting while the machine is suspended and
`CLOCK_MONOTONIC` does not. Their difference grows by the time spent asleep.
The check voids the arm if the difference grows by more than 1 second between
the start and the end of the arm, or if the kernel's suspend counter rises.
Either sensor seeing sleep is enough. Linux has no wall-clock gap rule, so an
NTP clock step cannot void an arm.

On Windows the check samples the wall clock every 2 s and voids the arm if two
samples are more than 30 s apart. This is the rule of `l2l_box.SuspendWatch` in
upstream Soup's `benchmarks/harness`. It misses a suspend shorter than about
30 s.

### Power

On Linux the check reads every power supply in `/sys/class/power_supply` that
is not a battery and whose `scope` is not `Device`, which skips a wireless
mouse battery, for example. The machine is on mains if any of them has an
`online` value other than 0. This follows the kernel's
`power_supply_is_system_supplied()`, with one difference. A machine with no
battery is on mains whatever its supplies say, so a desktop or cloud machine
gets ok with the reason "no battery".

The check reads the supplies at the start of the arm, every 2 s, and at the
end. One sample off mains voids the arm. An unplug shorter than 2 s can fall
between two samples.

On Windows the check reads the AC line from `GetSystemPowerStatus`. An unknown
AC line on a machine that may have a battery makes the sample unknown.

### Foreign reads

On Linux the check finds the disk that holds the path you pass as
`model_path`. It reads that disk's sector count from `/proc/diskstats` at the
start and end of the arm and multiplies it by 512. It does the same for every
other physical disk, summed. It also reads the bytes read by the watching
process and every process below it, from `/proc/<pid>/io`.

That process counter covers every disk, so the check cannot tell which disk
your own reads hit. It bounds the foreign reads on the watched disk instead:

- at most: the watched disk's reads minus the part of your own reads the other
  disks did not take;
- at least: the watched disk's reads minus all of your own reads.

The minimum is `foreign_read_bytes_min` and the maximum is
`foreign_read_bytes_max`. `foreign_read_bytes` holds the exact figure when the
two meet, else `null`.

| Bound | Outcome |
|---|---|
| minimum is 1 GB or more | void |
| maximum is under 1 GB | ok |
| the limit falls between them | unknown, with the bounds in the reason |

With 2 GB of your own reads on another disk and 1.5 GB read on the watched
disk, the minimum is 0 and the maximum is 1.5 GB, so the check is unknown. An
earlier version subtracted all 2 GB, got -0.5 GB and reported ok.

Run the workload inside the watching process or as its child, and wait for
each child to exit before the arm ends. Otherwise its reads count as foreign.
`f1_pilot.py` starts each run with `subprocess.run` inside the watch, which
does both.

The check is unknown when:

- no `model_path` is given;
- the path's file system has no block device, as on tmpfs, a btrfs subvolume
  or a network mount;
- the path spans several disks, as a RAID or LVM volume over two drives does;
- the disk is missing from `/proc/diskstats`;
- the bounds leave the 1 GB limit between them, as above;
- a child of the watching process has exited but nobody has waited for it.
  Linux denies its counter until then.

On Windows there is no disk-wide counter. The check reads each process's read
counter through psutil and voids the arm if one other process read 1 GB or
more. A process your user may not inspect goes unseen, and Windows counts a
process's reads from every disk, not only the disk under test. Without psutil
installed, the check is unknown with the reason "psutil not installed".
Windows records have no disk-wide total, so `unattributed_bytes` is always
`null` there.

## Root is never required

On Linux every sensor above is readable by a normal user: `/proc/diskstats`,
`/sys/power/suspend_stats/success`, `/sys/class/power_supply/*` and your own
processes' `/proc/<pid>/io`. `nvidia-smi` queries need no root either.

Without root, Linux denies `/proc/<pid>/io` of other users' processes, so the
module cannot name them. On the RTX 3050 laptop that was 283 to 318 processes
per arm across 40 runs, among them every root service. The record counts them in
`evidence.uninspectable_processes`. Their bytes still show up in the
disk-wide counter, so they still count toward the 1 GB limit.

## Readers and unattributed bytes

`foreign_reads` carries two extra fields.

- `readers` lists your own user's other processes that read during the arm,
  with their bytes, most first.
- `unattributed_bytes` is the foreign reads that `readers` does not explain.
  It is the foreign bytes minus the readers' bytes, floored at 0.

Unattributed bytes come from root services, other users' processes, the
kernel, and processes of your user that started and exited inside the arm
without a parent to inherit their counter. The floor is there because a
reader's counter covers every disk, so its bytes can exceed what the disk
under test saw.

A reader that exits during the arm can show up under its parent's name. Linux
adds a reaped child's counter to its parent's. When this page was written, a
`dd` started from a shell read 2 GB during a test arm, and the record named
the reader `zsh`. That arm's record was not kept.

Unattributed bytes do not change the outcome. The disk-wide counter decides.
`readers` and `unattributed_bytes` tell you where to look.

## Check outcomes

Every check ends in one of three outcomes, each with a `reason` and the
`evidence` the sensor read.

| Outcome | Meaning |
|---|---|
| `ok` | The sensor answered and saw nothing wrong. |
| `void` | The sensor answered and the arm is contaminated. |
| `unknown` | The sensor gave no answer. The reason says why. |

A sensor that raises or returns nothing makes its check unknown, never ok.
Unknown is not clean. If it were, a whole arm on battery whose power sensor
said nothing would pass as a clean measurement.

The arm record has one overall `outcome`:

1. `void` if any check is void, required or not;
2. otherwise `unknown` if any required check is unknown;
3. otherwise `ok`.

A gate whose required check is void or unknown on an arm it reads gets no
verdict. No verdict is not fail. It says the measurement cannot answer the
question, not that the code is slow.

### What else the record holds

Each arm record also holds `arm`, `round`, `run`, `label`, `gpu`, `required`,
the box stamps `box_stamp_before` and `box_stamp_after`, and `gpu_clock`. The
box stamps and the GPU clock record never void an arm. A field whose sensor
gave nothing is `null`, with the reason under the stamp's `unknown`.

`commit_charge` in a box stamp is `{"used_bytes", "limit_bytes"}`: the memory
the OS has promised to processes, and the most it will promise. Free RAM alone
misses a machine that is out of commit while RAM looks free, where allocations
start to fail or the pagefile grows. On Linux the values are `Committed_AS` and
`CommitLimit` from `/proc/meminfo`. On Windows the limit is `ullTotalPageFile`
from `GlobalMemoryStatusEx` and the charge is that limit minus
`ullAvailPageFile`. macOS is unknown. Linux does not enforce `CommitLimit` under
the default overcommit mode, so `used_bytes` can exceed `limit_bytes` there. The
RTX 3050 laptop records 16.2 GB committed against a 12.2 GB limit. Read the two
numbers as a trend between the stamps, not as a headroom figure.

`gpu` names the GPU every `nvidia-smi` call reads: `{"id": "1", "source":
"CUDA_VISIBLE_DEVICES"}`. The workload's `cuda:0` is the first entry of
`CUDA_VISIBLE_DEVICES`, an index or a UUID, so the sensors read that GPU. With
the variable unset they read GPU 0 and `source` is `default`. A variable that
leaves no GPU (`""`, `-1`, `NoDevFiles`) makes `id` null and every GPU field
unknown. Set `CUDA_VISIBLE_DEVICES` in the environment of the process that
runs the watch, before it starts. On a machine with different GPU models,
name the GPU by UUID, because CUDA and `nvidia-smi` can number them differently.

`gpu_clock` samples the SM clock and the active clock event reasons through
`nvidia-smi` every 2 s. It stores the clock's min, median and max, and how
many samples saw each reason, such as `sw_power_cap`, `sw_thermal_slowdown`
or `gpu_idle`. Without `nvidia-smi` it is unknown.

Read the GPU clock record, not only the outcome. On 2026-10-09 the RTX 3050
laptop ran a pilot while its GPU was held at 210 MHz under a 10 W power limit
on mains. Every check of every run was ok. The step took about 0.73 s against
0.069 s after a reboot cleared the limit. Only the GPU clock record showed it.
The median SM clock was 210 MHz, and `sw_power_cap` and `sw_thermal_slowdown`
were active in every sample. That pilot is not used for anything else.

## Committing validity rows and required checks

The module reports outcomes. The rule decides which of them gate a verdict.
Write that decision down before the first arm, in the gate's markdown file,
next to the rest of the decision rule. Each validity row names the check it
reads and whether that check is required.

```markdown
## The decision rule, written before the run

| validity row | check | required | if it fails |
|---|---|---|---|
| `suspend` is ok on every arm the verdict reads | `suspend` | yes | no verdict |
| `power` is ok on every arm the verdict reads | `power` | yes | no verdict |
| `foreign_reads` is ok on every arm the verdict reads | `foreign_reads` | yes | no verdict |
| ... the rule's own verdict rows ... | | | |
```

A row fails when its check is void or unknown on an arm the verdict reads.
The repeats rule below replaces a block with a void run before the verdict.
So a row usually fails on an unknown check. It fails on a void check only
when a void block was not replaced.

Pass the same required checks to every `ArmWatch`. The snippets on this page
need `benchmarks/harness` on `PYTHONPATH`.

```python
from arm_validity import ArmWatch

REQUIRED = ("suspend", "power", "foreign_reads")  # copied from the committed rule

with ArmWatch(arm="A", round=1, run=1, label="RTX 3050, real",
              required=REQUIRED, model_path=models) as watch:
    run_arm("A")
record = watch.record
```

`required` defaults to all three checks. Drop a check from it only when the
committed rule says why, for example a model on a network mount where
`foreign_reads` can only be unknown. A void check voids the arm whether it is
required or not.

Commit the rule before the first arm, and do not edit it after. A reviewer can
confirm the order by comparing the commit time with `box_stamp_before.unix_s`
of the first arm record.

## How many blocks: the repeats rule

A comparison of A against B runs in blocks of A, B, B, A, each run a fresh
process. Within a block, linear drift, such as a GPU warming up, adds the same
amount to the mean of A and the mean of B, so it cancels in their
difference. `block_order(count)` gives the order, so no script writes it by
hand.

The number of blocks to detect a 10% effect is

```
N = ceil(7.85 * (sigma / delta)^2),   at least 2
```

- `7.85` is (1.960 + 0.842)^2, the two z values for alpha 0.05 two-sided and
  power 0.8 in the sample size formula of the NIST/SEMATECH e-Handbook of
  Statistical Methods.
- `delta` is 10% of the mean of the A runs.
- `sigma` is the run-to-run spread within an arm, pooled over A and B:
  `sqrt((variance of the A runs + variance of the B runs) / 2)`.

Sigma for planning comes from a pilot on the machine being measured. Never
take it from another machine, another card, or a run on battery.

The procedure:

1. Run 5 blocks. These are the pilot, and they count toward N.
2. Compute N from the pilot and run blocks until there are N.
3. Recompute N once from all blocks. If it is larger, run blocks up to it.
4. Stop. N is not recomputed again, so the rule cannot keep adding blocks
   until the result looks right.

A block with a void run does not count. Keep its records and run a
replacement. A run whose outcome is unknown still counts toward N; whether the
gate gets a verdict is up to the required checks.

`blocks_to_add(blocks)` returns how many blocks are still due under these steps.
It skips blocks with a void run, so each one adds a replacement. It never asks
for fewer than the 5 pilot blocks, even when N is smaller. `repeats(blocks)`
returns sigma, the mean of A and N. A loop that runs one block at a time:

```python
from arm_validity import ArmWatch, Run, block_order, blocks_to_add, repeats

blocks, records = [], []
while blocks_to_add(blocks) > 0:
    block = []
    for position, arm in enumerate(block_order(1)[0]):
        round, run = 2 * len(blocks) + 1 + position // 2, 4 * len(blocks) + position + 1
        with ArmWatch(arm=arm, round=round, run=run, label=LABEL,
                      required=REQUIRED, model_path=models) as watch:
            value = run_arm(arm)  # a fresh process; the median of its timed steps
        records.append(watch.record)
        block.append(Run(value, watch.record["outcome"]))
    blocks.append(block)
print(repeats(blocks))
```

The design simulation tested the mean block difference, the mean of A minus the
mean of B in each block, with a two-sided t-test at alpha 0.05. A rule that uses
a different test should say so.

### Why a counted pilot and one top-up

A bootstrap simulation on the 13 gate-836 runs below, with 20,000 experiments
per condition and seed 1234, gave this rule 84% power and a 5.3% false-positive
rate at a median of 63 blocks. A 2-block pilot that is thrown away gave 72%
power. A thrown-away pilot of 4 to 8 blocks gave at most 77%.

### Additive noise

The only run-to-run spread data in the repo is `gate-836-bench-train-contract.md`
(RTX 5070 Laptop, on battery, GPU capped at 50 W). It repeated `soup bench
train` with the same config on four shapes. Its section 6 gives the fastest and
slowest median step of each, in seconds:

| shape | steps/warmup | fastest | slowest | ratio | difference |
|---|---|---:|---:|---:|---:|
| A (13 runs) | 12/4 | 0.1972 | 0.4746 | 2.41x | 0.2774 |
| A (3 runs) | 60/20 | 0.1965 | 0.4401 | 2.24x | 0.2436 |
| B (3 runs) | 12/4 | 5.2596 | 5.5039 | 1.046x | 0.2443 |
| C (3 runs) | 12/4 | 0.2045 | 0.3750 | 1.83x | 0.1705 |
| D (3 runs) | 200/40 | 0.2036 | 0.3854 | 1.89x | 0.1818 |

The ratio runs from 1.05x to 2.41x. The difference stays between 0.17 s and
0.28 s, while the fast-mode step time differs by 26x between shape B and the
rest. The noise adds a roughly fixed amount of time per step, now and then,
instead of scaling with the step. A longer window does not average it away:
shape D counted 160 steps and still spread 1.89x.

Three rules follow for a benchmark author:

- State noise in seconds per step, from your own pilot, not as a percentage.
  Shape B's 4.6% spread and shape A's 2.41x spread are the same 0.24 s on steps
  of 5.3 s and 0.2 s.
- Compare the effect you want to detect with the noise. The repeats rule sets
  delta to 10% of the mean of A, so delta shrinks with the step. An effect much
  smaller than the additive noise needs many blocks, since N grows with
  (sigma / delta)^2. An effect much larger needs few.
- Do not subtract or drop slow runs by hand. The record names no cause: its
  section 7 tested two explanations, another session on the box and CPU
  frequency scaling on battery, and refuted both. The validity checks void an arm only for the
  causes they watch.

These sizes belong to one card on battery. The RTX 3050 laptop on mains gave a
sigma of 0.00054 s in its own pilot (see the worked example). Run the pilot on
the machine you measure.

### Worked example: gate-836

This example shows the arithmetic only. Do not plan runs with its numbers.

`gate-836-bench-train-contract.md` ran shape A 13 times with the same config, 12
steps with 4 warm-up steps each. The medians of their timed steps were 0.1972 to
0.4746 s. Those 13 runs give:

- sigma = 0.1027 s, their standard deviation. With A and B the same config,
  this is what the pooled sigma estimates;
- mean = 0.3646 s, so delta = 0.03646 s;
- 7.85 * (0.1027 / 0.03646)^2 = 62.3, so N = 63 blocks.

Those runs were on an RTX 5070 Laptop on battery with the GPU capped at 50 W.
Nothing measured on battery counts, and numbers from one card are never
carried to another, so this sigma plans nothing.

For contrast, the RTX 3050 laptop ran its own pilot on mains on 2026-10-12, with
the same shape A config. Its record is in `results/f1-pilot/`. It gave sigma
0.00054 s and a mean of A of 0.0688 s. The formula gives 1 block, so N is the
minimum of 2, and the comparison stops after its 5 pilot blocks. That holds for
that config on that machine only.

To measure your machine's spread before a comparison, run `f1_pilot.py` with
the A config. It runs `soup bench train` with the same config for
A and B, so any difference between them is noise:

```bash
python benchmarks/harness/f1_pilot.py --config bench.yaml \
    --output /tmp/my-pilot/pilot.json
```

The script writes its report and a `runs` directory next to `--output`. Do
not point it at `results/f1-pilot/`, which holds the RTX 3050 record. In the
comparison itself, the pilot is the comparison's own first 5 blocks.

## Manual check on your laptop

The automated tests drive the checks with fake sensors. This procedure makes
each event happen for real and shows the check catching it. It takes about
15 minutes. You need a laptop with a battery and a terminal you can open twice.

Set up once, from the repository root:

```bash
export PYTHONPATH="$PWD/benchmarks/harness"
MODELS=~/.cache/huggingface/hub          # the disk under test holds this path
LABEL="RTX 3050, real"                   # your card, and "real"
OUT=benchmarks/results/f1-manual-check
mkdir -p "$OUT"
cat > /tmp/watch_arm.py <<'EOF'
import json
import sys
import time

from arm_validity import ArmWatch

name, seconds, label, model_path = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
with ArmWatch(arm=name, round=1, run=1, label=label, model_path=model_path) as watch:
    print(f"arm {name} running for {seconds:.0f} s", file=sys.stderr)
    time.sleep(seconds)
print(json.dumps(watch.record, indent=2))
print(f"arm {name}: {watch.record['outcome']}", file=sys.stderr)
EOF
```

Each arm prints its outcome when it ends. Run them one at a time, on mains
unless a step says otherwise.

1. **Clean.** Run `python3 /tmp/watch_arm.py clean 60 "$LABEL" "$MODELS" > "$OUT/clean.json"`
   and leave the machine alone. Expect `ok`. If a check is unknown, its
   reason says which sensor is missing.
2. **Sleep.** Run `python3 /tmp/watch_arm.py sleep 120 "$LABEL" "$MODELS" > "$OUT/sleep.json"`.
   In a second terminal run `systemctl suspend`, wait about 10 s, and wake
   the machine. Expect `checks.suspend` void.
3. **Battery.** Run `python3 /tmp/watch_arm.py battery 60 "$LABEL" "$MODELS" > "$OUT/battery.json"`.
   Unplug the charger for at least 10 s, then plug it back in. Expect
   `checks.power` void.
4. **Foreign read.** First make a 2 GB file on the disk under test:
   `dd if=/dev/urandom of=$HOME/f1-read-test.bin bs=1M count=2048 conv=fsync`.
   Then run `python3 /tmp/watch_arm.py foreign-read 60 "$LABEL" "$MODELS" > "$OUT/foreign-read.json"`,
   and in a second terminal read the file with direct I/O:
   `dd if=$HOME/f1-read-test.bin of=/dev/null bs=1M iflag=direct`.
   Expect `checks.foreign_reads` void with about 2.1 GB of foreign reads.
   `iflag=direct` bypasses the page cache, so every byte comes from the disk.
5. Delete the file: `rm $HOME/f1-read-test.bin`.

For step 4, check that `$HOME` and `$MODELS` are on the same disk:

```bash
python3 -c "from arm_validity import disk_device; import os, sys; print(disk_device(os.path.expanduser('~')), disk_device(sys.argv[1]))" "$MODELS"
```

Both names must match, `nvme0n1` for example.

## Not covered

- macOS. Every check is unknown with the reason "not implemented on macOS".
- Windows hardware. The test suite drives the Windows path with mocks only.
  Nobody has run it on a Windows machine.
- Lid state, and any event other than sleep, power and foreign reads. None of
  them voids an arm.
- Hibernation, a VM paused by its host, and containers. Nobody has tested the
  sleep and disk sensors in these cases.
- GPU throttling. The GPU clock record shows it but never voids an arm.
