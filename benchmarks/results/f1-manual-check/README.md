# F1 manual check: RTX 3050, real

Four arm records from the manual check in `benchmarks/METHOD.md`, run on
2026-10-12 on the RTX 3050 laptop. Each one makes a real event happen while
`ArmWatch` watches the arm. Every record has the label "RTX 3050, real".

| File | Event during the arm | Arm length | Outcome | Check that caught it |
|---|---|---|---|---|
| `clean.json` | none, on mains, machine left alone | 60 s | ok | none; all three checks ok |
| `sleep.json` | `systemctl suspend`, asleep about 29 s | 120 s awake | void | `suspend`: boot-time offset grew 29.064 s; suspend count rose from 0 to 1 |
| `battery.json` | charger unplugged, then plugged back in | 60 s | void | `power`: 8 of 32 power samples off mains, about 16 s |
| `foreign-read.json` | `dd` read a 2 GiB file with `iflag=direct` | 60 s | void | `foreign_reads`: at least 2,147,487,744 bytes on `nvme0n1` |

In every void record the other two checks are ok, so each event voided its
own check only.

The foreign-read record names the reader `zsh`, not `dd`. The `dd` ran in a
subshell and exited inside the arm, and Linux adds a reaped child's read
counter to its parent's. METHOD.md describes this under "Readers and
unattributed bytes".

Every record has the fields added after the first review: `gpu` (GPU 0, from
the default, because `CUDA_VISIBLE_DEVICES` was unset), `commit_charge` in
both box stamps, and the `foreign_read_bytes_min` and `foreign_read_bytes_max`
bounds. On this laptop the other disks read 0 bytes, since `nvme0n1` is the
only physical disk, so the two bounds meet and `foreign_read_bytes` is exact.
The bounds rule for several disks ran on synthetic counters only.

## Machine and versions

| Item | Value |
|---|---|
| Laptop | ASUS TUF Gaming F15 FX507ZC4 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, driver 595.99.02, power limit 80 W |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic |
| Python | 3.14.4, the system `python3`, no venv |
| systemd | 259 (259.5-0ubuntu3.4) |
| dd | uutils coreutils 0.10.0 |
| Code | `benchmarks/harness/arm_validity.py` at commit `a24acc1d` |
| Disk under test | `nvme0n1`, holding both `~/.cache/huggingface/hub` and `$HOME` |

## Commands

Setup, from the repository root:

```bash
export PYTHONPATH="$PWD/benchmarks/harness"
MODELS=~/.cache/huggingface/hub
LABEL="RTX 3050, real"
OUT=benchmarks/results/f1-manual-check
mkdir -p "$OUT"
python3 -c "from arm_validity import disk_device; import os, sys; print(disk_device(os.path.expanduser('~')), disk_device(sys.argv[1]))" "$MODELS"
# nvme0n1 nvme0n1
```

`/tmp/watch_arm.py` is the script from the manual check in METHOD.md, unchanged.

Clean, started 18:57:59 +05:

```bash
python3 /tmp/watch_arm.py clean 60 "$LABEL" "$MODELS" > "$OUT/clean.json"
```

Foreign read, started 18:59:11 +05. The test file was written and synced
before the arm started, and deleted after it ended:

```bash
dd if=/dev/urandom of=$HOME/f1-read-test.bin bs=1M count=2048 conv=fsync
python3 /tmp/watch_arm.py foreign-read 60 "$LABEL" "$MODELS" > "$OUT/foreign-read.json" &
sleep 10; dd if=$HOME/f1-read-test.bin of=/dev/null bs=1M iflag=direct; wait
rm $HOME/f1-read-test.bin
```

Battery, started 19:02:44 +05. A person unplugged the charger about 10 s
after the arm started and plugged it back in about 16 s later:

```bash
python3 /tmp/watch_arm.py battery 60 "$LABEL" "$MODELS" > "$OUT/battery.json"
```

Sleep, started 19:03:51 +05. The arm ran in the background, and a person ran
`systemctl suspend` in the same session right after, then woke the laptop
about 29 s later. The arm's 120 s timer does not count the time asleep, so the
record spans about 149 s of wall-clock time:

```bash
python3 /tmp/watch_arm.py sleep 120 "$LABEL" "$MODELS" > "$OUT/sleep.json" &
systemctl suspend
wait
```

## Records not kept

The first set of four records, run on 2026-10-09 with the code at `64724a82`,
was replaced after the PR 3 review changed the record shape. It is in git
history at `23bfe7fb`. That set had a 34 s suspend and an 18 s unplug.

A first sleep arm of 300 s ran with no suspend in it and came back ok. It
was not a sleep record, so it is not here.

## Not run

The Windows path was not run on Windows. The test suite drives it with mocks
only.
