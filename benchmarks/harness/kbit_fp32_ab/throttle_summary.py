"""Throttle reasons during the busy part (power > 300 W) of each T cell.  Reads raw/box/smi_throttle.csv (nvidia-smi -l 2 on the box,
columns: timestamp UTC, SM clock, power, temperature, clocks_throttle_reasons.active bitmask) and raw/box/cell_times.tsv."""
import collections
import datetime
import statistics
import sys

box = sys.argv[1]
rows = []
for line in open(f"{box}/smi_throttle.csv", encoding="utf-8"):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 5:
        continue
    try:
        ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
        rows.append((ts, float(parts[1].split()[0]), float(parts[2].split()[0]), float(parts[3]), int(parts[4], 16)))
    except ValueError:
        continue
BITS = {0x1: "gpu_idle", 0x2: "app_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
print(f"{len(rows)} samples; busy = power > 300 W inside the cell's start..end")
print("cell | busy samples | median SM clock MHz | median power W | share of busy samples with each throttle reason bit")
for line in open(f"{box}/cell_times.tsv", encoding="utf-8"):
    cell, t0, t1, _rc, _rest = line.rstrip("\n").split("\t")
    if not cell.endswith("-T"):
        continue
    busy = [r for r in rows if float(t0) <= r[0] <= float(t1) and r[2] > 300]
    if not busy:
        continue
    count = collections.Counter(name for r in busy for bit, name in BITS.items() if r[4] & bit)
    shares = {k: round(v / len(busy), 2) for k, v in count.items()} or "none"
    print(f"{cell} | {len(busy)} | {statistics.median(r[1] for r in busy):.0f} | {statistics.median(r[2] for r in busy):.0f} | {shares}")
