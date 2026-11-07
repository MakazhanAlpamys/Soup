"""RAM watchdog for one arm of ``benchmarks/probe-rtx5070-moonlight-fp8-source.md``.

Light on purpose (psutil and ``nvidia-smi`` only; it never imports torch). Start it before the
run, in its own window; it waits for the training process, follows it to its exit and then
exits itself:

    python benchmarks/harness/fp8_loss_check_watchdog.py \
        --match D:/soup-bench/fp8-loss-check/bf16/soup.yaml --out D:/soup-bench/fp8-loss-check/bf16

The training process is the ``python*`` process whose command line contains ``--match``
(compared case-insensitively, with ``\\`` read as ``/``) and the word ``train``; ``--pid``
names it directly instead. Every ``--period`` seconds (5) it appends one JSON line to
``<out>/watchdog.jsonl``: available RAM, the process tree's resident set, the process's peak
working set (Windows), the GPU's memory in use, utilisation, temperature, SM clock and power
from ``nvidia-smi``, the PIDs ``nvidia-smi`` lists as compute processes, whether the box is on
AC, and the free space of the ``--disk`` volume.

If available RAM falls below ``--min-free-gb`` (decimal GB, 1.5), it stops the process tree
(terminate, then kill after 10 s), records that it fired, and exits 3: the arm is then
INCONCLUSIVE, not a failure. A gap of more than ``--gap-s`` (30) between two samples is
recorded (the laptop slept), and so is any sample off AC. ``<out>/watchdog.json`` holds the
summary the record reads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

GB = 1e9
MIB = 2**20
NVIDIA_FIELDS = (
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "clocks.sm",
    "power.draw",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _norm(text: str) -> str:
    return text.replace("\\", "/").lower()


def _nvidia() -> dict:
    """One ``nvidia-smi`` reading; the error text instead of numbers when it fails."""
    reading: dict = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(NVIDIA_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        for field, value in zip(NVIDIA_FIELDS, (part.strip() for part in out.split(","))):
            try:
                reading[field] = float(value)
            except ValueError:
                reading[field] = value
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        reading["compute_pids"] = sorted(
            int(line) for line in (part.strip() for part in apps.splitlines()) if line.isdigit()
        )
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        reading["error"] = f"{type(exc).__name__}: {exc}"
    return reading


def _find(psutil, match: str, own_pid: int):
    """The root python process whose command line names ``match`` and ``train``."""
    needle = _norm(match)
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == own_pid:
                continue
            name = (proc.info["name"] or "").lower()
            cmdline = proc.info["cmdline"] or []
            joined = _norm(" ".join(cmdline))
            if name.startswith("python") and needle in joined and "train" in cmdline:
                found.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    pids = {proc.pid for proc in found}
    roots = [proc for proc in found if proc.ppid() not in pids]
    return roots[0] if len(roots) == 1 else None


def _tree_rss(psutil, proc) -> int:
    total = 0
    for member in [proc, *proc.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _stop_tree(psutil, proc) -> list:
    members = [*proc.children(recursive=True), proc]
    for member in members:
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(members, timeout=10)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            continue
    return [member.pid for member in members]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--match", help="a string in the training process's command line")
    target.add_argument("--pid", type=int)
    parser.add_argument("--out", required=True, help="the arm's folder")
    parser.add_argument("--min-free-gb", type=float, default=1.5)
    parser.add_argument("--period", type=float, default=5.0)
    parser.add_argument("--gap-s", type=float, default=30.0)
    parser.add_argument("--wait-s", type=float, default=900.0, help="for the process to start")
    parser.add_argument("--disk", default="D:/")
    args = parser.parse_args(argv)

    import psutil

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "watchdog.jsonl")
    summary_path = os.path.join(args.out, "watchdog.json")
    summary: dict = {
        "started": _now(),
        "match": args.match,
        "min_free_gb": args.min_free_gb,
        "period_s": args.period,
        "fired": False,
        "gaps": [],
        "off_ac_samples": 0,
        "foreign_compute_pids": [],
    }

    deadline = time.monotonic() + args.wait_s
    proc = None
    while proc is None:
        if args.pid is not None:
            try:
                proc = psutil.Process(args.pid)
            except psutil.NoSuchProcess:
                proc = None
        else:
            proc = _find(psutil, args.match, os.getpid())
        if proc is None:
            if time.monotonic() > deadline:
                summary["error"] = "training process not found"
                with open(summary_path, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
                print(summary["error"])
                return 2
            time.sleep(1.0)
    summary["pid"] = proc.pid
    summary["cmdline"] = proc.cmdline()
    print(f"watching pid {proc.pid}: {' '.join(proc.cmdline())}")

    samples = 0
    min_free = None
    max_gpu_mib = None
    max_temp = None
    max_tree_rss = 0
    peak_wset = 0
    last = time.monotonic()
    exit_code = 0
    with open(log_path, "a", encoding="utf-8") as log:
        while True:
            try:
                running = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                running = False
            if not running:
                break
            tick = time.monotonic()
            if tick - last > args.gap_s:
                summary["gaps"].append({"at": _now(), "seconds": round(tick - last, 1)})
            last = tick
            free = psutil.virtual_memory().available
            try:
                tree_rss = _tree_rss(psutil, proc)
                info = proc.memory_info()
                wset = int(getattr(info, "peak_wset", 0) or 0)
            except psutil.NoSuchProcess:
                break
            battery = psutil.sensors_battery()
            on_ac = None if battery is None else bool(battery.power_plugged)
            gpu = _nvidia()
            sample = {
                "t": _now(),
                "free_ram_gb": round(free / GB, 3),
                "tree_rss_gb": round(tree_rss / GB, 3),
                "peak_wset_gb": round(wset / GB, 3),
                "on_ac": on_ac,
                "disk_free_gb": round(psutil.disk_usage(args.disk).free / GB, 2),
                "gpu": gpu,
            }
            log.write(json.dumps(sample) + "\n")
            log.flush()
            samples += 1
            min_free = free if min_free is None else min(min_free, free)
            max_tree_rss = max(max_tree_rss, tree_rss)
            peak_wset = max(peak_wset, wset)
            if isinstance(gpu.get("memory.used"), float):
                max_gpu_mib = max(max_gpu_mib or 0.0, gpu["memory.used"])
            if isinstance(gpu.get("temperature.gpu"), float):
                max_temp = max(max_temp or 0.0, gpu["temperature.gpu"])
            foreign = [pid for pid in gpu.get("compute_pids", []) if pid != proc.pid]
            for pid in foreign:
                if pid not in summary["foreign_compute_pids"]:
                    summary["foreign_compute_pids"].append(pid)
            if on_ac is False:
                summary["off_ac_samples"] += 1
            if free < args.min_free_gb * GB:
                summary["fired"] = True
                summary["fired_at"] = _now()
                summary["fired_free_ram_gb"] = round(free / GB, 3)
                summary["stopped_pids"] = _stop_tree(psutil, proc)
                print(f"watchdog FIRED: {free / GB:.2f} GB free < {args.min_free_gb} GB")
                exit_code = 3
                break
            time.sleep(max(0.0, args.period - (time.monotonic() - tick)))

    summary.update(
        ended=_now(),
        samples=samples,
        min_free_ram_gb=None if min_free is None else round(min_free / GB, 3),
        max_tree_rss_gb=round(max_tree_rss / GB, 3),
        peak_wset_gb=round(peak_wset / GB, 3),
        max_gpu_memory_used_mib=max_gpu_mib,
        max_gpu_temperature_c=max_temp,
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
