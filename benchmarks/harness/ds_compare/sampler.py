#!/usr/bin/env python3
"""Resource samplers for one run: nvidia-smi every 0.5 s, host RSS of a process tree at 1 Hz.

``RunSampler`` is what ``orchestrate.py`` uses: ``start()`` before the child is launched,
``stop()`` after it exits, both CSVs sit next to the run json::

    rep1.gpu.csv   unix_time, memory_used_mib, utilization_gpu_pct, sm_clock_mhz
    rep1.rss.csv   unix_time, rss_bytes, processes

GPU samples come from ONE long-lived ``nvidia-smi --loop-ms=500`` process (the STEP 3
method, without a process spawn per sample). A second, slower query lists the compute
processes on the card, for the "another process is on the GPU" void check.

``summarize`` turns the CSVs into the numbers of protocol section 3, both for the whole run
(STEP 3's definition of peak VRAM) and for the training window the trace gives
(``begin_wall`` .. ``begin_wall + train_runtime``): mean utilization and the median SM clock
"while busy" are taken over the training window, where the model load and the final save do
not dilute them.

Standalone use::

    python sampler.py --pid 1234 --prefix results/x/rep1     # until the pid exits
    python sampler.py --summarize results/x/rep1
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

GPU_INTERVAL_MS = 500
APPS_INTERVAL_S = 2.5
RSS_INTERVAL_S = 1.0
BUSY_UTILIZATION_PCT = 0  # a sample is "busy" when utilization is above this


def _number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None  # "[N/A]" and friends


class RunSampler:
    def __init__(
        self,
        prefix: str,
        pid: int,
        gpu_index: int = 0,
        nvidia_smi: str = "nvidia-smi",
    ) -> None:
        self.prefix = prefix
        self.pid = pid
        self.gpu_index = gpu_index
        self.nvidia_smi = nvidia_smi
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.tree_pids: set[int] = set()
        self.apps: dict[int, dict[str, Any]] = {}
        self.gpu_available = shutil.which(nvidia_smi) is not None or os.path.exists(nvidia_smi)
        self.started_at: float | None = None
        self.stopped_at: float | None = None

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        self.started_at = time.time()
        os.makedirs(os.path.dirname(self.prefix) or ".", exist_ok=True)
        if self.gpu_available:
            self._smi = subprocess.Popen(
                [
                    self.nvidia_smi,
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used,utilization.gpu,clocks.sm",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={GPU_INTERVAL_MS}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._spawn(self._read_gpu)
            self._spawn(self._poll_apps)
        else:
            open(self.prefix + ".gpu.csv", "w", encoding="utf-8").close()
        self._spawn(self._poll_rss)

    def stop(self) -> None:
        self._stop.set()
        if self._smi is not None:
            self._kill_smi()
        for thread in self._threads:
            thread.join(timeout=10)
        self.stopped_at = time.time()

    def _kill_smi(self) -> None:
        """Stop the long-lived nvidia-smi, and anything a wrapper script started under it."""
        assert self._smi is not None
        try:
            import psutil

            for child in psutil.Process(self._smi.pid).children(recursive=True):
                child.kill()
        except Exception:  # noqa: BLE001 - already gone
            pass
        self._smi.terminate()
        try:
            self._smi.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._smi.kill()

    def _spawn(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    # -- workers ----------------------------------------------------------------------
    def _read_gpu(self) -> None:
        assert self._smi is not None and self._smi.stdout is not None
        with open(self.prefix + ".gpu.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["unix_time", "memory_used_mib", "utilization_gpu_pct", "sm_clock_mhz"])
            for line in self._smi.stdout:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                values = [_number(part) for part in parts]
                writer.writerow(
                    [f"{time.time():.3f}"] + ["" if value is None else value for value in values]
                )
                handle.flush()

    def _poll_apps(self) -> None:
        command = [
            self.nvidia_smi,
            f"--id={self.gpu_index}",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    command, capture_output=True, text=True, timeout=10, check=False
                ).stdout
            except (OSError, subprocess.TimeoutExpired):
                out = ""
            now = time.time()
            for line in out.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                pid = int(parts[0])
                used = _number(parts[1]) or 0.0
                entry = self.apps.setdefault(
                    pid,
                    {"name": parts[2] if len(parts) > 2 else "", "max_mib": 0.0, "first_seen": now},
                )
                entry["max_mib"] = max(entry["max_mib"], used)
                entry["last_seen"] = now
            self._stop.wait(APPS_INTERVAL_S)

    def _poll_rss(self) -> None:
        import psutil

        with open(self.prefix + ".rss.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["unix_time", "rss_bytes", "processes"])
            while not self._stop.is_set():
                try:
                    root = psutil.Process(self.pid)
                    procs = [root] + root.children(recursive=True)
                except psutil.Error:
                    procs = []
                total = 0
                for proc in procs:
                    try:
                        total += proc.memory_info().rss
                        self.tree_pids.add(proc.pid)
                    except psutil.Error:
                        continue
                writer.writerow([f"{time.time():.3f}", total, len(procs)])
                handle.flush()
                self._stop.wait(RSS_INTERVAL_S)

    # -- result -----------------------------------------------------------------------
    def foreign_gpu_processes(self) -> dict[str, Any]:
        """Compute processes on the card that are not in the run's process tree.

        Inside a container ``nvidia-smi`` often reports host PIDs, which never match the
        tree; then the check is declared unavailable instead of flagging our own processes.
        """
        visible = any(pid in self.tree_pids for pid in self.apps)
        foreign = sorted(pid for pid in self.apps if pid not in self.tree_pids) if visible else None
        return {
            "apps_seen": {str(pid): info for pid, info in sorted(self.apps.items())},
            "pid_namespace_visible": visible,
            "foreign_pids": foreign,
        }


def read_csv_rows(path: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in record.items():
                number = _number(value) if value not in (None, "") else None
                if number is not None:
                    parsed[key] = number
            if "unix_time" in parsed:
                rows.append(parsed)
    return rows


def summarize(prefix: str, window: tuple[float, float] | None = None) -> dict[str, Any]:
    """Protocol section 3 numbers from ``<prefix>.gpu.csv`` and ``<prefix>.rss.csv``."""
    gpu = read_csv_rows(prefix + ".gpu.csv")
    rss = read_csv_rows(prefix + ".rss.csv")

    def stats(rows: list[dict[str, float]]) -> dict[str, Any]:
        mem = [row["memory_used_mib"] for row in rows if "memory_used_mib" in row]
        util = [row["utilization_gpu_pct"] for row in rows if "utilization_gpu_pct" in row]
        busy_clock = [
            row["sm_clock_mhz"]
            for row in rows
            if row.get("utilization_gpu_pct", 0) > BUSY_UTILIZATION_PCT and "sm_clock_mhz" in row
        ]
        return {
            "samples": len(rows),
            "vram_peak_mib": max(mem) if mem else None,
            "gpu_util_mean_pct": statistics.fmean(util) if util else None,
            "sm_clock_busy_median_mhz": statistics.median(busy_clock) if busy_clock else None,
            "busy_samples": len(busy_clock),
        }

    out: dict[str, Any] = {
        "whole_run": stats(gpu),
        "rss_peak_bytes": max((row["rss_bytes"] for row in rss), default=None),
        "rss_samples": len(rss),
        "gpu_idle_baseline_mib": gpu[0].get("memory_used_mib") if gpu else None,
    }
    if window is not None:
        start, end = window
        inside = [row for row in gpu if start <= row["unix_time"] <= end]
        out["train_window"] = stats(inside)
        out["train_window"]["start"] = start
        out["train_window"]["end"] = end
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--prefix", help="output prefix for --pid mode")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--summarize", metavar="PREFIX")
    args = parser.parse_args(argv)
    if args.summarize:
        import json

        print(json.dumps(summarize(args.summarize), indent=2))
        return 0
    if args.pid is None or not args.prefix:
        parser.error("--pid and --prefix are required (or use --summarize)")
    import psutil

    sampler = RunSampler(args.prefix, args.pid, args.gpu, args.nvidia_smi)
    sampler.start()
    try:
        while psutil.pid_exists(args.pid):
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
