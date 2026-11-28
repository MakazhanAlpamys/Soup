"""Run all pilot phases under an observed real RAM constraint, retaining accounting."""

import argparse
import json
import os
import posixpath
import re
import subprocess
import time
from pathlib import Path

MAX_RAM = 32 * 2**30
REQUIRED_PHASES = {
    "download",
    "preparation",
    "training",
    "evaluation_before",
    "evaluation_after",
    "reload",
}


def ram_constraint(sample):
    maximum = sample.get("memory_max")
    if type(maximum) is int and 0 < maximum <= MAX_RAM:
        return {
            "kind": "cgroup_v" + str(sample.get("cgroup_version", 2)),
            "hard_limit_bytes": maximum,
        }
    total = sample.get("host_total", 0)
    if 28_000_000_000 <= total <= MAX_RAM and sample.get("host_swap_total") == 0:
        return {"kind": "physical_host", "hard_limit_bytes": total}
    raise ValueError("pilot requires an actual RAM limit <=32 GiB, not process RSS alone")


def summarize_ram(samples, constraint, returncode):
    phases = sorted({row["phase"] for row in samples})
    first = samples[0].get("events", {}).get("oom_kill")
    last = samples[-1].get("events", {}).get("oom_kill")
    oom_delta = last - first if first is not None and last is not None else None
    host_first, host_last = samples[0].get("host_oom_kills"), samples[-1].get("host_oom_kills")
    host_oom_delta = (
        host_last - host_first if host_first is not None and host_last is not None else None
    )
    swap_values = [row.get("swap_current") for row in samples]
    swap_peaks = [row.get("swap_peak") for row in samples]
    swap_known = all(value is not None for value in swap_values)
    swap_peak = max(
        (value for value in swap_values + swap_peaks if value is not None), default=None
    )
    events_known = oom_delta is not None or constraint["kind"] == "physical_host"
    if constraint["kind"].startswith("cgroup_v"):
        stable = all(r.get("memory_max") == constraint["hard_limit_bytes"] for r in samples)
    else:
        stable = all(
            r.get("host_total") == constraint["hard_limit_bytes"] and r.get("host_swap_total") == 0
            for r in samples
        )
    passed = (
        stable
        and returncode == 0
        and REQUIRED_PHASES.issubset(phases)
        and events_known
        and (oom_delta in (0, None))
        and host_oom_delta in (0, None)
        and swap_known
        and swap_peak == 0
    )
    currents = [r["memory_current"] for r in samples if r.get("memory_current") is not None]
    peaks = [r["memory_peak"] for r in samples if r.get("memory_peak") is not None]
    return {
        "ram_32gib_constrained_pilot_passed": passed,
        "constraint": constraint,
        "returncode": returncode,
        "constraint_stable": stable,
        "observed_phases": phases,
        "samples": len(samples),
        "oom_kill_delta": oom_delta,
        "swap_peak_observed_bytes": swap_peak,
        "host_oom_kill_delta": host_oom_delta,
        "max_sampled_cgroup_bytes": max(currents, default=None),
        "kernel_cgroup_peak_bytes": max(peaks, default=None),
        "max_sampled_process_rss_bytes": max(r["process_rss_bytes"] for r in samples),
        "start_unix": samples[0].get("unix"),
        "end_unix": samples[-1].get("unix"),
    }


def locate_memory_cgroup(proc):
    """Resolve this process's controller membership against the mounted root."""
    memberships = {}
    for line in (proc / "self/cgroup").read_text(encoding="utf-8").splitlines():
        hierarchy, controllers, member = line.split(":", 2)
        if "memory" in controllers.split(","):
            memberships[1] = member
        elif hierarchy == "0" and not controllers:
            memberships[2] = member

    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

    for line in (proc / "self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, right = line.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        version = 2 if filesystem[0] == "cgroup2" else 1
        if filesystem[0] not in ("cgroup", "cgroup2") or version not in memberships:
            continue
        if version == 1 and "memory" not in filesystem[2].split(","):
            continue
        root, mount = unescape(fields[3]), unescape(fields[4])
        member = memberships[version]
        if posixpath.commonpath([member, root]) != root:
            continue
        relative = posixpath.relpath(member, root)
        candidate = os.path.realpath(os.path.join(mount, *relative.split("/")))
        mount = os.path.realpath(mount)
        if os.path.commonpath([candidate, mount]) != mount:
            raise ValueError("memory-controller path escapes its mounted root")
        marker = "memory.max" if version == 2 else "memory.limit_in_bytes"
        if (Path(candidate) / marker).is_file():
            return Path(candidate), version
    return None, None


def snapshot(phase_file, *, cgroup=None, proc=Path("/proc")):
    if cgroup is None:
        cgroup, version = locate_memory_cgroup(proc)
    else:
        version = 2 if (cgroup / "memory.max").is_file() else 1

    def number(name):
        if cgroup is None:
            return None
        try:
            value = (cgroup / name).read_text(encoding="utf-8").strip()
            return None if value == "max" else int(value)
        except FileNotFoundError:
            return None

    def counters(path):
        try:
            return {
                key: int(value)
                for key, value in (
                    line.split() for line in path.read_text(encoding="utf-8").splitlines()
                )
            }
        except FileNotFoundError:
            return {}

    meminfo = {}
    for line in (proc / "meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.split()[0]) * 1024
    processes = []
    for path in proc.glob("[0-9]*/status"):
        try:
            fields = dict(
                line.split(":", 1) for line in path.read_text(encoding="utf-8").splitlines()
            )
            processes.append(
                {
                    "pid": int(path.parent.name),
                    **{
                        key: int(fields.get(key, "0 kB").split()[0]) * 1024
                        for key in ("VmRSS", "VmHWM", "VmSwap")
                    },
                }
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    phase = phase_file.read_text(encoding="utf-8").strip() if phase_file.is_file() else "setup"
    memory_stat = counters(cgroup / "memory.stat") if cgroup is not None else {}
    if version == 1:
        maximum = number("memory.limit_in_bytes")
        if maximum is not None and maximum >= 2**60:
            maximum = None  # v1 uses a large integer to represent no limit.
        combined = number("memory.memsw.limit_in_bytes")
        swap_max = max(0, combined - maximum) if combined is not None and maximum else None
        swap = memory_stat.get("total_swap", memory_stat.get("swap"))
        peak = number("memory.max_usage_in_bytes")
        current = number("memory.usage_in_bytes")
        events = counters(cgroup / "memory.oom_control")
        events["limit_hits"] = number("memory.failcnt")
        swap_peak = None  # Subtracting two independently timed peaks is invalid.
    else:
        maximum = number("memory.max")
        swap_max, swap = number("memory.swap.max"), number("memory.swap.current")
        current, peak = number("memory.current"), number("memory.peak")
        events = counters(cgroup / "memory.events") if cgroup is not None else {}
        swap_peak = number("memory.swap.peak")
    if swap is None and meminfo["SwapTotal"] == 0:
        swap = 0
    return {
        "unix": time.time(),
        "phase": phase,
        "memory_max": maximum,
        "cgroup_version": version,
        "cgroup_path": str(cgroup) if cgroup else None,
        "memory_current": current,
        "memory_peak": peak,
        "swap_max": swap_max,
        "swap_current": swap,
        "swap_peak": swap_peak,
        "events": events,
        "memory_stat": memory_stat,
        "host_total": meminfo["MemTotal"],
        "host_swap_total": meminfo["SwapTotal"],
        "host_oom_kills": counters(proc / "vmstat").get("oom_kill"),
        "host_swap_used": meminfo["SwapTotal"] - meminfo["SwapFree"],
        "processes": processes,
        "process_rss_bytes": sum(r["VmRSS"] for r in processes),
    }


def run(args):
    from rich.console import Console

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    phase = Path(args.phase_file)
    samples = [snapshot(phase)]
    constraint = ram_constraint(samples[0])  # Refuse before any download/cache preparation.
    with (output / "ram-samples.jsonl").open("x", encoding="utf-8") as handle:

        def record(row):
            handle.write(json.dumps(row) + "\n")
            handle.flush()

        record(samples[0])
        child = subprocess.Popen(args.command)
        try:
            while child.poll() is None:
                time.sleep(args.interval)
                samples.append(snapshot(phase))
                record(samples[-1])
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
            samples.append(snapshot(phase))
            record(samples[-1])
            result = summarize_ram(samples, constraint, child.returncode)
            (output / "ram-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    Console().print(result)
    return 0 if result["ram_32gib_constrained_pilot_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-file", required=True)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.command and options.command[0] == "--":
        options.command.pop(0)
    if not options.command or options.interval <= 0:
        parser.error("a command and a positive sample interval are required")
    raise SystemExit(run(options))
