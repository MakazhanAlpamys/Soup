# Pinned host-memory accounting (#655)

Measured by [Tristan Grech](https://github.com/tristangrech) on 2026-09-10 UTC.
The [original issue comment](https://github.com/MakazhanAlpamys/Soup/issues/655#issuecomment-5623582962)
is reproduced below without edits, including the host-counter drift, following the
[maintainer's request](https://github.com/MakazhanAlpamys/Soup/issues/655#issuecomment-5629542226).

For [#655](https://github.com/MakazhanAlpamys/Soup/issues/655), these measurements
support keeping the `/dev/shm` `statvfs` preflight out of the pinned-store path:
both pinned allocations increased `RssShmem` by exactly 4 GiB while leaving the
mount's usage unchanged. The mount's free space is therefore not a capacity limit
for these allocations on the tested stack, nor a substitute for host and cgroup
headroom checks. This does not establish behavior on other stacks or guarantee
that a full streaming run will fit.

## Original measurement record

The accounting prediction is reproduced on a Linux/CUDA RunPod container. Two fresh-process pinned runs each added exactly **4 GiB to `RssShmem`**, with no change to `RssAnon` or `/dev/shm` usage. The pageable control added exactly **4 GiB to `RssAnon`**, with no change to `RssShmem`.

Environment: NVIDIA L4; driver `570.195.03`; PyTorch `2.9.1+cu128`; CUDA runtime `12.8`; Python `3.12.3`; Ubuntu 24.04.3 container; Linux host kernel `6.14.0-33-generic`. Measurements taken on 2026-09-10 UTC. Neither `PYTORCH_ALLOC_CONF` nor `PYTORCH_CUDA_ALLOC_CONF` was set.

Each run used a fresh Python process. After CUDA initialization and a 1 MiB pinned warmup, it allocated `4 * 1024**3` bytes of `torch.uint8` host memory and touched every page with `fill_(1)`. `is_pinned()` was true for both pinned runs and false for the control. The process read its own `/proc/self/status` and `smaps_rollup` directly.

All numbers below are **KiB, before → after**. The allocation is 4,194,304 KiB.

| Counter | Pinned run 1 | Pinned run 2 | Pageable control |
|---|---:|---:|---:|
| Process `RssAnon` | 283648 → 283648 | 282624 → 282624 | 283648 → 4477952 |
| Process `RssShmem` | 10240 → 4204544 | 10240 → 4204544 | 10240 → 10240 |
| Process `VmLck` | 0 → 0 | 0 → 0 | 0 → 0 |
| Host `Shmem` | 381632 → 4575968 | 384296 → 4578760 | 384288 → 384456 |
| Host `Mlocked` | 27540 → 27540 | 27540 → 27572 | 27540 → 27540 |
| Host `Unevictable` | 30612 → 30612 | 30612 → 30644 | 30612 → 30612 |
| `/dev/shm` used | 0 → 0 | 0 → 0 | 0 → 0 |

`smaps_rollup.Pss_Shmem` independently increased by exactly 4 GiB in each pinned run and stayed unchanged for the control. Cgroup v2 `memory.stat.shmem` also increased by exactly 4 GiB for the pinned allocation. No cgroup OOM events occurred.

The container's `memory.max` was 93,999,996,928 bytes; its `/dev/shm` mount was 47,000,002,560 bytes. The script checked host and cgroup headroom before allocating. Host-wide counters can include activity from other tenants, so the small background changes in those columns should not be attributed to this allocation.

This supports the issue's predicted shared-memory accounting on this stack: the pinned allocation does not consume the `/dev/shm` mount. It does not establish the driver's internal implementation, prove the behavior on every driver/kernel, or measure reclaimability. I did not run the optional streaming test near 55% of host RAM or induce an OOM.

The temporary pod has been terminated. The exact probe below can be run twice with `--kind pinned` and once with `--kind pageable`; each invocation emits its environment and raw before/after counters as JSON. Probe SHA-256: `44bba1b76a9b41e927fe59666104bd44fcce63288f4af48887736118f7111c11`.


<details>
<summary>Exact probe used (save as probe.py)</summary>

```python
"""Soup #655: bounded pinned-host-memory accounting probe; one fresh process/run."""
import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

GIB = 1024**3


def kb_fields(path, wanted):
    result = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            result[key] = int(value.split()[0]) * 1024
    return result


def cgroup_snapshot():
    roots = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if ".." in Path(relative).parts:
            continue
        if not controllers:
            roots.append(Path("/sys/fs/cgroup") / relative.lstrip("/"))
        elif "memory" in controllers.split(","):
            roots.append(Path("/sys/fs/cgroup/memory") / relative.lstrip("/"))
    results = {}
    for root in dict.fromkeys(roots):
        for filename in ("memory.current", "memory.max", "memory.limit_in_bytes",
                         "memory.usage_in_bytes", "memory.stat", "memory.events"):
            path = root / filename
            if path.is_file():
                results[str(path)] = path.read_text().strip()
    return results


def snapshot():
    shm = os.statvfs("/dev/shm")
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "process_bytes": kb_fields("/proc/self/status", {
            "VmRSS", "VmLck", "RssAnon", "RssFile", "RssShmem"}),
        "smaps_rollup_bytes": kb_fields("/proc/self/smaps_rollup", {
            "Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Locked"}),
        "host_bytes": kb_fields("/proc/meminfo", {
            "MemTotal", "MemAvailable", "Shmem", "Mlocked", "Unevictable"}),
        "dev_shm_bytes": {
            "total": shm.f_blocks * shm.f_frsize,
            "used": (shm.f_blocks - shm.f_bfree) * shm.f_frsize,
            "available": shm.f_bavail * shm.f_frsize,
        },
        "cgroup": cgroup_snapshot(),
    }


def check_headroom(before, allocation_bytes):
    headrooms = [before["host_bytes"]["MemAvailable"]]
    cgroup = before["cgroup"]
    pairs = (("memory.max", "memory.current"),
             ("memory.limit_in_bytes", "memory.usage_in_bytes"))
    found = False
    for path, value in cgroup.items():
        for limit_name, usage_name in pairs:
            if Path(path).name != limit_name:
                continue
            usage = cgroup.get(str(Path(path).with_name(usage_name)))
            if usage is not None:
                found = True
                if value != "max":
                    headrooms.append(int(value) - int(usage))
    if not found:
        raise RuntimeError("Cannot verify container memory headroom")
    if min(headrooms) < 2 * allocation_bytes:
        raise RuntimeError("Less than 2x allocation headroom; refusing probe")
    return headrooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("pinned", "pageable"), required=True)
    parser.add_argument("--gib", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    environment = {
        "kernel": platform.release(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"], text=True).strip(),
        "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
        "allocator_environment": {key: os.environ.get(key) for key in (
            "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
        "os_release": Path("/etc/os-release").read_text().strip(),
    }
    torch.cuda.init()
    warmup = torch.empty(1024**2, dtype=torch.uint8, pin_memory=True)
    warmup.fill_(1)
    del warmup
    gc.collect()
    torch.cuda.synchronize()
    before = snapshot()
    allocation_bytes = args.gib * GIB
    headroom = check_headroom(before, allocation_bytes)
    start = time.monotonic()
    tensor = torch.empty(allocation_bytes, dtype=torch.uint8, device="cpu",
                         pin_memory=args.kind == "pinned")
    tensor.fill_(1)
    assert tensor.is_pinned() == (args.kind == "pinned")
    assert tensor[0].item() == 1 and tensor[-1].item() == 1
    time.sleep(0.25)
    after = snapshot()
    delta = {
        section: {key: after[section][key] - value
                  for key, value in before[section].items()}
        for section in ("process_bytes", "smaps_rollup_bytes", "host_bytes",
                        "dev_shm_bytes")
    }
    result = {
        "kind": args.kind, "allocation_bytes": allocation_bytes,
        "is_pinned": tensor.is_pinned(), "environment": environment,
        "headroom_bytes": headroom, "before": before, "after": after,
        "delta_bytes": delta, "duration_seconds": time.monotonic() - start,
        "note": "Process counters are local; host counters can include other tenants.",
    }
    del tensor
    gc.collect()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
```

```bash
python probe.py --kind pinned --gib 4
python probe.py --kind pinned --gib 4
python probe.py --kind pageable --gib 4
```
</details>
