"""soup doctor — check dependency compatibility and system health."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.constants import GITHUB_URL

console = Console()

# Dependencies to check: (import_name, package_name, min_version, required)
DEPS = [
    ("torch", "torch", "2.0.0", True),
    ("transformers", "transformers", "4.36.0", True),
    ("peft", "peft", "0.7.0", True),
    ("trl", "trl", "0.7.0", True),
    ("datasets", "datasets", "2.14.0", True),
    ("bitsandbytes", "bitsandbytes", "0.41.0", True),
    ("accelerate", "accelerate", "0.25.0", True),
    ("pydantic", "pydantic", "2.0.0", True),
    ("typer", "typer", "0.9.0", True),
    ("rich", "rich", "13.0.0", True),
    ("yaml", "pyyaml", "6.0", True),
    ("plotext", "plotext", "5.2.0", True),
    # Optional
    ("fastapi", "fastapi", "0.104.0", False),
    ("uvicorn", "uvicorn", "0.24.0", False),
    ("datasketch", "datasketch", "1.6.0", False),
    ("lm_eval", "lm-eval", "0.4.0", False),
    ("wandb", "wandb", "0.15.0", False),
    ("deepspeed", "deepspeed", "0.12.0", False),
    ("httpx", "httpx", "0.24.0", False),
    ("unsloth", "unsloth", "2024.8", False),
    ("PIL", "Pillow", "9.0.0", False),
    ("torchao", "torchao", "0.4.0", False),
    ("sglang", "sglang", "0.2.0", False),
    ("librosa", "librosa", "0.10.0", False),
]

# v0.40.1 Part C / C5 — packages whose major version we explicitly cap.
# Empty by default; entries gate the dependency table to flag a
# breaking-major upgrade (e.g. transformers 5.x) as INCOMPATIBLE rather
# than silently allowing it.
_MAX_EXCLUSIVE: dict[str, str] = {
    "transformers": "5.0.0",
}


def doctor(
    nccl: bool = typer.Option(
        False, "--nccl", help="Measure NCCL bandwidth and check against reference table."
    ),
):
    """Check system dependencies, GPU, and compatibility."""
    console.print("[bold]Soup Doctor[/] - checking your environment...\n")

    # System info
    dual_python_advisory = _detect_dual_python_interpreters()
    panel_body = (
        f"Python:   [bold]{sys.version.split()[0]}[/]\n"
        f"Platform: [bold]{platform.system()} {platform.release()}[/]\n"
        f"Arch:     [bold]{platform.machine()}[/]"
    )
    if dual_python_advisory:
        panel_body += f"\n[yellow]{dual_python_advisory}[/]"
    console.print(Panel(panel_body, title="System"))

    # GPU check
    _check_gpu()

    # Resources check
    _check_resources()

    # Dependencies table
    table = Table(title="Dependencies")
    table.add_column("Package", style="bold")
    table.add_column("Required", justify="center")
    table.add_column("Installed", justify="center")
    table.add_column("Min Version")
    table.add_column("Status")

    issues = []

    for import_name, pkg_name, min_ver, required in DEPS:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", getattr(mod, "VERSION", None))
            if version is None:
                # v0.40.1 Part D / M1 — some installs (notably ``rich``)
                # don't export ``__version__`` on the package surface;
                # importlib.metadata is canonical and works everywhere.
                try:
                    from importlib.metadata import (
                        PackageNotFoundError,
                    )
                    from importlib.metadata import (
                        version as _pkgver,
                    )

                    version = _pkgver(pkg_name)
                except (PackageNotFoundError, ImportError):
                    version = "?"
            version_str = str(version)

            # v0.40.1 Part C / C5 — flag transformers 5.x as INCOMPATIBLE
            # until the TRL/transformers 5.x migration lands.
            max_excl = _MAX_EXCLUSIVE.get(pkg_name)
            if max_excl and _version_ge(version_str, max_excl):
                status = f"[red]INCOMPATIBLE (need <{max_excl})[/]"
                issues.append(
                    f"Downgrade {pkg_name}: pip install '{pkg_name}>={min_ver},<{max_excl}'"
                )
            elif _version_ok(version_str, min_ver):
                status = "[green]OK[/]"
            else:
                status = f"[yellow]outdated (need >={min_ver})[/]"
                issues.append(f"Upgrade {pkg_name}: pip install '{pkg_name}>={min_ver}'")

            table.add_row(
                pkg_name,
                "yes" if required else "optional",
                version_str,
                f">={min_ver}",
                status,
            )
        except ImportError:
            if required:
                status = "[red]MISSING[/]"
                issues.append(f"Install {pkg_name}: pip install '{pkg_name}>={min_ver}'")
            else:
                status = "[dim]not installed[/]"

            table.add_row(
                pkg_name,
                "yes" if required else "optional",
                "-",
                f">={min_ver}",
                status,
            )

    console.print(table)

    # Check torchvision + torch compatibility
    _check_torchvision_compat(issues)

    if nccl:
        _run_nccl_check()

    # Summary
    if issues:
        console.print(f"\n[yellow]Found {len(issues)} issue(s):[/]")
        for issue in issues:
            console.print(f"  [red]>[/] {issue}")
        console.print(
            "\n[dim]Fix all: pip install -U "
            + " ".join(
                f"'{pkg_name}>={min_ver}'" for _, pkg_name, min_ver, required in DEPS if required
            )
            + "[/]"
        )
    else:
        console.print("\n[bold green]All checks passed![/] Your environment is ready.")

    console.print(f"\n[dim]GitHub: [link={GITHUB_URL}]{GITHUB_URL}[/link][/]")


def _get_mlx_info() -> dict:
    """Surface MLX info in the doctor report (never crashes on non-Apple)."""
    try:
        from soup_cli.utils.mlx import get_mlx_info
    except ImportError:
        return {"available": False}
    try:
        return get_mlx_info()
    except Exception:  # noqa: BLE001
        return {"available": False}


def _check_gpu():
    """Check GPU availability and display info."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for idx in range(gpu_count):
                name = torch.cuda.get_device_name(idx)
                mem = torch.cuda.get_device_properties(idx)
                total_gb = getattr(mem, "total_memory", getattr(mem, "total_mem", 0))
                total_gb = total_gb / (1024**3)
                gpus.append(f"  GPU {idx}: [bold]{name}[/] ({total_gb:.1f} GB)")
            gpu_info = "\n".join(gpus)
            cuda_ver = torch.version.cuda or "N/A"
            console.print(
                Panel(
                    f"CUDA:     [bold green]available[/] (v{cuda_ver})\n"
                    f"GPUs:     [bold]{gpu_count}[/]\n{gpu_info}",
                    title="GPU",
                )
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            console.print(
                Panel(
                    "Backend:  [bold green]MPS (Apple Silicon)[/]\n"
                    "Status:   [bold green]available[/]",
                    title="GPU",
                )
            )
        else:
            # v0.40.1 Part C / N3 — distinguish "no GPU hardware" from
            # "GPU hardware present, wrong torch wheel". When nvidia-smi
            # reports a GPU but torch lacks CUDA, the user installed the
            # CPU-only wheel — point them at the right reinstall command.
            advisory = _detect_gpu_hw_without_torch_cuda()
            console.print(
                Panel(
                    "Backend:  [bold yellow]CPU only[/]\n"
                    "Warning:  Training will be slow without GPU."
                    + (f"\n[dim]{advisory}[/]" if advisory else ""),
                    title="GPU",
                )
            )
    except ImportError:
        console.print(
            Panel(
                "Backend:  [red]unknown (torch not installed)[/]",
                title="GPU",
            )
        )


def _detect_gpu_hw_without_torch_cuda() -> str:
    """v0.40.1 Part C / N3 — return advisory string if nvidia-smi succeeds
    but torch lacks CUDA (i.e. user installed the CPU-only wheel).
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        completed = subprocess.run(  # noqa: S603 — argv list, no shell
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    gpu_name = (completed.stdout or "").strip().splitlines()[:1]
    raw_label = gpu_name[0] if gpu_name else "GPU"
    # v0.40.1 review fix — security: nvidia-smi output is embedded in a
    # Rich-markup string at the call site; escape `[`/`]` so a GPU name like
    # "NVIDIA Quadro [T4]" cannot break or inject markup.
    from rich.markup import escape as _markup_escape

    gpu_label = _markup_escape(raw_label)
    try:
        from importlib.metadata import version as _pkgver

        torch_version = _pkgver("torch")
    except Exception:  # noqa: BLE001
        torch_version = "?"
    return (
        f"GPU hardware present ({gpu_label}) but torch is the CPU build "
        f"(torch {torch_version}). To enable your GPU: "
        f"`pip install torch --index-url https://download.pytorch.org/whl/cu121`"
    )


def _detect_dual_python_interpreters() -> str:
    """v0.40.1 Part C / N4 — flag when ``soup`` runs under one Python and
    ``python`` on the user's PATH is a different interpreter.
    """
    import os
    import shutil

    soup_python = sys.executable
    path_python = shutil.which("python") or shutil.which("python3")
    if not path_python:
        return ""
    # v0.40.1 review fix — use os.path.realpath, not Path.resolve(), so
    # Windows 8.3 short names don't produce a false-positive advisory.
    try:
        if os.path.realpath(path_python) == os.path.realpath(soup_python):
            return ""
    except OSError:
        return ""
    return (
        f"`soup` runs under {soup_python}; `python` on your PATH is "
        f"{path_python}. site-packages may differ — for any `python -c` "
        f"check use the soup interpreter explicitly."
    )


_GB = 1024**3


def _get_ram_gb() -> str:
    """Get total system RAM in GB, with cross-platform fallbacks."""
    # Prefer psutil if installed
    try:
        import psutil

        return f"{psutil.virtual_memory().total / _GB:.0f} GB"
    except ImportError:
        pass

    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return f"{kb * 1024 / _GB:.0f} GB"
        except (OSError, ValueError):
            pass
    elif system == "Darwin":
        try:
            import subprocess

            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0:
                return f"{int(res.stdout.strip()) / _GB:.0f} GB"
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    elif system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return f"{stat.ullTotalPhys / _GB:.0f} GB"
        except (OSError, AttributeError):
            pass

    return "Unknown"


def _check_resources():
    """Check RAM and Disk space and display info."""
    import shutil

    table = Table(title="System Resources")
    table.add_column("Resource", style="bold")
    table.add_column("Value")

    table.add_row("RAM", _get_ram_gb())

    try:
        usage = shutil.disk_usage(".")
        disk_str = f"{usage.free / _GB:.0f} GB free"
    except OSError:
        disk_str = "Unknown"

    table.add_row("Disk", disk_str)
    console.print(table)
    console.print()


def _check_torchvision_compat(issues: list):
    """Check that torchvision version is compatible with torch."""
    try:
        import torch
        import torchvision

        torch_ver = torch.__version__.split("+")[0]
        tv_ver = torchvision.__version__.split("+")[0]
        torch_minor = ".".join(torch_ver.split(".")[:2])
        tv_minor = ".".join(tv_ver.split(".")[:2])

        # Known compatible pairs (torch minor -> torchvision minor)
        compat = {
            "2.6": "0.21",
            "2.5": "0.20",
            "2.4": "0.19",
            "2.3": "0.18",
            "2.2": "0.17",
            "2.1": "0.16",
            "2.0": "0.15",
        }
        expected_tv = compat.get(torch_minor)
        if expected_tv and not tv_minor.startswith(expected_tv):
            msg = (
                f"torchvision {tv_ver} may be incompatible with torch {torch_ver}. "
                f"Expected torchvision {expected_tv}.x"
            )
            console.print(f"  [yellow]Warning:[/] {msg}")
            issues.append(msg)
    except (ImportError, AttributeError):
        # AttributeError: torchvision circular import on some platforms
        pass


def _version_ok(installed: str, minimum: str) -> bool:
    """Check if installed version meets minimum requirement."""
    try:
        inst_parts = [int(x) for x in installed.split(".")[:3]]
        min_parts = [int(x) for x in minimum.split(".")[:3]]
        # Pad to same length
        while len(inst_parts) < 3:
            inst_parts.append(0)
        while len(min_parts) < 3:
            min_parts.append(0)
        return inst_parts >= min_parts
    except (ValueError, AttributeError):
        return True  # Can't parse, assume OK


def _version_ge(installed: str, threshold: str) -> bool:
    """v0.40.1 Part C / C5 — return True iff installed >= threshold.

    Used to flag major-version upgrades we haven't migrated to. Robust to
    suffixes like ``5.0.0.dev0`` (split on ``.``, parse leading ints only).
    """
    try:
        inst_parts: list[int] = []
        for chunk in installed.split(".")[:3]:
            digits = "".join(c for c in chunk if c.isdigit())
            inst_parts.append(int(digits) if digits else 0)
        thr_parts = [int(x) for x in threshold.split(".")[:3]]
        while len(inst_parts) < 3:
            inst_parts.append(0)
        while len(thr_parts) < 3:
            thr_parts.append(0)
        return inst_parts >= thr_parts
    except (ValueError, AttributeError):
        return False


def _run_nccl_check():
    """Run NCCL bandwidth test if requested."""
    try:
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp

        from soup_cli.utils.profiling_v0_43 import nccl_bandwidth_check
        from soup_cli.utils.topology import detect_topology
    except ImportError:
        console.print("\n[yellow]NCCL bandwidth check requires torch and torch.distributed.[/]")
        return

    if not torch.cuda.is_available() or not dist.is_available():
        console.print("\n[yellow]NCCL bandwidth check requires CUDA and torch.distributed.[/]")
        return

    topo = detect_topology()
    gpu_count = topo["gpu_count"]
    if gpu_count < 2:
        console.print("\n[yellow]NCCL bandwidth requires >=2 GPUs[/]")
        return

    name = torch.cuda.get_device_name(0).lower()
    if "h100" in name:
        gpu = "h100"
    elif "a100" in name:
        gpu = "a100"
    elif "v100" in name:
        gpu = "v100"
    elif "4090" in name:
        gpu = "rtx4090"
    elif "3090" in name:
        gpu = "rtx3090"
    else:
        gpu = "unknown"

    link = topo["interconnect"]

    manager = mp.Manager()
    return_dict = manager.dict()

    console.print(
        "\n[bold]Measuring NCCL bandwidth[/] "
        "(100 MB all_reduce, 3 warmup + 10 timed iters, median GB/s)..."
    )
    try:
        mp.spawn(_nccl_worker, args=(return_dict,), nprocs=2, join=True)
    except Exception as e:
        console.print(f"[red]Failed to measure NCCL bandwidth:[/] {e}")
        return

    if "gb_per_sec" in return_dict:
        measured = return_dict["gb_per_sec"]
        res = nccl_bandwidth_check(gpu=gpu, link=link, measured_gb_per_sec=measured)
        status = res["status"]
        if status == "OK":
            color = "green"
        elif status == "MINOR":
            color = "yellow"
        else:
            color = "red"

        expected = res.get("expected_gb_per_sec")
        expected_str = f" vs expected {expected:.1f}" if expected else ""

        console.print(
            f"  Result ({gpu.upper()} over {link.upper()}): "
            f"[{color}]{status}[/] ({measured:.1f} GB/s{expected_str})"
        )
    else:
        console.print("[red]Failed to capture NCCL bandwidth measurement.[/]")


# NCCL benchmark constants. 100 MB matches typical gradient-bucket size for
# medium models (7B fp16 ≈ 100-200 MB per bucket) so the measurement
# reflects real all-reduce traffic, not artificially small messages.
_NCCL_BENCHMARK_TENSOR_BYTES = 100 * 1024 * 1024  # 100 MB
_NCCL_BENCHMARK_WARMUP_ITERS = 3  # 3 warmups to amortise CUDA kernel JIT
_NCCL_BENCHMARK_TIMED_ITERS = 10  # 10 timed runs; report the median (robust to outliers)


def _nccl_worker(rank: int, return_dict):
    import os
    import statistics
    import time

    import torch
    import torch.distributed as dist

    # Snapshot prior env so we can restore after the spawn finishes —
    # avoids leaking MASTER_ADDR/MASTER_PORT into the parent doctor process
    # (matters when tests run multiple doctor invocations in one session).
    prior_addr = os.environ.get("MASTER_ADDR")
    prior_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    try:
        dist.init_process_group("nccl", rank=rank, world_size=2)

        # 100 MB tensor of float32 (4 bytes per element).
        num_elements = _NCCL_BENCHMARK_TENSOR_BYTES // 4
        tensor = torch.ones(num_elements, dtype=torch.float32, device=f"cuda:{rank}")

        # Warmup — runs CUDA kernel JIT + first-NCCL-collective handshake
        # out of the timing window. 3 iters is enough to stabilise.
        for _ in range(_NCCL_BENCHMARK_WARMUP_ITERS):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device=f"cuda:{rank}")

        # Per-iteration timing (10 samples). Median is more robust to
        # one-off jitter (kernel preemption, swap, etc.) than mean.
        per_iter_sec: list[float] = []
        for _ in range(_NCCL_BENCHMARK_TIMED_ITERS):
            start = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize(device=f"cuda:{rank}")
            per_iter_sec.append(time.perf_counter() - start)

        if rank == 0:
            median_elapsed = statistics.median(per_iter_sec)
            size_gb = (tensor.element_size() * tensor.numel()) / 1e9
            return_dict["gb_per_sec"] = size_gb / median_elapsed

        dist.destroy_process_group()
    finally:
        # Restore prior env exactly as we found it (or remove if missing).
        if prior_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = prior_addr
        if prior_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = prior_port
