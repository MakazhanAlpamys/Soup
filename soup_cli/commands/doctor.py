"""soup doctor — check dependency compatibility and system health."""

import platform
import sys

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


def doctor():
    """Check system dependencies, GPU, and compatibility."""
    console.print("[bold]Soup Doctor[/] - checking your environment...\n")

    # System info
    console.print(
        Panel(
            f"Python:   [bold]{sys.version.split()[0]}[/]\n"
            f"Platform: [bold]{platform.system()} {platform.release()}[/]\n"
            f"Arch:     [bold]{platform.machine()}[/]",
            title="System",
        )
    )

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
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
            version_str = str(version)

            if _version_ok(version_str, min_ver):
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

    # Summary
    if issues:
        console.print(f"\n[yellow]Found {len(issues)} issue(s):[/]")
        for issue in issues:
            console.print(f"  [red]>[/] {issue}")
        console.print("\n[dim]Fix all: pip install -U " + " ".join(
            f"'{pkg_name}>={min_ver}'"
            for _, pkg_name, min_ver, required in DEPS
            if required
        ) + "[/]")
    else:
        console.print("\n[bold green]All checks passed![/] Your environment is ready.")

    console.print(f"\n[dim]GitHub: [link={GITHUB_URL}]{GITHUB_URL}[/link][/]")


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
                total_gb = total_gb / (1024 ** 3)
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
            console.print(
                Panel(
                    "Backend:  [bold yellow]CPU only[/]\n"
                    "Warning:  Training will be slow without GPU.",
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


def _check_resources():
    """Check RAM and Disk space and display info."""
    import shutil

    table = Table(title="System Resources")
    table.add_column("Resource", style="bold")
    table.add_column("Value")

    # RAM
    ram_str = "Unknown"
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_str = f"{mem.total / (1024 ** 3):.0f} GB"
    except ImportError:
        pass

    if ram_str == "Unknown":
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            ram_str = f"{kb / (1024 ** 2):.0f} GB"
                            break
            except Exception:
                pass
        elif platform.system() == "Darwin":
            try:
                import subprocess
                res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
                if res.returncode == 0:
                    ram_str = f"{int(res.stdout.strip()) / (1024 ** 3):.0f} GB"
            except Exception:
                pass
        elif platform.system() == "Windows":
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
                ram_str = f"{stat.ullTotalPhys / (1024 ** 3):.0f} GB"
            except Exception:
                pass

    table.add_row("RAM", ram_str)

    # Disk
    try:
        usage = shutil.disk_usage(".")
        disk_str = f"{usage.free / (1024 ** 3):.0f} GB"
    except Exception:
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
            "2.6": "0.21", "2.5": "0.20", "2.4": "0.19",
            "2.3": "0.18", "2.2": "0.17", "2.1": "0.16", "2.0": "0.15",
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
