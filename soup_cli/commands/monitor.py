"""v0.44.0 Part A — `soup monitor` GPU live-monitor command.

Renders a Rich panel with one row per detected GPU: Util / Temp / VRAM /
Power. Polls `nvidia-smi` (Linux/Windows/CUDA) at the configured refresh
rate. Apple Silicon variant is a stub note in v0.44.0.
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.gpu_monitor import (
    GpuSample,
    detect_apple_silicon,
    query_nvidia_smi,
)

console = Console()


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:5.1f}%"


def _format_mb(value: float | None) -> str:
    return "—" if value is None else f"{value:7.0f} MB"


def _format_temp(value: float | None) -> str:
    return "—" if value is None else f"{value:4.0f}°C"


def _format_power(value: float | None) -> str:
    return "—" if value is None else f"{value:5.1f} W"


def _build_table(samples: list[GpuSample]) -> Table:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("GPU", justify="right")
    table.add_column("Name", overflow="fold")
    table.add_column("Util", justify="right")
    table.add_column("Mem Util", justify="right")
    table.add_column("VRAM Used", justify="right")
    table.add_column("VRAM Total", justify="right")
    table.add_column("Temp", justify="right")
    table.add_column("Power", justify="right")
    for sample in samples:
        table.add_row(
            str(sample.index),
            escape(sample.name),
            _format_pct(sample.util_gpu_pct),
            _format_pct(sample.util_mem_pct),
            _format_mb(sample.mem_used_mb),
            _format_mb(sample.mem_total_mb),
            _format_temp(sample.temp_c),
            _format_power(sample.power_w),
        )
    return table


def monitor(
    refresh: float = typer.Option(
        2.0,
        "--refresh",
        "-r",
        help="Refresh interval in seconds (0.25 to 30).",
    ),
    once: bool = typer.Option(
        False,
        "--once",
        help="Print one snapshot and exit (skip the live panel).",
    ),
) -> None:
    """Live GPU monitor: Util / Temp / VRAM / Power per GPU.

    Requires nvidia-smi on PATH. On Apple Silicon use Activity Monitor or
    powermetrics — full Apple Silicon support lands in v0.44.1.
    """
    if not (0.25 <= refresh <= 30.0):
        console.print("[red]--refresh must be in [0.25, 30][/]")
        raise typer.Exit(code=2)
    if detect_apple_silicon():
        console.print(
            "[yellow]Apple Silicon detected — `soup monitor` is "
            "Apple-Silicon-aware in v0.44.1.[/]\n"
            "Use Activity Monitor → Window → GPU History for now."
        )
    ok, samples = query_nvidia_smi()
    if not ok:
        console.print(
            "[yellow]nvidia-smi not found or returned non-zero. "
            "Install NVIDIA drivers + CUDA toolkit, or run on a GPU host.[/]"
        )
        raise typer.Exit(code=1)
    if once or not samples:
        console.print(Panel(_build_table(samples), title="Soup GPU Monitor"))
        return
    with Live(
        Panel(_build_table(samples), title="Soup GPU Monitor"),
        refresh_per_second=max(1.0, 1.0 / refresh),
        screen=False,
    ) as live:
        try:
            while True:
                time.sleep(refresh)
                ok, fresh = query_nvidia_smi()
                if not ok:
                    live.update(
                        Panel(
                            "[yellow]nvidia-smi unavailable[/]",
                            title="Soup GPU Monitor",
                        )
                    )
                    continue
                live.update(
                    Panel(_build_table(fresh), title="Soup GPU Monitor")
                )
        except KeyboardInterrupt:
            console.print("[dim]exit[/]")
