"""v0.44.0 Part A — `soup monitor` GPU live-monitor command.

Renders a Rich panel with one row per detected GPU: Util / Temp / VRAM /
Power. Polls `nvidia-smi` (Linux/Windows/CUDA) at the configured refresh
rate, or `powermetrics` on Apple Silicon.
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
    PowermetricsStatus,
    detect_apple_silicon,
    query_nvidia_smi,
    query_powermetrics,
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


def _powermetrics_error(status: PowermetricsStatus) -> str:
    if status is PowermetricsStatus.PERMISSION_DENIED:
        return (
            "[yellow]Apple GPU metrics require permission to run "
            "`/usr/bin/powermetrics`.[/]\n"
            "Run [bold]sudo -v[/] in a terminal, then rerun `soup monitor`. "
            "Soup uses `sudo -n`: it never prompts for or reads your password."
        )
    if status is PowermetricsStatus.UNAVAILABLE:
        return (
            "[yellow]`/usr/bin/powermetrics` is unavailable on this Mac. "
            "Use Activity Monitor → Window → GPU History.[/]"
        )
    if status is PowermetricsStatus.INVALID_OUTPUT:
        return (
            "[yellow]`powermetrics` returned no usable Apple GPU sample. "
            "Its plist schema may not expose `gpu_power` on this Mac.[/]"
        )
    return "[yellow]`powermetrics` failed or timed out while reading Apple GPU metrics.[/]"


def _monitor_apple_silicon(refresh: float, once: bool) -> None:
    title = "Soup GPU Monitor — Apple Silicon"
    result = query_powermetrics(refresh)
    if result.status is not PowermetricsStatus.OK:
        console.print(_powermetrics_error(result.status))
        raise typer.Exit(code=1)

    samples = list(result.samples)
    if once or not samples:
        console.print(Panel(_build_table(samples), title=title))
        return

    with Live(
        Panel(_build_table(samples), title=title),
        refresh_per_second=max(1.0, 1.0 / refresh),
        screen=False,
        console=console,
    ) as live:
        try:
            while True:
                # powermetrics itself blocks for the requested sample window;
                # sleeping here as well would double the refresh interval.
                fresh = query_powermetrics(refresh)
                if fresh.status is not PowermetricsStatus.OK:
                    live.update(Panel(_powermetrics_error(fresh.status), title=title))
                    # Failure paths can return immediately; avoid a busy loop.
                    time.sleep(refresh)
                    continue
                live.update(Panel(_build_table(list(fresh.samples)), title=title))
        except KeyboardInterrupt:
            console.print("[dim]exit[/]")


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

    Uses nvidia-smi on NVIDIA systems and powermetrics on Apple Silicon.
    """
    if not (0.25 <= refresh <= 30.0):
        console.print("[red]--refresh must be in [0.25, 30][/]")
        raise typer.Exit(code=2)
    if detect_apple_silicon():
        _monitor_apple_silicon(refresh, once)
        return
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
