"""soup airgap-bundle — build a one-shot offline tarball (v0.60.0 Part F)."""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


def airgap_bundle(
    output: str = typer.Option(..., "--output", "-o",
                               help="Output tarball path (cwd-contained)"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    dataset: Optional[List[str]] = typer.Option(
        None, "--dataset",
        help="Dataset directory (repeatable)",
    ),
    wheel: Optional[List[str]] = typer.Option(
        None, "--wheel",
        help="Wheel directory (repeatable)",
    ),
    kernel: Optional[List[str]] = typer.Option(
        None, "--kernel",
        help="CUDA / kernel directory (repeatable)",
    ),
    bundle_size_cap: float = typer.Option(
        100.0, "--bundle-size-cap",
        help="Cap in GiB (default 100). Build aborts when exceeded.",
    ),
):
    """Build a signed tarball with model + datasets + wheels + kernels (v0.60.0).

    Designed for one-way physical-media transfer through a data diode.
    Refuses to write a bundle larger than ``--bundle-size-cap`` GiB.
    Manifest is embedded inside as ``manifest.json`` with SHA-256 per file.
    """
    from soup_cli.utils.airgap_bundle import (
        AirgapBundlePlan,
        build_airgap_bundle,
    )

    if bundle_size_cap <= 0:
        console.print("[red]--bundle-size-cap must be > 0[/]")
        raise typer.Exit(2)
    cap_bytes_float = float(bundle_size_cap) * 1024 * 1024 * 1024
    # Convert to int; allow fractional cap to address sub-GiB tests.
    cap_bytes = int(cap_bytes_float)
    if cap_bytes < 1:
        cap_bytes = 1

    try:
        plan = AirgapBundlePlan(
            output=output,
            model_dir=model,
            dataset_dirs=tuple(dataset or ()),
            wheel_dirs=tuple(wheel or ()),
            kernel_dirs=tuple(kernel or ()),
            bundle_size_cap_bytes=cap_bytes,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid plan: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        manifest = build_airgap_bundle(plan)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Output:      [bold]{escape(output)}[/]\n"
            f"Model:       [bold]{escape(manifest.model_dir)}[/]\n"
            f"Datasets:    {len(manifest.datasets)}\n"
            f"Wheels:      {len(manifest.wheels)}\n"
            f"Kernels:     {len(manifest.kernels)}\n"
            f"Files:       {len(manifest.files)}\n"
            f"Total bytes: {manifest.total_bytes}\n"
            f"Cap (GiB):   {bundle_size_cap}\n"
            f"Soup ver:    {escape(manifest.soup_version)}\n"
            f"Created:     [dim]{escape(manifest.created_at)}[/]",
            title="Airgap bundle",
        )
    )
