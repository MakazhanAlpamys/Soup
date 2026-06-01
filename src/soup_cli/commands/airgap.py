"""soup airgap-bundle — build a one-shot offline tarball (v0.60.0 Part F)."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()

_MAX_RECEIPT_BYTES = 16 * 1024 * 1024  # 16 MiB


def _load_receipt_json(path: str) -> dict:
    """Load + validate a receipt JSON (cwd-contained, symlink-rejected, capped)."""
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    real = enforce_under_cwd_and_no_symlink(path, "repro-receipt")
    if os.path.getsize(real) > _MAX_RECEIPT_BYTES:
        raise ValueError(
            f"repro-receipt too large (> {_MAX_RECEIPT_BYTES} bytes)"
        )
    with open(real, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("repro-receipt must be a JSON object")
    return data


def _resolve_repro_receipt(explicit: Optional[str], model_dir: str) -> Optional[dict]:
    """Resolve the receipt dict: explicit flag, else auto-detect in model dir.

    An explicit ``--repro-receipt`` load failure is a hard error (exit 2). The
    auto-detected ``<model>/repro-receipt.json`` path is best-effort: any
    failure silently yields ``None`` (the bundle is still built without it).
    """
    if explicit:
        try:
            return _load_receipt_json(explicit)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            console.print(f"[red]--repro-receipt {escape(explicit)}: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
    # Auto-detect — best-effort. Note: when the receipt lives inside the model
    # dir it is embedded twice (top-level ``repro-receipt.json`` AND
    # ``model/repro-receipt.json`` via the model-dir walk). Both are faithful
    # byte-copies with distinct arcnames; the duplication is a few KB and
    # acceptable.
    candidate = os.path.join(model_dir, "repro-receipt.json")
    if os.path.isfile(candidate):
        try:
            return _load_receipt_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    return None


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
    repro_receipt: Optional[str] = typer.Option(
        None, "--repro-receipt",
        help=(
            "Embed a reproducibility receipt JSON (from `soup train "
            "--repro-receipt`) into the bundle. Auto-detected from "
            "<model>/repro-receipt.json when not given. v0.71.3 #188."
        ),
    ),
):
    """Build a signed tarball with model + datasets + wheels + kernels (v0.60.0).

    Designed for one-way physical-media transfer through a data diode.
    Refuses to write a bundle larger than ``--bundle-size-cap`` GiB.
    Manifest is embedded inside as ``manifest.json`` with SHA-256 per file.
    When a reproducibility receipt is supplied (or auto-detected in the model
    dir) it is embedded as ``repro-receipt.json`` (v0.71.3 #188).
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

    receipt = _resolve_repro_receipt(repro_receipt, model)

    try:
        plan = AirgapBundlePlan(
            output=output,
            model_dir=model,
            dataset_dirs=tuple(dataset or ()),
            wheel_dirs=tuple(wheel or ()),
            kernel_dirs=tuple(kernel or ()),
            bundle_size_cap_bytes=cap_bytes,
            repro_receipt=receipt,
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
