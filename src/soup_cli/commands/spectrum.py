"""soup spectrum — native Spectrum targeted-training scan (#266, v0.71.23).

``soup spectrum scan`` streams a model's ``.safetensors`` shards (NO model
load), computes a singular-value SNR per weight matrix (arXiv:2406.06623),
ranks layers within each module-type group and prints the top ``--top-percent``
as a ready-to-paste ``training.unfrozen_parameters`` YAML block. Paste it into
a ``soup.yaml`` to fine-tune only the high-SNR layers (full FT, LoRA off).

The scan is pure-numpy and runs on a CPU box even for very large models —
peak RSS is the largest single weight matrix, not the whole model.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # static types only — the import is numpy-lazy at runtime
    from soup_cli.utils.spectrum_scan import LayerSNR, ScanResult

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Spectrum SNR scan for targeted training (v0.71.23).",
)


@app.callback()
def _spectrum() -> None:
    """Spectrum SNR scan for targeted training (v0.71.23).

    An explicit callback keeps this a command group (Typer otherwise collapses
    a single-command app and drops the ``scan`` subcommand name).
    """


def _render_table(
    result: "ScanResult", kept: list[str], top_percent: float
) -> None:
    # Group by the same key as select_unfrozen_parameters (the layer-type
    # signature), so the "Kept" column can never desync from the selection.
    from soup_cli.utils.spectrum_scan import param_prefix

    kept_set = set(kept)
    groups: dict[str, list["LayerSNR"]] = defaultdict(list)
    for layer in result.layers:
        groups[layer.group].append(layer)

    table = Table(
        title=f"Spectrum SNR — {escape(result.model)} "
        f"(top {top_percent:g}% per group)"
    )
    table.add_column("Type")
    table.add_column("Group")
    table.add_column("Layers", justify="right")
    table.add_column("Kept", justify="right")
    table.add_column("SNR min / mean / max", justify="right")

    for group in sorted(groups):
        items = groups[group]
        snrs = [layer.snr for layer in items]
        n_kept = sum(1 for layer in items if param_prefix(layer.name) in kept_set)
        table.add_row(
            items[0].module_type,
            escape(group),
            str(len(items)),
            str(n_kept),
            f"{min(snrs):.3f} / {sum(snrs) / len(snrs):.3f} / {max(snrs):.3f}",
        )
    console.print(table)


@app.command()
def scan(
    model: str = typer.Option(
        ..., "--model", "-m", help="HF Hub id or local model directory."
    ),
    top_percent: float = typer.Option(
        50.0,
        "--top-percent",
        help="Keep the top N% of layers (by SNR) within each module type.",
    ),
    modules: str = typer.Option(
        "all",
        "--modules",
        help="Comma list of module types to scan: e.g. 'mlp,attn' — or 'all'.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the YAML patch to this path (must stay under cwd).",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Skip the ~/.soup/spectrum scan cache."
    ),
) -> None:
    """Scan a model's layer SNR and emit a ``training.unfrozen_parameters`` patch."""
    import yaml

    from soup_cli.utils.spectrum_scan import scan_model, select_unfrozen_parameters

    if not (0 < top_percent <= 100):
        console.print("[red]--top-percent must be in (0, 100].[/]")
        raise typer.Exit(2)

    # Validate the output path up front (before the scan work) so a bad
    # destination fails fast rather than after a long scan.
    if output is not None:
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(output, "output")
        except (ValueError, TypeError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

    try:
        result = scan_model(model, modules=modules, use_cache=not no_cache)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        console.print(f"[red]Spectrum scan failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:  # hub validation / unexpected — friendly, no traceback
        console.print(
            f"[red]Spectrum scan failed:[/] "
            f"{escape(type(exc).__name__)}: {escape(str(exc))}"
        )
        raise typer.Exit(1) from exc

    if not result.layers:
        console.print(
            "[yellow]No scannable 2-D weight matrices found "
            "(check --model and --modules).[/]"
        )
        raise typer.Exit(1)

    try:
        kept = select_unfrozen_parameters(
            result.layers, top_percent=top_percent, modules=modules
        )
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    _render_table(result, kept, top_percent)

    yaml_block = yaml.safe_dump(
        {"training": {"unfrozen_parameters": kept}},
        default_flow_style=False,
        sort_keys=False,
    )
    console.print(
        Panel(
            escape(yaml_block.rstrip()),
            title="training.unfrozen_parameters",
            border_style="cyan",
        )
    )
    console.print(
        f"[dim]{len(kept)} parameter group(s) unfrozen — paste the block above "
        f"into your soup.yaml (full fine-tuning, LoRA off).[/]"
    )

    if output is not None:
        from soup_cli.utils.paths import atomic_write_text

        atomic_write_text(yaml_block, output, field="output")
        console.print(f"[green]Wrote patch:[/] {escape(output)}")
