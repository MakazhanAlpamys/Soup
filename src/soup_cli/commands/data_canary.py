"""``soup data canary insert|check`` — Secret-Sharer canaries (v0.71.36).

insert: add K high-entropy secrets to a dataset + write the manifest.
check:  measure whether a model memorized them, by comparing each canary's
        loss against N never-inserted controls from the same secret space.

Exit 0 = OK/MINOR, 2 = MAJOR (a canary was memorized), 1 = runtime error.
Mirrors ``soup diagnose`` / ``soup ship`` exit conventions so CI can gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli.data.loader import load_raw_data
from soup_cli.utils.canary import (
    _harden_permissions,
    build_canary_report,
    canary_report_to_dict,
    canary_rows,
    generate_canaries,
    generate_controls,
    load_manifest,
    write_manifest,
)
from soup_cli.utils.live_eval import compute_pair_losses
from soup_cli.utils.paths import atomic_write_text, is_under_cwd

if TYPE_CHECKING:  # static types only — transformers stays a lazy import
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

console = Console()
app = typer.Typer(
    no_args_is_help=True, help="Dataset canaries (memorization probe)."
)

# Strip C0/DEL before any manifest-derived string reaches the terminal.
# rich.markup.escape() only neutralises Rich's own [...] tag syntax — a raw
# ESC byte survives it. The manifest is explicitly a shareable artifact, so
# a hostile one is a realistic input: an OSC 52 / cursor sequence in a
# `secret` could spoof the title bar or obscure the MAJOR verdict printed
# right below the table. Mirrors commands/data_doctor.py + commands/shrink.py.
# --output JSON is unaffected: json.dumps already \\u00XX-escapes these.
_CONTROL_STRIP_TABLE = {
    i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)
}
_CONTROL_STRIP_TABLE[0x7F] = None


def _for_terminal(text: str) -> str:
    return text.translate(_CONTROL_STRIP_TABLE)

# Fixed seed for the control draw: controls are a null distribution, not a
# secret, so reproducibility is the useful property here.
_CONTROL_SEED = 12345
_VERDICT_COLOUR = {"OK": "green", "MINOR": "yellow", "MAJOR": "red"}


def _load_pair(
    base: str, adapter: Optional[str], device: str
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase, str]":
    """Seam: load ``(model, tokenizer, device)``. Patched in tests."""
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    return load_model_and_tokenizer(base, adapter=adapter, device=device)


def _guard_under_cwd(target: str, label: str) -> None:
    if not is_under_cwd(Path(target)):
        console.print(
            f"[red]{label} path is outside the working directory: "
            f"{escape(str(target))}[/]"
        )
        raise typer.Exit(1)


@app.command()
def insert(
    path: str = typer.Argument(..., help="Path to dataset file"),
    output: str = typer.Option(..., "--output", "-o", help="Output JSONL"),
    manifest: str = typer.Option(
        ..., "--manifest",
        help="Where to write the canary manifest (CONTAINS THE SECRETS)",
    ),
    count: int = typer.Option(16, "--count", "-k", help="Number of canaries"),
    seed: int = typer.Option(0, "--seed", help="Canary generation seed"),
):
    """Insert K unique canaries into a dataset and record them."""
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {escape(str(file_path))}[/]")
        raise typer.Exit(1)
    _guard_under_cwd(output, "Output")
    _guard_under_cwd(manifest, "Manifest")

    rows = load_raw_data(file_path)
    try:
        canaries = generate_canaries(count=count, seed=seed)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)

    mixed = list(rows) + canary_rows(canaries)
    # Manifest FIRST. If the dataset were written first and the manifest
    # then failed (disk full, permissions), a canary-poisoned dataset would
    # survive on disk with nothing left to identify the secrets in it — a
    # user who missed the error and trained on it could never audit what
    # was inserted. Failing before the data is written leaves no artifact.
    try:
        write_manifest(canaries, manifest)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)
    try:
        atomic_write_text(
            "\n".join(json.dumps(row) for row in mixed) + "\n", output
        )
    except (ValueError, OSError) as exc:
        # The manifest now describes canaries that are in no dataset. Say so
        # rather than leaving a manifest that looks authoritative.
        console.print(
            f"[red]Could not write {escape(str(output))}: "
            f"{escape(str(exc))}[/]\n"
            f"[yellow]{escape(str(manifest))} was already written and now "
            "describes canaries that were NOT inserted — delete it or re-run."
            "[/]"
        )
        raise typer.Exit(1)

    console.print(
        f"[green]Inserted {len(canaries)} canaries:[/] "
        f"{len(rows)} -> {len(mixed)} rows\n"
        f"Output: [bold]{escape(output)}[/]\n"
        f"Manifest: [bold]{escape(manifest)}[/]"
    )
    console.print(
        "[yellow]The manifest contains the secret canaries. Do NOT commit it "
        "alongside the dataset it protects — anyone holding it can reproduce "
        "them.[/]"
    )


@app.command()
def check(
    manifest: str = typer.Option(..., "--manifest", help="Canary manifest"),
    base: str = typer.Option(..., "--base", help="Base model id or path"),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", help="LoRA adapter to check"
    ),
    controls: int = typer.Option(
        128, "--controls", help="Never-inserted controls to rank against"
    ),
    device: str = typer.Option("auto", "--device", help="auto/cpu/cuda"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the report as JSON"
    ),
):
    """Check a model for memorization of the manifest's canaries."""
    if output is not None:
        _guard_under_cwd(output, "Output")
    try:
        canaries = load_manifest(manifest)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)
    if not canaries:
        console.print("[red]Manifest contains no canaries.[/]")
        raise typer.Exit(1)

    try:
        control_set = generate_controls(
            count=controls, seed=_CONTROL_SEED,
            exclude={canary.secret for canary in canaries},
        )
    except (ValueError, TypeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)

    try:
        model, tokenizer, dev = _load_pair(base, adapter, device)
    except ImportError:
        console.print(
            "[red]soup data canary check needs PyTorch + transformers.[/]\n"
            # \[train] escaped: Rich would eat the bracket and print a
            # command that installs WITHOUT the extra.
            "Install with: [bold]pip install \"soup-cli\\[train]\"[/]"
        )
        raise typer.Exit(1)
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"[red]Could not load model: {escape(str(exc))}[/]")
        raise typer.Exit(1)

    pairs = [(canary.carrier, canary.secret) for canary in canaries]
    pairs += [(control.carrier, control.secret) for control in control_set]
    losses = compute_pair_losses(model, tokenizer, pairs, device=dev)
    # compute_pair_losses is index-aligned, so the split follows the counts.
    canary_losses = losses[: len(canaries)]
    control_losses = losses[len(canaries):]

    try:
        report = build_canary_report(
            canary_losses, control_losses,
            [canary.secret for canary in canaries],
        )
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1)

    table = Table(
        title=f"Canary exposure — {len(canaries)} canaries vs "
              f"{report.n_controls} controls"
    )
    table.add_column("Canary")
    table.add_column("Loss", justify="right")
    table.add_column("Percentile", justify="right")
    table.add_column("Memorized", justify="right")
    for exposure in report.exposures:
        loss_text = (
            "nan" if math.isnan(exposure.loss) else f"{exposure.loss:.4f}"
        )
        table.add_row(
            escape(_for_terminal(exposure.secret.strip())),
            loss_text,
            f"{exposure.percentile * 100:.1f}%",
            "[red]YES[/]" if exposure.memorized else "no",
        )
    console.print(table)
    colour = _VERDICT_COLOUR[report.verdict]
    console.print(f"[{colour}]Verdict: {report.verdict}[/]")

    if output is not None:
        # The report embeds every secret, so it is as sensitive as the
        # manifest and gets the same 0600 + warning. `insert` warns about
        # the manifest; without this the report would be the quiet leak.
        atomic_write_text(
            json.dumps(canary_report_to_dict(report), indent=2), output
        )
        _harden_permissions(output)
        console.print(f"[green]Report written:[/] [bold]{escape(output)}[/]")
        console.print(
            "[yellow]The report lists the canary secrets — treat it like the "
            "manifest and do not commit it.[/]"
        )

    if report.verdict == "MAJOR":
        raise typer.Exit(2)
