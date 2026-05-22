"""soup probe — post-train activation probes (v0.66.0).

Sub-commands:

- ``soup probe pack <base>``     — list calibrated probes for a base
- ``soup probe sleeper <run-id>``— defection probe (Part C)
- ``soup probe interference``    — pairwise adapter interference (Part D)
- ``soup probe sae-diff``        — SAE feature diff (Part A)

Composes with ``soup diagnose`` (v0.56.0) — the four probes here are
the v0.66.0 "Post-train X-rays" extension to the v0.56 diagnose surface.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Post-train activation probes (v0.66.0).",
)


# 16 MiB caps mirror v0.56.0 diagnose evidence policy.
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _read_json_evidence(path: str, *, field: str) -> Mapping[str, Any]:
    """Cwd-contained + symlink-rejected + size-capped JSON evidence loader.

    Mirrors v0.56.0 diagnose / v0.65.0 evidence loader policy. Returns a
    Mapping (not dict) to signal callers must not mutate the payload —
    review fix M4 (v0.66.0).
    """
    enforce_under_cwd_and_no_symlink(path, field)
    real = os.path.realpath(path)
    try:
        size = os.path.getsize(real)
    except OSError as exc:
        raise typer.BadParameter(
            f"{field}: cannot read — {type(exc).__name__}"
        ) from exc
    if size > _MAX_EVIDENCE_BYTES:
        raise typer.BadParameter(
            f"{field}: file too large ({size} > {_MAX_EVIDENCE_BYTES} bytes)"
        )
    try:
        with open(real, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{field}: invalid JSON — {exc}") from exc
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{field}: must be a JSON object (dict)")
    return data


# ---------------------------------------------------------------------------
# soup probe pack <base>
# ---------------------------------------------------------------------------


@app.command(name="pack")
def pack(
    base: Optional[str] = typer.Argument(
        None, help="Base model id (e.g. meta-llama/Llama-3-8B)"
    ),
    list_bases: bool = typer.Option(
        False, "--list", help="List all bases that ship a probe pack"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write pack manifest JSON to path"
    ),
):
    """Assemble or list calibrated probe packs for a base (v0.66.0)."""
    from soup_cli.utils.probe_pack import (
        get_probe_pack,
        list_probe_bases,
        render_pack_json,
        render_pack_markdown,
    )

    if list_bases:
        bases = list_probe_bases()
        table = Table(title="Probe pack bases")
        table.add_column("Base", style="bold")
        for b in bases:
            table.add_row(escape(b))
        console.print(table)
        return

    if base is None:
        console.print(
            "[red]Pass <base> or --list (see `soup probe pack --help`).[/]"
        )
        raise typer.Exit(2)

    try:
        result = get_probe_pack(base)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(render_pack_markdown(result))

    if output is not None:
        try:
            atomic_write_text(render_pack_json(result), output,
                              field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Cannot write --output: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote probe pack → {escape(output)}[/]")


# ---------------------------------------------------------------------------
# soup probe sleeper <base>
# ---------------------------------------------------------------------------


@app.command(name="sleeper")
def sleeper(
    base: str = typer.Argument(..., help="Base model id (must have a bundled probe)"),
    evidence: Optional[str] = typer.Option(
        None, "--evidence",
        help="JSON with 'activations': [[...], ...] (2D float matrix)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write probe result JSON to path"
    ),
):
    """Apply the bundled sleeper-agent defection probe to activations (v0.66.0).

    Without ``--evidence``, the command lists the probe metadata and exits 0
    (neutral OK report — matches v0.56.0 diagnose behaviour).
    """
    import numpy as np

    from soup_cli.utils.sleeper_probe import (
        BUNDLED_PROBES,
        SleeperProbeResult,
        render_sleeper_json,
        render_sleeper_markdown,
        run_sleeper_probe,
        validate_base_for_probe,
    )

    try:
        canonical = validate_base_for_probe(base)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    spec = BUNDLED_PROBES[canonical]
    if evidence is None:
        console.print(Panel(
            f"Base: [bold]{escape(canonical)}[/]\n"
            f"Hidden dim: {spec.hidden_dim}\n"
            f"Threshold: {spec.threshold:.3f}\n"
            f"Description: {escape(spec.description)}\n\n"
            "[dim]Pass --evidence <activations.json> to run the probe.[/]",
            title="Sleeper probe (no evidence)",
        ))
        # Emit neutral OK report — composes with v0.56 diagnose policy.
        neutral = SleeperProbeResult(
            base=canonical,
            num_tokens=0,
            defection_rate=0.0,
            max_score=0.0,
            verdict="OK",
        )
        if output is not None:
            try:
                atomic_write_text(
                    render_sleeper_json(neutral), output, field="--output"
                )
            except (TypeError, ValueError) as exc:
                console.print(f"[red]{escape(str(exc))}[/]")
                raise typer.Exit(2) from exc
            console.print(f"[green]Wrote neutral report → {escape(output)}[/]")
        return

    try:
        payload = _read_json_evidence(evidence, field="--evidence")
    except typer.BadParameter as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    acts_raw = payload.get("activations")
    if acts_raw is None:
        console.print("[red]--evidence: missing 'activations' field[/]")
        raise typer.Exit(2)
    try:
        activations = np.asarray(acts_raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]--evidence: invalid activations — {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        result = run_sleeper_probe(activations, canonical)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(render_sleeper_markdown(result))

    if output is not None:
        try:
            atomic_write_text(render_sleeper_json(result), output,
                              field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote probe result → {escape(output)}[/]")

    if result.verdict == "MAJOR":
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# soup probe interference <losses.json>
# ---------------------------------------------------------------------------


@app.command(name="interference")
def interference(
    losses: str = typer.Argument(
        ..., help="JSON: {adapters: [...], losses: {'a|b': loss_value, ...}}"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write matrix JSON to path"
    ),
):
    """Pairwise N×N adapter interference matrix from measured losses (v0.66.0).

    Input JSON shape:

      {"adapters": ["a", "b", "c"],
       "losses": {"a|a": 1.0, "a|b": 1.05, ..., "c|c": 1.2}}

    Keys ``"a|b"`` mean: loss measured on adapter A's domain when both
    A AND B are loaded. Diagonal entries ``"a|a"`` are the baseline.
    """
    from soup_cli.utils.interference import (
        build_interference_matrix,
        render_matrix_json,
        render_matrix_markdown,
    )

    try:
        payload = _read_json_evidence(losses, field="--losses")
    except typer.BadParameter as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    adapters_raw = payload.get("adapters")
    losses_raw = payload.get("losses")
    if not isinstance(adapters_raw, list) or not isinstance(losses_raw, dict):
        console.print(
            "[red]JSON must have 'adapters' (list) and 'losses' (dict).[/]"
        )
        raise typer.Exit(2)

    # Parse "a|b" keys
    parsed_losses: dict[tuple[str, str], float] = {}
    for key, value in losses_raw.items():
        if not isinstance(key, str) or "|" not in key:
            console.print(
                f"[red]Invalid losses key {key!r}; expected 'a|b' shape.[/]"
            )
            raise typer.Exit(2)
        # H3 review fix (v0.66.0): validate the value is numeric BEFORE
        # inserting into the parsed map. Otherwise a string/list/dict
        # value passes through and crashes deep inside the math kernel
        # with a confusing message.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            console.print(
                f"[red]losses[{escape(key)!r}] must be numeric, "
                f"got {type(value).__name__}[/]"
            )
            raise typer.Exit(2)
        target, co = key.split("|", 1)
        parsed_losses[(target, co)] = float(value)

    try:
        matrix = build_interference_matrix(tuple(adapters_raw), parsed_losses)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(render_matrix_markdown(matrix))

    if output is not None:
        try:
            atomic_write_text(render_matrix_json(matrix), output,
                              field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote matrix → {escape(output)}[/]")

    if matrix.worst_pair is not None and abs(matrix.worst_score) >= 0.20:
        # MAJOR interference — exit non-zero so CI gates can refuse the merge
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# soup probe sae-diff <sae> <pre.json> <post.json>
# ---------------------------------------------------------------------------


@app.command(name="sae-diff")
def sae_diff(
    sae: str = typer.Argument(..., help="Path to SAE safetensors checkpoint"),
    pre_acts: str = typer.Argument(..., help="JSON: {'activations': [[...]]}"),
    post_acts: str = typer.Argument(..., help="JSON: {'activations': [[...]]}"),
    top_k: int = typer.Option(20, "--top-k", min=1, max=10_000,
                              help="Top-K changed features to report"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write diff report JSON to path"
    ),
):
    """SAE feature diff between pre- and post-FT activations (v0.66.0)."""
    import numpy as np

    from soup_cli.utils.sae_diff import (
        compute_sae_diff,
        load_sae_weights,
        render_report_json,
        render_report_markdown,
    )

    try:
        sae_weights = load_sae_weights(sae)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (TypeError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        pre_payload = _read_json_evidence(pre_acts, field="pre_acts")
        post_payload = _read_json_evidence(post_acts, field="post_acts")
    except typer.BadParameter as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        pre = np.asarray(pre_payload.get("activations"), dtype=np.float32)
        post = np.asarray(post_payload.get("activations"), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid activations JSON: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        report = compute_sae_diff(pre, post, sae_weights, top_k=top_k)
    except (TypeError, ValueError, KeyError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(render_report_markdown(report))

    if output is not None:
        try:
            atomic_write_text(render_report_json(report), output,
                              field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote diff → {escape(output)}[/]")
