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
from collections.abc import Callable, Mapping
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
    weights: Optional[str] = typer.Option(
        None, "--weights",
        help=(
            "Operator-supplied calibrated probe weights (.npz / .npy / "
            ".safetensors). When set, replaces the synthetic seed fallback "
            "and accepts any base id (v0.71.8 #215)."
        ),
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write probe result JSON to path"
    ),
):
    """Apply the sleeper-agent defection probe to activations (v0.66.0).

    Without ``--evidence``, the command lists the probe metadata and exits 0
    (neutral OK report — matches v0.56.0 diagnose behaviour). Pass ``--weights``
    to use a real calibrated probe instead of the synthetic fallback (#215).
    """
    import numpy as np

    from soup_cli.utils.sleeper_probe import (
        BUNDLED_PROBES,
        SleeperProbeResult,
        load_probe_weights,
        render_sleeper_json,
        render_sleeper_markdown,
        run_sleeper_probe,
        validate_base_for_probe,
    )

    # When real weights are supplied the base id is free-form (the operator
    # brings their own probe). Otherwise the bundled allowlist is consulted.
    loaded_weights = None
    if weights is not None:
        try:
            loaded_weights = load_probe_weights(weights)
        except (TypeError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]--weights: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        canonical = base
        spec = None
    else:
        try:
            canonical = validate_base_for_probe(base)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        spec = BUNDLED_PROBES[canonical]

    if evidence is None and spec is not None:
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

    if evidence is None:
        # Reachable only with --weights set (the metadata panel needs a spec).
        console.print(
            "[red]--weights requires --evidence <activations.json> to run.[/]"
        )
        raise typer.Exit(2)

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
        result = run_sleeper_probe(activations, canonical, weights=loaded_weights)
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
# soup probe truth / harm <base>  (v0.71.8 #217)
# ---------------------------------------------------------------------------


def _run_kind_probe_cli(
    base: str,
    evidence: Optional[str],
    weights: Optional[str],
    output: Optional[str],
    *,
    kind: str,
    bundled: Mapping[str, Any],
    validate_base: Callable[[Any], str],
    run_probe: Callable[..., Any],
    render_json: Callable[[Any], str],
    render_markdown: Callable[[Any], str],
) -> None:
    """Shared CLI body for ``soup probe truth`` / ``soup probe harm`` (#217).

    Mirrors the sleeper command: no ``--evidence`` lists the probe metadata +
    emits a neutral OK report (matches v0.56.0 diagnose); ``--weights`` loads a
    real calibrated probe (any base id); exit 2 on a MAJOR verdict.
    """
    import numpy as np

    from soup_cli.utils.probe_kernel import ProbeResult, load_probe_weights

    loaded_weights = None
    spec = None
    if weights is not None:
        try:
            loaded_weights = load_probe_weights(weights)
        except (TypeError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]--weights: {escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        canonical = base
    else:
        try:
            canonical = validate_base(base)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        spec = bundled[canonical]

    if evidence is None and spec is not None:
        console.print(Panel(
            f"Base: [bold]{escape(canonical)}[/]\n"
            f"Hidden dim: {spec.hidden_dim}\n"
            f"Threshold: {spec.threshold:.3f}\n"
            f"Bands: 5% / 20% (OK / MINOR / MAJOR)\n"
            f"Description: {escape(spec.description)}\n\n"
            "[dim]Pass --evidence <activations.json> to run the probe.[/]",
            title=f"{kind.capitalize()} probe (no evidence)",
        ))
        neutral = ProbeResult(
            kind=kind, base=canonical, num_tokens=0, flag_rate=0.0,
            max_score=0.0, verdict="OK",
        )
        if output is not None:
            try:
                atomic_write_text(render_json(neutral), output, field="--output")
            except (TypeError, ValueError) as exc:
                console.print(f"[red]{escape(str(exc))}[/]")
                raise typer.Exit(2) from exc
            console.print(f"[green]Wrote neutral report → {escape(output)}[/]")
        return

    if evidence is None:
        console.print(
            "[red]--weights requires --evidence <activations.json> to run.[/]"
        )
        raise typer.Exit(2)

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
        result = run_probe(activations, canonical, weights=loaded_weights)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(render_markdown(result))

    if output is not None:
        try:
            atomic_write_text(render_json(result), output, field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote probe result → {escape(output)}[/]")

    if result.verdict == "MAJOR":
        raise typer.Exit(2)


@app.command(name="truth")
def truth(
    base: str = typer.Argument(..., help="Base model id (must have a bundled probe)"),
    evidence: Optional[str] = typer.Option(
        None, "--evidence",
        help="JSON with 'activations': [[...], ...] (2D float matrix)",
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights",
        help="Operator-supplied calibrated probe weights (.npz / .npy / .safetensors)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write probe result JSON to path"
    ),
):
    """Apply the TruthfulQA-style honesty probe to activations (v0.71.8 #217)."""
    from soup_cli.utils.truth_probe import (
        BUNDLED_TRUTH_PROBES,
        render_truth_json,
        render_truth_markdown,
        run_truth_probe,
        validate_base_for_truth,
    )

    _run_kind_probe_cli(
        base, evidence, weights, output, kind="truth",
        bundled=BUNDLED_TRUTH_PROBES, validate_base=validate_base_for_truth,
        run_probe=run_truth_probe, render_json=render_truth_json,
        render_markdown=render_truth_markdown,
    )


@app.command(name="harm")
def harm(
    base: str = typer.Argument(..., help="Base model id (must have a bundled probe)"),
    evidence: Optional[str] = typer.Option(
        None, "--evidence",
        help="JSON with 'activations': [[...], ...] (2D float matrix)",
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights",
        help="Operator-supplied calibrated probe weights (.npz / .npy / .safetensors)",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write probe result JSON to path"
    ),
):
    """Apply the HarmBench-style misuse probe to activations (v0.71.8 #217)."""
    from soup_cli.utils.harm_probe import (
        BUNDLED_HARM_PROBES,
        render_harm_json,
        render_harm_markdown,
        run_harm_probe,
        validate_base_for_harm,
    )

    _run_kind_probe_cli(
        base, evidence, weights, output, kind="harm",
        bundled=BUNDLED_HARM_PROBES, validate_base=validate_base_for_harm,
        run_probe=run_harm_probe, render_json=render_harm_json,
        render_markdown=render_harm_markdown,
    )


# ---------------------------------------------------------------------------
# soup probe interference <losses.json>   (--measure: v0.71.8 #218)
# ---------------------------------------------------------------------------

_MAX_MEASURE_ROWS = 10_000


def _load_jsonl_rows(path: str, *, field: str) -> list:
    """Cwd-contained, symlink-rejected JSONL loader (O_NOFOLLOW + size cap)."""
    canonical = enforce_under_cwd_and_no_symlink(path, field)
    rows: list = []
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(canonical, flags)
    # File-size cap on the SAME fd bounds a pathological newline-free line.
    if os.fstat(fd).st_size > 256 * 1024 * 1024:
        os.close(fd)
        raise ValueError(f"{field} file exceeds 256 MiB")
    with os.fdopen(fd, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= _MAX_MEASURE_ROWS:
                break
    return rows


def _parse_adapter_specs(specs: Optional[list[str]]) -> dict[str, str]:
    """Parse ``name=path`` adapter specs; cwd-contain each path."""
    from soup_cli.utils.paths import is_under_cwd

    if not specs:
        raise typer.BadParameter(
            "--measure requires ≥2 --adapter name=path specs"
        )
    out: dict[str, str] = {}
    for spec in specs:
        if not isinstance(spec, str) or "=" not in spec:
            raise typer.BadParameter(
                f"--adapter {spec!r}: expected name=path"
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise typer.BadParameter(f"--adapter {spec!r}: name and path required")
        if name in out:
            raise typer.BadParameter(f"--adapter: duplicate name {name!r}")
        if not is_under_cwd(path):
            raise typer.BadParameter(
                f"--adapter {name!r}: path must stay under the cwd"
            )
        out[name] = path
    return out


def _emit_matrix(matrix: Any, output: Optional[str]) -> None:
    """Render + write the interference matrix and exit 2 on MAJOR worst-pair."""
    from soup_cli.utils.interference import render_matrix_json, render_matrix_markdown

    console.print(render_matrix_markdown(matrix))
    if output is not None:
        try:
            atomic_write_text(render_matrix_json(matrix), output, field="--output")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Wrote matrix → {escape(output)}[/]")
    if matrix.worst_pair is not None and abs(matrix.worst_score) >= 0.20:
        raise typer.Exit(2)


def _run_interference_measure(
    measure: str,
    base_model: Optional[str],
    adapter_specs: Optional[list[str]],
    device: Optional[str],
    output: Optional[str],
) -> None:
    """Auto-measure interference losses then build + emit the matrix (#218)."""
    from soup_cli.utils.interference import build_interference_matrix

    if not base_model:
        console.print("[red]--measure requires --base-model <id>.[/]")
        raise typer.Exit(2)
    try:
        adapters = _parse_adapter_specs(adapter_specs)
    except typer.BadParameter as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        eval_rows = _load_jsonl_rows(measure, field="--measure")
    except (OSError, ValueError) as exc:
        console.print(f"[red]--measure: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    if not eval_rows:
        console.print("[red]--measure: eval suite has no usable rows.[/]")
        raise typer.Exit(2)

    from soup_cli.utils.interference_live import measure_interference_losses

    console.print(
        f"[cyan]--measure:[/] loading {len(adapters)} adapters on "
        f"{escape(base_model)} and measuring {len(eval_rows)} eval rows ..."
    )
    try:
        losses = measure_interference_losses(
            base_model, adapters, eval_rows, device=device
        )
    except (TypeError, ValueError, RuntimeError, ImportError) as exc:
        console.print(f"[red]--measure failed: {escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        matrix = build_interference_matrix(tuple(adapters), losses)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    _emit_matrix(matrix, output)


@app.command(name="interference")
def interference(
    losses: Optional[str] = typer.Argument(
        None, help="JSON: {adapters: [...], losses: {'a|b': loss_value, ...}}"
    ),
    measure: Optional[str] = typer.Option(
        None, "--measure",
        help=(
            "Auto-measure losses against this eval-suite JSONL instead of "
            "supplying a losses JSON. Requires --base-model + ≥2 --adapter "
            "(v0.71.8 #218)."
        ),
    ),
    base_model: Optional[str] = typer.Option(
        None, "--base-model", help="Base model id for --measure"
    ),
    adapter: Optional[list[str]] = typer.Option(
        None, "--adapter",
        help="name=path adapter spec for --measure (repeatable)",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device for --measure (cpu / cuda)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write matrix JSON to path"
    ),
):
    """Pairwise N×N adapter interference matrix (v0.66.0; --measure v0.71.8 #218).

    Two input modes:

    1. Operator-supplied losses JSON (positional): ``{"adapters": ["a", "b"],
       "losses": {"a|a": 1.0, "a|b": 1.05, ...}}`` where ``"a|b"`` is the loss
       on A's domain with both A AND B loaded and ``"a|a"`` is the baseline.
    2. ``--measure <eval_suite.jsonl> --base-model <id> --adapter a=path
       --adapter b=path`` auto-measures the losses by loading each adapter
       (and each co-loaded pair) and running the eval suite.
    """
    if measure is not None:
        _run_interference_measure(measure, base_model, adapter, device, output)
        return

    if losses is None:
        console.print(
            "[red]Provide a losses JSON, or use --measure <eval_suite> "
            "--base-model <id> --adapter name=path ...[/]"
        )
        raise typer.Exit(2)

    from soup_cli.utils.interference import build_interference_matrix

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

    _emit_matrix(matrix, output)


# ---------------------------------------------------------------------------
# soup probe sae-diff <sae> <pre.json> <post.json>
# ---------------------------------------------------------------------------


@app.command(name="sae-diff")
def sae_diff(
    sae: str = typer.Argument(
        ..., help="Path to SAE safetensors checkpoint (or repo id with --auto-download)"
    ),
    pre_acts: str = typer.Argument(..., help="JSON: {'activations': [[...]]}"),
    post_acts: str = typer.Argument(..., help="JSON: {'activations': [[...]]}"),
    top_k: int = typer.Option(20, "--top-k", min=1, max=10_000,
                              help="Top-K changed features to report"),
    auto_download: bool = typer.Option(
        False, "--auto-download",
        help=(
            "Treat <sae> as an HF Hub repo id and download it into "
            "~/.soup/sae-cache/ first (must be in HF_HUB_ALLOWLIST; v0.71.8 #216)."
        ),
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write diff report JSON to path"
    ),
):
    """SAE feature diff between pre- and post-FT activations (v0.66.0)."""
    import numpy as np

    from soup_cli.utils.sae_diff import (
        compute_sae_diff,
        download_sae,
        load_sae_weights,
        render_report_json,
        render_report_markdown,
    )

    try:
        if auto_download:
            console.print(
                f"[cyan]--auto-download:[/] fetching SAE {escape(sae)} "
                "into ~/.soup/sae-cache/ ..."
            )
            sae_weights = download_sae(sae)
        else:
            sae_weights = load_sae_weights(sae)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except (TypeError, ValueError, RuntimeError, KeyError, ImportError) as exc:
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
