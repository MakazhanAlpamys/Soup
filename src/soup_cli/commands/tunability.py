"""`soup tunability` — probe-train candidate bases + report Pareto frontier.

CLI surface for the v0.64.0 Part A pre-flight probe. Live LoRA probe lands
in v0.64.1 (see ``utils/tunability._default_probe`` docstring).
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.paths import is_under_cwd
from soup_cli.utils.tunability import (
    DEFAULT_CANDIDATES,
    CandidateBase,
    run_tunability,
    write_report,
)

console = Console()


def _resolve_candidates(names: Optional[str]) -> tuple[CandidateBase, ...]:
    """Resolve a comma-separated name list against ``DEFAULT_CANDIDATES``."""
    if not names:
        return DEFAULT_CANDIDATES
    requested = [n.strip().lower() for n in names.split(",") if n.strip()]
    if not requested:
        return DEFAULT_CANDIDATES
    by_name = {c.name.lower(): c for c in DEFAULT_CANDIDATES}
    resolved: list[CandidateBase] = []
    for name in requested:
        if name not in by_name:
            available = ", ".join(sorted(by_name.keys()))
            raise typer.BadParameter(
                f"unknown candidate {name!r}; known: {available}"
            )
        resolved.append(by_name[name])
    return tuple(resolved)


def tunability_cmd(
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Path to held-out JSONL slice used for the probe.",
    ),
    candidates: Optional[str] = typer.Option(
        None,
        "--candidates",
        help="Comma-separated subset of default candidates by name.",
    ),
    probe_steps: int = typer.Option(
        100,
        "--probe-steps",
        help="Steps per candidate probe (10-10000).",
    ),
    holdout_size: int = typer.Option(
        64,
        "--holdout-size",
        help="Number of holdout rows used for delta scoring (10-100000).",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path for the JSON report.",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Print planned candidates + cost estimate, exit without probing.",
    ),
    list_only: bool = typer.Option(
        False,
        "--list",
        help="List built-in candidate catalogue + exit.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help=(
            "Run a LIVE LoRA probe per candidate (loads each repo + trains "
            "--probe-steps on a tiny held-out slice). Without it, a "
            "deterministic offline heuristic is used."
        ),
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device for the live probe (cuda / cpu). Auto-detected when omitted.",
    ),
) -> None:
    """Probe-train candidate bases + report Pareto frontier (v0.64.0)."""
    if list_only:
        table = Table(title="Default tunability candidates")
        table.add_column("Name")
        table.add_column("Repo")
        table.add_column("Params (B)", justify="right")
        table.add_column("License")
        for c in DEFAULT_CANDIDATES:
            table.add_row(
                escape(c.name),
                escape(c.repo_id),
                f"{c.params_b:.2f}",
                escape(c.license_id),
            )
        console.print(table)
        return

    if not dataset:
        console.print(
            "[red]--dataset is required (use --list to see the catalogue, "
            "--plan-only to dry-run).[/]"
        )
        raise typer.Exit(2)

    # cwd containment + null-byte rejection on dataset
    if "\x00" in dataset:
        console.print("[red]dataset path must not contain null bytes[/]")
        raise typer.Exit(2)
    if not is_under_cwd(dataset):
        console.print(f"[red]dataset {escape(dataset)!r} is outside cwd[/]")
        raise typer.Exit(2)

    try:
        cands = _resolve_candidates(candidates)
    except typer.BadParameter as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    if plan_only:
        table = Table(title="Planned tunability sweep")
        table.add_column("Name")
        table.add_column("Params (B)", justify="right")
        table.add_column("License")
        for c in cands:
            table.add_row(
                escape(c.name),
                f"{c.params_b:.2f}",
                escape(c.license_id),
            )
        console.print(table)
        probe_kind = "LIVE LoRA probe" if live else "offline heuristic"
        console.print(
            Panel(
                f"[yellow]Plan-only.[/] Would run {len(cands)} probes "
                f"x {probe_steps} steps on holdout={holdout_size}.\n"
                f"Probe: {probe_kind}. Pass --live for a real LoRA probe.",
                title="tunability",
                border_style="yellow",
            )
        )
        return

    probe_fn = None
    if live:
        from soup_cli.utils.tunability import live_lora_probe

        def _live_probe_fn(cand, ds, *, probe_steps, holdout_size):  # noqa: ANN001
            return live_lora_probe(
                cand,
                ds,
                probe_steps=probe_steps,
                holdout_size=holdout_size,
                device=device,
            )

        probe_fn = _live_probe_fn

    try:
        report = run_tunability(
            candidates=cands,
            dataset_path=dataset,
            probe_steps=probe_steps,
            holdout_size=holdout_size,
            probe_fn=probe_fn,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    table = Table(title="Tunability sweep")
    table.add_column("Name")
    table.add_column("Delta", justify="right")
    table.add_column("Wall (s)", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("License")
    on_frontier = {r.candidate.name for r in report.frontier}
    for r in report.results:
        marker = "[green]*[/]" if r.candidate.name in on_frontier else " "
        table.add_row(
            f"{marker} {escape(r.candidate.name)}",
            f"{r.delta:+.4f}",
            f"{r.wall_clock_seconds:.0f}",
            f"{r.estimated_cost_usd:.4f}",
            escape(r.candidate.license_id),
        )
    console.print(table)
    console.print(
        Panel(
            f"Pareto frontier: {len(report.frontier)} / {len(report.results)} "
            "candidates. Delta is base_loss - probe_loss (higher = better).",
            title="tunability",
            border_style="green",
        )
    )

    if output:
        if "\x00" in output:
            console.print("[red]output path must not contain null bytes[/]")
            raise typer.Exit(2)
        if not is_under_cwd(output):
            console.print(f"[red]output {escape(output)!r} is outside cwd[/]")
            raise typer.Exit(2)
        try:
            write_report(report, output)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        console.print(f"[green]Report written to {escape(output)}[/]")


__all__ = ["tunability_cmd"]
