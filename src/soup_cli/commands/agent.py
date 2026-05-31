"""soup agent — Agent Forge: spec → tool-calling SFT dataset / train / eval.

v0.46.0 Part B. Live ``train`` and ``eval`` wrappers print the planned
sub-command rather than re-entering the Typer app in-process (matches the
``soup quantize`` design from v0.44.0 Part D).
"""

from __future__ import annotations

import json
import shlex
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=True)


@app.command()
def synth(
    spec: str = typer.Option(
        ..., "--spec", "-s",
        help="Path to OpenAPI / MCP / GraphQL spec (YAML or JSON, under cwd).",
    ),
    output: str = typer.Option(
        "agent_dataset.jsonl", "--output", "-o",
        help="Where to write the synthesised JSONL dataset (under cwd).",
    ),
    kind: Optional[str] = typer.Option(
        None, "--kind", "-k",
        help="Spec kind override: openapi | mcp | graphql. Auto-detected if omitted.",
    ),
    examples_per_endpoint: int = typer.Option(
        1, "--examples-per-endpoint", "-n", min=1, max=32,
        help="Number of synthetic rows to emit per endpoint.",
    ),
):
    """Parse an API spec and synthesise a tool-calling SFT dataset."""
    from soup_cli.utils.agent_forge import (
        load_spec_file,
        parse_spec,
        synthesise_dataset,
        write_dataset,
    )

    try:
        spec_dict = load_spec_file(spec)
    except (ValueError, FileNotFoundError, TypeError) as exc:
        console.print(f"[red]Spec load failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001 — yaml/json parse errors
        console.print(f"[red]Spec parse failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    try:
        endpoints, report = parse_spec(spec_dict, kind=kind)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Parse failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not endpoints:
        console.print(
            "[yellow]No endpoints discovered.[/] "
            f"Detected kind: [bold]{escape(report.spec_kind)}[/]."
        )
        for w in report.warnings[:5]:
            console.print(f"  [dim]- {escape(w)}[/]")
        raise typer.Exit(1)

    rows = synthesise_dataset(endpoints, examples_per_endpoint=examples_per_endpoint)
    try:
        out_path = write_dataset(rows, output)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Write failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    table = Table(title=f"Agent Forge — {escape(report.spec_kind)}")
    table.add_column("Tool", style="bold cyan")
    table.add_column("Method", style="magenta")
    table.add_column("Path")
    # ep.path is partly user-controlled (from the spec) — escape every cell
    # before handing to Rich Table (matches v0.43.0 Part B Tournament policy).
    for ep in endpoints[:20]:
        table.add_row(escape(ep.tool), escape(ep.method), escape(ep.path))
    console.print(table)
    if len(endpoints) > 20:
        console.print(f"[dim]... and {len(endpoints) - 20} more[/]")

    console.print(
        Panel(
            f"Spec kind:  [bold]{escape(report.spec_kind)}[/]\n"
            f"Endpoints:  [bold]{report.endpoint_count}[/] "
            f"(skipped duplicates: {report.skipped})\n"
            f"Rows:       [bold]{len(rows)}[/]\n"
            f"Output:     [bold]{escape(out_path)}[/]",
            title="[bold green]Agent Forge — synth complete[/]",
        )
    )
    for w in report.warnings[:5]:
        console.print(f"[yellow]warning:[/] {escape(w)}")


@app.command()
def train(
    spec: str = typer.Option(..., "--spec", "-s", help="API spec (under cwd)."),
    base: str = typer.Option(
        ..., "--base", "-b",
        help="Base model HF repo id to fine-tune.",
    ),
    dataset_out: str = typer.Option(
        "agent_dataset.jsonl", "--dataset-out",
        help="Where the synth step writes its dataset (under cwd).",
    ),
    output_dir: str = typer.Option(
        "./agent_train_output", "--output-dir",
        help="Where the planned soup train run will store checkpoints.",
    ),
    examples_per_endpoint: int = typer.Option(
        4, "--examples-per-endpoint", "-n", min=1, max=32,
    ),
):
    """One-shot wrapper: synth + planned soup train invocation (printed)."""
    from soup_cli.utils.agent_forge import (
        load_spec_file,
        parse_spec,
        synthesise_dataset,
        write_dataset,
    )

    # CRITICAL security fix: reject newline/NUL/oversize in --base and
    # --output-dir BEFORE building the recipe YAML string. A crafted
    # --base "evil\ntraining:\n  epochs: 9999" would inject YAML keys
    # into the rendered recipe.
    for label, value in (("--base", base), ("--output-dir", output_dir)):
        if not isinstance(value, str) or not value:
            console.print(f"[red]{label} must be a non-empty string[/]")
            raise typer.Exit(2)
        if "\x00" in value or "\n" in value or "\r" in value:
            console.print(f"[red]{label} contains NUL or newline[/]")
            raise typer.Exit(2)
        if len(value) > 4096:
            console.print(f"[red]{label} exceeds 4096 chars[/]")
            raise typer.Exit(2)

    try:
        spec_dict = load_spec_file(spec)
        endpoints, report = parse_spec(spec_dict)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        console.print(f"[red]Spec error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if not endpoints:
        console.print("[red]No endpoints discovered; aborting train.[/]")
        raise typer.Exit(1)

    rows = synthesise_dataset(endpoints, examples_per_endpoint=examples_per_endpoint)
    try:
        ds_path = write_dataset(rows, dataset_out)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Dataset write failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    planned_cmd = " ".join(
        shlex.quote(p) for p in [
            "soup", "train",
            "--config", "agent_train.yaml",
            "--yes",
        ]
    )
    recipe_yaml = (
        f"base: {base}\n"
        "task: sft\n"
        "data:\n"
        f"  train: {ds_path}\n"
        "  format: tool-calling\n"
        "training:\n"
        "  epochs: 3\n"
        "  lr: 2.0e-5\n"
        "  batch_size: auto\n"
        f"output: {output_dir}\n"
    )

    console.print(
        Panel(
            f"Spec:      [bold]{escape(spec)}[/]\n"
            f"Endpoints: [bold]{report.endpoint_count}[/]\n"
            f"Dataset:   [bold]{escape(ds_path)}[/]\n"
            f"Base:      [bold]{escape(base)}[/]\n"
            f"Output:    [bold]{escape(output_dir)}[/]\n\n"
            f"[bold]Planned recipe (agent_train.yaml):[/]\n{escape(recipe_yaml)}\n"
            f"[bold]Run:[/] {escape(planned_cmd)}",
            title="[bold green]Agent Forge — train plan[/]",
        )
    )
    console.print(
        "[yellow]Note:[/] live in-process training is intentionally not "
        "re-entered (Typer commands aren't safe to re-enter); copy the recipe "
        "into agent_train.yaml and run the command above."
    )


@app.command()
def eval(
    spec: str = typer.Option(..., "--spec", "-s", help="API spec (under cwd)."),
    predictions: str = typer.Option(
        ...,
        "--predictions",
        "-p",
        help=(
            "JSONL of model outputs, one per line, with at minimum "
            "{tool: <name>, arguments: {...}}."
        ),
    ),
):
    """Score predicted tool-calls against the spec's tool catalog.

    Each prediction row gets two checks:
    1. ``tool`` matches a known endpoint in the spec.
    2. ``arguments`` only references parameters declared on that endpoint.
    """
    import os
    import stat as _stat

    from soup_cli.utils.agent_forge import load_spec_file, parse_spec
    from soup_cli.utils.paths import is_under_cwd

    max_pred_lines = 1_000_000

    if not isinstance(predictions, str) or not predictions or "\x00" in predictions:
        console.print("[red]predictions path must be non-empty NUL-free string[/]")
        raise typer.Exit(1)
    if not is_under_cwd(predictions):
        console.print("[red]predictions path must stay under cwd[/]")
        raise typer.Exit(1)
    # Symlink TOCTOU defence (mirrors load_spec_file policy).
    try:
        if _stat.S_ISLNK(os.lstat(predictions).st_mode):
            console.print("[red]predictions path must not be a symlink[/]")
            raise typer.Exit(1)
    except FileNotFoundError:
        console.print("[red]predictions file not found[/]")
        raise typer.Exit(1) from None

    try:
        spec_dict = load_spec_file(spec)
        endpoints, _ = parse_spec(spec_dict)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        console.print(f"[red]Spec error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    tool_to_params = {ep.tool: set(ep.parameters) for ep in endpoints}

    total = 0
    tool_ok = 0
    args_ok = 0
    try:
        with open(predictions, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if total >= max_pred_lines:
                    break
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                tool = row.get("tool")
                if isinstance(tool, str) and tool in tool_to_params:
                    tool_ok += 1
                    args = row.get("arguments") or {}
                    if isinstance(args, dict):
                        invalid = [
                            k for k in args.keys()
                            if k not in tool_to_params[tool]
                        ]
                        if not invalid:
                            args_ok += 1
    except OSError as exc:
        console.print(f"[red]Predictions read failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if total == 0:
        console.print("[yellow]No predictions to score.[/]")
        raise typer.Exit(1)

    tool_pct = 100.0 * tool_ok / total
    args_pct = 100.0 * args_ok / total
    console.print(
        Panel(
            f"Predictions:    [bold]{total}[/]\n"
            f"Tool match:     [bold]{tool_ok}[/] ({tool_pct:.1f}%)\n"
            f"Args valid:     [bold]{args_ok}[/] ({args_pct:.1f}%)",
            title="[bold green]Agent Forge — eval[/]",
        )
    )
