"""soup registry — Local Model Registry CLI (v0.26.0 Part A)."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.registry.diff import config_diff, eval_delta
from soup_cli.registry.store import (
    AmbiguousRefError,
    RegistryStore,
    validate_name,
    validate_tag,
)

console = Console()

app = typer.Typer(no_args_is_help=True)


def _fail(message: str) -> None:
    console.print(f"[red]{escape(message)}[/]")
    raise typer.Exit(1)


def _require_entry(store: RegistryStore, ref: str) -> dict:
    try:
        eid = store.resolve(ref)
    except AmbiguousRefError as exc:
        _fail(str(exc))
    if eid is None:
        _fail(f"Registry entry not found: {ref}")
    entry = store.get(eid)
    if entry is None:
        _fail(f"Registry entry not found: {ref}")
    return entry


@app.command("push")
def push_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Training run id to register"),
    name: str = typer.Option(..., "--name", "-n", help="Registry entry name"),
    tag: str = typer.Option("v1", "--tag", "-t", help="Tag (e.g. v1, prod-2024)"),
    notes: str = typer.Option("", "--notes", help="Free-form notes"),
) -> None:
    """Register a completed training run as a registry entry."""
    try:
        validate_name(name)
        validate_tag(tag)
    except ValueError as exc:
        _fail(f"Invalid input: {exc}")

    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    try:
        if tracker.get_run(run_id) is None:
            _fail(f"Run not found: {run_id}")
        with RegistryStore() as store:
            try:
                entry_id = store.register_from_run(
                    tracker, run_id, name=name, tag=tag,
                    notes=notes or None,
                )
            except ValueError as exc:
                _fail(str(exc))
        console.print(Panel(
            f"Registered [cyan]{escape(name)}[/]:[magenta]{escape(tag)}[/]\n"
            f"Entry ID: [bold]{escape(entry_id)}[/]",
            title="Registry", border_style="green",
        ))
    finally:
        tracker.close()


@app.command("list")
def list_cmd(
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t"),
    base: Optional[str] = typer.Option(None, "--base", "-b"),
    task: Optional[str] = typer.Option(None, "--task"),
    limit: int = typer.Option(100, "--limit", "-l"),
) -> None:
    """List registry entries."""
    with RegistryStore() as store:
        entries = store.list(name=name, tag=tag, base=base, task=task,
                             limit=max(1, min(limit, 1000)))
        if not entries:
            console.print("[dim]No registry entries found.[/]")
            return

        table = Table(title="Registry")
        table.add_column("Entry ID", style="bold cyan", no_wrap=True)
        table.add_column("Name")
        table.add_column("Tags")
        table.add_column("Base Model", max_width=32)
        table.add_column("Task")
        table.add_column("Created", no_wrap=True)

        for entry in entries:
            short_id = entry["id"][:20]
            table.add_row(
                escape(short_id),
                escape(entry["name"]),
                escape(", ".join(entry.get("tags", []))),
                escape(entry["base_model"]),
                escape(entry["task"]),
                escape(entry["created_at"][:16].replace("T", " ")),
            )

        console.print(table)


@app.command("show")
def show_cmd(
    ref: str = typer.Argument(..., help="Entry id / prefix / name:tag"),
) -> None:
    """Show full details of a registry entry."""
    with RegistryStore() as store:
        entry = _require_entry(store, ref)

        table = Table(
            title=f"Registry Entry: {escape(entry['name'])}",
            show_header=False,
        )
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")

        fields = [
            ("id", entry["id"]),
            ("name", entry["name"]),
            ("tags", ", ".join(entry.get("tags", []))),
            ("base_model", entry["base_model"]),
            ("task", entry["task"]),
            ("run_id", entry.get("run_id") or "-"),
            ("entry_hash", entry.get("entry_hash", "")[:16] + "..."),
            ("created_at", entry["created_at"]),
            ("notes", entry.get("notes") or ""),
        ]
        for key, val in fields:
            table.add_row(escape(key), escape(str(val)))
        console.print(table)

        artifacts = store.get_artifacts(entry["id"])
        if artifacts:
            art_table = Table(title="Artifacts")
            art_table.add_column("Kind", style="magenta")
            art_table.add_column("Path")
            art_table.add_column("Size", justify="right")
            art_table.add_column("SHA256 (prefix)")
            for art in artifacts:
                art_table.add_row(
                    escape(art["kind"]),
                    escape(art["path"]),
                    f"{art['size_bytes'] / 1024 / 1024:.1f} MB"
                    if art["size_bytes"] > 1024 * 1024
                    else f"{art['size_bytes']} B",
                    escape(art["sha256"][:16]),
                )
            console.print(art_table)

        ancestors = store.get_ancestors(entry["id"])
        if ancestors:
            lin_table = Table(title="Ancestors")
            lin_table.add_column("Entry ID", style="cyan")
            lin_table.add_column("Name")
            lin_table.add_column("Relation")
            for anc in ancestors:
                lin_table.add_row(
                    escape(anc["id"]), escape(anc["name"]),
                    escape(anc.get("relation", "")),
                )
            console.print(lin_table)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Keyword to search for"),
    limit: int = typer.Option(50, "--limit", "-l"),
) -> None:
    """Search registry entries by name/base/task/notes."""
    with RegistryStore() as store:
        hits = store.search(query, limit=max(1, min(limit, 500)))
        if not hits:
            console.print(f"[dim]No entries match '{escape(query)}'.[/]")
            return

        table = Table(title=f"Search: '{escape(query)}'")
        table.add_column("Entry ID", style="bold cyan")
        table.add_column("Name")
        table.add_column("Base Model")
        table.add_column("Task")
        for entry in hits:
            table.add_row(
                escape(entry["id"]),
                escape(entry["name"]),
                escape(entry["base_model"]),
                escape(entry["task"]),
            )
        console.print(table)


@app.command("diff")
def diff_cmd(
    left: str = typer.Argument(..., help="Left entry ref"),
    right: str = typer.Argument(..., help="Right entry ref"),
) -> None:
    """Side-by-side config + eval delta between two entries."""
    with RegistryStore() as store:
        left_entry = _require_entry(store, left)
        right_entry = _require_entry(store, right)

        try:
            lcfg = json.loads(left_entry.get("config_json") or "{}")
            rcfg = json.loads(right_entry.get("config_json") or "{}")
        except (TypeError, ValueError):
            lcfg, rcfg = {}, {}

        changes = config_diff(lcfg, rcfg)
        table = Table(title=(
            f"Config diff: {escape(left_entry['id'])} "
            f"-> {escape(right_entry['id'])}"
        ))
        table.add_column("Path", style="cyan")
        table.add_column("Kind")
        table.add_column("Left")
        table.add_column("Right")
        if not changes:
            console.print("[green]No config differences.[/]")
        else:
            for change in changes:
                table.add_row(
                    escape(change.path),
                    escape(change.kind),
                    escape(str(change.left) if change.left is not None else "-"),
                    escape(str(change.right) if change.right is not None else "-"),
                )
            console.print(table)

        left_evals = store.get_eval_results(left_entry["id"])
        right_evals = store.get_eval_results(right_entry["id"])
        if left_evals or right_evals:
            deltas = eval_delta(left_evals, right_evals)
            eval_table = Table(title="Eval delta")
            eval_table.add_column("Benchmark", style="cyan")
            eval_table.add_column("Left", justify="right")
            eval_table.add_column("Right", justify="right")
            eval_table.add_column("Delta", justify="right")
            for row in deltas:
                delta_val = row["delta"]
                delta_str = f"{delta_val:+.3f}" if delta_val is not None else "-"
                eval_table.add_row(
                    escape(str(row["benchmark"])),
                    f"{row['left']:.3f}" if row["left"] is not None else "-",
                    f"{row['right']:.3f}" if row["right"] is not None else "-",
                    delta_str,
                )
            console.print(eval_table)


@app.command("promote")
def promote_cmd(
    ref: str = typer.Argument(..., help="Entry ref to promote"),
    tag: str = typer.Option(..., "--tag", "-t", help="Tag to assign"),
) -> None:
    """Add a tag to an existing entry (e.g. promote to 'prod')."""
    try:
        validate_tag(tag)
    except ValueError as exc:
        _fail(str(exc))

    with RegistryStore() as store:
        entry = _require_entry(store, ref)
        store.add_tag(entry["id"], tag)
        console.print(
            f"[green]Added tag[/] [magenta]{escape(tag)}[/] "
            f"to [cyan]{escape(entry['id'])}[/]."
        )


@app.command("delete")
def delete_cmd(
    ref: str = typer.Argument(..., help="Entry ref to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a registry entry and its lineage / artifact rows."""
    with RegistryStore() as store:
        entry = _require_entry(store, ref)
        if not yes:
            console.print(
                f"About to delete [cyan]{escape(entry['id'])}[/] "
                f"([bold]{escape(entry['name'])}[/]). Pass --yes to confirm."
            )
            # User cancelled / didn't pass --yes: this is a no-op, not an error.
            raise typer.Exit(0)
        store.delete(entry["id"])
        console.print(f"[green]Deleted[/] {escape(entry['id'])}.")
