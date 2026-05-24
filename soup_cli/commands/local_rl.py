"""``soup local-rl`` — personal-LLM flywheel daemon CLI (v0.68.0 Part E).

Subcommands:

- ``init`` — create the SQLite schema
- ``status`` — print counters
- ``record`` — append a thumbs-up/down record
- ``harvest`` — emit DPO pairs as JSONL
- ``train`` — nightly DPO/KTO/ORPO train (deferred to v0.68.1)
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.local_rl import (
    SUPPORTED_LOCAL_RL_BACKENDS,
    SUPPORTED_LOCAL_RL_TRAIN_METHODS,
    LocalRLConfig,
    harvest_dpo_pairs,
    init_local_rl_db,
    record_thumb,
    run_nightly_train,
)
from soup_cli.utils.paths import atomic_write_text

console = Console()

app = typer.Typer(no_args_is_help=True, help="Personal-LLM flywheel daemon (v0.68.0).")


@app.command(name="init")
def init_cmd(
    db: str = typer.Option(
        "local_rl.db", "--db", help="Path to local-RL SQLite database"
    ),
) -> None:
    """Create the local-RL SQLite schema."""
    try:
        init_local_rl_db(db)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"DB:     [bold]{escape(db)}[/]\n"
            f"Tables: [bold]interactions[/], [bold]thumbs[/]",
            title="soup local-rl init",
        )
    )


@app.command(name="status")
def status_cmd(
    db: str = typer.Option(
        "local_rl.db", "--db", help="Path to local-RL SQLite database"
    ),
) -> None:
    """Print counters from the local-RL database."""
    import os

    from soup_cli.utils.local_rl import validate_db_path

    try:
        validate_db_path(db)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    real = os.path.abspath(db)
    if not os.path.exists(real):
        console.print(f"[red]db not found: {escape(db)}[/]")
        raise typer.Exit(2)

    with sqlite3.connect(real) as conn:
        up = conn.execute(
            "SELECT COUNT(*) FROM thumbs WHERE thumb='up'"
        ).fetchone()[0]
        down = conn.execute(
            "SELECT COUNT(*) FROM thumbs WHERE thumb='down'"
        ).fetchone()[0]
        interactions = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]

    table = Table(title=f"soup local-rl status — {db}")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Interactions", str(interactions))
    table.add_row("Thumbs up", str(up))
    table.add_row("Thumbs down", str(down))
    console.print(table)


@app.command(name="record")
def record_cmd(
    db: str = typer.Option(
        "local_rl.db", "--db", help="Path to local-RL SQLite database"
    ),
    prompt: str = typer.Option(..., "--prompt", help="Prompt text"),
    response: str = typer.Option(..., "--response", help="Model response"),
    thumb: str = typer.Option(..., "--thumb", help="up / down"),
) -> None:
    """Append a thumbs-up / thumbs-down record."""
    try:
        record_thumb(db_path=db, prompt=prompt, response=response, thumb=thumb)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        f"[green]Recorded {escape(thumb)} for prompt {escape(prompt[:32])}…[/]"
    )


@app.command(name="harvest")
def harvest_cmd(
    db: str = typer.Option(
        "local_rl.db", "--db", help="Path to local-RL SQLite database"
    ),
    output: str = typer.Option(
        "dpo_pairs.jsonl", "--output", "-o", help="Output JSONL path"
    ),
) -> None:
    """Harvest DPO pairs from thumbs into a JSONL file."""
    try:
        pairs = harvest_dpo_pairs(db)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    lines = [json.dumps(dataclasses.asdict(p), ensure_ascii=False) for p in pairs]
    text = "\n".join(lines) + ("\n" if lines else "")
    try:
        atomic_write_text(text, output, field="output")
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"DB:     [bold]{escape(db)}[/]\n"
            f"Output: [bold]{escape(output)}[/]\n"
            f"Pairs:  [bold]{len(pairs)}[/]",
            title="soup local-rl harvest",
        )
    )


@app.command(name="train")
def train_cmd(
    db: str = typer.Option(
        "local_rl.db", "--db", help="Path to local-RL SQLite database"
    ),
    backend: str = typer.Option(
        "ollama",
        "--backend",
        help="Allowed: " + ", ".join(sorted(SUPPORTED_LOCAL_RL_BACKENDS)),
    ),
    model: str = typer.Option(..., "--model", help="Model id (Ollama tag or MLX path)"),
    train_method: str = typer.Option(
        "dpo",
        "--train-method",
        help="Allowed: " + ", ".join(sorted(SUPPORTED_LOCAL_RL_TRAIN_METHODS)),
    ),
) -> None:
    """Trigger the nightly DPO/KTO/ORPO train. Deferred to v0.68.1."""
    try:
        cfg = LocalRLConfig(
            backend=backend,
            model=model,
            db_path=db,
            train_method=train_method,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        run_nightly_train(cfg)
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]{escape(str(exc))}[/]",
                title="Live local-rl train deferred",
            )
        )
        raise typer.Exit(3) from exc
