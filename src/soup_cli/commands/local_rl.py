"""``soup local-rl`` — personal-LLM flywheel daemon CLI (v0.68.0 Part E).

Subcommands:

- ``init`` — create the SQLite schema
- ``status`` — print counters
- ``record`` — append a thumbs-up/down record
- ``harvest`` — emit DPO pairs as JSONL
- ``train`` — run (``--once``) or schedule the nightly DPO/KTO/ORPO train (v0.71.13 #229)
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
            f"Tables: [bold]interactions[/], [bold]thumbs[/], [bold]state[/]",
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
    model: str = typer.Option(
        ...,
        "--model",
        help="Training base — HF repo id or local path (NOT the Ollama tag)",
    ),
    train_method: str = typer.Option(
        "dpo",
        "--train-method",
        help="Allowed: " + ", ".join(sorted(SUPPORTED_LOCAL_RL_TRAIN_METHODS)),
    ),
    once: bool = typer.Option(
        False,
        "--once",
        help="Run the train now (ad-hoc); skip scheduler install.",
    ),
    min_pairs: int = typer.Option(
        10, "--min-pairs", help="Skip training when fewer pairs are harvested."
    ),
    output: str = typer.Option(
        "local_rl_adapter", "--output", "-o", help="Adapter output directory"
    ),
    scheduler_dir: str = typer.Option(
        "local-rl-scheduler",
        "--scheduler-dir",
        help="Directory to render the systemd / launchd scaffold into.",
    ),
    hour: int = typer.Option(3, "--hour", help="Daily train hour (local time)."),
    minute: int = typer.Option(0, "--minute", help="Daily train minute."),
) -> None:
    """Run (``--once``) or schedule the nightly DPO/KTO/ORPO train."""
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

    if not once:
        # Render the scheduler scaffold; never run systemctl / launchctl.
        import sys

        from soup_cli.utils.local_rl_scheduler import write_scheduler_files

        try:
            written = write_scheduler_files(
                scheduler_dir,
                soup_python=sys.executable,
                db_path=db,
                model=model,
                train_method=train_method,
                hour=hour,
                minute=minute,
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        files = "\n".join(f"  [bold]{escape(n)}[/]" for n in sorted(written))
        console.print(
            Panel(
                f"Rendered the nightly-train scaffold into "
                f"[bold]{escape(scheduler_dir)}[/]:\n{files}\n\n"
                "[dim]Linux:[/] cp soup-local-rl.{service,timer} ~/.config/systemd/user/ "
                "&& systemctl --user enable --now soup-local-rl.timer\n"
                "[dim]macOS:[/] cp com.soup.local-rl.plist ~/Library/LaunchAgents/ "
                "&& launchctl load ~/Library/LaunchAgents/com.soup.local-rl.plist\n"
                "[dim]Now:[/] re-run with [bold]--once[/] for an ad-hoc train.",
                title="soup local-rl train — scheduler scaffold",
            )
        )
        return

    import subprocess

    try:
        result = run_nightly_train(
            cfg, once=True, min_pairs=min_pairs, output_dir=output
        )
    except (TypeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    except subprocess.SubprocessError as exc:
        # The underlying `soup train` subprocess failed (e.g. OOM / bad base).
        console.print(
            Panel(
                f"[red]Training subprocess failed:[/] {escape(str(exc))}",
                title="soup local-rl train — failed",
            )
        )
        raise typer.Exit(1) from exc

    if result.status.startswith("skipped"):
        console.print(
            Panel(
                f"[yellow]Skipped:[/] {escape(result.reason)}",
                title="soup local-rl train — skipped",
            )
        )
        return

    console.print(
        Panel(
            f"Trained: [bold]{result.num_pairs}[/] pairs via "
            f"[bold]{escape(cfg.train_method)}[/]\n"
            f"Output:  [bold]{escape(result.output_dir or output)}[/]",
            title="soup local-rl train — done",
        )
    )
