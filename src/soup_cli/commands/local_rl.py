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
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
from soup_cli.utils.terminal import strip_control, terminal_text

console = Console()

app = typer.Typer(no_args_is_help=True, help="Personal-LLM flywheel daemon (v0.68.0).")


# Database, prompt, output and scheduler paths come from the operator, and error
# text from the checks on them, so every line that carries them is printed as
# ``Text``: shown as typed, never read as markup or ``:name:`` emoji, control
# characters removed (#2489).


def _refuse(message: object) -> None:
    console.print(terminal_text(message, "red"))


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
        _refuse(exc)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.assemble(
                "DB:     ",
                (strip_control(db), "bold"),
                "\nTables: ",
                ("interactions", "bold"),
                ", ",
                ("thumbs", "bold"),
                ", ",
                ("state", "bold"),
            ),
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
        _refuse(exc)
        raise typer.Exit(2) from exc

    real = os.path.abspath(db)
    if not os.path.exists(real):
        _refuse(f"db not found: {db}")
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

    # A ``Text`` title does not get the table's title style, so it is named here.
    table = Table(title=terminal_text(f"soup local-rl status — {db}", "table.title"))
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
        _refuse(exc)
        raise typer.Exit(2) from exc

    console.print(terminal_text(f"Recorded {thumb} for prompt {prompt[:32]}…", "green"))


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
        _refuse(exc)
        raise typer.Exit(2) from exc

    lines = [json.dumps(dataclasses.asdict(p), ensure_ascii=False) for p in pairs]
    text = "\n".join(lines) + ("\n" if lines else "")
    try:
        atomic_write_text(text, output, field="output")
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.assemble(
                "DB:     ",
                (strip_control(db), "bold"),
                "\nOutput: ",
                (strip_control(output), "bold"),
                "\nPairs:  ",
                (str(len(pairs)), "bold"),
            ),
            title="soup local-rl harvest",
        )
    )


@app.command(name="train")
def train_cmd(
    db: str = typer.Option(
        "local_rl.db", "--db", help="Path to local-RL SQLite database"
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help=(
            "Accepted for compatibility; has no effect on training. "
            "Allowed: " + ", ".join(sorted(SUPPORTED_LOCAL_RL_BACKENDS))
        ),
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
            backend="ollama" if backend is None else backend,
            model=model,
            db_path=db,
            train_method=train_method,
        )
    except (TypeError, ValueError) as exc:
        _refuse(exc)
        raise typer.Exit(2) from exc

    if backend is not None:
        console.print("[dim]Note: --backend has no effect on training.[/]")

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
                min_pairs=min_pairs,
                output_dir=output,
            )
        except (TypeError, ValueError) as exc:
            _refuse(exc)
            raise typer.Exit(2) from exc
        files = Text("\n").join(
            Text.assemble("  ", (strip_control(n), "bold")) for n in sorted(written)
        )
        console.print(
            Panel(
                Text.assemble(
                    "Rendered the nightly-train scaffold into ",
                    (strip_control(scheduler_dir), "bold"),
                    ":\n",
                    files,
                    "\n\n",
                    ("Linux:", "dim"),
                    " cp soup-local-rl.{service,timer} ~/.config/systemd/user/ "
                    "&& systemctl --user enable --now soup-local-rl.timer\n",
                    ("macOS:", "dim"),
                    " cp com.soup.local-rl.plist ~/Library/LaunchAgents/ "
                    "&& launchctl load ~/Library/LaunchAgents/com.soup.local-rl.plist\n",
                    ("Now:", "dim"),
                    " re-run with ",
                    ("--once", "bold"),
                    " for an ad-hoc train.",
                ),
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
        _refuse(exc)
        raise typer.Exit(2) from exc
    except subprocess.SubprocessError as exc:
        # The underlying `soup train` subprocess failed (e.g. OOM / bad base).
        console.print(
            Panel(
                Text.assemble(
                    ("Training subprocess failed:", "red"), " ", strip_control(exc)
                ),
                title="soup local-rl train — failed",
            )
        )
        raise typer.Exit(1) from exc

    if result.status.startswith("skipped"):
        console.print(
            Panel(
                Text.assemble(("Skipped:", "yellow"), " ", strip_control(result.reason)),
                title="soup local-rl train — skipped",
            )
        )
        return

    console.print(
        Panel(
            Text.assemble(
                "Trained: ",
                (str(result.num_pairs), "bold"),
                " pairs via ",
                (strip_control(cfg.train_method), "bold"),
                "\nOutput:  ",
                (strip_control(result.output_dir or output), "bold"),
            ),
            title="soup local-rl train — done",
        )
    )
