"""soup lock — shared run lockfile (v0.67.0 Part E).

Subcommands:

- ``soup lock write``: render a ``soup.lock`` from operator-supplied
  base-model / dataset / env hashes.
- ``soup lock check``: compare a tracked ``soup.lock`` against a
  freshly-computed closure; exit 3 on drift.
- ``soup lock show``: print a tracked lock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()

app = typer.Typer(no_args_is_help=True, help="Shared run lockfile (v0.67.0 Part E)")


@app.command(name="write")
def write_lock_cmd(
    base_model: str = typer.Option(..., "--base-model", help="HF model id / path"),
    base_sha: str = typer.Option(..., "--base-sha", help="64-hex base-model SHA"),
    dataset_sha: str = typer.Option(..., "--dataset-sha", help="64-hex dataset SHA"),
    env_hash: Optional[str] = typer.Option(
        None,
        "--env-hash",
        help="64-hex env hash. If omitted, auto-derived from --env-lock "
        "(soup-env.lock) via `soup env lock`.",
    ),
    env_lock: str = typer.Option(
        "soup-env.lock",
        "--env-lock",
        help="Path to a soup-env.lock to auto-derive --env-hash from "
        "(used only when --env-hash is omitted).",
    ),
    output: str = typer.Option("soup.lock", "--output", "-o", help="Output path"),
):
    """Render a ``soup.lock`` from the base/dataset/env hashes.

    When ``--env-hash`` is omitted, it is auto-derived from ``--env-lock``
    (default ``soup-env.lock``) so an operator who ran ``soup env lock`` does
    not have to copy the hash by hand (v0.71.1 #224).
    """
    from soup_cli import __version__
    from soup_cli.utils.soup_lock import SoupLock, compute_lock_closure, write_lock

    # v0.71.1 #224 — auto-glue: derive the env hash from soup-env.lock when
    # the operator did not pass --env-hash explicitly. Treat an empty string
    # the same as omitted so `--env-hash ""` auto-derives rather than tripping
    # the generic 64-hex closure error.
    if not env_hash:
        from soup_cli.utils.env_lock import compute_env_hash
        from soup_cli.utils.env_lock import read_lock as read_env_lock

        try:
            env_lock_obj = read_env_lock(env_lock)
        except FileNotFoundError as exc:
            console.print(
                f"[red]--env-hash not provided and {escape(env_lock)!s} not found.[/]\n"
                "Either pass --env-hash <64-hex> explicitly, or run "
                "`soup env lock` first to create soup-env.lock."
            )
            raise typer.Exit(2) from exc
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        env_hash = compute_env_hash(env_lock_obj)

    try:
        closure = compute_lock_closure(
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
        )
        lock = SoupLock(
            soup_version=__version__,
            base_model=base_model,
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
            closure_sha=closure,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        write_lock(lock, output)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Lock:        [bold]{escape(output)}[/]\n"
            f"Base:        [bold]{escape(base_model)}[/]\n"
            f"Closure SHA: [bold]{closure[:12]}…[/]",
            title="soup.lock written",
        )
    )


@app.command(name="show")
def show_lock_cmd(
    path: str = typer.Argument("soup.lock", help="Path to soup.lock"),
):
    """Print a tracked lock file."""
    from soup_cli.utils.soup_lock import read_lock

    try:
        lock = read_lock(path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Soup version:    [bold]{escape(lock.soup_version)}[/]\n"
            f"Base model:      [bold]{escape(lock.base_model)}[/]\n"
            f"Base SHA:        [bold]{lock.base_model_sha[:16]}…[/]\n"
            f"Dataset SHA:     [bold]{lock.dataset_sha[:16]}…[/]\n"
            f"Env hash:        [bold]{lock.env_hash[:16]}…[/]\n"
            f"Closure SHA:     [bold]{lock.closure_sha[:16]}…[/]\n"
            f"Created at:      [bold]{escape(lock.created_at)}[/]",
            title=f"soup.lock — {escape(path)}",
        )
    )


@app.command(name="check")
def check_lock_cmd(
    path: str = typer.Argument("soup.lock", help="Path to tracked soup.lock"),
    base_sha: str = typer.Option(..., "--base-sha", help="64-hex current base-model SHA"),
    dataset_sha: str = typer.Option(..., "--dataset-sha", help="64-hex current dataset SHA"),
    env_hash: str = typer.Option(..., "--env-hash", help="64-hex current env hash"),
    base_model: str = typer.Option(..., "--base-model", help="Current base model id"),
):
    """Refuse with exit 3 if the lock has drifted from current state."""
    from soup_cli import __version__
    from soup_cli.utils.soup_lock import (
        SoupLock,
        check_lock_drift,
        compute_lock_closure,
        read_lock,
    )

    try:
        expected = read_lock(path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        closure = compute_lock_closure(
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
        )
        # Use the existing soup_version + created_at from `expected` so the
        # comparison stays content-only (drift only counts the 5 content
        # fields per `check_lock_drift`).
        actual = SoupLock(
            soup_version=expected.soup_version,
            base_model=base_model,
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
            closure_sha=closure,
            created_at=expected.created_at,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    drift = check_lock_drift(expected, actual)
    if drift.ok:
        console.print(
            Panel(
                f"Lock:        [bold]{escape(path)}[/]\n"
                f"Status:      [green]OK[/] — closure matches",
                title="soup lock check",
            )
        )
        return

    console.print(
        Panel(
            f"Lock:    [bold]{escape(path)}[/]\n"
            f"Status:  [red]DRIFT[/]",
            title="soup lock check",
        )
    )
    for change in drift.changes:
        console.print(f"  [red]- {escape(change)}[/]")
    if __version__ != expected.soup_version:
        console.print(
            f"[yellow]Note: soup version changed "
            f"({escape(expected.soup_version)} -> {escape(__version__)})[/]"
        )
    raise typer.Exit(3)
