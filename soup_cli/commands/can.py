"""soup can — shareable .can artifact CLI (v0.26.0 Part E)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(no_args_is_help=True)


def _fail(message: str) -> None:
    console.print(f"[red]{escape(message)}[/]")
    raise typer.Exit(1)


@app.command(name="pack")
def pack_cmd(
    entry_id: str = typer.Option(
        ..., "--entry-id", help="Registry entry id / prefix / name:tag",
    ),
    out: str = typer.Option(
        ..., "--out", "-o", help="Output .can path",
    ),
    author: str = typer.Option(
        "unknown", "--author", help="Author handle",
    ),
    description: str = typer.Option(
        "", "--description", help="Free-form description",
    ),
) -> None:
    """Pack a registry entry into a shareable .can file."""
    from soup_cli.cans.pack import pack_entry

    try:
        path = pack_entry(
            entry_id=entry_id, out_path=out, author=author,
            description=description or None,
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(Panel(
        f"Packed [cyan]{escape(entry_id)}[/] -> [bold]{escape(str(path))}[/]",
        title="Soup Can", border_style="green",
    ))


@app.command(name="inspect")
def inspect_cmd(
    path: str = typer.Argument(..., help="Path to .can file"),
) -> None:
    """Preview a .can file's manifest without extracting."""
    from soup_cli.cans.unpack import inspect_can

    try:
        manifest = inspect_can(path)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"Cannot inspect can: {exc}")

    table = Table(title=f"Can: {escape(manifest.name)}", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    for key, val in [
        ("name", manifest.name),
        ("author", manifest.author),
        ("created_at", manifest.created_at),
        ("format_version", manifest.can_format_version),
        ("base_hash", (manifest.base_hash or "")[:16] + "..."),
        ("tags", ", ".join(manifest.tags)),
        ("description", manifest.description or ""),
    ]:
        table.add_row(escape(key), escape(str(val)))
    console.print(table)


@app.command(name="verify")
def verify_cmd(
    path: str = typer.Argument(..., help="Path to .can file"),
) -> None:
    """Verify a .can file's schema and config parseability."""
    from soup_cli.cans.verify import verify_can

    report = verify_can(path)
    if report.manifest_ok and report.config_ok:
        console.print(f"[green]OK:[/] {escape(report.message)}")
    else:
        console.print(f"[red]Verify failed:[/] {escape(report.message)}")
        raise typer.Exit(1)


@app.command(name="run")
def run_cmd(
    can_path: str = typer.Argument(..., help="Path to .can file"),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip the security confirmation panel (auto-trains, auto-fetches data)",
    ),
    deploy: bool = typer.Option(
        False, "--deploy",
        help="Run the can's deploy_targets after a successful train",
    ),
    extract_dir: Optional[str] = typer.Option(
        None, "--extract-dir",
        help="Where to extract (default: fresh tmp dir)",
    ),
    env_capture: Optional[str] = typer.Option(
        None, "--env-capture",
        help="Write env summary (pip freeze + GPU info) to this path",
    ),
) -> None:
    """Run a can end-to-end: extract -> train (-> optional deploy)."""
    from soup_cli.cans.run import run_can

    if not yes:
        try:
            from soup_cli.cans.unpack import inspect_can
            manifest = inspect_can(can_path)
        except (FileNotFoundError, ValueError) as exc:
            _fail(str(exc))
        console.print(Panel(
            "[yellow]`soup can run` will:[/]\n"
            "  - Extract the can\n"
            "  - Auto-fetch any data referenced inside\n"
            "  - Run [bold]soup train[/] against the embedded config\n\n"
            f"Recipe: [bold]{escape(manifest.name)}[/]\n"
            f"Author: {escape(manifest.author)}\n\n"
            "Pass [bold]--yes[/] to confirm.",
            title="Run can - confirm", border_style="yellow",
        ))
        raise typer.Exit(1)

    result = None
    try:
        result = run_can(
            can_path, yes=True, deploy=deploy,
            extract_dir=extract_dir, capture_env_to=env_capture,
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))

    # _fail raises typer.Exit; result remains None only if a future
    # exception type slips through. Belt-and-braces guard so a
    # NameError-style regression is impossible.
    if result is None:
        _fail("soup can run did not produce a result")
    if result.train_returncode != 0:
        console.print(
            f"[red]train failed (rc={result.train_returncode}). "
            f"Extract dir: {escape(str(result.extract_dir))}[/]"
        )
        raise typer.Exit(result.train_returncode)
    if result.deploy_returncode is not None and result.deploy_returncode != 0:
        console.print(
            f"[red]deploy failed (rc={result.deploy_returncode})[/]"
        )
        raise typer.Exit(result.deploy_returncode)
    console.print(
        f"[green]Can run complete[/] - extract dir: "
        f"[bold]{escape(str(result.extract_dir))}[/]"
    )


@app.command(name="publish")
def publish_cmd(
    can_path: str = typer.Argument(..., help="Path to .can file"),
    hf_hub: str = typer.Option(
        ..., "--hf-hub",
        help="HF Hub dataset repo (e.g. 'me/my-recipe-can')",
    ),
    private: bool = typer.Option(False, "--private"),
    commit_message: Optional[str] = typer.Option(
        None, "--message", "-m",
        help="Commit message (first line only, capped at 200 chars)",
    ),
) -> None:
    """Publish a .can file to HF Hub as a dataset."""
    from soup_cli.cans.publish import publish_can

    try:
        url = publish_can(
            can_path,
            repo_id=hf_hub,
            private=private,
            commit_message=commit_message,
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    except ImportError as exc:
        _fail(f"huggingface_hub not installed: {exc}")
    console.print(f"[green]Published[/] -> [bold]{escape(url)}[/]")


@app.command(name="fork")
def fork_cmd(
    source: str = typer.Argument(..., help="Source .can path"),
    out: str = typer.Option(..., "--out", "-o", help="Output forked .can path"),
    modify: list[str] = typer.Option(
        [], "--modify", help="Modification like 'training.lr=5e-5' (repeatable)",
    ),
    author: str = typer.Option("unknown", "--author"),
) -> None:
    """Fork a can with config modifications and re-pack."""
    from soup_cli.cans.pack import fork_can

    try:
        path = fork_can(
            source=source, out_path=out, modifications=list(modify),
            author=author,
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(
        f"[green]Forked[/] -> [bold]{escape(str(path))}[/]"
    )
