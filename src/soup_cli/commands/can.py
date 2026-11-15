"""soup can — shareable .can artifact CLI (v0.26.0 Part E)."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soup_cli.utils.terminal import strip_control

console = Console()
app = typer.Typer(no_args_is_help=True)


def _fail(message: str) -> None:
    console.print(Text(strip_control(message), style="red"))
    raise typer.Exit(1)


def _load_attestation_files(paths: Optional[List[str]]) -> list[dict]:
    """Load + shape-validate in-toto Statement JSON files (cwd-contained)."""
    from soup_cli.cans.schema import (
        _MAX_ATTESTATION_BYTES,
        validate_attestation_statement,
    )
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    statements: list[dict] = []
    for raw in paths or []:
        try:
            real = enforce_under_cwd_and_no_symlink(raw, "attest")
            # Security: gate the raw file size BEFORE json.load so a multi-GB
            # --attest file is rejected without being parsed into memory
            # (the 1 MiB cap in validate_attestation_statement only fires
            # AFTER the parse).
            if os.path.getsize(real) > _MAX_ATTESTATION_BYTES:
                raise ValueError(
                    f"attestation file too large (> {_MAX_ATTESTATION_BYTES} bytes)"
                )
            with open(real, encoding="utf-8-sig") as fh:
                stmt = json.load(fh)
            statements.append(validate_attestation_statement(stmt))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _fail(f"--attest {raw}: {exc}")
    return statements


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
    attest: Optional[List[str]] = typer.Option(
        None, "--attest",
        help=(
            "Embed an in-toto Statement JSON into the can manifest (v3, "
            "repeatable). Each file is shape-validated. v0.71.3."
        ),
    ),
) -> None:
    """Pack a registry entry into a shareable .can file."""
    from soup_cli.cans.pack import pack_entry

    attestations = _load_attestation_files(attest)
    try:
        path = pack_entry(
            entry_id=entry_id, out_path=out, author=author,
            description=description or None,
            attestations=attestations,
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(
        Panel(
            Text.assemble(
                "Packed ", (strip_control(entry_id), "cyan"),
                " -> ", (strip_control(path), "bold"),
            ),
            title="Soup Can",
            border_style="green",
        )
    )


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

    table = Table(title=Text(f"Can: {manifest.name}"), show_header=False)
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
        # ``key`` is one of the literals above; ``val`` comes from the can.
        table.add_row(key, Text(strip_control(val)))
    console.print(table)


@app.command(name="verify")
def verify_cmd(
    path: str = typer.Argument(..., help="Path to .can file"),
) -> None:
    """Verify a .can file's schema and config parseability."""
    from soup_cli.cans.verify import verify_can

    report = verify_can(path)
    if report.manifest_ok and report.config_ok:
        console.print(Text.assemble(("OK:", "green"), f" {strip_control(report.message)}"))
    else:
        console.print(Text.assemble(("Verify failed:", "red"), f" {strip_control(report.message)}"))
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
            Text.assemble(
                ("`soup can run` will:", "yellow"),
                "\n  - Extract the can\n"
                "  - Auto-fetch any data referenced inside\n"
                "  - Run ",
                ("soup train", "bold"),
                " against the embedded config\n\nRecipe: ",
                (manifest.name, "bold"),
                f"\nAuthor: {strip_control(manifest.author)}\n\nPass ",
                ("--yes", "bold"),
                " to confirm.",
            ),
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
            Text(
                f"train failed (rc={result.train_returncode}). "
                f"Extract dir: {strip_control(result.extract_dir)}",
                style="red",
            )
        )
        raise typer.Exit(result.train_returncode)
    if result.deploy_returncode is not None and result.deploy_returncode != 0:
        console.print(
            f"[red]deploy failed (rc={result.deploy_returncode})[/]"
        )
        raise typer.Exit(result.deploy_returncode)
    console.print(
        Text.assemble(
            ("Can run complete", "green"),
            " - extract dir: ",
            (strip_control(result.extract_dir), "bold"),
        )
    )


@app.command(name="publish")
def publish_cmd(
    can_path: str = typer.Argument(..., help="Path to .can file"),
    hf_hub: str = typer.Option(
        ..., "--hf-hub",
        help="HF Hub dataset repo (e.g. 'me/my-recipe-can')",
    ),
    private: bool = typer.Option(
        False, "--private", help="Create a private HF Hub dataset repository."
    ),
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
    console.print(Text.assemble(("Published", "green"), " -> ", (strip_control(url), "bold")))


@app.command(name="fork")
def fork_cmd(
    source: str = typer.Argument(..., help="Source .can path"),
    out: str = typer.Option(..., "--out", "-o", help="Output forked .can path"),
    modify: list[str] = typer.Option(
        [], "--modify", help="Modification like 'training.lr=5e-5' (repeatable)",
    ),
    author: str = typer.Option(
        "unknown", "--author", help="Author name recorded in the forked manifest."
    ),
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
    console.print(Text.assemble(("Forked", "green"), " -> ", (strip_control(path), "bold")))
