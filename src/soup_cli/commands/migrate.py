"""soup migrate — import configs from LLaMA-Factory, Axolotl, and Unsloth."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

SUPPORTED_SOURCES = ("llamafactory", "axolotl", "unsloth")


def migrate(
    source: str = typer.Option(
        ...,
        "--from",
        help="Source tool: llamafactory, axolotl, or unsloth",
    ),
    config_file: str = typer.Argument(
        ...,
        help="Path to the source config file (.yaml or .ipynb)",
    ),
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output path for generated soup.yaml",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print generated config without writing to file",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
):
    """Import a config from LLaMA-Factory, Axolotl, or Unsloth notebook."""
    from soup_cli.migrate.common import (
        config_to_yaml,
        validate_input_path,
        validate_output_path,
    )

    # Validate source
    if source not in SUPPORTED_SOURCES:
        console.print(
            f"[red]Unknown source: {source}[/]\n"
            f"Supported: {', '.join(SUPPORTED_SOURCES)}"
        )
        raise typer.Exit(1)

    # Validate input path
    input_path = Path(config_file)
    try:
        input_path = validate_input_path(input_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    # Keep the permissive content check for explicit .jsonl files. For every
    # other suffix, require two complete JSON objects on separate lines so a
    # single JSON config or notebook is not mistaken for training data.
    suffix = input_path.suffix.lower()
    is_jsonl = (
        suffix == ".jsonl" and _looks_like_jsonl(input_path)
    ) or (
        suffix != ".jsonl"
        and _looks_like_jsonl(input_path, require_multiple_objects=True)
    )
    if is_jsonl:
        console.print(
            f"[red]Expected a {source} YAML config; got JSONL "
            f"({input_path.name}) — did you pass the wrong file?[/]"
        )
        console.print(
            "[dim]Tip: `soup migrate` migrates competitor *configs*, not "
            "training data. Pass the .yaml / .ipynb file instead.[/]"
        )
        raise typer.Exit(2)

    # Validate output path
    output_path = Path(output)
    if not dry_run:
        try:
            output_path = validate_output_path(output_path)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1)

    # Run migration
    try:
        if source == "llamafactory":
            from soup_cli.migrate.llamafactory import migrate_llamafactory
            result = migrate_llamafactory(input_path)
        elif source == "axolotl":
            from soup_cli.migrate.axolotl import migrate_axolotl
            result = migrate_axolotl(input_path)
        elif source == "unsloth":
            from soup_cli.migrate.unsloth import migrate_unsloth
            result = migrate_unsloth(input_path)
    except ValueError as exc:
        console.print(f"[red]Migration failed:[/] {exc}")
        raise typer.Exit(1)

    # Show warnings (escape Rich markup from untrusted config values)
    migration_warnings = result.get("_warnings", [])
    if migration_warnings:
        from rich.markup import escape
        warning_text = "\n".join(f"  [yellow]![/] {escape(w)}" for w in migration_warnings)
        console.print(Panel(
            warning_text,
            title="[yellow]Migration Warnings[/]",
            border_style="yellow",
        ))

    # Generate YAML
    yaml_str = config_to_yaml(result)

    # Show generated config
    console.print(Panel(
        Syntax(yaml_str, "yaml", theme="monokai"),
        title=f"[bold green]Generated soup.yaml[/] (from {source})",
    ))

    if dry_run:
        console.print("[dim]Dry run -- no file written.[/]")
        return

    # Check for existing file
    if output_path.exists() and not yes:
        confirm = typer.confirm(
            f"File '{output}' already exists. Overwrite?"
        )
        if not confirm:
            console.print("[yellow]Aborted.[/]")
            raise typer.Exit(0)

    # Write output
    output_path.write_text(yaml_str, encoding="utf-8")
    console.print(f"[green]\u2713[/] Config written to [bold]{output}[/]")
    console.print(f"[dim]Next: soup train --config {output}[/]")


def _looks_like_jsonl(path: Path, *, require_multiple_objects: bool = False) -> bool:
    """Inspect a bounded prefix for explicit or structurally identifiable JSONL.

    ``utf-8-sig``, not ``utf-8``: a UTF-8 BOM decodes to U+FEFF, which
    ``str.strip()`` does not remove because it is not whitespace, so a BOM'd
    JSONL file would sniff as *not* JSONL and silently lose the friendly
    error. Windows tooling writes that BOM by default, PowerShell's
    ``Out-File`` included (#675 review).

    The read is bounded because a file with no newline is one single line. An
    unbounded line iteration pulled a measured 240.9 MB peak for a 120 MB
    one-liner. Unknown suffixes use the stricter mode: two non-blank lines must
    each be complete JSON objects, which separates JSONL from one JSON document.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            prefix = fh.read(65536)
    except OSError:
        return False

    lines = [line.strip() for line in prefix.splitlines() if line.strip()]
    if not lines:
        return False
    if not require_multiple_objects:
        return lines[0].startswith("{")

    object_count = 0
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(value, dict):
            return False
        object_count += 1
        if object_count == 2:
            return True
    return False
