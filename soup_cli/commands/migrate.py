"""soup migrate — import configs from LLaMA-Factory, Axolotl, and Unsloth."""

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

    # v0.40.1 Part D / N2 — friendly error when the user passes a JSONL
    # data file (`.jsonl`) instead of a YAML config. The sniff helper is
    # only invoked when the suffix says ``.jsonl`` (notebook .ipynb files
    # legitimately start with ``{`` — we must not falsely flag them).
    if input_path.suffix.lower() == ".jsonl":
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


def _looks_like_jsonl(path: Path) -> bool:
    """v0.40.1 Part D / N2 — sniff first non-blank line for `{` (JSONL)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                return stripped.startswith("{")
    except OSError:
        return False
    return False
