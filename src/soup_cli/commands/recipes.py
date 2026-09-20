"""soup recipes — browse and use ready-made configs for popular models."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=True)


@app.command(name="list")
def list_cmd():
    """List all available recipes."""
    from soup_cli.recipes.catalog import RECIPES

    table = Table(title="Soup Recipes")
    table.add_column("Name", style="bold cyan")
    table.add_column("Model", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Description")

    for name, recipe in RECIPES.items():
        table.add_row(name, recipe.model, recipe.task, recipe.size, recipe.description)

    console.print(table)


@app.command()
def show(
    name: str = typer.Argument(..., help="Recipe name"),
):
    """Show a recipe config (print YAML to stdout)."""
    from soup_cli.recipes.catalog import get_recipe

    recipe = get_recipe(name)
    if recipe is None:
        console.print(f"[red]Recipe not found: {name}[/]")
        # v0.40.1 Part E / M3 — fuzzy-match suggestion (mirrors Typer's
        # built-in "Did you mean" for unknown CLI flags).
        suggestions = _suggest_recipes(name)
        if suggestions:
            console.print(
                f"[dim]Did you mean: [bold]{', '.join(suggestions)}[/]?[/]"
            )
        console.print("[dim]Run 'soup recipes list' to see all recipes.[/]")
        raise typer.Exit(1)

    console.print(Panel(
        Syntax(recipe.yaml_str, "yaml", theme="monokai"),
        title=f"[bold green]{name}[/] -- {recipe.description}",
    ))


@app.command()
def use(
    name: str = typer.Argument(..., help="Recipe name"),
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output path for config file",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
):
    """Copy a recipe to soup.yaml (or custom path)."""
    from soup_cli.recipes.catalog import get_recipe

    recipe = get_recipe(name)
    if recipe is None:
        console.print(f"[red]Recipe not found: {name}[/]")
        raise typer.Exit(1)

    from soup_cli.migrate.common import validate_output_path

    try:
        output_path = validate_output_path(Path(output))
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    if output_path.exists() and not yes:
        confirm = typer.confirm(f"File '{output}' already exists. Overwrite?")
        if not confirm:
            console.print("[yellow]Aborted.[/]")
            raise typer.Exit(0)

    output_path.write_text(recipe.yaml_str, encoding="utf-8")
    console.print(f"[green]\u2713[/] Recipe [bold]{name}[/] written to [bold]{output}[/]")
    console.print(f"[dim]Next: soup train --config {output}[/]")


@app.command()
def search(
    query: Optional[str] = typer.Argument(None, help="Search keyword"),
    task: Optional[str] = typer.Option(None, "--task", help="Filter by task"),
    size: Optional[str] = typer.Option(None, "--size", help="Filter by model size"),
):
    """Search recipes by keyword, task, or model size."""
    from soup_cli.recipes.catalog import RECIPES, search_recipes

    results = search_recipes(query=query, task=task, size=size)

    if not results:
        console.print("[yellow]No recipes found matching your query.[/]")
        console.print(f"[dim]Available recipes: {len(RECIPES)}. Run 'soup recipes list'.[/]")
        return

    table = Table(title="Search Results")
    table.add_column("Name", style="bold cyan")
    table.add_column("Model", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Description")

    # Find name for each result
    for recipe in results:
        name = next(
            (n for n, r in RECIPES.items() if r is recipe),
            "?",
        )
        table.add_row(name, recipe.model, recipe.task, recipe.size, recipe.description)

    console.print(table)


@app.command()
def verify(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Verify one config instead of the whole catalogue.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Machine-readable rows on stdout."
    ),
    templates: bool = typer.Option(
        True, "--templates/--no-templates",
        help="Also verify src/soup_cli/templates and examples/ (catalogue mode).",
    ),
):
    """Check that a config can attach a LoRA adapter, without downloading weights.

    Builds each base's architecture on the meta device from ``config.json`` alone,
    then runs Soup's own target resolution and the real peft attach. Exits 1 only
    when something cannot attach: a gated repo, or one needing trust_remote_code,
    is reported as unverified rather than broken (#1116).
    """
    import json as _json

    from soup_cli.utils.attach_preflight import (
        PreflightReport,
        Verdict,
        build_on_meta,
        check_attach,
        load_hf_config,
    )

    configs = _configs_to_verify(config, templates)
    if not configs:
        console.print("[red]Nothing to verify.[/]")
        raise typer.Exit(1)

    report = PreflightReport()
    for name, cfg in configs:
        report.checks.append(
            check_attach(
                name, cfg, load_hf_config=load_hf_config, build_model=build_on_meta
            )
        )

    if json_out:
        console.print_json(
            _json.dumps(
                [
                    {
                        "name": c.name, "base": c.base, "task": c.task,
                        "verdict": c.verdict.value, "model_type": c.model_type,
                        "adapted": c.adapted, "expert_modules": c.expert_modules,
                        "vision_modules": c.vision_modules, "detail": c.detail,
                    }
                    for c in report.checks
                ]
            )
        )
    else:
        _print_preflight(report, Verdict)
    raise typer.Exit(report.exit_code)


def _configs_to_verify(config: Optional[str], templates: bool):
    """``(name, SoupConfig)`` for everything in scope; a config that will not
    parse is surfaced here rather than counted as an attach failure."""
    from soup_cli.config.loader import load_config_from_string

    pairs: list[tuple[str, str]] = []
    if config:
        path = Path(config)
        if not path.is_file():
            console.print(f"[red]Config not found:[/] {config}")
            raise typer.Exit(1)
        pairs.append((path.name, path.read_text(encoding="utf-8")))
    else:
        from soup_cli.recipes.catalog import RECIPES

        pairs.extend((name, recipe.yaml_str) for name, recipe in RECIPES.items())
        if templates:
            root = Path(__file__).resolve().parents[1]
            for base in (root / "templates", root.parents[1] / "examples"):
                if base.is_dir():
                    pairs.extend(
                        (str(p.relative_to(base.parent)), p.read_text(encoding="utf-8"))
                        for p in sorted(base.rglob("*.yaml"))
                    )

    loaded = []
    for name, text in pairs:
        try:
            loaded.append((name, load_config_from_string(text)))
        except Exception as exc:  # noqa: BLE001 --- a parse failure is #330's job
            console.print(f"[dim]skipped {name}: does not parse ({type(exc).__name__})[/]")
    return loaded


def _print_preflight(report, verdict_cls) -> None:
    """One row per config that did not simply attach, then the counts.

    Printing 91 green rows buries the 42 that matter, so ATTACHES is summarised
    and everything else is listed.
    """
    failures = report.failures
    unverified = report.by_verdict(verdict_cls.UNVERIFIED)

    if failures:
        table = Table(title="Cannot attach")
        table.add_column("Config", style="bold cyan")
        table.add_column("model_type", style="magenta")
        table.add_column("Stage", style="yellow")
        table.add_column("Why")
        for check in failures:
            table.add_row(
                check.name, str(check.model_type), check.stage, check.detail[:90]
            )
        console.print(table)

    if unverified:
        console.print(
            f"[yellow]{len(unverified)} config(s) could not be checked here[/] "
            "[dim](gated, remote code, or unreadable by this transformers) — "
            "not counted as failures.[/]"
        )

    attaches = report.by_verdict(verdict_cls.ATTACHES)
    experts = sum(1 for c in attaches if c.expert_modules)
    vision = [c.name for c in attaches if c.vision_modules]
    console.print(
        f"[green]{len(attaches)} attach[/], "
        f"[red]{len(failures)} cannot[/], "
        f"{len(report.by_verdict(verdict_cls.NO_ADAPTER))} have no adapter, "
        f"{len(unverified)} unverified."
    )
    if experts:
        console.print(f"[dim]{experts} adapted expert modules.[/]")
    if vision:
        console.print(f"[yellow]adapted a vision tower:[/] {', '.join(vision)}")


def _suggest_recipes(query: str, n: int = 3) -> list[str]:
    """v0.40.1 Part E / M3 — return up to ``n`` close-matching recipe ids."""
    from difflib import get_close_matches

    from soup_cli.recipes.catalog import RECIPES

    try:
        names = list(RECIPES.keys()) if hasattr(RECIPES, "keys") else [
            r.name for r in RECIPES
        ]
    except Exception:  # noqa: BLE001
        return []
    return get_close_matches(query, names, n=n, cutoff=0.6)
