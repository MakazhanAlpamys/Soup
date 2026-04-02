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
        console.print("[dim]Run 'soup recipes list' to see all recipes.[/]")
        raise typer.Exit(1)

    console.print(Panel(
        Syntax(recipe.yaml_str, "yaml", theme="monokai"),
        title=f"[bold green]{name}[/] — {recipe.description}",
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
