"""soup adapters — LoRA adapter management (list, info, compare)."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=True)


def _find_adapters(directory: Path, max_depth: int = 6) -> list[Path]:
    """Recursively find directories containing adapter_config.json.

    Limited to max_depth levels and 200 results to prevent runaway scans.
    """
    adapters = []
    for config_file in directory.rglob("adapter_config.json"):
        try:
            depth = len(config_file.relative_to(directory).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            adapters.append(config_file.parent)
        if len(adapters) >= 200:
            break
    return sorted(adapters)


def _read_adapter_config(adapter_path: Path) -> dict:
    """Read adapter_config.json from an adapter directory."""
    config_file = adapter_path / "adapter_config.json"
    with open(config_file, encoding="utf-8") as fh:
        return json.load(fh)


def _get_adapter_size(adapter_path: Path) -> str:
    """Get total size of adapter files on disk."""
    total_bytes = 0
    for file_path in adapter_path.iterdir():
        if file_path.is_file():
            total_bytes += file_path.stat().st_size

    if total_bytes < 1024:
        return f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"


@app.command(name="list")
def list_adapters(
    directory: str = typer.Argument(
        ".", help="Directory to scan for adapters (recursive)"
    ),
):
    """Scan a directory for LoRA adapters and list them."""
    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        console.print(f"[red]Directory not found: {directory}[/]")
        raise typer.Exit(1)

    adapters = _find_adapters(dir_path)

    if not adapters:
        console.print("[dim]No adapters found in this directory.[/]")
        return

    table = Table(title=f"Adapters in {dir_path}")
    table.add_column("Path", style="bold")
    table.add_column("Base Model")
    table.add_column("LoRA r")
    table.add_column("Type")
    table.add_column("Size")

    for adapter_path in adapters:
        try:
            config = _read_adapter_config(adapter_path)
            base = config.get("base_model_name_or_path", "unknown")
            lora_r = str(config.get("r", "?"))
            peft_type = config.get("peft_type", "?")
            size = _get_adapter_size(adapter_path)

            # Shorten path relative to scan directory
            try:
                rel_path = adapter_path.relative_to(dir_path)
            except ValueError:
                rel_path = adapter_path
            table.add_row(str(rel_path), base, lora_r, peft_type, size)
        except (json.JSONDecodeError, OSError):
            table.add_row(str(adapter_path), "[red]error[/]", "-", "-", "-")

    console.print(table)
    console.print(f"\n[dim]Found {len(adapters)} adapter(s).[/]")


@app.command()
def info(
    adapter: str = typer.Argument(..., help="Path to adapter directory"),
):
    """Show detailed metadata for a LoRA adapter."""
    adapter_path = Path(adapter).resolve()
    config_file = adapter_path / "adapter_config.json"

    if not adapter_path.exists():
        console.print(f"[red]Adapter not found: {adapter}[/]")
        raise typer.Exit(1)

    if not config_file.exists():
        console.print(f"[red]No adapter_config.json in: {adapter}[/]")
        raise typer.Exit(1)

    config = _read_adapter_config(adapter_path)
    size = _get_adapter_size(adapter_path)

    base_model = config.get("base_model_name_or_path", "unknown")
    lora_r = config.get("r", "?")
    lora_alpha = config.get("lora_alpha", "?")
    lora_dropout = config.get("lora_dropout", "?")
    peft_type = config.get("peft_type", "?")
    task_type = config.get("task_type", "?")
    target_modules = config.get("target_modules", [])

    if isinstance(target_modules, list):
        modules_str = ", ".join(target_modules)
    else:
        modules_str = str(target_modules)

    info_text = (
        f"Base model: [bold]{base_model}[/]\n"
        f"PEFT type:  [bold]{peft_type}[/]\n"
        f"Task:       [bold]{task_type}[/]\n"
        f"LoRA rank:  [bold]{lora_r}[/], alpha: [bold]{lora_alpha}[/], "
        f"dropout: [bold]{lora_dropout}[/]\n"
        f"Targets:    [bold]{modules_str}[/]\n"
        f"Size on disk: [bold]{size}[/]"
    )

    console.print(Panel(info_text, title=f"Adapter Info — {adapter_path.name}"))


@app.command()
def compare(
    adapter1: str = typer.Argument(..., help="Path to first adapter"),
    adapter2: str = typer.Argument(..., help="Path to second adapter"),
):
    """Compare two LoRA adapters side-by-side."""
    path1 = Path(adapter1).resolve()
    path2 = Path(adapter2).resolve()

    for label, path in [("Adapter 1", path1), ("Adapter 2", path2)]:
        config_file = path / "adapter_config.json"
        if not path.exists():
            console.print(f"[red]{label} not found: {path}[/]")
            raise typer.Exit(1)
        if not config_file.exists():
            console.print(f"[red]{label} has no adapter_config.json: {path}[/]")
            raise typer.Exit(1)

    config1 = _read_adapter_config(path1)
    config2 = _read_adapter_config(path2)

    table = Table(title="Adapter Comparison")
    table.add_column("Field", style="bold")
    table.add_column(path1.name, justify="center")
    table.add_column(path2.name, justify="center")

    # Fields to compare
    fields = [
        ("Base model", "base_model_name_or_path"),
        ("PEFT type", "peft_type"),
        ("Task type", "task_type"),
        ("LoRA rank (r)", "r"),
        ("LoRA alpha", "lora_alpha"),
        ("LoRA dropout", "lora_dropout"),
        ("Target modules", "target_modules"),
    ]

    for label, key in fields:
        val1 = config1.get(key, "-")
        val2 = config2.get(key, "-")

        # Format lists
        if isinstance(val1, list):
            val1 = ", ".join(str(item) for item in val1)
        if isinstance(val2, list):
            val2 = ", ".join(str(item) for item in val2)

        val1_str = str(val1)
        val2_str = str(val2)

        # Highlight differences
        if val1_str != val2_str:
            val1_str = f"[yellow]{val1_str}[/]"
            val2_str = f"[yellow]{val2_str}[/]"

        table.add_row(label, val1_str, val2_str)

    # Add size comparison
    size1 = _get_adapter_size(path1)
    size2 = _get_adapter_size(path2)
    table.add_row("Size on disk", size1, size2)

    console.print(table)
