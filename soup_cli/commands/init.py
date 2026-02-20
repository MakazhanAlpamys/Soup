"""soup init — interactive project setup wizard."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from soup_cli.config.schema import TEMPLATES

console = Console()


def init(
    template: str = typer.Option(
        None,
        "--template",
        "-t",
        help="Use a template: chat, code, medical",
    ),
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output config file path",
    ),
):
    """Create a new soup.yaml config interactively or from a template."""
    output_path = Path(output)

    if output_path.exists():
        overwrite = typer.confirm(f"{output_path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit()

    if template:
        if template not in TEMPLATES:
            console.print(f"[red]Unknown template: {template}[/]")
            console.print(f"Available: {', '.join(TEMPLATES.keys())}")
            raise typer.Exit(1)
        config_text = TEMPLATES[template]
        console.print(f"[green]Using template:[/] {template}")
    else:
        config_text = _interactive_wizard()

    output_path.write_text(config_text, encoding="utf-8")
    console.print(
        Panel(
            f"[bold green]Config saved to {output_path}[/]\n\n"
            f"Next step: [bold]soup train --config {output_path}[/]",
            title="Ready!",
        )
    )


def _interactive_wizard() -> str:
    """Walk user through config creation."""
    console.print(Panel("[bold]Soup Config Wizard[/]", subtitle="Let's set up your training"))

    base_model = Prompt.ask(
        "Base model",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    task = Prompt.ask("Task", choices=["sft", "dpo"], default="sft")
    data_path = Prompt.ask("Training data path", default="./data/train.jsonl")
    data_format = Prompt.ask("Data format", choices=["alpaca", "sharegpt", "chatml"], default="alpaca")
    epochs = Prompt.ask("Epochs", default="3")
    use_qlora = Prompt.ask("Use QLoRA (4-bit)?", choices=["yes", "no"], default="yes")

    quantization = "4bit" if use_qlora == "yes" else "none"

    return f"""# Soup training config
# Docs: https://github.com/MakazhanAlpamys/Soup

base: {base_model}
task: {task}

data:
  train: {data_path}
  format: {data_format}
  val_split: 0.1

training:
  epochs: {epochs}
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: {quantization}

output: ./output
"""
