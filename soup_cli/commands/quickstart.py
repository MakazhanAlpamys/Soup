"""soup quickstart — one command for a complete demo (create data + config + train)."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

# Minimal demo dataset — 20 instruction-following examples
DEMO_DATA = [
    {"instruction": "What is machine learning?", "input": "",
     "output": "Machine learning is a subset of AI where computers learn patterns from data."},
    {"instruction": "Explain what a neural network is.", "input": "",
     "output": "A neural network is a computing system inspired by biological neural networks."},
    {"instruction": "What is Python?", "input": "",
     "output": "Python is a high-level programming language known for its readability."},
    {"instruction": "Define overfitting.", "input": "",
     "output": "Overfitting is when a model learns noise in training data instead of patterns."},
    {"instruction": "What is a GPU?", "input": "",
     "output": "A GPU is a specialized processor designed for parallel computation."},
    {"instruction": "Explain LoRA.", "input": "",
     "output": "LoRA (Low-Rank Adaptation) is a technique to fine-tune large models efficiently."},
    {"instruction": "What is tokenization?", "input": "",
     "output": "Tokenization is the process of splitting text into smaller units called tokens."},
    {"instruction": "Define transfer learning.", "input": "",
     "output": "Transfer learning uses a pre-trained model as a starting point for a new task."},
    {"instruction": "What is an epoch?", "input": "",
     "output": "An epoch is one complete pass through the entire training dataset."},
    {"instruction": "Explain gradient descent.", "input": "",
     "output": "Gradient descent is an optimization algorithm that minimizes loss iteratively."},
    {"instruction": "What is a loss function?", "input": "",
     "output": "A loss function measures how far model predictions are from actual values."},
    {"instruction": "Define batch size.", "input": "",
     "output": "Batch size is the number of training samples processed before updating weights."},
    {"instruction": "What is quantization?", "input": "",
     "output": "Quantization reduces model precision (e.g., 32-bit to 4-bit) to save memory."},
    {"instruction": "Explain attention mechanism.", "input": "",
     "output": "Attention lets models focus on relevant parts of input when generating output."},
    {"instruction": "What is fine-tuning?", "input": "",
     "output": "Fine-tuning is training a pre-trained model on task-specific data."},
    {"instruction": "Define learning rate.", "input": "",
     "output": "Learning rate controls how much model weights change during each training step."},
    {"instruction": "What is a transformer?", "input": "",
     "output": "A transformer is a neural network architecture based on self-attention."},
    {"instruction": "Explain backpropagation.", "input": "",
     "output": "Backpropagation computes gradients by propagating errors backward through layers."},
    {"instruction": "What is RLHF?", "input": "",
     "output": "RLHF trains models using human feedback as a reward signal."},
    {"instruction": "Define inference.", "input": "",
     "output": "Inference is using a trained model to make predictions on new data."},
]

DEMO_CONFIG = """# Soup Quickstart Config — auto-generated demo
base: TinyLlama/TinyLlama-1.1B-Chat-v1.0

task: sft

data:
  train: ./quickstart_data.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 1
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
  quantization: "none"

output: ./quickstart_output
"""


def quickstart(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Create data and config only, do not train",
    ),
):
    """Run a complete demo: create sample data, config, and train."""
    console.print(
        Panel(
            "This will:\n"
            "  1. Create [bold]quickstart_data.jsonl[/] (20 examples)\n"
            "  2. Create [bold]quickstart_soup.yaml[/] config\n"
            "  3. Train a tiny LoRA adapter (~1 min on GPU)\n\n"
            "Model: [bold]TinyLlama/TinyLlama-1.1B-Chat-v1.0[/]",
            title="[bold]Soup Quickstart[/]",
        )
    )

    if not yes and not dry_run:
        confirm = typer.confirm("Continue?", default=True)
        if not confirm:
            console.print("[yellow]Cancelled.[/]")
            raise typer.Exit()

    # 1. Create demo data
    data_path = Path("quickstart_data.jsonl")
    if data_path.exists():
        console.print(f"[yellow]Data file already exists:[/] {data_path}")
    else:
        with open(data_path, "w", encoding="utf-8") as fh:
            for entry in DEMO_DATA:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        console.print(f"[green]Created:[/] {data_path} ({len(DEMO_DATA)} examples)")

    # 2. Create demo config
    config_path = Path("quickstart_soup.yaml")
    if config_path.exists():
        console.print(f"[yellow]Config file already exists:[/] {config_path}")
    else:
        config_path.write_text(DEMO_CONFIG, encoding="utf-8")
        console.print(f"[green]Created:[/] {config_path}")

    if dry_run:
        console.print("\n[yellow]Dry run - files created, skipping training.[/]")
        console.print(f"To train: [bold]soup train --config {config_path}[/]")
        raise typer.Exit()

    # 3. Train — invoke via subprocess so Typer resolves defaults properly
    console.print("\n[bold]Starting training...[/]\n")
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "soup_cli.cli", "train", "--config", str(config_path), "--yes"],
        check=False,
    )
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
