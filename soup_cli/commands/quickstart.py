"""soup quickstart — one command for a complete demo (create data + config + train)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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

_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_LOW_VRAM_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_LOW_VRAM_THRESHOLD_GB = 6.0


def _pick_quickstart_model() -> tuple[str, str | None]:
    """v0.40.1 Part C / G1 — pick the demo model based on detected VRAM.

    Returns ``(model_id, advisory)``. On <=6 GB GPUs (e.g. RTX 3050 4 GB)
    TinyLlama 1.1B doesn't fit and crashes at step 0; auto-switch to
    SmolLM2-135M (verified to train in 5 s on RTX 3050 4 GB).
    """
    try:
        import torch
    except ImportError:
        return _DEFAULT_MODEL, None
    if not torch.cuda.is_available():
        return _DEFAULT_MODEL, None
    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = float(getattr(props, "total_memory", 0)) / 1024**3
    except (RuntimeError, OSError):
        return _DEFAULT_MODEL, None
    if total_gb and total_gb <= _LOW_VRAM_THRESHOLD_GB:
        return (
            _LOW_VRAM_MODEL,
            f"Detected {total_gb:.1f} GB VRAM (≤{_LOW_VRAM_THRESHOLD_GB:.0f}) — "
            f"using {_LOW_VRAM_MODEL} instead of {_DEFAULT_MODEL}.",
        )
    return _DEFAULT_MODEL, None


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
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Output directory for data, config, and run artifacts "
            "(default: current directory)."
        ),
    ),
):
    """Run a complete demo: create sample data, config, and train."""
    import os

    from soup_cli.utils.paths import is_under_cwd

    model_id, advisory = _pick_quickstart_model()
    if advisory:
        console.print(f"[yellow]{advisory}[/]")

    # Resolve output directory (containment-checked)
    if output is None:
        out_dir = Path.cwd()
    else:
        if not is_under_cwd(output):
            console.print(
                f"[red]--output must stay under the current working directory; "
                f"got: {output}[/]"
            )
            raise typer.Exit(2)
        out_dir = Path(os.path.realpath(output))
        out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            "This will:\n"
            f"  1. Create [bold]{out_dir}/quickstart_data.jsonl[/] (20 examples)\n"
            f"  2. Create [bold]{out_dir}/quickstart_soup.yaml[/] config\n"
            "  3. Train a tiny LoRA adapter (~1 min on GPU)\n\n"
            f"Model: [bold]{model_id}[/]",
            title="[bold]Soup Quickstart[/]",
        )
    )

    if not yes and not dry_run:
        confirm = typer.confirm("Continue?", default=True)
        if not confirm:
            console.print("[yellow]Cancelled.[/]")
            raise typer.Exit()

    # 1. Create demo data
    data_path = out_dir / "quickstart_data.jsonl"
    if data_path.exists():
        console.print(f"[yellow]Data file already exists:[/] {data_path}")
    else:
        with open(data_path, "w", encoding="utf-8") as fh:
            for entry in DEMO_DATA:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        console.print(f"[green]Created:[/] {data_path} ({len(DEMO_DATA)} examples)")

    # 2. Create demo config
    config_path = out_dir / "quickstart_soup.yaml"
    if config_path.exists():
        console.print(f"[yellow]Config file already exists:[/] {config_path}")
    else:
        rendered = DEMO_CONFIG.replace(_DEFAULT_MODEL, model_id)
        # When --output is set, retarget data + run dirs into that dir.
        if output is not None:
            rendered = rendered.replace(
                "./quickstart_data.jsonl", str(data_path)
            ).replace("./quickstart_output", str(out_dir / "quickstart_output"))
        config_path.write_text(rendered, encoding="utf-8")
        console.print(f"[green]Created:[/] {config_path}")
    # Also write a `soup.yaml` symlink-style alias for tools that look for it.
    soup_yaml = out_dir / "soup.yaml"
    if not soup_yaml.exists() and output is not None:
        try:
            soup_yaml.write_text(
                config_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        except OSError:
            pass

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
