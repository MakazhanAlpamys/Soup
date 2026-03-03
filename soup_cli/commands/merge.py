"""soup merge — merge LoRA adapter with base model into a full model."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def merge(
    adapter: str = typer.Option(
        ...,
        "--adapter",
        "-a",
        help="Path to the LoRA adapter directory",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model ID. Auto-detected from adapter_config.json if not set.",
    ),
    output: str = typer.Option(
        "./merged",
        "--output",
        "-o",
        help="Output directory for the merged model",
    ),
    dtype: str = typer.Option(
        "float16",
        "--dtype",
        help="Data type for the merged model: float16, bfloat16, float32",
    ),
):
    """Merge a LoRA adapter with its base model into a full model."""
    adapter_path = Path(adapter)

    # --- Validate adapter ---
    if not adapter_path.exists():
        console.print(f"[red]Adapter path not found: {adapter_path}[/]")
        raise typer.Exit(1)

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        console.print(
            f"[red]Not a LoRA adapter: {adapter_path}[/]\n"
            "Expected adapter_config.json in the directory."
        )
        raise typer.Exit(1)

    # --- Resolve base model ---
    if not base:
        base = _detect_base_model(adapter_config_path)
        if not base:
            console.print(
                "[red]Cannot detect base model from adapter_config.json.[/]\n"
                "Please specify with [bold]--base[/] flag."
            )
            raise typer.Exit(1)

    # --- Validate dtype ---
    valid_dtypes = ("float16", "bfloat16", "float32")
    if dtype not in valid_dtypes:
        console.print(f"[red]Invalid dtype: {dtype}. Must be one of: {', '.join(valid_dtypes)}[/]")
        raise typer.Exit(1)

    output_path = Path(output)

    console.print(
        Panel(
            f"Adapter: [bold]{adapter_path}[/]\n"
            f"Base:    [bold]{base}[/]\n"
            f"Output:  [bold]{output_path}[/]\n"
            f"Dtype:   [bold]{dtype}[/]",
            title="Merge Plan",
        )
    )

    # --- Merge ---
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]

        console.print(f"[dim]Loading base model: {base}...[/]")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="cpu",
        )

        console.print(f"[dim]Loading LoRA adapter: {adapter_path}...[/]")
        model = PeftModel.from_pretrained(model, str(adapter_path))

        console.print("[dim]Merging weights...[/]")
        model = model.merge_and_unload()

        console.print(f"[dim]Saving merged model to {output_path}...[/]")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))

        console.print("[dim]Saving tokenizer...[/]")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        tokenizer.save_pretrained(str(output_path))

    except ImportError as exc:
        console.print(f"[red]Missing dependency: {exc}[/]")
        console.print("Run: [bold]pip install torch transformers peft[/]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Merge failed: {exc}[/]")
        raise typer.Exit(1)

    # Calculate output size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    size_str = _format_size(total_size)

    console.print(
        Panel(
            f"Output: [bold]{output_path}[/]\n"
            f"Size:   [bold]{size_str}[/]\n\n"
            f"Next steps:\n"
            f"  [bold]soup chat --model {output_path}[/]\n"
            f"  [bold]soup push --model {output_path} --repo user/model[/]\n"
            f"  [bold]soup export --model {output_path} --format gguf[/]",
            title="[bold green]Merge Complete![/]",
        )
    )


def _detect_base_model(adapter_config_path: Path) -> Optional[str]:
    """Read base_model_name_or_path from adapter_config.json."""
    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path")
    except (json.JSONDecodeError, OSError):
        return None


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
