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
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading models that ship custom Python via auto_map. "
            "Default deny (v0.36.0). Only enable if you trust the source."
        ),
    ),
    save_format: str = typer.Option(
        "fp16",
        "--save-format",
        help=(
            "Merged-checkpoint save format. fp16 (default) writes a "
            "standard fp16 merge. 4bit / 4bit_forced write a single "
            "BNB-4bit-quantized merge without the dequant-merge-requant "
            "cycle (v0.53.1 #142)."
        ),
    ),
):
    """Merge a LoRA adapter with its base model into a full model."""
    # v0.53.1 #142 — validate save_format up front
    from soup_cli.utils.save_formats import validate_merge_save_format
    try:
        save_format_canonical = validate_merge_save_format(save_format)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2)

    # v0.53.1 — early cwd containment on --output (security review M4).
    # Mirrors v0.20.0 / v0.40.2 policy: containment check fires at the CLI
    # boundary, not deferred to the deeper helper.
    from soup_cli.utils.paths import is_under_cwd as _is_under_cwd
    if not _is_under_cwd(output):
        console.print(
            f"[red]--output {output!r} must stay under cwd[/]"
        )
        raise typer.Exit(2)
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
        model_dtype = dtype_map[dtype]

        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code,
            resolve_trust_remote_code,
        )

        requires = model_requires_trust_remote_code(str(adapter_path)) or False
        trc = resolve_trust_remote_code(
            base,
            requested=trust_remote_code,
            console=console,
            requires_remote_code=requires,
        )

        console.print(f"[dim]Loading base model: {base}...[/]")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=model_dtype,
            trust_remote_code=trc,
            device_map="cpu",
        )

        console.print(f"[dim]Loading LoRA adapter: {adapter_path}...[/]")
        model = PeftModel.from_pretrained(model, str(adapter_path))

        console.print("[dim]Merging weights...[/]")
        model = model.merge_and_unload()

        if save_format_canonical == "fp16":
            console.print(f"[dim]Saving merged model to {output_path}...[/]")
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path))

            console.print("[dim]Saving tokenizer...[/]")
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_path), trust_remote_code=trc
            )
            tokenizer.save_pretrained(str(output_path))
        else:
            # v0.53.1 #142 — 4bit / 4bit_forced merged checkpoint.
            # Two-stage: first write an fp16 merge to a tempdir, then
            # reload with BNB-4bit config and save to output.
            import tempfile

            from soup_cli.utils.save_formats import merge_4bit

            with tempfile.TemporaryDirectory(
                prefix=".soup_4bit_merge_", dir=str(Path.cwd()),
            ) as staged:
                staged_path = Path(staged)
                console.print(
                    f"[dim]Staging fp16 merge in {staged_path.name}...[/]"
                )
                model.save_pretrained(str(staged_path))
                tokenizer = AutoTokenizer.from_pretrained(
                    str(adapter_path), trust_remote_code=trc
                )
                tokenizer.save_pretrained(str(staged_path))

                # Free the in-memory fp16 model before reloading 4bit
                del model
                console.print(
                    f"[dim]Re-loading + saving BNB-4bit merge "
                    f"({save_format_canonical}) to {output_path}...[/]"
                )
                merge_4bit(
                    merged_dir=str(staged_path),
                    output_dir=str(output_path),
                    forced=(save_format_canonical == "4bit_forced"),
                    dtype="bfloat16" if dtype == "bfloat16" else "float16",
                    trust_remote_code=trc,
                )

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
