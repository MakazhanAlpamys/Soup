"""v0.44.0 Part D — `soup quantize` standalone (ergonomic alias for `soup export`).

Prints the equivalent `soup export ...` invocation. We intentionally do NOT
in-process call `commands.export.export` because Typer commands aren't
designed for re-entry from another command — pre-validation of arguments,
signal handlers, and Rich Console lifetimes can interact badly. The user
gets a copy-pasteable command instead.
"""

from __future__ import annotations

import shlex

import typer
from rich.console import Console
from rich.markup import escape

console = Console()

VALID_FORMATS = frozenset({"gguf", "gptq", "awq", "onnx", "tensorrt"})


def quantize(
    model_path: str = typer.Argument(
        ...,
        help="Source checkpoint (safetensors directory or single .safetensors).",
    ),
    to: str = typer.Option(
        "gguf",
        "--to",
        help="Target format: gguf | gptq | awq | onnx | tensorrt.",
    ),
    bits: int = typer.Option(
        4,
        "--bits",
        help="Quantization bits (1-16; respected by gguf/gptq/awq).",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination directory (default: <model_path>-<to>).",
    ),
) -> None:
    """Quantize a model - ergonomic alias for `soup export --format <to>`.

    Example:
      soup quantize ./out --to gguf --bits 4
    """
    canonical = to.lower().strip()
    if canonical not in VALID_FORMATS:
        console.print(
            f"[red]--to must be one of {sorted(VALID_FORMATS)}; got {to!r}[/]"
        )
        raise typer.Exit(code=2)
    if isinstance(bits, bool) or not isinstance(bits, int):
        console.print("[red]--bits must be int[/]")
        raise typer.Exit(code=2)
    if not (1 <= bits <= 16):
        console.print("[red]--bits must be in [1, 16][/]")
        raise typer.Exit(code=2)
    parts = ["soup", "export", "--model", model_path, "--format", canonical]
    if canonical == "gguf":
        parts.extend(["--quant", f"q{bits}_K_M"])
    elif canonical in ("gptq", "awq"):
        parts.extend(["--bits", str(bits)])
    if output:
        parts.extend(["--output", output])
    rendered = " ".join(shlex.quote(part) for part in parts)
    console.print("[cyan]Run:[/]")
    console.print(f"  [bold]{escape(rendered)}[/]")
    console.print(
        "[dim]Tip: `soup export --help` lists every advanced quantization flag.[/]"
    )
