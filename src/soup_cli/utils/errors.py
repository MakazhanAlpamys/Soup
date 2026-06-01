"""Friendly error handling — maps raw exceptions to actionable messages."""

import traceback

from rich.console import Console
from rich.panel import Panel

console = Console(stderr=True)

# v0.71.0 — the heavy training stack (torch / transformers / peft / trl /
# datasets / bitsandbytes / accelerate) moved out of the core install into the
# `[train]` extra. A missing one of these surfaces this single, actionable fix.
# The `\\[` escapes the literal `[` for Rich markup (it renders as `[train]`).
_TRAIN_FIX = "Training needs the \\[train] extra. Run: pip install 'soup-cli\\[train]'"

# Map known error patterns to (short message, fix suggestion)
ERROR_MAP = [
    # CUDA OOM
    (
        "CUDA out of memory",
        "GPU ran out of memory during training.",
        (
            "Try --batch-size <half> or --grad-accum <double> (keeps effective "
            "batch size); or use quantization: 4bit, or use a smaller model."
        ),
    ),
    (
        "OutOfMemoryError",
        "GPU ran out of memory.",
        (
            "Try --batch-size <half> or --grad-accum <double> (keeps effective "
            "batch size); or use quantization: 4bit, or use a smaller model."
        ),
    ),
    # Missing optional deps
    (
        "No module named 'fastapi'",
        "FastAPI is not installed (needed for soup serve).",
        "Run: pip install 'soup-cli\\[serve]'",
    ),
    (
        "No module named 'uvicorn'",
        "Uvicorn is not installed (needed for soup serve).",
        "Run: pip install 'soup-cli\\[serve]'",
    ),
    (
        "No module named 'datasketch'",
        "Datasketch is not installed (needed for dedup).",
        "Run: pip install 'soup-cli\\[data]'",
    ),
    (
        "No module named 'lm_eval'",
        "lm-evaluation-harness is not installed (needed for eval).",
        "Run: pip install 'soup-cli\\[eval]'",
    ),
    (
        "No module named 'wandb'",
        "Weights & Biases is not installed.",
        "Run: pip install wandb",
    ),
    (
        "No module named 'deepspeed'",
        "DeepSpeed is not installed.",
        "Run: pip install 'soup-cli\\[deepspeed]'",
    ),
    (
        "No module named 'httpx'",
        "httpx is not installed (needed for data generate).",
        "Run: pip install 'soup-cli\\[generate]'",
    ),
    # Heavy training stack — all moved to the [train] extra in v0.71.0.
    (
        "No module named 'torch'",
        "PyTorch is not installed (needed for training).",
        _TRAIN_FIX,
    ),
    (
        "No module named 'transformers'",
        "Transformers is not installed (needed for training).",
        _TRAIN_FIX,
    ),
    (
        "No module named 'peft'",
        "PEFT is not installed (needed for LoRA training).",
        _TRAIN_FIX,
    ),
    (
        "No module named 'trl'",
        "TRL is not installed (needed for training).",
        _TRAIN_FIX,
    ),
    (
        "No module named 'datasets'",
        "Datasets is not installed (needed for training).",
        _TRAIN_FIX,
    ),
    (
        "No module named 'bitsandbytes'",
        "BitsAndBytes is not installed (needed for quantization).",
        _TRAIN_FIX,
    ),
    (
        "No module named 'accelerate'",
        "Accelerate is not installed (needed for training).",
        _TRAIN_FIX,
    ),
    # CPU / quantization issues
    (
        "expanded size of the tensor",
        "Model generation failed (empty tensors, likely GRPO/PPO on CPU).",
        "GRPO requires a CUDA GPU. For CPU training, use SFT or DPO instead.",
    ),
    (
        "expected m1 and m2 to have the same dtype",
        "Dtype mismatch (likely 4bit quantization on CPU).",
        "Use a GPU, or set quantization: none in your config for CPU training.",
    ),
    (
        "Your setup doesn't support bf16",
        "This training task requires GPU with bf16 support.",
        "Use a CUDA GPU, or try a simpler task (SFT/DPO work on CPU).",
    ),
    (
        "use_cpu",
        "This training task requires use_cpu flag on CPU-only systems.",
        "Use a CUDA GPU for best results, or upgrade trl: pip install -U trl",
    ),
    (
        "nms does not exist",
        "torchvision version is incompatible with torch.",
        "Run: pip install torchvision --force-reinstall (or check soup doctor).",
    ),
    # Connection errors
    (
        "ConnectionError",
        "Network connection failed.",
        "Check your internet connection. If downloading from HuggingFace, check HF_TOKEN.",
    ),
    (
        "HTTPError",
        "HTTP request failed.",
        "Check your internet connection and API keys (OPENAI_API_KEY, HF_TOKEN).",
    ),
    (
        "ConnectTimeout",
        "Connection timed out.",
        "Check your internet connection and try again.",
    ),
    # File not found
    (
        "No such file or directory",
        None,  # Will use the original message
        "Check the file path. Run 'soup init' to create a config.",
    ),
    # YAML errors
    (
        "yaml.scanner.ScannerError",
        "Invalid YAML syntax in config file.",
        "Check your soup.yaml for syntax errors (indentation, colons, quotes).",
    ),
    # Pydantic validation
    (
        "validation error",
        "Config validation failed.",
        "Check your soup.yaml values. Run 'soup init' to generate a valid config.",
    ),
    # Auth errors
    (
        "401",
        "Authentication failed.",
        "Check your API key or token (HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY).",
    ),
    (
        "403",
        "Access denied.",
        "Check your permissions. Some models require accepting a license on HuggingFace.",
    ),
]


def format_friendly_error(exc: Exception, verbose: bool = False) -> None:
    """Display a friendly error message for the given exception.

    In normal mode: 2-3 lines with error + fix suggestion.
    In verbose mode: full traceback.
    """
    exc_str = str(exc)
    exc_type = type(exc).__name__

    # Search for known error patterns
    for pattern, short_msg, fix in ERROR_MAP:
        if pattern in exc_str or pattern in exc_type:
            error_msg = short_msg or exc_str
            console.print(f"\n[bold red]Error:[/] {error_msg}")
            console.print(f"[green]Fix:[/] {fix}")
            if verbose:
                console.print()
                console.print(
                    Panel(
                        traceback.format_exc(),
                        title="[dim]Full Traceback[/]",
                        border_style="dim",
                    )
                )
            return

    # Unknown error — show type + message
    console.print(f"\n[bold red]Error:[/] {exc_type}: {exc_str}")
    console.print("[dim]Run with --verbose for the full traceback.[/]")
    if verbose:
        console.print()
        console.print(
            Panel(
                traceback.format_exc(),
                title="[dim]Full Traceback[/]",
                border_style="dim",
            )
        )
