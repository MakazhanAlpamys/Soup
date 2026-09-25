"""Friendly error handling — maps raw exceptions to actionable messages."""

import re
import traceback

from rich.console import Console
from rich.panel import Panel

console = Console(stderr=True)

# v0.71.0 — the heavy training stack (torch / transformers / peft / trl /
# datasets / bitsandbytes / accelerate) moved out of the core install into the
# `[train]` extra. A missing one of these surfaces this single, actionable fix.
# The `\\[` escapes the literal `[` for Rich markup (it renders as `[train]`).
_TRAIN_FIX = "Training needs the \\[train] extra. Run: pip install \"soup-cli\\[train]\""

# Map known error patterns to (short message, fix suggestion)
ERROR_MAP = [
    # CUDA OOM
    (
        "CUDA out of memory",
        "GPU ran out of memory during training.",
        (
            "Try --batch-size <half> or --grad-accum <double> "
            "(keeps effective batch size); enable gradient_checkpointing, "
            "use 4bit quantization, or use a smaller model."
        ),
    ),
    (
        "OutOfMemoryError",
        "GPU ran out of memory.",
        (
            "Try --batch-size <half> or --grad-accum <double> "
            "(keeps effective batch size); enable gradient_checkpointing, "
            "use 4bit quantization, or use a smaller model."
        ),
    ),
    # Missing optional deps
    (
        "No module named 'fastapi'",
        "FastAPI is not installed (needed for soup serve).",
        "Run: pip install \"soup-cli\\[serve]\"",
    ),
    (
        "No module named 'uvicorn'",
        "Uvicorn is not installed (needed for soup serve).",
        "Run: pip install \"soup-cli\\[serve]\"",
    ),
    (
        "No module named 'datasketch'",
        "Datasketch is not installed (needed for dedup).",
        "Run: pip install \"soup-cli\\[data]\"",
    ),
    (
        "No module named 'lm_eval'",
        "lm-evaluation-harness is not installed (needed for eval).",
        "Run: pip install \"soup-cli\\[eval]\"",
    ),
    (
        "No module named 'wandb'",
        "Weights & Biases is not installed.",
        "Run: pip install wandb",
    ),
    (
        "No module named 'deepspeed'",
        "DeepSpeed is not installed.",
        "Run: pip install \"soup-cli\\[deepspeed]\"",
    ),
    (
        "No module named 'httpx'",
        "httpx is not installed (needed for data generate).",
        "Run: pip install \"soup-cli\\[generate]\"",
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
    # peft's is_torchao_available() RAISES ImportError instead of returning
    # False when a too-old torchao is installed (peft/import_utils.py) — hit
    # nine frames inside get_peft_model on hosted notebooks (Colab, Kaggle)
    # that preinstall an old torchao. v0.73.x #389.
    (
        "Found an incompatible version of torchao",
        "Your installed peft refuses to load with the preinstalled torchao version.",
        (
            "Run: pip uninstall -y torchao (or upgrade it past the version peft "
            "demands). Soup does not need torchao unless you set "
            "training.quantization_aware. This usually comes from a hosted "
            "notebook (Colab, Kaggle) that preinstalls an old torchao."
        ),
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
    # Hugging Face gated/private model access
    (
        "gated repo",
        "This Hugging Face model requires authentication or license acceptance.",
        (
            "Run: huggingface-cli login or set HF_TOKEN. "
            "Also accept the model license on Hugging Face."
        ),
    ),
    (
        "gated-repo",
        "This Hugging Face model requires authentication or license acceptance.",
        (
        "Run: huggingface-cli login or set HF_TOKEN. "
        "Also accept the model license on Hugging Face."
        ),
    ),
    (
        "GatedRepoError",
        "This Hugging Face model requires authentication or license acceptance.",
        (
            "Run: huggingface-cli login or set HF_TOKEN. "
            "Also accept the model license on Hugging Face."
        ),
    ),

    # trust_remote_code errors
    (
        "trust_remote_code",
        "This model requires remote code execution approval.",
        (
            "Set trust_remote_code=True in your config or command "
            "if you trust the model source."
        ),
    ),
    (
        "requires you to execute the configuration file",
        "This model requires remote code execution approval.",
        (
            "Set trust_remote_code=True in your config or command "
            "if you trust the model source."
        ),
    ),
]


# 401/403 must be decided from the exception OBJECT (or explicit HTTP
# phrasing), never from bare digits — "train row 401:", a checkpoint path
# holding "4010", or a hash slice were all reported as auth failures and the
# real message was hidden (#1267). The old bare "401"/"403" ERROR_MAP entries
# are gone; this rule runs before the map so a genuine status also beats the
# generic "HTTPError" entry.
_AUTH_HINTS = {
    401: (
        "Authentication failed.",
        "Check your API key or token (HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY).",
    ),
    403: (
        "Access denied.",
        "Check your permissions. Some models require accepting a license on HuggingFace.",
    ),
}

# Text fallback for stacks that only put the status in the message. Every
# pattern requires HTTP context around the digits, so a row number, tensor
# size, path component, or hash slice cannot match.
_AUTH_STATUS_TEXT = (
    (re.compile(r"401\s+(?:Unauthorized|Client Error)"), 401),
    (re.compile(r"HTTP\s+(?:Error\s+)?[:=]?\s*401\b"), 401),
    (re.compile(r"Error code:\s*401\b"), 401),
    (re.compile(r"403\s+(?:Forbidden|Client Error)"), 403),
    (re.compile(r"HTTP\s+(?:Error\s+)?[:=]?\s*403\b"), 403),
    (re.compile(r"Error code:\s*403\b"), 403),
)

# These carry 401/403 but have their own, more specific ERROR_MAP entries that
# must keep winning, so the status rule declines them outright.
_STATUS_DECLINE_TYPES = ("GatedRepoError", "RepositoryNotFoundError")


def _exception_chain(exc: Exception):
    """Yield exc and the exceptions it wraps, without looping."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _auth_status(exc: Exception):
    """Return 401/403 when the exception is genuinely an auth failure.

    The status is read from the exception object first — `response.status_code`
    (requests / httpx / huggingface_hub), `.code` (urllib), or a direct
    `.status_code` — walking `__cause__`/`__context__` because peft and
    transformers wrap hub errors. Only when no object carries a status does the
    HTTP-phrasing text fallback apply.
    """
    chain = list(_exception_chain(exc))
    if any(type(e).__name__ in _STATUS_DECLINE_TYPES for e in chain):
        return None
    for e in chain:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(status, int) and status in _AUTH_HINTS:
            return status
        for attr in ("code", "status_code"):
            status = getattr(e, attr, None)
            if isinstance(status, int) and status in _AUTH_HINTS:
                return status
    for pattern, status in _AUTH_STATUS_TEXT:
        if pattern.search(str(exc)):
            return status
    return None


def format_friendly_error(exc: Exception, verbose: bool = False) -> None:
    """Display a friendly error message for the given exception.

    In normal mode: 2-3 lines with error + fix suggestion.
    In verbose mode: full traceback.
    """
    exc_str = str(exc)
    exc_type = type(exc).__name__

    matched = None
    status = _auth_status(exc)
    if status is not None:
        matched = _AUTH_HINTS[status]

    # Search for known error patterns
    if matched is None:
        for pattern, short_msg, fix in ERROR_MAP:
            if pattern in exc_str or pattern in exc_type:
                matched = (short_msg or exc_str, fix)
                break

    if matched is not None:
        error_msg, fix = matched
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
