"""soup tokenizer — BPE tokenizer training (v0.53.9 #15)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from soup_cli.utils.paths import is_under_cwd

console = Console()

app = typer.Typer(
    help="Tokenizer tools: train a BPE tokenizer from a JSONL corpus.",
    no_args_is_help=True,
)

_MAX_VOCAB_SIZE = 200_000
_MIN_VOCAB_SIZE = 256
_MAX_CORPUS_BYTES = 50 * 1024 * 1024  # 50 MB total cap
_MAX_LINE_BYTES = 8 * 1024  # 8 KB per-line cap


def _validate_under_cwd(path: str, *, label: str) -> str:
    if not isinstance(path, str) or not path:
        raise typer.BadParameter(f"{label} must be a non-empty path")
    if "\x00" in path:
        raise typer.BadParameter(f"{label} contains NUL byte")
    if not is_under_cwd(path):
        raise typer.BadParameter(
            f"{label} must stay under the current working directory: {path}"
        )
    # TOCTOU defence: lstat the RAW user-supplied path BEFORE realpath
    # resolution (matches v0.53.7 #106 policy). `os.path.realpath` would
    # follow the link and `lstat` on the resolved target would see a
    # regular file even when the user-supplied entry is a symlink.
    try:
        st = os.lstat(path)
    except OSError:
        # Missing-file path is handled by the caller's `is_file` check.
        return os.path.realpath(path)
    if stat.S_ISLNK(st.st_mode):
        raise typer.BadParameter(
            f"{label} target must not be a symlink"
        )
    return os.path.realpath(path)


def _extract_texts(input_path: Path) -> list[str]:
    """Read JSONL or plaintext lines into a list of training strings.

    Capped at 50 MB total + 8 KB per line to bound RAM on adversarial input.
    """
    texts: list[str] = []
    total_bytes = 0
    if input_path.suffix.lower() == ".jsonl":
        with open(input_path, encoding="utf-8") as fh:
            for raw in fh:
                if len(raw.encode("utf-8", errors="replace")) > _MAX_LINE_BYTES:
                    continue  # silently skip oversized line
                total_bytes += len(raw)
                if total_bytes > _MAX_CORPUS_BYTES:
                    break
                text = raw.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    for key in ("text", "content", "prompt"):
                        value = row.get(key)
                        if isinstance(value, str) and value:
                            texts.append(value)
                            break
                    else:
                        # ShareGPT-style messages list
                        msgs = row.get("messages")
                        if isinstance(msgs, list):
                            for msg in msgs:
                                if isinstance(msg, dict):
                                    content = msg.get("content")
                                    if isinstance(content, str) and content:
                                        texts.append(content)
                elif isinstance(row, str):
                    texts.append(row)
    else:
        with open(input_path, encoding="utf-8") as fh:
            for raw in fh:
                if len(raw.encode("utf-8", errors="replace")) > _MAX_LINE_BYTES:
                    continue
                total_bytes += len(raw)
                if total_bytes > _MAX_CORPUS_BYTES:
                    break
                text = raw.rstrip("\n\r")
                if text:
                    texts.append(text)
    return texts


@app.command(name="train")
def train(
    input_path: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to JSONL or .txt corpus (must stay under cwd).",
    ),
    vocab_size: int = typer.Option(
        32_000,
        "--vocab-size",
        "-v",
        help=f"Target vocabulary size ({_MIN_VOCAB_SIZE}-{_MAX_VOCAB_SIZE}).",
    ),
    output: str = typer.Option(
        "tokenizer_out",
        "--output",
        "-o",
        help="Output directory for tokenizer.json + vocab.json (under cwd).",
    ),
    min_frequency: int = typer.Option(
        2,
        "--min-frequency",
        help="Minimum pair frequency (>=1).",
    ),
    special_tokens: Optional[list[str]] = typer.Option(
        None,
        "--special-token",
        help="Add special token (repeatable). Defaults: <pad>, <s>, </s>, <unk>.",
    ),
) -> None:
    """Train a BPE tokenizer from a local corpus.

    Outputs `tokenizer.json` (Hugging Face format) and `vocab.json` to
    `--output`. Requires the `tokenizers` library (bundled with
    `transformers`).
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise typer.BadParameter("--vocab-size must be int")
    if not (_MIN_VOCAB_SIZE <= vocab_size <= _MAX_VOCAB_SIZE):
        raise typer.BadParameter(
            f"--vocab-size must be in [{_MIN_VOCAB_SIZE}, {_MAX_VOCAB_SIZE}]"
        )
    if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
        raise typer.BadParameter("--min-frequency must be int")
    if min_frequency < 1:
        raise typer.BadParameter("--min-frequency must be >= 1")

    input_real = _validate_under_cwd(input_path, label="--input")
    output_real = _validate_under_cwd(output, label="--output")

    resolved_input = Path(input_real)
    if not resolved_input.is_file():
        console.print(f"[red]Input file not found:[/] {input_path}")
        raise typer.Exit(1)
    output_path = Path(output_real)

    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    except ImportError as exc:
        console.print(
            "[red]The 'tokenizers' package is required.[/]\n"
            "Install with: [bold]pip install tokenizers[/]"
        )
        raise typer.Exit(1) from exc

    texts = _extract_texts(resolved_input)
    if not texts:
        console.print(
            f"[red]No usable training text found in[/] {resolved_input.name}"
        )
        raise typer.Exit(1)

    tokens = list(special_tokens or [])
    if not tokens:
        tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    # Sanitise: reject NUL / oversize entries; dedup preserves first-seen order.
    cleaned: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if not isinstance(tok, str) or not tok or "\x00" in tok or len(tok) > 128:
            raise typer.BadParameter(
                f"--special-token entry must be a non-empty NUL-free str <=128 chars: {tok!r}"
            )
        if tok in seen:
            continue
        seen.add(tok)
        cleaned.append(tok)

    console.print(
        Panel(
            f"Input:        [bold]{resolved_input.name}[/]\n"
            f"Texts:        [bold]{len(texts)}[/]\n"
            f"Vocab size:   [bold]{vocab_size}[/]\n"
            f"Output dir:   [bold]{output_path.name}[/]",
            title="BPE Tokenizer Training",
        )
    )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=cleaned,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    output_path.mkdir(parents=True, exist_ok=True)
    # Post-mkdir re-check: defends against a symlink planted between the
    # containment check and the write. Matches v0.43.0 Part D `copy_bundle_to`
    # TOCTOU policy.
    try:
        post_st = os.lstat(output_path)
    except OSError as exc:
        console.print(f"[red]Output directory not accessible:[/] {exc}")
        raise typer.Exit(1) from exc
    if stat.S_ISLNK(post_st.st_mode):
        console.print(
            "[red]Output directory is a symlink — refusing to write.[/]"
        )
        raise typer.Exit(1)
    if not is_under_cwd(str(output_path)):
        console.print("[red]Output directory escaped cwd after mkdir.[/]")
        raise typer.Exit(1)

    tokenizer_file = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    vocab_file = output_path / "vocab.json"
    vocab = tokenizer.get_vocab()
    vocab_file.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    console.print(
        f"[green]Trained tokenizer with[/] [bold]{len(vocab)}[/] tokens "
        f"-> [bold]{output_path.name}/{tokenizer_file.name}[/]"
    )
