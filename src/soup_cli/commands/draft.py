"""soup draft — train-your-own speculative-decoding draft (v0.71.33).

Three subcommands::

    soup draft distill --target <tuned> --draft-base <tiny> --data d.jsonl -o draft/
    soup draft measure --target <tuned> --draft draft/ --prompts p.jsonl
    soup draft list

``distill`` is thin orchestration over the existing ``task='distill'`` trainer:
it renders a validated distill config (student = the tiny draft base, teacher =
your tuned target), runs ``soup train`` as a subprocess, then merges the LoRA
adapter back into a DENSE checkpoint — a draft has to be loadable standalone as
``assistant_model=``. The draft is recorded in the local registry so
``soup serve --auto-spec`` picks it up.

``measure`` reports the teacher-forced acceptance rate (see ``utils/draft.py``)
plus plain-vs-assisted throughput. Exit codes mirror ``soup ship`` / ``soup
shrink``: 0 = ok, 2 = below ``--min-acceptance``, 1 = runtime error.

Draft and target MUST share a tokenizer (v1). Heavy imports are lazy.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli import __version__
from soup_cli.utils.adapter_fuse import fuse_adapter_into
from soup_cli.utils.draft import (
    AcceptanceReport,
    acceptance_rate,
    classify_acceptance,
    draft_report_to_dict,
    list_drafts,
    measure_acceptance,
    measure_throughput,
    register_draft,
    render_draft_panel,
    same_tokenizer,
)
from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

if TYPE_CHECKING:  # pragma: no cover — typing only, keeps the CLI import light
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

app = typer.Typer(help="Train + measure a speculative-decoding draft model.")
console = Console()

_MAX_INPUT_BYTES = 64 * 1024 * 1024
_MAX_PROMPT_ROWS = 10_000
_MAX_DATA_ROWS = 1_000_000
_DISTILL_TIMEOUT_SECONDS = 24 * 60 * 60
# batch 1 + grad checkpointing + a bounded max_length: teacher AND student are
# both resident during distillation, so keep the footprint consumer-GPU sized.
_DISTILL_BATCH_SIZE = 1
_DISTILL_MAX_LENGTH = 1024
_MAX_DISTILL_EPOCHS = 100
# How much of a failed subprocess's output to surface (mirrors shrink.py).
_SUBPROCESS_ERROR_TAIL_CHARS = 800

# Strip C0 / ESC / DEL before subprocess- or model-derived text hits the
# terminal (rich.markup.escape only neutralises [...] markup, not raw ESC
# bytes) — mirrors commands/shrink.py::_for_terminal.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None


def _for_terminal(text: str) -> str:
    return text.translate(_CONTROL_STRIP_TABLE)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _fail(message: str, code: int = 1) -> NoReturn:
    """Print a red error and exit. Raises internally so a forgotten ``raise``
    at a call site can never silently turn a failure into a no-op (mirrors
    ``commands/ship.py::_fail``)."""
    console.print(f"[red]{escape(message)}[/]")
    raise typer.Exit(code)


def _read_jsonl(path: str, label: str, max_rows: int) -> list[dict]:
    """Read a JSONL file: cwd-contained, O_NOFOLLOW, size- and row-capped."""
    enforce_under_cwd_and_no_symlink(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    rows: list[dict] = []
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_size > _MAX_INPUT_BYTES:
            raise ValueError(f"{label} exceeds {_MAX_INPUT_BYTES} bytes")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= max_rows:
                break
    return rows


def _vocab_size_of(model_id: str, trc: bool = False) -> int:
    """Vocab size from the model's config — no weight download."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trc)
    vocab = getattr(config, "vocab_size", None)
    if vocab is None:
        raise ValueError(f"{model_id} config has no vocab_size")
    return int(vocab)


def _resolve_trust(model_id: str) -> bool:
    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    requires = model_requires_trust_remote_code(model_id) or False
    return resolve_trust_remote_code(
        model_id, requested=False, console=console, requires_remote_code=requires
    )


def _load_pair_member(
    model_id: str, *, device: Optional[str] = None, trc: bool = False
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase", str]:
    """Load one half of the (target, draft) pair. Returns (model, tokenizer, device)."""
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    return load_model_and_tokenizer(
        model_id, device=device, trust_remote_code=trc, dtype="auto"
    )


# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------
def _build_distill_config_yaml(
    *,
    draft_base: str,
    target: str,
    data: str,
    out_dir: str,
    steps: int,
    data_rows: int,
) -> str:
    """Render the ``task: distill`` config: student = draft base, teacher = target.

    Every user-supplied string is embedded via ``json.dumps`` — a JSON string
    literal is always a valid YAML scalar, so a model id or path containing a
    newline cannot inject sibling YAML keys into the ``training:`` block.
    Mirrors ``commands/shrink.py::_build_heal_config_yaml``.
    """
    per_epoch = max(1, data_rows // _DISTILL_BATCH_SIZE)
    raw_epochs = math.ceil(steps / per_epoch)
    if raw_epochs > _MAX_DISTILL_EPOCHS:
        raise ValueError(
            f"--steps {steps} over {data_rows} rows expands to {raw_epochs} "
            f"epochs (> {_MAX_DISTILL_EPOCHS}); reduce --steps or grow --data."
        )
    epochs = max(1, raw_epochs)
    return (
        "base: {draft_base}\n"
        "task: distill\n"
        "output: {out}\n"
        "data:\n"
        "  train: {data}\n"
        "  format: auto\n"
        "  max_length: {max_length}\n"
        "training:\n"
        "  teacher_model: {target}\n"
        "  distill_divergence: forward_kl\n"
        "  distill_temperature: 2.0\n"
        "  epochs: {epochs}\n"
        "  batch_size: {batch}\n"
        "  gradient_checkpointing: true\n"
        "  quantization: none\n"
        "  lora:\n"
        "    r: 16\n"
        "    alpha: 32\n"
    ).format(
        draft_base=json.dumps(draft_base),
        out=json.dumps(out_dir),
        data=json.dumps(data),
        target=json.dumps(target),
        max_length=_DISTILL_MAX_LENGTH,
        epochs=epochs,
        batch=_DISTILL_BATCH_SIZE,
    )


def _run_distill(
    *,
    draft_base: str,
    target: str,
    data: str,
    out_dir: str,
    steps: int,
    data_rows: int,
    device: Optional[str] = None,
    trc: bool = False,
) -> None:
    """Distil the target into the draft base, then fuse the adapter to dense.

    Writes a validated distill config, runs ``soup train`` as a subprocess
    (argv list, no shell — mirrors ``commands/shrink.py::_run_heal``), and
    merges the resulting LoRA adapter into ``out_dir`` so the shipped draft is a
    single dense model loadable as ``assistant_model=``.
    """
    import subprocess
    import sys

    from soup_cli.config.loader import load_config_from_string

    yaml_text = _build_distill_config_yaml(
        draft_base=draft_base,
        target=target,
        data=data,
        out_dir=out_dir,
        steps=steps,
        data_rows=data_rows,
    )
    load_config_from_string(yaml_text)  # validate before spending a subprocess

    config_path = Path(out_dir).parent / "draft_distill_config.yaml"
    atomic_write_text(yaml_text, str(config_path), field="draft distill config")

    env = dict(os.environ)
    if device is not None and device.lower() == "cpu":
        # -1 is the canonical "hide every GPU"; "" trips an "Invalid device id"
        # assertion in some torch/accelerate paths.
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    argv = [
        sys.executable,
        "-m",
        "soup_cli.cli",
        "train",
        "--config",
        str(config_path),
        "--yes",
    ]
    try:
        result = subprocess.run(  # noqa: S603 — argv list, no shell.
            argv,
            capture_output=True,
            check=False,
            timeout=_DISTILL_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"draft distill exceeded {_DISTILL_TIMEOUT_SECONDS}s timeout"
        ) from exc
    if result.returncode != 0:
        combined = (result.stderr or b"").decode("utf-8", "replace") + (
            result.stdout or b""
        ).decode("utf-8", "replace")
        tail = _for_terminal(combined[-_SUBPROCESS_ERROR_TAIL_CHARS:])
        raise RuntimeError(f"draft distill failed (rc={result.returncode}): {tail}")

    # The trainer saves the student's LoRA adapter into out_dir. Merge it into a
    # dense checkpoint IN PLACE: transformers cannot use a PEFT adapter dir as
    # an assistant_model.
    fuse_adapter_into(base_dir=out_dir, adapter_dir=out_dir, trc=trc)


@app.command()
def distill(
    target: str = typer.Option(
        ..., "--target", help="The tuned model to speed up (the teacher)."
    ),
    draft_base: str = typer.Option(
        ...,
        "--draft-base",
        help="Tiny model to distil into (the student). Must share the target's "
        "tokenizer. A `soup shrink` output qualifies by construction.",
    ),
    data: str = typer.Option(
        ..., "--data", help="JSONL distillation set (chat / instruction rows)."
    ),
    output: str = typer.Option(
        "draft", "-o", "--output", help="Directory for the dense draft model."
    ),
    steps: int = typer.Option(
        500, "--steps", min=1, max=1_000_000, help="Approximate training steps."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="cpu | cuda (default: auto-detect)."
    ),
    no_register: bool = typer.Option(
        False,
        "--no-register",
        help="Do not record the draft in ~/.soup/drafts.json "
        "(it then won't be picked up by `soup serve --auto-spec`).",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Print the distill config and exit; write nothing."
    ),
) -> None:
    """Distil a target model into a tiny dense speculative-decoding draft."""
    try:
        enforce_under_cwd_and_no_symlink(output, "output dir")
        rows = _read_jsonl(data, "data path", _MAX_DATA_ROWS)
    except ValueError as exc:
        _fail(str(exc))
    if not rows:
        _fail(f"data file has no usable rows: {data}")

    target_trc = _resolve_trust(target)
    draft_trc = _resolve_trust(draft_base)

    # Same-tokenizer gate. Speculative decoding proposes DRAFT token ids into
    # the TARGET's vocabulary — a mismatch silently produces garbage rather
    # than failing, so refuse up front.
    try:
        target_vocab = _vocab_size_of(target, target_trc)
        draft_vocab = _vocab_size_of(draft_base, draft_trc)
    except Exception as exc:  # noqa: BLE001 — surface as a friendly CLI error
        _fail(f"could not read model config: {exc}")

    if target_vocab != draft_vocab:
        _fail(
            f"Draft and target must share a tokenizer, but vocab sizes differ "
            f"(target={target_vocab}, draft={draft_vocab}). Speculative decoding "
            f"proposes draft token ids into the target's vocabulary, so a "
            f"mismatched pair produces garbage rather than a speedup.\n"
            f"Pick a draft base from the target's own family (a `soup shrink` "
            f"output always qualifies), or run cross-tokenizer distillation "
            f"manually with task=distill + training.uld_strategy."
        )

    try:
        yaml_text = _build_distill_config_yaml(
            draft_base=draft_base,
            target=target,
            data=data,
            out_dir=output,
            steps=steps,
            data_rows=len(rows),
        )
    except ValueError as exc:
        _fail(str(exc))

    if plan_only:
        console.print(
            f"[bold]Plan[/] — distil [cyan]{escape(target)}[/] into "
            f"[cyan]{escape(draft_base)}[/] over {len(rows)} rows "
            f"(shared vocab {target_vocab})\n"
        )
        console.print(escape(yaml_text))
        console.print("[dim]--plan-only: nothing written.[/]")
        return

    console.print(
        f"[bold]Distilling[/] {escape(target)} -> {escape(draft_base)} "
        f"({len(rows)} rows, ~{steps} steps)"
    )
    try:
        _run_distill(
            draft_base=draft_base,
            target=target,
            data=data,
            out_dir=output,
            steps=steps,
            data_rows=len(rows),
            device=device,
            trc=draft_trc,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        _fail(f"distill failed: {exc}")

    if not no_register:
        register_draft(target, output)
        console.print(
            f"[green]Registered[/] as the local draft for {escape(target)} — "
            "`soup serve --auto-spec` will now pick it up."
        )

    console.print(
        f"[green]Draft written to[/] {escape(output)}\n"
        f"[dim]Next: soup draft measure --target {escape(target)} "
        f"--draft {escape(output)} --prompts <p.jsonl>[/]"
    )


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------
def _prompt_texts(rows: list[dict]) -> list[str]:
    """Best-effort prompt text (prompt / text / instruction / messages)."""
    prompts: list[str] = []
    for row in rows:
        for key in ("prompt", "text", "instruction", "content"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                prompts.append(value)
                break
        else:
            messages = row.get("messages")
            if isinstance(messages, list):
                user = [
                    msg.get("content", "")
                    for msg in messages
                    if isinstance(msg, dict) and msg.get("role") == "user"
                ]
                if user and isinstance(user[0], str) and user[0].strip():
                    prompts.append(user[0])
    return prompts


@app.command()
def measure(
    target: str = typer.Option(..., "--target", help="The model being served."),
    draft: str = typer.Option(
        ..., "--draft", help="The draft model (a `soup draft distill` output)."
    ),
    prompts: str = typer.Option(
        ..., "--prompts", help="JSONL prompts representative of production traffic."
    ),
    max_new_tokens: int = typer.Option(
        64, "--max-new-tokens", min=1, max=4096, help="Tokens to generate per prompt."
    ),
    num_assistant_tokens: int = typer.Option(
        5,
        "--num-assistant-tokens",
        min=1,
        max=64,
        help="Tokens the draft proposes per step (the assisted-generation knob).",
    ),
    device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda."),
    min_acceptance: Optional[float] = typer.Option(
        None,
        "--min-acceptance",
        min=0.0,
        max=1.0,
        help="Exit 2 if the acceptance rate falls below this (for CI gating).",
    ),
    output: Optional[str] = typer.Option(
        None, "-o", "--output", help="Write the report as JSON."
    ),
) -> None:
    """Report a draft's acceptance rate + throughput against its target."""
    try:
        rows = _read_jsonl(prompts, "prompts path", _MAX_PROMPT_ROWS)
    except ValueError as exc:
        _fail(str(exc))
    prompt_texts = _prompt_texts(rows)
    if not prompt_texts:
        _fail(f"prompts file yielded no usable prompt text: {prompts}")
    if output is not None:
        try:
            enforce_under_cwd_and_no_symlink(output, "output path")
        except ValueError as exc:
            _fail(str(exc))

    target_trc = _resolve_trust(target)
    draft_trc = _resolve_trust(draft)

    console.print(f"[dim]Loading target: {escape(target)}[/]")
    try:
        target_model, target_tok, resolved_device = _load_pair_member(
            target, device=device, trc=target_trc
        )
        console.print(f"[dim]Loading draft: {escape(draft)}[/]")
        draft_model, draft_tok, _ = _load_pair_member(
            draft, device=resolved_device, trc=draft_trc
        )
    except Exception as exc:  # noqa: BLE001 — friendly CLI error
        _fail(f"could not load the model pair: {exc}")

    if not same_tokenizer(target_tok, draft_tok):
        _fail(
            "Draft and target do not share a tokenizer. Speculative decoding "
            "proposes draft token ids into the target's vocabulary, so this "
            "pair cannot be used together — the draft's proposals would be "
            "meaningless. Distil a draft from this target with "
            "`soup draft distill`."
        )

    accepted, total = measure_acceptance(
        target_model,
        draft_model,
        target_tok,
        prompt_texts,
        max_new_tokens=max_new_tokens,
    )
    if total == 0:
        _fail(
            "the target generated no tokens for any prompt — nothing to measure "
            "(check the prompts file and --max-new-tokens)"
        )

    rate = acceptance_rate(accepted, total)
    verdict = classify_acceptance(rate)

    tok_s_plain = measure_throughput(
        target_model, target_tok, prompt_texts, max_new_tokens=max_new_tokens
    )
    tok_s_assisted = measure_throughput(
        target_model,
        target_tok,
        prompt_texts,
        assistant_model=draft_model,
        num_assistant_tokens=num_assistant_tokens,
        max_new_tokens=max_new_tokens,
    )
    # A measured 0.0 tok/s means "we could not time it", not "zero throughput";
    # normalise explicitly rather than leaning on 0.0 being falsy.
    plain = None if tok_s_plain <= 0 else tok_s_plain
    assisted = None if tok_s_assisted <= 0 else tok_s_assisted
    speedup = assisted / plain if (plain and assisted) else None

    report = AcceptanceReport(
        target=target,
        draft=draft,
        n_prompts=len(prompt_texts),
        n_generated_tokens=total,
        acceptance_rate=rate,
        verdict=verdict,
        tok_s_plain=plain,
        tok_s_assisted=assisted,
        speedup=speedup,
        num_assistant_tokens=num_assistant_tokens,
        soup_version=__version__,
    )
    console.print(render_draft_panel(report))

    if output is not None:
        atomic_write_text(
            json.dumps(draft_report_to_dict(report), indent=2),
            output,
            field="report path",
        )
        console.print(f"[dim]Report written to {escape(output)}[/]")

    if min_acceptance is not None and rate < min_acceptance:
        _fail(
            f"Acceptance {rate:.1%} is below the required {min_acceptance:.1%}.",
            code=2,
        )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
@app.command("list")
def list_registered() -> None:
    """List locally-trained drafts that `soup serve --auto-spec` can use."""
    entries = list_drafts()
    if not entries:
        console.print(
            "[yellow]No drafts registered.[/] Train one with "
            "`soup draft distill --target <model> --draft-base <tiny> "
            "--data <d.jsonl>`."
        )
        return

    table = Table(title="Local speculative-decoding drafts")
    table.add_column("Target", style="cyan")
    table.add_column("Draft")
    table.add_column("Acceptance", justify="right")
    table.add_column("Created", style="dim")
    for entry in entries:
        rate = entry.get("acceptance_rate")
        table.add_row(
            escape(str(entry.get("target", "?"))),
            escape(str(entry.get("draft", "?"))),
            "-" if rate is None else f"{float(rate) * 100:.1f}%",
            escape(str(entry.get("created", "?"))),
        )
    console.print(table)
