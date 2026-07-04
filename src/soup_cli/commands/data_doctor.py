"""``soup data doctor`` / ``soup data lint`` — Fine-tune Doctor (v0.71.27).

Thin Typer/Rich CLI layer over the pure engines in
``utils/data_doctor.py`` (chat-template compat report + loss-mask X-ray)
and ``utils/data_lint.py`` (preference-data linter). Mirrors
``commands/diagnose.py``'s render/exit-code conventions: exit 0 = OK/MINOR,
exit 2 = MAJOR (or a routing usage error — wrong format for this command),
exit 1 = runtime error (bad path, bad tokenizer, bad --output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soup_cli.data.formats import detect_format
from soup_cli.data.loader import load_raw_data
from soup_cli.utils import data_doctor as engine
from soup_cli.utils import data_lint as lint_engine
from soup_cli.utils.paths import atomic_write_text
from soup_cli.utils.trust_remote import model_requires_trust_remote_code, resolve_trust_remote_code

console = Console()

_PREFERENCE_FORMATS = ("dpo", "kto")

# C0 control bytes (keep tab/newline/CR) + DEL, stripped before ANY
# dataset-derived string reaches the terminal. rich.markup.escape() only
# neutralises Rich's own [...] tag syntax — it does not strip raw escape
# sequences, and dataset content (an unknown 'role' field, --show-mask's
# decoded token text) is untrusted: a crafted training row can carry a
# literal ESC byte that survives escape() and hits the terminal raw
# (title-bar spoofing, OSC 8 link-text spoofing, or obscuring a MAJOR
# verdict via cursor tricks). --output JSON is unaffected: json.dumps
# already \\u00XX-escapes control characters per the JSON spec.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None


def _for_terminal(text: str) -> str:
    return text.translate(_CONTROL_STRIP_TABLE)


def _verdict_style(verdict: str) -> str:
    return {"OK": "green", "MINOR": "yellow", "MAJOR": "red"}.get(verdict, "white")


_AnyCheck = Union[engine.DoctorCheck, lint_engine.LintCheck]


def _render_checks_table(title: str, checks: Sequence[_AnyCheck]) -> None:
    table = Table(title=title)
    table.add_column("Check", style="bold")
    table.add_column("Verdict")
    table.add_column("Message", overflow="fold")
    table.add_column("Evidence", overflow="fold")
    for check in checks:
        table.add_row(
            escape(check.name),
            f"[{_verdict_style(check.verdict)}]{check.verdict}[/]",
            escape(_for_terminal(check.message)),
            escape(_for_terminal(check.evidence)),
        )
    console.print(table)


def _render_doctor_report(report: engine.DoctorReport) -> None:
    _render_checks_table("soup data doctor", report.checks)
    console.print(
        Panel.fit(
            f"[bold {_verdict_style(report.overall)}]{report.overall}[/] — "
            f"{report.rows_scanned}/{report.total_rows} rows scanned",
            title="overall",
        )
    )


def _render_lint_report(report: lint_engine.LintReport) -> None:
    _render_checks_table(f"soup data lint ({escape(report.fmt)})", report.checks)
    console.print(
        Panel.fit(
            f"[bold {_verdict_style(report.overall)}]{report.overall}[/] — "
            f"{report.rows_scanned}/{report.total_rows} rows scanned",
            title="overall",
        )
    )


def _render_mask_previews(previews: Sequence[engine.MaskPreviewRow]) -> None:
    for preview in previews:
        text = Text()
        for token in preview.tokens:
            style = "black on green" if token.trained else "dim"
            # Byte-level BPE tokenizers round-trip arbitrary bytes exactly,
            # so a crafted training row can decode straight back to a raw
            # control sequence — strip before it ever reaches Text.append.
            text.append(_for_terminal(token.text), style=style)
        console.print(Panel(text, title=f"row {preview.row_index} ({escape(preview.strategy)})"))


def _write_report_json(payload: dict, output: str) -> None:
    atomic_write_text(json.dumps(payload, indent=2), output, field="output")


def _resolve_format(data: list, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    try:
        return detect_format(data)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


def _load_tokenizer(model: str, trust_remote_code: bool):
    """Probe + warn (mirrors every other model-loading command: chat.py,
    diff.py, export.py, merge.py, serve.py, infer.py, generate.py, and
    every trainer wrapper) before resolving, then delegate to the pure
    engine.
    """
    try:
        requires = model_requires_trust_remote_code(model) or False
        resolved_trust = resolve_trust_remote_code(model, trust_remote_code, console, requires)
        return engine.resolve_tokenizer(model, trust_remote_code=resolved_trust)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


def doctor(
    path: str = typer.Argument(..., help="Path to dataset file (jsonl/json/csv/parquet)."),
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Tokenizer id or local path to check chat-template compatibility against.",
    ),
    fmt: str = typer.Option(
        "auto", "--format", "-f", help="Dataset format (auto-detects by default).",
    ),
    max_length: int = typer.Option(
        2048, "--max-length", help="Training max_length — used for the truncation-risk check.",
    ),
    sample: int = typer.Option(200, "--sample", help="Number of rows to sample for the checks."),
    show_mask: Optional[int] = typer.Option(
        None, "--show-mask",
        help="Render N sample rows with per-token trained/masked colouring "
        "(the real collator path).",
    ),
    train_on_responses_only: bool = typer.Option(
        True, "--train-on-responses-only/--no-train-on-responses-only",
        help="Mask non-assistant tokens (mirrors data.train_on_responses_only).",
    ),
    train_on_messages_with_train_field: bool = typer.Option(
        False, "--train-on-messages-with-train-field",
        help="Per-message train:bool field masking (mirrors the same soup.yaml flag).",
    ),
    train_on_eot: bool = typer.Option(
        False, "--train-on-eot",
        help="Extend the trained span through the trailing EOS/EOT token.",
    ),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code", help="Allow loading a tokenizer that ships custom code.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the report JSON to this path.",
    ),
) -> None:
    """Chat-template compatibility report + loss-mask X-ray.

    Catches the top *silent* fine-tune failures before a single training
    step: missing/incompatible chat_template, missing EOS in the trained
    span (the #1 "model never stops generating" bug), duplicated BOS,
    unsupported system role, unknown message roles, and truncation risk.
    Pass --show-mask N to render N sample rows with per-token
    trained/masked colouring through the real answer-only / per-message /
    RAFT span-mask collator path.
    """
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(code=1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(code=1)

    resolved_fmt = _resolve_format(data, fmt)

    if resolved_fmt in _PREFERENCE_FORMATS:
        console.print(
            f"[red]Error:[/] format={escape(repr(resolved_fmt))} is preference data — "
            "use [bold]soup data lint[/] instead of [bold]soup data doctor[/]."
        )
        raise typer.Exit(code=2)

    if resolved_fmt == "raft" and show_mask is None:
        console.print(
            "[red]Error:[/] RAFT format has no chat-template compat report (no chat "
            "messages) — pass [bold]--show-mask N[/] to preview the RAFT span-mask."
        )
        raise typer.Exit(code=2)

    tokenizer = _load_tokenizer(model, trust_remote_code)

    exit_code = 0
    if resolved_fmt != "raft":
        try:
            report = engine.run_doctor(
                data, tokenizer, fmt=resolved_fmt, max_length=max_length,
                sample_size=sample, include_eot=train_on_eot,
                train_on_responses_only=train_on_responses_only,
                train_on_messages_with_train_field=train_on_messages_with_train_field,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/] {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        _render_doctor_report(report)
        if output:
            try:
                _write_report_json(report.to_dict(), output)
                console.print(f"[green]Wrote[/] {escape(output)}")
            except (OSError, ValueError) as exc:
                console.print(f"[red]Error:[/] cannot write --output: {escape(str(exc))}")
                raise typer.Exit(code=1) from exc
        if report.overall == "MAJOR":
            exit_code = 2

    if show_mask is not None:
        try:
            previews = engine.render_mask_preview(
                data, tokenizer, fmt=resolved_fmt, n=show_mask, max_length=max_length,
                train_on_responses_only=train_on_responses_only,
                train_on_messages_with_train_field=train_on_messages_with_train_field,
                include_eot=train_on_eot,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/] {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        if previews:
            _render_mask_previews(previews)
        else:
            console.print("[yellow]No rows could be rendered for --show-mask.[/]")

    if exit_code:
        raise typer.Exit(code=exit_code)


def lint(
    path: str = typer.Argument(..., help="Path to preference dataset file (dpo/kto)."),
    fmt: str = typer.Option("auto", "--format", "-f", help="Dataset format: auto, dpo, kto."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Optional tokenizer for exact token-length bias (default: word count).",
    ),
    sample: int = typer.Option(2000, "--sample", help="Number of rows to sample for the checks."),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code", help="Allow loading a tokenizer that ships custom code.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the report JSON to this path.",
    ),
) -> None:
    """Preference-data linter (dpo/orpo/simpo/ipo/bco/kto).

    Flags the top silent preference-data bugs: length bias (chosen
    systematically longer than rejected — reported as a Cohen's d effect
    size), KTO label imbalance, near-duplicate pairs, chosen == rejected
    rows, and the prompt echoed verbatim inside the completion.
    """
    file_path = Path(path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/]")
        raise typer.Exit(code=1)

    data = load_raw_data(file_path)
    if not data:
        console.print("[red]Dataset is empty.[/]")
        raise typer.Exit(code=1)

    resolved_fmt = _resolve_format(data, fmt)

    if resolved_fmt not in _PREFERENCE_FORMATS:
        console.print(
            f"[red]Error:[/] format={escape(repr(resolved_fmt))} is not preference data — "
            "soup data lint only supports dpo/orpo/simpo/ipo/bco (chosen/rejected) and kto. "
            "Use [bold]soup data doctor[/] instead for chat/SFT data."
        )
        raise typer.Exit(code=2)

    # Only dpo's length_bias check consults length_fn — skip the tokenizer
    # load entirely for kto so `--model` isn't a wasted download/load.
    length_fn = None
    if model and resolved_fmt == "dpo":
        tokenizer = _load_tokenizer(model, trust_remote_code)

        def length_fn(text: str) -> float:
            try:
                try:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                except TypeError:
                    ids = tokenizer.encode(text)
                return float(len(ids))
            except Exception:  # noqa: BLE001 — a row the tokenizer can't encode
                # still needs a length estimate; degrade to the word-count
                # proxy (matches the --model-less default) instead of
                # crashing the whole lint pass on one bad row.
                return float(len(text.split()))

    try:
        report = lint_engine.run_lint(data, resolved_fmt, sample_size=sample, length_fn=length_fn)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    _render_lint_report(report)

    if output:
        try:
            _write_report_json(report.to_dict(), output)
            console.print(f"[green]Wrote[/] {escape(output)}")
        except (OSError, ValueError) as exc:
            console.print(f"[red]Error:[/] cannot write --output: {escape(str(exc))}")
            raise typer.Exit(code=1) from exc

    if report.overall == "MAJOR":
        raise typer.Exit(code=2)


__all__ = ["doctor", "lint"]
