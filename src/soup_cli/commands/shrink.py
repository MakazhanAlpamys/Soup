"""soup shrink — depth-prune + distill-heal (v0.71.29, arXiv:2403.17887).

Top-level CLI command (NOT a sub-group). Ranks a model's decoder layers by the
angular distance of the residual stream across a contiguous block over a
calibration set, drops the least-important block, optionally distill-heals, and
emits a single dense smaller model with a before/after perplexity verdict::

    soup shrink --model <id|path> --drop-ratio 0.25 --calib calib.jsonl -o shrunk
    soup shrink --model <id|path> --drop-layers 6 --calib calib.jsonl \
        --heal heal.jsonl --heal-steps 200 -o shrunk

Exit codes: 0 = SHIP, 2 = DON'T SHIP, 1 = runtime error (mirrors soup ship /
soup diagnose). Heavy imports (torch/transformers) are lazy inside functions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink
from soup_cli.utils.shrink import (
    DECISION_SHIP,
    compute_layer_importance,
    decide_shrink,
    prune_model_layers,
    render_shrink_panel,
    resolve_drop_count,
    select_drop_block,
    shrink_verdict_to_dict,
)

console = Console()

# 64 MiB cap on the calibration JSONL (symlink-pointed-to-/dev/zero DoS guard).
_MAX_CALIB_BYTES = 64 * 1024 * 1024
_MAX_CALIB_ROWS = 10_000
_MAX_HEAL_STEPS = 1_000_000
_PPL_MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Path + data helpers
# ---------------------------------------------------------------------------
def _under_cwd(path: str, label: str) -> str:
    """Validate ``path`` stays under cwd and is not a symlink; return it."""
    enforce_under_cwd_and_no_symlink(path, label)
    return path


def _extract_text(row: object) -> str:
    """Best-effort prompt text from a calib row (text / prompt / messages)."""
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        for key in ("text", "prompt", "content", "instruction"):
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val
        messages = row.get("messages")
        if isinstance(messages, list):
            parts = [
                m.get("content", "")
                for m in messages
                if isinstance(m, dict) and isinstance(m.get("content"), str)
            ]
            joined = "\n".join(p for p in parts if p.strip())
            if joined.strip():
                return joined
    return ""


def _load_calib(path: str) -> list[str]:
    """Load calibration prompts from a JSONL file (cwd-contained, size-capped).

    Opens with ``O_NOFOLLOW`` and fstats the open fd (TOCTOU defence, mirrors
    ``commands/diagnose.py::_load_evidence``). Each non-empty line is a JSON
    object; the prompt text is extracted via :func:`_extract_text`.
    """
    _under_cwd(path, "calib path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise typer.BadParameter(f"calib path unreadable: {exc}") from exc
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_size > _MAX_CALIB_BYTES:
            raise typer.BadParameter(
                f"calib file exceeds {_MAX_CALIB_BYTES} bytes"
            )
        prompts: list[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a raw-text line (not JSON) as a plain prompt.
                row = line
            text = _extract_text(row)
            if text.strip():
                prompts.append(text)
            if len(prompts) >= _MAX_CALIB_ROWS:
                break
    if not prompts:
        raise typer.BadParameter("calib file yielded no usable prompt text")
    return prompts


# ---------------------------------------------------------------------------
# Model helpers (torch-lazy)
# ---------------------------------------------------------------------------
def _count_params(model: object) -> int:
    return sum(p.numel() for p in model.parameters())  # type: ignore[attr-defined]


def _perplexity(model: object, tokenizer: object, prompts: list[str], device: str) -> float:
    """Mean unconditional-LM perplexity of ``model`` over ``prompts``.

    ``exp(mean per-example cross-entropy)`` with ``labels = input_ids`` (the
    whole sequence is the target). Returns ``inf`` when no example is usable.
    """
    import math

    import torch

    losses: list[float] = []
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        for text in prompts:
            enc = tokenizer(  # type: ignore[operator]
                text,
                return_tensors="pt",
                truncation=True,
                max_length=_PPL_MAX_LENGTH,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)  # type: ignore[operator]
            loss = float(out.loss.item())
            if loss == loss:  # not NaN
                losses.append(loss)
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def _load_for_shrink(model_id: str, device: Optional[str], trust_remote_code: bool):
    """Load a model + tokenizer for shrinking (trust_remote_code probe + warn)."""
    from soup_cli.utils.live_eval import load_model_and_tokenizer
    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    requires = model_requires_trust_remote_code(model_id) or False
    trc = resolve_trust_remote_code(
        model_id, requested=trust_remote_code, console=console, requires_remote_code=requires
    )
    return load_model_and_tokenizer(model_id, device=device, trust_remote_code=trc)


def _render_importance_table(importances, chosen) -> None:
    table = Table(title="soup shrink — layer importance (lower = safer to drop)")
    table.add_column("Rank", justify="right")
    table.add_column("Block (start..end)")
    table.add_column("Angular distance", justify="right")
    table.add_column("Chosen")
    for rank, imp in enumerate(importances, start=1):
        end = imp.start + imp.block_size
        is_chosen = imp.start == chosen.start
        table.add_row(
            str(rank),
            f"{imp.start}..{end - 1}",
            f"{imp.angular_distance:.4f}",
            "<-- drop" if is_chosen else "",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------
def shrink(
    model: str = typer.Option(..., "--model", help="Model id or local path to shrink."),
    drop_ratio: Optional[float] = typer.Option(
        None, "--drop-ratio", help="Fraction of layers to drop (0-1); e.g. 0.25."
    ),
    drop_layers: Optional[int] = typer.Option(
        None, "--drop-layers", help="Explicit number of contiguous layers to drop."
    ),
    calib: str = typer.Option(
        ..., "--calib", help="Calibration JSONL (prompts) — must stay under cwd."
    ),
    tolerance: float = typer.Option(
        0.10, "--tolerance", help="Perplexity-regression tolerance for the verdict."
    ),
    output_dir: str = typer.Option(
        "./shrunk", "--output-dir", "-o", help="Directory for the shrunk model + report."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device for the importance/ppl passes (cuda / cpu)."
    ),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code", help="Allow custom modeling code (auto_map)."
    ),
    attach_to_registry: Optional[str] = typer.Option(
        None, "--attach-to-registry", help="Attach the shrink report to a registry entry id."
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only", help="Print the importance table + chosen block and exit."
    ),
) -> None:
    """Depth-prune a model (least-important contiguous block) + verdict."""
    try:
        _shrink_impl(
            model=model,
            drop_ratio=drop_ratio,
            drop_layers=drop_layers,
            calib=calib,
            tolerance=tolerance,
            output_dir=output_dir,
            device=device,
            trust_remote_code=trust_remote_code,
            attach_to_registry=attach_to_registry,
            plan_only=plan_only,
        )
    except typer.Exit:
        raise
    except (typer.BadParameter, ValueError) as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc


def _shrink_impl(
    *,
    model: str,
    drop_ratio: Optional[float],
    drop_layers: Optional[int],
    calib: str,
    tolerance: float,
    output_dir: str,
    device: Optional[str],
    trust_remote_code: bool,
    attach_to_registry: Optional[str],
    plan_only: bool,
) -> None:
    if not (0.0 <= tolerance <= 5.0):
        raise typer.BadParameter("--tolerance must be in [0.0, 5.0]")
    # Fail fast on the flag combination BEFORE loading a multi-GB model.
    if (drop_ratio is None) == (drop_layers is None):
        raise typer.BadParameter("set exactly one of --drop-ratio / --drop-layers")
    prompts = _load_calib(calib)

    console.print(f"[dim]Loading {escape(model)} ...[/]")
    mdl, tokenizer, dev = _load_for_shrink(model, device, trust_remote_code)
    # Reject an unsupported architecture up front (before the importance scan).
    from soup_cli.utils.shrink import shrink_arch_of

    shrink_arch_of(mdl)
    n_layers = int(mdl.config.num_hidden_layers)
    count = resolve_drop_count(n_layers, drop_ratio=drop_ratio, drop_layers=drop_layers)

    console.print(f"[dim]Scoring importance over {len(prompts)} calib prompts ...[/]")
    importances = compute_layer_importance(
        mdl, tokenizer, prompts, block_size=count, device=dev
    )
    chosen = select_drop_block(importances)
    _render_importance_table(importances, chosen)

    if plan_only:
        end = chosen.start + chosen.block_size
        console.print(
            Panel.fit(
                f"Would drop layers [bold]{chosen.start}..{end - 1}[/] "
                f"({count} of {n_layers}); angular distance "
                f"{chosen.angular_distance:.4f}",
                title="plan only",
            )
        )
        raise typer.Exit(0)

    params_before = _count_params(mdl)
    ppl_original = _perplexity(mdl, tokenizer, prompts, dev)

    # Prune in memory, save, then RELOAD (slicing leaves layer_idx stale;
    # from_pretrained rebuilds them contiguously — measure on the shipped dir).
    prune_model_layers(mdl, chosen.start, chosen.block_size)
    out_root = Path(output_dir)
    model_out = out_root / "model"
    model_out.mkdir(parents=True, exist_ok=True)
    mdl.save_pretrained(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    del mdl

    reloaded, tok2, dev2 = _load_for_shrink(str(model_out), device, trust_remote_code)
    layers_after = int(reloaded.config.num_hidden_layers)
    params_after = _count_params(reloaded)
    ppl_final = _perplexity(reloaded, tok2, prompts, dev2)
    params_saved_pct = (
        100.0 * (params_before - params_after) / params_before if params_before else 0.0
    )

    verdict = decide_shrink(
        ppl_original,
        ppl_final,
        tolerance=tolerance,
        layers_before=n_layers,
        layers_after=layers_after,
        params_saved_pct=params_saved_pct,
        healed=False,
    )
    console.print(render_shrink_panel(verdict))

    report_path = out_root / "shrink_report.json"
    report = shrink_verdict_to_dict(verdict)
    report["model"] = model
    report["dropped_block"] = [chosen.start, chosen.start + chosen.block_size - 1]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/] {escape(str(report_path))}")

    if attach_to_registry:
        _attach_to_registry(attach_to_registry, str(report_path))

    raise typer.Exit(0 if verdict.decision == DECISION_SHIP else 2)


def _attach_to_registry(registry_id: str, report_path: str) -> None:
    """Attach the shrink report JSON as a registry artifact (best-effort)."""
    try:
        from soup_cli.registry.attach import attach_artifact
    except Exception as exc:  # noqa: BLE001 — registry is optional
        console.print(
            f"[yellow]Warning:[/] could not import registry attach helper: {escape(str(exc))}"
        )
        return
    try:
        attach_artifact(registry_id, "shrink_report", report_path)
        console.print(
            f"[green]Attached[/] shrink_report to registry entry [bold]{escape(registry_id)}[/]"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Warning:[/] could not attach to registry: {escape(str(exc))}")
