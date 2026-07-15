"""soup card — generate a HuggingFace model card from a registry entry (v0.71.35).

Turns a Local Model Registry entry into a ready-to-publish ``MODELCARD.md``:
base model, training config, eval scorecard, provenance (config/data hashes),
lineage (ancestors), and a table of every registered artifact (adapter / merged
model / GGUF / diagnose report / eval results). Every public push becomes a
documented, provenance-carrying release.

Design notes:
  * ``build_model_card`` is a pure function over plain dicts (no store / no
    torch) so it is trivially unit-testable and reusable by ``soup push --card``.
  * User-controlled text is escaped for its output target: registry ``notes`` is
    free-form -> ``html.escape`` (blocks script/``javascript:`` injection on the
    HF README viewer); markdown table cells -> ``_safe_md_cell`` (reused from
    ``push.py``); YAML frontmatter scalars -> ``_yaml_dq`` (control-char strip +
    double-quote escaping so a hostile ``base_model`` cannot inject sibling keys).
  * Output is written through ``atomic_write_text`` (cwd-contained,
    symlink-rejected), consistent with ``write_bom`` / ``write_attestation``.
"""

from __future__ import annotations

import html
import json
from typing import Any, Mapping, Sequence

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from soup_cli.commands.push import (
    _render_eval_scorecard,
    _render_training_section,
    _safe_md_cell,
)
from soup_cli.registry.store import AmbiguousRefError, RegistryStore
from soup_cli.utils.paths import atomic_write_text

console = Console()

# Bounds on registry-sourced content (v0.71.35 security review). The registry
# places no length cap on `notes` / `base_model` and no row cap on artifacts, so
# a pathological entry could otherwise render an unbounded card.
_MAX_NOTES_CHARS = 8_000
_MAX_ROWS = 200


class CardError(ValueError):
    """Raised when a registry entry cannot be resolved for a card."""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n\n[truncated]"


# Export kinds that prove the artifact is a standalone (non-adapter) model.
_DENSE_KINDS = frozenset({"merged", "gguf", "onnx", "awq", "gptq", "tensorrt", "edited_model"})


def _is_adapter(artifacts: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    """Decide whether the entry describes a LoRA adapter or a standalone model.

    A registered artifact is definitive; otherwise fall back to the training
    config. This matters because the card is a provenance document — claiming
    "Full model" for a LoRA (and emitting ``library_name: transformers`` instead
    of ``peft``) is a false statement + a broken HF Hub card. Found by the
    v0.71.35 Step-6 smoke: a real LoRA run with no artifacts attached rendered
    as "Full model".
    """
    kinds = {a.get("kind") for a in artifacts or []}
    if "adapter" in kinds:
        return True
    if kinds & _DENSE_KINDS:
        return False
    training = config.get("training") or {}
    if not isinstance(training, dict):
        return False
    # Spectrum / LISA are full-FT: LoRA is off even though the dumped config
    # still carries a default `lora` block.
    if training.get("unfrozen_parameters") or training.get("lisa_enabled"):
        return False
    lora = training.get("lora") or {}
    if not isinstance(lora, dict):
        return False
    try:
        return int(lora.get("r") or 0) > 0
    except (TypeError, ValueError):
        return False


def _yaml_dq(value: object) -> str:
    """Serialise ``value`` as a YAML double-quoted scalar, stripping control
    chars and escaping ``\\`` / ``"`` so it cannot break out of the frontmatter.
    """
    text = "".join(ch for ch in str(value) if ord(ch) >= 0x20)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def build_model_card(
    entry: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    eval_results: Sequence[Mapping[str, Any]],
    ancestors: Sequence[Mapping[str, Any]],
) -> str:
    """Render a full HF model card (markdown) from a registry entry + joins.

    ``entry`` is a hydrated registry row; ``artifacts`` / ``eval_results`` /
    ``ancestors`` are the corresponding ``RegistryStore`` joins (may be empty).
    """
    name = str(entry.get("name") or "model")
    base = str(entry.get("base_model") or "")
    task = str(entry.get("task") or "sft")
    notes = entry.get("notes") or ""
    tags = entry.get("tags") or []
    created = str(entry.get("created_at") or "")
    config_hash = str(entry.get("config_hash") or "")
    data_hash = str(entry.get("data_hash") or "")
    run_id = str(entry.get("run_id") or "")

    try:
        config = json.loads(entry.get("config_json") or "{}")
        if not isinstance(config, dict):
            config = {}
    except (TypeError, ValueError):
        config = {}
    training_cfg = dict(config)
    training_cfg.setdefault("base", base)
    training_cfg.setdefault("task", task)

    # Row caps: bound the card regardless of how many rows the registry holds.
    artifacts = list(artifacts or [])[:_MAX_ROWS]
    ancestors = list(ancestors or [])[:_MAX_ROWS]
    eval_results = list(eval_results or [])[:_MAX_ROWS]

    scorecard: dict[str, Any] = {}
    for row in eval_results or []:
        bench = row.get("benchmark")
        score = row.get("score")
        if bench is not None and score is not None:
            scorecard[str(bench)] = score

    is_adapter = _is_adapter(artifacts, config)

    # --- YAML frontmatter --------------------------------------------------- #
    fm: list[str] = ["---", f"library_name: {'peft' if is_adapter else 'transformers'}"]
    if base:
        fm.append(f"base_model: {_yaml_dq(base)}")
    fm.append("tags:")
    for tag in ("soup-cli", "fine-tuned", "compliance"):
        fm.append(f"  - {tag}")
    fm.append("---")

    lines: list[str] = list(fm)
    lines += ["", f"# {_safe_md_cell(name)}", ""]
    lines.append("Fine-tuned with [Soup CLI](https://github.com/MakazhanAlpamys/Soup).")
    lines.append("")

    # --- Model details ------------------------------------------------------ #
    lines += ["## Model Details", "", "| Field | Value |", "| --- | --- |"]
    detail_rows = [
        ("Name", name),
        ("Base model", base),
        ("Task", task),
        ("Type", "LoRA adapter" if is_adapter else "Full model"),
    ]
    if run_id:
        detail_rows.append(("Run id", run_id))
    if created:
        detail_rows.append(("Created", created))
    if tags:
        detail_rows.append(("Tags", ", ".join(str(t) for t in tags)))
    for key, val in detail_rows:
        lines.append(f"| {key} | {_safe_md_cell(val)} |")
    lines.append("")

    # --- Training ----------------------------------------------------------- #
    training_section = _render_training_section(training_cfg)
    if training_section:
        lines.append(training_section)

    # --- Evaluation --------------------------------------------------------- #
    eval_section = _render_eval_scorecard(scorecard)
    if eval_section:
        lines.append(eval_section)

    # --- Provenance --------------------------------------------------------- #
    if config_hash or data_hash:
        lines += ["## Provenance", "", "| Hash | Value |", "| --- | --- |"]
        if config_hash:
            lines.append(f"| config sha256 | `{_safe_md_cell(config_hash)}` |")
        if data_hash:
            lines.append(f"| data sha256 | `{_safe_md_cell(data_hash)}` |")
        lines.append("")

    # --- Lineage ------------------------------------------------------------ #
    if ancestors:
        lines += ["## Lineage", "", "| Ancestor | Relation | Entry id |", "| --- | --- | --- |"]
        for anc in ancestors:
            lines.append(
                f"| {_safe_md_cell(anc.get('name', ''))} "
                f"| {_safe_md_cell(anc.get('relation', ''))} "
                f"| `{_safe_md_cell(str(anc.get('id', ''))[:24])}` |"
            )
        lines.append("")

    # --- Artifacts (provenance / compliance) -------------------------------- #
    if artifacts:
        lines += [
            "## Artifacts",
            "",
            "| Kind | File | SHA256 (prefix) |",
            "| --- | --- | --- |",
        ]
        for art in artifacts:
            path = str(art.get("path", ""))
            filename = path.replace("\\", "/").rsplit("/", 1)[-1]
            sha = str(art.get("sha256", ""))[:16]
            lines.append(
                f"| {_safe_md_cell(art.get('kind', ''))} "
                f"| {_safe_md_cell(filename)} "
                f"| `{_safe_md_cell(sha)}` |"
            )
        lines.append("")

    # --- Notes -------------------------------------------------------------- #
    if notes:
        lines += ["## Notes", "", _truncate(html.escape(str(notes)), _MAX_NOTES_CHARS), ""]

    # --- Usage -------------------------------------------------------------- #
    lines += ["## Usage", ""]
    if is_adapter:
        lines += [
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            # json.dumps so a base_model containing a quote cannot produce a
            # syntactically broken snippet (the value is inert docs, never run).
            f"base = {json.dumps(_safe_md_cell(base) or 'BASE_MODEL')}",
            "model = AutoModelForCausalLM.from_pretrained(base)",
            'model = PeftModel.from_pretrained(model, "YOUR_REPO")',
            "tokenizer = AutoTokenizer.from_pretrained(base)",
            "```",
            "",
        ]
    else:
        lines += [
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            'model = AutoModelForCausalLM.from_pretrained("YOUR_REPO")',
            'tokenizer = AutoTokenizer.from_pretrained("YOUR_REPO")',
            "```",
            "",
        ]

    return "\n".join(lines) + "\n"


def build_card_for_ref(ref: str) -> str:
    """Resolve ``ref`` in the registry and render its model card.

    Raises :class:`CardError` on ambiguous / missing refs (so callers such as
    ``soup push --card`` can map it to an exit code).
    """
    with RegistryStore() as store:
        try:
            eid = store.resolve(ref)
        except AmbiguousRefError as exc:
            raise CardError(str(exc)) from exc
        if eid is None:
            raise CardError(f"Registry entry not found: {ref}")
        entry = store.get(eid)
        if entry is None:
            raise CardError(f"Registry entry not found: {ref}")
        artifacts = store.get_artifacts(eid)
        evals = store.get_eval_results(eid)
        ancestors = store.get_ancestors(eid)
    return build_model_card(entry, artifacts, evals, ancestors)


def card(
    ref: str = typer.Argument(..., help="Registry entry id / prefix / name:tag"),
    output: str = typer.Option(
        "MODELCARD.md",
        "-o",
        "--output",
        help="Output markdown path (must stay under the current directory)",
    ),
) -> None:
    """Generate a HuggingFace model card from a registry entry."""
    try:
        markdown = build_card_for_ref(ref)
    except CardError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    try:
        written = atomic_write_text(markdown, output, field="card output")
    except (ValueError, OSError) as exc:
        console.print(f"[red]Cannot write card: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"Model card written to [bold]{escape(written)}[/]",
            title="soup card",
            border_style="green",
        )
    )


# Optional re-exports for callers that want the raw pieces.
__all__ = ["build_model_card", "build_card_for_ref", "card", "CardError"]
