"""Export behavior checker — issue #2048.

Verifies that an exported model artifact (merged safetensors, or a
directory produced by ``soup merge``) still behaves like the LoRA
adapter it came from.

The core question: *did the LoRA delta survive the export?*

Three models are compared on identical prompts with greedy decoding:

  base     — the unmodified base model.
  ref      — base + LoRA adapter (the reference behavior).
  exported — the exported artifact under test.

A prompt where the adapter did *not* change the base output is
uninformative (the delta is silent there).  For prompts where the
adapter *did* change the output, we check whether the exported artifact
follows the adapter (delta preserved) or reverts to the base (delta
lost).

Verdict taxonomy (matching the Soup gate taxonomy):
  PASS        — exported model matches the adapter on every active prompt,
                at least one prompt is active, and overall agreement meets
                the threshold.
  DELTA_LOST  — on at least one prompt where the adapter changed the
                output the exported model reverted to the base; exit 2.
  LOW_AGREE   — agreement fell below ``min_agreement`` without a clean
                delta-lost case; exit 2.
  NO_EVIDENCE — none of the prompts exposed an adapter change, so fidelity
                cannot be verified; exit 2.

Only HuggingFace safetensors directories are supported for live
generation.  Standalone GGUF files, MLX-fused directories, and other
formats are refused with exit 3 and a diagnostic naming the limitation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

from soup_cli.utils.terminal import for_terminal

if TYPE_CHECKING:
    from rich.table import Table


# ---------------------------------------------------------------------------
# Supported-format guard
# ---------------------------------------------------------------------------

#: Suffixes that belong to formats we cannot run live generation against.
_REFUSED_SUFFIXES: frozenset[str] = frozenset({
    ".gguf", ".onnx", ".plan", ".engine", ".trt",
})

#: Weight filenames that indicate an MLX-fused directory.
_MLX_WEIGHT_NAMES: frozenset[str] = frozenset({
    "weights.npz",
})


def classify_artifact(path: Path) -> str:
    """Return a short format label or raise ValueError for unsupported formats.

    Returns ``'hf'`` for directories that look like HuggingFace
    safetensors models (or merged LoRA outputs).

    Raises ``ValueError`` with a human-readable message for GGUF, ONNX,
    TensorRT, and MLX-fused artifacts so the CLI can exit with
    EXIT_USAGE_ERROR (3) and a clear diagnostic.
    """
    if not path.exists():
        raise ValueError(f"path not found: {path}")

    # --- Standalone file formats ---
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            raise ValueError(
                f"'{path}' is a standalone GGUF file. "
                "export-check does not support standalone GGUF files; "
                "run 'soup llama cli' or use llama-cli directly to "
                "validate GGUF generation."
            )
        if suffix in _REFUSED_SUFFIXES:
            raise ValueError(
                f"'{path}' has suffix '{suffix}'. "
                "export-check only supports HuggingFace merged-model directories "
                f"(safetensors). Unsupported format: '{suffix}'."
            )
        raise ValueError(
            f"'{path}' is a file with an unrecognised suffix '{path.suffix}'. "
            "export-check expects a directory containing safetensors weight files."
        )

    # --- Directory: check for MLX-fused weights ---
    try:
        dir_files = {entry.name for entry in os.scandir(path)}
    except OSError as exc:
        raise ValueError(f"cannot list '{path}': {exc}") from exc

    if _MLX_WEIGHT_NAMES & dir_files:
        raise ValueError(
            f"'{path}' appears to be an MLX-fused model directory (contains "
            f"{_MLX_WEIGHT_NAMES & dir_files}). "
            "export-check does not support MLX-fused artifacts; "
            "use mlx_lm.generate directly to validate MLX models."
        )

    # Presence of at least one safetensors file is the HF signal.
    # (adapter_model.safetensors, model.safetensors, model-00001-of-*.safetensors …)
    has_safetensors = any(
        name.endswith(".safetensors") or name.endswith(".bin")
        for name in dir_files
    )
    if not has_safetensors:
        raise ValueError(
            f"'{path}' does not look like a HuggingFace model directory — "
            "no *.safetensors or *.bin weight files found."
        )

    return "hf"


# ---------------------------------------------------------------------------
# Text-distance helper (no extra dependencies)
# ---------------------------------------------------------------------------

def _char_overlap_ratio(a: str, b: str) -> float:
    """Return the character-level Jaccard overlap between *a* and *b*.

    Returns 1.0 when both strings are identical, 0.0 when they share no
    characters.  Uses a simple set-of-positions approach (bigrams over
    characters) for a stable, no-dependency distance that works well for
    short generation outputs.
    """
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Build character bigrams
    def bigrams(s: str) -> list[str]:
        return [s[i:i + 2] for i in range(len(s) - 1)] or list(s)

    bg_a = bigrams(a)
    bg_b = bigrams(b)
    set_a = set(bg_a)
    set_b = set(bg_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _first_divergent_pos(a: str, b: str) -> Optional[int]:
    """Return the character index of the first difference, or None if equal."""
    for idx, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportCheckPromptRow:
    """Per-prompt comparison result."""

    prompt: str
    base_output: str
    ref_output: str
    exported_output: str
    #: True when the adapter changed the base output on this prompt.
    adapter_active: bool
    #: Similarity of exported vs reference (1.0 = identical).
    agree_export_ref: float
    #: Whether the exported output exactly matches the adapter reference output.
    exact_match_export_ref: bool
    #: Similarity of exported vs base (1.0 = identical).
    agree_export_base: float
    #: Position of first divergent character between exported and ref.
    first_divergent_char: Optional[int]
    #: "DELTA_LOST" | "PASS" | "INACTIVE" (adapter did not change output here)
    verdict: str


@dataclass(frozen=True)
class ExportCheckResult:
    """Aggregate result returned by :func:`run_export_check`."""

    rows: list[ExportCheckPromptRow]
    overall_agreement: float
    exact_match_rate: float
    active_prompt_count: int
    delta_lost: bool
    passed: bool
    stub: bool = False

    def to_json(self) -> str:
        import json

        # JSON is emitted directly to the terminal by the CLI. Sanitize only
        # this render-time copy so model comparison and result objects retain
        # their original prompt/output strings.
        rows = [asdict(row) for row in self.rows]
        for row in rows:
            for field in ("prompt", "base_output", "ref_output", "exported_output"):
                row[field] = for_terminal(row[field])

        payload: dict[str, Any] = {
            "overall_agreement": self.overall_agreement,
            "exact_match_rate": self.exact_match_rate,
            "active_prompt_count": self.active_prompt_count,
            "delta_lost": self.delta_lost,
            "passed": self.passed,
            "rows": rows,
        }
        if self.stub:
            payload["stub"] = True
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

# Default prompts used when the caller does not supply a prompts file.
# These are intentionally generic — they probe basic instruction-following
# without requiring any specific fine-tuning domain.
_DEFAULT_PROMPTS: tuple[str, ...] = (
    "Summarise this text in one sentence: The quick brown fox jumps over the lazy dog.",
    "What is 7 multiplied by 8?",
    "List three primary colors.",
    "Translate 'hello' to Spanish.",
    "Complete the sentence: The capital of France is",
)


def load_prompts_file(path: str) -> list[str]:
    """Read prompts from a JSONL or plain-text file.

    JSONL rows may contain a ``"prompt"`` key or be bare strings.
    Plain-text files yield one prompt per non-empty line.

    Raises ``ValueError`` on parse failures.
    """
    import json

    result: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: invalid JSON — {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{lineno}: JSONL row must be a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                prompt = obj.get("prompt", "")
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"{path}:{lineno}: 'prompt' field must be a string, "
                        f"got {type(prompt).__name__}"
                    )
                result.append(prompt)
            else:
                result.append(line)
    if not result:
        raise ValueError(f"'{path}' contains no prompts.")
    return result


def _compare_one(
    *,
    prompt: str,
    base_gen: Callable[[str], str],
    ref_gen: Callable[[str], str],
    exported_gen: Callable[[str], str],
) -> ExportCheckPromptRow:
    """Run a single prompt through all three generators and compute metrics."""
    base_out = base_gen(prompt)
    ref_out = ref_gen(prompt)
    exp_out = exported_gen(prompt)

    adapter_active = base_out != ref_out
    agree_exp_ref = _char_overlap_ratio(exp_out, ref_out)
    agree_exp_base = _char_overlap_ratio(exp_out, base_out)
    first_div = _first_divergent_pos(exp_out, ref_out)

    if not adapter_active:
        verdict = "INACTIVE"
    elif agree_exp_ref >= agree_exp_base:
        # Exported is at least as close to reference as to base → delta survived.
        verdict = "PASS"
    else:
        # Exported is closer to base than to reference → delta was lost.
        verdict = "DELTA_LOST"

    return ExportCheckPromptRow(
        prompt=prompt,
        base_output=base_out,
        ref_output=ref_out,
        exported_output=exp_out,
        adapter_active=adapter_active,
        agree_export_ref=agree_exp_ref,
        exact_match_export_ref=exp_out == ref_out,
        agree_export_base=agree_exp_base,
        first_divergent_char=first_div,
        verdict=verdict,
    )


def run_export_check(
    *,
    base_gen: Callable[[str], str],
    ref_gen: Callable[[str], str],
    exported_gen: Callable[[str], str],
    prompts: Optional[list[str]] = None,
    min_agreement: float = 0.80,
    stub: bool = False,
) -> ExportCheckResult:
    """Compare base, reference (adapter), and exported model on *prompts*.

    Parameters
    ----------
    base_gen:
        Generator for the unmodified base model.
    ref_gen:
        Generator for the reference model (base + LoRA adapter).
    exported_gen:
        Generator for the exported artifact under test.
    prompts:
        Prompts to run.  Defaults to :data:`_DEFAULT_PROMPTS` when ``None``.
    min_agreement:
        Minimum required average ``agree_export_ref`` score across prompts
        where the adapter changes the base model's output. The default 0.80
        allows minor wording drift while the closer-to-base check still
        rejects an export that has reverted to base behavior.
    stub:
        True when synthetic stubs are used instead of live models.

    Returns
    -------
    ExportCheckResult
    """
    effective_prompts = prompts if prompts is not None else list(_DEFAULT_PROMPTS)
    if not effective_prompts:
        raise ValueError("No prompts provided.")

    rows: list[ExportCheckPromptRow] = []
    for prompt in effective_prompts:
        row = _compare_one(
            prompt=prompt,
            base_gen=base_gen,
            ref_gen=ref_gen,
            exported_gen=exported_gen,
        )
        rows.append(row)

    active_rows = [row for row in rows if row.adapter_active]
    delta_lost = any(row.verdict == "DELTA_LOST" for row in active_rows)
    overall_agreement = (
        sum(row.agree_export_ref for row in active_rows) / len(active_rows)
        if active_rows
        else 0.0
    )
    exact_match_rate = (
        sum(row.exact_match_export_ref for row in active_rows) / len(active_rows)
        if active_rows
        else 0.0
    )
    passed = bool(active_rows) and (not delta_lost) and (overall_agreement >= min_agreement)

    return ExportCheckResult(
        rows=rows,
        overall_agreement=overall_agreement,
        exact_match_rate=exact_match_rate,
        active_prompt_count=len(active_rows),
        delta_lost=delta_lost,
        passed=passed,
        stub=stub,
    )


# ---------------------------------------------------------------------------
# Model generator factory (lazy — avoids module-level heavy imports)
# ---------------------------------------------------------------------------

def make_hf_generator(
    model_path: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Return a greedy HuggingFace generator for *model_path*.

    Raises ``ImportError`` when transformers is not installed, or
    ``OSError`` / ``ValueError`` on a bad path.
    """
    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be in [1, 4096]")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=False
    )
    model.eval()

    def _generate(prompt: str) -> str:
        if not prompt:
            return ""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


def make_hf_adapter_generator(
    adapter_path: str,
    base_model: str,
    *,
    max_new_tokens: int = 128,
) -> Callable[[str], str]:
    """Return a greedy HuggingFace+PEFT generator for *adapter_path* on *base_model*.

    Raises ``ImportError`` when transformers / peft are not installed.
    """
    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be in [1, 4096]")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    def _generate(prompt: str) -> str:
        if not prompt:
            return ""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


def stub_generator(label: str) -> Callable[[str], str]:
    """Return a deterministic stub generator for tests and ``--allow-stub``.

    Each stub returns a distinct fixed string so tests can probe whether
    the stub is being used correctly.
    """
    def _stub(prompt: str) -> str:  # noqa: ARG001
        return f"[stub:{label}]"

    return _stub


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _verdict_color(verdict: str) -> str:
    return {
        "PASS": "[green]PASS[/]",
        "INACTIVE": "[dim]INACTIVE[/]",
        "DELTA_LOST": "[red]DELTA_LOST[/]",
    }.get(verdict, verdict)


def render_table(result: ExportCheckResult) -> "Table":
    """Render an :class:`ExportCheckResult` as a Rich table."""
    from rich.table import Table

    title = "Export behavior check"
    if result.stub:
        title += " [yellow](stub — not a live measurement)[/]"
    table = Table(title=title)
    table.add_column("Prompt", style="cyan", max_width=40, no_wrap=True)
    table.add_column("Active?", justify="center")
    table.add_column("Agree ref", justify="right")
    table.add_column("Exact", justify="center")
    table.add_column("Agree base", justify="right")
    table.add_column("1st div", justify="right")
    table.add_column("Verdict")
    for row in result.rows:
        prompt = row.prompt[:38] + ".." if len(row.prompt) > 40 else row.prompt
        table.add_row(
            for_terminal(prompt),
            "yes" if row.adapter_active else "no",
            f"{row.agree_export_ref:.2f}",
            "yes" if row.exact_match_export_ref else "no",
            f"{row.agree_export_base:.2f}",
            str(row.first_divergent_char) if row.first_divergent_char is not None else "-",
            _verdict_color(row.verdict),
        )
    return table


def render_markdown(result: ExportCheckResult) -> str:
    lines = []
    if result.stub:
        lines.append("> **Warning**: Stub generators — not a live model measurement.\n")
    lines.extend([
        "| Prompt (truncated) | Active | Agree ref | Exact | Agree base | 1st div | Verdict |",
        "|--------------------|--------|-----------|-------|------------|---------|---------|",
    ])
    for row in result.rows:
        prompt_trunc = (row.prompt[:38] + "..") if len(row.prompt) > 40 else row.prompt
        prompt_trunc = for_terminal(prompt_trunc).replace("|", r"\|")
        lines.append(
            f"| {prompt_trunc} | {'yes' if row.adapter_active else 'no'} "
            f"| {row.agree_export_ref:.2f} "
            f"| {'yes' if row.exact_match_export_ref else 'no'} "
            f"| {row.agree_export_base:.2f} "
            f"| {row.first_divergent_char if row.first_divergent_char is not None else '-'} "
            f"| {row.verdict} |"
        )
    lines.append(f"\n**Adapter-active prompts:** {result.active_prompt_count}")
    lines.append(f"**Overall agreement:** {result.overall_agreement:.2f}")
    lines.append(f"**Exact-match rate:** {result.exact_match_rate:.2f}")
    lines.append(f"**Delta lost:** {'yes' if result.delta_lost else 'no'}")
    lines.append(f"**Result:** {'PASS' if result.passed else 'FAIL'}")
    return "\n".join(lines)


FORMAT_RENDERERS: dict[str, Callable[[ExportCheckResult], Union["Table", str]]] = {
    "table": render_table,
    "json": lambda r: r.to_json(),
    "markdown": render_markdown,
}


def ensure_format(fmt: str) -> None:
    """Raise ``ValueError`` for unknown format strings."""
    if fmt not in FORMAT_RENDERERS:
        raise ValueError(
            f"unknown format '{fmt}'. Use table | json | markdown"
        )


def render(
    result: ExportCheckResult,
    *,
    fmt: Literal["table", "json", "markdown"],
) -> Union["Table", str]:
    renderer = FORMAT_RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"unknown format '{fmt}'. Use table | json | markdown")
    return renderer(result)
