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
  PASS          — exported model matches the adapter on every active prompt,
                  at least one prompt is active, and overall agreement meets
                  the threshold.
  DELTA_LOST    — on at least one prompt where the adapter changed the
                  output the exported model reverted to the base; exit 2.
  LOW_AGREEMENT — agreement fell below ``min_agreement`` without a clean
                  delta-lost case; exit 2.
  NO_EVIDENCE   — none of the prompts exposed an adapter change, so fidelity
                  cannot be verified; exit 2.

Only HuggingFace safetensors directories are supported for live
generation.  Standalone GGUF files, MLX-fused directories, and other
formats are refused with exit 3 and a diagnostic naming the limitation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol, Union

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

    if "adapter_config.json" in dir_files:
        raise ValueError(
            f"'{path}' is a LoRA adapter directory, not a merged model. "
            "Pass the merged artifact to --exported, not the adapter."
        )

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
class GenerationOutput:
    """Generation output including raw decoded text and token IDs."""

    text: str
    token_ids: list[int]


class ModelEvaluator(Protocol):
    """Protocol for model evaluation in export-check."""

    def generate(self, prompt: str, max_new_tokens: int = 128) -> GenerationOutput:
        """Run greedy decoding on prompt and return text and generated token IDs."""
        ...

    def compute_nll(self, texts: list[str]) -> float:
        """Compute mean next-token negative log-likelihood on texts."""
        ...


class FunctionModelEvaluator:
    """Adapts a simple callable or stub generator into a ModelEvaluator."""

    def __init__(
        self,
        gen_fn: Callable[[str], str],
        nll_fn: Optional[Callable[[list[str]], float]] = None,
    ) -> None:
        self._gen_fn = gen_fn
        self._nll_fn = nll_fn

    def generate(self, prompt: str, max_new_tokens: int = 128) -> GenerationOutput:  # noqa: ARG002
        out_text = self._gen_fn(prompt)
        # Approximate tokens via utf-8 bytes for synthetic / stub functions without a tokenizer
        token_ids = list(out_text.encode("utf-8"))
        return GenerationOutput(text=out_text, token_ids=token_ids)

    def compute_nll(self, texts: list[str]) -> float:
        if self._nll_fn is not None:
            return self._nll_fn(texts)
        return 0.0


def _coerce_evaluator(
    evaluator_or_callable: Union[
        ModelEvaluator, Callable[[str], str], Callable[[str], GenerationOutput]
    ],
) -> ModelEvaluator:
    """Ensure an evaluator object conforms to ModelEvaluator."""
    if hasattr(evaluator_or_callable, "generate") and hasattr(evaluator_or_callable, "compute_nll"):
        return evaluator_or_callable  # type: ignore[return-value]

    # It's a callable
    def _adapted(prompt: str) -> str:
        res = evaluator_or_callable(prompt)  # type: ignore[operator]
        return res.text if isinstance(res, GenerationOutput) else str(res)

    return FunctionModelEvaluator(_adapted)


def _first_divergent_token(a: list[int], b: list[int]) -> Optional[int]:
    """Return the 0-based token index of the first difference, or None if equal."""
    for idx, (ta, tb) in enumerate(zip(a, b)):
        if ta != tb:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


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
    #: 0-based index of first divergent token between exported and ref (None if identical).
    first_divergent_token: Optional[int]
    #: Position of first divergent character between exported and ref (legacy char offset).
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
    base_nll: Optional[float] = None
    ref_nll: Optional[float] = None
    exported_nll: Optional[float] = None
    verdict: str = "PASS"
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
            "verdict": self.verdict,
            "base_nll": self.base_nll,
            "ref_nll": self.ref_nll,
            "exported_nll": self.exported_nll,
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

    Supports:
    - JSONL objects with a ``"prompt"`` key.
    - Alpaca-format objects with an ``"instruction"`` key (and optional ``"input"``).
    - ChatML-format objects with a ``"messages"`` key (extracts user/system messages).
    - JSONL bare JSON strings (e.g. ``"Hello world"``).
    - Plain text files (one prompt per non-empty line).

    Raises ``ValueError`` on parse failures with file name and line number.
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
                if "prompt" in obj:
                    prompt = obj["prompt"]
                    if not isinstance(prompt, str):
                        raise ValueError(
                            f"{path}:{lineno}: 'prompt' field must be a string, "
                            f"got {type(prompt).__name__}"
                        )
                    result.append(prompt)
                elif "instruction" in obj:
                    # Alpaca format: instruction + optional input
                    instruction = obj["instruction"]
                    if not isinstance(instruction, str):
                        raise ValueError(
                            f"{path}:{lineno}: 'instruction' field must be a string, "
                            f"got {type(instruction).__name__}"
                        )
                    inp = obj.get("input", "")
                    if inp and isinstance(inp, str):
                        prompt = f"{instruction}\n\n{inp}".strip()
                    else:
                        prompt = instruction
                    result.append(prompt)
                elif "messages" in obj:
                    # ChatML format: list of messages dicts [{"role": ..., "content": ...}]
                    msgs = obj["messages"]
                    if not isinstance(msgs, list):
                        raise ValueError(
                            f"{path}:{lineno}: 'messages' field must be a list, "
                            f"got {type(msgs).__name__}"
                        )
                    content_parts = []
                    for m in msgs:
                        if isinstance(m, dict) and "content" in m and isinstance(m["content"], str):
                            content_parts.append(m["content"])
                    if not content_parts:
                        raise ValueError(
                            f"{path}:{lineno}: 'messages' list contained no text content"
                        )
                    result.append("\n\n".join(content_parts))
                else:
                    raise ValueError(
                        f"{path}:{lineno}: JSON object missing recognizable prompt key "
                        f"(expected 'prompt', 'instruction', or 'messages')"
                    )
            elif line.startswith('"') and line.endswith('"') and len(line) >= 2:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        result.append(obj)
                    else:
                        result.append(line)
                except json.JSONDecodeError:
                    result.append(line)
            else:
                result.append(line)
    if not result:
        raise ValueError(f"'{path}' contains no prompts.")
    return result


def _compare_one(
    *,
    prompt: str,
    base_eval: ModelEvaluator,
    ref_eval: ModelEvaluator,
    exported_eval: ModelEvaluator,
) -> ExportCheckPromptRow:
    """Run a single prompt through all three evaluators and compute metrics."""
    base_res = base_eval.generate(prompt)
    ref_res = ref_eval.generate(prompt)
    exp_res = exported_eval.generate(prompt)

    base_out = base_res.text
    ref_out = ref_res.text
    exp_out = exp_res.text

    adapter_active = (base_out != ref_out) or (base_res.token_ids != ref_res.token_ids)
    agree_exp_ref = _char_overlap_ratio(exp_out, ref_out)
    agree_exp_base = _char_overlap_ratio(exp_out, base_out)
    first_div_char = _first_divergent_pos(exp_out, ref_out)
    first_div_tok = _first_divergent_token(exp_res.token_ids, ref_res.token_ids)

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
        exact_match_export_ref=(exp_out == ref_out),
        agree_export_base=agree_exp_base,
        first_divergent_token=first_div_tok,
        first_divergent_char=first_div_char,
        verdict=verdict,
    )


def _aggregate_rows_and_nll(
    *,
    prompts: list[str],
    base_outputs: list[GenerationOutput],
    ref_outputs: list[GenerationOutput],
    exp_outputs: list[GenerationOutput],
    base_nll_val: Optional[float],
    ref_nll_val: Optional[float],
    exp_nll_val: Optional[float],
    min_agreement: float,
    stub: bool,
) -> ExportCheckResult:
    """Shared aggregation logic used by run_export_check and
    run_export_check_from_factories.
    """
    rows: list[ExportCheckPromptRow] = []
    for prompt, base_res, ref_res, exp_res in zip(prompts, base_outputs, ref_outputs, exp_outputs):
        base_out = base_res.text
        ref_out = ref_res.text
        exp_out = exp_res.text

        adapter_active = (base_out != ref_out) or (base_res.token_ids != ref_res.token_ids)
        agree_exp_ref = _char_overlap_ratio(exp_out, ref_out)
        agree_exp_base = _char_overlap_ratio(exp_out, base_out)
        first_div_char = _first_divergent_pos(exp_out, ref_out)
        first_div_tok = _first_divergent_token(exp_res.token_ids, ref_res.token_ids)

        if not adapter_active:
            row_verdict = "INACTIVE"
        elif agree_exp_ref >= agree_exp_base:
            row_verdict = "PASS"
        else:
            row_verdict = "DELTA_LOST"

        rows.append(ExportCheckPromptRow(
            prompt=prompt,
            base_output=base_out,
            ref_output=ref_out,
            exported_output=exp_out,
            adapter_active=adapter_active,
            agree_export_ref=agree_exp_ref,
            exact_match_export_ref=(exp_out == ref_out),
            agree_export_base=agree_exp_base,
            first_divergent_token=first_div_tok,
            first_divergent_char=first_div_char,
            verdict=row_verdict,
        ))

    active_rows = [row for row in rows if row.adapter_active]
    delta_lost_generation = any(row.verdict == "DELTA_LOST" for row in active_rows)

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

    nll_delta_lost = False
    all_nlls_present = (
        base_nll_val is not None and ref_nll_val is not None and exp_nll_val is not None
    )
    if not stub and all_nlls_present:
        adapter_nll_delta = abs(base_nll_val - ref_nll_val)  # type: ignore[operator]
        if adapter_nll_delta > 1e-4:
            dist_exp_ref = abs(exp_nll_val - ref_nll_val)  # type: ignore[operator]
            dist_exp_base = abs(exp_nll_val - base_nll_val)  # type: ignore[operator]
            if dist_exp_base < dist_exp_ref:
                nll_delta_lost = True

    delta_lost = delta_lost_generation or nll_delta_lost

    if delta_lost:
        verdict = "DELTA_LOST"
        passed = False
    elif not active_rows:
        verdict = "NO_EVIDENCE"
        passed = False
    elif overall_agreement < min_agreement:
        verdict = "LOW_AGREEMENT"
        passed = False
    else:
        verdict = "PASS"
        passed = True

    return ExportCheckResult(
        rows=rows,
        overall_agreement=overall_agreement,
        exact_match_rate=exact_match_rate,
        active_prompt_count=len(active_rows),
        delta_lost=delta_lost,
        passed=passed,
        base_nll=base_nll_val,
        ref_nll=ref_nll_val,
        exported_nll=exp_nll_val,
        verdict=verdict,
        stub=stub,
    )


# Type alias for a model factory callable.
ModelFactory = Callable[[], ModelEvaluator]


def run_export_check_from_factories(
    *,
    base_factory: ModelFactory,
    ref_factory: ModelFactory,
    exported_factory: ModelFactory,
    prompts: Optional[list[str]] = None,
    min_agreement: float = 0.80,
    compute_nll: bool = True,
    stub: bool = False,
) -> ExportCheckResult:
    """Load, evaluate, and unload each model sequentially to avoid concurrent memory use.

    Each factory is called exactly once.  The model it returns is used for
    all prompts, then explicitly released before the next factory is called.
    This keeps only one model in memory at a time.

    Parameters
    ----------
    base_factory:
        Zero-argument callable that loads and returns the base ModelEvaluator.
    ref_factory:
        Zero-argument callable that loads and returns the reference (adapter) ModelEvaluator.
    exported_factory:
        Zero-argument callable that loads and returns the exported ModelEvaluator.
    prompts:
        Prompts to run. Defaults to :data:`_DEFAULT_PROMPTS` when ``None``.
    min_agreement:
        Minimum required average ``agree_export_ref`` score across active prompts.
    compute_nll:
        Whether to calculate held-out causal next-token NLL across prompts.
    stub:
        True when synthetic stubs are used instead of live models.

    Returns
    -------
    ExportCheckResult
    """
    import gc

    effective_prompts = prompts if prompts is not None else list(_DEFAULT_PROMPTS)
    if not effective_prompts:
        raise ValueError("No prompts provided.")

    # --- Pass 1: base model ---
    base_eval = base_factory()
    base_outputs = [base_eval.generate(p) for p in effective_prompts]
    base_nll_val: Optional[float] = None
    if compute_nll and not stub:
        base_nll_val = base_eval.compute_nll(effective_prompts)
    del base_eval
    gc.collect()

    # --- Pass 2: reference (adapter) model ---
    ref_eval = ref_factory()
    ref_outputs = [ref_eval.generate(p) for p in effective_prompts]
    ref_nll_val: Optional[float] = None
    if compute_nll and not stub:
        ref_nll_val = ref_eval.compute_nll(effective_prompts)
    del ref_eval
    gc.collect()

    # --- Pass 3: exported model ---
    exp_eval = exported_factory()
    exp_outputs = [exp_eval.generate(p) for p in effective_prompts]
    exp_nll_val: Optional[float] = None
    if compute_nll and not stub:
        exp_nll_val = exp_eval.compute_nll(effective_prompts)
    del exp_eval
    gc.collect()

    return _aggregate_rows_and_nll(
        prompts=effective_prompts,
        base_outputs=base_outputs,
        ref_outputs=ref_outputs,
        exp_outputs=exp_outputs,
        base_nll_val=base_nll_val,
        ref_nll_val=ref_nll_val,
        exp_nll_val=exp_nll_val,
        min_agreement=min_agreement,
        stub=stub,
    )


def run_export_check(
    *,
    base_gen: Union[ModelEvaluator, Callable[[str], str], Callable[[str], GenerationOutput]],
    ref_gen: Union[ModelEvaluator, Callable[[str], str], Callable[[str], GenerationOutput]],
    exported_gen: Union[ModelEvaluator, Callable[[str], str], Callable[[str], GenerationOutput]],
    prompts: Optional[list[str]] = None,
    min_agreement: float = 0.80,
    compute_nll: bool = True,
    stub: bool = False,
) -> ExportCheckResult:
    """Compare base, reference (adapter), and exported model on *prompts*.

    Parameters
    ----------
    base_gen:
        Evaluator or generator for the unmodified base model.
    ref_gen:
        Evaluator or generator for the reference model (base + LoRA adapter).
    exported_gen:
        Evaluator or generator for the exported artifact under test.
    prompts:
        Prompts to run. Defaults to :data:`_DEFAULT_PROMPTS` when ``None``.
    min_agreement:
        Minimum required average ``agree_export_ref`` score across prompts
        where the adapter changes the base model's output. The default 0.80
        allows minor wording drift while the closer-to-base check still
        rejects an export that has reverted to base behavior.
    compute_nll:
        Whether to calculate held-out causal next-token NLL across prompts.
    stub:
        True when synthetic stubs are used instead of live models.

    Returns
    -------
    ExportCheckResult
    """
    effective_prompts = prompts if prompts is not None else list(_DEFAULT_PROMPTS)
    if not effective_prompts:
        raise ValueError("No prompts provided.")

    base_eval = _coerce_evaluator(base_gen)
    ref_eval = _coerce_evaluator(ref_gen)
    exp_eval = _coerce_evaluator(exported_gen)

    base_outputs = [base_eval.generate(p) for p in effective_prompts]
    ref_outputs = [ref_eval.generate(p) for p in effective_prompts]
    exp_outputs = [exp_eval.generate(p) for p in effective_prompts]

    # Held-out NLL computation (all three evaluators are still in scope here)
    base_nll_val: Optional[float] = None
    ref_nll_val: Optional[float] = None
    exp_nll_val: Optional[float] = None
    if compute_nll and not stub:
        base_nll_val = base_eval.compute_nll(effective_prompts)
        ref_nll_val = ref_eval.compute_nll(effective_prompts)
        exp_nll_val = exp_eval.compute_nll(effective_prompts)

    return _aggregate_rows_and_nll(
        prompts=effective_prompts,
        base_outputs=base_outputs,
        ref_outputs=ref_outputs,
        exp_outputs=exp_outputs,
        base_nll_val=base_nll_val,
        ref_nll_val=ref_nll_val,
        exp_nll_val=exp_nll_val,
        min_agreement=min_agreement,
        stub=stub,
    )


# ---------------------------------------------------------------------------
# Model generator factory (lazy — avoids module-level heavy imports)
# ---------------------------------------------------------------------------

class HFCausalLMEvaluator:
    """Evaluates a causal LM: greedy text generation and masked NLL computation."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 128) -> GenerationOutput:
        if not prompt:
            return GenerationOutput(text="", token_ids=[])

        max_pos = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", 2048
        )
        max_prompt_len = max(1, max_pos - max_new_tokens)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = outputs[0][prompt_len:].tolist()
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return GenerationOutput(text=text, token_ids=new_token_ids)

    def compute_nll(self, texts: list[str]) -> float:
        import torch
        from torch.nn import functional as torch_f

        if not texts:
            return 0.0

        max_pos = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", 2048
        )
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_pos
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        loss_flat = torch_f.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        loss_flat = loss_flat.view(shift_labels.shape) * shift_mask
        total_loss = float(loss_flat.sum().item())
        total_tokens = float(shift_mask.sum().item())
        return total_loss / total_tokens if total_tokens > 0 else 0.0


def make_hf_evaluator(
    model_path: str,
    *,
    max_new_tokens: int = 128,
) -> HFCausalLMEvaluator:
    """Return a HuggingFace evaluator for *model_path*."""
    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be in [1, 4096]")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False)
    model.eval()
    return HFCausalLMEvaluator(model, tokenizer)


def make_hf_adapter_evaluator(
    adapter_path: str,
    base_model: str,
    *,
    max_new_tokens: int = 128,
) -> HFCausalLMEvaluator:
    """Return a HuggingFace+PEFT evaluator for *adapter_path* on *base_model*."""
    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be in [1, 4096]")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return HFCausalLMEvaluator(model, tokenizer)


def make_hf_generator(
    model_path: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,  # noqa: ARG001
) -> Callable[[str], str]:
    """Return a greedy HuggingFace generator for *model_path*.

    Raises ``ImportError`` when transformers is not installed, or
    ``OSError`` / ``ValueError`` on a bad path.
    """
    evaluator = make_hf_evaluator(model_path, max_new_tokens=max_new_tokens)

    def _generate(prompt: str) -> str:
        return evaluator.generate(prompt, max_new_tokens=max_new_tokens).text

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
    evaluator = make_hf_adapter_evaluator(
        adapter_path=adapter_path,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
    )

    def _generate(prompt: str) -> str:
        return evaluator.generate(prompt, max_new_tokens=max_new_tokens).text

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
        "LOW_AGREEMENT": "[red]LOW_AGREEMENT[/]",
        "NO_EVIDENCE": "[yellow]NO_EVIDENCE[/]",
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
    table.add_column("1st div tok", justify="right")
    table.add_column("Verdict")
    for row in result.rows:
        prompt = row.prompt[:38] + ".." if len(row.prompt) > 40 else row.prompt
        first_div = (
            str(row.first_divergent_token)
            if row.first_divergent_token is not None
            else (str(row.first_divergent_char) if row.first_divergent_char is not None else "-")
        )
        table.add_row(
            for_terminal(prompt),
            "yes" if row.adapter_active else "no",
            f"{row.agree_export_ref:.2f}",
            "yes" if row.exact_match_export_ref else "no",
            f"{row.agree_export_base:.2f}",
            first_div,
            _verdict_color(row.verdict),
        )
    return table


def render_markdown(result: ExportCheckResult) -> str:
    lines = []
    if result.stub:
        lines.append("> **Warning**: Stub generators — not a live model measurement.\n")
    lines.extend([
        "| Prompt (truncated) | Active | Agree ref | Exact | Agree base | 1st div tok | Verdict |",
        "|--------------------|--------|-----------|-------|------------|-------------|---------|",
    ])
    for row in result.rows:
        prompt_trunc = (row.prompt[:38] + "..") if len(row.prompt) > 40 else row.prompt
        prompt_trunc = for_terminal(prompt_trunc).replace("|", r"\|")
        first_div = (
            str(row.first_divergent_token)
            if row.first_divergent_token is not None
            else (str(row.first_divergent_char) if row.first_divergent_char is not None else "-")
        )
        lines.append(
            f"| {prompt_trunc} | {'yes' if row.adapter_active else 'no'} "
            f"| {row.agree_export_ref:.2f} "
            f"| {'yes' if row.exact_match_export_ref else 'no'} "
            f"| {row.agree_export_base:.2f} "
            f"| {first_div} "
            f"| {row.verdict} |"
        )
    lines.append(f"\n**Adapter-active prompts:** {result.active_prompt_count}")
    lines.append(f"**Overall agreement:** {result.overall_agreement:.2f}")
    lines.append(f"**Exact-match rate:** {result.exact_match_rate:.2f}")
    if result.base_nll is not None:
        lines.append(f"**Base NLL:** {result.base_nll:.4f}")
    if result.ref_nll is not None:
        lines.append(f"**Reference NLL:** {result.ref_nll:.4f}")
    if result.exported_nll is not None:
        lines.append(f"**Exported NLL:** {result.exported_nll:.4f}")
    lines.append(f"**Delta lost:** {'yes' if result.delta_lost else 'no'}")
    lines.append(f"**Result:** {'PASS' if result.passed else 'FAIL'} ({result.verdict})")
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
