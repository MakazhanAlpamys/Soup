"""Speculative-decoding draft engine (v0.71.33).

``soup draft`` distils a target model into a tiny *draft* model and reports how
often that draft would be accepted by the target during speculative decoding.

Two halves, deliberately separated:

* **Pure half** (this is the moat, and it is CPU-testable with no models):
  :func:`compute_acceptance`, :func:`classify_acceptance`,
  :func:`same_tokenizer`, the frozen :class:`AcceptanceReport`, its renderer,
  and the local draft registry.
* **Torch-lazy half**: :func:`measure_acceptance` / :func:`measure_throughput`
  import torch inside the function body.

**Acceptance rate.** ``transformers`` does not expose accepted-token counts
from assisted generation, so we measure the metric the speculative-decoding
literature reports (Medusa / EAGLE): *teacher-forced argmax agreement*. The
target greedy-generates a continuation; the draft forwards ONCE over that
sequence; alpha = the fraction of generated positions where the draft's argmax
equals the token the target actually produced. It is exact, deterministic and
cheap. It is NOT a wall-clock speedup prediction — a sampling-based
speculative run also depends on the rejection-resample cascade.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover — typing only; torch stays lazy at runtime
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Verdict bands. STRONG at >= 0.70 is where speculative decoding starts paying
# for the draft's forward pass on realistic hardware.
ACCEPTANCE_STRONG = 0.70
ACCEPTANCE_MODERATE = 0.50

VERDICT_STRONG = "STRONG"
VERDICT_MODERATE = "MODERATE"
VERDICT_WEAK = "WEAK"

# Probe strings for :func:`same_tokenizer`. Deliberately varied — ASCII words,
# digits, punctuation, non-ASCII, whitespace and a newline — because two
# tokenizers routinely agree on plain lowercase words and diverge everywhere
# else. A single-word probe would wave through an incompatible pair.
PROBE_CORPUS: tuple[str, ...] = (
    "Hello, world!",
    "The quick brown fox jumps over 13 lazy dogs.",
    "def fibonacci(n: int) -> int:\n    return n",
    "éàü 你好 русский",
    "1234567890 %$#@!",
)

# Local draft registry — mirrors the ~/.soup/spectrum cache (v0.71.23) and
# SOUP_REGISTRY_DB_PATH (v0.26.0) precedents.
_DRAFT_REGISTRY_ENV = "SOUP_DRAFT_REGISTRY_PATH"
_MAX_REGISTRY_ENTRIES = 200
_MAX_REGISTRY_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pure kernels
# ---------------------------------------------------------------------------
def count_accepted(draft_argmax: Sequence[int], target_ids: Sequence[int]) -> int:
    """Number of positions where the draft's argmax matches the target token.

    Both sequences must cover the SAME generated positions.

    Raises:
        ValueError: the two sequences differ in length — that would silently
            compare misaligned positions and report a meaningless number.
    """
    if len(draft_argmax) != len(target_ids):
        raise ValueError(
            "draft_argmax and target_ids must be the same length, got "
            f"{len(draft_argmax)} and {len(target_ids)}"
        )
    return sum(
        1 for proposed, actual in zip(draft_argmax, target_ids) if proposed == actual
    )


def compute_acceptance(
    draft_argmax: Sequence[int], target_ids: Sequence[int]
) -> float:
    """Acceptance rate for a SINGLE generated sequence.

    Public convenience kernel over :func:`count_accepted` (the corpus-level
    aggregate path uses ``count_accepted`` + :func:`acceptance_rate` instead, so
    the division happens once over the whole corpus rather than per sequence).
    An empty pair scores 0.0 — nothing was proposed, so nothing was accepted.
    """
    matched = count_accepted(draft_argmax, target_ids)  # also length-checks
    if not target_ids:
        return 0.0
    return matched / len(target_ids)


def acceptance_rate(accepted: int, total: int) -> float:
    """Aggregate acceptance over a corpus. ``total == 0`` scores 0.0."""
    if total < 0 or accepted < 0:
        raise ValueError("accepted and total must be non-negative")
    if accepted > total:
        raise ValueError(f"accepted ({accepted}) exceeds total ({total})")
    if total == 0:
        return 0.0
    return accepted / total


def classify_acceptance(rate: float) -> str:
    """Bucket an acceptance rate into ``STRONG`` / ``MODERATE`` / ``WEAK``."""
    if isinstance(rate, bool):
        raise TypeError(f"acceptance rate must not be bool, got {rate!r}")
    if not isinstance(rate, (int, float)):
        raise TypeError(
            f"acceptance rate must be a number, got {type(rate).__name__}"
        )
    value = float(rate)
    if not math.isfinite(value):
        raise ValueError(f"acceptance rate must be finite, got {rate!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"acceptance rate must be between 0 and 1, got {rate!r}")
    if value >= ACCEPTANCE_STRONG:
        return VERDICT_STRONG
    if value >= ACCEPTANCE_MODERATE:
        return VERDICT_MODERATE
    return VERDICT_WEAK


def same_tokenizer(
    tok_a: "PreTrainedTokenizerBase", tok_b: "PreTrainedTokenizerBase"
) -> bool:
    """True when two tokenizers are interchangeable for speculative decoding.

    Equal ``vocab_size`` AND identical ids over :data:`PROBE_CORPUS`. The probe
    matters: a vocab-size check alone passes two 32000-token tokenizers that
    disagree on every token, which would make the draft's proposals pure noise
    (and ``assistant_model=`` silently produce garbage rather than fail).

    A tokenizer that raises while encoding is treated as incompatible rather
    than crashing the caller.
    """
    try:
        if not hasattr(tok_a, "vocab_size") or not hasattr(tok_b, "vocab_size"):
            # A tokenizer that cannot report its vocab size cannot be proven
            # compatible — refuse rather than assume.
            return False
        if int(tok_a.vocab_size) != int(tok_b.vocab_size):
            return False
        for probe in PROBE_CORPUS:
            ids_a = tok_a.encode(probe, add_special_tokens=False)
            ids_b = tok_b.encode(probe, add_special_tokens=False)
            if list(ids_a) != list(ids_b):
                return False
    except Exception:  # noqa: BLE001 — a broken tokenizer is "not compatible"
        return False
    return True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptanceReport:
    """One-screen result of ``soup draft measure``."""

    target: str
    draft: str
    n_prompts: int
    n_generated_tokens: int
    acceptance_rate: float
    verdict: str
    tok_s_plain: Optional[float]
    tok_s_assisted: Optional[float]
    speedup: Optional[float]
    num_assistant_tokens: int
    soup_version: str


def draft_report_to_dict(report: AcceptanceReport) -> dict:
    """Serialise a report (``--output report.json``)."""
    return asdict(report)


def _fmt(value: Optional[float], suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def render_draft_panel(report: AcceptanceReport) -> Panel:
    """Rich panel — data and render stay separate (house style)."""
    colour = {
        VERDICT_STRONG: "green",
        VERDICT_MODERATE: "yellow",
        VERDICT_WEAK: "red",
    }.get(report.verdict, "white")

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Target", report.target)
    table.add_row("Draft", report.draft)
    table.add_row(
        "Acceptance",
        f"[bold {colour}]{report.acceptance_rate * 100:.1f}%[/] "
        f"([bold {colour}]{report.verdict}[/])",
    )
    table.add_row(
        "Sampled",
        f"{report.n_generated_tokens} tokens over {report.n_prompts} prompts",
    )
    table.add_row("Throughput", f"{_fmt(report.tok_s_plain, ' tok/s')} plain")
    table.add_row(
        "",
        f"{_fmt(report.tok_s_assisted, ' tok/s')} assisted "
        f"(draft={report.num_assistant_tokens} tok/step)",
    )
    table.add_row("Speedup", f"{_fmt(report.speedup, 'x')}")

    return Panel(
        table,
        title=f"[bold {colour}]Draft acceptance: {report.verdict}[/]",
        border_style=colour,
    )


# ---------------------------------------------------------------------------
# Local draft registry (~/.soup/drafts.json)
# ---------------------------------------------------------------------------
def draft_registry_path() -> str:
    """Path to the local draft registry (``SOUP_DRAFT_REGISTRY_PATH`` wins)."""
    override = os.environ.get(_DRAFT_REGISTRY_ENV)
    if override:
        return override
    return str(Path.home() / ".soup" / "drafts.json")


def _atomic_write_json(payload: dict, path: str) -> str:
    """Atomic JSON write into the draft registry.

    The registry lives under ``$HOME`` (not cwd), so it deliberately does NOT
    use ``paths.atomic_write_text`` (which enforces cwd containment) — mirrors
    ``spectrum_scan._atomic_write_json``. 0600 on POSIX: the file records local
    filesystem paths.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".soup.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        if os.name == "posix":
            try:
                os.chmod(tmp, 0o600)
            except OSError:  # pragma: no cover — best-effort
                pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return os.path.realpath(path)


def _read_registry() -> list[dict]:
    """Load registry entries. A missing/corrupt file reads as empty, never raises.

    ``soup serve --auto-spec`` calls into this on every start; a hand-edited or
    truncated JSON file must not take the server down.
    """
    path = draft_registry_path()
    try:
        if not os.path.isfile(path):
            return []
        if os.path.getsize(path) > _MAX_REGISTRY_BYTES:
            return []
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        drafts = data.get("drafts") if isinstance(data, dict) else None
        if not isinstance(drafts, list):
            return []
        return [entry for entry in drafts if isinstance(entry, dict)]
    except Exception:  # noqa: BLE001 — corrupt registry == no registry
        return []


def list_drafts() -> list[dict]:
    """Every registered draft, newest first."""
    return list(reversed(_read_registry()))


def register_draft(
    target: str, draft_dir: str, acceptance_rate: Optional[float] = None
) -> None:
    """Record ``target -> draft_dir`` so ``serve --auto-spec`` can find it.

    The target key is lower-cased to match ``spec_pairing.pick_draft_model``'s
    normalisation. Re-registering the same target replaces the old entry.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string")
    if not isinstance(draft_dir, str) or not draft_dir.strip():
        raise ValueError("draft_dir must be a non-empty string")
    if acceptance_rate is not None:
        classify_acceptance(acceptance_rate)  # validates bounds / finiteness

    key = target.strip().lower()
    entry = {
        "target": key,
        "draft": os.path.realpath(draft_dir),
        "acceptance_rate": (
            None if acceptance_rate is None else float(acceptance_rate)
        ),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries = [item for item in _read_registry() if item.get("target") != key]
    entries.append(entry)
    entries = entries[-_MAX_REGISTRY_ENTRIES:]

    _atomic_write_json({"drafts": entries}, draft_registry_path())


def lookup_draft(target: str) -> Optional[str]:
    """Locally-trained draft for ``target``, or None.

    An entry whose directory no longer exists (the user moved or deleted the
    draft) is skipped — a stale registry must degrade to "no draft", never to a
    crash inside ``soup serve``.
    """
    if not isinstance(target, str) or not target.strip():
        return None
    key = target.strip().lower()
    for entry in reversed(_read_registry()):
        if entry.get("target") != key:
            continue
        draft = entry.get("draft")
        if isinstance(draft, str) and os.path.isdir(draft):
            return draft
    return None


# ---------------------------------------------------------------------------
# Measurement (torch-lazy)
# ---------------------------------------------------------------------------
def measure_acceptance(
    target_model: "PreTrainedModel",
    draft_model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 64,
) -> tuple[int, int]:
    """Teacher-forced acceptance of ``draft_model`` against ``target_model``.

    For each prompt the target greedy-generates a continuation; the draft then
    forwards ONCE over the full sequence. Causal-LM alignment: ``logits[i]``
    predicts token ``i + 1``, so the draft's prediction for generated position
    ``p`` is read from ``logits[p - 1]``.

    Returns ``(accepted, total)`` summed over prompts.
    """
    import torch

    accepted = 0
    total = 0
    device = next(target_model.parameters()).device

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        prompt_len = int(input_ids.shape[1])

        gen_kwargs: dict = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # greedy: a sampled target makes alpha noisy
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        mask = encoded.get("attention_mask", None)
        if mask is not None:
            gen_kwargs["attention_mask"] = mask.to(device)

        with torch.no_grad():
            full_ids = target_model.generate(**gen_kwargs)

        generated = full_ids[0, prompt_len:]
        if generated.numel() == 0:
            continue

        draft_device = next(draft_model.parameters()).device
        with torch.no_grad():
            logits = draft_model(input_ids=full_ids.to(draft_device)).logits

        # logits[p - 1] predicts the token at position p. The first generated
        # token sits at index prompt_len, so its prediction is logits[prompt_len
        # - 1]; the last generated token needs no prediction beyond it, hence
        # the -1 upper bound.
        proposal_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
        proposals = proposal_logits.argmax(dim=-1).cpu().tolist()
        actual = generated.cpu().tolist()

        # The pure kernel does the comparison — one implementation, and the one
        # the off-by-one fixture pins.
        accepted += count_accepted(proposals, actual)
        total += len(actual)

    return accepted, total


def measure_throughput(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompts: Sequence[str],
    *,
    assistant_model: Optional["PreTrainedModel"] = None,
    num_assistant_tokens: int = 5,
    max_new_tokens: int = 64,
) -> float:
    """Wall-clock generation throughput (tokens/second), greedy decode.

    One warm-up generate is discarded (CUDA kernel autotuning / lazy module
    init), then the timed region is bracketed by ``cuda.synchronize()`` so the
    number is not measuring an unfinished async queue.
    """
    import torch

    if not prompts:
        return 0.0

    device = next(model.parameters()).device

    def _generate(prompt: str) -> int:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        kwargs: dict = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        mask = encoded.get("attention_mask", None)
        if mask is not None:
            kwargs["attention_mask"] = mask.to(device)
        if assistant_model is not None:
            kwargs["assistant_model"] = assistant_model
            kwargs["num_assistant_tokens"] = num_assistant_tokens
        with torch.no_grad():
            out = model.generate(**kwargs)
        return int(out.shape[1] - input_ids.shape[1])

    _generate(prompts[0])  # warm-up, discarded

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated = sum(_generate(prompt) for prompt in prompts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    return generated / elapsed
