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
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from rich.panel import Panel
from rich.table import Table

from soup_cli.utils.terminal import for_terminal

if TYPE_CHECKING:  # pragma: no cover — typing only; torch stays lazy at runtime
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Verdict bands on the acceptance rate alone. They do NOT say whether a pair pays:
# that depends on how much faster the draft is than its target (#843). The one pair
# measured at scale sat at 0.813 -- STRONG -- and ran at 0.481x of plain. See
# :func:`breakeven_acceptance` for the rate a given draft/target pair needs.
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


def count_accepted_spans(
    draft_pieces: Sequence[str], target_pieces: Sequence[str]
) -> int:
    """Number of target token positions accepted by draft proposals across tokenizers.

    Unlike :func:`count_accepted` which requires identical token IDs and identical
    tokenization boundaries, this operates on decoded text pieces. It aligns the
    character spans of the draft's proposed tokens against the target's tokens using
    exact span overlap when text matches, or :class:`difflib.SequenceMatcher` when
    decoded texts differ.

    A target token position is accepted if its full character span in the target
    sequence is matched by the draft's proposals.
    """
    import difflib

    from soup_cli.utils.uld import _MAX_ALIGN_CHARS, _char_spans

    if not target_pieces:
        return 0
    if not draft_pieces:
        return 0

    d_text, d_spans = _char_spans(draft_pieces)
    t_text, t_spans = _char_spans(target_pieces)

    if d_text == t_text:
        return len(target_pieces)

    d_text = d_text[:_MAX_ALIGN_CHARS]
    t_text = t_text[:_MAX_ALIGN_CHARS]

    matcher = difflib.SequenceMatcher(None, d_text, t_text, autojunk=False)
    matching_blocks = [b for b in matcher.get_matching_blocks() if b[2] > 0]
    if not matching_blocks:
        return 0

    accepted = 0
    for s, e in t_spans:
        if s == e:
            if any(t_start <= s <= t_start + size for _, t_start, size in matching_blocks):
                accepted += 1
        else:
            if any(t_start <= s and e <= t_start + size for _, t_start, size in matching_blocks):
                accepted += 1
    return accepted


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


def compute_acceptance_spans(
    draft_pieces: Sequence[str], target_pieces: Sequence[str]
) -> float:
    """Acceptance rate for a single sequence across tokenizers."""
    if not target_pieces:
        return 0.0
    matched = count_accepted_spans(draft_pieces, target_pieces)
    return matched / len(target_pieces)


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


# ---------------------------------------------------------------------------
# Break-even model (#843) -- torch-free
# ---------------------------------------------------------------------------
#: The draft lengths the CLI accepts (``--num-assistant-tokens``, ``--sweep-k``).
DRAFT_K_MIN = 1
DRAFT_K_MAX = 64

#: Printed beside every modelled number, so it is never read as a measurement.
MODEL_ASSUMPTIONS = "modelled; i.i.d. acceptance; excludes framework overhead"


def _check_acceptance(a: object) -> float:
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        raise TypeError(f"acceptance must be a number, got {a!r}")
    value = float(a)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"acceptance must be between 0 and 1, got {a!r}")
    return value


def _check_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {k!r}")
    if not DRAFT_K_MIN <= k <= DRAFT_K_MAX:
        raise ValueError(f"k must be in {DRAFT_K_MIN}..{DRAFT_K_MAX}, got {k}")
    return k


def _check_ratio(c: object) -> float:
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise TypeError(f"latency ratio must be a number, got {c!r}")
    value = float(c)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"latency ratio must be positive and finite, got {c!r}")
    return value


def latency_ratio(tok_s_plain: Optional[float], tok_s_draft: Optional[float]) -> Optional[float]:
    """``c``: the draft's per-token cost in target steps (plain tok/s / draft tok/s).

    ``None`` when either throughput is missing, non-positive or non-finite -- an
    unmeasured arm, never a ratio of 0 or infinity.
    """
    values = []
    for tok_s in (tok_s_plain, tok_s_draft):
        if isinstance(tok_s, bool) or not isinstance(tok_s, (int, float)):
            return None
        if not math.isfinite(tok_s) or tok_s <= 0:
            return None
        values.append(float(tok_s))
    return values[0] / values[1]


def expected_tokens_per_step(a: float, k: int) -> float:
    """Tokens one assisted step yields: ``E(a, k) = (1 - a^(k+1)) / (1 - a)``.

    The standard model with per-position acceptance ``a`` independent across
    positions: the target always contributes one token, plus each accepted draft
    token up to ``k``. ``k + 1`` at ``a = 1``, where the closed form divides by zero.
    """
    a = _check_acceptance(a)
    k = _check_k(k)
    if a == 1.0:
        return float(k + 1)
    return (1.0 - a ** (k + 1)) / (1.0 - a)


def modelled_speedup(a: float, k: int, c: float) -> float:
    """``S = E(a, k) / (k*c + 1)``: tokens per target-step-equivalent of cost.

    A CEILING, not a prediction. It assumes the verification pass over ``k + 1``
    tokens costs one target decode step (batch 1) and charges nothing for framework
    overhead, which on the one pair measured at scale halved the result again
    (modelled 0.955x, measured 0.481x, #303).
    """
    return expected_tokens_per_step(a, k) / (_check_k(k) * _check_ratio(c) + 1.0)


def breakeven_acceptance(k: int, c: float) -> Optional[float]:
    """The acceptance at which ``S = 1`` for draft length ``k``, or ``None``.

    ``None`` means no acceptance rate pays: even a perfect draft gives
    ``(k+1) / (k*c + 1) <= 1``, which is every ``k`` once ``c >= 1`` (a draft no
    faster than its target). ``S`` rises monotonically in ``a``, so the crossing is
    found by bisection.
    """
    k = _check_k(k)
    c = _check_ratio(c)
    if modelled_speedup(1.0, k, c) <= 1.0:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if modelled_speedup(mid, k, c) < 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def modelled_best_k(a: float, c: float) -> tuple[int, float]:
    """The draft length in the CLI's range that maximises :func:`modelled_speedup`.

    Returned with its speedup, which may be below 1: the best k of a pair that
    never pays is still reported, and the caller says it does not pay.
    """
    a = _check_acceptance(a)
    c = _check_ratio(c)
    best = max(range(DRAFT_K_MIN, DRAFT_K_MAX + 1), key=lambda k: modelled_speedup(a, k, c))
    return best, modelled_speedup(a, best, c)


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


def supports_universal_assisted_decoding() -> bool:
    """True if installed transformers supports cross-tokenizer assisted decoding (UAD)."""
    try:
        from transformers.generation import candidate_generator

        return hasattr(
            candidate_generator, "AssistedCandidateGeneratorDifferentTokenizers"
        )
    except Exception:
        return False


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
    #: Outcome of the best-effort assisted-throughput arm (#344 review): one of
    #: "pending" (arm not reached), "complete" (a positive tok/s was measured),
    #: "untimed" (arm returned no usable number), "crash" (arm raised), or
    #: "interrupted" (Ctrl-C). Without it the crash / untimed / interrupt reports
    #: are byte-identical on disk, so a failed arm is indistinguishable from an
    #: un-run one.
    assisted_status: str = "pending"
    #: #843. The draft decoding alone, and the break-even model it feeds. All are
    #: modelled (:data:`MODEL_ASSUMPTIONS`) except ``tok_s_draft``. ``draft_status``
    #: follows ``assisted_status``'s vocabulary. ``breakeven_acceptance`` is ``None``
    #: both when unmeasured and when no rate pays; ``latency_ratio`` tells them apart.
    tok_s_draft: Optional[float] = None
    draft_status: str = "pending"
    latency_ratio: Optional[float] = None
    breakeven_acceptance: Optional[float] = None
    modelled_best_k: Optional[int] = None
    modelled_speedup_best_k: Optional[float] = None
    #: ``--sweep-k``: one ``{"k", "tok_s_assisted", "speedup", "status"}`` per k.
    k_sweep: Optional[tuple] = None
    measured_best_k: Optional[int] = None


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
    table.add_row("Target", for_terminal(report.target))
    table.add_row("Draft", for_terminal(report.draft))
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

    if report.tok_s_draft is not None:
        table.add_row("Draft alone", f"{_fmt(report.tok_s_draft, ' tok/s')}")
    if report.latency_ratio is not None:
        table.add_row(
            "Latency ratio", f"{report.latency_ratio:.3f} (plain / draft-alone tok/s)"
        )
        if report.breakeven_acceptance is None:
            table.add_row(
                "Break-even",
                "[red]none -- no acceptance rate pays: the draft is not fast enough "
                "relative to the target[/]",
            )
        else:
            table.add_row(
                "Break-even",
                f"{report.breakeven_acceptance * 100:.1f}% acceptance at "
                f"k={report.num_assistant_tokens}",
            )
        if report.modelled_best_k is not None and report.modelled_speedup_best_k is not None:
            if report.modelled_speedup_best_k > 1.0:
                table.add_row(
                    "Best k",
                    f"k={report.modelled_best_k} -> {report.modelled_speedup_best_k:.2f}x "
                    "at the measured acceptance",
                )
            else:
                # A "best" k that is still a slowdown is not a recommendation.
                table.add_row(
                    "Best k",
                    "[red]no k pays at the measured acceptance[/] (closest: "
                    # Three decimals: at 0.995x, "1.00x" beside "no k pays" reads as
                    # a contradiction.
                    f"k={report.modelled_best_k} -> {report.modelled_speedup_best_k:.3f}x)",
                )
        table.add_row("", f"[dim]({MODEL_ASSUMPTIONS}; a ceiling, not a prediction)[/]")
    if report.k_sweep:
        for row in report.k_sweep:
            measured = (
                f"{_fmt(row.get('tok_s_assisted'), ' tok/s')} ({_fmt(row.get('speedup'), 'x')})"
                if row.get("status") == "complete"
                else row.get("status", "n/a")
            )
            table.add_row(f"Sweep k={row['k']}", measured)
        best = "n/a" if report.measured_best_k is None else f"k={report.measured_best_k}"
        table.add_row("Measured best k", best)

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


# Serialises threads INSIDE this process. The OS file lock below is per-process
# on both Windows (msvcrt) and POSIX (flock), so it does NOT serialise two
# threads of the same interpreter — both locks are needed.
_REGISTRY_THREAD_LOCK = threading.Lock()


@contextmanager
def _registry_lock():
    """Best-effort exclusive lock on the registry, in-process AND cross-process.

    The registry is a read-modify-write file (replace the entry for one target,
    keep the rest), so two ``soup draft distill`` runs finishing at the same
    time could otherwise lose one registration entirely — the second writer's
    snapshot predates the first writer's commit.

    Cross-process: a sidecar ``<registry>.lock`` file, mirroring
    ``utils/advise_history._append_with_lock`` (a separate file, so the lock
    never depends on the data file's seek position). If OS locking is
    unavailable on the host, the update proceeds unlocked — degraded, not
    broken.
    """
    with _REGISTRY_THREAD_LOCK:
        lock_path = draft_registry_path() + ".lock"
        handle = None
        try:
            os.makedirs(
                os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True
            )
            # O_NOFOLLOW so a pre-planted symlink at <registry>.lock can't
            # redirect the lock (defence-in-depth — nothing is written to it).
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_path, flags, 0o600)
            handle = os.fdopen(fd, "a+")
        except OSError:
            handle = None

        locked = False
        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt  # type: ignore[import-not-found]

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl  # type: ignore[import-not-found]

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                locked = True
            except (ImportError, OSError):
                locked = False
        try:
            yield
        finally:
            if handle is not None:
                try:
                    if locked and os.name == "nt":
                        import msvcrt  # type: ignore[import-not-found]

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    elif locked:
                        import fcntl  # type: ignore[import-not-found]

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
                handle.close()


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
        # O_NOFOLLOW: this runs on every `soup serve` startup, so a symlink
        # planted at ~/.soup/drafts.json must not be transparently followed.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            if os.fstat(handle.fileno()).st_size > _MAX_REGISTRY_BYTES:
                return []
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
    # Read-modify-write under a cross-process lock: two concurrent distill runs
    # must not lose one another's registration.
    with _registry_lock():
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
    draft_tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> tuple[int, int]:
    """Teacher-forced acceptance of ``draft_model`` against ``target_model``.

    For each prompt the target greedy-generates a continuation; the draft then
    forwards ONCE over the full sequence. Causal-LM alignment: ``logits[i]``
    predicts token ``i + 1``, so the draft's prediction for generated position
    ``p`` is read from ``logits[p - 1]``.

    When ``draft_tokenizer`` is provided and differs from ``tokenizer``, acceptance
    is measured using decoded character span alignment (:func:`count_accepted_spans`),
    allowing speculative evaluation across different vocabularies and tokenization
    boundaries.

    Note: Cross-tokenizer acceptance is a **lower bound**. A boundary merge (where
    the draft tokenizer merges the last prompt character with the first generated
    character) drops the straddling token, shortening the score by up to ``1/n_gen``.
    The bias is always downward.

    Returns ``(accepted, total)`` summed over prompts.
    """
    import torch

    accepted = 0
    total = 0
    device = next(target_model.parameters()).device
    draft_device = next(draft_model.parameters()).device

    is_same = draft_tokenizer is None or same_tokenizer(tokenizer, draft_tokenizer)

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        prompt_len = int(input_ids.shape[1])

        gen_kwargs: dict = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # greedy: a sampled target makes alpha noisy
            "repetition_penalty": 1.0,  # neutralise checkpoint penalty: raw argmax agreement
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

        actual = generated.cpu().tolist()

        if is_same:
            with torch.no_grad():
                logits = draft_model(input_ids=full_ids.to(draft_device)).logits

            # logits[p - 1] predicts the token at position p. The first generated
            # token sits at index prompt_len, so its prediction is logits[prompt_len
            # - 1]; the last generated token needs no prediction beyond it, hence
            # the -1 upper bound.
            proposal_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
            proposals = proposal_logits.argmax(dim=-1).cpu().tolist()

            # The pure kernel does the comparison — one implementation, and the one
            # the off-by-one fixture pins.
            accepted += count_accepted(proposals, actual)
            total += len(actual)
        else:
            assert draft_tokenizer is not None
            target_pieces = [
                tokenizer.decode([tid], skip_special_tokens=False) for tid in actual
            ]

            prompt_draft_enc = draft_tokenizer(prompt, return_tensors="pt")
            draft_prompt_len = int(prompt_draft_enc["input_ids"].shape[1])

            full_ids_list = full_ids[0].cpu().tolist()
            full_text = tokenizer.decode(full_ids_list, skip_special_tokens=False)
            draft_full_enc = draft_tokenizer(full_text, return_tensors="pt")
            draft_full_ids = draft_full_enc["input_ids"].to(draft_device)

            if draft_full_ids.shape[1] <= draft_prompt_len:
                continue

            total += len(target_pieces)

            with torch.no_grad():
                draft_logits = draft_model(input_ids=draft_full_ids).logits

            draft_proposal_logits = draft_logits[
                0, draft_prompt_len - 1 : draft_full_ids.shape[1] - 1, :
            ]
            draft_proposals = draft_proposal_logits.argmax(dim=-1).cpu().tolist()

            draft_pieces = [
                draft_tokenizer.decode([did], skip_special_tokens=False)
                for did in draft_proposals
            ]

            accepted += count_accepted_spans(draft_pieces, target_pieces)

    return accepted, total


def measure_throughput(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompts: Sequence[str],
    *,
    assistant_model: Optional["PreTrainedModel"] = None,
    assistant_tokenizer: Optional["PreTrainedTokenizerBase"] = None,
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
            if assistant_tokenizer is not None and not same_tokenizer(
                tokenizer, assistant_tokenizer
            ):
                if not supports_universal_assisted_decoding():
                    raise RuntimeError(
                        "Universal Assisted Decoding (cross-tokenizer speculative decoding) "
                        "requires transformers with cross-tokenizer support. "
                        "Please upgrade transformers (pip install --upgrade transformers)."
                    )
                kwargs["tokenizer"] = tokenizer
                kwargs["assistant_tokenizer"] = assistant_tokenizer
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
