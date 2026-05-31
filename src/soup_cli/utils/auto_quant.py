"""Auto-quant picker: try multiple quant formats, pick fastest-at-acceptable-quality.

Pure-Python decision engine. Actual model loading + eval is delegated to the
caller (v0.30.0 ships the picker + schema, trainer-side eval loop deferred
to v0.30.1 following the same pattern as v0.28.0 kernel_picker).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_DEFAULT_ORDER = ("gguf", "awq", "gptq", "fp8", "none")


def default_candidate_order() -> tuple[str, ...]:
    """Canonical order of quant formats to try.

    GGUF first because it's the widest-deployable; AWQ/GPTQ next for 4-bit
    quality; FP8 only on Hopper+; 'none' (baseline) last so we always have
    a fallback to prove any quant is actually helping.
    """
    return _DEFAULT_ORDER


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"candidate name must match {_VALID_NAME_RE.pattern}, got {name!r}"
        )


@dataclass(frozen=True)
class Candidate:
    """One candidate quant configuration + its measured score/latency."""

    name: str
    score: float  # quality in [0.0, 1.0]; higher is better
    latency_ms: float
    ok: bool  # False if eval crashed / threshold-invalid output

    def __post_init__(self) -> None:
        _validate_name(self.name)
        # ``bool`` is a subclass of ``int``; reject it explicitly so a caller
        # passing ``score=True`` doesn't sneak past as 1.0.
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or math.isnan(self.score)
        ):
            raise ValueError(f"score must be a finite float, got {self.score!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")
        if (
            not isinstance(self.latency_ms, (int, float))
            or isinstance(self.latency_ms, bool)
            or math.isnan(self.latency_ms)
        ):
            raise ValueError(f"latency_ms must be a finite float, got {self.latency_ms!r}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if not isinstance(self.ok, bool):
            raise ValueError(f"ok must be a bool, got {type(self.ok).__name__}")


def pick_best(
    candidates: Iterable[Candidate],
    *,
    min_score: float = 0.90,
) -> Candidate:
    """Pick the fastest candidate whose score >= min_score.

    Tie-break: first-encountered (stable, matches v0.28.0 kernel_picker).
    Raises ValueError if no candidate passes the threshold.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0.0, 1.0], got {min_score}")

    # Materialise so we can both filter and count from a generator input.
    all_candidates = list(candidates)
    pool = [c for c in all_candidates if c.ok and c.score >= min_score]
    if not pool:
        raise ValueError(
            f"no candidate passed min_score={min_score}; "
            f"ran {len(all_candidates)} candidates"
        )
    best = pool[0]
    for cand in pool[1:]:
        if cand.latency_ms < best.latency_ms:
            best = cand
    return best


def evaluate_candidate(
    name: str, *,
    eval_fn,
    prompts,
    min_correct_fraction: float = 0.5,
) -> Candidate:
    """Run ``eval_fn(prompt) -> (response, correct_bool)`` over a small prompt
    set, time it, and produce a Candidate.

    Latency is mean per-prompt ms. Score is the fraction correct. ``ok`` is
    True when score >= ``min_correct_fraction``.

    Robust to ``eval_fn`` crashes — any prompt that raises sets ``ok=False``
    and continues so a single bad prompt doesn't disqualify a candidate that
    works on the rest.
    """
    import time as _time

    if not prompts:
        raise ValueError("evaluate_candidate requires at least one prompt")

    correct = 0
    total = 0
    completed = 0  # for honest latency mean (excludes crashed prompts)
    started = _time.perf_counter()
    crashed = False
    for prompt in prompts:
        total += 1
        try:
            _resp, hit = eval_fn(prompt)
        except Exception:  # noqa: BLE001 — surface as eval failure
            crashed = True
            continue
        completed += 1
        if hit:
            correct += 1
    elapsed_ms = (_time.perf_counter() - started) * 1000.0 / max(1, completed)
    score = correct / total
    return Candidate(
        name=name,
        score=score,
        latency_ms=elapsed_ms,
        ok=(not crashed) and (score >= min_correct_fraction),
    )


def quant_name_to_vllm_kwargs(name: str) -> dict[str, str]:
    """v0.35.0 #61 — translate a picker candidate name into vLLM engine kwargs.

    Returns a dict suitable to splat into ``AsyncLLMEngine.from_engine_args``.
    Unknown / pass-through names return ``{}`` so the engine uses its default
    (i.e. baseline / no quantization).
    """
    _validate_name(name)
    mapping: dict[str, dict[str, str]] = {
        "awq": {"quantization": "awq"},
        "gptq": {"quantization": "gptq"},
        "fp8": {"quantization": "fp8"},
        # GGUF needs a path swap, not a kwarg — caller must handle separately.
        "gguf": {},
        "none": {},
    }
    return dict(mapping.get(name, {}))


def quant_name_to_bnb_kwargs(name: str) -> dict[str, bool]:
    """v0.35.0 #61 — translate a picker name into transformers BnB kwargs.

    Returns ``{"load_in_4bit": True}`` for awq/gptq (treated as 4-bit-class),
    and ``{}`` for fp8 / gguf / none / unknown — those formats are not
    expressible via BitsAndBytesConfig and are handled either by the engine
    directly (fp8) or via a path swap (gguf). Caller wraps these into a
    ``BitsAndBytesConfig``.
    """
    _validate_name(name)
    if name in ("awq", "gptq"):
        return {"load_in_4bit": True}
    return {}


def free_engine(engine: Any) -> None:
    """v0.35.0 #61 — best-effort engine release before re-instantiation.

    .. important::
       The caller is responsible for releasing their own reference *before*
       calling this helper — Python's ``del`` on a function parameter only
       removes the local binding, it cannot drop the caller's reference.
       Idiomatic use::

           free_engine(engine)
           engine = None  # caller drops their reference too

       This function exists primarily to invoke ``torch.cuda.empty_cache()``
       after the caller has already nulled out their reference. The
       parameter is accepted only so the call-site reads naturally.

    Failure to invoke ``empty_cache`` is a soft advisory — caller should
    NOT block on it.
    """
    del engine  # local binding only — see docstring caveat
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass


def try_reload_with_fallback(
    *,
    picked: "Candidate",
    all_candidates: list["Candidate"],
    build_fn: Callable[[str], Any],
) -> tuple["Candidate", Any]:
    """v0.35.0 #61 — attempt ``build_fn(picked.name)``, falling back to the
    next-highest-scored candidate on load failure.

    ``build_fn`` is a callable ``name -> engine`` that may raise on any
    backend-side load error (AWQ kernel missing, GPTQ checkpoint absent, etc).
    Returns ``(candidate_actually_used, engine)``. Raises ``RuntimeError``
    only when every candidate fails to load — at that point the server cannot
    bind and must abort.
    """
    # Build the fallback queue: picked first, then remaining candidates by
    # descending score (matches run_auto_quant_picker's soft-fallback policy).
    seen = {picked.name}
    queue: list[Candidate] = [picked]
    by_score = sorted(all_candidates, key=lambda c: -c.score)
    for cand in by_score:
        if cand.name not in seen:
            queue.append(cand)
            seen.add(cand.name)

    last_error: Exception | None = None
    for cand in queue:
        try:
            engine = build_fn(cand.name)
        except Exception as exc:  # noqa: BLE001 — caller's build is best-effort
            last_error = exc
            continue
        return cand, engine

    # Redact: type+message rather than repr — repr can include filesystem
    # paths from inside the backend's load chain (matches v0.34.0 crash.py
    # secret-redaction policy).
    last_summary = (
        f"{type(last_error).__name__}: {last_error}" if last_error else "<none>"
    )
    raise RuntimeError(
        f"every auto-quant candidate failed to load (tried {len(queue)}); "
        f"last error: {last_summary}"
    )


def run_auto_quant_picker(
    *, candidate_specs, prompts, min_score: float = 0.90,
) -> Candidate:
    """Run the full pick: evaluate each candidate, pick best by score+latency.

    ``candidate_specs`` is a sequence of ``(name, eval_fn)`` pairs. Each
    ``eval_fn`` takes a prompt and returns ``(response, correct_bool)``.

    Falls back to the highest-scoring candidate (regardless of threshold)
    when no candidate passes ``min_score``, so the server can still bind a
    port. The caller is expected to log the choice.
    """
    candidates = [
        evaluate_candidate(name, eval_fn=fn, prompts=prompts)
        for name, fn in candidate_specs
    ]
    try:
        return pick_best(candidates, min_score=min_score)
    except ValueError:
        # Soft fallback: pick the highest-scored candidate so the server
        # still has a valid choice. Documented as advisory in serve.py.
        ok_candidates = [c for c in candidates if c.ok]
        pool = ok_candidates or candidates
        return max(pool, key=lambda c: (c.score, -c.latency_ms))
