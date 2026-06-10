"""v0.53.1 #109 — soup deploy autopilot --measure helper.

Live Quant-Lobotomy measurement for each candidate quant in a deploy profile.
Wraps v0.26.0 :mod:`soup_cli.eval.quant_check` with disk-cache so that
repeated invocations on the same (base, profile, eval-tasks) tuple short-
circuit. Soft-fallback policy: when no candidate clears ``OK``, the
highest-delta candidate (least negative drop) is selected.

The actual model loading + generation is the caller's responsibility — this
module is pure-Python and takes opaque ``Callable[[str], str]`` generators
so it stays testable without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from rich.table import Table

# Mirrors v0.26.0 Part D thresholds (verdict OK = drop < 2%, MINOR = drop <
# 5%, MAJOR otherwise). Exposed for tests + consistency.
DEFAULT_MINOR_THRESHOLD: float = 0.02
DEFAULT_MAJOR_THRESHOLD: float = 0.05

_MAX_CACHE_BYTES: int = 1 * 1024 * 1024  # 1 MB cap on cache file
_MAX_CANDIDATES: int = 32
_MAX_BASE_LEN: int = 512  # mirrors v0.40.5 reward_model / v0.62.0 --base policy

# v0.71.22 #143 — closed candidate allowlist for the first-party generator
# factories. Mirrors the TrainingConfig.quantization Quant Menu surface
# ("none" = the unquantized baseline; HQQ uses the hqq:Nbit shape).
MEASURABLE_QUANT_CANDIDATES: frozenset[str] = frozenset(
    {"none", "4bit", "8bit", "gptq", "awq", "aqlm", "eetq", "mxfp4", "fp8"}
)
_HQQ_CANDIDATE_RE = re.compile(r"^hqq:[12348]bit$")

# Test + advanced-operator escape hatch: when set on this module, the deploy
# CLI uses these callables INSTEAD of the first-party transformers factories
# below (v0.71.22 #143 lifted the v0.53.1 placeholder deferral — the live
# loaders are ``build_before_generator`` / ``build_after_generator_factory``).
# NOT a public API; kept as the test seam.
_DEPLOY_MEASURE_BEFORE_GEN: Optional[Callable[[str], str]] = None
_DEPLOY_MEASURE_AFTER_FACTORY: Optional[
    Callable[[str], Callable[[str], str]]
] = None


@dataclass(frozen=True)
class MeasureResult:
    """Outcome of a single candidate measurement."""

    candidate: str   # e.g. "4bit" / "gptq" / "awq"
    before: float    # baseline (unquantized) score in [0, 1]
    after: float     # quantized score
    delta: float     # after - before (negative = worse)
    verdict: str     # "OK" | "MINOR" | "MAJOR"


def compute_cache_key(*, base_sha: str, profile_name: str, tasks_sha: str) -> str:
    """Build a deterministic cache key from the input tuple.

    Callers should pass FULL SHA-256 hex strings (64 chars) for the two
    digest inputs. The orchestrator in :mod:`soup_cli.commands.deploy`
    currently truncates ``base_sha`` to 16 hex chars at the call site —
    that 16-char truncation is acceptable for caching purposes (collision
    probability ≈ 1 in 2³² across ~4 billion cache entries) but is the
    operator-facing policy, not a property enforced here. Future callers
    that truncate further should expect collisions in proportion.
    """
    for name, value in (
        ("base_sha", base_sha),
        ("profile_name", profile_name),
        ("tasks_sha", tasks_sha),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be str, got {type(value).__name__}"
            )
        if not value:
            raise ValueError(f"{name} must be non-empty")
        if "\x00" in value:
            raise ValueError(f"{name} must not contain null bytes")
    hasher = hashlib.sha256()
    hasher.update(base_sha.encode("utf-8"))
    hasher.update(b"\x1f")
    hasher.update(profile_name.encode("utf-8"))
    hasher.update(b"\x1f")
    hasher.update(tasks_sha.encode("utf-8"))
    return hasher.hexdigest()[:32]


def sha_of_file(path: str) -> str:
    """SHA-256 of a file, used to fingerprint the eval tasks JSONL."""
    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {os.path.basename(path)!r}")
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _default_cache_path() -> str:
    """Return ``~/.soup/deploy_autopilot_cache.json`` (allowed override via env)."""
    override = os.environ.get("SOUP_DEPLOY_AUTOPILOT_CACHE")
    if override:
        # Reject null-bytes + control chars before doing any path resolution.
        if "\x00" in override or any(ord(ch) < 32 for ch in override):
            override = None
    if override:
        # Honor override only if it's plausibly safe (under home / cwd / temp)
        candidate = os.path.realpath(override)
        for safe_root in (
            os.path.realpath(os.path.expanduser("~")),
            os.path.realpath(os.getcwd()),
            os.path.realpath(tempfile.gettempdir()),
        ):
            try:
                if (
                    os.path.commonpath([candidate, safe_root]) == safe_root
                ):
                    return candidate
            except ValueError:
                continue
        # Fall through to default if override is unsafe
    return os.path.join(
        os.path.expanduser("~"), ".soup", "deploy_autopilot_cache.json"
    )


def load_cache(path: Optional[str] = None) -> dict:
    """Read the cache file. Returns ``{}`` if missing or malformed."""
    cache_path = path or _default_cache_path()
    if not isinstance(cache_path, str):
        return {}
    if not os.path.isfile(cache_path):
        return {}
    # TOCTOU defence: reject a symlink at the cache target before open()
    # (mirrors the existing guard in ``save_cache``).
    try:
        st = os.lstat(cache_path)
    except OSError:
        return {}
    if stat.S_ISLNK(st.st_mode):
        return {}
    if st.st_size > _MAX_CACHE_BYTES:
        return {}
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_cache(cache: dict, path: Optional[str] = None) -> None:
    """Atomically write the cache to disk. Best-effort: silent on failure."""
    cache_path = path or _default_cache_path()
    if not isinstance(cache_path, str):
        return
    dir_part = os.path.dirname(cache_path)
    if dir_part:
        try:
            os.makedirs(dir_part, exist_ok=True)
        except OSError:
            return
    # Reject symlink at target (TOCTOU defence)
    if os.path.lexists(cache_path):
        try:
            st = os.lstat(cache_path)
        except OSError:
            return
        if stat.S_ISLNK(st.st_mode):
            return
    serialised = json.dumps(cache, sort_keys=True, indent=2)
    if len(serialised) > _MAX_CACHE_BYTES:
        return
    tmp_fd = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".deploy_autopilot_cache_", suffix=".tmp",
            dir=dir_part or None,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(serialised)
            tmp_fd = None
            os.replace(tmp_name, cache_path)
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
        # Best-effort 0o600 on POSIX
        try:
            os.chmod(cache_path, 0o600)
        except OSError:
            pass
    except OSError:
        return


def _score_tasks(
    tasks_file: str, generate_fn: Callable[[str], str],
) -> float:
    """Average score across a JSONL task file (delegates to v0.25.0 eval)."""
    from soup_cli.eval.custom import load_eval_tasks, score_task

    tasks = load_eval_tasks(tasks_file)
    if not tasks:
        return 0.0
    total = 0.0
    for task in tasks:
        output = generate_fn(task.prompt)
        total += float(score_task(task, output).score)
    return total / len(tasks)


def measure_candidate(
    *,
    candidate: str,
    tasks_file: str,
    before_gen: Optional[Callable[[str], str]] = None,
    after_gen: Callable[[str], str],
    before_score: Optional[float] = None,
) -> MeasureResult:
    """Score one quant candidate against the baseline.

    Returns a :class:`MeasureResult` with classify_delta verdict.

    The baseline is deterministic across candidates, so the orchestrator
    (:func:`run_measure`) computes it ONCE and threads the float in via
    ``before_score`` — this avoids re-running full greedy generation over
    every task once per candidate (N× redundant GPU time) and keeps the
    baseline model out of VRAM while a quantized candidate is loaded.
    Direct / back-compat callers may instead pass ``before_gen`` and the
    baseline is scored here. Exactly one of the two must be supplied.
    """
    if isinstance(candidate, bool) or not isinstance(candidate, str):
        raise TypeError("candidate must be a non-bool str")
    if not candidate:
        raise ValueError("candidate must be non-empty")
    if "\x00" in candidate:
        raise ValueError("candidate must not contain null bytes")

    if before_score is not None:
        if isinstance(before_score, bool) or not isinstance(
            before_score, (int, float)
        ):
            raise TypeError("before_score must be a non-bool number")
        before = float(before_score)
    elif before_gen is not None:
        before = _score_tasks(tasks_file, before_gen)
    else:
        raise ValueError(
            "measure_candidate requires either before_gen or before_score"
        )
    after = _score_tasks(tasks_file, after_gen)
    delta = after - before
    if delta >= 0:
        verdict = "OK"
    else:
        drop = -delta
        if drop < DEFAULT_MINOR_THRESHOLD:
            verdict = "OK"
        elif drop < DEFAULT_MAJOR_THRESHOLD:
            verdict = "MINOR"
        else:
            verdict = "MAJOR"
    return MeasureResult(
        candidate=candidate, before=before, after=after,
        delta=delta, verdict=verdict,
    )


def pick_best(results: Sequence[MeasureResult]) -> Optional[MeasureResult]:
    """Soft-fallback policy from v0.33.0 #54.

    Returns the first ``OK`` candidate by insertion order; if none clears OK,
    returns the candidate with the highest ``delta`` (smallest drop relative
    to its baseline — the v0.33.0 #54 design intent). Returns ``None`` only
    for an empty sequence.
    """
    if not results:
        return None
    for r in results:
        if r.verdict == "OK":
            return r
    return max(results, key=lambda r: r.delta)


# --- v0.71.22 #143 — first-party transformers generator factories -----------


def validate_measure_candidate(candidate: object) -> str:
    """Validate + canonicalise a quant candidate for the live measure loop.

    Accepts the closed :data:`MEASURABLE_QUANT_CANDIDATES` allowlist plus the
    ``hqq:{1,2,3,4,8}bit`` shape. Case-insensitive; returns the lowercase
    canonical form. Bool rejected before the str check (project policy).
    """
    if isinstance(candidate, bool):
        raise TypeError("candidate must not be bool")
    if not isinstance(candidate, str):
        raise TypeError(f"candidate must be str, got {type(candidate).__name__}")
    if not candidate:
        raise ValueError("candidate must be non-empty")
    if "\x00" in candidate:
        raise ValueError("candidate must not contain null bytes")
    canonical = candidate.lower()
    if canonical in MEASURABLE_QUANT_CANDIDATES or _HQQ_CANDIDATE_RE.match(
        canonical
    ):
        return canonical
    supported = ", ".join(sorted(MEASURABLE_QUANT_CANDIDATES))
    # Truncate the echoed candidate (mirrors longlora._truncate_for_message
    # policy) so a pathologically long value can't bloat the error message.
    shown = candidate if len(candidate) <= 64 else candidate[:64] + "…"
    raise ValueError(
        f"candidate {shown!r} is not a measurable quant. Supported: "
        f"{supported}, hqq:Nbit (N in 1/2/3/4/8)"
    )


def _check_base_id(base: object) -> str:
    """Shape-validate a base model id / path for the factory builders."""
    # NOTE: this raises ValueError (not TypeError) for bool/non-str wrong-type
    # input — DELIBERATELY inconsistent with validate_measure_candidate (which
    # raises TypeError for the same class). The ValueError behaviour is pinned
    # by tests (test_v07122.test_builders_validate_max_new_tokens); changing it
    # would break them, so we keep it and document the seam here.
    if isinstance(base, bool) or not isinstance(base, str):
        raise ValueError("base must be a non-bool string")
    if not base:
        raise ValueError("base must be non-empty")
    if "\x00" in base:
        raise ValueError("base must not contain null bytes")
    if len(base) > _MAX_BASE_LEN:
        raise ValueError(f"base exceeds {_MAX_BASE_LEN} chars")
    return base


def _check_max_new_tokens(value: object) -> int:
    # NOTE: raises ValueError (not TypeError) for bool/non-int wrong-type input
    # — deliberately inconsistent with validate_measure_candidate's TypeError;
    # the ValueError behaviour is pinned by tests (see _check_base_id note).
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_new_tokens must be a non-bool int")
    if value < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {value}")
    return value


def _free_accelerator_memory() -> None:
    """Best-effort GC + CUDA cache flush between candidate model loads."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — torch missing / CUDA error: best-effort
        pass


def _import_transformers():
    """Lazy transformers import — module-level seam for tests."""
    import transformers

    return transformers


def _load_measure_model(
    base: str,
    *,
    quantization: str,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load ``base`` with the candidate quant via the Quant Menu loader.

    Returns a ``(model, tokenizer, device)`` triple shaped for
    :func:`soup_cli.utils.live_eval.make_generator`'s ``loaded=`` kwarg.
    ``quantization='none'`` loads the plain baseline. Heavy imports are lazy.
    """
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.live_eval import resolve_device
    from soup_cli.utils.quant_menu import build_quantization_config_for_loader

    transformers = _import_transformers()
    dev = resolve_device(device)
    tcfg = TrainingConfig(quantization=quantization)
    quant_config = build_quantization_config_for_loader(
        tcfg=tcfg, base=base, console=None
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base, trust_remote_code=trust_remote_code
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quant_config is not None:
        # Quantized loads place weights themselves (bnb/gptq/awq need
        # device_map on CUDA; a post-hoc .to() would break bnb layouts).
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant_config,
            device_map="auto" if dev == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base, trust_remote_code=trust_remote_code
        ).to(dev)
    model.eval()
    return model, tokenizer, dev


def _lazy_generator(
    base: str,
    *,
    quantization: str,
    device: Optional[str],
    max_new_tokens: int,
    trust_remote_code: bool,
) -> Callable[[str], str]:
    """Closure that loads the model on FIRST call (cache hits never load)."""
    holder: dict[str, Callable[[str], str]] = {}

    def _gen(prompt: str) -> str:
        if "fn" not in holder:
            _free_accelerator_memory()
            loaded = _load_measure_model(
                base,
                quantization=quantization,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            from soup_cli.utils import live_eval

            holder["fn"] = live_eval.make_generator(
                base, loaded=loaded, max_new_tokens=max_new_tokens
            )
        return holder["fn"](prompt)

    return _gen


def build_before_generator(
    base: str,
    *,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    trust_remote_code: bool = False,
) -> Callable[[str], str]:
    """First-party baseline generator for ``deploy autopilot --measure``.

    Lazy-loads the UNQUANTIZED ``base`` (greedy decode via
    :func:`soup_cli.utils.live_eval.make_generator`) on the first prompt —
    a cache-hit measure run never touches the model. (v0.71.22 #143)
    """
    _check_base_id(base)
    _check_max_new_tokens(max_new_tokens)
    return _lazy_generator(
        base,
        quantization="none",
        device=device,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


def build_after_generator_factory(
    base: str,
    *,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    trust_remote_code: bool = False,
) -> Callable[[str], Callable[[str], str]]:
    """First-party per-candidate quantized generator factory (#143).

    ``factory(candidate)`` validates the candidate eagerly (fail-fast at the
    measure loop boundary) and returns a generator that lazy-loads ``base``
    with that candidate's quant config from the Quant Menu loader on first
    use. CUDA memory from the previous candidate is freed before each load.
    """
    _check_base_id(base)
    _check_max_new_tokens(max_new_tokens)

    def factory(candidate: str) -> Callable[[str], str]:
        canonical = validate_measure_candidate(candidate)
        return _lazy_generator(
            base,
            quantization=canonical,
            device=device,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )

    # Marker so run_measure can pre-validate the whole candidate list up front
    # for the first-party path (M2) without touching injected test seams.
    factory._soup_first_party_measure_factory = True  # type: ignore[attr-defined]
    return factory


def run_measure(
    *,
    profile_name: str,
    base_sha: str,
    candidates: Sequence[str],
    tasks_file: str,
    before_gen: Callable[[str], str],
    after_gen_factory: Callable[[str], Callable[[str], str]],
    cache_path: Optional[str] = None,
) -> tuple[list[MeasureResult], bool]:
    """Run measurement across every candidate quant for a deploy profile.

    ``after_gen_factory(candidate)`` returns a fresh generator for that
    candidate (so the caller can lazy-load the quantized model per call).

    Returns ``(results, cache_hit)``. On a hit, ``results`` is loaded from
    disk and no eval is run.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of strings")
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    if len(candidates) > _MAX_CANDIDATES:
        raise ValueError(
            f"too many candidates ({len(candidates)}; cap {_MAX_CANDIDATES})"
        )

    # M2 — pre-validate the ENTIRE candidate list up front so a typo'd Nth
    # candidate fails BEFORE any (expensive) model load, rather than burning a
    # live load + eval on every preceding candidate. Only enforced when the
    # first-party factory is in play; an injected test seam factory is left
    # untouched (it may legitimately use non-Quant-Menu candidate names).
    if getattr(after_gen_factory, "_soup_first_party_measure_factory", False):
        for candidate in candidates:
            validate_measure_candidate(candidate)

    tasks_sha = sha_of_file(tasks_file)
    key = compute_cache_key(
        base_sha=base_sha, profile_name=profile_name, tasks_sha=tasks_sha,
    )

    cache = load_cache(cache_path)
    cached = cache.get(key)
    if isinstance(cached, dict):
        rows = cached.get("rows")
        if isinstance(rows, list):
            try:
                hit = [MeasureResult(**r) for r in rows]
                if all(isinstance(r, MeasureResult) for r in hit):
                    return hit, True
            except (TypeError, ValueError):
                pass

    # Cache miss — compute the deterministic baseline ONCE (M3), then run each
    # candidate against it. This keeps the baseline model out of VRAM while a
    # quantized candidate is loaded and avoids N× redundant baseline scoring.
    before_score = _score_tasks(tasks_file, before_gen)
    results: list[MeasureResult] = []
    for candidate in candidates:
        after_gen = after_gen_factory(candidate)
        result = measure_candidate(
            candidate=candidate, tasks_file=tasks_file,
            after_gen=after_gen, before_score=before_score,
        )
        results.append(result)

    cache[key] = {"rows": [asdict(r) for r in results]}
    save_cache(cache, cache_path)
    return results, False


def render_measure_table(results: Sequence[MeasureResult]) -> "Table":
    """Render a Rich table from a sequence of :class:`MeasureResult` rows."""
    from rich.markup import escape
    from rich.table import Table

    table = Table(title="Deploy autopilot — measured candidates")
    table.add_column("Candidate", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Verdict")
    for r in results:
        verdict_styled = {
            "OK": f"[green]{r.verdict}[/]",
            "MINOR": f"[yellow]{r.verdict}[/]",
            "MAJOR": f"[red]{r.verdict}[/]",
        }.get(r.verdict, escape(r.verdict))
        table.add_row(
            escape(r.candidate),
            f"{r.before:.3f}",
            f"{r.after:.3f}",
            f"{r.delta:+.3f}",
            verdict_styled,
        )
    return table
