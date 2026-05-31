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

# Test + advanced-operator escape hatch: when set on this module, the deploy
# CLI uses these callables instead of the v0.46.1 model-loading generators.
# Documented as v0.53.1 deferral — see the v0.53.1 Known Limitations entry
# in plan.md. NOT a public API; the live transformers / vLLM generators land
# alongside v0.53.2.
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
    before_gen: Callable[[str], str],
    after_gen: Callable[[str], str],
) -> MeasureResult:
    """Score one quant candidate against the baseline.

    Returns a :class:`MeasureResult` with classify_delta verdict.
    """
    if isinstance(candidate, bool) or not isinstance(candidate, str):
        raise TypeError("candidate must be a non-bool str")
    if not candidate:
        raise ValueError("candidate must be non-empty")
    if "\x00" in candidate:
        raise ValueError("candidate must not contain null bytes")

    before = _score_tasks(tasks_file, before_gen)
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

    # Cache miss — run the eval loop
    results: list[MeasureResult] = []
    for candidate in candidates:
        after_gen = after_gen_factory(candidate)
        result = measure_candidate(
            candidate=candidate, tasks_file=tasks_file,
            before_gen=before_gen, after_gen=after_gen,
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
