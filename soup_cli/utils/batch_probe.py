"""OOM-binary-search auto batch size + cache (v0.36.0 Part D).

Replaces sft.py's static-formula auto batch (which under-counts activations,
gradient buffers, and optimizer state and is frequently wrong on first run)
with a real try/halve loop. Mirrors LlamaFactory + Axolotl probes.

The probe runs ONE forward+backward+step per candidate before the real
training loop. To avoid re-probing on every run, the picked size is cached
in a JSON file keyed on the (model, max_length, quantization, lora_r, gpu)
tuple. Default cache path: ``~/.soup/batch_cache.json``. Override via
``SOUP_BATCH_CACHE_PATH`` env var (used by tests).

Pure-logic surface (binary-search loop, cache I/O, key normalisation) is
fully testable without CUDA. The CUDA-side ``probe_fn`` callable is supplied
by the trainer wrapper at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Optional

# Stay safe — never go below 1; never run forever.
_MIN_BATCH = 1
_DEFAULT_MAX_DOUBLINGS = 8

ProbeFn = Callable[[int], bool]


# ---------------------------------------------------------------------------
# Pure binary search
# ---------------------------------------------------------------------------


def probe_batch_size(
    probe: ProbeFn,
    *,
    start: int,
    ceiling: int,
    oom_exceptions: tuple[type[BaseException], ...],
    max_doublings: int = _DEFAULT_MAX_DOUBLINGS,
) -> int:
    """Try-halve-then-double loop. Returns the largest batch that ran OK.

    Strategy:

    1. Try ``start``. If OOM, halve until either it fits or hits ``_MIN_BATCH``.
    2. If start fits, double until OOM (or ``ceiling``). Back off by half
       to the last known-good size.

    Args:
        probe: Callable taking a batch size; returns ``True`` on success or
            raises one of ``oom_exceptions`` on OOM. Any other exception
            propagates unchanged.
        start: Initial batch size to try (must be >= 1).
        ceiling: Hard cap — never exceed this size.
        oom_exceptions: Tuple of exception classes to treat as OOM.
        max_doublings: Cap successful doublings to prevent runaway.

    Raises:
        ValueError: ``start <= 0`` or ``ceiling < start``.
        RuntimeError: Even ``batch_size=1`` OOMs.
    """
    if not isinstance(start, int) or isinstance(start, bool) or start <= 0:
        raise ValueError("start must be a positive int")
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < start:
        raise ValueError("ceiling must be an int >= start")

    # Halve until it fits.
    current = start
    last_good: Optional[int] = None
    while current >= _MIN_BATCH:
        try:
            ok = probe(current)
        except oom_exceptions:
            current = current // 2
            continue
        if ok:
            last_good = current
            break
        current = current // 2

    if last_good is None:
        raise RuntimeError(
            "OOM at batch_size=1 — model + max_length + quantization is too "
            "large for this GPU. Reduce data.max_length, enable 4bit "
            "quantization, or use FSDP / DeepSpeed."
        )

    # Double until OOM or ceiling.
    doublings = 0
    while doublings < max_doublings and last_good < ceiling:
        candidate = min(last_good * 2, ceiling)
        if candidate == last_good:
            break
        try:
            ok = probe(candidate)
        except oom_exceptions:
            break
        if not ok:
            break
        last_good = candidate
        doublings += 1

    return last_good


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_path() -> str:
    """Resolve the cache file path with containment.

    Override via ``SOUP_BATCH_CACHE_PATH`` env var is allowed but the path
    must stay under either the user's home directory or the current
    working directory. This prevents env-var poisoning from turning the
    cache write into an arbitrary-file-write primitive (e.g. crafted
    ``SOUP_BATCH_CACHE_PATH=/etc/cron.d/soup`` from a compromised shell
    profile or CI).
    """
    override = os.environ.get("SOUP_BATCH_CACHE_PATH")
    if override:
        import tempfile

        candidate = os.path.realpath(override)
        home = os.path.realpath(os.path.expanduser("~"))
        cwd = os.path.realpath(os.getcwd())
        tmp = os.path.realpath(tempfile.gettempdir())
        for anchor in (home, cwd, tmp):
            try:
                if os.path.commonpath([candidate, anchor]) == anchor:
                    return candidate
            except ValueError:
                continue
        # Out-of-bounds override — fall through to the safe default.
        return os.path.join(home, ".soup", "batch_cache.json")
    return os.path.join(os.path.expanduser("~"), ".soup", "batch_cache.json")


def make_cache_key(
    base: str,
    max_length: int,
    quantization: str,
    lora_r: int,
    gpu_name: str,
    gpu_memory_gb: int,
) -> str:
    """Stable string key for the cache. Hashed for filesystem safety."""
    for name, value in (
        ("max_length", max_length),
        ("lora_r", lora_r),
        ("gpu_memory_gb", gpu_memory_gb),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int (got {type(value).__name__})")
    raw = "|".join(
        [
            str(base),
            str(max_length),
            str(quantization),
            str(lora_r),
            str(gpu_name),
            str(gpu_memory_gb),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def load_cache() -> dict[str, int]:
    """Load the JSON cache. Returns ``{}`` on missing / malformed file."""
    path = _cache_path()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v > 0:
            out[k] = v
    return out


def save_cache_entry(key: str, value: int) -> None:
    """Insert/update one entry. Other entries are preserved."""
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("value must be a positive int")
    cache = load_cache()
    cache[key] = value
    path = _cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        # Best-effort 0600 — match v0.26.0 registry.db policy. Failure on
        # Windows / non-POSIX FS is silently ignored.
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        # Cache is best-effort — never crash training because the home dir
        # is read-only.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def pick_batch_size(
    *,
    static_estimate: int,
    strategy: str,
    base: str,
    max_length: int,
    quantization: str,
    lora_r: int,
    gpu_name: str,
    gpu_memory_gb: int,
    probe_fn: Optional[ProbeFn],
    oom_exceptions: Optional[tuple[type[BaseException], ...]] = None,
    console: Any = None,
) -> int:
    """Top-level batch picker. Honours strategy + cache + probe.

    Returns:
        Picked batch size (always >= 1). Falls back to ``static_estimate``
        when probing is unavailable or the strategy is "static". When
        ``strategy="probe"`` is explicit but ``probe_fn`` is ``None``, a
        yellow advisory is printed via ``console`` (if supplied).
    """
    if not isinstance(static_estimate, int) or static_estimate <= 0:
        raise ValueError("static_estimate must be a positive int")

    if strategy == "static":
        return static_estimate

    # auto / probe — same code path; difference is auto silently skips
    # probing when probe_fn is unavailable; explicit probe surfaces a warning.
    if probe_fn is None:
        if strategy == "probe" and console is not None:
            console.print(
                "[yellow]auto_batch_size_strategy='probe' requested but no "
                "probe_fn available — falling back to the static estimate. "
                "This is expected on CPU-only runs.[/]"
            )
        return static_estimate

    key = make_cache_key(base, max_length, quantization, lora_r, gpu_name, gpu_memory_gb)
    cache = load_cache()
    cached = cache.get(key)
    if cached:
        return cached

    if oom_exceptions is None:
        # Caller didn't pre-import torch — this is the trainer-side path.
        try:
            import torch
        except ImportError:
            return static_estimate
        oom_exceptions = (torch.cuda.OutOfMemoryError,)

    # ceiling = static * 4 — never go higher than 4x what the static formula
    # estimated, so a misconfigured probe can't run forever.
    ceiling = static_estimate * 4
    picked = probe_batch_size(
        probe_fn,
        start=static_estimate,
        ceiling=ceiling,
        oom_exceptions=oom_exceptions,
    )
    save_cache_entry(key, picked)
    return picked
