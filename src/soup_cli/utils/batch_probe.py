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
# Folded into every cache key. "v2" = probe gates on the measured peak (#649);
# keys without it were written by the exception-only probe and are ignored.
_CACHE_KEY_VERSION = "v2"

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
    """Stable string key for the cache. Hashed for filesystem safety.

    ``_CACHE_KEY_VERSION`` is folded into the hash so an entry written by an
    older probe is simply never found. Bumped in #649: the exception-only probe
    approved batches that spilled to host memory under WDDM and cached them, and
    a cached wrong answer is recomputed by nobody.
    """
    for name, value in (
        ("max_length", max_length),
        ("lora_r", lora_r),
        ("gpu_memory_gb", gpu_memory_gb),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int (got {type(value).__name__})")
    raw = "|".join(
        [
            _CACHE_KEY_VERSION,
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


# ---------------------------------------------------------------------------
# Live CUDA probe builder (v0.40.3 #64)
# ---------------------------------------------------------------------------


def _is_cuda_oom(exc: BaseException, torch: Any) -> bool:
    """Is ``exc`` the device running out of memory, in any of torch's spellings?

    The allocator raises ``torch.cuda.OutOfMemoryError``. An OOM that surfaces
    later at a synchronize point does not: torch >= 2.8 raises
    ``torch.AcceleratorError("CUDA error: out of memory")`` and older releases a
    plain ``RuntimeError`` with the same text. #649 observed the second form
    under WDDM, where the allocator had already spilled instead of raising.
    Anything else (an illegal access, a device assert) is not a fit answer and
    must propagate.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _probe_budget_bytes(torch: Any, device: str) -> Optional[int]:
    """Bytes this process can reach on ``device``: what it holds plus what is free.

    ``mem_get_info()`` is a device-level driver query, so it excludes VRAM held
    by other processes (the streaming pre-flight relies on the same reading,
    see :func:`~soup_cli.utils.layer_stream.resolve_available_vram_bytes`).
    Under WDDM it reports physical VRAM: the allocator's own warning in #649
    read ``free: 0`` while the step kept going in host memory. Returns ``None``
    when the driver cannot answer, in which case the caller keeps the
    exception-only criterion rather than inventing a budget.
    """
    try:
        free, _total = torch.cuda.mem_get_info(device)
        held = torch.cuda.memory_allocated(device)
        return int(free) + int(held)
    except Exception:  # noqa: BLE001 — a driver that cannot answer is not a fit answer
        return None


def make_cuda_probe_fn(
    model: Any,
    tokenizer: Any,
    *,
    max_length: int,
    device: str = "cuda",
) -> Optional[ProbeFn]:
    """Build a real CUDA ``probe_fn`` for :func:`pick_batch_size`.

    Returns a closure that, given a candidate batch size ``B``, runs ONE
    forward + backward step on a synthetic batch of ``B`` sequences of
    length ``max_length``. Returns ``False`` when the step raises an OOM in
    any of torch's spellings (:func:`_is_cuda_oom`) OR when it completes with
    a measured peak above what this process can reach on the device. Other
    exceptions propagate so misconfiguration surfaces.

    The second criterion is the #649 fix. Under WDDM (native Windows, WSL2)
    the allocator does not raise when dedicated VRAM runs out; it spills to
    host memory and the step completes an order of magnitude slower, so "did
    not throw" is not "fits". The gate reads ``max_memory_allocated`` after
    the step, deliberately not ``max_memory_reserved``: reserved runs
    1.08-1.41x allocated and gating on it refuses configurations that run
    (measured for :func:`~soup_cli.utils.layer_stream.decide_measured_fit`,
    which this mirrors). The threshold is the budget from
    :func:`_probe_budget_bytes`, not a fraction of it: the probe's peak
    already runs above the real step (12.5-14.3% in the streaming
    measurements), which is the direction that makes an exact comparison
    safe.

    Returns ``None`` on non-CUDA devices, when torch is unavailable, when
    ``cuda.is_available()`` is False, or when any of the inputs is missing
    — :func:`pick_batch_size` falls back to the static estimate via its
    probe-unavailable branch.

    Added in v0.40.3 (#64). SFT-only this release; non-SFT trainer
    expansion can come later.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be int")
    if max_length < 8:
        raise ValueError(f"max_length must be >= 8, got {max_length}")
    if model is None or tokenizer is None:
        return None
    if device != "cuda":
        return None

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    # Use len(tokenizer) — `vocab_size` returns the BASE vocab and excludes
    # added special tokens. On Llama-3 / Qwen tokenizers with appended
    # `<|pad|>` at id 128255, vocab_size=128000 would mod the pad_id back to
    # `255` (random byte token), invalidating the probe. `len(tokenizer)`
    # includes added tokens.
    try:
        vocab_size = int(len(tokenizer))
    except TypeError:
        vocab_size = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    if vocab_size <= 1:
        vocab_size = 32000
    pad_id = int(pad_id) % vocab_size

    def _probe(batch_size: int) -> bool:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be int")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # Zero grads BEFORE forward — defends against the synthetic
        # backward accumulating into the live training model's grad
        # buffers (matches v0.35.0 #45 benchmark_kernel_combos policy).
        try:
            model.zero_grad(set_to_none=True)
        except (AttributeError, RuntimeError):
            pass
        try:
            # Budget and peak reset BEFORE the step: `synchronize()` here can
            # surface an earlier async OOM, which the handler below classifies.
            torch.cuda.synchronize()
            budget = _probe_budget_bytes(torch, device)
            torch.cuda.reset_peak_memory_stats(device)
            ids = torch.full(
                (batch_size, max_length), pad_id, dtype=torch.long, device=device,
            )
            attn = torch.ones_like(ids)
            labels = ids.clone()
            outputs = model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = getattr(outputs, "loss", None)
            if loss is not None:
                # Drop intermediate tensor refs BEFORE backward so peak VRAM
                # reflects the realistic training step (matches v0.35.0 policy).
                del ids, attn, labels, outputs
                loss.backward()
            else:
                # Last resort — generic signal we got past forward.
                del ids, attn, labels, outputs
            torch.cuda.synchronize()
            if budget is None:
                return True
            peak = int(torch.cuda.max_memory_allocated(device))
            # Completed is not fitted (WDDM spill, #649): refuse on the peak.
            return peak <= budget
        except Exception as exc:  # noqa: BLE001 — classified, not swallowed
            if _is_cuda_oom(exc, torch):
                return False
            raise
        finally:
            try:
                model.zero_grad(set_to_none=True)
            except (AttributeError, RuntimeError):
                pass
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass

    return _probe
