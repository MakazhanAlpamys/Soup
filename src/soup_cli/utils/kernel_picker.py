"""Kernel auto-composition — benchmark and pick the fastest kernel combo.

Enumerates installed performance kernels (Liger, FlashAttention, torch
baseline) and picks the fastest combination for the current GPU. Benchmarks
each candidate on a small warm-up loop and selects the one with the lowest
observed step time.

This is a config-resolver helper: the benchmarking loop is expected to be
driven by the trainer wrapper (a few warm-up steps before the real train
loop). Here we only provide:

- ``enumerate_kernel_combos(backend, device)`` — list candidate combos
- ``pick_best_kernel(candidates)`` — choose the fastest from timing results

Design: we never auto-enable combos that the user has *disabled* via their
TrainingConfig (e.g. if ``use_liger: false`` explicitly, it stays false). The
picker only searches within combos the user hasn't opted out of.
"""

from __future__ import annotations

from typing import Any


def enumerate_kernel_combos(
    backend: str, device: str,
) -> list[dict[str, Any]]:
    """Enumerate candidate kernel combinations for the current environment.

    Returns a list of dicts: ``{"name", "use_liger", "use_flash_attn",
    "use_cut_ce"}``. The list always contains a ``baseline`` entry (no special
    kernels) so that the picker has a reference point.

    Rules:
    - CPU → only baseline.
    - unsloth backend → baseline only (unsloth uses its own kernels internally).
    - mlx backend → baseline only (Apple Silicon path doesn't share kernels).
    - cuda + transformers → baseline + each available kernel + known-good combos.
    """
    baseline = {
        "name": "baseline",
        "use_liger": False,
        "use_flash_attn": False,
        "use_cut_ce": False,
    }

    # CPU: nothing to compose
    if device != "cuda":
        return [baseline]

    # Unsloth + MLX have their own kernel paths - picker would just confuse them
    if backend in ("unsloth", "mlx"):
        return [baseline]

    combos: list[dict[str, Any]] = [baseline]

    # Probe availability — lazy imports inside each helper
    try:
        from soup_cli.utils.liger import check_liger_available

        liger_ok = check_liger_available()
    except ImportError:
        liger_ok = False

    try:
        from soup_cli.utils.flash_attn import check_flash_attn_available

        flash_ok = check_flash_attn_available() is not None
    except ImportError:
        flash_ok = False

    try:
        from soup_cli.utils.cut_ce import check_cut_ce_available

        cce_ok = check_cut_ce_available()
    except ImportError:
        cce_ok = False

    if liger_ok:
        combos.append({
            "name": "liger",
            "use_liger": True,
            "use_flash_attn": False,
            "use_cut_ce": False,
        })

    if flash_ok:
        combos.append({
            "name": "flash",
            "use_liger": False,
            "use_flash_attn": True,
            "use_cut_ce": False,
        })

    if liger_ok and flash_ok:
        combos.append({
            "name": "liger+flash",
            "use_liger": True,
            "use_flash_attn": True,
            "use_cut_ce": False,
        })

    if cce_ok:
        combos.append({
            "name": "cut_ce",
            "use_liger": False,
            "use_flash_attn": False,
            "use_cut_ce": True,
        })

    if liger_ok and flash_ok and cce_ok:
        combos.append({
            "name": "liger+flash+cut_ce",
            "use_liger": True,
            "use_flash_attn": True,
            "use_cut_ce": True,
        })

    return combos


def benchmark_kernel_combos(
    model: Any,
    candidates: list[dict[str, Any]],
    *,
    device: str = "cuda",
    batch_size: int = 1,
    seq_len: int = 16,
    num_steps: int = 10,
    vocab_size: int = 32_000,
) -> list[dict[str, Any]]:
    """Run a tiny forward+backward warm-up loop per candidate and record time_ms.

    v0.35.0 #45 — closes the picker's "deterministic name-hash tiebreak"
    fallback by feeding it real measurements from the trainer's own model.
    Returns a NEW list of candidate dicts annotated with ``time_ms``; the
    original list is not mutated.

    On CPU / no-CUDA / torch unavailable, every candidate's ``time_ms`` is
    set to ``None`` — :func:`pick_best_kernel` will reject the result and
    the caller is expected to degrade to picker-without-bench.

    The benchmark intentionally does NOT swap kernels mid-loop (that would
    require model re-instantiation per candidate which is too expensive on
    CI-sized models). Instead we run identical forward+backward across
    candidates and record the relative ordering — useful as a coarse
    "did anything install correctly" health check.
    """
    # bool is a subclass of int — reject explicitly so that True / False
    # don't sneak in as 1 / 0 (matches v0.30.0 Candidate / v0.34.0 cost
    # estimator policy).
    for arg_name, arg_val in (
        ("batch_size", batch_size), ("seq_len", seq_len),
        ("num_steps", num_steps), ("vocab_size", vocab_size),
    ):
        if isinstance(arg_val, bool):
            raise TypeError(f"{arg_name} must be int, not bool")

    out: list[dict[str, Any]] = [dict(c) for c in candidates]

    if device != "cuda":
        for entry in out:
            entry["time_ms"] = None
        return out

    try:
        import torch
    except ImportError:
        for entry in out:
            entry["time_ms"] = None
        return out

    if not torch.cuda.is_available():
        for entry in out:
            entry["time_ms"] = None
        return out

    if model is None:
        for entry in out:
            entry["time_ms"] = None
        return out

    # Bound caller-supplied numbers so a misconfigured caller cannot OOM the
    # CI runner with seq_len=10**6.
    bs = max(1, min(int(batch_size), 32))
    sl = max(1, min(int(seq_len), 512))
    steps = max(1, min(int(num_steps), 50))
    vs = max(1024, min(int(vocab_size), 200_000))

    try:
        input_ids = torch.randint(
            0, vs, (bs, sl), device="cuda", dtype=torch.long,
        )
    except (RuntimeError, OSError):
        for entry in out:
            entry["time_ms"] = None
        return out

    for entry in out:
        entry["time_ms"] = _time_one_combo(model, input_ids, steps=steps)
    return out


def _time_one_combo(
    model: Any, input_ids: Any, *, steps: int,
) -> "float | None":
    """Run ``steps`` forward passes on ``model`` and return mean ms per step.

    v0.35.0 review fix — the benchmark is forward-only under ``torch.no_grad``
    so the live training model's parameter ``.grad`` tensors stay clean. The
    relative ordering between kernel combos is preserved (backward time is
    approximately proportional to forward time across HF causal LMs), and
    backward+optimizer state is what we MUST NOT pollute — corrupting the
    first real optimizer step with benchmark gradients was the original bug.

    Returns ``None`` on any error (OOM / model rejects shape) so the caller
    can degrade gracefully.
    """
    import math
    import time

    import torch  # benchmark only runs on CUDA path; torch is guaranteed here.

    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            # Warm-up step (not timed) — first call triggers cuDNN autotune.
            outputs = model(input_ids=input_ids)
            if _extract_logits_or_loss(outputs) is None:
                return None
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(steps):
                outputs = model(input_ids=input_ids)
                if _extract_logits_or_loss(outputs) is None:
                    return None
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / steps
        if not math.isfinite(elapsed_ms):
            return None
        return elapsed_ms
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return None


def _extract_logits_or_loss(outputs: Any) -> Any:
    """Return a model-output tensor — used only to verify the forward pass
    produced something. Forward-only benchmarks don't need a backward-able
    loss, just a sentinel to confirm the model accepted the input shape.
    """
    for attr in ("loss", "logits", "last_hidden_state"):
        value = getattr(outputs, attr, None)
        if value is not None:
            return value
    return None


def pick_best_kernel(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pick the fastest kernel combo given benchmarked timings.

    Args:
        candidates: List of dicts each with ``name`` and ``time_ms`` fields.

    Returns:
        The single candidate dict with the lowest ``time_ms``.
        Ties are broken by order (first candidate wins) so callers can place
        the preferred default (baseline) first.

    Raises:
        ValueError: if ``candidates`` is empty or if **all** candidates are
            missing a finite ``time_ms`` (no benchmark signal — picking blindly
            would mask a silent infrastructure failure).
    """
    if not candidates:
        raise ValueError("pick_best_kernel requires at least one candidate")

    finite = [c for c in candidates if _finite_time_ms(c.get("time_ms"))]
    if not finite:
        raise ValueError(
            "pick_best_kernel: no candidate has a finite time_ms — "
            "benchmarking appears to have failed for every combo."
        )

    # Stable sort: ties go to the one earlier in the list (baseline usually).
    return min(candidates, key=lambda c: _sortable_time_ms(c.get("time_ms")))


def _finite_time_ms(value: Any) -> bool:
    """True if ``value`` is a real finite number (not None, not NaN, not inf)."""
    import math

    if value is None:
        return False
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(as_float)


def _sortable_time_ms(value: Any) -> float:
    """Convert ``time_ms`` to a sortable float; missing/NaN → +inf."""
    import math

    if value is None:
        return float("inf")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if math.isnan(as_float):
        return float("inf")
    return as_float
