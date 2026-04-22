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
