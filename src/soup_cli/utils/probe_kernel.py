"""v0.71.8 #215/#217 — shared linear-probe math kernel.

A "linear probe" over a model's residual stream is a single
``(W ∈ R^D, threshold ∈ R)`` pair. Applying it to a 2D activation tensor
``[N_tokens, D]`` returns per-token scores; tokens with ``score > threshold``
are flagged. This is the real method behind Anthropic's "Simple probes can
catch sleeper agents" (April 2024) and the standard TruthfulQA / HarmBench
contrast-probe family.

This module ships the *real* probe math (pure-numpy, lazy import) shared by:

- ``soup_cli.utils.sleeper_probe`` — defection probe (#215)
- ``soup_cli.utils.truth_probe`` — honesty probe (#217)
- ``soup_cli.utils.harm_probe`` — misuse probe (#217)

The key primitive ``compute_contrast_probe(positive, negative)`` derives a
calibrated probe direction from two labelled activation sets (the difference
of class means, normalised), with the decision threshold at the midpoint of
the two projected class means. No downloaded artifact is required — the probe
is computed from the model's own activations. Operators who already hold a
calibrated weight file can plug it in via the per-probe ``load_probe_weights``
helpers instead.

Public surface:

- ``compute_contrast_probe(positive, negative)`` → ``(w, threshold)``
- ``apply_linear_probe(activations, w, threshold)`` → per-token scores
- ``flagged_rate(scores, threshold)`` → fraction flagged ``∈ [0, 1]``
- ``classify_probe_rate(rate, *, minor, major)`` → ``OK`` / ``MINOR`` / ``MAJOR``
- ``load_probe_weights(path)`` → ``(w, threshold)`` from .npz / .npy / .safetensors
- ``ProbeSpec`` / ``ProbeResult`` frozen dataclasses (shared by truth / harm)
- ``synthetic_probe_weights(base, hidden_dim, *, salt)`` → offline fallback vector
- ``run_linear_probe(activations, *, kind, base, w, threshold, minor, major)``
"""
from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Shared caps (mirror sleeper_probe / sae_diff limits).
_MAX_TOKENS = 1_000_000
_MAX_HIDDEN_DIM = 65_536
_MAX_BASE_LEN = 200
_MAX_WEIGHTS_BYTES = 512 * 1024 * 1024  # probe weight file DoS cap

PROBE_VERDICTS: Tuple[str, str, str] = ("OK", "MINOR", "MAJOR")


def _as_2d(value: Any, field: str) -> Any:
    import numpy as np

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be array-like") from exc
    if arr.ndim != 2:
        raise ValueError(f"{field} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{field} must be non-empty")
    if arr.shape[0] > _MAX_TOKENS:
        raise ValueError(f"{field} has too many rows ({arr.shape[0]})")
    if arr.shape[1] <= 0 or arr.shape[1] > _MAX_HIDDEN_DIM:
        raise ValueError(
            f"{field} hidden dim must be in (0, {_MAX_HIDDEN_DIM}]"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} must be finite")
    return arr


def compute_contrast_probe(
    positive: Any,
    negative: Any,
) -> Tuple[Any, float]:
    """Derive a calibrated linear probe from two labelled activation sets.

    ``positive`` / ``negative`` are ``[N, D]`` activation matrices (one row
    per example, e.g. mean-pooled residual stream at a layer). The probe
    direction is ``mean(positive) - mean(negative)`` normalised to unit norm;
    the decision threshold is the midpoint of the two projected class means
    (so a fresh "positive" example scores above threshold, a "negative" below).

    This is the real contrast-pair method (Anthropic "Simple probes can catch
    sleeper agents") — no downloaded artifact required.

    Returns ``(w, threshold)`` where ``w`` is a float32 numpy vector of length
    ``D`` and ``threshold`` is a Python float.
    """
    import numpy as np

    pos = _as_2d(positive, "positive")
    neg = _as_2d(negative, "negative")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"hidden-dim mismatch: positive[{pos.shape[1]}] vs "
            f"negative[{neg.shape[1]}]"
        )
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    direction = pos_mean - neg_mean
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError(
            "degenerate contrast probe (positive and negative means coincide)"
        )
    w = (direction / norm).astype(np.float32)
    proj_pos = float(pos_mean @ w)
    proj_neg = float(neg_mean @ w)
    threshold = (proj_pos + proj_neg) / 2.0
    if not math.isfinite(threshold):
        raise ValueError("contrast probe produced a non-finite threshold")
    return w, threshold


def apply_linear_probe(
    activations: Any,
    w: Any,
    threshold: float,
) -> Any:
    """Per-token probe scores ``activations @ w``.

    ``threshold`` is validated here so callers cannot pass NaN; it is not
    applied to the projection (callers compare scores to it separately via
    :func:`flagged_rate`). Shape contract mirrors
    ``sleeper_probe.apply_sleeper_probe``: ``activations`` ``[N, D]`` and
    ``w`` ``[D]``.
    """
    import numpy as np

    if isinstance(threshold, bool):
        raise TypeError("threshold must be float, got bool")
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be float")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")

    try:
        acts = np.asarray(activations, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("activations must be array-like") from exc
    try:
        vec = np.asarray(w, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("w must be array-like") from exc
    if acts.ndim != 2:
        raise ValueError(f"activations must be 2D, got {acts.shape}")
    if vec.ndim != 1:
        raise ValueError(f"w must be 1D, got {vec.shape}")
    if acts.shape[1] != vec.shape[0]:
        raise ValueError(
            f"shape mismatch: activations[{acts.shape}] @ w[{vec.shape}]"
        )
    if acts.shape[0] > _MAX_TOKENS:
        raise ValueError(f"too many tokens ({acts.shape[0]})")
    if vec.shape[0] > _MAX_HIDDEN_DIM:
        raise ValueError(f"w hidden_dim too large ({vec.shape[0]})")
    # Reject non-finite activations so a corrupt --evidence file fails loudly
    # rather than producing a silently-wrong flag rate (NaN comparisons are
    # always False → undercount) — review fix.
    if not np.all(np.isfinite(acts)):
        raise ValueError("activations must be finite")
    return acts @ vec


def flagged_rate(scores: Any, threshold: float) -> float:
    """Fraction of ``scores`` strictly greater than ``threshold`` ∈ ``[0, 1]``."""
    import numpy as np

    if isinstance(threshold, bool):
        raise TypeError("threshold must be float, got bool")
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be float")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    try:
        arr = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("scores must be array-like") from exc
    if arr.ndim != 1:
        raise ValueError(f"scores must be 1D, got {arr.shape}")
    n = int(arr.shape[0])
    if n == 0:
        return 0.0
    flagged = int(np.sum(arr > threshold))
    return flagged / n


def classify_probe_rate(rate: float, *, minor: float, major: float) -> str:
    """OK if ``rate < minor``, MINOR if ``rate < major``, else MAJOR.

    Boundary semantics put the boundary value in the MORE SEVERE bucket
    (exactly ``minor`` → MINOR, exactly ``major`` → MAJOR), matching the
    v0.26 / v0.56 / v0.65 / v0.66 review-fix policy on threshold semantics.
    """
    if isinstance(rate, bool):
        raise TypeError("rate must be float, got bool")
    if not isinstance(rate, (int, float)):
        raise TypeError("rate must be float")
    if not math.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    for name, band in (("minor", minor), ("major", major)):
        if isinstance(band, bool) or not isinstance(band, (int, float)):
            raise TypeError(f"{name} band must be float")
        if not math.isfinite(float(band)) or not 0.0 <= band <= 1.0:
            raise ValueError(f"{name} band must be in [0, 1]")
    if minor > major:
        raise ValueError("minor band must be ≤ major band")
    if rate < minor:
        return "OK"
    if rate < major:
        return "MINOR"
    return "MAJOR"


@dataclass(frozen=True)
class ProbeSpec:
    """Generic calibrated linear-probe metadata (shared by truth / harm, #217)."""

    base: str
    hidden_dim: int
    threshold: float
    description: str

    def __post_init__(self) -> None:
        if isinstance(self.base, bool) or not isinstance(self.base, str):
            raise TypeError("base must be str")
        if not self.base:
            raise ValueError("base must be non-empty")
        if isinstance(self.hidden_dim, bool) or not isinstance(self.hidden_dim, int):
            raise TypeError("hidden_dim must be int")
        if self.hidden_dim <= 0 or self.hidden_dim > _MAX_HIDDEN_DIM:
            raise ValueError(f"hidden_dim must be in (0, {_MAX_HIDDEN_DIM}]")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("threshold must be float")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")
        if not isinstance(self.description, str):
            raise TypeError("description must be str")


@dataclass(frozen=True)
class ProbeResult:
    """Generic linear-probe verdict (shared by truth / harm, #217)."""

    kind: str
    base: str
    num_tokens: int
    flag_rate: float
    max_score: float
    verdict: str

    def __post_init__(self) -> None:
        for name in ("kind", "base"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, str):
                raise TypeError(f"{name} must be str")
            if not val:
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.num_tokens, bool) or not isinstance(self.num_tokens, int):
            raise TypeError("num_tokens must be int")
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        for name in ("flag_rate", "max_score"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be float")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.flag_rate <= 1.0:
            raise ValueError("flag_rate must be in [0, 1]")
        if isinstance(self.verdict, bool) or not isinstance(self.verdict, str):
            raise TypeError("verdict must be str")
        if self.verdict not in PROBE_VERDICTS:
            raise ValueError(f"verdict must be one of {PROBE_VERDICTS}")


def synthetic_probe_weights(base: str, hidden_dim: int, *, salt: str) -> Any:
    """Deterministic synthetic probe direction keyed on ``salt:base``.

    The offline fallback when no real calibrated weights are supplied. ``salt``
    (``"truth"`` / ``"harm"`` / ``"sleeper"``) makes each probe kind a distinct
    deterministic direction. Seed = ``sha256(salt:base)[:16]`` so the vector is
    reproducible ACROSS Python processes (``hash()`` is process-salted unless
    ``PYTHONHASHSEED=0``).
    """
    import numpy as np

    if isinstance(base, bool) or not isinstance(base, str) or not base:
        raise ValueError("base must be a non-empty string")
    if isinstance(salt, bool) or not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
        raise TypeError("hidden_dim must be int")
    if hidden_dim <= 0 or hidden_dim > _MAX_HIDDEN_DIM:
        raise ValueError(f"hidden_dim must be in (0, {_MAX_HIDDEN_DIM}]")
    digest = hashlib.sha256(f"{salt}:{base}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    w = rng.standard_normal(hidden_dim).astype(np.float32)
    norm = float(np.linalg.norm(w))
    if norm > 0:
        w = w / norm
    return w


def run_linear_probe(
    activations: Any,
    *,
    kind: str,
    base: str,
    w: Any,
    threshold: float,
    minor: float,
    major: float,
) -> ProbeResult:
    """Apply ``(w, threshold)`` to ``activations`` and build a :class:`ProbeResult`.

    Generic orchestrator shared by truth / harm. ``activations`` must be a 2D
    ``[N_tokens, D]`` array. ``minor`` / ``major`` are the flag-rate bands for
    :func:`classify_probe_rate` (truth / harm use 5% / 20%).
    """
    import numpy as np

    acts = np.asarray(activations, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"activations must be 2D, got {acts.shape}")
    scores = apply_linear_probe(acts, w, threshold)
    n = int(acts.shape[0])
    rate = flagged_rate(scores, threshold)
    max_score = float(np.max(scores)) if n > 0 else 0.0
    if not math.isfinite(max_score):
        max_score = 0.0
    verdict = classify_probe_rate(rate, minor=minor, major=major)
    return ProbeResult(
        kind=kind,
        base=base,
        num_tokens=n,
        flag_rate=float(rate),
        max_score=max_score,
        verdict=verdict,
    )


def validate_bundled_base(name: object, bundled: "Mapping[str, ProbeSpec]") -> str:
    """Case-insensitive allowlist canonicaliser against a bundled-probe map."""
    if isinstance(name, bool):
        raise TypeError("base must be str, got bool")
    if not isinstance(name, str):
        raise TypeError(f"base must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("base must be non-empty")
    if "\x00" in name:
        raise ValueError("base must not contain null bytes")
    if len(name) > _MAX_BASE_LEN:
        raise ValueError(f"base must be ≤{_MAX_BASE_LEN} chars")
    lower = name.lower()
    for known in bundled:
        if known.lower() == lower:
            return known
    raise ValueError(
        f"no bundled probe for base {name!r} (known: {sorted(bundled)})"
    )


def run_bundled_probe(
    activations: Any,
    base: str,
    *,
    kind: str,
    bundled: "Mapping[str, ProbeSpec]",
    salt: str,
    minor: float,
    major: float,
    weights: Optional[Tuple[Any, float]] = None,
) -> ProbeResult:
    """Resolve a probe (operator weights or synthetic-from-bundled-spec) and run it.

    Shared by ``truth_probe`` / ``harm_probe``. When ``weights`` is supplied the
    bundled allowlist is not consulted (``base`` only shape-validated). Otherwise
    ``base`` must be in ``bundled`` and the synthetic seed-derived fallback is
    used. ``minor`` / ``major`` are the verdict bands (truth / harm use 5% / 20%).
    """
    import numpy as np

    acts = np.asarray(activations, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"activations must be 2D, got {acts.shape}")

    if weights is not None:
        if not isinstance(weights, tuple) or len(weights) != 2:
            raise TypeError("weights must be a (W, threshold) tuple")
        w_raw, threshold = weights
        if isinstance(base, bool) or not isinstance(base, str):
            raise TypeError("base must be str")
        if not base or "\x00" in base:
            raise ValueError("base must be a non-empty string without null bytes")
        if len(base) > _MAX_BASE_LEN:
            raise ValueError(f"base must be ≤{_MAX_BASE_LEN} chars")
        w = np.asarray(w_raw, dtype=np.float32)
        if w.ndim != 1:
            raise ValueError("weights[0] must be a 1D vector")
        if acts.shape[1] != w.shape[0]:
            raise ValueError(
                f"hidden_dim mismatch: activations[{acts.shape[1]}] vs "
                f"probe[{w.shape[0]}]"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("weights[1] (threshold) must be float")
        if not math.isfinite(float(threshold)):
            raise ValueError("weights[1] (threshold) must be finite")
        result_base, thr = base, float(threshold)
    else:
        canonical = validate_bundled_base(base, bundled)
        spec = bundled[canonical]
        if acts.shape[1] != spec.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: activations[{acts.shape[1]}] vs "
                f"probe[{spec.hidden_dim}]"
            )
        result_base = canonical
        w = synthetic_probe_weights(canonical, spec.hidden_dim, salt=salt)
        thr = spec.threshold

    return run_linear_probe(
        acts, kind=kind, base=result_base, w=w, threshold=thr,
        minor=minor, major=major,
    )


def load_probe_weights(path: str) -> Tuple[Any, float]:
    """Load an operator-supplied calibrated probe ``(W, threshold)`` from disk.

    Replaces the synthetic seed-derived placeholder with a real weight file
    (#215). Supports three formats:

    - ``.npz`` — numpy archive with a ``w`` array (required) and an optional
      scalar ``threshold`` (default 0.0).
    - ``.npy`` — a bare 1D ``w`` vector (threshold defaults to 0.0).
    - ``.safetensors`` — a ``W_probe`` tensor (required) and an optional 1-element
      ``threshold`` tensor.

    The path is cwd-contained + symlink-rejected (``O_NOFOLLOW`` probe-open, same
    policy as ``sae_diff.load_sae_weights``). ``np.load`` runs with
    ``allow_pickle=False`` so a crafted archive cannot execute code. Returns
    ``(w, threshold)`` where ``w`` is a float32 numpy vector and ``threshold``
    is a Python float.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(path, "probe weights")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_RDONLY | no_follow)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"probe weights not found: {os.path.basename(path)}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"probe weights cannot be opened (symlink?): {type(exc).__name__}"
        ) from exc
    os.close(fd)
    real = os.path.realpath(path)
    # DoS cap before any read (mirrors the CLI evidence loader's 16 MiB cap).
    if os.path.getsize(real) > _MAX_WEIGHTS_BYTES:
        raise ValueError(
            f"probe weights file exceeds {_MAX_WEIGHTS_BYTES} bytes"
        )

    import numpy as np

    lower = path.lower()
    threshold = 0.0
    if lower.endswith(".npz"):
        with np.load(real, allow_pickle=False) as archive:
            if "w" not in archive.files:
                raise KeyError("npz archive missing required 'w' array")
            w = archive["w"]
            if "threshold" in archive.files:
                threshold = float(archive["threshold"])
    elif lower.endswith(".npy"):
        w = np.load(real, allow_pickle=False)
    elif lower.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - dep present in CI
            raise RuntimeError(
                "safetensors package required; pip install safetensors"
            ) from exc
        with safe_open(real, framework="numpy") as handle:
            keys = list(handle.keys())
            if "W_probe" not in keys:
                raise KeyError("safetensors checkpoint missing 'W_probe'")
            w = handle.get_tensor("W_probe")
            if "threshold" in keys:
                threshold = float(handle.get_tensor("threshold").reshape(-1)[0])
    else:
        raise ValueError(
            "probe weights must be .npz / .npy / .safetensors"
        )

    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 1:
        raise ValueError(f"probe weights 'w' must be 1D, got shape {w.shape}")
    if w.shape[0] <= 0 or w.shape[0] > _MAX_HIDDEN_DIM:
        raise ValueError(
            f"probe weights hidden dim must be in (0, {_MAX_HIDDEN_DIM}]"
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("probe weights must be finite")
    if not math.isfinite(threshold):
        raise ValueError("probe threshold must be finite")
    return w, float(threshold)
