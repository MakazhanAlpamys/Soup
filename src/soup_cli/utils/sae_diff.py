"""v0.66.0 Part A — SAE feature diff.

Sparse Autoencoder (SAE) pre/post feature attribution. Pure-numpy math; the
``safetensors`` package is lazy-imported so this module's import is cheap.

Public surface:

- ``SaeFeatureChange`` / ``SaeFeatureDiffReport`` frozen dataclasses
- ``HF_HUB_ALLOWLIST`` closed frozenset of known SAE repos
- ``validate_sae_repo(name)`` — case-insensitive canonicalisation
- ``encode_activations(activations, w_enc, b_enc=None)`` — sparse-ReLU encode
- ``compute_feature_diff(pre_feats, post_feats, *, top_k)`` — pure math
- ``load_sae_weights(path)`` — containment + symlink rejection
- ``compute_sae_diff(pre, post, sae, *, top_k)`` — end-to-end orchestrator
- ``render_report_json`` / ``render_report_markdown`` for CI consumption

OK / MINOR / MAJOR taxonomy not used here — feature diff is a descriptive
report, not a regression gate. Callers chain into ``soup diagnose`` if they
want verdict classification.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Optional, Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

# Closed allowlist of known SAE repos on HF Hub. We do NOT auto-download — the
# operator must already have the safetensors file locally. The allowlist only
# governs which repo IDs we will record in a report's provenance field so a
# tampered manifest cannot quietly cite a random repo.
HF_HUB_ALLOWLIST: frozenset[str] = frozenset({
    # DeepMind Gemma Scope (Gemma 2 9B / 27B residual-stream SAEs).
    "google/gemma-scope-2b-pt-res",
    "google/gemma-scope-9b-pt-res",
    "google/gemma-scope-27b-pt-res",
    # EleutherAI Pythia SAE family.
    "eleutherai/sae-pythia-70m-deduped",
    "eleutherai/sae-pythia-160m-deduped",
    # JBloomAus SAE Lens — Llama family.
    "jbloomaus/llama-2-7b-saes",
    "jbloomaus/llama-3-8b-saes",
    # OpenAI GPT-2 small SAE (research demo).
    "openai/sae-gpt2-small",
})

_MAX_REPO_LEN = 200
_MAX_TOP_K = 100_000
_MIN_TOP_K = 1
_MAX_FEATURES = 1_000_000  # 1M SAE features cap
_MAX_TOKENS = 1_000_000


# ---------------------------------------------------------------------------
# Frozen reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaeFeatureChange:
    """Per-feature pre/post-FT activation change."""

    feature_id: int
    delta: float  # mean post - mean pre (token-averaged)
    pre_mean: float
    post_mean: float

    def __post_init__(self) -> None:
        if isinstance(self.feature_id, bool) or not isinstance(
            self.feature_id, int
        ):
            raise TypeError("feature_id must be int")
        if self.feature_id < 0:
            raise ValueError("feature_id must be non-negative")
        for name in ("delta", "pre_mean", "post_mean"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be float")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class SaeFeatureDiffReport:
    """End-to-end SAE diff: top-K changed features + summary."""

    num_features: int
    num_tokens: int
    l2_drift: float
    changes: Tuple[SaeFeatureChange, ...]

    def __post_init__(self) -> None:
        for name in ("num_features", "num_tokens"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{name} must be int")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.l2_drift, bool) or not isinstance(
            self.l2_drift, (int, float)
        ):
            raise TypeError("l2_drift must be float")
        if not math.isfinite(float(self.l2_drift)):
            raise ValueError("l2_drift must be finite")
        if self.l2_drift < 0:
            raise ValueError("l2_drift must be non-negative")
        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be tuple")
        for change in self.changes:
            if not isinstance(change, SaeFeatureChange):
                raise TypeError("changes must be tuple of SaeFeatureChange")


# ---------------------------------------------------------------------------
# Repo allowlist validator
# ---------------------------------------------------------------------------


def validate_sae_repo(name: object) -> str:
    """Validate a SAE repo id against the closed allowlist.

    Case-insensitive: returns the canonical (lower-case) entry that matches.
    """
    if isinstance(name, bool):
        raise TypeError("repo name must be str, got bool")
    if not isinstance(name, str):
        raise TypeError(
            f"repo name must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("repo name must be non-empty")
    if "\x00" in name:
        raise ValueError("repo name must not contain null bytes")
    if len(name) > _MAX_REPO_LEN:
        raise ValueError(f"repo name must be ≤{_MAX_REPO_LEN} chars")
    lower = name.lower()
    if lower not in HF_HUB_ALLOWLIST:
        raise ValueError(
            f"repo {name!r} not in HF_HUB_ALLOWLIST "
            f"(known: {sorted(HF_HUB_ALLOWLIST)})"
        )
    return lower


# ---------------------------------------------------------------------------
# Math kernel — sparse encoder + diff
# ---------------------------------------------------------------------------


def encode_activations(
    activations: Any,
    w_enc: Any,
    b_enc: Optional[Any] = None,
) -> Any:
    """Sparse-ReLU encoder: ``ReLU(activations @ w_enc + b_enc)``.

    Parameters
    ----------
    activations : ``[N_tokens, D_model]`` numpy array of model hidden states
    w_enc       : ``[D_model, N_features]`` encoder weight matrix
    b_enc       : optional ``[N_features]`` encoder bias (default zeros)

    Returns
    -------
    ``[N_tokens, N_features]`` numpy array of sparse feature activations.
    """
    import numpy as np

    try:
        acts = np.asarray(activations, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("activations must be array-like") from exc
    try:
        weights = np.asarray(w_enc, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("W_enc must be array-like") from exc
    if acts.ndim != 2:
        raise ValueError(f"activations must be 2D, got shape {acts.shape}")
    if weights.ndim != 2:
        raise ValueError(f"W_enc must be 2D, got shape {weights.shape}")
    if acts.shape[1] != weights.shape[0]:
        raise ValueError(
            f"shape mismatch: activations[{acts.shape}] @ w_enc[{weights.shape}]"
        )
    if weights.shape[1] > _MAX_FEATURES:
        raise ValueError(
            f"SAE has >{_MAX_FEATURES} features (got {weights.shape[1]})"
        )
    if acts.shape[0] > _MAX_TOKENS:
        raise ValueError(
            f"activations has >{_MAX_TOKENS} tokens (got {acts.shape[0]})"
        )
    pre = acts @ weights
    if b_enc is not None:
        bias = np.asarray(b_enc, dtype=np.float32)
        if bias.ndim != 1 or bias.shape[0] != weights.shape[1]:
            raise ValueError(
                f"b_enc must be 1D with {weights.shape[1]} entries"
            )
        pre = pre + bias
    return np.maximum(pre, 0.0)


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool):
        raise TypeError("top_k must be int, got bool")
    if not isinstance(top_k, int):
        raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
    if top_k < _MIN_TOP_K:
        raise ValueError(f"top_k must be ≥{_MIN_TOP_K}")
    if top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be ≤{_MAX_TOP_K}")
    return top_k


def compute_feature_diff(
    pre_features: Any,
    post_features: Any,
    *,
    top_k: int,
) -> SaeFeatureDiffReport:
    """Compute per-feature mean activation diff and return top-K changes.

    Both arrays must have shape ``[N_tokens, N_features]``. ``top_k`` is
    clamped to ``N_features`` (so passing top_k=100 on a 10-feature SAE
    returns 10 changes, not an error).
    """
    import numpy as np

    _validate_top_k(top_k)

    try:
        pre = np.asarray(pre_features, dtype=np.float64)
        post = np.asarray(post_features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("pre/post must be array-like") from exc
    if pre.ndim != 2 or post.ndim != 2:
        raise ValueError("pre/post must be 2D feature arrays")
    if pre.shape != post.shape:
        raise ValueError(
            f"shape mismatch: pre={pre.shape}, post={post.shape}"
        )
    num_tokens, num_features = pre.shape
    if num_features > _MAX_FEATURES:
        raise ValueError(f"too many features ({num_features})")

    pre_mean = pre.mean(axis=0) if num_tokens > 0 else np.zeros(num_features)
    post_mean = post.mean(axis=0) if num_tokens > 0 else np.zeros(num_features)
    delta = post_mean - pre_mean
    abs_delta = np.abs(delta)

    # Top-K by absolute change. argpartition for O(N) then sort the K-prefix.
    actual_k = min(top_k, num_features)
    if actual_k <= 0:
        top_idx: Any = np.empty((0,), dtype=np.int64)
    elif actual_k >= num_features:
        # Just sort all
        top_idx = np.argsort(-abs_delta, kind="stable")
    else:
        part = np.argpartition(-abs_delta, actual_k - 1)[:actual_k]
        top_idx = part[np.argsort(-abs_delta[part], kind="stable")]

    changes = []
    for idx in top_idx:
        i = int(idx)
        d = float(delta[i])
        if not math.isfinite(d):
            # Skip non-finite features rather than crashing the report
            continue
        changes.append(
            SaeFeatureChange(
                feature_id=i,
                delta=d,
                pre_mean=float(pre_mean[i]),
                post_mean=float(post_mean[i]),
            )
        )

    l2 = float(np.sqrt(np.sum(delta * delta)))
    if not math.isfinite(l2):
        l2 = 0.0
    return SaeFeatureDiffReport(
        num_features=int(num_features),
        num_tokens=int(num_tokens),
        l2_drift=l2,
        changes=tuple(changes),
    )


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def load_sae_weights(path: str) -> Mapping[str, Any]:
    """Load a safetensors SAE checkpoint with cwd-containment + symlink reject.

    Returns a dict with at least ``w_enc`` (encoder matrix). Other keys
    (``b_enc``, ``W_dec``, etc.) are optional and forwarded as-is when present.
    """
    enforce_under_cwd_and_no_symlink(path, "sae_path")
    # H1 review fix: O_NOFOLLOW probe-open defeats symlink swap between
    # the containment check and the safetensors read. M1 review fix:
    # drop the dead `lstat(realpath)` (realpath already followed the
    # chain — the final-target lstat could never see a symlink).
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_RDONLY | no_follow)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"SAE weights not found: {os.path.basename(path)}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"SAE weights cannot be opened (symlink?): "
            f"{type(exc).__name__}"
        ) from exc
    os.close(fd)
    real = os.path.realpath(path)
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised when dep missing
        raise RuntimeError(
            "safetensors package required; pip install safetensors"
        ) from exc
    out: dict[str, Any] = {}
    with safe_open(real, framework="numpy") as f:
        # M5 review fix: pre-check key count BEFORE materializing tensors so
        # the DoS cap actually defends against pathological files.
        keys = list(f.keys())
        if len(keys) > 64:
            raise ValueError("SAE checkpoint has >64 tensors (suspicious)")
        # Validate every key shape upfront so we fail fast on invalid names.
        for key in keys:
            if not isinstance(key, str) or "\x00" in key:
                raise ValueError(f"invalid tensor key: {key!r}")
        for key in keys:
            out[key] = f.get_tensor(key)
    if "W_enc" not in out:
        raise KeyError("SAE checkpoint missing required 'W_enc' tensor")
    return out


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def compute_sae_diff(
    pre_activations: Any,
    post_activations: Any,
    sae: Mapping[str, Any],
    *,
    top_k: int,
) -> SaeFeatureDiffReport:
    """Encode pre/post activations through the SAE and diff the features."""
    # H2 review fix: validate top_k early at the orchestrator boundary so
    # we fail fast before the encode passes — same policy as v0.55.0 etc.
    _validate_top_k(top_k)
    if not isinstance(sae, Mapping):
        raise TypeError(f"sae must be Mapping, got {type(sae).__name__}")
    if "W_enc" not in sae:
        raise KeyError("sae must contain 'W_enc'")
    import numpy as np

    pre = np.asarray(pre_activations)
    post = np.asarray(post_activations)
    if pre.shape != post.shape:
        raise ValueError(
            f"pre/post activations shape mismatch: {pre.shape} vs {post.shape}"
        )
    w_enc = sae["W_enc"]
    b_enc = sae.get("b_enc")
    pre_feats = encode_activations(pre, w_enc, b_enc=b_enc)
    post_feats = encode_activations(post, w_enc, b_enc=b_enc)
    return compute_feature_diff(pre_feats, post_feats, top_k=top_k)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report_json(report: SaeFeatureDiffReport) -> str:
    """Canonical JSON for CI / Registry artifact."""
    if not isinstance(report, SaeFeatureDiffReport):
        raise TypeError("report must be SaeFeatureDiffReport")
    payload = {
        "num_features": report.num_features,
        "num_tokens": report.num_tokens,
        "l2_drift": report.l2_drift,
        "changes": [asdict(c) for c in report.changes],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def render_report_markdown(report: SaeFeatureDiffReport) -> str:
    """Human-readable markdown for PR comments.

    All numeric fields are rendered via Python format spec (safe). String
    fields would need Rich-markup escaping — review H2 (v0.66.0) flagged
    this as a forward-compat hazard if a future patch adds operator-
    controlled labels to SaeFeatureChange. For now there are no string
    fields, so the report is safe by construction.
    """
    if not isinstance(report, SaeFeatureDiffReport):
        raise TypeError("report must be SaeFeatureDiffReport")
    lines = [
        "# SAE feature diff",
        "",
        f"- features: **{report.num_features}**",
        f"- tokens: **{report.num_tokens}**",
        f"- L2 drift: **{report.l2_drift:.4f}**",
        "",
    ]
    if not report.changes:
        lines.append("_no changes (empty report)_")
        return "\n".join(lines) + "\n"
    lines.extend([
        "## Top-changed features",
        "",
        "| Feature | Δ activation | pre mean | post mean |",
        "| --- | --- | --- | --- |",
    ])
    for change in report.changes:
        lines.append(
            f"| {change.feature_id} "
            f"| {change.delta:+.4f} "
            f"| {change.pre_mean:.4f} "
            f"| {change.post_mean:.4f} |"
        )
    return "\n".join(lines) + "\n"


# Module-level: expose closed metadata via MappingProxyType for callers that
# iterate (matches v0.51.0+ pattern).
_REPO_METADATA: Mapping[str, str] = MappingProxyType({
    name: name for name in sorted(HF_HUB_ALLOWLIST)
})
