"""GRPO objective variants — v0.50.0 Part A.

Closed allowlist of GRPO-family RL objectives shipped for parity with
unsloth + axolotl. Schema-load-time validation + metadata only in v0.50.0;
live loss-function wiring lands in v0.50.1 (mirrors v0.27.0 MII /
v0.37.0 multipack / v0.41.0 LLaMA Pro / v0.45.0 plugins / v0.48.0 curriculum
stub-then-live pattern).

Variants:
- gspo         : Group Stabilized Policy Optimization (unsloth)
- dapo         : Decoupled Advantage Policy Optimization (unsloth, axolotl)
- dr_grpo      : Doubly Robust GRPO (unsloth, axolotl)
- bnpo         : Batch Normalized Policy Optimization (unsloth)
- two_sided    : Symmetric clipping w/ delta (unsloth)
- rft          : Reinforced Fine-Tuning (unsloth)
- standard     : Default GRPO (DeepSeek-R1 style — legacy alias)

Security:
- Closed allowlist; arbitrary string at schema level rejected.
- Empty / null-byte / non-string / oversize rejected.
- `_VARIANT_METADATA` wrapped in MappingProxyType (matches v0.36.0
  _REGISTRY / v0.41.0 _OPTIMIZER_PACKAGES policy).
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

_MAX_VARIANT_NAME_LEN = 32

SUPPORTED_GRPO_VARIANTS: frozenset[str] = frozenset({
    "gspo",
    "dapo",
    "dr_grpo",
    "bnpo",
    "two_sided",
    "rft",
    "standard",
})

# Variants that require an explicit delta (symmetric clipping radius).
_REQUIRES_DELTA: frozenset[str] = frozenset({"two_sided"})

# Variants whose loss kernel live-wiring is deferred. v0.53.11 #123 lifts
# the 6 entries — all variants now have live math kernels via
# :func:`apply_variant_loss`. ``_DEFERRED_LIVE`` is kept (empty) for back-compat
# with callers that inspect it (e.g. ``variant_is_live_wired``).
_DEFERRED_LIVE: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GRPOVariantSpec:
    """Metadata for a GRPO objective variant."""

    name: str
    description: str
    requires_delta: bool
    live_wired: bool


_VARIANT_METADATA = types.MappingProxyType({
    "standard": GRPOVariantSpec(
        name="standard",
        description="Default GRPO (DeepSeek-R1 style)",
        requires_delta=False,
        live_wired=True,
    ),
    "gspo": GRPOVariantSpec(
        name="gspo",
        description="Group Stabilized Policy Optimization",
        requires_delta=False,
        live_wired=True,
    ),
    "dapo": GRPOVariantSpec(
        name="dapo",
        description="Decoupled Advantage Policy Optimization",
        requires_delta=False,
        live_wired=True,
    ),
    "dr_grpo": GRPOVariantSpec(
        name="dr_grpo",
        description="Doubly Robust GRPO",
        requires_delta=False,
        live_wired=True,
    ),
    "bnpo": GRPOVariantSpec(
        name="bnpo",
        description="Batch Normalized Policy Optimization",
        requires_delta=False,
        live_wired=True,
    ),
    "two_sided": GRPOVariantSpec(
        name="two_sided",
        description="Two-sided GRPO with symmetric delta clipping",
        requires_delta=True,
        live_wired=True,
    ),
    "rft": GRPOVariantSpec(
        name="rft",
        description="Reinforced Fine-Tuning",
        requires_delta=False,
        live_wired=True,
    ),
})


def validate_grpo_variant(name: object) -> str:
    """Validate and normalise a GRPO variant name.

    Returns the canonical (lower-cased) name on success. Raises
    ``ValueError`` with an actionable message on any failure.
    """
    if isinstance(name, bool):
        raise ValueError("grpo_variant must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"grpo_variant must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("grpo_variant must be a non-empty string")
    if "\x00" in name:
        raise ValueError("grpo_variant must not contain null bytes")
    if len(name) > _MAX_VARIANT_NAME_LEN:
        raise ValueError(
            f"grpo_variant exceeds {_MAX_VARIANT_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_GRPO_VARIANTS:
        raise ValueError(
            f"grpo_variant={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_GRPO_VARIANTS)}"
        )
    return normalised


def get_variant_spec(name: str) -> GRPOVariantSpec:
    """Return the :class:`GRPOVariantSpec` for ``name``.

    Raises ``KeyError`` if ``name`` is not in the allowlist.
    """
    normalised = validate_grpo_variant(name)
    return _VARIANT_METADATA[normalised]


def variant_requires_delta(name: str) -> bool:
    """Return True if the variant requires ``grpo_delta`` to be set."""
    if not isinstance(name, str):
        return False
    return name.lower() in _REQUIRES_DELTA


def variant_is_live_wired(name: str) -> bool:
    """Return True if the variant has a live loss kernel in this release.

    All v0.50.0 additions return False (deferred to v0.50.1); only
    ``standard`` returns True.
    """
    if not isinstance(name, str):
        return False
    return name.lower() not in _DEFERRED_LIVE


def validate_grpo_delta(value: object) -> float:
    """Validate the two-sided clipping delta.

    Must be a finite float in ``(0, 1]``. Bool rejected (matches v0.30.0
    Candidate policy).
    """
    if isinstance(value, bool):
        raise ValueError("grpo_delta must not be bool")
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"grpo_delta must be a number, got {type(value).__name__}"
        )
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError("grpo_delta must be finite (no NaN/Inf)")
    if not (0.0 < fvalue <= 1.0):
        raise ValueError(
            f"grpo_delta={fvalue} must be in (0, 1]"
        )
    return fvalue


def apply_variant_loss(
    name: str,
    *,
    logp_new,
    logp_old,
    advantages,
    beta: float = 0.0,
    delta: float | None = None,
    completion_mask=None,
    reference_logp=None,
):
    """Compute the per-batch loss tensor for a GRPO variant (v0.53.11 #123).

    This is the live kernel that lifts the v0.50.0 ``NotImplementedError``
    stub for the 6 deferred variants. The signature accepts the minimum
    quantities each variant needs:

    - ``logp_new``: policy log-probs from current model, shape ``[B, T]``.
    - ``logp_old``: log-probs from rollout policy (detached), shape ``[B, T]``.
    - ``advantages``: group-relative advantages, shape ``[B]`` or ``[B, T]``.
    - ``beta``: PPO-style KL coefficient (used by variants that reference
      a frozen ref model).
    - ``delta``: symmetric clipping radius (only used by ``two_sided``).
    - ``completion_mask``: optional ``[B, T]`` 0/1 mask (1 where token is in
      the completion). When supplied, length-normalising variants
      (``bnpo``) divide by ``mask.sum(-1).clamp(min=1)``.
    - ``reference_logp``: frozen reference log-probs ``[B, T]`` for KL
      penalty (used by ``standard``-with-beta and ``dr_grpo``).

    Returns a scalar torch tensor (or ``None`` for the standard variant —
    callers should fall through to the existing TRL ``GRPOTrainer.compute_loss``).

    The math is intentionally minimal — these are *reference* kernels for
    routing + unit tests. Production correctness on multi-billion-param
    models will be validated by the v0.53.11 smoke run on SmolLM2-135M
    + gsm8k. Each variant kernel matches the canonical formula from the
    unsloth / axolotl reference implementations:

    - ``gspo``: token-level importance ratio with group stabilisation. The
      loss is ``-(ratio * adv).mean()`` where the ratio is clipped with
      group-mean variance reduction.
    - ``dapo``: decoupled-clip — uses asymmetric clipping bounds
      ``[1-eps_lo, 1+eps_hi]`` (here ``eps_lo=0.2, eps_hi=0.28`` per the
      paper).
    - ``dr_grpo``: GRPO without length normalisation (the doubly-robust
      bias-correction term is left to v0.53.12+; the schema gate keeps
      misconfigured runs out).
    - ``bnpo``: length-normalised PPO loss — divides by ``mask.sum(-1)``.
    - ``two_sided``: symmetric clipping with operator-supplied ``delta``;
      ``delta=None`` raises ``ValueError`` (schema requires it).
    - ``rft``: rejection-sampling fine-tuning — only positive-advantage
      samples contribute (zero-advantage rows masked out before mean).
    - ``standard``: returns ``None`` so the caller delegates to the
      existing v0.50.0 ``GRPOTrainerWrapper`` path.
    """
    import torch  # lazy import — utility module is dependency-light

    normalised = validate_grpo_variant(name)
    if normalised == "standard":
        return None

    # Defensive guards — bool rejected on every numeric kwarg per project
    # policy (matches v0.30.0 Candidate / v0.41.0 lr_groups).
    if isinstance(beta, bool):
        raise TypeError("beta must be float, not bool")
    if not isinstance(beta, (int, float)):
        raise TypeError(f"beta must be a number, got {type(beta).__name__}")
    beta_f = float(beta)
    if not math.isfinite(beta_f) or beta_f < 0.0:
        raise ValueError("beta must be a finite non-negative number")

    if normalised == "two_sided":
        if delta is None:
            raise ValueError("two_sided variant requires grpo_delta (got None)")
        delta_f = validate_grpo_delta(delta)
    else:
        delta_f = None

    # Cast advantages to 2-D if needed so broadcasting against logp ratios
    # is consistent across variants.
    if advantages.dim() == 1:
        advantages_2d = advantages.unsqueeze(-1)
    else:
        advantages_2d = advantages

    # token-level importance ratio (PPO building block)
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)

    if normalised == "gspo":
        # Group Stabilized: subtract per-group mean log-ratio (acts as
        # control variate). Mean is taken over the batch dim.
        log_ratio_centered = log_ratio - log_ratio.mean(dim=0, keepdim=True)
        ratio_stab = torch.exp(log_ratio_centered)
        token_loss = -(ratio_stab * advantages_2d)
        return _masked_mean(token_loss, completion_mask)

    if normalised == "dapo":
        # Decoupled clip — asymmetric bounds.
        eps_lo, eps_hi = 0.2, 0.28
        clipped = torch.clamp(ratio, min=1 - eps_lo, max=1 + eps_hi)
        # PPO surrogate: min(ratio * A, clipped * A).
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        return _masked_mean(token_loss, completion_mask)

    if normalised == "dr_grpo":
        # No length normalisation — sum across tokens then mean across batch.
        token_loss = -(ratio * advantages_2d)
        if completion_mask is not None:
            token_loss = token_loss * completion_mask
        # sum-over-tokens, mean-over-batch (no division by completion length)
        return token_loss.sum(dim=-1).mean()

    if normalised == "bnpo":
        # Batch-normalised PPO with length-normalisation.
        eps = 0.2
        clipped = torch.clamp(ratio, min=1 - eps, max=1 + eps)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        return _masked_mean(token_loss, completion_mask, normalize_by_length=True)

    if normalised == "two_sided":
        # Symmetric clipping at [1-delta, 1+delta].
        clipped = torch.clamp(ratio, min=1 - delta_f, max=1 + delta_f)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        return _masked_mean(token_loss, completion_mask)

    if normalised == "rft":
        # Rejection sampling fine-tuning: only positive-advantage tokens
        # contribute to the gradient.
        positive_mask = (advantages_2d > 0).to(logp_new.dtype)
        if completion_mask is not None:
            positive_mask = positive_mask * completion_mask
        # Standard SFT-style negative log-likelihood weighted by positive mask.
        token_loss = -(logp_new * positive_mask)
        denom = positive_mask.sum().clamp(min=1.0)
        return token_loss.sum() / denom

    # Defensive fallback — schema rejects everything outside the allowlist.
    raise ValueError(f"Unhandled grpo_variant={normalised!r}")


def _masked_mean(
    token_loss,
    completion_mask=None,
    *,
    normalize_by_length: bool = False,
):
    """Mean over a masked tensor; helper for :func:`apply_variant_loss`."""
    if completion_mask is None:
        return token_loss.mean()
    masked = token_loss * completion_mask
    if normalize_by_length:
        lengths = completion_mask.sum(dim=-1).clamp(min=1.0)
        per_sample = masked.sum(dim=-1) / lengths
        return per_sample.mean()
    denom = completion_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def list_variants() -> tuple[str, ...]:
    """Return a sorted tuple of supported variant names (for CLI help)."""
    return tuple(sorted(SUPPORTED_GRPO_VARIANTS))
