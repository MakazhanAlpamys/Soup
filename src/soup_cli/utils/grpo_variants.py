"""GRPO objective variants — v0.50.0 Part A.

Closed allowlist of GRPO-family RL objectives shipped for parity with
unsloth + axolotl. Schema-load-time validation + metadata only in v0.50.0;
live loss-function wiring lands in v0.50.1 (mirrors v0.27.0 MII /
v0.37.0 multipack / v0.41.0 LLaMA Pro / v0.45.0 plugins / v0.48.0 curriculum
stub-then-live pattern).

Variants:
- gspo         : Group Sequence Policy Optimization (Qwen)
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
        description="Group Sequence Policy Optimization",
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
    - ``beta``: KL coefficient (``GRPOConfig.beta``, i.e. ``grpo_beta``).
      Every variant applies it -- see *KL penalty* below; ``0`` means no KL
      term.
    - ``delta``: symmetric clipping radius (used by ``two_sided`` and, as the
      sequence-level clipping radius, by ``gspo``).
    - ``completion_mask``: optional ``[B, T]`` 0/1 mask (1 where token is in
      the completion). When supplied, length-normalising variants
      (``bnpo``) divide by ``mask.sum(-1).clamp(min=1)``.
    - ``reference_logp``: the frozen reference policy's per-token log-probs,
      shape ``[B, T]`` (trl's ``ref_per_token_logps``). Required when
      ``beta > 0``; not read at all when ``beta == 0``.

    Returns a scalar torch tensor (or ``None`` for the standard variant —
    callers should fall through to the existing TRL ``GRPOTrainer.compute_loss``,
    which applies ``beta`` itself).

    KL penalty (#1232). With ``beta > 0`` every variant adds trl 0.29's
    per-token KL estimate against the reference,
    ``kl_t = exp(ref_t - logp_t) - (ref_t - logp_t) - 1``, as
    ``per_token_loss + beta * kl_t`` ahead of the variant's own reduction --
    where ``GRPOTrainer._compute_loss`` puts it -- so ``beta`` weighs the KL
    against the policy term per token exactly as it does in trl. A policy
    equal to its reference adds exactly zero. ``beta > 0`` with
    ``reference_logp=None`` raises ``ValueError`` instead of dropping the
    term; ``beta == 0`` skips it and reproduces the pre-#1232 loss bit for
    bit.

    The math is intentionally minimal — these are *reference* kernels for
    routing + unit tests. Production correctness on multi-billion-param
    models will be validated by the v0.53.11 smoke run on SmolLM2-135M
    + gsm8k. Each variant kernel matches the canonical formula from the
    unsloth / axolotl reference implementations:

    - ``gspo``: Group Sequence Policy Optimization (Qwen, arXiv:2507.18071).
      Sequence-level importance ratio length-normalized by completion length:
      ``s_i = exp((1/|y_i|) * sum (log p_new - log p_old))``, optimized with
      sequence-level surrogate clipping: ``-min(s * A, clip(s, 1-eps, 1+eps) * A)``.
      KL: each sequence's masked mean ``kl_t`` joins that sequence's loss
      before the mean over non-empty sequences.
    - ``dapo``: decoupled-clip — uses asymmetric clipping bounds
      ``[1-eps_lo, 1+eps_hi]`` (here ``eps_lo=0.2, eps_hi=0.28`` per the
      paper). KL: per token, inside the masked token mean over the batch.
    - ``dr_grpo``: GRPO without length normalisation (the doubly-robust
      bias-correction term is left to v0.53.12+; the schema gate keeps
      misconfigured runs out). KL: per token, inside the masked token sum
      that is then averaged over the batch, like the policy term.
    - ``bnpo``: length-normalised PPO loss — divides by ``mask.sum(-1)``.
      KL: per token, inside the per-sequence length-normalised mean.
    - ``two_sided``: symmetric clipping with operator-supplied ``delta``;
      ``delta=None`` raises ``ValueError`` (schema requires it). KL: per
      token, inside the masked token mean over the batch.
    - ``rft``: rejection-sampling fine-tuning — only positive-advantage
      samples contribute (zero-advantage rows masked out before mean). KL:
      on those same accepted tokens only, over the same denominator, so a
      batch with no accepted completion still contributes zero.
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

    # #1232 — at beta == 0 the reference is not read at all, so the loss is the
    # pre-#1232 one bit for bit and a reference whose exp() overflows cannot
    # leak NaN into it through a 0 * inf.
    per_token_kl = (
        _per_token_kl(normalised, beta_f, logp_new, reference_logp) if beta_f > 0.0 else None
    )

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
        # Group Sequence Policy Optimization (Qwen, arXiv:2507.18071):
        # Sequence-level importance ratio length-normalized by completion length:
        #   s_i = exp( 1/|y_i| * sum_{t=1}^{|y_i|} (log p_new - log p_old) )
        # Optimized with sequence-level surrogate clipping:
        #   loss = -min(s_i * A_i, clip(s_i, 1-eps, 1+eps) * A_i).mean()
        if delta is not None:
            eps = validate_grpo_delta(delta)
        else:
            eps = 0.2

        if advantages.dim() == 1:
            adv_seq = advantages
        elif advantages.dim() == 2 and advantages.size(-1) == 1:
            adv_seq = advantages.squeeze(-1)
        elif completion_mask is not None:
            adv_seq = (advantages * completion_mask).sum(dim=-1) / completion_mask.sum(
                dim=-1
            ).clamp(min=1.0)
        else:
            adv_seq = advantages.mean(dim=-1)

        if completion_mask is not None:
            mask = completion_mask.to(dtype=log_ratio.dtype)
            lengths = mask.sum(dim=-1)
            valid_seq_mask = (lengths > 0).to(dtype=log_ratio.dtype)
            seq_log_ratio = (log_ratio * mask).sum(dim=-1) / lengths.clamp(min=1.0)
        else:
            seq_log_ratio = log_ratio.mean(dim=-1)
            valid_seq_mask = torch.ones_like(seq_log_ratio)

        seq_ratio = torch.exp(seq_log_ratio)
        surr1 = seq_ratio * adv_seq
        surr2 = torch.clamp(seq_ratio, min=1.0 - eps, max=1.0 + eps) * adv_seq
        seq_loss = -torch.min(surr1, surr2)
        if per_token_kl is not None:
            # Sequence level, like the policy term: trl's sequence-level
            # importance sampling adds beta * kl_t to the per-sequence loss and
            # takes each sequence's masked token mean.
            if completion_mask is not None:
                seq_kl = (per_token_kl * mask).sum(dim=-1) / lengths.clamp(min=1.0)
            else:
                seq_kl = per_token_kl.mean(dim=-1)
            seq_loss = seq_loss + beta_f * seq_kl
        denom = valid_seq_mask.sum().clamp(min=1.0)
        return (seq_loss * valid_seq_mask).sum() / denom

    if normalised == "dapo":
        # Decoupled clip — asymmetric bounds.
        eps_lo, eps_hi = 0.2, 0.28
        clipped = torch.clamp(ratio, min=1 - eps_lo, max=1 + eps_hi)
        # PPO surrogate: min(ratio * A, clipped * A).
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        token_loss = _with_kl(token_loss, per_token_kl, beta_f)
        return _masked_mean(token_loss, completion_mask)

    if normalised == "dr_grpo":
        # No length normalisation — sum across tokens then mean across batch.
        token_loss = -(ratio * advantages_2d)
        token_loss = _with_kl(token_loss, per_token_kl, beta_f)
        if completion_mask is not None:
            token_loss = token_loss * completion_mask
        # sum-over-tokens, mean-over-batch (no division by completion length)
        return token_loss.sum(dim=-1).mean()

    if normalised == "bnpo":
        # Batch-normalised PPO with length-normalisation.
        eps = 0.2
        clipped = torch.clamp(ratio, min=1 - eps, max=1 + eps)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        token_loss = _with_kl(token_loss, per_token_kl, beta_f)
        return _masked_mean(token_loss, completion_mask, normalize_by_length=True)

    if normalised == "two_sided":
        # Symmetric clipping at [1-delta, 1+delta].
        clipped = torch.clamp(ratio, min=1 - delta_f, max=1 + delta_f)
        token_loss = -torch.min(ratio * advantages_2d, clipped * advantages_2d)
        token_loss = _with_kl(token_loss, per_token_kl, beta_f)
        return _masked_mean(token_loss, completion_mask)

    if normalised == "rft":
        # Rejection sampling fine-tuning: only positive-advantage tokens
        # contribute to the gradient.
        positive_mask = (advantages_2d > 0).to(logp_new.dtype)
        if completion_mask is not None:
            positive_mask = positive_mask * completion_mask
        # Standard SFT-style negative log-likelihood weighted by positive mask.
        token_loss = -(logp_new * positive_mask)
        if per_token_kl is not None:
            # rft trains on the accepted completions only, so its KL covers the
            # same tokens over the same denominator (#1232): a batch with no
            # accepted completion still contributes zero.
            token_loss = token_loss + beta_f * per_token_kl * positive_mask
        denom = positive_mask.sum().clamp(min=1.0)
        return token_loss.sum() / denom

    # Defensive fallback — schema rejects everything outside the allowlist.
    raise ValueError(f"Unhandled grpo_variant={normalised!r}")


def _per_token_kl(variant: str, beta: float, logp_new, reference_logp):
    """trl 0.29's per-token KL estimate against the frozen reference (#1232).

    ``exp(ref - logp) - (ref - logp) - 1``: the expression
    ``GRPOTrainer._compute_loss`` weights by ``beta``. It is non-negative and
    exactly zero where the policy equals the reference. The reference is
    detached because it is a frozen policy, not a parameter.
    """
    import torch  # lazy import — utility module is dependency-light

    if reference_logp is None:
        # Inside the trainer this text reaches the user through the #159
        # fallback warning, where beta=0 is no remedy (grpo_beta must be > 0),
        # so it names the batch key trl should have filled instead.
        raise ValueError(
            f"grpo_variant={variant!r} with beta={beta} needs reference_logp, the "
            "frozen reference policy's per-token log-probs, for its KL penalty "
            "(#1232); in a trl batch that is 'ref_per_token_logps', and it is missing"
        )
    if tuple(reference_logp.shape) != tuple(logp_new.shape):
        raise ValueError(
            f"reference_logp shape {tuple(reference_logp.shape)} does not match "
            f"logp_new shape {tuple(logp_new.shape)}"
        )
    ref_minus_logp = reference_logp.detach() - logp_new
    return torch.exp(ref_minus_logp) - ref_minus_logp - 1


def compute_variant_mean_kl(
    variant: str,
    *,
    logp_new,
    reference_logp,
    completion_mask=None,
    advantages=None,
):
    """Compute the batch-mean per-token KL term for the given variant (#1263).

    For variants matching trl's reduction (gspo, dapo, dr_grpo, bnpo, two_sided),
    computes the completion-token masked mean of trl 0.29's per-token KL estimator:
    ``(per_token_kl * mask).sum() / mask.sum().clamp(min=1.0)``.

    For ``rft``, averages only over accepted completions (positive advantage tokens)
    using the same denominator as its loss:
    ``(per_token_kl * accepted_mask).sum() / accepted_mask.sum().clamp(min=1.0)``.
    """
    normalised = variant.strip().lower()
    per_token_kl = _per_token_kl(normalised, 1.0, logp_new, reference_logp)

    if normalised == "rft":
        if advantages is None:
            raise ValueError("rft variant requires advantages to identify accepted tokens")
        adv_2d = advantages.unsqueeze(-1) if advantages.dim() == 1 else advantages
        pos_mask = (adv_2d > 0).to(dtype=per_token_kl.dtype)
        if completion_mask is not None:
            pos_mask = pos_mask * completion_mask.to(dtype=per_token_kl.dtype)
        denom = pos_mask.sum().clamp(min=1.0)
        return (per_token_kl * pos_mask).sum() / denom

    if completion_mask is not None:
        c_mask = completion_mask.to(dtype=per_token_kl.dtype)
        denom = c_mask.sum().clamp(min=1.0)
        return (per_token_kl * c_mask).sum() / denom

    return per_token_kl.mean()


def _with_kl(token_loss, per_token_kl, beta: float):
    """``token_loss + beta * kl_t``, trl's placement; a no-op without a KL term."""
    if per_token_kl is None:
        return token_loss
    return token_loss + beta * per_token_kl


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
