"""v0.52.0 Part E — Energy-Based FT (EBFT) + Generalized DPO (GDPO) helpers.

Schema-only release: each algorithm has a closed allowlist of variant names
plus pure validators. Live loss kernels land in v0.52.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# Closed allowlists.
EBFT_VARIANTS: frozenset[str] = frozenset({"structured", "strided"})
GDPO_VARIANTS: frozenset[str] = frozenset({"standard", "length_normalized", "margin"})

_MAX_VARIANT_LEN: int = 32

_MIN_EBFT_TEMP: float = 1e-4
_MAX_EBFT_TEMP: float = 100.0


@dataclass(frozen=True)
class EBFTSpec:
    """Metadata for an EBFT variant. Frozen — immutable."""

    name: str
    description: str
    live_wired: bool


_EBFT_METADATA: Mapping[str, EBFTSpec] = MappingProxyType({
    "structured": EBFTSpec(
        name="structured",
        description="Structured Energy-Based FT (per-token energies)",
        live_wired=True,  # v0.53.2 #135 — kernel + attach hook shipped.
    ),
    "strided": EBFTSpec(
        name="strided",
        description="Strided Energy-Based FT (block-sampled energies)",
        live_wired=True,  # v0.53.2 #135
    ),
})


@dataclass(frozen=True)
class GDPOSpec:
    """Metadata for a GDPO variant. Frozen — immutable."""

    name: str
    description: str
    live_wired: bool


_GDPO_METADATA: Mapping[str, GDPOSpec] = MappingProxyType({
    "standard": GDPOSpec(
        name="standard",
        description="Standard GDPO (general preference objective)",
        live_wired=True,  # v0.53.2 #135 — kernel + DPO attach hook shipped.
    ),
    "length_normalized": GDPOSpec(
        name="length_normalized",
        description="Length-normalized GDPO (SimPO-style normalisation)",
        live_wired=True,  # v0.53.2 #135
    ),
    "margin": GDPOSpec(
        name="margin",
        description="Margin-augmented GDPO (DPO + margin term)",
        live_wired=True,  # v0.53.2 #135
    ),
})


def _validate_variant(name: object, allowed: frozenset[str], label: str) -> str:
    """Shared variant-name validator."""
    if isinstance(name, bool):
        raise TypeError(f"{label} must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(f"{label} must be str, got {type(name).__name__}")
    if not name:
        raise ValueError(f"{label} must be non-empty")
    if "\x00" in name:
        raise ValueError(f"{label} must not contain null bytes")
    if len(name) > _MAX_VARIANT_LEN:
        raise ValueError(
            f"{label} too int (max {_MAX_VARIANT_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(
            f"{label} {name!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_ebft_variant(name: object) -> str:
    """Validate an EBFT variant and return the canonical form."""
    return _validate_variant(name, EBFT_VARIANTS, "ebft_variant")


def validate_gdpo_variant(name: object) -> str:
    """Validate a GDPO variant and return the canonical form."""
    return _validate_variant(name, GDPO_VARIANTS, "gdpo_variant")


def get_ebft_spec(name: str) -> EBFTSpec:
    """Return the frozen :class:`EBFTSpec` for ``name`` or raise."""
    return _EBFT_METADATA[validate_ebft_variant(name)]


def get_gdpo_spec(name: str) -> GDPOSpec:
    """Return the frozen :class:`GDPOSpec` for ``name`` or raise."""
    return _GDPO_METADATA[validate_gdpo_variant(name)]


def validate_ebft_temperature(value: object) -> float:
    """Validate an EBFT temperature scalar in [1e-4, 100]. Rejects bool/NaN."""
    if isinstance(value, bool):
        raise TypeError(f"ebft_temperature must not be bool, got {value!r}")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"ebft_temperature must be float, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(
            f"ebft_temperature must be finite, got {value!r}"
        )
    if fval < _MIN_EBFT_TEMP:
        raise ValueError(
            f"ebft_temperature must be >= {_MIN_EBFT_TEMP}, got {fval}"
        )
    if fval > _MAX_EBFT_TEMP:
        raise ValueError(
            f"ebft_temperature must be <= {_MAX_EBFT_TEMP}, got {fval}"
        )
    return fval


def _check_task_backend(task: object, backend: object) -> None:
    """Shared bool/str guard for cross-compat helpers."""
    for name, value in (("task", task), ("backend", backend)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")


def validate_ebft_compat(*, task: str, backend: str) -> None:
    """Schema-time gate for ``ebft_variant`` — SFT-only, non-MLX."""
    _check_task_backend(task, backend)
    if backend == "mlx":
        raise ValueError(
            "ebft_variant is not supported on backend=mlx in v0.52.0"
        )
    if task != "sft":
        raise ValueError(
            f"ebft_variant requires task='sft'; got task={task!r}"
        )


def validate_gdpo_compat(*, task: str, backend: str) -> None:
    """Schema-time gate for ``gdpo_variant`` — DPO-family-only, non-MLX."""
    _check_task_backend(task, backend)
    if backend == "mlx":
        raise ValueError(
            "gdpo_variant is not supported on backend=mlx in v0.52.0"
        )
    if task not in ("dpo", "preference"):
        raise ValueError(
            f"gdpo_variant requires task in ('dpo', 'preference'); "
            f"got task={task!r}"
        )


def attach_ebft_compute_loss(trainer: object, tcfg: object) -> bool:
    """Wrap ``trainer.compute_loss`` so the EBFT term is added to CE (v0.53.2 #135).

    No-op when ``tcfg.ebft_variant`` is None. Otherwise the original
    ``compute_loss`` is preserved and called first (for the standard SFT
    cross-entropy), and the EBFT kernel is added to the returned loss.

    Idempotent: a sentinel attribute (``_soup_ebft_wrapped``) on the trainer
    prevents double-wrapping when ``setup()`` is called twice on the same
    trainer instance (security review v0.53.2 H1).

    Returns:
        True if the wrap was installed, False otherwise.
    """
    variant = getattr(tcfg, "ebft_variant", None)
    if variant is None:
        return False
    if getattr(trainer, "_soup_ebft_wrapped", False):
        return False
    raw_temp = getattr(tcfg, "ebft_temperature", None)
    temperature = float(raw_temp) if raw_temp is not None else 1.0
    canonical_variant = validate_ebft_variant(variant)
    original = trainer.compute_loss  # type: ignore[attr-defined]

    def wrapped(
        model: object,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: object = None,
    ):
        result = original(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        ce_loss, outputs = result
        labels = inputs.get("labels")
        if labels is None:
            return ce_loss, outputs if return_outputs else ce_loss
        ebft_term = apply_ebft_loss(
            outputs.logits,
            labels,
            variant=canonical_variant,
            temperature=temperature,
        )
        total = ce_loss + ebft_term
        return total, outputs if return_outputs else total

    trainer.compute_loss = wrapped  # type: ignore[attr-defined]
    trainer._soup_ebft_wrapped = True  # type: ignore[attr-defined]
    return True


def attach_gdpo_compute_loss(trainer: object, tcfg: object) -> bool:
    """Wrap TRL's ``DPOTrainer.dpo_loss`` so a GDPO variant is used (v0.53.2 #135).

    No-op when ``tcfg.gdpo_variant`` is None. Replaces the trainer's
    ``dpo_loss`` method (the stable TRL hook returning losses, chosen rewards,
    rejected rewards) with a thin wrapper that calls :func:`apply_gdpo_loss`.

    Returns:
        True if the wrap was installed, False otherwise.
    """
    variant = getattr(tcfg, "gdpo_variant", None)
    if variant is None:
        return False
    if getattr(trainer, "_soup_gdpo_wrapped", False):
        return False
    canonical_variant = validate_gdpo_variant(variant)
    raw_beta = getattr(tcfg, "dpo_beta", None)
    beta = float(raw_beta) if raw_beta is not None else 0.1
    raw_margin = getattr(tcfg, "dpo_margin", None)
    margin = float(raw_margin) if raw_margin is not None else 0.0
    original = getattr(trainer, "dpo_loss", None)
    if original is None:
        return False

    def wrapped(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        chosen_lens=None,
        rejected_lens=None,
        *args: object,
        **kwargs: object,
    ):
        # length_normalized variant pulls lengths from explicit args; TRL's
        # callers either pass them positionally (newer TRL with length-norm
        # support) or via **kwargs.
        lens_c = chosen_lens if chosen_lens is not None else kwargs.get("chosen_lens")
        lens_r = (
            rejected_lens
            if rejected_lens is not None
            else kwargs.get("rejected_lens")
        )
        loss = apply_gdpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=reference_chosen_logps,
            ref_rejected_logps=reference_rejected_logps,
            variant=canonical_variant,
            beta=beta,
            margin=margin,
            chosen_lens=lens_c,
            rejected_lens=lens_r,
        )
        # Recreate TRL's standard return shape: per-sample losses, chosen
        # rewards, rejected rewards. We broadcast the mean to a per-sample
        # tensor and compute simple rewards = beta * (pi - ref).
        per_sample_loss = loss.expand_as(policy_chosen_logps)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()
        return per_sample_loss, chosen_rewards, rejected_rewards

    trainer.dpo_loss = wrapped  # type: ignore[attr-defined]
    trainer._soup_gdpo_wrapped = True  # type: ignore[attr-defined]
    return True


def apply_ebft_loss(
    logits,
    labels,
    *,
    variant: str,
    temperature: float,
    stride: int = 4,
    ignore_index: int = -100,
):
    """Energy-Based Fine-Tuning loss kernel (v0.53.2 #135).

    EBFT treats per-token logits as energies (lower = more probable) and
    penalises high-energy correct tokens. Two variants:

    * ``structured`` — per-token energy summed over every non-ignored
      position, divided by the count.
    * ``strided`` — same kernel but only every ``stride``-th position
      contributes (faster on int sequences).

    The temperature scales the softmax sharpness: lower temperature
    sharpens the implicit distribution, producing harsher gradients.

    Args:
        logits: ``(batch, seq, vocab)`` float tensor.
        labels: ``(batch, seq)`` int tensor; ``ignore_index`` entries
            contribute nothing.
        variant: ``"structured"`` or ``"strided"``.
        temperature: in ``[1e-4, 100]`` (validated).
        stride: positive int used only for the strided variant.
        ignore_index: label id treated as padding (default ``-100``).

    Returns:
        Scalar tensor (zero-dim) carrying gradient.

    Raises:
        ValueError: shape mismatch, unknown variant, invalid temperature,
            non-positive stride.
        TypeError: ``stride`` not int or is bool.
    """
    import torch

    canonical_variant = validate_ebft_variant(variant)
    temp = validate_ebft_temperature(temperature)
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError(f"stride must be int, got {type(stride).__name__}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if logits.dim() != 3:
        raise ValueError(
            f"logits must be (batch, seq, vocab); got shape {tuple(logits.shape)}"
        )
    if labels.dim() != 2:
        raise ValueError(
            f"labels must be (batch, seq); got shape {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits/labels shape mismatch: {tuple(logits.shape[:2])} vs "
            f"{tuple(labels.shape)}"
        )

    batch, seq, _ = logits.shape
    valid_mask = labels.ne(ignore_index)
    if canonical_variant == "strided":
        positions = torch.zeros_like(valid_mask)
        positions[:, ::stride] = True
        valid_mask = valid_mask & positions

    if not valid_mask.any():
        # Preserve grad path by multiplying logits by zero.
        return (logits.sum() * 0.0).reshape(())

    # Per-token energy: negative log-softmax of the correct label, scaled by
    # 1 / temperature. Equivalent to a temperature-scaled cross-entropy.
    safe_labels = labels.clamp(min=0)
    log_probs = torch.log_softmax(logits / temp, dim=-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    nll = nll * valid_mask.to(nll.dtype)
    denom = valid_mask.sum().clamp(min=1).to(nll.dtype)
    return nll.sum() / denom


def apply_gdpo_loss(
    *,
    policy_chosen_logps,
    policy_rejected_logps,
    variant: str,
    beta: float,
    ref_chosen_logps=None,
    ref_rejected_logps=None,
    chosen_lens=None,
    rejected_lens=None,
    margin: float = 0.0,
):
    """Generalized DPO loss kernel (v0.53.2 #135).

    Three variants:

    * ``standard`` — closed-form DPO: ``-log σ(β·(Δπ - Δref))``.
      Requires reference log-probs.
    * ``length_normalized`` — SimPO-style: ``-log σ(β·(π_w/L_w - π_l/L_l))``;
      requires ``chosen_lens`` and ``rejected_lens``.
    * ``margin`` — DPO with an explicit margin: ``-log σ(β·(Δπ - Δref) -
      margin)``. Requires reference log-probs.

    All log-prob tensors are 1-D, batch-shaped (one entry per pair).

    Returns:
        Scalar tensor (mean across the batch) carrying gradient.

    Raises:
        ValueError: unknown variant, missing reference/lengths, beta bounds,
            shape mismatch, non-finite margin.
        TypeError: ``beta`` or ``margin`` bool / non-numeric.
    """
    import torch

    canonical_variant = validate_gdpo_variant(variant)

    if isinstance(beta, bool):
        raise TypeError(f"beta must not be bool, got {beta!r}")
    if not isinstance(beta, (int, float)):
        raise TypeError(f"beta must be float, got {type(beta).__name__}")
    fbeta = float(beta)
    if not math.isfinite(fbeta) or fbeta <= 0.0 or fbeta > 100.0:
        raise ValueError(f"beta must be in (0, 100], got {beta!r}")

    if isinstance(margin, bool):
        raise TypeError(f"margin must not be bool, got {margin!r}")
    if not isinstance(margin, (int, float)):
        raise TypeError(f"margin must be float, got {type(margin).__name__}")
    fmargin = float(margin)
    if not math.isfinite(fmargin):
        raise ValueError(f"margin must be finite, got {margin!r}")

    if policy_chosen_logps.shape != policy_rejected_logps.shape:
        raise ValueError(
            f"policy chosen/rejected shape mismatch: "
            f"{tuple(policy_chosen_logps.shape)} vs "
            f"{tuple(policy_rejected_logps.shape)}"
        )

    if canonical_variant in ("standard", "margin"):
        if ref_chosen_logps is None or ref_rejected_logps is None:
            raise ValueError(
                f"variant {canonical_variant!r} requires reference log-probs "
                "(ref_chosen_logps + ref_rejected_logps)"
            )
        if (
            ref_chosen_logps.shape != policy_chosen_logps.shape
            or ref_rejected_logps.shape != policy_rejected_logps.shape
        ):
            raise ValueError(
                "reference log-probs shape mismatch with policy log-probs"
            )
        pi_delta = policy_chosen_logps - policy_rejected_logps
        ref_delta = ref_chosen_logps - ref_rejected_logps
        logits = fbeta * (pi_delta - ref_delta)
        if canonical_variant == "margin":
            logits = logits - fmargin
        return -torch.nn.functional.logsigmoid(logits).mean()

    # length_normalized
    if chosen_lens is None or rejected_lens is None:
        raise ValueError(
            "variant 'length_normalized' requires chosen_lens and "
            "rejected_lens"
        )
    chosen_lens_f = torch.clamp(chosen_lens.float(), min=1.0)
    rejected_lens_f = torch.clamp(rejected_lens.float(), min=1.0)
    chosen_norm = policy_chosen_logps / chosen_lens_f
    rejected_norm = policy_rejected_logps / rejected_lens_f
    logits = fbeta * (chosen_norm - rejected_norm)
    return -torch.nn.functional.logsigmoid(logits).mean()
