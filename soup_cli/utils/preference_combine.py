"""Multi-objective preference loss combiner (v0.40.1 Part B runtime).

Closes the v0.40.0 Part D stub-then-live deferral: ``preference_loss_weights``
now actually combines 2-5 preference losses into one backward pass.

Each preference loss reduces to a pure function of the same forward-pass
quantities: ``policy_chosen_logps`` / ``policy_rejected_logps`` and (for
DPO / IPO) ``ref_chosen_logps`` / ``ref_rejected_logps``. Sharing one
forward pass keeps the cost ~equal to single-loss training.

Compatibility matrix (enforced at config-load + at runtime):

* DPO / IPO  — require a frozen reference model (β log-ratio family).
* SimPO / ORPO — reference-free.
* BCO — uses ``prompt + completion + label`` data, *incompatible* with
  paired ``prompt + chosen + rejected`` batches. Rejected at runtime when
  combined with anything else; users wanting to blend BCO with paired
  losses must run them as separate stages.

The helper itself is dependency-light — it only imports torch lazily so it
can be unit-tested on toy tensors without pulling TRL.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Mapping, Optional

if TYPE_CHECKING:
    import torch  # noqa: F401

PAIRED_LOSSES: frozenset = frozenset({"dpo", "simpo", "orpo", "ipo"})
REF_MODEL_LOSSES: frozenset = frozenset({"dpo", "ipo"})
REF_FREE_LOSSES: frozenset = frozenset({"simpo", "orpo"})
UNPAIRED_LOSSES: frozenset = frozenset({"bco"})


def validate_weight_compat(weights: Mapping[str, float]) -> None:
    """Enforce the BCO-incompatible-with-paired rule at runtime.

    Schema-level validation already restricts keys to the allowlist and
    bounds the sum to 1; this guard catches the data-format mismatch that
    only manifests at training time.
    """
    keys = set(weights.keys())
    if "bco" in keys and (keys - {"bco"}):
        raise ValueError(
            "preference_loss_weights cannot mix 'bco' with paired losses "
            "(dpo/simpo/orpo/ipo). BCO consumes prompt+completion+label rows, "
            "while paired losses consume prompt+chosen+rejected. Run BCO as a "
            "separate task=bco stage."
        )


def needs_reference_model(weights: Mapping[str, float]) -> bool:
    """True iff any active loss in the blend uses a frozen reference model."""
    return bool(set(weights) & REF_MODEL_LOSSES)


def _sigmoid(x):
    import torch

    return torch.sigmoid(x)


def _logsigmoid(x):
    import torch

    return torch.nn.functional.logsigmoid(x)


def compute_dpo_term(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta: float):
    """Standard DPO loss: ``-log σ(β · (Δπ - Δπ_ref))``.

    All log-prob args are summed-token log-likelihoods of the *response*
    only (matching TRL's ``DPOTrainer.compute_reference_log_probs`` shape).
    """
    if ref_chosen is None or ref_rejected is None:
        raise ValueError("DPO requires reference-model log-probs")
    pi_logratio = pol_chosen - pol_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (pi_logratio - ref_logratio)
    return -_logsigmoid(logits).mean()


def compute_ipo_term(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta: float):
    """IPO loss: squared-hinge regularised ``(Δπ - Δπ_ref - 1/(2β))²``."""
    if ref_chosen is None or ref_rejected is None:
        raise ValueError("IPO requires reference-model log-probs")
    if beta <= 0:
        raise ValueError(f"IPO beta must be > 0, got {beta}")
    pi_logratio = pol_chosen - pol_rejected
    ref_logratio = ref_chosen - ref_rejected
    target = 1.0 / (2.0 * beta)
    return ((pi_logratio - ref_logratio - target) ** 2).mean()


def compute_simpo_term(
    pol_chosen,
    pol_rejected,
    beta: float,
    gamma: float,
    chosen_lens=None,
    rejected_lens=None,
):
    """Reference-free length-normalised preference loss (SimPO).

    ``pol_chosen`` / ``pol_rejected`` are summed log-probs; lengths are the
    response token counts used to length-normalise. When ``chosen_lens`` is
    None, falls back to per-sample 1.0 (i.e. acts like length-blind DPO,
    with no reference).
    """
    import torch

    if chosen_lens is None or rejected_lens is None:
        chosen_norm = pol_chosen
        rejected_norm = pol_rejected
    else:
        # Avoid div by zero.
        chosen_lens = torch.clamp(chosen_lens.float(), min=1.0)
        rejected_lens = torch.clamp(rejected_lens.float(), min=1.0)
        chosen_norm = pol_chosen / chosen_lens
        rejected_norm = pol_rejected / rejected_lens
    logits = beta * (chosen_norm - rejected_norm) - gamma
    return -_logsigmoid(logits).mean()


def compute_orpo_term(pol_chosen, pol_rejected, alpha: float):
    """Reference-free odds-ratio preference loss (ORPO).

    Uses the response-log-prob formulation ``-log σ(log(p_w) - log(p_l) +
    log(1-p_l) - log(1-p_w))`` scaled by ``alpha``. Approximates the full
    ORPO loss without the SFT term — caller is expected to mix in SFT via
    its own weight if desired.
    """
    import torch

    log_odds_chosen = pol_chosen - torch.log1p(-torch.exp(pol_chosen).clamp(max=1 - 1e-7))
    log_odds_rejected = pol_rejected - torch.log1p(
        -torch.exp(pol_rejected).clamp(max=1 - 1e-7)
    )
    sigm_term = _logsigmoid(log_odds_chosen - log_odds_rejected)
    return (-alpha * sigm_term).mean()


def combine_losses(
    losses: Dict[str, "torch.Tensor"],
    weights: Mapping[str, float],
) -> "torch.Tensor":
    """Weighted sum of per-loss tensors, validated against ``weights``.

    Raises:
        ValueError: weight dict and loss dict keys differ, or weights don't
            sum to 1 within ±1e-6 (defence-in-depth — schema also enforces).
    """
    if not weights:
        raise ValueError("weights mapping must not be empty")
    if set(losses.keys()) != set(weights.keys()):
        raise ValueError(
            f"loss keys {sorted(losses)} != weight keys {sorted(weights)}"
        )
    # v0.40.1 review fix — defence-in-depth bool rejection (schema also
    # rejects bool, but the runtime path should not silently accept True/False).
    for name, weight in weights.items():
        if isinstance(weight, bool):
            raise TypeError(
                f"preference_loss_weights[{name!r}] must be float, not bool"
            )
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"weights must sum to 1.0 (±1e-6), got {total}")
    out = None
    for name, weight in weights.items():
        contrib = float(weight) * losses[name]
        out = contrib if out is None else out + contrib
    return out


def describe_blend(weights: Optional[Mapping[str, float]]) -> str:
    """Human-readable summary for advisory output."""
    if not weights:
        return "(none)"
    parts = [f"{w:.2f}·{n}" for n, w in sorted(weights.items())]
    return " + ".join(parts)
