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

import logging
import math
from typing import TYPE_CHECKING, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

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


def compute_orpo_term(
    pol_chosen,
    pol_rejected,
    alpha: float,
    chosen_lens=None,
    rejected_lens=None,
):
    """Reference-free odds-ratio preference loss (ORPO).

    Uses the response-log-prob formulation ``-log σ(log(p_w) - log(p_l) +
    log(1-p_l) - log(1-p_w))`` scaled by ``alpha``. Approximates the full
    ORPO loss without the SFT term — caller is expected to mix in SFT via
    its own weight if desired.

    ``pol_chosen`` / ``pol_rejected`` are *summed* sequence log-probs. On a real
    sequence the summed log-prob is ≈ −45, so ``exp()`` underflows to 0 and the
    ``log(1 − p)`` odds-ratio correction collapses to 0 — degenerating the loss
    to a plain log-prob difference. TRL avoids this with ``average_log_prob``;
    pass ``chosen_lens`` / ``rejected_lens`` (response token counts) to
    length-normalise the log-probs here so ``exp()`` yields a real per-token
    probability and the odds-ratio term is meaningful.
    """
    import torch

    if chosen_lens is not None and rejected_lens is not None:
        chosen_lens = torch.clamp(chosen_lens.float(), min=1.0)
        rejected_lens = torch.clamp(rejected_lens.float(), min=1.0)
        lp_chosen = pol_chosen / chosen_lens
        lp_rejected = pol_rejected / rejected_lens
    else:
        lp_chosen = pol_chosen
        lp_rejected = pol_rejected
    log_odds_chosen = lp_chosen - torch.log1p(-torch.exp(lp_chosen).clamp(max=1 - 1e-7))
    log_odds_rejected = lp_rejected - torch.log1p(
        -torch.exp(lp_rejected).clamp(max=1 - 1e-7)
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


def attach_weighted_preference_combine(trainer: object, weights: Mapping[str, float]) -> bool:
    """v0.53.11 #68 — wrap inner trainer's ``compute_loss`` for TRUE weighted blend.

    Replaces the v0.40.1 primary-loss approximation with a per-batch
    weighted sum across all named losses. After the inner trainer's
    ``compute_loss`` runs (computing the primary forward pass), we:

    1. Read the policy + reference summed log-probs that TRL stashes on
       the trainer / batch (``policy_chosen_logps``, ``policy_rejected_logps``,
       ``ref_chosen_logps``, ``ref_rejected_logps``).
    2. For each weighted loss in ``weights``, compute its term via the
       matching ``compute_*_term`` helper.
    3. Combine via :func:`combine_losses` and return the blended scalar.

    When the TRL trainer does not expose the per-batch log-probs (older
    TRL versions, custom forks), we fall back to the v0.40.1 primary-loss
    scaling — defence-in-depth so a TRL upgrade does not crash training.

    Idempotent: re-attaching detects the ``_soup_weighted_combine`` marker.
    """
    validate_weight_compat(weights)
    if not hasattr(trainer, "compute_loss"):
        return False
    original = trainer.compute_loss
    if getattr(original, "_soup_weighted_combine", False):
        return True
    snapshot = dict(weights)

    def wrapped(model, inputs, return_outputs=False, **kwargs):
        result = original(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            primary_loss, outputs = result
        else:
            primary_loss = result
            outputs = None

        # True weighted-sum path: read per-batch logps from inputs OR the
        # trainer's last-batch state.
        pol_chosen = _read_logps(inputs, "policy_chosen_logps")
        if pol_chosen is None:
            pol_chosen = _read_logps(inputs, "chosen_logps")
        pol_rejected = _read_logps(inputs, "policy_rejected_logps")
        if pol_rejected is None:
            pol_rejected = _read_logps(inputs, "rejected_logps")
        ref_chosen = _read_logps(inputs, "reference_chosen_logps")
        if ref_chosen is None:
            ref_chosen = _read_logps(inputs, "ref_chosen_logps")
        ref_rejected = _read_logps(inputs, "reference_rejected_logps")
        if ref_rejected is None:
            ref_rejected = _read_logps(inputs, "ref_rejected_logps")

        terms: dict = {}
        if pol_chosen is not None and pol_rejected is not None:
            # v0.53.11 review fix (security HIGH) — explicit None check
            # before float(). The previous `or 0.1` idiom triggered tensor
            # truth-value evaluation when TRL stored these as zero-element
            # tensors instead of Python floats.
            beta_attr = getattr(trainer, "beta", None)
            beta = float(beta_attr) if beta_attr is not None else 0.1
            for name in snapshot:
                try:
                    if name == "dpo":
                        if ref_chosen is None or ref_rejected is None:
                            logger.debug(
                                "weighted-combine: dpo term skipped — ref logps missing"
                            )
                            continue
                        terms["dpo"] = compute_dpo_term(
                            pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta
                        )
                    elif name == "ipo":
                        if ref_chosen is None or ref_rejected is None:
                            logger.debug(
                                "weighted-combine: ipo term skipped — ref logps missing"
                            )
                            continue
                        terms["ipo"] = compute_ipo_term(
                            pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta
                        )
                    elif name == "simpo":
                        gamma_attr = getattr(trainer, "simpo_gamma", None)
                        gamma = (
                            float(gamma_attr) if gamma_attr is not None else 1.0
                        )
                        terms["simpo"] = compute_simpo_term(
                            pol_chosen, pol_rejected, beta, gamma
                        )
                    elif name == "orpo":
                        alpha_attr = getattr(trainer, "orpo_alpha", None)
                        alpha = (
                            float(alpha_attr) if alpha_attr is not None else 1.0
                        )
                        # Length-normalise when the batch carries response
                        # lengths/labels — otherwise summed log-probs underflow
                        # exp() and the odds-ratio correction degenerates.
                        terms["orpo"] = compute_orpo_term(
                            pol_chosen, pol_rejected, alpha,
                            chosen_lens=_read_lens(inputs, "chosen"),
                            rejected_lens=_read_lens(inputs, "rejected"),
                        )
                    elif name == "bco":
                        # BCO is data-format-incompatible — already rejected
                        # by validate_weight_compat. Defensive skip.
                        continue
                except (TypeError, ValueError) as exc:
                    logger.debug(
                        "weighted-combine: %s term skipped — %s", name, exc
                    )
                    continue

        if len(terms) == len(snapshot):
            # All requested terms computed — return true weighted sum.
            blended = combine_losses(terms, snapshot)
        else:
            # Fallback: primary-loss scaling.
            primary_weight = snapshot.get(_pick_primary(snapshot), 1.0)
            blended = primary_loss * float(primary_weight)
        if return_outputs:
            return blended, outputs
        return blended

    wrapped._soup_weighted_combine = True  # type: ignore[attr-defined]
    trainer.compute_loss = wrapped  # type: ignore[assignment]
    return True


def _read_logps(obj, name: str):
    """Read a log-prob tensor from a mapping / object — TRL inputs vary."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        val = obj.get(name)
        if val is not None:
            return val
    return getattr(obj, name, None)


def _read_lens(obj, prefix: str):
    """Best-effort per-side response-length tensor for length-normalised terms.

    Tries an explicit ``<prefix>_lens`` key first, then derives the count from
    ``<prefix>_labels`` (non-``-100`` tokens = the loss-masked response). Returns
    None when neither is present, in which case the caller keeps the (degenerate)
    summed-log-prob behaviour rather than crashing.
    """
    lens = _read_logps(obj, f"{prefix}_lens")
    if lens is not None:
        return lens
    labels = _read_logps(obj, f"{prefix}_labels")
    if labels is None:
        return None
    try:
        import torch

        if not isinstance(labels, torch.Tensor):
            return None
        return (labels != -100).sum(dim=-1)
    except Exception:  # noqa: BLE001 — best-effort; degrade to no normalisation
        return None


def _pick_primary(weights: Mapping[str, float]) -> str:
    """Return the loss with the highest weight (deterministic on ties)."""
    return max(sorted(weights), key=lambda k: weights[k])


def describe_blend(weights: Optional[Mapping[str, float]]) -> str:
    """Human-readable summary for advisory output."""
    if not weights:
        return "(none)"
    parts = [f"{w:.2f}·{n}" for n, w in sorted(weights.items())]
    return " + ".join(parts)
