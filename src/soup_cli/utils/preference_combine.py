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
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch  # noqa: F401

PAIRED_LOSSES: frozenset = frozenset({"dpo", "simpo", "orpo", "ipo"})
REF_MODEL_LOSSES: frozenset = frozenset({"dpo", "ipo"})
REF_FREE_LOSSES: frozenset = frozenset({"simpo", "orpo"})
UNPAIRED_LOSSES: frozenset = frozenset({"bco"})

#: The beta every Soup preference wrapper builds with. Soup has no ``beta``
#: schema field — ``simpo.py`` leaves CPOConfig's own default in place and
#: ``dpo.py`` likewise — so this is the value trl's `CPOConfig` / `DPOTrainer`
#: carry, NOT ``training.orpo_beta`` (``orpo.py:211`` uses that one). Reading
#: beta off the primary trainer made the blend's objective depend on which loss
#: happened to be named first (#1425 review); ``tests/test_v05311.py`` pins this
#: constant against trl so a default change cannot drift silently.
DEFAULT_BETA: float = 0.1

#: Fallbacks for the two knobs `params` may omit. These mirror the schema
#: defaults (`schema.py`: `simpo_gamma=0.5`, `orpo_beta=0.1`) so a blend
#: attached without a config — an existing test harness, say — lands on the
#: same objective the standalone loss would use.
DEFAULT_SIMPO_GAMMA: float = 0.5
DEFAULT_ORPO_ALPHA: float = 0.1


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
    log(1-p_l) - log(1-p_w))`` scaled by ``alpha``.

    This is the odds-ratio term only: it leaves out the NLL term that a
    standalone ``preference_loss: orpo`` run adds (trl computes that NLL over
    prompt plus response, ``orpo_trainer.py``). A blend of ``simpo`` and
    ``orpo`` therefore trains neither loss's likelihood term — see the
    "Weighted Multi-Objective Preference Loss" section in ``docs/training.md``
    and #1425.

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


def blend_loss_params(cfg) -> dict:
    """Each blended loss's own hyperparameters, read from the config (#1425).

    The blend used to take ``beta`` / ``simpo_gamma`` / ``orpo_alpha`` off the
    primary trainer, which is whichever loss has the larger weight — so
    ``orpo_beta`` fed SimPO's beta on an ORPO-primary blend, ORPO's alpha was
    always ``1.0`` because no trainer has an ``orpo_alpha`` attribute, and the
    same weights written in a different YAML order trained a different
    objective. Config is the only place these are unambiguous.

    ``training.orpo_beta`` is ORPO's odds-ratio weight, so it becomes ORPO's
    ``alpha`` here; ``training.simpo_gamma`` is SimPO's margin. Beta has no
    schema field — every Soup wrapper leaves trl's default in place — so it
    comes from :data:`DEFAULT_BETA` rather than off a trainer.
    """
    tcfg = getattr(cfg, "training", None)
    if tcfg is None:  # defensive — a config always has `training`
        return {}
    return {
        "dpo": {"beta": DEFAULT_BETA},
        "ipo": {"beta": DEFAULT_BETA},
        "simpo": {
            "beta": DEFAULT_BETA,
            "gamma": float(getattr(tcfg, "simpo_gamma", DEFAULT_SIMPO_GAMMA)),
        },
        "orpo": {"alpha": float(getattr(tcfg, "orpo_beta", DEFAULT_ORPO_ALPHA))},
    }


def attach_weighted_preference_combine(
    trainer: object,
    weights: Mapping[str, float],
    params: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> bool:
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

    When the TRL trainer does not expose the per-batch log-probs (which is every
    trl 0.29 trainer — #1425), this raises, naming the terms it could not
    compute. It used to fall back to the v0.40.1 primary-loss scaling, which
    trained only the highest-weighted loss while reporting a blended number.

    ``params`` supplies each loss's own hyperparameters, keyed by loss name
    (``{"simpo": {"beta": ..., "gamma": ...}, "orpo": {"alpha": ...}}``). It
    must come from the config, not from the trainer: the trainer is whichever
    loss has the larger weight, so reading ``beta`` / ``simpo_gamma`` off it
    made the same weights in a different YAML order train a different
    objective (#1425 review). Anything not supplied falls back to
    :data:`DEFAULT_BETA`.

    Idempotent: re-attaching detects the ``_soup_weighted_combine`` marker.
    """
    validate_weight_compat(weights)
    if not hasattr(trainer, "compute_loss"):
        return False
    original = trainer.compute_loss
    if getattr(original, "_soup_weighted_combine", False):
        return True
    snapshot = dict(weights)
    loss_params: Dict[str, Mapping[str, float]] = {k: dict(v) for k, v in (params or {}).items()}

    def wrapped(model, inputs, return_outputs=False, **kwargs):
        result = original(model, inputs, return_outputs=return_outputs, **kwargs)
        # #1425: the primary loss is no longer a fallback (it is what made a
        # blend train one loss while reporting several). It is only unpacked
        # to pass `outputs` through when the caller asked for it.
        outputs = result[1] if return_outputs else None

        # True weighted-sum path: the per-batch logps, else the ones captured
        # from the primary trainer's own forward pass (#1425).
        captured = getattr(trainer, "_soup_policy_logps", None) or {}
        pol_chosen = captured.get("policy_chosen_logps")
        if pol_chosen is None:
            pol_chosen = _read_logps(inputs, "policy_chosen_logps")
        if pol_chosen is None:
            pol_chosen = _read_logps(inputs, "chosen_logps")
        pol_rejected = captured.get("policy_rejected_logps")
        if pol_rejected is None:
            pol_rejected = _read_logps(inputs, "policy_rejected_logps")
        if pol_rejected is None:
            pol_rejected = _read_logps(inputs, "rejected_logps")
        ref_chosen = _read_logps(inputs, "reference_chosen_logps")
        if ref_chosen is None:
            ref_chosen = _read_logps(inputs, "ref_chosen_logps")
        ref_rejected = _read_logps(inputs, "reference_rejected_logps")
        if ref_rejected is None:
            ref_rejected = _read_logps(inputs, "ref_rejected_logps")
        # Lengths come from the capture when it ran, else from the batch.
        chosen_lens = captured.get("chosen_lens")
        if chosen_lens is None:
            chosen_lens = _read_lens(inputs, "chosen")
        rejected_lens = captured.get("rejected_lens")
        if rejected_lens is None:
            rejected_lens = _read_lens(inputs, "rejected")

        terms: dict = {}
        if pol_chosen is not None and pol_rejected is not None:
            # #1425 review: every hyperparameter comes from `params` (the
            # config), never from the primary trainer. The trainer is whichever
            # loss has the larger weight, so `getattr` here made {simpo: 0.5,
            # orpo: 0.5} and {orpo: 0.5, simpo: 0.5} train different objectives.
            for name in snapshot:
                cfg_for_loss = loss_params.get(name, {})
                try:
                    if name == "dpo":
                        if ref_chosen is None or ref_rejected is None:
                            logger.debug(
                                "weighted-combine: dpo term skipped — ref logps missing"
                            )
                            continue
                        terms["dpo"] = compute_dpo_term(
                            pol_chosen,
                            pol_rejected,
                            ref_chosen,
                            ref_rejected,
                            float(cfg_for_loss.get("beta", DEFAULT_BETA)),
                        )
                    elif name == "ipo":
                        if ref_chosen is None or ref_rejected is None:
                            logger.debug(
                                "weighted-combine: ipo term skipped — ref logps missing"
                            )
                            continue
                        terms["ipo"] = compute_ipo_term(
                            pol_chosen,
                            pol_rejected,
                            ref_chosen,
                            ref_rejected,
                            float(cfg_for_loss.get("beta", DEFAULT_BETA)),
                        )
                    elif name == "simpo":
                        terms["simpo"] = compute_simpo_term(
                            pol_chosen,
                            pol_rejected,
                            float(cfg_for_loss.get("beta", DEFAULT_BETA)),
                            float(cfg_for_loss.get("gamma", DEFAULT_SIMPO_GAMMA)),
                            chosen_lens=chosen_lens,
                            rejected_lens=rejected_lens,
                        )
                    elif name == "orpo":
                        # `orpo_beta` is ORPO's odds-ratio weight; `orpo_alpha`
                        # never existed on any trainer, so the term used 1.0
                        # regardless of the config.
                        terms["orpo"] = compute_orpo_term(
                            pol_chosen,
                            pol_rejected,
                            float(cfg_for_loss.get("alpha", DEFAULT_ORPO_ALPHA)),
                            chosen_lens=chosen_lens,
                            rejected_lens=rejected_lens,
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
            # #1425: trl 0.29 puts no per-sequence log-probs on the batch, so
            # `terms` is empty on every step. Scaling the primary loss by its
            # weight trained only the highest-weighted loss and said nothing
            # about it, while setup had already printed
            # "0.70·dpo + 0.30·simpo". Refuse instead, naming the terms.
            missing = sorted(set(snapshot) - set(terms))
            computed = sorted(terms)
            raise ValueError(
                "preference_loss_weights: cannot compute "
                f"{', '.join(missing)} on this step"
                + (f" (computed: {', '.join(computed)})" if computed else "")
                + f". Weights requested: {', '.join(sorted(snapshot))}. The "
                "underlying trl exposes no per-sequence log-probs on the batch, "
                "so the blend cannot be evaluated; training would silently use "
                "the primary loss alone. Remove training.preference_loss_weights "
                "and set training.preference_loss to the single loss you want."
            )
        if return_outputs:
            return blended, outputs
        return blended

    wrapped._soup_weighted_combine = True  # type: ignore[attr-defined]
    trainer.compute_loss = wrapped  # type: ignore[assignment]
    return True


def attach_policy_logp_capture(trainer: object) -> bool:
    """Record the policy log-probs trl already computes (#1425).

    trl 0.29's CPO (the SimPO path) and ORPO trainers compute per-sequence
    chosen/rejected log-probs inside ``concatenated_forward`` and hand them
    straight back to ``get_batch_loss_metrics``; nothing puts them where a
    weighted blend can read them, which is why every blend on trl 0.29 fell
    through to the primary loss alone. trl's DPO/IPO trainers compute theirs
    inline in ``compute_loss`` and publish only means, so they have no such
    hook and return ``False`` here — their blends keep refusing.

    The captured values are converted to **summed** log-probs using the batch's
    own per-side ``chosen_labels`` / ``rejected_labels``, because the kernels in
    this module take sums plus lengths. Whether trl handed over sums or averages
    is not inspected: deriving the lengths and multiplying back gives the sum
    either way. That multiply-back is only correct for the paths where trl
    averaged — ``cpo_trainer.py`` averages for ``loss_type`` ``ipo``/``simpo``,
    and ``orpo_trainer.py`` always averages. Soup's SimPO wrapper always builds
    ``loss_type="simpo"``, so the averaging path is the live one today; a future
    wrapper using ``sigmoid`` or ``hinge`` would return sums and this would
    double-count the lengths.

    Returns True when the capture is in place.
    """
    if not hasattr(trainer, "concatenated_forward"):
        return False
    original = trainer.concatenated_forward
    if getattr(original, "_soup_logp_capture", False):
        return True

    def capturing(model, batch, *args, **kwargs):
        out = original(model, batch, *args, **kwargs)
        try:
            chosen_logps, rejected_logps = out[0], out[1]
        except (TypeError, IndexError, KeyError):  # pragma: no cover — trl shape drift
            # IndexError/TypeError: not a sequence (or too short). KeyError: a
            # dict-shaped return, which older trl DPOTrainers used — let it fall
            # through to the blend's refusal rather than raise out of a forward.
            return out
        _record_policy_logps(trainer, batch, chosen_logps, rejected_logps)
        return out

    capturing._soup_logp_capture = True
    trainer.concatenated_forward = capturing
    return True


def _response_lengths(batch: Any, side: str) -> Any:
    """Per-example response token count: everything not masked with ``-100``."""
    labels = batch.get(f"{side}_labels") if hasattr(batch, "get") else None
    if labels is None:
        return None
    try:
        return (labels != -100).sum(dim=-1)
    except (AttributeError, TypeError):  # pragma: no cover — non-tensor labels
        return None


def _record_policy_logps(trainer, batch, chosen_logps, rejected_logps) -> None:
    """Store summed policy log-probs and the lengths that produced them."""
    chosen_lens = _response_lengths(batch, "chosen")
    rejected_lens = _response_lengths(batch, "rejected")
    if chosen_lens is None or rejected_lens is None:
        # No per-side labels: keep what trl gave us and say we did not normalise.
        trainer._soup_policy_logps = {
            "policy_chosen_logps": chosen_logps,
            "policy_rejected_logps": rejected_logps,
            "chosen_lens": None,
            "rejected_lens": None,
        }
        return
    # trl averaged; the kernels take sums, so put the length back in.
    trainer._soup_policy_logps = {
        "policy_chosen_logps": chosen_logps * chosen_lens.to(chosen_logps.dtype),
        "policy_rejected_logps": rejected_logps * rejected_lens.to(
            rejected_logps.dtype,
        ),
        "chosen_lens": chosen_lens,
        "rejected_lens": rejected_lens,
    }


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


def describe_blend(weights: Optional[Mapping[str, float]]) -> str:
    """Human-readable summary for advisory output."""
    if not weights:
        return "(none)"
    parts = [f"{w:.2f}·{n}" for n, w in sorted(weights.items())]
    return " + ".join(parts)
