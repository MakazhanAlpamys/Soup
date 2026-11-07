"""MLX SimPO trainer — reference-free, length-normalised preference optimization on Apple Silicon.

The objective is TRL 0.29.1's ``experimental/cpo/cpo_trainer.py`` with
``loss_type="simpo"``, the trainer Soup's transformers backend runs for
``task: simpo`` (``trainer/simpo.py`` forwards ``simpo_gamma`` and
``cpo_alpha`` and no beta):

- ``a``, ``b``: the **mean** log-probability of the chosen / rejected completion
  tokens (``get_batch_logps(..., average_log_prob=True)``);
- preference term ``-logsigmoid(beta * (a - b) - simpo_gamma)`` per pair, the
  margin outside the temperature (TRL writes it ``beta * (a - b - gamma / beta)``);
  ``label_smoothing`` is ``CPOConfig``'s default 0, so its second term vanishes;
- ``beta`` is ``CPOConfig.beta``'s default, :data:`SIMPO_BETA`; Soup exposes no
  knob for it (``dpo_beta`` / ``orpo_beta`` are other objectives');
- loss = ``mean(preference) + cpo_alpha * NLL``, where the NLL is a token-mean
  cross-entropy over the chosen **completion** tokens only (CPO's labels mask
  the prompt with ``-100``), across the batch.

Tokenization is ORPO's: TRL's CPO and ORPO ``tokenize_row`` /
``build_tokenized_answer`` are the same code, so the rows, the masks and Soup's
length cap come from :mod:`soup_cli.trainer.mlx_orpo`. The completion mask is
both the score mask and the NLL mask. The loop is MLX DPO's.
"""

from __future__ import annotations

from typing import Any

from soup_cli.trainer.mlx_orpo import (
    ORPO_TRUNCATION_MODE,
    MLXORPOTrainerWrapper,
    collate_orpo,
    orpo_terms,
)

#: ``CPOConfig.beta``'s default in TRL 0.29.1, the SimPO temperature Soup's
#: transformers backend trains with.
SIMPO_BETA = 0.1


def simpo_terms(model: Any, tokens: Any, completion: Any, *,
                head_split: Any = None) -> tuple[Any, Any, Any]:
    """``(chosen mean logps, rejected mean logps, chosen NLL)`` for a collated batch.

    ORPO's terms with the completion mask in place of the sequence mask: CPO's
    NLL reads the same labels as the scores, so the prompt is not in it.
    """
    return orpo_terms(model, tokens, completion, completion, head_split=head_split)


def simpo_loss(
    chosen_mean_logps: Any,
    rejected_mean_logps: Any,
    chosen_nll: Any,
    *,
    beta: float,
    gamma: float,
    cpo_alpha: float,
) -> tuple[Any, dict]:
    """TRL's SimPO loss and its metrics from per-pair mean completion log-probs.

    Returns ``(loss, metrics)`` with ``preference_loss`` (the mean preference
    term, which TRL does not log) and TRL's logged quantities: ``nll_loss``,
    ``reward_chosen`` / ``reward_rejected`` (``beta * logp``),
    ``reward_accuracy`` (strict ``>``), ``reward_margin``, and
    ``logps_chosen`` / ``logps_rejected``, all batch means.
    """
    import mlx.core as mx
    import mlx.nn as nn

    a = chosen_mean_logps.astype(mx.float32)
    b = rejected_mean_logps.astype(mx.float32)
    logits = (a - b) - gamma / beta
    pref = -nn.log_sigmoid(beta * logits)
    loss = pref.mean() + cpo_alpha * chosen_nll
    chosen_rewards, rejected_rewards = beta * a, beta * b
    metrics = {
        "preference_loss": pref.mean(),
        "nll_loss": chosen_nll,
        "reward_chosen": chosen_rewards.mean(),
        "reward_rejected": rejected_rewards.mean(),
        "reward_accuracy": (chosen_rewards > rejected_rewards).astype(mx.float32).mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        "logps_chosen": a.mean(),
        "logps_rejected": b.mean(),
    }
    return loss, metrics


class MLXSimPOTrainerWrapper(MLXORPOTrainerWrapper):
    """SimPO on MLX: MLX ORPO's tokens, MLX DPO's loop, CPO's simpo loss."""

    objective = "SimPO"
    metric_names = (
        "preference_loss", "nll_loss", "reward_chosen", "reward_rejected", "reward_accuracy",
        "reward_margin", "logps_chosen", "logps_rejected",
    )

    def _batch_arrays(self, examples: list) -> tuple:
        import mlx.core as mx

        tokens, completion, _ = collate_orpo(examples)
        return mx.array(tokens), mx.array(completion)

    def _loss(self, model: Any, arrays: tuple, split: str, rows: Any) -> tuple:
        tokens, completion = arrays
        a, b, nll = simpo_terms(model, tokens, completion, head_split=self._head_split)
        tcfg = self.config.training
        return simpo_loss(
            a, b, nll, beta=SIMPO_BETA, gamma=float(tcfg.simpo_gamma),
            cpo_alpha=float(tcfg.cpo_alpha),
        )

    def _format_metrics(self, metrics: dict) -> str:
        return (
            f"nll {metrics['nll_loss']:.4f}, reward acc {metrics['reward_accuracy']:.2f}, "
            f"margin {metrics['reward_margin']:.4f}"
        )

    def _objective_metadata(self) -> dict:
        tcfg = self.config.training
        return {
            "task": "simpo",
            "simpo_beta": SIMPO_BETA,
            "simpo_gamma": float(tcfg.simpo_gamma),
            "cpo_alpha": float(tcfg.cpo_alpha),
            "simpo_loss": "trl-0.29.1 cpo simpo + cpo_alpha * chosen completion NLL",
            "reference": "none (reference-free)",
            "truncation_mode": ORPO_TRUNCATION_MODE,
        }
