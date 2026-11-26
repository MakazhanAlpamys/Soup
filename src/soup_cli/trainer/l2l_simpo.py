"""SimPO window loss for L2L (ticket A6-03).

SimPO is TRL's ``CPOTrainer`` with ``loss_type="simpo"``. Like ORPO it builds one micro-batch
from separate tensors ``chosen_*`` and ``rejected_*`` (right-padded) with no ``input_ids``,
stacks chosen over rejected in ``concatenated_inputs`` and runs the policy once. The loss is

    mean(-logsigmoid(beta * (p_chosen - p_rejected) - gamma)) + cpo_alpha * NLL

* ``p`` is the per-row mean log-probability of the response tokens (``get_batch_logps`` with
  ``average_log_prob``); the prompt is masked with ``-100`` in ``concatenated_labels``.
* ``NLL`` is a cross entropy over the response tokens of the chosen rows only, averaged over
  those tokens of the micro-batch. ORPO's NLL differs: it covers the prompt too.

There is no reference model, so the layers see the policy's ``[chosen; rejected]`` batch and
nothing else. HF divides a preference loss by ``k = len(window)``; the head returns the loss
already divided.

A pair whose short answer TRL truncates to zero tokens would make ``p`` 0/0. The wrapper
refuses such rows after the trainer is built (``SimPOTrainerWrapper``), on this path too.

torch is imported inside functions.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from soup_cli.trainer.l2l_trainer import (
    WindowLoss,
    refuse_unsupported_trainer,
    stack_pair_window,
)
from soup_cli.utils.l2l_step import LossHead


def refuse_unsupported_simpo(trainer: Any) -> None:
    """Refuse a built ``CPOTrainer`` whose loss is not the plain SimPO reproduced here.

    The Soup config exposes none of these settings, so they hold on every run today; the check
    keeps a TRL default that moves from silently changing what the L2L loss computes.
    """
    refuse_unsupported_trainer(
        trainer,
        (
            ("loss_type", "simpo", "loss_type"),
            ("label_smoothing", 0.0, "label_smoothing"),
            ("aux_loss_enabled", False, "aux_loss_enabled"),
        ),
        reproduces="training.l2l with task 'simpo' reproduces TRL's plain SimPO loss",
    )


def _cpo_trainer_class() -> Any:
    from soup_cli.trainer._trl_compat import resolve_trl_symbol

    return resolve_trl_symbol("CPOTrainer", "trl.experimental.cpo")


def simpo_loss(
    logits: Any,
    labels: Any,
    len_chosen: int,
    *,
    beta: float,
    gamma: float,
    cpo_alpha: float,
    trainer_cls: Any,
) -> Any:
    """TRL's SimPO loss of one micro-batch, in the order ``CPOTrainer`` computes it.

    ``logits`` are for the ``2 * len_chosen`` stacked rows; ``labels`` mask the prompt and the
    padding with ``-100``.
    """
    import torch
    import torch.nn.functional as functional

    all_logps = trainer_cls.get_batch_logps(logits, labels, average_log_prob=True)
    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    margin = (chosen_logps - rejected_logps) - gamma / beta
    losses = -functional.logsigmoid(beta * margin)

    chosen = logits[:len_chosen][..., :-1, :].contiguous()
    shifted = labels[:len_chosen][..., 1:].contiguous()
    nll = torch.nn.CrossEntropyLoss()(chosen.view(-1, chosen.shape[-1]), shifted.view(-1))
    return losses.mean() + cpo_alpha * nll


def simpo_window_loss(
    *, beta: float, gamma: float, cpo_alpha: float, padding_value: int
) -> WindowLoss:
    """The window loss of ``task: simpo``; ``padding_value`` is the tokenizer's pad id."""

    def read_window(
        window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
    ) -> Tuple[List[Any], List[Tuple[Any, int]]]:
        ids_list, stacked_list, chosen_rows = stack_pair_window(
            window,
            device,
            trainer_cls=_cpo_trainer_class(),
            padding_value=padding_value,
            neftune=neftune,
            has_dropout=has_dropout,
        )
        extras = [
            (stacked["concatenated_labels"], len_chosen)
            for stacked, len_chosen in zip(stacked_list, chosen_rows)
        ]
        return ids_list, extras

    def make_head(extras: List[Tuple[Any, int]], divisor: int) -> LossHead:
        trainer_cls = _cpo_trainer_class()

        def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            logits = causal.lm_head(owner.norm(hidden)[:, slice(0, None), :])
            labels, len_chosen = extras[mb]
            loss = simpo_loss(
                logits,
                labels,
                len_chosen,
                beta=beta,
                gamma=gamma,
                cpo_alpha=cpo_alpha,
                trainer_cls=trainer_cls,
            )
            return loss / divisor

        return head

    return WindowLoss(divide_by="micro_batches", read_window=read_window, make_head=make_head)
