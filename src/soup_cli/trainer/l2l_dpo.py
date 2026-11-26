"""DPO window loss for L2L (ticket A6-04).

TRL 0.29.1 ``DPOTrainer`` builds one micro-batch ``[chosen rows; rejected rows]`` with
``DataCollatorForPreference`` (``input_ids``, ``attention_mask``, ``completion_mask``,
right-padded). Its ``_compute_loss`` truncates the batch, runs the policy once with grad, runs
the model again under ``no_grad`` with the frozen ``ref`` adapter active (a ``PeftModel`` given
no ``ref_model`` always has one) and returns

    mean(-logsigmoid(beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))))

where each log-probability is the sum over the completion tokens of the row.

The L2L step walks the policy's ``[chosen; rejected]`` batch with grad and, through the same
loaded layer, the reference pass of :class:`~soup_cli.utils.l2l_step.ReferencePass`. The
reference head returns the two reference sums per micro-batch; the policy head reads them.
HF divides a preference loss by ``k = len(window)``; the head returns the loss already divided.

Only plain sigmoid DPO is reproduced: ``refuse_unsupported_dpo`` checks the built trainer for
the settings the formula assumes. The layers run without an attention mask, as for SFT; with
right padding and causal attention the output at a real position does not depend on the
padding after it, and the loss masks everything but completion tokens.

torch is imported inside functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Tuple

from soup_cli.trainer.l2l_trainer import (
    WindowLoss,
    refuse_mixed_shapes,
    refuse_unsupported_trainer,
    require_right_padded,
)
from soup_cli.utils.l2l_step import LossHead, ReferencePass

#: PEFT adapter name TRL gives the frozen copy of the policy it takes the reference from.
REFERENCE_ADAPTER = "ref"


def refuse_unsupported_dpo(trainer: Any) -> None:
    """Refuse a built ``DPOTrainer`` whose loss is not the plain sigmoid DPO reproduced here.

    The Soup config exposes none of these settings, so they hold on every run today; the check
    keeps a TRL default that moves from silently changing what the L2L loss computes.
    """
    refuse_unsupported_trainer(
        trainer,
        (
            ("loss_types", ["sigmoid"], "loss_type"),
            ("loss_weights", [1.0], "loss_weights"),
            ("ref_model", None, "ref_model"),
            ("precompute_ref_logps", False, "precompute_ref_log_probs"),
            ("ld_alpha", None, "ld_alpha"),
            ("f_divergence_type", "reverse_kl", "f_divergence_type"),
            ("label_smoothing", 0.0, "label_smoothing"),
            ("use_weighting", False, "use_weighting"),
        ),
        reproduces=(
            "training.l2l with task 'dpo' reproduces TRL's plain sigmoid DPO with the "
            f"'{REFERENCE_ADAPTER}' adapter as the reference"
        ),
    )


def _completion_logps(logits: Any, shift_labels: Any, shift_mask: Any) -> Any:
    """Sum of token log-probabilities over the completion tokens of each row, as TRL does."""
    from trl.trainer.utils import selective_log_softmax

    shift_logits = logits[..., :-1, :].contiguous()
    per_token = selective_log_softmax(shift_logits, shift_labels)
    per_token[shift_mask == 0] = 0.0
    return per_token.sum(dim=1)


def _completion_logps_of(causal: Any, owner: Any, hidden: Any, labels: Tuple[Any, Any]) -> Any:
    """Final norm, ``lm_head`` and the completion sums of one micro-batch's last hidden state."""
    logits = causal.lm_head(owner.norm(hidden)[:, slice(0, None), :])
    return _completion_logps(logits, *labels)


def dpo_loss(
    policy_logps: Any, ref_logps: Tuple[Any, Any], beta: float
) -> Any:
    """TRL's sigmoid DPO loss of one micro-batch from the stacked policy sums.

    ``policy_logps`` has the ``2n`` rows ``[chosen; rejected]``; ``ref_logps`` is the pair of
    reference sums ``(chosen, rejected)``.
    """
    import torch.nn.functional as functional

    chosen, rejected = policy_logps.chunk(2, dim=0)
    ref_chosen, ref_rejected = ref_logps
    delta = (chosen - ref_chosen) - (rejected - ref_rejected)
    return (-functional.logsigmoid(beta * delta)).mean()


@dataclass
class _Window:
    """What one window needs besides the ids: per micro-batch shifted labels and mask, and the
    reference pass whose outputs the policy head reads."""

    labels: List[Tuple[Any, Any]]
    reference: ReferencePass


def dpo_window_loss(*, args: Any) -> WindowLoss:
    """The window loss of ``task: dpo``; ``args`` is the ``DPOConfig`` the trainer was built with.

    It supplies ``beta`` and the ``max_length`` / ``truncation_mode`` TRL truncates each batch
    with, so the layers see the same tokens the policy forward of TRL sees.
    """
    beta = float(args.beta)

    def truncate(shim: Any, ids: Any, mask: Any, completion: Any) -> Tuple[Any, Any, Any]:
        from trl import DPOTrainer

        return DPOTrainer._truncate_inputs(shim, ids, mask, completion)

    def read_window(
        window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
    ) -> Tuple[List[Any], _Window]:
        ids_list: List[Any] = []
        labels: List[Tuple[Any, Any]] = []
        # ``_truncate_inputs`` reads only ``self.args.max_length`` and ``truncation_mode``.
        shim = SimpleNamespace(args=args)
        for inputs in window:
            ids, mask, completion = truncate(
                shim,
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device),
                inputs["completion_mask"].to(device),
            )
            require_right_padded(mask)
            ids_list.append(ids)
            labels.append((ids[..., 1:].contiguous(), completion[..., 1:].contiguous()))
        refuse_mixed_shapes(ids_list, neftune=neftune, has_dropout=has_dropout)

        def reference_head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            return _completion_logps_of(causal, owner, hidden, labels[mb]).chunk(2, dim=0)

        reference = ReferencePass(adapter=REFERENCE_ADAPTER, head=reference_head)
        return ids_list, _Window(labels=labels, reference=reference)

    def make_head(extras: _Window, divisor: int) -> LossHead:
        def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            policy = _completion_logps_of(causal, owner, hidden, extras.labels[mb])
            return dpo_loss(policy, extras.reference.outputs[mb], beta) / divisor

        return head

    return WindowLoss(
        divide_by="micro_batches",
        read_window=read_window,
        make_head=make_head,
        make_reference=lambda extras: extras.reference,
    )
