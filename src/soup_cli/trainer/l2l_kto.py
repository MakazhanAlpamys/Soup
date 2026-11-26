"""KTO window loss for L2L (ticket A6-05).

TRL 0.29.1 ``KTOTrainer`` reads unpaired rows ``(prompt, completion, label)``. At data
preparation it builds a second completion for every row, the KL completion: the answers of a
block of ``per_device_train_batch_size`` consecutive rows shifted by one inside the block
(``_get_kl_dataset``). A micro-batch is then one collator batch holding ``completion_*`` and
``KL_completion_*`` (right-padded, each at its own width) and ``label``. Per micro-batch TRL
runs four forwards, all through ``self.model``:

1. the policy over the KL completions, ``no_grad``;
2. the policy over the completions, with grad;
3. and 4. the same two forwards with every adapter switched off (``null_ref_context``), the
   reference; ``no_grad``. There is no ``ref`` adapter, unlike DPO.

Its loss is ``nanmean`` over the rows of the micro-batch of the sigmoid terms of ``kto_loss``,
with ``kl = clamp(mean(policy_KL - reference_KL), min=0)`` as a constant. An empty group of one
label is an empty tensor in the ``cat``, so a micro-batch of one label is finite.

The L2L step walks the policy's completions with grad and, through the same loaded layer, the
three other forwards as :class:`~soup_cli.utils.l2l_step.ReferencePass`es. The KL completions
have their own width, so those passes carry ids of their own. ``log-probabilities`` and the loss
come from TRL's ``KTOTrainer.get_batch_logps`` and ``KTOTrainer.kto_loss`` themselves. HF divides
a preference loss by ``k = len(window)``; the head returns the loss already divided.

The KL completion of a row is fixed when the data is prepared, from the rows around it. It is
the one of a row of the same micro-batch only if the micro-batches are those blocks: one
process, ``SequentialSampler`` (which KTO uses) and ``dataset_num_proc`` unset.
:func:`refuse_unaligned_kl_blocks` checks that at start. The layers run without an attention
mask, as for SFT; with right padding and causal attention the output at a real position does
not depend on the padding after it, and the log-probabilities mask everything but completion
tokens.

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


def refuse_unaligned_kl_blocks(args: Any, sampler: Any = None) -> None:
    """Refuse a run whose micro-batches are not the blocks the KL completions were built in.

    Call it with the ``KTOConfig`` before the trainer exists (``dataset_num_proc`` would fork
    workers that cut the blocks at their own shard borders, and more than one process gathers
    ``kl`` across processes), and again with the trainer's train sampler once it does.
    """
    num_proc = getattr(args, "dataset_num_proc", None)
    if num_proc is not None and int(num_proc) > 1:
        raise ValueError(
            f"training.l2l with task 'kto' needs dataset_num_proc to be unset or 1 (got "
            f"{num_proc}): TRL builds each row's KL completion inside a block of "
            "per_device_train_batch_size consecutive rows, and a parallel map cuts the blocks "
            "at its shard borders, so a micro-batch would not be one block."
        )
    world_size = int(getattr(args, "world_size", 1) or 1)
    if world_size > 1:
        raise ValueError(
            f"training.l2l with task 'kto' is single-process, but {world_size} processes "
            "were started: TRL averages the KL estimate over the processes, so a micro-batch's "
            "loss would depend on rows of other processes. Start one process."
        )
    if sampler is not None:
        from torch.utils.data import SequentialSampler

        if not isinstance(sampler, SequentialSampler):
            raise ValueError(
                "training.l2l with task 'kto' needs the SequentialSampler KTO uses: the KL "
                "completions are built inside blocks of consecutive rows, and a "
                f"{type(sampler).__name__} would not hand them out as those blocks."
            )


def refuse_unsupported_kto(trainer: Any) -> None:
    """Refuse a built ``KTOTrainer`` whose loss is not the plain KTO reproduced here.

    The Soup config exposes none of these settings, so they hold on every run today; the check
    keeps a TRL default that moves from silently changing what the L2L loss computes.
    """
    refuse_unsupported_trainer(
        trainer,
        (
            ("loss_type", "kto", "loss_type"),
            ("calculate_KL", True, "calculate_KL"),
            ("ref_model", None, "ref_model"),
            ("precompute_ref_log_probs", False, "precompute_ref_log_probs"),
            ("ref_adapter_name", None, "ref_adapter_name"),
            ("aux_loss_enabled", False, "aux_loss_enabled"),
        ),
        reproduces=(
            "training.l2l with task 'kto' reproduces TRL's plain KTO loss with the adapters "
            "switched off as the reference"
        ),
    )


def _logps_of(causal: Any, owner: Any, hidden: Any, labels: Any) -> Any:
    """Final norm, ``lm_head`` and the per-row log-probability sums, as TRL's KTO takes them."""
    from trl import KTOTrainer

    logits = causal.lm_head(owner.norm(hidden)[:, slice(0, None), :])
    return KTOTrainer.get_batch_logps(logits, labels, average_log_prob=False)


@dataclass
class _Window:
    """What one window needs besides the ids, per micro-batch: the labels of the completions
    and of the KL completions, the rows of each desirability, and the three extra passes whose
    outputs the policy head reads."""

    completion_labels: List[Any]
    kl_labels: List[Any]
    desirable: List[Any]
    undesirable: List[Any]
    policy_kl: ReferencePass
    reference: ReferencePass
    reference_kl: ReferencePass


def kto_window_loss(*, args: Any) -> WindowLoss:
    """The window loss of ``task: kto``; ``args`` is the ``KTOConfig`` the trainer was built with.

    It supplies ``beta``, ``loss_type`` and the group weights TRL's ``kto_loss`` reads.
    """

    def read_window(
        window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
    ) -> Tuple[List[Any], _Window]:
        import torch

        ids_list: List[Any] = []
        kl_ids: List[Any] = []
        completion_labels: List[Any] = []
        kl_labels: List[Any] = []
        desirable: List[Any] = []
        undesirable: List[Any] = []
        for inputs in window:
            ids_list.append(inputs["completion_input_ids"].to(device))
            kl_ids.append(inputs["KL_completion_input_ids"].to(device))
            require_right_padded(inputs["completion_attention_mask"])
            require_right_padded(inputs["KL_completion_attention_mask"])
            completion_labels.append(inputs["completion_labels"].to(device))
            kl_labels.append(inputs["KL_completion_labels"].to(device))
            label = torch.as_tensor(inputs["label"], dtype=torch.bool, device=device)
            desirable.append(torch.nonzero(label, as_tuple=False).view(-1))
            undesirable.append(torch.nonzero(~label, as_tuple=False).view(-1))
        refuse_mixed_shapes(ids_list, neftune=neftune, has_dropout=has_dropout)

        def kl_head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            return _logps_of(causal, owner, hidden, kl_labels[mb])

        def completion_head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            return _logps_of(causal, owner, hidden, completion_labels[mb])

        extras = _Window(
            completion_labels=completion_labels,
            kl_labels=kl_labels,
            desirable=desirable,
            undesirable=undesirable,
            # the policy as it stands, over the KL completions
            policy_kl=ReferencePass(head=kl_head, micro_batches=kl_ids),
            # TRL's null_ref_context: every adapter off
            reference=ReferencePass(head=completion_head, adapters_off=True),
            reference_kl=ReferencePass(head=kl_head, adapters_off=True, micro_batches=kl_ids),
        )
        return ids_list, extras

    def make_head(extras: _Window, divisor: int) -> LossHead:
        from trl import KTOTrainer

        def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            policy = _logps_of(causal, owner, hidden, extras.completion_labels[mb])
            reference = extras.reference.outputs[mb]
            desirable, undesirable = extras.desirable[mb], extras.undesirable[mb]
            shim = SimpleNamespace(
                calculate_KL=args.loss_type != "apo_zero_unpaired",
                beta=float(args.beta),
                loss_type=args.loss_type,
                desirable_weight=args.desirable_weight,
                undesirable_weight=args.undesirable_weight,
                # one process: gathering ``kl`` is the identity
                accelerator=SimpleNamespace(
                    gather_for_metrics=lambda value: value, device=policy.device
                ),
            )
            # ``shim`` stands in for ``self``: kto_loss reads only the attributes set above.
            losses, _chosen, _rejected, _kl = KTOTrainer.kto_loss(
                shim,  # type: ignore[arg-type]
                policy.index_select(0, desirable),
                policy.index_select(0, undesirable),
                extras.policy_kl.outputs[mb],
                reference.index_select(0, desirable),
                reference.index_select(0, undesirable),
                extras.reference_kl.outputs[mb],
            )
            return losses.nanmean() / divisor

        return head

    return WindowLoss(
        divide_by="micro_batches",
        read_window=read_window,
        make_head=make_head,
        make_reference=lambda extras: [extras.policy_kl, extras.reference, extras.reference_kl],
    )
