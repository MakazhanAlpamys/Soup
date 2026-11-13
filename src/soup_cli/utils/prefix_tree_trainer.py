"""Wire prefix-tree batches (``utils/prefix_tree.py``) into an HF Trainer.

``training.prefix_tree: true`` changes one thing about a run: the training
dataloader. Its batch sampler groups examples whose prefix trees fit the step's
budget (``batch_size`` rows of at most ``data.max_length`` tokens, the same
budget an ordinary batch has, so every VRAM pre-flight stays valid), and its
collator re-lays the stock collator's batch out as prefix-tree rows. The model,
the loss, the optimizer, evaluation and saving are untouched.

Mirrors ``utils/multipack_trainer.py``: a cached trainer subclass whose
``get_train_dataloader`` falls back to the base class until state is attached.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from soup_cli.utils.prefix_tree import (
    DEFAULT_MIN_GROUP_TOKENS,
    IGNORE_INDEX,
    PrefixTreeBatchSampler,
    PrefixTreeCollator,
    PrefixTreeVerdict,
    examples_from_batch,
    verify_prefix_tree_model,
)

_STATE_ATTR = "_soup_prefix_tree_state"

#: Tasks whose trainer goes through the wiring below.
PREFIX_TREE_TASKS = frozenset({"sft"})

#: Dataset rows handed to the collator at a time when reading the examples.
_READ_CHUNK = 64


def supports_prefix_tree(task: str) -> bool:
    """Whether ``training.prefix_tree`` is wired for ``task``.

    ``sft`` only: continued pretraining trains on every token, and rule 1 of
    ``utils/prefix_tree.py`` shares only tokens that are not trained on, so
    there would be nothing to share.
    """
    return task in PREFIX_TREE_TASKS


@dataclass
class PrefixTreeState:
    """What the trainer subclass needs to build its dataloader."""

    sampler: PrefixTreeBatchSampler
    max_tokens: int
    batch_size: int
    pad_token_id: int
    additive_dtype: Any = None
    #: The collator of the dataloader built last, for its running token counts.
    collator: Optional[PrefixTreeCollator] = None


def examples_from_dataset(
    dataset: Any, collator: Any, *, ignore_index: int = IGNORE_INDEX
) -> List[Tuple[List[int], List[int]]]:
    """Every dataset row as ``(input_ids, labels)``, exactly as ``collator`` builds it.

    Read through the collator on purpose: it is the collator that decides which
    labels are ignored (TRL derives them from ``assistant_masks`` at collation
    time), and the sampler has to plan with the examples the step will see.
    """
    if dataset is None:
        raise ValueError("no training dataset to read prefix-tree examples from")
    try:
        total = len(dataset)
    except TypeError as exc:
        raise ValueError(
            "training.prefix_tree needs a dataset with a length (a streaming / "
            "iterable dataset cannot be grouped by shared beginnings up front)"
        ) from exc
    examples: List[Tuple[List[int], List[int]]] = []
    for start in range(0, total, _READ_CHUNK):
        features = [dataset[index] for index in range(start, min(start + _READ_CHUNK, total))]
        examples.extend(examples_from_batch(collator(features), ignore_index=ignore_index))
    if len(examples) != total:
        raise RuntimeError(
            f"the collator returned {len(examples)} rows for {total} dataset rows; "
            "a prefix tree needs one row per example (is TRL packing on?)"
        )
    return examples


def attach_prefix_tree_state(
    trainer: Any,
    *,
    examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
    max_tokens: int,
    batch_size: int,
    pad_token_id: int,
    seed: int = 0,
    drop_last: bool = False,
    additive_dtype: Any = None,
    min_group_tokens: int = DEFAULT_MIN_GROUP_TOKENS,
) -> PrefixTreeState:
    """Plan the epoch's rows and stash what ``get_train_dataloader`` needs."""
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive int; got {max_tokens!r}")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("pad_token_id must be an int")
    sampler = PrefixTreeBatchSampler(
        examples,
        max_tokens=max_tokens,
        batch_size=batch_size,
        seed=seed,
        drop_last=drop_last,
        min_group_tokens=min_group_tokens,
    )
    state = PrefixTreeState(
        sampler=sampler,
        max_tokens=max_tokens,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        additive_dtype=additive_dtype,
    )
    setattr(trainer, _STATE_ATTR, state)
    return state


def prefix_tree_state(trainer: Any) -> Optional[PrefixTreeState]:
    return getattr(trainer, _STATE_ATTR, None)


@functools.lru_cache(maxsize=None)
def make_prefix_tree_trainer_class(base_cls: type) -> type:
    """A subclass of ``base_cls`` whose training dataloader yields prefix-tree batches.

    Cached, so two runs against the same base class share the subclass. Until
    :func:`attach_prefix_tree_state` has run, ``get_train_dataloader`` is the
    base class's. Evaluation and prediction dataloaders are never touched.
    """

    class PrefixTreeTrainer(base_cls):  # type: ignore[misc, valid-type]
        soup_prefix_tree: bool = True

        def get_train_dataloader(self):  # type: ignore[override]
            state = prefix_tree_state(self)
            train_dataset = getattr(self, "train_dataset", None)
            if state is None or train_dataset is None:
                return super().get_train_dataloader()

            from torch.utils.data import DataLoader

            accelerator = getattr(self, "accelerator", None)
            num_processes = getattr(accelerator, "num_processes", 1)
            if (
                isinstance(num_processes, int)
                and not isinstance(num_processes, bool)
                and num_processes > 1
            ):
                # Sharding rows across ranks is not validated: refuse rather than
                # let every rank train on the same rows.
                raise NotImplementedError(
                    "training.prefix_tree is single-process: it has not been validated "
                    f"across {num_processes} processes"
                )
            args = getattr(self, "args", None)
            num_workers = int(getattr(args, "dataloader_num_workers", 0) or 0)
            pin_memory = bool(getattr(args, "dataloader_pin_memory", False))
            collator = PrefixTreeCollator(
                self.data_collator,
                max_tokens=state.max_tokens,
                pad_token_id=state.pad_token_id,
                max_rows=state.batch_size,
                additive_dtype=state.additive_dtype,
            )
            state.collator = collator
            return DataLoader(
                train_dataset,
                batch_sampler=state.sampler,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    PrefixTreeTrainer.__name__ = f"PrefixTree{base_cls.__name__}"
    PrefixTreeTrainer.__qualname__ = PrefixTreeTrainer.__name__
    return PrefixTreeTrainer


def _model_float_dtype(model: Any) -> Any:
    import torch

    for parameter in model.parameters():
        if getattr(parameter, "is_meta", False):
            continue
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def verification_examples(
    examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
    rows: Sequence[Sequence[int]],
    *,
    limit: int = 3,
) -> List[Tuple[List[int], List[int]]]:
    """Real examples to check the model on: the first planned row that holds at
    least two, else the first two of the dataset (packed together they are two
    trees in one row, which exercises the mask and the positions as well)."""
    chosen: List[int] = []
    for row in rows:
        if len(row) >= 2:
            chosen = list(row[:limit])
            break
    if not chosen:
        chosen = list(range(min(2, len(examples))))
    return [(list(examples[index][0]), list(examples[index][1])) for index in chosen]


def check_prefix_tree_model(
    model: Any,
    *,
    pad_token_id: int,
    examples: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
    vocab_size: Optional[int] = None,
) -> PrefixTreeVerdict:
    """Decide HOW this model takes a prefix-tree row, or refuse.

    Tries the boolean mask first (what scaled-dot-product attention takes),
    then the additive mask in the model's dtype (what an attention that adds
    its mask to the scores takes). Returns the verdict of the form that
    reproduced the ordinary batch's logits; raises ``ValueError`` naming both
    measurements when neither did, so the run stops before its first step
    instead of training on rows the model does not compute faithfully.
    ``examples``: real training examples to check on (see
    :func:`verification_examples`); random tokens without them.
    """
    boolean = verify_prefix_tree_model(
        model, pad_token_id=pad_token_id, examples=examples, vocab_size=vocab_size
    )
    if boolean.exact:
        return boolean
    try:
        additive: Optional[PrefixTreeVerdict] = verify_prefix_tree_model(
            model,
            pad_token_id=pad_token_id,
            examples=examples,
            vocab_size=vocab_size,
            additive_dtype=_model_float_dtype(model),
        )
    except (RuntimeError, ValueError, TypeError):
        additive = None
    if additive is not None and additive.exact:
        return additive
    tried = f"with a boolean mask {boolean.reason}"
    if additive is not None:
        tried += f"; with an additive mask {additive.reason}"
    raise ValueError(
        "training.prefix_tree: this model does not compute a prefix-tree row as it "
        f"computes the same examples in an ordinary batch: {tried}. It needs a "
        "causal decoder that honours a 4D attention_mask and position_ids. Set "
        "training.prefix_tree: false."
    )
