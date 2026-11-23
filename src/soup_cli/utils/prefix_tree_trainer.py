"""Wire prefix-tree batches (``utils/prefix_tree.py``) into an HF Trainer.

``training.prefix_tree: true`` changes one thing about a run: the training
dataloader. Its batch sampler groups examples whose prefix trees fit the step's
budget (``batch_size`` rows of at most ``data.max_length`` tokens, the same
budget an ordinary batch has, so every VRAM pre-flight stays valid), and its
collator re-lays the stock collator's batch out as prefix-tree rows. The model,
the loss, the optimizer, evaluation and saving are untouched.

The batch keeps a 2D ``attention_mask`` (per token, how many of the row's
examples it belongs to), because the trainer reads it for its books: TRL counts
a step's tokens with its sum and weights the entropy metric by it. The 4D mask
travels under ``TREE_MASK_KEY`` and ``compute_loss`` hands it to the model in
the 2D mask's place, so ``num_tokens`` and ``entropy`` are those of the ordinary
batch holding the same examples.

Mirrors ``utils/multipack_trainer.py``: a cached trainer subclass whose
``get_train_dataloader`` falls back to the base class until state is attached.
"""

from __future__ import annotations

import functools
from array import array
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

#: Batch key of the 4D mask; ``attention_mask`` stays 2D for the trainer's metrics.
TREE_MASK_KEY = "prefix_tree_mask"

#: Planning holds every training example as two ``array('i')`` (ids, labels),
#: 8 bytes per token, and for a while a second copy of the shareable beginnings.
#: Measured peak on 10,000 examples / 4,000,000 tokens: 66 MB, 16 bytes per token.
PLAN_BYTES_PER_TOKEN = 16

#: Reading and planning are pure Python over every token: 7.5 s for the same
#: 4,000,000 tokens on a laptop CPU. A notice is printed above
#: ``PLAN_NOTICE_TOKENS`` so the wait before the first step is expected.
PLAN_SECONDS_PER_MILLION_TOKENS = 2.0
PLAN_NOTICE_TOKENS = 50_000_000


def planning_notice(examples: int, max_tokens: int) -> Optional[str]:
    """What planning may cost for ``examples`` rows of up to ``max_tokens``
    tokens, when that is worth saying before it starts; else ``None``.

    An upper bound: rows shorter than ``max_tokens`` cost less.
    """
    tokens = int(examples) * int(max_tokens)
    if tokens <= PLAN_NOTICE_TOKENS:
        return None
    gigabytes = tokens * PLAN_BYTES_PER_TOKEN / 1e9
    minutes = tokens / 1e6 * PLAN_SECONDS_PER_MILLION_TOKENS / 60
    return (
        f"planning reads all {examples:,} training examples (up to {tokens:,} tokens) "
        f"before the first step: up to about {gigabytes:.1f} GB of memory, released once "
        f"the rows are planned, and about {minutes:.0f} min on one CPU core"
    )


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
) -> List[Tuple[array, array]]:
    """Every dataset row as ``(input_ids, labels)``, exactly as ``collator`` builds it.

    Read through the collator on purpose: it is the collator that decides which
    labels are ignored (TRL derives them from ``assistant_masks`` at collation
    time), and the sampler has to plan with the examples the step will see.

    Each example is two ``array('i')``, 8 bytes per token (Python lists of ints
    measured about 80).
    """
    if dataset is None:
        raise ValueError("no training dataset to read prefix-tree examples from")
    try:
        total = len(dataset)
    except TypeError as exc:
        raise ValueError(
            "training.prefix_tree needs a dataset with a length (a streaming / "
            "iterable dataset cannot be grouped by shared beginnings up front). "
            "Set data.streaming: false or training.prefix_tree: false."
        ) from exc
    examples: List[Tuple[array, array]] = []
    for start in range(0, total, _READ_CHUNK):
        features = [dataset[index] for index in range(start, min(start + _READ_CHUNK, total))]
        examples.extend(
            (array("i", ids), array("i", labels))
            for ids, labels in examples_from_batch(collator(features), ignore_index=ignore_index)
        )
    if len(examples) != total:
        raise RuntimeError(
            f"the collator returned {len(examples)} rows for {total} dataset rows; "
            "a prefix tree needs one row per example (is TRL packing on?). "
            "Set training.prefix_tree: false."
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
    ``compute_loss`` is the base class's too, called with the 4D mask in the
    model's hands and the 2D one in the trainer's.
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
                    f"across {num_processes} processes. Set training.prefix_tree: false."
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
                tree_mask_key=TREE_MASK_KEY,
            )
            state.collator = collator
            return DataLoader(
                train_dataset,
                batch_sampler=state.sampler,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        def compute_loss(  # type: ignore[override]
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            def base():
                return super(PrefixTreeTrainer, self).compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            tree_mask = inputs.pop(TREE_MASK_KEY, None) if hasattr(inputs, "pop") else None
            if tree_mask is None:
                # An ordinary batch (evaluation, or no state attached).
                return base()

            def give_the_model_the_tree_mask(_module, args, kwargs):
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = tree_mask
                return args, kwargs

            # ``inputs["attention_mask"]`` stays 2D for what the base class reads
            # after the forward; only the model's own call sees the 4D mask.
            handle = model.register_forward_pre_hook(
                give_the_model_the_tree_mask, with_kwargs=True
            )
            try:
                return base()
            finally:
                handle.remove()

    PrefixTreeTrainer.__name__ = f"PrefixTree{base_cls.__name__}"
    PrefixTreeTrainer.__qualname__ = PrefixTreeTrainer.__name__
    return PrefixTreeTrainer


#: Model config fields that attention / MLP code reads as a functional dropout
#: rate in train mode (Llama, Mistral, Qwen2 and others read
#: ``attention_dropout``); dropout done by modules is found by walking them.
_FUNCTIONAL_DROPOUT_FIELDS = ("attention_dropout", "hidden_dropout")


def dropout_in_model(model: Any) -> List[str]:
    """Where ``model`` drops activations in train mode: dropout modules with a
    rate above 0 (LoRA's own included) and the functional rates of its config.

    A prefix-tree row computes a shared beginning once, so a mask drawn on it
    would be shared by every example; the ordinary batch draws one per example.
    The model check runs in eval mode and cannot see this.
    """
    import torch

    found: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0:
            found.append(f"{name or type(module).__name__} (p={module.p:g})")
    config = getattr(model, "config", None)
    for field in _FUNCTIONAL_DROPOUT_FIELDS:
        value = getattr(config, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            found.append(f"config.{field}={value:g}")
    return found


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
    max_tokens: Optional[int] = None,
) -> List[Tuple[List[int], List[int]]]:
    """Real examples to check the model on: the first planned row that holds at
    least two, else the first two of the dataset (packed together they are two
    trees in one row, which exercises the mask and the positions as well).

    The check packs its examples into one row, which must fit the row budget
    the sampler plans with (``max_tokens``, ``data.max_length``). Members of a
    planned row do by construction. The fallback pair may not (two unrelated
    examples of ``max_tokens`` each), so it is cut to its first
    ``max_tokens // 2`` tokens each: shorter copies of the same real text,
    used only by the check; training sees the examples whole.
    """
    chosen: List[int] = []
    for row in rows:
        if len(row) >= 2:
            chosen = list(row[:limit])
            break
    planned = bool(chosen)
    if not chosen:
        chosen = list(range(min(2, len(examples))))
    picked = [(list(examples[index][0]), list(examples[index][1])) for index in chosen]
    if (
        not planned
        and max_tokens is not None
        and sum(len(ids) for ids, _ in picked) > max_tokens
    ):
        keep = max_tokens // 2
        picked = [(ids[:keep], labels[:keep]) for ids, labels in picked]
    return picked


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
