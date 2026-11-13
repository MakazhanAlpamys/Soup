"""Prefix-tree batches: tokens several examples share are computed once.

In a causal language model the hidden state of a token depends only on the
tokens before it. Examples that begin with the same tokens (one system prompt,
one tool schema, one document, one prompt for several answers) therefore have
bit-identical states over that shared beginning, and an ordinary batch computes
them once per example. A prefix-tree row holds the shared beginning once and
the continuations after it:

- ``attention_mask`` is 4D: a token may attend to itself and to its ancestors in
  the tree, nothing else;
- ``position_ids`` is the token's depth in the tree, so every token sits at the
  position it had in its own example;
- ``labels`` are untouched for trained tokens.

Every token then sees exactly the context it saw in its own example, so the
loss and its gradient are those of the ordinary batch holding the same examples
(``tests/test_prefix_tree.py``: logits bit-identical on CPU, gradients to float
rounding). What changes is how many tokens the step runs: the tree's nodes, not
the sum of the examples' lengths.

Three rules keep that equality exact with the stock shifted cross-entropy:

1. Only tokens whose label is ignored are shared. A trained token appears once
   per example, as in the ordinary batch, so the token-mean of the loss keeps
   its weights. Sharing stops at an example's first trained token.
2. The stock loss scores the prediction made at row position ``i`` against the
   label at ``i + 1``. In a tree a token's predecessor is its parent, which is
   its row neighbour only for the first branch emitted under that parent. Where
   a trained token would follow a row neighbour that is not its parent, the
   parent is emitted again in front of it (same token, same position, same
   ancestors, label ignored): one extra token per such branch.
3. A token at the root of a tree has no predecessor. Its label is ignored, as
   the ordinary batch's shift ignores the label at position 0.

This module is pure Python apart from :func:`collate_prefix_tree` and
:func:`verify_prefix_tree_model`, which import torch when called.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

IGNORE_INDEX = -100

#: Shared tokens below which a common beginning does not decide which examples
#: are batched together (it is still shared when they happen to meet in a row).
#: Keeps a batch from being "every question that starts with the same two words".
DEFAULT_MIN_GROUP_TOKENS = 16

#: A prefix-tree row's logits may differ from the ordinary batch's by this
#: many times the model's own layout noise (the ordinary batch against the same
#: examples run one at a time). See :func:`verify_prefix_tree_model`.
VERIFY_NOISE_FACTOR = 4.0

#: The smallest accepted difference, as a fraction of the largest logit, for a
#: model whose layout noise measures as zero (CPU kernels often are exact).
VERIFY_FLOOR_BY_DTYPE = {
    "bfloat16": 4e-3,
    "float16": 1e-3,
    "float32": 1e-6,
    "float64": 1e-12,
}

#: A row's error must be at most this fraction of what a deliberately wrong row
#: produces (a plain causal mask; positions left at the row index).
VERIFY_CONTROL_FRACTION = 0.25


@dataclass(frozen=True)
class PrefixTreeRow:
    """One row of a prefix-tree batch.

    ``parents[i]`` is the row index of token ``i``'s parent, ``-1`` at a root.
    ``members`` are the caller's example indices held by this row, in the order
    they were added; ``paths[k][t]`` is the row index that carries token ``t``
    of ``members[k]`` (its hidden state, and the prediction for token ``t + 1``).
    """

    input_ids: Tuple[int, ...]
    labels: Tuple[int, ...]
    position_ids: Tuple[int, ...]
    parents: Tuple[int, ...]
    members: Tuple[int, ...]
    paths: Tuple[Tuple[int, ...], ...]

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def source_tokens(self) -> int:
        """Tokens the same examples hold as ordinary rows."""
        return sum(len(path) for path in self.paths)


class _RowBuilder:
    """Grows one row example by example."""

    __slots__ = ("ids", "labels", "positions", "parents", "members", "paths", "_children")

    def __init__(self) -> None:
        self.ids: List[int] = []
        self.labels: List[int] = []
        self.positions: List[int] = []
        self.parents: List[int] = []
        self.members: List[int] = []
        self.paths: List[Tuple[int, ...]] = []
        # Shareable nodes only: parent row index (-1 = root) -> token -> row index.
        self._children: Dict[int, Dict[int, int]] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def _shared(self, ids: Sequence[int], labels: Sequence[int], ignore: int) -> Tuple[int, int]:
        """``(tokens already in the tree, row index of the last of them or -1)``."""
        node = -1
        depth = 0
        for token, label in zip(ids, labels):
            if label != ignore:
                break
            child = self._children.get(node, {}).get(token)
            if child is None:
                break
            node = child
            depth += 1
        return depth, node

    def cost(self, ids: Sequence[int], labels: Sequence[int], ignore: int) -> int:
        """Tokens adding this example would append to the row."""
        depth, node = self._shared(ids, labels, ignore)
        remaining = len(ids) - depth
        if remaining == 0:
            return 0
        return remaining + int(self._needs_parent_again(node, labels[depth], ignore))

    def _needs_parent_again(self, node: int, first_label: int, ignore: int) -> bool:
        # Rule 2: a trained token must directly follow its parent in the row.
        return node != -1 and first_label != ignore and node != len(self.ids) - 1

    def add(self, member: int, ids: Sequence[int], labels: Sequence[int], ignore: int) -> None:
        depth, node = self._shared(ids, labels, ignore)
        path: List[int] = []
        walk = node
        while walk != -1:
            path.append(walk)
            walk = self.parents[walk]
        path.reverse()

        parent = node
        shareable = True
        if depth < len(ids) and self._needs_parent_again(node, labels[depth], ignore):
            self.ids.append(self.ids[node])
            self.labels.append(ignore)
            self.positions.append(self.positions[node])
            self.parents.append(self.parents[node])
            parent = len(self.ids) - 1
            path[-1] = parent
        for token, label in zip(ids[depth:], labels[depth:]):
            index = len(self.ids)
            self.ids.append(int(token))
            # Rule 3: a root has no predecessor to be scored against.
            self.labels.append(ignore if parent == -1 else int(label))
            self.positions.append(0 if parent == -1 else self.positions[parent] + 1)
            self.parents.append(parent)
            # Rule 1: the shareable part of a path ends at its first trained token.
            shareable = shareable and label == ignore
            if shareable:
                self._children.setdefault(parent, {})[int(token)] = index
            path.append(index)
            parent = index
        self.members.append(member)
        self.paths.append(tuple(path))

    def build(self) -> PrefixTreeRow:
        return PrefixTreeRow(
            input_ids=tuple(self.ids),
            labels=tuple(self.labels),
            position_ids=tuple(self.positions),
            parents=tuple(self.parents),
            members=tuple(self.members),
            paths=tuple(self.paths),
        )


def _normalise(
    index: int, ids: Sequence[int], labels: Sequence[int], ignore: int
) -> Tuple[List[int], List[int]]:
    """Plain int lists, with the label at position 0 ignored (rule 3): the stock
    shifted loss never scores an example's first token, so it is shareable."""
    if len(ids) != len(labels):
        raise ValueError(
            f"example {index}: {len(ids)} input ids but {len(labels)} labels; "
            "a prefix tree needs one label per token"
        )
    if len(ids) == 0:
        raise ValueError(f"example {index} is empty")
    return [int(token) for token in ids], [ignore] + [int(label) for label in labels[1:]]


def pack_prefix_tree(
    examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
    *,
    max_tokens: int,
    ignore_index: int = IGNORE_INDEX,
) -> List[PrefixTreeRow]:
    """Pack ``(input_ids, labels)`` examples, in order, into prefix-tree rows.

    Greedy and sequential: an example joins the current row if the tokens it
    adds fit under ``max_tokens``, else it starts a new row. A row always takes
    its first example, even one longer than ``max_tokens`` (the caller's
    tokeniser decides lengths; this never truncates or drops). Because a row
    starts from nothing, packing a run of examples that begins at a row boundary
    gives the same rows as packing the whole list: a sampler and a collator can
    each run this and agree.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive int; got {max_tokens!r}")
    rows: List[PrefixTreeRow] = []
    builder: Optional[_RowBuilder] = None
    for index, (raw_ids, raw_labels) in enumerate(examples):
        ids, labels = _normalise(index, raw_ids, raw_labels, ignore_index)
        added = 0 if builder is None else builder.cost(ids, labels, ignore_index)
        if builder is not None and len(builder) + added > max_tokens:
            rows.append(builder.build())
            builder = None
        if builder is None:
            builder = _RowBuilder()
        builder.add(index, ids, labels, ignore_index)
    if builder is not None:
        rows.append(builder.build())
    return rows


def ancestor_mask(parents: Sequence[int]) -> List[List[bool]]:
    """``mask[i][j]``: token ``i`` may attend to token ``j`` (itself or an ancestor).

    Parents precede their children in a row, so one pass suffices.
    """
    size = len(parents)
    mask: List[List[bool]] = []
    for index, parent in enumerate(parents):
        if not -1 <= parent < index:
            raise ValueError(
                f"parents[{index}] = {parent}: a parent must precede its child in the row"
            )
        row = [False] * size if parent == -1 else list(mask[parent])
        row[index] = True
        mask.append(row)
    return mask


def collate_prefix_tree(
    rows: Sequence[PrefixTreeRow],
    *,
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
    additive_dtype: Any = None,
) -> Dict[str, Any]:
    """Rows -> ``input_ids`` / ``labels`` / ``position_ids`` / 4D ``attention_mask``.

    The mask is boolean (True = may attend), which scaled-dot-product attention
    takes for any model dtype. ``additive_dtype`` instead gives the additive
    form (0 where attention is allowed, the dtype's most negative value
    elsewhere) for an attention implementation that adds its mask to the scores.
    Padding tokens attend to themselves only: a row with no allowed key would
    turn its softmax into NaN.
    """
    import torch

    if not rows:
        raise ValueError("no rows to collate")
    width = max(len(row) for row in rows)
    count = len(rows)
    input_ids = torch.full((count, width), int(pad_token_id), dtype=torch.long)
    labels = torch.full((count, width), int(ignore_index), dtype=torch.long)
    position_ids = torch.zeros((count, width), dtype=torch.long)
    allow = torch.zeros((count, width, width), dtype=torch.bool)
    diagonal = torch.arange(width)
    allow[:, diagonal, diagonal] = True
    for r, row in enumerate(rows):
        size = len(row)
        input_ids[r, :size] = torch.tensor(row.input_ids, dtype=torch.long)
        labels[r, :size] = torch.tensor(row.labels, dtype=torch.long)
        position_ids[r, :size] = torch.tensor(row.position_ids, dtype=torch.long)
        parents = row.parents
        for index in range(size):
            parent = parents[index]
            if parent != -1:
                allow[r, index, : index] = allow[r, parent, : index]
    mask = allow[:, None]
    if additive_dtype is not None:
        mask = torch.zeros(mask.shape, dtype=additive_dtype).masked_fill(
            ~mask, torch.finfo(additive_dtype).min
        )
    return {
        "input_ids": input_ids,
        "attention_mask": mask,
        "position_ids": position_ids,
        "labels": labels,
    }


@dataclass(frozen=True)
class PrefixTreeStats:
    """What a packing saves: tokens as ordinary rows against tree nodes."""

    examples: int
    rows: int
    source_tokens: int
    tree_tokens: int

    @property
    def ratio(self) -> float:
        return self.source_tokens / self.tree_tokens if self.tree_tokens else 1.0


def prefix_tree_stats(rows: Sequence[PrefixTreeRow]) -> PrefixTreeStats:
    return PrefixTreeStats(
        examples=sum(len(row.members) for row in rows),
        rows=len(rows),
        source_tokens=sum(row.source_tokens for row in rows),
        tree_tokens=sum(len(row) for row in rows),
    )


# --------------------------------------------------------------------------
# Which examples meet in a row
# --------------------------------------------------------------------------
def shareable_prefix(
    ids: Sequence[int], labels: Sequence[int], ignore_index: int
) -> Tuple[int, ...]:
    """The beginning of an example another example could share (rules 1 and 3)."""
    out: List[int] = []
    for at, (token, label) in enumerate(zip(ids, labels)):
        if at and label != ignore_index:
            break
        out.append(int(token))
    return tuple(out)


def _common(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def group_by_shared_prefix(
    prefixes: Sequence[Sequence[int]], *, min_group_tokens: int = DEFAULT_MIN_GROUP_TOKENS
) -> list:
    """Nest example indices by the beginnings they share.

    Returns a tree as nested lists whose leaves are example indices. A sub-list
    is a set of examples sharing at least ``min_group_tokens`` MORE tokens than
    the list around it does: one system prompt for the whole set gives one flat
    list, a document per few examples gives one sub-list per document, a
    document under a common system prompt gives both levels. Common beginnings
    shorter than that do not group, so a batch is not biased toward examples
    that merely start with the same few words.
    """
    if isinstance(min_group_tokens, bool) or not isinstance(min_group_tokens, int):
        raise TypeError("min_group_tokens must be an int")
    if min_group_tokens < 1:
        raise ValueError(f"min_group_tokens must be >= 1; got {min_group_tokens}")
    order = sorted(range(len(prefixes)), key=lambda index: tuple(prefixes[index]))
    shared = [
        _common(prefixes[order[at - 1]], prefixes[order[at]]) for at in range(1, len(order))
    ]

    def build(lo: int, hi: int, depth: int) -> list:
        # order[lo:hi] share ``depth`` tokens. Children: maximal runs whose
        # neighbours share at least ``depth + min_group_tokens``.
        out: list = []
        at = lo
        while at < hi:
            end = at + 1
            while end < hi and shared[end - 1] >= depth + min_group_tokens:
                end += 1
            if end - at == 1:
                out.append(order[at])
            else:
                out.append(build(at, end, min(shared[at : end - 1])))
            at = end
        return out

    return build(0, len(order), 0)


def _flatten_shuffled(tree: list, rng: random.Random) -> Iterator[int]:
    children = list(tree)
    rng.shuffle(children)
    for child in children:
        if isinstance(child, list):
            yield from _flatten_shuffled(child, rng)
        else:
            yield child


class PrefixTreeBatchSampler:
    """Batches of example indices whose prefix trees fit the step's token budget.

    ``examples[i]`` is ``(input_ids, labels)`` of dataset row ``i`` exactly as
    the collator will see it. Rows are formed ONCE, at construction: examples
    are ordered so that those sharing a long beginning are adjacent (random
    within and across such groups, see :func:`group_by_shared_prefix`) and
    packed greedily by :func:`pack_prefix_tree`. Each epoch then draws a new
    random grouping of those rows into batches of ``batch_size`` rows, so the
    number of batches is the same every epoch.

    A batch is yielded as its rows' examples, in order. Packing that list again
    (:class:`PrefixTreeCollator` does) never needs more than ``batch_size`` rows
    nor a row over ``max_tokens``: when a row's first examples fit into the
    previous row's slack, what is left of the row is a subset of it in the same
    order, whose tree has no more nodes and needs no more repeated parents. It
    can need fewer rows, which is the same examples in less space.
    """

    def __init__(
        self,
        examples: Sequence[Tuple[Sequence[int], Sequence[int]]],
        *,
        max_tokens: int,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
        ignore_index: int = IGNORE_INDEX,
        min_group_tokens: int = DEFAULT_MIN_GROUP_TOKENS,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive int; got {batch_size!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if not examples:
            raise ValueError("no examples: an empty dataset cannot drive a prefix-tree sampler")
        for index, (ids, labels) in enumerate(examples):
            _normalise(index, ids, labels, ignore_index)
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = bool(drop_last)
        self.epoch = 0
        prefixes = [shareable_prefix(ids, labels, ignore_index) for ids, labels in examples]
        tree = group_by_shared_prefix(prefixes, min_group_tokens=min_group_tokens)
        order = list(_flatten_shuffled(tree, random.Random(seed)))
        packed = pack_prefix_tree(
            [examples[index] for index in order], max_tokens=max_tokens, ignore_index=ignore_index
        )
        #: Dataset indices of each row, in packing order.
        self.rows: List[Tuple[int, ...]] = [
            tuple(order[member] for member in row.members) for row in packed
        ]
        self.stats = prefix_tree_stats(packed)

    def __len__(self) -> int:
        full, rest = divmod(len(self.rows), self.batch_size)
        return full if self.drop_last or not rest else full + 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        order = list(range(len(self.rows)))
        random.Random(self.seed * 1_000_003 + self.epoch).shuffle(order)
        self.epoch += 1
        for start in range(0, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if len(chosen) < self.batch_size and self.drop_last:
                return
            yield [index for row in chosen for index in self.rows[row]]


# --------------------------------------------------------------------------
# The collator
# --------------------------------------------------------------------------
#: Batch keys a prefix-tree batch rebuilds.
_REBUILT_KEYS = frozenset({"input_ids", "labels", "attention_mask", "position_ids"})


def examples_from_batch(
    batch: Any, *, ignore_index: int = IGNORE_INDEX
) -> List[Tuple[List[int], List[int]]]:
    """An ordinary collated batch -> per-example ``(input_ids, labels)`` lists.

    Padding is removed by the 2D ``attention_mask`` (left or right padding);
    without one, every position counts. Any other per-token tensor in the batch
    is refused: this would have to be carried through the tree, and silently
    dropping it would train on something other than what the collator built.
    """
    for key in ("input_ids", "labels"):
        if key not in batch:
            raise ValueError(f"a prefix tree needs {key!r} in the collated batch")
    extra = sorted(set(batch.keys()) - _REBUILT_KEYS)
    if extra:
        raise ValueError(
            f"the collated batch carries {extra}, which a prefix-tree batch cannot "
            "carry through; training.prefix_tree supports input_ids / labels / "
            "attention_mask batches only"
        )
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if tuple(input_ids.shape) != tuple(labels.shape) or input_ids.dim() != 2:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and labels {tuple(labels.shape)} must be "
            "the same 2D shape"
        )
    mask = batch.get("attention_mask")
    if mask is not None and tuple(mask.shape) != tuple(input_ids.shape):
        raise ValueError(
            f"attention_mask {tuple(mask.shape)} is not the 2D padding mask of "
            f"input_ids {tuple(input_ids.shape)}"
        )
    examples = []
    for row in range(input_ids.shape[0]):
        ids = input_ids[row]
        row_labels = labels[row]
        if mask is not None:
            keep = mask[row].bool()
            ids = ids[keep]
            row_labels = row_labels[keep]
        examples.append((ids.tolist(), row_labels.tolist()))
    return examples


class PrefixTreeCollator:
    """Wrap a collator: its ordinary batch becomes a prefix-tree batch.

    The inner collator still decides everything about the examples (tokens,
    truncation, which labels are ignored). This only re-lays them out, so
    whatever it is asked to collate comes out with the same trained tokens.
    ``max_rows`` (the step's row budget) is checked, not enforced: batches from
    :class:`PrefixTreeBatchSampler` fit by construction, and a batch that does
    not is a wiring error worth stopping on rather than a VRAM surprise.
    """

    def __init__(
        self,
        inner: Any,
        *,
        max_tokens: int,
        pad_token_id: int,
        max_rows: Optional[int] = None,
        ignore_index: int = IGNORE_INDEX,
        additive_dtype: Any = None,
    ) -> None:
        if not callable(inner):
            raise TypeError("inner must be a callable collator")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError("pad_token_id must be an int")
        self.inner = inner
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
        self.max_rows = max_rows
        self.ignore_index = ignore_index
        self.additive_dtype = additive_dtype
        #: Totals over every batch collated so far.
        self.source_tokens = 0
        self.tree_tokens = 0
        self.examples = 0

    def __call__(self, features: Any) -> Dict[str, Any]:
        batch = self.inner(features)
        examples = examples_from_batch(batch, ignore_index=self.ignore_index)
        rows = pack_prefix_tree(
            examples, max_tokens=self.max_tokens, ignore_index=self.ignore_index
        )
        if self.max_rows is not None and len(rows) > self.max_rows:
            raise RuntimeError(
                f"a batch of {len(examples)} examples packed into {len(rows)} prefix-tree "
                f"rows, over the step's {self.max_rows}; the sampler and the collator "
                "disagree about the examples"
            )
        stats = prefix_tree_stats(rows)
        self.source_tokens += stats.source_tokens
        self.tree_tokens += stats.tree_tokens
        self.examples += stats.examples
        return collate_prefix_tree(
            rows,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
            additive_dtype=self.additive_dtype,
        )


# --------------------------------------------------------------------------
# Does this model compute a prefix-tree row exactly?
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PrefixTreeVerdict:
    """:func:`verify_prefix_tree_model`'s measurement.

    All differences are the largest absolute difference in logits, at the same
    tokens, against the ordinary batch.

    ``error``: the prefix-tree row. ``noise``: the same examples run one at a
    time, i.e. two ORDINARY layouts of one computation, which is what this
    model's arithmetic disagrees with itself by in its dtype. ``bound``: the
    largest ``error`` accepted. ``control``: the row with every position left at
    its row index; ``mask_control``: the row with a plain causal mask, so
    branches see each other. Both wrong on purpose; ``None`` if the run failed.
    ``scale``: the largest ordinary logit. ``mask``: the mask form checked.
    ``reason``: empty when exact, else why not.
    """

    exact: bool
    error: float
    scale: float
    control: Optional[float]
    mask: str
    noise: float = 0.0
    bound: float = 0.0
    mask_control: Optional[float] = None
    reason: str = ""

    @property
    def relative_error(self) -> float:
        return self.error / self.scale if self.scale else float("inf")


def _verification_examples(vocab_size: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    rng = random.Random(seed)
    top = max(int(vocab_size), 8)

    def tokens(count: int) -> List[int]:
        return [rng.randrange(top // 4, top) for _ in range(count)]

    system = tokens(9)
    document = tokens(5)
    examples = []
    for prefix, answer in [
        (system + document, 4),
        (system + document, 6),
        (system + tokens(3), 5),
    ]:
        ids = prefix + tokens(answer)
        examples.append((ids, [IGNORE_INDEX] * len(prefix) + ids[len(prefix) :]))
    return examples


def verify_prefix_tree_model(
    model: Any,
    *,
    pad_token_id: int,
    examples: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None,
    vocab_size: Optional[int] = None,
    additive_dtype: Any = None,
    seed: int = 0,
) -> PrefixTreeVerdict:
    """Run a few examples through ``model`` as an ordinary batch and as one
    prefix-tree row, and compare the logits token by token.

    Nothing about the architecture is assumed by the trainer: a model that
    ignores ``position_ids``, adds a boolean mask to its scores, or biases
    attention by row distance fails here, before a step is taken. No gradient
    is computed and the model's training mode is restored.

    ``examples`` should be real training examples (two or three that the
    sampler put in one row). Half-precision arithmetic is not reproducible
    across batch shapes, and on tokens a model was never trained on it can be
    unstable enough to hide any difference: measured on an RTX 3050, a
    pretrained Qwen2.5-0.5B given random tokens disagreed with itself by 0.10
    to 1.05 of its logit range in bf16, against 1e-5 in fp32. Without
    ``examples``, random tokens are used (fine in fp32 / fp64 and for tests).

    Nothing is taken from a table. The model's own noise is measured (the
    ordinary batch against the same examples one at a time) and the row must:

    1. differ from the ordinary batch by no more than ``VERIFY_NOISE_FACTOR``
       times that noise (or the dtype's floor, when the noise is zero);
    2. differ by no more than ``VERIFY_CONTROL_FRACTION`` of what the plain
       causal mask does, and that control must itself stand clear of the noise:
       a model whose arithmetic cannot tell isolated examples from mixed ones
       cannot be verified in this dtype and is refused;
    3. likewise for wrong positions, when that control stands clear of the
       noise (a model without positional information has nothing to get wrong).
    """
    import torch

    if examples is None:
        if vocab_size is None:
            vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
        if vocab_size <= 0:
            raise ValueError("vocab_size is needed to build the verification tokens")
        examples = _verification_examples(vocab_size, seed)
    examples = [(list(ids), list(labels)) for ids, labels in examples]
    if len(examples) < 2:
        raise ValueError("at least two examples are needed to verify a prefix-tree row")
    rows = pack_prefix_tree(examples, max_tokens=sum(len(ids) for ids, _ in examples))
    row = rows[0]
    packed = collate_prefix_tree(rows, pad_token_id=pad_token_id, additive_dtype=additive_dtype)

    width = max(len(ids) for ids, _ in examples)
    plain_ids = torch.full((len(examples), width), int(pad_token_id), dtype=torch.long)
    plain_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for index, (ids, _labels) in enumerate(examples):
        plain_ids[index, : len(ids)] = torch.tensor(ids)
        plain_mask[index, : len(ids)] = 1

    # A streamed model's base weights are meta placeholders; its adapters are real.
    device = torch.device("cpu")
    dtype_name = "float32"
    found = False
    for candidate in model.parameters():
        if getattr(candidate, "is_meta", False):
            continue
        if not found:
            device = candidate.device
            found = True
        if candidate.is_floating_point():
            dtype_name = str(candidate.dtype).replace("torch.", "")
            break

    def logits(**inputs: Any) -> Any:
        moved = {key: value.to(device) for key, value in inputs.items()}
        return model(**moved).logits.detach().float().cpu()

    def difference(tree_logits: Any) -> float:
        worst = 0.0
        for member, path in zip(row.members, row.paths):
            got = tree_logits[0, list(path)]
            want = plain[member, : len(path)]
            worst = max(worst, float((got - want).abs().max()))
        return worst

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            plain = logits(input_ids=plain_ids, attention_mask=plain_mask)
            noise = 0.0
            for index, (ids, _labels) in enumerate(examples):
                alone = logits(input_ids=torch.tensor([ids]))
                noise = max(noise, float((alone[0] - plain[index, : len(ids)]).abs().max()))
            error = difference(
                logits(
                    input_ids=packed["input_ids"],
                    attention_mask=packed["attention_mask"],
                    position_ids=packed["position_ids"],
                )
            )

            def wrong(**changed: Any) -> Optional[float]:
                inputs = {
                    "input_ids": packed["input_ids"],
                    "attention_mask": packed["attention_mask"],
                    "position_ids": packed["position_ids"],
                }
                inputs.update(changed)
                try:
                    return difference(logits(**inputs))
                except (RuntimeError, ValueError, IndexError):
                    return None

            control = wrong(position_ids=torch.arange(len(row))[None])
            causal = torch.ones(len(row), len(row), dtype=torch.bool).tril()[None, None]
            if additive_dtype is not None:
                causal = torch.zeros(causal.shape, dtype=additive_dtype).masked_fill(
                    ~causal, torch.finfo(additive_dtype).min
                )
            mask_control = wrong(attention_mask=causal)
    finally:
        if was_training:
            model.train()
    scale = max(
        float(plain[index, : len(ids)].abs().max()) for index, (ids, _l) in enumerate(examples)
    )
    floor = VERIFY_FLOOR_BY_DTYPE.get(dtype_name, VERIFY_FLOOR_BY_DTYPE["float32"]) * scale
    bound = max(VERIFY_NOISE_FACTOR * noise, floor)

    def speaks(value: Optional[float]) -> bool:
        return value is not None and value == value and value > bound

    reason = ""
    if not (error == error and error <= bound):
        reason = (
            f"its logits differ from the ordinary batch's by {error:.3g}, more than "
            f"{bound:.3g} (this model's own layout noise is {noise:.3g})"
        )
    elif len(row) >= row.source_tokens and len(row.members) < 2:
        reason = "the verification examples did not form a row with two members"
    elif not speaks(mask_control):
        reason = (
            f"a plain causal mask, which lets the examples see each other, moves the "
            f"logits by only {mask_control}, within this model's own layout noise "
            f"({noise:.3g}) in {dtype_name}: nothing can be verified in this dtype"
        )
    elif error > VERIFY_CONTROL_FRACTION * mask_control:
        reason = (
            f"its logits are off by {error:.3g}, too close to what a plain causal "
            f"mask produces ({mask_control:.3g}): the 4D mask is not honoured"
        )
    elif speaks(control) and error > VERIFY_CONTROL_FRACTION * control:
        reason = (
            f"its logits are off by {error:.3g}, too close to what wrong positions "
            f"produce ({control:.3g}): position_ids are not honoured"
        )
    return PrefixTreeVerdict(
        exact=not reason,
        error=error,
        scale=scale,
        control=control,
        mask="bool" if additive_dtype is None else "additive",
        noise=noise,
        bound=bound,
        mask_control=mask_control,
        reason=reason,
    )
