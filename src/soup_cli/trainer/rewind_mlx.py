"""MLX side of the rewind flight recorder (see ``monitoring/rewind_log.py``).

Row identity comes from a dataset wrapper: both ``mlx_lm.tuner.trainer.iterate_batches``
(``trainer.py:143``) and Soup's ``masked_iterate_batches`` (``mlx_masking.py:276``) build
a batch with ``[dataset[j] for j in batch_idx[i]]``, so every row announces itself, in
batch order, in this process. One iteration of mlx-lm's loop = one record.

Per-row loss cannot leave the loss the way row ids do. ``train()`` compiles its step
with ``@partial(mx.compile, inputs=state, outputs=state)`` over
``state = [model.state, optimizer.state, mx.random.state]`` (``trainer.py:246-262``), so
the Python body of the loss runs only while tracing -- a per-row array stashed in a plain
dict would be the first batch forever. The channel out is the one mlx-lm already relies
on for the optimizer's own moments: **arrays reassigned into ``optimizer.state`` inside
the compiled region are written back after every call**. Creating the two keys *before*
``mx.compile`` sees ``state`` is what makes them part of the tracked state; measured on
mlx 0.32.2 / mlx-lm 0.31.3, values come back fresh each call and equal an uncompiled
reference.

Side effects in the loss are therefore *not* a per-iteration signal either -- the
recorder is driven by the dataset fetches, which run in plain Python, and the loss body
only arms it. The flush point falls out of the loop's sequencing (``trainer.py:319-328``):
``step(batch)`` -> ``mx.eval(state, ...)`` -> the ``for`` pulls the next batch, which
fetches ``dataset[j]``. So the wrapper's first ``__getitem__`` after a step is the first
moment the previous iteration's arrays are both written back and already evaluated.
The final iteration has no successor, hence the explicit ``flush()`` after ``train()``.

``evaluate()`` calls ``model.eval()`` (``trainer.py:186``) and the loop restores
``model.train()`` (``trainer.py:299``) while reusing the same ``loss`` callable, so
``model.training`` is what separates a training call from a validation one.

Nothing here may raise into training: a recorder that stops a run is worse than no
recorder. Every failure path ends in ``fail()``, which warns once and goes quiet.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Literal, Protocol, Sequence

from rich.console import Console
from rich.markup import escape

console = Console()

# Keys added to ``optimizer.state``. Namespaced so they cannot collide with an
# optimizer's own moment names, and stable because the reader has no schema for
# them -- they never reach disk, only ``flush()``.
ROW_LOSS_KEY = "rewind_row_loss"
ROW_TOKENS_KEY = "rewind_row_tokens"


class RewindSink(Protocol):
    """What this module needs of a log. ``RewindLog`` satisfies it; tests inject a list.

    Keeping the dependency at one method decouples the MLX wiring from the log
    format, and keeps this module free of the heavy-import question entirely.
    """

    def record_batch(
        self,
        *,
        step: int,
        micro: int,
        rows: Sequence[int],
        row_loss: Sequence[float],
        row_tokens: Sequence[int],
    ) -> None: ...


def _row_reduce(ce: Any, mask: Any):
    """Mean CE over a row's supervised tokens, and how many there were.

    The clamped denominator mirrors ``masked_loss``: a row with nothing
    supervised (every token truncated away, or a prompt-only row) has a zero
    numerator, so the clamp reports 0.0 rather than a nan. A nan here would
    reach the log and, worse, read as a spike in the very report this feature
    exists to produce.
    """
    import mlx.core as mx

    row_tokens = mask.sum(axis=1).astype(mx.float32)
    row_loss = (ce * mask).astype(mx.float32).sum(axis=1) / mx.maximum(row_tokens, 1)
    # Both float32: the arrays are written back into float32 state slots, and an
    # int array there would change the state's dtype and force one extra retrace.
    return row_loss, row_tokens


def per_row_masked(ce: Any, masks: Any):
    """Per-row reduction for the masked path: ``ce`` is raw ``(B, T-1)``, ``masks`` ``(B, T)``.

    ``masks[:, 1:]`` is the alignment, for the reason spelled out in
    ``mlx_masking.masked_loss``: target position ``i`` is original token
    ``i + 1``, so an unshifted mask supervises the token *before* each assistant
    token. Same shift as the scalar, or the per-row numbers would not sum to it.
    """
    return _row_reduce(ce, masks[:, 1:])


def per_row_span(ce: Any, lengths: Any):
    """Per-row reduction for mlx-lm's unmasked path, from its ``(offset, length)`` pair.

    Mirrors ``default_loss`` (``trainer.py:92-93``): one contiguous supervised
    span per row, rebuilt from the pair rather than carried as an array.
    """
    import mlx.core as mx

    steps = mx.arange(1, ce.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    return _row_reduce(ce, mask)


class MlxRewindState:
    """The FIFO of fetched row ids, the iteration counter, and the state handles.

    Constructed before ``mx.compile`` runs, because creating the two keys in
    ``optimizer.state`` is what enrolls them in the compiled step's tracked
    state. Their shape is ``(batch_size,)`` and never changes: both iterators drop
    the remainder, so every batch has the same row count and the compiled graph
    does not retrace on their account. ``batch_size`` is the PER-WORKER count --
    under a ``comm_group`` each process fetches ``args.batch_size // world_size``
    rows per iteration -- and a mismatch disables the recorder rather than
    misattributing rows.

    A dropped iteration still consumes a step number, so ``step`` always matches
    mlx-lm's own iteration count.
    """

    def __init__(
        self,
        sink: RewindSink,
        *,
        optimizer_state: Dict[str, Any],
        batch_size: int,
        grad_accum: int,
    ) -> None:
        import mlx.core as mx

        self.sink = sink
        self.optimizer_state = optimizer_state
        self.batch_size = int(batch_size)
        self.grad_accum = max(int(grad_accum), 1)
        self.failed = False
        self.pending = False
        self.dropped = 0
        self.it = 0
        self._ids: list[int] = []
        self._armed = False
        optimizer_state[ROW_LOSS_KEY] = mx.zeros((self.batch_size,), mx.float32)
        optimizer_state[ROW_TOKENS_KEY] = mx.zeros((self.batch_size,), mx.float32)

    def on_index(self, j: int) -> None:
        """Record one fetched row id. Order is the batch's order, so FIFO is enough.

        Completing a group of ``batch_size`` ids is also what marks the previous
        iteration flushable, for the reason in ``mark_pending``: the fetches are
        the only per-iteration signal that runs outside the compiled step.
        """
        if self.failed:
            return
        self._ids.append(int(j))
        if self._armed and len(self._ids) >= self.batch_size:
            self.pending = True

    def mark_pending(self) -> None:
        """A training call ran the loss body, so the two state keys are live.

        This cannot be the per-iteration signal, and that is not a style
        preference: the loss body is inside ``mx.compile``, so it executes only
        while tracing. Measured on mlx 0.32.2, three steps over two batch shapes
        trace twice and replay once -- a recorder driven by this alone logs the
        first iteration and then goes quiet forever (it did, on the first run of
        ``test_full_mechanism_*``). What it does do is *arm* the recorder, which
        keeps a wired-but-unwrapped loss from logging ids against zeroed arrays,
        and cover iteration one, whose ids were queued before this first trace.
        It re-runs on every retrace (a new sequence-length bucket); that is
        idempotent.
        """
        self._armed = True
        self.pending = True

    def flush(self) -> None:
        """Turn the previous iteration's arrays plus the oldest ids into one record."""
        if self.failed or not self.pending:
            return
        self.pending = False
        try:
            # Count the iteration before any early return: a dropped iteration
            # still owns a step number, or every later record would claim a step
            # one lower than the loss curve it is joined to.
            self.it += 1
            if len(self._ids) < self.batch_size:
                # Fewer ids than rows means the fetches and the loss calls have
                # drifted apart. Guessing would name innocent rows, so the record is
                # dropped -- and the leftovers with it, or they would seed the next
                # group and produce a record spanning two iterations.
                self.dropped += 1
                self._ids.clear()
                return
            if len(self._ids) > self.batch_size:
                # Under correct sequencing the queue holds exactly one batch here.
                # More means the dataset was read outside the batch loop -- e.g.
                # mlx-lm's length sort over a wrapper that is not a CacheDataset
                # (see wrap_dataset). A forensic tool that lies is worse than one
                # that stops.
                self.fail(
                    RuntimeError(
                        f"{len(self._ids)} row ids queued for a batch of "
                        f"{self.batch_size}; the dataset is being read outside the "
                        "training loop, so row identity cannot be trusted"
                    )
                )
                return
            row_loss = [float(x) for x in self.optimizer_state[ROW_LOSS_KEY].tolist()]
            row_tokens = [int(x) for x in self.optimizer_state[ROW_TOKENS_KEY].tolist()]
            if len(row_loss) != self.batch_size or len(row_tokens) != self.batch_size:
                self.fail(
                    RuntimeError(
                        f"per-row arrays are {len(row_loss)} wide for a batch of "
                        f"{self.batch_size}; batch_size must be the per-worker batch size"
                    )
                )
                return
            rows = self._ids[: self.batch_size]
            del self._ids[: self.batch_size]
            step = (self.it - 1) // self.grad_accum + 1
            micro = (self.it - 1) % self.grad_accum
        except Exception as exc:  # a bookkeeping bug: never raise into training
            self.fail(exc)
            return
        try:
            self.sink.record_batch(
                step=step,
                micro=micro,
                rows=rows,
                row_loss=row_loss,
                row_tokens=row_tokens,
            )
        except Exception as exc:  # the sink's own failure, reported separately
            self.fail(exc)

    def fail(self, exc: BaseException) -> None:
        """Disable the recorder after exactly one warning; later calls are silent."""
        if self.failed:
            return
        self.failed = True
        self.pending = False
        self._armed = False
        self._ids.clear()
        try:
            console.print(
                "[yellow]Rewind recorder disabled:[/] "
                f"{escape(type(exc).__name__)}: {escape(str(exc))}"
            )
        except Exception:  # a broken console must not stop training
            pass


@functools.lru_cache(maxsize=None)
def _recording_dataset_class() -> type:
    """Build (once) the ``CacheDataset`` subclass ``wrap_dataset`` instantiates.

    Lazy because ``mlx_lm`` is a heavy import. Cached so every wrapper shares one
    class, which keeps ``isinstance`` checks and ``repr`` stable.
    """
    from mlx_lm.tuner.datasets import CacheDataset  # heavy: imported at call time

    class IndexRecordingDataset(CacheDataset):
        def __init__(self, inner: Any, state: MlxRewindState) -> None:
            # CacheDataset.__init__ is deliberately NOT called. Re-wrapping an
            # already-cached dataset would make ``itemlen`` the length of the
            # ``(tokens, offset)`` tuple -- 2 for every row -- and fill a second
            # cache through ``inner.process``, which a CacheDataset lacks. All
            # three methods that read CacheDataset's own attributes are
            # overridden below, so ``_data`` / ``_proc_data`` are never consulted.
            self.__dict__["_inner"] = inner
            self.__dict__["_state"] = state

        def itemlen(self, idx: int) -> int:
            # The length sort must never reach __getitem__ (it would queue every
            # row id). Delegate, with upstream's own fallback for a plain inner.
            inner = self.__dict__["_inner"]
            fn = getattr(inner, "itemlen", None)
            return fn(idx) if fn is not None else len(inner[idx][0])

        def __len__(self) -> int:
            return len(self.__dict__["_inner"])

        def __getitem__(self, j):
            state = self.__dict__["_state"]
            state.flush()
            state.on_index(int(j))
            return self.__dict__["_inner"][j]

        def __getattr__(self, name: str) -> Any:
            # Forward-compat passthrough for anything a future mlx-lm probes.
            # Dunders are refused so copy/pickle never pick up the inner's own.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            try:
                inner = self.__dict__["_inner"]
            except KeyError:
                raise AttributeError(name) from None
            return getattr(inner, name)

    return IndexRecordingDataset


def wrap_dataset(inner: Any, state: MlxRewindState) -> Any:
    """Wrap the TRAINING dataset handed to mlx-lm's ``train()`` so row fetches announce ids.

    ``__getitem__`` is both the row-id source and the flush point: the first fetch
    after a step is the first moment the previous iteration's per-row arrays are
    written back and evaluated.

    The wrapper is a ``CacheDataset`` subclass because that ``isinstance`` is the
    only thing mlx-lm's ``iterate_batches`` checks before length-sorting through
    ``itemlen`` (``trainer.py:111-115``). A wrapper that fails it is sorted with
    ``len(dataset[idx][0])``, reading every row through ``__getitem__`` --
    measured: 8 fetches before the first 2-row batch over 6 rows, against 2/2/2
    once the check passes. Soup's ``masked_iterate_batches`` fetches exactly one
    batch per iteration either way.

    **Wrap the training dataset only.** mlx-lm runs ``evaluate()`` before the
    step of the iteration whose ids are already queued (``trainer.py:277-299`` vs
    ``:319``); a validation dataset wrapped with the same state would flush those
    ids against the previous iteration's losses.
    """
    return _recording_dataset_class()(inner, state)


def make_rewind_loss(state: MlxRewindState, *, kind: Literal["masked", "span"]):
    """Return ``loss(model, batch, third)`` with mlx-lm's ``(scalar, ntoks)`` contract.

    The reference body is reimplemented rather than wrapped: wrapping an inner
    loss and reducing again would mean a second ``model(inputs)``, doubling the
    forward cost of every step. One forward here feeds both the scalar the
    trainer sees -- the reference's expression unchanged, including
    ``masked_loss``'s ``maximum(ntoks, 1)`` clamp and ``default_loss``'s plain
    ``/ ntoks`` -- and the per-row arrays.

    ``third`` is the second element of whatever ``iterate_batches`` yields: a
    per-token mask for ``kind="masked"``, an ``(offset, length)`` pair for
    ``kind="span"``.
    """
    if kind not in ("masked", "span"):
        raise ValueError(f"kind must be 'masked' or 'span', got {kind!r}")

    def loss(model, batch, third):
        import mlx.core as mx
        import mlx.nn as nn

        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)

        raw = nn.losses.cross_entropy(logits, targets)
        if kind == "masked":
            mask = third[:, 1:]
            ntoks = mask.sum()
            scalar = (raw * mask).astype(mx.float32).sum() / mx.maximum(ntoks, 1)
        else:
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= third[:, 0:1], steps <= third[:, 1:])
            ntoks = mask.sum()
            scalar = (raw * mask).astype(mx.float32).sum() / ntoks

        # Validation reuses this callable; only training calls own an iteration.
        if getattr(model, "training", False) and not state.failed:
            if kind == "masked":
                row_loss, row_tokens = per_row_masked(raw, third)
            else:
                row_loss, row_tokens = per_row_span(raw, third)
            state.optimizer_state[ROW_LOSS_KEY] = row_loss
            state.optimizer_state[ROW_TOKENS_KEY] = row_tokens
            state.mark_pending()

        return scalar, ntoks

    return loss
