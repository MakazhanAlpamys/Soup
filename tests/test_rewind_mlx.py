"""The MLX side of the rewind flight recorder (`soup_cli.trainer.rewind_mlx`).

The interesting property under test is not arithmetic, it is *plumbing*: mlx-lm
compiles its training step with ``mx.compile(inputs=state, outputs=state)``, so a
per-row array stashed in a plain Python dict inside the loss would be the first
batch forever. ``test_full_mechanism_*`` drives the real shape of that loop --
compiled step, ``mx.eval(state)``, then the next batch pulled through the dataset
wrapper -- and compares every iteration against an uncompiled reference computed
on the weights *before* that iteration's update. A stale write-back fails it.
"""

from __future__ import annotations

from functools import partial

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.tuner.trainer import default_loss  # noqa: E402
from rich.console import Console  # noqa: E402

from soup_cli.trainer import rewind_mlx  # noqa: E402
from soup_cli.trainer.mlx_masking import masked_loss  # noqa: E402
from soup_cli.trainer.rewind_mlx import (  # noqa: E402
    ROW_LOSS_KEY,
    ROW_TOKENS_KEY,
    MlxRewindState,
    make_rewind_loss,
    per_row_masked,
    per_row_span,
    wrap_dataset,
)


class _Tiny(nn.Module):
    """The probe's module: three compiled steps run in well under a second."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.out = nn.Linear(8, 16)

    def __call__(self, x):
        return self.out(self.emb(x))


class _FakeSink:
    """Stands in for ``RewindLog``; the sink protocol is the whole coupling."""

    def __init__(self):
        self.records = []

    def record_batch(self, *, step, micro, rows, row_loss, row_tokens):
        self.records.append(
            {
                "step": step,
                "micro": micro,
                "rows": list(rows),
                "row_loss": list(row_loss),
                "row_tokens": list(row_tokens),
            }
        )


# Six rows of four tokens. Row 1's mask zeroes its prefix and row 4's its
# suffix, so a mask that is not shifted with the targets reduces differently.
_ROWS = [
    ([1, 2, 3, 4], [1, 1, 1, 1]),
    ([2, 3, 4, 5], [0, 0, 1, 1]),
    ([5, 6, 7, 8], [1, 1, 1, 1]),
    ([9, 1, 2, 3], [0, 1, 1, 1]),
    ([4, 5, 6, 7], [1, 1, 0, 0]),
    ([8, 9, 1, 2], [1, 1, 1, 1]),
]
# A fixed "shuffled" fetch order: records must follow fetch order, not row order.
_ORDER = [4, 1, 0, 5, 3, 2]

# Per-token CE and masks used by the hand-computed reduction tests.
_CE = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
_MASKS = np.array([[1, 1, 1, 1], [0, 0, 1, 1]], np.int32)
_LENGTHS = np.array([[1, 3], [2, 3]], np.int32)


def _fresh(*, grad_accum=1, batch_size=2, kind="masked"):
    """A model, an optimizer whose state carries the two rewind keys, and a sink."""
    mx.random.seed(0)
    model = _Tiny()
    opt = optim.Adam(learning_rate=1e-3)
    opt.init(model.trainable_parameters())
    sink = _FakeSink()
    state = MlxRewindState(
        sink, optimizer_state=opt.state, batch_size=batch_size, grad_accum=grad_accum
    )
    return model, opt, sink, state, make_rewind_loss(state, kind=kind)


def _batch(rows):
    toks, masks = zip(*rows)
    return mx.array(np.array(toks, np.int32)), mx.array(np.array(masks, np.int32))


def _reference_rows(model, batch, masks):
    """Per-row loss on the CURRENT weights, uncompiled, independent of the module."""
    ce = nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:])
    m = masks[:, 1:]
    row_loss = (ce * m).astype(mx.float32).sum(axis=1) / mx.maximum(m.sum(axis=1), 1)
    return row_loss.tolist(), [int(x) for x in m.sum(axis=1).tolist()]


def _drive(n_iters, *, grad_accum=1, batch_size=2):
    """Run mlx-lm's loop shape: compiled step, mx.eval(state), then the next batch.

    The batch generator is pulled lazily by ``zip``, so each batch's
    ``dataset[j]`` fetches happen *after* the previous step and its ``mx.eval``
    -- which is exactly the sequencing ``flush()`` depends on.
    """
    model, opt, sink, state, loss = _fresh(grad_accum=grad_accum, batch_size=batch_size)
    model.train()
    dataset = wrap_dataset(_ROWS, state)
    tracked = [model.state, opt.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, loss)

    @partial(mx.compile, inputs=tracked, outputs=tracked)
    def step(batch, masks):
        (lvalue, toks), grad = loss_value_and_grad(model, batch, masks)
        opt.update(model, grad)
        return lvalue, toks

    def batches():
        for i in range(0, len(_ORDER), batch_size):
            ids = _ORDER[i : i + batch_size]
            batch, masks = _batch([dataset[j] for j in ids])
            yield batch, masks, ids

    references = []
    for _, (batch, masks, ids) in zip(range(n_iters), batches()):
        row_loss, row_tokens = _reference_rows(model, batch, masks)
        references.append({"ids": list(ids), "row_loss": row_loss, "row_tokens": row_tokens})
        lvalue, toks = step(batch, masks)
        mx.eval(tracked, lvalue, toks)
    return sink, state, references


# --- 1. the per-row reductions against a hand computation -------------------


def test_per_row_masked_matches_numpy():
    m = _MASKS[:, 1:]
    want_loss = (_CE * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)
    row_loss, row_tokens = per_row_masked(mx.array(_CE), mx.array(_MASKS))
    # Row 1 is the discriminating one: 11/2 shifted, 6/1 unshifted.
    assert want_loss.tolist() == pytest.approx([2.0, 5.5], abs=1e-6)
    assert row_loss.tolist() == pytest.approx(want_loss.tolist(), abs=1e-6)
    assert [int(x) for x in row_tokens.tolist()] == m.sum(axis=1).tolist()


def test_per_row_span_matches_numpy():
    steps = np.arange(1, _CE.shape[1] + 1)
    m = ((steps >= _LENGTHS[:, 0:1]) & (steps <= _LENGTHS[:, 1:])).astype(np.int32)
    want_loss = (_CE * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)
    row_loss, row_tokens = per_row_span(mx.array(_CE), mx.array(_LENGTHS))
    assert want_loss.tolist() == pytest.approx([2.0, 5.5], abs=1e-6)
    assert row_loss.tolist() == pytest.approx(want_loss.tolist(), abs=1e-6)
    assert [int(x) for x in row_tokens.tolist()] == m.sum(axis=1).tolist()


# --- 2. the scalar the trainer sees is bit-for-bit the reference's -----------


@pytest.mark.parametrize("training", [True, False])
def test_masked_rewind_loss_matches_masked_loss(training):
    model, _opt, _sink, _state, loss = _fresh(kind="masked")
    model.train() if training else model.eval()
    batch, masks = _batch(_ROWS[:2])
    want, want_ntoks = masked_loss(model, batch, masks)
    got, got_ntoks = loss(model, batch, masks)
    mx.eval(want, got)
    assert got.item() == pytest.approx(want.item(), abs=1e-6)
    assert int(got_ntoks.item()) == int(want_ntoks.item())


def test_masked_rewind_loss_clamps_an_all_masked_batch():
    """`masked_loss` divides by max(ntoks, 1); a batch with nothing supervised
    must contribute exactly 0.0, not a nan that poisons every later gradient."""
    model, _opt, _sink, _state, loss = _fresh(kind="masked")
    model.train()
    batch, _ = _batch(_ROWS[:2])
    masks = mx.zeros(batch.shape, mx.int32)
    want, want_ntoks = masked_loss(model, batch, masks)
    got, got_ntoks = loss(model, batch, masks)
    mx.eval(want, got)
    assert want.item() == 0.0
    assert got.item() == pytest.approx(want.item(), abs=1e-6)
    assert int(got_ntoks.item()) == int(want_ntoks.item()) == 0


@pytest.mark.parametrize("training", [True, False])
def test_span_rewind_loss_matches_default_loss(training):
    model, _opt, _sink, _state, loss = _fresh(kind="span")
    model.train() if training else model.eval()
    batch, _ = _batch(_ROWS[:2])
    lengths = mx.array(_LENGTHS)
    want, want_ntoks = default_loss(model, batch, lengths)
    got, got_ntoks = loss(model, batch, lengths)
    mx.eval(want, got)
    assert got.item() == pytest.approx(want.item(), abs=1e-6)
    assert int(got_ntoks.item()) == int(want_ntoks.item())


# --- 3. the whole mechanism through a compiled step -------------------------


def test_full_mechanism_records_fetched_rows_and_fresh_losses():
    sink, state, references = _drive(3)
    state.flush()
    assert len(sink.records) == 3
    for record, reference in zip(sink.records, references):
        assert record["rows"] == reference["ids"]
        assert record["row_tokens"] == reference["row_tokens"]
        assert record["row_loss"] == pytest.approx(reference["row_loss"], abs=1e-4)
    # Distinct weights every iteration => distinct per-row losses. Equal values
    # across records would mean a stale write-back that happened to match once.
    assert len({tuple(r["row_loss"]) for r in sink.records}) == 3
    assert [(r["step"], r["micro"]) for r in sink.records] == [(1, 0), (2, 0), (3, 0)]
    assert state.dropped == 0
    assert state.pending is False
    assert state.failed is False


# --- 4. step / micro accounting ---------------------------------------------


def test_grad_accum_two_splits_iterations_into_step_and_micro():
    sink, state, _ = _drive(3, grad_accum=2)
    state.flush()
    assert [(r["step"], r["micro"]) for r in sink.records] == [(1, 0), (1, 1), (2, 0)]


# --- 5. evaluation must not record ------------------------------------------


def test_eval_pass_records_nothing():
    model, opt, sink, state, loss = _fresh()
    model.eval()
    batch, masks = _batch(_ROWS[:2])
    scalar, ntoks = loss(model, batch, masks)
    mx.eval(scalar, ntoks)
    assert state.pending is False
    assert opt.state[ROW_LOSS_KEY].tolist() == [0.0, 0.0]
    assert opt.state[ROW_TOKENS_KEY].tolist() == [0.0, 0.0]
    state.flush()
    assert sink.records == []


# --- 6. flush is driven by the pending flag, nothing else -------------------


def test_flush_without_a_pending_step_is_a_no_op():
    _model, _opt, sink, state, _loss = _fresh()
    state.flush()
    assert sink.records == []

    sink, state, _ = _drive(3)
    # The third iteration is still pending: nothing fetched after its step.
    assert len(sink.records) == 2
    state.flush()
    assert len(sink.records) == 3
    state.flush()
    assert len(sink.records) == 3


def test_dataset_wrapper_passes_through_length_and_attributes():
    class _Inner:
        def __init__(self):
            self.rows = _ROWS

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, j):
            return self.rows[j]

        def itemlen(self, j):
            return len(self.rows[j][0])

    _model, _opt, _sink, state, _loss = _fresh()
    dataset = wrap_dataset(_Inner(), state)
    assert len(dataset) == 6
    assert dataset.itemlen(0) == 4  # __getattr__ passthrough, no id recorded
    assert dataset[3] == _ROWS[3]


# --- 7. a failure disables the recorder after one warning -------------------


def test_fail_warns_once_then_stays_silent(monkeypatch):
    recorder = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_mlx, "console", recorder)
    _model, _opt, sink, state, _loss = _fresh()

    state.fail(RuntimeError("boom"))
    state.fail(RuntimeError("second failure"))

    lines = [line for line in recorder.export_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert "Rewind recorder disabled:" in lines[0]
    assert "boom" in lines[0]
    assert "second failure" not in recorder.export_text()

    assert state.failed is True
    state.pending = True
    state.flush()
    assert sink.records == []


def test_a_sink_that_raises_disables_rather_than_killing_training(monkeypatch):
    recorder = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_mlx, "console", recorder)

    class _Exploding:
        def record_batch(self, **_kwargs):
            raise OSError("disk full")

    mx.random.seed(0)
    model = _Tiny()
    opt = optim.Adam(learning_rate=1e-3)
    opt.init(model.trainable_parameters())
    state = MlxRewindState(_Exploding(), optimizer_state=opt.state, batch_size=2, grad_accum=1)
    state.on_index(0)
    state.on_index(1)
    state.mark_pending()
    state.flush()  # must not raise into training
    assert state.failed is True
    assert "disk full" in recorder.export_text()


def test_reads_outside_the_batch_loop_disable_rather_than_misattribute(monkeypatch):
    """mlx-lm's own `iterate_batches` sorts the dataset by length through
    `__getitem__` (`trainer.py:111-115`) because a wrapper is not a `CacheDataset`.
    That queues one id per row before the first batch; recording the oldest ids
    would name innocent rows, which is the one thing this feature must never do."""
    recorder = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_mlx, "console", recorder)
    _model, _opt, sink, state, _loss = _fresh()
    dataset = wrap_dataset(_ROWS, state)
    for j in range(len(_ROWS)):  # the length sort
        dataset[j]
    dataset[4], dataset[1]  # then the real batch
    state.mark_pending()
    state.flush()
    assert sink.records == []
    assert state.failed is True
    assert "row ids queued for a batch of 2" in recorder.export_text()


def test_a_short_id_queue_is_dropped_and_counted():
    _model, _opt, sink, state, _loss = _fresh()
    state.on_index(7)  # only one id for a batch_size=2 record
    state.mark_pending()
    state.flush()
    assert sink.records == []
    assert state.dropped == 1
    assert state.pending is False
    assert state.failed is False


# --- degradation paths found in review ---------------------------------------


def test_short_queue_leftovers_do_not_seed_the_next_group():
    """A dropped iteration's ids used to stay queued and join the next group,
    producing a record that spans two iterations with ``failed`` still False."""
    _model, _opt, sink, state, _loss = _fresh()
    state.on_index(7)  # one id for a batch of 2: this iteration is dropped
    state.mark_pending()
    state.flush()
    assert state.dropped == 1

    state.on_index(1)
    state.on_index(2)  # a full group completes the next iteration
    state.flush()

    assert [r["rows"] for r in sink.records] == [[1, 2]]
    assert state.failed is False


def test_a_dropped_iteration_still_consumes_its_step_number():
    """``step`` joins the log to the loss curve; a drop must not shift later steps."""
    _model, _opt, sink, state, _loss = _fresh()
    state.on_index(7)
    state.mark_pending()
    state.flush()  # iteration 1: dropped

    state.on_index(1)
    state.on_index(2)
    state.flush()  # iteration 2: recorded

    assert [r["step"] for r in sink.records] == [2]


def test_per_row_arrays_narrower_than_the_batch_disable(monkeypatch):
    """A per-worker batch smaller than the configured one must not be recorded."""
    recorder = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_mlx, "console", recorder)
    _model, opt, sink, state, _loss = _fresh(batch_size=4)
    opt.state[ROW_LOSS_KEY] = mx.zeros((2,), mx.float32)
    opt.state[ROW_TOKENS_KEY] = mx.zeros((2,), mx.float32)
    state.mark_pending()
    for j in range(4):
        state.on_index(j)

    state.flush()

    assert state.failed is True
    assert sink.records == []
    assert "per-worker batch size" in recorder.export_text()


def test_internal_failure_names_the_exception_type(monkeypatch):
    recorder = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_mlx, "console", recorder)
    _model, opt, sink, state, _loss = _fresh()
    state.on_index(0)
    state.on_index(1)
    state.mark_pending()
    del opt.state[ROW_LOSS_KEY]

    state.flush()  # must not raise

    assert state.failed is True
    assert "KeyError" in recorder.export_text()


def test_row_tokens_come_back_float32_from_both_reductions():
    """The state slots are float32; an int array written back forces a retrace."""
    _, masked_tokens = per_row_masked(mx.array(_CE), mx.array(_MASKS))
    _, span_tokens = per_row_span(mx.array(_CE), mx.array(_LENGTHS))
    assert masked_tokens.dtype == mx.float32
    assert span_tokens.dtype == mx.float32


class _Processed(list):
    """Rows already in mlx-lm's ``(tokens, offset)`` shape; ``process`` is identity."""

    def process(self, row):
        return row


def _cache_dataset():
    from mlx_lm.tuner.datasets import CacheDataset

    return CacheDataset(_Processed([(list(range(1, n + 3)), 0) for n in range(6)]))


def test_mlx_lm_iterate_batches_fetches_exactly_one_batch_per_iteration():
    """The wrapper passes ``isinstance(dataset, CacheDataset)``, so mlx-lm's length
    sort goes through ``itemlen`` and never reaches ``__getitem__``."""
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import iterate_batches

    _model, _opt, _sink, state, _loss = _fresh(batch_size=2)
    wrapped = wrap_dataset(_cache_dataset(), state)
    assert isinstance(wrapped, CacheDataset)

    batches = iterate_batches(dataset=wrapped, batch_size=2, max_seq_length=64)
    queued = []
    for _ in range(3):
        next(batches)
        queued.append(len(state._ids))  # unarmed: nothing is popped, ids accumulate

    assert queued == [2, 4, 6]


def test_control_a_plain_wrapper_is_read_row_by_row_for_the_length_sort():
    """The regression the CacheDataset base prevents, kept visible: without it the
    sort reads every row before the first batch."""
    from mlx_lm.tuner.trainer import iterate_batches

    _model, _opt, _sink, state, _loss = _fresh(batch_size=2)
    inner = _cache_dataset()

    class _Plain:
        def __len__(self):
            return len(inner)

        def __getitem__(self, j):
            state.on_index(int(j))
            return inner[j]

    next(iterate_batches(dataset=_Plain(), batch_size=2, max_seq_length=64))

    assert len(state._ids) == 6 + 2
