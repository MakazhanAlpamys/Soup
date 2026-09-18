"""Tests for the rewind.jsonl flight-recorder writer/reader.

Covers: header shape, batch round-trip, length-mismatch drops, NaN
encoding as JSON null, drop-not-disable on a bad scalar, disable-on-OSError
with a single warning, parent-dir creation, one-file-per-run rotation,
token-weighted step_loss, torn/malformed line skipping, malformed-file
errors, and fingerprint key-order independence.

Stdlib + rich only: a "numpy-like scalar" is simulated with a small class
whose ``__float__`` raises, never by importing numpy/torch/mlx.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.monitoring import rewind_log as rewind_log_module
from soup_cli.monitoring.rewind_log import (
    REWIND_LOG_VERSION,
    RewindLog,
    RewindLogError,
    dataset_fingerprint,
    read_rewind_log,
)


class _BadScalar:
    """Stand-in for a framework scalar that refuses ``float()``.

    Mirrors e.g. a multi-element array scalar: ``float(x)`` raises TypeError.
    Defined here so the test module stays free of torch/numpy/mlx imports.
    """

    def __float__(self) -> float:
        raise TypeError("only size-1 arrays can be converted to Python scalars")


def _make_log(tmp_path: Path, **overrides) -> RewindLog:
    kwargs = dict(
        path=tmp_path / "rewind.jsonl",
        backend="sft",
        task="chat",
        n_rows=100,
        batch_size=4,
        grad_accum=2,
        dataset_fingerprint="deadbeef",
    )
    kwargs.update(overrides)
    return RewindLog(**kwargs)


def _header_line(**overrides) -> str:
    header = {
        "kind": "header",
        "version": REWIND_LOG_VERSION,
        "backend": "sft",
        "task": "chat",
        "n_rows": 1,
        "batch_size": 1,
        "grad_accum": 1,
        "dataset_fingerprint": "x",
        "created": 0.0,
    }
    header.update(overrides)
    return json.dumps(header) + "\n"


def test_header_written_on_construction(tmp_path):
    log = _make_log(tmp_path)
    log.close()
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header["kind"] == "header"
    assert header["version"] == 1
    assert REWIND_LOG_VERSION == 1
    assert header["backend"] == "sft"
    assert header["task"] == "chat"
    assert header["n_rows"] == 100
    assert header["batch_size"] == 4
    assert header["grad_accum"] == 2
    assert header["dataset_fingerprint"] == "deadbeef"
    assert isinstance(header["created"], float)


def test_record_batch_round_trip_preserves_order_and_types(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[3, 7], row_loss=[0.5, 1.5], row_tokens=[10, 20])
    log.record_batch(step=1, micro=1, rows=[9], row_loss=[2.0], row_tokens=[5])
    log.close()

    # Control for the drop counter: a healthy run drops nothing and stays on.
    assert log.dropped == 0
    assert log.disabled is False

    run = read_rewind_log(log.path)
    assert run.steps() == [1]
    batches = run.batches_for(1)
    assert len(batches) == 2

    first = batches[0]
    assert first.step == 1
    assert first.micro == 0
    assert first.rows == (3, 7)
    assert first.row_loss == (0.5, 1.5)
    assert first.row_tokens == (10, 20)
    assert isinstance(first.rows[0], int)
    assert isinstance(first.row_loss[0], float)
    assert isinstance(first.row_tokens[0], int)

    second = batches[1]
    assert second.micro == 1
    assert second.rows == (9,)


def test_close_is_idempotent(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[4])
    log.close()
    log.close()
    run = read_rewind_log(log.path)
    assert len(run.batches_for(1)) == 1


def test_length_mismatch_is_dropped_without_disabling(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1, 2], row_loss=[0.1], row_tokens=[10, 10])
    assert log.dropped == 1
    # A shape bug in one batch must not cost the rest of the run.
    assert log.disabled is False
    log.record_batch(step=1, micro=1, rows=[5], row_loss=[0.2], row_tokens=[10])
    log.close()
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # header + the good batch only
    assert log.dropped == 1


def test_nan_loss_written_as_null_and_read_back_as_nan(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1], row_loss=[float("nan")], row_tokens=[10])
    log.close()

    raw = log.path.read_text(encoding="utf-8")
    batch_line = raw.splitlines()[1]
    assert "null" in batch_line
    assert "NaN" not in batch_line

    run = read_rewind_log(log.path)
    loss = run.batches_for(1)[0].row_loss[0]
    assert math.isnan(loss)


@pytest.mark.parametrize("arg", ["rows", "row_loss", "row_tokens"])
def test_generator_argument_is_dropped_not_raised(tmp_path, arg):
    """A generator has no ``__len__``; the writer must count it, not explode.

    ``rewind_hf`` calls ``record_batch`` with no enclosing try, so a raise
    escaping this method takes the whole training run down.
    """
    log = _make_log(tmp_path)
    kwargs = {"rows": [1, 2], "row_loss": [0.5, 0.5], "row_tokens": [10, 10]}
    kwargs[arg] = (v for v in kwargs[arg])

    log.record_batch(step=1, micro=0, **kwargs)  # must not raise
    assert log.dropped == 1
    assert log.disabled is False

    log.record_batch(step=2, micro=0, rows=[3], row_loss=[0.25], row_tokens=[8])
    log.close()
    assert log.dropped == 1
    assert read_rewind_log(log.path).steps() == [2]


@pytest.mark.parametrize("bad", [None, _BadScalar()], ids=["none", "numpy_like_scalar"])
def test_bad_loss_scalar_drops_batch_but_keeps_logging(tmp_path, bad):
    """One unusable scalar must not kill the log for the rest of the run."""
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1, 2], row_loss=[1.0, bad], row_tokens=[10, 10])
    assert log.dropped == 1
    assert log.disabled is False

    # The decisive part: a LATER valid batch is still written.
    log.record_batch(step=2, micro=0, rows=[3], row_loss=[0.25], row_tokens=[8])
    log.close()
    assert log.dropped == 1

    run = read_rewind_log(log.path)
    assert run.steps() == [2]
    assert run.batches_for(2)[0].row_loss == (0.25,)


def test_bad_row_id_drops_batch_but_keeps_logging(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=["not-an-int"], row_loss=[1.0], row_tokens=[10])
    assert log.dropped == 1
    assert log.disabled is False
    log.record_batch(step=2, micro=0, rows=[3], row_loss=[0.5], row_tokens=[4])
    log.close()
    assert read_rewind_log(log.path).steps() == [2]


@pytest.mark.parametrize("field", ["rows", "row_tokens"])
def test_infinite_integer_field_drops_batch_but_keeps_logging(tmp_path, field):
    """int(float("inf")) raises OverflowError, not ValueError -- it must still be a drop."""
    log = _make_log(tmp_path)
    kwargs = {"rows": [1], "row_loss": [1.0], "row_tokens": [10]}
    kwargs[field] = [float("inf")]
    log.record_batch(step=1, micro=0, **kwargs)
    assert log.dropped == 1
    assert log.disabled is False
    log.record_batch(step=2, micro=0, rows=[3], row_loss=[0.5], row_tokens=[4])
    log.close()
    assert read_rewind_log(log.path).steps() == [2]


def test_missing_parent_directory_is_created(tmp_path):
    """The trainer's output dir may not exist yet — mkdir, don't disable."""
    target = tmp_path / "runs" / "exp1" / "rewind.jsonl"
    assert not target.parent.exists()
    log = _make_log(tmp_path, path=target)
    assert log.disabled is False
    assert target.parent.is_dir()

    log.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[10])
    log.close()
    run = read_rewind_log(target)
    assert run.header["kind"] == "header"
    assert run.batches_for(1)[0].rows == (1,)


def test_previous_run_is_rotated_to_backup(tmp_path):
    path = tmp_path / "rewind.jsonl"
    first = _make_log(tmp_path, path=path)
    first.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[10])
    first.close()
    first_text = path.read_text(encoding="utf-8")
    assert len(first_text.splitlines()) == 2

    second = _make_log(tmp_path, path=path, backend="mlx")
    second.close()

    backup = tmp_path / "rewind.jsonl.1"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == first_text

    new_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(new_lines) == 1  # exactly one header line, no leftover batches
    assert json.loads(new_lines[0])["backend"] == "mlx"


def test_rotation_refuses_to_overwrite_a_symlink_backup(tmp_path):
    path = tmp_path / "rewind.jsonl"
    victim = tmp_path / "victim.txt"
    victim.write_text("do not clobber me\n", encoding="utf-8")

    first = _make_log(tmp_path, path=path)
    first.close()

    backup = tmp_path / "rewind.jsonl.1"
    try:
        backup.symlink_to(victim)
    except (NotImplementedError, OSError):  # unprivileged Windows
        pytest.skip("symlink creation not supported on this platform/user")

    second = _make_log(tmp_path, path=path, backend="mlx")
    second.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[10])
    second.close()

    assert backup.is_symlink()
    assert victim.read_text(encoding="utf-8") == "do not clobber me\n"
    # The new run still logs; rotation is best-effort, logging is not.
    assert second.disabled is False
    assert read_rewind_log(path).batches_for(1)[0].rows == (1,)


def test_oserror_on_append_disables_log_with_single_warning(tmp_path, monkeypatch):
    log = _make_log(tmp_path)

    real_open = Path.open
    call_count = {"n": 0}

    def flaky_open(self, mode="r", *args, **kwargs):
        if self == log.path and "a" in mode:
            call_count["n"] += 1
            raise OSError("disk full")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    # width=10_000 + soft_wrap: tmp_path can be long enough to wrap the path
    # out of a single line, which would make the `str(log.path) in output`
    # assertion fail for reasons that have nothing to do with the code.
    test_console = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(rewind_log_module, "console", test_console)

    log.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[10])
    assert log.disabled is True

    # export_text(clear=False) so a second export doesn't wipe the buffer —
    # rich clears by default.
    output_after_first = test_console.export_text(clear=False)
    assert "Rewind log disabled" in output_after_first
    assert str(log.path) in output_after_first

    # Second call while disabled: silent no-op, no new warning text.
    log.record_batch(step=2, micro=0, rows=[1], row_loss=[1.0], row_tokens=[10])
    output_after_second = test_console.export_text(clear=False)
    assert output_after_second == output_after_first
    assert output_after_second.count("Rewind log disabled") == 1
    assert call_count["n"] == 1


def test_step_loss_is_token_weighted(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1, 2], row_loss=[1.0, 3.0], row_tokens=[10, 30])
    log.record_batch(step=2, micro=0, rows=[1], row_loss=[float("nan")], row_tokens=[10])
    log.close()

    run = read_rewind_log(log.path)
    assert run.step_loss(1) == pytest.approx(2.5)
    assert run.step_loss(1) != pytest.approx(2.0)
    assert math.isnan(run.step_loss(2))
    assert run.step_loss(999) is None


def test_step_loss_zero_tokens_returns_none(tmp_path):
    """Zero supervised tokens means unmeasured, not "the lowest loss seen"."""
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[0])
    log.close()
    run = read_rewind_log(log.path)
    assert run.step_loss(1) is None
    assert run.step_loss(1) != 0.0


def test_header_only_file_reads_as_zero_batches(tmp_path):
    log = _make_log(tmp_path)
    log.close()
    run = read_rewind_log(log.path)
    assert run.batches == ()
    assert run.steps() == []
    assert run.header["backend"] == "sft"


def test_torn_multibyte_last_line_does_not_fail_the_file(tmp_path):
    """A reader racing the writer can catch a line split mid-UTF-8 character.

    Strict decoding raises for the WHOLE file; the reader must decode
    leniently and let the torn line fail ``json.loads`` on its own.
    """
    path = tmp_path / "rewind.jsonl"
    good = json.dumps(
        {
            "kind": "batch",
            "step": 1,
            "micro": 0,
            "rows": [1],
            "row_loss": [0.5],
            "row_tokens": [10],
        }
    )
    with path.open("wb") as fh:
        fh.write(_header_line().encode("utf-8"))
        fh.write((good + "\n").encode("utf-8"))
        # A partial line whose final byte is the lead byte of "é" (0xC3 0xA9).
        fh.write(b'{"kind": "batch", "step": 9, "micro": 0, "tag": "caf\xc3')

    # Control: the bytes really are torn — a strict read blows up.
    with pytest.raises(UnicodeDecodeError):
        path.read_text(encoding="utf-8")

    run = read_rewind_log(path)
    assert run.steps() == [1]  # torn line skipped, complete lines survive
    assert run.batches_for(1)[0].row_loss == (0.5,)


def test_batch_line_missing_step_or_micro_is_skipped(tmp_path):
    path = tmp_path / "rewind.jsonl"
    lines = [
        _header_line(),
        json.dumps({"kind": "batch", "micro": 0, "rows": [1], "row_loss": [1.0]}) + "\n",
        json.dumps({"kind": "batch", "step": 2, "rows": [1], "row_loss": [1.0]}) + "\n",
        json.dumps(
            {
                "kind": "batch",
                "step": 3,
                "micro": 0,
                "rows": [7],
                "row_loss": [0.25],
                "row_tokens": [4],
            }
        )
        + "\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")

    run = read_rewind_log(path)
    assert run.steps() == [3]
    assert run.batches_for(3)[0].rows == (7,)


def test_non_numeric_loss_on_disk_skips_only_that_line(tmp_path):
    path = tmp_path / "rewind.jsonl"
    bad = json.dumps(
        {
            "kind": "batch",
            "step": 1,
            "micro": 0,
            "rows": [1],
            "row_loss": ["oops"],
            "row_tokens": [4],
        }
    )
    good = json.dumps(
        {
            "kind": "batch",
            "step": 2,
            "micro": 0,
            "rows": [2],
            "row_loss": [0.5],
            "row_tokens": [4],
        }
    )
    path.write_text(_header_line() + bad + "\n" + good + "\n", encoding="utf-8")
    assert read_rewind_log(path).steps() == [2]


def test_unknown_kind_line_is_skipped(tmp_path):
    path = tmp_path / "rewind.jsonl"
    future = json.dumps({"kind": "checkpoint", "step": 5, "path": "ckpt-5"})
    good = json.dumps(
        {
            "kind": "batch",
            "step": 1,
            "micro": 0,
            "rows": [1],
            "row_loss": [1.0],
            "row_tokens": [4],
        }
    )
    path.write_text(_header_line() + future + "\n" + good + "\n", encoding="utf-8")
    assert read_rewind_log(path).steps() == [1]


def test_infinity_token_from_a_foreign_writer_reads_as_nan(tmp_path):
    """``json.loads`` accepts the non-standard ``Infinity`` token; we must not
    let an infinite loss through as a real number."""
    path = tmp_path / "rewind.jsonl"
    line = (
        '{"kind": "batch", "step": 1, "micro": 0, "rows": [1, 2], '
        '"row_loss": [Infinity, -Infinity], "row_tokens": [4, 4]}'
    )
    path.write_text(_header_line() + line + "\n", encoding="utf-8")

    run = read_rewind_log(path)
    losses = run.batches_for(1)[0].row_loss
    assert len(losses) == 2
    assert all(math.isnan(v) for v in losses)
    assert not any(math.isinf(v) for v in losses)
    assert math.isnan(run.step_loss(1))


def test_read_rewind_log_raises_for_missing_file(tmp_path):
    with pytest.raises(RewindLogError):
        read_rewind_log(tmp_path / "does_not_exist.jsonl")


def test_read_rewind_log_raises_for_missing_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"kind": "batch", "step": 1}) + "\n", encoding="utf-8")
    with pytest.raises(RewindLogError):
        read_rewind_log(path)


def test_read_rewind_log_raises_for_version_mismatch(tmp_path):
    path = tmp_path / "bad_version.jsonl"
    path.write_text(_header_line(version=99), encoding="utf-8")
    with pytest.raises(RewindLogError):
        read_rewind_log(path)


def test_read_rewind_log_accepts_a_str_path(tmp_path):
    log = _make_log(tmp_path)
    log.record_batch(step=1, micro=0, rows=[1], row_loss=[1.0], row_tokens=[4])
    log.close()
    assert read_rewind_log(str(log.path)).steps() == [1]


def test_dataset_fingerprint_key_order_independent_and_value_sensitive(tmp_path):
    rows_a = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    rows_b = [{"text": "hello", "id": 1}, {"text": "world", "id": 2}]
    rows_c = [{"id": 1, "text": "hello"}, {"id": 2, "text": "changed"}]

    fp_a = dataset_fingerprint(rows_a)
    fp_b = dataset_fingerprint(rows_b)
    fp_c = dataset_fingerprint(rows_c)

    assert fp_a == fp_b
    assert fp_a != fp_c
    assert isinstance(fp_a, str)
    assert len(fp_a) == 64  # sha256 hex digest length


class _CountingBatches(tuple):
    """A ``batches`` tuple that counts how many times it is iterated."""

    def __new__(cls, items):
        self = super().__new__(cls, items)
        self.iterations = 0
        return self

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


def test_reading_a_run_indexes_the_batches_once_not_once_per_step():
    """`find_spikes` asks for every step's batches, and `batches_for` used to
    scan the whole log each time: 0.08 s at 2,000 steps, 87 s at 50,000. The
    index is built once at construction, so the scan count must not grow with
    the number of steps."""
    from soup_cli.monitoring.rewind_log import BatchRecord, RewindRun
    from soup_cli.utils.rewind import find_spikes

    records = [
        BatchRecord(step=step, micro=0, rows=(step,), row_loss=(1.0,), row_tokens=(10,))
        for step in range(1, 201)
    ]
    batches = _CountingBatches(records)
    run = RewindRun(header={"kind": "header", "version": 1}, batches=batches)
    after_construction = batches.iterations

    find_spikes(run)
    for step in run.steps():
        run.batches_for(step)
        run.step_loss(step)

    assert after_construction == 1, "the index must be built in one pass"
    assert batches.iterations == 1, (
        f"the log was re-scanned {batches.iterations - 1} times after construction; "
        "reading is quadratic in the number of steps again"
    )

