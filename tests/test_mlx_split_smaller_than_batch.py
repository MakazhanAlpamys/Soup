"""#1851 — a split smaller than one batch on ``backend: mlx``.

mlx-lm trains and evaluates in batches of ``training.batch_size``, and both its
``iterate_batches`` and Soup's ``masked_iterate_batches`` raise on a dataset
smaller than one batch. The bundled ``alpaca_tiny.jsonl`` (10 rows, default
``val_split: 0.1`` -> 1 held-out row) with ``batch_size: 2`` died at the first
evaluation, after the model had loaded:

    ValueError: Dataset must have at least batch_size=2 examples but only has 1.

* A **validation** split smaller than one batch is not evaluated: the run warns,
  naming both numbers and that ``val_loss`` will be absent, and trains as the
  no-validation branch does. The default ``val_split`` keeps working, and the
  rows are not withheld in silence (#1223).
* A **training** split smaller than one batch cannot train at all, so it is
  refused in ``setup()``, before the model loads.

Driven through the shared fake-MLX harness: ``train()`` runs its real body.
"""

from __future__ import annotations

import sys
from io import StringIO

import pytest

from tests.conftest import strip_ansi
from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx


def _cfg(batch_size):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

    return SoupConfig(
        base="mlx-community/tiny",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./d.jsonl", format="plaintext"),
        training=TrainingConfig(epochs=1, batch_size=batch_size),
        output="./out",
    )


def _rows(n):
    return [{"text": f"row {i}"} for i in range(n)]


def _quiet_console(monkeypatch):
    from rich.console import Console

    from soup_cli.trainer import mlx_sft

    buffer = StringIO()
    monkeypatch.setattr(mlx_sft, "console", Console(file=buffer, width=400))
    return buffer


def _train(tmp_path, monkeypatch, *, batch_size, train_rows, val_rows):
    """Run the real ``train()``; return (kwargs mlx-lm's train got, printed text)."""
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    _install_fake_mlx(monkeypatch)
    seen: dict = {}
    monkeypatch.setattr(
        sys.modules["mlx_lm.tuner.trainer"], "train", lambda **kw: seen.update(kw)
    )
    printed = _quiet_console(monkeypatch)
    monkeypatch.chdir(tmp_path)

    wrapper = MLXSFTTrainerWrapper(_cfg(batch_size))
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": _rows(train_rows), "val": _rows(val_rows)}
    wrapper.train()
    return seen, printed.getvalue()


class TestAShortValidationSplitIsSkippedWithAWarning:
    def test_the_issue_case_trains_without_evaluating(self, tmp_path, monkeypatch):
        seen, printed = _train(tmp_path, monkeypatch, batch_size=2, train_rows=9, val_rows=1)

        assert seen["val_dataset"] is None
        args = seen["args"]
        # The no-validation cadence: evaluation never comes due.
        assert args.steps_per_eval > args.iters

        # Rich styles the numbers on a colour terminal; compare the plain text.
        text = " ".join(strip_ansi(printed).split())
        assert "validation split has 1 row" in text
        assert "training.batch_size (2)" in text
        assert "val_loss" in text

    def test_control_a_full_validation_batch_is_still_evaluated(self, tmp_path, monkeypatch):
        seen, printed = _train(tmp_path, monkeypatch, batch_size=2, train_rows=9, val_rows=2)

        assert seen["val_dataset"] is not None
        assert seen["args"].steps_per_eval <= seen["args"].iters
        assert "validation split has" not in printed

    def test_control_no_validation_split_says_nothing(self, tmp_path, monkeypatch):
        seen, printed = _train(tmp_path, monkeypatch, batch_size=4, train_rows=9, val_rows=0)

        assert seen["val_dataset"] is None
        assert "validation split has" not in printed

    def test_batch_size_auto_counts_as_one(self, tmp_path, monkeypatch):
        """MLX resolves ``batch_size: auto`` to 1, so one row is a full batch."""
        seen, _ = _train(tmp_path, monkeypatch, batch_size="auto", train_rows=3, val_rows=1)

        assert seen["val_dataset"] is not None


class TestAShortTrainingSplitIsRefusedBeforeTheModelLoads:
    def _setup(self, monkeypatch, *, batch_size, train_rows):
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        _install_fake_mlx(monkeypatch)
        loads: list = []

        def _load(path, *args, **kwargs):
            loads.append(path)
            return _FakeMlxModel(), object()

        monkeypatch.setattr(sys.modules["mlx_lm"], "load", _load, raising=False)
        _quiet_console(monkeypatch)
        wrapper = MLXSFTTrainerWrapper(_cfg(batch_size))
        return wrapper, loads

    def test_refused_naming_batch_size(self, monkeypatch):
        wrapper, loads = self._setup(monkeypatch, batch_size=8, train_rows=5)
        with pytest.raises(ValueError, match=r"training split has 5 row.*batch_size \(8\)"):
            wrapper.setup({"train": _rows(5), "val": []})
        assert loads == [], "the model loaded before the split was checked"

    def test_control_a_full_batch_loads(self, monkeypatch):
        wrapper, loads = self._setup(monkeypatch, batch_size=4, train_rows=4)
        wrapper.setup({"train": _rows(4), "val": _rows(1)})
        assert loads == ["mlx-community/tiny"]
