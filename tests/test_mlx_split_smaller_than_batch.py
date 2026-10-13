"""#1851 — a split smaller than one batch is refused before the model loads.

mlx-lm evaluates the validation split in batches of ``training.batch_size`` and
both its ``iterate_batches`` and Soup's ``masked_iterate_batches`` raise when a
dataset holds fewer rows than one batch. The bundled ``alpaca_tiny.jsonl`` (10
rows, default ``val_split: 0.1`` -> 1 held-out row) with ``batch_size: 2`` died
at the first evaluation, after the model had loaded and LoRA was attached:

    ValueError: Dataset must have at least batch_size=2 examples but only has 1.

The split is now checked in ``setup()``, before ``load_mlx_model``, and the
message names both numbers and the three ways out. The training split gets the
same check: it fails the same way, one step later.
"""

from __future__ import annotations

import pytest


def _wrapper(monkeypatch, *, batch_size):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer import mlx_sft

    def _must_not_load(*_args, **_kwargs):
        raise AssertionError("the model was loaded before the split was checked")

    monkeypatch.setattr("soup_cli.utils.mlx.load_mlx_model", _must_not_load)
    cfg = SoupConfig(
        base="mlx-community/tiny",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./d.jsonl", format="alpaca"),
        training=TrainingConfig(epochs=1, batch_size=batch_size),
        output="./out",
    )
    wrapper = mlx_sft.MLXSFTTrainerWrapper(cfg)
    monkeypatch.setattr(wrapper, "_require_mlx", lambda: None)
    return wrapper


def _rows(n):
    return [{"instruction": f"q{i}", "output": "a"} for i in range(n)]


class TestRefusedBeforeTheModelLoads:
    def test_the_issue_case_one_val_row_batch_two(self, monkeypatch):
        wrapper = _wrapper(monkeypatch, batch_size=2)
        with pytest.raises(ValueError) as excinfo:
            wrapper.setup({"train": _rows(9), "val": _rows(1)})
        message = str(excinfo.value)
        assert "validation split has 1 row" in message
        assert "training.batch_size (2)" in message
        # The three ways out, by the names the user writes in soup.yaml.
        assert "batch_size" in message and "val_split" in message
        assert "val_split: 0" in message

    def test_a_short_train_split_is_refused_too(self, monkeypatch):
        wrapper = _wrapper(monkeypatch, batch_size=8)
        with pytest.raises(ValueError, match="training split has 5 row"):
            wrapper.setup({"train": _rows(5), "val": []})


class TestControls:
    """The check must not refuse anything mlx-lm can run."""

    @pytest.mark.parametrize(
        "train, val, batch_size",
        [
            (9, 2, 2),   # exactly one validation batch
            (9, 0, 4),   # no validation split at all: nothing is evaluated
            (4, 1, 1),   # batch_size 1 fits any non-empty split
        ],
    )
    def test_reaches_the_model_load(self, monkeypatch, train, val, batch_size):
        wrapper = _wrapper(monkeypatch, batch_size=batch_size)
        with pytest.raises(AssertionError, match="model was loaded"):
            wrapper.setup({"train": _rows(train), "val": _rows(val)})

    def test_batch_size_auto_counts_as_one(self, monkeypatch):
        """MLX resolves ``batch_size: auto`` to 1 in ``train()``."""
        wrapper = _wrapper(monkeypatch, batch_size="auto")
        with pytest.raises(AssertionError, match="model was loaded"):
            wrapper.setup({"train": _rows(3), "val": _rows(1)})
