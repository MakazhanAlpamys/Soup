"""#2556: embedding negatives are on every row (triplet) or on none (contrastive).

The Transformers embedding trainer read the choice from the first row only, so a
partial mix was silently trained contrastive or crashed in the collator.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.data.formats import split_embedding_negatives
from tests.conftest import strip_ansi
from tests.test_issue1234_embedding_contrastive_batch_size import _tiny_bert_dir


def _with(negative=None):
    row = {"anchor": "hello", "positive": "world"}
    if negative is not None:
        row["negative"] = negative
    return row


def _mix_message(split, have, lack):
    return (
        f"embedding {split} set: {have} of {have + lack} rows have a 'negative' and "
        f"{lack} do not; negatives must be on every row (triplet) or on none (contrastive)"
    )


@pytest.mark.parametrize(
    ("rows", "have", "lack"),
    [
        ([_with(), _with("python"), _with("python")], 2, 1),
        ([_with("python"), _with(), _with()], 1, 2),
        ([_with("python"), _with("python"), _with()], 2, 1),
        ([_with("python"), _with("   \t"), _with("python")], 2, 1),
        ([_with("python"), {**_with(), "negative": None}, _with("python")], 2, 1),
        ([_with("python"), _with(""), _with("python")], 2, 1),
    ],
    ids=("first-lacks", "first-has", "last-lacks", "whitespace-only", "none", "empty"),
)
def test_a_partial_mix_is_refused_naming_the_counts(rows, have, lack):
    with pytest.raises(ValueError) as excinfo:
        split_embedding_negatives(rows, "train")
    assert str(excinfo.value) == _mix_message("train", have, lack)


def test_the_message_names_the_split():
    with pytest.raises(ValueError, match=r"^embedding validation set: 1 of 2 rows"):
        split_embedding_negatives([_with("python"), _with()], "validation")


def test_every_row_with_a_negative_is_triplet():
    assert split_embedding_negatives([_with("python"), _with(" a ")], "train") is True


@pytest.mark.parametrize(
    "rows",
    [
        [_with(), _with()],
        [_with(""), {**_with(), "negative": None}, _with("  ")],
        [],
    ],
    ids=("missing", "blank-or-null", "empty-set"),
)
def test_no_row_with_a_negative_is_contrastive(rows):
    assert split_embedding_negatives(rows, "train") is False


def _wrapper(tmp_path, monkeypatch, loss="triplet"):
    from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

    model_path = _tiny_bert_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": model_path,
                "task": "embedding",
                "data": {"train": "train.jsonl", "max_length": 64},
                "training": {
                    "batch_size": 4,
                    "embedding_loss": loss,
                    "quantization": "none",
                    "lora": {"r": 8, "alpha": 16, "target_modules": ["query", "value"]},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    return EmbeddingTrainerWrapper(cfg, device="cpu")


def _no_model_load(monkeypatch):
    import transformers

    def must_not_run(*args, **kwargs):
        pytest.fail("a partial mix of negatives reached the model load")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", must_not_run)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", must_not_run)


@pytest.mark.parametrize("loss", ["triplet", "contrastive"])
def test_setup_refuses_a_partial_train_mix_before_the_model_loads(tmp_path, monkeypatch, loss):
    wrapper = _wrapper(tmp_path, monkeypatch, loss=loss)
    _no_model_load(monkeypatch)
    train = [_with(), _with("python"), _with("python"), _with("python")]
    with pytest.raises(ValueError) as excinfo:
        wrapper.setup({"train": train})
    assert str(excinfo.value) == _mix_message("train", 3, 1)


def test_setup_refuses_a_partial_validation_mix_before_the_model_loads(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch)
    _no_model_load(monkeypatch)
    train = [_with("python") for _ in range(4)]
    val = [_with("python"), _with()]
    with pytest.raises(ValueError) as excinfo:
        wrapper.setup({"train": train, "val": val})
    assert str(excinfo.value) == _mix_message("validation", 1, 1)


def test_setup_trains_triplet_when_every_row_has_a_negative(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch)
    wrapper.setup({"train": [_with("python") for _ in range(4)]})
    assert wrapper.trainer._loss_type == "triplet"
    batch = next(iter(wrapper.trainer._trainer.get_train_dataloader()))
    assert "negative_input_ids" in batch


def test_setup_keeps_the_contrastive_fallback_when_no_row_has_one(tmp_path, monkeypatch, capsys):
    wrapper = _wrapper(tmp_path, monkeypatch)
    wrapper.setup({"train": [_with(), _with(""), _with(), _with()]})
    assert wrapper.trainer._loss_type == "contrastive"
    out = " ".join(strip_ansi(capsys.readouterr().out).split())
    assert "triplet loss requires 'negative' field" in out
    assert "Falling back to contrastive loss" in out


def test_setup_allows_a_triplet_train_set_with_a_validation_set_without_negatives(
    tmp_path, monkeypatch
):
    wrapper = _wrapper(tmp_path, monkeypatch)
    wrapper.setup({"train": [_with("python") for _ in range(4)], "val": [_with(), _with()]})
    assert wrapper.trainer._loss_type == "triplet"
