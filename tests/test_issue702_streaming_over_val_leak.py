"""`streaming: true` + `over` + `val_split` must not put a row on both sides (#702).

`interleave_datasets(..., stopping_strategy="all_exhausted")` recycles the
shorter stream to exhaust the longer one, so the materialised result contains
duplicates of the smaller source. The positional tail slice `_finalize` takes
for `val_split` treats a row and its recycled copy as separate entries, so it
could put one on each side. Validation loss on the recycled source then
measures memorisation and the run reports a suspiciously good number rather
than an error, which is why this is worth a regression test rather than a note.

Structurally identical to #680, fixed on the eager paths by #701. That fix
carves val out per source *before* padding, which does not transfer here: a
stream is not countable ahead of time. This path splits over *distinct* rows
after materialisation instead, then withholds every copy of a chosen val row
from train.

The control cases matter as much as the failing one: `probs`, `concat` and
`under` never duplicate a row on this path, and must keep the ordinary split.
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import pytest
from rich.console import Console

import soup_cli.data.loader as loader
from soup_cli.config.schema import SoupConfig
from soup_cli.data.loader import _row_key, _split_val, _split_val_deduplicated, load_dataset
from tests.test_issue459_interleave_streaming_hub import (
    _install_fake_streaming_datasets,
    _write_jsonl,
)


def _cfg(tmp_path: Path, *, strategy: str, val_split: float) -> SoupConfig:
    return SoupConfig.model_validate(
        {
            "base": "test-base",
            "task": "sft",
            "data": {
                "train": [str(tmp_path / "big.jsonl"), str(tmp_path / "small.jsonl")],
                "format": "plaintext",
                "streaming": True,
                "interleave": strategy,
                "val_split": val_split,
            },
            "training": {"epochs": 1},
            "output": str(tmp_path / "out"),
        }
    )


def _sources(tmp_path: Path, big: int = 100, small: int = 10) -> None:
    """The asymmetry from the issue: 100 rows against 10, so `over` recycles."""
    _write_jsonl(tmp_path / "big.jsonl", [f"a{i}" for i in range(big)])
    _write_jsonl(tmp_path / "small.jsonl", [f"b{i}" for i in range(small)])


def _overlap(result: dict) -> int:
    train_keys = {_row_key(row) for row in result["train"]}
    return sum(1 for row in result.get("val", []) if _row_key(row) in train_keys)


# ---------------------------------------------------------------------------
# The unit that does the work
# ---------------------------------------------------------------------------


def _recycled_interleave(big: int = 100, small: int = 10) -> list[dict]:
    rows_a = [{"text": f"a{i}"} for i in range(big)]
    rows_b = [{"text": f"b{i}"} for i in range(small)]
    out: list[dict] = []
    for i in range(big):
        out.append(rows_a[i])
        out.append(rows_b[i % small])
    return out


def test_the_positional_split_really_does_leak() -> None:
    """Pin the defect, so the fix below is measured against something real.

    Without this, `_split_val_deduplicated` returning disjoint sets would be
    unremarkable -- it would not show that the old behaviour was not.
    """
    rows = _recycled_interleave()
    train, val = _split_val(rows, 0.1)
    train_keys = {_row_key(row) for row in train}
    leaked = sum(1 for row in val if _row_key(row) in train_keys)

    assert leaked > 0, "expected the positional slice to leak recycled rows"


def test_deduplicated_split_is_disjoint_and_val_is_not_empty() -> None:
    rows = _recycled_interleave()
    train, val = _split_val_deduplicated(rows, 0.1)

    train_keys = {_row_key(row) for row in train}
    val_keys = {_row_key(row) for row in val}

    # Disjointness alone is trivially true when val is empty, which #701's own
    # tests were caught by, so both are asserted together.
    assert val, "val must not be empty"
    assert train, "train must not be empty"
    assert not (train_keys & val_keys)


def _multiset(rows: list[dict]) -> Counter:
    return Counter(_row_key(row) for row in rows)


def test_no_row_is_lost_by_the_deduplicated_split() -> None:
    """Every input row comes out on exactly one side, with its multiplicity.

    Stated about the rows, not re-derived from the implementation's filter:
    the output multiset must equal the input multiset.
    """
    rows = _recycled_interleave()
    train, val = _split_val_deduplicated(rows, 0.1)

    assert val
    assert _multiset(train) + _multiset(val) == _multiset(rows)


def test_duplicates_of_a_train_row_are_kept() -> None:
    """The oversampling of train rows survives the split."""
    rows = _recycled_interleave()
    train, _ = _split_val_deduplicated(rows, 0.1)

    assert len(train) > len({_row_key(row) for row in train}), (
        "train should still contain recycled duplicates of non-val rows"
    )


def test_genuinely_repeated_source_rows_are_not_lost() -> None:
    """Two distinct source rows that render to the same text (#729 review).

    Content keying cannot tell them from a recycled copy. The split used to
    put one in val and filter the other out of train, so 20 rows went in and
    19 came out. Val is now drawn from content that occurs once.
    """
    rows = [{"text": f"r{i}"} for i in range(18)]
    rows += [{"text": "DUPLICATE_CONTENT"}, {"text": "DUPLICATE_CONTENT"}]

    train, val = _split_val_deduplicated(rows, 0.3)

    assert val
    assert not (set(_multiset(train)) & set(_multiset(val)))
    assert _multiset(train) + _multiset(val) == _multiset(rows)


def test_too_little_unique_content_withholds_copies_loudly(monkeypatch) -> None:
    """When val can only be filled from repeated content, say how much train lost."""
    out = io.StringIO()
    monkeypatch.setattr(loader, "console", Console(file=out, width=400))
    rows = [{"text": f"r{i}"} for i in range(10) for _ in range(3)]

    train, val = _split_val_deduplicated(rows, 0.2)

    assert len(val) == 2
    assert not (set(_multiset(train)) & set(_multiset(val)))
    assert len(train) + len(val) + 4 == len(rows)
    assert "withheld 4 duplicate row(s) from train" in out.getvalue()


def test_the_recycled_case_withholds_nothing(monkeypatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(loader, "console", Console(file=out, width=400))

    _split_val_deduplicated(_recycled_interleave(), 0.1)

    assert "withheld" not in out.getvalue()


@pytest.mark.parametrize("val_split", [0.1, 0.5])
def test_a_single_distinct_row_refuses_rather_than_training_on_nothing(val_split) -> None:
    """Mirror of `_split_val_per_source`'s refusal on the eager path."""
    rows = [{"text": "same"}] * 20

    with pytest.raises(ValueError, match="leaves 0 training rows under streaming 'over'"):
        _split_val_deduplicated(rows, val_split)


def test_a_split_with_no_duplicates_matches_the_ordinary_split() -> None:
    """On input without duplicates the two functions must agree."""
    rows = [{"text": f"r{i}"} for i in range(50)]
    assert _split_val_deduplicated(rows, 0.2) == _split_val(rows, 0.2)


def test_row_key_ignores_key_order() -> None:
    assert _row_key({"a": 1, "b": 2}) == _row_key({"b": 2, "a": 1})


def test_row_key_serialises_an_unknown_type_through_repr() -> None:
    assert _row_key({"x": object()})


def test_row_key_falls_back_when_json_refuses_the_row() -> None:
    """A leak check must not become a crash: a circular value makes json raise."""
    looped: list = []
    looped.append(looped)

    assert _row_key({"x": looped}) == _row_key({"x": looped})


# ---------------------------------------------------------------------------
# End to end, through load_dataset
# ---------------------------------------------------------------------------


def test_streaming_over_with_val_split_has_no_overlap(tmp_path, monkeypatch) -> None:
    _sources(tmp_path)
    _install_fake_streaming_datasets(monkeypatch, [])

    result = load_dataset(_cfg(tmp_path, strategy="over", val_split=0.1).data)

    assert _overlap(result) == 0
    assert result["val"], "val must not be empty"
    assert result["train"], "train must not be empty"


def test_streaming_over_on_a_single_distinct_row_refuses(tmp_path, monkeypatch) -> None:
    _write_jsonl(tmp_path / "big.jsonl", ["same"] * 20)
    _write_jsonl(tmp_path / "small.jsonl", ["same"] * 2)
    _install_fake_streaming_datasets(monkeypatch, [])

    with pytest.raises(ValueError, match="leaves 0 training rows under streaming 'over'"):
        load_dataset(_cfg(tmp_path, strategy="over", val_split=0.5).data)


@pytest.mark.parametrize("strategy", ["concat", "under", "probs"])
def test_the_other_strategies_are_unchanged(tmp_path, monkeypatch, strategy) -> None:
    """Control: only `over` duplicates rows on this path.

    These three go through the ordinary `_finalize` split, and this asserts
    the fix did not quietly reroute them: the deduplicating split must never
    be called, and val must be non-empty so disjointness is not vacuous.
    """
    _sources(tmp_path)
    _install_fake_streaming_datasets(monkeypatch, [])

    def rerouted(*args, **kwargs):
        raise AssertionError(f"{strategy} was rerouted through the #702 split")

    monkeypatch.setattr(loader, "_split_val_deduplicated", rerouted)

    config = SoupConfig.model_validate(
        {
            "base": "test-base",
            "task": "sft",
            "data": {
                "train": [str(tmp_path / "big.jsonl"), str(tmp_path / "small.jsonl")],
                "format": "plaintext",
                "streaming": True,
                "interleave": (
                    {"strategy": "probs", "probs": [0.5, 0.5]}
                    if strategy == "probs"
                    else strategy
                ),
                "val_split": 0.1,
            },
            "training": {"epochs": 1},
            "output": str(tmp_path / "out"),
        }
    )
    result = load_dataset(config.data)

    assert _overlap(result) == 0
    assert result["train"]
    assert result["val"]


def test_streaming_over_without_val_split_is_untouched(tmp_path, monkeypatch) -> None:
    """`val_split: 0` has no split to corrupt, so the path must not change."""
    _sources(tmp_path)
    _install_fake_streaming_datasets(monkeypatch, [])

    result = load_dataset(_cfg(tmp_path, strategy="over", val_split=0.0).data)

    assert "val" not in result
    # Oversampling still happened: more rows out than the two sources hold.
    assert len(result["train"]) > 110
