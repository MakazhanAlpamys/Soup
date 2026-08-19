"""`data.interleave` is schema-validated but never consumed at training time (issue #443).

`parse_interleave`/`InterleaveSpec` have been fully implemented and unit-tested since
v0.42.0, and the schema has validated `data.interleave`'s shape since the same release
— but `load_dataset()` never called `parse_interleave`, so every multi-dataset mixture
request silently trained on nothing but `data.train`'s single path (the same gap #330
and #442 papered over in their respective renderers, deliberately deferring the real
fix to this issue).

The maintainer's decision on #443 (Option A, scope fixed by him): `DataConfig.train`
now accepts `str | list[str]`; a list requires `data.interleave`, applies to local file
paths only, and is combined into one row set BEFORE the existing `val_split` line in
`_finalize` runs (so a single path stays byte-identical). `packing` / `multipack` +
`interleave`, and `data.streaming` / an HF-hub dataset name + `interleave`, are all
rejected at parse time with a message naming the reason — streaming/hub-dataset
interleaving is out of scope here, filed as follow-up #459.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import DataConfig, SoupConfig
from soup_cli.data.loader import load_dataset


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text(
        "\n".join(f'{{"text": {t!r}}}' for t in texts).replace("'", '"') + "\n",
        encoding="utf-8",
    )


def _cfg(tmp_path: Path, **data_overrides) -> SoupConfig:
    data = {
        "train": str(tmp_path / "a.jsonl"),
        "format": "plaintext",
        "val_split": 0.0,
    }
    data.update(data_overrides)
    return SoupConfig.model_validate(
        {
            "base": "test-base",
            "task": "sft",
            "data": data,
            "training": {"epochs": 1},
            "output": str(tmp_path / "out"),
        }
    )


# ---------------------------------------------------------------------------
# Loader-level: the actual acceptance criteria
# ---------------------------------------------------------------------------


def test_single_path_train_output_is_byte_identical_to_baseline(tmp_path):
    # Golden test pinning the single-path branch — must not change AT ALL
    # as a side effect of wiring interleave in. Any refactor that touches
    # this branch's output shape fails here.
    _write_jsonl(tmp_path / "a.jsonl", ["A-0", "A-1", "A-2"])
    cfg = _cfg(tmp_path)
    result = load_dataset(cfg.data)
    assert result == {
        "train": [
            {"text": "A-0"},
            {"text": "A-1"},
            {"text": "A-2"},
        ]
    }


def test_data_interleave_is_actually_consumed_not_just_parsed(tmp_path):
    # The acceptance criterion in the maintainer's own words: a test that
    # fails if data.interleave is ignored — asserting on the rows that
    # reach the trainer, not on the parsed config. Before #443,
    # load_dataset() never looked at data.interleave at all, so this would
    # have silently returned only dataset A's rows (or crashed on a list
    # train_path) — either way, this test fails on the pre-#443 code.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="under",
    )
    result = load_dataset(cfg.data)
    texts = {row["text"] for row in result["train"]}
    assert any(t.startswith("A-") for t in texts)
    assert any(t.startswith("B-") for t in texts)


def test_interleave_concat(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(4)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="concat",
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 7
    assert sum(t.startswith("A-") for t in texts) == 4
    assert sum(t.startswith("B-") for t in texts) == 3


def test_interleave_under_truncates_to_smallest(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="under",
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 6
    assert sum(t.startswith("A-") for t in texts) == 3
    assert sum(t.startswith("B-") for t in texts) == 3


def test_interleave_over_upsamples_to_largest(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="over",
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 20
    assert sum(t.startswith("A-") for t in texts) == 10
    b_texts = [t for t in texts if t.startswith("B-")]
    assert len(b_texts) == 10
    # B only has 3 unique rows — over-sampling must repeat, not invent rows.
    counts = Counter(b_texts)
    assert set(counts) == {"B-0", "B-1", "B-2"}
    assert sum(counts.values()) == 10


def test_interleave_probs_matches_requested_ratio(tmp_path):
    # 30 + 10 = 40 total, probs chosen to divide evenly so the expected
    # counts are exact (no apportionment-rounding ambiguity).
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(30)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(10)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave={"strategy": "probs", "probs": [0.75, 0.25]},
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 40
    assert sum(t.startswith("A-") for t in texts) == 30
    assert sum(t.startswith("B-") for t in texts) == 10


def test_val_split_applied_once_after_mixing(tmp_path):
    # Proves "one slice, the same line as today in _finalize": the split
    # fraction must be computed off the COMBINED length, not per-source.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(8)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(2)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="concat",
        val_split=0.2,
    )
    result = load_dataset(cfg.data)
    combined_len = len(result["train"]) + len(result["val"])
    assert combined_len == 10
    assert len(result["val"]) == int(10 * 0.2)


# ---------------------------------------------------------------------------
# Schema-level: back-compat pin + the four parse-time refusals
# ---------------------------------------------------------------------------


def test_interleave_train_single_string_still_accepted_by_schema():
    # Schema-level companion to the loader's byte-identical pin: a bare
    # string data.train must keep validating exactly as before #443.
    cfg = DataConfig(train="d.jsonl", format="auto")
    assert cfg.train == "d.jsonl"
    assert cfg.interleave is None


def test_train_list_of_one_entry_rejected():
    with pytest.raises(ValidationError, match=">= 2 entries"):
        DataConfig(train=["only-one.jsonl"], interleave="concat")


def test_train_list_without_interleave_rejected(tmp_path):
    with pytest.raises(ValidationError, match="requires data.interleave"):
        _cfg(tmp_path, train=["a.jsonl", "b.jsonl"])


def test_interleave_with_packing_rejected_with_reason(tmp_path):
    with pytest.raises(ValidationError, match="fixed blocks"):
        SoupConfig.model_validate(
            {
                "base": "test-base",
                "task": "sft",
                "data": {
                    "train": [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
                    "interleave": "concat",
                    "format": "plaintext",
                },
                "training": {"epochs": 1, "packing": True},
                "output": str(tmp_path / "out"),
            }
        )


def test_interleave_with_multipack_rejected_with_reason(tmp_path):
    with pytest.raises(ValidationError, match="mixture ratio"):
        SoupConfig.model_validate(
            {
                "base": "test-base",
                "task": "sft",
                "data": {
                    "train": [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
                    "interleave": "concat",
                    "format": "plaintext",
                },
                "training": {"epochs": 1, "multipack": True},
                "output": str(tmp_path / "out"),
            }
        )


def test_interleave_with_streaming_rejected_with_reason(tmp_path):
    with pytest.raises(ValidationError, match="#459"):
        _cfg(
            tmp_path,
            train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
            interleave="concat",
            streaming=True,
        )


def test_interleave_with_hub_dataset_name_rejected_with_reason(tmp_path):
    with pytest.raises(ValidationError, match="#459"):
        _cfg(
            tmp_path,
            train=[str(tmp_path / "a.jsonl"), "org/some-dataset"],
            interleave="concat",
        )


def test_interleave_rendered_overlay_yaml_round_trips_through_loader(tmp_path):
    # Belt-and-braces: a config assembled purely from YAML text (not
    # Python dict kwargs) round-trips through the schema and the loader.
    _write_jsonl(tmp_path / "a.jsonl", ["A-0", "A-1"])
    _write_jsonl(tmp_path / "b.jsonl", ["B-0", "B-1"])
    yaml_text = (
        "base: test-base\n"
        "task: sft\n"
        "data:\n"
        f"  train:\n    - {tmp_path / 'a.jsonl'}\n    - {tmp_path / 'b.jsonl'}\n"
        "  interleave: concat\n"
        "  format: plaintext\n"
        "  val_split: 0.0\n"
        "training:\n"
        "  epochs: 1\n"
        f"output: {tmp_path / 'out'}\n"
    )
    cfg = load_config_from_string(yaml_text)
    result = load_dataset(cfg.data)
    assert len(result["train"]) == 4
