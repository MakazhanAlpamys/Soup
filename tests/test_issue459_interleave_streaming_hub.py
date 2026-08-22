"""`data.interleave` extended to `data.streaming: true` and HF-hub dataset
name lists (issue #459 — the follow-up #443 explicitly deferred).

#443 wired `data.interleave` for a `data.train` list of local file paths
only; its own schema validator refused `data.streaming: true` and any
remote-URI / HF-hub-name list entry outright, naming this issue as the
follow-up. #459 lifts that refusal for two DECIDED (not emergent) new
shapes, each dispatched by classifying every `data.train` entry as
`local` / `remote` / `hub` (`loader._classify_train_entry`,
`schema._validate_interleave_compat`'s inline `_kind`):

- All entries `local` / `remote`, `data.streaming: true` — delegates to
  `datasets.interleave_datasets` / `concatenate_datasets`
  (`loader._load_interleaved_streaming_datasets`). The strategy-name
  mapping is documented in that function's docstring and in
  `docs/data.md`; the acceptance criterion is that the strategies MEAN THE
  SAME THING as the local path, verified below by running one `probs`
  config through both paths and comparing the resulting proportions —
  not two tests that each pass in isolation.
- All entries `hub` (HF-hub dataset names) — eager per-entry loading via
  the existing hub loader, reusing the SAME `_combine_interleaved` the
  local path uses (`loader._load_interleaved_hub_datasets`). The
  `val_split`-vs-hub-split precedence is decided here: a combined
  validation split is honoured only when EVERY entry provides one;
  otherwise it's ignored (warned) and `val_split` applies to the combined
  train rows.
- Anything still unsupported (a streaming hub-name list; a list mixing
  hub names with local/remote entries) keeps refusing, by name — covered
  by tests/test_issue443_interleave_wiring.py's updated refusal tests.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

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
# Fake `datasets` module — no network, no real HF dependency.
# ---------------------------------------------------------------------------


class _FakeIterableDataset:
    """Stand-in for HF's IterableDataset — just an iterable of dict rows."""

    def __init__(self, rows):
        self._rows = list(rows)

    def shuffle(self, buffer_size=None):
        return self  # order isn't load-bearing here — see loader's _cycle_to note

    def __iter__(self):
        return iter(self._rows)


def _rows_from_jsonl(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _round_robin(pools: list[list[dict]], stopping_strategy: str) -> list[dict]:
    """Realistic-enough fake for interleave_datasets(streams, stopping_strategy=...)
    with no probabilities: alternate one row per source per round.
    """
    if any(len(p) == 0 for p in pools):
        return []
    out: list[dict] = []
    if stopping_strategy == "first_exhausted":
        n = min(len(p) for p in pools)
    elif stopping_strategy == "all_exhausted":
        n = max(len(p) for p in pools)
    else:
        raise AssertionError(f"unexpected stopping_strategy {stopping_strategy!r}")
    for i in range(n):
        for pool in pools:
            out.append(pool[i % len(pool)])
    return out


def _weighted_sample(
    pools: list[list[dict]], probabilities: list[float], stopping_strategy: str
) -> list[dict]:
    """Realistic-enough fake for interleave_datasets(streams, probabilities=...):
    weighted-without-replacement draw per source, stopping per policy. With
    large-enough pools this converges to `probabilities` — that convergence
    is exactly what the acceptance test below checks for.
    """
    rng = random.Random(0)
    ptrs = [0] * len(pools)
    out: list[dict] = []
    max_draws = sum(len(p) for p in pools) * 10
    for _ in range(max_draws):
        i = rng.choices(range(len(pools)), weights=probabilities, k=1)[0]
        if ptrs[i] >= len(pools[i]):
            if stopping_strategy == "first_exhausted":
                break
            ptrs[i] = 0
        out.append(pools[i][ptrs[i]])
        ptrs[i] += 1
    return out


def _install_fake_streaming_datasets(monkeypatch, calls: list):
    """Installs a fake `datasets` module recording every
    interleave_datasets/concatenate_datasets call (for the call-mapping
    assertions) while still combining real rows (for the non-degenerate /
    proportions assertions)."""

    def fake_load_dataset(builder, data_files=None, split=None, streaming=False):
        assert builder == "json"
        assert split == "train"
        assert streaming is True
        return _FakeIterableDataset(_rows_from_jsonl(data_files))

    def fake_concatenate_datasets(streams):
        calls.append(("concatenate_datasets", {}))
        combined = []
        for s in streams:
            combined.extend(list(s))
        return _FakeIterableDataset(combined)

    def fake_interleave_datasets(
        streams, probabilities=None, stopping_strategy="first_exhausted"
    ):
        calls.append((
            "interleave_datasets",
            {"probabilities": probabilities, "stopping_strategy": stopping_strategy},
        ))
        pools = [list(s) for s in streams]
        if probabilities is not None:
            combined = _weighted_sample(pools, probabilities, stopping_strategy)
        else:
            combined = _round_robin(pools, stopping_strategy)
        return _FakeIterableDataset(combined)

    fake_module = type(sys)("datasets")
    fake_module.load_dataset = fake_load_dataset
    fake_module.concatenate_datasets = fake_concatenate_datasets
    fake_module.interleave_datasets = fake_interleave_datasets
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


# ---------------------------------------------------------------------------
# Schema-level: the four dispatch outcomes from _validate_interleave_compat
# ---------------------------------------------------------------------------


def test_streaming_local_list_validates():
    cfg = DataConfig(
        train=["a.jsonl", "b.jsonl"], interleave="concat", format="plaintext", streaming=True
    )
    assert cfg.streaming is True


def test_streaming_remote_uri_list_validates():
    cfg = DataConfig(
        train=["s3://bucket/a.jsonl", "s3://bucket/b.jsonl"],
        interleave="under",
        format="plaintext",
        streaming=True,
    )
    assert cfg.streaming is True


def test_remote_uri_without_streaming_rejected(tmp_path):
    with pytest.raises(ValidationError, match="require data.streaming=true"):
        _cfg(
            tmp_path,
            train=[str(tmp_path / "a.jsonl"), "s3://bucket/b.jsonl"],
            interleave="concat",
        )


def test_all_hub_list_without_streaming_validates(tmp_path):
    cfg = _cfg(
        tmp_path,
        train=["org/dataset-a", "org/dataset-b"],
        interleave="concat",
        format="plaintext",
    )
    assert cfg.data.train == ["org/dataset-a", "org/dataset-b"]
    assert cfg.data.interleave == "concat"
    assert cfg.data.streaming is False


def test_all_hub_list_streaming_still_rejected(tmp_path):
    with pytest.raises(ValidationError, match="does not support data.streaming=true"):
        _cfg(
            tmp_path,
            train=["org/dataset-a", "org/dataset-b"],
            interleave="concat",
            streaming=True,
        )


def test_mixed_kinds_rejected(tmp_path):
    with pytest.raises(ValidationError, match="not a mix"):
        _cfg(
            tmp_path,
            train=[str(tmp_path / "a.jsonl"), "org/dataset-a"],
            interleave="concat",
        )


# ---------------------------------------------------------------------------
# Loader-level, streaming: strategy -> HF call mapping + non-degenerate
# ---------------------------------------------------------------------------


def test_streaming_concat_calls_concatenate_datasets(tmp_path, monkeypatch):
    calls: list = []
    _install_fake_streaming_datasets(monkeypatch, calls)
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(4)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="concat",
        streaming=True,
    )
    result = load_dataset(cfg.data)
    assert calls == [("concatenate_datasets", {})]
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 7
    assert any(t.startswith("A-") for t in texts)
    assert any(t.startswith("B-") for t in texts)


def test_streaming_under_calls_interleave_first_exhausted(tmp_path, monkeypatch):
    calls: list = []
    _install_fake_streaming_datasets(monkeypatch, calls)
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="under",
        streaming=True,
    )
    result = load_dataset(cfg.data)
    assert calls == [
        ("interleave_datasets", {"probabilities": None, "stopping_strategy": "first_exhausted"})
    ]
    texts = [row["text"] for row in result["train"]]
    assert any(t.startswith("A-") for t in texts)
    assert any(t.startswith("B-") for t in texts)
    # bounded by the smaller source, same shape as the local "under" path
    assert len(texts) == 6


def test_streaming_over_calls_interleave_all_exhausted(tmp_path, monkeypatch):
    calls: list = []
    _install_fake_streaming_datasets(monkeypatch, calls)
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="over",
        streaming=True,
    )
    result = load_dataset(cfg.data)
    assert calls == [
        ("interleave_datasets", {"probabilities": None, "stopping_strategy": "all_exhausted"})
    ]
    texts = [row["text"] for row in result["train"]]
    # upsampled to match the larger source, same shape as the local "over" path
    assert len(texts) == 20
    assert sum(t.startswith("A-") for t in texts) == 10
    assert sum(t.startswith("B-") for t in texts) == 10


def test_streaming_probs_calls_interleave_with_probabilities(tmp_path, monkeypatch):
    calls: list = []
    _install_fake_streaming_datasets(monkeypatch, calls)
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(30)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(10)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave={"strategy": "probs", "probs": [0.75, 0.25]},
        streaming=True,
    )
    result = load_dataset(cfg.data)
    assert calls == [(
        "interleave_datasets",
        {"probabilities": [0.75, 0.25], "stopping_strategy": "first_exhausted"},
    )]
    texts = [row["text"] for row in result["train"]]
    assert any(t.startswith("A-") for t in texts)
    assert any(t.startswith("B-") for t in texts)


def test_streaming_local_and_remote_uris_can_mix_one_list(tmp_path, monkeypatch):
    # loader._classify_train_entry allows local + remote to share the
    # streaming dispatch — both go through the same "json" builder.
    calls: list = []
    _install_fake_streaming_datasets(monkeypatch, calls)
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(4)])

    def fake_load_dataset(builder, data_files=None, split=None, streaming=False):
        assert builder == "json"
        if str(data_files).startswith("s3://"):
            return _FakeIterableDataset([{"text": f"B-{i}"} for i in range(3)])
        return _FakeIterableDataset(_rows_from_jsonl(data_files))

    monkeypatch.setattr(sys.modules["datasets"], "load_dataset", fake_load_dataset)

    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), "s3://bucket/b.jsonl"],
        interleave="concat",
        streaming=True,
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 7
    assert any(t.startswith("A-") for t in texts)
    assert any(t.startswith("B-") for t in texts)


# ---------------------------------------------------------------------------
# THE core acceptance test: one config, run both ways, proportions compared.
# ---------------------------------------------------------------------------


def test_probs_proportions_match_between_local_and_streaming_paths(tmp_path, monkeypatch):
    """Issue #459's own acceptance wording: "a test that runs one config
    both ways and compares the resulting proportions, not two tests that
    each pass in isolation."
    """
    calls: list = []
    _install_fake_streaming_datasets(monkeypatch, calls)
    # Large-enough pools that the weighted-without-replacement fake
    # converges close to the requested ratio before either source runs out.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(400)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(400)])
    probs = [0.7, 0.3]

    local_cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave={"strategy": "probs", "probs": probs},
        streaming=False,
    )
    streaming_cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave={"strategy": "probs", "probs": probs},
        streaming=True,
    )

    local_result = load_dataset(local_cfg.data)
    streaming_result = load_dataset(streaming_cfg.data)

    local_texts = [row["text"] for row in local_result["train"]]
    streaming_texts = [row["text"] for row in streaming_result["train"]]

    local_a_ratio = sum(t.startswith("A-") for t in local_texts) / len(local_texts)
    streaming_a_ratio = sum(t.startswith("A-") for t in streaming_texts) / len(streaming_texts)

    assert local_a_ratio == pytest.approx(0.7, abs=1e-6)
    assert streaming_a_ratio == pytest.approx(0.7, abs=0.05)
    assert local_a_ratio == pytest.approx(streaming_a_ratio, abs=0.05)


# ---------------------------------------------------------------------------
# Loader-level, hub: combining + validation-split precedence
# ---------------------------------------------------------------------------


def _install_fake_hub_datasets(monkeypatch, registry: dict):
    def fake_load_dataset(name):
        return registry[name]

    fake_module = type(sys)("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_hub_list_concat_combines_all_entries(tmp_path, monkeypatch):
    registry = {
        "org/a": {"train": [{"text": f"A-{i}"} for i in range(4)]},
        "org/b": {"train": [{"text": f"B-{i}"} for i in range(3)]},
    }
    _install_fake_hub_datasets(monkeypatch, registry)
    cfg = _cfg(tmp_path, train=["org/a", "org/b"], interleave="concat")
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 7
    assert sum(t.startswith("A-") for t in texts) == 4
    assert sum(t.startswith("B-") for t in texts) == 3


def test_hub_list_probs_not_silently_ignored(tmp_path, monkeypatch):
    # The #443-lesson test, applied to the hub path: fails if the key is
    # ignored — before #459, this shape refused at parse time; if the
    # loader silently dropped interleave, this would collapse to one
    # dataset's rows instead of the apportioned mixture.
    registry = {
        "org/a": {"train": [{"text": f"A-{i}"} for i in range(30)]},
        "org/b": {"train": [{"text": f"B-{i}"} for i in range(10)]},
    }
    _install_fake_hub_datasets(monkeypatch, registry)
    cfg = _cfg(
        tmp_path,
        train=["org/a", "org/b"],
        interleave={"strategy": "probs", "probs": [0.75, 0.25]},
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 40
    assert sum(t.startswith("A-") for t in texts) == 30
    assert sum(t.startswith("B-") for t in texts) == 10


def test_hub_validation_split_combined_when_every_entry_has_one(tmp_path, monkeypatch):
    registry = {
        "org/a": {
            "train": [{"text": f"A-{i}"} for i in range(4)],
            "validation": [{"text": f"Aval-{i}"} for i in range(2)],
        },
        "org/b": {
            "train": [{"text": f"B-{i}"} for i in range(4)],
            "validation": [{"text": f"Bval-{i}"} for i in range(2)],
        },
    }
    _install_fake_hub_datasets(monkeypatch, registry)
    cfg = _cfg(tmp_path, train=["org/a", "org/b"], interleave="concat", val_split=0.5)
    result = load_dataset(cfg.data)
    val_texts = [row["text"] for row in result["val"]]
    assert len(val_texts) == 4
    assert sum(t.startswith("Aval-") for t in val_texts) == 2
    assert sum(t.startswith("Bval-") for t in val_texts) == 2
    # val_split (0.5) must NOT have been applied — hub splits won.
    assert len(result["train"]) == 8


def test_hub_validation_split_ignored_when_only_some_entries_have_one(tmp_path, monkeypatch):
    registry = {
        "org/a": {
            "train": [{"text": f"A-{i}"} for i in range(8)],
            "validation": [{"text": f"Aval-{i}"} for i in range(2)],
        },
        "org/b": {"train": [{"text": f"B-{i}"} for i in range(2)]},
    }
    _install_fake_hub_datasets(monkeypatch, registry)
    cfg = _cfg(tmp_path, train=["org/a", "org/b"], interleave="concat", val_split=0.2)
    result = load_dataset(cfg.data)
    val_texts = [row["text"] for row in result["val"]]
    # The hub "validation" split (Aval-*) must be ignored entirely — a
    # partial hub split is not a decided mixture — and val_split applied
    # to the combined 10 train rows instead.
    assert not any(t.startswith("Aval-") for t in val_texts)
    combined_len = len(result["train"]) + len(result["val"])
    assert combined_len == 10
    assert len(result["val"]) == int(10 * 0.2)


def test_hub_dataset_missing_train_split_still_raises(tmp_path, monkeypatch):
    registry = {
        "org/a": {"train": [{"text": "A-0"}]},
        "org/b": {"validation": [{"text": "Bval-0"}]},
    }
    _install_fake_hub_datasets(monkeypatch, registry)
    cfg = _cfg(tmp_path, train=["org/a", "org/b"], interleave="concat")
    with pytest.raises(ValueError, match="no 'train' split"):
        load_dataset(cfg.data)


def test_load_hf_dataset_single_name_unchanged_by_refactor(tmp_path, monkeypatch):
    # Golden test for the _load_one_hub_dataset factoring — the single-name
    # branch (load_dataset() on a bare string, not a list) must produce
    # byte-identical output to before #459's refactor.
    registry = {"org/solo": {"train": [{"text": "X-0"}, {"text": "X-1"}]}}
    _install_fake_hub_datasets(monkeypatch, registry)
    cfg = _cfg(tmp_path, train="org/solo")
    result = load_dataset(cfg.data)
    assert result == {"train": [{"text": "X-0"}, {"text": "X-1"}]}
