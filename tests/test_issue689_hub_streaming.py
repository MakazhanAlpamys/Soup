"""#689 - data.streaming is ignored for HuggingFace Hub dataset names.

The schema advertises a pass-through to HF `streaming=True`. `_load_one_hub_dataset`
calls `hf_load(name)` with no kwargs and list-comprehends the split. Remote and
local-interleave streaming already forward the flag (`loader.py:547` / `:848`).

These tests spy the kwargs `datasets.load_dataset` actually receives. A row-count
check would stay green if the flag were forwarded and then the source materialised
anyway. All-hub `data.interleave` + streaming stays schema-rejected (#459).
"""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig
from soup_cli.data.loader import load_dataset


def _cfg(tmp_path, **data_overrides):
    data = {
        "train": "org/solo",
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


class _FakeStream:
    def __init__(self, rows):
        self._rows = list(rows)
        self.shuffle_calls = []

    def shuffle(self, buffer_size=None):
        self.shuffle_calls.append(buffer_size)
        return self

    def __iter__(self):
        return iter(self._rows)


def _install_spy(monkeypatch, registry):
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append({"name": name, "kwargs": dict(kwargs)})
        spec = registry[name]
        if kwargs.get("streaming"):
            out = {}
            for split, rows in spec.items():
                out[split] = _FakeStream(rows)
            return out
        return spec

    fake_module = type(sys)("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return calls


def test_hub_streaming_forwards_streaming_kwarg(tmp_path, monkeypatch):
    registry = {"org/solo": {"train": [{"text": "a"}, {"text": "b"}]}}
    calls = _install_spy(monkeypatch, registry)
    load_dataset(_cfg(tmp_path, streaming=True).data)
    assert calls, "load_dataset was not called"
    assert calls[0]["name"] == "org/solo"
    assert calls[0]["kwargs"].get("streaming") is True


def test_hub_streaming_false_does_not_pass_streaming(tmp_path, monkeypatch):
    registry = {"org/solo": {"train": [{"text": "a"}]}}
    calls = _install_spy(monkeypatch, registry)
    load_dataset(_cfg(tmp_path, streaming=False).data)
    assert calls[0]["name"] == "org/solo"
    assert "streaming" not in calls[0]["kwargs"]


def test_hub_streaming_shuffles_train_with_buffer_size(tmp_path, monkeypatch):
    train = _FakeStream([{"text": "a"}, {"text": "b"}])
    val = _FakeStream([{"text": "v"}])
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append({"name": name, "kwargs": dict(kwargs)})
        return {"train": train, "validation": val}

    fake_module = type(sys)("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    result = load_dataset(
        _cfg(tmp_path, streaming=True, buffer_size=1000).data
    )
    assert calls[0]["kwargs"].get("streaming") is True
    assert train.shuffle_calls == [1000]
    assert val.shuffle_calls == []
    assert [row["text"] for row in result["train"]] == ["a", "b"]
    assert [row["text"] for row in result["val"]] == ["v"]


def test_hub_streaming_caps_at_max_remote_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("soup_cli.data.loader.MAX_REMOTE_ROWS", 3)
    registry = {"org/solo": {"train": [{"text": f"r{i}"} for i in range(10)]}}
    _install_spy(monkeypatch, registry)
    result = load_dataset(_cfg(tmp_path, streaming=True).data)
    assert [row["text"] for row in result["train"]] == ["r0", "r1", "r2"]


def test_all_hub_list_streaming_still_rejected(tmp_path):
    with pytest.raises(ValidationError, match="does not support data.streaming=true"):
        _cfg(
            tmp_path,
            train=["org/dataset-a", "org/dataset-b"],
            interleave="concat",
            streaming=True,
        )

