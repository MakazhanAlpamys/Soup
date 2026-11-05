"""v0.77.1 — tests for embed device resolution (#2279).

Verifies that ``resolve_device("auto")`` treats "auto" identically to
``resolve_device(None)`` across CUDA available and unavailable environments,
rejects empty or invalid inputs, preserves explicit devices verbatim, and allows
both ``embed_texts`` and default-device CLI commands (``soup data topics`` and
``soup data dedup --semantic``) to execute without PyTorch ``to("auto")``
``RuntimeError``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import transformers
from transformers.tokenization_utils_base import BatchEncoding
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import embed, live_eval


class DummyModel(torch.nn.Module):
    """Minimal nn.Module stub that records device arguments passed to .to()."""

    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.recorded_devices: list[Any] = []

    def to(self, *args: Any, **kwargs: Any) -> DummyModel:
        if args:
            self.recorded_devices.append(args[0])
        elif "device" in kwargs:
            self.recorded_devices.append(kwargs["device"])
        return super().to(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 4
        dev = self.param.device
        base = torch.arange(batch_size, device=dev, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        last_hidden = torch.ones(
            (batch_size, seq_len, self.hidden_dim),
            dtype=torch.float32,
            device=dev,
        ) + base
        return types.SimpleNamespace(last_hidden_state=last_hidden)


class DummyTokenizer:
    """Minimal tokenizer stub returning valid BatchEncoding tensors."""

    def __call__(self, texts: Any, **kwargs: Any) -> BatchEncoding:
        if isinstance(texts, (str, bytes)):
            texts = [texts]
        batch_size = len(texts)
        seq_len = 4
        return BatchEncoding({
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        })


def test_resolve_device_auto_matches_none_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    expected = live_eval.resolve_device(None)
    assert expected == "cuda"
    assert live_eval.resolve_device("auto") == expected
    assert live_eval.resolve_device("  auto  ") == expected
    assert live_eval.resolve_device("AUTO") == expected
    assert live_eval.resolve_device("Auto") == expected


def test_resolve_device_auto_matches_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    expected = live_eval.resolve_device(None)
    assert expected == "cpu"
    assert live_eval.resolve_device("auto") == expected
    assert live_eval.resolve_device("  auto  ") == expected
    assert live_eval.resolve_device("AUTO") == expected
    assert live_eval.resolve_device("Auto") == expected


def test_resolve_device_torch_import_error_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("simulated torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    assert live_eval.resolve_device(None) == "cpu"
    assert live_eval.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_preserved() -> None:
    for explicit in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
        assert live_eval.resolve_device(explicit) == explicit


@pytest.mark.parametrize("invalid", ["", "   ", "\t", "\n", 123, True, False, [], {}])
def test_resolve_device_rejects_invalid_inputs(invalid: Any) -> None:
    with pytest.raises(ValueError, match="device must be a non-empty string or None"):
        live_eval.resolve_device(invalid)


def test_embed_texts_default_device_resolves_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: dummy_model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: dummy_tokenizer
    )

    vecs = embed.embed_texts(["hello world", "soup post training"])
    expected_dev = live_eval.resolve_device(None)

    assert dummy_model.recorded_devices == [expected_dev]
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, dummy_model.hidden_dim)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_embed_texts_explicit_device(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: dummy_model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: dummy_tokenizer
    )

    vecs = embed.embed_texts(["hello world"], device="cpu")
    assert dummy_model.recorded_devices == ["cpu"]
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (1, dummy_model.hidden_dim)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_data_topics_default_device_cli_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: dummy_model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: dummy_tokenizer
    )

    data_file = tmp_path / "dataset.jsonl"
    data_file.write_text(
        json.dumps({"text": "first prompt"}) + "\n"
        + json.dumps({"text": "second prompt"}) + "\n"
        + json.dumps({"text": "third prompt"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["data", "topics", str(data_file)])
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert dummy_model.recorded_devices == [live_eval.resolve_device(None)]


def test_data_dedup_semantic_default_device_cli_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: dummy_model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: dummy_tokenizer
    )

    data_file = tmp_path / "dataset.jsonl"
    data_file.write_text(
        json.dumps({"text": "unique sentence one"}) + "\n"
        + json.dumps({"text": "unique sentence two"}) + "\n"
        + json.dumps({"text": "unique sentence three"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["data", "dedup", str(data_file), "--semantic"])
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert dummy_model.recorded_devices == [live_eval.resolve_device(None)]
