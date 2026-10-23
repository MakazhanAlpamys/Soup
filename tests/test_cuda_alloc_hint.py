"""#2149: ``soup train`` sets the expandable_segments hint before anything starts CUDA.

``expandable_segments_status()`` only writes ``PYTORCH_CUDA_ALLOC_CONF`` while CUDA is
not initialised, and on the ``soup train`` path ``detect_device()`` initialised it long
before the streamed setup got there, so the hint never took on a CUDA run. The hint is
now set from ``train.py`` before ``detect_device()``, for streamed runs on Linux,
without overwriting a value the user exported. macOS and Windows are left alone.
"""

from __future__ import annotations

import os
import sys

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

HINT = "expandable_segments:True"


class _StopHereError(Exception):
    """Raised by the recording ``detect_device``: nothing after it matters here."""


@pytest.fixture(autouse=True)
def clean_alloc_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    # delenv on an absent key registers no undo, so set a sentinel first: teardown
    # then restores the unset state even after the code under test set it.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "sentinel_for_teardown")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")


def _value_at_detect_device(monkeypatch, tmp_path, *, stream_layers: bool) -> list:
    config = tmp_path / "soup.yaml"
    config.write_text(
        "base: owner/model\n"
        "task: sft\n"
        "data:\n"
        "  train: dummy.jsonl\n"
        "training:\n"
        "  quantization: none\n"
        "  batch_size: 1\n"
        f"  stream_layers: {'true' if stream_layers else 'false'}\n",
        encoding="utf-8",
    )
    (tmp_path / "dummy.jsonl").write_text('{"text": "hi"}\n', encoding="utf-8")
    seen: list = []

    def recording_detect_device(**kwargs):
        seen.append(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        raise _StopHereError

    monkeypatch.setattr("soup_cli.commands.train.detect_device", recording_detect_device)
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])
    assert isinstance(result.exception, _StopHereError), (result.output, repr(result.exception))
    return seen


def test_a_streamed_run_on_linux_has_the_hint_before_cuda_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")

    seen = _value_at_detect_device(monkeypatch, tmp_path, stream_layers=True)

    assert len(seen) == 1 and seen[0] is not None and HINT in seen[0]


def test_a_run_that_does_not_stream_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")

    assert _value_at_detect_device(monkeypatch, tmp_path, stream_layers=False) == [None]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_windows_and_macos_are_left_alone(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)

    assert _value_at_detect_device(monkeypatch, tmp_path, stream_layers=True) == [None]


def test_a_value_the_user_exported_is_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    seen = _value_at_detect_device(monkeypatch, tmp_path, stream_layers=True)

    assert seen == ["max_split_size_mb:128"]


def test_an_empty_export_counts_as_unset(monkeypatch, tmp_path):
    """``setdefault`` would keep ``""``: the hint stays off and the old reason comes back."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "")

    seen = _value_at_detect_device(monkeypatch, tmp_path, stream_layers=True)

    assert seen == [HINT]


def test_the_helper_does_not_import_torch(monkeypatch):
    from soup_cli.utils.gpu import configure_cuda_alloc_hint

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "torch", None)  # any `import torch` now raises

    configure_cuda_alloc_hint(stream_layers=True)

    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == HINT
