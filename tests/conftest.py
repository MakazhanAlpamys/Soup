"""Shared pytest fixtures and helpers."""

import functools
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

#: Rich/Pygments emit SGR escapes *between* the tokens of one logical line, so a
#: multi-token substring like "modality: text" is absent from raw output and
#: yaml.safe_load rejects \x1b outright (#633). 38 test files had grown their
#: own copy of this regex; new code should import this one.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: "str | None") -> str:
    """Return ``text`` with SGR escape sequences removed."""
    return _ANSI_ESCAPE.sub("", text or "")


#: Why a ``requires_symlink`` test was skipped. It names the account's missing capability, not
#: the platform: Windows creates symlinks fine with Developer Mode on or when elevated, which is
#: how the Windows CI cells run them (#832).
SYMLINK_SKIP_REASON = "this account cannot create symlinks (enable Developer Mode or run elevated)"


@functools.lru_cache(maxsize=None)
def can_symlink() -> bool:
    """Whether this process can create a symlink. Probed once per session."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "target")
        with open(target, "w", encoding="utf-8"):
            pass
        try:
            os.symlink(target, os.path.join(tmp, "link"))
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_symlink") is not None and not can_symlink():
        pytest.skip(SYMLINK_SKIP_REASON)


@pytest.fixture(autouse=True)
def _isolate_experiments_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the experiments DB at a per-test temp file.

    The MCP capacity gate now reads persisted runs from the tracker (issue
    #402), so a test must not see 'running' rows left in the real
    ``~/.soup/experiments.db`` by earlier tests or by the developer's own runs.
    A test that needs a specific DB overrides ``SOUP_DB_PATH`` itself; this only
    provides a clean, isolated default.
    """
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_alpaca_data(tmp_data_dir: Path) -> Path:
    """Create a sample alpaca-format JSONL file."""
    path = tmp_data_dir / "train.jsonl"
    samples = [
        {
            "instruction": "What is Python?",
            "input": "",
            "output": "Python is a programming language.",
        },
        {
            "instruction": "Explain gravity",
            "input": "",
            "output": "Gravity is a fundamental force.",
        },
        {
            "instruction": "Translate hello to Spanish",
            "input": "hello",
            "output": "hola",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return path


@pytest.fixture
def sample_config(tmp_path: Path, sample_alpaca_data: Path) -> Path:
    """Create a sample soup.yaml config."""
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        f"""base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: {sample_alpaca_data}
  format: alpaca
  val_split: 0.1
training:
  epochs: 1
  lr: 2e-5
  batch_size: 1
  lora:
    r: 8
    alpha: 16
  quantization: 4bit
output: {tmp_path / 'output'}
""",
        encoding="utf-8",
    )
    return config_path
