"""#1186 item 3: the "unsloth is installed" tip is only printed for tasks that have
an unsloth setup.

On every other task ``backend: unsloth`` is refused at config load
(:data:`UNSLOTH_SETUP_TASKS`), so suggesting it there sends the user into a
refusal. The tip stays for the tasks that do read it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.config.schema import UNSLOTH_SETUP_TASKS
from tests.conftest import strip_ansi

_TIP = "unsloth is installed"


def _plain(text: "str | None") -> str:
    return " ".join(strip_ansi(text or "").split())


#: The minimum extra fields each task needs to load, so the run reaches the tip.
_TASK_EXTRAS = {
    "sft": "",
    "distill": "  teacher_model: meta-llama/Llama-3.1-70B\n",
    "classifier": "  num_labels: 2\n",
}


def _write_config(tmp_path: Path, task: str) -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n',
        encoding="utf-8",
    )
    path = tmp_path / "soup.yaml"
    path.write_text(
        f"base: meta-llama/Llama-3.1-8B\ntask: {task}\n"
        "backend: transformers\n"
        f"data:\n  train: {train.as_posix()}\n  format: chatml\n"
        "training:\n  quantization: none\n"
        f"{_TASK_EXTRAS[task]}"
        "  lora: {r: 8, alpha: 16}\n"
        f"output: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def _dry_run(tmp_path, monkeypatch, task: str) -> str:
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.utils.unsloth.is_unsloth_available", lambda: True)
    config = _write_config(tmp_path, task)
    result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])
    output = _plain(result.output)
    # The config must load, or the tip is absent for the wrong reason.
    assert "Training Setup" in output, output
    return output


def test_the_tip_is_shown_for_a_task_with_an_unsloth_setup(tmp_path, monkeypatch):
    assert "sft" in UNSLOTH_SETUP_TASKS
    output = _dry_run(tmp_path, monkeypatch, "sft")
    assert _TIP in output, output


@pytest.mark.parametrize("task", ["distill", "classifier"])
def test_the_tip_is_not_shown_for_a_task_without_one(tmp_path, monkeypatch, task):
    assert task not in UNSLOTH_SETUP_TASKS
    output = _dry_run(tmp_path, monkeypatch, task)
    assert _TIP not in output, output
