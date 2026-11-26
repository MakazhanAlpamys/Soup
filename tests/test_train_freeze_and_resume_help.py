"""#3086 T-2 and T-3: two texts of ``soup train`` that lagged the code.

T-2: on ``backend: mlx`` the refusal of ``training.freeze_ratio`` /
``training.freeze_layers`` said layers are frozen only with ``backend:
transformers``, while native MLX text SFT with ``lora.r: 0`` (and ``quantization:
none``, which a full fine-tune requires) loads them and freezes. The refusal now
names that route wherever it exists; the other paths keep their text, which
``tests/test_issue1581_freeze_refused_off_the_text_path.py`` pins.

T-3: the help of ``--resume`` and ``--hf-resume`` named only
``training.relora_steps``. Both now name every run that is refused: ReLoRA, which
``train`` checks itself, and each branch of ``_resume_refusal``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import UNSUPPORTED_RESUME_TASKS, _resume_refusal
from tests.conftest import strip_ansi

_RESUME_OPTIONS = ("--resume", "--hf-resume")


def _option_block(option: str) -> str:
    """The help of one option in ``soup train --help``, whitespace collapsed.

    A row of the options table starts with ``│ -`` in the name column; the lines
    that wrap its help are indented there, so the block ends at the next row or
    at the bottom of the panel."""
    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0, result.output
    lines = strip_ansi(result.output).splitlines()
    start = next(
        index for index, line in enumerate(lines)
        if line.startswith(f"│ {option} ") or line.startswith(f"│ {option},")
    )
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.startswith("│ -") or not line.startswith("│"):
            break
        block.append(line)
    return " ".join(" ".join(line.strip("│ ") for line in block).split())


# ---------------------------------------------------------------------------
# T-2
# ---------------------------------------------------------------------------


def _mlx_sft(**training):
    return yaml.safe_dump({
        "base": "./local-llama", "backend": "mlx", "task": "sft",
        "data": {"train": "./train.jsonl"}, "output": "./out",
        "training": training,
    })


def test_soup_train_names_the_mlx_full_fine_tune_route(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        _mlx_sft(lora={"r": 16}, freeze_ratio=0.5), encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])

    flat = " ".join(strip_ansi(result.output).split())
    assert result.exit_code != 0, flat
    assert (
        "layers are frozen only on the text path with backend: transformers and no "
        "layer streaming, or in native MLX text SFT with lora.r: 0, so this run would "
        "train every layer. Use backend: transformers, or lora.r: 0 with quantization: "
        "none, or remove the key."
    ) in flat, flat


# ---------------------------------------------------------------------------
# T-3
# ---------------------------------------------------------------------------

_HAND_LISTED_REFUSALS = [
    # (task, backend, lora.r) that _resume_refusal refuses, and the words of the help.
    ("ppo", "transformers", 8, "task: ppo"),
    ("sft", "mlx", 0, "backend: mlx for a full fine-tune (lora.r: 0)"),
    ("grpo", "mlx", 8, "task: grpo"),
]


def _cfg(task, backend, rank):
    return SimpleNamespace(
        task=task, backend=backend, training=SimpleNamespace(lora=SimpleNamespace(r=rank)),
    )


@pytest.mark.parametrize("option", _RESUME_OPTIONS)
@pytest.mark.parametrize("words", [
    "training.relora_steps is configured",
    "task: unlearn",
    "task: ppo",
    "on backend: mlx for a full fine-tune (lora.r: 0) or task: grpo",
])
def test_each_resume_option_names_every_refused_run(option, words):
    block = _option_block(option)

    assert words in block, block


@pytest.mark.parametrize(("option", "kept"), [
    ("--resume", "Resume from checkpoint: path to checkpoint dir ('auto' for latest); "
                 "on the MLX backend, a path to a .safetensors adapter file instead."),
    ("--hf-resume", "Download the latest checkpoint branch from the --push-as repo and "
                    "resume from it. Requires --push-as;"),
])
def test_each_resume_option_keeps_its_other_sentences(option, kept):
    assert kept in _option_block(option)


@pytest.mark.parametrize("option", _RESUME_OPTIONS)
def test_the_help_names_every_task_that_cannot_resume(option):
    """A task added to the set fails here until both help texts name it."""
    block = _option_block(option)

    for task in sorted(UNSUPPORTED_RESUME_TASKS):
        assert _resume_refusal(_cfg(task, "transformers", 8), option) is not None
        assert f"task: {task}" in block, (task, block)


@pytest.mark.parametrize("option", _RESUME_OPTIONS)
@pytest.mark.parametrize(("task", "backend", "rank", "words"), _HAND_LISTED_REFUSALS)
def test_the_help_names_each_other_refusal(option, task, backend, rank, words):
    assert _resume_refusal(_cfg(task, backend, rank), option) is not None

    assert words in _option_block(option)


def test_a_run_the_help_leaves_out_can_resume():
    """The control: LoRA SFT on MLX is not in the list, and it is not refused."""
    assert _resume_refusal(_cfg("sft", "mlx", 8), "--resume") is None
    assert _resume_refusal(_cfg("sft", "transformers", 8), "--resume") is None
