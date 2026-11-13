"""Keep the MLX support and refusal docs aligned with config validation (#2223)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import MLX_TASKS, SoupConfig

DOCS = Path(__file__).parents[1] / "docs"
NON_SFT_MLX_TASKS = MLX_TASKS - {"sft"}


def _mlx_section(markdown: str, task: str) -> str:
    headings = list(re.finditer(r"^#### (.+?) on MLX\s*$", markdown, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(1).lower() == task:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
            return markdown[heading.start():end]
    raise AssertionError(f"Missing MLX documentation section for {task!r}")


@pytest.mark.parametrize("task", sorted(NON_SFT_MLX_TASKS))
def test_mlx_eval_steps_is_refused_for_each_non_sft_task(task: str) -> None:
    with pytest.raises(
        ValidationError,
        match=r"training\.eval_steps is not (?:supported|applied)",
    ):
        SoupConfig(
            base="test/model",
            task=task,
            backend="mlx",
            data={"train": "train.jsonl", "format": "chatml"},
            training={"eval_steps": 1},
        )


def test_mlx_sft_honors_eval_steps() -> None:
    cfg = SoupConfig(
        base="test/model",
        task="sft",
        backend="mlx",
        data={"train": "train.jsonl", "format": "chatml"},
        training={"eval_steps": 1},
    )
    assert cfg.training.eval_steps == 1


def test_mlx_eval_steps_docs_cover_supported_tasks() -> None:
    training = (DOCS / "training.md").read_text(encoding="utf-8")
    backends = (DOCS / "backends-and-ops.md").read_text(encoding="utf-8")
    preference_passage = re.search(
        r"The MLX preference\s+tasks \(([^)]*)\).*?refuse\s+(?:this\s+)?field\.",
        training,
        re.DOTALL | re.IGNORECASE,
    )
    assert preference_passage is not None, "Missing MLX eval_steps preference-task passage"
    documented_preferences = set(re.findall(r"`([a-z_]+)`", preference_passage.group(1)))
    # GRPO also refuses eval_steps but is documented in its own section below; SFT honors it.
    expected_preferences = MLX_TASKS - {"sft", "grpo"}
    assert documented_preferences == expected_preferences
    grpo_section = _mlx_section(backends, "grpo")
    assert re.search(r"MLX GRPO refuses\s+`training\.eval_steps`", grpo_section)
    assert documented_preferences | {"grpo"} == MLX_TASKS - {"sft"}


@pytest.mark.parametrize("task", sorted(NON_SFT_MLX_TASKS))
def test_mlx_dora_is_refused_for_each_non_sft_task(task: str) -> None:
    with pytest.raises(ValidationError, match="MLX DoRA requires task='sft'"):
        SoupConfig(
            base="test/model",
            task=task,
            backend="mlx",
            data={"train": "train.jsonl", "format": "chatml"},
            training={"quantization": "none", "lora": {"use_dora": True}},
        )


def test_mlx_dora_refusal_is_listed_in_each_non_sft_task_section() -> None:
    backends = (DOCS / "backends-and-ops.md").read_text(encoding="utf-8")
    for task in sorted(NON_SFT_MLX_TASKS):
        section = _mlx_section(backends, task)
        refusal = re.search(
            r"^- \*\*Refused at load:\*\*(.*?)(?=\n- \*\*|\n#### |\Z)",
            section,
            re.MULTILINE | re.DOTALL,
        )
        assert refusal is not None, f"Missing refused-at-load list for MLX {task}"
        assert "DoRA" in refusal.group(1), f"DoRA is missing from the MLX {task} refusal list"


def test_mlx_extra_lists_the_tasks_from_config() -> None:
    models = (DOCS / "models.md").read_text(encoding="utf-8")
    row = next(line for line in models.splitlines() if line.startswith("| `mlx` |"))
    documented = set(re.findall(r"\bSFT\b|\bDPO\b|\bORPO\b|\bSimPO\b|\bKTO\b|\bGRPO\b", row))
    assert {task.lower() for task in documented} == MLX_TASKS
    assert "`[train]` is not required" in row
