"""Keep the MLX support and refusal docs aligned with config validation (#2223)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import MLX_TASKS, SoupConfig

DOCS = Path(__file__).parents[1] / "docs"
NON_SFT_MLX_TASKS = MLX_TASKS - {"sft"}
EVAL_STEPS_REFUSAL_TASKS = MLX_TASKS - {"sft", "distill"}


def _mlx_section(markdown: str, task: str) -> str:
    headings = list(re.finditer(r"^#### (.+?) on MLX\s*$", markdown, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(1).lower() == task:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
            return markdown[heading.start():end]
    raise AssertionError(f"Missing MLX documentation section for {task!r}")


def _mlx_distill_section(training: str) -> str:
    section = re.search(
        r"^### Native MLX Token Distillation\s*$([\s\S]*?)(?=^### |\Z)",
        training,
        re.MULTILINE,
    )
    assert section is not None, "Missing native MLX distillation documentation section"
    return section.group(0)


@pytest.mark.parametrize("task", sorted(EVAL_STEPS_REFUSAL_TASKS))
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


def test_mlx_distill_honors_eval_steps() -> None:
    cfg = SoupConfig(
        base="test/model",
        task="distill",
        backend="mlx",
        data={"train": "train.jsonl", "format": "chatml"},
        training={"teacher_model": "test/teacher", "eval_steps": 1},
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
    # GRPO refuses eval_steps in its own section; distillation supports update-based checks.
    expected_preferences = MLX_TASKS - {"sft", "grpo", "distill"}
    assert documented_preferences == expected_preferences
    grpo_section = _mlx_section(backends, "grpo")
    assert re.search(r"MLX GRPO refuses\s+`training\.eval_steps`", grpo_section)
    distill_section = _mlx_distill_section(training)
    assert re.search(
        r"`eval_steps`\s+counts optimizer updates; validation also runs at the final update",
        distill_section,
    )
    assert documented_preferences | {"grpo", "distill"} == MLX_TASKS - {"sft"}


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
    training = (DOCS / "training.md").read_text(encoding="utf-8")
    for task in sorted(NON_SFT_MLX_TASKS):
        if task == "distill":
            section = _mlx_distill_section(training)
            assert re.search(r"DoRA.{0,100}refused", section, re.DOTALL | re.IGNORECASE)
            continue
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
    documented = set(re.findall(
        r"\bSFT\b|\bDPO\b|\bORPO\b|\bSimPO\b|\bKTO\b|\bGRPO\b|\bdistill\b",
        row,
    ))
    assert {task.lower() for task in documented} == MLX_TASKS
    assert "`[train]` is not required" in row
