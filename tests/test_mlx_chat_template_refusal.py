"""Regression tests for #1913 Finding A: `backend: mlx` silently ignored
`data.chat_template` — the MLX trainer renders the model's shipped template,
so a prefix-stable override was dropped and the run still failed the mask's
prefix-stability check (`ResponseMaskError`) deep inside training.

After the fix the combination is refused at config load with a message that
names the escape hatches (remove the override or use the transformers
backend).
"""

from __future__ import annotations

import pytest

from soup_cli.config.schema import SoupConfig

# Registered names and raw Jinja all take the same backend validation path.
TEMPLATES = ["chatml", "llama3", "{% for m in messages %}{{ m['content'] }}{% endfor %}"]


class TestMlxChatTemplateRefusal:
    @pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto"])
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_mlx_with_chat_template_refused_at_config_load(self, template: str, task: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            SoupConfig(
                base="test/model",
                task=task,
                backend="mlx",
                data={"train": "d.jsonl", "chat_template": template},
                training={"epochs": 1},
            )
        message = str(exc_info.value)
        assert "data.chat_template is not applied by backend: mlx" in message
        assert "backend: transformers" in message
        assert "Remove data.chat_template" in message

    @pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto"])
    def test_empty_chat_template_still_loads_on_mlx(self, task: str) -> None:
        # The field validator normalises "" to None, which is not an override.
        config = SoupConfig(
            base="test/model",
            task=task,
            backend="mlx",
            data={"train": "d.jsonl", "chat_template": ""},
            training={"epochs": 1},
        )
        assert config.data.chat_template is None

    @pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto"])
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_transformers_backend_still_accepts_chat_template(
        self, template: str, task: str,
    ) -> None:
        # The override is real on the transformers backend; only mlx drops it.
        config = SoupConfig(
            base="test/model",
            task=task,
            backend="transformers",
            data={"train": "d.jsonl", "chat_template": template},
            training={"epochs": 1},
        )
        assert config.data.chat_template == template

    @pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto"])
    def test_mlx_without_chat_template_still_allowed(self, task: str) -> None:
        config = SoupConfig(
            base="test/model",
            task=task,
            backend="mlx",
            data={"train": "d.jsonl"},
            training={"epochs": 1},
        )
        assert config.data.chat_template is None

    def test_unsloth_backend_still_accepts_chat_template(self) -> None:
        config = SoupConfig(
            base="test/model",
            task="sft",
            backend="unsloth",
            data={"train": "d.jsonl", "chat_template": "chatml"},
        )
        assert config.data.chat_template == "chatml"


class TestMlxGrpoAppliesChatTemplate:
    """#1976: MLX GRPO renders prompts with ``data.chat_template`` (and the
    transformers path's fallback), so the MLX refusal leaves it out."""

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_grpo_on_mlx_accepts_a_chat_template(self, template: str) -> None:
        config = SoupConfig(
            base="test/model",
            task="grpo",
            backend="mlx",
            data={"train": "d.jsonl", "chat_template": template},
            training={"epochs": 1, "quantization": "none"},
        )
        assert config.data.chat_template == template
