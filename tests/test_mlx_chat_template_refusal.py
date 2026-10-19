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

# One registered name and one raw Jinja string, so a check that only matches
# a literal spelling cannot pass the suite.
TEMPLATES = ["chatml", "llama3", "{% for m in messages %}{{ m['content'] }}{% endfor %}"]


class TestMlxChatTemplateRefusal:
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_mlx_with_chat_template_refused_at_config_load(self, template: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            SoupConfig(
                base="test/model",
                task="sft",
                backend="mlx",
                data={"train": "d.jsonl", "chat_template": template},
                training={"epochs": 1},
            )
        message = str(exc_info.value)
        assert "data.chat_template is not applied by backend: mlx" in message
        assert "backend: transformers" in message

    def test_empty_chat_template_still_loads_on_mlx(self) -> None:
        # The field validator normalises "" to None, which is not an override.
        config = SoupConfig(
            base="test/model",
            task="sft",
            backend="mlx",
            data={"train": "d.jsonl", "chat_template": ""},
            training={"epochs": 1},
        )
        assert config.data.chat_template is None

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_transformers_backend_still_accepts_chat_template(self, template: str) -> None:
        # The override is real on the transformers backend; only mlx drops it.
        config = SoupConfig(
            base="test/model",
            task="sft",
            backend="transformers",
            data={"train": "d.jsonl", "chat_template": template},
            training={"epochs": 1},
        )
        assert config.data.chat_template == template

    def test_mlx_without_chat_template_still_allowed(self) -> None:
        config = SoupConfig(
            base="test/model",
            task="sft",
            backend="mlx",
            data={"train": "d.jsonl"},
            training={"epochs": 1},
        )
        assert config.data.chat_template is None
