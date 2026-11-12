"""``training.vllm_sleep_mode: true`` is refused at load: GRPO has no vLLM rollouts.

The GRPO wrapper copied the flag into TRL's ``GRPOConfig.vllm_enable_sleep_mode``, but TRL
reads that only under ``use_vllm`` and Soup never sets ``use_vllm``: GRPO always generates
with the model's own ``generate``, so there is no vLLM engine to put to sleep and the key
did nothing. The field stays (its default ``false`` is accepted) so the message can name it.
The two gates that already refused the key on other tasks and on backend ``mlx`` come first
and keep their wording.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi

NO_EFFECT = (
    "training.vllm_sleep_mode has no effect: GRPO does not use a vLLM server for rollouts "
    "in this version, so there is no vLLM engine to put to sleep between rollouts. "
    "Refused as of v0.78.0. Delete the line."
)


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _yaml(task: str = "grpo", backend: str = "transformers", sleep_line: str = "") -> str:
    training = "  epochs: 1\n  lr: 1e-4\n"
    if task == "grpo":
        training += "  reward_fn: accuracy\n"
    if task == "dpo":
        data_format = "dpo"
    else:
        data_format = "chatml"
    return (
        "base: test-llama\n"
        f"task: {task}\n"
        f"backend: {backend}\n"
        f"data:\n  train: ./data.jsonl\n  format: {data_format}\n"
        "output: ./out\n"
        f"training:\n{training}{sleep_line}"
    )


class TestDefaultIsAccepted:
    def test_the_field_is_kept_and_defaults_to_false(self):
        assert "vllm_sleep_mode" in TrainingConfig.model_fields
        assert TrainingConfig().vllm_sleep_mode is False

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    def test_grpo_without_the_key_loads(self, backend):
        cfg = load_config_from_string(_yaml(backend=backend))
        assert cfg.training.vllm_sleep_mode is False

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    @pytest.mark.parametrize("value", ["false", "False", "no", "off"])
    def test_grpo_with_the_default_spelled_out_loads(self, backend, value):
        cfg = load_config_from_string(
            _yaml(backend=backend, sleep_line=f"  vllm_sleep_mode: {value}\n")
        )
        assert cfg.training.vllm_sleep_mode is False

    def test_the_training_block_alone_still_accepts_true(self):
        """The refusal lives in the SoupConfig cross-validator (it needs the task and
        the backend), so a bare TrainingConfig is unchanged."""
        assert TrainingConfig(vllm_sleep_mode=True).vllm_sleep_mode is True


class TestGrpoRefusesTheKey:
    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    @pytest.mark.parametrize("value", ["true", "True", "yes", "on", "1"])
    def test_every_non_default_spelling_is_refused_with_the_exact_message(
        self, backend, value,
    ):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(
                _yaml(backend=backend, sleep_line=f"  vllm_sleep_mode: {value}\n")
            )
        assert NO_EFFECT in str(excinfo.value)

    def test_the_schema_refuses_it_without_the_loader(self):
        with pytest.raises(ValueError) as excinfo:
            SoupConfig(
                base="test-llama",
                task="grpo",
                data={"train": "./data.jsonl"},
                training={"vllm_sleep_mode": True},
            )
        assert NO_EFFECT in str(excinfo.value)

    def test_the_message_names_the_key_and_says_why(self):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(_yaml(sleep_line="  vllm_sleep_mode: true\n"))
        message = str(excinfo.value)
        assert "training.vllm_sleep_mode" in message
        assert "GRPO does not use a vLLM server for rollouts in this version" in message
        assert "has no effect" in message
        assert "Refused as of v0.78.0. Delete the line." in message


class TestOtherGatesKeepTheirWording:
    """Behaviour of every other task and of backend mlx is what it was before."""

    @pytest.mark.parametrize("task", ["sft", "dpo"])
    def test_other_tasks_keep_the_task_gate(self, task):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(_yaml(task=task, sleep_line="  vllm_sleep_mode: true\n"))
        assert f"vllm_sleep_mode requires task='grpo'; got task='{task}'" in str(excinfo.value)
        assert NO_EFFECT not in str(excinfo.value)

    def test_ppo_keeps_the_task_gate(self):
        with pytest.raises(ValueError) as excinfo:
            SoupConfig(
                base="test-llama",
                task="ppo",
                data={"train": "./data.jsonl"},
                training={"vllm_sleep_mode": True},
            )
        assert "vllm_sleep_mode requires task='grpo'; got task='ppo'" in str(excinfo.value)
        assert NO_EFFECT not in str(excinfo.value)

    def test_grpo_on_mlx_keeps_the_backend_gate(self):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(
                _yaml(backend="mlx", sleep_line="  vllm_sleep_mode: true\n")
            )
        assert (
            "vllm_sleep_mode requires backend in ['transformers', 'unsloth']; "
            "got backend='mlx'"
        ) in str(excinfo.value)
        assert NO_EFFECT not in str(excinfo.value)

    @pytest.mark.parametrize("task", ["sft", "dpo", "ppo"])
    def test_other_tasks_without_the_key_are_unchanged(self, task):
        cfg = load_config_from_string(_yaml(task=task))
        assert cfg.training.vllm_sleep_mode is False


class TestTrainCommand:
    def test_soup_train_stops_at_the_config_with_exit_code_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data.jsonl").write_text(
            '{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8"
        )
        (tmp_path / "soup.yaml").write_text(
            _yaml(sleep_line="  vllm_sleep_mode: true\n"), encoding="utf-8"
        )

        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])

        output = _plain(result.output)
        assert result.exit_code == 1, (output, repr(result.exception))
        assert "Config validation error:" in output
        assert NO_EFFECT in output
        assert "vLLM sleep mode enabled via TRL GRPOConfig" not in output
