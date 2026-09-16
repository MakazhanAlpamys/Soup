"""Tests for PPO / Full RLHF Pipeline — config, data prep, template, routing, sweep."""

import pytest
import yaml

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── PPO Config Tests ──────────────────────────────────────────────────────


class TestPPOConfig:
    """Test PPO task config validation."""

    def test_ppo_task_accepted(self):
        """PPO task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "ppo"

    def test_ppo_epochs_default(self):
        """ppo_epochs should default to 4."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.ppo_epochs == 4

    def test_ppo_epochs_custom(self):
        """Custom ppo_epochs should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"ppo_epochs": 8},
        )
        assert cfg.training.ppo_epochs == 8

    def test_ppo_epochs_minimum(self):
        """ppo_epochs must be >= 1."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                task="ppo",
                data={"train": "./data.jsonl"},
                training={"ppo_epochs": 0},
            )

    def test_ppo_clip_ratio_default(self):
        """ppo_clip_ratio should default to 0.2."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.ppo_clip_ratio == pytest.approx(0.2)

    def test_ppo_clip_ratio_custom(self):
        """Custom ppo_clip_ratio should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"ppo_clip_ratio": 0.1},
        )
        assert cfg.training.ppo_clip_ratio == pytest.approx(0.1)

    def test_ppo_clip_ratio_must_be_positive(self):
        """ppo_clip_ratio must be > 0."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                task="ppo",
                data={"train": "./data.jsonl"},
                training={"ppo_clip_ratio": 0},
            )

    def test_ppo_clip_ratio_max_one(self):
        """ppo_clip_ratio must be <= 1.0."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                task="ppo",
                data={"train": "./data.jsonl"},
                training={"ppo_clip_ratio": 1.5},
            )

    def test_ppo_kl_penalty_default(self):
        """ppo_kl_penalty should default to 0.05."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.ppo_kl_penalty == pytest.approx(0.05)

    def test_ppo_kl_penalty_custom(self):
        """Custom ppo_kl_penalty should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"ppo_kl_penalty": 0.1},
        )
        assert cfg.training.ppo_kl_penalty == pytest.approx(0.1)

    def test_ppo_kl_penalty_zero_allowed(self):
        """ppo_kl_penalty can be 0 (no KL penalty)."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"ppo_kl_penalty": 0},
        )
        assert cfg.training.ppo_kl_penalty == 0

    def test_reward_model_default_none(self):
        """reward_model should default to None."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.reward_model is None

    def test_reward_model_custom_path(self):
        """reward_model should accept a path."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"reward_model": "./output_rm"},
        )
        assert cfg.training.reward_model == "./output_rm"

    def test_reward_model_hf_id(self):
        """reward_model should accept an HF model ID."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"reward_model": "OpenAssistant/reward-model-deberta-v3-large-v2"},
        )
        assert "reward-model" in cfg.training.reward_model

    def test_ppo_full_config(self):
        """Full PPO config should validate correctly."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="ppo",
            data={"train": "./data.jsonl", "format": "chatml", "max_length": 2048},
            training={
                "epochs": 1,
                "lr": 1e-6,
                "ppo_epochs": 4,
                "ppo_clip_ratio": 0.2,
                "ppo_kl_penalty": 0.05,
                "reward_model": "./output_rm",
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "ppo"
        assert cfg.training.ppo_epochs == 4
        assert cfg.training.reward_model == "./output_rm"
        assert cfg.data.max_length == 2048


# ─── Reward Model Config Tests ────────────────────────────────────────────


class TestRewardModelConfig:
    """Test reward_model task config validation."""

    def test_reward_model_task_accepted(self):
        """reward_model task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="reward_model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "reward_model"

    def test_reward_model_with_dpo_format(self):
        """reward_model should work with DPO data format."""
        cfg = SoupConfig(
            base="some-model",
            task="reward_model",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.task == "reward_model"
        assert cfg.data.format == "dpo"

    def test_reward_model_full_config(self):
        """Full reward model config should validate correctly."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="reward_model",
            data={"train": "./pref_data.jsonl", "format": "dpo"},
            training={
                "epochs": 1,
                "lr": 1e-5,
                "lora": {"r": 32, "alpha": 16},
                "quantization": "4bit",
            },
            output="./output_rm",
        )
        assert cfg.task == "reward_model"
        assert cfg.output == "./output_rm"


# ─── PPO Data Preparation Tests ───────────────────────────────────────────


class TestPreparePPODataset:
    """Test PPO dataset preparation."""

    def test_from_prompt_string(self):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [{"prompt": "What is 2+2?", "answer": "4"}]
        result = _prepare_ppo_dataset(data)
        assert len(result) == 1
        assert result[0]["prompt_text"] == "What is 2+2?"
        assert result[0]["answer"] == "4"

    def test_from_messages(self):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        ]
        result = _prepare_ppo_dataset(data)
        assert len(result) == 1
        # Should join system + user content
        assert "You are helpful." in result[0]["prompt_text"]
        assert "Hello" in result[0]["prompt_text"]

    def test_from_messages_with_chat_template(self):
        from types import SimpleNamespace

        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        def fake_apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
            parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
            rendered = "".join(parts)
            if add_generation_prompt:
                rendered += "<assistant>"
            return rendered

        tokenizer = SimpleNamespace(
            chat_template="custom",
            apply_chat_template=fake_apply_chat_template,
        )
        data = [
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "It pulls."},
                ]
            }
        ]
        result = _prepare_ppo_dataset(data, tokenizer=tokenizer)
        assert len(result) == 1
        assert (
            result[0]["prompt_text"]
            == "<system>Be concise.</system><user>Explain gravity.</user><assistant>"
        )

    def test_from_prompt_message_list_with_chat_template(self):
        from types import SimpleNamespace

        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        def fake_apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
            return "".join(f"[{m['role']}:{m['content']}]" for m in messages)

        tokenizer = SimpleNamespace(
            chat_template="custom",
            apply_chat_template=fake_apply_chat_template,
        )
        data = [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]
        result = _prepare_ppo_dataset(data, tokenizer=tokenizer)
        assert len(result) == 1
        assert result[0]["prompt_text"] == "[user:What is 2+2?]"
        assert result[0]["answer"] == "4"

    def test_from_messages_fallback_when_template_fails(self):
        from types import SimpleNamespace

        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        def broken_apply(*_args, **_kwargs):
            raise RuntimeError("broken template")

        tokenizer = SimpleNamespace(
            chat_template="custom",
            apply_chat_template=broken_apply,
        )
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello world"},
                ]
            }
        ]
        result = _prepare_ppo_dataset(data, tokenizer=tokenizer)
        assert len(result) == 1
        assert result[0]["prompt_text"] == "Hello world"

    def test_from_prompt_message_list(self):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]
        result = _prepare_ppo_dataset(data)
        assert "What is 2+2?" in result[0]["prompt_text"]
        assert result[0]["answer"] == "4"

    def test_from_alpaca_format(self):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [{"instruction": "Translate hello", "input": "", "output": "hola"}]
        result = _prepare_ppo_dataset(data)
        assert result[0]["prompt_text"] == "Translate hello"
        assert result[0]["answer"] == "hola"

    def test_multiple_rows(self):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [
            {"prompt": "Q1", "answer": "A1"},
            {"prompt": "Q2", "answer": "A2"},
            {"prompt": "Q3", "answer": "A3"},
        ]
        result = _prepare_ppo_dataset(data)
        assert len(result) == 3

    def test_prompt_without_answer(self):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        data = [{"prompt": "Tell me a joke"}]
        result = _prepare_ppo_dataset(data)
        assert result[0]["prompt_text"] == "Tell me a joke"
        assert "answer" not in result[0]


# ─── Reward Model Data Preparation Tests ──────────────────────────────────


class TestPrepareRewardDataset:
    """Test reward model dataset preparation."""

    def test_from_dpo_format(self):
        from soup_cli.trainer.reward_model import _prepare_reward_dataset

        data = [{"prompt": "What is AI?", "chosen": "AI is...", "rejected": "I dunno"}]
        result = _prepare_reward_dataset(data)
        assert len(result) == 1
        assert "What is AI?" in result[0]["chosen"]
        assert "AI is..." in result[0]["chosen"]
        assert "What is AI?" in result[0]["rejected"]
        assert "I dunno" in result[0]["rejected"]

    def test_chosen_rejected_message_lists(self):
        from soup_cli.trainer.reward_model import _prepare_reward_dataset

        data = [
            {
                "prompt": "Hello",
                "chosen": [{"role": "assistant", "content": "Hi there!"}],
                "rejected": [{"role": "assistant", "content": "Go away"}],
            }
        ]
        result = _prepare_reward_dataset(data)
        assert "Hi there!" in result[0]["chosen"]
        assert "Go away" in result[0]["rejected"]

    def test_without_prompt(self):
        from soup_cli.trainer.reward_model import _prepare_reward_dataset

        data = [{"chosen": "Good answer", "rejected": "Bad answer"}]
        result = _prepare_reward_dataset(data)
        assert result[0]["chosen"] == "Good answer"
        assert result[0]["rejected"] == "Bad answer"

    def test_multiple_rows(self):
        from soup_cli.trainer.reward_model import _prepare_reward_dataset

        data = [
            {"prompt": "Q1", "chosen": "Good1", "rejected": "Bad1"},
            {"prompt": "Q2", "chosen": "Good2", "rejected": "Bad2"},
        ]
        result = _prepare_reward_dataset(data)
        assert len(result) == 2

    def test_prompt_as_message_list(self):
        from soup_cli.trainer.reward_model import _prepare_reward_dataset

        data = [
            {
                "prompt": [{"role": "user", "content": "Hello"}],
                "chosen": "Hi!",
                "rejected": "Bye",
            }
        ]
        result = _prepare_reward_dataset(data)
        assert "Hello" in result[0]["chosen"]


# ─── RLHF Template Tests ─────────────────────────────────────────────────


class TestRLHFTemplate:
    """Test the RLHF template."""

    def test_rlhf_template_exists(self):
        assert "rlhf" in TEMPLATES

    def test_rlhf_template_valid_yaml(self):
        config = yaml.safe_load(TEMPLATES["rlhf"])
        assert config["task"] == "ppo"
        assert config["training"]["ppo_epochs"] == 4
        assert config["training"]["ppo_clip_ratio"] == 0.2
        assert config["training"]["ppo_kl_penalty"] == 0.05
        assert config["training"]["reward_model"] == "./output_rm"

    def test_rlhf_template_valid_config(self):
        raw = yaml.safe_load(TEMPLATES["rlhf"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "ppo"
        assert cfg.training.ppo_epochs == 4
        assert cfg.training.reward_model == "./output_rm"


# ─── Train Command Routing Tests ─────────────────────────────────────────


class TestPPOTrainRouting:
    """Test that train command routes to PPO trainer."""

    def test_ppo_import_exists(self):
        """PPOTrainerWrapper should be importable."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        assert PPOTrainerWrapper is not None

    def test_ppo_wrapper_init(self):
        """PPOTrainerWrapper should initialize without error."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "ppo"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_ppo_wrapper_with_reward_config(self):
        """PPOTrainerWrapper should accept reward model config."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"reward_model": "./output_rm"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.training.reward_model == "./output_rm"

    def test_ppo_wrapper_deepspeed(self):
        """PPOTrainerWrapper should accept deepspeed config."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
        )
        wrapper = PPOTrainerWrapper(cfg, device="cuda", deepspeed_config="/tmp/ds.json")
        assert wrapper.deepspeed_config == "/tmp/ds.json"


class TestRewardModelTrainRouting:
    """Test that train command routes to RewardModel trainer."""

    def test_reward_model_import_exists(self):
        """RewardModelTrainerWrapper should be importable."""
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        assert RewardModelTrainerWrapper is not None

    def test_reward_model_wrapper_init(self):
        """RewardModelTrainerWrapper should initialize without error."""
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="reward_model",
            data={"train": "./data.jsonl"},
        )
        wrapper = RewardModelTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "reward_model"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None


# ─── Sweep Shortcut Tests ────────────────────────────────────────────────


class TestPPOSweepParams:
    """Test PPO parameter shortcuts in sweep."""

    def test_ppo_epochs_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"ppo_epochs": 4}}
        _set_nested_param(config, "ppo_epochs", 8)
        assert config["training"]["ppo_epochs"] == 8

    def test_ppo_clip_ratio_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"ppo_clip_ratio": 0.2}}
        _set_nested_param(config, "ppo_clip_ratio", 0.1)
        assert config["training"]["ppo_clip_ratio"] == 0.1

    def test_ppo_kl_penalty_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"ppo_kl_penalty": 0.05}}
        _set_nested_param(config, "ppo_kl_penalty", 0.1)
        assert config["training"]["ppo_kl_penalty"] == 0.1

    def test_reward_model_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"reward_model": None}}
        _set_nested_param(config, "reward_model", "./my_rm")
        assert config["training"]["reward_model"] == "./my_rm"


# ─── Init Command Tests ──────────────────────────────────────────────────


class TestInitRLHF:
    """Test init command with RLHF template."""

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def test_init_rlhf_template(self, tmp_path):
        """soup init --template rlhf should create a valid config."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output_file = tmp_path / "soup_ppo.yaml"
        result = runner.invoke(app, ["init", "--template", "rlhf", "--output", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()
        config = yaml.safe_load(output_file.read_text())
        assert config["task"] == "ppo"

    def test_init_help_shows_rlhf(self):
        """soup init --help should mention rlhf template."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        clean = self._strip_ansi(result.output)
        assert "rlhf" in clean


# ─── CLI Registration Tests ─────────────────────────────────────────────


class TestTrainCliRegistration:
    """Test train command handles PPO and reward_model tasks."""

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def test_train_help_shows_config(self):
        """soup train --help should show --config option."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        clean = self._strip_ansi(result.output)
        assert "--config" in clean


# ─── Edge Cases ──────────────────────────────────────────────────────────


class TestPPOEdgeCases:
    """Test edge cases for PPO configuration."""

    def test_ppo_with_reward_fn_and_no_model(self):
        """PPO with reward_fn but no reward_model should be valid."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"reward_fn": "format"},
        )
        assert cfg.training.reward_fn == "format"
        assert cfg.training.reward_model is None

    def test_ppo_with_both_reward_sources(self):
        """PPO with both reward_model and reward_fn should be valid."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={"reward_model": "./output_rm", "reward_fn": "format"},
        )
        assert cfg.training.reward_model == "./output_rm"
        assert cfg.training.reward_fn == "format"

    def test_ppo_with_unsloth_backend(self):
        """PPO should accept unsloth backend."""
        cfg = SoupConfig(
            base="some-model",
            task="ppo",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"

    def test_reward_model_with_quantization(self):
        """Reward model should accept quantization settings."""
        cfg = SoupConfig(
            base="some-model",
            task="reward_model",
            data={"train": "./data.jsonl"},
            training={"quantization": "8bit"},
        )
        assert cfg.training.quantization == "8bit"

    def test_all_five_tasks(self):
        """All five task types should be valid."""
        for task in ["sft", "dpo", "grpo", "ppo", "reward_model"]:
            cfg = SoupConfig(
                base="some-model",
                task=task,
                data={"train": "./data.jsonl"},
            )
            assert cfg.task == task

    def test_invalid_task_rejected(self):
        """Invalid task should be rejected by validation."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                task="invalid_task",
                data={"train": "./data.jsonl"},
            )
