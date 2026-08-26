"""Tests for GRPO training — config, rewards, data preparation, template."""

import textwrap

import pytest

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestGRPOConfig:
    """Test GRPO task config validation."""

    def test_grpo_task_accepted(self):
        """GRPO task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "grpo"

    def test_grpo_beta_default(self):
        """grpo_beta should default to 0.1."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.grpo_beta == 0.1

    def test_grpo_beta_custom(self):
        """Custom grpo_beta should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
            training={"grpo_beta": 0.04},
        )
        assert cfg.training.grpo_beta == pytest.approx(0.04)

    def test_grpo_beta_must_be_positive(self):
        """grpo_beta must be > 0."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                task="grpo",
                data={"train": "./data.jsonl"},
                training={"grpo_beta": 0},
            )

    def test_num_generations_default(self):
        """num_generations should default to 4."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.num_generations == 4

    def test_num_generations_custom(self):
        """Custom num_generations should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
            training={"num_generations": 8},
        )
        assert cfg.training.num_generations == 8

    def test_num_generations_minimum(self):
        """num_generations must be >= 2."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                task="grpo",
                data={"train": "./data.jsonl"},
                training={"num_generations": 1},
            )

    def test_reward_fn_default(self):
        """reward_fn should default to 'accuracy'."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.reward_fn == "accuracy"

    def test_reward_fn_custom_path(self):
        """reward_fn should accept a custom file path."""
        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
            training={"reward_fn": "./my_reward.py"},
        )
        assert cfg.training.reward_fn == "./my_reward.py"

    def test_grpo_full_config(self):
        """Full GRPO config should validate correctly."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="grpo",
            data={"train": "./data.jsonl", "format": "sharegpt", "max_length": 4096},
            training={
                "epochs": 3,
                "lr": 1e-5,
                "grpo_beta": 0.1,
                "num_generations": 4,
                "reward_fn": "format",
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "grpo"
        assert cfg.training.reward_fn == "format"
        assert cfg.training.num_generations == 4
        assert cfg.data.max_length == 4096


# ─── Reward Function Tests ──────────────────────────────────────────────────


class TestAccuracyReward:
    """Test the accuracy reward function."""

    def test_exact_match(self):
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [[{"role": "assistant", "content": "The answer is #### 42"}]]
        rewards = accuracy_reward(completions, answer=["42"])
        assert rewards == [1.0]

    def test_boxed_match(self):
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [[{"role": "assistant", "content": "So \\boxed{42} is the result"}]]
        rewards = accuracy_reward(completions, answer=["42"])
        assert rewards == [1.0]

    def test_partial_match(self):
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [[{"role": "assistant", "content": "The answer is 42 degrees"}]]
        rewards = accuracy_reward(completions, answer=["42"])
        assert rewards == [0.5]

    def test_no_match(self):
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [[{"role": "assistant", "content": "I don't know"}]]
        rewards = accuracy_reward(completions, answer=["42"])
        assert rewards == [0.0]

    def test_multiple_completions(self):
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [
            [{"role": "assistant", "content": "#### 42"}],
            [{"role": "assistant", "content": "Wrong answer"}],
            [{"role": "assistant", "content": "The answer is 42"}],
        ]
        rewards = accuracy_reward(completions, answer=["42", "42", "42"])
        assert rewards == [1.0, 0.0, 0.5]

    def test_empty_completion(self):
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [[]]
        rewards = accuracy_reward(completions, answer=["42"])
        assert rewards == [0.0]


class TestFormatReward:
    """Test the format reward function."""

    def test_perfect_format(self):
        from soup_cli.trainer.rewards import format_reward

        content = "<think>Let me think step by step...</think>\nThe answer is 42."
        completions = [[{"role": "assistant", "content": content}]]
        rewards = format_reward(completions)
        assert rewards == [1.0]

    def test_think_only(self):
        from soup_cli.trainer.rewards import format_reward

        content = "<think>Thinking...</think>"
        completions = [[{"role": "assistant", "content": content}]]
        rewards = format_reward(completions)
        assert rewards == [0.5]

    def test_no_format(self):
        from soup_cli.trainer.rewards import format_reward

        completions = [[{"role": "assistant", "content": "Just a plain answer"}]]
        rewards = format_reward(completions)
        assert rewards == [0.0]

    def test_multiple_completions(self):
        from soup_cli.trainer.rewards import format_reward

        completions = [
            [{"role": "assistant", "content": "<think>A</think>\nB"}],
            [{"role": "assistant", "content": "No format"}],
        ]
        rewards = format_reward(completions)
        assert rewards == [1.0, 0.0]


class TestExtractAnswer:
    """Test answer extraction from model output."""

    def test_hash_format(self):
        from soup_cli.trainer.rewards import _extract_answer

        assert _extract_answer("Some work\n#### 42") == "42"

    def test_boxed_format(self):
        from soup_cli.trainer.rewards import _extract_answer

        assert _extract_answer("So \\boxed{42} is the answer") == "42"

    def test_no_answer(self):
        from soup_cli.trainer.rewards import _extract_answer

        assert _extract_answer("Just plain text") is None

    def test_multiple_hashes(self):
        from soup_cli.trainer.rewards import _extract_answer

        assert _extract_answer("#### step\n#### 42") == "42"


class TestLoadRewardFn:
    """Test reward function loading."""

    def test_load_builtin_accuracy(self):
        from soup_cli.trainer.rewards import accuracy_reward, load_reward_fn

        fn = load_reward_fn("accuracy")
        assert fn is accuracy_reward

    def test_load_builtin_format(self):
        from soup_cli.trainer.rewards import format_reward, load_reward_fn

        fn = load_reward_fn("format")
        assert fn is format_reward

    def test_load_custom_file(self, tmp_path):
        from soup_cli.trainer.rewards import load_reward_fn

        custom_file = tmp_path / "my_reward.py"
        custom_file.write_text(textwrap.dedent("""\
            def reward_fn(completions, **kwargs):
                return [1.0] * len(completions)
        """))
        fn = load_reward_fn(str(custom_file))
        result = fn([[{"content": "test"}]])
        assert result == [1.0]

    def test_load_custom_file_missing_fn(self, tmp_path):
        from soup_cli.trainer.rewards import load_reward_fn

        custom_file = tmp_path / "bad_reward.py"
        custom_file.write_text("x = 1\n")
        with pytest.raises(ValueError, match="must define a 'reward_fn'"):
            load_reward_fn(str(custom_file))

    def test_load_unknown_name(self):
        from soup_cli.trainer.rewards import load_reward_fn

        with pytest.raises(ValueError, match="Unknown reward function"):
            load_reward_fn("nonexistent")


# ─── Data Preparation Tests ─────────────────────────────────────────────────


class TestPrepareGRPODataset:
    """Test GRPO dataset preparation."""

    def test_from_prompt_string(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        data = [{"prompt": "What is 2+2?", "answer": "4"}]
        result = _prepare_grpo_dataset(data)
        assert len(result) == 1
        assert result[0]["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert result[0]["answer"] == "4"

    def test_from_messages(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        ]
        result = _prepare_grpo_dataset(data)
        assert len(result) == 1
        # The final assistant reference is removed from the prompt.
        assert len(result[0]["prompt"]) == 2
        assert result[0]["prompt"][0]["role"] == "system"
        assert result[0]["prompt"][1]["role"] == "user"

    def test_multi_turn_prompt_preserves_earlier_assistant_turns(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
            {"role": "assistant", "content": "Reference answer"},
        ]

        result = _prepare_grpo_dataset([{"messages": messages}])

        assert result[0]["prompt"] == messages[:-1]
        assert result[0]["answer"] == "Reference answer"

    def test_trailing_user_is_not_misread_as_an_answer_pair(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Still waiting"},
        ]

        result = _prepare_grpo_dataset([{"messages": messages}])

        assert result[0]["prompt"] == messages
        assert "answer" not in result[0]

    def test_from_prompt_message_list(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        data = [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]
        result = _prepare_grpo_dataset(data)
        assert result[0]["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert result[0]["answer"] == "4"

    def test_from_alpaca_format(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        data = [{"instruction": "Translate hello", "input": "", "output": "hola"}]
        result = _prepare_grpo_dataset(data)
        assert result[0]["prompt"] == [{"role": "user", "content": "Translate hello"}]
        assert result[0]["answer"] == "hola"

    def test_multiple_rows(self):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        data = [
            {"prompt": "Q1", "answer": "A1"},
            {"prompt": "Q2", "answer": "A2"},
            {"prompt": "Q3", "answer": "A3"},
        ]
        result = _prepare_grpo_dataset(data)
        assert len(result) == 3


# ─── Template Tests ──────────────────────────────────────────────────────────


class TestReasoningTemplate:
    """Test the reasoning/GRPO template."""

    def test_reasoning_template_exists(self):
        assert "reasoning" in TEMPLATES

    def test_reasoning_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["reasoning"])
        assert config["task"] == "grpo"
        assert config["training"]["grpo_beta"] == 0.1
        assert config["training"]["num_generations"] == 4
        assert config["training"]["reward_fn"] == "accuracy"

    def test_reasoning_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["reasoning"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "grpo"
        assert cfg.training.grpo_beta == 0.1


# ─── Train Command Routing Tests ─────────────────────────────────────────────


class TestGRPOTrainRouting:
    """Test that train command routes to GRPO trainer."""

    def test_grpo_import_exists(self):
        """GRPOTrainerWrapper should be importable."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        assert GRPOTrainerWrapper is not None

    def test_grpo_wrapper_init(self):
        """GRPOTrainerWrapper should initialize without error."""
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="grpo",
            data={"train": "./data.jsonl"},
        )
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "grpo"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None


# ─── Sweep Shortcut Tests ────────────────────────────────────────────────────


class TestGRPOSweepParams:
    """Test GRPO parameter shortcuts in sweep."""

    def test_grpo_beta_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"grpo_beta": 0.1}}
        _set_nested_param(config, "grpo_beta", 0.04)
        assert config["training"]["grpo_beta"] == 0.04

    def test_num_generations_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"num_generations": 4}}
        _set_nested_param(config, "num_generations", 8)
        assert config["training"]["num_generations"] == 8

    def test_reward_fn_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"reward_fn": "accuracy"}}
        _set_nested_param(config, "reward_fn", "format")
        assert config["training"]["reward_fn"] == "format"
