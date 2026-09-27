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

    def test_answer_phrase_followed_by_a_unit_scores_full_credit(self):
        # #1226: there is no 0.5 substring credit any more. After an answer phrase, "42 degrees"
        # is not a bare number, so its number is read from that clause: 42. (After '####' or
        # inside \boxed{} the answer must BE the number: '#### 42 apples' scores 0.0.)
        from soup_cli.trainer.rewards import accuracy_reward

        completions = [[{"role": "assistant", "content": "The answer is 42 degrees"}]]
        rewards = accuracy_reward(completions, answer=["42"])
        assert rewards == [1.0]

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
        # #1226: "The answer is 42" is an explicit answer now, not a 0.5 substring hit.
        assert rewards == [1.0, 0.0, 1.0]

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


# ─── #342 Gradient watchdog — real-trainer smoke tests ───────────────────────
# These download a model and run a real GRPOTrainer on CPU.
# Marked @pytest.mark.smoke — deselected by the 3x3 matrix, run by pytorch-smoke.


@pytest.mark.smoke
class TestGRPOStabilityWiring:
    """Assert exactly one GRPOStabilityCallback after a real GRPOTrainer setup."""

    def test_exactly_one_callback_attached(self, tmp_path):
        """After GRPOTrainerWrapper.setup(), exactly one GRPOStabilityCallback
        is in the trainer's callback list.  On main without this PR there are
        none.  Deleting the ensure_grpo_stability_callback call fails this.
        """
        from soup_cli.monitoring.grpo_stability_callback import (
            GRPOStabilityCallback,
        )
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        config = SoupConfig(
            base="hf-internal-testing/tiny-random-LlamaForCausalLM",
            task="grpo",
            data={"train": str(tmp_path / "train.jsonl"), "max_length": 64},
            training={
                "epochs": 1,
                "lr": 1e-4,
                "batch_size": 2,
                "num_generations": 2,
                "gradient_accumulation_steps": 2,
                "grpo_beta": 0.04,
                "lora": {"r": 8, "alpha": 16},
                "quantization": "none",
            },
        )
        # Write minimal training data
        import json
        train_file = tmp_path / "train.jsonl"
        for i in range(4):
            train_file.open("a").write(
                json.dumps({"prompt": f"What is {i}+{i}?", "answer": str(i * 2)}) + "\n"
            )

        wrapper = GRPOTrainerWrapper(config, device="cpu")
        dataset = {"train": [json.loads(line) for line in train_file.read_text().splitlines()]}
        wrapper.setup(dataset)

        cbs = [
            cb for cb in wrapper.trainer.callback_handler.callbacks
            if isinstance(cb, GRPOStabilityCallback)
        ]
        assert len(cbs) == 1, f"Expected exactly 1 GRPOStabilityCallback, got {len(cbs)}"


@pytest.mark.smoke
class TestGRPOWatchdogE2E:
    """End-to-end: NaN injection on a real GRPOTrainer, watchdog skips the step."""

    def test_nan_step_leaves_weights_unchanged(self, tmp_path):
        """Inject NaN into one lora_B gradient on microbatch 1 of optimizer
        step 2.  Assert every trainable weight is bit-identical before and
        after step 2, and all weights are finite.

        This test kills four mutations:
        - Moving the check into training_step (microbatch 2 gets applied)
        - Deleting ensure_grpo_stability_callback (weights go NaN)
        - Changing zero_grad(set_to_none=False) (weights move)
        - Removing the watchdog's zero_grad (weights go NaN)
        """
        import json

        import torch
        from trl import GRPOConfig, GRPOTrainer

        from soup_cli.utils.peft_wiring import ensure_grpo_stability_callback

        model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

        # Write minimal training data — 4 prompts
        train_file = tmp_path / "train.jsonl"
        for i in range(4):
            train_file.open("a").write(
                json.dumps({"prompt": f"What is {i}+{i}?", "answer": str(i * 2)}) + "\n"
            )

        # Load model + tokenizer + LoRA
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_config = LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)

        from datasets import Dataset

        ds = Dataset.from_list([
            {"prompt": f"What is {i}+{i}?"}
            for i in range(4)
        ])

        def reward_fn(completions, **kwargs):
            return [1.0] * len(completions)

        grpo_config = GRPOConfig(
            output_dir=str(tmp_path / "output"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            # With a constant reward and lora_B initialised to zero, every
            # gradient but the injected NaN is exactly zero, so an applied
            # step moves nothing unless weight decay is on.
            weight_decay=0.01,
            max_completion_length=32,
            num_generations=2,
            beta=0.04,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
            bf16=False,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=ds,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )

        # Wire the watchdog
        ensure_grpo_stability_callback(trainer)

        # Find a lora_B parameter and register a NaN-injecting hook
        # that fires on the second optimizer step's first microbatch
        target_param = None
        for name, param in model.named_parameters():
            if "lora_B" in name and param.requires_grad:
                target_param = param
                break
        assert target_param is not None, "No lora_B parameter found"

        step_counter = {"count": 0}

        def poison_hook(grad):
            step_counter["count"] += 1
            # Poison on the 3rd backward call = microbatch 1 of step 2
            # (step 1 has 2 microbatches, step 2 starts at call 3)
            if step_counter["count"] == 3:
                return torch.full_like(grad, float("nan"))
            return grad

        handle = target_param.register_hook(poison_hook)

        # Callback to snapshot weights after step 1 (before poisoned step 2)
        from transformers import TrainerCallback

        class _SnapshotCallback(TrainerCallback):
            def __init__(self):
                self.snapshot = {}

            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step == 1 and model is not None:
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            self.snapshot[n] = p.data.clone()

        snap_cb = _SnapshotCallback()
        trainer.add_callback(snap_cb)

        try:
            trainer.train()

            # Assert all weights are finite
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert torch.isfinite(param.data).all(), (
                        f"Parameter {name} has non-finite values after training"
                    )

            # Assert weights are bit-identical to the step-1 snapshot
            # (step 2 was poisoned, so the watchdog should have skipped it)
            assert snap_cb.snapshot, "Snapshot callback did not fire"
            for name, pre in snap_cb.snapshot.items():
                post = dict(model.named_parameters())[name].data
                assert torch.equal(pre, post), (
                    f"Parameter {name} changed during poisoned step 2 "
                    f"(watchdog failed to skip)"
                )
        finally:
            handle.remove()


@pytest.mark.smoke
class TestGRPORunRecordReachesTheAudit:
    def test_a_real_run_writes_the_record_and_the_audit_reads_it(self, tmp_path, monkeypatch):
        """save_state() puts trainer_state.json next to the adapter; the audit reads it."""
        import json
        import os
        import pathlib

        import yaml
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app as adapters_app
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        monkeypatch.chdir(tmp_path)
        raw = {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": "grpo",
            "data": {"train": "train.jsonl", "max_length": 64},
            "output": "out",
            "training": {
                "epochs": 1,
                "lr": 1e-4,
                "batch_size": 2,
                "num_generations": 2,
                "gradient_accumulation_steps": 2,
                "grpo_beta": 0.04,
                "reward_fn": "format",
                "lora": {"r": 8, "alpha": 16},
                "quantization": "none",
            },
        }
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(raw))
        rows = [{"prompt": f"What is {i}+{i}?", "answer": str(i * 2)} for i in range(4)]
        wrapper = GRPOTrainerWrapper(SoupConfig(**raw), device="cpu")
        wrapper.setup({"train": rows})
        wrapper.train()

        out = pathlib.Path(wrapper._output_dir)
        state = json.loads((out / "trainer_state.json").read_text())
        assert state["log_history"][-1] == {"nan_skip_count": 0, "nan_skip_fraction": 0.0}

        rel = os.path.relpath(out, tmp_path)
        res = CliRunner().invoke(adapters_app, ["audit", rel, "--config", "soup.yaml", "--json"])
        assert res.exit_code in (0, 2), (res.output, repr(res.exception))
        row = next(
            r for r in json.loads(res.stdout)["rows"] if r["setting"] == "nan_skip_fraction"
        )
        assert (row["status"], row["ran"]) == ("ok", 0.0)
