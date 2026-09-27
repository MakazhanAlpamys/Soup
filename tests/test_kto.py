"""Tests for KTO training — config, data format, template, routing, sweep."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestKTOConfig:
    """Test KTO task config validation."""

    def test_kto_task_accepted(self):
        """KTO task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "kto"

    def test_kto_beta_default(self):
        """kto_beta should default to 0.1."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.kto_beta == 0.1

    def test_kto_beta_custom(self):
        """Custom kto_beta should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
            training={"kto_beta": 0.05},
        )
        assert cfg.training.kto_beta == pytest.approx(0.05)

    def test_kto_beta_must_be_positive(self):
        """kto_beta must be > 0."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="kto",
                data={"train": "./data.jsonl"},
                training={"kto_beta": 0},
            )

    def test_kto_full_config(self):
        """Full KTO config should validate correctly."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="kto",
            data={"train": "./data.jsonl", "format": "kto", "max_length": 2048},
            training={
                "epochs": 3,
                "lr": 1e-5,
                "kto_beta": 0.2,
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "kto"
        assert cfg.training.kto_beta == pytest.approx(0.2)
        assert cfg.data.max_length == 2048

    def test_kto_data_format_accepted(self):
        """KTO format should be accepted in DataConfig."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl", "format": "kto"},
        )
        assert cfg.data.format == "kto"


# ─── Data Format Tests ─────────────────────────────────────────────────────


class TestKTODataFormat:
    """Test KTO data format detection and conversion."""

    def test_format_signature_exists(self):
        """KTO format signature should be registered."""
        from soup_cli.data.formats import FORMAT_SIGNATURES

        assert "kto" in FORMAT_SIGNATURES
        assert FORMAT_SIGNATURES["kto"] == {"prompt", "completion", "label"}

    def test_detect_kto_format(self):
        """Should auto-detect KTO format from data keys."""
        from soup_cli.data.formats import detect_format

        data = [{"prompt": "Q", "completion": "A", "label": True}]
        assert detect_format(data) == "kto"

    def test_detect_kto_with_extra_keys(self):
        """Should detect KTO format even with extra keys."""
        from soup_cli.data.formats import detect_format

        data = [{"prompt": "Q", "completion": "A", "label": False, "id": 1}]
        assert detect_format(data) == "kto"

    def test_convert_kto_desirable(self):
        """Should convert desirable KTO row correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "What is 2+2?", "completion": "4", "label": True}
        result = format_to_messages(row, "kto")
        assert result["prompt"] == "What is 2+2?"
        assert result["completion"] == "4"
        assert result["label"] is True

    def test_convert_kto_undesirable(self):
        """Should convert undesirable KTO row correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "What is 2+2?", "completion": "Fish", "label": False}
        result = format_to_messages(row, "kto")
        assert result["prompt"] == "What is 2+2?"
        assert result["completion"] == "Fish"
        assert result["label"] is False

    def test_convert_kto_label_coerced_to_bool(self):
        """Integer labels should be coerced to boolean."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "Q", "completion": "A", "label": 1}
        result = format_to_messages(row, "kto")
        assert result["label"] is True

        row_false = {"prompt": "Q", "completion": "A", "label": 0}
        result_false = format_to_messages(row_false, "kto")
        assert result_false["label"] is False

    def test_convert_kto_string_label_true(self):
        """String 'true'/'yes'/'1' should parse as True."""
        from soup_cli.data.formats import format_to_messages

        for val in ("true", "True", "TRUE", "yes", "1"):
            row = {"prompt": "Q", "completion": "A", "label": val}
            result = format_to_messages(row, "kto")
            assert result["label"] is True, f"Expected True for label={val!r}"

    def test_convert_kto_string_label_false(self):
        """String 'false'/'no'/'0' should parse as False (not truthy coercion)."""
        from soup_cli.data.formats import format_to_messages

        for val in ("false", "False", "FALSE", "no", "0"):
            row = {"prompt": "Q", "completion": "A", "label": val}
            result = format_to_messages(row, "kto")
            assert result["label"] is False, f"Expected False for label={val!r}"

    def test_convert_kto_string_label_invalid_returns_none(self):
        """Invalid string label should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "Q", "completion": "A", "label": "maybe"}
        result = format_to_messages(row, "kto")
        assert result is None


# ─── Template Tests ──────────────────────────────────────────────────────────


class TestKTOTemplate:
    """Test the KTO template."""

    def test_kto_template_exists(self):
        assert "kto" in TEMPLATES

    def test_kto_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["kto"])
        assert config["task"] == "kto"
        assert config["training"]["kto_beta"] == 0.1
        assert config["data"]["format"] == "kto"

    def test_kto_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["kto"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "kto"
        assert cfg.training.kto_beta == 0.1


# ─── Train Command Routing Tests ─────────────────────────────────────────────


class TestKTOTrainRouting:
    """Test that train command routes to KTO trainer."""

    def test_kto_import_exists(self):
        """KTOTrainerWrapper should be importable."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        assert KTOTrainerWrapper is not None

    def test_kto_wrapper_init(self):
        """KTOTrainerWrapper should initialize without error."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "kto"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_kto_wrapper_init_with_options(self):
        """KTOTrainerWrapper should accept all constructor options."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(
            cfg, device="cuda", report_to="wandb", deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"


# ─── Sweep Shortcut Tests ────────────────────────────────────────────────────


class TestKTOSweepParams:
    """Test KTO parameter shortcuts in sweep."""

    def test_kto_beta_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"kto_beta": 0.1}}
        _set_nested_param(config, "kto_beta", 0.05)
        assert config["training"]["kto_beta"] == 0.05

    def test_kto_beta_shortcut_creates_nested_key(self):
        """kto_beta shortcut should create nested training dict if missing."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "kto_beta", 0.2)
        assert config["training"]["kto_beta"] == pytest.approx(0.2)

    def test_sweep_run_single_routes_to_kto_trainer(self):
        """_run_single should instantiate KTOTrainerWrapper for kto task."""
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )

        fake_dataset = {
            "train": [
                {"prompt": "Q?", "completion": "A", "label": True},
            ]
        }
        fake_result = {
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "total_steps": 10,
            "duration_secs": 60.0,
            "output_dir": "./output",
            "duration": "1m",
        }

        fake_gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}
        with mock_patch("soup_cli.data.loader.load_dataset", return_value=fake_dataset), \
             mock_patch("soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU")), \
             mock_patch("soup_cli.utils.gpu.get_gpu_info", return_value=fake_gpu_info), \
             mock_patch("soup_cli.experiment.tracker.ExperimentTracker") as mock_tracker_cls, \
             mock_patch("soup_cli.monitoring.display.TrainingDisplay"), \
             mock_patch("soup_cli.trainer.kto.KTOTrainerWrapper.setup"), \
             mock_patch(
                 "soup_cli.trainer.kto.KTOTrainerWrapper.train", return_value=fake_result
             ) as mock_train:
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-kto-1"
            mock_tracker_cls.return_value = mock_tracker

            result = _run_single(cfg, {}, "kto_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-kto-1"


# ─── Config Validation Edge Cases ────────────────────────────────────────────


class TestKTOConfigEdgeCases:
    """Additional config validation edge cases for KTO."""

    def test_kto_beta_negative_rejected(self):
        """Negative kto_beta should be rejected."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                task="kto",
                data={"train": "./data.jsonl"},
                training={"kto_beta": -0.1},
            )

    def test_kto_beta_very_large_accepted(self):
        """Very large kto_beta values should be accepted (no upper bound)."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
            training={"kto_beta": 10.0},
        )
        assert cfg.training.kto_beta == pytest.approx(10.0)

    def test_kto_task_with_non_kto_format_accepted(self):
        """KTO task with a non-kto format in data config should be accepted by schema."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl", "format": "auto"},
        )
        assert cfg.task == "kto"
        assert cfg.data.format == "auto"

    def test_kto_config_unsloth_backend(self):
        """KTO task with unsloth backend should validate correctly."""
        cfg = SoupConfig(
            base="some-model",
            task="kto",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"
        assert cfg.task == "kto"

    def test_kto_tokenizer_stored_as_none_before_setup(self):
        """tokenizer attribute should be None before setup is called."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(cfg)
        assert wrapper.tokenizer is None

    def test_kto_output_dir_stored_as_none_before_setup(self):
        """_output_dir attribute should be None before setup is called."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(cfg)
        assert wrapper._output_dir is None


# ─── Data Format Detection Edge Cases ────────────────────────────────────────


class TestKTODataFormatEdgeCases:
    """Edge cases for KTO data format detection."""

    def test_detect_empty_dataset_raises(self):
        """detect_format on empty list should raise ValueError."""
        from soup_cli.data.formats import detect_format

        with pytest.raises(ValueError, match="Empty dataset"):
            detect_format([])

    def test_dpo_keys_do_not_match_kto(self):
        """Data with DPO keys (chosen, rejected) should not be detected as KTO."""
        from soup_cli.data.formats import detect_format

        data = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        assert detect_format(data) == "dpo"

    def test_kto_keys_do_not_match_dpo(self):
        """Data with KTO keys (completion, label) should not be detected as DPO."""
        from soup_cli.data.formats import detect_format

        data = [{"prompt": "Q", "completion": "A", "label": True}]
        assert detect_format(data) == "kto"

    def test_kto_checked_before_dpo_in_order(self):
        """KTO appears before DPO in check_order so KTO takes priority when keys overlap."""
        from soup_cli.data.formats import detect_format

        # A row that has KTO keys but NOT the DPO-only keys (chosen, rejected)
        data = [{"prompt": "Q", "completion": "A", "label": False}]
        result = detect_format(data)
        assert result == "kto"

    def test_detect_format_unknown_keys_raises(self):
        """Data with unrecognised keys should raise ValueError."""
        from soup_cli.data.formats import detect_format

        data = [{"question": "Q", "answer": "A"}]
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(data)

    def test_format_to_messages_unknown_format_raises(self):
        """format_to_messages with an unknown format name should raise ValueError."""
        from soup_cli.data.formats import format_to_messages

        with pytest.raises(ValueError, match="Unknown format"):
            format_to_messages({"prompt": "Q", "completion": "A", "label": True}, "kto_v2")

    def test_convert_kto_missing_prompt_returns_none(self):
        """Row missing required 'prompt' key should return None (exception caught)."""
        from soup_cli.data.formats import format_to_messages

        row = {"completion": "A", "label": True}
        result = format_to_messages(row, "kto")
        assert result is None

    def test_convert_kto_missing_completion_returns_none(self):
        """Row missing required 'completion' key should return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "Q", "label": True}
        result = format_to_messages(row, "kto")
        assert result is None

    def test_convert_kto_missing_label_returns_none(self):
        """Row missing required 'label' key should return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "Q", "completion": "A"}
        result = format_to_messages(row, "kto")
        assert result is None

    def test_convert_kto_string_label_whitespace_stripped(self):
        """String labels with surrounding whitespace should be stripped and parsed."""
        from soup_cli.data.formats import format_to_messages

        row_true = {"prompt": "Q", "completion": "A", "label": "  true  "}
        result_true = format_to_messages(row_true, "kto")
        assert result_true["label"] is True

        row_false = {"prompt": "Q", "completion": "A", "label": "  false  "}
        result_false = format_to_messages(row_false, "kto")
        assert result_false["label"] is False

    def test_convert_kto_preserves_extra_keys_not_passed_through(self):
        """Extra keys in the row are not forwarded — output has exactly prompt/completion/label."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "Q", "completion": "A", "label": True, "id": 42, "source": "web"}
        result = format_to_messages(row, "kto")
        assert set(result.keys()) == {"prompt", "completion", "label"}

    def test_convert_kto_none_label_coerced_to_false(self):
        """None label should be coerced to False via bool()."""
        from soup_cli.data.formats import format_to_messages

        row = {"prompt": "Q", "completion": "A", "label": None}
        result = format_to_messages(row, "kto")
        assert result["label"] is False


# ─── Train Method Guard Test ──────────────────────────────────────────────────


class TestKTOTrainGuard:
    """Test the RuntimeError guard when train() is called before setup()."""

    def test_train_before_setup_raises_runtime_error(self):
        """Calling train() before setup() should raise RuntimeError."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup\\(dataset\\) first"):
            wrapper.train()

    def test_train_error_message_mentions_setup(self):
        """RuntimeError message should mention setup()."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
        )
        wrapper = KTOTrainerWrapper(cfg)
        with pytest.raises(RuntimeError) as exc_info:
            wrapper.train()
        assert "setup" in str(exc_info.value).lower()


# ─── Train Method Result Structure ───────────────────────────────────────────


class TestKTOTrainResults:
    """Test the result dict returned by train() using a mocked trainer."""

    def _make_wrapper_with_mock_trainer(self, log_history=None, global_step=20):
        """Helper: return a KTOTrainerWrapper with trainer pre-injected."""
        from soup_cli.trainer.kto import KTOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="kto",
            data={"train": "./data.jsonl"},
            output="./output",
        )
        wrapper = KTOTrainerWrapper(cfg, device="cpu")
        mock_trainer = MagicMock()
        mock_trainer.train = MagicMock()
        mock_trainer.state.log_history = log_history if log_history is not None else []
        mock_trainer.state.global_step = global_step
        mock_trainer.save_model = MagicMock()
        wrapper.trainer = mock_trainer
        wrapper.tokenizer = MagicMock()
        wrapper._output_dir = "./output"
        return wrapper, mock_trainer

    def test_train_returns_expected_keys(self):
        """train() result dict must contain all expected keys."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.5}, {"loss": 0.8}], global_step=10
        )
        result = wrapper.train()
        assert "initial_loss" in result
        assert "final_loss" in result
        assert "duration" in result
        assert "duration_secs" in result
        assert "output_dir" in result
        assert "total_steps" in result

    def test_train_result_losses_from_log_history(self):
        """initial_loss and final_loss should come from trainer log_history."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}, {"loss": 1.0}, {"loss": 0.5}], global_step=30
        )
        result = wrapper.train()
        assert result["initial_loss"] == pytest.approx(2.0)
        assert result["final_loss"] == pytest.approx(0.5)

    def test_train_result_empty_log_history_returns_zero_losses(self):
        """When log_history has no 'loss' entries, losses should be 0."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(log_history=[], global_step=0)
        result = wrapper.train()
        assert result["initial_loss"] == 0
        assert result["final_loss"] == 0

    def test_train_result_total_steps_from_trainer_state(self):
        """total_steps should match trainer.state.global_step."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=42
        )
        result = wrapper.train()
        assert result["total_steps"] == 42

    def test_train_result_output_dir_matches(self):
        """output_dir in result should match wrapper._output_dir."""
        wrapper, _ = self._make_wrapper_with_mock_trainer()
        result = wrapper.train()
        assert result["output_dir"] == "./output"

    def test_train_result_duration_minutes_format(self):
        """Short durations (<1h) should produce 'Xm' format."""
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=5
        )

        # Patch time.time to control elapsed duration: 90 seconds
        call_count = [0]

        def fake_time():
            call_count[0] += 1
            return 0 if call_count[0] == 1 else 90

        with mock_patch("soup_cli.trainer.kto.time.time", side_effect=fake_time):
            result = wrapper.train()

        assert result["duration"] == "1m"

    def test_train_result_duration_hours_format(self):
        """Long durations (>=1h) should produce 'Xh Ym' format."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=100
        )

        call_count = [0]

        def fake_time():
            call_count[0] += 1
            return 0 if call_count[0] == 1 else 3720  # 1h 2m

        with mock_patch("soup_cli.trainer.kto.time.time", side_effect=fake_time):
            result = wrapper.train()

        assert result["duration"] == "1h 2m"

    def test_train_calls_save_model(self):
        """train() should call trainer.save_model with output_dir."""
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=5
        )
        wrapper.train()
        mock_trainer.save_model.assert_called_once_with("./output")

    def test_train_calls_tokenizer_save_pretrained(self):
        """train() should call tokenizer.save_pretrained with output_dir."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=5
        )
        wrapper.train()
        wrapper.tokenizer.save_pretrained.assert_called_once_with("./output")

    def test_train_passes_resume_checkpoint_to_trainer(self):
        """train() should forward resume_from_checkpoint to trainer.train()."""
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 1.0}], global_step=5
        )
        wrapper.train(resume_from_checkpoint="/ckpt/checkpoint-50")
        mock_trainer.train.assert_called_once_with(
            resume_from_checkpoint="/ckpt/checkpoint-50"
        )

    def test_train_log_history_skips_non_loss_entries(self):
        """Log entries without 'loss' key should not be counted in train_losses."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[
                {"eval_loss": 2.5},
                {"loss": 1.0},
                {"eval_loss": 1.2},
                {"loss": 0.6},
            ],
            global_step=4,
        )
        result = wrapper.train()
        assert result["initial_loss"] == pytest.approx(1.0)
        assert result["final_loss"] == pytest.approx(0.6)


# ─── CLI Init KTO Template Tests ─────────────────────────────────────────────


class TestKTOInitTemplate:
    """Test that soup init produces correct output for KTO."""

    def test_init_kto_template_creates_file(self, tmp_path):
        """soup init --template kto should write a file with kto task."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(app, ["init", "--template", "kto", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "task: kto" in content
        assert "kto_beta" in content
        assert "format: kto" in content

    def test_init_kto_template_produces_valid_config(self, tmp_path):
        """The file written by soup init --template kto should parse to a valid SoupConfig."""
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(app, ["init", "--template", "kto", "--output", str(output)])
        cfg = load_config(Path(output))
        assert cfg.task == "kto"
        assert cfg.training.kto_beta == pytest.approx(0.1)
        assert cfg.data.format == "kto"


# ─── Wizard KTO Path Tests ────────────────────────────────────────────────────


class TestKTOWizardPath:
    """Test the interactive wizard auto-sets format for KTO task."""

    def test_wizard_kto_task_sets_kto_format(self):
        """When the wizard receives task=kto, data format should be forced to 'kto'."""
        from soup_cli.commands.init import _interactive_wizard

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
            "some-model",
            "kto",
            "./data.jsonl",
            "3",
            "yes",
        ]):
            config_text = _interactive_wizard()

        assert "task: kto" in config_text
        assert "format: kto" in config_text
        assert "kto_beta: 0.1" in config_text

    def test_wizard_kto_does_not_prompt_for_format(self):
        """The wizard should NOT ask for data format when task=kto."""
        from soup_cli.commands.init import _interactive_wizard

        prompt_calls = []

        def record_prompt(question, **kwargs):
            prompt_calls.append(question)
            answers = {
                "Base model": "some-model",
                "Task": "kto",
                "Training data path": "./data.jsonl",
                "Epochs": "3",
                "Use QLoRA (4-bit)?": "yes",
            }
            return answers.get(question, kwargs.get("default", ""))

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=record_prompt):
            config_text = _interactive_wizard()

        # "Data format" prompt should not appear when task is kto
        assert not any("format" in call.lower() for call in prompt_calls)
        assert "format: kto" in config_text


# ─── Config Loader Round-trip Tests ──────────────────────────────────────────


class TestKTOConfigLoaderRoundTrip:
    """Test KTO template YAML survives round-trip through load_config_from_string."""

    def test_kto_template_round_trip(self):
        """TEMPLATES['kto'] should parse via load_config_from_string without error."""
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["kto"])
        assert cfg.task == "kto"
        assert cfg.training.kto_beta == pytest.approx(0.1)
        assert cfg.data.format == "kto"

    def test_kto_custom_yaml_round_trip(self):
        """Custom KTO YAML string should round-trip correctly."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: custom-model/llama-7b
task: kto

data:
  train: ./kto_data.jsonl
  format: kto
  max_length: 1024

training:
  epochs: 5
  lr: 5e-6
  kto_beta: 0.05
  quantization: none

output: ./kto_output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.task == "kto"
        assert cfg.training.kto_beta == pytest.approx(0.05)
        assert cfg.training.epochs == 5
        assert cfg.data.max_length == 1024
        assert cfg.output == "./kto_output"

    def test_kto_invalid_beta_in_yaml_raises_value_error(self):
        """YAML with invalid kto_beta should raise ValueError from load_config_from_string."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: some-model
task: kto
data:
  train: ./data.jsonl
training:
  kto_beta: -1.0
"""
        with pytest.raises(ValueError):
            load_config_from_string(yaml_str)
