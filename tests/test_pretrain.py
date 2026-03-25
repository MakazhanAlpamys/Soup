"""Tests for pretrain task — config, plaintext format, template, routing, sweep."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestPretrainConfig:
    """Test pretrain task config validation."""

    def test_pretrain_task_accepted(self):
        """pretrain task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "pretrain"

    def test_pretrain_default_config(self):
        """pretrain task should use default training config values."""
        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.epochs == 3
        assert cfg.training.lr == pytest.approx(2e-5)

    def test_pretrain_with_plaintext_format(self):
        """pretrain task with plaintext format should validate correctly."""
        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./corpus.txt", "format": "plaintext"},
        )
        assert cfg.data.format == "plaintext"

    def test_plaintext_format_accepted(self):
        """plaintext should be a valid data format."""
        cfg = SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl", "format": "plaintext"},
        )
        assert cfg.data.format == "plaintext"

    def test_pretrain_full_config(self):
        """Full pretrain config should validate correctly."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B",
            task="pretrain",
            data={"train": "./corpus.jsonl", "format": "plaintext", "max_length": 4096},
            training={
                "epochs": 1,
                "lr": 1e-5,
                "gradient_accumulation_steps": 8,
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "pretrain"
        assert cfg.data.max_length == 4096
        assert cfg.training.gradient_accumulation_steps == 8

    def test_pretrain_unsloth_backend(self):
        """pretrain task with unsloth backend should validate correctly."""
        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"
        assert cfg.task == "pretrain"


# ─── MoE Config Tests ──────────────────────────────────────────────────────


class TestMoEConfig:
    """Test MoE-specific config fields."""

    def test_moe_lora_default_false(self):
        """moe_lora should default to False."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.moe_lora is False

    def test_moe_lora_enabled(self):
        """moe_lora should be settable to True."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"moe_lora": True},
        )
        assert cfg.training.moe_lora is True

    def test_moe_aux_loss_coeff_default(self):
        """moe_aux_loss_coeff should default to 0.01."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.moe_aux_loss_coeff == pytest.approx(0.01)

    def test_moe_aux_loss_coeff_custom(self):
        """Custom moe_aux_loss_coeff should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"moe_aux_loss_coeff": 0.05},
        )
        assert cfg.training.moe_aux_loss_coeff == pytest.approx(0.05)

    def test_moe_aux_loss_coeff_zero_allowed(self):
        """moe_aux_loss_coeff=0 should be valid (disables aux loss)."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"moe_aux_loss_coeff": 0.0},
        )
        assert cfg.training.moe_aux_loss_coeff == 0.0

    def test_moe_aux_loss_coeff_negative_rejected(self):
        """Negative moe_aux_loss_coeff should be rejected."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"moe_aux_loss_coeff": -0.1},
            )


# ─── Plaintext Data Format Tests ──────────────────────────────────────────


class TestPlaintextDataFormat:
    """Test plaintext data format detection and conversion."""

    def test_format_signature_exists(self):
        """plaintext format signature should be registered."""
        from soup_cli.data.formats import FORMAT_SIGNATURES

        assert "plaintext" in FORMAT_SIGNATURES
        assert FORMAT_SIGNATURES["plaintext"] == {"text"}

    def test_detect_plaintext_format(self):
        """Should auto-detect plaintext format from data keys."""
        from soup_cli.data.formats import detect_format

        data = [{"text": "This is a document."}]
        assert detect_format(data) == "plaintext"

    def test_detect_plaintext_with_extra_keys(self):
        """Should detect plaintext format even with extra keys."""
        from soup_cli.data.formats import detect_format

        data = [{"text": "Document content.", "id": 1, "source": "web"}]
        assert detect_format(data) == "plaintext"

    def test_convert_plaintext(self):
        """Should convert plaintext row correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {"text": "This is raw text for pre-training."}
        result = format_to_messages(row, "plaintext")
        assert result["text"] == "This is raw text for pre-training."

    def test_convert_plaintext_preserves_text_only(self):
        """Output should have exactly the 'text' key."""
        from soup_cli.data.formats import format_to_messages

        row = {"text": "content", "id": 42, "source": "web"}
        result = format_to_messages(row, "plaintext")
        assert set(result.keys()) == {"text"}

    def test_convert_plaintext_empty_text_returns_none(self):
        """Empty text should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"text": ""}
        result = format_to_messages(row, "plaintext")
        assert result is None

    def test_convert_plaintext_whitespace_only_returns_none(self):
        """Whitespace-only text should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"text": "   \n  "}
        result = format_to_messages(row, "plaintext")
        assert result is None

    def test_convert_plaintext_missing_text_returns_none(self):
        """Row missing 'text' key should return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"content": "some text"}
        result = format_to_messages(row, "plaintext")
        assert result is None

    def test_plaintext_not_confused_with_chatml(self):
        """Data with only 'text' key should not be detected as chatml."""
        from soup_cli.data.formats import detect_format

        data = [{"text": "plain content"}]
        assert detect_format(data) == "plaintext"

    def test_chatml_not_detected_as_plaintext(self):
        """Data with 'messages' key should be detected as chatml, not plaintext."""
        from soup_cli.data.formats import detect_format

        data = [{"messages": [{"role": "user", "content": "hello"}]}]
        assert detect_format(data) == "chatml"


# ─── Text File Loading Tests ──────────────────────────────────────────────


class TestTxtFileLoading:
    """Test .txt file loading."""

    def test_load_txt_file(self, tmp_path):
        """Should load .txt file as list of {text: ...} dicts."""
        from soup_cli.data.loader import load_raw_data

        txt_file = tmp_path / "corpus.txt"
        txt_file.write_text("Line one\nLine two\nLine three\n", encoding="utf-8")
        data = load_raw_data(txt_file)
        assert len(data) == 3
        assert data[0] == {"text": "Line one"}
        assert data[1] == {"text": "Line two"}
        assert data[2] == {"text": "Line three"}

    def test_load_txt_skips_empty_lines(self, tmp_path):
        """Empty lines should be skipped."""
        from soup_cli.data.loader import load_raw_data

        txt_file = tmp_path / "corpus.txt"
        txt_file.write_text("Line one\n\n\nLine two\n", encoding="utf-8")
        data = load_raw_data(txt_file)
        assert len(data) == 2
        assert data[0] == {"text": "Line one"}
        assert data[1] == {"text": "Line two"}

    def test_load_txt_empty_file(self, tmp_path):
        """Empty .txt file should return empty list."""
        from soup_cli.data.loader import load_raw_data

        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("", encoding="utf-8")
        data = load_raw_data(txt_file)
        assert data == []

    def test_txt_extension_supported(self):
        """'.txt' should be in SUPPORTED_EXTENSIONS."""
        from soup_cli.data.loader import SUPPORTED_EXTENSIONS

        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_load_txt_strips_whitespace(self, tmp_path):
        """Lines should have leading/trailing whitespace stripped."""
        from soup_cli.data.loader import load_raw_data

        txt_file = tmp_path / "corpus.txt"
        txt_file.write_text("  hello  \n  world  \n", encoding="utf-8")
        data = load_raw_data(txt_file)
        assert data[0] == {"text": "hello"}
        assert data[1] == {"text": "world"}


# ─── Template Tests ──────────────────────────────────────────────────────


class TestPretrainTemplate:
    """Test the pretrain template."""

    def test_pretrain_template_exists(self):
        assert "pretrain" in TEMPLATES

    def test_pretrain_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["pretrain"])
        assert config["task"] == "pretrain"
        assert config["data"]["format"] == "plaintext"

    def test_pretrain_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["pretrain"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "pretrain"
        assert cfg.data.format == "plaintext"

    def test_moe_template_exists(self):
        assert "moe" in TEMPLATES

    def test_moe_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["moe"])
        assert config["training"]["moe_lora"] is True
        assert config["training"]["moe_aux_loss_coeff"] == 0.01

    def test_moe_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["moe"])
        cfg = SoupConfig(**raw)
        assert cfg.training.moe_lora is True
        assert cfg.training.moe_aux_loss_coeff == pytest.approx(0.01)


# ─── Train Command Routing Tests ──────────────────────────────────────────


class TestPretrainTrainRouting:
    """Test that train command routes to pretrain trainer."""

    def test_pretrain_import_exists(self):
        """PretrainTrainerWrapper should be importable."""
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        assert PretrainTrainerWrapper is not None

    def test_pretrain_wrapper_init(self):
        """PretrainTrainerWrapper should initialize without error."""
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )
        wrapper = PretrainTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "pretrain"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_pretrain_wrapper_init_with_options(self):
        """PretrainTrainerWrapper should accept all constructor options."""
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )
        wrapper = PretrainTrainerWrapper(
            cfg, device="cuda", report_to="wandb", deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"


# ─── Sweep Shortcut Tests ─────────────────────────────────────────────────


class TestPretrainSweepParams:
    """Test pretrain/MoE parameter shortcuts in sweep."""

    def test_moe_lora_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"moe_lora": False}}
        _set_nested_param(config, "moe_lora", True)
        assert config["training"]["moe_lora"] is True

    def test_moe_aux_loss_coeff_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"moe_aux_loss_coeff": 0.01}}
        _set_nested_param(config, "moe_aux_loss_coeff", 0.05)
        assert config["training"]["moe_aux_loss_coeff"] == pytest.approx(0.05)

    def test_moe_lora_shortcut_creates_nested_key(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "moe_lora", True)
        assert config["training"]["moe_lora"] is True

    def test_sweep_run_single_routes_to_pretrain_trainer(self):
        """_run_single should instantiate PretrainTrainerWrapper for pretrain task."""
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )

        fake_dataset = {
            "train": [{"text": "Some pre-training text."}]
        }
        fake_result = {
            "initial_loss": 3.0,
            "final_loss": 2.5,
            "total_steps": 10,
            "duration_secs": 120.0,
            "output_dir": "./output",
            "duration": "2m",
        }

        fake_gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}
        with mock_patch("soup_cli.data.loader.load_dataset", return_value=fake_dataset), \
             mock_patch("soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU")), \
             mock_patch("soup_cli.utils.gpu.get_gpu_info", return_value=fake_gpu_info), \
             mock_patch("soup_cli.experiment.tracker.ExperimentTracker") as mock_tracker_cls, \
             mock_patch("soup_cli.monitoring.display.TrainingDisplay"), \
             mock_patch("soup_cli.trainer.pretrain.PretrainTrainerWrapper.setup"), \
             mock_patch(
                 "soup_cli.trainer.pretrain.PretrainTrainerWrapper.train",
                 return_value=fake_result,
             ) as mock_train:
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-pretrain-1"
            mock_tracker_cls.return_value = mock_tracker

            result = _run_single(cfg, {}, "pretrain_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-pretrain-1"


# ─── Train Guard Test ────────────────────────────────────────────────────


class TestPretrainTrainGuard:
    """Test the RuntimeError guard when train() is called before setup()."""

    def test_train_before_setup_raises_runtime_error(self):
        """Calling train() before setup() should raise RuntimeError."""
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )
        wrapper = PretrainTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup\\(dataset\\) first"):
            wrapper.train()

    def test_train_error_message_mentions_setup(self):
        """RuntimeError message should mention setup()."""
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
        )
        wrapper = PretrainTrainerWrapper(cfg)
        with pytest.raises(RuntimeError) as exc_info:
            wrapper.train()
        assert "setup" in str(exc_info.value).lower()


# ─── Train Method Result Structure ──────────────────────────────────────────


class TestPretrainTrainResults:
    """Test the result dict returned by train() using a mocked trainer."""

    def _make_wrapper_with_mock_trainer(self, log_history=None, global_step=20):
        """Helper: return a PretrainTrainerWrapper with trainer pre-injected."""
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="pretrain",
            data={"train": "./data.jsonl"},
            output="./output",
        )
        wrapper = PretrainTrainerWrapper(cfg, device="cpu")
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
            log_history=[{"loss": 3.0}, {"loss": 2.5}], global_step=10
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
            log_history=[{"loss": 3.0}, {"loss": 2.5}, {"loss": 2.0}], global_step=30
        )
        result = wrapper.train()
        assert result["initial_loss"] == pytest.approx(3.0)
        assert result["final_loss"] == pytest.approx(2.0)

    def test_train_result_empty_log_history_returns_zero_losses(self):
        """When log_history has no 'loss' entries, losses should be 0."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(log_history=[], global_step=0)
        result = wrapper.train()
        assert result["initial_loss"] == 0
        assert result["final_loss"] == 0

    def test_train_result_total_steps_from_trainer_state(self):
        """total_steps should match trainer.state.global_step."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}], global_step=42
        )
        result = wrapper.train()
        assert result["total_steps"] == 42

    def test_train_result_output_dir_matches(self):
        """output_dir in result should match wrapper._output_dir."""
        wrapper, _ = self._make_wrapper_with_mock_trainer()
        result = wrapper.train()
        assert result["output_dir"] == "./output"

    def test_train_calls_save_model(self):
        """train() should call trainer.save_model with output_dir."""
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}], global_step=5
        )
        wrapper.train()
        mock_trainer.save_model.assert_called_once_with("./output")

    def test_train_calls_tokenizer_save_pretrained(self):
        """train() should call tokenizer.save_pretrained with output_dir."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}], global_step=5
        )
        wrapper.train()
        wrapper.tokenizer.save_pretrained.assert_called_once_with("./output")

    def test_train_passes_resume_checkpoint_to_trainer(self):
        """train() should forward resume_from_checkpoint to trainer.train()."""
        wrapper, mock_trainer = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}], global_step=5
        )
        wrapper.train(resume_from_checkpoint="/ckpt/checkpoint-50")
        mock_trainer.train.assert_called_once_with(
            resume_from_checkpoint="/ckpt/checkpoint-50"
        )

    def test_train_result_duration_minutes_format(self):
        """Short durations (<1h) should produce 'Xm' format."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}], global_step=5
        )

        call_count = [0]

        def fake_time():
            call_count[0] += 1
            return 0 if call_count[0] == 1 else 90

        with mock_patch("soup_cli.trainer.pretrain.time.time", side_effect=fake_time):
            result = wrapper.train()

        assert result["duration"] == "1m"

    def test_train_result_duration_hours_format(self):
        """Long durations (>=1h) should produce 'Xh Ym' format."""
        wrapper, _ = self._make_wrapper_with_mock_trainer(
            log_history=[{"loss": 2.0}], global_step=100
        )

        call_count = [0]

        def fake_time():
            call_count[0] += 1
            return 0 if call_count[0] == 1 else 3720  # 1h 2m

        with mock_patch("soup_cli.trainer.pretrain.time.time", side_effect=fake_time):
            result = wrapper.train()

        assert result["duration"] == "1h 2m"


# ─── CLI Init Template Tests ──────────────────────────────────────────────


class TestPretrainInitTemplate:
    """Test that soup init produces correct output for pretrain."""

    def test_init_pretrain_template_creates_file(self, tmp_path):
        """soup init --template pretrain should write a file with pretrain task."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(
            app, ["init", "--template", "pretrain", "--output", str(output)]
        )
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "task: pretrain" in content
        assert "format: plaintext" in content

    def test_init_pretrain_template_produces_valid_config(self, tmp_path):
        """The file written by soup init --template pretrain should parse."""
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(
            app, ["init", "--template", "pretrain", "--output", str(output)]
        )
        cfg = load_config(Path(output))
        assert cfg.task == "pretrain"
        assert cfg.data.format == "plaintext"

    def test_init_moe_template_creates_file(self, tmp_path):
        """soup init --template moe should write a file with moe_lora."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(
            app, ["init", "--template", "moe", "--output", str(output)]
        )
        assert result.exit_code == 0
        content = output.read_text()
        assert "moe_lora: true" in content


# ─── Wizard Pretrain Path Tests ──────────────────────────────────────────


class TestPretrainWizardPath:
    """Test the interactive wizard auto-sets format for pretrain task."""

    def test_wizard_pretrain_task_sets_plaintext_format(self):
        """When the wizard receives task=pretrain, data format should be 'plaintext'."""
        from soup_cli.commands.init import _interactive_wizard

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
            "some-model",
            "pretrain",
            "./corpus.txt",
            "1",
            "yes",
        ]):
            config_text = _interactive_wizard()

        assert "task: pretrain" in config_text
        assert "format: plaintext" in config_text

    def test_wizard_pretrain_does_not_prompt_for_format(self):
        """The wizard should NOT ask for data format when task=pretrain."""
        from soup_cli.commands.init import _interactive_wizard

        prompt_calls = []

        def record_prompt(question, **kwargs):
            prompt_calls.append(question)
            answers = {
                "Base model": "some-model",
                "Task": "pretrain",
                "Training data path": "./corpus.txt",
                "Epochs": "1",
                "Use QLoRA (4-bit)?": "yes",
            }
            return answers.get(question, kwargs.get("default", ""))

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=record_prompt):
            config_text = _interactive_wizard()

        # "Data format" prompt should not appear when task is pretrain
        assert not any("format" in call.lower() for call in prompt_calls)
        assert "format: plaintext" in config_text


# ─── Config Loader Round-trip Tests ──────────────────────────────────────


class TestPretrainConfigLoaderRoundTrip:
    """Test pretrain template YAML survives round-trip through load_config_from_string."""

    def test_pretrain_template_round_trip(self):
        """TEMPLATES['pretrain'] should parse via load_config_from_string."""
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["pretrain"])
        assert cfg.task == "pretrain"
        assert cfg.data.format == "plaintext"

    def test_moe_template_round_trip(self):
        """TEMPLATES['moe'] should parse via load_config_from_string."""
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["moe"])
        assert cfg.training.moe_lora is True
        assert cfg.training.moe_aux_loss_coeff == pytest.approx(0.01)

    def test_pretrain_custom_yaml_round_trip(self):
        """Custom pretrain YAML string should round-trip correctly."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: custom-model/llama-7b
task: pretrain

data:
  train: ./corpus.jsonl
  format: plaintext
  max_length: 4096

training:
  epochs: 1
  lr: 1e-5
  quantization: none

output: ./pretrain_output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.task == "pretrain"
        assert cfg.data.format == "plaintext"
        assert cfg.data.max_length == 4096
        assert cfg.output == "./pretrain_output"
