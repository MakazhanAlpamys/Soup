"""Tests for embedding task — config, data format, template, routing, sweep."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestEmbeddingConfig:
    """Test embedding task config validation."""

    def test_embedding_task_accepted(self):
        """embedding task should be a valid task type."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )
        assert cfg.task == "embedding"

    def test_embedding_default_config(self):
        """embedding task should use default training config values."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.epochs == 3
        assert cfg.training.lr == pytest.approx(2e-5)

    def test_embedding_with_embedding_format(self):
        """embedding task with embedding format should validate correctly."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl", "format": "embedding"},
        )
        assert cfg.data.format == "embedding"

    def test_embedding_format_accepted(self):
        """embedding should be a valid data format."""
        cfg = SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl", "format": "embedding"},
        )
        assert cfg.data.format == "embedding"

    def test_embedding_full_config(self):
        """Full embedding config should validate correctly."""
        cfg = SoupConfig(
            base="BAAI/bge-base-en-v1.5",
            task="embedding",
            data={"train": "./data.jsonl", "format": "embedding", "max_length": 512},
            training={
                "epochs": 3,
                "lr": 2e-5,
                "quantization": "none",
                "embedding_loss": "contrastive",
                "embedding_margin": 0.5,
                "embedding_pooling": "mean",
            },
        )
        assert cfg.task == "embedding"
        assert cfg.data.max_length == 512
        assert cfg.training.embedding_loss == "contrastive"
        assert cfg.training.embedding_margin == pytest.approx(0.5)
        assert cfg.training.embedding_pooling == "mean"

    def test_embedding_unsloth_backend(self):
        """embedding task with unsloth backend should validate correctly."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            backend="unsloth",
            data={"train": "./data.jsonl"},
        )
        assert cfg.backend == "unsloth"
        assert cfg.task == "embedding"


# ─── Embedding-specific Config Tests ─────────────────────────────────────


class TestEmbeddingTrainingConfig:
    """Test embedding-specific training config fields."""

    def test_embedding_loss_default_contrastive(self):
        """embedding_loss should default to 'contrastive'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.embedding_loss == "contrastive"

    def test_embedding_loss_triplet(self):
        """embedding_loss should accept 'triplet'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"embedding_loss": "triplet"},
        )
        assert cfg.training.embedding_loss == "triplet"

    def test_embedding_loss_cosine(self):
        """embedding_loss should accept 'cosine'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"embedding_loss": "cosine"},
        )
        assert cfg.training.embedding_loss == "cosine"

    def test_embedding_loss_invalid_rejected(self):
        """Invalid embedding_loss should be rejected."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"embedding_loss": "invalid"},
            )

    def test_embedding_margin_default(self):
        """embedding_margin should default to 0.5."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.embedding_margin == pytest.approx(0.5)

    def test_embedding_margin_custom(self):
        """Custom embedding_margin should be accepted."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"embedding_margin": 1.0},
        )
        assert cfg.training.embedding_margin == pytest.approx(1.0)

    def test_embedding_margin_zero_rejected(self):
        """Zero embedding_margin should be rejected (gt=0)."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"embedding_margin": 0.0},
            )

    def test_embedding_margin_negative_rejected(self):
        """Negative embedding_margin should be rejected."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"embedding_margin": -0.1},
            )

    def test_embedding_pooling_default_mean(self):
        """embedding_pooling should default to 'mean'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.training.embedding_pooling == "mean"

    def test_embedding_pooling_cls(self):
        """embedding_pooling should accept 'cls'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"embedding_pooling": "cls"},
        )
        assert cfg.training.embedding_pooling == "cls"

    def test_embedding_pooling_last(self):
        """embedding_pooling should accept 'last'."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
            training={"embedding_pooling": "last"},
        )
        assert cfg.training.embedding_pooling == "last"

    def test_embedding_pooling_invalid_rejected(self):
        """Invalid embedding_pooling should be rejected."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                data={"train": "./data.jsonl"},
                training={"embedding_pooling": "max"},
            )


# ─── Embedding Data Format Tests ─────────────────────────────────────────


class TestEmbeddingDataFormat:
    """Test embedding data format detection and conversion."""

    def test_format_signature_exists(self):
        """embedding format signature should be registered."""
        from soup_cli.data.formats import FORMAT_SIGNATURES

        assert "embedding" in FORMAT_SIGNATURES
        assert FORMAT_SIGNATURES["embedding"] == {"anchor", "positive"}

    def test_detect_embedding_format_pair(self):
        """Should auto-detect embedding format from anchor+positive keys."""
        from soup_cli.data.formats import detect_format

        data = [{"anchor": "What is Python?", "positive": "A programming language."}]
        assert detect_format(data) == "embedding"

    def test_detect_embedding_format_triplet(self):
        """Should auto-detect embedding format with triplet data."""
        from soup_cli.data.formats import detect_format

        data = [
            {
                "anchor": "What is Python?",
                "positive": "A programming language.",
                "negative": "A type of snake.",
            }
        ]
        assert detect_format(data) == "embedding"

    def test_convert_embedding_pair(self):
        """Should convert embedding pair row correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {"anchor": "query", "positive": "relevant doc"}
        result = format_to_messages(row, "embedding")
        assert result["anchor"] == "query"
        assert result["positive"] == "relevant doc"
        assert "negative" not in result

    def test_convert_embedding_triplet(self):
        """Should convert embedding triplet row correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {"anchor": "query", "positive": "relevant", "negative": "irrelevant"}
        result = format_to_messages(row, "embedding")
        assert result["anchor"] == "query"
        assert result["positive"] == "relevant"
        assert result["negative"] == "irrelevant"

    def test_convert_embedding_empty_anchor_returns_none(self):
        """Empty anchor should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"anchor": "", "positive": "text"}
        result = format_to_messages(row, "embedding")
        assert result is None

    def test_convert_embedding_empty_positive_returns_none(self):
        """Empty positive should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"anchor": "query", "positive": ""}
        result = format_to_messages(row, "embedding")
        assert result is None

    def test_convert_embedding_missing_anchor_returns_none(self):
        """Row missing 'anchor' key should return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"text": "some text", "positive": "relevant"}
        result = format_to_messages(row, "embedding")
        assert result is None

    def test_embedding_not_confused_with_dpo(self):
        """Embedding data should not be detected as DPO."""
        from soup_cli.data.formats import detect_format

        data = [{"anchor": "query", "positive": "relevant"}]
        assert detect_format(data) == "embedding"

    def test_convert_embedding_empty_negative_skipped(self):
        """Empty negative field should be excluded from result."""
        from soup_cli.data.formats import format_to_messages

        row = {"anchor": "query", "positive": "relevant", "negative": ""}
        result = format_to_messages(row, "embedding")
        assert "negative" not in result


# ─── Template Tests ──────────────────────────────────────────────────────


class TestEmbeddingTemplate:
    """Test the embedding template."""

    def test_embedding_template_exists(self):
        assert "embedding" in TEMPLATES

    def test_embedding_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["embedding"])
        assert config["task"] == "embedding"
        assert config["data"]["format"] == "embedding"
        assert config["training"]["embedding_loss"] == "contrastive"

    def test_embedding_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["embedding"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "embedding"
        assert cfg.data.format == "embedding"
        assert cfg.training.embedding_loss == "contrastive"
        assert cfg.training.embedding_pooling == "mean"


# ─── Train Command Routing Tests ──────────────────────────────────────────


class TestEmbeddingTrainRouting:
    """Test that train command routes to embedding trainer."""

    def test_embedding_import_exists(self):
        """EmbeddingTrainerWrapper should be importable."""
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        assert EmbeddingTrainerWrapper is not None

    def test_embedding_wrapper_init(self):
        """EmbeddingTrainerWrapper should initialize without error."""
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )
        wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "embedding"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_embedding_wrapper_init_with_options(self):
        """EmbeddingTrainerWrapper should accept all constructor options."""
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )
        wrapper = EmbeddingTrainerWrapper(
            cfg, device="cuda", report_to="wandb", deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"


# ─── Sweep Shortcut Tests ─────────────────────────────────────────────────


class TestEmbeddingSweepParams:
    """Test embedding parameter shortcuts in sweep."""

    def test_embedding_loss_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"embedding_loss": "contrastive"}}
        _set_nested_param(config, "embedding_loss", "triplet")
        assert config["training"]["embedding_loss"] == "triplet"

    def test_embedding_margin_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"embedding_margin": 0.5}}
        _set_nested_param(config, "embedding_margin", 1.0)
        assert config["training"]["embedding_margin"] == pytest.approx(1.0)

    def test_embedding_pooling_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"embedding_pooling": "mean"}}
        _set_nested_param(config, "embedding_pooling", "cls")
        assert config["training"]["embedding_pooling"] == "cls"

    def test_embedding_loss_shortcut_creates_nested_key(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "embedding_loss", "triplet")
        assert config["training"]["embedding_loss"] == "triplet"

    def test_sweep_run_single_routes_to_embedding_trainer(self):
        """_run_single should instantiate EmbeddingTrainerWrapper for embedding task."""
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )

        fake_dataset = {
            "train": [{"anchor": "query", "positive": "relevant"}]
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
             mock_patch("soup_cli.trainer.embedding.EmbeddingTrainerWrapper.setup"), \
             mock_patch(
                 "soup_cli.trainer.embedding.EmbeddingTrainerWrapper.train",
                 return_value=fake_result,
             ) as mock_train:
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-emb-1"
            mock_tracker_cls.return_value = mock_tracker

            result = _run_single(cfg, {}, "embedding_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-emb-1"


# ─── Train Guard Test ────────────────────────────────────────────────────


class TestEmbeddingTrainGuard:
    """Test the RuntimeError guard when train() is called before setup()."""

    def test_train_before_setup_raises_runtime_error(self):
        """Calling train() before setup() should raise RuntimeError."""
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )
        wrapper = EmbeddingTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup\\(dataset\\) first"):
            wrapper.train()

    def test_train_error_message_mentions_setup(self):
        """RuntimeError message should mention setup()."""
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
        )
        wrapper = EmbeddingTrainerWrapper(cfg)
        with pytest.raises(RuntimeError) as exc_info:
            wrapper.train()
        assert "setup" in str(exc_info.value).lower()


# ─── Train Method Result Structure ──────────────────────────────────────────


class TestEmbeddingTrainResults:
    """Test the result dict returned by train() using a mocked trainer."""

    def _make_wrapper_with_mock_trainer(self, log_history=None, global_step=20):
        """Helper: return an EmbeddingTrainerWrapper with trainer pre-injected."""
        from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
            output="./output",
        )
        wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
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

        with mock_patch("soup_cli.trainer.embedding.time.time", side_effect=fake_time):
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

        with mock_patch("soup_cli.trainer.embedding.time.time", side_effect=fake_time):
            result = wrapper.train()

        assert result["duration"] == "1h 2m"


# ─── Setup Transformers Integration Tests ────────────────────────────────


class TestEmbeddingSetupTransformers:
    """Test _setup_transformers integration for embedding trainer."""

    def test_lora_task_type_feature_extraction(self):
        """Embedding trainer should use TaskType.FEATURE_EXTRACTION."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
            training={"quantization": "none"},
        )

        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1000, 10000)

        with mock_patch("transformers.AutoModel.from_pretrained",
                        return_value=mock_model), \
             mock_patch("transformers.AutoTokenizer.from_pretrained"), \
             mock_patch("peft.get_peft_model", return_value=mock_model), \
             mock_patch("peft.LoraConfig") as mock_lora_config, \
             mock_patch("peft.prepare_model_for_kbit_training"):
            from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

            wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
            wrapper._setup_transformers(cfg, cfg.training)

            # Check that FEATURE_EXTRACTION task type was used
            from peft import TaskType
            call_kwargs = mock_lora_config.call_args[1]
            assert call_kwargs["task_type"] == TaskType.FEATURE_EXTRACTION

    def test_auto_target_modules_resolved_to_none(self):
        """target_modules='auto' should be resolved to None for peft."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
            training={"quantization": "none", "lora": {"target_modules": "auto"}},
        )

        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1000, 10000)

        with mock_patch("transformers.AutoModel.from_pretrained",
                        return_value=mock_model), \
             mock_patch("transformers.AutoTokenizer.from_pretrained"), \
             mock_patch("peft.get_peft_model", return_value=mock_model), \
             mock_patch("peft.LoraConfig") as mock_lora_config, \
             mock_patch("peft.prepare_model_for_kbit_training"):
            from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

            wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
            wrapper._setup_transformers(cfg, cfg.training)

            call_kwargs = mock_lora_config.call_args[1]
            assert call_kwargs["target_modules"] is None

    def test_dora_flag_forwarded(self):
        """use_dora should be forwarded to LoraConfig."""
        cfg = SoupConfig(
            base="some-model",
            task="embedding",
            data={"train": "./data.jsonl"},
            training={
                "quantization": "none",
                "lora": {"use_dora": True},
            },
        )

        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1000, 10000)

        with mock_patch("transformers.AutoModel.from_pretrained",
                        return_value=mock_model), \
             mock_patch("transformers.AutoTokenizer.from_pretrained"), \
             mock_patch("peft.get_peft_model", return_value=mock_model), \
             mock_patch("peft.LoraConfig") as mock_lora_config, \
             mock_patch("peft.prepare_model_for_kbit_training"):
            from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

            wrapper = EmbeddingTrainerWrapper(cfg, device="cpu")
            wrapper._setup_transformers(cfg, cfg.training)

            call_kwargs = mock_lora_config.call_args[1]
            assert call_kwargs["use_dora"] is True


# ─── CLI Init Template Tests ──────────────────────────────────────────────


class TestEmbeddingInitTemplate:
    """Test that soup init produces correct output for embedding."""

    def test_init_embedding_template_creates_file(self, tmp_path):
        """soup init --template embedding should write a file with embedding task."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(
            app, ["init", "--template", "embedding", "--output", str(output)]
        )
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "task: embedding" in content
        assert "format: embedding" in content
        assert "embedding_loss: contrastive" in content

    def test_init_embedding_template_produces_valid_config(self, tmp_path):
        """The file written by soup init --template embedding should parse."""
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(
            app, ["init", "--template", "embedding", "--output", str(output)]
        )
        cfg = load_config(Path(output))
        assert cfg.task == "embedding"
        assert cfg.data.format == "embedding"


# ─── Wizard Embedding Path Tests ──────────────────────────────────────────


class TestEmbeddingWizardPath:
    """Test the interactive wizard auto-sets format for embedding task."""

    def test_wizard_embedding_task_sets_embedding_format(self):
        """When the wizard receives task=embedding, data format should be 'embedding'."""
        from soup_cli.commands.init import _interactive_wizard

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
            "some-model",
            "embedding",
            "./data/pairs.jsonl",
            "3",
            "no",
        ]):
            config_text = _interactive_wizard()

        assert "task: embedding" in config_text
        assert "format: embedding" in config_text

    def test_wizard_embedding_does_not_prompt_for_format(self):
        """The wizard should NOT ask for data format when task=embedding."""
        from soup_cli.commands.init import _interactive_wizard

        prompt_calls = []

        def record_prompt(question, **kwargs):
            prompt_calls.append(question)
            answers = {
                "Base model": "some-model",
                "Task": "embedding",
                "Training data path": "./data/pairs.jsonl",
                "Epochs": "3",
                "Use QLoRA (4-bit)?": "no",
            }
            return answers.get(question, kwargs.get("default", ""))

        with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=record_prompt):
            config_text = _interactive_wizard()

        assert not any("format" in call.lower() for call in prompt_calls)
        assert "format: embedding" in config_text


# ─── Config Loader Round-trip Tests ──────────────────────────────────────


class TestEmbeddingConfigLoaderRoundTrip:
    """Test embedding template YAML survives round-trip."""

    def test_embedding_template_round_trip(self):
        """TEMPLATES['embedding'] should parse via load_config_from_string."""
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["embedding"])
        assert cfg.task == "embedding"
        assert cfg.data.format == "embedding"
        assert cfg.training.embedding_loss == "contrastive"

    def test_embedding_custom_yaml_round_trip(self):
        """Custom embedding YAML string should round-trip correctly."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: BAAI/bge-base-en-v1.5
task: embedding

data:
  train: ./data/pairs.jsonl
  format: embedding
  max_length: 512

training:
  epochs: 5
  lr: 1e-5
  quantization: none
  embedding_loss: triplet
  embedding_margin: 1.0
  embedding_pooling: cls

output: ./output_emb
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.task == "embedding"
        assert cfg.data.format == "embedding"
        assert cfg.training.embedding_loss == "triplet"
        assert cfg.training.embedding_margin == pytest.approx(1.0)
        assert cfg.training.embedding_pooling == "cls"
        assert cfg.output == "./output_emb"


# ─── Pooling Function Tests ─────────────────────────────────────────────


class TestPoolingFunction:
    """Test the _pool_embeddings helper function."""

    def test_mean_pooling(self):
        """Mean pooling should average non-padding tokens."""
        import torch

        from soup_cli.trainer.embedding import _pool_embeddings

        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        ])
        mask = torch.tensor([[1, 1, 0]])
        result = _pool_embeddings(hidden, mask, "mean")
        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(result, expected)

    def test_cls_pooling(self):
        """CLS pooling should return first token embedding."""
        import torch

        from soup_cli.trainer.embedding import _pool_embeddings

        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ])
        mask = torch.tensor([[1, 1, 1]])
        result = _pool_embeddings(hidden, mask, "cls")
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_last_pooling(self):
        """Last-token pooling should return last non-padding token."""
        import torch

        from soup_cli.trainer.embedding import _pool_embeddings

        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        ])
        mask = torch.tensor([[1, 1, 0]])
        result = _pool_embeddings(hidden, mask, "last")
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(result, expected)
