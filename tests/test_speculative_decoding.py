"""Tests for speculative decoding — CLI flags, model loading, serve integration."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

# ─── CLI Flag Tests ──────────────────────────────────────────────────────


class TestSpeculativeDecodingCLI:
    """Test that --speculative-decoding flag is accepted by soup serve."""

    def test_serve_speculative_flag_exists(self):
        """The serve command should accept --speculative-decoding flag."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--speculative-decoding" in result.output

    def test_serve_spec_tokens_flag_exists(self):
        """The serve command should accept --spec-tokens flag."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--spec-tokens" in result.output


# ─── Draft Model Loading Tests ────────────────────────────────────────────


class TestDraftModelLoading:
    """Test draft model loading for speculative decoding."""

    def test_load_draft_model_calls_from_pretrained(self):
        """_load_draft_model should call AutoModelForCausalLM.from_pretrained."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        with mock_patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ):
            from soup_cli.commands.serve import _load_draft_model

            result = _load_draft_model("small-model", "cuda")
            mock_model.eval.assert_called_once()
            assert result == mock_model

    def test_load_draft_model_cpu_device(self):
        """CPU device should use device_map='cpu'."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        with mock_patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ) as mock_load:
            from soup_cli.commands.serve import _load_draft_model

            _load_draft_model("small-model", "cpu")
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["device_map"] == "cpu"

    def test_load_draft_model_blocks_urls(self):
        """_load_draft_model should reject URL-based model paths (SSRF)."""
        import typer

        from soup_cli.commands.serve import _load_draft_model

        with pytest.raises(typer.Exit):
            _load_draft_model("https://evil.com/model", "cuda")

    def test_load_draft_model_blocks_http_urls(self):
        """_load_draft_model should reject http:// URLs."""
        import typer

        from soup_cli.commands.serve import _load_draft_model

        with pytest.raises(typer.Exit):
            _load_draft_model("http://evil.com/model", "cpu")

    def test_load_draft_model_cuda_dtype(self):
        """CUDA device should use device_map='auto' and float16."""
        import torch

        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        with mock_patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ) as mock_load:
            from soup_cli.commands.serve import _load_draft_model

            _load_draft_model("small-model", "cuda")
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["device_map"] == "auto"
            assert call_kwargs["dtype"] == torch.float16

    def test_load_draft_model_no_trust_remote_code(self):
        """_load_draft_model should NOT pass trust_remote_code."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        with mock_patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ) as mock_load:
            from soup_cli.commands.serve import _load_draft_model

            _load_draft_model("small-model", "cuda")
            call_kwargs = mock_load.call_args[1]
            assert "trust_remote_code" not in call_kwargs


# ─── Generate Response with Draft Model Tests ────────────────────────────


class TestGenerateWithDraftModel:
    """Test that _generate_response passes assistant_model correctly."""

    def test_generate_passes_assistant_model(self):
        """When assistant_model is provided, it should be in generate kwargs."""
        import torch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.decode.return_value = "response"

        mock_draft = MagicMock()

        from soup_cli.commands.serve import _generate_response

        _generate_response(
            mock_model, mock_tokenizer,
            [{"role": "user", "content": "hello"}],
            max_tokens=10,
            assistant_model=mock_draft,
            num_assistant_tokens=3,
        )

        gen_call = mock_model.generate.call_args
        assert gen_call[1]["assistant_model"] == mock_draft
        assert gen_call[1]["num_assistant_tokens"] == 3

    def test_generate_without_assistant_model(self):
        """When assistant_model is None, it should NOT be in generate kwargs."""
        import torch

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.decode.return_value = "response"

        from soup_cli.commands.serve import _generate_response

        _generate_response(
            mock_model, mock_tokenizer,
            [{"role": "user", "content": "hello"}],
            max_tokens=10,
        )

        gen_call = mock_model.generate.call_args
        assert "assistant_model" not in gen_call[1]


# ─── Create App with Draft Model Tests ──────────────────────────────────


class TestCreateAppWithDraftModel:
    """Test that _create_app accepts and uses draft_model parameter."""

    def test_create_app_accepts_draft_model(self):
        """_create_app should accept draft_model parameter."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_draft = MagicMock()

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            model_name="test",
            max_tokens_default=512,
            draft_model=mock_draft,
            num_speculative_tokens=5,
        )
        assert app is not None

    def test_create_app_without_draft_model(self):
        """_create_app should work without draft_model."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            model_name="test",
            max_tokens_default=512,
        )
        assert app is not None


# ─── vLLM Speculative Decoding Tests ────────────────────────────────────


class TestVllmSpeculativeDecoding:
    """Test vLLM engine creation with speculative decoding params."""

    def test_vllm_engine_spec_params_signature(self):
        """create_vllm_engine should accept speculative model params."""
        import inspect

        from soup_cli.utils.vllm import create_vllm_engine

        sig = inspect.signature(create_vllm_engine)
        assert "speculative_model" in sig.parameters
        assert "num_speculative_tokens" in sig.parameters

    def test_vllm_engine_spec_params_defaults(self):
        """Default speculative params should be None and 5."""
        import inspect

        from soup_cli.utils.vllm import create_vllm_engine

        sig = inspect.signature(create_vllm_engine)
        assert sig.parameters["speculative_model"].default is None
        assert sig.parameters["num_speculative_tokens"].default == 5
