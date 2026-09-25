"""Tests for soup infer — batch inference command."""

import json
from unittest.mock import MagicMock, PropertyMock

import pytest

from tests.conftest import strip_ansi


class _BatchTokenizer:
    def __init__(self, *, templated: bool):
        self.chat_template = "dummy" if templated else None
        self.pad_token_id = 0
        self.padding_side = "right"
        self.calls = []
        self.padding_sides = []
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return f"templated:{messages[0]['content']}"

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
    ):
        import torch

        is_batch = isinstance(texts, list)
        text_rows = texts if is_batch else [texts]
        self.calls.append(
            {
                "is_batch": is_batch,
                "add_special_tokens": add_special_tokens,
                "padding": padding,
                "return_tensors": return_tensors,
            }
        )
        self.padding_sides.append(self.padding_side)
        rows = [list(range(1, len(text) % 3 + 2)) for text in text_rows]
        width = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad_count = width - len(row)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_count + row)
                masks.append([0] * pad_count + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * pad_count)
                masks.append([1] * len(row) + [0] * pad_count)
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return f"response-{int(token_ids[-1])}"


class _DeterministicBatchModel:
    def __init__(self):
        import torch

        self.device = torch.device("cpu")
        self.generate_calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        import torch

        self.generate_calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "kwargs": kwargs,
            }
        )
        generated = []
        for row, mask in zip(input_ids, attention_mask, strict=True):
            signature = int(row[mask.bool()].sum())
            generated.append(torch.tensor([[100 + signature]]))
        return torch.cat([input_ids, torch.cat(generated)], dim=1)

# ─── Prompt Reading Tests ───────────────────────────────────────────────────


class TestReadPrompts:
    """Test prompt reading from files."""

    def test_read_jsonl_prompts(self, tmp_path):
        """Should read prompts from JSONL with 'prompt' field."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "prompts.jsonl"
        lines = [
            json.dumps({"prompt": "What is AI?"}),
            json.dumps({"prompt": "Explain gravity."}),
            json.dumps({"prompt": "Hello world."}),
        ]
        path.write_text("\n".join(lines))

        result = _read_prompts(path)
        assert len(result) == 3
        assert result[0] == "What is AI?"
        assert result[1] == "Explain gravity."
        assert result[2] == "Hello world."

    def test_read_plain_text_prompts(self, tmp_path):
        """Should read plain text lines as prompts."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "prompts.txt"
        path.write_text("What is AI?\nExplain gravity.\nHello world.\n")

        result = _read_prompts(path)
        assert len(result) == 3
        assert result[0] == "What is AI?"

    def test_read_skips_empty_lines(self, tmp_path):
        """Should skip empty lines."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "prompts.txt"
        path.write_text("Line one\n\n\nLine two\n\n")

        result = _read_prompts(path)
        assert len(result) == 2

    def test_read_empty_file(self, tmp_path):
        """Should return empty list for empty file."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "empty.jsonl"
        path.write_text("")

        result = _read_prompts(path)
        assert result == []

    def test_read_mixed_jsonl_and_text(self, tmp_path):
        """Should handle mixed JSONL and plain text lines."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "mixed.txt"
        lines = [
            json.dumps({"prompt": "From JSONL"}),
            "Plain text prompt",
        ]
        path.write_text("\n".join(lines))

        result = _read_prompts(path)
        assert len(result) == 2
        assert result[0] == "From JSONL"
        assert result[1] == "Plain text prompt"

    def test_read_jsonl_without_prompt_field(self, tmp_path):
        """JSONL without 'prompt' field should be treated as plain text."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "noprompt.jsonl"
        lines = [
            json.dumps({"text": "This has no prompt field"}),
        ]
        path.write_text("\n".join(lines))

        result = _read_prompts(path)
        assert len(result) == 1
        # Falls through to plain text since 'prompt' key missing
        assert "text" in result[0]


# ─── CLI Validation Tests ──────────────────────────────────────────────────


class TestInferCLI:
    """Test infer CLI command validation."""

    def test_input_file_not_found(self, tmp_path):
        """Should fail if input file doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", str(tmp_path),
            "--input", "/nonexistent/prompts.jsonl",
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_model_not_found(self, tmp_path):
        """Should fail if model path doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(json.dumps({"prompt": "test"}) + "\n")

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", "/nonexistent/model",
            "--input", str(prompts_file),
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_empty_input_file(self, tmp_path):
        """Should fail if input file has no prompts."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "empty.jsonl"
        prompts_file.write_text("")

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", str(tmp_path),
            "--input", str(prompts_file),
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert result.exit_code != 0
        assert "no prompts" in result.output.lower()

    def test_infer_command_registered(self):
        """Infer command should be registered in the app."""
        from soup_cli.cli import app

        command_names = [
            cmd.name or (cmd.callback.__name__ if cmd.callback else None)
            for cmd in app.registered_commands
        ]
        assert "infer" in command_names

    def test_infer_help(self):
        """Infer command should show help text."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["infer", "--help"])
        assert result.exit_code == 0
        assert "batch inference" in result.output.lower()
        assert "--batch-size" in strip_ansi(result.output)

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_batch_size_must_be_positive(self, tmp_path, batch_size, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(json.dumps({"prompt": "test"}) + "\n")
        monkeypatch.setattr(
            "soup_cli.commands.infer._load_model",
            lambda *args, **kwargs: (object(), object()),
        )

        result = CliRunner().invoke(
            app,
            [
                "infer",
                "--model", str(tmp_path),
                "--input", str(prompts_file),
                "--output", str(tmp_path / "out.jsonl"),
                "--batch-size", str(batch_size),
            ],
        )

        assert result.exit_code == 2
        assert "x>=1" in result.output

    def test_required_options_error(self):
        """Should fail if required options are missing."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["infer"])
        assert result.exit_code != 0

    def test_max_tokens_too_large_rejected(self, tmp_path):
        """--max-tokens above 16384 should be rejected by CLI."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(json.dumps({"prompt": "test"}) + "\n")

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", str(tmp_path),
            "--input", str(prompts_file),
            "--output", str(tmp_path / "out.jsonl"),
            "--max-tokens", "99999",
        ])
        assert result.exit_code != 0

    def test_max_tokens_zero_rejected(self, tmp_path):
        """--max-tokens 0 should be rejected by CLI."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(json.dumps({"prompt": "test"}) + "\n")

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", str(tmp_path),
            "--input", str(prompts_file),
            "--output", str(tmp_path / "out.jsonl"),
            "--max-tokens", "0",
        ])
        assert result.exit_code != 0


# ─── Load Model Tests ────────────────────────────────────────────────────


class TestLoadModel:
    """Test _load_model adapter detection and error paths."""

    def test_adapter_without_base_model_exits(self, tmp_path):
        """Should exit if adapter found but no base model detectable."""

        from soup_cli.commands.infer import _load_model

        # Create adapter_config.json without base_model_name_or_path
        adapter_config = tmp_path / "adapter_config.json"
        adapter_config.write_text(json.dumps({"peft_type": "LORA"}))

        from click.exceptions import Exit

        with pytest.raises(Exit):
            _load_model(str(tmp_path), None, "cpu")

    def test_adapter_with_corrupt_json_exits(self, tmp_path):
        """Should exit if adapter_config.json is corrupt and no --base given."""
        from soup_cli.commands.infer import _load_model

        adapter_config = tmp_path / "adapter_config.json"
        adapter_config.write_text("not valid json {{{")

        from click.exceptions import Exit

        with pytest.raises(Exit):
            _load_model(str(tmp_path), None, "cpu")

    def test_adapter_config_reads_base_model(self, tmp_path):
        """Should read base_model_name_or_path from adapter_config.json."""
        adapter_config = tmp_path / "adapter_config.json"
        adapter_config.write_text(json.dumps({
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
        }))

        config = json.loads(adapter_config.read_text())
        assert config["base_model_name_or_path"] == "meta-llama/Llama-3.1-8B-Instruct"

    def test_no_adapter_config_is_full_model(self, tmp_path):
        """Without adapter_config.json, model is treated as full model."""
        assert not (tmp_path / "adapter_config.json").exists()


# ─── Generate Function Tests ─────────────────────────────────────────────


class TestGenerate:
    """Test _generate function branches."""

    def _make_mock_model_and_tokenizer(self, has_chat_template=True):
        """Create mock model and tokenizer for testing _generate."""
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        if has_chat_template:
            mock_tokenizer.chat_template = "dummy_template"
            mock_tokenizer.apply_chat_template = MagicMock(
                return_value="<|user|>Hello<|assistant|>"
            )
        else:
            mock_tokenizer.chat_template = None

        # Mock tokenizer call (tokenizer(text, return_tensors="pt"))
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode = MagicMock(return_value="Hello there!")

        mock_model = MagicMock()
        type(mock_model).device = PropertyMock(return_value=torch.device("cpu"))
        mock_model.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4, 5, 6]])  # 3 input + 3 new tokens
        )

        return mock_model, mock_tokenizer

    def test_generate_with_chat_template(self):
        """Should use apply_chat_template when available."""
        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer(has_chat_template=True)
        messages = [{"role": "user", "content": "Hello"}]

        response, token_count = _generate(model, tokenizer, messages)

        tokenizer.apply_chat_template.assert_called_once()
        assert isinstance(response, str)
        assert token_count == 3  # 6 total - 3 input

    def test_generate_without_chat_template(self):
        """Should use fallback formatter when no chat_template."""
        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer(has_chat_template=False)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        response, token_count = _generate(model, tokenizer, messages)

        # Should NOT call apply_chat_template
        assert not hasattr(tokenizer.apply_chat_template, 'called') or \
            not tokenizer.apply_chat_template.called
        assert isinstance(response, str)
        assert token_count == 3

    def test_generate_greedy_temperature_zero(self):
        """Should set do_sample=False when temperature=0."""
        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer()
        messages = [{"role": "user", "content": "Hello"}]

        _generate(model, tokenizer, messages, temperature=0.0)

        # Check gen_kwargs passed to model.generate
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    def test_generate_sampling_temperature_positive(self):
        """Should set do_sample=True and include temperature when > 0."""
        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer()
        messages = [{"role": "user", "content": "Hello"}]

        _generate(model, tokenizer, messages, temperature=0.7)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == pytest.approx(0.7)
        assert "top_p" in call_kwargs

    def test_generate_returns_tuple(self):
        """_generate should return (text, token_count) tuple."""
        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer()
        messages = [{"role": "user", "content": "Hello"}]

        result = _generate(model, tokenizer, messages)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)

    def test_generate_token_count_from_tensor_shape(self):
        """Token count should come from tensor shape, not re-encoding."""
        import torch

        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer()
        # Return 7 tokens total, 3 are input => 4 new tokens
        model.generate.return_value = torch.tensor([[1, 2, 3, 10, 20, 30, 40]])
        messages = [{"role": "user", "content": "Hello"}]

        _, token_count = _generate(model, tokenizer, messages)
        assert token_count == 4  # 7 - 3 input tokens

    def test_generate_fallback_formats_roles(self):
        """Fallback formatter should include all role types."""
        from soup_cli.commands.infer import _generate

        model, tokenizer = self._make_mock_model_and_tokenizer(has_chat_template=False)
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ]

        _generate(model, tokenizer, messages)

        # Check that the tokenizer was called with formatted text
        tokenizer_call_args = tokenizer.call_args[0][0]
        assert "System: Be helpful" in tokenizer_call_args
        assert "User: What is 2+2?" in tokenizer_call_args
        assert "Assistant:" in tokenizer_call_args


class TestGenerateBatch:
    def test_batch_template_errors_are_not_silently_fallback(self):
        from soup_cli.commands.infer import _generate_batch

        class BrokenTemplateTokenizer(_BatchTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                raise ValueError("broken chat template")

        with pytest.raises(ValueError, match="broken chat template"):
            _generate_batch(
                _DeterministicBatchModel(),
                BrokenTemplateTokenizer(templated=True),
                ["first", "second"],
                max_tokens=1,
                temperature=0,
            )

    def test_batch_generation_uses_left_padding_and_one_call(self):
        from soup_cli.commands.infer import _generate_batch

        model = _DeterministicBatchModel()
        tokenizer = _BatchTokenizer(templated=True)

        results = _generate_batch(
            model,
            tokenizer,
            ["xx", "xxxx"],
            max_tokens=4,
            temperature=0,
        )

        assert len(results) == 2
        assert len(model.generate_calls) == 1
        assert tokenizer.padding_side == "right"
        assert tokenizer.padding_sides[-1] == "left"
        assert tokenizer.calls[-1] == {
            "is_batch": True,
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
        }
        assert model.generate_calls[0]["input_ids"][0, 0].item() == 0

    def test_batch_generation_counts_tokens_before_padding(self):
        import torch

        from soup_cli.commands.infer import _generate_batch

        class VariableLengthBatchModel:
            device = torch.device("cpu")

            def generate(self, input_ids, attention_mask, **kwargs):
                generated = []
                lengths = [2, 5]

                for index, (row, mask) in enumerate(
                    zip(input_ids, attention_mask, strict=True)
                ):
                    signature = int(row[mask.bool()].sum())
                    tokens = [
                        100 + signature + offset
                        for offset in range(lengths[index])
                    ]
                    generated.append(torch.tensor([tokens], dtype=torch.long))

                max_length = max(lengths)
                padded = [
                    torch.cat(
                        [
                            tokens,
                            torch.zeros(
                                (1, max_length - tokens.shape[1]),
                                dtype=torch.long,
                            ),
                        ],
                        dim=1,
                    )
                    for tokens in generated
                ]
                return torch.cat([input_ids, torch.cat(padded)], dim=1)

        model = VariableLengthBatchModel()
        tokenizer = _BatchTokenizer(templated=True)

        results = _generate_batch(
            model,
            tokenizer,
            ["first", "second"],
            max_tokens=5,
            temperature=0,
        )

        assert [token_count for _, token_count in results] == [2, 5]


    def test_batch_generation_stops_counting_at_eos(self):
        import torch

        from soup_cli.commands.infer import _generate_batch

        class EosBatchModel:
            device = torch.device("cpu")

            def generate(self, input_ids, attention_mask, **kwargs):
                generated = []
                for index, (row, mask) in enumerate(
                    zip(input_ids, attention_mask, strict=True)
                ):
                    signature = int(row[mask.bool()].sum())
                    if index == 0:
                        tokens = [
                            100 + signature,
                            99,
                            101 + signature,
                            102 + signature,
                            103 + signature,
                        ]
                    else:
                        tokens = [
                            110 + signature,
                            111 + signature,
                            112 + signature,
                            113 + signature,
                            114 + signature,
                        ]
                    generated.append(torch.tensor([tokens], dtype=torch.long))

                return torch.cat([input_ids, torch.cat(generated)], dim=1)

        model = EosBatchModel()
        tokenizer = _BatchTokenizer(templated=True)
        tokenizer.eos_token_id = 99

        results = _generate_batch(
            model,
            tokenizer,
            ["first", "second"],
            max_tokens=5,
            temperature=0,
        )

        assert [token_count for _, token_count in results] == [2, 5]


    def test_batch_generation_uses_attention_mask_for_padding(self):
        import torch

        from soup_cli.commands.infer import _generate_batch

        class MaskCheckingModel:
            device = torch.device("cpu")

            def generate(self, input_ids, attention_mask, **kwargs):
                assert torch.equal(
                    attention_mask == 0,
                    input_ids == 99,
                )
                generated = torch.tensor(
                    [[101], [102]],
                    dtype=torch.long,
                )
                return torch.cat([input_ids, generated], dim=1)

        model = MaskCheckingModel()
        tokenizer = _BatchTokenizer(templated=True)
        tokenizer.pad_token_id = 99

        results = _generate_batch(
            model,
            tokenizer,
            ["x", "xx"],
            max_tokens=1,
            temperature=0,
        )

        assert [response for response, _ in results] == [
            "response-101",
            "response-102",
        ]


    def test_batch_generation_restores_original_padding_side(self):
        from soup_cli.commands.infer import _generate_batch

        model = _DeterministicBatchModel()
        tokenizer = _BatchTokenizer(templated=True)

        _generate_batch(model, tokenizer, ["first", "second"], temperature=0)

        assert tokenizer.padding_sides[-1] == "left"
        assert tokenizer.padding_side == "right"

    def test_templated_and_untemplated_batch_encoding(self):
        from soup_cli.commands.infer import _generate_batch

        for templated, expected_special_tokens in ((True, False), (False, True)):
            model = _DeterministicBatchModel()
            tokenizer = _BatchTokenizer(templated=templated)

            _generate_batch(model, tokenizer, ["one", "two"], temperature=0)

            assert tokenizer.calls[-1]["add_special_tokens"] is expected_special_tokens

    def test_batch_generation_preserves_prompt_response_order(self):
        import torch

        from soup_cli.commands.infer import _generate_batch

        class OrderedBatchModel:
            device = torch.device("cpu")

            def generate(self, input_ids, attention_mask, **kwargs):
                generated = [
                    torch.tensor([[101 + index]], dtype=torch.long)
                    for index in range(input_ids.shape[0])
                ]
                return torch.cat([input_ids, torch.cat(generated)], dim=1)

        model = OrderedBatchModel()
        tokenizer = _BatchTokenizer(templated=True)

        results = _generate_batch(
            model,
            tokenizer,
            ["first", "second"],
            max_tokens=1,
            temperature=0,
        )

        assert [response for response, _ in results] == [
            "response-101",
            "response-102",
        ]


    def test_batch_size_one_uses_legacy_generate_path(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text(json.dumps({"prompt": "hello"}) + "\n")
        output_path = tmp_path / "output.jsonl"
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(
            "soup_cli.commands.infer._load_model",
            lambda *args, **kwargs: (object(), object()),
        )

        calls = []

        def fake_generate(model, tokenizer, messages, *, max_tokens, temperature):
            calls.append(messages)
            return "legacy-response", 3

        def fail_generate_batch(*args, **kwargs):
            raise AssertionError("_generate_batch must not be used for batch_size=1")

        monkeypatch.setattr("soup_cli.commands.infer._generate", fake_generate)
        monkeypatch.setattr(
            "soup_cli.commands.infer._generate_batch",
            fail_generate_batch,
        )

        result = CliRunner().invoke(
            app,
            [
                "infer",
                "--model", "fake/model",
                "--input", str(prompts_path),
                "--output", str(output_path),
                "--device", "cpu",
                "--batch-size", "1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert json.loads(output_path.read_text()) == {
            "prompt": "hello",
            "response": "legacy-response",
            "tokens_generated": 3,
        }


    def test_default_batch_size_uses_legacy_generate_path(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text(json.dumps({"prompt": "hello"}) + "\n")
        output_path = tmp_path / "output.jsonl"
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(
            "soup_cli.commands.infer._load_model",
            lambda *args, **kwargs: (object(), object()),
        )

        calls = []

        def fake_generate(model, tokenizer, messages, *, max_tokens, temperature):
            calls.append(messages)
            return "legacy-response", 3

        def fail_generate_batch(*args, **kwargs):
            raise AssertionError("_generate_batch must not be used for default batch_size=1")

        monkeypatch.setattr("soup_cli.commands.infer._generate", fake_generate)
        monkeypatch.setattr(
            "soup_cli.commands.infer._generate_batch",
            fail_generate_batch,
        )

        result = CliRunner().invoke(
            app,
            [
                "infer",
                "--model", "fake/model",
                "--input", str(prompts_path),
                "--output", str(output_path),
                "--device", "cpu",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert json.loads(output_path.read_text()) == {
            "prompt": "hello",
            "response": "legacy-response",
            "tokens_generated": 3,
        }


    def test_cli_batches_partial_chunk_in_input_order(self, tmp_path, monkeypatch):
        import torch
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_path = tmp_path / "prompts.jsonl"
        prompts = [f"prompt-{index}" for index in range(5)]
        prompts_path.write_text(
            "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
        )
        output_path = tmp_path / "output.jsonl"
        class UniqueBatchModel:
            device = torch.device("cpu")

            def __init__(self):
                self.next_token = 101
                self.generate_calls = []

            def generate(self, input_ids, attention_mask, **kwargs):
                self.generate_calls.append(input_ids.clone())
                rows = []
                for _ in range(input_ids.shape[0]):
                    rows.append(torch.tensor([[self.next_token]]))
                    self.next_token += 1
                return torch.cat([input_ids, torch.cat(rows)], dim=1)

        model = UniqueBatchModel()
        tokenizer = _BatchTokenizer(templated=True)
        monkeypatch.setattr(
            "soup_cli.commands.infer._load_model",
            lambda *args, **kwargs: (model, tokenizer),
        )
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(
            app,
            [
                "infer",
                "--model", "fake/model",
                "--input", str(prompts_path),
                "--output", str(output_path),
                "--device", "cpu",
                "--batch-size", "2",
                "--temperature", "0",
            ],
        )

        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert [(row["prompt"], row["response"]) for row in rows] == [
            (prompt, f"response-{101 + index}")
            for index, prompt in enumerate(prompts)
        ]
        assert len(model.generate_calls) == 3

    def test_batch_size_one_and_eight_have_identical_outputs(
        self, tmp_path, monkeypatch
    ):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts = [f"prompt-{index}" for index in range(5)]
        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text(
            "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts)
        )
        monkeypatch.chdir(tmp_path)
        outputs = []
        models = []
        for batch_size in (1, 8):
            model = _DeterministicBatchModel()
            tokenizer = _BatchTokenizer(templated=True)
            monkeypatch.setattr(
                "soup_cli.commands.infer._load_model",
                lambda *args, model=model, tokenizer=tokenizer, **kwargs: (
                    model,
                    tokenizer,
                ),
            )
            output_path = tmp_path / f"output-{batch_size}.jsonl"
            result = CliRunner().invoke(
                app,
                [
                    "infer",
                    "--model", "fake/model",
                    "--input", str(prompts_path),
                    "--output", str(output_path),
                    "--device", "cpu",
                    "--batch-size", str(batch_size),
                    "--temperature", "0",
                ],
            )
            assert result.exit_code == 0, result.output
            outputs.append([json.loads(line) for line in output_path.read_text().splitlines()])
            models.append(model)

        assert outputs[0] == outputs[1]
        assert len(models[0].generate_calls) == 5
        assert len(models[1].generate_calls) == 1


# ─── Import Tests ─────────────────────────────────────────────────────────


class TestInferImports:
    """Test that infer module is importable."""

    def test_import_infer_module(self):
        from soup_cli.commands.infer import infer
        assert infer is not None

    def test_import_read_prompts(self):
        from soup_cli.commands.infer import _read_prompts
        assert callable(_read_prompts)

    def test_import_load_model(self):
        from soup_cli.commands.infer import _load_model
        assert callable(_load_model)

    def test_import_generate(self):
        from soup_cli.commands.infer import _generate
        assert callable(_generate)
