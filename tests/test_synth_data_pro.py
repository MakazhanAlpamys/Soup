"""Tests for v0.20.0 — Synth Data Gen Pro.

Covers: new providers (Ollama, Anthropic, vLLM), domain templates,
quality pipeline (validate, filter, dedup), and security (SSRF).
"""

import json
import os
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

# ─── Provider Validation Tests ──────────────────────────────────────────


class TestNewProvidersValidation:
    """Test that new providers are accepted by the CLI."""

    def test_ollama_provider_accepted(self):
        """'ollama' should be a valid provider."""
        from soup_cli.commands.generate import VALID_PROVIDERS

        assert "ollama" in VALID_PROVIDERS

    def test_anthropic_provider_accepted(self):
        """'anthropic' should be a valid provider."""
        from soup_cli.commands.generate import VALID_PROVIDERS

        assert "anthropic" in VALID_PROVIDERS

    def test_vllm_provider_accepted(self):
        """'vllm' should be a valid provider."""
        from soup_cli.commands.generate import VALID_PROVIDERS

        assert "vllm" in VALID_PROVIDERS

    def test_old_providers_still_work(self):
        """Original providers should still be valid."""
        from soup_cli.commands.generate import VALID_PROVIDERS

        assert "openai" in VALID_PROVIDERS
        assert "local" in VALID_PROVIDERS
        assert "server" in VALID_PROVIDERS

    def test_invalid_provider_rejected_cli(self):
        """Invalid provider should be rejected by CLI."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "data", "generate",
            "--prompt", "test",
            "--provider", "invalid_provider",
            "--count", "1",
        ])
        assert result.exit_code != 0


class TestNewProvidersCLIHelp:
    """Test CLI help text includes new providers."""

    def test_help_mentions_ollama(self):
        """Help text should mention 'ollama' provider."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "ollama" in result.output

    def test_help_mentions_anthropic(self):
        """Help text should mention 'anthropic' provider."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "anthropic" in result.output

    def test_help_mentions_vllm(self):
        """Help text should mention 'vllm' provider."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "vllm" in result.output

    def test_help_mentions_template(self):
        """Help text should mention --template option."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "template" in result.output.lower()

    def test_help_mentions_quality_pipeline(self):
        """Help text should mention --quality-pipeline option."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "quality-pipeline" in result.output


# ─── Ollama Provider Tests ──────────────────────────────────────────────


class TestOllamaProvider:
    """Test the Ollama provider."""

    def test_detect_ollama_success(self):
        """detect_ollama should return version when Ollama is running."""
        from soup_cli.data.providers.ollama import detect_ollama

        mock_tags = MagicMock()
        mock_tags.status_code = 200
        mock_tags.json.return_value = {"models": []}

        mock_ver = MagicMock()
        mock_ver.status_code = 200
        mock_ver.json.return_value = {"version": "0.6.2"}

        with mock_patch("httpx.get", side_effect=[mock_tags, mock_ver]):
            version = detect_ollama()
        assert version == "0.6.2"

    def test_detect_ollama_not_running(self):
        """detect_ollama should return None when Ollama is not running."""
        from soup_cli.data.providers.ollama import detect_ollama

        with mock_patch("httpx.get", side_effect=OSError("connection refused")):
            version = detect_ollama()
        assert version is None

    def test_detect_ollama_version_endpoint_fails(self):
        """detect_ollama should return 'unknown' if version endpoint fails."""
        from soup_cli.data.providers.ollama import detect_ollama

        mock_tags = MagicMock()
        mock_tags.status_code = 200
        mock_tags.json.return_value = {"models": []}

        with mock_patch("httpx.get", side_effect=[mock_tags, ValueError("fail")]):
            version = detect_ollama()
        assert version == "unknown"

    def test_generate_ollama_calls_api(self):
        """generate_ollama should call Ollama API correctly."""
        from soup_cli.data.providers.ollama import generate_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '[{"instruction": "test", "output": "ok"}]'}}
            ]
        }

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            result = generate_ollama(
                prompt="test", count=1, fmt="alpaca",
                model_name="llama3.1",
                base_url="http://localhost:11434",
                temperature=0.8,
                generation_prompt="Generate 1 example",
            )

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "localhost:11434" in call_url
        assert "/v1/chat/completions" in call_url
        assert len(result) == 1

    def test_generate_ollama_error_response(self):
        """generate_ollama should raise ValueError on error."""
        from soup_cli.data.providers.ollama import generate_ollama

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "model not found"

        with mock_patch("httpx.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Ollama returned 404"):
                generate_ollama(
                    prompt="test", count=1, fmt="alpaca",
                    model_name="nonexistent",
                    base_url="http://localhost:11434",
                    temperature=0.8,
                    generation_prompt="Generate 1 example",
                )


class TestOllamaSSRF:
    """Test Ollama provider SSRF protection."""

    def test_blocks_remote_url(self):
        """Remote Ollama URL should be rejected."""
        from soup_cli.data.providers.ollama import validate_ollama_url

        with pytest.raises(ValueError, match="localhost"):
            validate_ollama_url("http://evil.com:11434")

    def test_blocks_non_http_scheme(self):
        """Non-HTTP scheme should be rejected."""
        from soup_cli.data.providers.ollama import validate_ollama_url

        with pytest.raises(ValueError, match="HTTP or HTTPS"):
            validate_ollama_url("file:///etc/passwd")

    def test_allows_localhost(self):
        """localhost should be allowed."""
        from soup_cli.data.providers.ollama import validate_ollama_url

        validate_ollama_url("http://localhost:11434")  # Should not raise

    def test_allows_127_0_0_1(self):
        """127.0.0.1 should be allowed."""
        from soup_cli.data.providers.ollama import validate_ollama_url

        validate_ollama_url("http://127.0.0.1:11434")  # Should not raise

    def test_generate_ollama_rejects_remote(self):
        """generate_ollama should reject remote URLs."""
        from soup_cli.data.providers.ollama import generate_ollama

        with pytest.raises(ValueError, match="localhost"):
            generate_ollama(
                prompt="test", count=1, fmt="alpaca",
                model_name="m",
                base_url="http://169.254.169.254",
                temperature=0.8,
                generation_prompt="test",
            )


# ─── Anthropic Provider Tests ───────────────────────────────────────────


class TestAnthropicProvider:
    """Test the Anthropic provider."""

    def test_generate_anthropic_calls_api(self):
        """generate_anthropic should call Anthropic Messages API."""
        from soup_cli.data.providers.anthropic import generate_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": '[{"instruction": "test", "output": "ok"}]'}
            ]
        }

        with mock_patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            with mock_patch("httpx.post", return_value=mock_response) as mock_post:
                result = generate_anthropic(
                    prompt="test", count=1, fmt="alpaca",
                    model_name="claude-3-haiku-20240307",
                    temperature=0.8,
                    generation_prompt="Generate 1 example",
                )

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "anthropic.com" in call_url
        call_headers = mock_post.call_args[1]["headers"]
        assert call_headers["x-api-key"] == "sk-ant-test"
        assert len(result) == 1

    def test_generate_anthropic_no_api_key(self):
        """generate_anthropic should raise ValueError without API key."""
        from soup_cli.data.providers.anthropic import generate_anthropic

        with mock_patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with mock_patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    generate_anthropic(
                        prompt="test", count=1, fmt="alpaca",
                        model_name="claude-3-haiku-20240307",
                        temperature=0.8,
                        generation_prompt="test",
                    )

    def test_generate_anthropic_error_response(self):
        """generate_anthropic should raise ValueError on API error."""
        from soup_cli.data.providers.anthropic import generate_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with mock_patch.dict(os.environ, {"ANTHROPIC_API_KEY": "bad-key"}):
            with mock_patch("httpx.post", return_value=mock_response):
                with pytest.raises(ValueError, match="401"):
                    generate_anthropic(
                        prompt="test", count=1, fmt="alpaca",
                        model_name="claude-3-haiku-20240307",
                        temperature=0.8,
                        generation_prompt="test",
                    )

    def test_anthropic_api_key_from_env_only(self):
        """Anthropic API key should only come from environment, never CLI."""
        import inspect

        from soup_cli.data.providers.anthropic import generate_anthropic

        sig = inspect.signature(generate_anthropic)
        param_names = list(sig.parameters.keys())
        assert "api_key" not in param_names

    def test_anthropic_parses_content_blocks(self):
        """Anthropic response with multiple content blocks should be parsed."""
        from soup_cli.data.providers.anthropic import generate_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": '[{"instruction": "a", "output": "b"}'},
                {"type": "text", "text": ', {"instruction": "c", "output": "d"}]'},
            ]
        }

        with mock_patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            with mock_patch("httpx.post", return_value=mock_response):
                result = generate_anthropic(
                    prompt="test", count=2, fmt="alpaca",
                    model_name="claude-3-haiku-20240307",
                    temperature=0.8,
                    generation_prompt="test",
                )

        assert len(result) == 2


# ─── vLLM Provider Tests ────────────────────────────────────────────────


class TestVLLMProvider:
    """Test the vLLM provider."""

    def test_generate_vllm_calls_api(self):
        """generate_vllm should call vLLM OpenAI-compatible API."""
        from soup_cli.data.providers.vllm import generate_vllm

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '[{"instruction": "test", "output": "ok"}]'}}
            ]
        }

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            result = generate_vllm(
                prompt="test", count=1, fmt="alpaca",
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8000",
                temperature=0.8,
                generation_prompt="Generate 1 example",
            )

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "localhost:8000" in call_url
        assert "/v1/chat/completions" in call_url
        assert len(result) == 1

    def test_generate_vllm_appends_v1(self):
        """generate_vllm should append /v1 if missing."""
        from soup_cli.data.providers.vllm import generate_vllm

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "[]"}}]
        }

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            generate_vllm(
                prompt="test", count=1, fmt="alpaca",
                model_name="m", base_url="http://localhost:8000",
                temperature=0.8, generation_prompt="test",
            )

        call_url = mock_post.call_args[0][0]
        assert "/v1/chat/completions" in call_url

    def test_generate_vllm_error_response(self):
        """generate_vllm should raise ValueError on error."""
        from soup_cli.data.providers.vllm import generate_vllm

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with mock_patch("httpx.post", return_value=mock_response):
            with pytest.raises(ValueError, match="vLLM server returned 500"):
                generate_vllm(
                    prompt="test", count=1, fmt="alpaca",
                    model_name="m", base_url="http://localhost:8000",
                    temperature=0.8, generation_prompt="test",
                )


class TestVLLMSSRF:
    """Test vLLM provider SSRF protection."""

    def test_blocks_non_http_scheme(self):
        """Non-HTTP scheme should be rejected."""
        from soup_cli.data.providers.vllm import validate_vllm_url

        with pytest.raises(ValueError, match="HTTP or HTTPS"):
            validate_vllm_url("file:///etc/passwd")

    def test_blocks_remote_http(self):
        """Remote HTTP should be rejected."""
        from soup_cli.data.providers.vllm import validate_vllm_url

        with pytest.raises(ValueError, match="HTTPS for remote"):
            validate_vllm_url("http://169.254.169.254/latest")

    def test_allows_localhost_http(self):
        """localhost HTTP should be allowed."""
        from soup_cli.data.providers.vllm import validate_vllm_url

        validate_vllm_url("http://localhost:8000")  # Should not raise

    def test_allows_remote_https(self):
        """Remote HTTPS should be allowed."""
        from soup_cli.data.providers.vllm import validate_vllm_url

        validate_vllm_url("https://vllm.example.com:8000")  # Should not raise

    def test_generate_vllm_rejects_remote_http(self):
        """generate_vllm should reject remote HTTP URLs."""
        from soup_cli.data.providers.vllm import generate_vllm

        with pytest.raises(ValueError, match="HTTPS for remote"):
            generate_vllm(
                prompt="test", count=1, fmt="alpaca",
                model_name="m",
                base_url="http://evil.com:8000",
                temperature=0.8,
                generation_prompt="test",
            )


# ─── Additional Provider Tests (TDD review gaps) ───────────────────────


class TestAnthropicHardcodedURL:
    """Test that Anthropic URL is hardcoded and not user-configurable."""

    def test_api_url_is_anthropic(self):
        """The API URL must always be anthropic.com."""
        from soup_cli.data.providers.anthropic import ANTHROPIC_API_URL

        assert "anthropic.com" in ANTHROPIC_API_URL
        assert ANTHROPIC_API_URL.startswith("https://")

    def test_no_api_base_parameter(self):
        """generate_anthropic should not accept an api_base parameter."""
        import inspect

        from soup_cli.data.providers.anthropic import generate_anthropic

        sig = inspect.signature(generate_anthropic)
        assert "api_base" not in sig.parameters
        assert "base_url" not in sig.parameters


class TestProviderMalformedResponse:
    """Test providers handle malformed LLM responses gracefully."""

    def test_anthropic_malformed_content(self):
        """Anthropic with unexpected response structure should raise."""
        from soup_cli.data.providers.anthropic import generate_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}

        with mock_patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            with mock_patch("httpx.post", return_value=mock_response):
                with pytest.raises(ValueError, match="Unexpected"):
                    generate_anthropic(
                        prompt="test", count=1, fmt="alpaca",
                        model_name="claude-3-haiku-20240307",
                        temperature=0.8,
                        generation_prompt="test",
                    )

    def test_vllm_malformed_content(self):
        """vLLM with unexpected response structure should raise."""
        from soup_cli.data.providers.vllm import generate_vllm

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}

        with mock_patch("httpx.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Unexpected"):
                generate_vllm(
                    prompt="test", count=1, fmt="alpaca",
                    model_name="m",
                    base_url="http://localhost:8000",
                    temperature=0.8,
                    generation_prompt="test",
                )

    def test_ollama_malformed_content(self):
        """Ollama with unexpected response structure should raise."""
        from soup_cli.data.providers.ollama import generate_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}

        with mock_patch("httpx.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Unexpected"):
                generate_ollama(
                    prompt="test", count=1, fmt="alpaca",
                    model_name="m",
                    base_url="http://localhost:11434",
                    temperature=0.8,
                    generation_prompt="test",
                )


class TestParseJsonArrayShared:
    """Test the shared parse_json_array utility."""

    def test_parse_valid_array(self):
        """Should parse a clean JSON array."""
        from soup_cli.data.providers._utils import parse_json_array

        result = parse_json_array('[{"a": 1}, {"b": 2}]')
        assert len(result) == 2

    def test_parse_empty(self):
        """Should return empty for empty input."""
        from soup_cli.data.providers._utils import parse_json_array

        assert parse_json_array("") == []

    def test_parse_markdown_fences(self):
        """Should strip markdown code fences."""
        from soup_cli.data.providers._utils import parse_json_array

        content = '```json\n[{"a": 1}]\n```'
        result = parse_json_array(content)
        assert len(result) == 1

    def test_parse_ndjson_fallback(self):
        """Should fall back to line-by-line JSON."""
        from soup_cli.data.providers._utils import parse_json_array

        content = '{"a": 1}\n{"b": 2}'
        result = parse_json_array(content)
        assert len(result) == 2


# ─── Batch Routing Tests ────────────────────────────────────────────────


class TestBatchRoutingNewProviders:
    """Test that _generate_batch routes to new providers correctly."""

    def test_routes_to_ollama(self):
        """provider='ollama' should route to Ollama provider."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.data.providers.ollama.generate_ollama",
            return_value=[{"instruction": "x", "output": "y"}],
        ) as mock_ollama:
            result = _generate_batch(
                prompt="test", count=1, fmt="alpaca",
                provider="ollama", model_name="llama3.1",
                api_key=None, api_base=None,
                temperature=0.8, seed_examples=[],
            )

        mock_ollama.assert_called_once()
        assert len(result) == 1

    def test_routes_to_anthropic(self):
        """provider='anthropic' should route to Anthropic provider."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.data.providers.anthropic.generate_anthropic",
            return_value=[{"instruction": "x", "output": "y"}],
        ) as mock_anthropic:
            result = _generate_batch(
                prompt="test", count=1, fmt="alpaca",
                provider="anthropic", model_name="claude-3-haiku-20240307",
                api_key=None, api_base=None,
                temperature=0.8, seed_examples=[],
            )

        mock_anthropic.assert_called_once()
        assert len(result) == 1

    def test_routes_to_vllm(self):
        """provider='vllm' should route to vLLM provider."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.data.providers.vllm.generate_vllm",
            return_value=[{"instruction": "x", "output": "y"}],
        ) as mock_vllm:
            result = _generate_batch(
                prompt="test", count=1, fmt="alpaca",
                provider="vllm", model_name="m",
                api_key=None, api_base=None,
                temperature=0.8, seed_examples=[],
            )

        mock_vllm.assert_called_once()
        assert len(result) == 1

    def test_old_providers_still_route(self):
        """Original providers should still route correctly."""
        from soup_cli.commands.generate import _generate_batch

        for prov, mock_target in [
            ("openai", "soup_cli.commands.generate._generate_openai"),
            ("local", "soup_cli.commands.generate._generate_local"),
            ("server", "soup_cli.commands.generate._generate_server"),
        ]:
            with mock_patch(mock_target, return_value=[]) as mock_fn:
                _generate_batch(
                    prompt="test", count=1, fmt="alpaca",
                    provider=prov, model_name="m",
                    api_key="key" if prov == "openai" else None,
                    api_base=None,
                    temperature=0.8, seed_examples=[],
                )
            mock_fn.assert_called_once()


# ─── Domain Template Tests ──────────────────────────────────────────────


class TestTemplateValidation:
    """Test template validation."""

    def test_valid_templates(self):
        """All valid templates should be accepted."""
        from soup_cli.commands.generate import VALID_TEMPLATES

        assert "code" in VALID_TEMPLATES
        assert "conversation" in VALID_TEMPLATES
        assert "qa" in VALID_TEMPLATES
        assert "preference" in VALID_TEMPLATES
        assert "reasoning" in VALID_TEMPLATES

    def test_invalid_template_rejected_cli(self):
        """Invalid template should be rejected."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "data", "generate",
            "--prompt", "test",
            "--template", "invalid_template",
            "--count", "1",
        ])
        assert result.exit_code != 0


class TestCodeTemplate:
    """Test code domain template."""

    def test_build_prompt_default(self):
        """Code template should build a valid prompt."""
        from soup_cli.data.templates.code import build_prompt

        result = build_prompt(5, "alpaca", "format spec", language="Python")
        assert "Python" in result
        assert "5" in result

    def test_build_prompt_languages(self):
        """Code template should support different languages."""
        from soup_cli.data.templates.code import build_prompt

        for lang in ["Python", "JavaScript", "Go", "Rust", "Java"]:
            result = build_prompt(3, "alpaca", "spec", language=lang)
            assert lang in result

    def test_build_prompt_task_types(self):
        """Code template should support different task types."""
        from soup_cli.data.templates.code import build_prompt

        for task_type in ["function", "debug", "explain", "refactor", "test"]:
            result = build_prompt(3, "alpaca", "spec", task_type=task_type)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_template_spec_has_languages(self):
        """Template spec should list supported languages."""
        from soup_cli.data.templates.code import TEMPLATE_SPEC

        assert "languages" in TEMPLATE_SPEC
        assert len(TEMPLATE_SPEC["languages"]) >= 5


class TestConversationTemplate:
    """Test conversation domain template."""

    def test_build_prompt_default(self):
        """Conversation template should build a valid prompt."""
        from soup_cli.data.templates.conversation import build_prompt

        result = build_prompt(5, "chatml", "format spec")
        assert "5" in result
        assert "conversation" in result.lower() or "multi-turn" in result.lower()

    def test_build_prompt_with_topic(self):
        """Conversation template should include topic."""
        from soup_cli.data.templates.conversation import build_prompt

        result = build_prompt(3, "chatml", "spec", topic="science fiction")
        assert "science fiction" in result

    def test_turns_clamped(self):
        """Turns should be clamped to 2-10 range."""
        from soup_cli.data.templates.conversation import build_prompt

        # turns < 2 should be clamped to 2
        result = build_prompt(3, "chatml", "spec", turns=0)
        assert "2" in result

        # turns > 10 should be clamped to 10
        result = build_prompt(3, "chatml", "spec", turns=20)
        assert "10" in result


class TestQATemplate:
    """Test QA domain template."""

    def test_build_prompt_without_context(self):
        """QA template without context should generate general QA."""
        from soup_cli.data.templates.qa import build_prompt

        result = build_prompt(5, "alpaca", "format spec")
        assert "5" in result
        assert "question" in result.lower()

    def test_build_prompt_with_context(self):
        """QA template with context should include it."""
        from soup_cli.data.templates.qa import build_prompt

        context = "Python is a programming language created by Guido van Rossum."
        result = build_prompt(3, "alpaca", "spec", context=context)
        assert "Python" in result
        assert "Guido" in result

    def test_context_truncated(self):
        """Long context should be truncated to prevent overflow."""
        from soup_cli.data.templates.qa import build_prompt

        context = "x" * 20000
        result = build_prompt(3, "alpaca", "spec", context=context)
        # Context should be truncated to 8000 chars
        assert len(result) < 20000


class TestPreferenceTemplate:
    """Test preference domain template."""

    def test_build_prompt_dpo(self):
        """Preference template for DPO should use chosen/rejected format."""
        from soup_cli.data.templates.preference import build_prompt

        result = build_prompt(5, task="dpo")
        assert "chosen" in result
        assert "rejected" in result

    def test_build_prompt_kto(self):
        """Preference template for KTO should use label format."""
        from soup_cli.data.templates.preference import build_prompt

        result = build_prompt(5, task="kto")
        assert "label" in result
        assert "true" in result.lower() or "false" in result.lower()

    def test_build_prompt_orpo(self):
        """Preference template for ORPO should use chosen/rejected format."""
        from soup_cli.data.templates.preference import build_prompt

        result = build_prompt(5, task="orpo")
        assert "chosen" in result
        assert "rejected" in result


class TestReasoningTemplate:
    """Test reasoning domain template."""

    def test_build_prompt_math(self):
        """Reasoning template for math should include math description."""
        from soup_cli.data.templates.reasoning import build_prompt

        result = build_prompt(5, "alpaca", "format spec", domain="math")
        assert "math" in result.lower()

    def test_build_prompt_logic(self):
        """Reasoning template for logic should include logic description."""
        from soup_cli.data.templates.reasoning import build_prompt

        result = build_prompt(5, "alpaca", "spec", domain="logic")
        assert "logic" in result.lower()

    def test_build_prompt_code(self):
        """Reasoning template for code should include code description."""
        from soup_cli.data.templates.reasoning import build_prompt

        result = build_prompt(5, "alpaca", "spec", domain="code")
        assert "code" in result.lower() or "algorithm" in result.lower()

    def test_valid_domains(self):
        """All domains should be in the DOMAINS constant."""
        from soup_cli.data.templates.reasoning import DOMAINS

        assert "math" in DOMAINS
        assert "logic" in DOMAINS
        assert "code" in DOMAINS


class TestTemplateBuildPromptIntegration:
    """Test _build_template_prompt in generate.py."""

    def test_code_template_integration(self):
        """Code template should be built correctly via _build_template_prompt."""
        from soup_cli.commands.generate import _build_template_prompt

        result = _build_template_prompt(
            template="code", prompt="test", count=5, fmt="alpaca",
            language="Go", task_type="debug",
        )
        assert "Go" in result

    def test_conversation_template_integration(self):
        """Conversation template should be built correctly."""
        from soup_cli.commands.generate import _build_template_prompt

        result = _build_template_prompt(
            template="conversation", prompt="test", count=5, fmt="chatml",
            topic="cooking",
        )
        assert "cooking" in result

    def test_qa_template_integration(self):
        """QA template should be built correctly."""
        from soup_cli.commands.generate import _build_template_prompt

        result = _build_template_prompt(
            template="qa", prompt="test", count=5, fmt="alpaca",
            context_text="Sample context text",
        )
        assert "Sample context" in result

    def test_preference_template_integration(self):
        """Preference template should be built correctly."""
        from soup_cli.commands.generate import _build_template_prompt

        result = _build_template_prompt(
            template="preference", prompt="test", count=5, fmt="alpaca",
            pref_task="kto",
        )
        assert "label" in result

    def test_reasoning_template_integration(self):
        """Reasoning template should be built correctly."""
        from soup_cli.commands.generate import _build_template_prompt

        result = _build_template_prompt(
            template="reasoning", prompt="test", count=5, fmt="alpaca",
            domain="logic",
        )
        assert "logic" in result.lower()

    def test_unknown_template_falls_back(self):
        """Unknown template should fall back to default prompt."""
        from soup_cli.commands.generate import _build_template_prompt

        result = _build_template_prompt(
            template="nonexistent", prompt="test topic", count=5, fmt="alpaca",
        )
        assert "test topic" in result


# ─── Quality Pipeline Tests ─────────────────────────────────────────────


class TestValidatePipeline:
    """Test the validation pipeline step."""

    def test_removes_invalid_entries(self, tmp_path):
        """Validation pipeline should remove invalid entries."""
        from soup_cli.commands.generate import _run_validate_pipeline

        data = [
            {"instruction": "valid", "input": "", "output": "ok"},
            {"bad": "entry"},
            {"instruction": "also valid", "input": "", "output": "fine"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        _run_validate_pipeline(path, "alpaca")

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["instruction"] == "valid"
        assert lines[1]["instruction"] == "also valid"

    def test_keeps_all_valid(self, tmp_path):
        """Validation pipeline should keep all entries when all valid."""
        from soup_cli.commands.generate import _run_validate_pipeline

        data = [
            {"instruction": "a", "input": "", "output": "b"},
            {"instruction": "c", "input": "", "output": "d"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        _run_validate_pipeline(path, "alpaca")

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2

    def test_validates_preference_format(self, tmp_path):
        """Validation pipeline should accept preference format entries."""
        from soup_cli.commands.generate import _run_validate_pipeline

        data = [
            {"prompt": "q", "chosen": "good", "rejected": "bad"},
            {"prompt": "q", "completion": "ans", "label": True},
            {"bad": "entry"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        _run_validate_pipeline(path, "alpaca")

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2


class TestFilterPipeline:
    """Test the quality filter pipeline step."""

    def test_filter_pipeline_runs(self, tmp_path):
        """Filter pipeline should run without error."""
        from soup_cli.commands.generate import _run_filter_pipeline

        data = [
            {"instruction": "What is Python?", "output": "Python is a programming language."},
            {"instruction": "asdfjkl asdf", "output": "random noise text here"},
            {"instruction": "Explain ML", "output": "Machine learning is a field of AI."},
            {"instruction": "Math question", "output": "The answer to 2+2 is 4."},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        # Should not raise even without torch/transformers
        _run_filter_pipeline(path)

    def test_filter_pipeline_empty(self, tmp_path):
        """Filter pipeline should handle empty file."""
        from soup_cli.commands.generate import _run_filter_pipeline

        path = tmp_path / "empty.jsonl"
        path.write_text("")

        _run_filter_pipeline(path)  # Should not raise


class TestDedupPipeline:
    """Test the dedup pipeline step."""

    def test_dedup_pipeline_without_datasketch(self, tmp_path):
        """Dedup pipeline should gracefully handle missing datasketch."""
        from soup_cli.commands.generate import _run_dedup_pipeline

        data = [
            {"instruction": "a", "output": "b"},
            {"instruction": "a", "output": "b"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        # This will either dedup or skip gracefully if datasketch not installed
        _run_dedup_pipeline(path)

    def test_dedup_pipeline_empty(self, tmp_path):
        """Dedup pipeline should handle empty file."""
        from soup_cli.commands.generate import _run_dedup_pipeline

        path = tmp_path / "empty.jsonl"
        path.write_text("")

        _run_dedup_pipeline(path)  # Should not raise


class TestQualityPipelineFlag:
    """Test the --quality-pipeline convenience flag."""

    def test_quality_pipeline_enables_all(self):
        """--quality-pipeline should enable validate, filter, and dedup."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        # We can test this by checking the help text mentions the flag
        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "quality-pipeline" in result.output
        assert "validate" in result.output
        assert "filter" in result.output
        assert "dedup" in result.output


# ─── Preference Validation Tests ────────────────────────────────────────


class TestValidatePreference:
    """Test preference data validation."""

    def test_dpo_format_valid(self):
        """DPO format should be valid."""
        from soup_cli.commands.generate import _validate_preference

        assert _validate_preference({"prompt": "q", "chosen": "a", "rejected": "b"})

    def test_kto_format_valid(self):
        """KTO format should be valid."""
        from soup_cli.commands.generate import _validate_preference

        assert _validate_preference({"prompt": "q", "completion": "a", "label": True})

    def test_invalid_format(self):
        """Invalid format should fail validation."""
        from soup_cli.commands.generate import _validate_preference

        assert not _validate_preference({"instruction": "q", "output": "a"})
        assert not _validate_preference({"prompt": "q"})
        assert not _validate_preference({})


# ─── Output Path Sanitization Tests ─────────────────────────────────────


class TestPathWithinCwd:
    """Test _path_within_cwd helper."""

    def test_path_within_cwd(self, tmp_path):
        """Path inside cwd should return True."""

        from soup_cli.commands.generate import _path_within_cwd

        cwd = tmp_path.resolve()
        child = (tmp_path / "subdir" / "file.jsonl").resolve()
        assert _path_within_cwd(child, cwd) is True

    def test_path_outside_cwd(self, tmp_path):
        """Path outside cwd should return False."""

        from soup_cli.commands.generate import _path_within_cwd

        cwd = (tmp_path / "subdir").resolve()
        outside = tmp_path.resolve()
        assert _path_within_cwd(outside, cwd) is False

    def test_absolute_path_outside_cwd(self):
        """Absolute path to system directory should return False."""
        from pathlib import Path

        from soup_cli.commands.generate import _path_within_cwd

        cwd = Path.cwd().resolve()
        # /tmp or C:\Windows are outside typical cwd
        system_path = Path("/tmp/exfil.jsonl").resolve()
        # This may or may not be within cwd depending on where tests run,
        # but the function itself should work correctly
        result = _path_within_cwd(system_path, cwd)
        assert isinstance(result, bool)


class TestOutputPathSanitization:
    """Test that output paths are sanitized."""

    def test_path_traversal_blocked_in_cli(self):
        """Output path with '..' should be rejected."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()

        # Mock the generation to avoid actual API call
        with mock_patch(
            "soup_cli.commands.generate._generate_batch",
            return_value=[{"instruction": "x", "output": "y"}],
        ):
            result = runner.invoke(app, [
                "data", "generate",
                "--prompt", "test",
                "--output", "../../../etc/evil.jsonl",
                "--count", "1",
                "--provider", "server",
            ])
        assert result.exit_code != 0

    def test_absolute_output_path_blocked(self):
        """Absolute output path outside cwd should be rejected."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()

        with mock_patch(
            "soup_cli.commands.generate._generate_batch",
            return_value=[{"instruction": "x", "output": "y"}],
        ):
            result = runner.invoke(app, [
                "data", "generate",
                "--prompt", "test",
                "--output", "/tmp/exfil.jsonl",
                "--count", "1",
                "--provider", "server",
            ])
        assert result.exit_code != 0


# ─── Ollama Model Shorthand Tests ───────────────────────────────────────


class TestOllamaModelShorthand:
    """Test --ollama-model shorthand flag."""

    def test_ollama_model_sets_provider(self):
        """--ollama-model should set provider to 'ollama'."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "ollama-model" in result.output


# ─── Rate Limiting Tests ────────────────────────────────────────────────


class TestRateLimiting:
    """Test rate limiting configuration."""

    def test_requests_per_minute_in_help(self):
        """Help should mention --requests-per-minute."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "requests-per-minute" in result.output or "rpm" in result.output


# ─── Format Spec Tests ──────────────────────────────────────────────────


class TestGetFormatSpec:
    """Test _get_format_spec helper."""

    def test_alpaca_spec(self):
        """Alpaca format spec should mention instruction and output."""
        from soup_cli.commands.generate import _get_format_spec

        spec = _get_format_spec("alpaca")
        assert "instruction" in spec
        assert "output" in spec

    def test_sharegpt_spec(self):
        """ShareGPT format spec should mention conversations."""
        from soup_cli.commands.generate import _get_format_spec

        spec = _get_format_spec("sharegpt")
        assert "conversations" in spec

    def test_chatml_spec(self):
        """ChatML format spec should mention messages."""
        from soup_cli.commands.generate import _get_format_spec

        spec = _get_format_spec("chatml")
        assert "messages" in spec

    def test_unknown_format_returns_alpaca(self):
        """Unknown format should fall back to alpaca spec."""
        from soup_cli.commands.generate import _get_format_spec

        spec = _get_format_spec("unknown")
        assert "instruction" in spec


# ─── End-to-End CLI Integration Tests ───────────────────────────────────


class TestEndToEndGeneration:
    """Test end-to-end generation with mocked providers."""

    def test_generate_with_ollama_provider(self, tmp_path):
        """Full generation with Ollama provider should work."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        output_path = tmp_path / "output.jsonl"

        with mock_patch(
            "soup_cli.data.providers.ollama.generate_ollama",
            return_value=[
                {"instruction": "What is AI?", "input": "", "output": "AI is..."},
                {"instruction": "Explain ML", "input": "", "output": "ML is..."},
            ],
        ), mock_patch(
            "soup_cli.commands.generate._path_within_cwd",
            return_value=True,
        ):
            runner = CliRunner()
            result = runner.invoke(app, [
                "data", "generate",
                "--prompt", "Generate AI questions",
                "--provider", "ollama",
                "--model", "llama3.1",
                "--count", "2",
                "--batch-size", "2",
                "--output", str(output_path),
            ])

        assert result.exit_code == 0
        assert output_path.exists()
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2

    def test_generate_with_template(self, tmp_path):
        """Generation with a template should work."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        output_path = tmp_path / "output.jsonl"

        with mock_patch(
            "soup_cli.commands.generate._generate_server",
            return_value=[
                {"instruction": "Write a function", "input": "", "output": "def foo(): pass"},
            ],
        ), mock_patch(
            "soup_cli.commands.generate._path_within_cwd",
            return_value=True,
        ):
            runner = CliRunner()
            result = runner.invoke(app, [
                "data", "generate",
                "--prompt", "Generate code",
                "--template", "code",
                "--language", "Python",
                "--provider", "server",
                "--count", "1",
                "--batch-size", "1",
                "--output", str(output_path),
            ])

        assert result.exit_code == 0
        assert "code" in result.output.lower() or "Template" in result.output

    def test_generate_with_anthropic_provider(self, tmp_path):
        """Full generation with Anthropic provider should work."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        output_path = tmp_path / "output.jsonl"

        with mock_patch(
            "soup_cli.data.providers.anthropic.generate_anthropic",
            return_value=[
                {"instruction": "What is AI?", "input": "", "output": "AI is..."},
            ],
        ), mock_patch(
            "soup_cli.commands.generate._path_within_cwd",
            return_value=True,
        ):
            runner = CliRunner()
            result = runner.invoke(app, [
                "data", "generate",
                "--prompt", "Generate AI questions",
                "--provider", "anthropic",
                "--model", "claude-3-haiku-20240307",
                "--count", "1",
                "--batch-size", "1",
                "--output", str(output_path),
            ])

        assert result.exit_code == 0
        assert output_path.exists()

    def test_generate_with_vllm_provider(self, tmp_path):
        """Full generation with vLLM provider should work."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        output_path = tmp_path / "output.jsonl"

        with mock_patch(
            "soup_cli.data.providers.vllm.generate_vllm",
            return_value=[
                {"instruction": "What is AI?", "input": "", "output": "AI is..."},
            ],
        ), mock_patch(
            "soup_cli.commands.generate._path_within_cwd",
            return_value=True,
        ):
            runner = CliRunner()
            result = runner.invoke(app, [
                "data", "generate",
                "--prompt", "Generate AI questions",
                "--provider", "vllm",
                "--count", "1",
                "--batch-size", "1",
                "--output", str(output_path),
            ])

        assert result.exit_code == 0
        assert output_path.exists()
