"""Tests for local server data generation provider (--provider server)."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

# ─── Provider Validation Tests ───────────────────────────────────────────


class TestServerProviderValidation:
    """Test that 'server' is a valid provider for soup data generate."""

    def test_server_provider_routes_without_error(self):
        """'server' should route to _generate_server without error."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.commands.generate._generate_server",
            return_value=[],
        ):
            result = _generate_batch(
                prompt="test",
                count=1,
                fmt="alpaca",
                provider="server",
                model_name="m",
                api_key=None,
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )
        assert result == []

    def test_invalid_provider_rejected(self):
        """Invalid providers should cause an exit."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "test",
                "--provider",
                "invalid_provider",
                "--count",
                "1",
            ],
        )
        assert result.exit_code != 0

    def test_server_provider_in_help_text(self):
        """The help text should mention 'server' provider."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "generate", "--help"])
        assert "server" in result.output


# ─── _generate_server Function Tests ─────────────────────────────────────


class TestGenerateServer:
    """Test the _generate_server function."""

    def test_generate_server_calls_httpx(self):
        """_generate_server should call httpx.post with correct URL."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '[{"instruction": "test", "output": "response"}]'}}]
        }

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            _generate_server(
                prompt="Generate test data",
                count=1,
                fmt="alpaca",
                model_name="test-model",
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "localhost:8000" in call_url
        assert "/chat/completions" in call_url

    def test_generate_server_no_auth_header(self):
        """_generate_server should NOT include Authorization header."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="test-model",
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        call_headers = mock_post.call_args[1]["headers"]
        assert "Authorization" not in call_headers

    def test_generate_server_custom_api_base(self):
        """_generate_server should use custom api_base."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="test-model",
                api_base="http://localhost:11434/v1",
                temperature=0.8,
                seed_examples=[],
            )

        call_url = mock_post.call_args[0][0]
        assert "localhost:11434" in call_url

    def test_generate_server_appends_v1_if_missing(self):
        """If api_base doesn't end with /v1, it should be appended."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="test-model",
                api_base="http://localhost:8000",
                temperature=0.8,
                seed_examples=[],
            )

        call_url = mock_post.call_args[0][0]
        assert "/v1/chat/completions" in call_url

    def test_generate_server_error_response(self):
        """_generate_server should raise ValueError on non-200 response."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with mock_patch("httpx.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Server returned 500"):
                _generate_server(
                    prompt="test",
                    count=1,
                    fmt="alpaca",
                    model_name="test-model",
                    api_base=None,
                    temperature=0.8,
                    seed_examples=[],
                )

    def test_generate_server_parses_json_array(self):
        """_generate_server should parse JSON array from response content."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": ('[{"instruction": "What is 2+2?", "input": "", "output": "4"}]')
                    }
                }
            ]
        }

        with mock_patch("httpx.post", return_value=mock_response):
            result = _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="test-model",
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        assert len(result) == 1
        assert result[0]["instruction"] == "What is 2+2?"

    def test_generate_server_timeout(self):
        """_generate_server should use 300s timeout for local servers."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with mock_patch("httpx.post", return_value=mock_response) as mock_post:
            _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="test-model",
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        assert mock_post.call_args[1]["timeout"] == 300.0


# ─── _generate_batch Routing Tests ───────────────────────────────────────


class TestGenerateBatchRouting:
    """Test that _generate_batch routes to the correct provider."""

    def test_batch_routes_to_server(self):
        """provider='server' should route to _generate_server."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.commands.generate._generate_server",
            return_value=[{"instruction": "x", "output": "y"}],
        ) as mock_server:
            result = _generate_batch(
                prompt="test",
                count=1,
                fmt="alpaca",
                provider="server",
                model_name="test-model",
                api_key=None,
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        mock_server.assert_called_once()
        assert len(result) == 1

    def test_batch_routes_to_openai(self):
        """provider='openai' should route to _generate_openai."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.commands.generate._generate_openai",
            return_value=[],
        ) as mock_openai:
            _generate_batch(
                prompt="test",
                count=1,
                fmt="alpaca",
                provider="openai",
                model_name="gpt-4o-mini",
                api_key="sk-test",
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        mock_openai.assert_called_once()

    def test_batch_routes_to_local(self):
        """provider='local' should route to _generate_local."""
        from soup_cli.commands.generate import _generate_batch

        with mock_patch(
            "soup_cli.commands.generate._generate_local",
            return_value=[],
        ) as mock_local:
            _generate_batch(
                prompt="test",
                count=1,
                fmt="alpaca",
                provider="local",
                model_name="some-model",
                api_key=None,
                api_base=None,
                temperature=0.8,
                seed_examples=[],
            )

        mock_local.assert_called_once()


# ─── _parse_json_array Tests ─────────────────────────────────────────────


class TestParseJsonArray:
    """Test the JSON array parsing function used by all providers."""

    def test_parse_valid_json_array(self):
        """Should parse a clean JSON array."""
        from soup_cli.commands.generate import _parse_json_array

        result = _parse_json_array('[{"a": 1}, {"b": 2}]')
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_parse_markdown_code_fence(self):
        """Should strip markdown code fences."""
        from soup_cli.commands.generate import _parse_json_array

        content = '```json\n[{"instruction": "test", "output": "ok"}]\n```'
        result = _parse_json_array(content)
        assert len(result) == 1
        assert result[0]["instruction"] == "test"

    def test_parse_json_with_surrounding_text(self):
        """Should extract JSON array from surrounding text."""
        from soup_cli.commands.generate import _parse_json_array

        content = 'Here are the examples:\n[{"a": 1}]\nDone!'
        result = _parse_json_array(content)
        assert len(result) == 1

    def test_parse_ndjson_fallback(self):
        """Should fall back to line-by-line JSON parsing."""
        from soup_cli.commands.generate import _parse_json_array

        content = '{"a": 1}\n{"b": 2}\n{"c": 3}'
        result = _parse_json_array(content)
        assert len(result) == 3

    def test_parse_empty_array(self):
        """Should return empty list for empty array."""
        from soup_cli.commands.generate import _parse_json_array

        assert _parse_json_array("[]") == []

    def test_parse_invalid_json_returns_empty(self):
        """Should return empty list for completely invalid JSON."""
        from soup_cli.commands.generate import _parse_json_array

        assert _parse_json_array("not json at all") == []

    def test_parse_filters_non_dict_items(self):
        """Should filter out non-dict items from the array."""
        from soup_cli.commands.generate import _parse_json_array

        result = _parse_json_array('[{"a": 1}, "string", 42, {"b": 2}]')
        assert len(result) == 2

    def test_parse_code_fence_without_language(self):
        """Should strip code fences without language specifier."""
        from soup_cli.commands.generate import _parse_json_array

        content = '```\n[{"x": 1}]\n```'
        result = _parse_json_array(content)
        assert len(result) == 1


# ─── _validate_example Tests ─────────────────────────────────────────────


class TestValidateExample:
    """Test format validation for generated examples."""

    def test_validate_alpaca_valid(self):
        from soup_cli.commands.generate import _validate_example

        assert _validate_example({"instruction": "Q", "output": "A"}, "alpaca")

    def test_validate_alpaca_missing_output(self):
        from soup_cli.commands.generate import _validate_example

        assert not _validate_example({"instruction": "Q"}, "alpaca")

    def test_validate_sharegpt_valid(self):
        from soup_cli.commands.generate import _validate_example

        row = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ]
        }
        assert _validate_example(row, "sharegpt")

    def test_validate_sharegpt_too_few_turns(self):
        from soup_cli.commands.generate import _validate_example

        row = {"conversations": [{"from": "human", "value": "Hi"}]}
        assert not _validate_example(row, "sharegpt")

    def test_validate_chatml_valid(self):
        from soup_cli.commands.generate import _validate_example

        row = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        assert _validate_example(row, "chatml")

    def test_validate_chatml_empty_messages(self):
        from soup_cli.commands.generate import _validate_example

        assert not _validate_example({"messages": []}, "chatml")

    def test_validate_unknown_format(self):
        from soup_cli.commands.generate import _validate_example

        assert not _validate_example({"a": 1}, "unknown")


# ─── SSRF Validation Tests ───────────────────────────────────────────────


class TestServerSSRFValidation:
    """Test SSRF protection in _generate_server."""

    def test_server_blocks_non_http_scheme(self):
        """file:// scheme should be rejected."""
        from soup_cli.commands.generate import _generate_server

        with pytest.raises(ValueError, match="HTTP or HTTPS"):
            _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="m",
                api_base="file:///etc/passwd",
                temperature=0.8,
                seed_examples=[],
            )

    def test_server_blocks_remote_http(self):
        """Remote HTTP (non-localhost) should be rejected."""
        from soup_cli.commands.generate import _generate_server

        with pytest.raises(ValueError, match="HTTPS for remote"):
            _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="m",
                api_base="http://169.254.169.254/latest",
                temperature=0.8,
                seed_examples=[],
            )

    def test_server_allows_localhost_http(self):
        """HTTP to localhost should be allowed."""
        from soup_cli.commands.generate import _generate_server

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "[]"}}]}

        with mock_patch("httpx.post", return_value=mock_response):
            result = _generate_server(
                prompt="test",
                count=1,
                fmt="alpaca",
                model_name="m",
                api_base="http://127.0.0.1:8000",
                temperature=0.8,
                seed_examples=[],
            )
        assert result == []
