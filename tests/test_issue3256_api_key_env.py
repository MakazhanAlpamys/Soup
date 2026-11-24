"""`soup data generate --api-key-env` reads the key from a named variable (#3256).

The accepted scope is the generic gap from the maintainer's decision on #3256:
no vendor-named provider, just a way to name the environment variable that
holds the credential for a custom --api-base, so a second endpoint does not
have to reuse OPENAI_API_KEY.
"""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

CUSTOM_BASE = "https://api.example.com/v1"
CUSTOM_VAR = "CUSTOM_API_KEY"


class _FakeResponse:
    status_code = 200

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([{"instruction": "Q?", "output": "A."}])
                    }
                }
            ]
        }


def _run_generate(tmp_path, monkeypatch, *extra_args):
    """Invoke `soup data generate` with a stubbed HTTP client, capturing calls."""
    calls = []

    def fake_post_capped(url, headers=None, json=None, **kwargs):
        calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        return _FakeResponse()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.utils.http_response.post_capped", fake_post_capped)
    result = runner.invoke(
        app,
        [
            "data",
            "generate",
            "--prompt",
            "Test prompt",
            "--count",
            "1",
            "--batch-size",
            "1",
            "--provider",
            "openai",
            "--model",
            "test-model",
            "--api-base",
            CUSTOM_BASE,
            "--requests-per-minute",
            "6000",
            "--output",
            str(tmp_path / "output.jsonl"),
            *extra_args,
        ],
    )
    return result, calls


def test_named_variable_is_read(tmp_path, monkeypatch):
    """The named variable supplies the credential for a custom --api-base."""
    monkeypatch.setenv(CUSTOM_VAR, "sk-custom")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result, calls = _run_generate(
        tmp_path, monkeypatch, "--api-key-env", CUSTOM_VAR
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(calls) == 1
    assert calls[0]["url"] == f"{CUSTOM_BASE}/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-custom"
    assert calls[0]["json"]["model"] == "test-model"


def test_named_variable_beats_openai_key(tmp_path, monkeypatch):
    """With --api-key-env given, OPENAI_API_KEY is not consulted."""
    monkeypatch.setenv(CUSTOM_VAR, "sk-custom")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    result, calls = _run_generate(
        tmp_path, monkeypatch, "--api-key-env", CUSTOM_VAR
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-custom"


def test_explicit_api_key_beats_named_variable(tmp_path, monkeypatch):
    """An explicit --api-key still wins over --api-key-env."""
    monkeypatch.setenv(CUSTOM_VAR, "sk-custom")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result, calls = _run_generate(
        tmp_path,
        monkeypatch,
        "--api-key",
        "sk-explicit",
        "--api-key-env",
        CUSTOM_VAR,
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-explicit"


def test_without_option_still_reads_openai_key(tmp_path, monkeypatch):
    """Without --api-key-env, today's behavior is unchanged."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    result, calls = _run_generate(tmp_path, monkeypatch)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-openai"


def test_missing_named_variable_names_it(tmp_path, monkeypatch):
    """A missing named variable gives the same kind of message as today's key."""
    monkeypatch.delenv(CUSTOM_VAR, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result, _ = _run_generate(tmp_path, monkeypatch, "--api-key-env", CUSTOM_VAR)

    assert result.exit_code == 1
    output = strip_ansi(result.output)
    assert f"Set {CUSTOM_VAR} env var" in output


@pytest.mark.parametrize("bad_name", ["123-bad", "has space", "MY[KEY]"])
def test_invalid_name_is_refused(tmp_path, monkeypatch, bad_name):
    """An invalid environment variable name is refused, naming the option."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "data",
            "generate",
            "--prompt",
            "Test prompt",
            "--count",
            "1",
            "--api-key-env",
            bad_name,
        ],
    )

    assert result.exit_code == 1
    output = strip_ansi(result.output)
    assert "--api-key-env" in output
    assert "not a valid environment variable name" in output
