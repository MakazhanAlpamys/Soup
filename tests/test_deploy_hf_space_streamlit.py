"""Tests for Hugging Face Streamlit Space SDK guidance (#2746)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import HfHubHTTPError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.deploy import _is_streamlit_sdk_refusal, hf_space
from tests.conftest import strip_ansi

runner = CliRunner()
SYNTHETIC_SDK_REFUSAL = "Invalid space_sdk 'streamlit'; supported SDKs are gradio and docker."


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _run_deploy(monkeypatch, tmp_path, *arguments: str, api=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    if api is None:
        api = MagicMock()
    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: api)
    result = runner.invoke(
        app,
        [
            "deploy", "hf-space", "--model", "user/model", "--space", "user/space",
            "--yes", *arguments,
        ],
    )
    return result, api


def _hub_error(server_message: str, *, status_code: int = 400) -> HfHubHTTPError:
    request = SimpleNamespace(url="https://huggingface.co/api/repos/create")
    response = SimpleNamespace(headers={}, request=request, status_code=status_code)
    return HfHubHTTPError(
        f"{status_code} Client Error", response=response, server_message=server_message,
    )


@pytest.mark.parametrize("failure_step", ["create_repo", "create_commit"])
def test_streamlit_sdk_refusal_is_named_and_keeps_hub_response(
    monkeypatch, tmp_path, failure_step: str,
) -> None:
    api = MagicMock()
    getattr(api, failure_step).side_effect = _hub_error(SYNTHETIC_SDK_REFUSAL)

    result, _ = _run_deploy(
        monkeypatch, tmp_path, "--template", "streamlit-chat", api=api,
    )

    output = _plain(result.output)
    assert result.exit_code == 1
    assert "Hugging Face no longer creates new Streamlit Spaces" in output
    assert "--template gradio-chat" in output
    assert SYNTHETIC_SDK_REFUSAL in output


@pytest.mark.parametrize(
    "server_message",
    ["Unsupported SDK", "Unsupported Streamlit runtime"],
)
def test_sdk_refusal_requires_field_and_value_in_server_message(server_message: str) -> None:
    error = _hub_error(server_message)
    # The exception string can contain a request URL or SDK-looking text; it
    # must not substitute for the Hub's structured server message.
    error.args = ("Invalid space_sdk 'streamlit'",)

    assert not _is_streamlit_sdk_refusal(error, "streamlit")


@pytest.mark.parametrize(
    ("message", "status_code"),
    [
        ("authentication failed", 401),
        ("payment required", 402),
        ("server unavailable", 503),
    ],
)
def test_unrelated_hub_errors_keep_their_original_message(
    monkeypatch, tmp_path, message: str, status_code: int,
) -> None:
    api = MagicMock()
    api.create_repo.side_effect = _hub_error(message, status_code=status_code)

    result, _ = _run_deploy(
        monkeypatch, tmp_path, "--template", "streamlit-chat", api=api,
    )

    output = _plain(result.output)
    assert result.exit_code == 1
    assert "Space deploy failed" in output
    assert "Hub response:" not in output
    assert "Use --template gradio-chat instead" not in output


def test_gradio_sdk_refusal_is_not_mapped(monkeypatch, tmp_path) -> None:
    _write_custom_template(tmp_path, sdk="gradio")
    api = MagicMock()
    api.create_repo.side_effect = _hub_error("Invalid space_sdk 'gradio'")

    result, _ = _run_deploy(monkeypatch, tmp_path, "--template-dir", "custom", api=api)

    output = _plain(result.output)
    assert result.exit_code == 1
    assert "Space deploy failed" in output
    assert "Hub response:" not in output
    assert "Use --template gradio-chat instead" not in output


def test_connection_error_is_not_mapped(monkeypatch, tmp_path) -> None:
    api = MagicMock()
    api.create_repo.side_effect = ConnectionError("connection timed out")

    result, _ = _run_deploy(
        monkeypatch, tmp_path, "--template", "streamlit-chat", api=api,
    )

    output = _plain(result.output)
    assert result.exit_code == 1
    assert "Space deploy failed" in output
    assert "Hub response:" not in output
    assert "connection timed out" in output


def test_custom_streamlit_sdk_refusal_is_mapped(monkeypatch, tmp_path) -> None:
    _write_custom_template(tmp_path, sdk="streamlit")
    api = MagicMock()
    api.create_repo.side_effect = _hub_error(SYNTHETIC_SDK_REFUSAL)

    result, _ = _run_deploy(monkeypatch, tmp_path, "--template-dir", "custom", api=api)

    output = _plain(result.output)
    assert result.exit_code == 1
    assert "Hugging Face no longer creates new Streamlit Spaces" in output
    assert "--template gradio-chat" in output
    assert SYNTHETIC_SDK_REFUSAL in output


def test_non_hub_sdk_error_is_not_mapped(monkeypatch, tmp_path) -> None:
    api = MagicMock()
    api.create_repo.side_effect = RuntimeError(SYNTHETIC_SDK_REFUSAL)

    result, _ = _run_deploy(
        monkeypatch, tmp_path, "--template", "streamlit-chat", api=api,
    )

    output = _plain(result.output)
    assert result.exit_code == 1
    assert "Space deploy failed" in output
    assert "Hub response:" not in output


def test_builtin_streamlit_prints_v078_deprecation_warning(monkeypatch, tmp_path) -> None:
    result, api = _run_deploy(monkeypatch, tmp_path, "--template", "streamlit-chat")

    output = _plain(result.output).lower()
    assert result.exit_code == 0, (output, repr(result.exception))
    assert "deprecation warning (v0.78.0)" in output
    api.create_repo.assert_called_once()


def test_builtin_gradio_has_no_streamlit_warning(monkeypatch, tmp_path) -> None:
    result, _ = _run_deploy(monkeypatch, tmp_path, "--template", "gradio-chat")

    output = _plain(result.output).lower()
    assert result.exit_code == 0, (output, repr(result.exception))
    assert "deprecation warning" not in output


def _write_custom_template(tmp_path, *, sdk: str) -> None:
    template = tmp_path / "custom"
    template.mkdir()
    (template / "app.py").write_text(f"import {sdk}\n", encoding="utf-8")
    (template / "README.md").write_text(f"---\nsdk: {sdk}\n---\n", encoding="utf-8")
    (template / "requirements.txt").write_text(f"{sdk}>=1.0\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("sdk", "warning_expected"), [("streamlit", True), ("gradio", False)],
)
def test_custom_template_warning_matches_detected_sdk(
    monkeypatch, tmp_path, sdk: str, warning_expected: bool,
) -> None:
    _write_custom_template(tmp_path, sdk=sdk)

    result, api = _run_deploy(monkeypatch, tmp_path, "--template-dir", "custom")

    output = _plain(result.output).lower()
    assert result.exit_code == 0, (output, repr(result.exception))
    assert ("deprecation warning (v0.78.0)" in output) is warning_expected
    assert api.create_repo.call_args.kwargs["space_sdk"] == sdk


def test_help_and_docstring_explain_streamlit_deprecation() -> None:
    help_result = runner.invoke(app, ["deploy", "hf-space", "--help"])
    output = _plain(help_result.output).lower()

    assert help_result.exit_code == 0
    assert "streamlit-chat" in output
    assert "gradio-chat" in output
    assert "no longer creates" in output
    assert hf_space.__doc__ is not None
    docstring = _plain(hf_space.__doc__).lower()
    assert "streamlit-chat" in docstring
    assert "v0.78.0" in docstring
