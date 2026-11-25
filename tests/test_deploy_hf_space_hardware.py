"""Tests for the optional Hugging Face Space hardware and ZeroGPU path (#2745)."""

from __future__ import annotations

import ast
import hashlib
import re
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.deploy import render_space_template

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text or "").split())


def _deploy(monkeypatch, tmp_path, *arguments: str, api=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    if api is None:
        api = MagicMock()
    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", lambda **_: api)
    result = runner.invoke(
        app,
        [
            "deploy", "hf-space",
            "--model", "user/model",
            "--space", "user/space",
            "--yes",
            *arguments,
        ],
    )
    return result, api


def _uploaded_files(api) -> dict[str, bytes]:
    operations = api.create_commit.call_args.kwargs["operations"]
    return {operation.path_in_repo: operation.path_or_fileobj for operation in operations}


def _digests(rendered: dict[str, str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in rendered.items()
    }


def test_default_gradio_render_remains_byte_for_byte_unchanged() -> None:
    assert _digests(render_space_template("gradio-chat", model_repo="user/model")) == {
        "app.py": "015681a39733aa13c89d721898a40971c7cae0564c726aede8be18a77daa9113",
        "requirements.txt": "1f17adad35e12f7c34f0ed2ce8b6bb5ef357901235051e11c29f1e13a984e18b",
        "README.md": "df71349c810f32183c228fcfef0535ddf85804dff6532c97a0a8d7cb0f670d54",
    }


def test_default_streamlit_render_remains_byte_for_byte_unchanged() -> None:
    assert _digests(render_space_template("streamlit-chat", model_repo="user/model")) == {
        "app.py": "3f902fa9e4173f910a129082ed0185fb5d6350f55159721f94ccb25b675db032",
        "requirements.txt": "56afaf4f0d303d7173de71d81374ffca9584ccb5f100742f42fb21994dc63fdf",
        "README.md": "7d6bb7a5afd0a2c147d1399993362024b1e9bafbf7fff8b8cc338aaf9b4197fd",
    }


def test_zero_gpu_cli_uploads_decorated_respond_and_spaces_requirement(
    monkeypatch, tmp_path,
) -> None:
    result, api = _deploy(monkeypatch, tmp_path, "--hardware", "zero-a10g")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    files = _uploaded_files(api)
    app_source = files["app.py"].decode("utf-8")
    module = ast.parse(app_source)
    respond = next(
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "respond"
    )
    assert any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "spaces"
        and decorator.attr == "GPU"
        for decorator in respond.decorator_list
    )
    assert "spaces" in files["requirements.txt"].decode("utf-8").splitlines()
    assert api.create_repo.call_args.kwargs["space_hardware"] == "zero-a10g"


def test_t4_cli_uploads_exact_default_gradio_render(monkeypatch, tmp_path) -> None:
    result, api = _deploy(monkeypatch, tmp_path, "--hardware", "t4-small")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert _uploaded_files(api) == {
        name: content.encode("utf-8")
        for name, content in render_space_template("gradio-chat", model_repo="user/model").items()
    }
    assert api.create_repo.call_args.kwargs["space_hardware"] == "t4-small"


def test_default_hardware_option_is_omitted_from_hub_call(monkeypatch, tmp_path) -> None:
    result, api = _deploy(monkeypatch, tmp_path)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "space_hardware" not in api.create_repo.call_args.kwargs


@pytest.mark.parametrize(
    "slug_argument",
    ["", "-leading", "trailing-", "double--hyphen", "with_underscore", "a" * 33],
)
def test_invalid_hardware_slug_fails_before_any_hub_call(
    monkeypatch, tmp_path, slug_argument: str,
) -> None:
    api = MagicMock()
    result, api = _deploy(
        monkeypatch, tmp_path, f"--hardware={slug_argument}", api=api,
    )

    assert result.exit_code == 1
    assert _plain(result.output) == (
        "Invalid --hardware: expected a lowercase hardware slug of at most 32 letters, "
        "digits, and single hyphens."
    )
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()


def test_32_character_hardware_slug_is_accepted(monkeypatch, tmp_path) -> None:
    slug = "a" * 32
    result, api = _deploy(monkeypatch, tmp_path, f"--hardware={slug}")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert api.create_repo.call_args.kwargs["space_hardware"] == slug


def test_zerogpu_refuses_builtin_streamlit_before_any_hub_call(monkeypatch, tmp_path) -> None:
    api = MagicMock()
    result, api = _deploy(
        monkeypatch,
        tmp_path,
        "--hardware", "zero-a10g",
        "--template", "streamlit-chat",
        api=api,
    )

    assert result.exit_code == 1
    assert _plain(result.output) == (
        "ZeroGPU does not support the built-in streamlit-chat template; "
        "use --template gradio-chat."
    )
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()


def test_custom_template_is_uploaded_unchanged_with_hardware(monkeypatch, tmp_path) -> None:
    template = tmp_path / "custom"
    template.mkdir()
    source_files = {
        "app.py": "import gradio as gr\n",
        "README.md": "---\nsdk: gradio\n---\nCustom app\n",
        "requirements.txt": "gradio>=4\n",
    }
    for name, content in source_files.items():
        (template / name).write_text(content, encoding="utf-8")

    result, api = _deploy(
        monkeypatch,
        tmp_path,
        "--template-dir", "custom",
        "--hardware", "zero-a10g",
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert api.create_repo.call_args.kwargs["space_hardware"] == "zero-a10g"
    assert _uploaded_files(api) == {
        name: content.encode("utf-8") for name, content in source_files.items()
    }
