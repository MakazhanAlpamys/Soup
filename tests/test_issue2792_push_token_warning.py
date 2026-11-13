"""soup push --hub hf: the --token deprecation is about the flag, not HF_TOKEN (#2792).

``--token`` is declared with ``envvar="HF_TOKEN"``, so Typer fills it from the
environment as well. The warning and the missing-token hint must tell the two
sources apart: a user who already exported HF_TOKEN is doing what the warning
asks and should not see it.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DEPRECATION = "--token is deprecated"


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    # No cached HF login and no hub tokens unless a test sets one.
    empty = tmp_path / "home"
    empty.mkdir()
    for var in ("HOME", "USERPROFILE", "HF_HOME"):
        monkeypatch.setenv(var, str(empty))
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return out


@pytest.fixture
def hf_api_tokens(monkeypatch):
    """Record the token handed to the HF client and stop before any network call."""
    seen: list = []

    def _fake_get_hf_api(token=None, endpoint=None):
        seen.append(token)
        raise ImportError("stop before upload")

    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", _fake_get_hf_api)
    return seen


def _push(*extra):
    from soup_cli.cli import app

    return runner.invoke(app, ["push", "--model", "out", "--repo", "user/m", *extra])


def test_hf_token_from_env_prints_no_deprecation(model_dir, hf_api_tokens, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
    result = _push()
    assert _DEPRECATION not in _clean(result.output)
    # The env token is still the one used: resolution is unchanged.
    assert hf_api_tokens == ["hf_FROM_ENV"]


def test_hf_token_typed_on_command_line_still_warns(model_dir, hf_api_tokens):
    result = _push("--token", "hf_TYPED")
    assert _DEPRECATION in _clean(result.output)
    assert hf_api_tokens == ["hf_TYPED"]


def test_hf_token_typed_with_env_set_still_warns(model_dir, hf_api_tokens, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_FROM_ENV")
    result = _push("--token", "hf_TYPED")
    assert _DEPRECATION in _clean(result.output)
    assert hf_api_tokens == ["hf_TYPED"]


def test_hf_missing_token_hint_does_not_suggest_deprecated_flag(model_dir, hf_api_tokens):
    result = _push()
    out = _clean(result.output)
    assert result.exit_code == 1
    assert "No HuggingFace token found" in out
    assert "HF_TOKEN" in out
    assert "huggingface-cli login" in out
    assert "--token" not in out
    assert hf_api_tokens == []


def test_non_hf_missing_token_hint_keeps_token_flag(model_dir, monkeypatch):
    # On other hubs --token is the supported way, not a deprecated one.
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    result = _push("--hub", "modelscope")
    out = _clean(result.output)
    assert result.exit_code == 1
    assert "--token YOUR_TOKEN" in out
    assert "MODELSCOPE_API_TOKEN" in out
