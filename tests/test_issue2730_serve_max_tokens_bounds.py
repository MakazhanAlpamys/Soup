"""#2730: ``soup serve --max-tokens`` is checked before anything loads.

The value is the default for every request that omits ``max_tokens``, on every
backend. Unchecked, ``--max-tokens 0`` (or a negative value) started the server
and then failed each such request with HTTP 500, and a value above the
per-request cap of 16384 bypassed it.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import serve as serve_cmd
from tests.conftest import strip_ansi

runner = CliRunner()


def _serve(*args: str):
    # The model path does not exist, so a value that passes the check stops at
    # "Model path not found" instead of loading anything.
    return runner.invoke(app, ["serve", "--model", "no-such-model-dir", *args])


@pytest.mark.parametrize("value", ["0", "-1", "-5", "16385", "100000"])
@pytest.mark.parametrize("backend", ["transformers", "vllm", "sglang", "mii"])
def test_an_out_of_range_default_is_refused_before_the_model(value, backend, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = _serve("--backend", backend, f"--max-tokens={value}")
    out = strip_ansi(result.output)
    assert result.exit_code == 2, out
    assert "--max-tokens" in out and "[1, 16384]" in out
    assert "Model path not found" not in out


@pytest.mark.parametrize("value", ["1", "512", "16384"])
def test_an_in_range_default_passes_the_check(value, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = _serve(f"--max-tokens={value}")
    out = strip_ansi(result.output)
    assert "--max-tokens:" not in out
    assert "Model path not found" in out


def test_the_flag_and_the_request_field_share_one_range():
    field = serve_cmd._MAX_TOKENS_MIN, serve_cmd._MAX_TOKENS_MAX
    assert field == (1, 16384)
