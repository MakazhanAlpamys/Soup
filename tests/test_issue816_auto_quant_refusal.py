"""Regression tests for #816: --auto-quant must not guess from timer noise."""

from __future__ import annotations

import inspect
import re

import pytest
from typer.testing import CliRunner


def _plain(text: str) -> str:
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


@pytest.mark.parametrize("backend", ["transformers", "vllm", "sglang", "mii"])
def test_auto_quant_refuses_before_any_backend_can_apply_a_guess(backend: str) -> None:
    from soup_cli.cli import app

    result = CliRunner().invoke(
        app,
        ["serve", "--model", "unused", "--backend", backend, "--auto-quant"],
    )

    output = _plain(result.output)
    assert result.exit_code == 2
    assert "--auto-quant is unavailable" in output
    assert "Refusing instead of guessing" in output
    assert "picked:" not in output
    assert "binding vLLM" not in output


def test_auto_quant_result_is_independent_of_perf_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    from soup_cli.cli import app

    ticks = iter([1000.0, 0.0, 50.0, -50.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(ticks))
    runner = CliRunner()

    first = runner.invoke(app, ["serve", "--model", "unused", "--auto-quant"])
    second = runner.invoke(app, ["serve", "--model", "unused", "--auto-quant"])

    assert first.exit_code == second.exit_code == 2
    assert _plain(first.output) == _plain(second.output)


def test_serve_has_no_stub_picker_or_quantization_forwarding() -> None:
    from soup_cli.commands.serve import serve

    source = inspect.getsource(serve)
    assert "run_auto_quant_picker" not in source
    assert "quant_name_to_vllm_kwargs" not in source
    assert "return (\"\", True)" not in source
    assert "quantization=None" in source


def test_help_describes_refusal_instead_of_a_live_eval() -> None:
    from soup_cli.cli import app
    from soup_cli.commands.serve import serve

    result = CliRunner().invoke(app, ["serve", "--help"], terminal_width=220)
    output = _plain(result.output)
    help_text = inspect.signature(serve).parameters["auto_quant"].default.help

    assert result.exit_code == 0
    assert "--auto-quant" in output
    assert "refuses instead of guessing" in help_text
    assert "pick fastest-at-acceptable-quality" not in help_text
