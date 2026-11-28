"""Issue #3278 — `soup registry list` / `search` help states the --limit range.

Both commands describe --limit as a maximum, but each clamps it: list floors
at 1 and caps at 1,000, search floors at 1 and caps at 500. A user reading the
option help cannot predict those values, so each help string states its own
range and that out-of-range values are clamped. The clamping itself is
unchanged; the store-limit test stays the behavioral control.
"""

from __future__ import annotations

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize(
    ("command", "limit_range", "default"),
    [("list", "1-1000", "100"), ("search", "1-500", "50")],
)
def test_limit_help_states_range_and_clamping(command, limit_range, default, monkeypatch):
    monkeypatch.setenv("COLUMNS", "500")
    result = make_cli_runner().invoke(app, ["registry", command, "--help"])
    assert result.exit_code == 0, (result.output, result.exception)
    output = _plain(result)
    assert limit_range in output
    assert "out-of-range values are clamped" in output
    assert f"[default: {default}]" in output
