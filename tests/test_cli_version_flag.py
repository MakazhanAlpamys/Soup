"""Tests for eager `soup --version` CLI flag (#2101)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from soup_cli import __version__
from soup_cli.cli import app
from soup_cli.utils.constants import GITHUB_URL
from tests.conftest import strip_ansi


def _plain(text: str) -> str:
    """Whitespace-collapsed, ANSI-stripped output."""
    return " ".join(strip_ansi(text).split())


class TestVersionFlag:
    def test_soup_version_flag_exits_zero_and_prints_version(self):
        result = CliRunner().invoke(app, ["--version"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert f"soup v{__version__}" in output
        assert GITHUB_URL in output

    def test_soup_version_flag_matches_version_subcommand_output(self):
        flag_result = CliRunner().invoke(app, ["--version"])
        cmd_result = CliRunner().invoke(app, ["version"])
        assert flag_result.exit_code == 0, (flag_result.output, repr(flag_result.exception))
        assert cmd_result.exit_code == 0, (cmd_result.output, repr(cmd_result.exception))
        assert _plain(flag_result.output) == _plain(cmd_result.output)

    def test_soup_version_flag_is_eager_over_help(self):
        result = CliRunner().invoke(app, ["--version", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert f"soup v{__version__}" in output
        assert "Usage" not in output

    def test_version_subcommand_remains_functional(self):
        json_result = CliRunner().invoke(app, ["version", "--json"])
        assert json_result.exit_code == 0, (json_result.output, repr(json_result.exception))
        data = json.loads(json_result.output)
        assert data["version"] == __version__
