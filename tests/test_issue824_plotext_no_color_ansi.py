"""Tests for issue #824: plotext histogram bypasses Rich and emits ANSI escape codes."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.plotext_compat import render_histogram, render_line

runner = CliRunner()


def test_cli_runner_captures_histogram_without_ansi(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        json.dumps({"instruction": "hi", "output": "hello world"}) + "\n"
        + json.dumps({"instruction": "describe soup", "output": "a fine tool"}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["data", "stats", str(data_file)])
    assert result.exit_code == 0
    # The histogram should now be captured by CliRunner
    assert "Text Length Distribution" in result.output
    # Must not contain raw escape bytes
    assert "\x1b" not in result.output


def test_subprocess_data_stats_no_color_has_zero_ansi_escapes(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        "\n".join(
            json.dumps({"instruction": f"prompt {i}", "output": f"answer {'x' * (i * 10)}"})
            for i in range(1, 15)
        )
        + "\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "NO_COLOR": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONLEGACYWINDOWSSTDIO": "0",
    }
    cmd = [sys.executable, "-m", "soup_cli", "data", "stats", str(data_file)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0
    assert b"\x1b" not in proc.stdout
    decoded = proc.stdout.decode("utf-8", errors="replace")
    assert "Text Length Distribution" in decoded


def test_subprocess_data_stats_redirected_stdout_has_zero_ansi_escapes(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        json.dumps({"instruction": "q", "output": "a"}) + "\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONLEGACYWINDOWSSTDIO": "0",
    }
    env.pop("NO_COLOR", None)

    cmd = [sys.executable, "-m", "soup_cli", "data", "stats", str(data_file)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0
    # Standard output redirected to pipe is not a tty/terminal, so no ANSI codes
    assert b"\x1b" not in proc.stdout
    decoded = proc.stdout.decode("utf-8", errors="replace")
    assert "Text Length Distribution" in decoded


def test_plotext_compat_uncolorize_on_non_terminal_console() -> None:
    import plotext

    buf = io.StringIO()
    cons = Console(file=buf, force_terminal=False, color_system=None)

    render_histogram(
        plotext,
        [10, 20, 30, 40, 50],
        bins=5,
        title="Test Distribution",
        xlabel="Length",
        ylabel="Count",
        console=cons,
    )

    output = buf.getvalue()
    assert "Test Distribution" in output
    assert "\x1b" not in output


def test_plotext_compat_render_line_uncolorize(monkeypatch: pytest.MonkeyPatch) -> None:
    import plotext

    buf = io.StringIO()
    cons = Console(file=buf, force_terminal=True, color_system="standard")
    monkeypatch.setenv("NO_COLOR", "1")

    render_line(
        plotext,
        [1, 2, 3],
        [1.0, 0.5, 0.2],
        label="train_loss",
        title="Loss Curve",
        xlabel="Step",
        ylabel="Loss",
        console=cons,
    )

    output = buf.getvalue()
    assert "Loss Curve" in output
    assert "\x1b" not in output
