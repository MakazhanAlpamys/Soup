"""#1808 — ``soup loop watch --detach`` validates the daemon's options before it forks.

The parent spawned the child with stdout and stderr on ``/dev/null`` and printed
"watch detached" without checking the options the child validates at startup
(``WatchConfig.__post_init__``), so a ``--poll-interval 86400`` or
``--max-iterations -1`` reported success while the child exited 2 unseen. The
parent now builds the same config first and fails with the same message.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

from .conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture()
def initialised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["loop", "init", "m", "--eval", "e.json", "--baseline", "b"]
    )
    assert result.exit_code == 0, result.output
    return tmp_path


def _detach(*extra: str):
    with mock.patch("soup_cli.commands.loop.subprocess.Popen") as popen:
        popen.return_value.pid = 123
        result = CliRunner().invoke(app, ["loop", "watch", "--detach", *extra])
    return result, popen


@pytest.mark.parametrize(
    "extra,message",
    [
        pytest.param(["--poll-interval", "86400"], "poll_interval_sec must be in [1, 3600]",
                     id="poll-interval"),
        pytest.param(["--poll-interval", "0.5"], "poll_interval_sec must be in [1, 3600]",
                     id="poll-interval-low"),
        pytest.param(["--max-iterations", "-1"], "max_iterations must be a non-negative int",
                     id="max-iterations"),
    ],
)
def test_invalid_options_fail_in_the_parent_and_spawn_no_child(initialised, extra, message):
    result, popen = _detach(*extra)
    out = _plain(result.output)
    assert result.exit_code == 2, (result.exit_code, out)
    assert "invalid watch config" in out
    assert message in out
    assert "watch detached" not in out
    popen.assert_not_called()


def test_control_valid_options_still_detach(initialised):
    result, popen = _detach("--poll-interval", "60", "--max-iterations", "1")
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "watch detached pid=123" in out
    popen.assert_called_once()
    argv = popen.call_args.args[0]
    assert "--foreground" in argv and "--poll-interval" in argv


def test_control_the_foreground_path_keeps_its_own_message(initialised):
    result = CliRunner().invoke(
        app, ["loop", "watch", "--foreground", "--poll-interval", "86400"]
    )
    out = _plain(result.output)
    assert result.exit_code == 2, (result.exit_code, out)
    assert "poll_interval_sec must be in [1, 3600]" in out
