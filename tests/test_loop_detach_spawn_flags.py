"""``soup loop watch --detach`` starts the child in its own session / process group.

The child used to inherit the terminal: closing it (SIGHUP, console close) or Ctrl+C in it
reached the daemon and its training child, and the child's Ctrl+C recorded a failed train. Now
the child is spawned with ``start_new_session=True`` (POSIX) or ``DETACHED_PROCESS |
CREATE_NEW_PROCESS_GROUP`` (Windows), and with its stdin on the null device.
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_detach as loop_detach
from soup_cli.cli import app

runner = CliRunner()
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


class TestSpawnFlagsPerPlatform:
    def test_posix_starts_a_new_session(self):
        kwargs = loop_detach.detached_popen_kwargs("posix")

        assert kwargs == {"stdin": subprocess.DEVNULL, "start_new_session": True}

    def test_windows_detaches_the_console_and_starts_a_new_process_group(self):
        kwargs = loop_detach.detached_popen_kwargs("nt")

        assert kwargs == {
            "stdin": subprocess.DEVNULL,
            "creationflags": DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        }
        assert "start_new_session" not in kwargs

    def test_the_default_is_the_platform_this_runs_on(self):
        assert loop_detach.detached_popen_kwargs() == loop_detach.detached_popen_kwargs(os.name)


class TestTheCommandPassesThem:
    @pytest.fixture
    def initialised(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(loop_detach.DETACHED_ENV, raising=False)
        monkeypatch.setattr(loop_detach, "process_is_alive", lambda pid: False)
        result = runner.invoke(
            app, ["loop", "init", "m", "--eval", "e.json", "--baseline", "b"]
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        return tmp_path

    def test_popen_gets_the_detaching_arguments(self, initialised):
        with mock.patch("soup_cli.commands.loop.subprocess.Popen") as popen:
            popen.return_value.pid = 123
            result = runner.invoke(app, ["loop", "watch", "--detach"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        kwargs = popen.call_args.kwargs
        for key, value in loop_detach.detached_popen_kwargs().items():
            assert kwargs[key] == value, key
        assert kwargs["stdout"] is not None and kwargs["stderr"] == subprocess.STDOUT
