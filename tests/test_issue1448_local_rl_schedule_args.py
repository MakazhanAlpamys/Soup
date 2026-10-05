"""#1448 — ``soup local-rl train`` (scheduled mode) drops ``--min-pairs`` and ``-o``.

Without ``--once``, ``soup local-rl train`` renders a systemd ``.service`` /
``.timer`` and a launchd ``.plist`` that run ``soup local-rl train --once``
every night. The rendered command carried ``--db``, ``--model`` and
``--train-method`` only, so the nightly job trained on as few as 10 pairs into
``local_rl_adapter/`` whatever ``--min-pairs`` and ``--output`` the user had
asked for, and printed nothing about it. ``--once`` already passed both.

This file pins:

* the rendered ``.service`` and ``.plist`` carry both values;
* the rendered command, parsed by the real CLI, reaches ``run_nightly_train``
  with the same ``min_pairs`` and ``output_dir`` that ``--once`` would pass;
* a ``--min-pairs`` the nightly run would refuse is refused when the scaffold
  is rendered, before any unit file is written;
* the defaults are rendered explicitly, so the nightly job matches ``--once``.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import sys

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.local_rl import init_local_rl_db
from soup_cli.utils.local_rl_scheduler import (
    LAUNCHD_PLIST_NAME,
    SYSTEMD_SERVICE_NAME,
    build_train_argv,
)
from tests.conftest import strip_ansi

runner = CliRunner()


def _schedule(*extra: str) -> "object":
    return runner.invoke(
        app,
        [
            "local-rl", "train",
            "--db", "local_rl.db",
            "--model", "org/base",
            "--scheduler-dir", "sched",
            *extra,
        ],
    )


def _service_argv(tmp_path) -> list:
    text = (tmp_path / "sched" / SYSTEMD_SERVICE_NAME).read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("ExecStart="))
    return shlex.split(line[len("ExecStart="):], posix=True)


def _plist_argv(tmp_path) -> list:
    with open(tmp_path / "sched" / LAUNCHD_PLIST_NAME, "rb") as fh:
        return list(plistlib.load(fh)["ProgramArguments"])


def _value_after(argv: list, flag: str) -> str:
    assert flag in argv, (flag, argv)
    return argv[argv.index(flag) + 1]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_local_rl_db("local_rl.db")
    return tmp_path


class TestTheRenderedJobCarriesTheOptions:
    def test_systemd_service(self, workdir):
        result = _schedule("--min-pairs", "200", "-o", "adapters/nightly")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        argv = _service_argv(workdir)
        assert _value_after(argv, "--min-pairs") == "200"
        assert _value_after(argv, "--output") == "adapters/nightly"

    def test_launchd_plist(self, workdir):
        result = _schedule("--min-pairs", "200", "-o", "adapters/nightly")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        argv = _plist_argv(workdir)
        assert _value_after(argv, "--min-pairs") == "200"
        assert _value_after(argv, "--output") == "adapters/nightly"

    def test_an_output_path_with_a_space_survives_systemd_quoting(self, workdir):
        result = _schedule("-o", "my adapters/nightly")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert _value_after(_service_argv(workdir), "--output") == "my adapters/nightly"
        assert _value_after(_plist_argv(workdir), "--output") == "my adapters/nightly"

    def test_the_defaults_are_rendered_too(self, workdir):
        """The nightly job must do what ``--once`` with the same flags does,
        including when the user left both at their defaults."""
        result = _schedule()
        assert result.exit_code == 0, (result.output, repr(result.exception))

        for argv in (_service_argv(workdir), _plist_argv(workdir)):
            assert _value_after(argv, "--min-pairs") == "10"
            assert _value_after(argv, "--output") == "local_rl_adapter"


def test_the_rendered_command_reaches_the_runner_with_the_same_values(
    workdir, monkeypatch
):
    """Run the scheduled command through the real CLI and record what the
    runner receives: a flag rendered under the wrong name would fail here."""
    import soup_cli.commands.local_rl as local_rl_cmd
    from soup_cli.utils.local_rl import NightlyTrainResult

    seen = {}

    def fake_run(cfg, *, once, min_pairs, output_dir):
        seen.update(once=once, min_pairs=min_pairs, output_dir=output_dir)
        return NightlyTrainResult(
            status="skipped_insufficient_pairs", num_pairs=0, output_dir=None,
            reason="test",
        )

    argv = build_train_argv(
        soup_python=sys.executable,
        db_path="local_rl.db",
        model="org/base",
        train_method="kto",
        min_pairs=200,
        output_dir="adapters/nightly",
    )
    assert argv[1:3] == ["-m", "soup_cli.cli"]
    monkeypatch.setattr(local_rl_cmd, "run_nightly_train", fake_run)

    result = runner.invoke(app, argv[3:])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert seen == {"once": True, "min_pairs": 200, "output_dir": "adapters/nightly"}


@pytest.mark.parametrize("min_pairs", ["0", "-5", "1000001"])
def test_a_min_pairs_the_nightly_run_would_refuse_is_refused_now(workdir, min_pairs):
    """Rendering it would install a job that fails every night."""
    result = _schedule("--min-pairs", min_pairs)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "min_pairs must be in [1, 1000000]" in " ".join(strip_ansi(result.output).split())
    assert not os.path.exists(workdir / "sched")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"min_pairs": True}, TypeError),
        ({"min_pairs": 2.5}, TypeError),
        ({"min_pairs": 0}, ValueError),
        ({"output_dir": ""}, ValueError),
        ({"output_dir": "a\x00b"}, ValueError),
        ({"output_dir": "a\nExecStartPre=/bin/sh"}, ValueError),
    ],
)
def test_build_train_argv_validates_the_new_arguments(workdir, kwargs, error):
    with pytest.raises(error):
        build_train_argv(
            soup_python=sys.executable,
            db_path="local_rl.db",
            model="org/base",
            train_method="dpo",
            **kwargs,
        )
