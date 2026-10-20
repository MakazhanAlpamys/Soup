"""``soup train --gpus N`` ends with the exit code of the launcher it hands over to.

With ``--gpus N`` (N > 1) the command re-runs itself under ``accelerate
launch``. It did that with ``os.execvp`` on every platform:

* on POSIX ``execvp`` replaces the process, so the launcher's exit status is
  the command's. Nothing to change; the end-to-end test below pins it.
* Windows has no exec. ``os.execvp`` there starts the launcher as a NEW
  process and ends the calling one with status 0 at once. The shell got its
  prompt back while the run was still going, and ``soup train --gpus 2 &&
  next-step`` ran ``next-step`` straight away, whatever the launcher returned
  later.

``soup_cli.utils.launcher.run_launcher`` now owns the hand-over: exec on POSIX,
run-and-wait on Windows, and the command exits with the code it reports.

The end-to-end test puts a fake ``accelerate`` that exits 7 on ``PATH`` and
runs the real command line in a subprocess, on whatever platform the suite
runs on. The unit tests drive both branches on every platform through the
module's ``_IS_WINDOWS`` switch.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli
from soup_cli.cli import app
from soup_cli.utils import launcher, topology
from tests.conftest import strip_ansi

_DISTRIBUTED_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "ACCELERATE_MIXED_PRECISION",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
)

_CONFIG = (
    "base: test/model\n"
    "task: sft\n"
    "data: {train: data.jsonl, format: alpaca}\n"
    "training: {epochs: 1, lr: 1e-4, batch_size: 1}\n"
)


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


def _exit_with(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


# ---------------------------------------------------------------------------
# run_launcher: the run-and-wait branch (what Windows uses)
# ---------------------------------------------------------------------------


class TestRunAndWait:
    @pytest.fixture(autouse=True)
    def _wait_branch(self, monkeypatch):
        monkeypatch.setattr(launcher, "_IS_WINDOWS", True)

    @pytest.mark.parametrize("code", [0, 1, 2, 7, 113])
    def test_the_launcher_exit_code_is_returned(self, code):
        assert launcher.run_launcher(_exit_with(code)) == code

    def test_it_waits_for_the_launcher_to_finish(self, tmp_path):
        marker = tmp_path / "finished.txt"
        script = (
            "import sys, time\n"
            "time.sleep(0.6)\n"
            f"open({str(marker)!r}, 'w').write('done')\n"
            "sys.exit(7)\n"
        )

        code = launcher.run_launcher([sys.executable, "-c", script])

        # Returned only after the child wrote its marker, and with its code.
        assert marker.read_text() == "done"
        assert code == 7

    def test_the_launcher_inherits_the_environment(self, tmp_path, monkeypatch):
        # The multi-node path exports NCCL_* with os.environ.setdefault just
        # before the hand-over; the launcher must see them.
        monkeypatch.setenv("NCCL_SOUP_TEST", "from-parent")
        seen = "os.environ.get('NCCL_SOUP_TEST') == 'from-parent'"
        script = f"import os, sys; sys.exit(5 if {seen} else 6)"

        assert launcher.run_launcher([sys.executable, "-c", script]) == 5

    def test_a_launcher_that_is_not_on_path_raises_oserror_naming_it(self):
        with pytest.raises(OSError, match="no-such-launcher-soup-test"):
            launcher.run_launcher(["no-such-launcher-soup-test", "launch"])

    def test_no_shell_is_involved(self, tmp_path, monkeypatch):
        # An argument that a shell would split or expand arrives as one argument.
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "argv.txt"
        script = f"import sys; open({str(out)!r}, 'w').write('|'.join(sys.argv[1:]))"
        tricky = "a b & echo x > y"

        assert launcher.run_launcher([sys.executable, "-c", script, tricky, "$HOME"]) == 0
        assert out.read_text() == f"{tricky}|$HOME"
        assert not (tmp_path / "y").exists()

    def test_ctrl_c_keeps_waiting_for_the_launcher(self, monkeypatch):
        # The console delivers Ctrl+C to the launcher too; the parent must not
        # return before it, or the shell is told a code the run did not produce.
        waits: list[str] = []

        class _Process:
            def __init__(self, argv):
                self.argv = argv

            def wait(self):
                waits.append("wait")
                if len(waits) < 3:
                    raise KeyboardInterrupt
                return 130

            def kill(self):  # pragma: no cover - must not be called
                raise AssertionError("the launcher was killed instead of awaited")

        monkeypatch.setattr(subprocess, "Popen", _Process)

        assert launcher.run_launcher(_exit_with(0)) == 130
        assert waits == ["wait", "wait", "wait"]

    def test_an_empty_argv_is_refused(self):
        with pytest.raises(ValueError, match="launcher argv must not be empty"):
            launcher.run_launcher([])


class TestTheLauncherIsLookedUpOnPathOnly:
    """``PATH`` directories and ``PATHEXT`` suffixes; never the current directory."""

    @pytest.fixture
    def dirs(self, tmp_path, monkeypatch):
        first, second, cwd = tmp_path / "first", tmp_path / "second", tmp_path / "project"
        for directory in (first, second, cwd):
            directory.mkdir()
        monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second)]))
        monkeypatch.setenv("PATHEXT", ".exe;.cmd")
        # Windows' own switch for "do not search the current directory". Some
        # shells export it; without it a lookup that honours the Windows
        # default would find the file in the project directory.
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
        monkeypatch.chdir(cwd)
        return first, second, cwd

    @pytest.mark.parametrize("suffix", [".exe", ".cmd"])
    def test_a_pathext_suffix_is_tried_in_each_path_directory(self, dirs, suffix):
        _first, second, _cwd = dirs
        (second / f"soup-fake-launcher{suffix}").write_text("", encoding="utf-8")

        found = launcher._find_on_path("soup-fake-launcher")

        assert found is not None
        assert os.path.samefile(found, second / f"soup-fake-launcher{suffix}")

    def test_the_first_path_directory_wins(self, dirs):
        first, second, _cwd = dirs
        (first / "soup-fake-launcher.cmd").write_text("", encoding="utf-8")
        (second / "soup-fake-launcher.exe").write_text("", encoding="utf-8")

        found = launcher._find_on_path("soup-fake-launcher")

        assert found is not None
        assert os.path.samefile(found, first / "soup-fake-launcher.cmd")

    def test_a_file_in_the_current_directory_is_not_the_launcher(self, dirs, monkeypatch):
        _first, _second, cwd = dirs
        (cwd / "soup-fake-launcher.exe").write_text("", encoding="utf-8")
        (cwd / "soup-fake-launcher.cmd").write_text("", encoding="utf-8")

        assert launcher._find_on_path("soup-fake-launcher") is None
        monkeypatch.setattr(launcher, "_IS_WINDOWS", True)
        with pytest.raises(OSError, match="launcher not found on PATH"):
            launcher.run_launcher(["soup-fake-launcher", "launch"])

    def test_an_empty_path_entry_does_not_mean_the_current_directory(self, dirs, monkeypatch):
        first, second, cwd = dirs
        monkeypatch.setenv("PATH", os.pathsep.join([str(first), "", str(second)]))
        (cwd / "soup-fake-launcher.exe").write_text("", encoding="utf-8")

        assert launcher._find_on_path("soup-fake-launcher") is None

    def test_a_name_that_already_has_a_suffix_is_used_as_it_is(self, dirs):
        first, _second, _cwd = dirs
        (first / "soup-fake-launcher.cmd").write_text("", encoding="utf-8")

        found = launcher._find_on_path("soup-fake-launcher.cmd")

        assert found is not None
        assert os.path.samefile(found, first / "soup-fake-launcher.cmd")
        assert launcher._find_on_path("soup-fake-launcher.bat") is None

    def test_an_explicit_path_is_not_searched_for(self, dirs):
        assert launcher._find_on_path(sys.executable) == sys.executable
        assert launcher._find_on_path(os.path.join("missing-dir", "tool.exe")) is None


# ---------------------------------------------------------------------------
# run_launcher: the exec branch (POSIX)
# ---------------------------------------------------------------------------


class TestExec:
    def test_posix_replaces_the_process_with_the_launcher(self, monkeypatch):
        monkeypatch.setattr(launcher, "_IS_WINDOWS", False)
        calls: list[tuple] = []

        def _execvp(file, argv):
            calls.append((file, list(argv)))
            raise SystemExit(99)  # a real execvp never returns

        monkeypatch.setattr(os, "execvp", _execvp)
        monkeypatch.setattr(
            subprocess, "Popen", lambda *a, **k: pytest.fail("POSIX must exec, not spawn")
        )

        with pytest.raises(SystemExit):
            launcher.run_launcher(["accelerate", "launch", "--num_processes", "2"])

        assert calls == [("accelerate", ["accelerate", "launch", "--num_processes", "2"])]

    def test_the_switch_follows_the_platform(self):
        assert launcher._IS_WINDOWS is (os.name == "nt")


# ---------------------------------------------------------------------------
# soup train --gpus N: the command exits with what run_launcher reports
# ---------------------------------------------------------------------------


@pytest.fixture
def two_gpu_run(tmp_path, monkeypatch):
    for var in _DISTRIBUTED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setattr(launcher, "is_in_distributed", lambda: False)
    # Keep the in-process runs off the real GPU probe; the end-to-end test uses it.
    monkeypatch.setattr(
        topology, "detect_topology", lambda: {"gpu_count": 2, "interconnect": "pcie"}
    )

    def invoke(*extra: str):
        return CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--gpus", "2", "--yes", *extra]
        )

    return invoke


class TestTrainExitsWithTheLauncherCode:
    @pytest.mark.parametrize("code", [0, 1, 7, 113])
    def test_exit_code_is_the_launcher_code(self, two_gpu_run, monkeypatch, code):
        seen: list[list[str]] = []

        def _run(argv):
            seen.append(list(argv))
            return code

        monkeypatch.setattr(launcher, "run_launcher", _run)

        result = two_gpu_run()

        assert result.exit_code == code, (result.output, repr(result.exception))
        assert len(seen) == 1
        assert seen[0][:4] == ["accelerate", "launch", "--num_processes", "2"]
        assert "--no-reexec" in seen[0]

    def test_a_launcher_that_cannot_start_exits_1_with_the_hint(self, two_gpu_run, monkeypatch):
        def _run(argv):
            raise FileNotFoundError(2, "launcher not found on PATH", argv[0])

        monkeypatch.setattr(launcher, "run_launcher", _run)

        result = two_gpu_run()

        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "accelerate launch failed:" in out
        assert "launcher not found on PATH" in out
        assert "Use --no-reexec to print the launch command." in out

    def test_no_reexec_prints_the_command_and_does_not_launch(self, two_gpu_run, monkeypatch):
        monkeypatch.setattr(
            launcher, "run_launcher", lambda argv: pytest.fail("must not launch")
        )

        result = two_gpu_run("--no-reexec")

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "Multi-GPU launch required" in _plain(result.output)


# ---------------------------------------------------------------------------
# End to end: a fake `accelerate` on PATH that exits 7
# ---------------------------------------------------------------------------


def _fake_accelerate(bin_dir: Path, marker: Path, code: int) -> None:
    """A launcher named ``accelerate`` that records its arguments and exits ``code``."""
    bin_dir.mkdir()
    if os.name == "nt":
        script = bin_dir / "accelerate.cmd"
        script.write_text(
            f'@echo %*> "{marker}"\r\n@exit {code}\r\n', encoding="utf-8", newline=""
        )
    else:
        script = bin_dir / "accelerate"
        script.write_text(
            f'#!/bin/sh\nprintf \'%s \' "$@" > "{marker}"\nexit {code}\n',
            encoding="utf-8",
            newline="",
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.mark.parametrize("code", [7, 0])
def test_soup_train_gpus_returns_the_exit_code_of_accelerate(tmp_path, code):
    """The real command line, in a real process, on this platform's hand-over path."""
    marker = tmp_path / "launched.txt"
    _fake_accelerate(tmp_path / "bin", marker, code)
    (tmp_path / "soup.yaml").write_text(_CONFIG, encoding="utf-8")

    env = {key: value for key, value in os.environ.items() if key not in _DISTRIBUTED_ENV}
    env["PATH"] = str(tmp_path / "bin") + os.pathsep + env.get("PATH", "")
    # The child must import the tree under test, not another installed copy.
    env["PYTHONPATH"] = str(Path(soup_cli.__file__).resolve().parents[1])
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["HF_HUB_OFFLINE"] = "1"
    env["NO_COLOR"] = "1"

    proc = subprocess.run(
        [sys.executable, "-m", "soup_cli.cli", "train", "--config", "soup.yaml",
         "--gpus", "2", "--yes"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300, check=False,
        encoding="utf-8", errors="replace",
    )

    # The launcher ran, with the launch argv, before the command returned ...
    assert marker.is_file(), (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    launched = marker.read_text(encoding="utf-8", errors="replace")
    assert "launch --num_processes 2" in launched
    assert "--no-reexec" in launched
    # ... and its exit code is the command's.
    assert proc.returncode == code, (proc.stdout[-2000:], proc.stderr[-2000:])
