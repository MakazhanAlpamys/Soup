"""Regression tests for issue #1464: benchmark container cleanup on timeout or interruption."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


def _result(path: Path, *, passed: bool, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tests_outcomes": [passed], **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_docker_log(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [json.loads(line) for line in lines]


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "soup.db"))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker_py = bin_dir / "fake_docker.py"

    fake_docker_code = (
        "import json\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "\n"
        'log_file = os.environ.get("FAKE_DOCKER_LOG")\n'
        "if log_file:\n"
        '    with open(log_file, "a", encoding="utf-8") as f:\n'
        '        f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        "\n"
        "args = sys.argv[1:]\n"
        "if not args:\n"
        "    sys.exit(0)\n"
        "\n"
        "subcmd = args[0]\n"
        'if subcmd in ("info", "image"):\n'
        '    if subcmd == "info":\n'
        '        sys.stdout.write("24.0.0\\n")\n'
        "    sys.exit(0)\n"
        "\n"
        'marker_file = os.environ.get("FAKE_DOCKER_MARKER")\n'
        "\n"
        'if subcmd == "run":\n'
        '    mode = os.environ.get("FAKE_DOCKER_RUN", "hang")\n'
        '    if mode == "exit0":\n'
        "        sys.exit(0)\n"
        '    elif mode == "exit17":\n'
        "        sys.exit(17)\n"
        "    else:\n"
        "        start = time.monotonic()\n"
        "        cap = 30.0\n"
        "        while time.monotonic() - start < cap:\n"
        "            if marker_file and os.path.exists(marker_file):\n"
        "                break\n"
        "            time.sleep(0.05)\n"
        "        sys.exit(0)\n"
        "\n"
        'elif subcmd == "rm":\n'
        '    mode = os.environ.get("FAKE_DOCKER_RM", "ok")\n'
        '    if mode == "ok":\n'
        "        if marker_file:\n"
        "            try:\n"
        '                with open(marker_file, "w", encoding="utf-8") as f:\n'
        '                    f.write("removed\\n")\n'
        "            except OSError:\n"
        "                pass\n"
        "        sys.exit(0)\n"
        '    elif mode == "fail":\n'
        '        sys.stderr.write("docker: error removing container\\n")\n'
        "        sys.exit(1)\n"
        '    elif mode == "hang":\n'
        "        time.sleep(30.0)\n"
        "        sys.exit(0)\n"
        "\n"
        "sys.exit(0)\n"
    )
    fake_docker_py.write_text(fake_docker_code, encoding="utf-8")

    docker_sh = bin_dir / "docker"
    docker_sh.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{fake_docker_py}" "$@"\n',
        encoding="utf-8",
    )
    if hasattr(os, "chmod"):
        os.chmod(docker_sh, 0o755)

    docker_cmd = bin_dir / "docker.cmd"
    docker_cmd.write_text(
        f'@echo off\n"{sys.executable}" "{fake_docker_py}" %*\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    log_path = tmp_path / "docker_log.jsonl"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log_path))
    marker_path = tmp_path / "docker_removed_marker"
    monkeypatch.setenv("FAKE_DOCKER_MARKER", str(marker_path))
    monkeypatch.setenv("FAKE_DOCKER_RUN", "hang")
    monkeypatch.setenv("FAKE_DOCKER_RM", "ok")

    corpus = tmp_path / "polyglot"
    corpus.mkdir(exist_ok=True)


class DockerSpy:
    """Wrap subprocess.run to shorten timeouts, record invocations, and track events."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.events: list[str] = []
        self.shorten_rm = False
        self.raise_ctrl_c = False
        self.real_run = subprocess.run
        monkeypatch.setattr(subprocess, "run", self._run)

    def _run(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        call_kwargs = dict(kwargs)
        self.calls.append((argv, call_kwargs))

        if len(argv) > 1:
            if argv[1] == "run":
                call_kwargs["timeout"] = 1
                if self.raise_ctrl_c:
                    try:
                        return self.real_run(argv, *args, **call_kwargs)
                    except subprocess.TimeoutExpired:
                        raise KeyboardInterrupt from None
            elif argv[1] == "rm":
                self.events.append("docker_rm")
                if self.shorten_rm:
                    call_kwargs["timeout"] = 1

        run_cmd = argv
        if call_kwargs.get("shell") and isinstance(cmd, (list, tuple)):
            import shlex

            run_cmd = " ".join(shlex.quote(str(a)) for a in argv)

        return self.real_run(run_cmd, *args, **call_kwargs)


@pytest.fixture
def docker_spy(monkeypatch: pytest.MonkeyPatch) -> DockerSpy:
    return DockerSpy(monkeypatch)


def test_timeout_names_and_removes_container(docker_spy: DockerSpy, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    plain_output = _plain(result.output)
    assert "timed out after 60 seconds" in plain_output

    run_calls = [
        argv for argv, _ in docker_spy.calls if len(argv) > 1 and argv[1] == "run"
    ]
    assert run_calls, "docker run was not called"
    run_argv = run_calls[0]

    name_arg = next((arg for arg in run_argv if arg.startswith("--name=")), None)
    assert name_arg is not None, "docker run did not contain --name"
    container_name = name_arg.split("=", 1)[1]
    assert re.fullmatch(r"soup-aider-[0-9a-f]{12}", container_name)

    log_entries = _read_docker_log(tmp_path / "docker_log.jsonl")
    assert ["rm", "-f", container_name] in log_entries


def test_rm_uses_same_executable_and_no_shell(docker_spy: DockerSpy) -> None:
    from soup_cli.eval.aider_polyglot import DOCKER_REMOVE_TIMEOUT_SECONDS

    runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )

    run_calls = [
        (argv, kw)
        for argv, kw in docker_spy.calls
        if len(argv) > 1 and argv[1] == "run"
    ]
    assert run_calls, "docker run was not called"
    run_argv, _ = run_calls[0]

    name_arg = next((arg for arg in run_argv if arg.startswith("--name=")), None)
    assert name_arg is not None, "docker run did not specify --name"
    container_name = name_arg.split("=", 1)[1]

    rm_calls = [
        (argv, kw)
        for argv, kw in docker_spy.calls
        if len(argv) > 1 and argv[1] == "rm"
    ]
    assert rm_calls, "docker rm was not called"
    rm_argv, rm_kwargs = rm_calls[0]

    assert rm_kwargs.get("shell") is False
    assert rm_kwargs.get("timeout") == DOCKER_REMOVE_TIMEOUT_SECONDS
    assert rm_argv[0] == run_argv[0]
    assert rm_argv == [run_argv[0], "rm", "-f", container_name]


def test_failing_rm_emits_one_line_warning(
    docker_spy: DockerSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_RM", "fail")
    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    plain_output = _plain(result.output)
    assert "timed out after 60 seconds" in plain_output

    match = re.search(r"Could not remove container (soup-aider-[0-9a-f]{12})", plain_output)
    assert match is not None, "Warning with container name was not printed"
    container_name = match.group(1)

    warning_msg = (
        f"Could not remove container {container_name}; remove it with: "
        f"docker rm -f {container_name}"
    )
    assert warning_msg in plain_output

    lines = _ANSI_RE.sub("", result.output).splitlines()
    assert any(f"remove it with: docker rm -f {container_name}" in line for line in lines)

    pos_timeout = plain_output.find("timed out after 60 seconds")
    pos_warn = plain_output.find(f"Could not remove container {container_name}")
    assert pos_timeout < pos_warn

    # Also verify OSError during rm is caught and emits the warning
    original_run = docker_spy.real_run

    def oserror_run(cmd: Any, *a: Any, **kw: Any) -> Any:
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        if len(argv) > 1 and argv[1] == "rm":
            raise OSError("Simulated rm failure")
        return original_run(cmd, *a, **kw)

    monkeypatch.setattr(docker_spy, "real_run", oserror_run)
    result_err = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )
    assert result_err.exit_code == 1, (result_err.output, repr(result_err.exception))
    assert "Could not remove container" in _plain(result_err.output)


def test_hanging_rm_times_out_and_emits_warning(
    docker_spy: DockerSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_RM", "hang")
    docker_spy.shorten_rm = True
    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    plain_output = _plain(result.output)
    assert "timed out after 60 seconds" in plain_output

    match = re.search(r"Could not remove container (soup-aider-[0-9a-f]{12})", plain_output)
    assert match is not None, "Warning with container name was not printed"
    container_name = match.group(1)

    warning_msg = (
        f"Could not remove container {container_name}; remove it with: "
        f"docker rm -f {container_name}"
    )
    assert warning_msg in plain_output

    pos_timeout = plain_output.find("timed out after 60 seconds")
    pos_warn = plain_output.find(f"Could not remove container {container_name}")
    assert pos_timeout < pos_warn


def test_ctrl_c_removes_container_and_exits_130(
    docker_spy: DockerSpy, tmp_path: Path
) -> None:
    docker_spy.raise_ctrl_c = True
    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )
    # Typer 0.27.2 converts KeyboardInterrupt to exit code 130
    assert result.exit_code == 130, (result.output, repr(result.exception))

    run_calls = [
        argv for argv, _ in docker_spy.calls if len(argv) > 1 and argv[1] == "run"
    ]
    assert run_calls, "docker run was not called"
    name_arg = next((arg for arg in run_calls[0] if arg.startswith("--name=")), None)
    assert name_arg is not None, "docker run did not specify --name"
    container_name = name_arg.split("=", 1)[1]

    log_entries = _read_docker_log(tmp_path / "docker_log.jsonl")
    assert any(
        len(entry) > 1 and entry[0] == "rm" and container_name in entry
        for entry in log_entries
    )


def test_ctrl_c_failing_rm_emits_one_line_warning(
    docker_spy: DockerSpy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_RM", "fail")
    docker_spy.raise_ctrl_c = True
    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--timeout",
            "60",
        ],
    )
    assert result.exit_code == 130, (result.output, repr(result.exception))
    plain_output = _plain(result.output)

    match = re.search(r"Could not remove container (soup-aider-[0-9a-f]{12})", plain_output)
    assert match is not None, "Warning with container name was not printed"
    container_name = match.group(1)

    warning_msg = (
        f"Could not remove container {container_name}; remove it with: "
        f"docker rm -f {container_name}"
    )
    assert warning_msg in plain_output

    lines = _ANSI_RE.sub("", result.output).splitlines()
    assert any(f"remove it with: docker rm -f {container_name}" in line for line in lines)


def test_unit_container_name_and_command_validation(tmp_path: Path) -> None:
    import soup_cli.eval.aider_polyglot as aider_eval

    if not hasattr(aider_eval, "new_container_name"):
        pytest.fail("new_container_name() is not implemented on main")

    name1 = aider_eval.new_container_name()
    name2 = aider_eval.new_container_name()
    assert re.fullmatch(r"^soup-aider-[0-9a-f]{12}$", name1)
    assert re.fullmatch(r"^soup-aider-[0-9a-f]{12}$", name2)
    assert name1 != name2

    with pytest.raises(aider_eval.AiderEvalError, match="Docker container name"):
        aider_eval.build_docker_command(
            docker="/usr/bin/docker",
            image="aider-benchmark",
            model="openai/test",
            exercises_dir=tmp_path,
            output_dir=tmp_path,
            threads=1,
            num_tests=1,
            container_name="-bad",
        )

    cmd_no_name = aider_eval.build_docker_command(
        docker="/usr/bin/docker",
        image="aider-benchmark",
        model="openai/test",
        exercises_dir=tmp_path,
        output_dir=tmp_path,
        threads=1,
        num_tests=1,
    )
    assert not any(arg.startswith("--name") for arg in cmd_no_name)

    cmd_with_name = aider_eval.build_docker_command(
        docker="/usr/bin/docker",
        image="aider-benchmark",
        model="openai/test",
        exercises_dir=tmp_path,
        output_dir=tmp_path,
        threads=1,
        num_tests=1,
        container_name=name1,
    )
    assert f"--name={name1}" in cmd_with_name
    rm_idx = cmd_with_name.index("--rm")
    assert cmd_with_name[rm_idx + 1] == f"--name={name1}"


def test_control_success_does_not_call_rm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DOCKER_RUN", "exit0")
    output = tmp_path / "results"
    _result(
        output / "python/exercises/practice/clock/.aider.results.json",
        passed=True,
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
        ],
    )
    assert result.exit_code == 0, result.output
    log_entries = _read_docker_log(tmp_path / "docker_log.jsonl")
    assert not any(entry and entry[0] == "rm" for entry in log_entries)


def test_control_nonzero_exit_does_not_call_rm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_RUN", "exit17")
    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
        ],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "exited with status 17" in _plain(result.output)
    log_entries = _read_docker_log(tmp_path / "docker_log.jsonl")
    assert not any(entry and entry[0] == "rm" for entry in log_entries)


def test_control_tracker_still_closes_on_timeout(
    docker_spy: DockerSpy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    tracker.start_run({}, "cpu", "test", {}, run_id="run_aider_test_tracker")
    tracker.close()

    original_close = ExperimentTracker.close

    def tracking_close(self: Any, *args: Any, **kwargs: Any) -> Any:
        docker_spy.events.append("tracker_close")
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentTracker, "close", tracking_close)
    docker_spy.events.clear()

    result = runner.invoke(
        app,
        [
            "eval",
            "aider",
            "--model",
            "openai/example",
            "--output",
            "results",
            "--exercises-dir",
            "polyglot",
            "--run-id",
            "run_aider_test_tracker",
            "--timeout",
            "60",
        ],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "tracker_close" in docker_spy.events

    if "docker_rm" in docker_spy.events:
        # Tracker close must occur before container removal (Mutation 13)
        assert docker_spy.events.index("tracker_close") < docker_spy.events.index("docker_rm")
