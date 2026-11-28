"""Setup lifecycle controls; local POSIX processes only."""

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.harness.streaming_controls.setup_supervisor import supervise

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")


def script(tmp_path, body):
    path = tmp_path / "input.sh"
    path.write_text(body)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_exit_code_and_log_survive_failed_setup(tmp_path):
    path, sha = script(tmp_path, "echo diagnosis; exit 141\n")
    result = supervise(path, sha, tmp_path / "out", 2, time.time() + 10)
    assert result["status"] == "FAILED" and result["returncode"] == 141
    assert (tmp_path / "out/stdout_stderr.log").read_text() == "diagnosis\n"
    assert json.loads((tmp_path / "out/result.json").read_text()) == result


def test_timeout_does_not_turn_into_success(tmp_path):
    path, sha = script(tmp_path, "echo entered; exec sleep 20\n")
    result = supervise(path, sha, tmp_path / "out", 0.2, time.time() + 10)
    assert result["status"] == "TIMEOUT"
    assert result["returncode"] == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)


@pytest.mark.parametrize("problem", ["hash", "expired", "oversized", "existing"])
def test_invalid_authority_never_executes_script(tmp_path, problem):
    path, sha = script(tmp_path, "touch " + shlex.quote(str(tmp_path / "executed")) + "\n")
    out = tmp_path / "out"
    if problem == "existing":
        out.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        supervise(
            path,
            "0" * 64 if problem == "hash" else sha,
            out,
            1801 if problem == "oversized" else 2,
            time.time() - 1 if problem == "expired" else time.time() + 10,
        )
    assert not (tmp_path / "executed").exists()


def test_closed_client_pipe_and_hangup_do_not_kill_setup(tmp_path):
    path, sha = script(tmp_path, "echo first; sleep 0.7; echo last\n")
    out = tmp_path / "out"
    module = Path(__file__).parents[1] / "benchmarks/harness/streaming_controls/setup_supervisor.py"
    child = subprocess.Popen(
        [
            sys.executable,
            str(module),
            "--script",
            str(path),
            "--sha256",
            sha,
            "--output",
            str(out),
            "--max-seconds",
            "3",
            "--deadline-unix-s",
            str(time.time() + 10),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        child.stdout.close()
        end = time.monotonic() + 4
        while time.monotonic() < end:
            receipt = out / "result.json"
            if receipt.exists() and json.loads(receipt.read_text())["status"] == "RUNNING":
                break
            time.sleep(0.01)
        else:
            pytest.fail("setup did not start")
        os.kill(child.pid, signal.SIGHUP)
        assert child.wait(timeout=5) == 0, child.stderr.read().decode()
        assert (out / "stdout_stderr.log").read_text() == "first\nlast\n"
        assert json.loads(receipt.read_text())["status"] == "PASS"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stderr.close()


def test_supervisor_termination_stops_its_running_child(tmp_path):
    path, sha = script(tmp_path, "exec sleep 20\n")
    out = tmp_path / "out"
    module = Path(__file__).parents[1] / "benchmarks/harness/streaming_controls/setup_supervisor.py"
    child = subprocess.Popen(
        [
            sys.executable,
            str(module),
            "--script",
            str(path),
            "--sha256",
            sha,
            "--output",
            str(out),
            "--max-seconds",
            "10",
            "--deadline-unix-s",
            str(time.time() + 15),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        end = time.monotonic() + 4
        while time.monotonic() < end:
            receipt = out / "result.json"
            if receipt.exists():
                state = json.loads(receipt.read_text())
                if state["status"] == "RUNNING":
                    break
            time.sleep(0.01)
        else:
            pytest.fail("setup did not start")
        child.terminate()
        assert child.wait(timeout=5) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(state["pid"], 0)
        assert json.loads(receipt.read_text())["status"] == "FAILED"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_successful_shell_cannot_leave_background_writer_alive(tmp_path):
    marker = tmp_path / "escaped"
    path, sha = script(
        tmp_path, "(sleep 0.6; echo leaked > " + shlex.quote(str(marker)) + ") &\nexit 0\n"
    )
    result = supervise(path, sha, tmp_path / "out", 2, time.time() + 10)
    assert result["status"] == "PASS"
    assert result["group_cleanup_confirmed"] is True
    time.sleep(0.8)
    assert not marker.exists()


def test_measurement_admits_5400_seconds_without_expanding_setup_limit(tmp_path):
    path, sha = script(tmp_path, "echo measured; exit 0\n")
    result = supervise(path, sha, tmp_path / "out", 5400, time.time() + 10, purpose="measurement")
    assert result["status"] == "PASS"
    assert result["purpose"] == "measurement"
    assert result["group_cleanup_confirmed"]
    assert (tmp_path / "out/stdout_stderr.log").read_text() == "measured\n"


@pytest.mark.parametrize(
    "purpose,bound,message",
    [
        ("setup", 5400, "Unexpired deadline and setup bound <=1800s required"),
        ("measurement", 5401, "Unexpired deadline and measurement bound <=5400s required"),
        ("unknown", 1, "Unknown supervisor purpose"),
    ],
)
def test_phase_bound_rejection_happens_before_launch(tmp_path, purpose, bound, message):
    marker = tmp_path / "executed"
    path, sha = script(tmp_path, "touch " + shlex.quote(str(marker)) + "\n")
    with pytest.raises(ValueError, match=f"^{message}$"):
        supervise(path, sha, tmp_path / "out", bound, time.time() + 10, purpose=purpose)
    assert not marker.exists()
    assert not (tmp_path / "out").exists()


def test_measurement_obeys_shared_deadline_despite_larger_phase_bound(tmp_path):
    path, sha = script(tmp_path, "exec sleep 20\n")
    result = supervise(path, sha, tmp_path / "out", 5400, time.time() + 0.2, purpose="measurement")
    assert result["status"] == "TIMEOUT"
    assert result["group_cleanup_confirmed"]
    assert result["elapsed_seconds"] < 3
