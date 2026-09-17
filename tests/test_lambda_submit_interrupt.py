"""`soup train --cloud lambda --cloud-submit` survives Ctrl+C until the controller exits.

The controller terminates the paid Lambda instance in its ``finally`` block. A
Ctrl+C in the terminal reaches the controller AND the parent `soup` process;
``subprocess.run`` answers the parent's KeyboardInterrupt by killing the
controller 0.25 s later, which can leave the instance running. The parent must
keep waiting instead.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ENV = {
    "LAMBDA_API_KEY": "test-key",
    "LAMBDA_SSH_KEY_NAME": "test-ssh-key",
    "LAMBDA_SSH_PRIVATE_KEY": "/tmp/test-private-key",
}
_SRC = Path(__file__).resolve().parents[1] / "src"


def _plan(stub_path: str = "x.py"):
    from soup_cli.cloud._common import CloudPlan

    return CloudPlan(
        cloud="lambda",
        gpu="a100",
        output_dir="./out",
        stub_path=stub_path,
        stub_text="",
        run_command=f"python {stub_path}",
    )


class _FakePopen:
    instances: list[_FakePopen] = []

    def __init__(self, argv, *, env=None, wait_results=()):
        self.argv = argv
        self.env = env
        self.wait_results = list(wait_results)
        self.wait_calls = 0
        self.kill_calls = 0
        self.terminate_calls = 0
        _FakePopen.instances.append(self)

    def wait(self, timeout=None):
        self.wait_calls += 1
        result = self.wait_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def kill(self):
        self.kill_calls += 1

    def terminate(self):
        self.terminate_calls += 1

    def send_signal(self, sig):
        self.kill_calls += 1


def _patch_popen(monkeypatch, wait_results):
    _FakePopen.instances = []

    def factory(argv, **kwargs):
        return _FakePopen(argv, env=kwargs.get("env"), wait_results=wait_results)

    def forbidden_run(*args, **kwargs):
        raise AssertionError("submit_lambda_run must not use subprocess.run")

    monkeypatch.setattr(subprocess, "Popen", factory)
    monkeypatch.setattr(subprocess, "run", forbidden_run)


def test_keyboard_interrupt_keeps_waiting_and_never_kills(monkeypatch, capsys):
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _patch_popen(monkeypatch, [KeyboardInterrupt(), KeyboardInterrupt(), 0])

    assert submit_lambda_run(_plan(), env=dict(_ENV)) == 0

    assert len(_FakePopen.instances) == 1
    proc = _FakePopen.instances[0]
    assert proc.argv == [sys.executable, "x.py"]
    assert proc.env == _ENV
    assert proc.wait_calls == 3
    assert proc.kill_calls == 0
    assert proc.terminate_calls == 0
    err = " ".join(capsys.readouterr().err.split())
    assert "waiting for the Lambda controller to terminate the instance" in err


def test_returns_controller_exit_code(monkeypatch):
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _patch_popen(monkeypatch, [3])

    assert submit_lambda_run(_plan(), env=dict(_ENV)) == 3
    assert _FakePopen.instances[0].wait_calls == 1


_CONTROLLER = """\
import pathlib
import time

pathlib.Path("ready").write_text("ready")
try:
    while True:
        time.sleep(0.1)
finally:
    time.sleep(1)
    pathlib.Path("marker").write_text("terminated")
"""

_PARENT = """\
import os
import sys

from soup_cli.cloud._common import CloudPlan
from soup_cli.cloud.lambda_labs import submit_lambda_run

plan = CloudPlan(
    cloud="lambda", gpu="a100", output_dir="./out", stub_path="x.py",
    stub_text="", run_command="python x.py",
)
env = dict(os.environ)
env.update(LAMBDA_API_KEY="k", LAMBDA_SSH_KEY_NAME="n", LAMBDA_SSH_PRIVATE_KEY="/tmp/p")
sys.exit(submit_lambda_run(plan, env=env))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group SIGINT")
def test_real_sigint_lets_controller_finish_cleanup(tmp_path):
    (tmp_path / "x.py").write_text(_CONTROLLER, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(_SRC))
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT],
        cwd=tmp_path,
        env=env,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").exists():
            assert parent.poll() is None, "parent exited before the controller started"
            assert time.monotonic() < deadline, "controller never started"
            time.sleep(0.05)
        # What a terminal Ctrl+C does: SIGINT to the whole foreground group.
        os.killpg(parent.pid, signal.SIGINT)
        parent.wait(timeout=15)
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait()
    assert (tmp_path / "marker").read_text() == "terminated"
