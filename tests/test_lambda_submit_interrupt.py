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
        # Each child answers from its own copy of wait_results, so a caller that
        # re-spawned on every interrupt would loop forever instead of failing.
        # One controller per submit is the contract; say so here.
        if _FakePopen.instances:
            raise AssertionError("the controller must be spawned once")
        return _FakePopen(argv, env=kwargs.get("env"), wait_results=wait_results)

    def forbidden_run(*args, **kwargs):
        raise AssertionError("submit_lambda_run must not use subprocess.run")

    monkeypatch.setattr(subprocess, "Popen", factory)
    monkeypatch.setattr(subprocess, "run", forbidden_run)


def test_keyboard_interrupt_keeps_waiting_and_never_kills(monkeypatch, capsys):
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _patch_popen(monkeypatch, [KeyboardInterrupt(), KeyboardInterrupt(), 0])

    # An escaping KeyboardInterrupt would abort the whole pytest session rather
    # than fail this test, so it is converted into an ordinary failure.
    try:
        result = submit_lambda_run(_plan(), env=dict(_ENV))
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped submit_lambda_run")
    assert result == 0

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


class _StderrInterruptedOnce:
    """A stderr whose first ``write`` raises, as a Ctrl+C landing on it does."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self._raised = False

    def write(self, text):
        self.writes.append(text)
        if not self._raised:
            self._raised = True
            raise KeyboardInterrupt
        return len(text)

    def flush(self):
        pass


class _InterruptOnFirstWaitLookup:
    """A child whose first ``.wait`` ATTRIBUTE access raises KeyboardInterrupt.

    That is the shape of a SIGINT delivered the instant ``Popen`` returns: the
    controller is already running and the parent has not reached the statement
    that waits for it.
    """

    def __init__(self, argv, env=None):
        self.argv = argv
        self.env = env
        self.lookups = 0
        self.wait_calls = 0
        self.kill_calls = 0
        self.terminate_calls = 0

    def __getattr__(self, name):
        if name != "wait":
            raise AttributeError(name)
        self.__dict__["lookups"] += 1
        if self.lookups == 1:
            raise KeyboardInterrupt
        return self._wait

    def _wait(self, timeout=None):
        self.wait_calls += 1
        return 0

    def kill(self):
        self.kill_calls += 1

    def terminate(self):
        self.terminate_calls += 1


def test_interrupt_while_writing_the_notice_still_waits(monkeypatch):
    """A Ctrl+C that lands on the notice write must not unwind the parent (#1073).

    The notice is written from the interrupt handler, which a ``try`` around
    ``proc.wait()`` alone cannot cover: the second KeyboardInterrupt escaped
    ``submit_lambda_run`` while the controller was still terminating the paid
    instance. It stands for the same class of gap as the window between
    ``Popen`` returning and the waiting loop being entered -- code that runs
    while a child exists, outside the region that guarantees a wait.
    """
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _patch_popen(monkeypatch, [KeyboardInterrupt(), 0])
    stderr = _StderrInterruptedOnce()
    monkeypatch.setattr(sys, "stderr", stderr)

    try:
        result = submit_lambda_run(_plan(), env=dict(_ENV))
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped submit_lambda_run")
    assert result == 0

    proc = _FakePopen.instances[0]
    assert proc.wait_calls == 2
    assert proc.kill_calls == 0
    assert proc.terminate_calls == 0
    assert any("terminate the instance" in text for text in stderr.writes)


def test_interrupt_at_spawn_time_still_waits(monkeypatch):
    """The child exists the moment ``Popen`` returns, so every path ends in a wait."""
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _FakePopen.instances = []
    children: list[_InterruptOnFirstWaitLookup] = []

    def factory(argv, **kwargs):
        if children:
            raise AssertionError("the controller must be spawned once")
        child = _InterruptOnFirstWaitLookup(argv, env=kwargs.get("env"))
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", factory)

    try:
        result = submit_lambda_run(_plan(), env=dict(_ENV))
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped submit_lambda_run")
    assert result == 0

    assert len(children) == 1, "the controller must not be spawned twice"
    child = children[0]
    assert child.lookups == 2
    assert child.wait_calls == 1
    assert child.kill_calls == 0
    assert child.terminate_calls == 0


def test_system_exit_also_waits_for_the_controller(monkeypatch, capsys):
    """A handler that turns a signal into SystemExit must not skip the cleanup either."""
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _patch_popen(monkeypatch, [SystemExit(1), 0])

    assert submit_lambda_run(_plan(), env=dict(_ENV)) == 0
    proc = _FakePopen.instances[0]
    assert proc.wait_calls == 2
    assert proc.kill_calls == 0
    assert proc.terminate_calls == 0
    assert "terminate the instance" in capsys.readouterr().err


def test_interrupt_before_the_child_exists_propagates(monkeypatch):
    """No controller was started, so there is nothing to wait for."""
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    calls = []

    def factory(argv, **kwargs):
        calls.append(argv)
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", factory)

    with pytest.raises(KeyboardInterrupt):
        submit_lambda_run(_plan(), env=dict(_ENV))
    assert len(calls) == 1, "a failed spawn must not be retried"


def test_a_failing_wait_surfaces_instead_of_spinning(monkeypatch):
    """Interrupts are absorbed; a broken ``wait()`` must not loop forever."""
    from soup_cli.cloud.lambda_labs import submit_lambda_run

    _patch_popen(monkeypatch, [OSError("waitpid failed")])

    with pytest.raises(OSError, match="waitpid failed"):
        submit_lambda_run(_plan(), env=dict(_ENV))
    assert _FakePopen.instances[0].wait_calls == 1


_CONTROLLER = """\
import os
import pathlib
import time

pathlib.Path("ready").write_text("ready")
try:
    while True:
        time.sleep(0.1)
finally:
    time.sleep(1)
    # Written to a temp name in the same directory and renamed into place, so a
    # missing marker means the parent exited first and can never mean "the
    # marker was caught half written".
    tmp = pathlib.Path("marker.tmp")
    tmp.write_text("terminated")
    os.replace(tmp, "marker")
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
