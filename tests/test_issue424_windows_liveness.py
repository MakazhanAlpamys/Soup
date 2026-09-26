"""Windows STILL_ACTIVE (259) exit-code ambiguity in process liveness (issue #424).

`GetExitCodeProcess` reports `STILL_ACTIVE` (259) for a running process, but 259
is also a legal process exit code — a process that exits *with* 259 used to be
indistinguishable from one still running and read ALIVE forever. The fix uses
`WaitForSingleObject(handle, 0)`: the process handle becomes signalled the
instant the process exits, independent of its exit code.
"""

from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from soup_cli.experiment import tracker as tracker_module
from soup_cli.mcp_server import execution as execution_module
from soup_cli.utils.process_liveness import process_is_alive

_STILL_ACTIVE_EXIT_CODE = 259

_WIN32_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="exercises the Windows-only liveness primitive"
)


@_WIN32_ONLY
def test_exit_code_259_reads_as_dead():
    """The exact #424 repro: a process that exits with code 259 must read DEAD."""
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.exit({_STILL_ACTIVE_EXIT_CODE})"]
    )
    proc.wait()
    assert proc.returncode == _STILL_ACTIVE_EXIT_CODE

    assert process_is_alive(proc.pid) is False


@_WIN32_ONLY
def test_control_ordinary_exit_code_reads_as_dead():
    """CONTROL: an ordinary exit code must also read DEAD (sanity check)."""
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    proc.wait()

    assert process_is_alive(proc.pid) is False


@_WIN32_ONLY
def test_control_live_process_reads_as_alive():
    """CONTROL: a genuinely running process must still read ALIVE."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Give it a moment to actually start before checking.
        time.sleep(0.2)
        assert process_is_alive(proc.pid) is True
    finally:
        proc.terminate()
        proc.wait()


@_WIN32_ONLY
def test_win32_never_falls_back_to_os_kill():
    """A revert to `os.kill(pid, 0)` on Windows must fail this test by name,
    not by the whole CI matrix dying with a stray KeyboardInterrupt (`os.kill`
    with signal 0 sends a console Ctrl+C on Windows rather than checking
    existence).
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.2)
        with patch("soup_cli.utils.process_liveness.os.kill") as mock_kill:
            process_is_alive(proc.pid)
        mock_kill.assert_not_called()
    finally:
        proc.terminate()
        proc.wait()


@pytest.mark.parametrize(
    "bad_pid", ["not-a-pid", 0, -1, True, False, None, 3.5]
)
def test_invalid_pid_returns_false_not_raise(bad_pid):
    """A non-int, bool, or non-positive pid must not crash a caller."""
    assert process_is_alive(bad_pid) is False


def test_tracker_and_execution_share_one_primitive():
    """Both call sites must go through the same shared primitive, not two
    copies that can drift (the original defect this issue reports)."""
    assert tracker_module._process_is_alive is process_is_alive
    assert execution_module._pid_is_alive is process_is_alive
