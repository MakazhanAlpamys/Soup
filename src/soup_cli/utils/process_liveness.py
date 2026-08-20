"""Cross-platform process-liveness check, shared by the experiment tracker and
the MCP execution manager (issue #424).

Extracted from ``experiment/tracker.py::_windows_process_is_alive`` and
``mcp_server/execution.py::_windows_pid_is_alive`` — the two were byte-identical
copies whose docstrings had already started to diverge. ``utils/adapter_fuse.py``
is the precedent for pulling a duplicated primitive out from under two callers
before they drift further.
"""

from __future__ import annotations

import os
import sys

# Windows liveness constants. GetExitCodeProcess returns STILL_ACTIVE for a
# running process, but STILL_ACTIVE (259) is also a LEGAL process exit code —
# a process that exits *with* code 259 used to be indistinguishable from one
# still running by exit-code alone. WaitForSingleObject is the disambiguator:
# the process handle becomes signalled the instant the process terminates,
# independent of what its exit code was.
_WIN_STILL_ACTIVE = 259
_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN_SYNCHRONIZE = 0x00100000
_WIN_ERROR_ACCESS_DENIED = 5
_WIN_WAIT_OBJECT_0 = 0x0
_WIN_WAIT_TIMEOUT = 0x102


def _windows_process_is_alive(pid: int) -> bool:
    """Liveness check for Windows, where ``os.kill(pid, 0)`` cannot be used.

    On Windows ``CTRL_C_EVENT`` is ``0``, so ``os.kill(pid, 0)`` routes signal 0
    to ``GenerateConsoleCtrlEvent`` — it sends a console Ctrl+C to the process
    group and never checks existence (it raises KeyboardInterrupt in-process
    instead). Query the process directly: open a handle with
    ``PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE`` (``WaitForSingleObject``
    needs ``SYNCHRONIZE`` on top of the query right, or the wait itself fails)
    and wait on it with a zero timeout — the handle becomes signalled
    (``WAIT_OBJECT_0``) the instant the process exits, whatever its exit code.
    A failed open means the pid is gone (ERROR_INVALID_PARAMETER) unless it is
    only ERROR_ACCESS_DENIED, which means the process exists but we lack rights.

    ``GetExitCodeProcess`` alone cannot make this call: it reports ``STILL_ACTIVE``
    (259) for a running process, but 259 is also a legal exit code, so a process
    that exits *with* 259 would read as alive forever (issue #424). It is kept
    here only as a fallback for the case where the wait itself fails.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(
        _WIN_PROCESS_QUERY_LIMITED_INFORMATION | _WIN_SYNCHRONIZE, False, pid
    )
    if not handle:
        return ctypes.get_last_error() == _WIN_ERROR_ACCESS_DENIED
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result == _WIN_WAIT_OBJECT_0:
            # Signalled: the process has exited, regardless of its exit code.
            return False
        if wait_result == _WIN_WAIT_TIMEOUT:
            # Not signalled within 0ms: still running.
            return True
        # WAIT_FAILED or an unexpected value — fall back to the exit-code read.
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True  # opened but could not query — assume alive (conservative)
        return exit_code.value == _WIN_STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def process_is_alive(pid: int) -> bool:
    """Return True if a process with this PID currently exists.

    POSIX uses signal 0 — a genuine existence/permission check that delivers no
    signal: a PID owned by another user raises PermissionError and counts as
    alive; only a missing process (ProcessLookupError / ESRCH) is dead. Any
    other OSError is treated as not-alive so a corrupt record does not wedge a
    caller (reconcile-on-read, the MCP capacity gate) shut forever.

    Windows needs a different primitive entirely: ``os.kill(pid, 0)`` there
    sends a console Ctrl+C (``CTRL_C_EVENT == 0``) rather than checking
    existence, so liveness goes through OpenProcess + WaitForSingleObject (see
    ``_windows_process_is_alive``).

    A non-int, bool, or non-positive ``pid`` returns False rather than reaching
    either platform primitive: on POSIX, ``os.kill(0, 0)`` targets the caller's
    own process group and would read permanently alive, and on any platform a
    corrupt/invalid pid must not be able to crash a caller (issue #424).

    Known limitation — PID reuse: a dead run whose PID has since been recycled
    by an unrelated process reads as alive forever, so reconciliation never
    fires for it. That is inherent to PID-liveness (a recorded process
    start-time comparison would be needed to close it) and is out of scope for
    issue #424.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
