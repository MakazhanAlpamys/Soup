"""Start a child so that it and everything it starts can be ended as one unit.

``Popen.kill()`` ends the direct child only. A process that code run in a sandbox starts
(``subprocess.Popen``, a shell's ``&``, a double fork) is a grandchild: it outlives the
child, keeps running without any time limit, holds the sandbox's output pipes open and, on
Windows, keeps the sandbox directory as its working directory so the directory cannot be
removed.

* POSIX: the child is started as the leader of a new session (``start_new_session=True``),
  so its process group is its own pid and ``os.killpg`` ends the child and every process
  that is still in that group, whether or not it is still linked to the child as a parent.
  ``SIGKILL`` cannot be caught or ignored, so a child that ignores ``SIGTERM`` ends too.
* Windows: the child is assigned to a job object created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Every process it starts afterwards joins the job
  (the job does not allow breakaway), ``TerminateJobObject`` ends all of them, and closing
  the handle ends any that are left.

Best-effort, like the sandbox itself: a process that deliberately leaves the group
(``setsid`` / ``setpgid``) after its parent is gone is out of the POSIX group's reach, and
a process the child starts in the few milliseconds between ``CreateProcess`` and the job
assignment (before the interpreter has run any of the caller's code) is not in the job.
"""

from __future__ import annotations

import contextlib
import functools
import os
import signal
import subprocess
import sys
import threading
import time
import types

_WIN_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_WIN_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WIN_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_WIN_TERMINATED_BY_US = 1
_WAIT_POLL_SECONDS = 0.01


def popen_kwargs() -> dict:
    """Extra ``subprocess.Popen`` keyword arguments for a child run as a ``ProcessTree``."""
    if sys.platform == "win32":
        return {}
    return {"start_new_session": True}


@functools.lru_cache(maxsize=1)
def _job_api() -> types.SimpleNamespace:
    """kernel32's job-object calls with their argument types (Windows only, cached)."""
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class Accounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return types.SimpleNamespace(
        ctypes=ctypes,
        kernel32=kernel32,
        extended_limits=ExtendedLimits,
        accounting=Accounting,
    )


def _create_job_for(process: subprocess.Popen) -> "int | None":
    """A kill-on-close job object holding ``process``, or ``None`` when it cannot be set up."""
    api = _job_api()
    kernel32 = api.kernel32
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = api.extended_limits()
    limits.BasicLimitInformation.LimitFlags = _WIN_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        job,
        _WIN_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        api.ctypes.byref(limits),
        api.ctypes.sizeof(limits),
    ) and kernel32.AssignProcessToJobObject(job, int(process._handle))
    if not ok:
        kernel32.CloseHandle(job)
        return None
    return int(job)


class ProcessTree:
    """A started child plus everything it starts, ended together.

    Start the child with ``**popen_kwargs()``, then wrap it. ``kill`` is idempotent and
    never raises; ``wait_gone`` waits (Windows) until the job holds no process; ``close``
    ends whatever is left and releases the job. Safe to call from several threads.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._lock = threading.Lock()
        self._job: "int | None" = None
        if sys.platform == "win32":
            with contextlib.suppress(OSError, ValueError, AttributeError):
                self._job = _create_job_for(process)

    @property
    def has_job(self) -> bool:
        """Windows: whether the child is held by a job object. POSIX: always False."""
        return self._job is not None

    def kill(self) -> None:
        """End the child and every process it started that this can still reach."""
        with self._lock:
            if sys.platform == "win32":
                if self._job is not None:
                    with contextlib.suppress(OSError):
                        _job_api().kernel32.TerminateJobObject(
                            self._job, _WIN_TERMINATED_BY_US
                        )
            else:
                with contextlib.suppress(OSError):
                    os.killpg(self._process.pid, signal.SIGKILL)
        # The direct child as well: covers a group or job that could not be used.
        with contextlib.suppress(OSError):
            self._process.kill()

    def wait_gone(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds until no process of the tree is left.

        Windows only: the processes a job ends release their handles (the working
        directory among them) a moment after ``TerminateJobObject`` returns. POSIX has
        nothing to wait for: the signal is delivered and the files are free. Returns
        whether the tree is gone.
        """
        with self._lock:
            job = self._job
        if job is None:
            return True
        api = _job_api()
        deadline = time.monotonic() + timeout
        while True:
            info = api.accounting()
            ok = api.kernel32.QueryInformationJobObject(
                job,
                _WIN_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                api.ctypes.byref(info),
                api.ctypes.sizeof(info),
                None,
            )
            if not ok or info.ActiveProcesses == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_WAIT_POLL_SECONDS)

    def close(self) -> None:
        """Release the job; closing the handle ends any process still in it."""
        with self._lock:
            job, self._job = self._job, None
        if job is not None:
            with contextlib.suppress(OSError):
                _job_api().kernel32.CloseHandle(job)
