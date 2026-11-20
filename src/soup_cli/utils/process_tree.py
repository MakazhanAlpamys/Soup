"""Start a child so that it and everything it starts can be ended as one unit.

``Popen.kill()`` ends the direct child only. A process that code run in a sandbox starts
is a grandchild: it outlives the child, keeps running without any time limit, holds the
sandbox's output pipes open and, on Windows, keeps the sandbox directory as its working
directory so the directory cannot be removed.

* POSIX: the child is started as the leader of a new session (``start_new_session=True``),
  so its process group is its own pid and ``os.killpg`` ends the child and every process
  that is still in that group, whether or not it is still linked to the child as a parent.
  ``SIGKILL`` cannot be caught or ignored, so a child that ignores ``SIGTERM`` ends too.
* Windows: the child is created suspended, assigned to a job object created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` and only then allowed to run, so it has not
  started anything before it is in the job. Every process it starts afterwards joins the
  job (the job does not allow breakaway), including the interpreter that a launcher (a
  virtual environment's ``python.exe``) starts, ``TerminateJobObject`` ends all of them,
  and closing the handle ends any that are left.

``wait_gone`` reports True only when it has verified that nothing of the tree is left; when
it cannot ask (no job could be created, a query failed) it reports False.

Best-effort, like the sandbox itself: a process that moves itself into another process
group is out of the POSIX group's reach, and a Windows host that refuses the job (for
example one that does not allow nested jobs) leaves only the direct child to end. On POSIX
the child no longer shares the terminal's foreground group, so Ctrl-C and a terminal
hangup do not reach it directly: an interrupted caller ends the tree in its ``finally``,
but a caller that is killed outright runs no ``finally`` and the group lives until its
code ends it (Windows ends the job when the controller exits).
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
_WIN_CREATE_SUSPENDED = 0x00000004
_WAIT_POLL_SECONDS = 0.01


def popen_kwargs() -> dict:
    """Extra ``subprocess.Popen`` keyword arguments for a child run as a ``ProcessTree``.

    Windows: the child starts suspended; ``ProcessTree`` lets it run once it is in the job.
    """
    if sys.platform == "win32":
        return {"creationflags": _WIN_CREATE_SUSPENDED}
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
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    return types.SimpleNamespace(
        ctypes=ctypes,
        kernel32=kernel32,
        ntdll=ntdll,
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

    Start the child with ``**popen_kwargs()``, then wrap it at once (on Windows the child
    waits, suspended, until this constructor has put it in the job). ``kill`` is
    idempotent and never raises; ``wait_gone`` waits until no process of the tree is left
    and says whether that was verified; ``close`` ends whatever is left and releases the
    job. Safe to call from several threads.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._lock = threading.Lock()
        self._job: "int | None" = None
        if sys.platform == "win32":
            with contextlib.suppress(OSError, ValueError, AttributeError):
                self._job = _create_job_for(process)
            self._resume()

    def _resume(self) -> None:
        """Windows: let the child run (it was started suspended by ``popen_kwargs``).

        Harmless for a child that is already running. A child that cannot be resumed
        would wait for ever, so it is ended instead.
        """
        try:
            status = _job_api().ntdll.NtResumeProcess(int(self._process._handle))
        except (OSError, ValueError, AttributeError):
            status = -1
        if status != 0:
            with contextlib.suppress(OSError):
                self._process.kill()

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

        The processes that were ended release their handles (the working directory among
        them) a moment after the kill returns. Returns True only when it was verified that
        none is left (Windows: the job holds no active process; POSIX: the process group
        is empty). Returns False when processes are still there after ``timeout`` and
        also when it cannot tell: no job on Windows, a failed query, a group that cannot
        be probed.
        """
        if sys.platform == "win32":
            return self._wait_job_empty(timeout)
        deadline = time.monotonic() + timeout
        while True:
            try:
                os.killpg(self._process.pid, 0)
            except ProcessLookupError:
                return True
            except OSError:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(_WAIT_POLL_SECONDS)

    def _wait_job_empty(self, timeout: float) -> bool:
        with self._lock:
            job = self._job
        if job is None:
            return False
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
            if not ok:
                return False
            if info.ActiveProcesses == 0:
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
