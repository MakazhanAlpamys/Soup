"""How the loop starts the children it waits for, and how it names the way one ended.

The watch daemon of ``soup loop watch --detach`` has no console (``DETACHED_PROCESS``; see
``loop_detach.detached_popen_kwargs``). On Windows a console program started by a process
with no console gets a NEW console of its own, with a WINDOW: measured on this project's dev
box, a stub child of a detached parent reports ``GetConsoleWindow() != 0`` and a console
process list of one. That window is a console nobody is looking at, owned by one child, and
the only way a console control event (Ctrl+C, Ctrl+Break, close) can reach it. The ``soup
train`` child of a detached watcher died in exactly that shape: exit ``0xC000013A``
(``STATUS_CONTROL_C_EXIT``), nothing in its log but ``Loading config from ...``, while the
same command in the foreground ran (review 5, F2).

Every child the loop starts therefore gets ``CREATE_NO_WINDOW`` on Windows: it still has a
console (so a ``git`` or ``nvidia-smi`` it starts in turn flashes no window of its own), but
that console has no window, no keyboard and no close button. POSIX is unchanged: nothing is
added (``start_new_session`` stays the detached daemon's own business).

A child that ends by a signal (POSIX, a negative return code) or by an NTSTATUS (Windows, an
unsigned code of ``0xC0000000`` and up) is reported by name and code (``describe_exit``), so
``last_failure`` says ``STATUS_CONTROL_C_EXIT`` and not a bare ``3221225786``.
"""

from __future__ import annotations

import os
import signal
import sys
from typing import Optional

#: ``subprocess.CREATE_NO_WINDOW``, spelled out for the interpreters that lack the constant.
CREATE_NO_WINDOW = 0x08000000


def python_child_argv(*arguments: str) -> "list[str]":
    """The argv of a Python child of the loop: this interpreter, ``-P``, then ``arguments``.

    ``python -m ...`` and ``python -c ...`` put the current directory first on ``sys.path``, so a
    file in the project directory named like a library (``torch.py``, ``yaml.py``) would be
    imported by the child, which carries the daemon's environment. ``-P`` (Python 3.11, the
    declared floor) leaves the current directory off ``sys.path``. Every child the loop starts is
    built here.
    """
    return [sys.executable, "-P", *arguments]

#: The NTSTATUS codes a Python or CUDA child ends with, by name and in words.
_NTSTATUS = {
    0xC000013A: (
        "STATUS_CONTROL_C_EXIT",
        "the process was ended by a console Ctrl+C, Ctrl+Break or close event",
    ),
    0xC0000005: ("STATUS_ACCESS_VIOLATION", "the process crashed on an invalid memory access"),
    0xC00000FD: ("STATUS_STACK_OVERFLOW", "the process overflowed its stack"),
    0xC0000017: ("STATUS_NO_MEMORY", "the process could not get the memory it asked for"),
    0xC000001D: ("STATUS_ILLEGAL_INSTRUCTION", "the process ran an instruction the CPU lacks"),
    0xC0000135: ("STATUS_DLL_NOT_FOUND", "a DLL the process needs was not found"),
    0xC0000139: ("STATUS_ENTRYPOINT_NOT_FOUND", "a DLL lacks a function the process needs"),
    0xC0000142: ("STATUS_DLL_INIT_FAILED", "a DLL of the process failed to initialise"),
    0xC0000374: ("STATUS_HEAP_CORRUPTION", "the process corrupted its heap"),
    0xC0000409: ("STATUS_STACK_BUFFER_OVERRUN", "the process aborted (a fast-fail)"),
    0xC0000194: ("STATUS_POSSIBLE_DEADLOCK", "the process hit a possible deadlock"),
}


#: POSIX signal names whose number is the same on Linux and macOS, for a Windows machine that
#: reads a return code it did not produce (``signal.Signals`` there lacks most of them).
_COMMON_SIGNALS = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV",
    13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
}


def child_popen_kwargs(os_name: Optional[str] = None) -> dict:
    """``subprocess`` keyword arguments for a child of the loop: the creation flags, if any.

    Windows: ``CREATE_NO_WINDOW`` (a console with no window; see the module docstring). Any
    other system: ``{}``. ``os_name`` replaces ``os.name`` (tests).
    """
    if (os.name if os_name is None else os_name) == "nt":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


def describe_exit(code: object, os_name: Optional[str] = None) -> str:
    """``exit code 1``, or the code with the name of the signal / NTSTATUS that ended the child.

    ``exit code 3221225786 (STATUS_CONTROL_C_EXIT 0xC000013A: the process was ended by ...)`` for
    a Windows status, ``exit code -9 (killed by signal SIGKILL)`` for a POSIX signal. The text
    always starts ``exit code <code>`` (what a person greps for). A status is recognised by its
    size (an exit status on POSIX is 0..255, a signal is -1..-64), so the name does not depend on
    the system this runs on; only a small negative code needs ``os_name`` (a signal on POSIX,
    an ordinary code on Windows). Anything that is not an int is ``no exit code``.
    """
    if isinstance(code, bool) or not isinstance(code, int):
        return "no exit code"
    if code >= 0x80000000 or code < -255:
        status = code & 0xFFFFFFFF  # the signed 32-bit reading of the same status
        if status in _NTSTATUS:
            name, words = _NTSTATUS[status]
            return f"exit code {code} ({name} 0x{status:08X}: {words})"
        return f"exit code {code} (NTSTATUS 0x{status:08X})"
    if code < 0 and (os.name if os_name is None else os_name) != "nt":
        try:
            name = signal.Signals(-code).name
        except ValueError:  # this system's enum lacks it (Windows has no SIGKILL, say)
            name = _COMMON_SIGNALS.get(-code, "")
        return f"exit code {code} (killed by signal {name or -code})"
    return f"exit code {code}"
