"""UTF-8 stdio bootstrap for Windows CLI sessions (v0.40.1 Part A).

Windows defaults `sys.stdout`/`sys.stderr` to the OEM codepage (cp1251 /
cp1252 / cp932 / …), which crashes Rich/Typer with `UnicodeEncodeError` the
moment we print β, ✓, → or any box-drawing character. This single call at the
CLI entrypoint re-encodes both streams to UTF-8 before any Rich console is
constructed, eliminating the entire mojibake / crash family in one place.

POSIX terminals are already UTF-8, so the function is effectively a no-op
there — guarded by ``sys.platform == "win32"`` to keep behaviour explicit.
"""

from __future__ import annotations

import os
import sys


def force_utf8_stdio() -> None:
    """Force UTF-8 on stdout / stderr; safe to call multiple times.

    On Windows: reconfigures the text streams to UTF-8 and seeds
    ``PYTHONIOENCODING=utf-8`` so child processes inherit. On POSIX: no-op.

    All errors are swallowed — if reconfigure fails (redirected pipe, frozen
    binary, weird shim) we prefer that the CLI continue to start over crashing
    at import time.
    """
    if sys.platform != "win32":
        return

    # Subprocess inheritance — preserve user override if any.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            # Detached / non-text streams — best effort only.
            continue
