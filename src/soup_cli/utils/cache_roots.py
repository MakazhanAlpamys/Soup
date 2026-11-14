"""Operator-set cache roots: one containment rule, one warning (#2610).

``SOUP_LAYER_STREAM_CACHE_DIR`` (layer-streaming shard cache) and
``SOUP_SPECTRUM_CACHE_DIR`` (spectrum scan cache) pick the folder a cache lives in.
Both are bounded to ``$HOME``, the working directory and the temp directory: a value
outside all three, or one carrying a C0 control character, is dropped and the default
(``~/.soup/layer-stream``, ``~/.soup/spectrum``) is used. The boundary is deliberate
and unchanged here.

What this module adds is the report. Dropping an operator's value without a word
sent a shard cache to the small system disk with nothing in the output saying why, so
the fallback now writes one WARNING per process for each variable and value, naming
the variable, the rejected value (rendered so it cannot move the terminal), the reason
and the folder used instead. The resolvers run several times per process (``soup train``
validates the stripe roots, builds the shard folder and runs the doctor probes), hence
the once-guard.

The channel is ``logging``, as in the neighbouring ``layer_shard`` / ``spectrum_scan``
warnings. ``soup``'s CLI callback sets the root logger's level from ``--log-level`` and
installs no root handler, so a ``soup_cli.*`` record reaches the user through the stdlib
last-resort handler on stderr at the default tier and is dropped at ``quiet``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading

logger = logging.getLogger(__name__)

LAYER_STREAM_CACHE_ENV = "SOUP_LAYER_STREAM_CACHE_DIR"
SPECTRUM_CACHE_ENV = "SOUP_SPECTRUM_CACHE_DIR"

REASON_OUTSIDE = "it is outside $HOME, the working directory and the temp directory"
REASON_CONTROL = "it contains control characters"

_FIX_OUTSIDE = (
    "To use that folder, run soup from a working directory on the same drive, "
    "or put the folder under $HOME."
)
_FIX_CONTROL = "Set it again without the control characters."

#: Characters of the rejected value shown in the warning. A Windows environment value
#: can run to 32 KB; the head of a path is what identifies it.
_MAX_SHOWN = 200

#: ``(variable, value)`` pairs already reported by this process.
_WARNED: set[tuple[str, str]] = set()
_WARNED_LOCK = threading.Lock()


def containment_bounds() -> list[str]:
    """Real paths of the three folders an operator-set cache root may live under."""
    return [
        os.path.realpath(os.path.expanduser("~")),
        os.path.realpath(os.getcwd()),
        os.path.realpath(tempfile.gettempdir()),
    ]


def _visible(text: str) -> str:
    """``text`` with every C0 control character shown as ``\\xNN``, then terminal-safe.

    ``for_terminal`` keeps TAB, LF and CR (a value ending in the carriage return of a
    Windows-edited file would overwrite the line, a newline would forge a second log
    line), so those are spelled out first. Whatever else ``for_terminal`` strips (DEL,
    C1, bidi and zero-width format characters) cannot be told apart in the output, which
    is the point.
    """
    from soup_cli.utils.terminal import for_terminal

    spelled = "".join(f"\\x{ord(ch):02x}" if ord(ch) < 0x20 else ch for ch in text)
    if len(spelled) > _MAX_SHOWN:
        spelled = spelled[:_MAX_SHOWN] + "..."
    return for_terminal(spelled)


def _warn_once(variable: str, value: str, reason: str, fix: str, used: str) -> None:
    key = (variable, value)
    with _WARNED_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    logger.warning(
        "%s='%s' is ignored: %s. Using %s instead. %s",
        variable,
        _visible(value),
        reason,
        _visible(used),
        fix,
    )


def env_cache_root(variable: str, default: str) -> str:
    """The cache root ``variable`` selects, or ``default``.

    ``variable`` unset or empty gives ``default`` silently. A value without control
    characters whose real path lies under ``$HOME``, the working directory or the
    temp directory is returned as that real path (the caller creates it). Any other
    value gives ``default`` and one warning per process and value.
    """
    override = os.environ.get(variable)
    if not override:
        return default
    if any(ord(ch) < 0x20 for ch in override):
        _warn_once(variable, override, REASON_CONTROL, _FIX_CONTROL, default)
        return default
    from soup_cli.utils.paths import is_under

    candidate = os.path.realpath(os.path.expanduser(override))
    if any(is_under(candidate, bound) for bound in containment_bounds()):
        return candidate
    _warn_once(variable, override, REASON_OUTSIDE, _FIX_OUTSIDE, default)
    return default
