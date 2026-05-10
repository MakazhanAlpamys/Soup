"""v0.44.0 Part B — `.checkpoint_now` trigger-file watcher.

Touch `<output_dir>/.checkpoint_now` to force the next eval-step to save.
Pure-Python; the trainer callback polls `should_save_now()` between steps.
"""

from __future__ import annotations

import os
import stat
from typing import Optional

from soup_cli.utils.paths import is_under_cwd

TRIGGER_FILENAME = ".checkpoint_now"


def trigger_path(output_dir: str) -> str:
    """Return the absolute trigger-file path for `output_dir`.

    Path containment: the resolved trigger path must stay under cwd to
    prevent a crafted output_dir like `/etc` from causing the watcher to
    poll a sensitive directory.
    """
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be str")
    if not output_dir:
        raise ValueError("output_dir must be non-empty")
    if "\x00" in output_dir:
        raise ValueError("output_dir contains NUL byte")
    candidate = os.path.realpath(os.path.join(output_dir, TRIGGER_FILENAME))
    if not is_under_cwd(candidate):
        raise ValueError(
            f"trigger path is outside cwd: {os.path.basename(candidate)}"
        )
    return candidate


def should_save_now(output_dir: str) -> bool:
    """Return True iff the trigger file exists. Never raises on missing dir."""
    try:
        path = trigger_path(output_dir)
    except (TypeError, ValueError):
        return False
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def consume_trigger(output_dir: str) -> bool:
    """Atomically consume the trigger: delete the file, return True if deleted.

    Used by the trainer callback after a successful save so that the next
    step doesn't re-save.
    """
    try:
        path = trigger_path(output_dir)
    except (TypeError, ValueError):
        return False
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def write_trigger(output_dir: str, *, contents: Optional[str] = None) -> str:
    """Helper for `soup train` to manually create the trigger file (testing
    + scripting). Returns the resolved trigger path."""
    path = trigger_path(output_dir)
    body = contents if contents is not None else ""
    if not isinstance(body, str):
        raise TypeError("contents must be str or None")
    if "\x00" in body:
        raise ValueError("contents contains NUL byte")
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    # TOCTOU defence: refuse to write through a pre-existing symlink at the
    # trigger path (matches v0.33.0 #22 / v0.43.0 Part C policy).
    try:
        link_stat = os.lstat(path)
    except FileNotFoundError:
        link_stat = None
    if link_stat is not None and stat.S_ISLNK(link_stat.st_mode):
        raise OSError(
            f"refusing to write through symlink at {os.path.basename(path)}"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path
