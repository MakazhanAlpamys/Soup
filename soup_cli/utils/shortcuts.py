"""v0.44.0 Part B — Desktop / Start Menu / .desktop shortcut creator.

Pure-Python: builds the shortcut file content for the host platform.
Returns the rendered text + suggested filename without writing — the caller
(`soup install-shortcut`) does the actual write under cwd containment.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class ShortcutSpec:
    """Rendered shortcut content + suggested filename."""

    filename: str
    content: str
    platform: str


# `Exec=` and `--name` may contain user-controlled paths in the future; we
# disallow shell metacharacters defensively even though today the inputs are
# fixed strings. Any change to those callers must keep the allowlist policy.
_NAME_DISALLOWED = frozenset("`'\"\\\n\r\x00$;&|<>")


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be str")
    if not name:
        raise ValueError("name must be non-empty")
    if len(name) > 64:
        raise ValueError("name must be <= 64 chars")
    if any(char in _NAME_DISALLOWED for char in name):
        raise ValueError("name contains disallowed character")
    return name


def _validate_command(command: str) -> str:
    if not isinstance(command, str):
        raise TypeError("command must be str")
    if not command:
        raise ValueError("command must be non-empty")
    if "\x00" in command or "\n" in command or "\r" in command:
        raise ValueError("command contains control char")
    if len(command) > 1024:
        raise ValueError("command must be <= 1024 chars")
    return command


def detect_platform() -> str:
    """Return one of {linux, darwin, windows, unknown}."""
    system = platform.system().lower()
    if system in ("linux", "darwin", "windows"):
        return system
    return "unknown"


def build_linux_desktop_entry(*, name: str, command: str) -> ShortcutSpec:
    """Build a freedesktop.org `.desktop` entry."""
    name = _validate_name(name)
    command = _validate_command(command)
    body = (
        "[Desktop Entry]\n"
        f"Type=Application\n"
        f"Name={name}\n"
        f"Exec={command}\n"
        "Terminal=true\n"
        "Categories=Development;\n"
    )
    safe_filename = name.lower().replace(" ", "-") + ".desktop"
    return ShortcutSpec(filename=safe_filename, content=body, platform="linux")


def build_macos_command_file(*, name: str, command: str) -> ShortcutSpec:
    """Build a `.command` script (double-clickable on macOS Finder)."""
    name = _validate_name(name)
    command = _validate_command(command)
    body = "#!/usr/bin/env bash\n" f"exec {command}\n"
    safe_filename = name.lower().replace(" ", "-") + ".command"
    return ShortcutSpec(filename=safe_filename, content=body, platform="darwin")


def build_windows_cmd(*, name: str, command: str) -> ShortcutSpec:
    """Build a Windows `.cmd` launcher.

    Note: a true Start-Menu .lnk needs `pywin32` or `winshell`, which we keep
    out of our deps. The `.cmd` file is a portable alternative the user can
    pin to taskbar/Start menu manually.
    """
    name = _validate_name(name)
    command = _validate_command(command)
    body = "@echo off\r\n" f"{command} %*\r\n"
    safe_filename = name.lower().replace(" ", "-") + ".cmd"
    return ShortcutSpec(filename=safe_filename, content=body, platform="windows")


def build_for_current_platform(*, name: str, command: str) -> ShortcutSpec:
    """Build the right shortcut for the host. Raises on `unknown`."""
    plat = detect_platform()
    if plat == "linux":
        return build_linux_desktop_entry(name=name, command=command)
    if plat == "darwin":
        return build_macos_command_file(name=name, command=command)
    if plat == "windows":
        return build_windows_cmd(name=name, command=command)
    raise NotImplementedError(f"Shortcut creation not supported on platform: {plat}")
