"""``soup local-rl train`` scheduler scaffold (v0.71.13 #229).

Render a systemd-user ``local-rl.service`` + ``.timer`` (Linux) or a
``com.soup.local-rl.plist`` (macOS launchd) that invokes
``soup local-rl train --once`` daily at a configurable local time.

This module only **renders** the unit files (and writes them to a chosen
directory) — it never runs ``systemctl`` / ``launchctl`` itself, so it is
fully testable on any OS. The CLI prints the manual install command. This
mirrors v0.46 deploy-autopilot / v0.68 apple-adapter (render-not-execute).

Public surface:

- ``build_train_argv(...)`` — argv list the scheduler invokes
- ``render_systemd_service(...)`` / ``render_systemd_timer(...)``
- ``render_launchd_plist(...)``
- ``write_scheduler_files(target_dir, ...)`` — atomic write of the units
- ``SYSTEMD_SERVICE_NAME`` / ``SYSTEMD_TIMER_NAME`` / ``LAUNCHD_PLIST_NAME``
"""

from __future__ import annotations

import os
from typing import Dict, List

from soup_cli.utils.local_rl import (
    MIN_PAIRS_DEFAULT,
    validate_db_path,
    validate_local_rl_train_method,
    validate_min_pairs,
    validate_output_dir,
)
from soup_cli.utils.paths import atomic_write_text

SYSTEMD_SERVICE_NAME = "soup-local-rl.service"
SYSTEMD_TIMER_NAME = "soup-local-rl.timer"
LAUNCHD_PLIST_NAME = "com.soup.local-rl.plist"

_MIN_HOUR = 0
_MAX_HOUR = 23
_MIN_MINUTE = 0
_MAX_MINUTE = 59
_MAX_MODEL_LEN = 512


def _validate_model(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("model must not be bool")
    if not isinstance(value, str):
        raise TypeError("model must be str")
    if not value:
        raise ValueError("model must be non-empty")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(
            "model must not contain NUL / newline / carriage return"
        )
    if len(value) > _MAX_MODEL_LEN:
        raise ValueError(f"model length {len(value)} > {_MAX_MODEL_LEN}")
    return value


def _validate_hm(hour: object, minute: object) -> "tuple[int, int]":
    for name, value, lo, hi in (
        ("hour", hour, _MIN_HOUR, _MAX_HOUR),
        ("minute", minute, _MIN_MINUTE, _MAX_MINUTE),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int")
        if value < lo or value > hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return int(hour), int(minute)  # type: ignore[arg-type]


def _validate_soup_python(value: object) -> str:
    """The python executable that runs ``-m soup_cli.cli``."""
    if isinstance(value, bool):
        raise TypeError("soup_python must not be bool")
    if not isinstance(value, str):
        raise TypeError("soup_python must be str")
    if not value:
        raise ValueError("soup_python must be non-empty")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("soup_python must not contain NUL / newline")
    return value


def build_train_argv(
    *,
    soup_python: str,
    db_path: str,
    model: str,
    train_method: str,
    min_pairs: int = MIN_PAIRS_DEFAULT,
    output_dir: str = "local_rl_adapter",
) -> List[str]:
    """Return the argv list the scheduler invokes (no shell).

    ``min_pairs`` and ``output_dir`` are always rendered, so the scheduled run
    does what ``soup local-rl train --once`` with the same flags does (#1448).
    They are validated here with the runner's own checks, so a value the
    nightly run would refuse is refused before the unit is written.
    """
    soup_python = _validate_soup_python(soup_python)
    validate_db_path(db_path)
    model = _validate_model(model)
    method = validate_local_rl_train_method(train_method)
    min_pairs = validate_min_pairs(min_pairs)
    output_dir = validate_output_dir(output_dir)
    if "\n" in output_dir or "\r" in output_dir:
        raise ValueError("output_dir must not contain newline / carriage return")
    return [
        soup_python,
        "-m",
        "soup_cli.cli",
        "local-rl",
        "train",
        "--once",
        "--db",
        db_path,
        "--model",
        model,
        "--train-method",
        method,
        "--min-pairs",
        str(min_pairs),
        "--output",
        output_dir,
    ]


def _systemd_quote(arg: str) -> str:
    """Double-quote an argv element for a systemd ``ExecStart`` line.

    systemd splits ``ExecStart`` on whitespace unless quoted; ``"`` and
    ``\\`` inside a quoted token must be escaped. Defence-in-depth: refuse
    any newline / CR (a quoted token cannot span lines — this would inject a
    new directive). The argv builder already rejects these on db_path/model.
    """
    if "\n" in arg or "\r" in arg:
        raise ValueError("systemd argument must not contain newline / CR")
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_systemd_service(
    *,
    soup_python: str,
    db_path: str,
    model: str,
    train_method: str,
    min_pairs: int = MIN_PAIRS_DEFAULT,
    output_dir: str = "local_rl_adapter",
) -> str:
    """Render the systemd-user ``.service`` unit."""
    argv = build_train_argv(
        soup_python=soup_python,
        db_path=db_path,
        model=model,
        train_method=train_method,
        min_pairs=min_pairs,
        output_dir=output_dir,
    )
    exec_start = " ".join(_systemd_quote(a) for a in argv)
    return (
        "[Unit]\n"
        "Description=Soup local-RL nightly DPO/KTO/ORPO train\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={_systemd_quote(os.getcwd())}\n"
        f"ExecStart={exec_start}\n"
    )


def render_systemd_timer(*, hour: int = 3, minute: int = 0) -> str:
    """Render the systemd-user ``.timer`` unit (daily at ``hour:minute``)."""
    hour, minute = _validate_hm(hour, minute)
    return (
        "[Unit]\n"
        "Description=Soup local-RL nightly train timer\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar=*-*-* {hour:02d}:{minute:02d}:00\n"
        "Persistent=true\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


def render_launchd_plist(
    *,
    soup_python: str,
    db_path: str,
    model: str,
    train_method: str,
    hour: int = 3,
    minute: int = 0,
    min_pairs: int = MIN_PAIRS_DEFAULT,
    output_dir: str = "local_rl_adapter",
) -> str:
    """Render a macOS launchd ``.plist`` (daily ``StartCalendarInterval``)."""
    from xml.sax.saxutils import escape as _xml_escape

    hour, minute = _validate_hm(hour, minute)
    argv = build_train_argv(
        soup_python=soup_python,
        db_path=db_path,
        model=model,
        train_method=train_method,
        min_pairs=min_pairs,
        output_dir=output_dir,
    )
    args_xml = "\n".join(
        f"        <string>{_xml_escape(a)}</string>" for a in argv
    )
    label = LAUNCHD_PLIST_NAME[: -len(".plist")]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "    <key>Label</key>\n"
        f"    <string>{_xml_escape(label)}</string>\n"
        "    <key>ProgramArguments</key>\n"
        "    <array>\n"
        f"{args_xml}\n"
        "    </array>\n"
        "    <key>WorkingDirectory</key>\n"
        f"    <string>{_xml_escape(os.getcwd())}</string>\n"
        "    <key>StartCalendarInterval</key>\n"
        "    <dict>\n"
        "        <key>Hour</key>\n"
        f"        <integer>{hour}</integer>\n"
        "        <key>Minute</key>\n"
        f"        <integer>{minute}</integer>\n"
        "    </dict>\n"
        "</dict>\n"
        "</plist>\n"
    )


def write_scheduler_files(
    target_dir: str,
    *,
    soup_python: str,
    db_path: str,
    model: str,
    train_method: str,
    hour: int = 3,
    minute: int = 0,
    min_pairs: int = MIN_PAIRS_DEFAULT,
    output_dir: str = "local_rl_adapter",
) -> Dict[str, str]:
    """Render both the systemd units and the launchd plist into ``target_dir``.

    All three are written so an operator on either OS finds the right scaffold.
    Returns ``{filename: realpath}``. Writes are atomic + cwd-contained +
    symlink-rejected via ``atomic_write_text``.
    """
    if not isinstance(target_dir, str) or not target_dir or "\x00" in target_dir:
        raise ValueError("target_dir must be a non-empty NUL-free string")

    service = render_systemd_service(
        soup_python=soup_python,
        db_path=db_path,
        model=model,
        train_method=train_method,
        min_pairs=min_pairs,
        output_dir=output_dir,
    )
    timer = render_systemd_timer(hour=hour, minute=minute)
    plist = render_launchd_plist(
        soup_python=soup_python,
        db_path=db_path,
        model=model,
        train_method=train_method,
        hour=hour,
        minute=minute,
        min_pairs=min_pairs,
        output_dir=output_dir,
    )

    # Render (and so validate) everything before creating the directory.
    os.makedirs(target_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for name, body in (
        (SYSTEMD_SERVICE_NAME, service),
        (SYSTEMD_TIMER_NAME, timer),
        (LAUNCHD_PLIST_NAME, plist),
    ):
        path = os.path.join(target_dir, name)
        atomic_write_text(body, path, field="scheduler_file")
        written[name] = os.path.realpath(path)
    return written


__all__ = [
    "SYSTEMD_SERVICE_NAME",
    "SYSTEMD_TIMER_NAME",
    "LAUNCHD_PLIST_NAME",
    "build_train_argv",
    "render_systemd_service",
    "render_systemd_timer",
    "render_launchd_plist",
    "write_scheduler_files",
]
