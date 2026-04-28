"""Smart logging tiers (v0.34.0 Part A).

Maps a user-facing `--log-level` flag (quiet/normal/verbose/debug) onto
Python's stdlib logging levels and installs a Rich-formatted handler on the
top-level "soup" logger.

Tier semantics:
    quiet   -> only ERROR and worse; drop info/warning chatter
    normal  -> WARNING (default; matches prior CLI behaviour)
    verbose -> INFO; per-step monitoring messages
    debug   -> DEBUG; trainer internals + library warnings surface
"""

from __future__ import annotations

import enum
import logging
from typing import Tuple

LOGGER_NAME = "soup"


class LogLevel(enum.Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


LOG_LEVELS: Tuple[str, ...] = tuple(level.value for level in LogLevel)


_PY_LEVEL_BY_TIER = {
    LogLevel.QUIET: logging.ERROR,
    LogLevel.NORMAL: logging.WARNING,
    LogLevel.VERBOSE: logging.INFO,
    LogLevel.DEBUG: logging.DEBUG,
}


def parse_log_level(value: str) -> LogLevel:
    """Parse a CLI string into a LogLevel. Case-insensitive."""
    if not isinstance(value, str):
        raise ValueError(f"log level must be a string, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError("log level contains null byte")
    try:
        return LogLevel(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(LOG_LEVELS)
        raise ValueError(f"invalid log level {value!r}; choose one of: {valid}") from exc


def resolve_python_log_level(tier: LogLevel) -> int:
    """Map a tier onto a stdlib logging level integer."""
    return _PY_LEVEL_BY_TIER[tier]


def setup_logging(tier: LogLevel) -> logging.Logger:
    """Configure the top-level "soup" logger for the given tier.

    Idempotent — repeated calls update the level. When the tier changes
    relative to a prior call (e.g. NORMAL → DEBUG), the existing handler
    is replaced so format toggles like ``show_time``/``show_path`` reflect
    the new tier.
    """
    py_level = resolve_python_log_level(tier)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(py_level)

    # Drop any prior soup handler whose tier no longer matches so we can
    # rebuild with the right RichHandler kwargs for the new tier.
    for existing in list(logger.handlers):
        if getattr(existing, "_soup_log_tier", None) is not None:
            if getattr(existing, "_soup_log_tier", None) != tier:
                logger.removeHandler(existing)

    if not any(getattr(handler, "_soup_log_tier", None) == tier for handler in logger.handlers):
        handler: logging.Handler
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                show_time=tier == LogLevel.DEBUG,
                show_path=tier == LogLevel.DEBUG,
                rich_tracebacks=True,
                markup=False,
            )
        except Exception:  # pragma: no cover - rich is a hard dep, defensive
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(levelname)s %(name)s: %(message)s")
            )
        handler._soup_log_tier = tier  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False

    for handler in logger.handlers:
        handler.setLevel(py_level)

    return logger
