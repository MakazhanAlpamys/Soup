"""Tests for v0.34.0 Part A — smart logging tiers."""

from __future__ import annotations

import logging
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.log_level import (
    LOG_LEVELS,
    LogLevel,
    parse_log_level,
    resolve_python_log_level,
    setup_logging,
)

# Rich help renderer can split a flag like --log-level with ANSI colour
# escapes between `--`, `-log`, `-level` when the terminal is narrow (CI
# runners have a narrower default width than local shells). Strip ANSI so
# substring assertions are robust across all CI matrix jobs.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)

class TestLogLevelEnum:
    def test_four_tiers_defined(self):
        assert set(LOG_LEVELS) == {"quiet", "normal", "verbose", "debug"}

    def test_parse_valid(self):
        assert parse_log_level("quiet") == LogLevel.QUIET
        assert parse_log_level("normal") == LogLevel.NORMAL
        assert parse_log_level("verbose") == LogLevel.VERBOSE
        assert parse_log_level("debug") == LogLevel.DEBUG

    def test_parse_case_insensitive(self):
        assert parse_log_level("QUIET") == LogLevel.QUIET
        assert parse_log_level("Verbose") == LogLevel.VERBOSE

    def test_parse_invalid(self):
        with pytest.raises(ValueError, match="invalid log level"):
            parse_log_level("loud")

    def test_parse_null_byte_rejected(self):
        with pytest.raises(ValueError):
            parse_log_level("quiet\x00")

    def test_resolve_python_level(self):
        assert resolve_python_log_level(LogLevel.QUIET) == logging.ERROR
        assert resolve_python_log_level(LogLevel.NORMAL) == logging.WARNING
        assert resolve_python_log_level(LogLevel.VERBOSE) == logging.INFO
        assert resolve_python_log_level(LogLevel.DEBUG) == logging.DEBUG


class TestSetupLogging:
    def test_returns_root_logger(self):
        logger = setup_logging(LogLevel.NORMAL)
        assert logger.level == logging.WARNING

    def test_quiet_suppresses_info(self):
        logger = setup_logging(LogLevel.QUIET)
        assert not logger.isEnabledFor(logging.INFO)

    def test_debug_enables_debug(self):
        logger = setup_logging(LogLevel.DEBUG)
        assert logger.isEnabledFor(logging.DEBUG)

    def test_idempotent(self):
        # Calling twice should not duplicate handlers
        setup_logging(LogLevel.NORMAL)
        n_handlers = len(logging.getLogger("soup").handlers)
        setup_logging(LogLevel.NORMAL)
        assert len(logging.getLogger("soup").handlers) == n_handlers

    def test_tier_change_replaces_handler(self):
        setup_logging(LogLevel.NORMAL)
        setup_logging(LogLevel.DEBUG)
        # Exactly one tier-tagged handler should remain — the DEBUG one.
        tagged = [
            handler for handler in logging.getLogger("soup").handlers
            if getattr(handler, "_soup_log_tier", None) is not None
        ]
        assert len(tagged) == 1
        assert tagged[0]._soup_log_tier == LogLevel.DEBUG  # type: ignore[attr-defined]

    def test_parse_non_string_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            parse_log_level(42)  # type: ignore[arg-type]


class TestIssue273ForeignHandlerContract:
    """Deterministically pin the ``74->73`` branch of ``setup_logging`` (#273).

    The branch is the loop-back edge taken when a handler on the ``soup``
    logger carries **no** ``_soup_log_tier`` tag — a handler Soup did not
    install. Whether ambient logger state happens to exercise it varies by
    platform and test ordering, so these tests force the state explicitly:
    an embedding application's handlers must survive reconfiguration, while
    a Soup handler with a stale tier is still removed.
    """

    @pytest.fixture(autouse=True)
    def _isolate_soup_logger(self):
        logger = logging.getLogger("soup")
        saved_handlers = list(logger.handlers)
        saved_level = logger.level
        saved_propagate = logger.propagate
        yield
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in saved_handlers:
            logger.addHandler(handler)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate

    def test_foreign_handler_survives_reconfigure(self):
        # Killing check: dropping the ``is not None`` guard in setup_logging
        # removes this handler and fails this test by name.
        foreign = logging.NullHandler()
        assert not hasattr(foreign, "_soup_log_tier")

        logger = logging.getLogger("soup")
        logger.addHandler(foreign)
        setup_logging(LogLevel.NORMAL)

        assert foreign in logger.handlers

    def test_stale_soup_handler_removed_with_foreign_present(self):
        foreign = logging.NullHandler()
        stale = logging.NullHandler()
        stale._soup_log_tier = LogLevel.DEBUG  # type: ignore[attr-defined]

        logger = logging.getLogger("soup")
        logger.addHandler(foreign)
        logger.addHandler(stale)
        setup_logging(LogLevel.NORMAL)

        assert stale not in logger.handlers
        assert foreign in logger.handlers
        tagged = [
            handler for handler in logger.handlers
            if getattr(handler, "_soup_log_tier", None) is not None
        ]
        assert len(tagged) == 1
        assert tagged[0]._soup_log_tier == LogLevel.NORMAL  # type: ignore[attr-defined]


class TestCliFlag:
    def test_log_level_help_visible(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "--log-level" in _strip_ansi(result.output), result.output

    def test_log_level_invalid_rejected(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--log-level", "loud", "version"])
        assert result.exit_code != 0

    def test_log_level_quiet_accepted(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--log-level", "quiet", "version"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
