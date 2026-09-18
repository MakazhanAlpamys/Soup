"""Canonical exit codes for Soup gate and verdict commands.

Taxonomy (project-wide standard):
- 0 (EXIT_OK): Gate passed (SHIP / OK / MINOR / valid data).
- 1 (EXIT_RUNTIME_ERROR): Unexpected execution failure (internal error, crash).
- 2 (EXIT_GATE_FAILED): Model or artifact failed the gate threshold
  (DON'T SHIP / MAJOR / regression / drift / unusable data).
- 3 (EXIT_USAGE_ERROR): User invocation, flag, or input-data error
  (missing file, invalid format, unparseable input, bad flag).
"""

from __future__ import annotations

import click.exceptions
import typer.core

EXIT_OK: int = 0
EXIT_RUNTIME_ERROR: int = 1
EXIT_GATE_FAILED: int = 2
EXIT_USAGE_ERROR: int = 3

_USAGE_ERRORS: tuple[type[Exception], ...] = (click.exceptions.UsageError,)


class _GateUsageErrorMixin:
    """Mixin that catches Click/Typer UsageErrors and sets exit_code = EXIT_USAGE_ERROR (3)."""

    def make_context(self, info_name, args, parent=None, **extra):
        try:
            return super().make_context(info_name, args, parent=parent, **extra)
        except _USAGE_ERRORS as exc:
            exc.exit_code = EXIT_USAGE_ERROR
            raise


class GateCommand(_GateUsageErrorMixin, typer.core.TyperCommand):
    """TyperCommand for gate commands: invalid flags / usage errors exit 3 instead of 2."""


class GateGroup(_GateUsageErrorMixin, typer.core.TyperGroup):
    """TyperGroup for gate commands (e.g. `soup ship`): usage errors exit 3 instead of 2."""
