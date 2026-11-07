"""The Click capabilities used by the installed Typer, without version checks.

Typer may use standalone Click or its own copy. Their exceptions and enum
members have different identities even when both packages are installed.
Resolve the implementation from Typer's real command base, rather than
assuming that ``import click`` describes the objects in our command tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from typer.core import TyperGroup

_core = import_module(TyperGroup.__bases__[0].__module__)
_exceptions = import_module(f"{_core.__package__}.exceptions")

UsageError = _exceptions.UsageError
MissingParameter = _exceptions.MissingParameter
ParameterSource = _core.ParameterSource


def is_group(command: object) -> bool:
    """Whether a command exposes a subcommand mapping (Click or Typer)."""
    return isinstance(getattr(command, "commands", None), Mapping)


def is_option(parameter: object) -> bool:
    """Whether a real CLI parameter declares itself an option."""
    return getattr(parameter, "param_type_name", None) == "option"


def is_argument(parameter: object) -> bool:
    """Whether a real CLI parameter declares itself a positional argument."""
    return getattr(parameter, "param_type_name", None) == "argument"
