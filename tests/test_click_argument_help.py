"""Positional-argument descriptions must reach ``--help`` (click 8.5.0 dropped them).

With click 8.5.0 installed, ``soup <command> --help`` listed every positional
argument as ``ADAPTER  TEXT  [required]`` with no description: click 8.5's
``Argument.__init__`` assigns its own ``help`` after typer set the one from
``typer.Argument(help=...)``, so every positional argument in the CLI lost its
description. Option help is unaffected, which is why nothing else in the suite noticed.
No typer release under Soup's ``<0.21`` cap compensates, and click 8.4.2 renders
correctly, so the fix is a bound on click in ``pyproject.toml``. It is declared
explicitly because Soup imports click directly (``commands/autopilot.py``,
``utils/argv_redaction.py``, ``utils/exit_codes.py``) and typer pulls it in only
transitively.

Two tests, because they fail for different reasons: the rendering test fails on any
environment that resolved click 8.5 (whatever the cause), and the declaration test
fails when the pin is removed from ``pyproject.toml`` even on an environment that
happens to hold a good click.

``pyproject.toml`` is parsed with a regex, not ``tomllib``: ``tomllib`` is 3.11+ and
this repo supports 3.10 (the convention of ``tests/test_issue636_torch_floor.py``).
Before widening the bound, check that ``soup adapters check --help`` still shows its
description with the newest click and the newest typer Soup allows.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from packaging.requirements import Requirement
from packaging.version import Version
from typer.testing import CliRunner

from soup_cli.cli import app

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

# Rich/Typer emits PER-CHARACTER ANSI escapes on a colour-capable runner (Linux and
# macOS CI) and draws box borders between the fragments of a wrapped help line;
# Windows auto-disables colour. Strip both, then collapse whitespace, before any
# substring match on CLI output (same helper as test_issue338_awq_gptq_deprecated.py).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX_RE = re.compile(r"[─-╿]")


def _plain(text: str) -> str:
    """ANSI- and border-stripped, whitespace-collapsed CLI output."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


@pytest.mark.parametrize(
    ("argv", "description"),
    [
        (["adapters", "check", "--help"], "Path to adapter directory"),
        (["data", "clean", "--help"], "Path to input dataset file (JSONL)"),
    ],
)
def test_a_positional_argument_description_reaches_help(argv, description):
    result = CliRunner().invoke(app, argv)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert description in _plain(result.output), (
        f"`soup {' '.join(argv)}` does not show the argument description "
        f"{description!r}; click 8.5.0 drops positional-argument help "
        f"(pyproject.toml bounds click below 8.5)",
        _plain(result.output),
    )


def _core_requirements() -> list[Requirement]:
    """The ``[project] dependencies`` array of ``pyproject.toml``, as requirements."""
    text = PYPROJECT.read_text(encoding="utf-8")
    table = re.search(r"^\[project\]\s*$(.*?)^\[", text, re.M | re.S)
    assert table, "pyproject.toml has no [project] table"
    array = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", table.group(1), re.M | re.S)
    assert array, "no dependencies = [...] array in [project]"
    # Either TOML string quote, an optional trailing comma and an optional trailing
    # comment; comment-only lines do not match.
    entries = [
        double or single
        for double, single in re.findall(
            r"""^\s*(?:"([^"]+)"|'([^']+)')\s*,?\s*(?:#.*)?$""", array.group(1), re.M
        )
    ]
    assert entries, "the core dependencies parsed to no requirement strings"
    return [Requirement(entry) for entry in entries]


def test_the_parser_sees_the_known_core_entries():
    # Guards the regex itself: a parser that silently returned one entry would make
    # the pin assertions below vacuous in the passing direction.
    names = {requirement.name.lower() for requirement in _core_requirements()}
    assert {"typer", "rich", "pydantic", "pyyaml", "plotext"} <= names


def test_click_is_declared_and_bounded_below_8_5():
    clicks = [req for req in _core_requirements() if req.name.lower() == "click"]
    assert len(clicks) == 1, (
        "pyproject.toml must declare click once in [project] dependencies, bounded "
        "below 8.5: click 8.5.0 drops every positional-argument description from "
        f"--help; found {[str(req) for req in clicks]}"
    )
    specifier = clicks[0].specifier
    assert Version("8.5.0") not in specifier, f"click 8.5.0 is allowed by {specifier}"
    assert Version("8.5") not in specifier, f"click 8.5 is allowed by {specifier}"
    # The bound is an upper one, not a pin to nothing: the release that works is allowed.
    assert Version("8.4.2") in specifier, f"click 8.4.2 is refused by {specifier}"
    # Soup imports click directly and typer needs click 8: the floor stays at 8.0.
    assert Version("8.0.0") in specifier, f"click 8.0.0 is refused by {specifier}"
