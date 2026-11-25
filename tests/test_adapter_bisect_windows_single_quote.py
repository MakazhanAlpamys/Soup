"""On Windows, ``--opt='a b'`` in ``soup adapters bisect --eval-command`` is one word (#3088, B-7).

The Windows splitter grouped a single-quoted run only when the quote started a word, so
``--opt='a b'`` was ``--opt='a`` and ``b'`` while ``--opt="a b"`` was ``--opt=a b``. A ``'``
inside a word now groups when the next ``'`` of the command ends a word. Every other ``'``
stays a literal character as before: an apostrophe (``don't``, ``a'b c``, ``O'Brien``), an
unclosed quote, a quote glued to more text, a ``'`` inside double quotes, and one after a
backslash.
"""

from __future__ import annotations

import json
import ntpath
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import adapters
from tests.conftest import strip_ansi


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("run --opt='a b' x", ["run", "--opt=a b", "x"]),
        ('run --opt="a b" x', ["run", "--opt=a b", "x"]),
        ("run '{ckpt}' x", ["run", "{ckpt}", "x"]),
        ("run --opt='a b'", ["run", "--opt=a b"]),
        ("run --a='x y' --b='z w'", ["run", "--a=x y", "--b=z w"]),
        ("run --opt='a b'\tx", ["run", "--opt=a b", "x"]),
        ("run --ckpt='{ckpt}'", ["run", "--ckpt={ckpt}"]),
        # An escaped \" opens no group, so the ' after it still ends the word.
        (r"tool --opt='a \"b' x", ["tool", '--opt=a \\"b', "x"]),
    ],
)
def test_a_quote_inside_a_word_groups_when_its_closing_quote_ends_the_word(command, expected):
    assert adapters._split_windows_command(command) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("run don't", ["run", "don't"]),
        ("run a'b c", ["run", "a'b", "c"]),
        ("run --opt='a b", ["run", "--opt='a", "b"]),
        ("run --opt='a b'c", ["run", "--opt='a", "b'c"]),
        ("run don't stop 'x'", ["run", "don't", "stop", "x"]),
        ("run don't 'x y'", ["run", "don't", "x y"]),
        ("run it's Bob's", ["run", "it's", "Bob's"]),
        ('run "it\'s"', ["run", "it's"]),
        (r"run x\'a b'", ["run", r"x\'a", "b'"]),
        (r"C:\Users\O'Brien\tool.bat {ckpt}", [r"C:\Users\O'Brien\tool.bat", "{ckpt}"]),
        # The next ' is inside a later double-quoted argument, so it closes nothing.
        ("tool don't \"James' file.txt\"", ["tool", "don't", "James' file.txt"]),
        # A ' that ends a word opens no group, even when a later ' also ends one.
        ("tool dogs' and cats' x", ["tool", "dogs'", "and", "cats'", "x"]),
        ("tool --opt=' a'", ["tool", "--opt='", "a'"]),
    ],
)
def test_any_other_quote_is_a_literal_character_as_before(command, expected):
    assert adapters._split_windows_command(command) == expected


def test_a_word_that_starts_with_an_unclosed_quote_is_still_refused():
    with pytest.raises(ValueError, match="No closing quotation"):
        adapters._split_windows_command("run 'a b")


def test_the_help_names_the_spellings_that_group():
    result = CliRunner().invoke(app, ["adapters", "bisect", "--help"])
    flat = " ".join(strip_ansi(result.output).split())
    assert "(--opt='a b')" in flat
    assert "(don't) is a literal character" in flat


def test_the_windows_command_receives_the_grouped_option(tmp_path, monkeypatch):
    # The Windows path of the command, taken on any platform with ``os.name`` set to "nt".
    monkeypatch.chdir(tmp_path)
    program = r"C:\tools\eval.bat"
    simulated = SimpleNamespace(**vars(os))
    simulated.name = "nt"
    simulated.path = SimpleNamespace(**vars(ntpath))
    simulated.path.isfile = lambda path: path == program
    simulated.access = lambda path, mode: True
    monkeypatch.setattr(adapters, "os", simulated)
    calls = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    result = CliRunner().invoke(app, [
        "adapters", "bisect", "ckpt-1", "ckpt-2",
        "--eval-command", rf"{program} eval.py --name='my run' --ckpt='{{ckpt}}'",
        "--output", "bisect.json",
    ])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == [
        [program, "eval.py", "--name=my run", "--ckpt=ckpt-1"],
        [program, "eval.py", "--name=my run", "--ckpt=ckpt-2"],
    ]
    assert json.loads(Path("bisect.json").read_text())["verdict"] == "ALL_OK"
