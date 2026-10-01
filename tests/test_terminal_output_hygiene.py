"""No module under ``src/soup_cli`` takes ``repr()`` of an escaped string.

``rich.markup.escape`` -- and ``soup_cli.utils.terminal.for_terminal``, which
calls it -- neutralises a tag by putting one backslash in front of its ``[``.
``repr()`` doubles every backslash, so ``f"{escape(x)!r}"`` hands Rich two of
them, which Rich reads as a literal backslash followed by a LIVE tag: the
escape is undone by the quoting. A value containing ``[/]`` then raises
``MarkupError`` wherever it is printed (``soup attest verify`` exited 1,
"verifier unavailable", instead of 3 that way), and any other tag restyles
the line. The order that works is quote first, escape second:
``for_terminal(repr(x))``.

This ratchet is deliberately narrow. In every tracked module under
``src/soup_cli``, at any scope, it flags exactly two spellings:

* an f-string replacement field with the ``!r`` conversion whose value is a
  call to an escaping helper, and
* ``repr(<call to an escaping helper>)``.

An escaping helper is a name the module binds to ``rich.markup.escape`` or
``soup_cli.utils.terminal.for_terminal`` (``as`` aliases and plain
re-assignments such as ``_safe = for_terminal`` included), or the attribute
call ``<module>.escape`` / ``<module>.for_terminal`` on a name bound to one
of those two modules. ``re.escape`` and ``html.escape`` are never flagged.
It does not look for values printed with no escaping at all -- that needs
data flow, not a syntax check -- so a clean scan here says nothing about
those.
"""

from __future__ import annotations

import ast
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.errors import MarkupError

REPO_ROOT = Path(__file__).resolve().parent.parent

_RICH_MARKUP = "rich.markup"
_TERMINAL = "soup_cli.utils.terminal"

#: module -> the attribute of it whose return value is Rich-escaped text.
_HELPERS = {_RICH_MARKUP: "escape", _TERMINAL: "for_terminal"}


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "src/soup_cli"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line.endswith(".py")]


class _EscapingNames:
    """What the escaping helpers are called in one module."""

    def __init__(self, tree: ast.AST) -> None:
        #: local names bound to a helper function
        self.functions: set[str] = set()
        #: dotted local name of a helper module -> the helper attribute
        self.modules: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                for alias in node.names:
                    local = alias.asname or alias.name
                    if _HELPERS.get(node.module or "") == alias.name:
                        self.functions.add(local)
                    full = f"{node.module}.{alias.name}"
                    if full in _HELPERS:
                        self.modules[local] = _HELPERS[full]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _HELPERS:
                        self.modules[alias.asname or alias.name] = _HELPERS[alias.name]
        self._follow_reassignments(tree)

    def _follow_reassignments(self, tree: ast.AST) -> None:
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
        ]
        changed = True
        while changed:
            changed = False
            for node in assignments:
                if node.value.id not in self.functions:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in self.functions:
                        self.functions.add(target.id)
                        changed = True

    def is_escaping_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in self.functions
        if isinstance(func, ast.Attribute):
            return self.modules.get(ast.unparse(func.value)) == func.attr
        return False


def repr_of_escaped_lines(source: str) -> list[int]:
    """Line numbers of every ``{helper(...)!r}`` and ``repr(helper(...))``."""
    tree = ast.parse(source)
    names = _EscapingNames(tree)
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FormattedValue)
            and node.conversion == ord("r")
            and names.is_escaping_call(node.value)
        ):
            lines.append(node.value.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "repr"
            and len(node.args) == 1
            and names.is_escaping_call(node.args[0])
        ):
            lines.append(node.lineno)
    return sorted(lines)


class TestNoReprOfEscapedTextInSrc:
    def test_the_source_tree_has_none(self) -> None:
        offenders: list[str] = []
        for path in _tracked_python_files():
            rel = path.relative_to(REPO_ROOT).as_posix()
            try:
                lines = repr_of_escaped_lines(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                pytest.fail(f"{rel} does not parse ({exc.msg} at line {exc.lineno})")
            offenders.extend(f"{rel}:{line}" for line in lines)
        assert not offenders, (
            "repr() applied to an escaped string re-activates the markup it "
            "escaped. Quote first, escape second -- for_terminal(repr(x)) -- "
            "instead of {escape(x)!r}:\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_reads_the_source_tree(self) -> None:
        """A listing that silently came back short would pass vacuously."""
        scanned = _tracked_python_files()
        assert len(scanned) > 300, f"only {len(scanned)} files listed"
        names = {path.name for path in scanned}
        assert {"attest.py", "terminal.py", "probe.py"} <= names


class TestTheDetector:
    """The tree is clean, so these are the proof that the scan can fail."""

    @pytest.mark.parametrize(
        "source",
        [
            "from rich.markup import escape\nmsg = f'backend {escape(name)!r}'\n",
            "from rich.markup import escape as _esc\nmsg = f'{_esc(name)!r}'\n",
            "from soup_cli.utils.terminal import for_terminal\nmsg = f'{for_terminal(name)!r}'\n",
            "from soup_cli.utils.terminal import for_terminal\n_safe = for_terminal\n"
            "msg = f'{_safe(name)!r}'\n",
            "from rich import markup\nmsg = f'{markup.escape(name)!r}'\n",
            "import rich.markup\nmsg = f'{rich.markup.escape(name)!r}'\n",
            "from soup_cli.utils import terminal\nmsg = f'{terminal.for_terminal(name)!r}'\n",
            "from rich.markup import escape\nmsg = 'x ' + repr(escape(name))\n",
            "def show(name):\n    from rich.markup import escape\n"
            "    return f'[red]{escape(name)!r}[/]'\n",
        ],
        ids=[
            "escape", "escape-alias", "for-terminal", "reassigned", "module-attribute",
            "dotted-module", "terminal-module", "explicit-repr", "function-local-import",
        ],
    )
    def test_each_spelling_is_flagged(self, source: str) -> None:
        assert len(repr_of_escaped_lines(source)) == 1, source

    @pytest.mark.parametrize(
        "source",
        [
            "from soup_cli.utils.terminal import for_terminal\n"
            "msg = f'backend {for_terminal(repr(name))}'\n",
            "from rich.markup import escape\nmsg = f'{escape(name)}'\n",
            "msg = f'{name!r}'\n",
            "import re\nmsg = f'{re.escape(name)!r}'\n",
            "from html import escape\nmsg = f'{escape(name)!r}'\n",
            "from soup_cli.utils.terminal import strip_control\n"
            "msg = f'{strip_control(name)!r}'\n",
        ],
        ids=[
            "quote-then-escape", "escape-without-repr", "plain-repr", "re-escape",
            "html-escape", "strip-only",
        ],
    )
    def test_other_spellings_are_not_flagged(self, source: str) -> None:
        assert repr_of_escaped_lines(source) == [], source


class TestSitesTheRatchetFoundKeepTheirExitCode:
    """Two of the sites this ratchet found on its first run, through the CLI.

    Both refuse a path outside the working directory with exit 2 and quote
    the path back; with ``{escape(path)!r}`` a path containing ``[/]`` raised
    ``MarkupError`` instead, which the CLI reports as exit 1.
    """

    @pytest.mark.parametrize(
        "module_name, argv, expected",
        [
            ("env", ["env", "lock", "-o", "../[/]x.lock"], "output '../[/]x.lock' is outside cwd"),
            (
                "tunability",
                ["tunability", "--dataset", "../[/]d.jsonl"],
                "dataset '../[/]d.jsonl' is outside cwd",
            ),
        ],
        ids=["env-lock", "tunability"],
    )
    def test_a_path_containing_markup_is_quoted_literally(
        self, tmp_path, monkeypatch, module_name, argv, expected
    ) -> None:
        import importlib

        from typer.testing import CliRunner

        from soup_cli.cli import app

        module = importlib.import_module(f"soup_cli.commands.{module_name}")
        monkeypatch.chdir(tmp_path)
        buffer = StringIO()
        monkeypatch.setattr(module, "console", Console(file=buffer, color_system=None, width=200))
        result = CliRunner().invoke(app, argv)
        assert result.exit_code == 2, (buffer.getvalue(), repr(result.exception))
        assert expected in buffer.getvalue(), buffer.getvalue()


class TestWhyTheOrderMatters:
    """The behaviour the ratchet exists for, shown on Rich itself."""

    def test_repr_after_escape_leaves_a_live_closing_tag(self) -> None:
        from rich.markup import escape

        with pytest.raises(MarkupError):
            Console(file=StringIO()).print(f"[yellow]backend {escape('[/]')!r} refused[/]")

    def test_escape_after_repr_prints_the_quoted_text(self) -> None:
        from soup_cli.utils.terminal import for_terminal

        buffer = StringIO()
        Console(file=buffer, color_system=None, width=200).print(
            f"[yellow]backend {for_terminal(repr('[/]'))} refused[/]"
        )
        assert buffer.getvalue().strip() == "backend '[/]' refused"
