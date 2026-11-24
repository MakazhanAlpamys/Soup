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
``src/soup_cli`` it flags exactly these spellings:

* an f-string replacement field with the ``!r`` conversion whose value is a
  call to an escaping helper, and ``repr(<call to an escaping helper>)``, at
  any scope;
* the same two applied to a plain name -- ``{name!r}`` / ``repr(name)`` --
  when that name is assigned from a call to an escaping helper in the same
  scope (``name = for_terminal(x)`` ... ``{name!r}``). A nested function or
  class is its own scope, so a closure is not followed, and any such
  assignment marks the name for the whole scope.

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


_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _reprs_of(nodes, matches) -> list[int]:
    """Lines of ``{x!r}`` / ``repr(x)`` among ``nodes`` where ``matches(x)``."""
    lines: list[int] = []
    for node in nodes:
        if (
            isinstance(node, ast.FormattedValue)
            and node.conversion == ord("r")
            and matches(node.value)
        ):
            lines.append(node.value.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "repr"
            and len(node.args) == 1
            and matches(node.args[0])
        ):
            lines.append(node.lineno)
    return lines


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node inside ``scope`` that does not belong to a nested scope."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _escaped_names(nodes, names: _EscapingNames) -> set[str]:
    """Names assigned from a call to an escaping helper among ``nodes``."""
    escaped: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Assign) and names.is_escaping_call(node.value):
            escaped.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and isinstance(node.target, ast.Name)
            and names.is_escaping_call(node.value)
        ):
            escaped.add(node.target.id)
    return escaped


def repr_of_escaped_lines(source: str) -> list[int]:
    """Line numbers of every ``repr()`` taken of escaped text.

    ``{helper(...)!r}`` / ``repr(helper(...))`` anywhere, plus ``{name!r}`` /
    ``repr(name)`` where ``name`` is assigned from a helper call in the same
    scope.
    """
    tree = ast.parse(source)
    names = _EscapingNames(tree)
    lines = _reprs_of(ast.walk(tree), names.is_escaping_call)
    for scope in ast.walk(tree):
        if not isinstance(scope, _SCOPES):
            continue
        nodes = _scope_nodes(scope)
        escaped = _escaped_names(nodes, names)
        if escaped:
            lines.extend(
                _reprs_of(
                    nodes,
                    lambda value, escaped=escaped: (
                        isinstance(value, ast.Name) and value.id in escaped
                    ),
                )
            )
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


class TestTheDetectorFollowsAssignedNames:
    """``name = for_terminal(x)`` followed by ``{name!r}`` in the same scope."""

    @pytest.mark.parametrize(
        "source",
        [
            "def warn(audio):\n"
            "    from soup_cli.utils.terminal import for_terminal\n"
            "    name = for_terminal(audio)\n"
            "    return f'[yellow]Skipped {name!r}[/]'\n",
            "from rich.markup import escape\nname = escape(value)\nmsg = 'x ' + repr(name)\n",
            "from rich.markup import escape\nname: str = escape(value)\nmsg = f'{name!r}'\n",
            "from rich.markup import escape\n"
            "class Report:\n"
            "    label = escape(value)\n"
            "    line = f'{label!r}'\n",
        ],
        ids=["function-scope", "module-scope-explicit-repr", "annotated", "class-scope"],
    )
    def test_each_spelling_is_flagged(self, source: str) -> None:
        assert len(repr_of_escaped_lines(source)) == 1, source

    @pytest.mark.parametrize(
        "source",
        [
            "from soup_cli.utils.terminal import for_terminal\n"
            "name = for_terminal(repr(value))\nmsg = f'{name}'\n",
            "from rich.markup import escape\nname = escape(value)\nmsg = f'{name}'\n",
            "from rich.markup import escape\n"
            "def first(value):\n"
            "    name = escape(value)\n"
            "    return name\n"
            "def second(name):\n"
            "    return f'{name!r}'\n",
            "name = value.strip()\nmsg = f'{name!r}'\n",
            "import re\nname = re.escape(value)\nmsg = f'{name!r}'\n",
            "from soup_cli.utils.terminal import for_terminal\n"
            "_safe = for_terminal\nmsg = f'{_safe!r}'\n",
        ],
        ids=[
            "quoted-first", "escaped-without-repr", "other-scope", "plain-assignment",
            "re-escape", "function-alias",
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


_TERMINAL_TEXT_SAMPLES = [
    pytest.param(r"a\[b", r"a\[b", id="backslash-open-bracket"),
    pytest.param(r"x\[", r"x\[", id="trailing-backslash-open-bracket"),
    pytest.param("[1]", "[1]", id="digits-in-brackets"),
    pytest.param("[/x]", "[/x]", id="closing-tag"),
    pytest.param("pre\x07\x9b\u202epost", "prepost", id="control-bytes"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
]

_MIGRATED_SITES = [
    "helper",
    "env-status-missing-lock",
    "env-check-missing-lock",
    "env-fix-missing-lock",
    "env-fix-wrote-requirements",
    "init-link-refusal",
    "init-saved-panel",
    "shrink-registry-warning",
    "shrink-invalid-heal-config",
]


def _collapse_cli_output(text: str) -> str:
    from tests.conftest import strip_ansi

    for ctrl in ("\x07", "\x9b", "\u202e"):
        assert ctrl not in text, repr(text)
    return " ".join(strip_ansi(text).split())


def _run_migrated_site(site: str, raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import errno
    import sys
    from types import SimpleNamespace

    import typer
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.commands import env as env_cmd
    from soup_cli.commands import init as init_cmd
    from soup_cli.commands import shrink as shrink_cmd
    from soup_cli.utils.env_lock import EnvEntry, EnvLock
    from soup_cli.utils.terminal import terminal_text

    monkeypatch.chdir(tmp_path)
    buf = StringIO()
    con = Console(file=buf, width=300)

    def _raise_fnf(_path: str) -> None:
        raise FileNotFoundError("no lock")

    if site == "helper":
        con.print(terminal_text(raw, style="bold"))
        return None, buf.getvalue()

    if site == "env-status-missing-lock":
        monkeypatch.setattr(env_cmd, "console", con)
        monkeypatch.setattr(env_cmd, "read_lock", _raise_fnf)
        res = CliRunner().invoke(app, ["env", "status", "--lock", raw])
        return res.exit_code, buf.getvalue()

    if site == "env-check-missing-lock":
        monkeypatch.setattr(env_cmd, "console", con)
        monkeypatch.setattr(env_cmd, "read_lock", _raise_fnf)
        monkeypatch.setattr(
            env_cmd, "current_declared_bounds_check", lambda: SimpleNamespace(ok=True)
        )
        res = CliRunner().invoke(app, ["env", "check", "--lock", raw])
        return res.exit_code, buf.getvalue()

    if site == "env-fix-missing-lock":
        monkeypatch.setattr(env_cmd, "console", con)
        monkeypatch.setattr(env_cmd, "read_lock", _raise_fnf)
        res = CliRunner().invoke(app, ["env", "fix", "--lock", raw])
        return res.exit_code, buf.getvalue()

    if site == "env-fix-wrote-requirements":
        lock = EnvLock(
            soup_version="0.77.2",
            python_version="3.11.0",
            platform="linux",
            cuda_version=None,
            entries=(EnvEntry("torch", "2.5.1", "pip"),),
            created_at="2026-11-10T00:00:00Z",
        )
        monkeypatch.setattr(env_cmd, "console", con)
        monkeypatch.setattr(env_cmd, "read_lock", lambda _p: lock)
        monkeypatch.setattr(env_cmd, "write_requirements_txt", lambda _l, _p: None)
        res = CliRunner().invoke(app, ["env", "fix", "-o", raw])
        return res.exit_code, buf.getvalue()

    if site == "init-link-refusal":
        def _raise_eloop(*_args, **_kwargs) -> None:
            raise OSError(errno.ELOOP, "loop")

        monkeypatch.setattr(init_cmd, "console", con)
        monkeypatch.setattr(init_cmd, "refuse_linked_dirs", _raise_eloop)
        res = CliRunner().invoke(app, ["init", "--template", "chat", "--output", raw])
        return res.exit_code, buf.getvalue()

    if site == "init-saved-panel":
        monkeypatch.setattr(init_cmd, "console", con)
        monkeypatch.setattr(init_cmd, "_refuse_link_at", lambda _out: None)
        monkeypatch.setattr(init_cmd, "_write_config", lambda _out, _txt: None)
        res = CliRunner().invoke(app, ["init", "--template", "chat", "--output", raw, "--force"])
        return res.exit_code, buf.getvalue()

    if site == "shrink-registry-warning":
        def _fail_attach(*_args, **_kwargs) -> None:
            raise RuntimeError(f"missing {raw}")

        monkeypatch.setattr(shrink_cmd, "console", con)
        monkeypatch.setitem(
            sys.modules,
            "soup_cli.registry.attach",
            SimpleNamespace(attach_artifact=_fail_attach),
        )
        shrink_cmd._attach_to_registry("reg-1", "report.json")
        return None, buf.getvalue()

    if site == "shrink-invalid-heal-config":
        def _bad_heal(_yaml: str) -> None:
            raise ValueError(f"bad field {raw}")

        monkeypatch.setattr(shrink_cmd, "console", con)
        monkeypatch.setattr(shrink_cmd, "_build_heal_config_yaml", lambda **_kwargs: "stub: 1")
        monkeypatch.setattr("soup_cli.config.loader.load_config_from_string", _bad_heal)
        with pytest.raises(typer.Exit) as exc_info:
            shrink_cmd._run_heal(
                pruned_dir="p",
                teacher="t",
                heal_data="d",
                steps=1,
                out_dir="o",
                heal_rows=1,
                device=None,
                trc=False,
            )
        return exc_info.value.exit_code, buf.getvalue()

    raise AssertionError(f"unknown site: {site}")


@pytest.mark.parametrize("site", _MIGRATED_SITES)
@pytest.mark.parametrize(("raw", "expected"), _TERMINAL_TEXT_SAMPLES)
def test_terminal_text_and_migrated_sites_preserve_literal_output(
    site: str, raw: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, output = _run_migrated_site(site, raw, tmp_path, monkeypatch)
    plain = _collapse_cli_output(output)
    panel_expected = str(Path(expected))

    expected_by_site = {
        "helper": (None, [expected]),
        "env-status-missing-lock": (1, [f"No lock file at {expected}; run `soup env lock` first."]),
        "env-check-missing-lock": (1, [f"No lock file at {expected}; run `soup env lock` first."]),
        "env-fix-missing-lock": (1, [f"No lock file at {expected}; run `soup env lock` first."]),
        "env-fix-wrote-requirements": (0, [f"Wrote requirements to {expected}"]),
        "init-link-refusal": (
            1,
            [f"{expected} is a symbolic link, junction or other reparse point;"],
        ),
        "init-saved-panel": (
            0,
            [
                f"Config saved to {panel_expected}",
                f"Next step: soup train --config {panel_expected}",
            ],
        ),
        "shrink-registry-warning": (
            None,
            [f"Warning: could not attach to registry: missing {expected}"],
        ),
        "shrink-invalid-heal-config": (
            1,
            [f"Invalid rendered heal config: bad field {expected}"],
        ),
    }
    want_code, want_substrings = expected_by_site[site]
    assert exit_code == want_code, (site, plain)
    for snippet in want_substrings:
        assert snippet in plain, (site, snippet, plain)
