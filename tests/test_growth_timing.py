"""``tests/_growth_timing.py``: the Linux-only gate on the linear-growth timing tests.

``assert_grows_linearly`` times a scan at ``n`` and at ``8n``. ``re`` behaves the same on every
OS, but a macOS or Windows runner adds cold regex compilation and timer noise that a Linux
runner does not, and every Linux CI cell runs these tests, so ``linux_only`` skips them
elsewhere. ``SOUP_RUN_GROWTH_TESTS=1`` runs them anywhere, for a local check.

Two things are pinned here:

* the gate itself (``TestTheGateRunsOnLinuxOrOnRequest``): a skipped test is the same colour as a
  passing one, so both edges of the condition are tested;
* that every test calling ``assert_grows_linearly`` carries the gate
  (``TestEveryGrowthTestCarriesTheGate``), so a sixth file cannot forget it. The scanner is
  shown able to fail on seeded sources, as in ``test_symlink_skips_are_capability_based.py``.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

from tests._growth_timing import RUN_ENV_VAR, _runs_here, linux_only

TESTS_DIR = Path(__file__).resolve().parent
HELPER = "assert_grows_linearly"
GATE = "linux_only"


def _is_helper_module(name: str | None) -> bool:
    return bool(name) and name.split(".")[-1] == "_growth_timing"


def _imported_names(tree: ast.Module) -> tuple[bool, set[str], set[str]]:
    """Whether the module imports ``_growth_timing``, and the local names of helper and gate."""
    imports = False
    helper_names = {HELPER}
    gate_names = {GATE}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _is_helper_module(node.module):
                imports = True
                for alias in node.names:
                    if alias.name == HELPER and alias.asname:
                        helper_names.add(alias.asname)
                    if alias.name == GATE and alias.asname:
                        gate_names.add(alias.asname)
            elif any(_is_helper_module(alias.name) for alias in node.names):
                imports = True
        elif isinstance(node, ast.Import):
            imports = imports or any(_is_helper_module(alias.name) for alias in node.names)
    return imports, helper_names, gate_names


def _names_in(node: ast.AST, names: set[str]) -> bool:
    """Whether ``node`` is, or holds, a reference to one of ``names`` (``gate`` or ``mod.gate``)."""
    return any(
        (isinstance(part, ast.Name) and part.id in names)
        or (isinstance(part, ast.Attribute) and part.attr in names)
        for part in ast.walk(node)
    )


def _carries(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, names: set[str]) -> bool:
    return any(_names_in(decorator, names) for decorator in node.decorator_list)


def scan_source(source: str, filename: str = "<string>") -> tuple[bool, int, list[str]]:
    """Return ``(imports the helper, calls found, one problem string per ungated call)``."""
    tree = ast.parse(source, filename=filename)
    imports, helper_names, gate_names = _imported_names(tree)
    if not imports:
        return False, 0, []

    module_gated = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        and _names_in(node.value, gate_names)
        for node in tree.body
    )
    problems: list[str] = []
    calls = 0

    def visit(node: ast.AST, scope: list[ast.AST]) -> None:
        nonlocal calls
        if (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id in helper_names)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == HELPER)
            )
        ):
            calls += 1
            where = f"{filename}:{node.lineno}"
            functions = [s for s in scope if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if module_gated:
                pass
            elif not functions:
                problems.append(
                    f"{where}: {HELPER} runs at import time, where no mark can skip it; "
                    f"move it into a test or set pytestmark = {GATE}"
                )
            else:
                entry = functions[0]  # the outermost function: the test, or the helper it hides in
                if not entry.name.startswith("test"):
                    problems.append(
                        f"{where}: {HELPER} is called from {entry.name}(), not from a test, so "
                        f"the mark on its callers cannot be checked here; set module-level "
                        f"pytestmark = {GATE}"
                    )
                elif not any(
                    _carries(s, gate_names)
                    for s in scope
                    if s is entry or isinstance(s, ast.ClassDef)
                ):
                    problems.append(
                        f"{where}: {entry.name} calls {HELPER} but neither it, its class nor the "
                        f"module carries @{GATE} (tests/_growth_timing.py)"
                    )
        child_scope = scope + [node] if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) else scope
        for child in ast.iter_child_nodes(node):
            visit(child, child_scope)

    visit(tree, [])
    return True, calls, problems


class TestTheGateRunsOnLinuxOrOnRequest:
    """A skipped test and a passing test are the same colour: pin both edges."""

    @pytest.mark.parametrize("platform", ["linux", "linux2"])
    def test_it_runs_on_linux(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.delenv(RUN_ENV_VAR, raising=False)
        assert _runs_here() is True

    @pytest.mark.parametrize("platform", ["darwin", "win32", "cygwin", "freebsd14"])
    def test_it_does_not_run_elsewhere_by_default(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.delenv(RUN_ENV_VAR, raising=False)
        assert _runs_here() is False

    @pytest.mark.parametrize("platform", ["darwin", "win32"])
    def test_the_environment_variable_runs_it_anywhere(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setenv(RUN_ENV_VAR, "1")
        assert _runs_here() is True

    @pytest.mark.parametrize("value", ["", "0", "no", "true"])
    def test_only_one_turns_the_escape_hatch_on(self, monkeypatch, value):
        """A near miss ("true", "0") must not silently run a 30-second timing test."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv(RUN_ENV_VAR, value)
        assert _runs_here() is False

    def test_the_variable_is_named_in_the_reason(self):
        reason = linux_only.kwargs["reason"]
        assert RUN_ENV_VAR == "SOUP_RUN_GROWTH_TESTS"
        assert f"{RUN_ENV_VAR}=1" in reason
        assert "Linux" in reason and "OS" in reason

    def test_the_mark_is_a_skipif_on_the_gate(self):
        assert linux_only.name == "skipif"
        assert linux_only.args == (not _runs_here(),)


class TestEveryGrowthTestCarriesTheGate:
    def test_every_test_that_times_growth_is_gated(self):
        importers: dict[str, int] = {}
        problems: list[str] = []
        for path in sorted(TESTS_DIR.rglob("*.py")):
            if path.name == "_growth_timing.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "_growth_timing" not in text:
                continue
            try:
                imports, calls, found = scan_source(text, path.name)
            except SyntaxError:
                continue
            if imports:
                importers[path.name] = calls
            problems += found
        assert not problems, "\n".join(problems)

        # The scan must be looking at something: the five files of #2069, each with a call.
        known = {
            "test_issue1226_clause_scanner_is_linear.py",
            "test_issue1329_email_pattern_is_linear.py",
            "test_issue1346_rewrites_are_linear.py",
            "test_issue1347_boxed_scan.py",
            "test_issue1528_contains_is_linear.py",
        }
        assert known <= set(importers), sorted(known - set(importers))
        assert all(importers[name] for name in known), importers

    def test_a_sixth_file_without_the_gate_is_caught(self):
        source = textwrap.dedent(
            """
            from tests._growth_timing import assert_grows_linearly

            def test_new_scan_is_linear():
                assert_grows_linearly(len, lambda k: "a" * k, 10)
            """
        )
        imports, calls, problems = scan_source(source, "test_new.py")
        assert (imports, calls) == (True, 1)
        assert len(problems) == 1 and "test_new_scan_is_linear" in problems[0]
        assert "@linux_only" in problems[0]

    @pytest.mark.parametrize(
        "source",
        [
            # on the test
            """
            from tests._growth_timing import assert_grows_linearly, linux_only

            @linux_only
            def test_a():
                assert_grows_linearly(len, str, 1)
            """,
            # under a parametrize, imported under another name
            """
            import pytest
            from tests._growth_timing import assert_grows_linearly as grows
            from tests._growth_timing import linux_only as only_linux

            @pytest.mark.parametrize("n", [1, 2])
            @only_linux
            def test_a(n):
                grows(len, str, n)
            """,
            # on the class
            """
            from tests._growth_timing import assert_grows_linearly, linux_only

            @linux_only
            class TestA:
                def test_a(self):
                    assert_grows_linearly(len, str, 1)
            """,
            # module-wide, as a list
            """
            import pytest
            from tests._growth_timing import assert_grows_linearly, linux_only

            pytestmark = [pytest.mark.unit, linux_only]

            def _check():
                assert_grows_linearly(len, str, 1)

            def test_a():
                _check()
            """,
            # through the module
            """
            from tests import _growth_timing

            @_growth_timing.linux_only
            def test_a():
                _growth_timing.assert_grows_linearly(len, str, 1)
            """,
        ],
    )
    def test_every_way_of_carrying_the_gate_is_accepted(self, source):
        imports, calls, problems = scan_source(textwrap.dedent(source))
        assert imports and calls == 1 and problems == []

    @pytest.mark.parametrize(
        "source, expected",
        [
            # the mark is on a neighbour, not on the test that times
            (
                """
                from tests._growth_timing import assert_grows_linearly, linux_only

                @linux_only
                def test_a():
                    pass

                def test_b():
                    assert_grows_linearly(len, str, 1)
                """,
                "test_b",
            ),
            # the class holds a gated test and an ungated one
            (
                """
                from tests._growth_timing import assert_grows_linearly, linux_only

                class TestA:
                    @linux_only
                    def test_a(self):
                        pass

                    def test_b(self):
                        assert_grows_linearly(len, str, 1)
                """,
                "test_b",
            ),
            # a decorator of the same name that is not the shared one
            (
                """
                import pytest
                from tests._growth_timing import assert_grows_linearly

                @pytest.mark.skipif(False, reason="x")
                def test_a():
                    assert_grows_linearly(len, str, 1)
                """,
                "test_a",
            ),
            # a helper hides the call and the module is not gated
            (
                """
                from tests._growth_timing import assert_grows_linearly, linux_only

                def _check():
                    assert_grows_linearly(len, str, 1)

                @linux_only
                def test_a():
                    _check()
                """,
                "_check()",
            ),
            # a call that runs at import time
            (
                """
                from tests._growth_timing import assert_grows_linearly, linux_only

                assert_grows_linearly(len, str, 1)
                """,
                "import time",
            ),
        ],
    )
    def test_each_way_of_missing_the_gate_is_caught(self, source, expected):
        _imports, calls, problems = scan_source(textwrap.dedent(source), "test_x.py")
        assert calls == 1 and len(problems) == 1, problems
        assert expected in problems[0]

    def test_a_file_that_does_not_import_the_helper_is_not_scanned(self):
        source = "def test_a():\n    assert_grows_linearly(len, str, 1)\n"
        assert scan_source(source) == (False, 0, [])
