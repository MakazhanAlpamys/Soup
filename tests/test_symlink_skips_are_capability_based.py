"""Repo-wide ratchet: a symlink test is gated on whether this account can create one (#832).

The suite used to gate symlink tests by **platform**: ``skipif(os.name == "nt")`` with some
twenty spellings of "POSIX symlink", imperative ``if os.name == "nt": pytest.skip(...)``, and
hand-rolled ``try: os.symlink(...) except OSError: pytest.skip(...)`` probes. Windows creates
symlinks fine with Developer Mode on or when elevated, and the Windows CI cells can, so about a
hundred path-containment and symlink-refusal tests were skipped there for no reason. One real
Windows defect was hiding behind them (#820). Meanwhile four tests had no guard at all and failed
with ``WinError 1314`` on an account without the privilege.

The rule this file enforces:

- A test that creates a symlink carries ``@pytest.mark.requires_symlink`` (on the test, its class,
  or the module's ``pytestmark``). ``tests/conftest.py`` probes once per session and skips it with
  a reason that names the missing capability.
- A platform skip whose reason mentions symlinks is allowed only when it names the POSIX behaviour
  the test needs (``O_NOFOLLOW`` today). "POSIX symlink semantics" does not say which.

``TestTheScannerCanActuallyFail`` is load-bearing: at the time of writing the scanner finds zero
offenders, and a scanner with nothing to find must still be shown able to find something.
"""

from __future__ import annotations

import ast
import pathlib
import re
import textwrap

TESTS_DIR = pathlib.Path(__file__).parent

_SYMLINK = re.compile(r"symlink|symbolic link", re.IGNORECASE)
_PLATFORM = re.compile(
    r"\bos\.name\b|\bsys\.platform\b|\bplatform\.system\(|__import__\('os'\)\.name"
    r"|__import__\('sys'\)\.platform|hasattr\(_?os, 'symlink'\)"
)
#: POSIX behaviours a platform skip may name instead of "symlink semantics".
_NAMED_POSIX_BEHAVIOUR = re.compile(r"\bO_NOFOLLOW\b")
_CREATES_SYMLINK = re.compile(r"(?<![\w.])_?os\.symlink\(|\.symlink_to\(")


def _reason(call: ast.Call) -> str:
    for keyword in call.keywords:
        if keyword.arg == "reason":
            return ast.unparse(keyword.value)
    return " ".join(ast.unparse(arg) for arg in call.args[1:])


def _has_requires_symlink(decorators: list[ast.expr]) -> bool:
    return any(ast.unparse(d) == "pytest.mark.requires_symlink" for d in decorators)


def scan_source(source: str, filename: str = "<string>") -> list[str]:
    """Return one ``file:line: problem`` string per offending site in ``source``."""
    tree = ast.parse(source, filename=filename)
    problems = []

    module_marked = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
        and "pytest.mark.requires_symlink" in ast.unparse(node.value)
        for node in tree.body
    )

    for node in ast.walk(tree):
        is_skipif = isinstance(node, ast.Call) and ast.unparse(node.func).endswith("mark.skipif")
        if is_skipif and node.args:
            condition = ast.unparse(node.args[0])
            reason = _reason(node)
            if (
                _PLATFORM.search(condition)
                and _SYMLINK.search(reason + condition)
                and not _NAMED_POSIX_BEHAVIOUR.search(reason)
            ):
                problems.append(
                    f"{filename}:{node.lineno}: platform skipif for a symlink test; use "
                    "@pytest.mark.requires_symlink, or name the POSIX behaviour in the reason"
                )

        if isinstance(node, ast.If) and _PLATFORM.search(ast.unparse(node.test)):
            for stmt in node.body:
                text = ast.unparse(stmt)
                about_symlinks = _SYMLINK.search(text + ast.unparse(node.test))
                if text.startswith("pytest.skip(") and about_symlinks:
                    if not _NAMED_POSIX_BEHAVIOUR.search(text):
                        problems.append(
                            f"{filename}:{node.lineno}: imperative platform skip for a symlink "
                            "test; use @pytest.mark.requires_symlink"
                        )

        if isinstance(node, ast.Try):
            body = "\n".join(ast.unparse(s) for s in node.body)
            handlers = [ast.unparse(s) for h in node.handlers for s in h.body]
            skips = any(h.startswith("pytest.skip(") for h in handlers)
            if _CREATES_SYMLINK.search(body) and skips:
                problems.append(
                    f"{filename}:{node.lineno}: hand-rolled symlink probe; "
                    "use @pytest.mark.requires_symlink"
                )

    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        if not _CREATES_SYMLINK.search(ast.unparse(node)):
            continue
        owner = next((c for c in classes if node in c.body), None)
        marked = (
            module_marked
            or _has_requires_symlink(node.decorator_list)
            or (owner is not None and _has_requires_symlink(owner.decorator_list))
        )
        guarded_by_named_skip = any(
            isinstance(d, ast.Call) and _NAMED_POSIX_BEHAVIOUR.search(ast.unparse(d))
            for d in node.decorator_list
        )
        if not marked and not guarded_by_named_skip:
            problems.append(
                f"{filename}:{node.lineno}: {node.name} creates a symlink without "
                "@pytest.mark.requires_symlink"
            )
    return problems


def test_symlink_tests_are_gated_on_capability_not_platform() -> None:
    problems = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        problems.extend(scan_source(path.read_text(encoding="utf-8"), path.name))
    assert not problems, "\n".join(problems)


class TestTheScannerCanActuallyFail:
    def test_flags_a_platform_skipif(self) -> None:
        source = textwrap.dedent(
            """
            import os
            import pytest

            @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
            def test_x(tmp_path):
                pass
            """
        )
        assert len(scan_source(source)) == 1

    def test_flags_a_module_level_marker(self) -> None:
        source = textwrap.dedent(
            """
            import sys
            import pytest

            POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="symlink needs admin")
            """
        )
        assert len(scan_source(source)) == 1

    def test_flags_an_imperative_platform_skip(self) -> None:
        source = textwrap.dedent(
            """
            import os
            import pytest

            @pytest.mark.requires_symlink
            def test_x(tmp_path):
                if os.name == "nt":
                    pytest.skip("symlink semantics differ on Windows")
                os.symlink(tmp_path / "a", tmp_path / "b")
            """
        )
        assert len(scan_source(source)) == 1

    def test_flags_a_hand_rolled_probe(self) -> None:
        source = textwrap.dedent(
            """
            import os
            import pytest

            @pytest.mark.requires_symlink
            def test_x(tmp_path):
                try:
                    os.symlink(tmp_path / "a", tmp_path / "b")
                except OSError:
                    pytest.skip("symlinks unavailable")
            """
        )
        assert len(scan_source(source)) == 1

    def test_flags_an_unguarded_symlink_test(self) -> None:
        source = textwrap.dedent(
            """
            def test_x(tmp_path):
                (tmp_path / "link").symlink_to(tmp_path / "real")
            """
        )
        assert len(scan_source(source)) == 1

    def test_accepts_the_marker_on_a_test_its_class_or_the_module(self) -> None:
        source = textwrap.dedent(
            """
            import os
            import pytest

            @pytest.mark.requires_symlink
            def test_a(tmp_path):
                os.symlink(tmp_path / "a", tmp_path / "b")

            @pytest.mark.requires_symlink
            class TestB:
                def test_b(self, tmp_path):
                    (tmp_path / "link").symlink_to(tmp_path / "real")
            """
        )
        assert scan_source(source) == []
        header = "import pytest\npytestmark = pytest.mark.requires_symlink\n"
        module_marked = header + textwrap.dedent(
            """
            def test_c(tmp_path):
                (tmp_path / "link").symlink_to(tmp_path / "real")
            """
        )
        assert scan_source(module_marked) == []

    def test_accepts_a_platform_skip_that_names_the_posix_behaviour(self) -> None:
        source = textwrap.dedent(
            """
            import os
            import pytest

            @pytest.mark.skipif(os.name == "nt", reason="relies on O_NOFOLLOW (#820)")
            def test_x(tmp_path):
                os.symlink(tmp_path / "a", tmp_path / "b")
            """
        )
        assert scan_source(source) == []


def test_can_symlink_probe_matches_this_account(tmp_path) -> None:
    import os

    from tests.conftest import can_symlink

    try:
        os.symlink(tmp_path / "target", tmp_path / "link")
    except (OSError, NotImplementedError):
        expected = False
    else:
        expected = True
    assert can_symlink() is expected
