"""Issue #322 — `soup mcp serve` on both mcp 1.x and 2.x.

mcp 2.0.0 removed two things Soup was built on: the
``@server.list_tools()`` / ``@server.call_tool()`` decorators, and
``mcp.shared.memory.create_connected_server_and_client_session``. v0.72.3
capped the extra at ``<2`` to unblock a release, which pinned anyone who wanted
2.x in the same environment.

The bridge is a **capability probe**, not a version comparison. That choice is
not stylistic: this repository has twice derived a bound by reading source or a
version table and been wrong both times — see ``trainer/_trl_compat.py``, whose
module docstring records the rule it earned. A probe cannot go stale, because it
asks the object in front of it.

The round-trip behaviour itself is asserted by ``test_v07128.py``'s
``TestServerRoundTrip``, which now runs unchanged on either major. What is
pinned here is the machinery that lets it: the probe, the shared helpers, and
the fact that neither is a version check in disguise.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import types as pytypes

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SERVER_SRC = _ROOT / "src" / "soup_cli" / "mcp_server" / "server.py"


@pytest.fixture(autouse=True)
def _need_mcp():
    pytest.importorskip("mcp")


class TestTheProbeMatchesTheInstalledSdk:
    def test_probe_agrees_with_the_server_constructor(self):
        """The probe's answer must be the SDK's answer, whichever major is here."""
        from mcp.server.lowlevel import Server

        from soup_cli.mcp_server.server import _uses_callback_handlers

        takes_callbacks = "on_list_tools" in inspect.signature(Server.__init__).parameters
        assert _uses_callback_handlers() is takes_callbacks

    def test_the_two_registration_styles_are_mutually_exclusive(self):
        """A build that satisfied neither branch would construct a server with no
        handlers at all and fail only at call time, so pin that exactly one of
        the two registration surfaces exists."""
        from mcp.server.lowlevel import Server

        from soup_cli.mcp_server.server import _uses_callback_handlers

        has_decorators = hasattr(Server, "list_tools") and hasattr(Server, "call_tool")
        assert has_decorators is not _uses_callback_handlers()

    def test_the_probe_is_not_a_version_comparison(self):
        """The rule `_trl_compat.py` earned, enforced rather than remembered.

        Read as syntax rather than as text: the module docstring *mentions*
        ``mcp.__version__`` precisely to say it is not consulted, so a
        substring check fails on the sentence that explains the rule.
        """
        import ast

        tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "__version__", ast.unparse(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("version", "parse_version"), ast.unparse(node)


class TestTheServerBuildsOnWhicheverMajorIsInstalled:
    def test_build_server_advertises_every_spec(self):
        from soup_cli.mcp_server.registry import build_registry
        from soup_cli.mcp_server.server import build_server

        specs = build_registry(allow_mutating=False, allow_execute=False)
        assert build_server(specs) is not None

    def test_the_shared_dispatch_is_used_by_both_adapters(self):
        """Both branches must route through one implementation. Two copies of
        the sanitize/serialize logic is how the majors would drift apart in
        behaviour while both kept passing their own tests."""
        import ast

        tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
        definitions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_dispatch_tool"
        ]
        call_sites = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_dispatch_tool"
        ]
        assert len(definitions) == 1, "two implementations would drift apart"
        assert len(call_sites) == 2, "one call site per SDK major, no more"

    def test_an_unknown_tool_reports_an_error_rather_than_raising(self):
        from soup_cli.mcp_server.server import _dispatch_tool, _DispatchError

        with pytest.raises(_DispatchError) as excinfo:
            _dispatch_tool({}, "no_such_tool", {})
        assert "unknown tool" in str(excinfo.value)

    def test_a_handler_failure_carries_no_path_or_stack(self):
        from soup_cli.mcp_server.server import _dispatch_tool, _DispatchError

        class _Spec:
            name = "boom"

            @staticmethod
            def handler(_args):
                raise RuntimeError(r"C:\secret\path.py exploded")

        with pytest.raises(_DispatchError) as excinfo:
            _dispatch_tool({"boom": _Spec}, "boom", {})
        message = str(excinfo.value)
        assert message == "internal error (RuntimeError)"
        assert "secret" not in message


class TestTheDualMajorReaders:
    """The field renames are read by meaning, and refuse to guess."""

    def test_is_error_reads_either_spelling(self):
        from tests.mcp_roundtrip import is_error

        assert is_error(pytypes.SimpleNamespace(isError=True)) is True
        assert is_error(pytypes.SimpleNamespace(is_error=True)) is True
        assert is_error(pytypes.SimpleNamespace(isError=False)) is False

    def test_is_error_raises_when_the_field_moves_again(self):
        """Returning False here would turn a third rename into "no tool call
        ever failed" -- a silent pass is the one outcome a test helper must not
        produce."""
        from tests.mcp_roundtrip import is_error

        with pytest.raises(AttributeError):
            is_error(pytypes.SimpleNamespace())

    def test_input_schema_reads_either_spelling(self):
        from tests.mcp_roundtrip import input_schema

        assert input_schema(pytypes.SimpleNamespace(inputSchema={"type": "object"}))
        assert input_schema(pytypes.SimpleNamespace(input_schema={"type": "object"}))

    def test_input_schema_raises_when_the_field_moves_again(self):
        from tests.mcp_roundtrip import input_schema

        with pytest.raises(AttributeError):
            input_schema(pytypes.SimpleNamespace())


class TestTheConstraintNoLongerExcludes2x:
    def test_the_extra_admits_a_2x_release(self):
        """The acceptance box asks that the constraint stop excluding 2.x.
        Asserted against the specifier itself rather than the literal text, so
        reformatting the pin cannot quietly re-exclude it."""
        from packaging.requirements import Requirement

        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        line = next(ln for ln in pyproject.splitlines() if ln.startswith("mcp = ["))
        spec = Requirement(re.search(r'"([^"]+)"', line).group(1)).specifier
        assert "2.0.0" in spec
        assert "1.10.0" in spec
        # The floor #296 measured must not come back down.
        assert "1.9.4" not in spec
        # ...and the CEILING must stay pinned too. The thesis of this PR is
        # that a bound nobody has verified should not ship: mcp 2.x was
        # exercised before lifting the cap to <3, and 3.x has been exercised
        # by nobody. Without these, deleting the upper bound entirely -- or
        # widening it to <4 -- leaves the suite green, which is the exact
        # failure class the cap exists to prevent (review of #498).
        assert "3.0.0" not in spec
        assert "4.0.0" not in spec

    def test_the_roundtrip_helper_is_not_collected_as_a_test_module(self):
        """`tests/mcp_roundtrip.py` is imported by test modules; if it were ever
        renamed to `test_*` pytest would collect its helpers as tests."""
        assert (_ROOT / "tests" / "mcp_roundtrip.py").is_file()
        assert not (_ROOT / "tests" / "test_mcp_roundtrip.py").exists()
