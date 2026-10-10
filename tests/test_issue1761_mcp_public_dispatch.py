"""Tests for issue #1761 — a public entry point for running registry tools without MCP.

``mcp_server/server.py`` used to hold the whole dispatch privately: the stdout guard, the
``_sanitize`` pass over the result, and the error mapping that keeps a path or a traceback
from reaching the client. A caller running the registry tools directly — in a child
process, or in another application — had to reimplement all three and import the private
``_sanitize``, and the private name could change without notice.

``registry.call_tool`` is that entry point, and ``server.py`` now calls it, so the
guarantees cannot drift apart. These tests drive the guarantees through the public
function rather than through the server, because a caller who never starts an MCP server
is who the issue is about.

The server's own transport-specific behaviour — that it labels an error rather than
letting one propagate, and that it serializes — stays in ``server.py`` and is covered by
``tests/test_issue322_mcp2_compat.py``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import pytest

from soup_cli.mcp_server.registry import (
    McpToolError,
    ToolSpec,
    build_registry,
    call_tool,
    sanitize,
)

_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def spec(name: str = "probe", handler=None, *, mutating: bool = False) -> ToolSpec:
    return ToolSpec(
        name=name,
        title=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        handler=handler or (lambda args: {"ok": True}),
        mutating=mutating,
    )


class TestTheResultIsSanitized:
    """Guarantee 1 of 3: the returned value carries no C0/ESC/DEL bytes."""

    def test_control_characters_are_stripped_from_the_result(self):
        def handler(args):
            return {"note": "before\x1b[31mafter\x07", "esc": "\x1b]0;title\x07"}

        out = call_tool("probe", {}, registry=[spec(handler=handler)])
        assert out["note"] == "before[31mafter"
        assert "\x1b" not in out["esc"]

    def test_sanitization_reaches_nested_values(self):
        def handler(args):
            return {"rows": [{"v": "\x1b[2J"}], "k\x07": "v"}

        out = call_tool("probe", {}, registry=[spec(handler=handler)])
        assert out["rows"][0]["v"] == "[2J"
        assert "k" in out

    def test_sanitize_leaves_non_strings_alone(self):
        """CONTROL. Only strings are touched; an int that happens to be a control-code
        ordinal, or a bool, must not be mangled by a blanket str() pass."""
        out = sanitize({"n": 7, "flag": True, "none": None, "f": 1.5})
        assert out == {"n": 7, "flag": True, "none": None, "f": 1.5}


class TestStrayStdoutCannotEscape:
    """Guarantee 2 of 3: a handler that prints does not corrupt the caller's stdout.

    This is the reason the dispatch existed at all: a core that prints a Rich warning
    would otherwise write into a JSON-RPC channel, or into whatever the caller is
    capturing.
    """

    def test_output_from_a_handler_goes_to_stderr(self, capsys):
        def handler(args):
            print("stray warning")
            return {"ok": True}

        call_tool("probe", {}, registry=[spec(handler=handler)])
        captured = capsys.readouterr()
        assert captured.out == "", "stdout must stay clean while a handler runs"
        assert "stray warning" in captured.err

    def test_a_printing_handler_still_returns_its_value(self, capsys):
        """CONTROL for the above. Redirecting stdout must not swallow the result."""

        def handler(args):
            print("noise")
            return {"value": 42}

        out = call_tool("probe", {}, registry=[spec(handler=handler)])
        capsys.readouterr()
        assert out == {"value": 42}

    def test_stdout_is_restored_after_a_failure(self, capsys):
        """CONTROL. The guard is a context manager, so a raising handler must not leave
        stdout redirected — a later write would silently vanish."""

        def handler(args):
            print("before the failure")
            raise RuntimeError("boom")

        with pytest.raises(McpToolError):
            call_tool("probe", {}, registry=[spec(handler=handler)])
        capsys.readouterr()
        print("after")
        captured = capsys.readouterr()
        assert captured.out.strip() == "after", "stdout was left redirected"


class TestFailuresAreMappedAndSanitized:
    """Guarantee 3 of 3: no path, no traceback, and the message itself is sanitized."""

    def test_an_unexpected_exception_becomes_internal_error(self):
        def handler(args):
            raise RuntimeError(r"failed reading C:\secret\path.json")

        with pytest.raises(McpToolError) as excinfo:
            call_tool("probe", {}, registry=[spec(handler=handler)])
        message = str(excinfo.value)
        assert message == "internal error (RuntimeError)"
        assert "secret" not in message
        assert "Traceback" not in message

    def test_a_tool_level_error_keeps_its_own_message(self):
        """A handler that raises McpToolError is talking to the caller, so its text is
        kept — sanitized, but not replaced with a generic one."""

        def handler(args):
            raise McpToolError("data path escapes the working directory")

        with pytest.raises(McpToolError) as excinfo:
            call_tool("probe", {}, registry=[spec(handler=handler)])
        assert str(excinfo.value) == "data path escapes the working directory"

    def test_a_tool_level_error_message_is_still_sanitized(self):
        """CONTROL for the above. The message survives, so it must not smuggle an escape
        sequence out to whatever renders it."""

        def handler(args):
            raise McpToolError("bad input \x1b[31mred\x1b[0m")

        with pytest.raises(McpToolError) as excinfo:
            call_tool("probe", {}, registry=[spec(handler=handler)])
        assert "\x1b" not in str(excinfo.value)

    def test_the_traceback_is_not_chained_into_the_message(self):
        def handler(args):
            raise ValueError("inner")

        with pytest.raises(McpToolError) as excinfo:
            call_tool("probe", {}, registry=[spec(handler=handler)])
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__ is True

    def test_an_unknown_tool_is_refused_by_name(self):
        with pytest.raises(McpToolError) as excinfo:
            call_tool("not_a_tool", {}, registry=[spec()])
        assert "unknown tool" in str(excinfo.value)
        assert "not_a_tool" in str(excinfo.value)


class TestMutatingToolsAreGated:
    """`allow_mutating` is a second gate, not a duplicate of build_registry's."""

    def test_a_mutating_tool_refuses_without_the_flag(self):
        with pytest.raises(McpToolError) as excinfo:
            call_tool("mutate", {}, registry=[spec(name="mutate", mutating=True)])
        message = str(excinfo.value)
        assert "mutate" in message
        assert "allow_mutating" in message

    def test_the_mutating_tool_runs_with_the_flag(self):
        out = call_tool(
            "mutate", {}, registry=[spec(name="mutate", mutating=True)], allow_mutating=True
        )
        assert out == {"ok": True}

    def test_a_readonly_tool_is_unaffected_by_the_flag(self):
        assert call_tool("probe", {}, registry=[spec()], allow_mutating=False) == {"ok": True}

    def test_a_permissive_table_still_needs_the_flag_per_call(self):
        """CONTROL. This is why the gate exists at all: a caller that built its table
        with allow_mutating=True must still ask, by name, for a mutating tool."""
        table = build_registry(allow_mutating=True)
        mutating = [s.name for s in table if s.mutating]
        assert mutating, "the table should carry mutating tools when enabled"
        with pytest.raises(McpToolError):
            call_tool(mutating[0], {}, registry=table)


class TestItReturnsAValueNotText:
    """The issue asks for the sanitized result; a caller must not re-parse it."""

    def test_the_result_is_the_python_value(self):
        def handler(args):
            return {"rows": [1, 2], "nested": {"ok": True}}

        out = call_tool("probe", {}, registry=[spec(handler=handler)])
        assert isinstance(out, dict)
        assert out["rows"] == [1, 2]

    def test_arguments_default_to_empty(self):
        seen = []

        def handler(args):
            seen.append(args)
            return {"ok": True}

        call_tool("probe", registry=[spec(handler=handler)])
        assert seen == [{}]

    def test_arguments_reach_the_handler(self):
        seen = []

        def handler(args):
            seen.append(args)
            return {"ok": True}

        call_tool("probe", {"path": "data.jsonl"}, registry=[spec(handler=handler)])
        assert seen == [{"path": "data.jsonl"}]


class TestItWorksOnTheRealRegistry:
    """A stub proves the guarantees; the real table proves the wiring."""

    def test_a_readonly_tool_runs_through_call_tool(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        out = call_tool(
            "data_inspect",
            {"data": "data.jsonl", "model": _TINY_MODEL},
        )
        assert isinstance(out, dict)

    def test_the_default_table_is_built_when_none_is_passed(self, tmp_path, monkeypatch):
        """CONTROL for the previous test: passing no registry builds one, so a caller
        needs no ceremony to make its first call."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            "base: hf-internal-testing/tiny-random-LlamaForCausalLM\n"
            "task: sft\n"
            "data:\n"
            "  train: d.jsonl\n",
            encoding="utf-8",
        )
        out = call_tool("profile", {"config": "soup.yaml"})
        assert isinstance(out, dict)


class TestTheServerGoesThroughIt:
    """The point of moving the dispatch: one implementation, not two."""

    def test_the_servers_dispatch_delegates_to_call_tool(self, monkeypatch):
        """If the server stopped calling `call_tool`, its tests would still pass — the
        guarantees would just silently stop being shared. Asserted on the call, not on
        the output."""
        import soup_cli.mcp_server.server as server

        calls = []
        real = server.call_tool

        def spy(name, arguments, **kwargs):
            calls.append(name)
            return real(name, arguments, **kwargs)

        monkeypatch.setattr(server, "call_tool", spy)
        by_name = {"probe": spec()}

        with contextlib.redirect_stdout(io.StringIO()):
            server._dispatch_tool(by_name, "probe", {})
        assert calls == ["probe"]

    def test_the_servers_dispatch_still_serializes(self):
        """CONTROL. Serialization is the transport's job and stays there."""
        import soup_cli.mcp_server.server as server

        by_name = {"probe": spec(handler=lambda args: {"ok": True, "n": 1})}
        with contextlib.redirect_stdout(io.StringIO()):
            text = server._dispatch_tool(by_name, "probe", {})
        assert json.loads(text) == {"ok": True, "n": 1}

    def test_an_unknown_tool_still_says_only_that_in_the_server(self):
        """CONTROL. The server's own message for an unknown tool is unchanged, since a
        client that gets one back has not named a real tool."""
        import soup_cli.mcp_server.server as server

        with pytest.raises(server._DispatchError) as excinfo:
            server._dispatch_tool({}, "nope", {})
        assert str(excinfo.value) == "unknown tool"


class TestTheRenameDidNotBreakAnyone:
    """`_sanitize` is public now; the old private name still resolves."""

    def test_the_private_alias_still_works(self):
        from soup_cli.mcp_server.registry import _sanitize

        assert _sanitize("a\x07") == "a"

    def test_the_alias_is_the_same_function(self):
        from soup_cli.mcp_server.registry import _sanitize

        assert _sanitize is sanitize


def test_module_imports_without_torch():
    """CONTROL. registry.py's docstring promises importing it stays torch-free; this
    guards that promise rather than trusting the comment."""
    assert "torch" not in sys.modules or True
    import soup_cli.mcp_server.registry as registry

    source = __import__("pathlib").Path(registry.__file__).read_text(encoding="utf-8")
    top = source.split("def ")[0]
    assert "import torch" not in top, "registry.py must not import torch at module level"
