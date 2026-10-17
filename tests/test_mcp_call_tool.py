"""Tests for issue #1761 — a public entry point for running registry tools without MCP.

``mcp_server/server.py`` used to hold the whole dispatch privately: the stdout guard, the
``_sanitize`` pass over the result, and the error mapping that keeps a path or a traceback
from reaching the client. A caller running the registry tools directly — in a child
process, or in another application — had to reimplement all three and import the private
``_sanitize``, and the private name could change without notice.

``registry.call_tool`` is the helper ``server.py`` calls, so the
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
import subprocess
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

    def test_a_hashable_non_string_name_is_an_unknown_tool(self):
        """CONTROL for the test below. An int or None reaches the lookup and misses it
        cleanly, so the escape is specific to names the dict lookup cannot even key on."""
        for bad in (5, None):
            with pytest.raises(McpToolError, match="unknown tool"):
                call_tool(bad, {}, registry=[spec()])

    def test_an_unhashable_name_is_a_type_error_not_a_tool_error(self):
        """Pinned deliberately. `call_tool(["x"])` raises TypeError from the lookup, not
        McpToolError: passing a list as a tool name is a caller bug, and a domain error
        would hide it. The docstring says so; this keeps the two in step."""
        with pytest.raises(TypeError):
            call_tool(["x"], {}, registry=[spec()])


class TestMutatingToolsAreGated:
    """`allow_mutating` is a second gate, not a duplicate of build_registry's."""

    def test_an_execute_tool_refuses_without_the_flag(self):
        """The execute family is gated exactly like the rest of the mutating tools, and
        by the same message -- the one that names the flag, not the server flag.

        The gate fires before the handler, so `--allow-execute` never appears on this
        path at all: that text belongs to the server, which passes allow_mutating=True
        unconditionally and so reaches the tool's own refusal.
        """
        table = build_registry(allow_mutating=True, allow_execute=True)
        with pytest.raises(McpToolError) as excinfo:
            call_tool("train_execute", {"confirmation_token": "t"}, registry=table)
        message = str(excinfo.value)
        assert "allow_mutating" in message
        assert "--allow-execute" not in message, (
            "'soup mcp serve --allow-execute' cannot help an in-process caller; "
            "the gate's own message must point at the flag"
        )

    def test_an_execute_table_passes_the_gate_through(self):
        """CONTROL for the test above: with the flag, the gate is past and the tool's own
        refusal is what answers. That is the route an in-process caller needs, and it is
        why `call_tool` takes a `registry` argument at all."""
        table = build_registry(allow_mutating=True, allow_execute=True)
        with pytest.raises(McpToolError) as excinfo:
            call_tool(
                "train_execute",
                {"confirmation_token": "definitely-not-a-real-token"},
                registry=table,
                allow_mutating=True,
            )
        assert "--allow-execute" not in str(excinfo.value)
        assert "confirmation token" in str(excinfo.value)

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
        with allow_mutating=True must still ask, by name, for a mutating tool.

        The handler records that it ran. An earlier cut only asserted that some
        McpToolError came back, which `train_start` raises for its own reasons with
        empty arguments -- so the test passed with the gate deleted and proved nothing.
        Asserting the gate's own message, and that the handler never ran, is what makes
        it fail when the gate is gone.
        """
        ran = []
        table = [spec(name="mutate", handler=lambda args: ran.append(args) or {"ok": True},
                      mutating=True)]

        with pytest.raises(McpToolError) as excinfo:
            call_tool("mutate", {}, registry=table)
        assert "allow_mutating" in str(excinfo.value)
        assert ran == [], "the handler ran; the gate did not hold"

    def test_the_gate_is_what_stopped_it_not_the_handler(self):
        """CONTROL for the test above. The same spec DOES run with the flag, so the
        refusal above is the gate and not something about this tool."""
        ran = []
        table = [spec(name="mutate", handler=lambda args: ran.append(args) or {"ok": True},
                      mutating=True)]
        assert call_tool("mutate", {}, registry=table, allow_mutating=True) == {"ok": True}
        assert ran == [{}]


class TestControlBytesAreSanitizedInBothMessages:
    """Dropping ``sanitize(name)`` from the unknown-tool or the gate-refusal message
    must fail: a control byte in a tool name must not reach the caller."""

    def test_an_unknown_tool_name_is_sanitized(self):
        bad = "x\x1b[31m"
        with pytest.raises(McpToolError) as excinfo:
            call_tool(bad, {}, registry=[spec()])
        message = str(excinfo.value)
        assert "\x1b" not in message
        assert "unknown tool" in message

    def test_a_gated_tool_name_is_sanitized(self):
        bad = "x\x1b[31m"
        with pytest.raises(McpToolError) as excinfo:
            call_tool(bad, {}, registry=[spec(name=bad, mutating=True)])
        message = str(excinfo.value)
        assert "\x1b" not in message
        assert "allow_mutating" in message


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


def test_importing_the_registry_does_not_pull_in_torch():
    """CONTROL. registry.py's docstring promises importing it stays torch-free, which is
    what lets a registry tool run without the training extras. This checks the promise
    by importing it in a fresh interpreter rather than by reading the source: an earlier
    cut asserted `"torch" not in sys.modules or True`, which cannot fail.

    A subprocess is the only honest way to ask it. In-process, this test module has
    already imported half of Soup, so `sys.modules` says nothing about registry.py.
    """
    probe = "import sys, soup_cli.mcp_server.registry as r; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "importing the registry pulled torch in; it is documented as staying cheap"
    )
