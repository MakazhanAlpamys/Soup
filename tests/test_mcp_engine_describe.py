"""Direct registry contract for the startup-only MCP engine description."""

from __future__ import annotations

import builtins
import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest

from soup_cli import __version__
from soup_cli.mcp_server import registry as reg

_DESCRIBE = "engine_describe"
_INVALID_ARGUMENTS = "soup.engine.invalid_arguments"
_PRIVATE = "private-engine-input"
_READ_NAMES = (
    "advise",
    "data_doctor",
    "data_inspect",
    "data_score",
    "data_validate",
    "diagnose_evidence",
    "engine_describe",
    "profile",
    "recipes_search",
    "recipes_show",
    "registry_list",
    "registry_show",
    "runs_list",
    "runs_show",
    "ship_evidence",
)
_PLAN_NAMES = ("export", "train_start")
_EXECUTE_NAMES = ("export_execute", "train_execute")


def test_registered_tool_names_match_client_constraints():
    names = [spec.name for spec in reg.build_registry(allow_mutating=True, allow_execute=True)]

    for name in names:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name), name
    assert "engine_describe" in names


@pytest.mark.parametrize(
    ("mutating", "execute", "size"),
    [
        pytest.param(False, False, 15, id="read"),
        pytest.param(True, False, 17, id="plan"),
        pytest.param(False, True, 19, id="execute-implies-mutating"),
        pytest.param(True, True, 19, id="explicit-both"),
    ],
)
def test_manifest_pins_exact_sorted_names_levels_and_effective_gates(mutating, execute, size):
    registry = reg.build_registry(allow_mutating=mutating, allow_execute=execute)
    expected = {name: "read" for name in _READ_NAMES}
    if size >= 17:
        expected.update(dict.fromkeys(_PLAN_NAMES, "plan"))
    if size == 19:
        expected.update(dict.fromkeys(_EXECUTE_NAMES, "execute"))

    # The manifest belongs to the built registry. Query it without the dispatch mutating flag.
    manifest = reg.call_tool(_DESCRIBE, registry=registry)

    assert set(manifest) == {"contract_version", "soup_version", "gates", "tools"}
    assert type(manifest["contract_version"]) is int
    assert manifest["contract_version"] == 1
    assert isinstance(manifest["soup_version"], str)
    assert manifest["soup_version"] == __version__
    assert set(manifest["gates"]) == {"mutating", "execute"}
    assert manifest["gates"]["mutating"] is (size >= 17)
    assert manifest["gates"]["execute"] is execute
    assert isinstance(manifest["tools"], list)
    assert len(manifest["tools"]) == size
    assert manifest["tools"] == [
        {"name": name, "level": level} for name, level in sorted(expected.items())
    ]
    # The list of tool specs still includes callable stubs behind closed gates.
    assert len(registry) == len(_READ_NAMES) + len(_PLAN_NAMES) + len(_EXECUTE_NAMES)


@pytest.mark.parametrize("transport", [None, "stdio", "http", "sse"])
def test_manifest_carries_only_the_configured_static_transport(transport):
    table = reg.build_registry(allow_mutating=False, transport=transport)

    manifest = reg.call_tool(_DESCRIBE, registry=table)

    if transport is None:
        assert "transport" not in manifest
    else:
        assert type(manifest["transport"]) is str
        assert manifest["transport"] == transport


def test_transport_uses_the_static_enum_without_echoing_an_invalid_value():
    with pytest.raises(ValueError, match="MCP transport") as fault:
        reg.build_registry(allow_mutating=False, transport=_PRIVATE)
    assert _PRIVATE not in str(fault.value)


def test_omitted_and_empty_arguments_succeed_with_the_empty_object_schema():
    table = reg.build_registry(allow_mutating=False)
    descriptor = next(spec for spec in table if spec.name == _DESCRIBE)

    assert descriptor.input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert descriptor.level == "read"
    assert descriptor.mutating is False
    assert descriptor.annotations["readOnlyHint"] is True

    expected = reg.call_tool(_DESCRIBE, {}, registry=table)
    assert reg.call_tool(_DESCRIBE, registry=table) == expected
    assert reg.call_tool(_DESCRIBE, None, registry=table) == expected
    assert reg.call_tool(_DESCRIBE, {}) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param(0.0, id="float-zero"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-array"),
        pytest.param((), id="empty-tuple-in-process"),
        pytest.param(True, id="true"),
        pytest.param(4, id="integer"),
        pytest.param(_PRIVATE, id="string"),
        pytest.param([_PRIVATE], id="array"),
        pytest.param({_PRIVATE: "plain"}, id="extra-key"),
        pytest.param({"extra": _PRIVATE}, id="extra-value"),
    ],
)
def test_bad_arguments_are_rejected_with_a_fixed_safe_message(arguments):
    table = reg.build_registry(allow_mutating=False)
    with pytest.raises(reg.McpToolError) as baseline:
        reg.call_tool(_DESCRIBE, {"unexpected": True}, registry=table)

    with pytest.raises(reg.McpToolError) as excinfo:
        reg.call_tool(_DESCRIBE, arguments, registry=table)

    message = str(excinfo.value)
    assert message.split(":", 1)[0] == _INVALID_ARGUMENTS
    assert message == str(baseline.value)
    assert _PRIVATE not in message


def test_handler_returns_fresh_nested_objects_each_time():
    table = reg.build_registry(allow_mutating=True, transport="http")
    descriptor = next(spec for spec in table if spec.name == _DESCRIBE)
    expected = reg.call_tool(_DESCRIBE, registry=table)

    first = descriptor.handler({})
    first["gates"]["mutating"] = False
    first["tools"][0]["level"] = "other"
    first["tools"].pop()
    first["transport"] = "stdio"

    second = descriptor.handler({})
    assert second == expected
    assert reg.call_tool(_DESCRIBE, registry=table) == expected


def _probe_spec(handler, name="probe"):
    return reg.ToolSpec(
        name=name,
        title="Probe",
        description="A test-only registry entry.",
        input_schema={"type": "object"},
        handler=handler,
        level="read",
        annotations={"readOnlyHint": True},
    )


@pytest.mark.parametrize(
    "declare", ["_readonly_specs", "_mutating_specs"], ids=["read", "mutating"],
)
def test_registry_rejects_reserved_engine_describe_collision(monkeypatch, declare):
    original = getattr(reg, declare)
    collision = _probe_spec(lambda _: {}, name="engine_describe")
    monkeypatch.setattr(reg, declare, lambda **kwargs: [*original(**kwargs), collision])

    with pytest.raises(ValueError, match="'engine_describe' is reserved"):
        reg.build_registry(allow_mutating=True, allow_execute=True)


def test_manifest_snapshots_the_shared_tool_declaration_at_build_time(monkeypatch):
    probe = _probe_spec(lambda _: {}, name="added-read")
    original = reg._readonly_specs
    monkeypatch.setattr(reg, "_readonly_specs", lambda: [*original(), probe])
    table = reg.build_registry(allow_mutating=False)

    expected = reg.call_tool(_DESCRIBE, registry=table)
    assert expected["tools"][0] == {"name": "added-read", "level": "read"}
    assert len(expected["tools"]) == 16

    table.remove(probe)
    assert reg.call_tool(_DESCRIBE, registry=table) == expected


@pytest.mark.parametrize("arguments", [False, 0, "", [], {"extra": "retained"}])
def test_other_tools_keep_falsey_normalization_and_the_supplied_registry(arguments):
    table = [_probe_spec(lambda args: {"received": args})]

    result = reg.call_tool("probe", arguments, registry=table)

    assert result == {"received": arguments or {}}


@pytest.mark.parametrize(
    ("name", "allow_mutating", "error_word"),
    [
        ("train_start", False, "allow_mutating=True"),
        ("train_start", True, "--allow-mutating"),
        ("export_execute", True, "--allow-execute"),
    ],
)
def test_existing_stubs_still_refuse_at_their_original_gate(name, allow_mutating, error_word):
    table = reg.build_registry(allow_mutating=False)

    with pytest.raises(reg.McpToolError) as excinfo:
        reg.call_tool(name, {}, registry=table, allow_mutating=allow_mutating)

    assert error_word in str(excinfo.value)


def test_describe_never_accesses_live_state_or_processes_after_build(monkeypatch):
    from soup_cli.experiment import tracker
    from soup_cli.mcp_server import execution
    from soup_cli.registry import store

    accessed = []

    class ExecutionTripwire:
        def __getattr__(self, name):
            accessed.append(f"execution.{name}")
            raise AssertionError(f"describe accessed execution.{name}")

    table = reg.build_registry(
        allow_mutating=True, allow_execute=True, transport="stdio", execution=ExecutionTripwire(),
    )
    targets = (
        (tracker, "ExperimentTracker"),
        (store, "RegistryStore"),
        (sqlite3, "connect"),
        (reg, "ExecutionManager"),
        (execution, "ExecutionManager"),
        (builtins, "open"),
        (Path, "open"),
        (Path, "mkdir"),
        (Path, "stat"),
        (Path, "iterdir"),
        (os, "getcwd"),
        (os, "getenv"),
        (os, "open"),
        (os, "stat"),
        (os, "listdir"),
        (type(os.environ), "get"),
        (type(os.environ), "__getitem__"),
        (subprocess, "run"),
        (subprocess, "Popen"),
    )

    def forbidden(name):
        def fail(*args, **kwargs):
            accessed.append(name)
            raise AssertionError(f"describe accessed {name}")
        return fail

    with monkeypatch.context() as guard:
        for owner, attribute in targets:
            guard.setattr(owner, attribute, forbidden(attribute))
        manifest = reg.call_tool(_DESCRIBE, registry=table)

    assert accessed == []
    assert manifest["gates"] == {"mutating": True, "execute": True}
    assert manifest["transport"] == "stdio"
