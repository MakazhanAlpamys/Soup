"""Every tool ``soup mcp serve`` lists carries an MCP annotation level.

Before this, the 14 read-only tools and the two plan-only tools were advertised with
``annotations=None``, so a client reading ``tools/list`` could not tell a tool that only
reads from one that is gated behind ``--allow-mutating``. A tool now has exactly one of
three levels, and the level is the annotation:

* ``read``    -> ``readOnlyHint: true``
* ``plan``    -> ``readOnlyHint: false, destructiveHint: false`` (additive: a run directory
  and a token under ``--allow-execute``, nothing deleted or overwritten)
* ``execute`` -> ``readOnlyHint: false, destructiveHint: true`` (unchanged)

The pinned table below is deliberately a literal: adding a tool or moving one between
levels is a visible diff here, and it does not import the production helper, so a change
to the helper cannot silently re-pin the expectation.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import soup_cli.mcp_server.registry as reg

#: The annotation each level must produce. Literal on purpose (see module docstring).
LEVEL_ANNOTATIONS = {
    "read": {"readOnlyHint": True},
    "plan": {"readOnlyHint": False, "destructiveHint": False},
    "execute": {"readOnlyHint": False, "destructiveHint": True},
}

#: Every tool the server lists, with its level.
EXPECTED_LEVEL = {
    "advise": "read",
    "data_inspect": "read",
    "data_validate": "read",
    "data_score": "read",
    "data_doctor": "read",
    "recipes_search": "read",
    "recipes_show": "read",
    "runs_list": "read",
    "runs_show": "read",
    "registry_list": "read",
    "registry_show": "read",
    "profile": "read",
    "diagnose_evidence": "read",
    "ship_evidence": "read",
    "train_start": "plan",
    "export": "plan",
    "train_execute": "execute",
    "export_execute": "execute",
}

#: ``openWorldHint`` is set on exactly two tools, where the code settles it. ``data_doctor``
#: loads a tokenizer for a model id the client chose, which can reach the Hub. ``profile``
#: prices a config with ``allow_network=False``: it reads the model's local directory and
#: cache and sends no request. Every other tool is left unset because the code does not
#: settle it (``data_score`` may start presidio, which can fetch a language model; ``advise``
#: has a live-probe path).
EXPECTED_OPEN_WORLD = {"data_doctor": True, "profile": False}

#: The three gate settings ``soup mcp serve`` can run with.
GATES = [
    pytest.param(False, False, id="gates-off"),
    pytest.param(True, False, id="allow-mutating"),
    pytest.param(True, True, id="allow-execute"),
]


def _expected(name: str) -> dict:
    expected = dict(LEVEL_ANNOTATIONS[EXPECTED_LEVEL[name]])
    if name in EXPECTED_OPEN_WORLD:
        expected["openWorldHint"] = EXPECTED_OPEN_WORLD[name]
    return expected


def _specs(allow_mutating: bool = False, allow_execute: bool = False) -> list[reg.ToolSpec]:
    return reg.build_registry(allow_mutating=allow_mutating, allow_execute=allow_execute)


def _names_at(level: str) -> list[str]:
    return [name for name, value in EXPECTED_LEVEL.items() if value == level]


class TestPinnedLevelTable:
    def test_the_table_names_exactly_the_tools_the_registry_builds(self):
        built = {spec.name for spec in _specs()}
        assert built == set(EXPECTED_LEVEL), (
            "a tool was added, removed or renamed: give it a level in EXPECTED_LEVEL",
            sorted(built ^ set(EXPECTED_LEVEL)),
        )

    def test_the_table_has_fourteen_read_two_plan_two_execute(self):
        assert len(_names_at("read")) == 14
        assert len(_names_at("plan")) == 2
        assert len(_names_at("execute")) == 2
        assert len(EXPECTED_LEVEL) == 18

    @pytest.mark.parametrize(("allow_mutating", "allow_execute"), GATES)
    def test_every_spec_has_the_annotations_its_level_pins(
        self, allow_mutating, allow_execute
    ):
        for spec in _specs(allow_mutating, allow_execute):
            assert spec.annotations == _expected(spec.name), spec.name


class TestEachLevel:
    @pytest.mark.parametrize(("allow_mutating", "allow_execute"), GATES)
    def test_every_spec_has_annotations(self, allow_mutating, allow_execute):
        for spec in _specs(allow_mutating, allow_execute):
            assert isinstance(spec.annotations, dict) and spec.annotations, spec.name
            assert isinstance(spec.annotations.get("readOnlyHint"), bool), spec.name

    def test_read_tools_are_read_only_and_not_gated(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("read"):
            assert by_name[name].annotations["readOnlyHint"] is True, name
            assert by_name[name].mutating is False, name

    def test_read_tools_carry_no_destructive_hint(self):
        # MCP defines destructiveHint as meaningful only when readOnlyHint is false.
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("read"):
            assert "destructiveHint" not in by_name[name].annotations, name
            assert "idempotentHint" not in by_name[name].annotations, name

    def test_plan_tools_are_explicitly_not_read_only(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("plan"):
            annotations = by_name[name].annotations
            assert annotations["readOnlyHint"] is False, name
            assert annotations["destructiveHint"] is False, name
            assert by_name[name].mutating is True, name

    def test_execute_tools_keep_what_they_had(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("execute"):
            assert by_name[name].annotations == {
                "readOnlyHint": False,
                "destructiveHint": True,
            }, name
            assert by_name[name].mutating is True, name

    def test_open_world_is_set_only_where_the_code_settles_it(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name, spec in by_name.items():
            if name in EXPECTED_OPEN_WORLD:
                assert spec.annotations["openWorldHint"] is EXPECTED_OPEN_WORLD[name], name
            else:
                assert "openWorldHint" not in spec.annotations, name

    def test_mutating_flag_and_read_only_hint_agree(self):
        # `mutating` is what the gate reads; the hint is what the client reads.
        for spec in _specs(True, True):
            assert spec.mutating is (not spec.annotations["readOnlyHint"]), spec.name


class TestAnnotationsForHelper:
    @pytest.mark.parametrize("level", sorted(LEVEL_ANNOTATIONS))
    def test_each_level_yields_its_pinned_annotation(self, level):
        assert reg.annotations_for(level) == LEVEL_ANNOTATIONS[level]

    def test_an_unknown_level_is_refused_naming_the_accepted_levels(self):
        with pytest.raises(ValueError) as exc:
            reg.annotations_for("write")
        assert "unknown tool level 'write'" in str(exc.value)
        assert "execute, plan, read" in str(exc.value)

    def test_each_call_returns_a_fresh_dict(self):
        first = reg.annotations_for("read")
        first["readOnlyHint"] = False
        first["extra"] = True
        assert reg.annotations_for("read") == {"readOnlyHint": True}

    def test_open_world_is_added_only_when_asked(self):
        assert reg.annotations_for("read", open_world=True) == {
            "readOnlyHint": True,
            "openWorldHint": True,
        }
        assert reg.annotations_for("read", open_world=False) == {
            "readOnlyHint": True,
            "openWorldHint": False,
        }
        assert "openWorldHint" not in reg.annotations_for("read")


class TestRegistryRefusesAnUnannotatedTool:
    """A new tool cannot be listed without a level."""

    @staticmethod
    def _registry_with(monkeypatch, *, source: str, index: int, change) -> None:
        original = getattr(reg, source)

        def patched(*args, **kwargs):
            specs = list(original(*args, **kwargs))
            specs[index] = change(specs[index])
            return specs

        monkeypatch.setattr(reg, source, patched)

    def test_a_read_tool_without_annotations_stops_the_build(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_readonly_specs",
            index=0,
            change=lambda spec: replace(spec, annotations=None),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=False)
        message = str(exc.value)
        assert "tool 'advise' has no MCP annotations" in message
        assert "annotations_for" in message

    def test_a_plan_tool_without_annotations_stops_the_build(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_mutating_specs",
            index=0,
            change=lambda spec: replace(spec, annotations=None),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=True)
        assert "tool 'train_start' has no MCP annotations" in str(exc.value)

    def test_an_execute_tool_without_annotations_stops_the_build(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_mutating_specs",
            index=3,
            change=lambda spec: replace(spec, annotations=None),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=True)
        assert "tool 'export_execute' has no MCP annotations" in str(exc.value)

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param({}, id="empty"),
            pytest.param({"destructiveHint": False}, id="no-read-only-hint"),
            pytest.param({"readOnlyHint": "yes"}, id="read-only-hint-not-a-bool"),
            pytest.param({"readOnlyHint": 1}, id="read-only-hint-int"),
        ],
    )
    def test_annotations_that_do_not_state_the_level_stop_the_build(self, monkeypatch, bad):
        self._registry_with(
            monkeypatch,
            source="_readonly_specs",
            index=2,
            change=lambda spec: replace(spec, annotations=bad),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=False)
        assert "tool 'data_validate' has no MCP annotations" in str(exc.value)

    def test_a_gated_tool_cannot_advertise_itself_read_only(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_mutating_specs",
            index=1,
            change=lambda spec: replace(spec, annotations=reg.annotations_for("read")),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=True)
        assert (
            "tool 'export' is gated as mutating but advertises readOnlyHint true"
            in str(exc.value)
        )

    def test_an_ungated_tool_cannot_advertise_itself_as_not_read_only(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_readonly_specs",
            index=5,
            change=lambda spec: replace(spec, annotations=reg.annotations_for("plan")),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=False)
        assert (
            "tool 'recipes_search' is not gated as mutating but advertises readOnlyHint false"
            in str(exc.value)
        )

    def test_call_tool_with_its_own_registry_is_not_affected(self):
        # `call_tool(registry=[...])` takes the caller's table as given: the guard is on
        # building the server's table, so an ad-hoc spec in a test or an embedding keeps
        # working without annotations.
        probe = reg.ToolSpec(
            name="probe",
            title="probe",
            description="probe",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args: {"ok": True},
        )
        assert probe.annotations is None
        assert reg.call_tool("probe", {}, registry=[probe]) == {"ok": True}


class TestGatingIsUnchanged:
    """The annotations describe the tools; they do not move a gate."""

    def test_plan_tools_refuse_with_the_same_message_when_gates_are_off(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("plan"):
            with pytest.raises(reg.McpToolError) as exc:
                by_name[name].handler({})
            assert str(exc.value) == (
                f"'{name}' can change state and is disabled; restart with "
                "'soup mcp serve --allow-mutating' to enable (still plan-only in v1)."
            )

    @pytest.mark.parametrize(("allow_mutating", "allow_execute"), GATES[:2])
    def test_execute_tools_refuse_with_the_same_message_without_allow_execute(
        self, allow_mutating, allow_execute
    ):
        by_name = {spec.name: spec for spec in _specs(allow_mutating, allow_execute)}
        for name in _names_at("execute"):
            with pytest.raises(reg.McpToolError) as exc:
                by_name[name].handler({"confirmation_token": "token123"})
            assert str(exc.value) == (
                f"'{name}' can execute commands and is disabled; restart with "
                "'soup mcp serve --allow-execute' to enable."
            )

    def test_call_tool_still_refuses_a_mutating_tool_without_allow_mutating(self):
        specs = _specs(allow_mutating=True)
        for name in _names_at("plan") + _names_at("execute"):
            with pytest.raises(reg.McpToolError) as exc:
                reg.call_tool(name, {}, registry=specs, allow_mutating=False)
            assert str(exc.value) == (
                f"'{name}' can change state and is disabled; pass "
                "allow_mutating=True to enable it."
            )

    def test_the_registry_is_still_eighteen_tools_in_every_gate_setting(self):
        for allow_mutating, allow_execute in ((False, False), (True, False), (True, True)):
            assert len(_specs(allow_mutating, allow_execute)) == 18


class TestAnnotationsReachTheClient:
    @pytest.fixture(autouse=True)
    def _need_mcp(self):
        pytest.importorskip("mcp", reason="needs the [mcp] extra (the SDK's Tool type)")

    @staticmethod
    def _wire(tool) -> dict:
        """The annotations as they travel: camelCase keys, unset hints omitted."""
        dumped = tool.model_dump(by_alias=True, exclude_none=True)
        return dumped.get("annotations") or {}

    @pytest.mark.parametrize(("allow_mutating", "allow_execute"), GATES)
    def test_tool_descriptors_carry_every_tools_annotations(
        self, allow_mutating, allow_execute
    ):
        from soup_cli.mcp_server.server import _tool_descriptors

        tools = _tool_descriptors(_specs(allow_mutating, allow_execute))
        assert {tool.name for tool in tools} == set(EXPECTED_LEVEL)
        for tool in tools:
            assert self._wire(tool) == _expected(tool.name), tool.name

    def test_tools_list_over_a_real_session_carries_the_annotations(self):
        from soup_cli.mcp_server.server import build_server
        from tests.mcp_roundtrip import connected_session

        server = build_server(_specs(allow_mutating=True))

        async def _go():
            async with connected_session(server) as client:
                return await client.list_tools()

        result = asyncio.run(_go())
        assert {tool.name for tool in result.tools} == set(EXPECTED_LEVEL)
        for tool in result.tools:
            assert self._wire(tool) == _expected(tool.name), tool.name
