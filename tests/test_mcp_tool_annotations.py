"""Every tool ``soup mcp serve`` lists carries a gate LEVEL and honest MCP annotations.

Two separate things, kept separate on purpose:

* **Level** (``read`` | ``plan`` | ``execute``) is the gate class: ``read`` always runs,
  ``plan`` is gated by ``--allow-mutating``, ``execute`` by ``--allow-execute``. It is a
  field on ``ToolSpec`` and decides ``ToolSpec.mutating``. The 15 / 2 / 2 split is pinned.
* **Annotations** are MCP hints about side effects, with MCP's own meaning: ``readOnlyHint``
  true says the tool does not modify its environment. So a read-LEVEL tool whose code path
  can write (open a database, migrate a schema, reconcile a stale row, fill a cache) is
  advertised ``readOnlyHint: false, destructiveHint: false``.

The pinned table below is a literal: adding a tool, moving one between levels or changing
a hint is a visible diff here, and it does not import the production helper, so a change to
the helper cannot silently re-pin the expectation. The reason next to each tool is the code
path that settles its ``readOnlyHint``.
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest

import soup_cli.mcp_server.registry as reg
from soup_cli.registry.store import RegistryStore

_RO = {"readOnlyHint": True}
# No idempotentHint on any of them: a repeated call is not promised to change nothing
# (a registry read re-applies file permissions), so no repeat-call guarantee is declared.
_WRITES = {"readOnlyHint": False, "destructiveHint": False}
_PLAN = {"readOnlyHint": False, "destructiveHint": False}
_EXECUTE = {"readOnlyHint": False, "destructiveHint": True}

#: name -> (level, hints, why ``readOnlyHint`` is what it is). Every tool the server lists.
TABLE = {
    "advise": (
        "read",
        _RO,
        "reads one dataset file and scores it in memory; the MCP path passes no model, "
        "so the live probes never run",
    ),
    "data_inspect": ("read", _RO, "reads one dataset file and computes stats in memory"),
    "data_validate": ("read", _RO, "reads one dataset file and checks its format in memory"),
    "data_score": (
        "read",
        {**_WRITES, "openWorldHint": True},
        "with the optional presidio-analyzer installed, its default engine downloads a "
        "missing spaCy model",
    ),
    "data_doctor": (
        "read",
        {**_WRITES, "openWorldHint": True},
        "AutoTokenizer.from_pretrained fills the Hugging Face cache for a client-chosen model",
    ),
    "recipes_search": ("read", _RO, "filters the in-memory recipe catalog"),
    "recipes_show": ("read", _RO, "looks one recipe up in the in-memory catalog"),
    "runs_list": (
        "read",
        _WRITES,
        "ExperimentTracker() creates and migrates the database and reading reconciles rows",
    ),
    "runs_show": (
        "read",
        _WRITES,
        "ExperimentTracker() creates and migrates the database and reading reconciles rows",
    ),
    "registry_list": (
        "read",
        _WRITES,
        "RegistryStore() creates and migrates the database, switches it to WAL, sets perms",
    ),
    "registry_show": (
        "read",
        _WRITES,
        "RegistryStore() creates and migrates the database, switches it to WAL, sets perms",
    ),
    "profile": (
        "read",
        {**_RO, "openWorldHint": False},
        "allow_network=False reads the local model directory and cache, never downloads",
    ),
    "diagnose_evidence": ("read", _RO, "reads one evidence JSON and builds a report in memory"),
    "ship_evidence": ("read", _RO, "reads one evidence JSON and computes a verdict in memory"),
    "engine_describe": ("read", _RO, "returns read-only startup metadata in memory"),
    "train_start": (
        "plan",
        _PLAN,
        "not read-only: under --allow-execute it makes a run directory and issues a token",
    ),
    "export": (
        "plan",
        _PLAN,
        "not read-only: under --allow-execute it issues a one-time token",
    ),
    "train_execute": ("execute", _EXECUTE, "spawns a training process"),
    "export_execute": ("execute", _EXECUTE, "spawns an export process"),
}

#: The read-LEVEL tools whose code path can write. Pinned: a tool leaves this list only when
#: its write path is gone, and then its table row flips to ``readOnlyHint: true`` with it.
WRITE_CAPABLE_READ_TOOLS = frozenset(
    {
        "data_score",
        "data_doctor",
        "runs_list",
        "runs_show",
        "registry_list",
        "registry_show",
    }
)

#: The three gate settings ``soup mcp serve`` can run with.
GATES = [
    pytest.param(False, False, id="gates-off"),
    pytest.param(True, False, id="allow-mutating"),
    pytest.param(True, True, id="allow-execute"),
]


def _specs(allow_mutating: bool = False, allow_execute: bool = False) -> list[reg.ToolSpec]:
    return reg.build_registry(allow_mutating=allow_mutating, allow_execute=allow_execute)


def _names_at(level: str) -> list[str]:
    return [name for name, row in TABLE.items() if row[0] == level]


def _hints(name: str) -> dict:
    return TABLE[name][1]


class TestPinnedTable:
    def test_the_table_names_exactly_the_tools_the_registry_builds(self):
        built = {spec.name for spec in _specs()}
        assert built == set(TABLE), (
            "a tool was added, removed or renamed: give it a row in TABLE",
            sorted(built ^ set(TABLE)),
        )

    def test_the_levels_split_fifteen_two_two(self):
        assert len(_names_at("read")) == 15
        assert len(_names_at("plan")) == 2
        assert len(_names_at("execute")) == 2
        assert len(TABLE) == 19

    def test_every_tool_has_a_one_line_reason(self):
        for name, (_, _, reason) in TABLE.items():
            assert reason and "\n" not in reason, name

    @pytest.mark.parametrize(("allow_mutating", "allow_execute"), GATES)
    def test_every_spec_has_the_level_and_hints_the_table_pins(
        self, allow_mutating, allow_execute
    ):
        for spec in _specs(allow_mutating, allow_execute):
            assert spec.level == TABLE[spec.name][0], spec.name
            assert spec.annotations == _hints(spec.name), spec.name

    @pytest.mark.parametrize(("allow_mutating", "allow_execute"), GATES)
    def test_every_spec_has_annotations_with_a_boolean_read_only_hint(
        self, allow_mutating, allow_execute
    ):
        for spec in _specs(allow_mutating, allow_execute):
            assert isinstance(spec.annotations, dict) and spec.annotations, spec.name
            assert isinstance(spec.annotations.get("readOnlyHint"), bool), spec.name


class TestLevelIsTheGateClass:
    def test_mutating_is_true_exactly_for_plan_and_execute(self):
        for spec in _specs(True, True):
            assert spec.mutating is (TABLE[spec.name][0] != "read"), spec.name

    def test_a_write_capable_read_tool_is_still_ungated(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in WRITE_CAPABLE_READ_TOOLS:
            assert by_name[name].level == "read", name
            assert by_name[name].mutating is False, name

    def test_a_write_capable_read_tool_still_runs_with_the_gates_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
        out = reg.call_tool("runs_list", {}, allow_mutating=False)
        assert out == {"runs": [], "count": 0}


class TestReadOnlyHintIsTruthful:
    def test_read_only_hint_is_true_for_exactly_the_tools_with_no_write_path(self):
        by_name = {spec.name: spec for spec in _specs()}
        read_only = {name for name, spec in by_name.items() if spec.annotations["readOnlyHint"]}
        assert read_only == set(_names_at("read")) - WRITE_CAPABLE_READ_TOOLS
        assert read_only == {
            "advise",
            "data_inspect",
            "data_validate",
            "recipes_search",
            "recipes_show",
            "profile",
            "diagnose_evidence",
            "ship_evidence",
            "engine_describe",
        }

    def test_a_tool_with_a_write_path_is_never_advertised_read_only(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in WRITE_CAPABLE_READ_TOOLS:
            annotations = by_name[name].annotations
            assert annotations["readOnlyHint"] is False, name
            assert annotations["destructiveHint"] is False, name

    def test_the_pinned_write_capable_list_agrees_with_the_table(self):
        in_table = {
            name
            for name, (level, hints, _) in TABLE.items()
            if level == "read" and hints["readOnlyHint"] is False
        }
        assert in_table == WRITE_CAPABLE_READ_TOOLS

    def test_no_tool_declares_an_idempotent_hint(self):
        # A repeated registry read re-applies file permissions (RegistryStore._tighten_perms
        # runs os.chmod on every construction), a run read reconciles stale rows, and the
        # doctor / scorer can fill a cache: none of them can promise that a repeat call
        # changes nothing, so the hint is OMITTED (not set to false) for all of them.
        by_name = {spec.name: spec for spec in _specs()}
        carrying = {name for name, spec in by_name.items() if "idempotentHint" in spec.annotations}
        assert carrying == set(), carrying
        for name in WRITE_CAPABLE_READ_TOOLS:
            assert by_name[name].annotations["readOnlyHint"] is False, name
            assert by_name[name].annotations["destructiveHint"] is False, name

    def test_registry_reads_declare_no_repeat_call_guarantee_because_they_reapply_perms(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in ("registry_list", "registry_show"):
            assert "idempotentHint" not in by_name[name].annotations, name
        # The reason, pinned to the code: every construction runs _tighten_perms, which chmods.
        assert "self._tighten_perms()" in inspect.getsource(RegistryStore.__init__)
        assert "os.chmod(" in inspect.getsource(RegistryStore._tighten_perms)

    def test_read_only_tools_carry_no_destructive_or_idempotent_hint(self):
        # MCP defines both as meaningful only when readOnlyHint is false.
        for spec in _specs():
            if spec.annotations["readOnlyHint"]:
                assert "destructiveHint" not in spec.annotations, spec.name
                assert "idempotentHint" not in spec.annotations, spec.name

    def test_plan_and_execute_tools_are_never_read_only(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("plan") + _names_at("execute"):
            assert by_name[name].annotations["readOnlyHint"] is False, name

    def test_execute_tools_keep_what_they_had(self):
        by_name = {spec.name: spec for spec in _specs()}
        for name in _names_at("execute"):
            assert by_name[name].annotations == {
                "readOnlyHint": False,
                "destructiveHint": True,
            }, name

    def test_open_world_is_set_only_where_the_code_settles_it(self):
        by_name = {spec.name: spec for spec in _specs()}
        expected = {"data_score": True, "data_doctor": True, "profile": False}
        for name, spec in by_name.items():
            if name in expected:
                assert spec.annotations["openWorldHint"] is expected[name], name
            else:
                assert "openWorldHint" not in spec.annotations, name


class TestTheWritePathsAreReal:
    """Witnesses for the four tracker / registry tools: they do write.

    If one of these tests starts failing because the tool became write-free, that is the
    follow-up landing: flip the tool's row in TABLE to ``readOnlyHint: true`` and take it out
    of WRITE_CAPABLE_READ_TOOLS in the same change.
    """

    def test_runs_list_creates_the_experiments_database(self, tmp_path, monkeypatch):
        db_path = tmp_path / "experiments.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
        assert not db_path.exists()
        reg.call_tool("runs_list", {})
        assert db_path.exists()

    def test_runs_show_rewrites_a_stale_running_row(self, tmp_path, monkeypatch):
        import soup_cli.experiment.tracker as tracker_module

        db_path = tmp_path / "experiments.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
        tracker = tracker_module.ExperimentTracker(db_path=db_path)
        try:
            tracker.launch_run(
                run_id="run_stale_1",
                kind="train",
                config_dict={},
                command_digest="digest",
                log_path="run.log",
            )
            tracker.mark_running("run_stale_1", pid=4242)
        finally:
            tracker.close()
        monkeypatch.setattr(tracker_module, "_process_is_alive", lambda pid: False)

        def stored_status() -> str:
            with closing(sqlite3.connect(str(db_path))) as conn:
                return conn.execute(
                    "SELECT status FROM runs WHERE run_id = ?", ("run_stale_1",)
                ).fetchone()[0]

        assert stored_status() == "running"
        reg.call_tool("runs_show", {"run_id": "run_stale_1"})
        assert stored_status() == "terminated"

    def test_registry_list_creates_the_registry_database(self, tmp_path, monkeypatch):
        db_path = tmp_path / "registry.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))
        assert not db_path.exists()
        reg.call_tool("registry_list", {})
        assert db_path.exists()

    def test_registry_show_creates_the_registry_database_even_on_a_miss(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "registry.db"
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(db_path))
        assert not db_path.exists()
        with pytest.raises(reg.McpToolError) as exc:
            reg.call_tool("registry_show", {"ref": "reg_missing"})
        assert str(exc.value) == "registry entry not found"
        assert db_path.exists()


class TestAnnotationsForHelper:
    @pytest.mark.parametrize(
        ("level", "kwargs", "expected"),
        [
            pytest.param("read", {}, {"readOnlyHint": True}, id="read"),
            pytest.param(
                "read",
                {"writes": True},
                {"readOnlyHint": False, "destructiveHint": False},
                id="read-writes",
            ),
            pytest.param(
                "plan", {}, {"readOnlyHint": False, "destructiveHint": False}, id="plan"
            ),
            pytest.param(
                "execute", {}, {"readOnlyHint": False, "destructiveHint": True}, id="execute"
            ),
        ],
    )
    def test_each_level_yields_its_pinned_annotation(self, level, kwargs, expected):
        assert reg.annotations_for(level, **kwargs) == expected

    def test_an_unknown_level_is_refused_naming_the_accepted_levels(self):
        with pytest.raises(ValueError) as exc:
            reg.annotations_for("write")
        assert "unknown tool level 'write'" in str(exc.value)
        assert "execute, plan, read" in str(exc.value)

    @pytest.mark.parametrize("level", ["plan", "execute"])
    def test_writes_is_refused_for_levels_that_are_never_read_only(self, level):
        with pytest.raises(ValueError) as exc:
            reg.annotations_for(level, writes=True)
        assert (
            f"'writes' applies to read-level tools only; a {level!r} tool is never read-only"
            in str(exc.value)
        )

    def test_idempotent_is_refused_on_a_read_only_tool(self):
        with pytest.raises(ValueError) as exc:
            reg.annotations_for("read", idempotent=True)
        assert "idempotentHint is meaningful only when readOnlyHint is false" in str(exc.value)

    def test_idempotent_and_open_world_are_added_only_when_asked(self):
        assert reg.annotations_for("read", writes=True, idempotent=True) == {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
        }
        assert reg.annotations_for("read", open_world=False) == {
            "readOnlyHint": True,
            "openWorldHint": False,
        }
        assert "openWorldHint" not in reg.annotations_for("read")
        assert "idempotentHint" not in reg.annotations_for("read", writes=True)

    def test_each_call_returns_a_fresh_dict(self):
        first = reg.annotations_for("read")
        first["readOnlyHint"] = False
        first["extra"] = True
        assert reg.annotations_for("read") == {"readOnlyHint": True}

    def test_level_fields_decides_level_mutating_and_annotations_together(self):
        assert reg.level_fields("read") == {
            "level": "read",
            "mutating": False,
            "annotations": {"readOnlyHint": True},
        }
        assert reg.level_fields("read", writes=True)["mutating"] is False
        assert reg.level_fields("plan")["mutating"] is True
        assert reg.level_fields("execute") == {
            "level": "execute",
            "mutating": True,
            "annotations": {"readOnlyHint": False, "destructiveHint": True},
        }


class TestRegistryRefusesAToolThatDoesNotStateItself:
    """A new tool cannot be listed without a level and honest hints."""

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
        assert "level_fields" in message

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
    def test_annotations_without_a_boolean_read_only_hint_stop_the_build(
        self, monkeypatch, bad
    ):
        self._registry_with(
            monkeypatch,
            source="_readonly_specs",
            index=2,
            change=lambda spec: replace(spec, annotations=bad),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=False)
        assert "tool 'data_validate' has no MCP annotations" in str(exc.value)

    @pytest.mark.parametrize("bad_level", [None, "write", ""])
    def test_a_tool_without_a_known_level_stops_the_build(self, monkeypatch, bad_level):
        self._registry_with(
            monkeypatch,
            source="_readonly_specs",
            index=0,
            change=lambda spec: replace(spec, level=bad_level),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=False)
        assert "tool 'advise' has no level; give it one of execute, plan, read" in str(exc.value)

    def test_a_read_level_tool_cannot_be_gated(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_readonly_specs",
            index=5,
            change=lambda spec: replace(spec, mutating=True),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=False)
        assert "tool 'recipes_search' is at level 'read' but mutating=True" in str(exc.value)

    def test_a_plan_level_tool_cannot_be_ungated(self, monkeypatch):
        self._registry_with(
            monkeypatch,
            source="_mutating_specs",
            index=1,
            change=lambda spec: replace(spec, mutating=False),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=True)
        assert "tool 'export' is at level 'plan' but mutating=False" in str(exc.value)

    @pytest.mark.parametrize(
        ("index", "name", "level"),
        [
            pytest.param(0, "train_start", "plan", id="plan"),
            pytest.param(2, "train_execute", "execute", id="execute"),
        ],
    )
    def test_only_a_read_level_tool_may_advertise_read_only(
        self, monkeypatch, index, name, level
    ):
        self._registry_with(
            monkeypatch,
            source="_mutating_specs",
            index=index,
            change=lambda spec: replace(spec, annotations={"readOnlyHint": True}),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=True)
        assert (
            f"tool '{name}' is at level '{level}' but advertises readOnlyHint true; "
            "only 'read' tools may"
        ) in str(exc.value)

    @pytest.mark.parametrize(
        ("source", "index", "name"),
        [
            pytest.param("_readonly_specs", 7, "runs_list", id="read"),
            pytest.param("_mutating_specs", 0, "train_start", id="plan"),
            pytest.param("_mutating_specs", 2, "train_execute", id="execute"),
        ],
    )
    def test_a_tool_that_is_not_read_only_must_state_destructive_hint(
        self, monkeypatch, source, index, name
    ):
        self._registry_with(
            monkeypatch,
            source=source,
            index=index,
            change=lambda spec: replace(spec, annotations={"readOnlyHint": False}),
        )
        with pytest.raises(ValueError) as exc:
            reg.build_registry(allow_mutating=True)
        assert f"tool '{name}' advertises readOnlyHint false without destructiveHint" in str(
            exc.value
        )

    def test_call_tool_with_its_own_registry_is_not_affected(self):
        # `call_tool(registry=[...])` takes the caller's table as given: the guard is on
        # building the server's table, so an ad-hoc spec in a test or an embedding keeps
        # working without a level or annotations.
        probe = reg.ToolSpec(
            name="probe",
            title="probe",
            description="probe",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args: {"ok": True},
        )
        assert probe.annotations is None and probe.level is None
        assert reg.call_tool("probe", {}, registry=[probe]) == {"ok": True}


class TestGatingIsUnchanged:
    """The level and the annotations describe the tools; they do not move a gate."""

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

    def test_the_registry_is_still_nineteen_tools_in_every_gate_setting(self):
        for allow_mutating, allow_execute in ((False, False), (True, False), (True, True)):
            assert len(_specs(allow_mutating, allow_execute)) == 19


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
        assert {tool.name for tool in tools} == set(TABLE)
        for tool in tools:
            assert self._wire(tool) == _hints(tool.name), tool.name

    def test_tools_list_over_a_real_session_carries_the_annotations(self):
        from soup_cli.mcp_server.server import build_server
        from tests.mcp_roundtrip import connected_session

        server = build_server(_specs(allow_mutating=True))

        async def _go():
            async with connected_session(server) as client:
                return await client.list_tools()

        result = asyncio.run(_go())
        assert {tool.name for tool in result.tools} == set(TABLE)
        for tool in result.tools:
            assert self._wire(tool) == _hints(tool.name), tool.name
