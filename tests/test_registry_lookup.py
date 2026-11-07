"""Registry lookups: list by run_id, and a name:tag that matches more than one entry (#1763).

``push`` always creates a new entry, so two pushes with the same name and tag leave twins.
``resolve("name:tag")`` used to return the newest twin silently; it now raises
``AmbiguousRefError`` naming the candidates, as an ambiguous id prefix already did. The CLI
paths that resolve a ref must report that as an error, not a traceback or a silent pick.
"""

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.registry.store import AmbiguousRefError, RegistryStore
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture()
def push(tmp_path, monkeypatch):
    """Isolated registry and branches dir under cwd; returns a push helper."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    (tmp_path / "branches").mkdir()
    monkeypatch.setenv("SOUP_BRANCHES_DIR", str(tmp_path / "branches"))

    def _push(name="model", tag="v1", run_id=None):
        with RegistryStore() as store:
            return store.push(name=name, tag=tag, base_model="org/base", task="sft",
                              run_id=run_id, config={"lr": 0.0001})

    return _push


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _ids(entries) -> list:
    return [entry["id"] for entry in entries]


class TestListByRunId:
    def test_filters_by_run_id(self, push):
        first = push(run_id="run-a")
        other = push(name="other", run_id="run-b")
        second = push(tag="v2", run_id="run-a")
        with RegistryStore() as store:
            assert set(_ids(store.list(run_id="run-a"))) == {first, second}
            assert _ids(store.list(run_id="run-b")) == [other]
            assert store.list(run_id="run-missing") == []

    def test_combines_with_the_other_filters(self, push):
        first = push(run_id="run-a")
        push(tag="v2", run_id="run-a")
        push(run_id="run-b")
        with RegistryStore() as store:
            assert _ids(store.list(run_id="run-a", tag="v1")) == [first]
            assert store.list(run_id="run-a", name="other") == []

    def test_without_run_id_nothing_is_filtered(self, push):
        ids = {push(run_id="run-a"), push(name="other")}
        with RegistryStore() as store:
            assert set(_ids(store.list())) == ids


class TestResolveNameTag:
    def test_a_single_match_resolves(self, push):
        entry = push()
        push(tag="v2")
        with RegistryStore() as store:
            assert store.resolve("model:v1") == entry
            assert store.resolve("registry://model:v1") == entry

    def test_no_match_is_none(self, push):
        push()
        with RegistryStore() as store:
            assert store.resolve("model:v9") is None
            assert store.resolve("nobody:v1") is None

    @pytest.mark.parametrize("ref", ["model:v1", "registry://model:v1"])
    def test_twins_raise_naming_both_ids(self, push, ref):
        first, second = push(), push()
        with RegistryStore() as store, pytest.raises(AmbiguousRefError) as excinfo:
            store.resolve(ref)
        message = str(excinfo.value)
        assert "'model:v1' matches 2 entries" in message
        assert first in message and second in message
        assert "use the full id" in message

    def test_twins_still_resolve_by_full_id(self, push):
        first, second = push(), push()
        with RegistryStore() as store:
            assert store.resolve(first) == first
            assert store.resolve(second) == second

    def test_only_the_shared_tag_is_ambiguous(self, push):
        push()
        push()
        other = push(tag="v2")
        with RegistryStore() as store:
            assert store.resolve("model:v2") == other

    def test_a_long_candidate_list_is_capped(self, push):
        ids = [push() for _ in range(7)]
        with RegistryStore() as store, pytest.raises(AmbiguousRefError) as excinfo:
            store.resolve("model:v1")
        message = str(excinfo.value)
        assert "matches 7 entries" in message
        assert sum(entry_id in message for entry_id in ids) == 5
        assert "and 2 more" in message

    def test_an_ambiguous_prefix_is_unchanged(self, push):
        push()
        push(name="other")
        with RegistryStore() as store, pytest.raises(AmbiguousRefError) as excinfo:
            store.resolve("reg_")
        assert "supply a longer prefix or the full id" in str(excinfo.value)


class TestCallers:
    """Each path that resolves a user-supplied ref reports the ambiguity and changes nothing."""

    def test_registry_show(self, push):
        first, second = push(), push()
        result = runner.invoke(app, ["registry", "show", "model:v1"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        out = _plain(result)
        assert "matches 2 entries" in out and first in out and second in out

    def test_can_pack(self, push, tmp_path):
        push()
        push()
        result = runner.invoke(app, ["can", "pack", "--entry-id", "model:v1", "--out", "model.can"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "matches 2 entries" in _plain(result)
        assert not (tmp_path / "model.can").exists()

    def test_eval_gate_baseline(self, push, tmp_path):
        push()
        push()
        (tmp_path / "gate.yaml").write_text(yaml.safe_dump({
            "suite": "gate",
            "tasks": [{"type": "custom", "name": "acc", "tasks": "evals/acc.jsonl",
                       "scorer": "exact", "threshold": 0.5}],
        }), encoding="utf-8")
        result = runner.invoke(app, ["eval", "gate", "--suite", "gate.yaml",
                                     "--baseline", "registry://model:v1"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "matches 2 entries" in _plain(result)

    def test_resolve_baseline_used_by_ship_and_the_training_gate(self, push):
        from soup_cli.eval.gate import resolve_baseline

        push()
        push()
        with pytest.raises(AmbiguousRefError):
            resolve_baseline("registry://model:v1")

    @pytest.mark.parametrize("ref", ["registry://model:v1", "registry://reg_"])
    def test_eval_quant_check(self, push, tmp_path, ref):
        push()
        push()
        (tmp_path / "tasks.jsonl").write_text('{"prompt": "1+1", "answer": "2"}\n',
                                              encoding="utf-8")
        result = runner.invoke(app, ["eval", "quant-check", "--before", ref,
                                     "--after", "after.gguf", "--tasks", "tasks.jsonl"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        out = _plain(result)
        assert "--before" in out and "matches 2 entries" in out

    def test_attach_artifact(self, push, tmp_path):
        from soup_cli.registry.attach import attach_artifact

        first, second = push(), push()
        (tmp_path / "report.json").write_text("{}", encoding="utf-8")
        with pytest.raises(AmbiguousRefError):
            attach_artifact("model:v1", path="report.json", kind="diagnose_report")
        with RegistryStore() as store:
            assert store.get_artifacts(first) == [] and store.get_artifacts(second) == []

    def test_adapters_branch_from_registry(self, push, tmp_path):
        push()
        push()
        result = runner.invoke(app, ["adapters", "branch", "fromtag",
                                     "--from-registry", "model:v1"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "matches 2 entries" in _plain(result)
        assert not (tmp_path / "branches" / "fromtag.json").exists()
        assert not (tmp_path / "fromtag.from-registry.yaml").exists()

    def test_adapters_branch_attach_to_registry(self, push, tmp_path):
        first, second = push(), push()
        (tmp_path / "soup.yaml").write_text("base: org/base\ntask: sft\n", encoding="utf-8")
        result = runner.invoke(app, ["adapters", "branch", "attached", "-c", "soup.yaml",
                                     "--base", "org/base", "--attach-to-registry", "model:v1"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "matches 2 entries" in _plain(result)
        with RegistryStore() as store:
            assert store.get_artifacts(first) == [] and store.get_artifacts(second) == []

    def test_card(self, push):
        from soup_cli.commands.card import CardError, build_card_for_ref

        push()
        push()
        with pytest.raises(CardError, match="matches 2 entries"):
            build_card_for_ref("model:v1")

    def test_mcp_registry_show(self, push):
        from soup_cli.mcp_server import registry as mcp_registry

        push()
        push()
        with pytest.raises(mcp_registry.McpToolError, match="ambiguous"):
            mcp_registry.tool_registry_show({"ref": "model:v1"})
