"""Tests for Local Model Registry / Provenance Vault (Part A of v0.26.0)."""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_hash_config_is_deterministic(self, tmp_path):
        from soup_cli.registry.hashing import hash_config

        config = {"base": "llama", "task": "sft", "training": {"lr": 2e-5}}
        assert hash_config(config) == hash_config(config)

    def test_hash_config_order_independent(self):
        from soup_cli.registry.hashing import hash_config

        config_a = {"base": "llama", "task": "sft"}
        config_b = {"task": "sft", "base": "llama"}
        assert hash_config(config_a) == hash_config(config_b)

    def test_hash_config_changes_on_mutation(self):
        from soup_cli.registry.hashing import hash_config

        config_a = {"base": "llama", "task": "sft"}
        config_b = {"base": "llama", "task": "dpo"}
        assert hash_config(config_a) != hash_config(config_b)

    def test_hash_file_matches_content(self, tmp_path):
        from soup_cli.registry.hashing import hash_file

        f = tmp_path / "data.jsonl"
        f.write_text("hello world\n", encoding="utf-8")
        digest = hash_file(str(f))
        assert len(digest) == 64  # sha256 hex
        assert digest == hash_file(str(f))

    def test_hash_file_different_content_different_hash(self, tmp_path):
        from soup_cli.registry.hashing import hash_file

        f_a = tmp_path / "a.txt"
        f_a.write_text("hello", encoding="utf-8")
        f_b = tmp_path / "b.txt"
        f_b.write_text("world", encoding="utf-8")
        assert hash_file(str(f_a)) != hash_file(str(f_b))

    def test_hash_file_missing_raises(self, tmp_path):
        from soup_cli.registry.hashing import hash_file

        with pytest.raises(FileNotFoundError):
            hash_file(str(tmp_path / "nope.bin"))

    def test_hash_entry_combines_config_and_data(self, tmp_path):
        from soup_cli.registry.hashing import hash_entry

        data_file = tmp_path / "data.jsonl"
        data_file.write_text("x", encoding="utf-8")
        config = {"base": "m1", "task": "sft"}
        digest = hash_entry(config=config, data_path=str(data_file), base_model="m1")
        assert len(digest) == 64
        assert digest == hash_entry(
            config=config, data_path=str(data_file), base_model="m1"
        )

    def test_hash_entry_changes_with_base(self, tmp_path):
        from soup_cli.registry.hashing import hash_entry

        data_file = tmp_path / "data.jsonl"
        data_file.write_text("x", encoding="utf-8")
        config = {"base": "m1", "task": "sft"}
        a = hash_entry(config=config, data_path=str(data_file), base_model="m1")
        b = hash_entry(config=config, data_path=str(data_file), base_model="m2")
        assert a != b

    def test_hash_entry_without_data_path(self):
        from soup_cli.registry.hashing import hash_entry

        config = {"base": "m1", "task": "sft"}
        digest = hash_entry(config=config, data_path=None, base_model="m1")
        assert len(digest) == 64


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_name_accepted(self):
        from soup_cli.registry.store import validate_name

        validate_name("my-model_v1.0")  # no raise

    def test_name_rejects_path_separator(self):
        from soup_cli.registry.store import validate_name

        with pytest.raises(ValueError, match="invalid"):
            validate_name("../evil")

    def test_name_rejects_null_byte(self):
        from soup_cli.registry.store import validate_name

        with pytest.raises(ValueError, match="invalid"):
            validate_name("bad\x00name")

    def test_name_rejects_empty(self):
        from soup_cli.registry.store import validate_name

        with pytest.raises(ValueError, match="empty"):
            validate_name("")

    def test_name_rejects_too_long(self):
        from soup_cli.registry.store import validate_name

        with pytest.raises(ValueError, match="int"):
            validate_name("a" * 300)

    def test_valid_tag_accepted(self):
        from soup_cli.registry.store import validate_tag

        validate_tag("v3")
        validate_tag("prod-2024")

    def test_tag_rejects_slash(self):
        from soup_cli.registry.store import validate_tag

        with pytest.raises(ValueError):
            validate_tag("v/3")

    def test_tag_rejects_empty(self):
        from soup_cli.registry.store import validate_tag

        with pytest.raises(ValueError, match="empty"):
            validate_tag("")

    def test_tag_rejects_too_long(self):
        from soup_cli.registry.store import validate_tag

        with pytest.raises(ValueError, match="int"):
            validate_tag("a" * 100)

    @pytest.mark.parametrize(
        "bad",
        ["../evil", "/abs/path", "bad\x00name", "", "a" * 300, "@invalid",
         "-leading-dash", ".leading-dot"],
    )
    def test_validate_name_rejects_invalid(self, bad):
        from soup_cli.registry.store import validate_name

        with pytest.raises(ValueError):
            validate_name(bad)


# ---------------------------------------------------------------------------
# Store CRUD
# ---------------------------------------------------------------------------


class TestRegistryStore:
    def _store(self, tmp_path) -> "RegistryStore":  # noqa: F821
        from soup_cli.registry.store import RegistryStore

        return RegistryStore(db_path=tmp_path / "reg.db")

    def test_init_creates_tables(self, tmp_path):
        store = self._store(tmp_path)
        conn = sqlite3.connect(tmp_path / "reg.db")
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        store.close()
        assert {"registry_entries", "registry_artifacts",
                "registry_lineage", "registry_tags"}.issubset(names)

    def test_push_and_get(self, tmp_path):
        store = self._store(tmp_path)
        entry_id = store.push(
            name="helpful-chat",
            tag="v1",
            base_model="llama-3.1-8b",
            task="sft",
            run_id=None,
            config={"base": "llama-3.1-8b", "task": "sft"},
            data_path=None,
            notes="initial",
        )
        assert entry_id

        entry = store.get(entry_id)
        assert entry is not None
        assert entry["name"] == "helpful-chat"
        assert entry["base_model"] == "llama-3.1-8b"
        assert entry["task"] == "sft"
        assert "v1" in entry["tags"]
        store.close()

    def test_push_duplicate_tag_under_name_replaces(self, tmp_path):
        store = self._store(tmp_path)
        e1 = store.push(name="model-a", tag="v1", base_model="b", task="sft",
                        run_id=None, config={"x": 1})
        e2 = store.push(name="model-a", tag="v1", base_model="b", task="sft",
                        run_id=None, config={"x": 2})
        # both entries exist but v1 points to the most recent by default
        listing = store.list(name="model-a")
        ids = [e["id"] for e in listing]
        assert e1 in ids
        assert e2 in ids
        # Latest tagged v1 should be e2
        latest = store.resolve("model-a:v1")
        assert latest == e2
        store.close()

    def test_list_all(self, tmp_path):
        store = self._store(tmp_path)
        for i in range(3):
            store.push(name=f"m{i}", tag="v1", base_model="b", task="sft",
                       run_id=None, config={"i": i})
        entries = store.list()
        assert len(entries) >= 3
        store.close()

    def test_list_filter_by_tag(self, tmp_path):
        store = self._store(tmp_path)
        store.push(name="m1", tag="v1", base_model="b", task="sft",
                   run_id=None, config={})
        store.push(name="m2", tag="v2", base_model="b", task="sft",
                   run_id=None, config={})
        entries = store.list(tag="v1")
        assert all("v1" in e["tags"] for e in entries)

    def test_list_filter_by_base(self, tmp_path):
        store = self._store(tmp_path)
        store.push(name="m1", tag="v1", base_model="llama-3.1-8b", task="sft",
                   run_id=None, config={})
        store.push(name="m2", tag="v1", base_model="qwen-3-8b", task="sft",
                   run_id=None, config={})
        entries = store.list(base="llama-3.1-8b")
        assert len(entries) == 1
        assert entries[0]["base_model"] == "llama-3.1-8b"

    def test_search(self, tmp_path):
        store = self._store(tmp_path)
        store.push(name="medical-chat", tag="v1", base_model="llama",
                   task="sft", run_id=None, config={}, notes="medical QA")
        store.push(name="legal-chat", tag="v1", base_model="llama",
                   task="sft", run_id=None, config={}, notes="legal QA")
        hits = store.search("medical")
        assert len(hits) == 1
        assert hits[0]["name"] == "medical-chat"

    def test_search_is_case_insensitive(self, tmp_path):
        store = self._store(tmp_path)
        store.push(name="Medical-Chat", tag="v1", base_model="llama",
                   task="sft", run_id=None, config={})
        hits = store.search("medical")
        assert len(hits) == 1

    def test_delete(self, tmp_path):
        store = self._store(tmp_path)
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        assert store.delete(eid) is True
        assert store.get(eid) is None
        assert store.delete(eid) is False

    def test_resolve_by_name_tag(self, tmp_path):
        store = self._store(tmp_path)
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        assert store.resolve("m1:v1") == eid

    def test_resolve_by_prefix(self, tmp_path):
        store = self._store(tmp_path)
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        assert store.resolve(eid[:8]) == eid

    def test_resolve_registry_uri(self, tmp_path):
        store = self._store(tmp_path)
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        assert store.resolve(f"registry://{eid}") == eid

    def test_resolve_unknown_returns_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store.resolve("nonexistent-id-xyz") is None

    def test_resolve_ambiguous_prefix_raises(self, tmp_path):
        """Two entries sharing a prefix must raise, not silently return None."""
        # Monkeypatch _generate_entry_id to force a prefix collision
        import soup_cli.registry.store as store_mod
        from soup_cli.registry.store import AmbiguousRefError, RegistryStore
        original = store_mod._generate_entry_id

        counter = {"n": 0}

        def gen_colliding() -> str:
            counter["n"] += 1
            return f"reg_samepre_abc{counter['n']:06d}"

        store_mod._generate_entry_id = gen_colliding
        try:
            store = RegistryStore(db_path=tmp_path / "reg.db")
            store.push(name="m1", tag="v1", base_model="b",
                       task="sft", run_id=None, config={})
            store.push(name="m2", tag="v1", base_model="b",
                       task="sft", run_id=None, config={})
            with pytest.raises(AmbiguousRefError):
                store.resolve("reg_samepre_abc")
            store.close()
        finally:
            store_mod._generate_entry_id = original

    def test_push_rejects_empty_base_model(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError, match="base_model"):
            store.push(name="m1", tag="v1", base_model="",
                       task="sft", run_id=None, config={})
        store.close()

    def test_push_rejects_empty_task(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError, match="task"):
            store.push(name="m1", tag="v1", base_model="b",
                       task="", run_id=None, config={})
        store.close()

    def test_push_with_data_path_records_data_hash(self, tmp_path):
        store = self._store(tmp_path)
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("hello\n", encoding="utf-8")
        eid = store.push(
            name="m1", tag="v1", base_model="b", task="sft",
            run_id=None, config={"x": 1}, data_path=str(data_file),
        )
        entry = store.get(eid)
        assert entry is not None
        assert entry["data_hash"] is not None
        assert len(entry["data_hash"]) == 64
        store.close()

    def test_list_filter_by_task(self, tmp_path):
        store = self._store(tmp_path)
        store.push(name="m1", tag="v1", base_model="b", task="sft",
                   run_id=None, config={})
        store.push(name="m2", tag="v1", base_model="b", task="dpo",
                   run_id=None, config={})
        entries = store.list(task="sft")
        assert len(entries) == 1
        assert entries[0]["task"] == "sft"
        store.close()

    def test_list_respects_limit(self, tmp_path):
        store = self._store(tmp_path)
        for i in range(5):
            store.push(name=f"m{i}", tag="v1", base_model="b", task="sft",
                       run_id=None, config={})
        entries = store.list(limit=2)
        assert len(entries) == 2
        store.close()

    def test_search_empty_query_returns_empty(self, tmp_path):
        store = self._store(tmp_path)
        store.push(name="m1", tag="v1", base_model="b", task="sft",
                   run_id=None, config={})
        assert store.search("") == []
        store.close()

    def test_add_tag_nonexistent_raises(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            store.add_tag("nonexistent_entry", "v2")
        store.close()

    def test_delete_cascade_cleans_child_rows(self, tmp_path):
        """FK ON DELETE CASCADE must remove artifacts, tags, lineage."""
        import sqlite3

        from soup_cli.registry.store import RegistryStore

        db_path = tmp_path / "reg.db"
        store = RegistryStore(db_path=db_path)
        parent = store.push(name="parent", tag="v1", base_model="b",
                            task="sft", run_id=None, config={})
        child = store.push(name="child", tag="v1", base_model="b",
                           task="sft", run_id=None, config={})
        artifact_file = tmp_path / "art.bin"
        artifact_file.write_bytes(b"x")
        store.add_artifact(entry_id=parent, kind="adapter",
                           path=str(artifact_file), enforce_cwd=False)
        store.add_lineage(child_id=child, parent_id=parent,
                          relation="forked_from")
        store.close()

        # Use a fresh connection to check after delete
        store = RegistryStore(db_path=db_path)
        assert store.delete(parent) is True
        store.close()

        conn = sqlite3.connect(db_path)
        arts = conn.execute(
            "SELECT COUNT(*) FROM registry_artifacts WHERE entry_id = ?",
            (parent,),
        ).fetchone()[0]
        tags = conn.execute(
            "SELECT COUNT(*) FROM registry_tags WHERE entry_id = ?",
            (parent,),
        ).fetchone()[0]
        lineage = conn.execute(
            "SELECT COUNT(*) FROM registry_lineage "
            "WHERE child_id = ? OR parent_id = ?",
            (parent, parent),
        ).fetchone()[0]
        conn.close()
        assert arts == 0
        assert tags == 0
        assert lineage == 0

    def test_get_eval_results_with_no_run_id_returns_empty(self, tmp_path):
        store = self._store(tmp_path)
        eid = store.push(name="m1", tag="v1", base_model="b",
                         task="sft", run_id=None, config={})
        assert store.get_eval_results(eid) == []
        store.close()

    def test_get_eval_results_missing_entry_returns_empty(self, tmp_path):
        store = self._store(tmp_path)
        assert store.get_eval_results("nonexistent") == []
        store.close()

    def test_context_manager_closes_connection(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        with RegistryStore(db_path=tmp_path / "reg.db") as store:
            store.push(name="m1", tag="v1", base_model="b",
                       task="sft", run_id=None, config={})
        # After __exit__, _conn must be reset to None
        assert store._conn is None


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestRegistryArtifacts:
    def test_add_and_list_artifact(self, tmp_path, monkeypatch):
        from soup_cli.registry.store import RegistryStore

        monkeypatch.chdir(tmp_path)
        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        artifact_file = tmp_path / "adapter.safetensors"
        artifact_file.write_bytes(b"fake weights")
        store.add_artifact(
            entry_id=eid, kind="adapter", path=str(artifact_file),
        )
        arts = store.get_artifacts(eid)
        assert len(arts) == 1
        assert arts[0]["kind"] == "adapter"
        assert arts[0]["sha256"]
        assert arts[0]["size_bytes"] > 0
        store.close()

    def test_add_artifact_rejects_unknown_kind(self, tmp_path, monkeypatch):
        from soup_cli.registry.store import RegistryStore

        monkeypatch.chdir(tmp_path)
        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        f = tmp_path / "x.bin"
        f.write_bytes(b"x")
        with pytest.raises(ValueError, match="kind"):
            store.add_artifact(entry_id=eid, kind="malicious", path=str(f))
        store.close()

    def test_add_artifact_accepts_judge_calibration_kind(self, tmp_path, monkeypatch):
        # v0.71.1 #214 — judge_calibration is a valid artifact kind.
        from soup_cli.registry.store import RegistryStore

        monkeypatch.chdir(tmp_path)
        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        f = tmp_path / "calib.json"
        f.write_text('{"calibrated": true}', encoding="utf-8")
        store.add_artifact(entry_id=eid, kind="judge_calibration", path=str(f))
        arts = store.get_artifacts(eid)
        assert any(a["kind"] == "judge_calibration" for a in arts)
        store.close()

    def test_add_artifact_default_enforces_cwd(self, tmp_path, monkeypatch):
        """By default, artifacts outside cwd are rejected."""
        from soup_cli.registry.store import RegistryStore

        # cd somewhere else so tmp_path is NOT under cwd
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        store = RegistryStore(db_path=workdir / "reg.db")
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"x")
        with pytest.raises(ValueError, match="outside|cwd"):
            store.add_artifact(entry_id=eid, kind="adapter",
                               path=str(outside))
        store.close()

    def test_add_artifact_missing_file_raises(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})
        with pytest.raises(FileNotFoundError):
            store.add_artifact(
                entry_id=eid, kind="adapter",
                path=str(tmp_path / "missing.bin"),
            )
        store.close()


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class TestLineage:
    def test_add_and_walk_lineage(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        parent = store.push(name="base", tag="v1", base_model="b", task="sft",
                            run_id=None, config={})
        child = store.push(name="fork", tag="v1", base_model="b", task="sft",
                           run_id=None, config={})
        store.add_lineage(child_id=child, parent_id=parent, relation="forked_from")

        ancestors = store.get_ancestors(child)
        assert parent in [a["id"] for a in ancestors]

        descendants = store.get_descendants(parent)
        assert child in [d["id"] for d in descendants]
        store.close()

    def test_lineage_rejects_unknown_relation(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        a = store.push(name="a", tag="v1", base_model="b", task="sft",
                       run_id=None, config={})
        b = store.push(name="b", tag="v1", base_model="b", task="sft",
                       run_id=None, config={})
        with pytest.raises(ValueError, match="relation"):
            store.add_lineage(child_id=a, parent_id=b, relation="hacked")
        store.close()

    def test_lineage_prevents_self_reference(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        a = store.push(name="a", tag="v1", base_model="b", task="sft",
                       run_id=None, config={})
        with pytest.raises(ValueError, match="self"):
            store.add_lineage(child_id=a, parent_id=a, relation="forked_from")
        store.close()

    def test_lineage_prevents_indirect_cycle(self, tmp_path):
        """A→B, then B→A would close a cycle — must be rejected."""
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        a = store.push(name="a", tag="v1", base_model="b", task="sft",
                       run_id=None, config={})
        b = store.push(name="b", tag="v1", base_model="b", task="sft",
                       run_id=None, config={})
        store.add_lineage(child_id=a, parent_id=b, relation="forked_from")
        with pytest.raises(ValueError, match="cycle"):
            store.add_lineage(child_id=b, parent_id=a,
                              relation="forked_from")
        store.close()

    def test_lineage_rejects_missing_child(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        parent = store.push(name="p", tag="v1", base_model="b", task="sft",
                            run_id=None, config={})
        with pytest.raises(ValueError, match="child"):
            store.add_lineage(child_id="nonexistent", parent_id=parent,
                              relation="forked_from")
        store.close()

    def test_lineage_rejects_missing_parent(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        child = store.push(name="c", tag="v1", base_model="b", task="sft",
                           run_id=None, config={})
        with pytest.raises(ValueError, match="parent"):
            store.add_lineage(child_id=child, parent_id="nonexistent",
                              relation="forked_from")
        store.close()


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_config_diff_detects_changes(self):
        from soup_cli.registry.diff import config_diff

        left = {"base": "llama", "training": {"lr": 2e-5, "epochs": 3}}
        right = {"base": "llama", "training": {"lr": 5e-5, "epochs": 3}}
        changes = config_diff(left, right)
        assert any("lr" in c.path for c in changes)
        # Only one change expected
        assert len(changes) == 1
        change = changes[0]
        assert change.kind == "changed"

    def test_config_diff_detects_added_removed(self):
        from soup_cli.registry.diff import config_diff

        left = {"a": 1}
        right = {"a": 1, "b": 2}
        changes = config_diff(left, right)
        kinds = [c.kind for c in changes]
        assert "added" in kinds

        changes = config_diff(right, left)
        kinds = [c.kind for c in changes]
        assert "removed" in kinds

    def test_eval_delta(self):
        from soup_cli.registry.diff import eval_delta

        left = [{"benchmark": "mmlu", "score": 0.6}]
        right = [{"benchmark": "mmlu", "score": 0.65}]
        delta = eval_delta(left, right)
        assert delta[0]["benchmark"] == "mmlu"
        assert abs(delta[0]["delta"] - 0.05) < 1e-6

    def test_eval_delta_benchmark_only_on_one_side(self):
        """If a benchmark appears on one side only, delta is None."""
        from soup_cli.registry.diff import eval_delta

        left = [{"benchmark": "mmlu", "score": 0.6}]
        right = [{"benchmark": "gsm8k", "score": 0.7}]
        deltas = eval_delta(left, right)
        by_bench = {d["benchmark"]: d for d in deltas}
        assert by_bench["mmlu"]["delta"] is None
        assert by_bench["mmlu"]["right"] is None
        assert by_bench["gsm8k"]["delta"] is None
        assert by_bench["gsm8k"]["left"] is None

    def test_config_diff_identical_configs(self):
        from soup_cli.registry.diff import config_diff

        cfg = {"base": "llama", "training": {"lr": 2e-5}}
        assert config_diff(cfg, cfg) == []

    def test_hash_config_with_nested_lists(self):
        from soup_cli.registry.hashing import hash_config

        # Config with lists (LoRA target_modules) — must hash deterministically
        cfg = {"lora": {"target_modules": ["q_proj", "v_proj"]}}
        h1 = hash_config(cfg)
        h2 = hash_config({"lora": {"target_modules": ["q_proj", "v_proj"]}})
        assert h1 == h2


# ---------------------------------------------------------------------------
# Auto-register hook (tracker extension)
# ---------------------------------------------------------------------------


class TestTrackerHook:
    def test_register_from_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.registry.store import RegistryStore

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "llama-3.1-8b", "task": "sft"},
            device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))
        store = RegistryStore(db_path=tmp_path / "reg.db")
        entry_id = store.register_from_run(tracker, run_id, tag="auto",
                                           name="auto-chat")
        entry = store.get(entry_id)
        assert entry is not None
        assert entry["run_id"] == run_id
        assert entry["base_model"] == "llama-3.1-8b"
        store.close()
        tracker.close()

    def test_register_from_missing_run_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.registry.store import RegistryStore

        tracker = ExperimentTracker()
        store = RegistryStore(db_path=tmp_path / "reg.db")
        with pytest.raises(ValueError, match="run"):
            store.register_from_run(tracker, "nonexistent_run", tag="v1",
                                    name="m1")
        store.close()
        tracker.close()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestRegistryCLI:
    def _setup_dbs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.chdir(tmp_path)

    def test_list_empty(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        result = runner.invoke(app, ["registry", "list"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "No registry entries" in result.output or "empty" in result.output.lower()

    def test_push_and_list(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        # Create a finished run first
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "llama", "task": "sft"},
            device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))
        tracker.close()

        result = runner.invoke(
            app, ["registry", "push", "--run-id", run_id,
                  "--name", "my-chat", "--tag", "v1"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "my-chat" in result.output or "Registered" in result.output

        result = runner.invoke(app, ["registry", "list"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "my-chat" in result.output

    def test_push_rejects_invalid_name(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "llama", "task": "sft"},
            device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))
        tracker.close()

        result = runner.invoke(
            app, ["registry", "push", "--run-id", run_id,
                  "--name", "../evil", "--tag", "v1"],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "invalid" in result.output.lower() or "error" in result.output.lower()

    def test_push_rejects_missing_run(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        result = runner.invoke(
            app, ["registry", "push", "--run-id", "no_such_run",
                  "--name", "m1", "--tag", "v1"],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_show(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="m1", tag="v1", base_model="llama",
                         task="sft", run_id=None,
                         config={"base": "llama", "task": "sft"},
                         notes="demo")
        store.close()

        result = runner.invoke(app, ["registry", "show", eid])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "m1" in result.output
        assert "llama" in result.output

    def test_show_missing(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        result = runner.invoke(app, ["registry", "show", "nope"])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_search_cli(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        store.push(name="medical-chat", tag="v1", base_model="llama",
                   task="sft", run_id=None, config={}, notes="medical")
        store.close()

        result = runner.invoke(app, ["registry", "search", "medical"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "medical-chat" in result.output

    def test_diff_cli(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        a = store.push(name="m1", tag="v1", base_model="llama",
                       task="sft", run_id=None,
                       config={"training": {"lr": 2e-5}})
        b = store.push(name="m1", tag="v2", base_model="llama",
                       task="sft", run_id=None,
                       config={"training": {"lr": 5e-5}})
        store.close()

        result = runner.invoke(app, ["registry", "diff", a, b])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "lr" in result.output

    def test_delete_cli(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="m1", tag="v1", base_model="llama",
                         task="sft", run_id=None, config={})
        store.close()

        result = runner.invoke(
            app, ["registry", "delete", eid, "--yes"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_delete_cli_without_yes_is_noop(self, tmp_path, monkeypatch):
        """Without --yes, command is an informational no-op (exit 0)."""
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="m1", tag="v1", base_model="llama",
                         task="sft", run_id=None, config={})
        store.close()

        result = runner.invoke(app, ["registry", "delete", eid])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--yes" in result.output or "confirm" in result.output.lower()
        # Entry should still exist
        store = RegistryStore()
        assert store.get(eid) is not None
        store.close()

    def test_promote_cli_happy_path(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="m1", tag="v1", base_model="llama",
                         task="sft", run_id=None, config={})
        store.close()

        result = runner.invoke(
            app, ["registry", "promote", eid, "--tag", "prod"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

        store = RegistryStore()
        entry = store.get(eid)
        assert "prod" in entry["tags"]
        store.close()

    def test_promote_cli_rejects_invalid_tag(self, tmp_path, monkeypatch):
        self._setup_dbs(tmp_path, monkeypatch)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="m1", tag="v1", base_model="llama",
                         task="sft", run_id=None, config={})
        store.close()

        result = runner.invoke(
            app, ["registry", "promote", eid, "--tag", "bad/slash"],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "invalid" in result.output.lower()

    def test_push_rejects_missing_run_shows_message(self, tmp_path, monkeypatch):
        """Error output must mention the run so operators can diagnose."""
        self._setup_dbs(tmp_path, monkeypatch)
        result = runner.invoke(
            app, ["registry", "push", "--run-id", "no_such_run",
                  "--name", "m1", "--tag", "v1"],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "not found" in result.output.lower() \
            or "no such" in result.output.lower()


# ---------------------------------------------------------------------------
# History command
# ---------------------------------------------------------------------------


class TestHistoryCLI:
    def test_history_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["history", "ghost"])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_history_shows_lineage(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.chdir(tmp_path)
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        parent = store.push(name="base", tag="v1", base_model="llama",
                            task="sft", run_id=None, config={})
        child = store.push(name="base", tag="v2", base_model="llama",
                           task="sft", run_id=None, config={})
        store.add_lineage(child_id=child, parent_id=parent,
                          relation="forked_from")
        store.close()

        result = runner.invoke(app, ["history", "base"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "v1" in result.output and "v2" in result.output

    def test_history_invalid_name_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["history", "../evil"])
        assert result.exit_code != 0, (result.output, repr(result.exception))
        assert "invalid" in result.output.lower()


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_artifact_path_traversal_rejected(self, tmp_path, monkeypatch):
        from soup_cli.registry.store import RegistryStore

        # Attempt to register an artifact outside CWD
        monkeypatch.chdir(tmp_path)
        store = RegistryStore(db_path=tmp_path / "reg.db")
        eid = store.push(name="m1", tag="v1", base_model="b", task="sft",
                         run_id=None, config={})

        outside = tmp_path.parent / "outside_artifact.bin"
        outside.write_bytes(b"x")
        try:
            with pytest.raises(ValueError, match="outside|cwd"):
                # Default enforce_cwd=True should reject
                store.add_artifact(entry_id=eid, kind="adapter",
                                   path=str(outside))
        finally:
            # Avoid polluting parent dir across CI reruns
            outside.unlink(missing_ok=True)
        store.close()

    def test_search_sql_injection_escaped(self, tmp_path):
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        store.push(name="clean", tag="v1", base_model="b", task="sft",
                   run_id=None, config={})
        # SQL injection attempt — should simply return 0 hits, not crash
        hits = store.search("'; DROP TABLE registry_entries; --")
        assert hits == []
        # Table still exists and row count unchanged
        entries = store.list()
        assert len(entries) == 1
        store.close()

    def test_search_like_wildcard_is_escaped(self, tmp_path):
        """A query of `%` must not match every row."""
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        store.push(name="alpha", tag="v1", base_model="b",
                   task="sft", run_id=None, config={})
        store.push(name="beta", tag="v1", base_model="b",
                   task="sft", run_id=None, config={})
        # `%` is a SQL LIKE wildcard; escaping must turn it into a literal
        hits = store.search("%")
        assert hits == []
        store.close()

    def test_resolve_like_wildcard_is_escaped(self, tmp_path):
        """Prefix resolve must not treat `%` as a wildcard."""
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore(db_path=tmp_path / "reg.db")
        store.push(name="m1", tag="v1", base_model="b",
                   task="sft", run_id=None, config={})
        assert store.resolve("%") is None
        store.close()

    def test_tag_with_null_byte_rejected(self):
        from soup_cli.registry.store import validate_tag

        with pytest.raises(ValueError):
            validate_tag("bad\x00tag")

    def test_store_resolves_real_path_for_containment(self, tmp_path):
        """Uses realpath + commonpath, not Path.resolve + relative_to."""
        from soup_cli.registry.store import _is_under

        inside = tmp_path / "sub" / "file.bin"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(b"x")
        assert _is_under(inside, tmp_path) is True
        # Windows short-name path containment (mimicked via tmp_path parent)
        outside = tmp_path.parent / "other.bin"
        assert _is_under(outside, tmp_path) is False


# ---------------------------------------------------------------------------
# Integration: auto-register after training finishes
# ---------------------------------------------------------------------------


class TestAutoRegister:
    def test_autoregister_candidate_not_committed(self, tmp_path, monkeypatch):
        """on_train_end should record a candidate, but user must push."""
        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "exp.db"))
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))

        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.registry.store import RegistryStore

        tracker = ExperimentTracker()
        run_id = tracker.start_run(
            config_dict={"base": "llama", "task": "sft"},
            device="cpu", device_name="cpu", gpu_info={},
        )
        tracker.finish_run(run_id, 2.0, 0.5, 100, 60.0, str(tmp_path / "out"))

        # Until push, registry is empty
        store = RegistryStore()
        assert store.list() == []

        eid = store.register_from_run(tracker, run_id, tag="auto",
                                      name="autoregistered")
        assert store.get(eid) is not None
        store.close()
        tracker.close()
