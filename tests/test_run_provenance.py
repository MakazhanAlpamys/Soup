"""Recorded base identities survive database upgrades and copied adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.registry import RegistryStore, hash_entry, hash_file

REVISION = "a" * 40
OTHER_REVISION = "b" * 40
TRAINING_HASH = "c" * 64
BASE = "owner/base"


@pytest.fixture(autouse=True)
def explicit_offline_state(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


def _run(tracker, *, base=BASE):
    run_id = tracker.start_run({"base": base, "task": "sft"}, "cpu", "CPU", {})
    tracker.finish_run(run_id, 1.0, 0.5, 1, 1.0, "/tmp/out")
    return run_id


def test_legacy_run_upgrade_is_nullable_and_idempotent(tmp_path):
    from soup_cli.experiment.tracker import _SCHEMA_SQL

    db = tmp_path / "legacy.db"
    # Strip only the new columns to reproduce a database from before this feature.
    legacy = "\n".join(
        line for line in _SCHEMA_SQL.splitlines()
        if not any(name in line for name in
                   ("base_revision", "training_data_sha256"))
    )
    with sqlite3.connect(db) as conn:
        conn.executescript(legacy)
        conn.execute(
            "INSERT INTO runs(run_id, created_at, config_json) VALUES (?, ?, ?)",
            ("legacy", "2020-01-01", json.dumps({"base": BASE})),
        )
    for _ in range(2):
        tracker = ExperimentTracker(db)
        row = tracker.get_run("legacy")
        assert row["base_revision"] is None
        assert row["base_revision_source"] is None
        assert row["training_data_sha256"] is None
        tracker.close()


def test_concurrent_database_initialization_preserves_recorded_identity(tmp_path):
    db = tmp_path / "concurrent.db"
    tracker = ExperimentTracker(db)
    run_id = _run(tracker)
    tracker.record_base_revision(run_id, revision=REVISION, source="hub_commit")
    tracker.close()

    def reopen(_):
        tracker = ExperimentTracker(db)
        try:
            return tracker.get_run(run_id)["base_revision"]
        finally:
            tracker.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(reopen, range(8))) == [REVISION] * 8


def test_concurrent_upgrade_of_legacy_database_leaves_old_identity_unknown(tmp_path):
    from soup_cli.experiment.tracker import _SCHEMA_SQL

    db = tmp_path / "legacy-concurrent.db"
    legacy = "\n".join(
        line for line in _SCHEMA_SQL.splitlines()
        if not any(name in line for name in ("base_revision", "training_data_sha256"))
    )
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(legacy)
        conn.execute(
            "INSERT INTO runs(run_id, created_at, config_json) VALUES ('old', '2020', '{}')",
        )

    def upgrade(_):
        tracker = ExperimentTracker(db)
        try:
            row = tracker.get_run("old")
            return row["base_revision"], row["base_revision_source"], row["training_data_sha256"]
        finally:
            tracker.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(upgrade, range(8))) == [(None, None, None)] * 8


def test_restarted_run_does_not_reuse_the_previous_identity(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _run(tracker)
    tracker.record_base_revision(run_id, revision=REVISION, source="hub_commit")
    tracker.start_run({"base": "other/base"}, "cpu", "CPU", {}, run_id=run_id)
    row = tracker.get_run(run_id)
    assert row["base_model"] == "other/base"
    assert row["base_revision"] is None
    assert row["base_revision_source"] is None
    assert row["training_data_sha256"] is None
    tracker.close()


def test_moving_hub_resolves_once_for_all_base_loads(monkeypatch):
    from soup_cli.utils import run_provenance as provenance

    probe = Mock(side_effect=[SimpleNamespace(sha=REVISION), SimpleNamespace(sha=OTHER_REVISION)])
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=probe))
    identity = provenance.BaseRevision(BASE)
    with provenance.base_revision_scope(identity):
        # Model, tokenizer, processor and config all use exactly the same execution identity.
        for _ in range(4):
            assert provenance.base_revision_kwargs(BASE) == {"revision": REVISION}
        assert provenance.base_revision_kwargs("different/teacher") == {}
        assert provenance.base_revision_kwargs("different/reward") == {}
    assert provenance.base_revision_kwargs(BASE) == {}
    assert probe.call_count == 1
    assert identity.revision == REVISION
    assert identity.source == "hub_commit"


@pytest.mark.parametrize("mode", ["offline", "failed", "invalid-sha", "unsupported"])
def test_unknown_identity_preserves_existing_loader_flow(monkeypatch, mode):
    from soup_cli.utils import run_provenance as provenance

    probe = Mock(return_value=SimpleNamespace(sha="main"))
    if mode == "failed":
        probe.side_effect = OSError("offline, with a sensitive diagnostic")
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=probe))
    if mode == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    identity = provenance.BaseRevision(BASE, binding_supported=mode != "unsupported")
    with provenance.base_revision_scope(identity):
        assert provenance.base_revision_kwargs(BASE) == {}
        assert provenance.base_revision_kwargs(BASE) == {}
    assert identity.revision is None
    assert identity.source != "hub_commit"
    assert probe.call_count == (0 if mode in {"offline", "unsupported"} else 1)


def test_scopes_restore_identity_after_nested_run_and_exception(monkeypatch):
    from soup_cli.utils import run_provenance as provenance

    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(
        model_info=lambda repo, **kwargs: SimpleNamespace(
            sha=REVISION if repo == BASE else OTHER_REVISION,
        ),
    ))
    with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
        with pytest.raises(RuntimeError), provenance.base_revision_scope(
            provenance.BaseRevision("other/base"),
        ):
            assert provenance.base_revision_kwargs("other/base") == {"revision": OTHER_REVISION}
            raise RuntimeError("aborted inner run")
        assert provenance.base_revision_kwargs(BASE) == {"revision": REVISION}
    assert provenance.base_revision_kwargs(BASE) == {}


def test_cache_path_identity_is_metadata_and_does_not_pin_local_bytes(tmp_path, monkeypatch):
    from soup_cli.utils import run_provenance as provenance

    snapshot = tmp_path / "models--owner--base" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(provenance, "get_hf_api", Mock(side_effect=AssertionError("no network")))
    identity = provenance.BaseRevision(str(snapshot))
    with provenance.base_revision_scope(identity):
        assert provenance.base_revision_kwargs(str(snapshot)) == {}
    assert identity.revision == REVISION
    assert identity.source == "cache_path"
    provenance.write_run_provenance(snapshot, identity.metadata("run-cache"))
    assert provenance.recorded_base_kwargs(str(snapshot), str(snapshot)) == {}
    assert provenance.BaseRevision(str(tmp_path)).revision is None


def test_copied_adapter_recovers_revision_without_original_database(tmp_path, monkeypatch):
    from soup_cli.utils import run_provenance as provenance

    source = tmp_path / "original"
    source.mkdir()
    identity = provenance.BaseRevision(BASE)
    identity.revision, identity.source = REVISION, "hub_commit"
    provenance.write_run_provenance(source, identity.metadata("run-original"))
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    shutil.rmtree(source)
    monkeypatch.setattr(provenance, "get_hf_api", Mock(side_effect=AssertionError("no lookup")))
    assert provenance.recorded_base_kwargs(str(copied), BASE) == {"revision": REVISION}
    assert provenance.read_run_provenance(copied)["run_id"] == "run-original"
    # Legacy adapters and explicit --base overrides preserve their supported behavior.
    assert provenance.recorded_base_kwargs(str(copied), "other/base") == {}
    assert provenance.recorded_base_kwargs(str(tmp_path / "legacy"), BASE) == {}


@pytest.mark.parametrize(
    "payload", ["not JSON", "[]", "{}", '{"schema_version":999}', "x" * 70000],
    ids=["syntax", "list", "missing", "future-version", "oversized"],
)
def test_unreadable_or_invalid_sidecar_stays_unknown(tmp_path, payload):
    from soup_cli.utils import run_provenance as provenance

    (tmp_path / provenance.PROVENANCE_FILENAME).write_text(payload, encoding="utf-8")
    assert provenance.recorded_base_kwargs(str(tmp_path), BASE) == {}


def test_oversized_sidecar_with_a_valid_json_prefix_is_not_accepted(tmp_path):
    from soup_cli.utils import run_provenance as provenance

    identity = provenance.BaseRevision(BASE)
    identity.revision, identity.source = REVISION, "hub_commit"
    payload = json.dumps(identity.metadata("run-1")) + " " * 70000
    (tmp_path / provenance.PROVENANCE_FILENAME).write_text(payload, encoding="utf-8")
    assert provenance.recorded_base_kwargs(str(tmp_path), BASE) == {}


def test_registration_copies_historical_identity_without_reopening_data(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.db")
    store = RegistryStore(tmp_path / "registry.db")
    run_id = _run(tracker)
    tracker.record_base_revision(run_id, revision=REVISION, source="hub_commit")
    tracker._get_conn().execute(
        "UPDATE runs SET training_data_sha256=? WHERE run_id=?", (TRAINING_HASH, run_id),
    )
    tracker._get_conn().commit()
    entry = store.get(store.register_from_run(tracker, run_id, name="model", tag="v1"))
    assert entry["base_revision"] == REVISION
    assert entry["base_revision_source"] == "hub_commit"
    assert entry["training_data_sha256"] == TRAINING_HASH
    assert entry["data_hash"] == TRAINING_HASH
    assert entry["entry_hash"] == hash_entry(
        config={"base": BASE, "task": "sft"}, data_path=None, base_model=BASE,
        data_sha256=TRAINING_HASH, base_revision=REVISION, base_revision_source="hub_commit",
    )
    store.close()
    tracker.close()


@pytest.mark.parametrize("source_state", ["changed", "deleted"])
def test_registration_keeps_training_digest_after_source_mutation(tmp_path, source_state):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"original training bytes"}\n', encoding="utf-8")
    historical = hash_file(str(data))
    config = {"base": BASE, "task": "sft", "data": {"train": str(data)}}
    tracker = ExperimentTracker(tmp_path / "runs.db")
    try:
        run_id = tracker.start_run(config, "cpu", "CPU", {})
        tracker.finish_run(run_id, 1.0, 0.5, 1, 1.0, str(tmp_path / "out"))
        tracker._get_conn().execute(
            "UPDATE runs SET training_data_sha256=? WHERE run_id=?", (historical, run_id),
        )
        tracker._get_conn().commit()
        if source_state == "changed":
            data.write_text("different bytes after training\n", encoding="utf-8")
            assert hash_file(str(data)) != historical
        else:
            data.unlink()
        with RegistryStore(tmp_path / "registry.db") as store:
            entry = store.get(store.register_from_run(tracker, run_id, name="model", tag="v1"))
        assert entry["training_data_sha256"] == historical
        assert entry["data_hash"] == historical
        assert entry["entry_hash"] == hash_entry(
            config=config, data_path=None, base_model=BASE, data_sha256=historical,
        )
    finally:
        tracker.close()


@pytest.mark.parametrize(
    "adapter_config, expected",
    [({"base_model_name_or_path": BASE}, BASE),
     ({"base_model_name_or_path": BASE, "revision": REVISION}, BASE),
     ({"base_model_name_or_path": "other/base"}, None),
     ({"base_model_name_or_path": BASE, "revision": OTHER_REVISION}, None),
     ({"base_model_name_or_path": BASE, "revision": "main"}, None),
     ({}, None), ([], None)],
)
def test_implicit_adapter_base_requires_matching_local_config(tmp_path, adapter_config, expected):
    from soup_cli.utils import run_provenance as provenance

    identity = provenance.BaseRevision(BASE)
    identity.revision, identity.source = REVISION, "hub_commit"
    provenance.write_run_provenance(tmp_path, identity.metadata("run-original"))
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    assert provenance.recorded_adapter_base(str(tmp_path)) == expected


def test_implicit_adapter_base_preserves_missing_malformed_and_oversized_config(tmp_path):
    from soup_cli.utils import run_provenance as provenance

    identity = provenance.BaseRevision(BASE)
    identity.revision, identity.source = REVISION, "hub_commit"
    provenance.write_run_provenance(tmp_path, identity.metadata("run-original"))
    assert provenance.recorded_adapter_base(str(tmp_path)) is None
    for payload in ("not JSON", json.dumps({"base_model_name_or_path": BASE}) + " " * 70000):
        (tmp_path / "adapter_config.json").write_text(payload, encoding="utf-8")
        assert provenance.recorded_adapter_base(str(tmp_path)) is None


def test_explicit_registered_file_takes_precedence_without_erasing_training_hash(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.db")
    store = RegistryStore(tmp_path / "registry.db")
    run_id = _run(tracker)
    tracker._get_conn().execute(
        "UPDATE runs SET training_data_sha256=? WHERE run_id=?", (TRAINING_HASH, run_id),
    )
    tracker._get_conn().commit()
    data = tmp_path / "registered.jsonl"
    data.write_text("changed after training\n", encoding="utf-8")
    entry = store.get(store.register_from_run(
        tracker, run_id, name="model", tag="v1", data_path=str(data),
    ))
    assert entry["data_hash"] == hash_file(str(data))
    assert entry["training_data_sha256"] == TRAINING_HASH
    assert entry["entry_hash"] == hash_entry(
        config={"base": BASE, "task": "sft"}, data_path=str(data), base_model=BASE,
    )
    store.close()
    tracker.close()


def test_legacy_registration_keeps_original_hash_and_config_schema(tmp_path):
    from soup_cli.config.schema import SoupConfig

    tracker = ExperimentTracker(tmp_path / "runs.db")
    store = RegistryStore(tmp_path / "registry.db")
    run_id = _run(tracker)
    entry = store.get(store.register_from_run(tracker, run_id, name="model", tag="v1"))
    assert entry["data_hash"] is None
    assert entry["base_revision"] is None
    assert entry["entry_hash"] == hash_entry(
        config={"base": BASE, "task": "sft"}, data_path=None, base_model=BASE,
    )
    assert "base_revision" not in SoupConfig.model_fields
    assert "training_data_sha256" not in SoupConfig.model_fields
    store.close()
    tracker.close()


def test_registry_retries_compare_revision_and_historical_training_identity(tmp_path):
    from soup_cli.registry.store import IdempotencyConflictError

    registered = tmp_path / "registered.jsonl"
    registered.write_text("registered input\n", encoding="utf-8")
    with RegistryStore(tmp_path / "registry.db") as store:
        request = dict(
            name="model", tag="v1", base_model=BASE, task="sft", run_id="run-1",
            config={}, base_revision=REVISION, base_revision_source="hub_commit",
            training_data_sha256=TRAINING_HASH, idempotency_key="job-1",
            data_path=str(registered),
        )
        entry_id = store.push(**request)
        assert store.push(**request) == entry_id
        with pytest.raises(IdempotencyConflictError):
            store.push(**{**request, "base_revision": OTHER_REVISION})
        with pytest.raises(IdempotencyConflictError):
            store.push(**{**request, "training_data_sha256": "d" * 64})
        with pytest.raises(IdempotencyConflictError):
            store.push(**{**request, "base_revision_source": "cache_path"})


def test_entry_hash_uses_the_same_registered_digest_that_is_stored(tmp_path, monkeypatch):
    from soup_cli.registry import store as store_module

    file_hash = Mock(side_effect=[TRAINING_HASH, "d" * 64])
    monkeypatch.setattr(store_module, "hash_file", file_hash)
    with RegistryStore(tmp_path / "registry.db") as store:
        entry = store.get(store.push(
            name="model", tag="v1", base_model=BASE, task="sft", run_id=None,
            config={}, data_path="file-that-may-change-between-reads",
        ))
    assert file_hash.call_count == 1
    assert entry["data_hash"] == TRAINING_HASH
    assert entry["entry_hash"] == hash_entry(
        config={}, data_path=None, base_model=BASE, data_sha256=TRAINING_HASH,
    )


def test_entry_fingerprint_binds_both_recorded_identities_and_the_source():
    envelope = {
        "config": '{"base":"owner/base"}', "base_model": BASE,
        "data_sha256": TRAINING_HASH, "base_revision": REVISION,
        "base_revision_source": "hub_commit",
    }
    expected = hashlib.sha256(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    assert hash_entry(
        config={"base": BASE}, data_path=None, base_model=BASE, data_sha256=TRAINING_HASH,
        base_revision=REVISION, base_revision_source="hub_commit",
    ) == expected


def test_recorded_revision_and_source_are_visible_in_run_and_registry_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from tests.conftest import strip_ansi

    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _run(tracker)
    tracker.record_base_revision(run_id, revision=REVISION, source="hub_commit")
    tracker.finish_run(run_id, 1.0, 0.5, 1, 1.0, str(tmp_path / "out"))
    with RegistryStore(tmp_path / "registry.db") as store:
        entry_id = store.register_from_run(tracker, run_id, name="model", tag="v1")
    tracker.close()
    for args in (["runs", "show", run_id], ["registry", "show", entry_id]):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, (result.output, result.exception)
        output = strip_ansi(result.output)
        assert REVISION in output
        assert "hub_commit" in output
