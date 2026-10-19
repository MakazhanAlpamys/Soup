"""Complete bounded source-byte hashes; unknown is never a partial digest."""

from __future__ import annotations

import hashlib
import os
import shutil
from types import SimpleNamespace

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.registry import RegistryStore, hash_entry
from soup_cli.utils import training_identity as identity
from soup_cli.utils.run_provenance import (
    BaseRevision,
    read_run_provenance,
    write_run_provenance,
)


def _config(source, **data):
    cfg = SoupConfig(base="local/model", data={"train": str(source)})
    return cfg.model_copy(update={"data": cfg.data.model_copy(update=data)})


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".csv", ".parquet", ".txt", ".JSONL"])
@pytest.mark.parametrize("payload", [b"", b"\xef\xbb\xbfraw\r\n\x00bytes\xff"])
def test_exact_complete_bytes(tmp_path, suffix, payload):
    source = tmp_path / ("data" + suffix)
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert identity.capture_training_data_hash(_config(source)) == expected


@pytest.mark.parametrize("updates", [
    {"train": ["first.jsonl", "second.jsonl"]}, {"format": "pre_tokenized"},
    {"tokenized_path": "cache"}, {"replay": "old.jsonl"}, {"interleave": "concat"},
    {"forget_set": "forget.jsonl"}, {"retain_set": "retain.jsonl"}, {"streaming": True},
    {"train": "https://example.invalid/data.jsonl"}, {"train": "owner/dataset"},
    {"train": "\\\\server\\share\\data.jsonl"}, {"train": "//server/share/data.jsonl"},
])
def test_unsupported_source_never_opens_a_member(tmp_path, monkeypatch, updates):
    source = tmp_path / "data.jsonl"
    source.write_text("present", encoding="utf-8")
    cfg = _config(source, **updates)
    monkeypatch.setattr(identity.os, "open", lambda *args: pytest.fail("unsupported input opened"))
    assert identity.capture_training_data_hash(cfg) is None


@pytest.mark.parametrize(
    "field,value", [("modality", "vision"), ("task", "unlearn"), ("task", "tts")],
)
def test_external_source_modes_are_unknown(tmp_path, field, value):
    source = tmp_path / "data.jsonl"
    source.write_text("present", encoding="utf-8")
    cfg = _config(source).model_copy(update={field: value})
    assert identity.capture_training_data_hash(cfg) is None


def test_missing_directory_and_unknown_suffix(tmp_path):
    assert identity.capture_training_data_hash(_config(tmp_path / "missing.jsonl")) is None
    directory = tmp_path / "directory.jsonl"
    directory.mkdir()
    assert identity.capture_training_data_hash(_config(directory)) is None
    unknown = tmp_path / "data.bin"
    unknown.write_bytes(b"not a raw loader input")
    assert identity.capture_training_data_hash(_config(unknown)) is None


def test_byte_budget_and_boundary(tmp_path, monkeypatch):
    source = tmp_path / "data.jsonl"
    source.write_bytes(b"12345")
    assert identity.capture_training_data_hash(_config(source), max_bytes=5) == hashlib.sha256(
        b"12345",
    ).hexdigest()
    monkeypatch.setattr(identity.os, "open", lambda *args: pytest.fail("oversized input opened"))
    assert identity.capture_training_data_hash(_config(source), max_bytes=4) is None


def test_time_limit_after_partial_read_is_unknown(tmp_path, monkeypatch):
    source = tmp_path / "data.jsonl"
    source.write_bytes(b"12345")
    monkeypatch.setattr(identity, "_CHUNK_BYTES", 2)
    ticks = iter([0.0, 0.0, 6.0])
    monkeypatch.setattr(identity.time, "monotonic", lambda: next(ticks, 6.0))
    assert identity.capture_training_data_hash(_config(source), timeout_seconds=5) is None


@pytest.mark.parametrize("operation", ["fstat", "stat"])
@pytest.mark.parametrize("elapsed", [0.0, 4.999, 5.0, 6.0])
def test_deadline_after_final_metadata_read(tmp_path, monkeypatch, operation, elapsed):
    source = tmp_path / "data.jsonl"
    payload = b"complete source bytes\n"
    source.write_bytes(payload)
    config = _config(source)
    original = getattr(identity.os, operation)
    clock = {"now": 0.0}
    calls = 0

    def metadata(*args, **kwargs):
        nonlocal calls
        info = original(*args, **kwargs)
        # Keep the real metadata unchanged; only account for time spent in the
        # final descriptor/path operation after the last pre-metadata check.
        if operation == "fstat" or str(args[0]) == str(source):
            calls += 1
            if calls == 2:
                clock["now"] = elapsed
        return info

    monkeypatch.setattr(identity.os, operation, metadata)
    monkeypatch.setattr(identity.time, "monotonic", lambda: clock["now"])
    digest = identity.capture_training_data_hash(config, timeout_seconds=5)
    assert calls == 2
    expected = hashlib.sha256(payload).hexdigest() if elapsed < 5 else None
    assert digest == expected


@pytest.mark.parametrize("mode", ["grow", "truncate", "rewrite", "read-error"])
def test_changes_or_read_failure_do_not_publish_a_digest(tmp_path, monkeypatch, mode):
    source = tmp_path / "data.jsonl"
    source.write_bytes(b"123456")
    monkeypatch.setattr(identity, "_CHUNK_BYTES", 2)
    fdopen = identity.os.fdopen
    reads = []
    handles = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            reads.append(size)
            chunk = self.handle.read(size)
            if len(reads) == 1:
                if mode == "read-error":
                    raise OSError("unreadable remainder")
                payload = {"grow": b"12345678", "truncate": b"1", "rewrite": b"abcdef"}[mode]
                source.write_bytes(payload)
                # Explicitly change metadata; this is an accidental-mutation check.
                os.utime(source, ns=(1_000_000_000, 2_000_000_000))
            return chunk

    def wrap(*args):
        handle = fdopen(*args)
        handles.append(handle)
        return Reader(handle)

    monkeypatch.setattr(identity.os, "fdopen", wrap)
    assert identity.capture_training_data_hash(_config(source), max_bytes=7) is None
    assert reads and max(reads) <= 2
    assert handles and all(handle.closed for handle in handles)


def test_path_replacement_metadata_is_checked(tmp_path, monkeypatch):
    source = tmp_path / "data.jsonl"
    source.write_bytes(b"12345")
    real_stat = identity.os.stat
    calls = 0

    def replaced(path, *args, **kwargs):
        nonlocal calls
        info = real_stat(path, *args, **kwargs)
        if str(path) == str(source):
            calls += 1
            if calls == 2:
                return SimpleNamespace(**{
                    field: getattr(info, field) + (1 if field == "st_ino" else 0)
                    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                })
        return info

    monkeypatch.setattr(identity.os, "stat", replaced)
    assert identity.capture_training_data_hash(_config(source)) is None
    assert calls == 2


def test_produced_digest_survives_copy_and_registry_override(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_bytes(b"original source\n")
    config = _config(source)
    digest = identity.capture_training_data_hash(config)
    assert digest == hashlib.sha256(b"original source\n").hexdigest()
    tracker = ExperimentTracker(tmp_path / "runs.db")
    try:
        run_id = tracker.start_run(config.model_dump(), "cpu", "CPU", {})
        tracker.record_training_data_hash(run_id, digest)
        metadata = BaseRevision(config.base).metadata(run_id)
        metadata["training_data_sha256"] = tracker.get_run(run_id)["training_data_sha256"]
        artifact = tmp_path / "artifact"
        write_run_provenance(artifact, metadata)
        copied = tmp_path / "copy"
        shutil.copytree(artifact, copied)
        source.write_bytes(b"later registered file\n")
        assert read_run_provenance(copied)["training_data_sha256"] == digest
        with RegistryStore(tmp_path / "registry.db") as store:
            historical = store.get(store.register_from_run(
                tracker, run_id, name="model", tag="old",
            ))
            assert historical["data_hash"] == historical["training_data_sha256"] == digest
            assert historical["entry_hash"] == hash_entry(
                config=config.model_dump(), data_path=None, base_model=config.base,
                data_sha256=digest,
            )
            explicit = store.get(store.register_from_run(
                tracker, run_id, name="model", tag="new", data_path=str(source),
            ))
            assert explicit["data_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert explicit["training_data_sha256"] == digest
    finally:
        tracker.close()


def test_record_unknown_and_reject_partial_or_malformed_hash(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.db")
    try:
        run_id = tracker.start_run({"base": "local/model", "task": "sft"}, "cpu", "CPU", {})
        tracker.record_training_data_hash(run_id, None)
        assert tracker.get_run(run_id)["training_data_sha256"] is None
        for invalid in ("partial", "A" * 64, True):
            with pytest.raises(ValueError, match="SHA-256"):
                tracker.record_training_data_hash(run_id, invalid)
        assert tracker.get_run(run_id)["training_data_sha256"] is None
    finally:
        tracker.close()
