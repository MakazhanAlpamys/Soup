"""Regression coverage for #374's layer-streaming disk lifecycle."""

from __future__ import annotations

import json
import os

import pytest

GB = 1_000_000_000


def test_hub_cache_snapshot_omits_local_materialization(monkeypatch) -> None:
    import huggingface_hub

    from soup_cli.utils.hubs import snapshot_download

    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/cached/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = snapshot_download(
        "org/model",
        cache_dir=None,
        allow_patterns=["*.safetensors"],
        namespace_check=False,
    )

    assert result == "/cached/snapshot"
    assert "local_dir" not in captured
    assert "cache_dir" not in captured


def test_regular_hf_cache_weights_are_used_in_place(tmp_path, monkeypatch) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"regular-cache-entry")
    calls = []

    def fake_snapshot_download(_model, *, cache_dir, **_kwargs):
        calls.append(cache_dir)
        if cache_dir is not None:
            pytest.fail("a regular HF cache entry must not be materialized again")
        return str(snapshot)

    monkeypatch.setattr(hubs, "snapshot_download", fake_snapshot_download)
    plans = []
    resolved = resolve_model_weights("org/model", before_materialize=plans.append)

    assert resolved == str(snapshot)
    assert calls == [None]
    assert plans[0].materialized_copy_bytes == 0
    assert plans[0].materialize_bytes == 0


def test_refusal_callback_runs_before_a_symlinked_snapshot_is_copied(
    tmp_path,
    monkeypatch,
) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    blob = tmp_path / "blob"
    blob.write_bytes(b"cached-weight")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blob)
    calls = []

    def fake_snapshot_download(_model, *, cache_dir, **_kwargs):
        calls.append(cache_dir)
        return str(snapshot)

    monkeypatch.setattr(hubs, "snapshot_download", fake_snapshot_download)

    class RefusedError(Exception):
        pass

    def refuse(plan) -> None:
        assert plan.needs_materialization
        assert plan.materialize_bytes == len(b"cached-weight")
        raise RefusedError

    with pytest.raises(RefusedError):
        resolve_model_weights("org/model", before_materialize=refuse)

    assert calls == [None], "the materializing download happened before pre-flight"


def test_hf_blob_metadata_reuses_an_existing_materialized_copy(
    tmp_path,
    monkeypatch,
) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import plan_model_weights

    blob_id = "a" * 64
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / blob_id
    blob.write_bytes(b"immutable-blob")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blob)

    cache_root = tmp_path / "spectrum"
    materialized = cache_root / "weights" / "org__model"
    materialized.mkdir(parents=True)
    (materialized / "model.safetensors").write_bytes(b"immutable-blob")
    metadata = materialized / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "model.safetensors.metadata").write_text(
        f"commit\n{blob_id}\n123.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        hubs,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )

    plan = plan_model_weights("org/model")
    assert plan.weights_dir == str(materialized)
    assert plan.materialize_bytes == 0
    assert plan.source_files[0][2] == (materialized / "model.safetensors").stat().st_mtime_ns


def test_spectrum_override_remains_contained_and_is_used(tmp_path, monkeypatch) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import plan_model_weights

    blob = tmp_path / "blob"
    blob.write_bytes(b"weight")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blob)
    cache_root = tmp_path / "spectrum-cache"
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        hubs,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )

    plan = plan_model_weights("org/model")
    assert os.path.commonpath([plan.weights_dir, str(cache_root)]) == str(cache_root)


class RecordingConsole:
    def __init__(self) -> None:
        self.items = []

    def print(self, item) -> None:
        self.items.append(item)


def test_disk_preflight_refuses_combined_writes_on_one_volume(
    tmp_path,
    monkeypatch,
) -> None:
    import soup_cli.trainer.stream_setup as setup

    recorder = RecordingConsole()
    monkeypatch.setattr(setup, "console", recorder)
    monkeypatch.setattr(setup, "_disk_volume", lambda _path: (7, 100 * GB))

    with pytest.raises(ValueError, match="needs 110.00 GB.*only 100.00 GB is free"):
        setup._render_stream_disk_preflight(
            source_bytes=80 * GB,
            materialized_copy_bytes=80 * GB,
            materialize_bytes=60 * GB,
            materialized_path=str(tmp_path / "weights"),
            shard_bytes=70 * GB,
            shard_write_bytes=50 * GB,
            shard_path=str(tmp_path / "shards"),
        )

    assert len(recorder.items) == 1
    panel_text = str(recorder.items[0].renderable)
    assert "Projected total on disk: 230.00 GB" in panel_text
    assert "Additional writes before training: 110.00 GB" in panel_text


def test_disk_preflight_checks_distinct_volumes_independently(
    tmp_path,
    monkeypatch,
) -> None:
    import soup_cli.trainer.stream_setup as setup

    recorder = RecordingConsole()
    monkeypatch.setattr(setup, "console", recorder)

    def volumes(path):
        return (1, 70 * GB) if path.endswith("weights") else (2, 60 * GB)

    monkeypatch.setattr(setup, "_disk_volume", volumes)
    setup._render_stream_disk_preflight(
        source_bytes=80 * GB,
        materialized_copy_bytes=80 * GB,
        materialize_bytes=60 * GB,
        materialized_path=str(tmp_path / "weights"),
        shard_bytes=70 * GB,
        shard_write_bytes=50 * GB,
        shard_path=str(tmp_path / "shards"),
    )
    assert len(recorder.items) == 1


def test_fingerprint_mismatch_names_the_changed_component(tmp_path) -> None:
    from soup_cli.utils.layer_shard import (
        fingerprint_source_files,
        inspect_shard_cache,
    )

    old_files = (("model.safetensors", 100, 10),)
    new_files = (("model.safetensors", 100, 11),)
    payload = {
        "n_layers": 1,
        "layer_keys": ["self_attn.q_proj.weight"],
        "extra_keys": [],
        "dtype": "float32",
        "total_params": 4,
        "arch": "llama",
        "soup_version": "test",
        "source_fingerprint": fingerprint_source_files(old_files),
        "source_files": old_files,
    }
    (tmp_path / "index.json").write_text(json.dumps(payload), encoding="utf-8")

    cached, reason = inspect_shard_cache(
        str(tmp_path),
        "float32",
        fingerprint_source_files(new_files),
        new_files,
        "none",
        True,
        "",
    )
    assert cached is None
    assert "mtime_ns changed" in reason
    assert "model.safetensors" in reason


def test_real_reshard_notice_and_index_record_source_components(tmp_path) -> None:
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    source = weights / "model.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "model.embed_tokens.weight": torch.ones(4, 2),
        },
        str(source),
    )
    first_messages = []
    shard_checkpoint(
        str(weights),
        str(shards),
        dtype="float32",
        notify=first_messages.append,
    )
    first_index = read_shard_index(str(shards))
    assert first_index.source_files[0][0] == "model.safetensors"

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    messages = []
    shard_checkpoint(
        str(weights),
        str(shards),
        dtype="float32",
        notify=messages.append,
    )
    assert len(messages) == 1
    assert "mtime_ns changed" in messages[0]
    assert "model.safetensors" in messages[0]
