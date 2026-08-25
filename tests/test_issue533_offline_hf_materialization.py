"""Offline Hugging Face cache materialization regression tests (#533)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

COMMIT = "a" * 40
WEIGHT_BLOB = "b" * 64
CONFIG_BLOB = "c" * 64


def _cached_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    import torch
    from safetensors.torch import save_file

    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / COMMIT
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    weight = blobs / WEIGHT_BLOB
    config = blobs / CONFIG_BLOB
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.eye(2)},
        str(weight),
    )
    config.write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / WEIGHT_BLOB)
    (snapshot / "config.json").symlink_to(Path("../../blobs") / CONFIG_BLOB)
    return snapshot, weight, config


@pytest.mark.skipif(os.name == "nt", reason="standard HF cache uses POSIX symlinks")
def test_complete_cached_snapshot_materializes_without_a_second_hub_call(
    tmp_path,
    monkeypatch,
) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    snapshot, weight, _config = _cached_snapshot(tmp_path)
    cache_root = tmp_path / "soup-cache"
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))
    calls = []

    def cached_only(_model, *, cache_dir, **_kwargs):
        calls.append(cache_dir)
        if cache_dir is not None:
            pytest.fail("materialization must not make a second Hub metadata call")
        return str(snapshot)

    monkeypatch.setattr(hubs, "snapshot_download", cached_only)

    resolved = Path(resolve_model_weights("org/model"))

    assert calls == [None]
    assert resolved == cache_root / "weights" / "org__model"
    assert not (resolved / "model.safetensors").is_symlink()
    assert (resolved / "model.safetensors").read_bytes() == weight.read_bytes()
    assert (resolved / "config.json").read_text(encoding="utf-8") == (
        '{"model_type":"llama"}\n'
    )
    metadata = (
        resolved
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    ).read_text(encoding="utf-8").splitlines()
    assert metadata[:2] == [COMMIT, WEIGHT_BLOB]

    index = shard_checkpoint(
        str(resolved),
        str(tmp_path / "shards"),
        dtype="float32",
        arch="llama",
    )
    assert index.n_layers == 1


@pytest.mark.skipif(os.name == "nt", reason="standard HF cache uses POSIX symlinks")
def test_resolved_commit_is_carried_by_the_materialization_plan(
    tmp_path,
    monkeypatch,
) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    snapshot, _weight, _config = _cached_snapshot(tmp_path)
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    monkeypatch.setattr(
        hubs,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )

    plan = plan_model_weights("org/model")

    assert plan.source_revision == COMMIT
    assert plan.source_dir == str(snapshot)

    with pytest.raises(ValueError, match="without its resolved commit"):
        materialize_model_weights(replace(plan, source_revision=None))


@pytest.mark.skipif(os.name == "nt", reason="standard HF cache uses POSIX symlinks")
def test_missing_blob_fails_without_publishing_a_partial_copy(
    tmp_path,
    monkeypatch,
) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    snapshot, weight, _config = _cached_snapshot(tmp_path)
    cache_root = tmp_path / "soup-cache"
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        hubs,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )
    plan = plan_model_weights("org/model")
    weight.unlink()

    with pytest.raises(FileNotFoundError, match="cached snapshot file"):
        materialize_model_weights(plan)

    assert not os.path.lexists(plan.weights_dir)


@pytest.mark.skipif(os.name == "nt", reason="standard HF cache uses POSIX symlinks")
def test_snapshot_symlink_cannot_escape_the_hf_blob_store(tmp_path, monkeypatch) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    snapshot, _weight, _config = _cached_snapshot(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"unsafe":true}\n', encoding="utf-8")
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(outside)
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    monkeypatch.setattr(
        hubs,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )
    plan = plan_model_weights("org/model")

    with pytest.raises(ValueError, match="outside the Hugging Face blob store"):
        materialize_model_weights(plan)

    assert not os.path.lexists(plan.weights_dir)


@pytest.mark.skipif(os.name == "nt", reason="control requires a POSIX symlink")
def test_materialized_target_symlink_cannot_redirect_cache_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    snapshot, _weight, _config = _cached_snapshot(tmp_path)
    cache_root = tmp_path / "soup-cache"
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        hubs,
        "snapshot_download",
        lambda *_args, **_kwargs: str(snapshot),
    )
    plan = plan_model_weights("org/model")
    victim = cache_root / "weights" / "victim"
    victim.mkdir(parents=True)
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    Path(plan.weights_dir).symlink_to(victim, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        materialize_model_weights(plan)

    assert marker.read_text(encoding="utf-8") == "keep"
