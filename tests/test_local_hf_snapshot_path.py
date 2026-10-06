"""Layer streaming given a local path to a Hugging Face cache snapshot.

``snapshots/<commit>/*.safetensors`` are symlinks into ``blobs/``. The sharder and the
scanner skip symlinks, so a local snapshot path must be copied to regular files the same
way the snapshot behind a Hub id is, and share that copy's cache slot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

COMMIT = "a" * 40
ETAG = "b" * 64


def _snapshot(tmp_path: Path, *, repo: str = "models--org--model", commit: str = COMMIT) -> Path:
    import torch
    from safetensors.torch import save_file

    root = tmp_path / repo
    blobs = root / "blobs"
    snapshot = root / "snapshots" / commit
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.eye(2)}, str(blobs / ETAG))
    (blobs / ("d" * 40)).write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / ETAG)
    (snapshot / "config.json").symlink_to(Path("../../blobs") / ("d" * 40))
    return snapshot


@pytest.mark.requires_symlink
def test_local_snapshot_path_is_materialized(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    snapshot = _snapshot(tmp_path)
    cache_root = tmp_path / "soup-cache"
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))

    resolved = Path(resolve_model_weights(str(snapshot)))

    assert resolved == cache_root / "weights" / "org__model"
    assert not (resolved / "model.safetensors").is_symlink()
    index = shard_checkpoint(
        str(resolved), str(tmp_path / "shards"), dtype="float32", arch="llama"
    )
    assert index.n_layers == 1


@pytest.mark.requires_symlink
def test_local_path_reuses_the_hub_id_copy(tmp_path, monkeypatch) -> None:
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import plan_model_weights, resolve_model_weights

    snapshot = _snapshot(tmp_path)
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    monkeypatch.setattr(hubs, "snapshot_download", lambda *_a, **_k: str(snapshot))
    resolve_model_weights("org/model")

    plan = plan_model_weights(str(snapshot))

    assert plan.materialize_bytes == 0
    assert plan.materialized_copy_bytes > 0


@pytest.mark.requires_symlink
def test_symlinked_weights_outside_a_snapshot_layout_are_refused(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights

    target = _snapshot(tmp_path) / "model.safetensors"
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "model.safetensors").symlink_to(target)
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))

    with pytest.raises(ValueError, match="symlinked .safetensors files outside"):
        plan_model_weights(str(plain))


@pytest.mark.requires_symlink
def test_snapshot_outside_an_hf_cache_name_keeps_a_path_derived_slot(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import model_slug, plan_model_weights

    snapshot = _snapshot(tmp_path, repo="my-model-cache")
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))

    plan = plan_model_weights(str(snapshot))

    assert Path(plan.weights_dir).name == model_slug(str(snapshot))
    assert Path(plan.weights_dir).name != "org__model"
