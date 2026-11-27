"""Layer streaming over huggingface_hub's cache-wide shared blob store.

Since huggingface_hub 1.32, a Xet download stores its bytes once per cache in
``<cache>/blobs/<2-hex>/<xet-hash>`` (a directory marked with
``.huggingface-shared-blobs``); the per-repo ``blobs/<etag>`` entry becomes a
relative symlink to it. Soup must copy such a snapshot, and still refuse links
that leave both stores.
"""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import pytest

COMMIT = "a" * 40
ETAG = "b" * 64
XET_HASH = "c" * 64
CONFIG_BLOB = "d" * 40


def _shared_store_snapshot(tmp_path: Path, *, marker: bool = True) -> tuple[Path, Path]:
    import torch
    from safetensors.torch import save_file

    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / COMMIT
    store = tmp_path / "blobs"
    for directory in (blobs, snapshot, store / XET_HASH[:2]):
        directory.mkdir(parents=True)
    if marker:
        (store / ".huggingface-shared-blobs").write_text("1\n", encoding="utf-8")

    shared = store / XET_HASH[:2] / XET_HASH
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.eye(2)}, str(shared))
    (blobs / ETAG).symlink_to(Path("../../blobs") / XET_HASH[:2] / XET_HASH)
    (blobs / CONFIG_BLOB).write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / ETAG)
    (snapshot / "config.json").symlink_to(Path("../../blobs") / CONFIG_BLOB)
    return snapshot, shared


def _use_snapshot(monkeypatch, tmp_path: Path, snapshot: Path) -> Path:
    from soup_cli.utils import hubs

    cache_root = tmp_path / "soup-cache"
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(hubs, "snapshot_download", lambda *_a, **_k: str(snapshot))
    return cache_root


@pytest.mark.requires_symlink
def test_shared_store_snapshot_is_materialized(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    snapshot, shared = _shared_store_snapshot(tmp_path)
    cache_root = _use_snapshot(monkeypatch, tmp_path, snapshot)

    resolved = Path(resolve_model_weights("org/model"))

    assert resolved == cache_root / "weights" / "org__model"
    assert not (resolved / "model.safetensors").is_symlink()
    assert (resolved / "model.safetensors").read_bytes() == shared.read_bytes()
    metadata = (
        resolved / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    ).read_text(encoding="utf-8").splitlines()
    # the recorded blob id is the repo's etag, not the shared store's Xet hash
    assert metadata[:2] == [COMMIT, ETAG]
    index = shard_checkpoint(
        str(resolved), str(tmp_path / "shards"), dtype="float32", arch="llama"
    )
    assert index.n_layers == 1


@pytest.mark.requires_symlink
def test_shared_store_copy_is_reused(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights, resolve_model_weights

    snapshot, _shared = _shared_store_snapshot(tmp_path)
    _use_snapshot(monkeypatch, tmp_path, snapshot)
    resolve_model_weights("org/model")

    plan = plan_model_weights("org/model")

    assert plan.materialize_bytes == 0
    assert plan.materialized_copy_bytes > 0


@pytest.mark.requires_symlink
def test_unmarked_sibling_blobs_dir_is_not_trusted(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    snapshot, _shared = _shared_store_snapshot(tmp_path, marker=False)
    _use_snapshot(monkeypatch, tmp_path, snapshot)
    plan = plan_model_weights("org/model")

    with pytest.raises(ValueError, match="outside the Hugging Face blob store"):
        materialize_model_weights(plan)

    assert not os.path.lexists(plan.weights_dir)


@pytest.mark.requires_symlink
def test_link_leaving_both_stores_is_still_refused(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    snapshot, _shared = _shared_store_snapshot(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"unsafe":true}\n', encoding="utf-8")
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(outside)
    _use_snapshot(monkeypatch, tmp_path, snapshot)
    plan = plan_model_weights("org/model")

    with pytest.raises(ValueError, match="outside the Hugging Face blob store"):
        materialize_model_weights(plan)

    assert not os.path.lexists(plan.weights_dir)


def _snapshot_with_store(tmp_path: Path, store: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    repo = tmp_path / "hub" / "models--org--model"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / COMMIT
    for directory in (blobs, snapshot, store / XET_HASH[:2]):
        directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.eye(2)},
        str(store / XET_HASH[:2] / XET_HASH),
    )
    (blobs / ETAG).symlink_to(Path("../../blobs") / XET_HASH[:2] / XET_HASH)
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / ETAG)
    return snapshot


def _plan(monkeypatch, tmp_path: Path, snapshot: Path):
    from soup_cli.utils.spectrum_scan import plan_model_weights

    _use_snapshot(monkeypatch, tmp_path, snapshot)
    return plan_model_weights("org/model")


@pytest.mark.requires_symlink
def test_symlinked_marker_is_not_trusted(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    store = tmp_path / "hub" / "blobs"
    snapshot = _snapshot_with_store(tmp_path, store)
    real_marker = tmp_path / "elsewhere-marker"
    real_marker.write_text("1\n", encoding="utf-8")
    (store / ".huggingface-shared-blobs").symlink_to(real_marker)
    plan = _plan(monkeypatch, tmp_path, snapshot)

    with pytest.raises(ValueError, match="outside the Hugging Face blob store"):
        materialize_model_weights(plan)
    assert not os.path.lexists(plan.weights_dir)


@pytest.mark.requires_symlink
def test_symlinked_store_directory_is_not_trusted(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    outside = tmp_path / "outside-store"
    snapshot = _snapshot_with_store(tmp_path, outside)
    (outside / ".huggingface-shared-blobs").write_text("1\n", encoding="utf-8")
    (tmp_path / "hub" / "blobs").symlink_to(outside, target_is_directory=True)
    plan = _plan(monkeypatch, tmp_path, snapshot)

    with pytest.raises(ValueError, match="outside the Hugging Face blob store"):
        materialize_model_weights(plan)
    assert not os.path.lexists(plan.weights_dir)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("winerror", [None, 5], ids=["native-rename", "windows-access-denied"])
def test_concurrent_materialization_uses_the_winners_copy(
    tmp_path, monkeypatch, winerror: int | None
) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    snapshot, shared = _shared_store_snapshot(tmp_path)
    plan = _plan(monkeypatch, tmp_path, snapshot)
    real_replace = os.replace
    raced = False

    def racing_replace(src: str, dst: str) -> None:
        nonlocal raced
        if not raced:
            raced = True
            materialize_model_weights(plan)
            if winerror is not None:
                collision = PermissionError(errno.EACCES, "Access is denied", dst)
                collision.winerror = winerror
                raise collision
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", racing_replace)
    resolved = Path(materialize_model_weights(plan))

    assert raced and resolved == Path(plan.weights_dir)
    assert (resolved / "model.safetensors").read_bytes() == shared.read_bytes()
    assert list(resolved.parent.iterdir()) == [resolved]


@pytest.mark.requires_symlink
@pytest.mark.parametrize(
    ("mismatch", "error", "message"),
    [
        ("revision", errno.EEXIST, "File exists"),
        ("blob", errno.EEXIST, "File exists"),
        ("manifest", errno.EEXIST, "File exists"),
        (None, errno.EIO, "Input/output error"),
    ],
)
def test_concurrent_materialization_propagates_publish_error(
    tmp_path, monkeypatch, mismatch: str | None, error: int, message: str
) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    snapshot, shared = _shared_store_snapshot(tmp_path)
    plan = _plan(monkeypatch, tmp_path, snapshot)
    target = Path(plan.weights_dir)
    publish_error = OSError(error, message, str(target))
    revision = "e" * 40 if mismatch == "revision" else COMMIT
    blob = "f" * 64 if mismatch == "blob" else ETAG
    weight = shared.read_bytes()
    if mismatch == "manifest":
        weight = weight[:-1]
    metadata = f"{revision}\n{blob}\n0\n"
    metadata_path = target / ".cache/huggingface/download/model.safetensors.metadata"

    def racing_replace(src: str, dst: str) -> None:
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text(metadata, encoding="utf-8")
        (target / "model.safetensors").write_bytes(weight)
        raise publish_error

    monkeypatch.setattr(os, "replace", racing_replace)
    with pytest.raises(OSError) as exc:
        materialize_model_weights(plan)

    assert exc.value is publish_error
    assert exc.value.strerror == message
    assert metadata_path.read_text(encoding="utf-8") == metadata
    assert (target / "model.safetensors").read_bytes() == weight
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.requires_symlink
@pytest.mark.parametrize(
    ("redirect", "message"),
    [
        ("target", "materialized weights path must not be a symlink"),
        ("parent", "materialized weights path changed during snapshot publish"),
    ],
)
def test_concurrent_materialization_refuses_a_redirected_winner(
    tmp_path, monkeypatch, redirect: str, message: str
) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    snapshot, shared = _shared_store_snapshot(tmp_path)
    plan = _plan(monkeypatch, tmp_path, snapshot)
    target = Path(plan.weights_dir)
    outside = tmp_path / "outside"
    real_replace = os.replace

    def racing_replace(src: str, dst: str) -> None:
        if redirect == "target":
            shutil.copytree(src, outside)
            target.symlink_to(outside, target_is_directory=True)
        else:
            real_replace(target.parent, outside)
            target.parent.symlink_to(outside, target_is_directory=True)
            shutil.copytree(src, dst)
        raise FileExistsError(errno.EEXIST, "File exists", dst)

    monkeypatch.setattr(os, "replace", racing_replace)
    with pytest.raises(ValueError, match=f"^{message}$"):
        materialize_model_weights(plan)

    assert (target / "model.safetensors").read_bytes() == shared.read_bytes()
    assert list(target.parent.iterdir()) == [target]
