"""Layer streaming copies a Hugging Face snapshot only from inside its cache.

A snapshot's files are links into the repo's ``blobs`` directory, and with
huggingface_hub 1.32 or later an entry there may link on into the cache-wide shared
store. Those links stay inside the cache and are followed. The ``blobs`` directory itself
has to be a real directory: when it is a link (a symlink, or a junction on Windows) it is
refused by name, wherever it points. A junction inside a snapshot is refused like a
directory symlink, because ``os.walk`` descends into one.

A copy made earlier is reused only while its snapshot would still be copied: when the
layout changes into one of the refused ones after the first copy, the next run is refused
the same way and the copy is left as it is.

A test that needs no symlink builds its snapshot from regular files and a hand-made copy
plan, so it also runs on a Windows account that cannot create symlinks. The junction check
is also tested with a supplied answer, so that it is covered on every OS.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

COMMIT = "a" * 40
WEIGHT_BLOB = "b" * 64
CONFIG_BLOB = "c" * 40
XET_HASH = "e" * 64
WEIGHTS = b"weights kept inside the cache"
CONFIG = '{"model_type":"llama"}\n'
OUTSIDE = '{"kept":"outside the blob store"}\n'
IS_A_LINK = "is a link and is not followed"
#: Where a linked blob store may point. Every one is refused: the last two stay inside the
#: repo folder, which for a local snapshot path is whatever folder the snapshot sits in.
LINK_TARGETS = [
    "outside-the-cache",
    "beside-the-repo-folder",
    "inside-the-repo-folder",
    "the-repo-folder-itself",
]

needs_junction = pytest.mark.skipif(
    os.name != "nt", reason="Windows junctions only exist on Windows"
)


def _repo(tmp_path: Path) -> Path:
    """``hub/models--org--model`` with a blob store and an empty snapshot directory."""
    repo = tmp_path / "hub" / "models--org--model"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / COMMIT).mkdir(parents=True)
    (repo / "blobs" / WEIGHT_BLOB).write_bytes(WEIGHTS)
    (repo / "blobs" / CONFIG_BLOB).write_text(CONFIG, encoding="utf-8")
    return repo


def _link_snapshot(repo: Path, *, config: str = CONFIG_BLOB) -> Path:
    """Fill the snapshot the way huggingface_hub does: relative links into ``blobs``."""
    snapshot = repo / "snapshots" / COMMIT
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / WEIGHT_BLOB)
    (snapshot / "config.json").symlink_to(Path("../../blobs") / config)
    return snapshot


def _link_target(tmp_path: Path, repo: Path, where: str) -> Path:
    """The directory a linked blob store points at, for one of ``LINK_TARGETS``."""
    return {
        "outside-the-cache": tmp_path / "outside" / "store",
        "beside-the-repo-folder": repo.parent / "unrelated",
        "inside-the-repo-folder": repo / "blob-store",
        "the-repo-folder-itself": repo,
    }[where]


def _link_blob_store(repo: Path, target: Path, *, junction: bool = False) -> None:
    """Move the files of ``<repo>/blobs`` into ``target`` and make ``blobs`` a link to it."""
    target.mkdir(parents=True, exist_ok=True)
    for blob in (repo / "blobs").iterdir():
        blob.rename(target / blob.name)
    (repo / "blobs").rmdir()
    if junction:
        import _winapi

        _winapi.CreateJunction(str(target), str(repo / "blobs"))
    else:
        (repo / "blobs").symlink_to(target, target_is_directory=True)


def _use_shared_store(repo: Path, store: Path) -> None:
    """Move the weight into ``store`` laid out as huggingface_hub's marked shared store."""
    (store / XET_HASH[:2]).mkdir(parents=True)
    (store / ".huggingface-shared-blobs").write_text("1\n", encoding="utf-8")
    (repo / "blobs" / WEIGHT_BLOB).rename(store / XET_HASH[:2] / XET_HASH)


def _blob_store_link(repo: Path) -> str:
    """The link's own path as the refusal prints it: resolved parents, the link kept."""
    return os.path.join(os.path.realpath(repo), "blobs")


def _weights_root(tmp_path: Path) -> Path:
    """Soup's ``weights`` directory; a refusal must leave it uncreated."""
    return tmp_path / "soup-cache" / "weights"


def _plan(monkeypatch, tmp_path: Path, snapshot: Path):
    """The copy plan a Hub id gets when huggingface_hub returns ``snapshot``."""
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import plan_model_weights

    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    monkeypatch.setattr(hubs, "snapshot_download", lambda *_a, **_k: str(snapshot))
    return plan_model_weights("org/model")


def _plan_of_regular_files(monkeypatch, tmp_path: Path, repo: Path):
    """Write the snapshot's two files as regular files and return a copy plan for them.

    No symlink is needed. It also points Soup's cache into ``tmp_path``. Not for a snapshot
    that already holds links: the writes would go through them.
    """
    from soup_cli.utils.spectrum_scan import ModelWeightsPlan

    snapshot = repo / "snapshots" / COMMIT
    (snapshot / "model.safetensors").write_bytes(WEIGHTS)
    (snapshot / "config.json").write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    return ModelWeightsPlan(
        model="org/model",
        source_dir=str(snapshot),
        weights_dir=str(_weights_root(tmp_path) / "org__model"),
        source_bytes=len(WEIGHTS),
        materialized_copy_bytes=len(WEIGHTS),
        materialize_bytes=len(WEIGHTS),
        source_files=(("model.safetensors", len(WEIGHTS), 0),),
        source_revision=COMMIT,
    )


def _assert_copied(resolved: Path) -> None:
    assert not (resolved / "model.safetensors").is_symlink()
    assert (resolved / "model.safetensors").read_bytes() == WEIGHTS
    assert (resolved / "config.json").read_text(encoding="utf-8") == CONFIG


def _assert_refused_as_a_link(refusal, repo: Path, tmp_path: Path) -> None:
    message = str(refusal.value)
    assert repr(_blob_store_link(repo)) in message
    assert "Soup copies weights only from inside the Hugging Face cache" in message
    assert not _weights_root(tmp_path).exists()


# --- the links huggingface_hub creates, and regular files, keep working -------------


@pytest.mark.requires_symlink
@pytest.mark.parametrize("layout", ["repo-blob-store", "shared-blob-store"])
def test_links_that_stay_inside_the_cache_are_copied(tmp_path, monkeypatch, layout) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    if layout == "shared-blob-store":
        _use_shared_store(repo, repo.parent / "blobs")
        (repo / "blobs" / WEIGHT_BLOB).symlink_to(
            Path("../../blobs") / XET_HASH[:2] / XET_HASH
        )
    plan = _plan(monkeypatch, tmp_path, _link_snapshot(repo))

    _assert_copied(Path(materialize_model_weights(plan)))


@pytest.mark.requires_symlink
def test_regular_file_beside_a_linked_weight_is_copied(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    snapshot = repo / "snapshots" / COMMIT
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / WEIGHT_BLOB)
    (snapshot / "config.json").write_text(CONFIG, encoding="utf-8")
    plan = _plan(monkeypatch, tmp_path, snapshot)

    _assert_copied(Path(materialize_model_weights(plan)))


def test_snapshot_of_regular_files_is_copied(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    plan = _plan_of_regular_files(monkeypatch, tmp_path, _repo(tmp_path))

    _assert_copied(Path(materialize_model_weights(plan)))


def test_real_subdirectory_inside_a_snapshot_is_copied(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    subdirectory = repo / "snapshots" / COMMIT / "onnx"
    subdirectory.mkdir()
    (subdirectory / "config.json").write_text(CONFIG, encoding="utf-8")
    plan = _plan_of_regular_files(monkeypatch, tmp_path, repo)

    resolved = Path(materialize_model_weights(plan))

    _assert_copied(resolved)
    assert (resolved / "onnx" / "config.json").read_text(encoding="utf-8") == CONFIG


# --- a linked blob store is refused, wherever it points -----------------------------


@pytest.mark.requires_symlink
@pytest.mark.parametrize("where", LINK_TARGETS)
def test_blob_store_symlink_is_refused(tmp_path, monkeypatch, where) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    target = _link_target(tmp_path, repo, where)
    _link_blob_store(repo, target)
    (target / "notes.json").write_text(OUTSIDE, encoding="utf-8")
    plan = _plan(monkeypatch, tmp_path, _link_snapshot(repo, config="notes.json"))

    with pytest.raises(ValueError, match=IS_A_LINK) as refusal:
        materialize_model_weights(plan)

    _assert_refused_as_a_link(refusal, repo, tmp_path)


@needs_junction
@pytest.mark.parametrize("where", LINK_TARGETS)
def test_blob_store_junction_is_refused(tmp_path, monkeypatch, where) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    _link_blob_store(repo, _link_target(tmp_path, repo, where), junction=True)
    plan = _plan_of_regular_files(monkeypatch, tmp_path, repo)

    with pytest.raises(ValueError, match=IS_A_LINK) as refusal:
        materialize_model_weights(plan)

    _assert_refused_as_a_link(refusal, repo, tmp_path)


def test_blob_store_reported_as_a_junction_is_refused(tmp_path, monkeypatch) -> None:
    """The junction half of the check on every OS: a real directory, a supplied answer."""
    from soup_cli.utils import spectrum_scan

    repo = _repo(tmp_path)
    plan = _plan_of_regular_files(monkeypatch, tmp_path, repo)
    monkeypatch.setattr(
        spectrum_scan, "_is_junction", lambda path: os.path.basename(path) == "blobs"
    )

    with pytest.raises(ValueError, match=IS_A_LINK) as refusal:
        spectrum_scan.materialize_model_weights(plan)

    _assert_refused_as_a_link(refusal, repo, tmp_path)


@pytest.mark.requires_symlink
def test_dangling_blob_store_link_is_refused_as_a_link(tmp_path, monkeypatch) -> None:
    """A linked blob store is named as a link even when its target is gone."""
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    plan = _plan(monkeypatch, tmp_path, _link_snapshot(repo))
    target = _link_target(tmp_path, repo, "outside-the-cache")
    _link_blob_store(repo, target)
    for blob in target.iterdir():
        blob.unlink()
    target.rmdir()

    with pytest.raises(ValueError, match=IS_A_LINK) as refusal:
        materialize_model_weights(plan)

    _assert_refused_as_a_link(refusal, repo, tmp_path)


@pytest.mark.parametrize("left_behind", ["nothing", "a-regular-file"])
def test_missing_blob_store_is_reported(tmp_path, monkeypatch, left_behind) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    for blob in (repo / "blobs").iterdir():
        blob.unlink()
    (repo / "blobs").rmdir()
    if left_behind == "a-regular-file":
        (repo / "blobs").write_text("not a directory\n", encoding="utf-8")
    plan = _plan_of_regular_files(monkeypatch, tmp_path, repo)

    with pytest.raises(FileNotFoundError, match="cached Hugging Face blob store is missing"):
        materialize_model_weights(plan)

    assert not _weights_root(tmp_path).exists()


@needs_junction
@pytest.mark.requires_symlink
def test_shared_store_that_is_a_junction_is_not_used(tmp_path, monkeypatch) -> None:
    """The shared store has the same rule; a junction there leaves the weight outside."""
    import _winapi

    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    elsewhere = tmp_path / "outside" / "shared"
    _use_shared_store(repo, elsewhere)
    _winapi.CreateJunction(str(elsewhere), str(repo.parent / "blobs"))
    (repo / "blobs" / WEIGHT_BLOB).symlink_to(Path("../../blobs") / XET_HASH[:2] / XET_HASH)
    plan = _plan(monkeypatch, tmp_path, _link_snapshot(repo))

    with pytest.raises(ValueError, match="points outside the Hugging Face blob store"):
        materialize_model_weights(plan)

    assert not _weights_root(tmp_path).exists()


# --- a linked directory inside a snapshot is refused --------------------------------


def _outside_directory(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.json").write_text(OUTSIDE, encoding="utf-8")
    return outside


@pytest.mark.requires_symlink
def test_symlinked_directory_inside_a_snapshot_is_refused(tmp_path, monkeypatch) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    snapshot = _link_snapshot(repo)
    (snapshot / "extra").symlink_to(_outside_directory(tmp_path), target_is_directory=True)
    plan = _plan(monkeypatch, tmp_path, snapshot)

    with pytest.raises(ValueError, match="directory symlink is not allowed: 'extra'"):
        materialize_model_weights(plan)

    assert not _weights_root(tmp_path).exists()


@needs_junction
def test_junction_inside_a_snapshot_is_refused(tmp_path, monkeypatch) -> None:
    import _winapi

    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    _winapi.CreateJunction(
        str(_outside_directory(tmp_path)), str(repo / "snapshots" / COMMIT / "extra")
    )
    plan = _plan_of_regular_files(monkeypatch, tmp_path, repo)

    with pytest.raises(ValueError, match="directory junction is not allowed: 'extra'"):
        materialize_model_weights(plan)

    assert not _weights_root(tmp_path).exists()


def test_directory_reported_as_a_junction_is_refused(tmp_path, monkeypatch) -> None:
    """The snapshot walk's refusal on every OS: a real directory, a supplied answer."""
    from soup_cli.utils import spectrum_scan

    repo = _repo(tmp_path)
    (repo / "snapshots" / COMMIT / "extra").mkdir()
    plan = _plan_of_regular_files(monkeypatch, tmp_path, repo)
    monkeypatch.setattr(
        spectrum_scan, "_is_junction", lambda path: os.path.basename(path) == "extra"
    )

    with pytest.raises(ValueError, match="directory junction is not allowed: 'extra'"):
        spectrum_scan.materialize_model_weights(plan)

    assert not _weights_root(tmp_path).exists()


@pytest.mark.parametrize(
    ("reparse_tag", "expected"),
    [(0xA0000003, True), (0xA000000C, False), (None, False)],
    ids=["mount-point", "symlink", "no-reparse-tag"],
)
def test_junction_check_reads_the_mount_point_tag(monkeypatch, reparse_tag, expected) -> None:
    """On every OS: the stat result is supplied, the comparison is the code's own."""
    from soup_cli.utils import spectrum_scan

    fields = {} if reparse_tag is None else {"st_reparse_tag": reparse_tag}
    info = SimpleNamespace(**fields)
    monkeypatch.setattr(spectrum_scan, "os", SimpleNamespace(lstat=lambda _path: info))

    assert spectrum_scan._is_junction("snapshots/extra") is expected


# --- the same rule for a local path to a snapshot (#1511) ---------------------------
#
# A local snapshot path is copied only when its weights are symlinks, so these need one.


@pytest.mark.requires_symlink
@pytest.mark.parametrize("where", LINK_TARGETS)
def test_local_snapshot_path_with_a_linked_blob_store_is_refused(
    tmp_path, monkeypatch, where
) -> None:
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    repo = _repo(tmp_path)
    target = _link_target(tmp_path, repo, where)
    _link_blob_store(repo, target)
    (target / "notes.json").write_text(OUTSIDE, encoding="utf-8")
    snapshot = _link_snapshot(repo, config="notes.json")
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))

    with pytest.raises(ValueError, match=IS_A_LINK) as refusal:
        resolve_model_weights(str(snapshot))

    _assert_refused_as_a_link(refusal, repo, tmp_path)


@pytest.mark.requires_symlink
def test_local_snapshot_in_a_plain_folder_cannot_reach_that_folder(tmp_path, monkeypatch) -> None:
    """No repo folder at all: ``snapshots`` sits in a folder that holds other files."""
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    folder = tmp_path / "downloads"
    (folder / "snapshots" / COMMIT).mkdir(parents=True)
    (folder / WEIGHT_BLOB).write_bytes(WEIGHTS)
    (folder / "notes.json").write_text(OUTSIDE, encoding="utf-8")
    (folder / "blobs").symlink_to(folder, target_is_directory=True)
    snapshot = _link_snapshot(folder, config="notes.json")
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))

    with pytest.raises(ValueError, match=IS_A_LINK) as refusal:
        resolve_model_weights(str(snapshot))

    _assert_refused_as_a_link(refusal, folder, tmp_path)


@needs_junction
@pytest.mark.requires_symlink
def test_local_snapshot_path_with_a_junction_inside_is_refused(tmp_path, monkeypatch) -> None:
    import _winapi

    from soup_cli.utils.spectrum_scan import resolve_model_weights

    snapshot = _link_snapshot(_repo(tmp_path))
    _winapi.CreateJunction(str(_outside_directory(tmp_path)), str(snapshot / "extra"))
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))

    with pytest.raises(ValueError, match="directory junction is not allowed: 'extra'"):
        resolve_model_weights(str(snapshot))

    assert not _weights_root(tmp_path).exists()


# --- a copy made earlier is reused only while its snapshot would still be copied ----
#
# The reuse answer compares link names and sizes, so these need a snapshot of symlinks.


def _copy_once(monkeypatch, tmp_path: Path, snapshot: Path) -> Path:
    """Copy a snapshot whose layout is accepted and return the copy's directory."""
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    resolved = Path(materialize_model_weights(_plan(monkeypatch, tmp_path, snapshot)))
    _assert_copied(resolved)
    return resolved


def _record_copies(monkeypatch) -> list[str]:
    """Every file the copy step writes from now on."""
    from soup_cli.utils import spectrum_scan

    copied: list[str] = []
    copy_file = spectrum_scan._copy_regular_file

    def _recording(source: str, destination: str) -> None:
        copied.append(os.path.basename(destination))
        copy_file(source, destination)

    monkeypatch.setattr(spectrum_scan, "_copy_regular_file", _recording)
    return copied


def _refusal_of_the_second_call(monkeypatch, resolved: Path, error, match: str, model="org/model"):
    """Plan and copy again: the copy is planned afresh, refused, and the old one is kept."""
    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    stamp = (resolved / "model.safetensors").stat().st_mtime_ns
    copied = _record_copies(monkeypatch)
    plan = plan_model_weights(model)
    assert plan.materialize_bytes == len(WEIGHTS), "the earlier copy was planned for reuse"

    with pytest.raises(error, match=match) as refusal:
        materialize_model_weights(plan)

    assert copied == []
    _assert_copied(resolved)
    assert (resolved / "model.safetensors").stat().st_mtime_ns == stamp
    return refusal


@pytest.mark.requires_symlink
@pytest.mark.parametrize("layout", ["repo-blob-store", "shared-blob-store", "real-subdirectory"])
def test_copy_of_an_unchanged_snapshot_is_reused_without_copying(
    tmp_path, monkeypatch, layout
) -> None:
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    if layout == "shared-blob-store":
        _use_shared_store(repo, repo.parent / "blobs")
        (repo / "blobs" / WEIGHT_BLOB).symlink_to(
            Path("../../blobs") / XET_HASH[:2] / XET_HASH
        )
    snapshot = _link_snapshot(repo)
    if layout == "real-subdirectory":
        (snapshot / "onnx").mkdir()
        (snapshot / "onnx" / "config.json").write_text(CONFIG, encoding="utf-8")
    resolved = _copy_once(monkeypatch, tmp_path, snapshot)
    stamp = (resolved / "model.safetensors").stat().st_mtime_ns
    copied = _record_copies(monkeypatch)

    plan = _plan(monkeypatch, tmp_path, snapshot)

    assert plan.materialize_bytes == 0
    assert plan.materialized_copy_bytes == len(WEIGHTS)
    assert Path(materialize_model_weights(plan)) == resolved
    assert copied == []
    assert (resolved / "model.safetensors").stat().st_mtime_ns == stamp


@pytest.mark.requires_symlink
def test_copy_is_replaced_when_a_weight_links_to_another_blob(tmp_path, monkeypatch) -> None:
    """Same size, another blob name: the name is what the reuse answer compares."""
    from soup_cli.utils.spectrum_scan import materialize_model_weights

    repo = _repo(tmp_path)
    snapshot = _link_snapshot(repo)
    resolved = _copy_once(monkeypatch, tmp_path, snapshot)
    other = WEIGHTS[::-1]
    (repo / "blobs" / ("f" * 64)).write_bytes(other)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / ("f" * 64))
    copied = _record_copies(monkeypatch)

    plan = _plan(monkeypatch, tmp_path, snapshot)

    assert plan.materialize_bytes == len(other)
    assert Path(materialize_model_weights(plan)) == resolved
    assert sorted(copied) == ["config.json", "model.safetensors"]
    assert (resolved / "model.safetensors").read_bytes() == other


@pytest.mark.requires_symlink
@pytest.mark.parametrize("where", LINK_TARGETS)
def test_copy_is_not_reused_once_the_blob_store_is_a_symlink(tmp_path, monkeypatch, where) -> None:
    repo = _repo(tmp_path)
    resolved = _copy_once(monkeypatch, tmp_path, _link_snapshot(repo))
    _link_blob_store(repo, _link_target(tmp_path, repo, where))

    refusal = _refusal_of_the_second_call(monkeypatch, resolved, ValueError, IS_A_LINK)

    assert repr(_blob_store_link(repo)) in str(refusal.value)


@needs_junction
@pytest.mark.requires_symlink
@pytest.mark.parametrize("where", LINK_TARGETS)
def test_copy_is_not_reused_once_the_blob_store_is_a_junction(tmp_path, monkeypatch, where) -> None:
    repo = _repo(tmp_path)
    resolved = _copy_once(monkeypatch, tmp_path, _link_snapshot(repo))
    _link_blob_store(repo, _link_target(tmp_path, repo, where), junction=True)

    refusal = _refusal_of_the_second_call(monkeypatch, resolved, ValueError, IS_A_LINK)

    assert repr(_blob_store_link(repo)) in str(refusal.value)


@pytest.mark.requires_symlink
def test_copy_is_not_reused_once_the_blob_store_is_reported_as_a_junction(
    tmp_path, monkeypatch
) -> None:
    """The junction half of the reuse answer on every OS: a supplied answer."""
    from soup_cli.utils import spectrum_scan

    repo = _repo(tmp_path)
    resolved = _copy_once(monkeypatch, tmp_path, _link_snapshot(repo))
    monkeypatch.setattr(
        spectrum_scan, "_is_junction", lambda path: os.path.basename(path) == "blobs"
    )

    refusal = _refusal_of_the_second_call(monkeypatch, resolved, ValueError, IS_A_LINK)

    assert repr(_blob_store_link(repo)) in str(refusal.value)


@pytest.mark.requires_symlink
def test_copy_is_not_reused_once_the_snapshot_holds_a_directory_symlink(
    tmp_path, monkeypatch
) -> None:
    snapshot = _link_snapshot(_repo(tmp_path))
    resolved = _copy_once(monkeypatch, tmp_path, snapshot)
    (snapshot / "extra").symlink_to(_outside_directory(tmp_path), target_is_directory=True)

    _refusal_of_the_second_call(
        monkeypatch, resolved, ValueError, "directory symlink is not allowed: 'extra'"
    )


@needs_junction
@pytest.mark.requires_symlink
def test_copy_is_not_reused_once_the_snapshot_holds_a_junction(tmp_path, monkeypatch) -> None:
    import _winapi

    snapshot = _link_snapshot(_repo(tmp_path))
    resolved = _copy_once(monkeypatch, tmp_path, snapshot)
    _winapi.CreateJunction(str(_outside_directory(tmp_path)), str(snapshot / "extra"))

    _refusal_of_the_second_call(
        monkeypatch, resolved, ValueError, "directory junction is not allowed: 'extra'"
    )


@pytest.mark.requires_symlink
def test_copy_is_not_reused_once_a_snapshot_directory_is_reported_as_a_junction(
    tmp_path, monkeypatch
) -> None:
    """The snapshot walk's half of the reuse answer on every OS: a supplied answer."""
    from soup_cli.utils import spectrum_scan

    snapshot = _link_snapshot(_repo(tmp_path))
    resolved = _copy_once(monkeypatch, tmp_path, snapshot)
    (snapshot / "extra").mkdir()
    monkeypatch.setattr(
        spectrum_scan, "_is_junction", lambda path: os.path.basename(path) == "extra"
    )

    _refusal_of_the_second_call(
        monkeypatch, resolved, ValueError, "directory junction is not allowed: 'extra'"
    )


@pytest.mark.requires_symlink
def test_copy_is_not_reused_once_a_weight_links_outside_the_blob_store(
    tmp_path, monkeypatch
) -> None:
    """Same link name and size as the recorded blob, kept in another directory."""
    repo = _repo(tmp_path)
    snapshot = _link_snapshot(repo)
    resolved = _copy_once(monkeypatch, tmp_path, snapshot)
    (repo / "elsewhere").mkdir()
    (repo / "elsewhere" / WEIGHT_BLOB).write_bytes(WEIGHTS[::-1])
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(Path("../../elsewhere") / WEIGHT_BLOB)

    _refusal_of_the_second_call(
        monkeypatch, resolved, ValueError, "points outside the Hugging Face blob store"
    )


@pytest.mark.requires_symlink
def test_copy_is_not_reused_once_a_snapshot_file_has_no_blob(tmp_path, monkeypatch) -> None:
    """A layout the copy step reports as incomplete is planned afresh, not refused early."""
    repo = _repo(tmp_path)
    resolved = _copy_once(monkeypatch, tmp_path, _link_snapshot(repo))
    (repo / "blobs" / CONFIG_BLOB).unlink()

    _refusal_of_the_second_call(
        monkeypatch, resolved, FileNotFoundError, "'config.json' is missing its blob"
    )


@pytest.mark.requires_symlink
@pytest.mark.parametrize("change", ["blob-store-symlink", "directory-symlink-in-the-snapshot"])
def test_local_snapshot_path_copy_is_not_reused_once_a_link_appears(
    tmp_path, monkeypatch, change
) -> None:
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    repo = _repo(tmp_path)
    snapshot = _link_snapshot(repo)
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    resolved = Path(resolve_model_weights(str(snapshot)))
    _assert_copied(resolved)
    if change == "blob-store-symlink":
        _link_blob_store(repo, _link_target(tmp_path, repo, "outside-the-cache"))
        match = IS_A_LINK
    else:
        (snapshot / "extra").symlink_to(_outside_directory(tmp_path), target_is_directory=True)
        match = "directory symlink is not allowed: 'extra'"

    _refusal_of_the_second_call(monkeypatch, resolved, ValueError, match, model=str(snapshot))
