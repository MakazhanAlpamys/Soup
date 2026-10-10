"""Which cache slot holds the regular-file copy of a Hugging Face snapshot.

Layer streaming keeps one regular-file copy of a symlinked snapshot per slot under
``<soup cache>/weights`` and reuses it while file names, sizes, the commit and the blob names
still match. The Hub id ``org/model`` uses the slot ``org__model``. A local path
``.../models--org--model/snapshots/<commit>`` uses that slot only when its repo folder is the
directory the Hub id resolves to in the active Hugging Face cache. A snapshot folder anywhere
else gets a slot named after its own resolved location, so a copy taken from one place is
never returned for a request that names another place.

The active cache is the one huggingface_hub uses when no ``cache_dir`` is passed. It fixes
that directory when it is imported, so these tests set ``constants.HF_HUB_CACHE`` rather than
an environment variable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

COMMIT = "a" * 40
BLOB = "b" * 64
REPO = "models--org--model"
HUB_BYTES = b"bytes kept in the active cache"
ELSEWHERE_BYTES = b"BYTES KEPT IN ANOTHER FOLDER.."
THIRD_BYTES = b"bytes kept in a third folder.."
LOCAL_SLOT_RE = re.compile(r"[A-Za-z0-9._-]{1,64}@[0-9a-f]{32}")

needs_junction = pytest.mark.skipif(
    os.name != "nt", reason="Windows junctions only exist on Windows"
)
windows_only = pytest.mark.skipif(
    os.name != "nt", reason="os.path.normcase folds letter case only on Windows"
)


def test_the_three_payloads_have_one_size() -> None:
    """The reuse check compares sizes, so the folders must differ in bytes only."""
    assert len(HUB_BYTES) == len(ELSEWHERE_BYTES) == len(THIRD_BYTES)
    assert len({HUB_BYTES, ELSEWHERE_BYTES, THIRD_BYTES}) == 3


def _snapshot(root: Path, data: bytes, *, repo: str = REPO) -> Path:
    """``<root>/<repo>/snapshots/<commit>`` laid out as huggingface_hub does, with one weight."""
    repo_dir = root / repo
    (repo_dir / "blobs").mkdir(parents=True)
    snapshot = repo_dir / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (repo_dir / "blobs" / BLOB).write_bytes(data)
    (snapshot / "model.safetensors").symlink_to(Path("../../blobs") / BLOB)
    return snapshot


def _use_hf_cache(monkeypatch, cache: Path, snapshot: Path) -> None:
    """Make ``cache`` the active Hugging Face cache and ``snapshot`` what a Hub id resolves to."""
    from huggingface_hub import constants

    from soup_cli.utils import hubs

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hubs, "snapshot_download", lambda *_a, **_k: str(snapshot))


def _weights(directory: Path) -> bytes:
    weight = directory / "model.safetensors"
    assert not weight.is_symlink()
    return weight.read_bytes()


def _is_under(path: Path, base: Path) -> bool:
    resolved = os.path.realpath(path)
    resolved_base = os.path.realpath(base)
    return os.path.commonpath([resolved, resolved_base]) == resolved_base


@pytest.fixture
def slots(tmp_path, monkeypatch) -> Path:
    """Soup's ``weights`` directory, placed under ``tmp_path``."""
    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "soup-cache"))
    return tmp_path / "soup-cache" / "weights"


@pytest.fixture
def copied(monkeypatch) -> list[str]:
    """The name of every file the materializer copies, in order."""
    from soup_cli.utils import spectrum_scan

    names: list[str] = []
    copy_file = spectrum_scan._copy_regular_file

    def counting(source: str, destination: str) -> None:
        names.append(os.path.basename(destination))
        copy_file(source, destination)

    monkeypatch.setattr(spectrum_scan, "_copy_regular_file", counting)
    return names


# ---------------------------------------------------------------------------
# A folder outside the active cache never shares the Hub id's slot
# ---------------------------------------------------------------------------
@pytest.mark.requires_symlink
def test_hub_id_copies_its_own_snapshot_after_a_same_named_folder_elsewhere(
    tmp_path, monkeypatch, slots
) -> None:
    from soup_cli.utils.spectrum_scan import (
        materialize_model_weights,
        plan_model_weights,
        resolve_model_weights,
    )

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    elsewhere = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)

    first = Path(resolve_model_weights(str(elsewhere)))
    plan = plan_model_weights("org/model")
    second = Path(materialize_model_weights(plan))

    assert plan.needs_materialization
    assert second == slots / "org__model"
    assert _weights(second) == HUB_BYTES
    assert first != second
    assert _weights(first) == ELSEWHERE_BYTES


@pytest.mark.requires_symlink
def test_same_named_folder_elsewhere_copies_its_own_snapshot_after_the_hub_id(
    tmp_path, monkeypatch, slots
) -> None:
    from soup_cli.utils.spectrum_scan import (
        materialize_model_weights,
        plan_model_weights,
        resolve_model_weights,
    )

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    elsewhere = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)

    first = Path(resolve_model_weights("org/model"))
    plan = plan_model_weights(str(elsewhere))
    second = Path(materialize_model_weights(plan))

    assert plan.needs_materialization
    assert first == slots / "org__model"
    assert second != first
    assert _weights(second) == ELSEWHERE_BYTES
    assert _weights(first) == HUB_BYTES


@pytest.mark.requires_symlink
def test_two_same_named_folders_get_two_slots(tmp_path, monkeypatch, slots) -> None:
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    one = _snapshot(tmp_path / "downloads-1", ELSEWHERE_BYTES)
    two = _snapshot(tmp_path / "downloads-2", THIRD_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)

    first = Path(resolve_model_weights(str(one)))
    second = Path(resolve_model_weights(str(two)))

    assert len({first, second, slots / "org__model"}) == 3
    assert _weights(first) == ELSEWHERE_BYTES
    assert _weights(second) == THIRD_BYTES
    assert not (slots / "org__model").exists()


@pytest.mark.requires_symlink
def test_two_long_paths_with_one_beginning_get_two_slots(tmp_path, monkeypatch, slots) -> None:
    """A slot named after the first 128 characters of the path would merge these two."""
    from soup_cli.utils.spectrum_scan import plan_model_weights

    shared = tmp_path / ("p" * max(8, 140 - len(str(tmp_path))))
    one = _snapshot(shared / "one", ELSEWHERE_BYTES, repo="my-model-cache")
    two = _snapshot(shared / "two", THIRD_BYTES, repo="my-model-cache")
    assert len(str(shared)) >= 140

    first = Path(plan_model_weights(str(one)).weights_dir)
    second = Path(plan_model_weights(str(two)).weights_dir)

    assert first != second
    assert first.name.startswith("my-model-cache@")
    assert second.name.startswith("my-model-cache@")


@pytest.mark.requires_symlink
def test_folder_in_the_cache_gets_its_own_slot_when_the_cache_is_unknown(
    tmp_path, monkeypatch, slots
) -> None:
    """Without huggingface_hub there is no active cache to compare with."""
    from soup_cli.utils import hubs
    from soup_cli.utils.spectrum_scan import plan_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)
    monkeypatch.setattr(hubs, "hf_cache_dir", lambda: None)

    plan = plan_model_weights(str(hub_snapshot))

    assert LOCAL_SLOT_RE.fullmatch(Path(plan.weights_dir).name)


# ---------------------------------------------------------------------------
# The folder the Hub id resolves to keeps sharing its slot
# ---------------------------------------------------------------------------
def _cache_layout(tmp_path: Path, layout: str) -> tuple[Path, Path]:
    """Build the cache and return ``(root in HF_HUB_CACHE, root in the local path)``.

    The linked layouts are a cache that was moved to another place and linked back.
    """
    real = tmp_path / "disk" / "hf-cache"
    _snapshot(real, HUB_BYTES)
    if layout == "plain":
        return real, real
    kind, env_root, path_root = layout.split("/")
    link = tmp_path / "hf-cache"
    if kind == "junction":
        import _winapi

        _winapi.CreateJunction(str(real), str(link))
    else:
        link.symlink_to(real, target_is_directory=True)
    roots = {"link": link, "real": real}
    return roots[env_root], roots[path_root]


CACHE_LAYOUTS = [
    "plain",
    "symlink/link/link",
    "symlink/link/real",
    "symlink/real/link",
    pytest.param("junction/link/link", marks=needs_junction),
    pytest.param("junction/link/real", marks=needs_junction),
    pytest.param("junction/real/link", marks=needs_junction),
]


@pytest.mark.requires_symlink
@pytest.mark.parametrize("layout", CACHE_LAYOUTS)
@pytest.mark.parametrize("order", ["local path first", "hub id first"])
def test_local_path_of_the_active_cache_shares_the_hub_id_slot(
    tmp_path, monkeypatch, slots, copied, layout, order
) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights, resolve_model_weights

    env_root, path_root = _cache_layout(tmp_path, layout)
    relative = Path(REPO) / "snapshots" / COMMIT
    _use_hf_cache(monkeypatch, env_root, env_root / relative)
    requests = [str(path_root / relative), "org/model"]
    if order == "hub id first":
        requests.reverse()

    first = Path(resolve_model_weights(requests[0]))
    after_first = list(copied)
    stamp = (first / "model.safetensors").stat().st_mtime_ns
    plan = plan_model_weights(requests[1])
    second = Path(resolve_model_weights(requests[1]))

    assert first == second == slots / "org__model"
    assert after_first == ["model.safetensors"]
    assert not plan.needs_materialization
    assert copied == after_first
    assert (second / "model.safetensors").stat().st_mtime_ns == stamp
    assert _weights(second) == HUB_BYTES
    assert [entry.name for entry in slots.iterdir()] == ["org__model"]


# ---------------------------------------------------------------------------
# A folder outside the cache: one location, one slot
# ---------------------------------------------------------------------------
@pytest.mark.requires_symlink
def test_folder_elsewhere_reuses_its_own_copy(tmp_path, monkeypatch, slots, copied) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights, resolve_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    elsewhere = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)

    first = Path(resolve_model_weights(str(elsewhere)))
    stamp = (first / "model.safetensors").stat().st_mtime_ns
    plan = plan_model_weights(str(elsewhere))
    second = Path(resolve_model_weights(str(elsewhere)))

    assert second == first
    assert not plan.needs_materialization
    assert plan.materialized_copy_bytes == len(ELSEWHERE_BYTES)
    assert copied == ["model.safetensors"]
    assert (second / "model.safetensors").stat().st_mtime_ns == stamp
    assert _weights(second) == ELSEWHERE_BYTES


def _spell(snapshot: Path, spelling: str, tmp_path: Path, monkeypatch) -> str:
    """Another way to write the path of ``snapshot``."""
    if spelling == "trailing separator":
        return str(snapshot) + os.sep
    if spelling == "dot-dot segment":
        return os.path.join(str(snapshot.parent), "..", "snapshots", COMMIT)
    if spelling == "relative":
        monkeypatch.chdir(snapshot.parent)
        return COMMIT
    if spelling == "other letter case":
        return str(snapshot).swapcase()
    if spelling == "through a linked parent":
        link = tmp_path / "shortcut"
        link.symlink_to(snapshot.parents[2], target_is_directory=True)
        return str(link / snapshot.parents[1].name / "snapshots" / COMMIT)
    raise AssertionError(spelling)


SPELLINGS = [
    "trailing separator",
    "dot-dot segment",
    "relative",
    "through a linked parent",
    pytest.param("other letter case", marks=windows_only),
]


@pytest.mark.requires_symlink
@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("repo", [REPO, "my-model-cache"])
def test_every_spelling_of_a_folder_elsewhere_maps_to_one_slot(
    tmp_path, monkeypatch, slots, copied, spelling, repo
) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights, resolve_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    elsewhere = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES, repo=repo)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)
    first = Path(resolve_model_weights(str(elsewhere)))

    plan = plan_model_weights(_spell(elsewhere, spelling, tmp_path, monkeypatch))

    assert Path(plan.weights_dir) == first
    assert not plan.needs_materialization
    assert copied == ["model.safetensors"]


@windows_only
@pytest.mark.requires_symlink
def test_letter_case_is_folded_where_the_volume_reports_a_path_as_spelled(
    tmp_path, monkeypatch, slots
) -> None:
    """Some volumes (RAM disks, a few network drives) cannot report a final path.

    ``os.path.realpath`` then returns the path as it was written, letter case included.
    """
    from soup_cli.utils.spectrum_scan import plan_model_weights

    cache = tmp_path / "hf-cache"
    relative = Path(REPO) / "snapshots" / COMMIT
    _use_hf_cache(monkeypatch, Path(str(cache).upper()), _snapshot(cache, HUB_BYTES))
    elsewhere = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES)
    lower_parent, upper_parent = str(tmp_path).lower(), str(tmp_path).upper()
    assert lower_parent != upper_parent

    def slot(parent: str, folder: str) -> str:
        return Path(plan_model_weights(os.path.join(parent, folder, relative)).weights_dir).name

    with monkeypatch.context() as patch:
        patch.setattr(os.path, "realpath", lambda path, **_kw: os.path.abspath(path))
        in_cache = slot(lower_parent, "hf-cache")
        spelled_lower = slot(lower_parent, "downloads")
        spelled_upper = slot(upper_parent, "downloads")

    assert in_cache == "org__model"
    assert spelled_lower == spelled_upper
    assert spelled_lower == Path(plan_model_weights(str(elsewhere)).weights_dir).name


# ---------------------------------------------------------------------------
# The name of a slot that belongs to a folder
# ---------------------------------------------------------------------------
@pytest.mark.requires_symlink
def test_slot_of_a_folder_elsewhere_is_one_bounded_name_inside_the_cache(
    tmp_path, monkeypatch, slots
) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)
    long_repo = "models--org--" + "x" * 55
    names = {}
    for repo in (REPO, "my model (copy)", long_repo):
        snapshot = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES, repo=repo)

        slot = Path(plan_model_weights(str(snapshot)).weights_dir)

        assert LOCAL_SLOT_RE.fullmatch(slot.name), slot.name
        assert slot.parent == slots
        assert _is_under(slot, slots)
        assert Path(plan_model_weights(str(snapshot)).weights_dir) == slot
        names[repo] = slot.name
    assert names["my model (copy)"].startswith("my_model_copy@")
    assert names[long_repo].startswith(long_repo[:64] + "@")
    assert len(names[long_repo]) == 64 + 1 + 32


@pytest.mark.requires_symlink
def test_slot_of_a_folder_elsewhere_names_the_folder_without_its_path(
    tmp_path, monkeypatch, slots
) -> None:
    from soup_cli.utils.spectrum_scan import plan_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    elsewhere = _snapshot(tmp_path / "private-folder-name", ELSEWHERE_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)

    plan = plan_model_weights(str(elsewhere))

    assert Path(plan.weights_dir).name.startswith(REPO + "@")
    assert "private-folder-name" not in os.path.relpath(plan.weights_dir, slots)
    assert plan.model == "org/model"
    assert plan.source_dir == os.path.realpath(elsewhere)


@pytest.mark.requires_symlink
def test_no_model_name_maps_onto_the_slot_of_a_folder(tmp_path, monkeypatch, slots) -> None:
    """``@`` is not in the alphabet of a slug, so the two kinds of slot cannot meet."""
    from soup_cli.utils.spectrum_scan import model_slug, plan_model_weights

    hub_snapshot = _snapshot(tmp_path / "hf-cache", HUB_BYTES)
    elsewhere = _snapshot(tmp_path / "downloads", ELSEWHERE_BYTES)
    _use_hf_cache(monkeypatch, tmp_path / "hf-cache", hub_snapshot)

    name = Path(plan_model_weights(str(elsewhere)).weights_dir).name

    assert model_slug(name) != name
    assert model_slug("org/" + name) != name
    assert model_slug(name.replace("--", "/")) != name


def test_snapshots_directory_at_a_volume_root_still_gets_a_name() -> None:
    from soup_cli.utils.spectrum_scan import _local_snapshot_slot

    root = os.path.abspath(os.sep)

    name = _local_snapshot_slot(os.path.join(root, "snapshots", COMMIT))

    assert re.fullmatch(r"local@[0-9a-f]{32}", name)


# ---------------------------------------------------------------------------
# Regular files are read where they are
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("folder", ["plain", os.path.join(REPO, "snapshots", COMMIT)])
def test_directory_of_regular_files_is_read_in_place(
    tmp_path, monkeypatch, slots, copied, folder
) -> None:
    """Also a cache snapshot written as regular files, as on an account without symlinks."""
    from huggingface_hub import constants

    from soup_cli.utils.spectrum_scan import materialize_model_weights, plan_model_weights

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    directory = tmp_path / "downloads" / folder
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(ELSEWHERE_BYTES)

    plan = plan_model_weights(str(directory))

    assert plan.weights_dir == plan.source_dir == os.path.realpath(directory)
    assert not plan.needs_materialization
    assert plan.materialized_copy_bytes == 0
    assert materialize_model_weights(plan) == os.path.realpath(directory)
    assert copied == []
    assert not slots.exists()
