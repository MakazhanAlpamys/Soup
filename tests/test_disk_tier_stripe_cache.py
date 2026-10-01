"""R4 — the index records where each layer lives, and the cache check reads it."""

import json
import os

import pytest

from soup_cli.utils.layer_shard import (
    ShardIndex,
    _atomic_write_index,
    _write_stripe_marker,
    extras_shard_path,
    inspect_shard_cache,
    layer_paths,
    layer_shard_path,
    read_shard_index,
    stripe_dirs,
)
from soup_cli.utils.stripe_roots import primary_cache_identity, stripe_folder_name

FP = "f" * 64


def _index(n_layers=3, stripe_roots=(), layer_roots=()):
    return ShardIndex(
        n_layers=n_layers, layer_keys=("w",), extra_keys=("e",), dtype="float32",
        total_params=1, arch="llama", soup_version="test", source_fingerprint=FP,
        stripe_roots=tuple(stripe_roots), layer_roots=tuple(layer_roots),
    )


def _cache(tmp_path, *, striped):
    """A cache on disk with EMPTY shard files: existence is all the check reads."""
    out = tmp_path / "cache" / "model"
    out.mkdir(parents=True)
    stripe_root = tmp_path / "stripe"
    stripe_root.mkdir()
    roots = (os.path.realpath(stripe_root),) if striped else ()
    index = _index(stripe_roots=roots, layer_roots=(0, 1, 0) if striped else ())
    dirs = stripe_dirs(str(out), roots)
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    for path in layer_paths(str(out), index):
        open(path, "wb").close()
    open(extras_shard_path(str(out)), "wb").close()
    for position, directory in enumerate(dirs[1:], start=1):
        _write_stripe_marker(
            directory, fingerprint=FP, position=position, n_roots=len(dirs),
            primary=primary_cache_identity(str(out)),
        )
    _atomic_write_index(index, str(out))
    return str(out), roots


def _inspect(out, roots):
    return inspect_shard_cache(out, "float32", FP, (), "none", False, "", "", stripe_roots=roots)


def _write_raw_index(out, **extra):
    payload = {
        "n_layers": 3, "layer_keys": ["w"], "extra_keys": ["e"], "dtype": "float32",
        "total_params": 1, **extra,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class TestTheIndexFields:
    def test_an_index_without_them_reads_as_one_root(self, tmp_path):
        _write_raw_index(str(tmp_path))
        index = read_shard_index(str(tmp_path))
        assert index.stripe_roots == () and index.layer_roots == ()

    def test_they_round_trip(self, tmp_path):
        out, roots = _cache(tmp_path, striped=True)
        index = read_shard_index(out)
        assert index.stripe_roots == roots and index.layer_roots == (0, 1, 0)

    def test_a_placement_that_misses_a_layer_is_refused(self, tmp_path):
        _write_raw_index(str(tmp_path), stripe_roots=["/x"], layer_roots=[0, 1])
        with pytest.raises(ValueError, match="names 2 layers"):
            read_shard_index(str(tmp_path))

    def test_a_placement_naming_a_root_the_index_lacks_is_refused(self, tmp_path):
        _write_raw_index(str(tmp_path), stripe_roots=["/x"], layer_roots=[0, 2, 0])
        with pytest.raises(ValueError, match="names a root"):
            read_shard_index(str(tmp_path))

    def test_stripe_roots_without_a_placement_is_refused(self, tmp_path):
        _write_raw_index(str(tmp_path), stripe_roots=["/x"])
        with pytest.raises(ValueError, match="layer_roots is empty"):
            read_shard_index(str(tmp_path))


class TestLayerPaths:
    def test_one_root_is_todays_layout(self, tmp_path):
        out = str(tmp_path / "model")
        assert layer_paths(out, _index()) == [layer_shard_path(out, idx) for idx in range(3)]

    def test_striped_layers_live_in_this_caches_folder_on_their_root(self, tmp_path):
        out = str(tmp_path / "cache" / "model")
        stripe = str(tmp_path / "stripe")
        paths = layer_paths(out, _index(stripe_roots=(stripe,), layer_roots=(0, 1, 0)))
        assert paths == [
            layer_shard_path(out, 0),
            layer_shard_path(os.path.join(stripe, stripe_folder_name(out)), 1),
            layer_shard_path(out, 2),
        ]


class TestTheCacheCheck:
    def test_a_matching_striped_cache_is_reusable(self, tmp_path):
        out, roots = _cache(tmp_path, striped=True)
        index, reason = _inspect(out, roots)
        assert index is not None, reason

    def test_asking_for_different_roots_invalidates_and_says_so(self, tmp_path):
        out, _ = _cache(tmp_path, striped=True)
        index, reason = _inspect(out, ())
        assert index is None and "stripe roots changed" in reason

    def test_a_missing_marker_invalidates_naming_it(self, tmp_path):
        out, roots = _cache(tmp_path, striped=True)
        marker = os.path.join(stripe_dirs(out, roots)[1], "stripe.json")
        os.remove(marker)
        index, reason = _inspect(out, roots)
        assert index is None and marker in reason

    def test_a_marker_from_another_cache_invalidates(self, tmp_path):
        out, roots = _cache(tmp_path, striped=True)
        folder = stripe_dirs(out, roots)[1]
        _write_stripe_marker(
            folder, fingerprint="0" * 64, position=1, n_roots=2,
            primary=primary_cache_identity(out),
        )
        index, reason = _inspect(out, roots)
        assert index is None and "another cache" in reason

    def test_a_missing_layer_on_the_stripe_root_is_named_by_full_path(self, tmp_path):
        out, roots = _cache(tmp_path, striped=True)
        gone = layer_shard_path(stripe_dirs(out, roots)[1], 1)
        os.remove(gone)
        index, reason = _inspect(out, roots)
        assert index is None and gone in reason

    def test_a_missing_local_layer_keeps_the_short_name(self, tmp_path):
        out, _ = _cache(tmp_path, striped=False)
        os.remove(layer_shard_path(out, 2))
        _, reason = _inspect(out, ())
        assert reason == "cached shard 'layer_002.safetensors' is missing"

    @pytest.mark.skipif(os.name != "nt", reason="Windows path spelling")
    def test_a_windows_respelling_of_the_same_root_does_not_invalidate(self, tmp_path):
        """Review Focus 3: case and a trailing separator must not cost a 4-minute re-shard."""
        out, roots = _cache(tmp_path, striped=True)
        index, reason = _inspect(out, (roots[0].upper() + "\\",))
        assert index is not None, reason


def test_a_reused_index_is_built_from_the_requested_roots_not_its_own_spelling(tmp_path):
    """Security LOW-5: the roots compare after normpath, which collapses `x/..` lexically while
    the kernel resolves it physically. A reused index must carry the roots the caller
    validated, so every later path is built from those."""
    out, roots = _cache(tmp_path, striped=True)
    path = os.path.join(out, "index.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["stripe_roots"] = [os.path.join(roots[0], "not-there", "..")]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    index, reason = _inspect(out, roots)
    assert index is not None, reason
    assert index.stripe_roots == roots
