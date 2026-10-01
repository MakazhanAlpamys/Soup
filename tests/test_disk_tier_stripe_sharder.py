"""R4 — the sharder places layer i on root i mod N, and keeps the cache honest about it."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from soup_cli.utils.layer_shard import (  # noqa: E402
    layer_paths,
    layer_shard_path,
    shard_checkpoint,
    stripe_dirs,
)
from soup_cli.utils.stripe_roots import StripeRootError, primary_cache_identity  # noqa: E402


def _weights(tmp_path, n_layers=3):
    torch.manual_seed(0)
    state = {}
    for idx in range(n_layers):
        pre = f"model.layers.{idx}."
        state[pre + "self_attn.q_proj.weight"] = torch.randn(8, 8)
        state[pre + "mlp.down_proj.weight"] = torch.randn(8, 16)
        state[pre + "input_layernorm.weight"] = torch.randn(8)
    state["model.embed_tokens.weight"] = torch.randn(32, 8)
    state["model.norm.weight"] = torch.randn(8)
    src = tmp_path / "weights"
    src.mkdir()
    save_file(state, str(src / "model.safetensors"))
    return str(src)


def _folder(out, stripe):
    """This primary cache's own folder inside the stripe root (slug + hash, fix wave B)."""
    return stripe_dirs(out, (os.path.realpath(stripe),))[1]


@pytest.fixture
def layout(tmp_path):
    stripe = tmp_path / "stripe"
    stripe.mkdir()
    return _weights(tmp_path), str(tmp_path / "cache" / "model"), str(stripe)


def test_odd_layers_land_on_the_stripe_root_and_even_ones_stay(layout):
    src, out, stripe = layout
    index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    assert index.layer_roots == (0, 1, 0)
    assert index.stripe_roots == (os.path.realpath(stripe),)
    assert os.path.exists(layer_shard_path(out, 0))
    assert os.path.exists(layer_shard_path(_folder(out, stripe), 1))
    assert not os.path.exists(layer_shard_path(out, 1))
    assert os.path.exists(layer_shard_path(out, 2))


def test_every_layer_is_byte_identical_to_the_single_root_shard(layout, tmp_path):
    src, out, stripe = layout
    single = shard_checkpoint(src, str(tmp_path / "cache" / "single"), dtype="float32")
    striped = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    for mine, theirs in zip(
        layer_paths(out, striped), layer_paths(str(tmp_path / "cache" / "single"), single)
    ):
        a, b = load_file(mine), load_file(theirs)
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), (mine, key)


def test_the_marker_describes_this_cache(layout):
    import json

    src, out, stripe = layout
    index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    with open(os.path.join(_folder(out, stripe), "stripe.json"), encoding="utf-8") as handle:
        assert json.load(handle) == {
            "source_fingerprint": index.source_fingerprint, "position": 1, "n_roots": 2,
            "primary_cache": primary_cache_identity(out),
        }


def test_an_unchanged_layout_is_reused_without_rewriting(layout):
    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    moved = layer_shard_path(_folder(out, stripe), 1)
    before = os.stat(moved).st_mtime_ns
    said = []
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,), notify=said.append)
    assert said == [] and os.stat(moved).st_mtime_ns == before


def test_turning_striping_on_reshards_and_drops_the_stale_local_copy(layout):
    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32")
    assert os.path.exists(layer_shard_path(out, 1))
    said = []
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,), notify=said.append)
    assert any("stripe roots changed" in line for line in said), said
    assert not os.path.exists(layer_shard_path(out, 1))


def test_turning_striping_off_names_the_abandoned_folder(layout):
    """Review Focus 4: ~18 GB left on the second drive must not go unmentioned."""
    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    said = []
    index = shard_checkpoint(src, out, dtype="float32", notify=said.append)
    assert index.layer_roots == () and os.path.exists(layer_shard_path(out, 1))
    folder = _folder(out, stripe)
    assert any(folder in line and "Delete it" in line for line in said), said


def test_a_stripe_root_that_does_not_exist_is_refused_before_any_write(layout, tmp_path):
    src, out, _ = layout
    with pytest.raises(StripeRootError, match="existing directory"):
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(str(tmp_path / "gone"),))
    assert not os.path.exists(layer_shard_path(out, 0))


def test_an_interrupted_reshard_leaves_the_old_index_consistent(layout, monkeypatch):
    """Fix round 1, Important finding: the index is the cache's commit point — every file it
    names must exist at every instant. Deleting the stale root-0 copy of a moved layer must
    wait until the new index (which no longer names it) is safely on disk, or a crash between
    the delete and the index write leaves the still-current (old) index naming a file that is
    no longer there."""
    import soup_cli.utils.layer_shard as layer_shard_mod
    from soup_cli.utils.layer_shard import read_shard_index

    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("interrupted")

    with monkeypatch.context() as m:
        m.setattr(layer_shard_mod, "_atomic_write_index", _boom)
        with pytest.raises(RuntimeError, match="interrupted"):
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))

    assert os.path.exists(layer_shard_path(out, 1))
    assert read_shard_index(out).layer_roots == ()

    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    assert not os.path.exists(layer_shard_path(out, 1))


def test_a_stale_copy_that_will_not_delete_still_returns_the_committed_index(layout, monkeypatch):
    """Fix round 2, Important finding: the deferred delete runs AFTER the index is already
    committed, so a failure to delete (a leftover handle holding the file open, most plausibly
    on Windows) must not turn a genuinely successful, correctly-committed shard into a
    caller-visible exception. It should warn and move on instead."""
    import soup_cli.utils.layer_shard as layer_shard_mod
    from soup_cli.utils.layer_shard import read_shard_index

    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32")
    stale_path = layer_shard_path(out, 1)
    real_remove = layer_shard_mod.os.remove

    def _flaky_remove(path, *args, **kwargs):
        if os.path.normcase(os.path.normpath(str(path))) == os.path.normcase(
            os.path.normpath(stale_path)
        ):
            raise PermissionError("in use")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(layer_shard_mod.os, "remove", _flaky_remove)

    said = []
    index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,), notify=said.append)

    assert index.layer_roots == (0, 1, 0)
    assert any(stale_path in line and "Could not delete" in line for line in said), said
    assert read_shard_index(out).layer_roots == (0, 1, 0)
