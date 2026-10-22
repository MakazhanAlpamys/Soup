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


def test_content_reshard_quarantines_index_before_layer_writes(layout, monkeypatch):
    import soup_cli.utils.layer_shard as layer_shard_mod

    src, out, _ = layout
    shard_checkpoint(src, out, dtype="float32")
    real_save = layer_shard_mod._atomic_save
    layer_writes = 0

    def _stop_after_two_layers(blob, path):
        nonlocal layer_writes
        if os.path.basename(path).startswith("layer_"):
            layer_writes += 1
            if layer_writes > 2:
                raise RuntimeError("interrupted")
        return real_save(blob, path)

    with monkeypatch.context() as patch:
        patch.setattr(layer_shard_mod, "_atomic_save", _stop_after_two_layers)
        with pytest.raises(RuntimeError, match="interrupted"):
            shard_checkpoint(src, out, dtype="bfloat16")

    assert not os.path.exists(os.path.join(out, "index.json"))
    assert os.path.exists(os.path.join(out, "index.json.resharding"))

    said = []
    repaired = shard_checkpoint(src, out, dtype="float32", notify=said.append)

    assert repaired.dtype == "float32"
    assert any("previous re-shard did not finish" in line for line in said), said
    assert not os.path.exists(os.path.join(out, "index.json.resharding"))
    for path in layer_paths(out, repaired):
        tensors = load_file(path)
        assert all(tensor.dtype == torch.float32 for tensor in tensors.values())


def test_index_write_interrupt_leaves_a_miss_for_the_next_run(layout, monkeypatch):
    import soup_cli.utils.layer_shard as layer_shard_mod

    src, out, _ = layout
    shard_checkpoint(src, out, dtype="float32")

    def _stop_index(*_args, **_kwargs):
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(layer_shard_mod, "_atomic_write_index", _stop_index)
        with pytest.raises(RuntimeError, match="interrupted"):
            shard_checkpoint(src, out, dtype="bfloat16")

    assert not os.path.exists(os.path.join(out, "index.json"))
    assert os.path.exists(os.path.join(out, "index.json.resharding"))

    repaired = shard_checkpoint(src, out, dtype="float32")
    assert repaired.dtype == "float32"
    assert not os.path.exists(os.path.join(out, "index.json.resharding"))


def test_cache_hit_reports_unindexed_layer_and_temp_files(layout):
    src, out, stripe = layout
    index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    orphan_layer = layer_shard_path(out, 999)
    orphan_temp = os.path.join(_folder(out, stripe), ".soup.leftover.tmp")
    save_file({"orphan.weight": torch.zeros(1)}, orphan_layer)
    with open(orphan_temp, "w", encoding="utf-8") as handle:
        handle.write("incomplete")

    said = []
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,), notify=said.append)

    assert index.dtype == "float32"
    assert any(orphan_layer in line for line in said), said
    assert any(orphan_temp in line for line in said), said
    assert os.path.exists(orphan_layer)
    assert os.path.exists(orphan_temp)


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


def test_a_full_stripe_drive_names_the_root_and_the_variable(layout, monkeypatch):
    """Final review Minor 8: ENOSPC on the second drive mid-shard was a bare OSError naming
    neither the stripe root nor the variable. The old index stays the commit point."""
    import soup_cli.utils.layer_shard as layer_shard_mod
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.stripe_roots import STRIPE_DIRS_ENV

    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32")
    folder = _folder(out, stripe)
    real_save = layer_shard_mod._atomic_save

    def full_drive(blob, path):
        if os.path.normcase(os.path.dirname(path)) == os.path.normcase(folder):
            raise OSError(28, "No space left on device")
        return real_save(blob, path)

    monkeypatch.setattr(layer_shard_mod, "_atomic_save", full_drive)
    with pytest.raises(OSError) as caught:
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    message = str(caught.value)
    assert caught.value.errno == 28, message
    assert STRIPE_DIRS_ENV in message and os.path.realpath(stripe) in message, message
    assert "No space left" in message, message
    assert read_shard_index(out).layer_roots == ()


def test_content_quarantine_covers_quant_and_source_fingerprint(layout):
    import soup_cli.utils.layer_shard as layer_shard_mod

    src, out, _ = layout
    index = shard_checkpoint(src, out, dtype="float32")
    common = {
        "out_dir": out,
        "dtype": index.dtype,
        "double_quant": index.double_quant,
        "quant_device": index.quant_device,
        "external_mode": index.external_mode,
    }

    assert layer_shard_mod._content_change_requires_index_quarantine(
        **common, fingerprint=index.source_fingerprint, quant="nf4"
    )
    assert layer_shard_mod._content_change_requires_index_quarantine(
        **common, fingerprint="different-source", quant=index.quant
    )


def test_previous_index_reader_falls_back_to_resharding_marker(layout):
    import soup_cli.utils.layer_shard as layer_shard_mod

    src, out, _ = layout
    original = shard_checkpoint(src, out, dtype="float32")
    layer_shard_mod._quarantine_shard_index(out)

    previous = layer_shard_mod._read_previous_shard_index(out)

    assert previous is not None
    assert previous.dtype == original.dtype
    assert previous.source_fingerprint == original.source_fingerprint


def test_successful_reshard_scans_for_unindexed_files_after_commit(layout, monkeypatch):
    import soup_cli.utils.layer_shard as layer_shard_mod
    from soup_cli.utils.layer_shard import read_shard_index

    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32")
    observed = []
    real_notify = layer_shard_mod._notify_unindexed_cache_files

    def _observe(directory, index, notify):
        committed = read_shard_index(directory)
        observed.append((committed.layer_roots, index.layer_roots))
        return real_notify(directory, index, notify)

    monkeypatch.setattr(layer_shard_mod, "_notify_unindexed_cache_files", _observe)
    index = shard_checkpoint(
        src, out, dtype="float32", stripe_roots=(stripe,), notify=lambda _line: None
    )

    assert observed == [(index.layer_roots, index.layer_roots)]


@pytest.mark.parametrize("stop_at", ["layers", "index"])
def test_content_reshard_interrupt_is_a_miss_with_striping(layout, monkeypatch, stop_at):
    import soup_cli.utils.layer_shard as layer_shard_mod

    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
    real_save = layer_shard_mod._atomic_save
    layer_writes = 0

    def _stop_after_two_layers(blob, path):
        nonlocal layer_writes
        if os.path.basename(path).startswith("layer_"):
            layer_writes += 1
            if layer_writes > 2:
                raise RuntimeError("interrupted")
        return real_save(blob, path)

    def _stop_index(*_args, **_kwargs):
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        if stop_at == "layers":
            patch.setattr(layer_shard_mod, "_atomic_save", _stop_after_two_layers)
        else:
            patch.setattr(layer_shard_mod, "_atomic_write_index", _stop_index)
        with pytest.raises(RuntimeError, match="interrupted"):
            shard_checkpoint(src, out, dtype="bfloat16", stripe_roots=(stripe,))

    said = []
    repaired = shard_checkpoint(
        src, out, dtype="float32", stripe_roots=(stripe,), notify=said.append
    )
    assert any("previous re-shard did not finish" in line for line in said), said
    for path in layer_paths(out, repaired):
        tensors = load_file(path)
        assert all(tensor.dtype == torch.float32 for tensor in tensors.values())


def test_three_root_layout_reports_files_left_on_a_kept_stripe(tmp_path):
    src = _weights(tmp_path, n_layers=6)
    out = str(tmp_path / "cache" / "model")
    first = tmp_path / "stripe-one"
    second = tmp_path / "stripe-two"
    first.mkdir()
    second.mkdir()
    first_root = str(first)
    second_root = str(second)

    shard_checkpoint(src, out, dtype="float32", stripe_roots=(first_root,))
    first_folder = _folder(out, first_root)
    leftovers = [layer_shard_path(first_folder, idx) for idx in (3, 5)]
    assert all(os.path.exists(path) for path in leftovers)

    said = []
    shard_checkpoint(
        src,
        out,
        dtype="float32",
        stripe_roots=(first_root, second_root),
        notify=said.append,
    )

    assert all(os.path.exists(path) for path in leftovers)
    for path in leftovers:
        assert any(path in line and "bytes" in line for line in said), (path, said)


def test_interrupt_before_deferred_delete_is_reported_on_next_hit(layout, monkeypatch):
    import soup_cli.utils.layer_shard as layer_shard_mod

    src, out, stripe = layout
    shard_checkpoint(src, out, dtype="float32")
    stale = layer_shard_path(out, 1)
    real_remove = layer_shard_mod.os.remove

    def _interrupt(path, *args, **kwargs):
        if os.path.normcase(os.path.normpath(str(path))) == os.path.normcase(
            os.path.normpath(stale)
        ):
            raise KeyboardInterrupt
        return real_remove(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(layer_shard_mod.os, "remove", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))

    said = []
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,), notify=said.append)
    assert any(stale in line and "Unindexed" in line for line in said), said


def test_notice_escapes_rich_markup_in_cache_paths(tmp_path):
    from io import StringIO

    from rich.console import Console

    src = _weights(tmp_path)
    out = str(tmp_path / "cache" / "model")
    stripe = tmp_path / "stripe-[bold]"
    stripe.mkdir()
    stripe_root = str(stripe)
    shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe_root,))
    orphan = os.path.join(_folder(out, stripe_root), ".soup.leftover.tmp")
    with open(orphan, "w", encoding="utf-8") as handle:
        handle.write("incomplete")

    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=1000)
    shard_checkpoint(
        src, out, dtype="float32", stripe_roots=(stripe_root,), notify=console.print
    )

    rendered = stream.getvalue()
    assert "stripe-[bold]" in rendered
    assert orphan in rendered


def test_completed_marker_delete_failure_is_non_fatal(layout, monkeypatch):
    import soup_cli.utils.layer_shard as layer_shard_mod
    from soup_cli.utils.layer_shard import read_shard_index

    src, out, _ = layout
    shard_checkpoint(src, out, dtype="float32")
    marker = os.path.join(out, "index.json.resharding")
    real_remove = layer_shard_mod.os.remove

    def _deny_marker(path, *args, **kwargs):
        if os.path.normcase(os.path.normpath(str(path))) == os.path.normcase(
            os.path.normpath(marker)
        ):
            raise PermissionError("in use")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(layer_shard_mod.os, "remove", _deny_marker)
    said = []
    index = shard_checkpoint(src, out, dtype="bfloat16", notify=said.append)

    assert index.dtype == "bfloat16"
    assert read_shard_index(out).dtype == "bfloat16"
    assert any(marker in line and "Could not delete" in line for line in said), said
