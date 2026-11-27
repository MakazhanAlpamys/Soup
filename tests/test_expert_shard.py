"""Expert shard storage (``utils/expert_shard.py``).

The storage layer is shared by the expert runtimes. The runtime keeps importing
its names from ``utils/expert_stream.py``; the shard format is version 5 with the
index fields this tree has always written. A cache written before the storage
layer was split out must load unchanged, and an index written with the extra
fields must load too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SHARD_NAMES = (
    "DOWN",
    "EXPERT_LAYOUT_PER_EXPERT",
    "EXPERT_PARTS",
    "GATE_UP",
    "ExpertShardInfo",
    "ExpertShardReader",
    "aligned_staging",
    "expert_key",
    "expert_shard_name",
    "write_expert_shard",
    "_checked_shape",
    "_expected_tensors",
    "_stored_torch_dtype",
    "_MAX_EXPERTS",
    "_MAX_EXPERT_BYTES",
)


def _pair(offset=0.0):
    import torch

    return (
        torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset,
        torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset,
    )


def _written(tmp_path, count=3):
    from soup_cli.utils.expert_shard import write_expert_shard

    path = str(tmp_path / "experts_000.safetensors")
    info = write_expert_shard(
        path, [_pair(100.0 * expert) for expert in range(count)], quantise=False
    )
    return path, info


def _reader(path, info, count=3, **kwargs):
    from soup_cli.utils.expert_shard import ExpertShardReader

    return ExpertShardReader(
        path,
        count=count,
        shapes=kwargs.pop("shapes", info.shapes),
        quant_specs=kwargs.pop("quant_specs", info.quant_specs),
        dtype="float32",
        direct=False,
        **kwargs,
    )


# --------------------------------------------------------------------------
# the runtime's names still resolve, and are the storage layer's objects
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", SHARD_NAMES)
def test_expert_stream_re_exports_the_storage_layer(name):
    from soup_cli.utils import expert_shard, expert_stream

    assert getattr(expert_stream, name) is getattr(expert_shard, name)


@pytest.mark.parametrize("name", ["_MAX_EXPERTS", "_MAX_EXPERT_BYTES"])
def test_a_limit_set_on_expert_stream_reaches_the_storage_layer(monkeypatch, name):
    from soup_cli.utils import expert_shard, expert_stream

    original = getattr(expert_shard, name)
    with monkeypatch.context() as patch:
        patch.setattr(expert_stream, name, 7)
        assert getattr(expert_shard, name) == 7
    assert getattr(expert_shard, name) == original
    assert getattr(expert_stream, name) == original


def test_the_format_version_is_unchanged_and_torch_stays_lazy():
    import subprocess
    import sys

    from soup_cli.utils import layer_shard

    assert layer_shard._SHARD_FORMAT_VERSION == 5
    import soup_cli

    src = str(Path(soup_cli.__file__).resolve().parent.parent)
    code = (
        f"import sys; sys.path.insert(0, {src!r});"
        "import soup_cli.utils.expert_shard, soup_cli.utils.expert_stream;"
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------
# a cache written by this tree's writer loads through the shared code
# --------------------------------------------------------------------------
def _checkpoint(root: Path, count: int = 4):
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    tensors = {"model.norm.weight": torch.ones(4)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = torch.eye(4)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.arange(count).float()
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = torch.ones(4, 3)
        for expert in range(count):
            for projection, shape in (
                ("gate_proj", (3, 4)),
                ("up_proj", (3, 4)),
                ("down_proj", (4, 3)),
            ):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = (
                    torch.arange(12).reshape(shape).float()
                    + 100 * expert
                    + 1000 * layer
                    + (10 if projection == "up_proj" else 0)
                )
    save_file(tensors, str(root / "model.safetensors"))
    return tensors


def _main_cache(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    source, out = tmp_path / "source", tmp_path / "cache"
    expected = _checkpoint(source)
    index = shard_checkpoint(
        str(source),
        str(out),
        dtype="float32",
        arch="deepseek_v3",
        num_hidden_layers=2,
        expert_layout="per_expert",
    )
    return expected, out, index


def _assert_cache_reads_back(expected, out, index):
    import torch

    from soup_cli.utils.expert_shard import aligned_staging
    from soup_cli.utils.layer_shard import expert_paths, read_shard_index

    reread = read_shard_index(str(out))
    assert reread == index
    assert reread.expert_layers == (0, 1) and reread.expert_count == 4
    assert reread.format_version == 5
    for layer, path in expert_paths(str(out), reread).items():
        from soup_cli.utils.expert_shard import ExpertShardReader

        reader = ExpertShardReader(
            path,
            count=reread.expert_count,
            shapes=reread.expert_shapes,
            quant_specs=reread.expert_quant_specs,
            dtype=reread.dtype,
            direct=False,
        )
        owner, staging = aligned_staging(reader.max_span)
        try:
            for expert in (3, 0, 2, 1):
                host = reader.read_host(expert, staging)
                prefix = f"model.layers.{layer}.mlp.experts.{expert}."
                want = torch.cat(
                    (expected[prefix + "gate_proj.weight"], expected[prefix + "up_proj.weight"]),
                    0,
                )
                assert torch.equal(host[f"{expert}.gate_up"], want)
                assert torch.equal(host[f"{expert}.down"], expected[prefix + "down_proj.weight"])
        finally:
            reader.close()
        assert owner.numel() >= reader.max_span


def test_an_index_written_by_this_trees_writer_loads_with_identical_tensors(tmp_path):
    expected, out, index = _main_cache(tmp_path)
    _assert_cache_reads_back(expected, out, index)


def test_this_trees_writer_adds_no_extra_index_fields(tmp_path):
    _, out, _ = _main_cache(tmp_path)
    payload = json.loads((out / "index.json").read_text())
    assert payload["format_version"] == 5
    assert payload["expert_layout"] == "per_expert"
    assert "fp32_router_weights" in payload
    for field in ("expert_layout_version", "fp32_keys", "stacked_experts"):
        assert field not in payload, field


def test_an_extra_field_index_loads_through_the_same_reader(tmp_path):
    from soup_cli.utils.layer_shard import read_shard_index

    expected, out, index = _main_cache(tmp_path)
    path = out / "index.json"
    payload = json.loads(path.read_text())
    payload["expert_layout_version"] = 1
    payload["fp32_keys"] = ["mlp.gate.e_score_correction_bias"]
    payload["stacked_experts"] = True
    path.write_text(json.dumps(payload))
    assert read_shard_index(str(out)) == index
    _assert_cache_reads_back(expected, out, index)


@pytest.mark.parametrize("version", [2, 0, True, 1.0, "1"])
def test_an_unknown_expert_layout_version_is_refused(tmp_path, version):
    from soup_cli.utils.layer_shard import read_shard_index

    _, out, _ = _main_cache(tmp_path)
    path = out / "index.json"
    payload = json.loads(path.read_text())
    payload["expert_layout_version"] = version
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported expert layout version"):
        read_shard_index(str(out))


# --------------------------------------------------------------------------
# reader hardening
# --------------------------------------------------------------------------
def test_the_reader_round_trips_every_expert(tmp_path):
    import torch

    from soup_cli.utils.expert_shard import aligned_staging

    path, info = _written(tmp_path)
    reader = _reader(path, info)
    owner, staging = aligned_staging(reader.max_span)
    try:
        for expert in (2, 0, 1):
            host = reader.read_host(expert, staging)
            gate_up, down = _pair(100.0 * expert)
            assert torch.equal(host[f"{expert}.gate_up"], gate_up)
            assert torch.equal(host[f"{expert}.down"], down)
    finally:
        reader.close()
    assert owner.numel() > 0


@pytest.mark.parametrize("expert", [True, 1.5, "0", None])
def test_read_host_refuses_a_non_integer_expert(tmp_path, expert):
    from soup_cli.utils.expert_shard import aligned_staging

    path, info = _written(tmp_path)
    reader = _reader(path, info)
    owner, staging = aligned_staging(reader.max_span)
    try:
        with pytest.raises(IndexError, match=r"outside \[0, 3\)"):
            reader.read_host(expert, staging)
    finally:
        reader.close()
    assert owner.numel() > 0


def _staging_cases():
    import torch

    from soup_cli.utils.expert_shard import aligned_staging

    def two_dimensional(size):
        owner, view = aligned_staging(size)
        return owner, view.reshape(1, -1)

    def wrong_dtype(size):
        owner, _ = aligned_staging(size)
        return owner, torch.zeros(size, dtype=torch.float32)

    def misaligned(size):
        owner, _ = aligned_staging(size + 8)
        pad = (-owner.data_ptr()) % 4096
        return owner, owner[pad + 1 : pad + 1 + size]

    def strided(size):
        owner = torch.zeros(2 * size + 4096, dtype=torch.uint8)
        return owner, owner[::2][:size]

    return {
        "two-dimensional": two_dimensional,
        "wrong-dtype": wrong_dtype,
        "misaligned": misaligned,
        "strided": strided,
    }


@pytest.mark.parametrize("case", ["two-dimensional", "wrong-dtype", "misaligned", "strided"])
def test_read_host_refuses_staging_that_is_not_sector_aligned_cpu_bytes(tmp_path, case):
    path, info = _written(tmp_path)
    reader = _reader(path, info)
    owner, staging = _staging_cases()[case](reader.max_span)
    try:
        with pytest.raises(
            ValueError, match="staging must be contiguous sector-aligned CPU uint8 memory"
        ):
            reader.read_host(0, staging)
    finally:
        reader.close()
    assert owner is not None


def test_read_host_refuses_staging_smaller_than_one_expert(tmp_path):
    from soup_cli.utils.expert_shard import aligned_staging

    path, info = _written(tmp_path)
    reader = _reader(path, info)
    owner, staging = aligned_staging(reader.max_span - 1)
    try:
        with pytest.raises(ValueError, match="staging holds"):
            reader.read_host(0, staging)
    finally:
        reader.close()
    assert owner.numel() > 0


def test_reader_refuses_a_file_with_a_tensor_the_index_does_not_name(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    path, info = _written(tmp_path)
    blob = load_file(path)
    blob["stray"] = torch.zeros(2)
    save_file(blob, path)
    with pytest.raises(ValueError, match="has unexpected tensors"):
        _reader(path, info)


def test_reader_refuses_nf4_specs_for_only_one_part(tmp_path):
    from soup_cli.utils.layer_shard import NF4WeightSpec

    path, info = _written(tmp_path)
    spec = NF4WeightSpec(
        shape=(6, 4), dtype="float32", blocksize=64, quant_type="nf4", nested=False,
        nested_blocksize=0,
    )
    with pytest.raises(ValueError, match="NF4 specs must name both gate_up and down, or neither"):
        _reader(path, info, quant_specs={"gate_up": spec})


def test_reader_refuses_shapes_that_are_not_one_expert_mlp(tmp_path):
    path, info = _written(tmp_path)
    shapes = {"gate_up": (6, 4), "down": (3, 4)}
    with pytest.raises(ValueError, match="gate_up and down shapes are not one expert MLP"):
        _reader(path, info, shapes=shapes)


@pytest.mark.parametrize("count", [True, 0, 4097, 1.5])
def test_reader_refuses_a_bad_expert_count(tmp_path, count):
    path, info = _written(tmp_path)
    with pytest.raises(ValueError, match="expert count must be an int in"):
        _reader(path, info, count=count)


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "buffered"])
@pytest.mark.parametrize("failure", ["header", "after-header"])
def test_a_failed_constructor_leaves_the_file_free_to_replace_or_delete(
    tmp_path, monkeypatch, direct, failure
):
    """The handle must be closed by the constructor itself, with the exception retained.

    ``excinfo`` keeps the half-built reader alive, so a finaliser cannot be what closes
    it. On Windows an open handle makes the replace or delete below fail.
    """
    import os

    from soup_cli.utils import expert_shard, safetensors_reader
    from soup_cli.utils.expert_shard import ExpertShardReader

    path, info = _written(tmp_path)
    handles = []
    real_open = ExpertShardReader._open

    def recording_open(self):
        handle = real_open(self)
        if not handles or handles[-1] is not handle:
            handles.append(handle)
        return handle

    monkeypatch.setattr(ExpertShardReader, "_open", recording_open)
    if failure == "header":

        def boom(_path):
            raise ValueError("injected failure")

        monkeypatch.setattr(safetensors_reader, "read_header_with_identity", boom)
    else:

        def boom(*args, **kwargs):
            raise ValueError("injected failure")

        monkeypatch.setattr(expert_shard, "_expected_tensors", boom)
    with pytest.raises(ValueError, match="injected failure") as excinfo:
        ExpertShardReader(
            path, count=3, shapes=info.shapes, quant_specs={}, dtype="float32", direct=direct
        )
    assert handles and all(handle.closed for handle in handles)
    assert excinfo.value is not None
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"x")
    os.replace(replacement, path)
    os.unlink(path)
    assert not os.path.exists(path)


@pytest.mark.parametrize("bad", [(True, 4), (6.0, 4), (6, 4, 1), (0, 4), (6,)])
def test_shapes_must_be_two_positive_integer_dimensions(tmp_path, bad):
    path, info = _written(tmp_path)
    with pytest.raises(ValueError, match="expert gate_up shape must"):
        _reader(path, info, shapes={"gate_up": bad, "down": (4, 3)})


# --------------------------------------------------------------------------
# writer hardening
# --------------------------------------------------------------------------
def test_writer_refuses_an_expert_that_is_not_a_gate_up_down_pair(tmp_path):
    import torch

    from soup_cli.utils.expert_shard import write_expert_shard

    gate_up, down = _pair()
    with pytest.raises(ValueError, match="expert 1 must supply exactly gate_up and down"):
        write_expert_shard(
            str(tmp_path / "x.safetensors"),
            [(gate_up, down), (gate_up, down, torch.zeros(1))],
            quantise=False,
        )
    with pytest.raises(ValueError, match="expert 0 must supply exactly gate_up and down"):
        write_expert_shard(str(tmp_path / "y.safetensors"), [(gate_up,)], quantise=False)


def test_writer_refuses_a_layer_with_no_expert(tmp_path):
    from soup_cli.utils.expert_shard import write_expert_shard

    with pytest.raises(ValueError, match="a MoE layer needs at least one expert"):
        write_expert_shard(str(tmp_path / "x.safetensors"), [], quantise=False)


def test_writer_refuses_matrices_that_are_not_one_mlp(tmp_path):
    import torch

    from soup_cli.utils.expert_shard import write_expert_shard

    with pytest.raises(ValueError, match="are not one expert MLP"):
        write_expert_shard(
            str(tmp_path / "x.safetensors"),
            [(torch.zeros(6, 4), torch.zeros(3, 4))],
            quantise=False,
        )
    assert not (tmp_path / "x.safetensors").exists()


def test_writer_refuses_a_layer_over_the_expert_limit(tmp_path, monkeypatch):
    from soup_cli.utils import expert_shard

    monkeypatch.setattr(expert_shard, "_MAX_EXPERTS", 2)
    with pytest.raises(ValueError, match="a MoE layer may hold at most 2 experts"):
        expert_shard.write_expert_shard(
            str(tmp_path / "x.safetensors"), [_pair(), _pair(), _pair()], quantise=False
        )


def _fake_quantiser(*, blocksize=64, drop=(), extra=None, packed_rows=None, code_shift=0.0):
    import torch

    from soup_cli.utils.layer_shard import ABSMAX_SUFFIX, NF4WeightSpec

    def fake(tensor, *, double_quant, device):
        weights = tensor.numel()
        rows = (weights + 1) // 2 if packed_rows is None else packed_rows
        sidecars = {
            "": torch.zeros((rows, 1), dtype=torch.uint8),
            ABSMAX_SUFFIX: torch.ones(-(-weights // 64)),
        }
        for suffix in drop:
            del sidecars[suffix]
        if extra:
            sidecars[extra] = torch.zeros(1)
        spec = NF4WeightSpec(
            shape=tuple(tensor.shape), dtype="float32", blocksize=blocksize,
            quant_type="nf4", nested=False, nested_blocksize=0,
        )
        code = torch.arange(16, dtype=torch.float32) + code_shift
        return sidecars, spec, code, None

    return fake


def _write_nf4(tmp_path, monkeypatch, fake, count=1):
    from soup_cli.utils import layer_shard
    from soup_cli.utils.expert_shard import write_expert_shard

    monkeypatch.setattr(layer_shard, "_quantize_nf4", fake)
    return write_expert_shard(
        str(tmp_path / "x.safetensors"), [_pair(float(e)) for e in range(count)], quantise=True
    )


def test_a_valid_nf4_codec_output_is_written_and_read_back(tmp_path, monkeypatch):
    from soup_cli.utils.expert_shard import aligned_staging

    info = _write_nf4(tmp_path, monkeypatch, _fake_quantiser())
    assert set(info.quant_specs) == {"gate_up", "down"}
    reader = _reader(str(tmp_path / "x.safetensors"), info, count=1)
    owner, staging = aligned_staging(reader.max_span)
    try:
        host = reader.read_host(0, staging)
        assert set(host) == {"0.gate_up", "0.gate_up::absmax", "0.down", "0.down::absmax"}
    finally:
        reader.close()
    assert owner.numel() > 0


@pytest.mark.parametrize(
    "fake,match",
    [
        (_fake_quantiser(blocksize=32), "NF4 codec returned unexpected blocksize or format"),
        (_fake_quantiser(extra="::surprise"), "NF4 codec returned unexpected sidecars"),
        (_fake_quantiser(drop=("::absmax",)), "NF4 codec returned unexpected sidecars"),
        (_fake_quantiser(packed_rows=3), "NF4 codec returned unexpected shape or dtype"),
    ],
    ids=["blocksize", "extra-sidecar", "missing-sidecar", "packed-size"],
)
def test_writer_refuses_a_codec_that_breaks_its_contract(tmp_path, monkeypatch, fake, match):
    with pytest.raises(ValueError, match=match):
        _write_nf4(tmp_path, monkeypatch, fake)
    assert not (tmp_path / "x.safetensors").exists()


def test_writer_refuses_nf4_code_tables_that_differ_between_experts(tmp_path, monkeypatch):
    shifts = iter(range(100))

    def drifting(tensor, *, double_quant, device):
        return _fake_quantiser(code_shift=float(next(shifts)))(
            tensor, double_quant=double_quant, device=device
        )

    with pytest.raises(ValueError, match="the NF4 code table differs at"):
        _write_nf4(tmp_path, monkeypatch, drifting, count=2)
    assert not (tmp_path / "x.safetensors").exists()


# --------------------------------------------------------------------------
# untrusted index and header text: bounded, no silent duplicate keys
# --------------------------------------------------------------------------
def test_an_oversized_shard_index_is_refused_before_parsing(tmp_path, monkeypatch):
    from soup_cli.utils import layer_shard

    _, out, _ = _main_cache(tmp_path)
    size = (out / "index.json").stat().st_size
    monkeypatch.setattr(layer_shard, "_MAX_INDEX_BYTES", size - 1)
    with pytest.raises(ValueError, match=r"is larger than \d+ bytes; refusing to parse it"):
        layer_shard.read_shard_index(str(out))
    monkeypatch.setattr(layer_shard, "_MAX_INDEX_BYTES", size)
    assert layer_shard.read_shard_index(str(out)).expert_count == 4


def test_a_duplicate_key_in_the_shard_index_is_refused(tmp_path):
    from soup_cli.utils.layer_shard import read_shard_index

    _, out, _ = _main_cache(tmp_path)
    path = out / "index.json"
    text = path.read_text()
    assert text.rstrip().endswith("}")
    path.write_text(text.rstrip()[:-1] + ', "expert_count": 1}')
    with pytest.raises(ValueError, match="duplicate key 'expert_count'"):
        read_shard_index(str(out))


def test_a_shard_index_that_is_not_an_object_is_refused(tmp_path):
    from soup_cli.utils.layer_shard import read_shard_index

    _, out, _ = _main_cache(tmp_path)
    (out / "index.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="is not a JSON object"):
        read_shard_index(str(out))


def test_a_duplicate_tensor_name_in_a_safetensors_header_is_refused(tmp_path):
    import struct

    from soup_cli.utils.safetensors_reader import read_header_with_identity

    body = (
        '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
        '"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}'
    ).encode()
    path = tmp_path / "dup.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + bytes(8))
    with pytest.raises(ValueError, match="duplicate key 'a'"):
        read_header_with_identity(str(path))


# --------------------------------------------------------------------------
# containment: the RESOLVED file must sit inside the store root
# --------------------------------------------------------------------------
@pytest.mark.requires_symlink
def test_a_symlinked_experts_file_pointing_outside_the_store_is_refused(tmp_path):
    import os

    from soup_cli.utils.layer_shard import expert_paths

    _, out, index = _main_cache(tmp_path)
    leaf = out / "experts_000.safetensors"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(leaf.read_bytes())
    leaf.unlink()
    os.symlink(outside, leaf)
    with pytest.raises(ValueError, match="experts_000.safetensors' resolves outside the store"):
        expert_paths(str(out), index)


@pytest.mark.requires_symlink
def test_a_symlinked_layer_file_pointing_outside_the_store_is_refused(tmp_path):
    import os

    from soup_cli.utils.layer_shard import layer_paths

    _, out, index = _main_cache(tmp_path)
    leaf = out / "layer_000.safetensors"
    outside = tmp_path / "outside_layer.safetensors"
    outside.write_bytes(leaf.read_bytes())
    leaf.unlink()
    os.symlink(outside, leaf)
    with pytest.raises(ValueError, match="layer_000.safetensors' resolves outside the store"):
        layer_paths(str(out), index)


@pytest.mark.requires_symlink
def test_a_symlink_that_stays_inside_the_store_is_accepted(tmp_path):
    import os

    from soup_cli.utils.layer_shard import expert_paths

    _, out, index = _main_cache(tmp_path)
    leaf = out / "experts_000.safetensors"
    moved = out / "moved.bin"
    leaf.rename(moved)
    os.symlink(moved, leaf)
    assert expert_paths(str(out), index)[0] == str(leaf)


@pytest.mark.parametrize(
    "name",
    [
        "..",
        ".",
        "",
        "../x.safetensors",
        "a/b.safetensors",
        "a\\b",
        "a\x00b",
        "a\x1fb",
        "a\x08",
        "a\x7f",
        "C:x",
        "C:\\x",
        "/abs",
        ".hidden",
        "a b",
        "a\n",
        "NUL",
        "con.safetensors",
        "COM1.bin",
    ],
)
def test_a_shard_file_name_that_is_not_bare_is_refused(tmp_path, name):
    from soup_cli.utils.layer_shard import contained_leaf

    with pytest.raises(ValueError, match="is not a bare file name"):
        contained_leaf(str(tmp_path), name)


def test_a_bare_name_inside_the_root_is_accepted(tmp_path):
    from soup_cli.utils.layer_shard import contained_leaf

    assert contained_leaf(str(tmp_path), "experts_000.safetensors") == str(
        tmp_path / "experts_000.safetensors"
    )


# --------------------------------------------------------------------------
# a cache written by the sharder BEFORE this storage layer existed
# --------------------------------------------------------------------------
FIXTURE = Path(__file__).parent / "fixtures" / "expert_shard_main_7b7697307"


def _fixture_source():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_expert_shard_main_7b7697307",
        Path(__file__).parent / "fixtures" / "make_expert_shard_main_7b7697307.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_tensors()


def _pre_port_readers(kind):
    from soup_cli.utils.expert_shard import ExpertShardReader
    from soup_cli.utils.layer_shard import expert_paths, read_shard_index

    root = FIXTURE / kind
    index = read_shard_index(str(root))
    readers = {
        layer: ExpertShardReader(
            path,
            count=index.expert_count,
            shapes=index.expert_shapes,
            quant_specs=index.expert_quant_specs,
            dtype=index.dtype,
            direct=False,
        )
        for layer, path in expert_paths(str(root), index).items()
    }
    return root, index, readers


def _expert_source(source, layer, expert):
    import torch

    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
    return (
        torch.cat((source[prefix + "gate_proj.weight"], source[prefix + "up_proj.weight"]), 0),
        source[prefix + "down_proj.weight"],
    )


def test_the_pre_port_fixture_has_the_index_this_tree_always_wrote():
    for kind in ("fp32", "nf4"):
        payload = json.loads((FIXTURE / kind / "index.json").read_text())
        assert payload["format_version"] == 5 and payload["expert_layout"] == "per_expert"
        assert "expert_layout_version" not in payload and "fp32_keys" not in payload


def test_a_float32_cache_written_before_the_split_reads_back_identically():
    import torch

    from soup_cli.utils.expert_shard import aligned_staging

    source = _fixture_source()
    _, index, readers = _pre_port_readers("fp32")
    assert index.expert_layers == (0, 1) and index.expert_count == 3
    assert not index.expert_quant_specs
    for layer, reader in readers.items():
        owner, staging = aligned_staging(reader.max_span)
        try:
            for expert in (2, 0, 1):
                host = reader.read_host(expert, staging)
                gate_up, down = _expert_source(source, layer, expert)
                assert torch.equal(host[f"{expert}.gate_up"], gate_up)
                assert torch.equal(host[f"{expert}.down"], down)
        finally:
            reader.close()
        assert owner.numel() > 0


def _assert_quant_state_equal(key, rebuilt, reference):
    """Compare a rebuilt NF4 ``QuantState`` with the stored state, one field at a time."""
    import torch

    assert rebuilt.blocksize == reference.blocksize, f"{key}: blocksize"
    assert rebuilt.quant_type == reference.quant_type, f"{key}: quant_type"
    assert rebuilt.dtype == reference.dtype, f"{key}: dtype"
    assert tuple(rebuilt.shape) == tuple(reference.shape), f"{key}: shape"
    assert torch.equal(rebuilt.code, reference.code), f"{key}: NF4 code table"
    assert torch.equal(rebuilt.absmax, reference.absmax), f"{key}: absmax"
    assert torch.equal(rebuilt.offset, reference.offset), f"{key}: nested offset"
    assert rebuilt.state2.blocksize == reference.state2.blocksize, f"{key}: nested blocksize"
    assert torch.equal(rebuilt.state2.code, reference.state2.code), f"{key}: nested code table"
    assert torch.equal(rebuilt.state2.absmax, reference.state2.absmax), f"{key}: nested absmax"


def test_a_nested_nf4_cache_written_before_the_split_reads_back_identically():
    """Real bitsandbytes NF4 with the default double quantisation, written by the old sharder."""
    pytest.importorskip(
        "bitsandbytes.functional", reason="NF4 fixture needs a working bitsandbytes"
    )
    import torch
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.expert_shard import aligned_staging
    from soup_cli.utils.layer_shard import (
        ABSMAX_SUFFIX,
        NESTED_ABSMAX_SUFFIX,
        NESTED_OFFSET_SUFFIX,
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    root, index, readers = _pre_port_readers("nf4")
    assert index.double_quant is True
    assert all(spec.nested for spec in index.expert_quant_specs.values())
    # Quantisation can vary across bitsandbytes versions. Use the cache as the oracle.
    stored_specs = json.loads((root / "index.json").read_text())["expert_quant_specs"]
    extras = load_file(str(root / "extras.safetensors"))
    codes = {name: extras[name].clone() for name in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY)}
    for layer, reader in readers.items():
        stored = load_file(str(root / f"experts_{layer:03d}.safetensors"))
        owner, staging = aligned_staging(reader.max_span)
        try:
            for expert in (1, 2, 0):
                host = reader.read_host(expert, staging)
                for part in ("gate_up", "down"):
                    key = f"{expert}.{part}"
                    spec = stored_specs[part]
                    reference = QuantState(
                        absmax=stored[key + ABSMAX_SUFFIX],
                        shape=torch.Size(spec["shape"]),
                        dtype=getattr(torch, spec["dtype"]),
                        blocksize=spec["blocksize"],
                        code=extras[NF4_CODE_KEY],
                        quant_type=spec["quant_type"],
                        offset=stored[key + NESTED_OFFSET_SUFFIX],
                        state2=QuantState(
                            absmax=stored[key + NESTED_ABSMAX_SUFFIX],
                            code=extras[NF4_NESTED_CODE_KEY],
                            blocksize=spec["nested_blocksize"],
                            dtype=torch.float32,
                        ),
                    )
                    rebuilt = rebuild_quant_state(
                        key, host, index.expert_quant_specs[part], codes
                    )
                    assert torch.equal(host[key], stored[key]), f"{key}: packed NF4 bytes differ"
                    _assert_quant_state_equal(key, rebuilt, reference)
                    assert torch.equal(
                        dequantize_4bit(host[key], quant_state=rebuilt),
                        dequantize_4bit(stored[key], quant_state=reference),
                    ), f"{key}: dequantised values differ from the stored state"
        finally:
            reader.close()
        assert owner.numel() > 0
