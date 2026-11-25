"""Expert shard storage (``utils/expert_shard.py``).

The storage layer is shared by the expert runtimes. The runtime keeps importing
its names from ``utils/expert_stream.py``; the shard format is version 5 with the
index fields this tree has always written. A cache written before the storage
layer was split out must load unchanged, and an index that also carries the
soup-rnd runtime's fields must load too.
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


def test_this_trees_writer_adds_no_index_field_of_the_rnd_runtime(tmp_path):
    _, out, _ = _main_cache(tmp_path)
    payload = json.loads((out / "index.json").read_text())
    assert payload["format_version"] == 5
    assert payload["expert_layout"] == "per_expert"
    assert "fp32_router_weights" in payload
    for field in ("expert_layout_version", "fp32_keys", "stacked_experts"):
        assert field not in payload, field


def test_an_rnd_style_index_loads_through_the_same_reader(tmp_path):
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


def test_reader_closes_its_handle_when_the_header_is_unreadable(tmp_path, monkeypatch):
    from soup_cli.utils import safetensors_reader

    path, info = _written(tmp_path)
    opened = []
    real_open_direct = safetensors_reader.open_direct

    def boom(_path):
        raise ValueError("injected header failure")

    monkeypatch.setattr(safetensors_reader, "read_header_with_identity", boom)
    def tracking_open(target):
        handle = real_open_direct(target)
        opened.append(handle)
        return handle

    monkeypatch.setattr(safetensors_reader, "open_direct", tracking_open)
    from soup_cli.utils.expert_shard import ExpertShardReader

    # ``excinfo`` keeps the half-built reader alive, so ``__del__`` cannot be what
    # closes the handle: only the constructor's own cleanup can.
    with pytest.raises(ValueError, match="injected header failure") as excinfo:
        ExpertShardReader(
            path,
            count=3,
            shapes=info.shapes,
            quant_specs={},
            dtype="float32",
            direct=True,
        )
    assert opened and all(handle.closed for handle in opened)
    assert excinfo.value is not None


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
