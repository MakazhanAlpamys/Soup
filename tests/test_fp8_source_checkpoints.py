"""B3 - block-scaled FP8 source checkpoints (DeepSeek-V3 / Kimi K2 layout).

DeepSeek-V3 and Kimi K2 publish their linear weights as ``float8_e4m3fn`` with an
fp32 ``<name>.weight_scale_inv`` per ``weight_block_size`` block (128 x 128). The
dequantised weight is ``q * scale_inv`` (the suffix names DeepSeek's quantiser's
point of view, not the operation applied here).

Before B3 the sharder cast the raw e4m3 values straight to the target dtype and
copied ``weight_scale_inv`` into the layer shard as an ordinary tensor: a
checkpoint whose weights are off by orders of magnitude, written without a word.
``TestSilentCastGuard`` pins the refusal that replaces that behaviour.

The gates, as fixed in the design note before any measurement:

* **G1** the new path's FP8 -> bf16 is BIT-EXACT against an independent decoder
  that reads e4m3 from its bit fields and never calls a ``float8`` cast.
* **G2** the NF4 bytes the sharder writes are byte-identical to the shipped
  ``_quantize_nf4`` run on the G1 reference.
* **G3** the NF4 reconstruction of what the sharder wrote stays within
  ``0.17 x absmax`` of the independent reference in every 64-element NF4 codec
  block; an all-zero block must reconstruct to exact zeros. 0.17 is a fixed
  threshold: a test failing it is a finding, never a reason to raise it.
* **G4** every negative control fails G1. The fixture is built so it must: each
  FP8 block has a distinct power-of-two scale (none equal to 1) and holds an
  element at the e4m3 maximum, and rows within a block are distinct.
* **G5** dequantisation works in block-aligned row chunks, so no fp32
  intermediate is larger than one chunk.
"""

import json
import math
import os

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: DeepSeek-V3's ``quantization_config.weight_block_size``.
BLOCK = (128, 128)
E4M3_MAX = 448.0
#: The G3 threshold, fixed before any FP8 measurement (design note 3a).
G3_BOUND = 0.17
NF4_CODEC_BLOCK = 64

#: (short name, shape). Square grids (2 x 2) so a transposed scale grid is not
#: hidden by shape, a non-square grid (1 x 3), and a partial edge block (200 rows
#: -> 2 block rows, the second 72 rows tall), which DeepSeek has in
#: ``kv_a_proj_with_mqa`` (576 rows).
FP8_WEIGHTS = (
    ("self_attn.q_proj.weight", (256, 256)),
    ("mlp.down_proj.weight", (128, 384)),
    ("self_attn.kv_a_proj_with_mqa.weight", (200, 256)),
)
FP8_SUFFIXES = frozenset(name for name, _ in FP8_WEIGHTS)
FP8_CONFIG = {
    "model_type": "llama",
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(BLOCK),
    },
}


# ==========================================================================
# the independent reference decoder
# ==========================================================================
def _e4m3_table():
    """Every e4m3fn byte decoded from its bit fields, with no float8 cast.

    Layout: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. Exponent 0
    is subnormal (``m/8 * 2^-6``). e4m3fn has no infinities: only ``S.1111.111``
    is NaN, so exponent 15 is otherwise an ordinary binade and the maximum is
    ``1.75 * 2^8 = 448``.
    """
    import torch

    values = []
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values.append(float("nan"))
        elif exponent == 0:
            values.append(sign * (mantissa / 8.0) * 2.0**-6)
        else:
            values.append(sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return torch.tensor(values, dtype=torch.float64)


def _reference_decode(q, scale, block=BLOCK):
    """G1's reference: table decode, fp32 multiply, one rounding to bf16."""
    import torch

    rows, cols = q.shape
    values = _e4m3_table()[q.view(torch.uint8).long()].to(torch.float32)
    expanded = (
        scale.to(torch.float32)
        .repeat_interleave(block[0], dim=0)[:rows]
        .repeat_interleave(block[1], dim=1)[:, :cols]
    )
    return (values * expanded).to(torch.bfloat16)


# ==========================================================================
# fixtures
# ==========================================================================
class _Exponents:
    """Hands out distinct power-of-two scales: 2^-3, 2^-4, ... never 1."""

    def __init__(self, start=-3):
        self.next = start

    def take(self):
        value = 2.0**self.next
        self.next -= 1
        return value


def _fp8_weight(shape, exponents, *, generator, block=BLOCK):
    """One controlled FP8 weight and its scale grid.

    Every block gets its own power-of-two scale and one element at +-448, and
    no two rows inside a block are equal: the conditions under which any wrong,
    swapped or missing scale and any row permutation changes a decoded value.
    """
    import torch

    rows, cols = shape
    grid = (math.ceil(rows / block[0]), math.ceil(cols / block[1]))
    raw = (torch.rand(shape, generator=generator) * 2 - 1) * 440.0
    q = raw.to(torch.float8_e4m3fn)
    scale = torch.empty(grid, dtype=torch.float32)
    for br in range(grid[0]):
        for bc in range(grid[1]):
            scale[br, bc] = exponents.take()
            r0, c0 = br * block[0], bc * block[1]
            sign = 1.0 if (br + bc) % 2 == 0 else -1.0
            q[r0 + (br + bc) % min(block[0], rows - r0), c0] = torch.tensor(
                sign * E4M3_MAX
            ).to(torch.float8_e4m3fn)
            tile = q[r0 : r0 + block[0], c0 : c0 + block[1]].view(torch.uint8)
            assert tile.unique(dim=0).shape[0] == tile.shape[0], "rows must be distinct"
    return q, scale


def _build_fp8_tensors(n_layers=2, seed=0):
    """(source tensors, G1 references) for a small llama-shaped FP8 checkpoint."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    exponents = _Exponents()
    tensors, reference = {}, {}
    for idx in range(n_layers):
        pre = f"model.layers.{idx}."
        for name, shape in FP8_WEIGHTS:
            q, scale = _fp8_weight(shape, exponents, generator=generator)
            tensors[pre + name] = q
            tensors[pre + name + "_scale_inv"] = scale
            reference[pre + name] = _reference_decode(q, scale)
        tensors[pre + "input_layernorm.weight"] = torch.randn(
            256, generator=generator
        ).to(torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randn(
        64, 256, generator=generator
    ).to(torch.bfloat16)
    tensors["model.norm.weight"] = torch.randn(256, generator=generator).to(torch.bfloat16)
    return tensors, reference


def _write_checkpoint(directory, tensors, config=FP8_CONFIG, files=None):
    """Write ``tensors`` (optionally split across files) and ``config.json``."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    if files is None:
        files = {"model.safetensors": list(tensors)}
    for filename, keys in files.items():
        save_file(
            {key: tensors[key].contiguous() for key in keys}, str(directory / filename)
        )
    if config is not None:
        (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


def _shard(tmp_path, tensors, *, config=FP8_CONFIG, name="cache", **kwargs):
    from soup_cli.utils.layer_shard import shard_checkpoint

    src = _write_checkpoint(tmp_path / f"{name}-src", tensors, config)
    out = str(tmp_path / name)
    kwargs.setdefault("dtype", "bfloat16")
    kwargs.setdefault("arch", "llama")
    return src, out, shard_checkpoint(src, out, **kwargs)


def _layer(out, idx):
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import layer_shard_path

    return load_file(layer_shard_path(out, idx))


# ==========================================================================
# the reference decoder itself
# ==========================================================================
class TestReferenceDecoder:
    def test_table_agrees_with_torch_on_every_finite_byte(self):
        """The reference is only worth something if it is right. It is checked
        once against torch's own cast here, and never calls that cast itself."""
        import torch

        table = _e4m3_table()
        every_byte = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
        theirs = every_byte.to(torch.float64)
        finite = ~torch.isnan(table)
        assert int(finite.sum()) == 254  # only 0x7F and 0xFF are NaN
        assert torch.equal(table[finite], theirs[finite])
        assert torch.isnan(theirs[~finite]).all()

    def test_table_landmarks(self):
        table = _e4m3_table()
        assert table[0x7E].item() == E4M3_MAX
        assert table[0xFE].item() == -E4M3_MAX
        assert table[0x01].item() == 2.0**-9  # smallest subnormal
        assert table[0x08].item() == 2.0**-6  # smallest normal


# ==========================================================================
# the silent cast the PR removes
# ==========================================================================
class TestSilentCastGuard:
    def test_fp8_weight_without_scale_or_config_is_refused(self, tmp_path):
        """Before B3 this sharded without a word and stored values up to 448
        where the real weight is ~0.01."""
        tensors, _ = _build_fp8_tensors()
        for key in [k for k in tensors if k.endswith("_scale_inv")]:
            del tensors[key]
        with pytest.raises(ValueError, match="float8"):
            _shard(tmp_path, tensors, config={"model_type": "llama"})

    def test_fp8_with_scales_but_no_fp8_quantization_config_is_refused(self, tmp_path):
        tensors, _ = _build_fp8_tensors()
        with pytest.raises(ValueError, match="quantization_config"):
            _shard(tmp_path, tensors, config={"model_type": "llama"})

    def test_read_tensor_refuses_float8_directly(self, tmp_path):
        """The last line of defence: no other path may cast e4m3 bytes."""
        import torch
        from safetensors import safe_open

        from soup_cli.utils.layer_shard import _read_tensor

        src = _write_checkpoint(
            tmp_path / "src", {"w": torch.zeros(4, 4).to(torch.float8_e4m3fn)}, None
        )
        with safe_open(os.path.join(src, "model.safetensors"), framework="pt") as handle:
            with pytest.raises(ValueError, match="float8"):
                _read_tensor(handle, "w", "bfloat16")


# ==========================================================================
# G1: bit-exact FP8 -> bf16
# ==========================================================================
class TestG1BitExact:
    def test_sharded_weights_equal_the_independent_decode(self, tmp_path):
        import torch

        tensors, reference = _build_fp8_tensors()
        _, out, _ = _shard(tmp_path, tensors)
        for idx in range(2):
            blob = _layer(out, idx)
            for name, _ in FP8_WEIGHTS:
                got = blob[name]
                want = reference[f"model.layers.{idx}.{name}"]
                assert got.dtype == torch.bfloat16
                assert torch.equal(got, want), (idx, name)

    def test_scales_never_reach_the_shard(self, tmp_path):
        tensors, _ = _build_fp8_tensors()
        _, out, index = _shard(tmp_path, tensors)
        for idx in range(2):
            assert not [k for k in _layer(out, idx) if "scale_inv" in k]
        assert not [k for k in index.layer_keys if "scale_inv" in k]

    def test_unquantised_tensors_pass_through_unchanged(self, tmp_path):
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import extras_shard_path

        tensors, _ = _build_fp8_tensors()
        _, out, _ = _shard(tmp_path, tensors)
        assert torch.equal(
            _layer(out, 1)["input_layernorm.weight"],
            tensors["model.layers.1.input_layernorm.weight"],
        )
        extras = load_file(extras_shard_path(out))
        assert torch.equal(extras["model.norm.weight"], tensors["model.norm.weight"])

    @pytest.mark.parametrize("chunk_rows", [128, 256, None])
    def test_chunk_size_does_not_change_a_single_bit(self, tmp_path, chunk_rows):
        import torch
        from safetensors import safe_open

        from soup_cli.utils.fp8_source import dequantize_fp8_blockwise

        tensors, reference = _build_fp8_tensors(n_layers=1)
        src = _write_checkpoint(tmp_path / "src", tensors)
        key = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        with safe_open(os.path.join(src, "model.safetensors"), framework="pt") as handle:
            got = dequantize_fp8_blockwise(
                handle.get_slice(key),
                handle.get_tensor(key + "_scale_inv"),
                block=BLOCK,
                dtype="bfloat16",
                key=key,
                chunk_rows=chunk_rows,
            )
        assert torch.equal(got, reference[key])


# ==========================================================================
# G2 and G3: what the sharder writes with quant="nf4"
# ==========================================================================
def _nf4_shard(tmp_path, tensors, double_quant, name="nf4"):
    from soup_cli.utils.layer_shard import QUANT_NF4

    return _shard(
        tmp_path,
        tensors,
        name=name,
        quant=QUANT_NF4,
        quant_suffixes=FP8_SUFFIXES,
        double_quant=double_quant,
        quant_device="cpu",
    )


def _dequant_from_shard(out, index, idx, name):
    from bitsandbytes.functional import dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import extras_shard_path
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    blob = _layer(out, idx)
    codes = load_file(extras_shard_path(out))
    state = rebuild_quant_state(name, blob, index.quant_specs[name], codes)
    return dequantize_4bit(blob[name], state)


def _g3_ratios(reconstructed, reference):
    """Per NF4 codec block: (max |error|, absmax), both in fp32."""
    got = reconstructed.float().reshape(-1, NF4_CODEC_BLOCK)
    want = reference.float().reshape(-1, NF4_CODEC_BLOCK)
    return (got - want).abs().amax(dim=1), want.abs().amax(dim=1)


#: None is the default path (``training.bnb_4bit_use_double_quant`` unset), which
#: for an FP8 source resolves to double-quant OFF; False is the explicit setting.
#: An explicit True is refused for FP8 sources (``TestDoubleQuantForFp8Sources``).
@pytest.mark.parametrize("double_quant", [None, False])
class TestG2G3NF4:
    def test_g2_nf4_bytes_match_the_codec_on_the_reference(self, tmp_path, double_quant):
        import torch

        from soup_cli.utils.layer_shard import _quantize_nf4

        tensors, reference = _build_fp8_tensors()
        _, out, _ = _nf4_shard(tmp_path, tensors, double_quant)
        for idx in range(2):
            blob = _layer(out, idx)
            for name, _ in FP8_WEIGHTS:
                want, _, _, _ = _quantize_nf4(
                    reference[f"model.layers.{idx}.{name}"],
                    double_quant=False,
                    device="cpu",
                )
                for sidecar, tensor in want.items():
                    assert torch.equal(blob[name + sidecar], tensor), (idx, name, sidecar)

    def test_g3_every_codec_block_is_within_the_bound(self, tmp_path, double_quant):
        import torch

        tensors, reference = _build_fp8_tensors()
        _, out, index = _nf4_shard(tmp_path, tensors, double_quant)
        for idx in range(2):
            for name, _ in FP8_WEIGHTS:
                back = _dequant_from_shard(out, index, idx, name)
                assert torch.isfinite(back).all()
                err, absmax = _g3_ratios(back, reference[f"model.layers.{idx}.{name}"])
                worst = (err / absmax).max().item()
                assert (err <= G3_BOUND * absmax).all(), (idx, name, worst)

    def test_g3_zero_block_reconstructs_to_exact_zeros(self, tmp_path, double_quant):
        """A zero block has absmax 0, so the bound is 0: every output exactly
        +-0 and finite. Lives in its own fixture because a zero block cannot
        hold the 448 element the G4 fixture needs."""
        import torch

        tensors, _ = _build_fp8_tensors()
        key = "model.layers.0.self_attn.q_proj.weight"
        q = tensors[key].clone()
        q[:128, :128] = torch.zeros(128, 128).to(torch.float8_e4m3fn)
        tensors[key] = q
        reference = _reference_decode(q, tensors[key + "_scale_inv"])
        _, out, index = _nf4_shard(tmp_path, tensors, double_quant)
        back = _dequant_from_shard(out, index, 0, "self_attn.q_proj.weight")
        assert torch.isfinite(back).all()
        zero_block = back[:128, :128]
        assert (zero_block == 0).all()
        err, absmax = _g3_ratios(back, reference)
        assert (err <= G3_BOUND * absmax).all()


# ==========================================================================
# G4: negative controls, each guaranteed to fail G1
# ==========================================================================
_Q = "model.layers.0.self_attn.q_proj.weight"
_Q_SCALE = _Q + "_scale_inv"


def _scale_times_two(tensors):
    tensors[_Q_SCALE] = tensors[_Q_SCALE].clone()
    tensors[_Q_SCALE][0, 1] *= 2


def _scale_from_another_layer(tensors):
    tensors[_Q_SCALE] = tensors["model.layers.1.self_attn.q_proj.weight_scale_inv"].clone()


def _scale_grid_transposed(tensors):
    tensors[_Q_SCALE] = tensors[_Q_SCALE].t().contiguous()


def _divide_instead_of_multiply(tensors):
    tensors[_Q_SCALE] = 1.0 / tensors[_Q_SCALE]


def _rows_permuted(tensors):
    q = tensors[_Q].clone()
    q[[0, 1]] = q[[1, 0]]
    tensors[_Q] = q


@pytest.mark.parametrize(
    "fault",
    [
        _scale_times_two,
        _scale_from_another_layer,
        _scale_grid_transposed,
        _divide_instead_of_multiply,
        _rows_permuted,
    ],
)
def test_g4_every_control_fails_g1(tmp_path, fault):
    import torch

    tensors, reference = _build_fp8_tensors()
    fault(tensors)
    _, out, _ = _shard(tmp_path, tensors)
    assert not torch.equal(_layer(out, 0)["self_attn.q_proj.weight"], reference[_Q])


def test_g4_missing_scale_is_refused_by_name(tmp_path):
    tensors, _ = _build_fp8_tensors()
    del tensors[_Q_SCALE]
    with pytest.raises(ValueError, match="self_attn.q_proj.weight"):
        _shard(tmp_path, tensors)


# ==========================================================================
# refusals at load
# ==========================================================================
class TestRefusals:
    @pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0.0, -0.5])
    def test_scale_must_be_finite_and_positive(self, tmp_path, bad):
        tensors, _ = _build_fp8_tensors()
        tensors[_Q_SCALE] = tensors[_Q_SCALE].clone()
        tensors[_Q_SCALE][1, 1] = bad
        with pytest.raises(ValueError, match="scale"):
            _shard(tmp_path, tensors)

    def test_scale_grid_of_the_wrong_shape(self, tmp_path):
        tensors, _ = _build_fp8_tensors()
        tensors[_Q_SCALE] = tensors[_Q_SCALE][:1].clone()
        with pytest.raises(ValueError, match="scale grid"):
            _shard(tmp_path, tensors)

    def test_scale_without_its_weight(self, tmp_path):
        import torch

        tensors, _ = _build_fp8_tensors()
        tensors["model.layers.0.mlp.ghost_proj.weight_scale_inv"] = torch.ones(1, 1)
        with pytest.raises(ValueError, match="ghost_proj"):
            _shard(tmp_path, tensors)

    def test_scale_in_the_next_file_is_decoded(self, tmp_path, monkeypatch):
        """Not a refusal: DeepSeek-V3 puts 155 of its 45,808 scales in the file
        after their weight, where a layer crosses a file boundary. Found by the
        first real-shard run, which this case would otherwise have refused.
        Run with ONE live handle so the read cannot lean on both files staying
        open."""
        import torch

        from soup_cli.utils import layer_shard

        class OneHandle(layer_shard._SourceHandles):
            def __init__(self, shards, opener, capacity=1):
                super().__init__(shards, opener, capacity=1)

        monkeypatch.setattr(layer_shard, "_SourceHandles", OneHandle)
        tensors, reference = _build_fp8_tensors()
        keys = list(tensors)
        files = {
            "model-00001-of-00002.safetensors": [k for k in keys if k != _Q_SCALE],
            "model-00002-of-00002.safetensors": [_Q_SCALE],
        }
        src = _write_checkpoint(tmp_path / "src", tensors, FP8_CONFIG, files)
        out = str(tmp_path / "out")
        layer_shard.shard_checkpoint(src, out, dtype="bfloat16", arch="llama")
        assert torch.equal(_layer(out, 0)["self_attn.q_proj.weight"], reference[_Q])

    @pytest.mark.parametrize(
        "patch, message",
        [
            ({"fmt": "e5m2"}, "e4m3"),
            ({"weight_block_size": None}, "weight_block_size"),
            ({"weight_block_size": [128]}, "weight_block_size"),
            ({"weight_block_size": [0, 128]}, "weight_block_size"),
        ],
    )
    def test_unsupported_fp8_config(self, tmp_path, patch, message):
        tensors, _ = _build_fp8_tensors()
        config = json.loads(json.dumps(FP8_CONFIG))
        config["quantization_config"].update(patch)
        if patch.get("weight_block_size", ...) is None:
            del config["quantization_config"]["weight_block_size"]
        with pytest.raises(ValueError, match=message):
            _shard(tmp_path, tensors, config=config)

    def test_fp8_config_with_bf16_weights_shards_normally(self, tmp_path):
        """A bf16 re-export that kept the FP8 config has nothing to dequantise."""
        import torch

        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64).to(torch.bfloat16),
            "model.embed_tokens.weight": torch.randn(16, 64).to(torch.bfloat16),
        }
        _, out, _ = _shard(tmp_path, tensors)
        assert torch.equal(
            _layer(out, 0)["self_attn.q_proj.weight"],
            tensors["model.layers.0.self_attn.q_proj.weight"],
        )


# ==========================================================================
# G5: bounded fp32 intermediates
# ==========================================================================
def test_g5_no_fp32_intermediate_exceeds_one_chunk(tmp_path, monkeypatch):
    import torch
    from safetensors import safe_open

    from soup_cli.utils import fp8_source

    seen = []
    original = fp8_source._dequantize_rows

    def spy(out_rows, q_rows, scale_rows, block_cols):
        seen.append(tuple(q_rows.shape))
        return original(out_rows, q_rows, scale_rows, block_cols)

    monkeypatch.setattr(fp8_source, "_dequantize_rows", spy)
    tensors, reference = _build_fp8_tensors(n_layers=1)
    src = _write_checkpoint(tmp_path / "src", tensors)
    key = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"  # 200 rows: 128 + 72
    with safe_open(os.path.join(src, "model.safetensors"), framework="pt") as handle:
        got = fp8_source.dequantize_fp8_blockwise(
            handle.get_slice(key),
            handle.get_tensor(key + "_scale_inv"),
            block=BLOCK,
            dtype="bfloat16",
            key=key,
            chunk_rows=128,
        )
    assert seen == [(128, 256), (72, 256)]
    assert torch.equal(got, reference[key])


def test_g5_chunk_rows_are_block_aligned():
    from soup_cli.utils.fp8_source import chunk_rows_for

    for cols in (256, 7168, 18432):
        rows = chunk_rows_for(cols, block_rows=128)
        assert rows >= 128 and rows % 128 == 0
        assert rows * cols * 4 <= max(64 * 2**20, 128 * cols * 4)


# ==========================================================================
# cache identity
# ==========================================================================
def test_changed_fp8_config_reshards(tmp_path):
    """The config now decides the bytes a source produces, so it is part of the
    source fingerprint for FP8 checkpoints."""
    from soup_cli.utils.layer_shard import shard_checkpoint

    tensors, _ = _build_fp8_tensors(n_layers=1)
    src, out, _ = _shard(tmp_path, tensors)
    config = json.loads(json.dumps(FP8_CONFIG))
    config["quantization_config"]["activation_scheme"] = "static"
    with open(os.path.join(src, "config.json"), "w") as handle:
        json.dump(config, handle)
    notes = []
    shard_checkpoint(src, out, dtype="bfloat16", arch="llama", notify=notes.append)
    assert any("Re-sharding" in note for note in notes), notes


# ==========================================================================
# double-quant for FP8 sources: the supported configuration
# ==========================================================================
class TestDoubleQuantForFp8Sources:
    """With double-quant on, the bitsandbytes NF4 codec failed B3's
    preregistered G3 gate on real DeepSeek-V3 weights (worst 2.1503 against
    0.17) and passed with it off (worst 0.1531). The gate is not relaxed;
    instead an FP8 source shards without double-quant by default, and asking
    for it explicitly is refused by name."""

    def test_default_path_shards_without_double_quant(self, tmp_path):
        from soup_cli.utils.layer_shard import QUANT_NF4

        tensors, _ = _build_fp8_tensors(n_layers=1)
        _, _, index = _shard(
            tmp_path,
            tensors,
            quant=QUANT_NF4,
            quant_suffixes=FP8_SUFFIXES,
            quant_device="cpu",
        )
        assert index.double_quant is False
        assert all(spec.nested is False for spec in index.quant_specs.values())

    def test_explicit_double_quant_is_refused_with_the_reason(self, tmp_path):
        from soup_cli.utils.layer_shard import QUANT_NF4

        tensors, _ = _build_fp8_tensors(n_layers=1)
        with pytest.raises(ValueError, match="gate-b3-fp8-source"):
            _shard(
                tmp_path,
                tensors,
                quant=QUANT_NF4,
                quant_suffixes=FP8_SUFFIXES,
                double_quant=True,
                quant_device="cpu",
            )

    def test_bf16_source_keeps_the_shipped_default(self, tmp_path):
        """Nothing changes for any other checkpoint: unset still means on."""
        import torch

        from soup_cli.utils.layer_shard import QUANT_NF4

        generator = torch.Generator().manual_seed(1)
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                128, 128, generator=generator
            ).to(torch.bfloat16),
            "model.embed_tokens.weight": torch.randn(16, 128, generator=generator).to(
                torch.bfloat16
            ),
        }
        _, _, index = _shard(
            tmp_path,
            tensors,
            config={"model_type": "llama"},
            quant=QUANT_NF4,
            quant_suffixes={"self_attn.q_proj.weight"},
            quant_device="cpu",
        )
        assert index.double_quant is True
        assert index.quant_specs["self_attn.q_proj.weight"].nested is True

    @pytest.mark.parametrize(
        "requested, fp8, quant, expected",
        [
            (None, True, "nf4", False),
            (False, True, "nf4", False),
            (None, False, "nf4", True),
            (True, False, "nf4", True),
            (False, False, "nf4", False),
            (True, True, "none", True),
            (None, True, "none", True),
        ],
    )
    def test_resolution_table(self, requested, fp8, quant, expected):
        from soup_cli.utils.fp8_source import resolve_double_quant

        assert resolve_double_quant(requested, fp8_source=fp8, quant=quant) is expected

    def test_model_config_detection(self):
        import enum
        from types import SimpleNamespace

        from soup_cli.utils.fp8_source import is_block_fp8_model_config

        class Method(str, enum.Enum):
            FP8 = "fp8"

        assert is_block_fp8_model_config(
            SimpleNamespace(quantization_config={"quant_method": "fp8"})
        )
        assert is_block_fp8_model_config(
            SimpleNamespace(quantization_config=SimpleNamespace(quant_method=Method.FP8))
        )
        assert not is_block_fp8_model_config(SimpleNamespace())
        assert not is_block_fp8_model_config(
            SimpleNamespace(quantization_config={"quant_method": "bitsandbytes"})
        )
        assert not is_block_fp8_model_config(SimpleNamespace(quantization_config={}))

    def test_stream_setup_resolves_through_the_helper(self):
        """The trainer and the sharder must agree, so the trainer's setup calls the
        same resolver; a later edit back to ``tcfg.double_quant_on`` fails here."""
        import inspect

        from soup_cli.trainer import stream_setup

        source = inspect.getsource(stream_setup)
        assert "stream_double_quant(tcfg, model_config, quant=quant)" in source

    @pytest.mark.parametrize(
        "requested, fp8, expected",
        [(None, True, False), (False, True, False), (None, False, True), (False, False, False)],
    )
    def test_stream_double_quant(self, requested, fp8, expected):
        from types import SimpleNamespace

        from soup_cli.utils.fp8_source import stream_double_quant

        tcfg = SimpleNamespace(
            double_quant_requested=requested, double_quant_on=requested is not False
        )
        config = SimpleNamespace(quantization_config={"quant_method": "fp8"} if fp8 else None)
        assert stream_double_quant(tcfg, config, quant="nf4") is expected

    def test_stream_double_quant_refuses_explicit_true_for_fp8(self):
        from types import SimpleNamespace

        from soup_cli.utils.fp8_source import stream_double_quant

        tcfg = SimpleNamespace(double_quant_requested=True, double_quant_on=True)
        config = SimpleNamespace(quantization_config={"quant_method": "fp8"})
        with pytest.raises(ValueError, match="not supported for a block-scaled FP8"):
            stream_double_quant(tcfg, config, quant="nf4")


# ==========================================================================
# the real-shard benchmark harness, on a cross-file scale
# ==========================================================================
def test_benchmark_reads_a_scale_from_its_own_file(tmp_path):
    """benchmarks/harness/fp8_source_probe.py must verify G1/G3 when a weight and
    its ``weight_scale_inv`` sit in different source files, as 155 pairs do in
    DeepSeek-V3. It used to read the scale from the weight's file and stop with
    a missing-tensor error after sharding had already succeeded."""
    import importlib.util
    import sys
    from pathlib import Path

    harness = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "fp8_source_probe.py"
    spec = importlib.util.spec_from_file_location("fp8_source_probe", harness)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        tensors, _ = _build_fp8_tensors()
        keys = list(tensors)
        files = {
            "model-00001-of-00002.safetensors": [k for k in keys if k != _Q_SCALE],
            "model-00002-of-00002.safetensors": [_Q_SCALE],
        }
        src = _write_checkpoint(tmp_path / "src", tensors, FP8_CONFIG, files)
        out_json = tmp_path / "result.json"
        code = module.main(
            [
                "--source", src,
                "--out", str(tmp_path / "out"),
                "--double-quant", "off",
                "--json", str(out_json),
            ]
        )
    finally:
        sys.modules.pop(spec.name, None)
    assert code == 0
    summary = json.loads(out_json.read_text())["summary"]
    n_fp8 = sum(1 for key in tensors if key + "_scale_inv" in tensors)
    assert summary["g1_tensors"] == n_fp8
    assert summary["g1_pass"] is True
    assert summary["g3_pass"] is True


def test_benchmark_runs_without_unix_only_apis(tmp_path, monkeypatch):
    """The whole probe must run where ``resource`` and ``os.statvfs`` do not
    exist (Windows): same cross-file checkpoint as above, with both removed.

    ``statvfs`` is hidden from the probe's own ``os`` and not deleted from the
    real module: on POSIX ``shutil.disk_usage`` is itself built on it.
    """
    import importlib.util
    import sys
    from pathlib import Path

    harness = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "fp8_source_probe.py"
    spec = importlib.util.spec_from_file_location("fp8_source_probe", harness)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A None entry makes ``import resource`` raise ImportError, as on Windows.
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    class _OsWithoutStatvfs:
        def __getattr__(self, name):
            if name == "statvfs":
                raise AttributeError("module 'os' has no attribute 'statvfs'")
            return getattr(os, name)

    monkeypatch.setattr(module, "os", _OsWithoutStatvfs())

    tensors, _ = _build_fp8_tensors()
    files = {
        "model-00001-of-00002.safetensors": [k for k in tensors if k != _Q_SCALE],
        "model-00002-of-00002.safetensors": [_Q_SCALE],
    }
    src = _write_checkpoint(tmp_path / "src", tensors, FP8_CONFIG, files)
    out_json = tmp_path / "result.json"
    code = module.main(
        [
            "--source", src,
            "--out", str(tmp_path / "out"),
            "--double-quant", "off",
            "--json", str(out_json),
        ]
    )
    assert code == 0
    result = json.loads(out_json.read_text())
    assert result["summary"]["g1_pass"] is True
    assert result["summary"]["g3_pass"] is True
    free = result["box"]["disk_free_bytes"]
    assert isinstance(free, int) and free > 0
    assert result["g5"]["peak_rss_bytes_through_sharding"] is None
    assert result["g5"]["peak_rss_note"]
