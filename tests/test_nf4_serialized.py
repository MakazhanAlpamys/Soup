"""Tests for ``soup_cli.utils.nf4_serialized``: byte-for-byte import of a bnb NF4 checkpoint.

The sources are REAL bitsandbytes serialisations built on the CPU (``quantize_4bit`` then
``QuantState.as_dict(packed=True)``), not hand-made tensors, so the key names, dtypes and
the JSON blob are exactly what the Hub checkpoints carry.
"""

import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
bnb_functional = pytest.importorskip("bitsandbytes.functional")
safetensors_torch = pytest.importorskip("safetensors.torch")

from soup_cli.utils import nf4_serialized  # noqa: E402
from soup_cli.utils.nf4_serialized import (  # noqa: E402
    import_nf4_checkpoint,
    plan_nf4_import,
    verify_nf4_import,
)

_STATE = ".quant_state.bitsandbytes__nf4"
_Q = "model.layers.0.self_attn.q_proj.weight"
_PROJ = r"\.layers\.\d+\.(self_attn|mlp)\.\w+_proj\.weight$"
_QUANT = {
    "quant_method": "bitsandbytes",
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_storage": "uint8",
    "bnb_4bit_compute_dtype": "bfloat16",
}
_SHAPE = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
    max_position_embeddings=64,
)
# dense (non-quantised) tensors per decoder layer, by architecture
_DENSE_PER_LAYER = {"llama": 2, "qwen3": 4, "mistral": 2}


def _config(model_type, tie):
    from transformers import LlamaConfig, MistralConfig, Qwen3Config

    if model_type == "llama":
        return LlamaConfig(tie_word_embeddings=tie, **_SHAPE)
    if model_type == "qwen3":
        return Qwen3Config(tie_word_embeddings=tie, head_dim=16, **_SHAPE)
    return MistralConfig(tie_word_embeddings=tie, **_SHAPE)


def edit_state(tensors, key, **fields):
    """Rewrite fields of one weight's quant_state JSON blob."""
    blob = tensors[key + _STATE]
    payload = json.loads(blob.numpy().tobytes())
    payload.update(fields)
    raw = json.dumps(payload).encode()
    tensors[key + _STATE] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def build_source(
    root, model_type="llama", *, tie=False, blocksize=64, nested=True, files=1, tweak=None,
    quant_config=None,
):
    """Write a tiny serialised-bnb checkpoint into ``root``; return its tensors."""
    import re

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = _config(model_type, tie)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    tensors = {}
    for key, value in model.state_dict().items():
        if tie and key == "lm_head.weight":
            continue
        if re.search(_PROJ, key):
            packed, state = bnb_functional.quantize_4bit(
                value.detach().contiguous(),
                blocksize=blocksize,
                compress_statistics=nested,
                quant_type="nf4",
            )
            tensors[key] = packed.contiguous()
            for part, tensor in state.as_dict(packed=True).items():
                tensors[f"{key}.{part}"] = tensor.detach().clone().contiguous()
        else:
            tensors[key] = value.detach().clone().contiguous()
    if tweak:
        tweak(tensors)
    os.makedirs(root, exist_ok=True)
    config = json.loads(cfg.to_json_string())
    config["torch_dtype"] = "bfloat16"
    config["quantization_config"] = quant_config if quant_config is not None else dict(_QUANT)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    names = sorted(tensors)
    if files == 1:
        safetensors_torch.save_file(tensors, os.path.join(root, "model.safetensors"))
    else:
        weight_map = {}
        step = -(-len(names) // files)
        for number in range(files):
            file_name = f"model-{number + 1:05d}-of-{files:05d}.safetensors"
            part = {name: tensors[name] for name in names[number * step : (number + 1) * step]}
            safetensors_torch.save_file(part, os.path.join(root, file_name))
            weight_map.update({name: file_name for name in part})
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
            json.dump({"metadata": {}, "weight_map": weight_map}, handle)
    return tensors


def load_dir(root):
    """Every tensor of every ``*.safetensors`` in ``root`` (a test helper: maps is fine here)."""
    merged = {}
    for name in sorted(os.listdir(root)):
        if name.endswith(".safetensors"):
            merged.update(safetensors_torch.load_file(os.path.join(root, name)))
    return merged


def raw(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


@pytest.fixture
def work(tmp_path):
    return str(tmp_path / "src"), str(tmp_path / "out")


# ----------------------------------------------------------------------
# happy path
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model_type", "tie", "files", "blocksize"),
    [
        ("llama", False, 1, 64),
        ("qwen3", False, 2, 64),
        ("mistral", False, 1, 64),
        ("llama", True, 1, 64),
        ("llama", False, 1, 4096),
        ("qwen3", False, 2, 4096),
    ],
)
def test_import_is_byte_for_byte_and_decodes_like_bitsandbytes(
    work, model_type, tie, files, blocksize
):
    from bitsandbytes.functional import QuantState, dequantize_4bit

    from soup_cli.utils.layer_shard import (
        _SHARD_FORMAT_VERSION,
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
        large_shard_path,
        layer_shard_path,
        read_shard_index,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    source, out = work
    build_source(source, model_type, tie=tie, files=files, blocksize=blocksize)
    result = import_nf4_checkpoint(source, out, chunk_bytes=1024)

    src = load_dir(source)
    index = read_shard_index(out)
    assert index == result.index
    assert (index.quant, index.double_quant, index.quant_device) == ("nf4", True, "serialized")
    assert index.format_version == _SHARD_FORMAT_VERSION
    assert index.dtype == "bfloat16" and index.n_layers == 2 and index.arch == model_type
    assert set(index.quant_specs) == {
        f"self_attn.{n}_proj.weight" for n in "qkvo"
    } | {f"mlp.{n}_proj.weight" for n in ("gate", "up", "down")}
    assert result.verified is True and result.layer_count == 2
    assert {spec.blocksize for spec in index.quant_specs.values()} == {blocksize}
    large = () if tie else ("lm_head.weight", "model.embed_tokens.weight")
    assert index.large_keys == large

    # count and size are derived from the shard files themselves, not from the plan
    shard_files = [layer_shard_path(out, idx) for idx in range(2)] + [extras_shard_path(out)]
    shard_files += [large_shard_path(out, key) for key in large]
    stored = [safetensors_torch.load_file(path) for path in shard_files]
    assert result.tensor_count == sum(len(shard) for shard in stored)
    assert result.payload_bytes == sum(
        tensor.numel() * tensor.element_size() for shard in stored for tensor in shard.values()
    )
    per_layer = 7 * 4 + _DENSE_PER_LAYER[model_type]
    assert result.tensor_count == 2 * per_layer + (2 if tie else 3) + 2

    # every payload equals the source payload
    suffix_of = {"": "", "::absmax": ".absmax", "::nested_absmax": ".nested_absmax"}
    extras = stored[2]
    for layer, shard in enumerate(stored[:2]):
        for key, tensor in shard.items():
            base, sep, tail = key.partition("::")
            if sep and tail == "nested_offset":
                offset = json.loads(src[f"model.layers.{layer}.{base}{_STATE}"].numpy().tobytes())
                assert tensor.shape == () and tensor.dtype == torch.float32
                assert tensor.item() == offset["nested_offset"]
                continue
            tail = "::" + tail if sep else ""
            source_key = f"model.layers.{layer}.{base}{suffix_of.get(tail, '')}"
            assert torch.equal(raw(tensor), raw(src[source_key])), key
    for key, tensor in extras.items():
        if key == NF4_CODE_KEY:
            assert torch.equal(raw(tensor), raw(src[_Q + ".quant_map"]))
        elif key == NF4_NESTED_CODE_KEY:
            assert torch.equal(raw(tensor), raw(src[_Q + ".nested_quant_map"]))
        else:
            assert torch.equal(raw(tensor), raw(src[key])), key
    for shard in stored[3:]:
        for key, tensor in shard.items():
            assert torch.equal(raw(tensor), raw(src[key])), key

    # MANDATORY: Soup's rebuilt state decodes bit-exactly like bitsandbytes' own decode of
    # the SOURCE (a QuantState unpacked from the serialised tensors, as bnb's loader does)
    codes = {key: extras[key] for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY)}
    checked = 0
    for layer, shard in enumerate(stored[:2]):
        for short, spec in index.quant_specs.items():
            full = f"model.layers.{layer}.{short}"
            reference_state = QuantState.from_dict(
                {
                    "absmax": src[full + ".absmax"],
                    "quant_map": src[full + ".quant_map"],
                    "nested_absmax": src[full + ".nested_absmax"],
                    "nested_quant_map": src[full + ".nested_quant_map"],
                    "quant_state" + ".bitsandbytes__nf4": src[full + _STATE],
                },
                device=torch.device("cpu"),
            )
            expected = dequantize_4bit(src[full], reference_state)
            ours = dequantize_4bit(shard[short], rebuild_quant_state(short, shard, spec, codes))
            assert expected.dtype == torch.bfloat16 and expected.abs().sum() > 0
            assert torch.equal(expected, ours), full
            checked += 1
    assert checked == 14


def test_cache_inspection_accepts_the_store(work):
    from soup_cli.utils.layer_shard import (
        _discover_safetensors,
        _source_file_components,
        checkpoint_source_components,
        fingerprint_source_files,
        inspect_shard_cache,
    )

    source, out = work
    build_source(source)
    result = import_nf4_checkpoint(source, out)
    components = checkpoint_source_components(
        source, _source_file_components(_discover_safetensors(source)), include_config=False
    )
    cached, reason = inspect_shard_cache(
        out, "bfloat16", fingerprint_source_files(components), components, "nf4", True,
        "serialized",
    )
    assert cached is not None, reason
    assert cached == result.index
    # a native (cpu/cuda) quantisation request must NOT silently reuse serialised bytes
    missed, why = inspect_shard_cache(
        out, "bfloat16", fingerprint_source_files(components), components, "nf4", True, "cuda"
    )
    assert missed is None and "device" in why.lower(), why


def test_verify_import_rereads_store_and_source(work):
    source, out = work
    build_source(source)
    done = import_nf4_checkpoint(source, out, verify=False)
    assert done.verified is False
    again = verify_nf4_import(source, out)
    assert again.verified is True
    assert (again.tensor_count, again.payload_bytes) == (done.tensor_count, done.payload_bytes)


def test_verify_catches_a_flipped_byte(work):
    source, out = work
    build_source(source)
    import_nf4_checkpoint(source, out)
    path = os.path.join(out, "layer_001.safetensors")
    with open(path, "r+b") as handle:
        handle.seek(-3, os.SEEK_END)
        byte = handle.read(1)
        handle.seek(-3, os.SEEK_END)
        handle.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match=r"layer_001\.safetensors: .* differs from the source"):
        verify_nf4_import(source, out)


def test_failed_verification_leaves_no_index(work, monkeypatch):
    source, out = work
    build_source(source)

    def corrupt(path, spans, chunk_bytes):
        real(path, spans, chunk_bytes)
        if path.endswith("layer_000.safetensors"):
            with open(path, "r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                last = handle.read(1)
                handle.seek(-1, os.SEEK_END)
                handle.write(bytes([last[0] ^ 1]))

    real = nf4_serialized._write_shard
    monkeypatch.setattr(nf4_serialized, "_write_shard", corrupt)
    with pytest.raises(ValueError, match="differs from the source"):
        import_nf4_checkpoint(source, out)
    assert not os.path.exists(os.path.join(out, "index.json"))
    assert os.listdir(out) == []


def test_copy_reads_are_bounded_and_nothing_is_mapped(work, monkeypatch):
    source, out = work
    build_source(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("the importer must not map a checkpoint file")

    import mmap

    import safetensors

    monkeypatch.setattr(safetensors, "safe_open", forbidden)
    monkeypatch.setattr(safetensors_torch, "safe_open", forbidden)
    monkeypatch.setattr(mmap, "mmap", forbidden)
    import_nf4_checkpoint(source, out, chunk_bytes=512)

    reads = []

    class Spy:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            reads.append(size)
            return self.handle.read(size)

        def __getattr__(self, name):
            return getattr(self.handle, name)

    span = next(
        item for item in plan_nf4_import(source).shards["extras.safetensors"].values()
    )
    big = max(
        (s for blob in plan_nf4_import(source).shards.values() for s in blob.values()),
        key=lambda s: s.nbytes,
    )
    assert big.nbytes > 512 >= 1 and span is not None
    with open(big.path, "rb") as handle, open(os.path.join(out, "scratch.bin"), "wb") as target:
        nf4_serialized._copy_range(Spy(handle), target, big, 512)
    assert reads and max(reads) <= 512 and len(reads) == -(-big.nbytes // 512)


def test_chunk_bytes_is_bounded(work):
    source, out = work
    build_source(source)
    for bad in (0, -1, nf4_serialized.MAX_CHUNK_BYTES + 1, True, 1.5):
        with pytest.raises(ValueError, match="chunk_bytes"):
            import_nf4_checkpoint(source, out, chunk_bytes=bad)
    assert not os.path.exists(out)


def test_plan_writes_nothing_and_counts(work):
    source, out = work
    build_source(source, "qwen3")
    before = sorted(os.listdir(os.path.dirname(source)))
    plan = plan_nf4_import(source)
    assert sorted(os.listdir(os.path.dirname(source))) == before
    assert plan.tensor_count == 2 * (28 + 4) + 3 + 2


# ----------------------------------------------------------------------
# rejections: the message names the tensor/field and the bad value, nothing is written
# ----------------------------------------------------------------------
def _rename_to_fp4(tensors):
    for key in [k for k in tensors if k.endswith(_STATE)]:
        tensors[key[: -len(_STATE)] + ".quant_state.bitsandbytes__fp4"] = tensors.pop(key)


def _int8(tensors):
    key = "model.layers.0.mlp.up_proj.weight"
    tensors[key + ".SCB"] = torch.zeros(128, dtype=torch.float32)


def _real_int8(tensors):
    """A projection serialised by a real ``Linear8bitLt`` (int8 weight, SCB, weight_format)."""
    import bitsandbytes as bnb

    key = "model.layers.0.mlp.up_proj.weight"
    layer = bnb.nn.Linear8bitLt(64, 128, bias=False, has_fp16_weights=False)
    layer.weight = bnb.nn.Int8Params(
        torch.randn(128, 64, dtype=torch.float16), requires_grad=False, has_fp16_weights=False
    )
    layer.weight = layer.weight._quantize("cpu")
    state = layer.state_dict()
    assert set(state) == {"weight", "SCB", "weight_format"}, set(state)
    for suffix in (".absmax", ".nested_absmax", ".quant_map", ".nested_quant_map", _STATE):
        del tensors[key + suffix]
    tensors[key] = state["weight"].clone()
    tensors[key + ".SCB"] = state["SCB"].clone()
    tensors[key + ".weight_format"] = state["weight_format"].clone()


def _tampered_table_everywhere(tensors):
    """A different but self-consistent NF4 table: every projection carries the same one."""
    for key in [name for name in tensors if name.endswith(".quant_map")]:
        table = tensors[key].clone()
        table[3] += 0.25
        tensors[key] = table


def _drop(suffix):
    def tweak(tensors):
        del tensors[_Q + suffix]

    return tweak


def _negative_scale(tensors):
    scales = tensors[_Q + ".nested_absmax"].clone()
    scales[0] = -1.0
    tensors[_Q + ".nested_absmax"] = scales


def _nan_scale(tensors):
    scales = tensors[_Q + ".nested_absmax"].clone()
    scales[0] = float("nan")
    tensors[_Q + ".nested_absmax"] = scales


def _other_table(tensors):
    key = "model.layers.1.mlp.down_proj.weight.quant_map"
    table = tensors[key].clone()
    table[3] += 0.25
    tensors[key] = table


def _packed_float(tensors):
    tensors[_Q] = tensors[_Q].to(torch.float32)


def _norm_fp16(tensors):
    key = "model.layers.0.input_layernorm.weight"
    tensors[key] = tensors[key].to(torch.float16)


def _bad_nested_offset_json(tensors):
    blob = tensors[_Q + _STATE]
    text = blob.numpy().tobytes().decode()
    text = text.replace('"nested_offset": ', '"nested_offset": NaN, "x": ')
    tensors[_Q + _STATE] = torch.frombuffer(bytearray(text.encode()), dtype=torch.uint8).clone()


def _extra_projection(tensors):
    tensors["model.layers.0.mlp.extra_proj.weight.absmax"] = tensors[_Q].clone()


def _blob_float(tensors):
    tensors[_Q + _STATE] = tensors[_Q + _STATE].to(torch.float32)


def _absmax_float(tensors):
    tensors[_Q + ".absmax"] = tensors[_Q + ".absmax"].to(torch.float32)


def _nested_map_short(tensors):
    tensors[_Q + ".nested_quant_map"] = tensors[_Q + ".nested_quant_map"][:255].clone()


def _nan_nested_table(tensors):
    for key in [name for name in tensors if name.endswith(".nested_quant_map")]:
        table = tensors[key].clone()
        table[0] = float("nan")
        tensors[key] = table


def _huge_offset_literal(tensors):
    text = tensors[_Q + _STATE].numpy().tobytes().decode()
    head = text[: text.index('"nested_offset"')]
    text = head + '"nested_offset": 1e999}'
    tensors[_Q + _STATE] = torch.frombuffer(bytearray(text.encode()), dtype=torch.uint8).clone()


def _mixed_blocksize(tensors):
    """Layer 1's q_proj re-quantised at blocksize 128: valid tensors, a different spec."""
    from bitsandbytes.functional import QuantState, dequantize_4bit, quantize_4bit

    key = "model.layers.1.self_attn.q_proj.weight"
    state = QuantState.from_dict(
        {
            "absmax": tensors[key + ".absmax"],
            "quant_map": tensors[key + ".quant_map"],
            "nested_absmax": tensors[key + ".nested_absmax"],
            "nested_quant_map": tensors[key + ".nested_quant_map"],
            "quant_state.bitsandbytes__nf4": tensors[key + _STATE],
        },
        device=torch.device("cpu"),
    )
    packed, fresh = quantize_4bit(
        dequantize_4bit(tensors[key], state),
        blocksize=128,
        compress_statistics=True,
        quant_type="nf4",
    )
    tensors[key] = packed.contiguous()
    for part, tensor in fresh.as_dict(packed=True).items():
        tensors[f"{key}.{part}"] = tensor.detach().clone().contiguous()


def _nested_absmax_long(tensors):
    tensors[_Q + ".nested_absmax"] = tensors[_Q + ".nested_absmax"].repeat(2)


def _two_spellings(tensors):
    tensors["language_model.model.norm.weight"] = tensors["model.norm.weight"].clone()


REJECTIONS = [
    ("fp4", {"tweak": _rename_to_fp4}, "FP4"),
    ("blob-not-u8", {"tweak": _blob_float}, f"{_Q}{_STATE}: the quant_state blob must be a 1-D U8"),
    ("blob-extra-field", {"tweak": lambda t: edit_state(t, _Q, extra=1)}, "unexpected fields"),
    ("absmax-float", {"tweak": _absmax_float}, f"{_Q}.absmax: stored as F32"),
    (
        "nested-absmax-long",
        {"tweak": _nested_absmax_long},
        f"{_Q}.nested_absmax: stored as F32 [2]",
    ),
    (
        "blob-too-large",
        {"tweak": lambda t: edit_state(t, _Q, pad="x" * 70000)},
        "quant_state blob is",
    ),
    ("two-spellings", {"tweak": _two_spellings}, "spelled two ways"),
    (
        "nested-map-short",
        {"tweak": _nested_map_short},
        f"{_Q}.nested_quant_map: stored as F32 [255]",
    ),
    ("nested-map-nan", {"tweak": _nan_nested_table}, "code table holds a non-finite value"),
    ("offset-huge-literal", {"tweak": _huge_offset_literal}, "is not a finite number"),
    ("spec-differs", {"tweak": _mixed_blocksize}, "quantisation spec differs from other layers"),
    ("int8-key", {"tweak": _int8}, "8-bit"),
    (
        "int8-real",
        {"tweak": _real_int8},
        "model.layers.0.mlp.up_proj.weight.SCB: bitsandbytes 8-bit",
    ),
    (
        "table-not-nf4",
        {"tweak": _tampered_table_everywhere},
        "model.layers.0.mlp.down_proj.weight.quant_map: the stored NF4 code table is not "
        "bitsandbytes' own NF4 table",
    ),
    ("not-nested", {"nested": False}, "not double-quantised"),
    ("blocksize-32", {"blocksize": 32}, "blocksize is 32"),
    (
        "blocksize-100",
        {"tweak": lambda t: edit_state(t, _Q, blocksize=100)},
        "blocksize is 100",
    ),
    (
        "nested-blocksize",
        {"tweak": lambda t: edit_state(t, _Q, nested_blocksize=8)},
        "nested_blocksize is 8",
    ),
    (
        "nested-dtype",
        {"tweak": lambda t: edit_state(t, _Q, nested_dtype="float16")},
        "nested_dtype is 'float16'",
    ),
    ("dtype", {"tweak": lambda t: edit_state(t, _Q, dtype="float16")}, "dtype is 'float16'"),
    ("quant-type", {"tweak": lambda t: edit_state(t, _Q, quant_type="fp4")}, "quant_type is 'fp4'"),
    ("shape", {"tweak": lambda t: edit_state(t, _Q, shape=[64, 65])}, "shape is [64, 65]"),
    (
        "offset-inexact",
        {"tweak": lambda t: edit_state(t, _Q, nested_offset=0.1)},
        "not exactly representable in float32",
    ),
    (
        "offset-overflow",
        {"tweak": lambda t: edit_state(t, _Q, nested_offset=1e300)},
        "does not fit in float32",
    ),
    ("offset-nan", {"tweak": _bad_nested_offset_json}, "non-finite JSON value NaN"),
    ("missing-absmax", {"tweak": _drop(".absmax")}, f"{_Q}.absmax"),
    ("missing-blob", {"tweak": _drop(_STATE)}, f"{_Q}{_STATE}"),
    ("negative-scale", {"tweak": _negative_scale}, f"{_Q}.nested_absmax: nested scale -1.0"),
    ("nan-scale", {"tweak": _nan_scale}, f"{_Q}.nested_absmax: nested scale nan"),
    ("table-differs", {"tweak": _other_table}, "code table differs"),
    ("packed-not-u8", {"tweak": _packed_float}, f"{_Q}: stored as F32"),
    ("norm-fp16", {"tweak": _norm_fp16}, "input_layernorm.weight: stored as F16"),
    (
        "unexpected-key",
        {"tweak": _extra_projection},
        "extra_proj.weight.absmax",
    ),
]


@pytest.mark.parametrize(("case", "kwargs", "message"), REJECTIONS, ids=[c[0] for c in REJECTIONS])
def test_unsupported_sources_are_refused_before_anything_is_written(work, case, kwargs, message):
    source, out = work
    build_source(source, **kwargs)
    with pytest.raises(ValueError) as excinfo:
        import_nf4_checkpoint(source, out)
    assert message in str(excinfo.value), (case, str(excinfo.value))
    assert not os.path.exists(out)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"bnb_4bit_use_double_quant": False}, "bnb_4bit_use_double_quant is False"),
        ({"bnb_4bit_quant_type": "fp4"}, "bnb_4bit_quant_type is 'fp4'"),
        ({"load_in_4bit": False}, "load_in_4bit is False"),
        ({"bnb_4bit_compute_dtype": "float16"}, "bnb_4bit_compute_dtype is 'float16'"),
        ({"bnb_4bit_compute_dtype": "float32"}, "bnb_4bit_compute_dtype is 'float32'"),
        ({"quant_method": "gptq"}, "quant_method is 'gptq'"),
        ({"load_in_8bit": True, "load_in_4bit": False}, "8-bit"),
        ({"bnb_4bit_quant_storage": "bfloat16"}, "bnb_4bit_quant_storage is 'bfloat16'"),
    ],
)
def test_config_quantization_is_checked(work, change, message):
    source, out = work
    build_source(source, quant_config={**_QUANT, **change})
    with pytest.raises(ValueError, match="quantization_config") as excinfo:
        import_nf4_checkpoint(source, out)
    assert message in str(excinfo.value)


def test_config_without_quantization_is_refused(work):
    source, out = work
    build_source(source)
    path = os.path.join(source, "config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    del config["quantization_config"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    with pytest.raises(ValueError, match="no quantization_config"):
        import_nf4_checkpoint(source, out)


def test_a_tensor_in_two_files_is_refused(work):
    source, out = work
    build_source(source, files=2)
    first = os.path.join(source, "model-00001-of-00002.safetensors")
    second = os.path.join(source, "model-00002-of-00002.safetensors")
    one = safetensors_torch.load_file(first)
    two = safetensors_torch.load_file(second)
    name = sorted(one)[0]
    two[name] = one[name]
    safetensors_torch.save_file(two, second)
    with pytest.raises(ValueError, match="appears in both"):
        import_nf4_checkpoint(source, out)


def test_a_truncated_file_is_refused(work):
    source, out = work
    build_source(source)
    path = os.path.join(source, "model.safetensors")
    os.truncate(path, os.path.getsize(path) - 10)
    with pytest.raises(ValueError, match="truncated"):
        import_nf4_checkpoint(source, out)
    assert not os.path.exists(out)


def test_trailing_bytes_are_refused(work):
    source, out = work
    build_source(source)
    with open(os.path.join(source, "model.safetensors"), "ab") as handle:
        handle.write(b"\0" * 16)
    with pytest.raises(ValueError, match="unindexed trailing data"):
        import_nf4_checkpoint(source, out)


def test_a_header_with_a_duplicate_key_is_refused(work):
    source, out = work
    build_source(source)
    entry = '"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    header = ("{" + entry + "," + entry + "}").encode()
    header += b" " * (-len(header) % 8)
    with open(os.path.join(source, "model.safetensors"), "wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header + b"\0")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        import_nf4_checkpoint(source, out)


def test_weight_map_path_escape_is_refused(work):
    source, out = work
    build_source(source, files=2)
    path = os.path.join(source, "model.safetensors.index.json")
    with open(path) as handle:
        weight_map = json.load(handle)["weight_map"]
    # Soup's file selection keeps only the basename, so this names real files; the
    # importer must still refuse a name that is a path
    with open(path, "w") as handle:
        json.dump({"weight_map": {k: "../" + v for k, v in weight_map.items()}}, handle)
    with pytest.raises(ValueError, match="path escape refused"):
        import_nf4_checkpoint(source, out)


def test_output_inside_the_source_is_refused(work):
    source, _out = work
    build_source(source)
    with pytest.raises(ValueError, match="must not contain each other"):
        import_nf4_checkpoint(source, os.path.join(source, "shards"))


def test_a_nonempty_output_directory_is_refused(work):
    source, out = work
    build_source(source)
    os.makedirs(out)
    with open(os.path.join(out, "keep.txt"), "w") as handle:
        handle.write("x")
    with pytest.raises(FileExistsError, match="not empty"):
        import_nf4_checkpoint(source, out)
    assert os.listdir(out) == ["keep.txt"]


@pytest.mark.requires_symlink
def test_symlinked_output_and_config_are_refused(work, tmp_path):
    source, out = work
    build_source(source)
    real_out = tmp_path / "real_out"
    real_out.mkdir()
    os.symlink(real_out, out, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        import_nf4_checkpoint(source, out)
    assert os.listdir(real_out) == []
    config = os.path.join(source, "config.json")
    os.replace(config, str(tmp_path / "config.real"))
    os.symlink(str(tmp_path / "config.real"), config)
    with pytest.raises(ValueError, match="symlink"):
        plan_nf4_import(source)


# ----------------------------------------------------------------------
# guards that need their own scenario
# ----------------------------------------------------------------------
def test_a_shared_weight_must_keep_its_shape_across_layers(work, monkeypatch):
    source, out = work
    key = "model.layers.1.input_layernorm.weight"
    build_source(source, tweak=lambda t: t.__setitem__(key, torch.zeros(65, dtype=torch.bfloat16)))
    real = nf4_serialized._build_expectations

    def widened(root):
        arch, layers, expectations = real(root)
        expectations[key] = (False, (65,))
        return arch, layers, expectations

    monkeypatch.setattr(nf4_serialized, "_build_expectations", widened)
    with pytest.raises(ValueError, match="shape differs from the same weight in other layers"):
        import_nf4_checkpoint(source, out)


def test_a_source_that_changes_during_the_import_is_refused(work, monkeypatch):
    source, out = work
    build_source(source)
    real = nf4_serialized._write_shard

    def write_then_touch(path, spans, chunk_bytes):
        real(path, spans, chunk_bytes)
        with open(os.path.join(source, "model.safetensors"), "ab") as handle:
            handle.write(bytes(1))

    monkeypatch.setattr(nf4_serialized, "_write_shard", write_then_touch)
    with pytest.raises(ValueError, match="changed during the import"):
        import_nf4_checkpoint(source, out)
    assert os.listdir(out) == []


def test_verify_catches_a_missing_tensor_and_a_changed_layout(work):
    source, out = work
    build_source(source)
    import_nf4_checkpoint(source, out)
    path = os.path.join(out, "extras.safetensors")
    shard = safetensors_torch.load_file(path)
    original = dict(shard)
    del shard["model.norm.weight"]
    safetensors_torch.save_file(shard, path)
    with pytest.raises(ValueError, match="tensors in the shard differ from the plan"):
        verify_nf4_import(source, out)
    original["model.norm.weight"] = original["model.norm.weight"].view(torch.float16)
    safetensors_torch.save_file(original, path)
    with pytest.raises(ValueError, match="was written with a different layout"):
        verify_nf4_import(source, out)


def test_verify_catches_an_index_that_does_not_match(work):
    source, out = work
    build_source(source)
    import_nf4_checkpoint(source, out)
    path = os.path.join(out, "index.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["total_params"] += 1
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="does not describe this source"):
        verify_nf4_import(source, out)


_ENTRY = '"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
MALFORMED_HEADERS = [
    (_ENTRY + ',"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}', "leave a gap"),
    (_ENTRY + ',"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}', "overlap"),
    ('"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}', "unsupported stored dtype"),
    ('"a":{"dtype":"U8","shape":[1.5],"data_offsets":[0,1]}', "invalid shape"),
    ('"a":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}', "does not hold 2 bytes"),
]


@pytest.mark.parametrize(("entries", "message"), MALFORMED_HEADERS)
def test_malformed_safetensors_headers_are_refused(work, entries, message):
    source, out = work
    build_source(source)
    header = ("{" + entries + "}").encode()
    header += b" " * (-len(header) % 8)
    with open(os.path.join(source, "model.safetensors"), "wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header + bytes(3))
    with pytest.raises(ValueError, match=message):
        import_nf4_checkpoint(source, out)


def _edit_weight_map(source, edit):
    path = os.path.join(source, "model.safetensors.index.json")
    with open(path) as handle:
        weight_map = json.load(handle)["weight_map"]
    edit(weight_map)
    with open(path, "w") as handle:
        json.dump({"weight_map": weight_map}, handle)


def test_the_weight_map_must_list_every_tensor(work):
    source, out = work
    build_source(source, files=2)
    _edit_weight_map(source, lambda weight_map: weight_map.pop(sorted(weight_map)[0]))
    with pytest.raises(ValueError, match="does not list 1 tensor"):
        import_nf4_checkpoint(source, out)


def test_the_weight_map_must_name_the_file_that_holds_the_tensor(work):
    source, out = work

    def swap(weight_map):
        first = sorted(weight_map)[0]
        weight_map[first] = next(n for n in set(weight_map.values()) if n != weight_map[first])

    build_source(source, files=2)
    _edit_weight_map(source, swap)
    with pytest.raises(ValueError, match="whose header does not hold it"):
        import_nf4_checkpoint(source, out)


def test_a_header_that_claims_more_than_the_file_holds_is_refused(work):
    source, out = work
    build_source(source)
    with open(os.path.join(source, "model.safetensors"), "wb") as handle:
        handle.write(struct.pack("<Q", 2**24 + 1) + b"{}")
    with pytest.raises(ValueError, match="truncated or corrupt safetensors header"):
        import_nf4_checkpoint(source, out)


def test_a_file_that_resolves_outside_the_checkpoint_is_refused(tmp_path, monkeypatch):
    root = os.path.realpath(str(tmp_path / "ckpt"))
    os.makedirs(root)
    with open(os.path.join(root, "config.json"), "w") as handle:
        handle.write("{}")
    elsewhere = str(tmp_path / "elsewhere.json")
    with open(elsewhere, "w") as handle:
        handle.write("{}")
    real = os.path.realpath

    def fake(path, *args, **kwargs):
        return elsewhere if str(path).endswith("config.json") else real(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", fake)
    with pytest.raises(ValueError, match="resolves outside the checkpoint directory"):
        nf4_serialized._contained_file(root, "config.json", "checkpoint")


def test_an_architecture_with_experts_or_layer_buffers_is_refused(work, monkeypatch):
    source, out = work
    build_source(source)
    monkeypatch.setattr("soup_cli.utils.layer_stream.stream_arch_of", lambda config: "deepseek_v3")
    with pytest.raises(ValueError, match="dense decoder architectures only"):
        import_nf4_checkpoint(source, out)
    assert not os.path.exists(out)


# ----------------------------------------------------------------------
# review round: different drives, size guards, read bounds, labels, links, skip modules
# ----------------------------------------------------------------------
def _drives_differ(monkeypatch, *paths):
    """Make ``os.path.commonpath`` fail the way Windows does for two drives, for this pair only."""
    real = os.path.commonpath
    pair = {os.path.normcase(os.path.realpath(path)) for path in paths}

    def commonpath(items):
        items = list(items)
        if len(items) == 2 and {os.path.normcase(item) for item in items} == pair:
            raise ValueError("Paths don't have the same drive")
        return real(items)

    monkeypatch.setattr(os.path, "commonpath", commonpath)


def test_a_source_and_an_output_on_different_drives_are_disjoint(work, monkeypatch):
    source, out = work
    build_source(source)
    _drives_differ(monkeypatch, source, out)
    result = import_nf4_checkpoint(source, out)
    assert result.verified is True


def test_a_source_inside_the_output_is_refused(work, tmp_path):
    source, _out = work
    build_source(source)
    with pytest.raises(ValueError, match="must not contain each other"):
        import_nf4_checkpoint(source, str(tmp_path))


def test_the_huggingface_config_keys_are_read_through_constants():
    assert nf4_serialized.HF_DOUBLE_QUANT_KEY == "bnb_4bit_use_double_quant"
    assert nf4_serialized.HF_QUANT_STORAGE_KEY == "bnb_4bit_quant_storage"


def test_a_drive_mismatch_is_a_refusal_not_a_crash(tmp_path, monkeypatch):
    root = os.path.realpath(str(tmp_path / "ckpt"))
    os.makedirs(root)
    with open(os.path.join(root, "config.json"), "w") as handle:
        handle.write("{}")
    _drives_differ(monkeypatch, root, os.path.join(root, "config.json"))
    with pytest.raises(ValueError, match="resolves outside the checkpoint directory"):
        nf4_serialized._contained_file(root, "config.json", "checkpoint")


def _write_header_file(source, header_text, body=bytes(3)):
    header = header_text.encode()
    header += b" " * (-len(header) % 8)
    with open(os.path.join(source, "model.safetensors"), "wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header + body)


EXTRA_MALFORMED_HEADERS = [
    ('"a":{"dtype":"U8","shape":[2097152,2097152],"data_offsets":[0,1]}', "above the bound"),
    ('"a":{"dtype":"U8","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,1]}', "invalid shape"),
    ('"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}', "invalid shape"),
    ('"a":{"dtype":"U8","shape":"ab","data_offsets":[0,1]}', "invalid shape"),
    ('"a":{"dtype":"U8","shape":5,"data_offsets":[0,1]}', "invalid shape"),
    ('"a":{"dtype":"U8","shape":null,"data_offsets":[0,1]}', "invalid shape"),
    ('"a":{"dtype":"U8","shape":[1]}', "invalid header entry"),
    ('"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"x":1}', "invalid header entry"),
    ('"a":5', "invalid header entry"),
    ('"a":{"dtype":"U8","shape":[1],"data_offsets":[0]}', "invalid data_offsets"),
    ('"a":{"dtype":"U8","shape":[1],"data_offsets":[0,"1"]}', "invalid data_offsets"),
    ('"a":{"dtype":"U8","shape":[1],"data_offsets":[true,1]}', "invalid data_offsets"),
    ('"a":{"dtype":"U8","shape":[1],"data_offsets":[-1,0]}', "does not hold 1 bytes"),
    ('"a":{"dtype":5,"shape":[1],"data_offsets":[0,1]}', "unsupported stored dtype"),
]


@pytest.mark.parametrize(("entries", "message"), EXTRA_MALFORMED_HEADERS)
def test_more_malformed_safetensors_headers_are_refused(work, entries, message):
    source, out = work
    build_source(source)
    _write_header_file(source, "{" + entries + "}")
    with pytest.raises(ValueError, match=message):
        import_nf4_checkpoint(source, out)
    assert not os.path.exists(out)


def test_a_header_that_is_not_an_object_is_refused(work):
    source, out = work
    build_source(source)
    _write_header_file(source, "[1]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        import_nf4_checkpoint(source, out)


def test_a_file_shorter_than_eight_bytes_is_refused(work):
    source, out = work
    build_source(source)
    with open(os.path.join(source, "model.safetensors"), "wb") as handle:
        handle.write(b"abc")
    with pytest.raises(ValueError, match=r"no header length\); re-download"):
        import_nf4_checkpoint(source, out)


def test_a_header_with_too_many_tensors_is_refused(work):
    source, out = work
    build_source(source)
    entry = '"t%d":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}'
    entries = ",".join(entry % number for number in range(nf4_serialized._MAX_TENSORS + 1))
    _write_header_file(source, "{" + entries + "}", body=b"")
    with pytest.raises(ValueError, match="more than 200000 tensors"):
        import_nf4_checkpoint(source, out)


def test_a_deeply_nested_header_is_refused(work):
    source, out = work
    build_source(source)
    _write_header_file(source, '{"a":' + "[" * 100000 + "]" * 100000 + "}")
    with pytest.raises(ValueError, match="JSON is nested too deeply"):
        import_nf4_checkpoint(source, out)


def test_an_oversized_config_is_refused_unread(work):
    source, out = work
    build_source(source)
    with open(os.path.join(source, "config.json"), "wb") as handle:
        handle.seek(nf4_serialized._MAX_HEADER_BYTES)
        handle.write(b" ")
    with pytest.raises(ValueError, match=r"config\.json is larger than 16777216 bytes"):
        import_nf4_checkpoint(source, out)


def test_an_oversized_index_file_is_refused_unread(tmp_path):
    path = tmp_path / "big.json"
    with open(path, "wb") as handle:
        handle.seek(nf4_serialized._MAX_HEADER_BYTES)
        handle.write(b" ")
    with pytest.raises(ValueError, match="index is larger than 16777216 bytes"):
        nf4_serialized._read_json_file(str(path), "index")


class _ReadSpy:
    """A file handle that records the size of every ``read()`` made through it."""

    def __init__(self, handle, sizes):
        self._handle = handle
        self._sizes = sizes

    def read(self, size=-1):
        self._sizes.append(size)
        return self._handle.read(size)

    def __getattr__(self, name):
        return getattr(self._handle, name)


def test_every_stage_of_the_import_uses_the_callers_chunk_size(work, monkeypatch):
    source, out = work
    build_source(source)
    seen = {"write": [], "copy": [], "compare": []}
    reads = {"copy": [], "compare": []}
    real_write = nf4_serialized._write_shard
    real_copy = nf4_serialized._copy_range
    real_compare = nf4_serialized._compare

    def write(path, spans, chunk_bytes):
        seen["write"].append(chunk_bytes)
        return real_write(path, spans, chunk_bytes)

    def copy(source_handle, target, span, chunk_bytes):
        seen["copy"].append(chunk_bytes)
        return real_copy(_ReadSpy(source_handle, reads["copy"]), target, span, chunk_bytes)

    def compare(stored, start, span, chunk_bytes, file_name):
        seen["compare"].append(chunk_bytes)
        spy = _ReadSpy(stored, reads["compare"])
        return real_compare(spy, start, span, chunk_bytes, file_name)

    monkeypatch.setattr(nf4_serialized, "_write_shard", write)
    monkeypatch.setattr(nf4_serialized, "_copy_range", copy)
    monkeypatch.setattr(nf4_serialized, "_compare", compare)
    import_nf4_checkpoint(source, out, chunk_bytes=512)
    for stage, chunks in seen.items():
        assert chunks and set(chunks) == {512}, (stage, set(chunks))
    for stage, sizes in reads.items():
        assert sizes and max(sizes) <= 512, (stage, max(sizes))


def test_untrusted_names_in_messages_are_made_safe_for_the_terminal(work):
    source, out = work
    build_source(source)
    entry = '"a\\u001b[bold]x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    _write_header_file(source, "{" + entry + "," + entry + "}")
    with pytest.raises(ValueError, match="duplicate JSON key") as excinfo:
        import_nf4_checkpoint(source, out)
    message = str(excinfo.value)
    assert "\x1b" not in message and "a\\[bold]x" in message, message


def test_unexpected_quant_state_fields_are_made_safe_for_the_terminal(work):
    source, out = work
    build_source(source, tweak=lambda t: edit_state(t, _Q, **{"[bold]x": 1}))
    with pytest.raises(ValueError, match="unexpected fields") as excinfo:
        import_nf4_checkpoint(source, out)
    assert "\\[bold]x" in str(excinfo.value)


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("fp4", {"tweak": _rename_to_fp4}),
        ("int8", {"tweak": _real_int8}),
        ("not-nested", {"nested": False}),
    ],
)
def test_quantisation_refusals_say_what_to_use_instead(work, case, kwargs):
    source, out = work
    build_source(source, **kwargs)
    with pytest.raises(ValueError, match="training.quantization: 4bit"):
        import_nf4_checkpoint(source, out)


# ---- Hugging Face cache snapshots: links into the repo's blob store ------------------------
def make_hf_cache(tmp_path, source, *, shared_store=False):
    """Lay ``source`` out the way huggingface_hub does: blobs/<etag> and snapshots/<sha>/<name>."""
    import shutil

    hub = tmp_path / "hub"
    repo = hub / "models--org--tiny"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for index, name in enumerate(sorted(os.listdir(source))):
        blob = blobs / f"{index:040x}"
        shutil.copy2(os.path.join(source, name), blob)
        if shared_store and name.endswith(".safetensors"):
            store = hub / "blobs"
            store.mkdir(exist_ok=True)
            (store / ".huggingface-shared-blobs").write_text("")
            os.replace(blob, store / f"xet{index}")
            os.symlink(os.path.join("..", "..", "blobs", f"xet{index}"), blob)
        os.symlink(os.path.join("..", "..", "blobs", blob.name), snapshot / name)
    return repo, snapshot


@pytest.mark.requires_symlink
@pytest.mark.parametrize(("shared_store", "files"), [(False, 1), (True, 2)])
def test_a_hugging_face_snapshot_of_links_is_imported(tmp_path, shared_store, files):
    source = str(tmp_path / "src")
    out = str(tmp_path / "out")
    build_source(source, files=files)
    _repo, snapshot = make_hf_cache(tmp_path, source, shared_store=shared_store)
    result = import_nf4_checkpoint(str(snapshot), out)
    assert result.verified is True and result.layer_count == 2
    assert result.tensor_count == 2 * (28 + 2) + 3 + 2
    # the cache identity names the file as the snapshot does, not by its blob's etag
    weights = sorted(name for name in os.listdir(source) if name.endswith(".safetensors"))
    assert [name for name, _size, _mtime in result.index.source_files] == weights
    assert verify_nf4_import(str(snapshot), out).verified is True


@pytest.mark.requires_symlink
@pytest.mark.parametrize("victim", ["config.json", "model.safetensors"])
def test_a_snapshot_link_that_escapes_the_cache_is_refused(tmp_path, victim):
    source = str(tmp_path / "src")
    build_source(source)
    _repo, snapshot = make_hf_cache(tmp_path, source)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"{}")
    link = snapshot / victim
    link.unlink()
    os.symlink(str(outside), link)
    with pytest.raises(ValueError, match="points outside the Hugging Face blob store"):
        import_nf4_checkpoint(str(snapshot), str(tmp_path / "out"))
    assert not os.path.exists(tmp_path / "out")


@pytest.mark.requires_symlink
def test_a_blob_entry_that_links_outside_the_cache_is_refused(tmp_path):
    source = str(tmp_path / "src")
    build_source(source)
    repo, snapshot = make_hf_cache(tmp_path, source)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"{}")
    blob = os.path.realpath(snapshot / "config.json")
    os.unlink(blob)
    os.symlink(str(outside), blob)
    with pytest.raises(ValueError, match="points outside the Hugging Face blob store"):
        import_nf4_checkpoint(str(snapshot), str(tmp_path / "out"))


@pytest.mark.requires_symlink
def test_a_linked_blob_store_is_refused_by_name(tmp_path):
    import shutil

    source = str(tmp_path / "src")
    build_source(source)
    repo, snapshot = make_hf_cache(tmp_path, source)
    moved = tmp_path / "elsewhere"
    shutil.move(str(repo / "blobs"), str(moved))
    os.symlink(str(moved), str(repo / "blobs"), target_is_directory=True)
    with pytest.raises(ValueError, match="is a link and is not followed"):
        import_nf4_checkpoint(str(snapshot), str(tmp_path / "out"))


@pytest.mark.requires_symlink
def test_a_link_outside_a_snapshot_directory_is_still_refused(tmp_path):
    source = str(tmp_path / "src")
    build_source(source)
    real = tmp_path / "config.real"
    os.replace(os.path.join(source, "config.json"), real)
    os.symlink(str(real), os.path.join(source, "config.json"))
    with pytest.raises(ValueError, match="is not a Hugging Face cache snapshot"):
        plan_nf4_import(source)


# ---- quantization_config.llm_int8_skip_modules --------------------------------------------
@pytest.mark.parametrize("skipped", [["lm_head"], ["vision_tower", "model.embed_tokens"], []])
def test_skip_modules_that_cover_no_quantised_projection_are_harmless(work, skipped):
    source, _out = work
    build_source(source, quant_config={**_QUANT, "llm_int8_skip_modules": skipped})
    assert plan_nf4_import(source).tensor_count == 2 * (28 + 2) + 3 + 2


@pytest.mark.parametrize(
    "skipped",
    [["model.layers.0.mlp.gate_proj"], ["lm_head", "gate_proj"], ["layers.1.self_attn.q_proj"]],
)
def test_skip_modules_that_cover_a_quantised_projection_are_refused(work, skipped):
    source, out = work
    build_source(source, quant_config={**_QUANT, "llm_int8_skip_modules": skipped})
    with pytest.raises(ValueError, match="llm_int8_skip_modules lists") as excinfo:
        import_nf4_checkpoint(source, out)
    message = str(excinfo.value)
    assert skipped[-1] in message and "covers the quantised projection model.layers." in message
    assert not os.path.exists(out)


@pytest.mark.parametrize("skipped", ["lm_head", [5], [""], {"a": 1}, ["x"] * 4097])
def test_skip_modules_must_be_a_list_of_names(work, skipped):
    source, out = work
    build_source(source, quant_config={**_QUANT, "llm_int8_skip_modules": skipped})
    with pytest.raises(ValueError, match="llm_int8_skip_modules must be a list of module names"):
        import_nf4_checkpoint(source, out)


def test_a_missing_compute_dtype_is_refused(work):
    source, out = work
    config = {key: value for key, value in _QUANT.items() if key != "bnb_4bit_compute_dtype"}
    build_source(source, quant_config=config)
    with pytest.raises(ValueError, match="bnb_4bit_compute_dtype is None"):
        import_nf4_checkpoint(source, out)
