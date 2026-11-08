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
    ("model_type", "tie", "files"),
    [("llama", False, 1), ("qwen3", False, 2), ("mistral", False, 1), ("llama", True, 1)],
)
def test_import_is_byte_for_byte_and_decodes_like_bitsandbytes(work, model_type, tie, files):
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
    build_source(source, model_type, tie=tie, files=files)
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


REJECTIONS = [
    ("fp4", {"tweak": _rename_to_fp4}, "FP4"),
    ("int8-key", {"tweak": _int8}, "8-bit"),
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


def test_symlinked_output_and_config_are_refused(work, tmp_path):
    source, out = work
    build_source(source)
    real_out = tmp_path / "real_out"
    real_out.mkdir()
    try:
        os.symlink(real_out, out, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this host")
    with pytest.raises(ValueError, match="symlink"):
        import_nf4_checkpoint(source, out)
    assert os.listdir(real_out) == []
    config = os.path.join(source, "config.json")
    os.replace(config, str(tmp_path / "config.real"))
    os.symlink(str(tmp_path / "config.real"), config)
    with pytest.raises(ValueError, match="symlink"):
        plan_nf4_import(source)
