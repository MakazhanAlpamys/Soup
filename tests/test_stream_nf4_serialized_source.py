"""Layer streaming trains on a pre-quantised bitsandbytes NF4 source (``unsloth/*-bnb-4bit``).

#2103 made layer streaming refuse every quantised source by name, bitsandbytes included:
the float sharder would have streamed the packed nibbles as if they were weights. This
file pins the one narrow exception that followed (maintainer ruling "ak"): a source that
is double-quantised NF4 with bf16 compute, under ``training.quantization: 4bit``, is
imported byte for byte (``utils/nf4_serialized.py``) instead of being re-sharded, and
trains. Everything else keeps its #2103 refusal and wording: 8-bit, FP4, a source that
is not double-quantised, GPTQ, AWQ, compressed-tensors, and the SAME NF4 source under
``quantization: none``.

Sections:

- the config-alone gate (``utils/fp8_block.py``), which decides before any weight is read;
- the sharder (``utils/layer_shard.py``): dispatch to the importer, cache reuse across
  run devices, a changed source, a failed import;
- the disk pre-flight, which sizes a serialised source from its payload bytes;
- the trainer setup, end to end on the CPU: a real ``SFTTrainerWrapper.setup`` over a
  serialised tiny Llama, a step with a finite loss, and the streamed model's logits and
  LoRA gradients equal to the RESIDENT bitsandbytes model loaded from the same files.
"""

from __future__ import annotations

import json
import os
import struct
import types

import pytest

_TAIL = "Layer streaming cannot dequantise it"
_POINTER = "Point training.base at a bf16 or FP8-block copy of this model"

#: ``quantization_config`` of an ``unsloth/*-bnb-4bit`` repository.
NESTED_NF4 = {
    "quant_method": "bitsandbytes",
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_storage": "uint8",
    "bnb_4bit_compute_dtype": "bfloat16",
}

#: The admitted architectures: every streamed one the importer does not exclude.
ADMITTED_ARCHS = (
    "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3_text", "phi", "phi3",
)

_SUFFIXES = tuple(
    [f"self_attn.{name}_proj.weight" for name in "qkvo"]
    + [f"mlp.{name}_proj.weight" for name in ("gate", "up", "down")]
)


def _without(config, *keys):
    return {key: value for key, value in config.items() if key not in keys}


def _gate(config, *, arch="llama", allow=False):
    from soup_cli.utils.fp8_block import check_source_quantization

    return check_source_quantization(config, arch=arch, allow_serialized_nf4=allow)


def _refusal(config, *, arch="llama", allow=False) -> str:
    from soup_cli.utils.fp8_block import UnsupportedQuantizedSourceError

    with pytest.raises(UnsupportedQuantizedSourceError) as caught:
        _gate(config, arch=arch, allow=allow)
    return str(caught.value)


# ==========================================================================
# the config-alone gate
# ==========================================================================
@pytest.mark.parametrize("arch", ADMITTED_ARCHS)
def test_a_nested_bf16_nf4_config_is_admitted_for_a_4bit_run(arch):
    from soup_cli.utils.fp8_block import is_serialized_nf4_config

    assert _gate(NESTED_NF4, arch=arch, allow=True) is None
    assert is_serialized_nf4_config(NESTED_NF4, arch=arch) is True


def test_a_config_without_the_storage_key_is_admitted_as_transformers_reads_it():
    """``bnb_4bit_quant_storage`` defaults to uint8; older repositories do not write it."""
    config = _without(NESTED_NF4, "bnb_4bit_quant_storage")
    assert _gate(config, allow=True) is None


@pytest.mark.parametrize("value", [None, "gptq", ["bitsandbytes"], 4, {}])
def test_a_value_that_is_not_a_bitsandbytes_mapping_is_not_a_serialized_nf4_config(value):
    from soup_cli.utils.fp8_block import is_serialized_nf4_config

    assert is_serialized_nf4_config(value, arch="llama") is False


def test_the_same_config_is_still_refused_in_its_old_words_without_the_4bit_run():
    """The default is the #2103 refusal; the exception is opt-in by the 4bit run."""
    message = _refusal(NESTED_NF4)
    assert "declares quant_method 'bitsandbytes' (4-bit)" in message
    assert _TAIL in message and _POINTER in message
    assert _refusal(NESTED_NF4, allow=False) == message


_STILL_REFUSED = {
    "int8": {"quant_method": "bitsandbytes", "load_in_8bit": True},
    "int8-beside-the-nf4-fields": dict(NESTED_NF4, load_in_4bit=False, load_in_8bit=True),
    "both-bit-widths-set": dict(NESTED_NF4, load_in_8bit=True),
    "fp4": dict(NESTED_NF4, bnb_4bit_quant_type="fp4"),
    "nf4-spelled-upper-case": dict(NESTED_NF4, bnb_4bit_quant_type="NF4"),
    "no-quant-type": _without(NESTED_NF4, "bnb_4bit_quant_type"),
    "not-nested": dict(NESTED_NF4, bnb_4bit_use_double_quant=False),
    "nested-flag-absent": _without(NESTED_NF4, "bnb_4bit_use_double_quant"),
    "nested-flag-an-int": dict(NESTED_NF4, bnb_4bit_use_double_quant=1),
    "float16-compute": dict(NESTED_NF4, bnb_4bit_compute_dtype="float16"),
    "float32-compute": dict(NESTED_NF4, bnb_4bit_compute_dtype="float32"),
    "compute-absent": _without(NESTED_NF4, "bnb_4bit_compute_dtype"),
    "bf16-storage": dict(NESTED_NF4, bnb_4bit_quant_storage="bfloat16"),
    "four-bit-flag-off": dict(NESTED_NF4, load_in_4bit=False),
    "four-bit-flag-a-string": dict(NESTED_NF4, load_in_4bit="true"),
    "no-method": _without(NESTED_NF4, "quant_method"),
    "method-in-another-case": dict(NESTED_NF4, quant_method="BitsAndBytes"),
    "method-bnb-4bit-spelling": dict(NESTED_NF4, quant_method="bitsandbytes_4bit"),
    "gptq": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    "awq": {"quant_method": "awq", "bits": 4, "version": "gemm"},
    "compressed-tensors": {"quant_method": "compressed-tensors", "format": "pack-quantized"},
}


@pytest.mark.parametrize("spelling", sorted(_STILL_REFUSED))
def test_every_other_quantised_config_is_refused_in_its_old_words_even_for_a_4bit_run(spelling):
    config = _STILL_REFUSED[spelling]
    old = _refusal(config, allow=False)
    assert _refusal(config, allow=True) == old
    assert _TAIL in old and _POINTER in old


@pytest.mark.parametrize("arch", ["deepseek_v3", "qwen4_exp"])
def test_an_architecture_the_importer_excludes_keeps_the_refusal(arch):
    old = _refusal(NESTED_NF4, arch=arch, allow=False)
    assert _refusal(NESTED_NF4, arch=arch, allow=True) == old


def test_the_refused_architectures_are_the_importers():
    """One list in two places would drift: the gate must never admit what the importer
    refuses by architecture, so they are asserted equal."""
    from soup_cli.utils import nf4_serialized
    from soup_cli.utils.fp8_block import SERIALIZED_NF4_REFUSED_ARCHS

    assert tuple(SERIALIZED_NF4_REFUSED_ARCHS) == tuple(nf4_serialized._UNSUPPORTED_ARCHS)


def test_a_transformers_bitsandbytes_config_object_is_read_like_the_dict():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    nested = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    assert _gate(nested, allow=True) is None
    # transformers' own defaults are fp4, no double quantisation, float32 compute
    plain = transformers.BitsAndBytesConfig(load_in_4bit=True)
    assert "'bitsandbytes' (4-bit)" in _refusal(plain, allow=True)


def test_the_gate_decides_from_the_config_alone_before_any_header(tmp_path):
    """Admitting must not read a weight, and refusing must not either: a directory whose
    only file is garbage gets past (or is stopped by) the config gate without touching it."""
    from soup_cli.utils.layer_shard import QUANT_NF4, QUANT_NONE, shard_checkpoint

    directory = tmp_path / "w"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"\xff" * 64)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization_config": NESTED_NF4}), encoding="utf-8"
    )
    with pytest.raises(ValueError) as refused:
        shard_checkpoint(
            str(directory), str(tmp_path / "s0"), dtype="bfloat16", arch="llama",
            quant=QUANT_NONE,
        )
    assert _TAIL in str(refused.value) and "'bitsandbytes' (4-bit)" in str(refused.value)
    # Admitted: the gate passes and the garbage file is what fails, in the IMPORTER's words,
    # not in the gate's.
    with pytest.raises(Exception) as admitted:
        shard_checkpoint(
            str(directory), str(tmp_path / "s1"), dtype="bfloat16", arch="llama",
            quant=QUANT_NF4, quant_suffixes=_SUFFIXES,
        )
    assert _TAIL not in str(admitted.value)


# ==========================================================================
# a real serialised checkpoint
# ==========================================================================
def _build(tmp_path, model_type="llama", name="src", **kwargs):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("bitsandbytes.functional")
    pytest.importorskip("safetensors.torch")
    from tests.test_nf4_serialized import build_source

    root = str(tmp_path / name)
    build_source(root, model_type, **kwargs)
    return root


def _shard(source, out, **overrides):
    from soup_cli.utils.layer_shard import QUANT_NF4, shard_checkpoint

    arguments = dict(
        dtype="bfloat16", arch="llama", quant=QUANT_NF4, quant_suffixes=_SUFFIXES,
        double_quant=True,
    )
    arguments.update(overrides)
    return shard_checkpoint(source, str(out), **arguments)


def _files(directory):
    return sorted(os.listdir(directory))


def _bytes(directory):
    contents = {}
    for name in _files(directory):
        with open(os.path.join(directory, name), "rb") as handle:
            contents[name] = handle.read()
    return contents


def test_the_sharder_imports_a_serialised_source_byte_for_byte(tmp_path):
    from soup_cli.utils.nf4_serialized import import_nf4_checkpoint

    source = _build(tmp_path)
    index = _shard(source, tmp_path / "cache", quant_device="cuda")
    direct = tmp_path / "direct"
    expected = import_nf4_checkpoint(source, str(direct)).index

    assert index == expected
    assert (index.quant, index.double_quant, index.quant_device) == ("nf4", True, "serialized")
    assert index.dtype == "bfloat16" and index.arch == "llama"
    # the very bytes the importer writes, and nothing else: no staging folder is left over
    assert _bytes(tmp_path / "cache") == _bytes(direct)


def test_the_importer_and_the_sharder_agree_on_the_serialized_device_name():
    from soup_cli.utils.layer_shard import SERIALIZED_QUANT_DEVICE

    assert SERIALIZED_QUANT_DEVICE == "serialized"


def test_without_a_4bit_run_the_same_source_is_still_refused_and_nothing_is_written(tmp_path):
    from soup_cli.utils.layer_shard import QUANT_NONE

    source = _build(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError) as refused:
        _shard(source, cache, quant=QUANT_NONE, quant_suffixes=())
    assert str(refused.value) == _refusal(NESTED_NF4)
    assert not cache.exists() or _files(cache) == []


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"dtype": "float32"}, "bfloat16"),
        ({"dtype": "float16"}, "bfloat16"),
        ({"double_quant": False}, "double"),
        ({"stripe_roots": ("stripe-a",)}, "stripe"),
    ],
    ids=["float32", "float16", "no-double-quant", "stripe-roots"],
)
def test_a_request_the_store_cannot_serve_is_refused_by_name_before_a_byte_is_written(
    tmp_path, overrides, needle
):
    source = _build(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match=needle):
        _shard(source, cache, **overrides)
    assert not cache.exists() or _files(cache) == []


def test_quant_suffixes_that_disagree_with_the_import_commit_nothing(tmp_path):
    source = _build(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="quant_suffixes"):
        _shard(source, cache, quant_suffixes=_SUFFIXES[:-1])
    assert not cache.exists() or _files(cache) == []


def test_a_second_run_reuses_the_cache_whatever_device_it_runs_on(tmp_path, monkeypatch):
    """The index says quant_device 'serialized'; a run passes its own device ('cuda',
    'cpu'). Reading that as a changed device would re-import on every run."""
    from soup_cli.utils import nf4_serialized

    source = _build(tmp_path)
    cache = tmp_path / "cache"
    first = _shard(source, cache, quant_device="cpu")
    before = _bytes(cache)

    def _must_not_import(*_args, **_kwargs):
        raise AssertionError("the cache was imported again")

    monkeypatch.setattr(nf4_serialized, "import_nf4_checkpoint", _must_not_import)
    for device in ("cuda", "cuda:0", "cpu", "mps"):
        assert _shard(source, cache, quant_device=device) == first
    assert _bytes(cache) == before
    assert not os.path.exists(cache / "index.json.resharding")


def _cache_check(cache, source, device):
    from soup_cli.utils.layer_shard import (
        _discover_safetensors,
        _fingerprint_components,
        _source_file_components,
        inspect_shard_cache,
    )

    components = _source_file_components(_discover_safetensors(source))
    return inspect_shard_cache(
        str(cache), "bfloat16", _fingerprint_components(components), components, "nf4", True,
        device,
    )


def _quarantine_check(cache, source, device):
    from soup_cli.utils.layer_shard import (
        _content_change_requires_index_quarantine,
        _discover_safetensors,
        _fingerprint_components,
        _source_file_components,
    )

    components = _source_file_components(_discover_safetensors(source))
    return _content_change_requires_index_quarantine(
        str(cache), dtype="bfloat16", fingerprint=_fingerprint_components(components),
        quant="nf4", double_quant=True, quant_device=device, external_mode="",
    )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_a_serialized_index_matches_any_run_device_in_the_cache_check(tmp_path, device):
    source = _build(tmp_path)
    cache = tmp_path / "cache"
    _shard(source, cache)
    reusable, reason = _cache_check(cache, source, device)
    assert reusable is not None, reason
    assert _quarantine_check(cache, source, device) is False


def test_only_a_serialized_index_ignores_the_run_device(tmp_path):
    """A cache Soup quantised ON a device still depends on it (CPU and CUDA disagree on the
    nested statistics): the leniency is for the importer's index and nothing else."""
    from dataclasses import replace

    from soup_cli.utils.layer_shard import _atomic_write_index, read_shard_index

    source = _build(tmp_path)
    cache = tmp_path / "cache"
    _shard(source, cache)
    _atomic_write_index(replace(read_shard_index(str(cache)), quant_device="cpu"), str(cache))
    reusable, reason = _cache_check(cache, source, "cuda")
    assert reusable is None and "quantization device changed" in reason
    assert _quarantine_check(cache, source, "cuda") is True
    reusable, _reason = _cache_check(cache, source, "cpu")
    assert reusable is not None
    assert _quarantine_check(cache, source, "cpu") is False


def test_a_changed_source_is_imported_again_over_the_stale_cache(tmp_path):
    from soup_cli.utils.layer_shard import _discover_safetensors, read_shard_index

    source = _build(tmp_path)
    cache = tmp_path / "cache"
    first = _shard(source, cache)
    path = _discover_safetensors(source)[0]
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    second = _shard(source, cache)
    assert second.source_fingerprint != first.source_fingerprint
    assert read_shard_index(str(cache)) == second
    assert (second.quant, second.quant_device) == ("nf4", "serialized")
    assert not os.path.exists(cache / "index.json.resharding")
    assert not [name for name in _files(cache) if name.startswith(".")]
    reusable, reason = _cache_check(cache, source, "cuda")
    assert reusable is not None, reason


def test_a_failed_import_changes_neither_the_previous_cache_nor_leaves_a_staging_folder(
    tmp_path, monkeypatch
):
    from soup_cli.utils import nf4_serialized
    from soup_cli.utils.layer_shard import _discover_safetensors

    source = _build(tmp_path)
    cache = tmp_path / "cache"
    _shard(source, cache)
    before = _bytes(cache)
    path = _discover_safetensors(source)[0]
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

    real = nf4_serialized.import_nf4_checkpoint

    def _import_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(nf4_serialized, "import_nf4_checkpoint", _import_then_fail)
    with pytest.raises(OSError, match="disk full"):
        _shard(source, cache)
    assert _bytes(cache) == before


def test_an_interrupted_install_leaves_a_cache_nothing_will_reuse(tmp_path, monkeypatch):
    """The index is the commit point: it goes last, and the old one is taken out of play
    before the first shard is replaced, so a crash between the two never leaves an index
    describing files of two imports."""
    from soup_cli.utils.nf4_serialized import import_nf4_checkpoint

    source = _build(tmp_path)
    cache = tmp_path / "cache"
    _shard(source, cache)
    real_replace = os.replace
    root = os.path.realpath(cache)

    def _flaky(src, dst, *args, **kwargs):
        # only the install into the cache folder; the importer renames inside its staging one
        if (
            os.path.realpath(os.path.dirname(dst)) == root
            and os.path.basename(dst) == "layer_001.safetensors"
        ):
            raise OSError("handle held open")
        return real_replace(src, dst, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", _flaky)
        with pytest.raises(OSError, match="handle held open"):
            _shard(source, cache, force=True)
    assert not os.path.exists(cache / "index.json")
    assert os.path.exists(cache / "index.json.resharding")
    assert not [name for name in _files(cache) if name.startswith(".soup-nf4-import.")]
    reusable, reason = _cache_check(cache, source, "cuda")
    assert reusable is None and "did not finish" in reason

    repaired = _shard(source, cache)
    direct = tmp_path / "direct"
    assert repaired == import_nf4_checkpoint(source, str(direct)).index
    assert _bytes(cache) == _bytes(direct)


def test_a_source_the_importer_refuses_is_refused_with_its_words_and_writes_nothing(tmp_path):
    from tests.test_nf4_serialized import edit_state

    source = _build(tmp_path, tweak=lambda tensors: edit_state(
        tensors, "model.layers.0.self_attn.q_proj.weight", dtype="float16"
    ))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError) as refused:
        _shard(source, cache)
    assert "model.layers.0.self_attn.q_proj.weight" in str(refused.value)
    assert not cache.exists() or _files(cache) == []


# ==========================================================================
# the disk pre-flight sizes a serialised source from its payload
# ==========================================================================
def _source_shards(source):
    return sorted(
        os.path.join(source, name) for name in os.listdir(source) if name.endswith(".safetensors")
    )


def _estimate(source, **overrides):
    from soup_cli.utils.layer_shard import QUANT_NF4, estimate_shard_cache_bytes

    with open(os.path.join(source, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    arguments = dict(
        dtype="bfloat16", arch="llama", quant=QUANT_NF4, quant_suffixes=_SUFFIXES,
        double_quant=True, quantization_config=config["quantization_config"],
    )
    arguments.update(overrides)
    return estimate_shard_cache_bytes(_source_shards(source), **arguments)


@pytest.mark.parametrize(
    ("model_type", "tie", "files"),
    [("llama", False, 1), ("qwen3", False, 2), ("llama", True, 1)],
    ids=["llama", "qwen3-two-files", "llama-tied"],
)
def test_the_estimate_is_the_payload_the_importer_writes(tmp_path, model_type, tie, files):
    from soup_cli.utils.nf4_serialized import import_nf4_checkpoint

    source = _build(tmp_path, model_type, tie=tie, files=files)
    estimate = _estimate(source, arch=model_type)
    result = import_nf4_checkpoint(source, str(tmp_path / "out"))

    assert sum(estimate.layer_bytes) + estimate.other_bytes == result.payload_bytes
    assert len(estimate.layer_bytes) == result.layer_count
    assert estimate.code_table_bytes == (16 + 256) * 4
    out = tmp_path / "out"
    on_disk = sum(os.path.getsize(out / name) for name in os.listdir(out))
    # headers and the index are bounded from above, never under-counted
    assert estimate.total_bytes >= on_disk
    headers = 0
    for name in os.listdir(out):
        if name.endswith(".safetensors"):
            with open(out / name, "rb") as handle:
                headers += 8 + struct.unpack("<Q", handle.read(8))[0]
    assert estimate.header_bytes >= headers
    assert estimate.index_bytes >= os.path.getsize(out / "index.json")


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [({"dtype": "float32"}, "bfloat16"), ({"double_quant": False}, "double")],
    ids=["float32", "no-double-quant"],
)
def test_the_estimate_refuses_a_request_the_store_cannot_serve(tmp_path, overrides, needle):
    source = _build(tmp_path)
    with pytest.raises(ValueError, match=needle):
        _estimate(source, **overrides)


def test_the_estimate_without_a_4bit_run_is_the_old_refusal(tmp_path):
    from soup_cli.utils.layer_shard import QUANT_NONE

    source = _build(tmp_path)
    with pytest.raises(ValueError) as refused:
        _estimate(source, quant=QUANT_NONE, quant_suffixes=())
    assert str(refused.value) == _refusal(NESTED_NF4)


def test_the_estimate_refuses_a_source_with_no_layers_a_gap_or_too_many():
    from soup_cli.utils.layer_shard import _estimate_serialized_nf4

    def refusal(tensors):
        with pytest.raises(ValueError) as caught:
            _estimate_serialized_nf4(tensors, source_files=1)
        return str(caught.value)

    assert "no decoder layer weights" in refusal({"model.embed_tokens.weight": ("BF16", (4, 4))})
    assert "not contiguous" in refusal({"model.layers.1.mlp.down_proj.weight": ("BF16", (4, 4))})
    assert "too many decoder layers" in refusal(
        {"model.layers.600.mlp.down_proj.weight": ("BF16", (4, 4))}
    )
    assert "stored type 'C64'" in refusal(
        {"model.layers.0.mlp.down_proj.weight.absmax": ("C64", (2,))}
    )


@pytest.mark.parametrize("extra", [".SCB", ".quant_state.bitsandbytes__fp4"])
def test_the_estimate_refuses_a_tensor_that_is_not_in_the_nf4_serialisation(tmp_path, extra):
    """The pre-flight runs before the weights are copied: a bitsandbytes tensor outside the
    NF4 vocabulary is refused here in the shared words rather than sized."""
    import torch
    from safetensors.torch import load_file, save_file

    source = _build(tmp_path)
    path = _source_shards(source)[0]
    tensors = load_file(path)
    tensors["model.layers.0.self_attn.q_proj.weight" + extra] = torch.zeros(4, dtype=torch.uint8)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="bitsandbytes"):
        _estimate(source)


# ==========================================================================
# the trainer setup
# ==========================================================================
def _drive_gate(tmp_path, monkeypatch, quantization):
    """Run the real ``_setup_streaming_transformers`` up to the weight resolve."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    class _ResolvedError(Exception):
        pass

    def _resolve(*_args, **_kwargs):
        raise _ResolvedError

    model_cfg = types.SimpleNamespace(
        model_type="llama", hidden_size=64, num_hidden_layers=2, vocab_size=128,
        intermediate_size=128, quantization_config=dict(NESTED_NF4),
    )
    monkeypatch.setattr(
        "soup_cli.trainer.stream_setup.StreamingSetupMixin._stream_meta_skeleton",
        classmethod(lambda cls, base, *, dtype, quant: types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.quantised_layer_suffixes",
        lambda _model: frozenset(_SUFFIXES),
    )
    monkeypatch.setattr(
        "soup_cli.utils.moe.resolve_moe_lora_targets",
        lambda _probe, _tcfg, targets, **_kwargs: targets,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _device: "bfloat16"
    )
    try:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_resolve, arch="llama", model_cfg=model_cfg,
            tcfg_overrides={"quantization": quantization},
        )
    except _ResolvedError:
        return "reached the weight resolve"
    return "finished"


def test_the_trainer_admits_the_source_under_4bit_before_resolving_weights(
    tmp_path, monkeypatch
):
    assert _drive_gate(tmp_path, monkeypatch, "4bit") == "reached the weight resolve"


def test_the_trainer_still_refuses_the_same_source_under_quantization_none(
    tmp_path, monkeypatch
):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a base that cannot stream")

    model_cfg = types.SimpleNamespace(
        model_type="llama", hidden_size=64, num_hidden_layers=2, vocab_size=128,
        intermediate_size=128, quantization_config=dict(NESTED_NF4),
    )
    with pytest.raises(ValueError) as refused:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="llama",
            model_cfg=model_cfg, tcfg_overrides={"quantization": "none"},
        )
    assert str(refused.value) == _refusal(NESTED_NF4)


def test_the_schema_still_refuses_8bit_streaming_before_any_source_is_read():
    from pydantic import ValidationError

    from soup_cli.config.loader import load_config_from_string

    text = """
base: unsloth/Qwen3-32B-bnb-4bit
task: sft
backend: transformers
modality: text
data:
  train: data.jsonl
  format: alpaca
training:
  batch_size: 1
  gradient_accumulation_steps: 1
  quantization: 8bit
  stream_layers: true
  lora:
    r: 8
"""
    with pytest.raises((ValidationError, ValueError)) as refused:
        load_config_from_string(text)
    assert "quantization='none' or" in str(refused.value)


def test_a_float16_card_is_refused_before_the_weights_are_resolved(tmp_path, monkeypatch):
    """The store is bfloat16 byte for byte; a T4 streams float16 and would reshard (or
    mismatch) on every run."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a run that cannot use the store")

    model_cfg = types.SimpleNamespace(
        model_type="llama", hidden_size=64, num_hidden_layers=2, vocab_size=128,
        intermediate_size=128, quantization_config=dict(NESTED_NF4),
    )
    monkeypatch.setattr(
        "soup_cli.trainer.stream_setup.StreamingSetupMixin._stream_meta_skeleton",
        classmethod(lambda cls, base, *, dtype, quant: types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.quantised_layer_suffixes",
        lambda _model: frozenset(_SUFFIXES),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _device: "float16"
    )
    with pytest.raises(ValueError, match="bfloat16"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="llama",
            model_cfg=model_cfg, tcfg_overrides={"quantization": "4bit"},
        )


def test_a_striped_cache_is_refused_before_the_weights_are_resolved(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a run that cannot use the store")

    model_cfg = types.SimpleNamespace(
        model_type="llama", hidden_size=64, num_hidden_layers=2, vocab_size=128,
        intermediate_size=128, quantization_config=dict(NESTED_NF4),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _device: "bfloat16"
    )
    monkeypatch.setattr(
        "soup_cli.utils.stripe_roots.resolve_stripe_roots", lambda *_a, **_k: ("other-drive",)
    )
    with pytest.raises(ValueError, match="stripe"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="llama",
            model_cfg=model_cfg, tcfg_overrides={"quantization": "4bit"},
        )


def test_a_run_without_double_quantization_is_refused_before_the_weights_are_resolved(
    tmp_path, monkeypatch
):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _must_not_resolve(*_args, **_kwargs):
        raise AssertionError("weights resolved for a run that cannot use the store")

    model_cfg = types.SimpleNamespace(
        model_type="llama", hidden_size=64, num_hidden_layers=2, vocab_size=128,
        intermediate_size=128, quantization_config=dict(NESTED_NF4),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _device: "bfloat16"
    )
    with pytest.raises(ValueError, match="double"):
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_must_not_resolve, arch="llama",
            model_cfg=model_cfg,
            tcfg_overrides={"quantization": "4bit", "double_quant_on": False},
        )


# ==========================================================================
# end to end on the CPU: soup train's streamed SFT over a serialised tiny Llama
# ==========================================================================
def _bf16_card(monkeypatch):
    """The CPU streams float32; an Ampere GPU streams bfloat16, which is what a serialised
    store holds. Pretend the card is the latter (the matmuls themselves run on the CPU)."""
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _device: "bfloat16"
    )


def _wrapper(tmp_path, monkeypatch, source, *, training=None):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_v07202 import _write_tiny_tokenizer

    _write_tiny_tokenizer(source)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    settings = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "quantization": "4bit",
        "stream_layers": True,
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "gradient_checkpointing": True,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    settings.update(training or {})
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": source,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": settings,
                "output": str(tmp_path / "out"),
            }
        )
    )
    dataset = {
        "train": [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello world"},
                ]
            }
            for _ in range(4)
        ]
    }
    return SFTTrainerWrapper(cfg, device="cpu"), dataset


@pytest.fixture
def served(tmp_path, monkeypatch, aten_half_matmuls):
    """``(wrapper, dataset, source)``: a serialised tiny Llama and the trainer over it."""
    _bf16_card(monkeypatch)
    source = _build(tmp_path)
    wrapper, dataset = _wrapper(tmp_path, monkeypatch, source)
    return wrapper, dataset, source


def _cache_folder(tmp_path):
    (name,) = os.listdir(tmp_path / "cache")
    return tmp_path / "cache" / name


def test_setup_imports_the_serialised_base_and_builds_the_streamed_trainer(served, tmp_path):
    from soup_cli.utils.layer_shard import read_shard_index

    wrapper, dataset, _source = served
    wrapper.setup(dataset)
    assert wrapper.trainer is not None
    assert wrapper.model.is_loaded_in_4bit is True
    index = read_shard_index(str(_cache_folder(tmp_path)))
    assert (index.quant, index.double_quant, index.quant_device) == ("nf4", True, "serialized")


def test_a_streamed_step_has_a_finite_loss_and_trains_every_layers_adapter(served):
    import torch

    from tests.test_v07202 import _randomise_lora_b

    wrapper, dataset, _source = served
    wrapper.setup(dataset)
    model = wrapper.model
    ids = torch.randint(1, 13, (1, 12), generator=torch.Generator().manual_seed(4))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    model.train()
    _randomise_lora_b(model)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    before = [param.detach().clone() for param in params]
    loss = model(**batch).loss
    assert torch.isfinite(loss), loss
    loss.backward()
    layers = set()
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name and ".layers." in name:
            assert param.grad is not None, name
            layers.add(int(name.split(".layers.")[1].split(".")[0]))
    assert layers == {0, 1}
    optimizer.step()
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, params))
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(model(**batch).loss)


def _resident_of(source, adapter_dtype):
    """The RESIDENT bitsandbytes model of the same files: ``from_pretrained`` over the
    serialised directory, adapters attached as the streamed model attaches them, and held
    in the dtype the streamed run keeps its adapters in (a bf16 run trains bf16 adapters;
    PEFT would otherwise default this reference to fp32, which rounds differently)."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    from tests.test_v07202 import _tiny_lora

    model = AutoModelForCausalLM.from_pretrained(
        source, device_map={"": "cpu"}, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    peft_model = get_peft_model(model, _tiny_lora())
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(adapter_dtype)
    return peft_model


def test_the_streamed_model_equals_the_resident_bitsandbytes_model_of_the_same_files(served):
    """Strictest equality: ``torch.equal`` on the logits, the loss and every LoRA gradient, in
    train mode (the resident path then dequantises and calls ``F.linear``, as the streamed
    one does; its eval path repacks for a CPU kernel and differs by one bf16 ulp, which
    test_v07202 bounds separately). First with the adapters inert (lora_B = 0: the base
    weights alone), then with them live."""
    import torch

    from tests.test_v07202 import _randomise_lora_b, _sync_adapters

    wrapper, dataset, source = served
    wrapper.setup(dataset)
    streamed = wrapper.model
    adapter_dtype = next(
        param.dtype for name, param in streamed.named_parameters() if "lora_" in name
    )
    resident = _resident_of(source, adapter_dtype)
    assert _sync_adapters(streamed, resident) > 0, "vacuous: no adapter was copied"

    ids = torch.randint(1, 13, (1, 16), generator=torch.Generator().manual_seed(5))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    streamed.train()
    resident.train()
    assert torch.equal(streamed(**batch).logits, resident(**batch).logits), "base weights"

    _randomise_lora_b(resident)
    assert _sync_adapters(streamed, resident) > 0
    out_streamed = streamed(**batch)
    out_resident = resident(**batch)
    assert torch.equal(out_streamed.logits, out_resident.logits)
    assert torch.equal(out_streamed.loss, out_resident.loss)

    out_streamed.loss.backward()
    out_resident.loss.backward()

    def grads(model):
        return {
            name.replace(".inner.", "."): param.grad
            for name, param in model.named_parameters()
            if param.requires_grad and "lora_" in name
        }

    got, want = grads(streamed), grads(resident)
    assert got.keys() == want.keys() and got
    for name in got:
        assert got[name] is not None and want[name] is not None, name
        assert torch.equal(got[name], want[name]), name


def test_a_second_setup_reuses_the_imported_cache(served, tmp_path, monkeypatch):
    from soup_cli.utils import nf4_serialized

    wrapper, dataset, source = served
    wrapper.setup(dataset)
    folder = _cache_folder(tmp_path)
    before = _bytes(folder)
    stamp = os.stat(folder / "index.json").st_mtime_ns

    def _must_not_import(*_args, **_kwargs):
        raise AssertionError("the pre-quantised cache was imported again")

    monkeypatch.setattr(nf4_serialized, "import_nf4_checkpoint", _must_not_import)
    again, dataset = _wrapper(tmp_path, monkeypatch, source)
    again.setup(dataset)
    assert _bytes(folder) == before
    assert os.stat(folder / "index.json").st_mtime_ns == stamp
    assert not any(name.startswith(".") or name.endswith(".resharding") for name in _files(folder))


def test_the_disk_preflight_sizes_the_cache_from_the_payload(tmp_path, monkeypatch):
    """First run: the import is charged its payload (not the coarse flat rate, not
    'approximately'). Second run: the cache is reusable, so nothing is written."""
    import soup_cli.trainer.stream_setup as stream_setup

    pytest.importorskip("peft")
    _bf16_card(monkeypatch)
    source = _build(tmp_path)
    recorded = []
    monkeypatch.setattr(
        stream_setup, "_render_stream_disk_preflight", lambda **kwargs: recorded.append(kwargs)
    )
    expected = _estimate(source).total_bytes
    wrapper, dataset = _wrapper(tmp_path, monkeypatch, source)
    wrapper.setup(dataset)
    again, dataset = _wrapper(tmp_path, monkeypatch, source)
    again.setup(dataset)

    first, second = recorded
    assert first["shard_bytes"] == expected and first["shard_write_bytes"] == expected
    assert first["approximate"] is False
    folder = _cache_folder(tmp_path)
    assert second["shard_write_bytes"] == 0
    assert second["shard_bytes"] == sum(os.path.getsize(folder / name) for name in _files(folder))


def test_a_cpu_run_without_a_bfloat16_card_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
):
    """No _bf16_card here: the CPU streams float32, which a bfloat16 store cannot serve."""
    pytest.importorskip("peft")
    source = _build(tmp_path)
    wrapper, dataset = _wrapper(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match="bfloat16"):
        wrapper.setup(dataset)
    cache = tmp_path / "cache"
    folders = os.listdir(cache) if cache.exists() else []
    assert [name for name in folders if os.listdir(cache / name)] == []
