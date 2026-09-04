"""#602 — Qwen4-Exp PLE rows may stay on a read-only SSD mapping."""

from __future__ import annotations

import hashlib
import json
import os
import re

import pytest


def _tiny_qwen4_config(*, ple: bool = True):
    try:
        from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig
    except ImportError:
        pytest.skip("installed Transformers release does not include qwen4_exp yet")

    text = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1] if ple else [],
        ple_embed_dim=16,
        heads_per_ngram=2,
        ngram_size=3,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        ple_conv_kernel_size=2,
        use_cache=False,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=2,
    )
    return Qwen4ExpConfig(text_config=text.to_dict())


def _save_tiny_qwen4(path):
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(602)
    model = AutoModelForCausalLM.from_config(
        _tiny_qwen4_config(), dtype=torch.float32
    )
    model.save_pretrained(path, safe_serialization=True)
    return model


def _ple_entry(weights_dir):
    from safetensors import safe_open

    found = []
    for name in sorted(os.listdir(weights_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(weights_dir, name)
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                match = re.search(r"ngram_embedding\.shard_(\d+)\.weight$", key)
                if match:
                    tensor = handle.get_tensor(key)
                    found.append((int(match.group(1)), name, key, tensor))
                elif key.endswith(".ple.ple_embedding.ngram_embedding.weight"):
                    found.append((0, name, key, handle.get_tensor(key)))
    if found:
        _part, name, key, tensor = sorted(found)[0]
        return name, key, tensor
    raise AssertionError("tiny Qwen4 checkpoint has no PLE table")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("torch_dtype", "safe_dtype"),
    [("float32", "F32"), ("float16", "F16"), ("bfloat16", "BF16")],
)
def test_sparse_reader_preserves_supported_dtypes(tmp_path, torch_dtype, safe_dtype):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.qwen4_ple import SafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    source = torch.arange(24, dtype=torch.float32).view(6, 4).to(
        getattr(torch, torch_dtype)
    )
    path = weights / "rows.safetensors"
    save_file({"rows": source}, path)
    reader = SafeTensorRowReader(
        str(weights),
        source_file=path.name,
        source_key="rows",
        expected_shape=tuple(source.shape),
        expected_dtype=safe_dtype,
    )

    ids = torch.tensor([5, 1, 5, 0])
    actual = reader.gather(ids)

    assert actual.dtype == source.dtype
    torch.testing.assert_close(actual, source[ids], rtol=0, atol=0)
    reader.close()


def test_sparse_reader_matches_resident_rows_without_mutating_source(tmp_path):
    import torch

    from soup_cli.utils.qwen4_ple import SafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    _save_tiny_qwen4(weights)
    filename, key, resident = _ple_entry(weights)
    path = weights / filename
    before_hash = _sha256(path)
    before_stat = path.stat()

    reader = SafeTensorRowReader(
        str(weights),
        source_file=filename,
        source_key=key,
        expected_shape=tuple(resident.shape),
        expected_dtype="F32",
    )
    last = resident.shape[0] - 1
    row_ids = torch.tensor([[last, 0, last], [0, last, 0]])
    gathered = reader.gather(row_ids)

    torch.testing.assert_close(gathered, resident[row_ids], rtol=0, atol=0)
    with pytest.raises(TypeError):
        reader._mapping[reader.spec.start] = 0
    reader.close()
    assert reader.closed is True
    assert _sha256(path) == before_hash
    after_stat = path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_sparse_reader_rejects_escape_and_out_of_range_rows(tmp_path):
    import torch

    from soup_cli.utils.qwen4_ple import SafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    _save_tiny_qwen4(weights)
    filename, key, resident = _ple_entry(weights)

    with pytest.raises(ValueError, match="filename"):
        SafeTensorRowReader(
            str(weights),
            source_file="../model.safetensors",
            source_key=key,
            expected_shape=tuple(resident.shape),
            expected_dtype="F32",
        )

    reader = SafeTensorRowReader(
        str(weights),
        source_file=filename,
        source_key=key,
        expected_shape=tuple(resident.shape),
        expected_dtype="F32",
    )
    with pytest.raises(IndexError, match="outside"):
        reader.gather(torch.tensor([resident.shape[0]]))
    reader.close()


def test_ple_header_rejects_shape_range_mismatch_and_past_eof():
    from soup_cli.utils.qwen4_ple import _tensor_rows_from_header

    with pytest.raises(ValueError, match="byte range"):
        _tensor_rows_from_header(
            {"ple": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 4]}},
            data_start=32,
            file_size=64,
            source_key="ple",
        )
    with pytest.raises(ValueError, match="past end"):
        _tensor_rows_from_header(
            {"ple": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
            data_start=32,
            file_size=47,
            source_key="ple",
        )


def test_qwen4_sharder_keeps_ple_in_original_checkpoint(tmp_path):
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import (
        QWEN4_PLE_WEIGHT_SUFFIX,
        layer_shard_path,
        read_shard_index,
        shard_checkpoint,
    )

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    _save_tiny_qwen4(weights)

    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )

    assert index.external_mode == "qwen4_ple"
    assert len(index.external_tensors) == 1
    external_key = next(iter(index.external_tensors))
    assert external_key.endswith(QWEN4_PLE_WEIGHT_SUFFIX)
    assert len(index.external_tensors[external_key].parts) > 1
    assert external_key not in index.layer_keys
    reloaded = read_shard_index(str(shards))
    assert reloaded.external_tensors == index.external_tensors
    assert (
        shard_checkpoint(
            str(weights), str(shards), dtype="float32", arch="qwen4_exp"
        )
        == index
    )
    for layer_idx in range(index.n_layers):
        with safe_open(layer_shard_path(str(shards), layer_idx), framework="pt") as handle:
            assert all("ngram_embedding.weight" not in key for key in handle.keys())


@pytest.mark.parametrize(
    ("bits", "words", "scale", "bias", "expected"),
    [
        (
            4,
            [3437096703, 2291772091, 1146447479, 1122867],
            -2.066666603088379,
            31.0,
            [0.0, 0.0, 2.0666675567626953, 2.0666675567626953],
        ),
        (
            5,
            [1975416799, 978769862, 902792293, 3343013046, 4469268],
            -1.0,
            31.0,
            [0.0, 1.0, 2.0, 3.0],
        ),
        (
            6,
            [
                2011676543,
                3144664893,
                2251909030,
                375498526,
                2735620389,
                2164256,
            ],
            -0.4920634925365448,
            31.0,
            [0.0, 0.9841270446777344, 1.9682540893554688, 2.952381134033203],
        ),
        (
            8,
            [
                3874486271,
                3318666974,
                2779624893,
                2223805596,
                1667986299,
                1112167002,
                556347706,
                528409,
            ],
            -0.12156862765550613,
            31.0,
            [0.0, 0.9725494384765625, 1.945098876953125, 3.039215087890625],
        ),
    ],
)
def test_oq_affine_decoder_matches_mlx_vectors(bits, words, scale, bias, expected):
    import torch

    from soup_cli.utils.oq_affine import AffineQuantSpec, dequantize_affine

    actual = dequantize_affine(
        torch.tensor([words], dtype=torch.uint32),
        torch.tensor([[scale]]),
        torch.tensor([[bias]]),
        spec=AffineQuantSpec(bits=bits, group_size=32),
        dtype="float32",
    )

    torch.testing.assert_close(
        actual[0, :4], torch.tensor(expected), rtol=0, atol=0
    )


def test_qwen4_sharder_dequantizes_omlx_oq_without_copying_companions(tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    save_file(
        {
            "language_model.model.layers.0.proj.weight": torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint32
            ),
            "language_model.model.layers.0.proj.scales": torch.ones(
                (2, 1), dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.proj.biases": torch.zeros(
                (2, 1), dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.conv1d.weight": torch.arange(
                24, dtype=torch.float32
            ).reshape(2, 12, 1),
            (
                "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding.shards.0.weight"
            ): torch.tensor([[0, 1, 2, 3]] * 3, dtype=torch.uint32),
            (
                "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding.shards.0.scales"
            ): torch.ones((3, 1), dtype=torch.bfloat16),
            (
                "language_model.model.layers.0.ple.ple_embedding."
                "ngram_embedding.shards.0.biases"
            ): torch.zeros((3, 1), dtype=torch.bfloat16),
            "vision_tower.ignored.weight": torch.ones(2, 2),
            "mtp.ignored.weight": torch.ones(2, 2),
        },
        weights / "model.safetensors",
    )
    (weights / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                }
            }
        ),
        encoding="utf-8",
    )

    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )

    assert index.n_layers == 1
    assert index.layer_keys == ("conv1d.weight", "proj.weight")
    assert len(index.external_tensors) == 1
    external = next(iter(index.external_tensors.values()))
    assert external.shape == (3, 32)
    assert external.bits == 4
    with safe_open(layer_shard_path(str(shards), 0), framework="pt") as handle:
        assert handle.keys() == ["conv1d.weight", "proj.weight"]
        actual = handle.get_tensor("proj.weight")
        conv = handle.get_tensor("conv1d.weight")
    assert actual.shape == (2, 32)
    assert conv.shape == (2, 1, 12)
    assert not any(key.endswith((".scales", ".biases")) for key in index.layer_keys)


def test_oq_ple_reader_dequantizes_only_selected_read_only_rows(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import OQExternalTensorPart, OQExternalTensorSpec
    from soup_cli.utils.qwen4_ple import OQShardedSafeTensorRowReader

    weights = tmp_path / "weights"
    weights.mkdir()
    source = weights / "model.safetensors"
    bits = 5
    rows = []
    for offset in range(4):
        stream = sum(((value + offset) % 32) << (bits * value) for value in range(32))
        rows.append([(stream >> (32 * word)) & 0xFFFFFFFF for word in range(bits)])
    packed = torch.tensor(rows, dtype=torch.uint32)
    scales = torch.tensor([[0.5], [1.0], [1.5], [2.0]], dtype=torch.bfloat16)
    biases = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.bfloat16)
    save_file(
        {"ple.weight": packed, "ple.scales": scales, "ple.biases": biases},
        source,
    )
    part = OQExternalTensorPart(
        source_file=source.name,
        weight_key="ple.weight",
        scales_key="ple.scales",
        biases_key="ple.biases",
        packed_shape=tuple(packed.shape),
        stats_shape=tuple(scales.shape),
        packed_dtype="U32",
        stats_dtype="BF16",
    )
    spec = OQExternalTensorSpec(
        parts=(part,),
        shape=(4, 32),
        dtype="float32",
        bits=bits,
        group_size=32,
        mode="affine",
    )
    before = _sha256(source)
    reader = OQShardedSafeTensorRowReader(str(weights), spec)
    ids = torch.tensor([[3, 1, 3]])

    actual = reader.gather(ids)

    quantized = torch.tensor(
        [[(value + offset) % 32 for value in range(32)] for offset in (3, 1, 3)],
        dtype=torch.float32,
    )
    expected = quantized * scales[[3, 1, 3]].float() + biases[[3, 1, 3]].float()
    torch.testing.assert_close(actual.squeeze(0), expected, rtol=0, atol=0)
    assert reader.nbytes == packed.numel() * 4 + (scales.numel() + biases.numel()) * 2
    reader.close()
    assert _sha256(source) == before


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("ngram_source", ["disk", "ram"])
def test_tiny_qwen4_ple_matches_resident_forward_loss_and_lora_gradients(
    tmp_path, device, ngram_source
):
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("needs an Apple Silicon MPS device")
    resident = _save_tiny_qwen4(weights).to(device).eval()
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights),
        weights_dir=str(weights),
        shard_dir=str(shards),
        index=index,
        lora_config=LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "in_proj_qkv"],
            task_type=TaskType.CAUSAL_LM,
        ),
        device=device,
        dtype="float32",
        buffers=2,
        pin=False,
        tier="ram",
        ngram_source=ngram_source,
    )
    streamed.eval()
    batch = torch.tensor([[1, 3, 4, 5]], device=device)

    with torch.no_grad():
        expected = resident(input_ids=batch, labels=batch)
    actual = streamed(input_ids=batch, labels=batch)

    if device == "cpu":
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    else:
        # MPS may select a different reduction schedule once PEFT wraps a
        # projection, even with its zero-initialised B matrix. The MPS gate uses
        # a narrow numerical tolerance; the CPU oracle above remains bit-exact
        # and guards the PLE row mapping itself.
        torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-4, atol=2e-8)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=3e-4, atol=2e-8)
    actual.loss.backward()
    trainable = [parameter for parameter in streamed.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert runtime.external_sources
    reader = runtime.external_sources[0]
    if ngram_source == "disk":
        assert len(reader.containers) == 1
        assert len({id(part._mapping) for part in reader.parts}) == 1
    runtime.close()
    assert reader.closed is True


def test_qwen4_streaming_gate_and_ngram_config():
    from types import SimpleNamespace

    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.layer_stream import stream_arch_of

    assert stream_arch_of(SimpleNamespace(model_type="qwen4_exp")) == "qwen4_exp"
    assert (
        stream_arch_of(SimpleNamespace(model_type="qwen4_exp_text"))
        == "qwen4_exp"
    )
    assert TrainingConfig(stream_ngram_source="disk").stream_ngram_source == "disk"
    with pytest.raises(ValueError, match="stream_ngram_source"):
        TrainingConfig(stream_ngram_source="network")


def test_qwen4_streaming_refuses_unsupported_task_by_name():
    from soup_cli.trainer.stream_setup import _validate_qwen4_streaming_mode

    with pytest.raises(ValueError, match="task='sft'"):
        _validate_qwen4_streaming_mode(
            arch="qwen4_exp", task="dpo", quant="none"
        )


def test_qwen4_streaming_refuses_quantized_base_by_name():
    from soup_cli.trainer.stream_setup import _validate_qwen4_streaming_mode

    with pytest.raises(ValueError, match="quantization='none'"):
        _validate_qwen4_streaming_mode(
            arch="qwen4_exp", task="sft", quant="nf4"
        )


def test_qwen4_ple_disk_streaming_refuses_non_ssd_by_name():
    from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_disk

    with pytest.raises(ValueError, match="needs an SSD or NVMe"):
        _validate_qwen4_ngram_disk(disk_kind="hdd", weights_dir="/slow/model")


def test_qwen4_stream_ngram_source_warns_when_checkpoint_has_no_ple():
    from soup_cli.trainer.stream_setup import _warn_if_ngram_source_unused

    messages = []
    _warn_if_ngram_source_unused(
        arch="qwen4_exp", requested="disk", ngram_bytes=0, notify=messages.append
    )

    assert len(messages) == 1
    assert "has no effect" in messages[0]


def test_external_ple_descriptor_reapplies_a_production_sized_element_cap():
    from soup_cli.utils.layer_shard import ExternalTensorPart, ExternalTensorSpec

    oversized = ExternalTensorPart(
        source_file="model.safetensors",
        source_key="ple.weight",
        shape=(2**36 + 1, 1),
        dtype="F32",
    )

    with pytest.raises(ValueError, match="element cap"):
        ExternalTensorSpec(parts=(oversized,), shape=oversized.shape, dtype="F32")


def test_qwen4_ngram_policy_covers_oq_ram_and_auto_defaults():
    from soup_cli.trainer.stream_setup import _resolve_qwen4_ngram_source

    common = {
        "store_total": 20,
        "ngram_bytes": 30,
        "free_ram": 100,
        "resident_ram": 0,
        "total_ram": None,
        "stream_source": "auto",
    }
    with pytest.raises(ValueError, match="oQ PLE embeddings require"):
        _resolve_qwen4_ngram_source(oq_ngram=True, requested="ram", **common)
    assert (
        _resolve_qwen4_ngram_source(
            oq_ngram=True, requested="auto", **common
        )
        == "disk"
    )
    assert (
        _resolve_qwen4_ngram_source(
            oq_ngram=False, requested="auto", **common
        )
        == "ram"
    )
    assert (
        _resolve_qwen4_ngram_source(
            oq_ngram=False,
            requested="auto",
            **{**common, "ngram_bytes": 70},
        )
        == "disk"
    )
    assert (
        _resolve_qwen4_ngram_source(
            oq_ngram=False,
            requested="auto",
            **_qwen4_ple_physical_limit_case(common),
        )
        == "disk"
    )


def test_qwen4_ngram_auto_counts_resident_ram_against_free_budget():
    from soup_cli.trainer.stream_setup import _resolve_qwen4_ngram_source

    assert (
        _resolve_qwen4_ngram_source(
            oq_ngram=False,
            requested="auto",
            store_total=3_000,
            ngram_bytes=1_000,
            free_ram=10_000,
            resident_ram=3_000,
            total_ram=100_000,
            stream_source="auto",
        )
        == "disk"
    )


def _qwen4_ple_physical_limit_case(common):
    physical_ram = 30_000
    base_store = 10_000
    ngram_store = 5_000
    resident_store = 3_000
    return {
        **common,
        "store_total": base_store,
        "ngram_bytes": ngram_store,
        "free_ram": physical_ram,
        "resident_ram": resident_store,
        "total_ram": physical_ram,
    }


def test_qwen4_ram_ple_refusal_is_independent_of_base_stream_source():
    from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

    with pytest.raises(ValueError, match="stream_ngram_source='ram'"):
        _validate_qwen4_ngram_ram_fit(
            stream_source="auto",
            ngram_source="ram",
            required_ram=81,
            free_ram=100,
        )
    _validate_qwen4_ngram_ram_fit(
        stream_source="auto",
        ngram_source="disk",
        required_ram=10_000,
        free_ram=100,
    )
    with pytest.raises(ValueError, match="stream_source='ram'"):
        _validate_qwen4_ngram_ram_fit(
            stream_source="ram",
            ngram_source="disk",
            required_ram=81,
            free_ram=100,
        )


def _drive_qwen4_streaming_setup(tmp_path, monkeypatch, resolve_weights=None):
    """Drive the real `_setup_streaming_transformers()` for a qwen4_exp base.

    THE harness. Extracted by the maintainer after review so the seam test and
    the planner-fingerprint test below share one copy: a second, drifting copy
    of 150 lines of monkeypatching is how a seam test stops matching the seam.
    Returns the list of fail-closed policies the setup actually reached.

    `resolve_weights` replaces the `resolve_model_weights` stub. The default
    stub ignores the `before_materialize` callback, which is where the shard-
    cache fingerprint is computed -- so a test about that callback has to supply
    its own and actually call it.
    """
    import sys
    import types

    import soup_cli.trainer.stream_setup as stream_setup

    harness_total_ram = 2_000_000
    harness_resident_ram = 4_096

    class _Wrapper(stream_setup.StreamingSetupMixin):
        def __init__(self):
            self.device = "cpu"
            self._trust_remote_code = False

        def _stream_budget_lines(self, *_args, **_kwargs):
            return (), None

    cfg = types.SimpleNamespace(
        base="qwen4-test",
        task="sft",
        data=types.SimpleNamespace(max_length=8),
    )
    tcfg = types.SimpleNamespace(
        quantization="none",
        double_quant_on=True,
        stream_source="auto",
        stream_ngram_source="auto",
        stream_buffers=2,
        stream_disk_kind=None,
        stream_pin=None,
        seed=7,
        moe_lora=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        lora=types.SimpleNamespace(
            r=2,
            alpha=4,
            dropout=0.0,
            target_modules=["q_proj"],
            use_dora=False,
            use_rslora=False,
        ),
    )
    model_cfg = types.SimpleNamespace(
        model_type="qwen4_exp",
        text_config=types.SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=1,
            vocab_size=8,
            intermediate_size=8,
        ),
    )
    external = types.SimpleNamespace(bits=4, nbytes=1_024, storage_nbytes=256)
    index = types.SimpleNamespace(
        n_layers=1,
        total_params=4,
        quant="none",
        quant_specs={},
        external_tensors={"model.layers.0.ple.weight": external},
    )
    layer_specs = [{"self_attn.q_proj.weight": ((2, 2), "float32")}]
    plan = types.SimpleNamespace(
        tier="ram", store_bytes=16, large_store_bytes=0, pinned=False, notes=()
    )
    runtime = types.SimpleNamespace(
        stats=lambda: {
            "tier": "ram",
            "store_bytes": 16,
            "pinned": False,
            "n_layers": 1,
            "buffers": 2,
            "buffer_bytes": 8,
            "large_buffer_bytes": 0,
        }
    )
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(from_pretrained=None),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=object(),
            TaskType=types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        ),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: tokenizer
    )
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", lambda *_a, **_k: model_cfg
    )
    monkeypatch.setattr("peft.LoraConfig", lambda **kwargs: types.SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.stream_arch_of", lambda *_a, **_k: "qwen4_exp"
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_shard.resolve_shard_dir",
        lambda *_a, **_k: str(tmp_path / "shards"),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_shard.shard_checkpoint", lambda *_a, **_k: index
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_shard.source_weight_bytes", lambda *_a, **_k: 1_024
    )
    monkeypatch.setattr(
        "soup_cli.utils.spectrum_scan.resolve_model_weights",
        resolve_weights
        or (lambda *_a, **_k: str(tmp_path / "weights")),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.free_ram_bytes", lambda: 1_000_000
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.total_ram_bytes", lambda: harness_total_ram
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.RamSource.layer_specs_from_shards",
        lambda *_a, **_k: layer_specs,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.extras_resident_bytes",
        lambda *_a, **_k: harness_resident_ram,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.large_layer_store_bytes",
        lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.large_layer_buffer_bytes",
        lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.resolve_disk_kind",
        lambda *_a, **_k: types.SimpleNamespace(kind="ssd"),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream.render_stream_panel", lambda *_a, **_k: "panel"
    )
    monkeypatch.setattr(
        "soup_cli.utils.peft_wiring.resolve_lora_target_modules",
        lambda *_a, **_k: ["q_proj"],
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.build_streamed_model",
        lambda **_kwargs: (types.SimpleNamespace(), runtime),
    )

    reached = []
    captured = {}

    def _mode(**_kwargs):
        reached.append("mode")

    def _source(**kwargs):
        reached.append("source")
        captured["source"] = kwargs
        return "disk"

    def _disk(**_kwargs):
        reached.append("disk")

    def _ram(**kwargs):
        reached.append("ram")
        captured["ram"] = kwargs

    monkeypatch.setattr(stream_setup, "_validate_qwen4_streaming_mode", _mode)
    monkeypatch.setattr(stream_setup, "_resolve_qwen4_ngram_source", _source)
    monkeypatch.setattr(stream_setup, "_validate_qwen4_ngram_disk", _disk)

    def _plan(**kwargs):
        captured["plan"] = kwargs
        return plan

    monkeypatch.setattr(stream_setup, "_validate_qwen4_ngram_ram_fit", _ram)
    monkeypatch.setattr("soup_cli.utils.layer_stream.build_stream_plan", _plan)

    _Wrapper()._setup_streaming_transformers(cfg, tcfg)
    return reached, captured, harness_total_ram, harness_resident_ram


def test_setup_reaches_every_qwen4_fail_closed_policy(tmp_path, monkeypatch):
    reached, _captured, _total_ram, _resident_ram = _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch
    )

    assert reached == ["mode", "source", "disk", "ram"]


def test_setup_passes_total_ram_to_qwen4_planner_and_refusal(tmp_path, monkeypatch):
    _reached, captured, total_ram, resident_ram = _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch
    )

    assert captured["source"]["total_ram"] == total_ram
    assert captured["source"]["resident_ram"] == resident_ram
    assert captured["ram"]["total_ram"] == total_ram
    assert captured["ram"]["resident_ram"] == resident_ram
    assert captured["plan"]["total_ram_bytes"] == total_ram
    assert captured["plan"]["embed_bytes"] == resident_ram


def test_planner_fingerprints_config_json_for_qwen4_but_not_for_llama(
    tmp_path, monkeypatch
):
    """The planner half of the oQ cache fingerprint, which nothing pinned.

    `layer_shard` hashes `config.json` into the shard-cache fingerprint for an
    external-mode checkpoint, and `test_qwen4_config_json_change_invalidates_oq_cache`
    covers that side. The PLANNER computes the same fingerprint independently
    (`stream_setup.py`, `include_config=arch == "qwen4_exp"`), and setting it to
    a constant `False` there passed 694 streaming tests.

    That is not display-only. A planner that reports a stale cache as valid
    zeroes `shard_write_bytes`, which feeds the disk pre-flight that refuses
    before sharding -- so the required space is under-reported, the refusal
    never fires, and the run dies out of disk mid-shard: the exact failure the
    pre-flight exists to prevent.

    Two assertions, because either alone is weak. The first pins the call site
    -- a constant `False` fails it. The second pins that the flag is
    load-bearing rather than decorative: with it, editing only `config.json`
    moves the fingerprint; without it, the fingerprint is unchanged.
    """
    import json
    import types

    import torch
    from safetensors.torch import save_file

    import soup_cli.utils.layer_shard as layer_shard
    from soup_cli.utils.layer_shard import (
        checkpoint_source_components,
        fingerprint_source_files,
    )

    seen = []
    real = layer_shard.checkpoint_source_components

    def _recording(weights_dir, source_files, *, include_config):
        seen.append(include_config)
        return real(weights_dir, source_files, include_config=include_config)

    monkeypatch.setattr(layer_shard, "checkpoint_source_components", _recording)

    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    blob = weights / "model.safetensors"
    save_file(
        {
            "language_model.model.layers.0.proj.weight": torch.tensor(
                [[0, 1, 2, 3]], dtype=torch.uint32
            ),
            "language_model.model.layers.0.proj.scales": torch.ones(
                (1, 1), dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.proj.biases": torch.zeros(
                (1, 1), dtype=torch.bfloat16
            ),
            "language_model.model.norm.weight": torch.ones(2),
        },
        blob,
    )
    (weights / "config.json").write_text(
        json.dumps({"quantization_config": {"bits": 4, "group_size": 32}}),
        encoding="utf-8",
    )
    stat = blob.stat()
    plan = types.SimpleNamespace(
        weights_dir=str(weights),
        source_bytes=stat.st_size,
        materialized_copy_bytes=0,
        materialize_bytes=0,
        needs_materialization=False,
        source_files=((str(blob), stat.st_size, int(stat.st_mtime)),),
    )

    def _resolve(_base, *, before_materialize=None):
        if before_materialize is not None:
            before_materialize(plan)
        return str(weights)

    _drive_qwen4_streaming_setup(tmp_path, monkeypatch, resolve_weights=_resolve)

    assert seen == [True], (
        "the planner must hash config.json for a qwen4_exp base; a constant "
        "False here silently reuses a stale oQ shard cache. saw: " + repr(seen)
    )

    probe = tmp_path / "fingerprint_probe"
    probe.mkdir()
    probe_blob = probe / "model.safetensors"
    probe_blob.write_bytes(b"weights")
    config = probe / "config.json"
    config.write_text(json.dumps({"a": 1}), encoding="utf-8")

    def _digest(include_config):
        stat = probe_blob.stat()
        components = ((probe_blob.name, stat.st_size, stat.st_mtime_ns),)
        return fingerprint_source_files(
            checkpoint_source_components(
                str(probe), components, include_config=include_config
            )
        )

    before_on, before_off = _digest(True), _digest(False)
    config.write_text(json.dumps({"a": 1}, indent=2), encoding="utf-8")

    assert _digest(True) != before_on, "config.json must move the fingerprint"
    assert _digest(False) == before_off, (
        "without include_config the same edit is invisible -- which is "
        "what makes the call site above load-bearing"
    )


def test_non_qwen_companion_suffixes_do_not_enable_oq(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "weights"
    weights.mkdir()
    save_file(
        {
            "model.layers.0.proj.weight": torch.ones(2, 2),
            "model.layers.0.proj.scales": torch.ones(2, 1),
            "model.layers.0.proj.biases": torch.zeros(2, 1),
            "model.norm.weight": torch.ones(2),
        },
        weights / "model.safetensors",
    )

    index = shard_checkpoint(str(weights), str(tmp_path / "shards"), dtype="float32")

    assert index.layer_keys == ("proj.biases", "proj.scales", "proj.weight")


def test_qwen4_config_json_change_invalidates_oq_cache(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "weights"
    shards = tmp_path / "shards"
    weights.mkdir()
    save_file(
        {
            "language_model.model.layers.0.proj.weight": torch.tensor(
                [[0, 1, 2, 3]], dtype=torch.uint32
            ),
            "language_model.model.layers.0.proj.scales": torch.ones(
                (1, 1), dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.proj.biases": torch.zeros(
                (1, 1), dtype=torch.bfloat16
            ),
            "language_model.model.norm.weight": torch.ones(2),
        },
        weights / "model.safetensors",
    )
    config = {
        "quantization_config": {"bits": 4, "group_size": 32, "mode": "affine"}
    }
    config_path = weights / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    first = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="qwen4_exp"
    )

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    notices = []
    second = shard_checkpoint(
        str(weights),
        str(shards),
        dtype="float32",
        arch="qwen4_exp",
        notify=notices.append,
    )

    assert first.source_fingerprint != second.source_fingerprint
    assert any(name == "config.json" for name, _size, _mtime in second.source_files)
    assert any("config.json" in notice for notice in notices)


def test_disk_embedding_returns_rows_on_registered_weight_device():
    import torch

    from soup_cli.utils.qwen4_ple import _disk_embedding

    class _Reader:
        dtype = "float32"
        shape = (4, 3)

        def gather(self, row_ids):
            return torch.ones((*row_ids.shape, self.shape[1]))

    embedding = _disk_embedding(_Reader()).to("meta")
    actual = embedding(torch.tensor([0, 1]))

    assert actual.device.type == "meta"


def test_external_tensor_bytes_reports_oq_packed_storage():
    from types import SimpleNamespace

    from soup_cli.utils.qwen4_ple import external_tensor_bytes

    spec = SimpleNamespace(nbytes=1_024, storage_nbytes=256)
    assert external_tensor_bytes({"ple.weight": spec}) == 256


def test_qwen4_oq_torch_floor_matches_project_and_doctor():
    import re
    from pathlib import Path

    from soup_cli.commands.doctor import DEPS

    root = Path(__file__).parents[1]
    project = (root / "pyproject.toml").read_text(encoding="utf-8")

    # Parse the `train` extra's own list rather than scanning the whole file.
    # A substring test is satisfied by the string appearing anywhere --
    # a comment, another extra, or a docstring -- so it would keep passing
    # after the floor had been moved or removed from the place that matters.
    # `tomllib` is 3.11+ and this repo supports 3.10, so this is a bounded
    # regex over one block rather than a TOML parse.
    block = re.search(r"^train = \[(.*?)^\]", project, re.S | re.M)
    assert block is not None, "pyproject.toml has no `train` extra"
    entries = re.findall(r'"([^"]+)"', block.group(1))
    torch_entries = [e for e in entries if e.split(">")[0].strip() == "torch"]

    assert torch_entries == ["torch>=2.5.0"], (
        "the `train` extra must declare exactly one torch floor, and it must be "
        "2.5.0 -- transformers 5.16.1's own torch extra forces torch>=2.5, "
        "which subsumes Qwen4/oQ's uint32 need at 2.3 (#636); found "
        f"{torch_entries}"
    )
    assert next(item for item in DEPS if item[0] == "torch")[2] == "2.5.0", (
        "`soup doctor` keeps a literal copy of the torch floor; it must equal "
        "the one pyproject.toml declares, pinned by "
        "tests/test_issue636_torch_floor.py (#636)"
    )


def test_qwen4_gate_record_and_changelog_are_discoverable_and_credited():
    """The gate record is indexed and the changelog entry keeps its credit.

    The credit lives in a per-PR fragment before a release and in the assembled
    ``CHANGELOG.md`` after one (#487/#490), so this reads BOTH and requires the
    string in whichever currently carries it. The first version pinned the
    fragment path alone and turned red on every cell the moment v0.74.0
    consumed it -- an assembly that is supposed to happen, failing a test whose
    subject is the credit rather than where the credit is stored.
    """
    from pathlib import Path

    root = Path(__file__).parents[1]
    benchmark_index = (root / "benchmarks" / "README.md").read_text(encoding="utf-8")

    sources = [root / "CHANGELOG.md"] + sorted(
        (root / "changelog.d").rglob("603.*.md")
    )
    texts = [p.read_text(encoding="utf-8") for p in sources if p.is_file()]
    assert texts, "neither CHANGELOG.md nor a 603 fragment is readable"

    assert "gate-qwen4-ple-m4-max.md" in benchmark_index, (
        "benchmarks/README.md no longer indexes the Qwen4 PLE gate record, so "
        "the measurement behind this feature is not discoverable"
    )
    assert any("(#602 by @Amix29 in #603)" in t for t in texts), (
        "the Qwen4 PLE changelog entry lost its `(#602 by @Amix29 in #603)` "
        f"credit; searched {[str(p.relative_to(root)) for p in sources]}"
    )
