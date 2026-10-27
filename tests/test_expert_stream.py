"""Expert-granularity streaming (``utils/expert_stream.py``).

The gate is bit-exactness against transformers' eager experts loop: the
streamed module must equal the resident one forward AND backward, on the
block alone and inside a streamed DeepSeek-V3 model. With ``two_reads`` it must
also equal the default streamed path while the checkpoint's recompute reads no
expert.
"""

from __future__ import annotations

import pytest

from tests._windows_ci import skip_on_windows_ci

CUDA = pytest.mark.gpu(reason="NF4 dequantisation needs bitsandbytes on CUDA")
#: ``StreamedExperts.two_reads``: off is the default node layout; on puts the routing
#: multiply inside the expert's autograd node so a checkpoint recompute reads nothing.
TWO_READS = pytest.mark.parametrize("two_reads", [False, True], ids=["default", "two-reads"])


# --------------------------------------------------------------------------
# block level
# --------------------------------------------------------------------------
def _deepseek_v3_experts(seed):
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts

    config = DeepseekV3Config(
        hidden_size=32, moe_intermediate_size=16, n_routed_experts=8, n_group=2,
        topk_group=1, num_experts_per_tok=2,
    )
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    module = DeepseekV3Experts(config)
    with torch.no_grad():
        module.gate_up_proj.normal_(0.0, 0.2)
        module.down_proj.normal_(0.0, 0.2)
    module.requires_grad_(False)
    return module, 8, 2


def _olmoe_experts(seed):
    import torch
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_experts=8, num_experts_per_tok=2)
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    module = OlmoeExperts(config)
    with torch.no_grad():
        module.gate_up_proj.normal_(0.0, 0.2)
        module.down_proj.normal_(0.0, 0.2)
    module.requires_grad_(False)
    return module, 8, 2


def _streamed_copy(tmp_path, module, n_experts, two_reads=False):
    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        ExpertShardReader,
        streamed_experts_class,
        write_expert_shard,
    )

    path = str(tmp_path / "experts_001.safetensors")
    pairs = (
        (module.gate_up_proj[e].detach(), module.down_proj[e].detach())
        for e in range(n_experts)
    )
    info = write_expert_shard(path, pairs, quantise=False)
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs={}, dtype="float32"
    )
    streamed = streamed_experts_class()(
        reader, ExpertFetcher(device="cpu", dtype="float32"), module.act_fn, 1, two_reads=two_reads
    )
    return streamed, reader


def _routing(n_tokens, n_experts, top_k, hidden, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(n_tokens, n_experts, generator=generator)
    top_k_index = scores.topk(top_k, dim=-1).indices
    top_k_weights = torch.rand(n_tokens, top_k, generator=generator).requires_grad_(True)
    states = torch.randn(n_tokens, hidden, generator=generator).requires_grad_(True)
    grad_out = torch.randn(n_tokens, hidden, generator=generator)
    return states, top_k_index, top_k_weights, grad_out


@TWO_READS
@pytest.mark.parametrize(
    "build", [_deepseek_v3_experts, _olmoe_experts], ids=["deepseek_v3", "olmoe"]
)
def test_streamed_experts_forward_equals_eager_bit_exactly(tmp_path, build, two_reads):
    import torch

    module, n_experts, top_k = build(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts, two_reads)
    states, index, weights, _ = _routing(24, n_experts, top_k, 32, 9)
    with torch.no_grad():
        want = module(states, index, weights)
        got = streamed(states, index, weights)
    assert torch.equal(got, want), (got - want).abs().max().item()
    reader.close()


@TWO_READS
@pytest.mark.parametrize(
    "build", [_deepseek_v3_experts, _olmoe_experts], ids=["deepseek_v3", "olmoe"]
)
def test_streamed_experts_backward_equals_eager_bit_exactly(tmp_path, build, two_reads):
    """Input and routing-weight gradients, the two a frozen expert passes on."""
    import torch

    module, n_experts, top_k = build(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts, two_reads)
    states, index, weights, grad_out = _routing(24, n_experts, top_k, 32, 9)

    module(states, index, weights).backward(grad_out)
    want_states, want_weights = states.grad.clone(), weights.grad.clone()
    states.grad = None
    weights.grad = None
    streamed(states, index, weights).backward(grad_out)

    assert float(want_states.abs().sum()) > 0.0
    assert torch.equal(states.grad, want_states), (states.grad - want_states).abs().max().item()
    assert torch.equal(weights.grad, want_weights)
    reader.close()


@TWO_READS
def test_streamed_experts_routing_weight_gradient_when_the_input_needs_none(tmp_path, two_reads):
    """Nothing upstream trains, so the expert input needs no gradient, while the routing
    weight does: its gradient must be the weight's own, not the input's."""
    import torch

    module, n_experts, top_k = _deepseek_v3_experts(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts, two_reads)
    states, index, weights, grad_out = _routing(24, n_experts, top_k, 32, 9)
    states.requires_grad_(False)

    module(states, index, weights).backward(grad_out)
    want = weights.grad.clone()
    weights.grad = None
    streamed(states, index, weights).backward(grad_out)

    assert float(want.abs().sum()) > 0.0
    assert torch.equal(weights.grad, want), (weights.grad - want).abs().max().item()
    reader.close()


# The bf16 case runs a CPU bf16 matmul under autocast, and part of GitHub's windows-latest
# fleet dies on it with 0xc000001d (illegal instruction) in the eager reference module's
# forward; the crash kills the xdist worker and the cell later times out (#382, #1314).
# Only that case is skipped, and only on a Windows CI runner: the fp32 control, a developer's
# Windows box, ubuntu and macos all still run it.
@TWO_READS
@pytest.mark.parametrize(
    "autocast",
    [
        pytest.param(False, id="fp32-control"),
        pytest.param(True, id="cpu-bf16-autocast", marks=skip_on_windows_ci),
    ],
)
def test_streamed_experts_backward_preserves_forward_autocast(tmp_path, autocast, two_reads):
    """Recomputation must differentiate the forward precision after its AMP region exits."""
    import torch

    module, n_experts, top_k = _deepseek_v3_experts(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts, two_reads)
    states, index, weights, grad_out = _routing(24, n_experts, top_k, 32, 9)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            want = module(states, index, weights)
        want.backward(grad_out)
        want_states, want_weights = states.grad.clone(), weights.grad.clone()
        states.grad = None
        weights.grad = None

        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            got = streamed(states, index, weights)
        got.backward(grad_out)

        assert torch.equal(got, want)
        assert float(want_states.abs().sum()) > 0.0
        assert torch.equal(states.grad, want_states), (states.grad - want_states).abs().max().item()
        assert torch.equal(weights.grad, want_weights)
    finally:
        reader.close()


@TWO_READS
def test_each_routed_expert_is_read_once_forward_and_once_backward(tmp_path, two_reads):
    """The cost the docstring states: an expert nobody routed to is never read."""
    import torch

    module, n_experts, _ = _deepseek_v3_experts(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts, two_reads)
    states = torch.randn(6, 32, requires_grad=True)
    index = torch.tensor([[0, 3]] * 6)
    weights = torch.full((6, 2), 0.5, requires_grad=True)
    out = streamed(states, index, weights)
    assert reader.fetches == 2
    out.sum().backward()
    assert reader.fetches == 4
    reader.close()


def test_fetched_weights_are_owned_not_views_of_the_staging_buffer(tmp_path):
    """Every read lands in one shared staging buffer; a fetch must hand back
    tensors the next read cannot overwrite (on a CPU target `.to` would not copy)."""
    import torch

    module, n_experts, _ = _deepseek_v3_experts(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts)
    first = [tensor.clone() for tensor in streamed.fetch(0)]
    held = streamed.fetch(0)
    streamed.fetch(1)
    assert torch.equal(held[0], first[0]) and torch.equal(held[1], first[1])
    assert not torch.equal(first[0], module.gate_up_proj[1])
    reader.close()


@TWO_READS
def test_expert_weights_are_released_after_each_expert(tmp_path, two_reads):
    """The VRAM bound: nothing keeps a fetched expert alive once its MLP ran, not
    the autograd graph after forward, not anything after backward."""
    import gc
    import weakref


    module, n_experts, top_k = _deepseek_v3_experts(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts, two_reads)
    refs = []
    real_fetch = streamed.fetch

    def tracking_fetch(expert):
        gate_up, down = real_fetch(expert)
        refs.extend((weakref.ref(gate_up), weakref.ref(down)))
        return gate_up, down

    streamed.fetch = tracking_fetch
    states, index, weights, grad_out = _routing(24, n_experts, top_k, 32, 9)
    out = streamed(states, index, weights)
    gc.collect()
    assert refs and all(ref() is None for ref in refs), "forward kept expert weights alive"
    refs.clear()
    out.backward(grad_out)
    gc.collect()
    assert refs and all(ref() is None for ref in refs), "backward kept expert weights alive"
    reader.close()


def test_reader_direct_and_buffered_reads_agree(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import (
        ExpertShardReader,
        aligned_staging,
        write_expert_shard,
    )

    path = str(tmp_path / "experts_001.safetensors")
    pairs = [(torch.randn(6, 4), torch.randn(4, 3)) for _ in range(3)]
    info = write_expert_shard(path, pairs, quantise=False)
    readers = [
        ExpertShardReader(
            path, count=3, shapes=info.shapes, quant_specs={}, dtype="float32", direct=direct
        )
        for direct in (True, False)
    ]
    assert readers[1].direct is False
    for expert in range(3):
        views = []
        for reader in readers:
            _owner, staging = aligned_staging(reader.max_span)
            views.append({k: v.clone() for k, v in reader.read_host(expert, staging).items()})
        assert views[0].keys() == views[1].keys()
        for key in views[0]:
            assert torch.equal(views[0][key], views[1][key])
        assert torch.equal(views[0][f"{expert}.gate_up"], pairs[expert][0])
    with pytest.raises(IndexError):
        readers[0].read_host(3, aligned_staging(readers[0].max_span)[1])
    with pytest.raises(ValueError, match="staging holds"):
        readers[0].read_host(0, aligned_staging(16)[1])
    for reader in readers:
        reader.close()


@pytest.mark.parametrize("expert", [-1, 3, 4])
def test_read_host_refuses_an_expert_outside_the_file(tmp_path, expert):
    """-1 would otherwise read the last expert through Python's negative index."""
    import torch

    from soup_cli.utils.expert_stream import (
        ExpertShardReader,
        aligned_staging,
        write_expert_shard,
    )

    path = str(tmp_path / "experts_001.safetensors")
    pairs = [(torch.randn(6, 4), torch.randn(4, 3)) for _ in range(3)]
    info = write_expert_shard(path, pairs, quantise=False)
    reader = ExpertShardReader(path, count=3, shapes=info.shapes, quant_specs={}, dtype="float32")
    with pytest.raises(IndexError, match=r"outside \[0, 3\)"):
        reader.read_host(expert, aligned_staging(reader.max_span)[1])
    reader.close()


def test_a_reader_holds_no_handle_until_its_first_fetch(tmp_path):
    """An install that fails after the readers were built must not leave the
    experts files open (Windows keeps an open file locked until collection)."""
    import torch

    from soup_cli.utils.expert_stream import (
        ExpertShardReader,
        aligned_staging,
        write_expert_shard,
    )

    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False)
    reader = ExpertShardReader(path, count=1, shapes=info.shapes, quant_specs={}, dtype="float32")
    assert reader._handle is None
    staging = aligned_staging(reader.max_span)[1]
    reader.read_host(0, staging)
    first = reader._handle
    assert first is not None
    reader.read_host(0, staging)
    # One handle for the reader's life: a reopen per fetch would leak the last one.
    assert reader._handle is first
    reader.close()
    assert reader._handle is None


def test_a_file_replaced_after_the_header_was_read_is_refused_at_first_fetch(tmp_path):
    import os

    import torch

    from soup_cli.utils.expert_stream import (
        ExpertShardReader,
        aligned_staging,
        write_expert_shard,
    )

    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False)
    reader = ExpertShardReader(path, count=1, shapes=info.shapes, quant_specs={}, dtype="float32")
    replacement = str(tmp_path / "other.safetensors")
    write_expert_shard(replacement, [(torch.ones(6, 4), torch.ones(4, 3))], quantise=False)
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="changed on disk since its header was read"):
        reader.read_host(0, aligned_staging(reader.max_span)[1])
    assert reader._handle is None


def test_install_refuses_mismatched_layers_and_expert_adapters(tmp_path):
    import torch.nn as nn

    from soup_cli.utils.expert_stream import ExpertFetcher, install_expert_streaming

    module, n_experts, _ = _deepseek_v3_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    fetcher = ExpertFetcher(device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="no experts file for it"):
        install_expert_streaming([_holder(module)], {}, fetcher)
    with pytest.raises(ValueError, match="which has no routed-experts module"):
        install_expert_streaming([nn.Module()], {0: reader}, fetcher)
    adapted = _holder(module)
    adapted.mlp.experts.lora_A = nn.Parameter(module.gate_up_proj.new_zeros(2, 2))
    with pytest.raises(ValueError, match="adapters on the routed experts"):
        install_expert_streaming([adapted], {0: reader}, fetcher)
    reader.close()


def test_two_read_install_refuses_a_moe_block_without_its_router(tmp_path):
    """Two reads replay the router's decisions, so the router must be found."""
    from soup_cli.utils.expert_stream import ExpertFetcher, install_expert_streaming

    module, n_experts, _ = _deepseek_v3_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    fetcher = ExpertFetcher(device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="no router module at `.gate`"):
        install_expert_streaming([_holder(module)], {0: reader}, fetcher, two_reads=True)
    reader.close()


def test_two_read_install_refuses_a_decoder_that_saves_the_moe_output(tmp_path):
    """The recompute's MoE output is zero. FlexOlmo normalises it before the residual add,
    and that RMSNorm saves its input: a zero there would give wrong gradients silently.
    OLMoE is refused as well: no test compares the real layer's gradients with the resident
    model's. Only decoder graphs checked to consume the MoE output with views and additions
    are admitted; the default three-read install still accepts both layers."""
    from transformers import FlexOlmoConfig, OlmoeConfig
    from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoDecoderLayer
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    from soup_cli.utils.expert_stream import ExpertFetcher, install_expert_streaming

    module, n_experts, _ = _olmoe_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    fetcher = ExpertFetcher(device="cpu", dtype="float32")
    shape = dict(
        hidden_size=32, intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        num_attention_heads=4, num_key_value_heads=4,
    )
    for layer_cls, config in (
        (FlexOlmoDecoderLayer, FlexOlmoConfig(**shape)),
        (OlmoeDecoderLayer, OlmoeConfig(**shape)),
    ):
        with pytest.raises(ValueError, match=f"is a {layer_cls.__name__}:"):
            install_expert_streaming([layer_cls(config, 0)], {0: reader}, fetcher, two_reads=True)
        assert install_expert_streaming([layer_cls(config, 0)], {0: reader}, fetcher) == 1
    reader.close()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("expert_layout", "experts", "unknown expert_layout"),
        ("expert_layers", [2, 1], "distinct ascending"),
        ("expert_layers", [9], "distinct ascending"),
        ("expert_count", 5000, "do not describe one layout"),
        ("expert_shapes", {"gate_up": [64, 0], "down": [64, 32]}, "do not describe one layout"),
    ],
)
def test_read_shard_index_refuses_a_malformed_expert_layout(tmp_path, field, value, match):
    import json

    from soup_cli.utils.layer_shard import read_shard_index

    weights, _ = _tiny_deepseek_v3(tmp_path)
    _shard(tmp_path, weights, expert_layout="per_expert")
    path = tmp_path / "shards" / "index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_shard_index(str(tmp_path / "shards"))


def test_missing_experts_file_is_a_cache_miss(tmp_path):
    import os

    from soup_cli.utils.layer_shard import expert_paths, inspect_shard_cache

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    os.remove(expert_paths(str(tmp_path / "shards"), index)[2])
    cached, reason = inspect_shard_cache(
        str(tmp_path / "shards"), "float32", index.source_fingerprint, index.source_files,
        "none", False, "", "", stripe_roots=(), expert_layout="per_expert",
    )
    assert cached is None and "experts_002.safetensors" in reason


def test_resharding_to_the_fused_layout_removes_the_experts_files(tmp_path):
    import os

    from soup_cli.utils.layer_shard import shard_cache_disk_bytes

    weights, _ = _tiny_deepseek_v3(tmp_path)
    per_expert = _shard(tmp_path, weights, expert_layout="per_expert")
    shards = tmp_path / "shards"
    assert shard_cache_disk_bytes(str(shards), per_expert) > sum(
        os.path.getsize(path) for path in shards.glob("layer_*.safetensors")
    )
    fused = _shard(tmp_path, weights)
    assert fused.expert_layout == ""
    assert not list(shards.glob("experts_*.safetensors"))


def test_per_expert_layout_refuses_a_checkpoint_without_per_expert_keys_before_writing(tmp_path):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    weights = tmp_path / "w"
    weights.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(8, 4),
            "model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 6, 4),
            "model.layers.0.mlp.experts.down_proj": torch.zeros(2, 4, 3),
        },
        str(weights / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="found no per-expert routed weights"):
        shard_checkpoint(
            str(weights), str(tmp_path / "s"), dtype="float32", arch="deepseek_v3",
            expert_layout="per_expert",
        )
    assert not list((tmp_path / "s").glob("layer_*.safetensors"))


# --------------------------------------------------------------------------
# experts file + reader
# --------------------------------------------------------------------------
def test_expert_shard_stores_gate_rows_before_up_rows(tmp_path):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.expert_stream import write_expert_shard

    gate_up = torch.cat((torch.full((3, 4), 1.0), torch.full((3, 4), 2.0)))
    down = torch.full((4, 3), 3.0)
    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(gate_up, down), (gate_up + 10, down + 10)], quantise=False)
    stored = load_file(path)
    assert info.count == 2 and dict(info.shapes) == {"gate_up": (6, 4), "down": (4, 3)}
    assert torch.equal(stored["1.gate_up"][:3], torch.full((3, 4), 11.0))
    assert torch.equal(stored["1.down"], torch.full((4, 3), 13.0))


def test_expert_shard_refuses_an_inconsistent_expert(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import write_expert_shard

    good = (torch.zeros(6, 4), torch.zeros(4, 3))
    bad = (torch.zeros(8, 4), torch.zeros(4, 3))
    with pytest.raises(ValueError, match="expert 1 gate_up has shape"):
        write_expert_shard(str(tmp_path / "x.safetensors"), [good, bad], quantise=False)


def test_reader_refuses_a_file_missing_an_expert(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False)
    with pytest.raises(ValueError, match="has no tensor '1.gate_up'"):
        ExpertShardReader(path, count=2, shapes=info.shapes, quant_specs={}, dtype="float32")


def _nf4_spec(shape, nested=False):
    from soup_cli.utils.layer_shard import NF4WeightSpec

    return NF4WeightSpec(
        shape=shape, dtype="bfloat16", blocksize=64, quant_type="nf4",
        nested=nested, nested_blocksize=256 if nested else 0,
    )


def _nf4_file(path, *, packed_bytes, absmax_blocks):
    """A one-expert NF4 experts file with chosen sidecar sizes (no bitsandbytes)."""
    import torch
    from safetensors.torch import save_file

    blob = {}
    for part, weights in (("gate_up", 6 * 64), ("down", 64 * 3)):
        n_packed = packed_bytes if part == "gate_up" else weights // 2
        n_blocks = absmax_blocks if part == "gate_up" else weights // 64
        blob[f"0.{part}"] = torch.zeros((n_packed, 1), dtype=torch.uint8)
        blob[f"0.{part}::absmax"] = torch.ones(n_blocks)
    save_file(blob, str(path))
    return str(path)


def test_reader_accepts_nf4_sidecars_of_exactly_the_spec_size(tmp_path):
    from soup_cli.utils.expert_stream import ExpertShardReader

    path = _nf4_file(tmp_path / "e.safetensors", packed_bytes=6 * 64 // 2, absmax_blocks=6)
    specs = {"gate_up": _nf4_spec((6, 64)), "down": _nf4_spec((64, 3))}
    reader = ExpertShardReader(
        path, count=1, shapes={"gate_up": (6, 64), "down": (64, 3)},
        quant_specs=specs, dtype="bfloat16",
    )
    reader.close()


@pytest.mark.parametrize(
    ("packed_bytes", "absmax_blocks"), [(16, 6), (6 * 64 // 2, 2)], ids=["packed", "absmax"]
)
def test_reader_refuses_nf4_sidecars_smaller_than_the_spec(tmp_path, packed_bytes, absmax_blocks):
    """The dequantisation reads prod(spec.shape) weights natively without a bounds
    check, so a header-valid file holding fewer bytes than the spec must be refused
    before any read (soup-c2's security read of this reader)."""
    from soup_cli.utils.expert_stream import ExpertShardReader

    path = _nf4_file(
        tmp_path / "e.safetensors", packed_bytes=packed_bytes, absmax_blocks=absmax_blocks
    )
    specs = {"gate_up": _nf4_spec((6, 64)), "down": _nf4_spec((64, 3))}
    with pytest.raises(ValueError, match="the index expects"):
        ExpertShardReader(
            path, count=1, shapes={"gate_up": (6, 64), "down": (64, 3)},
            quant_specs=specs, dtype="bfloat16",
        )


def test_reader_refuses_an_nf4_spec_that_disagrees_with_the_index_shape(tmp_path):
    from soup_cli.utils.expert_stream import ExpertShardReader

    path = _nf4_file(tmp_path / "e.safetensors", packed_bytes=6 * 64 // 2, absmax_blocks=6)
    specs = {"gate_up": _nf4_spec((12, 64)), "down": _nf4_spec((64, 3))}
    with pytest.raises(ValueError, match="is not the index's"):
        ExpertShardReader(
            path, count=1, shapes={"gate_up": (6, 64), "down": (64, 3)},
            quant_specs=specs, dtype="bfloat16",
        )


def test_reader_refuses_a_dense_dtype_other_than_the_index_dtype(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False)
    with pytest.raises(ValueError, match="the index expects torch.bfloat16"):
        ExpertShardReader(path, count=1, shapes=info.shapes, quant_specs={}, dtype="bfloat16")


def test_reader_refuses_an_expert_over_the_byte_limit(tmp_path, monkeypatch):
    import torch

    import soup_cli.utils.expert_stream as expert_stream

    path = str(tmp_path / "experts_001.safetensors")
    info = expert_stream.write_expert_shard(
        path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False
    )
    monkeypatch.setattr(expert_stream, "_MAX_EXPERT_BYTES", 1024)
    with pytest.raises(ValueError, match="over the 1024-byte limit"):
        expert_stream.ExpertShardReader(
            path, count=1, shapes=info.shapes, quant_specs={}, dtype="float32"
        )


def test_reader_refuses_a_file_swapped_while_it_opens(tmp_path, monkeypatch):
    """The header is read through the identity of the handle the reads use."""
    import dataclasses

    import torch

    import soup_cli.utils.safetensors_reader as safetensors_reader
    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False)
    real = safetensors_reader.read_header_with_identity

    def swapped(target):
        header, identity = real(target)
        return header, dataclasses.replace(identity, mtime_ns=identity.mtime_ns + 1)

    monkeypatch.setattr(safetensors_reader, "read_header_with_identity", swapped)
    with pytest.raises(ValueError, match="changed while it was being opened"):
        ExpertShardReader(path, count=1, shapes=info.shapes, quant_specs={}, dtype="float32")


def test_reader_refuses_a_file_with_more_experts_than_the_index(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(tmp_path / "experts_001.safetensors")
    pair = (torch.zeros(6, 4), torch.zeros(4, 3))
    info = write_expert_shard(path, [pair, (pair[0] + 1, pair[1] + 1)], quantise=False)
    with pytest.raises(ValueError, match="holds more than 1 experts"):
        ExpertShardReader(path, count=1, shapes=info.shapes, quant_specs={}, dtype="float32")


# --------------------------------------------------------------------------
# a whole streamed DeepSeek-V3 with per-expert layout vs the resident model
# --------------------------------------------------------------------------
def _tiny_deepseek_v3(tmp_path):
    import torch
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    torch.manual_seed(11)
    config = DeepseekV3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, n_group=2,
        topk_group=1, num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=24,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, max_position_embeddings=128,
    )
    model = DeepseekV3ForCausalLM(config).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator))
    weights = tmp_path / "model"
    model.save_pretrained(str(weights))
    # The streamed experts follow the eager loop; the resident control must too.
    model.set_experts_implementation("eager")
    return str(weights), model


def _lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_b_proj", "kv_b_proj", "o_proj"], task_type=TaskType.CAUSAL_LM,
    )


def _shard(tmp_path, weights, dtype="float32", **kwargs):
    from soup_cli.utils.layer_shard import shard_checkpoint

    return shard_checkpoint(
        weights, str(tmp_path / "shards"), dtype=dtype, arch="deepseek_v3",
        num_hidden_layers=3, **kwargs,
    )


def _paired(tmp_path, two_read_experts=False):
    import torch
    from peft import get_peft_model

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, resident = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
        lora_config=_lora(), device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
        two_read_experts=two_read_experts,
    )
    ref = get_peft_model(resident, _lora())
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    src = {k.replace(".inner.", "."): v for k, v in model.state_dict().items() if "lora_" in k}
    dst = {k: v for k, v in ref.state_dict().items() if "lora_" in k}
    assert src and set(src) == set(dst)
    for key, value in src.items():
        dst[key].copy_(value)
    model.eval()
    ref.eval()
    ids = torch.randint(0, 128, (2, 12), generator=generator)
    return model, runtime, ref, ids, index


def test_per_expert_layout_moves_routed_experts_out_of_the_layer_file(tmp_path):
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import expert_paths, layer_paths, read_shard_index

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    assert index.expert_layout == "per_expert"
    assert index.expert_layers == (1, 2)
    assert index.expert_count == 8
    assert dict(index.expert_shapes) == {"gate_up": (64, 64), "down": (64, 32)}
    assert read_shard_index(str(tmp_path / "shards")) == index
    layer_one = load_file(layer_paths(str(tmp_path / "shards"), index)[1])
    assert not any(key.startswith("mlp.experts") for key in layer_one)
    assert "mlp.shared_experts.gate_proj.weight" in layer_one
    assert set(expert_paths(str(tmp_path / "shards"), index)) == {1, 2}


def test_streamed_per_expert_forward_matches_resident_eager_bit_exactly(tmp_path):
    import torch

    model, runtime, ref, ids, _ = _paired(tmp_path)
    with torch.no_grad():
        got = model(input_ids=ids, labels=ids)
        want = ref(input_ids=ids, labels=ids)
    assert torch.equal(got.logits, want.logits), (got.logits - want.logits).abs().max().item()
    assert torch.equal(got.loss, want.loss)
    assert runtime.stats()["expert_fetches"] > 0
    assert runtime.expert_readers
    assert all(reader._handle is not None for reader in runtime.expert_readers)
    runtime.close()
    # Closing the runtime releases every experts file (Windows keeps an open file locked).
    assert all(reader._handle is None for reader in runtime.expert_readers)


@pytest.mark.parametrize(
    ("two_read_experts", "reads"), [(False, 3), (True, 2)], ids=["default", "two-reads"]
)
def test_streamed_per_expert_backward_matches_resident_lora_gradients(
    tmp_path, two_read_experts, reads
):
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    model, runtime, ref, ids, _ = _paired(tmp_path, two_read_experts)
    loss = model(input_ids=ids, labels=ids).loss
    forward_reads = [reader.fetches for reader in runtime.expert_readers]
    loss.backward()
    # Per MoE layer, under the checkpointed decoder layer: the default path reads a
    # routed expert three times (forward, the checkpoint's recompute, the expert's
    # own backward); with two reads the recompute reads none.
    assert len(forward_reads) == 2 and all(count > 0 for count in forward_reads)
    assert [reader.fetches for reader in runtime.expert_readers] == [
        reads * count for count in forward_reads
    ]
    ref(input_ids=ids, labels=ids).loss.backward()

    def grads(net):
        return {
            name: parameter.grad
            for name, parameter in canonical_named_parameters(net)
            if "lora_" in name and parameter.grad is not None
        }

    streamed, resident = grads(model), grads(ref)
    assert streamed and set(streamed) == set(resident)
    # Layer 0's adapters sit below both MoE layers: their gradient crossed
    # every streamed expert's backward.
    assert any(".layers.0." in name and float(g.abs().sum()) > 0 for name, g in streamed.items())
    for name, gradient in streamed.items():
        assert torch.equal(gradient, resident[name]), name
    runtime.close()


def _three_way(tmp_path, dtype):
    """The resident eager model and two streamed copies of it, the default path
    and two reads, holding equal adapters."""
    import torch
    from peft import get_peft_model
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, dtype, expert_layout="per_expert")
    streamed = []
    for two_read_experts in (False, True):
        model, runtime = build_streamed_model(
            model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
            lora_config=_lora(), device="cpu", dtype=dtype, buffers=2, pin=False, seed=3,
            two_read_experts=two_read_experts,
        )
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
        streamed.append((model.eval(), runtime))
    resident = DeepseekV3ForCausalLM.from_pretrained(weights, dtype=getattr(torch, dtype))
    resident.set_experts_implementation("eager")
    ref = get_peft_model(resident, _lora()).eval()
    adapters = [
        {k.replace(".inner.", "."): v for k, v in model.state_dict().items() if "lora_" in k}
        for model, _ in streamed
    ]
    dst = {k: v for k, v in ref.state_dict().items() if "lora_" in k}
    assert adapters[0] and set(adapters[0]) == set(adapters[1]) == set(dst)
    for key, value in adapters[0].items():
        assert torch.equal(adapters[1][key], value), key
        dst[key].copy_(value)
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(29))
    return ref, streamed[0], streamed[1], ids


def _step_capturing(net, ids, autocast=False):
    """One forward + backward; every tensor the three-way parity compares.

    Hooks capture the gradient at each decoder layer's input (dX) and at both
    outputs of each MoE router that carry one: the routing weights, and the
    logits. The router weight is frozen in training; here it is made to require
    grad, as the CUDA gate does, to show the gradient reaches it. A streamed
    router weight is a view of a pool slot, so each wrapper call substitutes a
    private leaf copy of it instead; the forward call's leaf is the one autograd
    accumulates into, the recompute's never receives a gradient.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    captured = {}
    expected_lora = {
        name: parameter for name, parameter in canonical_named_parameters(net) if "lora_" in name
    }
    assert expected_lora and all(parameter.requires_grad for parameter in expected_lora.values())

    def keep(key):
        def hook(grad):
            captured[key] = grad.detach().clone()

        return hook

    def on_input(idx):
        def hook(_module, args):
            if args[0].requires_grad:
                args[0].register_hook(keep(("dX", idx)))

        return hook

    def on_router(idx):
        # Also fires in a checkpoint recompute, whose tensors never receive a gradient.
        def hook(_module, _args, output):
            output[0].register_hook(keep(("router_logits", idx)))
            output[1].register_hook(keep(("routing_weights", idx)))

        return hook

    def private_router_weight(real, leaves):
        def substituted(buffers):
            weights = dict(real(buffers))
            leaf = weights["mlp.gate.weight"].detach().clone().requires_grad_(True)
            leaves.append(leaf)
            weights["mlp.gate.weight"] = leaf
            return weights

        return substituted

    handles = []
    router_leaves = {}
    for idx, layer in enumerate(net.get_submodule("base_model.model.model.layers")):
        handles.append(layer.register_forward_pre_hook(on_input(idx)))
        inner = getattr(layer, "inner", None)
        router = getattr((inner or layer).mlp, "gate", None)
        if router is None:
            continue
        handles.append(router.register_forward_hook(on_router(idx)))
        if inner is None:
            router.weight.requires_grad_(True)
            router_leaves[idx] = [router.weight]
        else:
            router_leaves[idx] = []
            layer._substituted_weights = private_router_weight(
                layer._substituted_weights, router_leaves[idx]
            )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = net(input_ids=ids, labels=ids)
        out.loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    captured["logits"] = out.logits.detach()
    captured["loss"] = out.loss.detach()
    for idx, leaves in router_leaves.items():
        grads = [leaf.grad for leaf in leaves if leaf.grad is not None]
        assert len(grads) == 1, (idx, len(grads))
        captured[("router_weight", idx)] = grads[0].clone()
    for name, parameter in expected_lora.items():
        assert parameter.grad is not None, f"missing expected adapter gradient: {name}"
        captured[("grad", name)] = parameter.grad.detach().clone()
    return captured


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param("fp32", id="fp32"),
        pytest.param("bf16", id="bf16", marks=skip_on_windows_ci),
        pytest.param("bf16-autocast", id="bf16-autocast", marks=skip_on_windows_ci),
    ],
)
def test_two_read_experts_equal_the_resident_model_and_the_default_path(tmp_path, cell):
    """Logits, loss, dX, the router-output gradients, the router weight's gradient and
    every LoRA gradient of the resident eager model, the default streamed path and two
    reads, bit for bit. The bf16-autocast cell runs the forward under CPU autocast and
    the backward outside."""
    import torch

    ref, (default, default_runtime), (two, two_runtime), ids = _three_way(
        tmp_path, "bfloat16" if cell == "bf16" else "float32"
    )
    autocast = cell == "bf16-autocast"
    try:
        want = _step_capturing(ref, ids, autocast)
        got_default = _step_capturing(default, ids, autocast)
        got = _step_capturing(two, ids, autocast)
    finally:
        default_runtime.close()
        two_runtime.close()
    kinds = {key[0] for key in want if isinstance(key, tuple)}
    assert kinds == {"dX", "router_logits", "routing_weights", "router_weight", "grad"}
    for kind in ("routing_weights", "router_weight"):
        assert {key for key in want if key[0] == kind} == {(kind, 1), (kind, 2)}
        assert float(want[(kind, 1)].abs().sum()) > 0.0
    assert set(got) == set(got_default) == set(want)
    for key, value in want.items():
        assert torch.equal(got_default[key], value), key
        assert torch.equal(got[key], value), key


def test_two_read_recompute_reuses_the_forward_routing_decisions(tmp_path):
    """Between forward and backward every router's correction bias is negated and
    scaled. It steers only WHICH experts are chosen, so a recompute that selected
    again would change the gradients; one that reuses the forward's decisions
    leaves them equal. A fresh forward under the new bias shows it does choose
    other experts."""
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    model, runtime, _, ids, _ = _paired(tmp_path, two_read_experts=True)
    layers = model.get_submodule("base_model.model.model.layers")
    routers = [layer.inner.mlp.gate for layer in layers if hasattr(layer.inner.mlp, "gate")]
    assert len(routers) == 2

    def grads():
        return {
            name: parameter.grad.clone()
            for name, parameter in canonical_named_parameters(model)
            if parameter.grad is not None
        }

    def choices_of(step):
        chosen = []
        handles = [
            router.register_forward_hook(
                lambda _m, _a, out: chosen.append(out[2].sort(dim=-1).values)
            )
            for router in routers
        ]
        try:
            return step(), chosen
        finally:
            for handle in handles:
                handle.remove()

    model(input_ids=ids, labels=ids).loss.backward()
    want = grads()
    model.zero_grad(set_to_none=True)

    loss, before = choices_of(lambda: model(input_ids=ids, labels=ids).loss)
    with torch.no_grad():
        for router in routers:
            router.e_score_correction_bias.mul_(-10.0)
    loss.backward()
    got = grads()
    with torch.no_grad():
        _, after = choices_of(lambda: model(input_ids=ids, labels=ids))
    runtime.close()

    assert len(before) == len(after) == 2
    assert all(not torch.equal(old, new) for old, new in zip(before, after))
    assert want and set(got) == set(want)
    for name, gradient in want.items():
        assert torch.equal(got[name], gradient), name


@pytest.mark.parametrize("two_read_experts", [False, True], ids=["default", "two-reads"])
def test_streamed_model_holds_one_expert_at_a_time(tmp_path, two_read_experts):
    """The VRAM bound through the checkpointed decoder layer: when an expert is
    fetched, no expert fetched before it is still alive (forward, checkpoint
    recompute, backward), and none outlives the step."""
    import weakref

    model, runtime, _, ids, _ = _paired(tmp_path, two_read_experts)
    alive_at_fetch = []
    refs = []
    for module in model.modules():
        if type(module).__name__ != "StreamedExperts":
            continue

        def tracking(expert, real=module.fetch):
            alive_at_fetch.append(sum(ref() is not None for ref in refs))
            gate_up, down = real(expert)
            refs.extend((weakref.ref(gate_up), weakref.ref(down)))
            return gate_up, down

        module.fetch = tracking
    model(input_ids=ids, labels=ids).loss.backward()
    runtime.close()
    assert alive_at_fetch and max(alive_at_fetch) == 0
    assert all(ref() is None for ref in refs)


def test_two_read_replay_refuses_a_router_call_the_forward_did_not_make(tmp_path):
    """The recompute hands each router call the decision recorded at its position; a
    call that does not match it must fail loudly, never route with another's choice."""
    import torch
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter

    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        install_expert_streaming,
        two_read_checkpoint_contexts,
    )

    module, n_experts, _ = _olmoe_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    router = OlmoeTopKRouter(OlmoeConfig(hidden_size=32, num_experts=8, num_experts_per_tok=2))
    holder = _holder(module, gate=router)
    install_expert_streaming(
        [holder], {0: reader}, ExpertFetcher(device="cpu", dtype="float32"), two_reads=True
    )
    forward, replay = two_read_checkpoint_contexts()
    with forward:
        holder.mlp.gate(torch.randn(6, 32))
    with replay, pytest.raises(RuntimeError, match="which the forward did not take"):
        holder.mlp.gate(torch.randn(7, 32))
    reader.close()


def test_two_read_replay_hands_out_owned_copies_of_the_recorded_decisions(tmp_path):
    """A caller may edit a router's outputs in place (OLMoE-style routers divide the
    top-k values in place). That must change neither what the record holds nor what
    any later recompute receives."""
    import torch
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter

    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        install_expert_streaming,
        two_read_checkpoint_contexts,
    )

    module, n_experts, _ = _olmoe_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    router = OlmoeTopKRouter(OlmoeConfig(hidden_size=32, num_experts=8, num_experts_per_tok=2))
    holder = _holder(module, gate=router)
    install_expert_streaming(
        [holder], {0: reader}, ExpertFetcher(device="cpu", dtype="float32"), two_reads=True
    )
    forward, replay = two_read_checkpoint_contexts()
    hidden = torch.randn(6, 32)
    with forward:
        decided = holder.mlp.gate(hidden)
    want = decided[2].clone()
    decided[2].zero_()  # the forward's caller edits its result in place
    for _ in range(2):  # two recomputes, as under retain_graph
        with replay:
            replayed = holder.mlp.gate(hidden)
            assert torch.equal(replayed[2], want)
            replayed[2].zero_()  # an edit of a replayed result must not reach the next one
    reader.close()


def test_two_read_experts_are_refused_on_a_fused_cache(tmp_path):
    """Accepted and ignored would train with three reads while reporting two."""
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights)
    with pytest.raises(ValueError, match="expert_layout='per_expert'"):
        build_streamed_model(
            model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
            lora_config=_lora(), device="cpu", dtype="float32", buffers=2, pin=False,
            two_read_experts=True,
        )


def test_layout_change_is_a_cache_miss_naming_the_layout(tmp_path):
    from soup_cli.utils.layer_shard import inspect_shard_cache

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    cached, reason = inspect_shard_cache(
        str(tmp_path / "shards"), "float32", index.source_fingerprint, index.source_files,
        "none", False, "", "", stripe_roots=(), expert_layout="",
    )
    assert cached is None
    assert "expert layout changed" in reason
    reused = _shard(tmp_path, weights, expert_layout="per_expert")
    assert reused == index


def test_per_expert_layout_is_refused_for_an_unadmitted_arch(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    with pytest.raises(ValueError, match="admitted: deepseek_v3"):
        shard_checkpoint(
            str(tmp_path), str(tmp_path / "s"), dtype="float32", arch="qwen4_exp",
            expert_layout="per_expert",
        )


def test_unknown_expert_layout_is_refused(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    with pytest.raises(ValueError, match="unsupported expert_layout"):
        shard_checkpoint(
            str(tmp_path), str(tmp_path / "s"), dtype="float32", arch="deepseek_v3",
            expert_layout="experts",
        )


def test_install_refuses_a_model_whose_experts_disagree_with_the_file(tmp_path):
    from soup_cli.utils.expert_stream import ExpertFetcher, install_expert_streaming

    module, n_experts, _ = _deepseek_v3_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    reader.count = 4  # what a stale index would claim
    holder = _holder(module)
    with pytest.raises(ValueError, match="reshard the checkpoint"):
        install_expert_streaming(
            [holder], {0: reader}, ExpertFetcher(device="cpu", dtype="float32")
        )
    reader.close()


def _holder(experts, gate=None):
    """A minimal decoder layer around one MoE block. It carries the class name of an
    admitted two-read decoder, so router and replay checks are reached, not the
    decoder-graph admission that ``test_two_read_install_refuses_a_decoder...`` covers."""
    import torch.nn as nn

    class DeepseekV3DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = experts
            if gate is not None:
                self.mlp.gate = gate

    return DeepseekV3DecoderLayer()


# --------------------------------------------------------------------------
# NF4 experts (CUDA)
# --------------------------------------------------------------------------
@CUDA
def test_sharder_writes_nf4_per_expert_files_a_streamed_model_runs_on(tmp_path):
    """The NF4 path of shard_checkpoint(expert_layout="per_expert") end to end."""
    pytest.importorskip("bitsandbytes")
    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )

    weights, _ = _tiny_deepseek_v3(tmp_path)
    probe = build_meta_skeleton(weights, dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    index = shard_checkpoint(
        weights, str(tmp_path / "shards"), dtype="bfloat16", arch="deepseek_v3", quant="nf4",
        quant_suffixes=suffixes, quant_device="cuda", num_hidden_layers=3,
        expert_layout="per_expert",
    )
    assert index.expert_quant_specs["gate_up"].nested is True
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
        lora_config=_lora(), device="cuda", dtype="bfloat16", buffers=2, pin=False,
        quant="nf4", double_quant=True,
    )
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(1)).cuda()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert runtime.stats()["expert_fetches"] > 0
    runtime.close()


@CUDA
def test_nf4_streamed_experts_equal_the_dequantised_resident_block(tmp_path):
    pytest.importorskip("bitsandbytes")
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        ExpertShardReader,
        streamed_experts_class,
        write_expert_shard,
    )
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    module, n_experts, top_k = _deepseek_v3_experts(5)
    module = module.to("cuda", torch.bfloat16)
    path = str(tmp_path / "experts_001.safetensors")
    pairs = [
        (module.gate_up_proj[e].detach(), module.down_proj[e].detach())
        for e in range(n_experts)
    ]
    info = write_expert_shard(path, iter(pairs), quantise=True, double_quant=True, device="cuda")
    with torch.no_grad():
        for e, (gate_up, down) in enumerate(pairs):
            for target, tensor in ((module.gate_up_proj, gate_up), (module.down_proj, down)):
                packed, state = quantize_4bit(
                    tensor, blocksize=64, compress_statistics=True, quant_type="nf4"
                )
                target[e].copy_(dequantize_4bit(packed, quant_state=state))
    codes = {NF4_CODE_KEY: info.codes[0].cuda(), NF4_NESTED_CODE_KEY: info.codes[1].cuda()}
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs, dtype="bfloat16"
    )
    streamed = streamed_experts_class()(
        reader, ExpertFetcher(device="cuda", dtype="bfloat16", codes=codes), module.act_fn, 1
    )
    states, index, weights, _ = _routing(24, n_experts, top_k, 32, 9)
    states = states.detach().to("cuda", torch.bfloat16)
    index = index.cuda()
    weights = weights.detach().to("cuda", torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(streamed(states, index, weights), module(states, index, weights))
    reader.close()


# --------------------------------------------------------------------------
# the synthetic store writer and the one-step harness (benchmarks/harness)
# --------------------------------------------------------------------------
def _harness(name):
    import importlib.util
    import sys
    from pathlib import Path

    harness = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location(name, harness / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_tiny_store_is_a_cache_the_runtime_accepts(tmp_path):
    """The writer skips the sharder, so its index and files must be exactly
    what the sharder would write: the index reads back, every experts file
    opens with the index's shapes and NF4 specs, and every linear has its
    sidecars."""
    from safetensors import safe_open

    from soup_cli.utils.expert_stream import ExpertShardReader
    from soup_cli.utils.layer_shard import expert_paths, layer_paths, read_shard_index

    store = str(tmp_path / "store")
    summary = _harness("synth_moe_store").write_store(store, "tiny", log=lambda _line: None)
    assert summary["label"] == "random weights, shape only"
    index = read_shard_index(store)
    assert index.arch == "deepseek_v3" and index.quant == "nf4"
    assert index.expert_layers == (1, 2) and index.expert_count == 8
    for layer, path in expert_paths(store, index).items():
        reader = ExpertShardReader(
            path, count=index.expert_count, shapes=index.expert_shapes,
            quant_specs=index.expert_quant_specs, dtype=index.dtype,
        )
        reader.close()
    with safe_open(layer_paths(store, index)[1], framework="pt") as handle:
        keys = set(handle.keys())
    assert "self_attn.q_a_proj.weight::absmax" in keys
    assert "self_attn.q_a_proj.weight::nested_offset" in keys
    assert index.double_quant is True
    assert index.expert_quant_specs["down"].nested is True
    assert "mlp.shared_experts.down_proj.weight" in keys
    assert "mlp.gate.e_score_correction_bias" in keys


def test_synthetic_writer_refuses_an_existing_cache(tmp_path, capsys):
    store = tmp_path / "store"
    store.mkdir()
    (store / "index.json").write_text("{}", encoding="utf-8")
    assert _harness("synth_moe_store").main(["--out", str(store), "--shape", "tiny"]) == 2
    assert "already holds a shard cache" in capsys.readouterr().out


@CUDA
def test_one_step_on_the_synthetic_tiny_store(tmp_path):
    import json

    pytest.importorskip("bitsandbytes")
    store = str(tmp_path / "store")
    _harness("synth_moe_store").write_store(store, "tiny", log=lambda _line: None)
    out = tmp_path / "step.json"
    assert _harness("moe_stream_step").main(
        ["--store", store, "--out", str(out), "--seq", "16", "--steps", "2", "--no-pin"]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["label"] == "random weights, shape only"
    assert len(report["steps"]) == 2
    for step in report["steps"]:
        assert step["loss_finite"] and step["grads_finite"]
        assert step["expert_fetches"] > 0
