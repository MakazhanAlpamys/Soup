"""Layer streaming's disk pre-flight sizes the shard cache the way the sharder writes it.

`soup train` with `training.stream_layers` shards the base into a per-layer cache and,
before writing a byte, refuses when the cache volume cannot hold it. The estimate used
to charge the NF4 rate to every parameter. In the default `fused` expert layout the
sharder keeps the stacked routed experts at the stream dtype, so for Moonlight-16B-A3B
(`deepseek_v3`) the pre-flight said ~8.4 GB and the sharder then wrote 30.6 GB: a disk
could pass the check and fill mid-shard.

The estimate now reads the source safetensors HEADERS (shapes and dtypes, no tensor is
loaded or mapped) and applies the sharder's own storage decisions to each tensor:
which decoder keys become NF4 (the `quant_suffixes` the meta probe found), which stay
float32, what the routed experts become in each layout, and the exact NF4 sidecar bytes.
"""

import json
import math
import os
import struct
import types

import pytest

# --------------------------------------------------------------------------
# a synthetic DeepSeek-V3-shaped checkpoint (real headers, zero payload)
# --------------------------------------------------------------------------
HIDDEN, DENSE_INTER, MOE_INTER, EXPERTS, VOCAB = 64, 128, 32, 4, 128
N_LAYERS = 3  # layer 0 dense (first_k_dense_replace=1), layers 1-2 MoE
_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}

#: What the meta probe reports for this layout: every nn.Linear of a decoder layer.
#: The router weight, the norms, the bias and the fused experts are not in it.
SUFFIXES = frozenset(
    {
        "self_attn.q_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    }
)


def _moe_tensors(*, source_dtype="BF16", extra_layers=0, n_layers=N_LAYERS):
    """name -> (safetensors dtype, shape) for a DeepSeek-V3-shaped base."""
    dt = source_dtype
    tensors = {
        "model.embed_tokens.weight": (dt, (VOCAB, HIDDEN)),
        "lm_head.weight": (dt, (VOCAB, HIDDEN)),
        "model.norm.weight": (dt, (HIDDEN,)),
    }
    for idx in range(n_layers + extra_layers):
        prefix = f"model.layers.{idx}."
        tensors[prefix + "input_layernorm.weight"] = (dt, (HIDDEN,))
        tensors[prefix + "post_attention_layernorm.weight"] = (dt, (HIDDEN,))
        tensors[prefix + "self_attn.q_proj.weight"] = (dt, (HIDDEN, HIDDEN))
        tensors[prefix + "self_attn.o_proj.weight"] = (dt, (HIDDEN, HIDDEN))
        if idx == 0:
            tensors[prefix + "mlp.gate_proj.weight"] = (dt, (DENSE_INTER, HIDDEN))
            tensors[prefix + "mlp.up_proj.weight"] = (dt, (DENSE_INTER, HIDDEN))
            tensors[prefix + "mlp.down_proj.weight"] = (dt, (HIDDEN, DENSE_INTER))
            continue
        tensors[prefix + "mlp.gate.weight"] = (dt, (EXPERTS, HIDDEN))
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = ("F32", (EXPERTS,))
        for name in ("gate_proj", "up_proj"):
            tensors[prefix + f"mlp.shared_experts.{name}.weight"] = (dt, (MOE_INTER, HIDDEN))
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = (dt, (HIDDEN, MOE_INTER))
        for expert in range(EXPERTS):
            for name in ("gate_proj", "up_proj"):
                key = prefix + f"mlp.experts.{expert}.{name}.weight"
                tensors[key] = (dt, (MOE_INTER, HIDDEN))
            tensors[prefix + f"mlp.experts.{expert}.down_proj.weight"] = (
                dt,
                (HIDDEN, MOE_INTER),
            )
    return tensors


def _dense_tensors(*, source_dtype="BF16"):
    """A Llama-shaped base: the same layout without the MoE layers."""
    return _moe_tensors(source_dtype=source_dtype, n_layers=1)


def _write_checkpoint(directory, tensors, name="model.safetensors"):
    """A safetensors file with a real header and an all-zero payload; returns its path."""
    directory.mkdir(parents=True, exist_ok=True)
    header, cursor = {}, 0
    for key, (dtype, shape) in tensors.items():
        size = math.prod(shape) * _ITEMSIZE[dtype]
        offsets = [cursor, cursor + size]
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": offsets}
        cursor += size
    raw = json.dumps(header).encode("utf-8")
    path = directory / name
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * cursor)
    return str(path)


#: Headers `read_header` cannot turn into shapes, by the exception each one raises there:
#: a null or nested dimension and a null offset are TypeErrors, an infinite dimension is
#: an OverflowError, 200 KB of nested brackets a RecursionError, the rest ValueErrors.
_MALFORMED = {
    "shape-null": '{"t":{"dtype":"BF16","shape":[null],"data_offsets":[0,2]}}',
    "shape-nested": '{"t":{"dtype":"BF16","shape":[[2]],"data_offsets":[0,4]}}',
    "offsets-null": '{"t":{"dtype":"BF16","shape":[2],"data_offsets":[null,4]}}',
    "shape-infinite": '{"t":{"dtype":"BF16","shape":[Infinity],"data_offsets":[0,4]}}',
    "nested-brackets": "[" * 200_000,
    "not-json": "not json at all",
    "unknown-dtype": '{"t":{"dtype":"C64","shape":[2],"data_offsets":[0,16]}}',
}
#: An FP8 header used to be unreadable here and fell back. It is read now, and an FP8
#: tensor that is not a DeepSeek-V3 block weight with its scale is a checkpoint the
#: sharder refuses: refused here in its words (tests/test_fp8_block_streaming.py).
_FP8_WITHOUT_SCALE = '{"t":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}'


def _layer_stack(n_layers):
    """The smallest checkpoint with `n_layers` contiguous decoder layers."""
    tensors = {"model.embed_tokens.weight": ("BF16", (4, 2))}
    for idx in range(n_layers):
        tensors[f"model.layers.{idx}.input_layernorm.weight"] = ("BF16", (2,))
    return tensors


def _write_raw(path, header_text, payload=b"\0" * 8):
    """A file with an arbitrary header text after its length field, then a payload."""
    raw = header_text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return str(path)


def _header_and_payload(path):
    """(bytes of the length field + header, bytes of tensor data) of one safetensors file."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
    return 8 + length, os.path.getsize(path) - 8 - length


def _nf4(numel, double_quant=True):
    """The oracle: packed nibbles + one absmax per 64 (uint8 + nested fp32 per 256 + offset)."""
    blocks = -(-numel // 64)
    packed = (numel + 1) // 2
    if not double_quant:
        return packed + 4 * blocks
    return packed + blocks + 4 * -(-blocks // 256) + 4


def _expected(layout, *, nf4, double_quant=True, itemsize=2, n_layers=N_LAYERS):
    """(per-layer payload, other payload) the sharder writes for `_moe_tensors()`."""

    def stored(numel, quantised):
        return _nf4(numel, double_quant) if quantised else numel * itemsize

    layers = []
    for idx in range(n_layers):
        total = 2 * HIDDEN * itemsize  # the two norms stay at the stream dtype
        total += 2 * stored(HIDDEN * HIDDEN, nf4)  # q_proj, o_proj
        if idx == 0:
            total += 3 * stored(DENSE_INTER * HIDDEN, nf4)
        else:
            total += EXPERTS * HIDDEN * itemsize  # the router weight is not an nn.Linear
            total += EXPERTS * 4  # e_score_correction_bias is kept at float32
            total += 3 * stored(MOE_INTER * HIDDEN, nf4)  # the shared expert
            if layout == "per_expert":
                # one (gate_up, down) pair per expert, each matrix quantised on its own
                total += EXPERTS * (
                    stored(2 * MOE_INTER * HIDDEN, nf4) + stored(HIDDEN * MOE_INTER, nf4)
                )
            else:
                # fused: stacked 3-D gate_up_proj / down_proj at the stream dtype
                total += EXPERTS * 3 * MOE_INTER * HIDDEN * itemsize
        layers.append(total)
    other = (2 * VOCAB * HIDDEN + HIDDEN) * itemsize
    if nf4:
        other += 16 * 4 + (256 * 4 if double_quant else 0)  # the shared code tables
    return tuple(layers), other


def _estimate(paths, *, layout="", quant="nf4", suffixes=SUFFIXES, **kwargs):
    from soup_cli.utils.layer_shard import estimate_shard_cache_bytes

    kwargs.setdefault("dtype", "bfloat16")
    kwargs.setdefault("arch", "deepseek_v3")
    return estimate_shard_cache_bytes(
        paths,
        quant=quant,
        quant_suffixes=suffixes if quant == "nf4" else (),
        expert_layout=layout,
        **kwargs,
    )


def _coarse(source_bytes, quant="nf4"):
    """The pre-fix estimate: the NF4 rate charged to every parameter."""
    from soup_cli.utils.layer_stream import estimate_stream_store_bytes

    return estimate_stream_store_bytes(source_bytes, dtype="bfloat16", quant=quant)


# --------------------------------------------------------------------------
# NF4 byte accounting
# --------------------------------------------------------------------------
class TestNf4WeightBytes:
    @pytest.mark.parametrize(
        ("numel", "double_quant", "expected"),
        [
            (4096, True, 2048 + 64 + 4 + 4),
            (4096, False, 2048 + 4 * 64),
            (63, True, 32 + 1 + 4 + 4),  # odd numel: one packed byte holds one nibble
            (63, False, 32 + 4),
            (42624, True, 21312 + 666 + 4 * 3 + 4),  # 666 blocks -> three nested blocks
            (2816 * 2048, True, 2975108),  # one Moonlight expert gate_up matrix
            (2816 * 2048, False, 3244032),
        ],
    )
    def test_hand_counted_sizes(self, numel, double_quant, expected):
        from soup_cli.utils.layer_shard import nf4_weight_bytes

        assert nf4_weight_bytes(numel, double_quant=double_quant) == expected

    @pytest.mark.parametrize("double_quant", [True, False], ids=["double", "single"])
    @pytest.mark.parametrize("shape", [(64, 64), (7, 9), (128, 333), (300, 2048)])
    def test_matches_what_the_sharder_stores(self, shape, double_quant):
        """Pinned against `_quantize_nf4` (bitsandbytes on CPU), the sharder's own call."""
        pytest.importorskip("bitsandbytes")
        torch = pytest.importorskip("torch")
        from soup_cli.utils.layer_shard import (
            _nf4_sidecar_suffixes,
            _quantize_nf4,
            nf4_weight_bytes,
        )

        weight = torch.randn(shape, generator=torch.Generator().manual_seed(5)).to(
            torch.bfloat16
        )
        sidecars, _spec, _code, _nested = _quantize_nf4(
            weight, double_quant=double_quant, device="cpu"
        )
        stored = sum(value.numel() * value.element_size() for value in sidecars.values())
        assert nf4_weight_bytes(weight.numel(), double_quant=double_quant) == stored
        # the header allowance names the same stored keys the sharder writes
        assert sorted(_nf4_sidecar_suffixes(double_quant)) == sorted(sidecars)

    def test_an_empty_weight_is_refused(self):
        from soup_cli.utils.layer_shard import nf4_weight_bytes

        with pytest.raises(ValueError, match="positive"):
            nf4_weight_bytes(0, double_quant=True)


# --------------------------------------------------------------------------
# the estimate, from headers
# --------------------------------------------------------------------------
class TestEstimateFromHeaders:
    def test_fused_nf4_keeps_the_routed_experts_at_the_stream_dtype(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _estimate([path], layout="")
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=True)

    def test_per_expert_nf4_quantises_each_expert_matrix(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _estimate([path], layout="per_expert")
        assert (got.layer_bytes, got.other_bytes) == _expected("per_expert", nf4=True)

    @pytest.mark.parametrize("layout", ["", "per_expert"], ids=["fused", "per_expert"])
    def test_single_quant_uses_float32_absmax(self, tmp_path, layout):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _estimate([path], layout=layout, double_quant=False)
        expected = _expected(layout or "fused", nf4=True, double_quant=False)
        assert (got.layer_bytes, got.other_bytes) == expected

    @pytest.mark.parametrize("layout", ["", "per_expert"], ids=["fused", "per_expert"])
    def test_unquantised_charges_every_tensor_at_the_stream_dtype(self, tmp_path, layout):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _estimate([path], layout=layout, quant="none")
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=False)
        # a bf16 source at the bf16 stream dtype: the payload is the source payload
        source_payload = sum(
            math.prod(shape) * _ITEMSIZE[dtype] for dtype, shape in _moe_tensors().values()
        )
        assert sum(got.layer_bytes) + got.other_bytes == source_payload

    def test_a_float32_source_is_charged_at_the_stream_dtype(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors(source_dtype="F32"))
        got = _estimate([path], layout="")
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=True)

    def test_a_float32_stream_dtype_doubles_the_unquantised_part(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _estimate([path], layout="", dtype="float32")
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=True, itemsize=4)

    def test_the_fused_cache_is_far_above_the_old_nf4_rate(self, tmp_path):
        """The defect itself: the routed experts are most of an MoE base."""
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        fused = _estimate([path], layout="")
        per_expert = _estimate([path], layout="per_expert")
        assert fused.total_bytes > 2 * _coarse(os.path.getsize(path))
        # an MoE layer: stacked experts at bf16 against one NF4 matrix per expert
        assert fused.layer_bytes[1] > 2 * per_expert.layer_bytes[1]

    def test_layers_past_num_hidden_layers_are_not_charged(self, tmp_path):
        """DeepSeek-V3 / Kimi K2 MTP layers: the sharder drops them, so does the estimate."""
        path = _write_checkpoint(tmp_path / "m", _moe_tensors(extra_layers=1))
        got = _estimate([path], layout="", num_hidden_layers=N_LAYERS)
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=True)
        kept = _estimate([path], layout="")
        assert len(kept.layer_bytes) == N_LAYERS + 1

    def test_the_tensors_may_span_several_files(self, tmp_path):
        tensors = list(_moe_tensors().items())
        half = len(tensors) // 2
        first = _write_checkpoint(tmp_path / "m", dict(tensors[:half]), "a.safetensors")
        second = _write_checkpoint(tmp_path / "m", dict(tensors[half:]), "b.safetensors")
        got = _estimate([first, second], layout="")
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=True)

    def test_a_dense_unquantised_base_is_charged_its_source_payload(self, tmp_path):
        """The pre-fix estimate for quant none was the source size; that does not move."""
        path = _write_checkpoint(tmp_path / "m", _dense_tensors())
        got = _estimate([path], arch="llama", quant="none")
        _header, source_payload = _header_and_payload(path)
        assert sum(got.layer_bytes) + got.other_bytes == source_payload

    def test_a_dense_nf4_base_keeps_embeddings_and_norms_at_the_stream_dtype(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _dense_tensors())
        got = _estimate([path], arch="llama")
        layers, other = _expected("fused", nf4=True, n_layers=1)
        assert (got.layer_bytes, got.other_bytes) == (layers, other)
        # the old rate charged the embeddings at NF4 too, so it sat below this
        assert got.total_bytes > _coarse(os.path.getsize(path))

    def test_headers_and_the_index_are_counted_on_top_of_the_payload(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _estimate([path], layout="per_expert")
        assert got.header_bytes > 0 and got.index_bytes > 0
        payload = sum(got.layer_bytes) + got.other_bytes
        assert got.total_bytes == payload + got.header_bytes + got.index_bytes

    def test_one_tensor_spelled_twice_is_refused_like_the_sharder(self, tmp_path):
        tensors = _moe_tensors()
        first = _write_checkpoint(tmp_path / "m", tensors, "a.safetensors")
        twin = {"model.norm.weight": tensors["model.norm.weight"]}
        second = _write_checkpoint(tmp_path / "m", twin, "b.safetensors")
        with pytest.raises(ValueError, match="twice"):
            _estimate([first, second])

    def test_a_missing_decoder_layer_is_refused_like_the_sharder(self, tmp_path):
        tensors = {
            key: value
            for key, value in _moe_tensors().items()
            if not key.startswith("model.layers.1.")
        }
        path = _write_checkpoint(tmp_path / "m", tensors)
        with pytest.raises(ValueError, match="not contiguous"):
            _estimate([path])

    def test_a_checkpoint_without_decoder_layers_is_refused(self, tmp_path):
        tensors = {"model.embed_tokens.weight": ("BF16", (VOCAB, HIDDEN))}
        path = _write_checkpoint(tmp_path / "m", tensors)
        with pytest.raises(ValueError, match="no decoder layer weights"):
            _estimate([path], quant="none")

    def test_qwen4_exp_is_left_to_its_own_estimators(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        assert _estimate([path], arch="qwen4_exp", quant="none") is None

    def test_nf4_without_quant_suffixes_is_refused_like_the_sharder(self, tmp_path):
        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        with pytest.raises(ValueError, match="quant_suffixes"):
            _estimate([path], suffixes=())

    @pytest.mark.parametrize("spelling", sorted(_MALFORMED), ids=sorted(_MALFORMED))
    def test_a_header_that_cannot_be_parsed_is_a_source_header_error(self, tmp_path, spelling):
        from soup_cli.utils.layer_shard import SourceHeaderError

        bad = _write_raw(tmp_path / "bad.safetensors", _MALFORMED[spelling])
        with pytest.raises(SourceHeaderError, match="bad.safetensors"):
            _estimate([bad])

    def test_a_missing_source_file_is_a_source_header_error(self, tmp_path):
        from soup_cli.utils.layer_shard import SourceHeaderError

        with pytest.raises(SourceHeaderError):
            _estimate([str(tmp_path / "gone.safetensors")])

    def test_arguments_are_checked_before_any_header_is_read(self, tmp_path):
        """A bad request is a bug to raise, never a header problem to fall back on."""
        from soup_cli.utils.layer_shard import SourceHeaderError

        missing = str(tmp_path / "gone.safetensors")
        for kwargs, needle in (
            ({"dtype": "int8"}, "unsupported dtype"),
            ({"layout": "experts"}, "unsupported expert_layout"),
            ({"suffixes": ()}, "quant_suffixes"),
        ):
            with pytest.raises(ValueError, match=needle) as exc:
                _estimate([missing], **kwargs)
            assert not isinstance(exc.value, SourceHeaderError)

    def test_a_layer_index_past_the_sharder_bound_is_refused_cheaply(self, tmp_path):
        """The sharder refuses more than 512 layers; sizing anything by the index first
        would allocate per index (~780 MB at 8,000,000, ~190 MB here) before saying so."""
        import tracemalloc

        tensors = _moe_tensors()
        tensors["model.layers.2000000.input_layernorm.weight"] = ("BF16", (HIDDEN,))
        path = _write_checkpoint(tmp_path / "m", tensors)
        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match=r"too many decoder layers \(2000001 > 512\)"):
                _estimate([path])
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 8 * 2**20, peak

    @pytest.mark.parametrize("n_layers", [512, 513])
    def test_the_layer_cap_counts_layers_like_the_sharder(self, tmp_path, n_layers):
        """The sharder takes max index + 1 and refuses above 512: 512 layers (indices
        0..511) shard, 513 do not. The estimate draws the line in the same place."""
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        from soup_cli.utils.layer_shard import shard_checkpoint

        weights = tmp_path / "m"
        path = _write_checkpoint(weights, _layer_stack(n_layers))
        if n_layers == 512:
            assert len(_estimate([path], arch="llama", quant="none").layer_bytes) == 512
            index = shard_checkpoint(str(weights), str(tmp_path / "out"), arch="llama")
            assert index.n_layers == 512
            return
        message = r"too many decoder layers \(513 > 512\)"
        with pytest.raises(ValueError, match=message):
            _estimate([path], arch="llama", quant="none")
        with pytest.raises(ValueError, match=message):
            shard_checkpoint(str(weights), str(tmp_path / "out"), arch="llama")

    @pytest.mark.parametrize("n_tensors", [10, 11])
    def test_the_tensor_cap_matches_the_sharder(self, tmp_path, monkeypatch, n_tensors):
        """At most _MAX_TOTAL_TENSORS tensors, the sharder's own constant and words.
        The cap is lowered to 10 here so the test stays a few KB."""
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        import soup_cli.utils.layer_shard as layer_shard

        monkeypatch.setattr(layer_shard, "_MAX_TOTAL_TENSORS", 10)
        weights = tmp_path / "m"
        path = _write_checkpoint(weights, _layer_stack(n_tensors - 1))
        if n_tensors == 10:
            assert len(_estimate([path], arch="llama", quant="none").layer_bytes) == 9
            return
        message = "checkpoint declares more than 10 tensors"
        with pytest.raises(ValueError, match=message):
            _estimate([path], arch="llama", quant="none")
        with pytest.raises(ValueError, match=message):
            layer_shard.shard_checkpoint(str(weights), str(tmp_path / "out"), arch="llama")

    def test_the_tensor_cap_counts_across_files(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_shard as layer_shard

        monkeypatch.setattr(layer_shard, "_MAX_TOTAL_TENSORS", 10)
        items = list(_layer_stack(10).items())  # 11 tensors, six and five per file
        first = _write_checkpoint(tmp_path / "m", dict(items[:6]), "a.safetensors")
        second = _write_checkpoint(tmp_path / "m", dict(items[6:]), "b.safetensors")
        with pytest.raises(ValueError, match="checkpoint declares more than 10 tensors"):
            _estimate([first, second], arch="llama", quant="none")

    def test_an_incomplete_expert_is_refused_like_the_sharder(self, tmp_path):
        tensors = _moe_tensors()
        del tensors["model.layers.1.mlp.experts.2.up_proj.weight"]
        path = _write_checkpoint(tmp_path / "m", tensors)
        with pytest.raises(ValueError, match="expert"):
            _estimate([path], layout="per_expert")


# --------------------------------------------------------------------------
# striping: the per-root shares come from the same estimate
# --------------------------------------------------------------------------
class TestStripeSharesUseTheEstimate:
    def test_shares_follow_the_estimated_layer_bytes(self, tmp_path):
        from soup_cli.trainer.stream_setup import estimate_stripe_write_shares

        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        est = _estimate([path], layout="")
        shares, approximate = estimate_stripe_write_shares(
            str(tmp_path / "m"), total_bytes=est.total_bytes, n_roots=2,
            dtype="bfloat16", quant="nf4", arch="deepseek_v3", estimate=est,
        )
        assert approximate is False
        assert sum(shares) == est.total_bytes
        # layers 0 and 2 plus everything outside the layers on root 0, layer 1 on root 1
        payload = sum(est.layer_bytes) + est.other_bytes
        root1 = est.layer_bytes[1]
        assert shares[1] == pytest.approx(est.total_bytes * root1 / payload, abs=1)

    def test_without_an_estimate_the_header_heuristic_is_unchanged(self, tmp_path):
        from soup_cli.trainer.stream_setup import estimate_stripe_write_shares

        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        est = _estimate([path], layout="")
        heuristic, _ = estimate_stripe_write_shares(
            str(tmp_path / "m"), total_bytes=est.total_bytes, n_roots=2,
            dtype="bfloat16", quant="nf4", arch="deepseek_v3",
        )
        exact, _ = estimate_stripe_write_shares(
            str(tmp_path / "m"), total_bytes=est.total_bytes, n_roots=2,
            dtype="bfloat16", quant="nf4", arch="deepseek_v3", estimate=est,
        )
        assert heuristic != exact  # else the test above proves nothing


# --------------------------------------------------------------------------
# which files the pre-flight reads
# --------------------------------------------------------------------------
_REVISION = "a" * 40


def _snapshot(tmp_path):
    repo = tmp_path / "hub" / "models--org--moe"
    snapshot = repo / "snapshots" / _REVISION
    (repo / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    return repo, snapshot


class TestPlannedSourceShards:
    def test_a_ready_weight_dir_is_listed_like_the_sharder_lists_it(self, tmp_path):
        from soup_cli.trainer.stream_setup import _planned_source_shards

        path = _write_checkpoint(tmp_path / "m", _moe_tensors())
        plan = types.SimpleNamespace(weights_dir=str(tmp_path / "m"), needs_materialization=False)
        assert _planned_source_shards(plan) == [os.path.realpath(path)]

    def test_a_copy_not_made_yet_is_read_from_its_snapshot(self, tmp_path):
        from soup_cli.trainer.stream_setup import _planned_source_shards

        _repo, snapshot = _snapshot(tmp_path)
        path = _write_checkpoint(snapshot, _moe_tensors())
        plan = types.SimpleNamespace(
            weights_dir=str(tmp_path / "copy-not-made-yet"),
            source_dir=str(snapshot),
            source_revision=_REVISION,
            needs_materialization=True,
        )
        assert [os.path.realpath(p) for p in _planned_source_shards(plan)] == [
            os.path.realpath(path)
        ]

    def test_a_snapshot_without_its_revision_is_not_read(self, tmp_path):
        """The copy step refuses it; the pre-flight falls back instead of guessing."""
        from soup_cli.trainer.stream_setup import _planned_source_shards

        _repo, snapshot = _snapshot(tmp_path)
        _write_checkpoint(snapshot, _moe_tensors())
        plan = types.SimpleNamespace(
            weights_dir=str(tmp_path / "copy"),
            source_dir=str(snapshot),
            source_revision=None,
            needs_materialization=True,
        )
        with pytest.raises(ValueError, match="revision"):
            _planned_source_shards(plan)

    def test_only_top_level_weight_files_are_read(self, tmp_path):
        """The sharder lists the copy's top level; a nested file is copied, never sharded."""
        from soup_cli.trainer.stream_setup import _planned_source_shards

        _repo, snapshot = _snapshot(tmp_path)
        path = _write_checkpoint(snapshot, _moe_tensors())
        _write_checkpoint(snapshot / "onnx", _moe_tensors())
        plan = types.SimpleNamespace(
            weights_dir=str(tmp_path / "copy"),
            source_dir=str(snapshot),
            source_revision=_REVISION,
            needs_materialization=True,
        )
        assert [os.path.realpath(p) for p in _planned_source_shards(plan)] == [
            os.path.realpath(path)
        ]

    @pytest.mark.requires_symlink
    def test_a_symlinked_snapshot_is_read_through_its_blob(self, tmp_path):
        from soup_cli.trainer.stream_setup import _planned_source_shards

        repo, snapshot = _snapshot(tmp_path)
        blob = _write_checkpoint(repo / "blobs", _moe_tensors(), name="etag123")
        (snapshot / "model.safetensors").symlink_to(os.path.join("..", "..", "blobs", "etag123"))
        plan = types.SimpleNamespace(
            weights_dir=str(tmp_path / "copy"),
            source_dir=str(snapshot),
            source_revision=_REVISION,
            needs_materialization=True,
        )
        assert [os.path.realpath(p) for p in _planned_source_shards(plan)] == [
            os.path.realpath(blob)
        ]

    @pytest.mark.requires_symlink
    def test_a_link_out_of_the_blob_store_is_refused(self, tmp_path):
        from soup_cli.trainer.stream_setup import _planned_source_shards

        _repo, snapshot = _snapshot(tmp_path)
        outside = _write_checkpoint(tmp_path / "elsewhere", _moe_tensors())
        (snapshot / "model.safetensors").symlink_to(outside)
        plan = types.SimpleNamespace(
            weights_dir=str(tmp_path / "copy"),
            source_dir=str(snapshot),
            source_revision=_REVISION,
            needs_materialization=True,
        )
        with pytest.raises(ValueError, match="outside the Hugging Face blob store"):
            _planned_source_shards(plan)


# --------------------------------------------------------------------------
# what falls back to the approximate figure, and what does not
# --------------------------------------------------------------------------
def _ready_plan(directory):
    return types.SimpleNamespace(weights_dir=str(directory), needs_materialization=False)


def _preflight(plan, **kwargs):
    from soup_cli.trainer.stream_setup import _preflight_cache_estimate

    kwargs.setdefault("dtype", "bfloat16")
    kwargs.setdefault("arch", "deepseek_v3")
    kwargs.setdefault("quant", "nf4")
    kwargs.setdefault("quant_suffixes", SUFFIXES)
    kwargs.setdefault("double_quant", True)
    kwargs.setdefault("expert_layout", "")
    kwargs.setdefault("num_hidden_layers", N_LAYERS)
    return _preflight_cache_estimate(plan, **kwargs)


class TestTheFallbackIsOnlyForUnreadableSources:
    @pytest.mark.parametrize("spelling", sorted(_MALFORMED), ids=sorted(_MALFORMED))
    def test_a_malformed_header_falls_back(self, tmp_path, spelling):
        _write_raw(tmp_path / "m" / "model.safetensors", _MALFORMED[spelling])
        assert _preflight(_ready_plan(tmp_path / "m")) is None

    def test_an_fp8_tensor_without_its_scale_is_refused_not_approximated(self, tmp_path):
        from soup_cli.utils.fp8_block import UnsupportedFP8Error

        _write_raw(tmp_path / "m" / "model.safetensors", _FP8_WITHOUT_SCALE)
        with pytest.raises(UnsupportedFP8Error, match="declares no FP8 quantization_config"):
            _preflight(_ready_plan(tmp_path / "m"))

    def test_a_source_that_cannot_be_listed_falls_back(self, tmp_path):
        plan = types.SimpleNamespace(
            weights_dir=str(tmp_path / "copy"), source_dir=str(tmp_path),
            source_revision=None, needs_materialization=True,
        )
        assert _preflight(plan) is None

    def test_a_readable_source_is_estimated(self, tmp_path):
        _write_checkpoint(tmp_path / "m", _moe_tensors())
        got = _preflight(_ready_plan(tmp_path / "m"))
        assert (got.layer_bytes, got.other_bytes) == _expected("fused", nf4=True)

    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"dtype": "int8"}, "unsupported dtype"),
            ({"expert_layout": "experts"}, "unsupported expert_layout"),
            ({"quant_suffixes": ()}, "quant_suffixes"),
        ],
        ids=["dtype", "layout", "suffixes"],
    )
    def test_a_bad_request_is_raised_not_approximated(self, tmp_path, kwargs, needle):
        _write_checkpoint(tmp_path / "m", _moe_tensors())
        with pytest.raises(ValueError, match=needle):
            _preflight(_ready_plan(tmp_path / "m"), **kwargs)

    def test_a_checkpoint_the_sharder_refuses_is_refused_here(self, tmp_path):
        """Same words, before any copy or shard is written."""
        tensors = _moe_tensors()
        tensors["model.layers.600.input_layernorm.weight"] = ("BF16", (HIDDEN,))
        _write_checkpoint(tmp_path / "m", tensors)
        with pytest.raises(ValueError, match=r"too many decoder layers \(601 > 512\)"):
            _preflight(_ready_plan(tmp_path / "m"), num_hidden_layers=None)

    def test_too_many_tensors_is_refused_not_approximated(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_shard as layer_shard

        monkeypatch.setattr(layer_shard, "_MAX_TOTAL_TENSORS", 10)
        _write_checkpoint(tmp_path / "m", _layer_stack(10))
        with pytest.raises(ValueError, match="checkpoint declares more than 10 tensors"):
            _preflight(
                _ready_plan(tmp_path / "m"), arch="llama", quant="none", quant_suffixes=(),
                num_hidden_layers=None,
            )

    def test_qwen4_exp_never_lists_its_source(self, tmp_path, monkeypatch):
        """Its own estimators size it; a snapshot walk would be wasted work."""
        import soup_cli.trainer.stream_setup as stream_setup

        def _must_not_list(_plan):
            raise AssertionError("qwen4_exp source listed")

        monkeypatch.setattr(stream_setup, "_planned_source_shards", _must_not_list)
        plan = _ready_plan(tmp_path)
        assert _preflight(plan, arch="qwen4_exp", quant="none", quant_suffixes=()) is None


# --------------------------------------------------------------------------
# the setup charges it (the shared driver in tests/test_issue602_qwen4_ple_streaming.py)
# --------------------------------------------------------------------------
def _deepseek_cfg():
    return types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=HIDDEN, num_hidden_layers=N_LAYERS,
        vocab_size=VOCAB, intermediate_size=DENSE_INTER,
    )


def _drive_preflight(tmp_path, monkeypatch, *, layout, tensors=None, free=10**15,
                     plan_overrides=None, quantization="4bit", double_quant=True,
                     suffixes=SUFFIXES, header_text=None):
    """Run the real setup up to the sharder; return the pre-flight's keyword arguments.

    The resolver hands the real `_disk_preflight` a plan over a synthetic checkpoint
    (or, with `header_text`, a file with that header), the meta probe reports
    `suffixes`, and the volume reports `free` bytes.
    """
    import soup_cli.trainer.stream_setup as stream_setup
    import soup_cli.utils.layer_stream_runtime as layer_stream_runtime
    import soup_cli.utils.moe as moe
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    weights = tmp_path / "weights"
    if header_text is None:
        path = _write_checkpoint(weights, tensors or _moe_tensors())
    else:
        path = _write_raw(weights / "model.safetensors", header_text)
    (weights / "config.json").write_text('{"model_type": "deepseek_v3"}', encoding="utf-8")
    size = os.path.getsize(path)
    plan = types.SimpleNamespace(
        weights_dir=str(weights), source_dir=str(weights), source_bytes=size,
        materialized_copy_bytes=0, materialize_bytes=0, needs_materialization=False,
        source_files=(("model.safetensors", size, 0),), source_revision=None,
    )
    for name, value in (plan_overrides or {}).items():
        setattr(plan, name, value)

    def _resolve(_base, *, before_materialize=None):
        if before_materialize is not None:
            before_materialize(plan)
        return str(weights)

    probes = []
    monkeypatch.setattr(
        stream_setup.StreamingSetupMixin, "_stream_meta_skeleton",
        classmethod(lambda cls, *a, **k: probes.append(k) or object()),
    )
    monkeypatch.setattr(layer_stream_runtime, "quantised_layer_suffixes", lambda _probe: suffixes)
    monkeypatch.setattr(moe, "resolve_moe_lora_targets", lambda *a, **k: None)
    monkeypatch.setattr(stream_setup, "_disk_volume", lambda _path: (1, free))
    seen = []
    real = stream_setup._render_stream_disk_preflight

    def _spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(stream_setup, "_render_stream_disk_preflight", _spy)
    real_shares = stream_setup.estimate_stripe_write_shares

    def _shares_spy(*args, **kwargs):
        seen.append({"shares_call": kwargs})
        return real_shares(*args, **kwargs)

    monkeypatch.setattr(stream_setup, "estimate_stripe_write_shares", _shares_spy)
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, resolve_weights=_resolve, arch="deepseek_v3",
        model_cfg=_deepseek_cfg(),
        tcfg_overrides={
            "stream_expert_layout": layout,
            "quantization": quantization,
            "double_quant_on": double_quant,
        },
    )
    shares_calls =[entry["shares_call"] for entry in seen if "shares_call" in entry]
    panels = [entry for entry in seen if "shares_call" not in entry]
    return panels, path, shares_calls


def _stream_dtype():
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    return resolve_stream_dtype("cpu")


class TestTheSetupChargesTheEstimate:
    @pytest.mark.parametrize("layout", ["fused", "per_expert"])
    def test_the_preflight_charges_the_header_estimate(self, tmp_path, monkeypatch, layout):
        panels, path, shares_calls = _drive_preflight(tmp_path, monkeypatch, layout=layout)
        expected = _estimate(
            [path], layout="" if layout == "fused" else layout, dtype=_stream_dtype(),
            num_hidden_layers=N_LAYERS,
        )
        assert len(panels) == 1
        assert panels[0]["shard_bytes"] == expected.total_bytes
        assert panels[0]["shard_write_bytes"] == expected.total_bytes
        assert panels[0]["approximate"] is False
        # a striped cache splits that same estimate over its roots
        assert [call["estimate"] for call in shares_calls] == [expected]

    def test_a_disk_that_fits_the_old_rate_but_not_the_fused_cache_is_refused(
        self, tmp_path, monkeypatch
    ):
        path = _write_checkpoint(tmp_path / "probe", _moe_tensors())
        old = _coarse(os.path.getsize(path))
        new = _estimate([path], layout="", dtype=_stream_dtype()).total_bytes
        assert old < new
        with pytest.raises(ValueError) as exc:
            _drive_preflight(tmp_path, monkeypatch, layout="fused", free=(old + new) // 2)
        text = str(exc.value)
        assert "layer-shard cache" in text
        assert "Refusing before copying or sharding" in text

    def test_a_disk_that_fits_the_fused_cache_passes(self, tmp_path, monkeypatch):
        path = _write_checkpoint(tmp_path / "probe", _moe_tensors())
        new = _estimate([path], layout="", dtype=_stream_dtype()).total_bytes
        panels, _path, _calls = _drive_preflight(
            tmp_path, monkeypatch, layout="fused", free=new
        )
        assert panels[0]["shard_bytes"] == new

    @pytest.mark.parametrize("spelling", sorted(_MALFORMED), ids=sorted(_MALFORMED))
    def test_unreadable_headers_fall_back_to_the_old_rate_and_say_approximately(
        self, tmp_path, monkeypatch, spelling
    ):
        from soup_cli.utils.layer_stream import estimate_stream_store_bytes

        panels, path, _calls = _drive_preflight(
            tmp_path, monkeypatch, layout="fused", header_text=_MALFORMED[spelling]
        )
        assert panels[0]["shard_bytes"] == estimate_stream_store_bytes(
            os.path.getsize(path), dtype=_stream_dtype(), quant="nf4"
        )
        assert panels[0]["approximate"] is True

    @pytest.mark.parametrize("layout", ["fused", "per_expert"])
    def test_single_quant_runs_are_charged_float32_absmax(self, tmp_path, monkeypatch, layout):
        """double_quant: false stores a float32 absmax per block, ~9% more than NF4 with it."""
        panels, path, _calls = _drive_preflight(
            tmp_path, monkeypatch, layout=layout, double_quant=False
        )
        expected = _estimate(
            [path], layout="" if layout == "fused" else layout, dtype=_stream_dtype(),
            double_quant=False, num_hidden_layers=N_LAYERS,
        )
        assert panels[0]["shard_bytes"] == expected.total_bytes
        double = _estimate([path], layout="", dtype=_stream_dtype())
        assert expected.total_bytes != double.total_bytes  # else this pins nothing

    def test_layers_past_the_configs_count_are_not_charged(self, tmp_path, monkeypatch):
        """DeepSeek-V3 / Kimi K2 MTP layers: the sharder drops them against the config."""
        panels, path, _calls = _drive_preflight(
            tmp_path, monkeypatch, layout="fused", tensors=_moe_tensors(extra_layers=1)
        )
        dropped = _estimate([path], dtype=_stream_dtype(), num_hidden_layers=N_LAYERS)
        kept = _estimate([path], dtype=_stream_dtype())
        assert len(dropped.layer_bytes) == N_LAYERS
        assert panels[0]["shard_bytes"] == dropped.total_bytes != kept.total_bytes

    def test_a_checkpoint_the_sharder_refuses_is_refused_before_sharding(
        self, tmp_path, monkeypatch
    ):
        tensors = _moe_tensors()
        del tensors["model.layers.1.mlp.experts.2.up_proj.weight"]
        with pytest.raises(ValueError, match="incomplete expert projections"):
            _drive_preflight(tmp_path, monkeypatch, layout="per_expert", tensors=tensors)

    def test_a_probe_that_found_nothing_to_quantise_is_refused(self, tmp_path, monkeypatch):
        """quant='nf4' with no suffixes is a setup bug, never an approximate figure."""
        with pytest.raises(ValueError, match="quant_suffixes"):
            _drive_preflight(tmp_path, monkeypatch, layout="fused", suffixes=frozenset())

    def test_an_unquantised_run_is_charged_its_header_estimate(self, tmp_path, monkeypatch):
        panels, path, _calls = _drive_preflight(
            tmp_path, monkeypatch, layout="fused", quantization="none"
        )
        expected = _estimate(
            [path], quant="none", dtype=_stream_dtype(), num_hidden_layers=N_LAYERS
        )
        assert panels[0]["shard_bytes"] == expected.total_bytes


# --------------------------------------------------------------------------
# qwen4_exp keeps its own figures; only the source-size one is labelled a guess
# --------------------------------------------------------------------------
def _qwen4_panel_text(tmp_path, monkeypatch, *, oq):
    """The pre-flight panel text of the real setup over a tiny qwen4_exp checkpoint.

    With ``oq`` the checkpoint carries affine companions (``.scales``/``.biases``) and a
    quantization_config, so ``estimate_oq_stream_cache_bytes`` sizes it from its
    headers; without, the panel shows the source-size figure.
    """
    import io
    import json

    import torch
    from rich.console import Console
    from safetensors.torch import save_file

    import soup_cli.trainer.stream_setup as stream_setup
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    weights = tmp_path / "weights"
    weights.mkdir()
    tensors = {"language_model.model.norm.weight": torch.ones(2)}
    if oq:
        tensors["language_model.model.layers.0.proj.weight"] = torch.tensor(
            [[0, 1, 2, 3]], dtype=torch.uint32
        )
        tensors["language_model.model.layers.0.proj.scales"] = torch.ones(
            (1, 1), dtype=torch.bfloat16
        )
        tensors["language_model.model.layers.0.proj.biases"] = torch.zeros(
            (1, 1), dtype=torch.bfloat16
        )
        config = {"quantization_config": {"bits": 4, "group_size": 32}}
    else:
        tensors["language_model.model.layers.0.proj.weight"] = torch.ones(
            (2, 2), dtype=torch.bfloat16
        )
        config = {}
    blob = weights / "model.safetensors"
    save_file(tensors, str(blob))
    (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
    stat = blob.stat()
    plan = types.SimpleNamespace(
        weights_dir=str(weights), source_bytes=stat.st_size, materialized_copy_bytes=0,
        materialize_bytes=0, needs_materialization=False,
        source_files=((str(blob), stat.st_size, int(stat.st_mtime)),),
    )

    def _resolve(_base, *, before_materialize=None):
        if before_materialize is not None:
            before_materialize(plan)
        return str(weights)

    buffer = io.StringIO()
    monkeypatch.setattr(
        stream_setup, "console", Console(file=buffer, width=400, color_system=None)
    )
    monkeypatch.setattr(stream_setup, "_disk_volume", lambda _path: (1, 10**15))
    _drive_qwen4_streaming_setup(tmp_path, monkeypatch, resolve_weights=_resolve)
    return " ".join(buffer.getvalue().split())


class TestQwen4ExpPanelLabel:
    def test_an_oq_checkpoint_is_sized_from_its_headers_not_labelled(
        self, tmp_path, monkeypatch
    ):
        text = _qwen4_panel_text(tmp_path, monkeypatch, oq=True)
        assert "Layer-shard cache:" in text
        assert "Layer-shard cache: approximately" not in text

    def test_a_non_oq_checkpoint_shows_its_source_size_as_approximate(
        self, tmp_path, monkeypatch
    ):
        text = _qwen4_panel_text(tmp_path, monkeypatch, oq=False)
        assert "Layer-shard cache: approximately" in text


# --------------------------------------------------------------------------
# the estimate against what the real sharder writes (CPU NF4)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_moonlight(tmp_path_factory):
    """A small `deepseek_v3` base laid out like Moonlight (untied head, q_lora_rank None),
    saved at bf16 with one 2-D tensor per expert, plus the probe's quant suffixes."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    torch.manual_seed(3)
    config = transformers.DeepseekV3Config(
        vocab_size=256, hidden_size=256, intermediate_size=512, moe_intermediate_size=256,
        num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, n_group=2,
        topk_group=1, num_experts_per_tok=2, kv_lora_rank=32, q_lora_rank=None,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    model = transformers.DeepseekV3ForCausalLM(config).to(torch.bfloat16)
    weights = tmp_path_factory.mktemp("tiny-moonlight") / "model"
    model.save_pretrained(str(weights))
    del model
    probe = build_meta_skeleton(str(weights), dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    return str(weights), suffixes


@pytest.fixture(scope="module")
def tiny_dense(tmp_path_factory):
    """A one-layer tied Llama: the smallest cache, where the index's fixed part shows."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    torch.manual_seed(4)
    config = transformers.LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    weights = tmp_path_factory.mktemp("tiny-dense") / "model"
    model.save_pretrained(str(weights))
    del model
    probe = build_meta_skeleton(str(weights), dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    return str(weights), suffixes


def _against_the_sharder(tmp_path, weights, suffixes, *, arch, layout, quant, double_quant,
                         num_hidden_layers):
    """Shard for real (CPU NF4) and hold the estimate against the files written.

    Returns ``(estimate, bytes of the cache files)`` after asserting that the payload
    is exact per decoder layer and outside them, and that the header and index
    allowances each cover what was written.
    """
    from soup_cli.utils.expert_stream import expert_shard_name
    from soup_cli.utils.layer_shard import (
        _INDEX_NAME,
        _discover_safetensors,
        estimate_shard_cache_bytes,
        layer_shard_path,
        shard_cache_disk_bytes,
        shard_checkpoint,
    )

    common = dict(
        dtype="bfloat16", arch=arch, quant=quant,
        quant_suffixes=suffixes if quant == "nf4" else (), double_quant=double_quant,
        expert_layout=layout, num_hidden_layers=num_hidden_layers,
    )
    estimate = estimate_shard_cache_bytes(_discover_safetensors(weights), **common)
    out = tmp_path / "cache"
    index = shard_checkpoint(weights, str(out), quant_device="cpu", **common)
    actual = shard_cache_disk_bytes(str(out), index)

    headers = 0
    for idx in range(index.n_layers):
        header, payload = _header_and_payload(layer_shard_path(str(out), idx))
        headers += header
        experts = out / expert_shard_name(idx)
        if experts.exists():
            header_x, payload_x = _header_and_payload(str(experts))
            headers += header_x
            payload += payload_x
        assert estimate.layer_bytes[idx] == payload, idx
    other = 0
    for name in os.listdir(out):
        if name == "extras.safetensors" or name.startswith("large_"):
            header, payload = _header_and_payload(str(out / name))
            headers += header
            other += payload
    assert estimate.other_bytes == other
    assert estimate.header_bytes >= headers
    assert estimate.index_bytes >= os.path.getsize(out / _INDEX_NAME)
    assert estimate.total_bytes >= actual
    return estimate, actual


@pytest.mark.parametrize(
    ("layout", "quant", "double_quant"),
    [
        ("", "nf4", True),
        ("per_expert", "nf4", True),
        ("", "nf4", False),
        ("per_expert", "nf4", False),
        ("", "none", True),
        ("per_expert", "none", True),
    ],
    ids=["fused-nf4", "per_expert-nf4", "fused-nf4-single", "per_expert-nf4-single",
         "fused-bf16", "per_expert-bf16"],
)
def test_the_moe_estimate_matches_the_cache_the_sharder_writes(
    tiny_moonlight, tmp_path, layout, quant, double_quant
):
    weights, suffixes = tiny_moonlight
    estimate, actual = _against_the_sharder(
        tmp_path, weights, suffixes, arch="deepseek_v3", layout=layout, quant=quant,
        double_quant=double_quant, num_hidden_layers=3,
    )
    assert estimate.total_bytes <= actual * 1.05


def test_a_minimal_checkpoint_index_is_covered(tmp_path):
    """Two tensors: the index is mostly its scalar fields, which the fixed part covers."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    weights = tmp_path / "model"
    _write_checkpoint(
        weights,
        {
            "model.embed_tokens.weight": ("BF16", (16, 8)),
            "model.layers.0.input_layernorm.weight": ("BF16", (8,)),
        },
    )
    _against_the_sharder(
        tmp_path, str(weights), (), arch="llama", layout="", quant="none",
        double_quant=True, num_hidden_layers=None,
    )


@pytest.mark.parametrize("quant", ["nf4", "none"])
def test_the_dense_estimate_matches_the_cache_the_sharder_writes(tiny_dense, tmp_path, quant):
    weights, suffixes = tiny_dense
    _against_the_sharder(
        tmp_path, weights, suffixes, arch="llama", layout="", quant=quant,
        double_quant=True, num_hidden_layers=None,
    )
