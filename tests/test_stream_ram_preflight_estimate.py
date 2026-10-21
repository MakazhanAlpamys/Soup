"""Layer streaming's early RAM check charges what the RAM tier will hold, from the headers.

`soup train` with `training.stream_layers: true` and `training.stream_source: ram` refuses a
base the RAM tier cannot hold. One check runs BEFORE sharding, so that such a base is refused
before minutes of disk writes; the exact one runs after, on the shard headers. The early one
charged the flat NF4 rate to the source files' size on disk. A fused MoE base under
`quantization: 4bit` was under-counted (Moonlight-16B-A3B: ~8.2 GB against a 30.6 GB cache,
whose stacked routed experts stay at the stream dtype), and an FP8 block source about 2x (one
byte per weight on disk, the dequantised weight in the cache). Such a run was refused only by
the exact check, after the sharder had written ~30 GB.

The early check now charges the disk pre-flight's header estimate
(`layer_shard.estimate_shard_cache_bytes`, read once per setup) for exactly what the exact
check counts: the decoder-layer files, the embedding/head shards and the resident extras, not
the per_expert experts files and not the NF4 code tables. The flat rate is the fallback only
when the headers cannot be listed or read, and its refusal says approximately.

Every setup-level case drives the real `_setup_streaming_transformers` through the shared
harness in `tests/test_issue602_qwen4_ple_streaming.py`, the way
`tests/test_stream_disk_preflight_moe.py` drives the disk pre-flight. The harness stubs the
sharder, so an empty list of sharder calls means no shard was written.
"""

import json
import math
import os
import types

import pytest

from tests.test_stream_disk_preflight_moe import (
    _MALFORMED,
    DENSE_INTER,
    HIDDEN,
    N_LAYERS,
    SUFFIXES,
    VOCAB,
    _deepseek_cfg,
    _dense_tensors,
    _moe_tensors,
    _write_checkpoint,
    _write_raw,
)

#: The meta probe's suffixes for the dense (Llama-shaped) base: every decoder nn.Linear.
DENSE_SUFFIXES = frozenset(
    {
        "self_attn.q_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
)
_BASE = "qwen4-test"  # the harness's cfg.base


def _stream_dtype():
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    return resolve_stream_dtype("cpu")


def _llama_cfg():
    return types.SimpleNamespace(
        model_type="llama", hidden_size=HIDDEN, num_hidden_layers=1, vocab_size=VOCAB,
        intermediate_size=DENSE_INTER,
    )


def _with_config(directory, model_type, num_hidden_layers):
    (directory / "config.json").write_text(
        json.dumps({"model_type": model_type, "num_hidden_layers": num_hidden_layers}),
        encoding="utf-8",
    )
    return directory


def _moe_dir(tmp_path, tensors=None):
    weights = tmp_path / "weights"
    _write_checkpoint(weights, tensors or _moe_tensors())
    return _with_config(weights, "deepseek_v3", N_LAYERS)


def _dense_dir(tmp_path):
    weights = tmp_path / "weights"
    _write_checkpoint(weights, _dense_tensors())
    return _with_config(weights, "llama", 1)


def _source_bytes(weights):
    return sum(
        os.path.getsize(os.path.join(weights, name))
        for name in os.listdir(weights)
        if name.endswith(".safetensors")
    )


def _estimate(weights, *, arch="deepseek_v3", quant="nf4", suffixes=SUFFIXES, layout="",
              num_hidden_layers=N_LAYERS, double_quant=True, quantization_config=None):
    """The header estimate the setup computes for this run, called directly."""
    from soup_cli.utils.layer_shard import _discover_safetensors, estimate_shard_cache_bytes

    return estimate_shard_cache_bytes(
        _discover_safetensors(str(weights)), dtype=_stream_dtype(), arch=arch, quant=quant,
        quant_suffixes=suffixes if quant == "nf4" else (), double_quant=double_quant,
        expert_layout=layout, num_hidden_layers=num_hidden_layers,
        quantization_config=quantization_config,
    )


def _flat(source_bytes, quant="nf4"):
    """The old early charge: the NF4 rate to every byte on disk."""
    from soup_cli.utils.layer_stream import estimate_stream_store_bytes

    return estimate_stream_store_bytes(source_bytes, dtype=_stream_dtype(), quant=quant)


def _free_for_budget(budget):
    """Free RAM whose 70% RAM-tier headroom is ``budget`` (to within a byte)."""
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM

    return int(budget / RAM_TIER_HEADROOM)


def _drive(tmp_path, monkeypatch, *, weights, arch="deepseek_v3", model_cfg=None,
           layout="fused", quantization="4bit", double_quant=True, suffixes=SUFFIXES,
           stream_source="ram", free_ram=10**15, total_ram=10**15, run_preflight=True,
           plan_overrides=None):
    """Run the real setup through the shared harness; return what it saw.

    ``early``: the keyword arguments of every early-check call; ``shards``: every sharder
    call (stubbed, so empty means nothing was written); ``reads``: how many times the source
    headers were read for an estimate; ``error``: the ValueError the setup raised, if any.
    ``run_preflight=False`` is a resolver that does not call the disk pre-flight back.
    """
    import soup_cli.trainer.stream_setup as stream_setup
    import soup_cli.utils.layer_shard as layer_shard
    import soup_cli.utils.layer_stream_runtime as layer_stream_runtime
    import soup_cli.utils.moe as moe
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    monkeypatch.delenv("SOUP_LAYER_STREAM_STRIPE_DIRS", raising=False)
    weights = str(weights)
    size = _source_bytes(weights)
    plan = types.SimpleNamespace(
        weights_dir=weights, source_dir=weights, source_bytes=size,
        materialized_copy_bytes=0, materialize_bytes=0, needs_materialization=False,
        source_files=(("model.safetensors", size, 0),), source_revision=None,
    )
    for name, value in (plan_overrides or {}).items():
        setattr(plan, name, value)

    def _resolve(_base, *, before_materialize=None):
        if run_preflight and before_materialize is not None:
            before_materialize(plan)
        return weights

    seen = {"early": [], "shards": [], "reads": 0}
    real_early = stream_setup._validate_early_ram_fit

    def _early_spy(**kwargs):
        seen["early"].append(kwargs)
        return real_early(**kwargs)

    real_read = layer_shard._read_source_tensors

    def _read_spy(paths):
        seen["reads"] += 1
        return real_read(paths)

    monkeypatch.setattr(stream_setup, "_validate_early_ram_fit", _early_spy)
    monkeypatch.setattr(layer_shard, "_read_source_tensors", _read_spy)
    monkeypatch.setattr(
        stream_setup.StreamingSetupMixin, "_stream_meta_skeleton",
        classmethod(lambda cls, *a, **k: object()),
    )
    monkeypatch.setattr(layer_stream_runtime, "quantised_layer_suffixes", lambda _probe: suffixes)
    monkeypatch.setattr(moe, "resolve_moe_lora_targets", lambda *a, **k: None)
    monkeypatch.setattr(stream_setup, "_disk_volume", lambda _path: (1, 10**15))
    try:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_resolve, arch=arch,
            model_cfg=model_cfg or _deepseek_cfg(), shard_calls=seen["shards"],
            free_ram=free_ram, total_ram=total_ram,
            tcfg_overrides={
                "stream_expert_layout": layout,
                "quantization": quantization,
                "double_quant_on": double_quant,
                "stream_source": stream_source,
            },
        )
    except ValueError as exc:
        seen["error"] = exc
    return seen


def _text(seen):
    assert "error" in seen, "the setup was not refused"
    return " ".join(str(seen["error"]).split())


def _passed(seen):
    assert "error" not in seen, seen.get("error")
    return seen


# --------------------------------------------------------------------------
# 1-2. a fused MoE base: refused early by the header estimate, passes when it fits
# --------------------------------------------------------------------------
class TestAFusedMoeBase:
    def test_over_free_ram_by_the_estimate_but_not_the_flat_rate_is_refused_before_sharding(
        self, tmp_path, monkeypatch
    ):
        weights = _moe_dir(tmp_path)
        need = _estimate(weights).ram_tier_bytes
        flat = _flat(_source_bytes(weights))
        assert flat < need  # the fused experts stay at the stream dtype
        free = _free_for_budget((flat + need) // 2)
        seen = _drive(tmp_path, monkeypatch, weights=weights, free_ram=free)
        text = _text(seen)
        assert "training.stream_source='ram'" in text
        assert "needs" in text and "of RAM once sharded" in text
        assert "sized from its safetensors headers" in text
        assert "Refusing before sharding" in text
        assert "stream_source='auto' to fall back to the NVMe disk tier" in text
        assert "approximately" not in text
        assert seen["shards"] == []  # nothing was sharded
        assert [call["ram_bytes"] for call in seen["early"]] == [need]
        assert seen["early"][0]["approximate"] is False
        # the on-disk figure is the source size (the harness reports 1,024 bytes)
        assert seen["early"][0]["source_bytes"] == 1_024

    def test_the_old_flat_rate_would_have_let_it_through(self, tmp_path):
        """The control: at that free RAM the flat figure is under the headroom."""
        from soup_cli.trainer.stream_setup import _validate_early_ram_fit

        weights = _moe_dir(tmp_path)
        need = _estimate(weights).ram_tier_bytes
        flat = _flat(_source_bytes(weights))
        free = _free_for_budget((flat + need) // 2)
        _validate_early_ram_fit(
            base=_BASE, source_bytes=_source_bytes(weights), ram_bytes=flat,
            approximate=True, quantised=True, free_ram=free, total_ram=None,
        )

    @pytest.mark.parametrize("layout", ["fused", "per_expert"])
    def test_a_base_that_fits_passes_and_is_sharded(self, tmp_path, monkeypatch, layout):
        weights = _moe_dir(tmp_path)
        need = _estimate(weights, layout="" if layout == "fused" else layout).ram_tier_bytes
        free = math.ceil(need / 0.7) + 2
        seen = _passed(
            _drive(tmp_path, monkeypatch, weights=weights, layout=layout, free_ram=free)
        )
        assert [call["ram_bytes"] for call in seen["early"]] == [need]
        assert len(seen["shards"]) == 1

    def test_the_per_expert_experts_files_are_not_charged(self, tmp_path, monkeypatch):
        """They are read from the cache volume on every tier; the RAM tier never holds them."""
        weights = _moe_dir(tmp_path)
        per_expert = _estimate(weights, layout="per_expert")
        assert per_expert.expert_file_bytes > 0
        seen = _passed(_drive(tmp_path, monkeypatch, weights=weights, layout="per_expert"))
        charged = seen["early"][0]["ram_bytes"]
        assert charged == sum(per_expert.layer_bytes) - per_expert.expert_file_bytes + (
            per_expert.other_bytes - per_expert.code_table_bytes
        )
        assert charged < sum(per_expert.layer_bytes) + per_expert.other_bytes

    def test_over_physical_ram_is_refused_before_sharding(self, tmp_path, monkeypatch):
        """The exact check's second rule, 55% of physical RAM, applies early too."""
        weights = _moe_dir(tmp_path)
        need = _estimate(weights).ram_tier_bytes
        seen = _drive(
            tmp_path, monkeypatch, weights=weights, free_ram=10**15, total_ram=need
        )
        text = _text(seen)
        assert "more than 55% of physical RAM" in text
        assert "Refusing before sharding" in text
        assert seen["shards"] == []


# --------------------------------------------------------------------------
# the estimate is read once: reused from the disk pre-flight
# --------------------------------------------------------------------------
class TestTheHeadersAreReadOnce:
    def test_the_early_check_reuses_the_disk_preflights_estimate(self, tmp_path, monkeypatch):
        weights = _moe_dir(tmp_path)
        seen = _passed(_drive(tmp_path, monkeypatch, weights=weights))
        assert seen["reads"] == 1
        assert seen["early"][0]["approximate"] is False

    def test_without_a_disk_preflight_the_resolved_directory_is_sized(
        self, tmp_path, monkeypatch
    ):
        """A resolver that never called the pre-flight back: the flat rate is not a fallback
        for readable headers, so the resolved directory is sized instead, once."""
        weights = _moe_dir(tmp_path)
        need = _estimate(weights).ram_tier_bytes
        seen = _passed(_drive(tmp_path, monkeypatch, weights=weights, run_preflight=False))
        assert seen["reads"] == 1
        assert [call["ram_bytes"] for call in seen["early"]] == [need]
        assert seen["early"][0]["approximate"] is False

    def test_auto_is_not_checked_early_and_reads_nothing_more(self, tmp_path, monkeypatch):
        """Only `ram` insists on the RAM tier. `auto` takes the disk tier when RAM is short,
        which is decided after sharding from the real store, so nothing before sharding
        refuses or warns on any RAM figure, even with no RAM free at all."""
        weights = _moe_dir(tmp_path)
        seen = _passed(
            _drive(
                tmp_path, monkeypatch, weights=weights, stream_source="auto",
                free_ram=1, run_preflight=False,
            )
        )
        assert seen["early"] == []
        assert seen["reads"] == 0
        assert len(seen["shards"]) == 1

    def test_a_qwen4_exp_base_is_left_to_its_exact_check(self, tmp_path, monkeypatch):
        """Its source size includes the external PLE table: charging it early would refuse
        the `stream_ngram_source: disk` run that path exists for."""
        import soup_cli.trainer.stream_setup as stream_setup
        from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

        calls = []
        monkeypatch.setattr(
            stream_setup, "_validate_early_ram_fit", lambda **kwargs: calls.append(kwargs)
        )
        shards = []
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, shard_calls=shards, free_ram=100,
            tcfg_overrides={"stream_source": "ram"},
        )
        assert calls == []
        assert len(shards) == 1


# --------------------------------------------------------------------------
# 3. an FP8 block source is charged its dequantised size
# --------------------------------------------------------------------------
#: Large enough that the weights, not the safetensors header, dominate the file size.
_FP8_HIDDEN, _FP8_INTER, _FP8_VOCAB, _FP8_EXPERTS, _FP8_LAYERS = 256, 128, 64, 2, 2
_FP8_BLOCK = (128, 128)  # DeepSeek-V3's own block size
_FP8_SUFFIXES = frozenset({"self_attn.o_proj.weight"})


def _fp8_pair(tmp_path):
    """(FP8 block checkpoint, the same weights dequantised, its quantization_config).

    Built with `tests/test_fp8_block_streaming.py`'s quantiser and writer: a DeepSeek-V3
    layout whose attention output and routed experts are FP8 with a scale per block, the
    embeddings, head, norms and router bf16.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from tests.test_fp8_block_streaming import _dequantised, _to_fp8, _write

    generator = torch.Generator().manual_seed(5)

    def normal(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    hidden, inter = _FP8_HIDDEN, _FP8_INTER
    state = {
        "model.embed_tokens.weight": normal(_FP8_VOCAB, hidden),
        "model.norm.weight": normal(hidden),
        "lm_head.weight": normal(_FP8_VOCAB, hidden),
    }
    linears = []
    for layer in range(_FP8_LAYERS):
        prefix = f"model.layers.{layer}."
        state[prefix + "input_layernorm.weight"] = normal(hidden)
        linears.append(prefix + "self_attn.o_proj.weight")
        state[linears[-1]] = normal(hidden, hidden)
        state[prefix + "mlp.gate.weight"] = normal(_FP8_EXPERTS, hidden)
        state[prefix + "mlp.gate.e_score_correction_bias"] = torch.randn(
            _FP8_EXPERTS, generator=generator
        )
        for expert in range(_FP8_EXPERTS):
            for name, shape in (
                ("gate_proj", (inter, hidden)),
                ("up_proj", (inter, hidden)),
                ("down_proj", (hidden, inter)),
            ):
                linears.append(prefix + f"mlp.experts.{expert}.{name}.weight")
                state[linears[-1]] = normal(*shape)
    fp8_state = _to_fp8(state, linears, _FP8_BLOCK)
    fp8 = _write(tmp_path / "fp8", fp8_state, block=_FP8_BLOCK, layers=_FP8_LAYERS)
    ref = _write(
        tmp_path / "ref", _dequantised(fp8_state, _FP8_BLOCK), quant=None, layers=_FP8_LAYERS
    )
    with open(os.path.join(fp8, "config.json"), encoding="utf-8") as handle:
        quantization_config = json.load(handle)["quantization_config"]
    return fp8, ref, quantization_config


def _fp8_cfg(quantization_config):
    return types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=_FP8_HIDDEN, num_hidden_layers=_FP8_LAYERS,
        vocab_size=_FP8_VOCAB, intermediate_size=_FP8_INTER,
        quantization_config=quantization_config,
    )


class TestAnFp8BlockSource:
    @pytest.mark.parametrize("quantization", ["none", "4bit"])
    def test_it_is_charged_its_dequantised_size(self, tmp_path, monkeypatch, quantization):
        fp8, ref, quantization_config = _fp8_pair(tmp_path)
        quant = "nf4" if quantization == "4bit" else "none"
        dequantised = _estimate(
            ref, quant=quant, suffixes=_FP8_SUFFIXES, num_hidden_layers=_FP8_LAYERS
        )
        seen = _passed(
            _drive(
                tmp_path, monkeypatch, weights=fp8, model_cfg=_fp8_cfg(quantization_config),
                quantization=quantization, suffixes=_FP8_SUFFIXES,
            )
        )
        assert [call["ram_bytes"] for call in seen["early"]] == [dequantised.ram_tier_bytes]
        assert seen["early"][0]["approximate"] is False
        # one byte per FP8 weight on disk: the old charge was well under the dequantised one
        assert 2 * _flat(_source_bytes(fp8), quant=quant) < dequantised.ram_tier_bytes

    def test_one_that_fits_its_files_but_not_its_dequantised_weights_is_refused_early(
        self, tmp_path, monkeypatch
    ):
        fp8, ref, quantization_config = _fp8_pair(tmp_path)
        need = _estimate(ref, quant="none", num_hidden_layers=_FP8_LAYERS).ram_tier_bytes
        flat = _flat(_source_bytes(fp8), quant="none")
        assert flat < need
        seen = _drive(
            tmp_path, monkeypatch, weights=fp8, model_cfg=_fp8_cfg(quantization_config),
            quantization="none", free_ram=_free_for_budget((flat + need) // 2),
        )
        text = _text(seen)
        assert "sized from its safetensors headers" in text
        assert "Refusing before sharding" in text
        assert seen["shards"] == []


# --------------------------------------------------------------------------
# 4. the early and the late check agree for the same base
# --------------------------------------------------------------------------
def _random_checkpoint(directory, tensors, *, model_type, num_hidden_layers, seed=0):
    """Real values (NF4 of an all-zero block is degenerate), the same headers as ``tensors``."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    generator = torch.Generator().manual_seed(seed)
    state = {
        key: torch.randn(*shape, generator=generator).to(dtypes[dtype])
        for key, (dtype, shape) in tensors.items()
    }
    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "model.safetensors"))
    return _with_config(directory, model_type, num_hidden_layers)


def _late_charge(weights, out, shard_kwargs):
    """Shard for real with the setup's own arguments; return what the exact check charges.

    The same calls the setup makes after sharding: the decoder-layer files' tensors, the
    embed/head shards and the resident extras (``store_total + embed_bytes``).
    """
    from soup_cli.utils.layer_shard import layer_paths, shard_checkpoint
    from soup_cli.utils.layer_stream import dtype_bytes
    from soup_cli.utils.layer_stream_runtime import (
        RamSource,
        extras_resident_bytes,
        large_layer_store_bytes,
    )

    index = shard_checkpoint(str(weights), str(out), **shard_kwargs)
    layer_specs = RamSource.layer_specs_from_paths(layer_paths(str(out), index))
    layer_store = sum(
        math.prod(shape) * dtype_bytes(stored)
        for per_layer in layer_specs
        for shape, stored in per_layer.values()
    )
    return layer_store + large_layer_store_bytes(str(out), index) + extras_resident_bytes(str(out))


_AGREEMENT_CASES = [
    ("deepseek_v3", "fused", "4bit", True),
    ("deepseek_v3", "per_expert", "4bit", True),
    ("deepseek_v3", "fused", "4bit", False),
    ("deepseek_v3", "fused", "none", True),
    ("deepseek_v3", "per_expert", "none", True),
    ("llama", "fused", "4bit", True),
    ("llama", "fused", "none", True),
]


@pytest.mark.parametrize(
    ("arch", "layout", "quantization", "double_quant"), _AGREEMENT_CASES,
    ids=["moe-fused-nf4", "moe-per_expert-nf4", "moe-fused-nf4-single", "moe-fused-plain",
         "moe-per_expert-plain", "dense-nf4", "dense-plain"],
)
def test_the_early_charge_is_what_the_check_after_sharding_charges(
    tmp_path, monkeypatch, arch, layout, quantization, double_quant
):
    if quantization == "4bit":
        pytest.importorskip("bitsandbytes")
    moe_base = arch == "deepseek_v3"
    weights = _random_checkpoint(
        tmp_path / "weights", _moe_tensors() if moe_base else _dense_tensors(),
        model_type=arch, num_hidden_layers=N_LAYERS if moe_base else 1,
    )
    seen = _passed(
        _drive(
            tmp_path, monkeypatch, weights=weights, arch=arch,
            model_cfg=_deepseek_cfg() if moe_base else _llama_cfg(), layout=layout,
            quantization=quantization, double_quant=double_quant,
            suffixes=SUFFIXES if moe_base else DENSE_SUFFIXES,
        )
    )
    (early,) = seen["early"]
    (shard_kwargs,) = seen["shards"]
    monkeypatch.undo()  # the harness's stubs off: shard and measure for real
    late = _late_charge(weights, tmp_path / "cache", shard_kwargs)
    assert early["approximate"] is False
    assert early["ram_bytes"] == late


@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_an_fp8_sources_early_charge_is_what_the_check_after_sharding_charges(
    tmp_path, monkeypatch, quantization
):
    if quantization == "4bit":
        pytest.importorskip("bitsandbytes")
    fp8, _ref, quantization_config = _fp8_pair(tmp_path)
    seen = _passed(
        _drive(
            tmp_path, monkeypatch, weights=fp8, model_cfg=_fp8_cfg(quantization_config),
            quantization=quantization, suffixes=_FP8_SUFFIXES,
        )
    )
    (early,) = seen["early"]
    (shard_kwargs,) = seen["shards"]
    monkeypatch.undo()
    assert early["ram_bytes"] == _late_charge(fp8, tmp_path / "cache", shard_kwargs)


def _refused(check):
    try:
        check()
    except ValueError:
        return True
    return False


@pytest.mark.parametrize(
    ("ram", "free", "total", "refused"),
    [
        (6_999, 10_000, None, False),
        (7_000, 10_000, None, True),  # 70% of free RAM reached: strict, like the late check
        (5_499, 10**9, 10_000, False),
        (5_500, 10**9, 10_000, True),  # 55% of physical RAM reached
        (10**12, 10**9, None, True),  # psutil without a physical total: free RAM alone
    ],
    ids=["under-free", "at-free", "under-physical", "at-physical", "no-total"],
)
def test_the_early_rule_is_the_late_rule(ram, free, total, refused):
    """The two decide alike on the same bytes: one budget, so the early check is never the
    stricter or the looser of the two."""
    from soup_cli.trainer.stream_setup import (
        _validate_early_ram_fit,
        _validate_qwen4_ngram_ram_fit,
    )

    early = _refused(
        lambda: _validate_early_ram_fit(
            base=_BASE, source_bytes=0, ram_bytes=ram, approximate=False, quantised=True,
            free_ram=free, total_ram=total,
        )
    )
    late = _refused(
        lambda: _validate_qwen4_ngram_ram_fit(
            stream_source="ram", ngram_source="disk", required_ram=ram, free_ram=free,
            resident_ram=0, total_ram=total,
        )
    )
    assert early is late is refused


# --------------------------------------------------------------------------
# 5. headers that cannot be read or listed: the old rate, labelled approximately
# --------------------------------------------------------------------------
class TestTheFallback:
    @pytest.mark.parametrize("spelling", sorted(_MALFORMED), ids=sorted(_MALFORMED))
    def test_unreadable_headers_fall_back_to_the_flat_rate_and_say_approximately(
        self, tmp_path, monkeypatch, spelling
    ):
        weights = tmp_path / "weights"
        _write_raw(weights / "model.safetensors", _MALFORMED[spelling])
        _with_config(weights, "deepseek_v3", N_LAYERS)
        seen = _drive(tmp_path, monkeypatch, weights=weights, free_ram=1)
        text = _text(seen)
        assert "approximately" in text
        assert "once quantised to NF4" in text
        assert "safetensors headers could not be listed or read" in text
        assert "Refusing before sharding" in text
        assert seen["shards"] == []
        assert seen["reads"] == 1  # the disk pre-flight's; the early check did not retry
        # the harness reports 1,024 bytes on disk: the flat rate of that is the charge
        assert [call["ram_bytes"] for call in seen["early"]] == [_flat(1_024)]
        assert seen["early"][0]["approximate"] is True

    def test_a_source_that_cannot_be_listed_falls_back(self, tmp_path, monkeypatch):
        """A symlinked snapshot without its revision: the copy step refuses it later."""
        weights = _moe_dir(tmp_path)
        seen = _passed(
            _drive(
                tmp_path, monkeypatch, weights=weights,
                plan_overrides={"needs_materialization": True, "source_revision": None},
            )
        )
        assert seen["early"][0]["approximate"] is True
        assert seen["early"][0]["ram_bytes"] == _flat(1_024)

    def test_an_unquantised_fallback_says_approximately_in_ram(self, tmp_path, monkeypatch):
        weights = tmp_path / "weights"
        _write_raw(weights / "model.safetensors", _MALFORMED["not-json"])
        _with_config(weights, "deepseek_v3", N_LAYERS)
        seen = _drive(tmp_path, monkeypatch, weights=weights, quantization="none", free_ram=1)
        text = _text(seen)
        assert "approximately" in text and "GB in RAM" in text
        assert "once quantised to NF4" not in text

    def test_a_fallback_that_fits_passes(self, tmp_path, monkeypatch):
        weights = tmp_path / "weights"
        _write_raw(weights / "model.safetensors", _MALFORMED["not-json"])
        _with_config(weights, "deepseek_v3", N_LAYERS)
        seen = _passed(_drive(tmp_path, monkeypatch, weights=weights))
        assert seen["early"][0]["approximate"] is True
        assert len(seen["shards"]) == 1


# --------------------------------------------------------------------------
# 6. dense bases
# --------------------------------------------------------------------------
class TestADenseBase:
    def test_an_nf4_dense_base_is_charged_its_unquantised_embeddings_and_norms(
        self, tmp_path, monkeypatch
    ):
        """Embeddings, the head and the norms stay at the stream dtype, which the flat NF4
        rate charged at 4 bits: refused early once the real figure is over the headroom."""
        weights = _dense_dir(tmp_path)
        need = _estimate(
            weights, arch="llama", suffixes=DENSE_SUFFIXES, num_hidden_layers=None
        ).ram_tier_bytes
        flat = _flat(_source_bytes(weights))
        assert flat < need
        seen = _drive(
            tmp_path, monkeypatch, weights=weights, arch="llama", model_cfg=_llama_cfg(),
            suffixes=DENSE_SUFFIXES, free_ram=_free_for_budget((flat + need) // 2),
        )
        text = _text(seen)
        assert "sized from its safetensors headers" in text
        assert "Refusing before sharding" in text
        assert seen["shards"] == []
        assert [call["ram_bytes"] for call in seen["early"]] == [need]

    @pytest.mark.parametrize("quantization", ["4bit", "none"])
    def test_a_dense_base_that_fits_passes(self, tmp_path, monkeypatch, quantization):
        weights = _dense_dir(tmp_path)
        quant = "nf4" if quantization == "4bit" else "none"
        need = _estimate(
            weights, arch="llama", quant=quant, suffixes=DENSE_SUFFIXES, num_hidden_layers=None
        ).ram_tier_bytes
        seen = _passed(
            _drive(
                tmp_path, monkeypatch, weights=weights, arch="llama", model_cfg=_llama_cfg(),
                quantization=quantization, suffixes=DENSE_SUFFIXES,
                free_ram=math.ceil(need / 0.7) + 2,
            )
        )
        assert [call["ram_bytes"] for call in seen["early"]] == [need]
        assert len(seen["shards"]) == 1

    def test_a_dense_unquantised_base_is_charged_what_it_streams_at(
        self, tmp_path, monkeypatch
    ):
        """No NF4, so no code tables; every tensor at the stream dtype (float32 on CPU)."""
        weights = _dense_dir(tmp_path)
        estimate = _estimate(
            weights, arch="llama", quant="none", suffixes=(), num_hidden_layers=None
        )
        assert estimate.code_table_bytes == 0 and estimate.expert_file_bytes == 0
        seen = _passed(
            _drive(
                tmp_path, monkeypatch, weights=weights, arch="llama", model_cfg=_llama_cfg(),
                quantization="none", suffixes=DENSE_SUFFIXES,
            )
        )
        assert seen["early"][0]["ram_bytes"] == sum(estimate.layer_bytes) + estimate.other_bytes


# --------------------------------------------------------------------------
# review round: psutil without figures, both ceilings, the printed figures, a bad checkpoint
# --------------------------------------------------------------------------
def _message(check):
    with pytest.raises(ValueError) as excinfo:
        check()
    return " ".join(str(excinfo.value).split())


def _early(**overrides):
    from soup_cli.trainer.stream_setup import _validate_early_ram_fit

    kwargs = dict(
        base=_BASE, source_bytes=0, ram_bytes=0, approximate=False, quantised=True,
        free_ram=10**15, total_ram=None,
    )
    kwargs.update(overrides)
    return _validate_early_ram_fit(**kwargs)


def _late(*, ram, free, total):
    from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

    return _validate_qwen4_ngram_ram_fit(
        stream_source="ram", ngram_source="disk", required_ram=ram, free_ram=free,
        resident_ram=0, total_ram=total,
    )


class TestWhenPsutilCannotReportRam:
    """After sharding, an unknown free RAM is replaced by ten times the store (with a
    notice), so only the 55% physical ceiling can still refuse, and only when the total is
    known. The early check does the same: no free-RAM rule, the physical one if it can."""

    def test_no_figures_at_all_proceed_to_sharding_with_the_notice(self, tmp_path, monkeypatch):
        import io

        from rich.console import Console

        import soup_cli.trainer.stream_setup as stream_setup

        buffer = io.StringIO()
        monkeypatch.setattr(
            stream_setup, "console", Console(file=buffer, width=400, color_system=None)
        )
        seen = _passed(
            _drive(
                tmp_path, monkeypatch, weights=_moe_dir(tmp_path), free_ram=None,
                total_ram=None,
            )
        )
        assert len(seen["shards"]) == 1
        text = " ".join(buffer.getvalue().split())
        assert "psutil unavailable" in text and "cannot size the RAM tier" in text
        assert text.count("psutil unavailable") == 1  # said once, by the check after sharding

    def test_unknown_free_ram_still_meets_the_physical_ceiling(self, tmp_path, monkeypatch):
        weights = _moe_dir(tmp_path)
        need = _estimate(weights).ram_tier_bytes
        seen = _drive(tmp_path, monkeypatch, weights=weights, free_ram=None, total_ram=need)
        text = _text(seen)
        assert "more than 55% of physical RAM" in text
        assert "Refusing before sharding" in text
        assert seen["shards"] == []

    def test_unknown_free_ram_under_the_physical_ceiling_passes(self, tmp_path, monkeypatch):
        weights = _moe_dir(tmp_path)
        need = _estimate(weights).ram_tier_bytes
        seen = _passed(
            _drive(tmp_path, monkeypatch, weights=weights, free_ram=None, total_ram=need * 2)
        )
        assert [call["ram_bytes"] for call in seen["early"]] == [need]
        assert len(seen["shards"]) == 1

    @pytest.mark.parametrize(("free", "total"), [(None, None), (None, 10**15), (10**15, None)])
    def test_an_unknown_figure_never_refuses_on_its_own(self, free, total):
        _early(ram_bytes=10**12, free_ram=free, total_ram=total)


def test_over_both_ceilings_the_physical_one_is_named_as_after_sharding():
    """The late check tests 55% of physical RAM first; the early refusal names it too."""
    early = _message(lambda: _early(ram_bytes=8_000, free_ram=10_000, total_ram=10_000))
    late = _message(lambda: _late(ram=8_000, free=10_000, total=10_000))
    for text in (early, late):
        assert "55% of physical RAM (0.0 GB)" in text, text
        assert "of RAM is free" not in text, text


class TestTheRefusalPrintsTheFiguresItCompared:
    def test_the_header_estimate(self):
        text = _message(
            lambda: _early(
                base="org/model", source_bytes=16_000_000_000, ram_bytes=30_600_000_000,
                free_ram=20_000_000_000,
            )
        )
        assert (
            "training.stream_source='ram' but org/model needs 30.6 GB of RAM once sharded"
            in text
        )
        assert "(sized from its safetensors headers; 16.0 GB on disk)" in text
        assert "and only 20.0 GB of RAM is free" in text

    def test_the_fallback(self):
        text = _message(
            lambda: _early(
                base="org/model", source_bytes=16_000_000_000, ram_bytes=4_100_000_000,
                approximate=True, free_ram=5_000_000_000,
            )
        )
        assert (
            "org/model is 16.0 GB on disk, approximately 4.1 GB once quantised to NF4" in text
        )
        assert "and only 5.0 GB of RAM is free" in text

    def test_the_physical_ceiling(self):
        text = _message(
            lambda: _early(
                source_bytes=16_000_000_000, ram_bytes=30_600_000_000,
                total_ram=40_000_000_000,
            )
        )
        assert "needs 30.6 GB of RAM once sharded" in text
        assert "16.0 GB on disk" in text
        assert "more than 55% of physical RAM (40.0 GB)" in text


@pytest.mark.parametrize("run_preflight", [True, False], ids=["after-preflight", "own-estimate"])
def test_a_checkpoint_the_sharder_refuses_is_refused_not_approximated(
    tmp_path, monkeypatch, run_preflight
):
    """Only an unreadable header falls back; a checkpoint the sharder would refuse is
    refused in its words, before any shard, whichever check sizes it first."""
    tensors = _moe_tensors()
    del tensors["model.layers.1.mlp.experts.2.up_proj.weight"]
    seen = _drive(
        tmp_path, monkeypatch, weights=_moe_dir(tmp_path, tensors), run_preflight=run_preflight
    )
    assert "incomplete expert projections" in _text(seen)
    assert seen["early"] == []
    assert seen["shards"] == []
