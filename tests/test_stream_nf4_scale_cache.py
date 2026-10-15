"""Layer streaming: ``training.stream_nf4_scale_cache`` keeps expanded NF4 scales resident.

A double-quantised NF4 weight stores its per-block absmax as uint8 codes plus a
nested absmax and offset. On bitsandbytes' dequantise + linear path every visit
first expands those to fp32 (``dequantize_blockwise`` + ``+= offset``) and only
then dequantises the 4-bit weight: two extra kernels and a temporary per projection,
on the forward and again on the checkpoint recompute. The base is frozen, so the
fp32 scales of a (layer, weight) never change. With the opt-in on, the streamed
layer expands them once and keeps them on the compute device, at a persistent
cost of 4 bytes per 64-element block (416 MiB for Mistral-7B) that the VRAM
pre-flight charges up front.

The fused custom-GEMM path is left alone: it dequantises the nested scales inside
its kernel, and feeding it pre-expanded scales changes the arithmetic.
"""

import pytest

from soup_cli.config.loader import load_config_from_string


def _cfg(**training):
    body = "\n".join(f"  {k}: {v}" for k, v in training.items())
    return (
        "base: HuggingFaceTB/SmolLM2-135M\n"
        "task: sft\n"
        "data:\n"
        "  train: d.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 1\n"
        "  lora:\n"
        "    r: 8\n" + body + "\n"
    )


class TestSchema:
    def test_defaults_off(self):
        cfg = load_config_from_string(_cfg(stream_layers="true", quantization="4bit"))
        assert cfg.training.stream_nf4_scale_cache is False

    def test_parses_with_streamed_nf4(self):
        cfg = load_config_from_string(
            _cfg(stream_layers="true", quantization="4bit", stream_nf4_scale_cache="true")
        )
        assert cfg.training.stream_nf4_scale_cache is True

    def test_refused_while_streaming_is_off(self):
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="false", stream_nf4_scale_cache="true"))
        assert "stream_nf4_scale_cache" in str(exc.value)
        assert "stream_layers" in str(exc.value)

    def test_refused_without_4bit(self):
        with pytest.raises(Exception) as exc:
            load_config_from_string(
                _cfg(stream_layers="true", quantization="none", stream_nf4_scale_cache="true")
            )
        message = str(exc.value)
        assert "stream_nf4_scale_cache" in message
        assert "quantization='4bit'" in message

    def test_refused_without_double_quant(self):
        with pytest.raises(Exception) as exc:
            load_config_from_string(
                _cfg(
                    stream_layers="true",
                    quantization="4bit",
                    bnb_4bit_use_double_quant="false",
                    stream_nf4_scale_cache="true",
                )
            )
        message = str(exc.value)
        assert "stream_nf4_scale_cache" in message
        assert "bnb_4bit_use_double_quant" in message


def _spec(shape, nested=True, blocksize=64):
    from soup_cli.utils.layer_shard import NF4WeightSpec

    return NF4WeightSpec(
        shape=tuple(shape),
        dtype="bfloat16",
        blocksize=blocksize,
        quant_type="nf4",
        nested=nested,
        nested_blocksize=256,
    )


class TestCacheBytes:
    def test_mistral_7b_is_416_mib(self):
        from soup_cli.utils.layer_stream import nf4_scale_cache_bytes

        hidden, kv, inter = 4096, 1024, 14336
        specs = {
            "self_attn.q_proj.weight": _spec((hidden, hidden)),
            "self_attn.k_proj.weight": _spec((kv, hidden)),
            "self_attn.v_proj.weight": _spec((kv, hidden)),
            "self_attn.o_proj.weight": _spec((hidden, hidden)),
            "mlp.gate_proj.weight": _spec((inter, hidden)),
            "mlp.up_proj.weight": _spec((inter, hidden)),
            "mlp.down_proj.weight": _spec((hidden, inter)),
        }
        assert nf4_scale_cache_bytes(specs, n_layers=32) == 436_207_616

    def test_a_partial_block_rounds_up(self):
        from soup_cli.utils.layer_stream import nf4_scale_cache_bytes

        # 65 elements -> 2 blocks of 64 -> 8 bytes per layer.
        assert nf4_scale_cache_bytes({"w": _spec((5, 13))}, n_layers=3) == 24

    def test_non_nested_weights_cost_nothing(self):
        from soup_cli.utils.layer_stream import nf4_scale_cache_bytes

        specs = {"a": _spec((64, 64), nested=False), "b": _spec((64, 64))}
        assert nf4_scale_cache_bytes(specs, n_layers=2) == 2 * 64 * 4

    def test_the_block_size_is_read_from_the_spec(self):
        from soup_cli.utils.layer_stream import nf4_scale_cache_bytes

        assert nf4_scale_cache_bytes({"w": _spec((128, 128), blocksize=128)}, n_layers=1) == 512


class TestPeakEstimate:
    def _estimate(self, **extra):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        return estimate_stream_peak_vram(
            layer_bytes=1000,
            buffers=2,
            extras_bytes=0,
            adapter_params=10,
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            n_layers=4,
            seq_len=4,
            **extra,
        )

    def test_persistent_bytes_add_exactly(self):
        assert self._estimate(persistent_bytes=12_345) == self._estimate() + 12_345

    def test_default_adds_nothing(self):
        assert self._estimate(persistent_bytes=0) == self._estimate()


class _Lora:
    r = 8
    target_modules = ["q_proj", "v_proj"]


class _ModelConfig:
    vocab_size = 64
    hidden_size = 8
    intermediate_size = 16
    num_hidden_layers = 4


class _Data:
    max_length = 4


class _Config:
    data = _Data()


class _TrainConfig:
    batch_size = 1
    stream_buffers = 2
    stream_vram_probe = False
    lora = _Lora()

    def __init__(self, cache):
        self.stream_nf4_scale_cache = cache


class _Index:
    n_layers = 4

    def __init__(self):
        self.quant_specs = {"self_attn.q_proj.weight": _spec((64, 64))}


class TestPreflightCharge:
    def _captured_persistent_bytes(self, monkeypatch, cache):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin
        from soup_cli.utils import layer_stream

        real = layer_stream.estimate_stream_peak_vram
        seen = {}

        def capture(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", capture)
        lines, _plan = StreamingSetupMixin()._stream_budget_lines(
            _Config(),
            _TrainConfig(cache),
            model_config=_ModelConfig(),
            layer_bytes=1000,
            embed_bytes=0,
            index=_Index(),
            on_cuda=False,
        )
        return seen.get("persistent_bytes", 0), "\n".join(lines)

    def test_the_cache_is_charged_when_on(self, monkeypatch):
        charged, text = self._captured_persistent_bytes(monkeypatch, cache=True)
        # 4 layers x 64 blocks x 4 bytes.
        assert charged == 4 * 64 * 4
        assert "stream_nf4_scale_cache" in text

    def test_nothing_is_charged_when_off(self, monkeypatch):
        charged, text = self._captured_persistent_bytes(monkeypatch, cache=False)
        assert charged == 0
        assert "stream_nf4_scale_cache" not in text


def _nf4_stream(tmp_path, *, cache, n_layers=2, double_quant=True, device="cpu", dtype="float32"):
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    from soup_cli.utils.layer_shard import QUANT_NF4, shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )
    from tests.test_v07202 import _tiny_llama_dir, _tiny_lora

    weights, _resident, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers)
    probe = build_meta_skeleton(weights, dtype=dtype, quant=QUANT_NF4)
    suffixes = quantised_layer_suffixes(probe)
    del probe
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(
        weights,
        shards,
        dtype=dtype,
        arch="llama",
        quant=QUANT_NF4,
        quant_suffixes=suffixes,
        quant_device=device,
        double_quant=double_quant,
    )
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=_tiny_lora(),
        device=device,
        dtype=dtype,
        buffers=2,
        pin=False,
        seed=3,
        quant=QUANT_NF4,
        double_quant=double_quant,
        nf4_scale_cache=cache,
    )
    return model, runtime, index


def _count_blockwise_dequant(monkeypatch):
    import bitsandbytes.functional as bnb_functional

    real = bnb_functional.dequantize_blockwise
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(bnb_functional, "dequantize_blockwise", counting)
    return calls


def _step(model, optimizer, ids):
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer.step()
    return out.logits.detach().clone(), out.loss.detach().clone(), grads


class TestRuntime:
    def test_off_builds_no_cache(self, tmp_path):
        _model, runtime, _index = _nf4_stream(tmp_path, cache=False)
        assert runtime.scale_cache is None
        assert runtime.stats()["nf4_scale_cache_bytes"] == 0

    def test_one_step_fills_every_nested_weight_once(self, tmp_path):
        import torch

        from soup_cli.utils.layer_stream import nf4_scale_cache_bytes

        model, runtime, index = _nf4_stream(tmp_path, cache=True)
        nested = sum(1 for spec in index.quant_specs.values() if spec.nested)
        assert nested > 0
        assert len(runtime.scale_cache) == 0

        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        model(input_ids=ids, labels=ids).loss.backward()

        assert len(runtime.scale_cache) == index.n_layers * nested
        # Every layer is filled, so the planned bytes are exactly what is held.
        assert runtime.scale_cache.nbytes == nf4_scale_cache_bytes(
            index.quant_specs, n_layers=index.n_layers
        )
        assert runtime.stats()["nf4_scale_cache_bytes"] == runtime.scale_cache.nbytes

    def test_a_filled_cache_skips_the_nested_expansion(self, tmp_path, monkeypatch):
        import torch

        model, runtime, _index = _nf4_stream(tmp_path, cache=True)
        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        model(input_ids=ids, labels=ids).loss.backward()
        filled = len(runtime.scale_cache)

        calls = _count_blockwise_dequant(monkeypatch)
        model(input_ids=ids, labels=ids).loss.backward()
        assert calls["n"] == 0
        assert len(runtime.scale_cache) == filled

    def test_without_the_cache_every_visit_expands(self, tmp_path, monkeypatch):
        """The control for the test above: the counter does see the expansion."""
        import torch

        model, _runtime, _index = _nf4_stream(tmp_path, cache=False)
        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        calls = _count_blockwise_dequant(monkeypatch)
        model(input_ids=ids, labels=ids).loss.backward()
        assert calls["n"] > 0

    def test_training_is_bit_identical_with_and_without_the_cache(self, tmp_path):
        import torch

        runs = {}
        for cache in (False, True):
            torch.manual_seed(0)
            model, _runtime, _index = _nf4_stream(tmp_path / str(cache), cache=cache)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-2
            )
            generator = torch.Generator().manual_seed(1)
            steps = []
            for _ in range(3):
                ids = torch.randint(0, 64, (1, 8), generator=generator)
                steps.append(_step(model, optimizer, ids))
            runs[cache] = steps

        nonzero = False
        for (logits_a, loss_a, grads_a), (logits_b, loss_b, grads_b) in zip(
            runs[False], runs[True]
        ):
            assert torch.equal(logits_a, logits_b)
            assert torch.equal(loss_a, loss_b)
            assert len(grads_a) == len(grads_b)
            for grad_a, grad_b in zip(grads_a, grads_b):
                assert torch.equal(grad_a, grad_b)
                nonzero = nonzero or bool(grad_a.abs().sum() > 0)
        assert nonzero, "vacuous: every gradient was zero"

    def test_close_releases_the_cache(self, tmp_path):
        import torch

        model, runtime, _index = _nf4_stream(tmp_path, cache=True)
        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        model(input_ids=ids, labels=ids).loss.backward()
        assert len(runtime.scale_cache) > 0

        runtime.close()
        assert len(runtime.scale_cache) == 0
        assert runtime.scale_cache.nbytes == 0

    def test_the_fused_path_keeps_the_nested_state(self, tmp_path, monkeypatch):
        """The custom GEMM folds the nested dequantisation into its kernel; giving
        it pre-expanded scales changed the loss in the study this ports. The fused
        arm must receive the original nested state and must not fill the cache."""
        import torch
        from bitsandbytes.functional import dequantize_4bit

        import soup_cli.utils.layer_stream_runtime as runtime_module

        model, runtime, _index = _nf4_stream(tmp_path, cache=True)
        seen = []

        def fake_fused(x, packed, quant_state, bias=None):
            seen.append(bool(quant_state.nested))
            weight = dequantize_4bit(packed, quant_state)
            return torch.nn.functional.linear(x, weight.to(x.dtype), bias)

        monkeypatch.setattr(
            runtime_module, "_can_use_checkpoint_visible_nf4_gemm", lambda *_a, **_k: True
        )
        monkeypatch.setattr(runtime_module, "checkpoint_visible_nf4_linear", fake_fused)

        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            model(input_ids=ids)
        assert seen and all(seen)
        assert len(runtime.scale_cache) == 0


class TestEdges:
    def test_shards_without_double_quant_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no nested NF4 weights"):
            _nf4_stream(tmp_path, cache=True, double_quant=False)

    def test_a_key_reused_for_another_shape_is_refused(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.functional import quantize_4bit

        from soup_cli.utils.layer_stream_runtime import NF4ScaleCache

        cache = NF4ScaleCache()
        _packed, first = quantize_4bit(
            torch.randn(64, 64), compress_statistics=True, quant_type="nf4"
        )
        _packed, second = quantize_4bit(
            torch.randn(32, 128), compress_statistics=True, quant_type="nf4"
        )
        cache.dense_state((0, "w"), first)
        with pytest.raises(RuntimeError, match="was built for shape"):
            cache.dense_state((0, "w"), second)

    def test_filled_under_inference_mode_then_trained(self, tmp_path):
        """An eval or generation pass may be the first to visit a layer; the
        scales it leaves behind must serve training unchanged."""
        import torch

        runs = {}
        for cache in (False, True):
            torch.manual_seed(0)
            model, _runtime, _index = _nf4_stream(tmp_path / str(cache), cache=cache)
            ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(2))
            with torch.inference_mode():
                model(input_ids=ids)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-2
            )
            runs[cache] = [_step(model, optimizer, ids) for _ in range(2)]

        for (logits_a, loss_a, grads_a), (logits_b, loss_b, grads_b) in zip(
            runs[False], runs[True]
        ):
            assert torch.equal(logits_a, logits_b)
            assert torch.equal(loss_a, loss_b)
            for grad_a, grad_b in zip(grads_a, grads_b):
                assert torch.equal(grad_a, grad_b)


class TestSetupWiring:
    """The real setup path, from YAML to the runtime builder (spied), on CPU."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_setting_reaches_the_runtime_builder(self, tmp_path, monkeypatch, flag):
        pytest.importorskip("torch")
        pytest.importorskip("bitsandbytes")
        from unittest.mock import MagicMock

        import soup_cli.utils.layer_stream as ls
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from tests.test_issue623_stream_pin_allocation import (
            _FREE_BYTES,
            _cfg_yaml,
            _tiny_checkpoint,
        )

        weights = _tiny_checkpoint(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            "soup_cli.utils.spectrum_scan.resolve_model_weights", lambda *_a, **_k: weights
        )
        monkeypatch.setattr(ls, "free_ram_bytes", lambda: _FREE_BYTES)
        monkeypatch.setattr(
            ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("nvme")
        )
        captured = {}

        def spy_build(**kwargs):
            captured.update(kwargs)
            runtime = MagicMock()
            runtime.tier = "ram"
            runtime.stats.return_value = {
                "tier": "ram",
                "store_bytes": 1_000_000,
                "disk_bytes": 0,
                "pinned": False,
                "read_ahead": None,
                "buffers": 2,
                "buffer_bytes": 4_000,
                "n_layers": 3,
            }
            return MagicMock(), runtime

        monkeypatch.setattr(rt, "build_streamed_model", spy_build)
        text = _cfg_yaml(
            weights, "", extra_training_yaml=f"  stream_nf4_scale_cache: {str(flag).lower()}\n"
        ).replace("quantization: none", "quantization: 4bit")
        cfg = load_config_from_string(text)
        wrapper = SFTTrainerWrapper(cfg)
        wrapper.device = "cpu"
        wrapper._setup_streaming_transformers(cfg, cfg.training)

        assert captured["nf4_scale_cache"] is flag


class TestCuda:
    """bitsandbytes' CUDA kernels on the dense path, with and without the cache."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("force_dense", [True, False])
    def test_bit_identical_on_cuda(self, tmp_path, monkeypatch, force_dense):
        torch = pytest.importorskip("torch")
        import soup_cli.utils.layer_stream_runtime as runtime_module

        if force_dense:
            # Tiny shapes would otherwise go to the fused GEMM, which the cache
            # never touches; forcing the dense arm is what puts the cache to work.
            monkeypatch.setattr(
                runtime_module, "_can_use_checkpoint_visible_nf4_gemm", lambda *_a, **_k: False
            )
        runs = {}
        for cache in (False, True):
            torch.manual_seed(0)
            model, runtime, _index = _nf4_stream(
                tmp_path / str(cache), cache=cache, device="cuda", dtype="bfloat16"
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-2
            )
            generator = torch.Generator().manual_seed(1)
            steps = []
            for _ in range(3):
                ids = torch.randint(0, 64, (1, 8), generator=generator).cuda()
                steps.append(_step(model, optimizer, ids))
            runs[cache] = steps
            if cache:
                filled = len(runtime.scale_cache)
            runtime.close()

        assert (filled > 0) is force_dense
        for (logits_a, loss_a, grads_a), (logits_b, loss_b, grads_b) in zip(
            runs[False], runs[True]
        ):
            assert torch.equal(logits_a, logits_b)
            assert torch.equal(loss_a, loss_b)
            for grad_a, grad_b in zip(grads_a, grads_b):
                assert torch.equal(grad_a, grad_b)
