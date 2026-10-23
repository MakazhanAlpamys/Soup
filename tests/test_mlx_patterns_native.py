"""Offline native per-module LoRA construction and artifact contracts."""

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.llama import Model, ModelArgs  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from soup_cli.cli import app  # noqa: E402
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402
from tests.test_mlx_dora_native import _checkpoint, _config  # noqa: E402


def _base(path, quantized=False):
    import mlx.nn as nn

    _checkpoint(path)
    config = json.loads((path / "config.json").read_text())
    config.update(num_hidden_layers=2, num_key_value_heads=2)
    mx.random.seed(7)
    model = Model(ModelArgs(**config))
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        config["quantization"] = {"group_size": 32, "bits": 4}
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(weights)
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("quantized", [False, True])
def test_public_patterns_construct_local_shapes_and_scales(tmp_path, monkeypatch, quantized):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    base = _base(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "adapter", dora=False, quantized=quantized)
    cfg.training.lora.rank_pattern = {"layers.0.self_attn.q_proj": 2, "q_proj": 3}
    cfg.training.lora.alpha_pattern = {"q_proj": 12, "v_proj": 20}
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"alpha beta gamma alpha"}\n' * 2)
    cfg.data.train = str(source)
    cfg.data.format = "plaintext"
    cfg.data.val_split = 0
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())
    observed = []
    captured = {}
    apply = MLXSFTTrainerWrapper._apply_lora

    def capture(self, model):
        captured["base"] = dict(tree_flatten(model.parameters()))
        mx.eval(captured["base"])
        apply(self, model)
        captured["model"] = model
        captured["initial"] = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(captured["initial"])
        assert sum(value.size for value in captured["initial"].values()) == 1408
        for index, layer in enumerate(model.layers):
            query, value = layer.self_attn.q_proj, layer.self_attn.v_proj
            rank = 2 if index == 0 else 3
            assert query.lora_a.shape == (64, rank)
            assert query.lora_b.shape == (rank, 64)
            assert query.scale == 12 / rank
            assert value.lora_a.shape == (64, 4)
            assert value.lora_b.shape == (4, 32)
            assert value.scale == 5
            observed.append(index)

    monkeypatch.setattr(MLXSFTTrainerWrapper, "_apply_lora", capture)
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    monkeypatch.chdir(tmp_path)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert observed == [0, 1]
    model = captured["model"]
    trained = dict(tree_flatten(model.parameters()))
    for name, value in captured["base"].items():
        parent, leaf = name.rsplit(".", 1)
        actual = trained.get(f"{parent}.linear.{leaf}", trained.get(name))
        assert mx.array_equal(actual, value).item(), name
    assert all(not mx.array_equal(value, captured["initial"][name]).item()
               for name, value in tree_flatten(model.trainable_parameters()))
    model.eval()
    probe = mx.array([[5, 6, 7]])
    expected = model(probe)
    from mlx_lm import load

    from soup_cli.utils.mlx_generation import load_native_model

    original, _ = load(str(base))
    assert mx.abs(expected - original(probe)).max().item() > 1e-4
    restored, _ = load_native_model(str(tmp_path / "adapter"))
    assert mx.abs(restored(probe) - expected).max().item() < 1e-6
    metadata = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert "lora_parameters" not in metadata
    assert len(metadata["soup_lora"]["modules"]) == 4
    with pytest.raises(AttributeError, match="lora_parameters"):
        load(str(base), adapter_path=str(tmp_path / "adapter"))
    import os
    import subprocess
    import sys
    from pathlib import Path

    outfile = tmp_path / "fresh.npy"
    script = (
        "import sys; import mlx.core as mx; "
        "from soup_cli.utils.mlx_generation import load_native_model; "
        "m,_=load_native_model(sys.argv[1]); "
        "mx.save(sys.argv[2],m(mx.array([[5,6,7]])))"
    )
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "adapter"), str(outfile)],
                   env=env, check=True, timeout=60, capture_output=True, text=True)
    assert mx.abs(mx.load(str(outfile)) - expected).max().item() < 1e-6
    from soup_cli.utils.mlx_merge import merge_native_adapter

    fused = merge_native_adapter(str(tmp_path / "adapter"), str(tmp_path / "fused"),
                                 dtype="float32", dequantize=quantized)
    merged, _ = load(str(fused))
    assert mx.allclose(merged(probe), expected, atol=2e-5, rtol=2e-5).item()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rslora", [False, True])
def test_non_square_nonzero_update_uses_local_rank_alpha(tmp_path, quantized, rslora):
    import math

    import numpy as np
    from mlx_lm import load

    base = _base(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "adapter", dora=False, quantized=quantized)
    cfg.training.lora.rank_pattern = {"v_proj": 2}
    cfg.training.lora.alpha_pattern = {"v_proj": 6}
    cfg.training.lora.use_rslora = rslora
    model, _ = load(str(base))
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper._apply_lora(model)
    layer = model.layers[1].self_attn.v_proj
    assert layer.lora_a.shape == (64, 2) and layer.lora_b.shape == (2, 32)
    assert layer.scale == 6 / (math.sqrt(2) if rslora else 2)
    x = np.arange(192, dtype=np.float32).reshape(1, 3, 64) / 100
    factors_a = np.arange(128, dtype=np.float32).reshape(64, 2) / 500
    factors_b = np.arange(64, dtype=np.float32).reshape(2, 32) / 300
    layer.lora_a, layer.lora_b = mx.array(factors_a), mx.array(factors_b)
    model.eval()
    delta = np.einsum("bti,ir,ro->bto", x, factors_a, factors_b) * layer.scale
    expected = np.array(layer.linear(mx.array(x))) + delta
    np.testing.assert_allclose(np.array(layer(mx.array(x))), expected, rtol=2e-5, atol=2e-5)
    assert np.abs(delta).max() > 0.1
    import mlx.nn as nn

    assert isinstance(layer.linear, nn.QuantizedLinear if quantized else nn.Linear)
    if quantized:
        assert layer.linear.bits == 4


@pytest.mark.parametrize("kind", ["missing", "embedding", "switch", "8bit"])
def test_patterns_refuse_invalid_targets_before_partial_conversion(tmp_path, kind):
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.tuner.lora import LoRALinear

    base = _base(tmp_path / "base")
    cfg = _config(base, tmp_path / "adapter", dora=False)
    cfg.training.lora.rank_pattern = {"q_proj": 2}
    good, _ = load(str(base))
    MLXSFTTrainerWrapper(cfg)._apply_lora(good)
    assert isinstance(good.layers[0].self_attn.q_proj, LoRALinear)
    model, _ = load(str(base))
    value = model.layers[0].self_attn.v_proj
    if kind == "missing":
        cfg.training.lora.target_modules = ["self_attn.q_proj", "missing"]
    elif kind == "embedding":
        model.layers[0].self_attn.v_proj = nn.Embedding(32, 64)
    elif kind == "switch":
        model.layers[0].self_attn.v_proj = SwitchLinear(64, 32, 2)
    else:
        model.layers[0].self_attn.v_proj = nn.QuantizedLinear.from_linear(value, bits=8,
                                                                        group_size=32)
    with pytest.raises(ValueError, match="matched dense or 4-bit linear targets"):
        MLXSFTTrainerWrapper(cfg)._apply_lora(model)
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
    assert dict(tree_flatten(model.trainable_parameters()))


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_nonexistent_patterns_do_not_freeze_or_convert_model(tmp_path, field):
    import mlx.nn as nn
    from mlx_lm import load

    base = _base(tmp_path / "base")
    cfg = _config(base, tmp_path / "adapter", dora=False)
    setattr(cfg.training.lora, field, {"missing": 2})
    model, _ = load(str(base))
    before = set(dict(tree_flatten(model.trainable_parameters())))
    with pytest.raises(ValueError, match=field + " matches no selected module"):
        MLXSFTTrainerWrapper(cfg)._apply_lora(model)
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
    assert set(dict(tree_flatten(model.trainable_parameters()))) == before


def _trained(tmp_path, quantized=False):
    base = _base(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "adapter", dora=False, quantized=quantized)
    cfg.training.lora.alpha_pattern = {"q_proj": 12}
    cfg.training.lora.rank_pattern = {"v_proj": 2}
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": [{"text": "alpha beta gamma alpha"}] * 2})
    wrapper.train()
    return base, cfg, wrapper


@pytest.mark.parametrize("quantized", [False, True])
def test_strict_artifact_and_warm_start_contracts(tmp_path, monkeypatch, quantized):
    import copy

    from mlx_lm import load

    from soup_cli.utils.mlx_generation import load_native_model

    monkeypatch.chdir(tmp_path)
    base, cfg, wrapper = _trained(tmp_path, quantized)
    adapter = tmp_path / "adapter"
    weights_file = adapter / "adapters.safetensors"
    metadata_file = adapter / "adapter_config.json"
    weights = mx.load(str(weights_file))
    mx.eval(weights)
    metadata = json.loads(metadata_file.read_text())
    probe = mx.array([[5, 6, 7]])
    wrapper.model.eval()
    expected = wrapper.model(probe)
    # Weight-only resume rebuilds the same per-module recipe first.
    resumed = MLXSFTTrainerWrapper(cfg)
    resumed.setup({"train": [{"text": "alpha beta gamma"}] * 2})
    resumed._apply_lora(resumed.model)
    resumed._load_checkpoint_weights(str(weights_file))
    resumed.model.eval()
    assert mx.abs(resumed.model(probe) - expected).max().item() < 1e-6
    for change in ("missing", "extra", "shape", "nonfinite"):
        altered = dict(weights)
        key = next(iter(altered))
        if change == "missing":
            altered.pop(key)
        elif change == "extra":
            altered["unexpected.lora_a"] = mx.ones((1,))
        elif change == "shape":
            altered[key] = mx.ones((1,))
        else:
            altered[key] = mx.full(altered[key].shape, float("nan"))
        mx.save_safetensors(str(weights_file), altered)
        before = dict(tree_flatten(resumed.model.trainable_parameters()))
        mx.eval(before)
        with pytest.raises(ValueError, match="tensor keys/shapes|finite floating"):
            load_native_model(str(adapter))
        with pytest.raises(ValueError, match="tensor keys/shapes/values"):
            resumed._load_checkpoint_weights(str(weights_file))
        after = dict(tree_flatten(resumed.model.trainable_parameters()))
        assert all(mx.array_equal(value, after[name]).item() for name, value in before.items())
    mx.save_safetensors(str(weights_file), weights)
    for change in ("missing", "extra", "scale", "layers"):
        altered = copy.deepcopy(metadata)
        modules = altered["soup_lora"]["modules"]
        key = next(iter(modules))
        if change == "missing":
            modules.pop(key)
        elif change == "extra":
            modules["model.layers.99.self_attn.q_proj"] = dict(modules[key])
        elif change == "scale":
            modules[key]["scale"] += 1
        else:
            altered["num_layers"] = 1
        metadata_file.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="per-module targets|metadata drift|complete base"):
            load_native_model(str(adapter))
    metadata_file.write_text(json.dumps(metadata))
    # Same tensor dimensions but a different requested alpha must fail warm-start.
    changed_cfg = cfg.model_copy(deep=True)
    changed_cfg.training.lora.alpha_pattern = {"q_proj": 16}
    changed = MLXSFTTrainerWrapper(changed_cfg)
    changed.model, _ = load(str(base))
    changed._apply_lora(changed.model)
    with pytest.raises(ValueError, match="metadata is incompatible"):
        changed._load_checkpoint_weights(str(weights_file))
    uniform_cfg = cfg.model_copy(deep=True)
    uniform_cfg.training.lora.rank_pattern = None
    uniform_cfg.training.lora.alpha_pattern = None
    uniform = MLXSFTTrainerWrapper(uniform_cfg)
    uniform.model, _ = load(str(base))
    uniform._apply_lora(uniform.model)
    with pytest.raises(ValueError, match="requires matching rank/alpha patterns"):
        uniform._load_checkpoint_weights(str(weights_file))
    metadata_file.unlink()
    with pytest.raises(ValueError, match="requires valid per-module metadata"):
        resumed._load_checkpoint_weights(str(weights_file))


@pytest.mark.parametrize("variant", ["packing", "lorafa", "loraplus", "pretrain"])
def test_pattern_combinations_refused_with_ordinary_valid_control(tmp_path, variant):
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear

    base = _base(tmp_path / "base")
    cfg = _config(base, tmp_path / "adapter", dora=False)
    cfg.training.lora.rank_pattern = {"q_proj": 2}
    good, _ = load(str(base))
    MLXSFTTrainerWrapper(cfg)._apply_lora(good)
    assert isinstance(good.layers[0].self_attn.q_proj, LoRALinear)
    if variant == "packing":
        cfg.training.packing = True
    elif variant == "lorafa":
        cfg.training.use_lorafa = True
    elif variant == "loraplus":
        cfg.training.loraplus_lr_ratio = 2
    else:
        cfg.task = "pretrain"
    model, _ = load(str(base))
    with pytest.raises(ValueError, match="ordinary LoRA/rsLoRA text SFT only"):
        MLXSFTTrainerWrapper(cfg)._apply_lora(model)


def test_periodic_adapter_is_reconstructible_after_interruption(tmp_path, monkeypatch):
    import mlx.nn as nn
    from mlx_lm.tuner import trainer

    from soup_cli.utils.mlx_generation import load_native_model

    monkeypatch.chdir(tmp_path)
    base = _base(tmp_path / "base")
    cfg = _config(base, tmp_path / "adapter", dora=False)
    cfg.training.lora.alpha_pattern = {"q_proj": 12}
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": [{"text": "alpha beta gamma"}] * 2})

    def interrupted(**kwargs):
        # Observe metadata before the upstream loop can write periodic weights.
        path = tmp_path / "adapter"
        assert (path / "adapter_config.json").is_file()
        model = kwargs["model"]
        layer = model.layers[0].self_attn.q_proj
        layer.lora_b = mx.ones_like(layer.lora_b) * 0.02
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(path / "adapters.safetensors"), weights)
        loaded, _ = load_native_model(str(path))
        assert isinstance(loaded.layers[0].self_attn.q_proj.linear, nn.Linear)
        model.eval()
        assert mx.allclose(loaded(mx.array([[5, 6]])), model(mx.array([[5, 6]])),
                           atol=1e-6, rtol=1e-6).item()
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "train", interrupted)
    with pytest.raises(KeyboardInterrupt):
        wrapper.train()


@pytest.mark.parametrize("mode", ["rank-only", "alpha-only", "qualified-target"])
def test_independent_patterns_and_explicit_qualified_target(tmp_path, mode):
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear

    base = _base(tmp_path / "base")
    cfg = _config(base, tmp_path / "adapter", dora=False)
    if mode == "alpha-only":
        cfg.training.lora.alpha_pattern = {"q_proj": 12}
    else:
        cfg.training.lora.rank_pattern = {"q_proj" if mode == "rank-only" else "v_proj": 2}
    if mode == "qualified-target":
        cfg.training.lora.target_modules = ["model.layers.1.self_attn.v_proj"]
    model, _ = load(str(base))
    MLXSFTTrainerWrapper(cfg)._apply_lora(model)
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, LoRALinear)}
    if mode == "qualified-target":
        assert set(modules) == {"model.layers.1.self_attn.v_proj"}
        layer = modules["model.layers.1.self_attn.v_proj"]
        assert layer.lora_a.shape == (64, 2) and layer.lora_b.shape == (2, 32)
        assert layer.scale == 4
        assert sum(value.size for _, value in tree_flatten(model.trainable_parameters())) == 192
    else:
        assert len(modules) == 4
        for name, layer in modules.items():
            query = name.endswith("q_proj")
            expected_rank = 2 if query and mode == "rank-only" else 4
            expected_alpha = 12 if query and mode == "alpha-only" else 8
            assert layer.lora_a.shape[1] == expected_rank
            assert layer.lora_b.shape[0] == expected_rank
            assert layer.scale == expected_alpha / expected_rank


def test_incompatible_existing_output_preserves_old_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, cfg, _ = _trained(tmp_path)
    # Same-recipe reuse is an adjacent valid control with real training/save.
    repeated = MLXSFTTrainerWrapper(cfg)
    rows = {"train": [{"text": "alpha beta gamma alpha"}] * 2}
    repeated.setup(rows)
    repeated.train()
    output = tmp_path / "adapter"
    before = {p.name: p.read_bytes() for p in output.glob("*") if p.is_file()}
    changed = cfg.model_copy(deep=True)
    changed.training.lora.alpha_pattern = {"q_proj": 16}
    wrapper = MLXSFTTrainerWrapper(changed)
    wrapper.setup(rows)
    with pytest.raises(ValueError, match="fresh output directory"):
        wrapper.train()
    assert {p.name: p.read_bytes() for p in output.glob("*") if p.is_file()} == before
