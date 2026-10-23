"""Real MLX multi-group AdamW, compiled state and native LoRA+ lifecycle."""

import copy
import json
from functools import partial

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402
from mlx_lm.tuner.lora import LoRALinear  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from soup_cli.cli import app  # noqa: E402
from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer  # noqa: E402
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper, _clipping_optimizer  # noqa: E402
from tests.test_mlx_dora_native import _config  # noqa: E402
from tests.test_mlx_patterns_native import _base  # noqa: E402


def _full_like(value, fill):
    return mx.full(value.shape, fill, dtype=value.dtype)


def _model(sparse=False):
    mx.random.seed(19)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model.freeze()
    model.layers = [LoRALinear.from_base(layer, r=2, scale=3, dropout=0)
                    for layer in model.layers]
    for layer in model.layers:
        layer.lora_a = mx.full((4, 2), .2)
        layer.lora_b = mx.full((2, 4), .3)
    if sparse:
        model.layers.insert(0, nn.Identity())
    return model


def _plan(scheduler="constant", warmup=0, decay=0, total=5):
    return plan_optimizer(lr=.005, optimizer="adamw", scheduler=scheduler,
                          warmup_ratio=warmup, weight_decay=decay, total_updates=total)


def _plus(model, ratio, plan):
    from soup_cli.trainer.mlx_loraplus import build_loraplus_optimizer

    return build_loraplus_optimizer(plan, model, ratio)


@pytest.mark.parametrize("scheduler", ["constant", "constant_with_warmup", "linear", "cosine"])
@pytest.mark.parametrize("warmup", [0, .4])
@pytest.mark.parametrize("ratio", [1, 2, 16])
def test_actual_updates_match_two_independent_adamw(scheduler, warmup, ratio):
    from dataclasses import replace

    model = _model()
    reference = _model()
    plan = _plan(scheduler, warmup, decay=.03)
    actual = _plus(model, ratio, plan)
    refs = [build_optimizer(plan), build_optimizer(replace(plan, peak_lr=plan.peak_lr * ratio))]
    single = build_optimizer(plan)
    control = _model()
    for step in range(5):
        grads = tree_map(lambda x: _full_like(x, (step + 1) * .2),
                         model.trainable_parameters())
        previous = dict(tree_flatten(model.trainable_parameters()))
        actual.update(model, grads)
        for suffix, opt in zip(("lora_a", "lora_b"), refs):
            group = tree_unflatten([(name, grad) for name, grad in tree_flatten(grads)
                                    if name.endswith(suffix)])
            opt.update(reference, group)
        if ratio == 1:
            single.update(control, grads)
        mx.eval(model.parameters(), reference.parameters(), actual.state)
        assert int(actual.step.item()) == step + 1
        for name, value in tree_flatten(model.trainable_parameters()):
            assert mx.allclose(value, dict(tree_flatten(reference.parameters()))[name],
                               atol=1e-7, rtol=1e-6).item(), name
            if ratio == 1:
                assert mx.array_equal(value, dict(tree_flatten(control.parameters()))[name]).item()
            if float(actual.learning_rate.item()) == 0:
                assert mx.array_equal(value, previous[name]).item()
        for group, reference_optimizer in zip(actual.optimizers, refs):
            actual_state = dict(tree_flatten(group.state))
            expected_state = dict(tree_flatten(reference_optimizer.state))
            assert actual_state.keys() == expected_state.keys()
            assert all(mx.allclose(value, expected_state[name], atol=1e-7, rtol=1e-6).item()
                       for name, value in actual_state.items())
        a, b = actual.optimizers
        assert int(a.step.item()) == int(b.step.item()) == step + 1
        assert float(b.learning_rate.item()) == pytest.approx(
            ratio * float(a.learning_rate.item()), rel=2e-6, abs=1e-9)
    assert any(not mx.array_equal(v, previous[n]).item()
               for n, v in tree_flatten(model.trainable_parameters()))


def test_global_clipping_once_matches_reference_and_distinguishes_split_clipping():
    from dataclasses import replace

    model, reference, split_model = _model(), _model(), _model()
    plan = _plan()
    clipped = _clipping_optimizer(_plus(model, 16, plan), .5)
    references = [build_optimizer(plan), build_optimizer(replace(plan, peak_lr=.08))]
    split_refs = [build_optimizer(plan), build_optimizer(replace(plan, peak_lr=.08))]
    for step in range(3):
        flat = [(name, _full_like(value, (.1 if name.endswith("lora_a") else 10)
                                    if step % 2 == 0 else (10 if name.endswith("lora_a") else .1)))
                for name, value in tree_flatten(model.trainable_parameters())]
        grads = tree_unflatten(flat)
        clipped.update(model, grads)
        global_grads, _ = optim.clip_grad_norm(grads, .5)
        for suffix, ref, split in zip(("lora_a", "lora_b"), references, split_refs):
            group = tree_unflatten([(n, g) for n, g in tree_flatten(global_grads)
                                    if n.endswith(suffix)])
            ref.update(reference, group)
            local, _ = optim.clip_grad_norm(tree_unflatten(
                [(n, g) for n, g in flat if n.endswith(suffix)]), .5)
            split.update(split_model, local)
    for name, value in tree_flatten(model.trainable_parameters()):
        assert mx.allclose(value, dict(tree_flatten(reference.parameters()))[name],
                           atol=1e-7, rtol=1e-6).item()
    assert max(mx.abs(v - dict(tree_flatten(split_model.parameters()))[n]).max().item()
               for n, v in tree_flatten(model.trainable_parameters())) > 1e-4


def test_compiled_accumulation_and_stable_recorder_state():
    models = [_model(sparse=True), _model(sparse=True)]
    opts = [_plus(model, 2, _plan("linear", .4)) for model in models]
    for model, opt, accumulation in zip(models, opts, (1, 2)):
        opt.init(model.trainable_parameters())
        captured = opt.state
        captured["rewind_row_loss"] = mx.array([0.])
        state = [model.state, captured]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(batch, previous, update):
            loss, grad = nn.value_and_grad(model, lambda m: mx.mean(m(batch) ** 2))(model)
            grad = tree_map(lambda g, old: g + old, grad, previous) if previous else grad
            captured["rewind_row_loss"] = mx.reshape(loss, (1,))
            if update:
                opt.update(model, tree_map(lambda g: g / accumulation, grad))
                grad = None
            return loss, grad

        previous = None
        for micro in range(4 * accumulation):
            _, previous = step(mx.ones((1, 4)), previous, (micro + 1) % accumulation == 0)
            mx.eval(state, previous)
            assert int(opt.step.item()) == (micro + 1) // accumulation
            assert opt.state is captured
            assert float(captured["rewind_row_loss"].item()) > 0
    for (name, value), (other, expected) in zip(
        tree_flatten(models[0].trainable_parameters()),
        tree_flatten(models[1].trainable_parameters())
    ):
        assert name == other
        assert mx.allclose(value, expected, atol=1e-6, rtol=1e-6).item()
    saved = copy.deepcopy(opts[0].state)
    restored = _plus(models[0], 2, _plan("linear", .4))
    restored.state = saved
    assert int(restored.step.item()) == 4
    assert restored.state["rewind_row_loss"].item() > 0
    gradients = tree_map(mx.ones_like, models[0].trainable_parameters())
    expected = opts[0].apply_gradients(gradients, models[0])
    actual = restored.apply_gradients(gradients, models[0])
    for (name, value), (other, reference) in zip(tree_flatten(actual), tree_flatten(expected)):
        assert name == other
        assert mx.array_equal(value, reference).item()
    restored.learning_rate = .002
    assert restored.optimizers[1].learning_rate.item() == pytest.approx(.004)


def test_unknown_or_missing_group_trainables_are_refused():
    model = _model()
    assert _plus(model, 2, _plan())
    model.extra = mx.ones((2, 2))
    with pytest.raises(ValueError, match="A/B"):
        _plus(model, 2, _plan())
    del model["extra"]
    model.layers[0].freeze(keys=["lora_a"])
    with pytest.raises(ValueError, match="A/B"):
        _plus(model, 2, _plan())


@pytest.mark.parametrize("quantized, checkpointing, accumulation, rslora", [
    (False, False, 1, False), (True, False, 1, True),
    (False, True, 2, True), (True, True, 2, False),
])
def test_public_native_plus_updates_reload_and_warm_start(
    tmp_path, monkeypatch, quantized, checkpointing, accumulation, rslora
):
    monkeypatch.chdir(tmp_path)
    base = _base(tmp_path / "base", quantized)
    cfg = _config(base, tmp_path / "adapter", dora=False, quantized=quantized)
    cfg.training.loraplus_lr_ratio = 16
    cfg.training.lora.use_rslora = rslora
    cfg.training.gradient_checkpointing = checkpointing
    cfg.training.gradient_accumulation_steps = accumulation
    data = tmp_path / "rows.jsonl"
    data.write_text('{"text":"alpha beta gamma alpha beta gamma"}\n' * 4)
    cfg.data.train, cfg.data.format, cfg.data.val_split = str(data), "plaintext", 0
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())
    import soup_cli.trainer.mlx_loraplus as plus

    build = plus.build_loraplus_optimizer
    optimizers = []

    def observe_optimizer(plan, model, ratio):
        optimizer = build(plan, model, ratio)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(plus, "build_loraplus_optimizer", observe_optimizer)
    snapshots = []
    original = MLXSFTTrainerWrapper._apply_lora

    def capture(self, model):
        frozen = dict(tree_flatten(model.parameters()))
        mx.eval(frozen)
        original(self, model)
        initial = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(initial)
        snapshots.append((self, frozen, initial))

    monkeypatch.setattr(MLXSFTTrainerWrapper, "_apply_lora", capture)
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(optimizers) == 1
    assert int(optimizers[0].step.item()) == 4 // accumulation
    batches = [json.loads(line) for line in (tmp_path / "adapter/rewind.jsonl").read_text()
               .splitlines() if json.loads(line)["kind"] == "batch"]
    assert len(batches) == 4
    assert all(record["row_tokens"][0] > 0 and record["row_loss"][0] > 0 for record in batches)
    wrapper, frozen, initial = snapshots[-1]
    params = dict(tree_flatten(wrapper.model.parameters()))
    for name, value in frozen.items():
        parent, leaf = name.rsplit(".", 1)
        assert mx.array_equal(value, params.get(f"{parent}.linear.{leaf}", params.get(name))).item()
    assert all(not mx.array_equal(v, initial[n]).item()
               for n, v in tree_flatten(wrapper.model.trainable_parameters()))
    metadata = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert metadata["loraplus_lr_ratio"] == 16
    assert metadata["peak_lr_b"] == pytest.approx(16 * cfg.training.lr)
    assert metadata["schedule_unit"] == "optimizer updates"
    from soup_cli.utils.mlx_generation import load_native_model

    fresh, _ = load_native_model(str(tmp_path / "adapter"))
    wrapper.model.eval()
    probe = mx.array([[5, 6, 7]])
    assert mx.allclose(fresh(probe), wrapper.model(probe), atol=1e-6, rtol=1e-6).item()
    # An actual weights-only warm start: capture optimizer before the native loop.
    starts = []

    def capture_optimizer(plan, model, ratio):
        optimizer = build(plan, model, ratio)
        starts.append(int(optimizer.step.item()))
        loaded = dict(tree_flatten(model.trainable_parameters()))
        assert all(mx.array_equal(v, loaded[n]).item()
                   for n, v in tree_flatten(wrapper.model.trainable_parameters()))
        return optimizer

    monkeypatch.setattr(plus, "build_loraplus_optimizer", capture_optimizer)
    cfg.output = str(tmp_path / "warm")
    resumed = MLXSFTTrainerWrapper(cfg)
    resumed.setup({"train": [{"text": "alpha beta gamma alpha beta gamma"}] * 4})
    resumed.train(resume_from_checkpoint=str(tmp_path / "adapter/adapters.safetensors"))
    assert starts == [0]


@pytest.mark.parametrize("kind", ["missing", "embedding", "8bit"])
def test_plus_targets_refuse_before_freezing(tmp_path, kind):
    from mlx_lm import load

    base = _base(tmp_path / "base")
    model, _ = load(str(base))
    cfg = _config(base, tmp_path / "out", dora=False)
    cfg.training.loraplus_lr_ratio = 2
    MLXSFTTrainerWrapper(cfg)._apply_lora(model)
    # A fresh valid dense base remains untouched on the negative path.
    model, _ = load(str(base))
    if kind == "8bit":
        nn.quantize(model, bits=8, group_size=32)
    cfg.training.lora.target_modules = {
        "missing": ["self_attn.missing"], "embedding": ["model.embed_tokens"],
        "8bit": ["self_attn.q_proj"],
    }[kind]
    before = dict(tree_flatten(model.trainable_parameters()))
    with pytest.raises(ValueError, match="matched dense or 4-bit linear targets"):
        MLXSFTTrainerWrapper(cfg)._apply_lora(model)
    assert dict(tree_flatten(model.trainable_parameters())).keys() == before.keys()
    assert not any(isinstance(module, LoRALinear) for _, module in model.named_modules())


def test_gradient_routing_rejects_unknown_tensor_after_valid_construction():
    model = _model()
    optimizer = _plus(model, 2, _plan())
    gradients = tree_map(mx.ones_like, model.trainable_parameters())
    gradients["unexpected"] = mx.ones((2, 2))
    with pytest.raises(ValueError, match="exactly.*A/B"):
        optimizer.update(model, gradients)
    assert int(optimizer.step.item()) == 0


@pytest.mark.parametrize("change", ["rank", "scale", "targets", "model", "missing_metadata"])
def test_existing_output_preserves_checkpoint_on_recipe_change(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    base = _base(tmp_path / "base")
    cfg = _config(base, tmp_path / "adapter", dora=False)
    cfg.training.loraplus_lr_ratio = 2
    rows = {"train": [{"text": "alpha beta gamma alpha"}] * 2}

    def run(config):
        wrapper = MLXSFTTrainerWrapper(config)
        wrapper.setup(rows)
        wrapper.train()

    run(cfg)
    # Compatible reuse remains supported, including a changed learning-rate ratio.
    cfg.training.loraplus_lr_ratio = 16
    run(cfg)
    output = tmp_path / "adapter"
    changed = cfg.model_copy(deep=True)
    if change == "rank":
        changed.training.lora.r = 2
    elif change == "scale":
        changed.training.lora.alpha = 16
    elif change == "targets":
        changed.training.lora.target_modules = ["self_attn.v_proj"]
    elif change == "model":
        changed.base = str(_base(tmp_path / "other-base"))
    else:
        (output / "adapter_config.json").unlink()
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="fresh output directory"):
        run(changed)
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before
