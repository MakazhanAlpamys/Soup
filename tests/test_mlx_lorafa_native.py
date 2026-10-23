"""Offline real MLX projection, compiled loop and complete artifact lifecycle."""

import json

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper  # noqa: E402
from tests.test_mlx_dora_native import _config  # noqa: E402
from tests.test_mlx_patterns_native import _base  # noqa: E402


@pytest.mark.parametrize(
    "quantized,checkpointing,accumulation,masked",
    [
        (False, False, 1, False),
        (True, False, 1, True),
        (False, True, 2, True),
        (True, True, 2, False),
    ],
)
def test_native_fa_update_save_reload(
    tmp_path, monkeypatch, quantized, checkpointing, accumulation, masked
):
    monkeypatch.chdir(tmp_path)
    cfg = _config(
        _base(tmp_path / "base", quantized), tmp_path / "adapter", dora=False, quantized=quantized
    )
    cfg.training.use_lorafa = True
    cfg.training.gradient_checkpointing = checkpointing
    cfg.training.gradient_accumulation_steps = accumulation
    cfg.data.train_on_responses_only = masked
    rows = (
        [
            {
                "messages": [
                    {"role": "user", "content": "alpha"},
                    {"role": "assistant", "content": "beta"},
                    {"role": "user", "content": "gamma"},
                    {"role": "assistant", "content": "alpha beta"},
                ]
            }
        ]
        * 4
        if masked
        else [{"text": "alpha beta gamma alpha beta gamma"}] * 4
    )
    trainer = MLXSFTTrainerWrapper(cfg)
    trainer.setup({"train": rows})
    initial_base = dict(tree_flatten(trainer.model.parameters()))
    mx.eval(initial_base)
    snapshots = {}
    original = trainer._apply_lora

    def apply(model):
        original(model)
        snapshots.update(dict(tree_flatten(model.trainable_parameters())))
        mx.eval(snapshots)

    monkeypatch.setattr(trainer, "_apply_lora", apply)
    result = trainer.train()
    assert result["total_steps"] == 4
    records = [
        json.loads(line) for line in (tmp_path / "adapter/rewind.jsonl").read_text().splitlines()
    ]
    batches = [row for row in records if row["kind"] == "batch"]
    assert len(batches) == 4
    assert all(row["row_tokens"][0] > 0 and row["row_loss"][0] > 0 for row in batches)
    params = dict(tree_flatten(trainer.model.parameters()))
    for name, value in initial_base.items():
        parent, leaf = name.rsplit(".", 1)
        assert mx.array_equal(value, params.get(f"{parent}.linear.{leaf}", params.get(name))).item()
    assert snapshots
    for name, before in snapshots.items():
        if name.endswith(".lora_a"):
            assert mx.array_equal(before, params[name]).item()
        else:
            assert name.endswith(".lora_b")
            assert not mx.array_equal(before, params[name]).item()
    assert all(n.endswith(".lora_b") for n, _ in tree_flatten(trainer.model.trainable_parameters()))
    for file in (tmp_path / "adapter").glob("*adapters.safetensors"):
        saved = mx.load(str(file))
        assert saved.keys() == snapshots.keys()
        assert all(mx.all(mx.isfinite(value)).item() for value in saved.values())
    metadata = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert metadata["lorafa"]["reference"] == "peft==0.21.2"
    assert metadata["lorafa"]["clipping"] == "raw_global_before_projection"
    from soup_cli.utils.mlx_generation import load_native_model

    fresh, _ = load_native_model(str(tmp_path / "adapter"))
    trainer.model.eval()
    probe = mx.array([[5, 6, 7]])
    assert mx.allclose(fresh(probe), trainer.model(probe), atol=1e-6, rtol=1e-6).item()
    from mlx_lm import load

    from soup_cli.utils.mlx_merge import merge_native_adapter

    merged = merge_native_adapter(
        str(tmp_path / "adapter"), str(tmp_path / "merged"), dtype="float32", dequantize=quantized
    )
    fused, _ = load(str(merged))
    assert mx.allclose(fused(probe), fresh(probe), atol=2e-5, rtol=2e-5).item()


def _optimizer_fixture(
    rank=2, width=4, output=3, degenerate=False, scale=2.0, b_zero=False, decay=0.07
):
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
    from soup_cli.trainer.mlx_optim import plan_optimizer

    mx.random.seed(31)
    model = nn.Sequential(
        LoRALinear.from_base(nn.Linear(width, output), r=rank, scale=scale, dropout=0)
    )
    model.freeze()
    layer = model.layers[0]
    layer.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    # Deliberately correlated/non-orthogonal columns; numerical policy truncates
    # singular values below rank*FP32 epsilon after the explicit regularizer.
    layer.lora_a = mx.arange(width * rank, dtype=mx.float32).reshape(width, rank) / 17 + 0.1
    if degenerate:
        layer.lora_a = mx.repeat(layer.lora_a[:, :1], rank, axis=1)
    layer.lora_b = mx.zeros((rank, output)) if b_zero else mx.full((rank, output), 0.2)
    plan = plan_optimizer(
        lr=0.003,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0,
        weight_decay=decay,
        total_updates=5,
    )
    optimizer = build_lorafa_optimizer(plan, model)
    return model, optimizer


@pytest.mark.parametrize(
    "rank,width,output,degenerate",
    [(1, 3, 5, False), (2, 4, 3, False), (3, 3, 3, False), (2, 4, 3, True)],
)
@pytest.mark.parametrize("scale", [0.5, 3.0])
@pytest.mark.parametrize("decay", [0.0, 0.07])
@pytest.mark.parametrize("b_zero", [False, True])
def test_projection_and_moments_against_independent_numpy(
    rank, width, output, degenerate, scale, b_zero, decay
):
    import numpy as np
    from mlx.utils import tree_unflatten

    model, optimizer = _optimizer_fixture(rank, width, output, degenerate, scale, b_zero, decay)
    layer = model.layers[0]
    a, b = np.array(layer.lora_a), np.array(layer.lora_b)
    gram = a.T.dot(a) + np.eye(rank, dtype=np.float32) * np.float32(1e-8)
    u, singular, vh = np.linalg.svd(gram, full_matrices=False)
    inverse = np.zeros_like(singular)
    keep = singular > rank * np.finfo(np.float32).eps * singular.max()
    inverse[keep] = 1 / singular[keep]
    projection = (vh.T * inverse) @ u.T / scale**2
    m, v = np.zeros_like(b), np.zeros_like(b)
    for step in range(1, 6):
        gradient = np.sin(np.arange(rank * output).reshape(rank, output) + step).astype("float32")
        corrected = projection @ gradient
        m, v = 0.9 * m + 0.1 * corrected, 0.999 * v + 0.001 * corrected**2
        b = (b - 0.003 * np.sqrt(1 - 0.999**step) / (1 - 0.9**step) * m / (np.sqrt(v) + 1e-6)) * (
            1 - 0.003 * decay
        )
        optimizer.update(model, tree_unflatten([("layers.0.lora_b", mx.array(gradient))]))
        mx.eval(model.parameters(), optimizer.state)
        state = optimizer.state["layers"][0]["lora_b"]
        np.testing.assert_allclose(np.array(layer.lora_b), b, atol=3e-6, rtol=3e-5)
        np.testing.assert_allclose(np.array(state["m"]), m, atol=2e-5, rtol=5e-5)
        np.testing.assert_allclose(np.array(state["v"]), v, atol=2e-5, rtol=8e-5)
        np.testing.assert_array_equal(np.array(layer.lora_a), a)
        assert int(optimizer.step.item()) == step


def test_frozen_a_preserves_activation_gradient_and_earlier_b():
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
    from soup_cli.trainer.mlx_optim import plan_optimizer

    model = nn.Sequential(
        LoRALinear.from_base(nn.Linear(3, 3), r=2, scale=2, dropout=0),
        LoRALinear.from_base(nn.Linear(3, 2), r=2, scale=2, dropout=0),
    )
    model.freeze()
    for layer in model.layers:
        layer.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        layer.lora_a = mx.full(layer.lora_a.shape, 0.2)
        layer.lora_b = mx.full(layer.lora_b.shape, 0.3)
    plan = plan_optimizer(
        lr=0.003,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0,
        weight_decay=0,
        total_updates=2,
    )
    build_lorafa_optimizer(plan, model)
    x = mx.array([[0.2, -0.3, 0.7]])
    derivative = mx.grad(lambda value: mx.sum(model(value) ** 2))(x)
    epsilon = 0.001
    for index in range(3):
        perturb = mx.eye(3)[index : index + 1] * epsilon
        numerical = (mx.sum(model(x + perturb) ** 2) - mx.sum(model(x - perturb) ** 2)) / (
            2 * epsilon
        )
        assert derivative[0, index].item() == pytest.approx(numerical.item(), rel=0.01, abs=1e-5)
    _, gradients = nn.value_and_grad(model, lambda m: mx.sum(m(x) ** 2))(model)
    assert mx.abs(gradients["layers"][0]["lora_b"]).max().item() > 1e-6
    assert all(name.endswith(".lora_b") for name, _ in tree_flatten(gradients))


def test_projection_state_roundtrip_and_loaded_a_cache(tmp_path):
    import copy

    from mlx.utils import tree_map

    from soup_cli.trainer.mlx_lorafa import adapter_tensors, load_fa_checkpoint

    model, optimizer = _optimizer_fixture()
    gradient = tree_map(lambda p: mx.full(p.shape, 0.4), model.trainable_parameters())
    optimizer.update(model, gradient)
    mx.eval(model.parameters(), optimizer.state)
    other, restored = _optimizer_fixture()
    other.update(model.parameters())
    restored.state = copy.deepcopy(optimizer.state)
    optimizer.update(model, gradient)
    restored.update(other, gradient)
    mx.eval(model.parameters(), other.parameters())
    assert mx.array_equal(model.layers[0].lora_b, other.layers[0].lora_b).item()
    # A different checkpoint must produce a different cached projector and update.
    weights = adapter_tensors(model)
    weights["layers.0.lora_a"] = weights["layers.0.lora_a"] * 3
    file = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(file), weights)
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
    from soup_cli.trainer.mlx_optim import plan_optimizer

    fresh = nn.Sequential(LoRALinear.from_base(nn.Linear(4, 3), r=2, scale=2, dropout=0))
    fresh.freeze()
    fresh.layers[0].unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    load_fa_checkpoint(fresh, str(file))
    plan = plan_optimizer(
        lr=0.003,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0,
        weight_decay=0.07,
        total_updates=5,
    )
    loaded = build_lorafa_optimizer(plan, fresh)
    assert not mx.allclose(
        loaded.state["lorafa"]["projections"]["layers.0.lora_b"],
        optimizer.state["lorafa"]["projections"]["layers.0.lora_b"],
    ).item()
    assert int(loaded.step.item()) == 0
    broken = dict(weights)
    broken.pop("layers.0.lora_a")
    mx.save_safetensors(str(file), broken)
    before = fresh.layers[0].lora_b
    with pytest.raises(ValueError, match="complete compatible"):
        load_fa_checkpoint(fresh, str(file))
    assert mx.array_equal(fresh.layers[0].lora_b, before).item()


def test_actual_pinned_peft_cpu_reference(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    import numpy as np
    from mlx.utils import tree_unflatten

    python = os.environ.get("SOUP_PEFT_REFERENCE_PYTHON")
    if not python:
        pytest.skip("separate pinned PEFT 0.21.2 CPU reference environment not selected")
    cases, fixtures = [], []
    for rank, width, output, scale, zero in [
        (1, 3, 5, 0.5, True),
        (2, 4, 3, 3.0, False),
        (3, 5, 4, 2.0, False),
    ]:
        model, optimizer = _optimizer_fixture(rank, width, output, scale=scale, b_zero=zero)
        # Full-rank but deliberately non-orthogonal A for a stable CPU/Metal comparison.
        a = np.array(model.layers[0].lora_a)
        a[:rank] += np.eye(rank, dtype=np.float32) * 0.4
        # Rebuild from actual changed A, never reuse the previous derived state.
        fresh, _ = _optimizer_fixture(rank, width, output, scale=scale, b_zero=zero)
        fresh.layers[0].lora_a = mx.array(a)
        fresh.layers[0].unfreeze(keys=["lora_a"], recurse=False)
        from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
        from soup_cli.trainer.mlx_optim import plan_optimizer

        plan = plan_optimizer(
            lr=0.003,
            optimizer="adamw_torch",
            scheduler="constant",
            warmup_ratio=0,
            weight_decay=0.07,
            total_updates=5,
        )
        optimizer = build_lorafa_optimizer(plan, fresh)
        gradients = [
            np.sin(np.arange(rank * output).reshape(rank, output) + step).astype("float32").tolist()
            for step in range(1, 6)
        ]
        cases.append(
            dict(
                a=a.tolist(),
                b=np.array(fresh.layers[0].lora_b).tolist(),
                scale=scale,
                gradients=gradients,
            )
        )
        fixtures.append((fresh, optimizer))
    script = Path(__file__).with_name("_lorafa_peft_reference.py")
    result = subprocess.run(
        [python, str(script)],
        input=json.dumps(cases),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    reference = json.loads(result.stdout)
    assert reference["version"] == "0.21.2"
    assert reference["source_sha256"] == (
        "b9f1616b3120796bd765bc2bd59390bc998d5f81b0b66bf26d58129e2e0bc82b"
    )
    for case, (model, optimizer), expected in zip(cases, fixtures, reference["cases"]):
        for gradient, values in zip(case["gradients"], expected):
            optimizer.update(model, tree_unflatten([("layers.0.lora_b", mx.array(gradient))]))
            mx.eval(model.parameters(), optimizer.state)
            state = optimizer.state["layers"][0]["lora_b"]
            for actual, key in [
                (model.layers[0].lora_b, "b"),
                (state["m"], "m"),
                (state["v"], "v"),
            ]:
                np.testing.assert_allclose(np.array(actual), values[key], atol=3e-6, rtol=1e-4)
    (tmp_path / "peft-reference.json").write_text(json.dumps(reference))


@pytest.mark.parametrize("scheduler", ["constant", "constant_with_warmup", "linear", "cosine"])
def test_compiled_accumulation_update_schedules_and_raw_clipping(scheduler):
    from functools import partial

    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_map

    from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
    from soup_cli.trainer.mlx_optim import build_lr_schedule, plan_optimizer
    from soup_cli.trainer.mlx_sft import _clipping_optimizer

    models, optimizers, histories = [], [], []
    for accumulation in (1, 2):
        model, _ = _optimizer_fixture()
        model.layers[0].unfreeze(keys=["lora_a"], recurse=False)
        plan = plan_optimizer(
            lr=0.003,
            optimizer="adamw_torch",
            scheduler=scheduler,
            warmup_ratio=0.4,
            weight_decay=0.07,
            total_updates=5,
        )
        optimizer = build_lorafa_optimizer(plan, model)
        clipped = _clipping_optimizer(optimizer, 0.15)
        optimizer.init(model.trainable_parameters())
        state = [model.state, optimizer.state]
        grad = nn.value_and_grad(model, lambda m, x: mx.mean(m(x) ** 2))

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, previous, update):
            _, gradients = grad(model, x)
            if previous is not None:
                gradients = tree_map(lambda a, b: a + b, gradients, previous)
            if update:
                gradients = tree_map(lambda g: g / accumulation, gradients)
                clipped.update(model, gradients)
                return None
            return gradients

        history = []
        for update in range(5):
            previous = None
            batch = mx.array([[0.2, -0.3, 0.7, 0.1], [-0.4, 0.2, 0.5, -0.8]]) * (update + 1)
            # Explicit oracle for clipping order, projection and optimizer moments.
            _, raw = grad(model, batch)
            raw, _ = optim.clip_grad_norm(raw, 0.15)
            corrected = (
                optimizer.state["lorafa"]["projections"]["layers.0.lora_b"]
                @ raw["layers"][0]["lora_b"]
            )
            before_m = optimizer.state["layers"][0]["lora_b"]["m"]
            expected_m = 0.9 * before_m + 0.1 * corrected
            previous_b = np.array(model.layers[0].lora_b)
            for index in range(accumulation):
                data = batch if accumulation == 1 else batch[index : index + 1]
                previous = step(data, previous, index == accumulation - 1)
                mx.eval(state, previous)
            np.testing.assert_allclose(
                np.array(optimizer.state["layers"][0]["lora_b"]["m"]),
                np.array(expected_m),
                rtol=2e-5,
                atol=1e-6,
            )
            rate = build_lr_schedule(plan)
            expected_rate = rate(mx.array(update)).item() if callable(rate) else rate
            assert optimizer.learning_rate.item() == pytest.approx(
                expected_rate, rel=3e-6, abs=1e-9
            )
            if expected_rate == 0:
                np.testing.assert_array_equal(np.array(model.layers[0].lora_b), previous_b)
            assert optimizer.step.item() == update + 1
            history.append(np.array(model.layers[0].lora_b))
        models.append(model)
        optimizers.append(optimizer)
        histories.append(history)
    np.testing.assert_allclose(histories[0], histories[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rows,expected_iterations", [(1, 2), (3, 2), (5, 4)])
def test_public_accumulation_tail_policy_and_warm_start(
    tmp_path, monkeypatch, rows, expected_iterations
):
    from typer.testing import CliRunner

    import soup_cli.trainer.mlx_lorafa as fa
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    cfg = _config(_base(tmp_path / "base"), tmp_path / "out", dora=False)
    cfg.training.use_lorafa = True
    cfg.training.gradient_accumulation_steps = 2
    source = tmp_path / "data.jsonl"
    source.write_text('{"text":"alpha beta gamma alpha beta"}\n' * rows)
    cfg.data.train, cfg.data.format, cfg.data.val_split = str(source), "plaintext", 0
    file = tmp_path / "config.json"
    file.write_text(cfg.model_dump_json())
    original = fa.build_lorafa_optimizer
    observed = []

    def capture(plan, model):
        optimizer = original(plan, model)
        observed.append(optimizer)
        return optimizer

    monkeypatch.setattr(fa, "build_lorafa_optimizer", capture)
    result = CliRunner().invoke(app, ["train", "--config", str(file), "--yes"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert observed[-1].step.item() == expected_iterations // 2
    metadata = json.loads((tmp_path / "out/adapter_config.json").read_text())
    assert metadata["iters"] == expected_iterations
    saved = mx.load(str(tmp_path / "out/adapters.safetensors"))
    cfg.output = str(tmp_path / "warm")
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": [{"text": "alpha beta gamma alpha beta"}] * 2})
    snapshots = []

    def capture_loaded(plan, model):
        tensors = fa.adapter_tensors(model)
        assert all(mx.array_equal(saved[n], tensors[n]).item() for n in saved)
        optimizer = original(plan, model)
        snapshots.append(optimizer.step.item())
        return optimizer

    monkeypatch.setattr(fa, "build_lorafa_optimizer", capture_loaded)
    wrapper.train(resume_from_checkpoint=str(tmp_path / "out/adapters.safetensors"))
    assert snapshots == [0]


def test_public_existing_output_guard_preserves_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(_base(tmp_path / "base"), tmp_path / "adapter", dora=False)
    cfg.training.use_lorafa = True
    rows = {"train": [{"text": "alpha beta gamma alpha beta"}] * 2}
    for _ in range(2):
        wrapper = MLXSFTTrainerWrapper(cfg)
        wrapper.setup(rows)
        wrapper.train()
    output = tmp_path / "adapter"
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    cfg.training.lora.alpha = 16
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup(rows)
    with pytest.raises(ValueError, match="fresh output directory"):
        wrapper.train()
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("kind", ["missing", "embedding", "8bit"])
def test_actual_model_target_refusal_before_freezing(tmp_path, kind):
    import mlx.nn as nn
    from mlx_lm import load

    base = _base(tmp_path / "base")
    model, _ = load(str(base))
    if kind == "8bit":
        nn.quantize(model, bits=8, group_size=32)
    cfg = _config(base, tmp_path / "out", dora=False)
    cfg.training.use_lorafa = True
    cfg.training.lora.target_modules = {
        "missing": ["self_attn.missing"],
        "embedding": ["model.embed_tokens"],
        "8bit": ["self_attn.q_proj"],
    }[kind]
    before = dict(tree_flatten(model.trainable_parameters()))
    with pytest.raises(ValueError, match="matched dense or 4-bit linear targets"):
        MLXSFTTrainerWrapper(cfg)._apply_lora(model)
    assert dict(tree_flatten(model.trainable_parameters())).keys() == before.keys()


def test_epsilon_bias_correction_and_post_update_decay_are_observable():
    import numpy as np
    from mlx.utils import tree_unflatten

    from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
    from soup_cli.trainer.mlx_optim import plan_optimizer

    model, _ = _optimizer_fixture()
    model.layers[0].unfreeze(keys=["lora_a"], recurse=False)
    plan = plan_optimizer(
        lr=0.003,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0,
        weight_decay=3,
        total_updates=1,
    )
    optimizer = build_lorafa_optimizer(plan, model)
    initial = np.array(model.layers[0].lora_b)
    raw = mx.full(initial.shape, 1e-7)
    gradient = np.array(optimizer.state["lorafa"]["projections"]["layers.0.lora_b"] @ raw)
    m, v = 0.1 * gradient, 0.001 * gradient**2
    delta = 0.003 * np.sqrt(1 - 0.999) / (1 - 0.9) * m / (np.sqrt(v) + 1e-6)
    expected = (initial - delta) * (1 - 0.003 * 3)
    wrong_decay = initial * (1 - 0.003 * 3) - delta
    wrong_epsilon = (initial - 0.003 * np.sqrt(1 - 0.999) / (1 - 0.9) * m / (np.sqrt(v) + 1e-8)) * (
        1 - 0.003 * 3
    )
    assert np.max(np.abs(expected - wrong_decay)) > 1e-6
    assert np.max(np.abs(expected - wrong_epsilon)) > 1e-6
    optimizer.update(model, tree_unflatten([("layers.0.lora_b", raw)]))
    np.testing.assert_allclose(np.array(model.layers[0].lora_b), expected, atol=5e-8, rtol=1e-6)


@pytest.mark.parametrize(
    "kind,diagnostic",
    [
        ("nan_a", "finite A/B"),
        ("inf_b", "finite A/B"),
        ("bf16", "compatible FP32"),
        ("shape", "compatible FP32"),
        ("scale_nan", "positive scale"),
        ("scale_huge", "positive scale"),
        ("gram_overflow", "finite FP32 Gram"),
    ],
)
def test_invalid_factors_refuse_before_a_is_frozen(kind, diagnostic):
    from soup_cli.trainer.mlx_lorafa import build_lorafa_optimizer
    from soup_cli.trainer.mlx_optim import plan_optimizer

    model, _ = _optimizer_fixture()
    layer = model.layers[0]
    layer.unfreeze(keys=["lora_a"], recurse=False)
    if kind == "nan_a":
        layer.lora_a = mx.full(layer.lora_a.shape, float("nan"))
    elif kind == "inf_b":
        layer.lora_b = mx.full(layer.lora_b.shape, float("inf"))
    elif kind == "bf16":
        layer.lora_a = layer.lora_a.astype(mx.bfloat16)
    elif kind == "shape":
        layer.lora_b = mx.zeros((3, 3))
    elif kind == "gram_overflow":
        layer.lora_a = mx.full(layer.lora_a.shape, 1e30)
    else:
        layer.scale = float("nan") if kind == "scale_nan" else 1e30
    before = dict(tree_flatten(model.trainable_parameters()))
    plan = plan_optimizer(
        lr=0.003,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0,
        weight_decay=0,
        total_updates=1,
    )
    with pytest.raises(ValueError, match=diagnostic):
        build_lorafa_optimizer(plan, model)
    assert dict(tree_flatten(model.trainable_parameters())).keys() == before.keys()


@pytest.mark.parametrize(
    "explicit,masked", [(False, False), (False, True), (True, False), (True, True)]
)
def test_native_validation_callbacks_and_update_cadence(tmp_path, monkeypatch, explicit, masked):
    import math

    import soup_cli.trainer.mlx_lorafa as fa

    monkeypatch.chdir(tmp_path)
    cfg = _config(_base(tmp_path / "base"), tmp_path / "adapter", dora=False)
    cfg.training.use_lorafa = True
    cfg.training.gradient_accumulation_steps = 2
    cfg.data.train_on_responses_only = masked
    if explicit:
        cfg.training.eval_steps = 1
    row = (
        {
            "messages": [
                {"role": "user", "content": "alpha"},
                {"role": "assistant", "content": "beta gamma"},
            ]
        }
        if masked
        else {"text": "alpha beta gamma alpha beta"}
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.setup({"train": [row] * 4, "val": [row] * 2})
    observed = []
    original = fa.train_lorafa

    def capture(**kwargs):
        callback = kwargs["training_callback"]
        report = callback.on_val_loss_report

        def record(info):
            observed.append((info["iteration"], info["val_loss"]))
            report(info)

        callback.on_val_loss_report = record
        return original(**kwargs)

    monkeypatch.setattr(fa, "train_lorafa", capture)
    wrapper.train()
    assert [iteration for iteration, _ in observed] == ([2, 4] if explicit else [0, 1, 2, 3])
    assert all(math.isfinite(value) and value > 0 for _, value in observed)
    if explicit:
        assert [(row["update"], row["optimizer_step"]) for row in wrapper._eval_log] == [
            (1, 1),
            (2, 2),
        ]
    assert wrapper.model.training
