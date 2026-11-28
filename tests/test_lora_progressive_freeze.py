"""Progressive freezing and its optimizer-window record."""

import json
from types import SimpleNamespace

import pytest


def _model():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(6))
    for layer in model.layers:
        layer.lora_weight = torch.nn.Parameter(torch.ones(1))
    return model


def _callback(tmp_path, **settings):
    from soup_cli.utils.layer_select import LayerSelectPolicy, _LayerSelectCallback

    return _LayerSelectCallback(
        LayerSelectPolicy("progressive", **settings),
        dataset=None,
        collator=None,
        batch_size=1,
        output_dir=str(tmp_path),
    )


def _start(callback, model, path, step=0):
    args = SimpleNamespace(device="cpu", output_dir=str(path))
    state = SimpleNamespace(global_step=step)
    callback.on_train_begin(args, state, model=model)
    return args, state


def _step(callback, args, state, norms, optimizer=None):
    import torch

    for idx in callback.selection.layers:
        param = callback.by_layer[idx][0][1]
        norm = norms(idx, state.global_step + 1)
        param.grad = None if norm is None else torch.full_like(param, norm)
    callback.on_pre_optimizer_step(args, state)
    state.global_step += 1
    callback.on_step_end(args, state, optimizer=optimizer)


def _record(path):
    return json.loads((path / "lora_layers.json").read_text(encoding="utf-8"))


def test_schedule_freezes_after_step_and_clears_only_frozen_state(tmp_path):
    from rich.console import Console

    from tests.conftest import strip_ansi

    model = _model()
    cb = _callback(tmp_path, freeze_step=2, freeze_below=3, interval_steps=1)
    cb.console = Console(record=True, force_terminal=True, width=120)
    args, state = _start(cb, model, tmp_path)
    params = [layer.lora_weight for layer in model.layers]
    optimizer = SimpleNamespace(state={param: {"step": 1} for param in params})

    _step(cb, args, state, lambda idx, step: 1.0, optimizer)
    assert all(param.requires_grad for param in params)
    _step(cb, args, state, lambda idx, step: 1.0, optimizer)
    assert all(not param.requires_grad and param.grad is None for param in params[:3])
    assert all(param.requires_grad for param in params[3:])
    assert set(optimizer.state) == set(params[3:])
    assert _record(tmp_path)["history"][-1] == {"step": 2, "layers": [3, 4, 5]}
    assert (
        "LoRA layers (progressive): step 2: layers below 3 frozen, training 3 of 6"
        in strip_ansi(cb.console.export_text(styles=True))
    )


@pytest.mark.parametrize(("cap", "cut"), [(None, 3), (2, 2), (6, 5)])
def test_rule_cap_preserves_a_trainable_layer(tmp_path, cap, cut):
    cb = _callback(tmp_path, interval_steps=2, max_freeze_below=cap)
    args, state = _start(cb, _model(), tmp_path)
    for _ in range(4):
        _step(cb, args, state, lambda idx, step: 1.0)
    assert _record(tmp_path)["history"][-1] == {"step": 4, "layers": list(range(cut, 6))}


def test_rule_stops_at_first_unstable_layer_and_deadline_raises_cut(tmp_path):
    cb = _callback(
        tmp_path, interval_steps=2, freeze_deadline=5, min_freeze_below=3, max_freeze_below=4
    )
    args, state = _start(cb, _model(), tmp_path)
    for _ in range(5):
        _step(cb, args, state, lambda idx, step: float(step) if idx == 1 else 1.0)
    assert _record(tmp_path)["history"] == [
        {"step": 0, "layers": list(range(6))},
        {"step": 4, "layers": list(range(1, 6))},
        {"step": 5, "layers": list(range(3, 6))},
    ]


def test_deadline_never_lowers_the_cut(tmp_path):
    cb = _callback(tmp_path, interval_steps=1, freeze_deadline=3, min_freeze_below=2)
    args, state = _start(cb, _model(), tmp_path)
    for _ in range(3):
        _step(cb, args, state, lambda idx, step: 1.0)
    assert cb.selection.history == [(0, list(range(6))), (2, [3, 4, 5])]


@pytest.mark.parametrize(("smoothing", "cut"), [(0.0, 0), (0.8, 3)])
def test_rule_smoothing_warms_up_and_rides_over_an_outlier(tmp_path, smoothing, cut):
    cb = _callback(tmp_path, interval_steps=1, freeze_smoothing=smoothing)
    args, state = _start(cb, _model(), tmp_path)
    for step in range(1, 4):
        _step(cb, args, state, lambda idx, step: 1.6 if idx == 0 and step == 2 else 1.0)
        if step < 3:
            assert cb.selection.layers == list(range(6))
    assert cb.selection.layers == list(range(cut, 6))


@pytest.mark.parametrize("smoothing", [0.0, 0.8])
def test_nonfinite_windows_are_skipped_without_advancing_the_rule(tmp_path, smoothing):
    cb = _callback(tmp_path, interval_steps=1, freeze_smoothing=smoothing)
    args, state = _start(cb, _model(), tmp_path)
    for _ in range(3):
        _step(cb, args, state, lambda idx, step: float("inf") if step in (1, 3) else 2.0)
    cb.on_train_end(args, state)
    record = _record(tmp_path)
    assert record["layers"] == list(range(6))
    assert record["skipped_windows"] == [1, 3]
    assert record["norms"] == [{"step": 2, "layers": {str(idx): 2.0 for idx in range(6)}}]


def test_scaler_skip_invalidates_even_a_window_with_finite_adapter_norms(tmp_path):
    cb = _callback(tmp_path, interval_steps=2)
    args, state = _start(cb, _model(), tmp_path)
    for step in range(1, 5):
        optimizer = SimpleNamespace(step_was_skipped=(step == 3))
        _step(cb, args, state, lambda idx, step: 1.0, optimizer)
    cb.on_train_end(args, state)
    record = _record(tmp_path)
    assert record["layers"] == list(range(6))
    assert record["skipped_windows"] == [4]
    assert [window["step"] for window in record["norms"]] == [2]


def test_features_only_use_completed_valid_gradient_windows(tmp_path):
    cb = _callback(tmp_path, feature_step=8, interval_steps=2, freeze_step=10, freeze_below=2)
    args, state = _start(cb, _model(), tmp_path)
    for step in range(1, 9):
        optimizer = SimpleNamespace(step_was_skipped=(step == 5))
        _step(cb, args, state, lambda idx, step: float("inf") if step == 1 else step,
              optimizer)
    record = _record(tmp_path)
    assert record["skipped_windows"] == [2, 6]
    assert [entry["step"] for entry in record["norms"]] == [4, 8]
    assert record["features"]["layers"]["0"] == {"grad_mean": 11.0, "grad_cv": 0.363636}


def test_grad_features_ignore_legacy_nonfinite_windows():
    from soup_cli.utils.layer_features import grad_features

    norms = [(10, {0: 100.0, 1: float("inf")}), (20, {0: 2.0}), (30, {0: 4.0})]
    features = grad_features(norms)
    assert features[0]["grad_mean"] == pytest.approx(3.0)
    assert features[0]["grad_cv"] == pytest.approx(1.0 / 3.0)
    assert 1 not in features


def test_features_resume_from_old_checkpoint_and_persist_after_capture(tmp_path, caplog):
    settings = {"interval_steps": 1, "freeze_step": 4, "freeze_below": 2}
    original = _callback(tmp_path, **settings)
    args, state = _start(original, _model(), tmp_path)
    _step(original, args, state, lambda idx, step: 2.0)
    original.on_save(args, state)
    checkpoint = tmp_path / "checkpoint-1"
    assert "feature_step" not in _record(checkpoint)["policy"]

    output = tmp_path / "resumed"
    resumed = _callback(output, feature_step=2, **settings)
    resumed.resume_checkpoint = str(checkpoint)
    args, state = _start(resumed, _model(), output, step=1)
    assert (
        "training.lora_layers: activation change requires a probe from the start of the run"
        in caplog.text
    )
    _step(resumed, args, state, lambda idx, step: 4.0)
    resumed.on_save(args, state)
    features = _record(output / "checkpoint-2")["features"]
    assert features["layers"]["0"] == {"grad_mean": 3.0, "grad_cv": 0.333333}

    final_output = tmp_path / "final"
    final = _callback(final_output, feature_step=2, **settings)
    final.resume_checkpoint = str(output / "checkpoint-2")
    args, state = _start(final, _model(), final_output, step=2)
    _step(final, args, state, lambda idx, step: 8.0)
    final.on_train_end(args, state)
    assert _record(final_output)["features"] == features


@pytest.mark.parametrize(("checkpoint_step", "skipped"), [(5, False), (5, True), (7, False)])
def test_resume_restores_rule_history_and_partial_window(tmp_path, checkpoint_step, skipped):
    settings = {"interval_steps": 2, "freeze_smoothing": 0.8}
    cb = _callback(tmp_path, **settings)
    args, state = _start(cb, _model(), tmp_path)
    for step in range(1, checkpoint_step + 1):
        optimizer = SimpleNamespace(step_was_skipped=(step == 5 and skipped))
        _step(cb, args, state, lambda idx, step: 1.0, optimizer)
    cb.on_save(args, state)

    resumed = _callback(tmp_path / "resumed", **settings)
    resumed.resume_checkpoint = str(tmp_path / f"checkpoint-{checkpoint_step}")
    resume_args, resume_state = _start(
        resumed, _model(), tmp_path / "resumed", step=checkpoint_step
    )
    assert resumed.selection.layers == cb.selection.layers
    for _ in range(checkpoint_step, 8):
        _step(cb, args, state, lambda idx, step: 1.0)
        _step(resumed, resume_args, resume_state, lambda idx, step: 1.0)
    cb.on_train_end(args, state)
    resumed.on_train_end(resume_args, resume_state)
    assert _record(tmp_path) == _record(tmp_path / "resumed")
    assert cb.selection.history[-1] == (8 if skipped else 6, [3, 4, 5])


@pytest.mark.parametrize("smoothing, freeze_step", [(0.0, 6), (0.8, 8)])
@pytest.mark.parametrize(
    ("checkpoint_step", "legacy_empty_entry"), [(3, False), (4, False), (4, True), (5, False)]
)
def test_empty_windows_do_not_advance_rule_across_resume(
    tmp_path, smoothing, freeze_step, checkpoint_step, legacy_empty_entry
):
    settings = {"interval_steps": 2, "freeze_smoothing": smoothing}
    cb = _callback(tmp_path, **settings)
    args, state = _start(cb, _model(), tmp_path)

    def norms(_idx, step):
        if step in (3, 4, 5):
            return None
        return 2.0 if step == 6 else 1.0

    for _ in range(checkpoint_step):
        _step(cb, args, state, norms)
    cb.on_save(args, state)
    checkpoint = tmp_path / f"checkpoint-{checkpoint_step}"
    record = _record(checkpoint)
    assert record["pending_norms"] == {}
    if legacy_empty_entry:
        record["norms"] = [record["norms"][0], {"step": 4, "layers": {}}]
        record["skipped_windows"] = []
        (checkpoint / "lora_layers.json").write_text(json.dumps(record), encoding="utf-8")

    resumed = _callback(tmp_path / "resumed", **settings)
    resumed.resume_checkpoint = str(checkpoint)
    resume_args, resume_state = _start(
        resumed, _model(), tmp_path / "resumed", step=checkpoint_step
    )
    for step in range(checkpoint_step + 1, freeze_step + 1):
        _step(cb, args, state, norms)
        _step(resumed, resume_args, resume_state, norms)
        if step < freeze_step:
            assert resumed.selection.layers == list(range(6))
    cb.on_train_end(args, state)
    resumed.on_train_end(resume_args, resume_state)
    record = _record(tmp_path / "resumed")
    assert record["history"] == [
        {"step": 0, "layers": list(range(6))},
        {"step": freeze_step, "layers": [3, 4, 5]},
    ]
    assert record["skipped_windows"] == [4]
    assert [entry["step"] for entry in record["norms"]] == [2, *range(6, freeze_step + 1, 2)]
    assert record == _record(tmp_path)


@pytest.mark.parametrize(
    "settings",
    [{"freeze_step": 1, "freeze_below": 6}, {"freeze_deadline": 1, "min_freeze_below": 6}],
)
def test_cut_that_would_freeze_every_adapter_fails_before_training(tmp_path, settings):
    cb = _callback(tmp_path, **settings)
    with pytest.raises(ValueError) as exc:
        _start(cb, _model(), tmp_path)
    assert str(exc.value) == "training.lora_layers: freezing below 6 leaves no adapted layer"


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"k": 2}, "method='progressive' takes neither k nor layers"),
        ({"layers": [0]}, "method='progressive' takes neither k nor layers"),
        ({"freeze_step": 1}, "freeze_step and freeze_below go together"),
        ({"min_freeze_below": 2}, "freeze_deadline and min_freeze_below go together"),
        (
            {"freeze_deadline": 5, "min_freeze_below": 3, "max_freeze_below": 2},
            "min_freeze_below must be at most max_freeze_below",
        ),
        (
            {"freeze_step": 1, "freeze_below": 2, "freeze_threshold": 0.2},
            "freeze_threshold only applies to the progressive rule",
        ),
        (
            {"method": "fixed", "layers": [1], "freeze_smoothing": 0.0},
            "freeze_smoothing only applies to method 'progressive'",
        ),
        (
            {"method": "lisa", "k": 2, "freeze_threshold": 0.10},
            "freeze_threshold only applies to method 'progressive'",
        ),
    ],
)
def test_progressive_config_guards(block, message):
    from pydantic import ValidationError

    from soup_cli.config.schema import LoraLayersConfig

    with pytest.raises(ValidationError) as exc:
        LoraLayersConfig.model_validate({"method": "progressive", **block})
    assert exc.value.errors()[0]["msg"] == f"Value error, training.lora_layers: {message}"


def test_trainer_keeps_frozen_adapter_weights_at_their_last_update(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from transformers import Trainer, TrainingArguments

    from soup_cli.config.schema import LoraLayersConfig
    from soup_cli.utils.layer_select import attach_layer_select_callback

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = _model().layers

        def forward(self, labels):
            weight = torch.stack([layer.lora_weight for layer in self.layers]).sum()
            return {"loss": (weight - labels.float().mean()).square()}

    model = Model()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=3,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=1,
            learning_rate=0.01,
            save_steps=1,
            report_to=[],
        ),
        train_dataset=[{"labels": 0.0}] * 6,
    )
    cfg = LoraLayersConfig(method="progressive", freeze_step=1, freeze_below=2, interval_steps=1)
    attach_layer_select_callback(trainer, SimpleNamespace(lora_layers=cfg), str(tmp_path))
    trainer.train()

    checkpoint = tmp_path / "checkpoint-1"
    first_step = load_file(str(checkpoint / "model.safetensors"))
    for idx, layer in enumerate(model.layers):
        assert layer.lora_weight.item() < 1.0
        assert layer.lora_weight.equal(first_step[f"layers.{idx}.lora_weight"]) == (idx < 2)
        assert (layer.lora_weight in trainer.optimizer.state) == (idx >= 2)
    assert _record(checkpoint)["layers"] == [2, 3, 4, 5]
    assert [entry["step"] for entry in _record(tmp_path)["norms"]] == [1, 2, 3]
