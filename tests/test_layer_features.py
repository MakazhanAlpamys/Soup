"""Opt-in recording of progressive LoRA training signals."""

import json
import random
from types import SimpleNamespace

import pytest

from tests.test_layer_select import _cfg, _model


def _record(path):
    return json.loads((path / "lora_layers.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (
            {"method": "fixed", "layers": [0], "feature_step": 2},
            "feature_step only applies to method 'progressive'",
        ),
        (
            {"method": "random", "k": 1, "feature_rows": 2},
            "feature_rows only applies to method 'progressive'",
        ),
        ({"feature_rows": 16}, "feature_rows requires feature_step"),
    ],
)
def test_feature_config_requires_progressive_recording(block, message):
    from pydantic import ValidationError

    from soup_cli.config.schema import LoraLayersConfig

    with pytest.raises(ValidationError) as error:
        LoraLayersConfig.model_validate({"method": "progressive", **block})
    assert error.value.errors()[0]["msg"] == f"Value error, training.lora_layers: {message}"


def test_feature_config_accepts_a_scheduled_freeze():
    cfg = _cfg(lora_layers={"method": "progressive", "freeze_step": 2, "freeze_below": 1,
                           "feature_step": 2, "feature_rows": 3})
    assert cfg.training.lora_layers.feature_step == 2
    assert cfg.training.lora_layers.feature_rows == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_step", 0, "Input should be greater than or equal to 1"),
        ("feature_rows", 0, "Input should be greater than or equal to 1"),
        ("feature_rows", 1025, "Input should be less than or equal to 1024"),
    ],
)
def test_feature_config_enforces_recording_bounds(field, value, message):
    from pydantic import ValidationError

    from soup_cli.config.schema import LoraLayersConfig

    with pytest.raises(ValidationError) as error:
        LoraLayersConfig.model_validate({"method": "progressive", "feature_step": 2}
                                        | {field: value})
    assert [(item["loc"], item["msg"]) for item in error.value.errors()] == [((field,), message)]


def test_update_features_match_dense_products():
    torch = pytest.importorskip("torch")

    from soup_cli.utils.layer_features import adapter_update_features

    model = _model()
    layers = model.get_base_model().model.layers
    for name, param in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.normal_(0, 0.1)
    actual = adapter_update_features(dict(enumerate(layers)))
    assert set(actual) == set(range(len(layers)))
    for idx, layer in enumerate(layers):
        updates, ranks = [], []
        for module in layer.modules():
            if not hasattr(module, "lora_A"):
                continue
            delta = module.lora_B["default"].weight @ module.lora_A["default"].weight
            singular = torch.linalg.svdvals(delta.detach() * module.scaling["default"])
            squared = singular.square()
            updates.append(float(squared.sum().sqrt()))
            ranks.append(float(squared.sum().square() / squared.square().sum()))
        assert actual[idx]["update"] == pytest.approx(sum(updates), rel=1e-5)
        assert actual[idx]["update_rank"] == pytest.approx(sum(ranks) / len(ranks), rel=1e-5)


@pytest.mark.parametrize("shape", [(40, 8), (8, 40)])
def test_linear_cka_matches_the_centered_dense_definition(shape):
    torch = pytest.importorskip("torch")

    from soup_cli.utils.layer_features import linear_cka

    rng = torch.Generator().manual_seed(7)
    before = torch.randn(shape, generator=rng)
    after = torch.randn(shape, generator=rng)
    left, right = before - before.mean(0), after - after.mean(0)
    expected = (left.T @ right).norm().square() / (
        (left.T @ left).norm() * (right.T @ right).norm()
    )
    assert linear_cka(before, after) == pytest.approx(float(expected), abs=1e-6)
    assert linear_cka(before, 3 * before + 1) == pytest.approx(1.0, abs=1e-6)


def test_loss_summary_handles_zero_and_nonfinite_logged_losses():
    from soup_cli.utils.layer_features import loss_summary

    logs = [{"loss": value} for value in (0.0, float("nan"), 1.0, 2.0, 3.0)]
    logs.append({"eval_loss": 5.0})
    assert loss_summary(logs) == {
        "first_quarter": 0.0, "last_quarter": 3.0, "drop": None, "steps": 4,
    }


@pytest.mark.parametrize("capture_baseline", [True, False])
def test_probe_preserves_left_padding_and_keyword_hidden_states(capture_baseline):
    torch = pytest.importorskip("torch")

    from soup_cli.utils.layer_features import LayerProbe

    class Block(torch.nn.Module):
        def forward(self, *, hidden_states):
            return (hidden_states.flip(-1),)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            assert input_ids.tolist() == [[0, 0, 4, 7]]
            assert attention_mask.tolist() == [[0, 0, 1, 1]]
            assert position_ids.tolist() == [[0, 0, 0, 1]]
            assert not use_cache
            horizontal = attention_mask * input_ids
            vertical = (1 - attention_mask) * torch.ones_like(input_ids)
            return self.layers[0](hidden_states=torch.stack((horizontal, vertical), dim=-1))

    model = Model()
    probe = LayerProbe(dict(enumerate(model.layers)))
    batch = {
        "input_ids": torch.tensor([[0, 0, 4, 7]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1]]),
        "position_ids": torch.tensor([[0, 0, 0, 1]]),
    }
    probe.start(model, [batch], "cpu", capture_baseline=capture_baseline)
    features = probe.finish(model, "cpu")
    expected = {"turn": 0.5}
    if capture_baseline:
        assert probe.before[0].tolist() == [[0.0, 4.0], [0.0, 7.0]]
        expected.update({"cka_change": 0.0, "cka_own": 0.0})
    assert features[0] == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["llama", "qwen3_5"])
def test_features_are_recorded_without_changing_training(tmp_path, kind):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from soup_cli.utils.layer_select import attach_layer_select_callback

    rows = [{"input_ids": list(range(i, i + 10))} for i in range(8)]

    def collate(items):
        # Extra probe collation must leave both training RNGs at their original state.
        offset = random.randrange(4) + int(torch.randint(4, (1,)).item())
        ids = torch.tensor([item["input_ids"] for item in items]) + offset
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}

    def run(output, **feature):
        model = _model(kind)
        cfg = _cfg(lora_layers={"method": "progressive", "freeze_step": 6, "freeze_below": 2,
                                "interval_steps": 2, **feature})
        trainer = transformers.Trainer(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=str(output),
                per_device_train_batch_size=1,
                max_steps=8,
                learning_rate=1e-2,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
                seed=0,
                data_seed=0,
                disable_tqdm=True,
            ),
            train_dataset=rows,
            data_collator=collate,
        )
        attach_layer_select_callback(trainer, cfg.training, str(output))
        trainer.train()
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        params = {name: value.detach().clone() for name, value in model.named_parameters()
                  if "lora_" in name}
        return losses, params

    plain_losses, plain_params = run(tmp_path / "plain")
    losses, params = run(tmp_path / "record", feature_step=6, feature_rows=3)
    assert plain_losses == losses
    assert params.keys() == plain_params.keys()
    assert all(torch.equal(value, plain_params[name]) for name, value in params.items())
    plain_record = _record(tmp_path / "plain")
    record = _record(tmp_path / "record")
    assert "features" not in plain_record
    features = record.pop("features")
    assert record == plain_record
    assert features["step"] == 6
    assert features["loss"]["steps"] == 5
    assert features["n_layers"] == (6 if kind == "llama" else 4)
    assert set(features["layers"]) == {str(idx) for idx in range(features["n_layers"])}
    for scores in features["layers"].values():
        assert set(scores) == {
            "update", "update_rank", "grad_mean", "grad_cv", "cka_change", "cka_own", "turn",
        }
        for key in ("cka_change", "cka_own", "turn"):
            assert 0.0 <= scores[key] <= 1.0


def test_failed_probe_restores_modes_rng_and_hooks(tmp_path, caplog):
    torch = pytest.importorskip("torch")

    from soup_cli.utils.layer_select import LayerSelectPolicy, _LayerSelectCallback

    class BrokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].lora_weight = torch.nn.Parameter(torch.ones(1))
            self.layers[0].eval()

        def forward(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert not self.training
            random.random()
            torch.rand(1)
            raise RuntimeError("probe failed on purpose")

    def collate(items):
        random.random()
        torch.rand(1)
        return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

    model = BrokenModel()
    callback = _LayerSelectCallback(
        LayerSelectPolicy("progressive", feature_step=1),
        dataset=[{}],
        collator=collate,
        batch_size=1,
        output_dir=str(tmp_path),
    )
    python_rng = random.getstate()
    torch_rng = torch.random.get_rng_state()
    args = SimpleNamespace(device="cpu")
    callback.on_train_begin(args, SimpleNamespace(global_step=0), model=model)
    assert model.training and not model.layers[0].training
    assert not model.layers[0]._forward_hooks
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert "training.lora_layers: the feature probe failed: probe failed on purpose" in caplog.text
    callback.on_step_end(args, SimpleNamespace(global_step=1), model=model)
    assert _record(tmp_path)["features"] == {"step": 1, "n_layers": 1, "layers": {}, "loss": None}
