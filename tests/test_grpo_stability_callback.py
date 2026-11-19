"""#2601: the GRPO gradient watchdog named every parameter when one gradient was NaN.

``GRPOStabilityCallback.on_pre_optimizer_step`` (#342) scans ``p.grad`` and records
the names of the parameters whose gradient is not finite. The Hugging Face
``Trainer`` calls that hook after ``_clip_grad_norm``. With ``max_grad_norm: 0`` it
still calls ``clip_grad_norm_(params, inf)`` in ``_get_grad_norm`` to log the norm.
Either way a non-finite total norm makes the clip coefficient NaN, and every
gradient is multiplied by it, so the scan found every trainable parameter. The
step was skipped correctly, but the names could not point at the cause.

The callback now records, from a ``register_post_accumulate_grad_hook`` on each
trainable parameter, whether its accumulated gradient was finite before the clip,
and names those parameters. When no such record exists, the names are the ones
read after the clip, and the record says so (``nan_param_names_source``).

The models are three ``nn.Linear`` layers on the CPU. A tensor hook on
``b.weight`` makes its gradient non-finite in one chosen backward.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class _ThreeLayers(nn.Module):
    """``a -> b -> c``. ``poison`` maps a forward number (from 1) to parameter
    names and factors; each of those gradients is multiplied by its factor in
    the backward of that forward."""

    def __init__(self, poison: dict[int, dict[str, float]]):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 1)
        self.forwards = 0
        for name, param in self.named_parameters():
            param.register_hook(self._poisoner(name))
        self._poison = poison

    def _poisoner(self, name: str):
        def hook(grad):
            factor = self._poison.get(self.forwards, {}).get(name)
            return grad if factor is None else grad * factor

        return hook

    def forward(self, x, labels=None):
        self.forwards += 1
        y = self.c(self.b(self.a(x))).squeeze(-1)
        return {"loss": ((y - labels) ** 2).mean()}


class _Rows(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"x": torch.full((4,), float(i + 1)), "labels": torch.tensor(float(i))}


def _train(tmp_path, model, *, max_grad_norm, accumulation=1, max_steps=1):
    from transformers import Trainer, TrainingArguments

    from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

    callback = GRPOStabilityCallback()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        learning_rate=0.1,
        use_cpu=True,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=_Rows(), callbacks=[callback])
    trainer.train()
    return trainer.state.log_history, callback


def _skips(log_history):
    return [e for e in log_history if "nan_skip_step" in e]


def _summary(log_history):
    return [e for e in log_history if "nan_skip_count" in e][-1]


class TestNamesComeFromBeforeTheClip:
    @pytest.mark.parametrize(
        ("factor", "max_grad_norm", "accumulation", "poisoned_forward"),
        [
            pytest.param(float("nan"), 1.0, 1, 1, id="nan-clip"),
            pytest.param(float("nan"), 0.0, 1, 1, id="nan-no-clip"),
            pytest.param(float("inf"), 0.0, 1, 1, id="inf-no-clip"),
            pytest.param(float("inf"), 1.0, 1, 1, id="inf-clip"),
            pytest.param(float("nan"), 1.0, 2, 1, id="nan-first-of-two-microbatches"),
            pytest.param(float("nan"), 1.0, 2, 2, id="nan-last-of-two-microbatches"),
        ],
    )
    def test_only_the_poisoned_parameter_is_named(
        self, tmp_path, factor, max_grad_norm, accumulation, poisoned_forward
    ):
        model = _ThreeLayers({poisoned_forward: {"b.weight": factor}})
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        history, _ = _train(
            tmp_path, model, max_grad_norm=max_grad_norm, accumulation=accumulation
        )

        (skip,) = _skips(history)
        assert skip["nan_param_names"] == ["b.weight"]
        assert skip["nan_param_count"] == 1
        assert skip["nan_param_names_source"] == "before_clip"
        summary = _summary(history)
        assert summary["nan_param_names"] == ["b.weight"]
        assert summary["nan_param_names_source"] == "before_clip"
        # The step is still skipped: no weight moved.
        for name, param in model.named_parameters():
            assert torch.equal(param.detach(), before[name]), name

    def test_the_warning_names_the_poisoned_parameter(self, tmp_path, caplog):
        model = _ThreeLayers({1: {"b.weight": float("nan")}})

        with caplog.at_level(logging.WARNING):
            _train(tmp_path, model, max_grad_norm=1.0)

        assert "non-finite gradients detected in 1 parameter(s) [b.weight]" in caplog.text
        assert "after clipping" not in caplog.text

    def test_a_later_step_names_only_its_own_parameter(self, tmp_path):
        model = _ThreeLayers({1: {"b.weight": float("nan")}, 3: {"c.bias": float("nan")}})

        history, _ = _train(tmp_path, model, max_grad_norm=1.0, max_steps=3)

        assert [(s["nan_skip_step"], s["nan_param_names"]) for s in _skips(history)] == [
            (1, ["b.weight"]),
            (3, ["c.bias"]),
        ]
        assert _summary(history)["nan_param_names"] == ["b.weight", "c.bias"]

    def test_two_parameters_in_one_step_keep_their_order(self, tmp_path):
        model = _ThreeLayers({1: {"c.bias": float("nan"), "b.weight": float("inf")}})

        history, _ = _train(tmp_path, model, max_grad_norm=1.0)

        (skip,) = _skips(history)
        assert skip["nan_param_names"] == ["b.weight", "c.bias"]

    def test_frozen_parameters_get_no_hook(self, tmp_path):
        model = _ThreeLayers({1: {"b.weight": float("nan")}})
        model.a.requires_grad_(False)

        history, _ = _train(tmp_path, model, max_grad_norm=1.0)

        (skip,) = _skips(history)
        assert skip["nan_param_names"] == ["b.weight"]
        assert skip["nan_param_names_source"] == "before_clip"

    def test_a_clean_run_records_no_skip(self, tmp_path):
        history, _ = _train(tmp_path, _ThreeLayers({}), max_grad_norm=1.0, max_steps=2)

        assert _skips(history) == []
        assert _summary(history) == {"nan_skip_count": 0, "nan_skip_fraction": 0.0}

    def test_the_hooks_are_removed_when_training_ends(self, tmp_path):
        model = _ThreeLayers({})
        _, callback = _train(tmp_path, model, max_grad_norm=1.0)

        callback._grad_finite.clear()
        model(torch.ones(4), labels=torch.tensor(0.0))["loss"].backward()

        assert callback._grad_finite == {}


def _state(global_step=4, log_history=None):
    return SimpleNamespace(global_step=global_step, log_history=log_history or [])


def _poisoned_module():
    """A real module whose gradients were made non-finite with no hook registered,
    as on a path where ``on_train_begin`` never ran."""
    model = _ThreeLayers({})
    model(torch.ones(4), labels=torch.tensor(0.0))["loss"].backward()
    model.b.weight.grad.fill_(float("nan"))
    model.c.bias.grad.fill_(float("nan"))
    return model


class TestWithoutARecordFromBeforeTheClip:
    def test_the_names_read_after_the_clip_are_labelled(self, caplog):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        callback = GRPOStabilityCallback()
        state = _state()
        model = _poisoned_module()

        with caplog.at_level(logging.WARNING):
            callback.on_pre_optimizer_step(None, state, SimpleNamespace(), model=model)

        (skip,) = _skips(state.log_history)
        assert skip["nan_param_names"] == ["b.weight", "c.bias"]
        assert skip["nan_param_names_source"] == "after_clip"
        assert "read after clipping" in caplog.text
        assert model.b.weight.grad is None

    def test_one_step_read_after_the_clip_labels_the_summary(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        callback = GRPOStabilityCallback()
        state = _state(global_step=0)
        callback.on_train_begin(None, state, SimpleNamespace(), model=_ThreeLayers({}))
        # A step whose names came from before the clip...
        callback._grad_finite = {"b.weight": torch.tensor(False)}
        callback.on_pre_optimizer_step(None, state, SimpleNamespace(), model=_poisoned_module())
        # ...and one without a record.
        callback.on_pre_optimizer_step(None, state, SimpleNamespace(), model=_poisoned_module())
        callback.on_train_end(None, state, SimpleNamespace())

        sources = [s["nan_param_names_source"] for s in _skips(state.log_history)]
        assert sources == ["before_clip", "after_clip"]
        assert _summary(state.log_history)["nan_param_names_source"] == "after_clip"

    def test_resume_restores_the_source(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        callback = GRPOStabilityCallback()
        state = _state(
            global_step=20,
            log_history=[
                {
                    "nan_skip_step": 10,
                    "nan_param_names": ["b.weight"],
                    "nan_param_names_source": "after_clip",
                },
            ],
        )
        callback.on_train_begin(None, state, SimpleNamespace())
        callback.on_train_end(None, state, SimpleNamespace())

        summary = _summary(state.log_history)
        assert summary["nan_param_names"] == ["b.weight"]
        assert summary["nan_param_names_source"] == "after_clip"
