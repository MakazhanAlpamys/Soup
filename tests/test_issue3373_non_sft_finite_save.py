"""Regression coverage for issue #3373: shared finite-state save guard on DPO and GRPO."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from soup_cli.trainer import dpo as dpo_mod
from soup_cli.trainer import grpo as grpo_mod
from soup_cli.trainer import sft as sft_mod
from soup_cli.trainer.finite_state import assert_finite_training_state


def _make_dpo_wrapper(
    tmp_path: Path, *, log_history: list[dict], relora_steps: int | None = None
):
    wrapper = dpo_mod.DPOTrainerWrapper.__new__(dpo_mod.DPOTrainerWrapper)
    wrapper.config = SimpleNamespace(
        output=str(tmp_path / "out"),
        task="dpo",
        training=SimpleNamespace(
            dpo_beta=0.1,
            dpo_beta_end=None,
            dpo_beta_schedule="constant",
            dpo_ref_regen_epochs=None,
            relora_steps=relora_steps,
        ),
    )
    wrapper.device = "cpu"
    wrapper._output_dir = str(tmp_path / "out")
    wrapper._batch_size = 1
    wrapper.tokenizer = MagicMock()
    wrapper._training_context = lambda *_a, **_kw: nullcontext()
    wrapper._attach_streamed_save_guard = lambda: None
    wrapper._assert_streamed_adapter_saved = lambda _out: None

    state = SimpleNamespace(log_history=log_history, global_step=len(log_history))
    args = SimpleNamespace(fp16=False, bf16=False)
    trainer = MagicMock()
    trainer.state = state
    trainer.args = args
    trainer.model = None
    wrapper.trainer = trainer
    return wrapper


def _make_grpo_wrapper(
    tmp_path: Path, *, log_history: list[dict], relora_steps: int | None = None
):
    wrapper = grpo_mod.GRPOTrainerWrapper.__new__(grpo_mod.GRPOTrainerWrapper)
    wrapper.config = SimpleNamespace(
        output=str(tmp_path / "out"),
        task="grpo",
        training=SimpleNamespace(relora_steps=relora_steps),
    )
    wrapper.device = "cpu"
    wrapper._output_dir = str(tmp_path / "out")
    wrapper._batch_size = 1
    wrapper.tokenizer = MagicMock()

    state = SimpleNamespace(log_history=log_history, global_step=len(log_history))
    args = SimpleNamespace(fp16=False, bf16=False)
    trainer = MagicMock()
    trainer.state = state
    trainer.args = args
    trainer.model = None
    wrapper.trainer = trainer
    return wrapper


class TestSharedFiniteTrainingStateHelper:
    def test_finite_dpo_and_grpo_histories_accepted(self) -> None:
        assert_finite_training_state(
            [{"loss": 0.65, "grad_norm": 0.8, "train_loss": 0.65, "step": 1}]
        )
        assert_finite_training_state(
            [{"loss": 0.12, "grad_norm": 1.1, "eval_loss": 0.25, "step": 2}],
            metric_key="loss",
        )

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_final_nan_and_inf_loss_values_rejected(self, bad_value: float) -> None:
        with pytest.raises(
            RuntimeError,
            match=r"non-finite training metric loss=.*step 4.*refusing to save",
        ):
            assert_finite_training_state(
                [{"loss": 0.5, "step": 1}, {"loss": bad_value, "step": 4}]
            )

    def test_transient_nan_followed_by_finite_final_value_accepted(self) -> None:
        assert_finite_training_state(
            [
                {"loss": float("nan"), "grad_norm": float("inf"), "step": 1},
                {"loss": 0.42, "grad_norm": 0.75, "step": 5},
            ]
        )

    def test_custom_metric_key_checked_and_standard_key_deduplicated(self) -> None:
        with pytest.raises(
            RuntimeError,
            match=r"non-finite training metric custom_kl=nan.*step 3.*refusing to save",
        ):
            assert_finite_training_state(
                [{"loss": 0.5, "custom_kl": float("nan"), "step": 3}],
                metric_key="custom_kl",
            )

        # Transient non-finite custom metric recovers by final entry.
        assert_finite_training_state(
            [
                {"loss": 0.5, "custom_kl": float("nan"), "step": 1},
                {"loss": 0.4, "custom_kl": 0.02, "step": 2},
            ],
            metric_key="custom_kl",
        )

        # Standard metric key passed explicitly still raises once on final failure.
        with pytest.raises(
            RuntimeError,
            match=r"non-finite training metric loss=inf.*step 2.*refusing to save",
        ):
            assert_finite_training_state(
                [{"loss": float("inf"), "step": 2}], metric_key="loss"
            )

    def test_non_finite_trainable_parameters_rejected_and_frozen_ignored(self) -> None:
        torch = pytest.importorskip("torch")

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.frozen = torch.nn.Parameter(
                    torch.tensor([float("nan")]), requires_grad=False
                )
                self.trainable = torch.nn.Linear(2, 2, bias=False)
                self.register_buffer("buf", torch.tensor([float("inf")]))

        model = _Model()
        # Frozen NaN parameter and buffer must not fail when trainable weight is finite.
        assert_finite_training_state([], model=model)

        with torch.no_grad():
            model.trainable.weight[0, 1] = float("inf")
        with pytest.raises(
            RuntimeError,
            match=r"non-finite trainable parameter 'trainable\.weight'.*refusing to save",
        ):
            assert_finite_training_state([], model=model)

    def test_sft_compatibility_alias_remains_patchable(self, monkeypatch) -> None:
        assert sft_mod._assert_finite_training_state is assert_finite_training_state
        calls = []
        monkeypatch.setattr(
            sft_mod, "_assert_finite_training_state", lambda *a, **kw: calls.append((a, kw))
        )
        sft_mod._assert_finite_training_state([{"loss": float("nan"), "step": 1}], model=None)
        assert len(calls) == 1


class TestDpoAndGrpoFiniteSaveGuard:
    @pytest.mark.parametrize("relora_steps", [None, 10])
    def test_dpo_finite_history_calls_guard_before_save_paths(
        self, tmp_path: Path, monkeypatch, relora_steps: int | None
    ) -> None:
        events: list[str] = []
        relora_calls: list[tuple] = []

        monkeypatch.setattr(dpo_mod, "align_trainable_dtype_for_fp16", lambda *a, **kw: None)
        monkeypatch.setattr(
            dpo_mod, "manual_save_context", lambda _trainer: nullcontext()
        )
        monkeypatch.setattr(
            "soup_cli.utils.v028_features.activation_offloading_context",
            lambda *_a, **_kw: nullcontext(),
        )
        monkeypatch.setattr(
            "soup_cli.utils.dpo_variants.build_dpo_variant_callbacks",
            lambda **_kw: [],
        )
        monkeypatch.setattr(
            "soup_cli.utils.peft_wiring.save_model_with_relora",
            lambda *args: (events.append("relora_save"), relora_calls.append(args)),
        )

        def _recording_guard(log_history, model=None, *, metric_key="loss"):
            events.append("guard")
            assert_finite_training_state(log_history, model=model, metric_key=metric_key)

        monkeypatch.setattr(dpo_mod, "assert_finite_training_state", _recording_guard)

        wrapper = _make_dpo_wrapper(
            tmp_path,
            log_history=[{"loss": 0.4, "grad_norm": 0.5, "step": 2}],
            relora_steps=relora_steps,
        )
        wrapper.trainer.save_model.side_effect = lambda *_a: events.append("save_model")

        result = wrapper.train()
        assert result["final_loss"] == pytest.approx(0.4)
        expected_save = "save_model" if relora_steps is None else "relora_save"
        assert events == ["guard", expected_save]

    @pytest.mark.parametrize("relora_steps", [None, 10])
    @pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")])
    def test_dpo_aborts_before_regular_or_relora_save_on_non_finite_loss(
        self, tmp_path: Path, monkeypatch, relora_steps: int | None, bad_loss: float
    ) -> None:
        relora_calls: list[tuple] = []
        monkeypatch.setattr(dpo_mod, "align_trainable_dtype_for_fp16", lambda *a, **kw: None)
        monkeypatch.setattr(
            dpo_mod, "manual_save_context", lambda _trainer: nullcontext()
        )
        monkeypatch.setattr(
            "soup_cli.utils.v028_features.activation_offloading_context",
            lambda *_a, **_kw: nullcontext(),
        )
        monkeypatch.setattr(
            "soup_cli.utils.dpo_variants.build_dpo_variant_callbacks",
            lambda **_kw: [],
        )
        monkeypatch.setattr(
            "soup_cli.utils.peft_wiring.save_model_with_relora",
            lambda *args: relora_calls.append(args),
        )

        wrapper = _make_dpo_wrapper(
            tmp_path,
            log_history=[{"loss": 0.5, "step": 1}, {"loss": bad_loss, "step": 3}],
            relora_steps=relora_steps,
        )
        with pytest.raises(RuntimeError, match=r"non-finite training metric loss="):
            wrapper.train()

        wrapper.trainer.save_model.assert_not_called()
        assert relora_calls == []
        wrapper.tokenizer.save_pretrained.assert_not_called()

    @pytest.mark.parametrize("relora_steps", [None, 10])
    def test_grpo_finite_history_calls_guard_before_save_paths(
        self, tmp_path: Path, monkeypatch, relora_steps: int | None
    ) -> None:
        events: list[str] = []
        relora_calls: list[tuple] = []

        monkeypatch.setattr(grpo_mod, "align_trainable_dtype_for_fp16", lambda *a, **kw: None)
        monkeypatch.setattr(
            grpo_mod, "manual_save_context", lambda _trainer: nullcontext()
        )
        monkeypatch.setattr(
            "soup_cli.utils.v028_features.activation_offloading_context",
            lambda *_a, **_kw: nullcontext(),
        )
        monkeypatch.setattr(
            "soup_cli.utils.peft_wiring.save_model_with_relora",
            lambda *args: (events.append("relora_save"), relora_calls.append(args)),
        )

        def _recording_guard(log_history, model=None, *, metric_key="loss"):
            events.append("guard")
            assert_finite_training_state(log_history, model=model, metric_key=metric_key)

        monkeypatch.setattr(grpo_mod, "assert_finite_training_state", _recording_guard)

        wrapper = _make_grpo_wrapper(
            tmp_path,
            log_history=[{"loss": 0.3, "grad_norm": 0.7, "step": 2}],
            relora_steps=relora_steps,
        )
        wrapper.trainer.save_model.side_effect = lambda *_a: events.append("save_model")

        result = wrapper.train()
        assert result["final_loss"] == pytest.approx(0.3)
        expected_save = "save_model" if relora_steps is None else "relora_save"
        assert events == ["guard", expected_save]

    @pytest.mark.parametrize("relora_steps", [None, 10])
    @pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")])
    def test_grpo_aborts_before_regular_or_relora_save_on_non_finite_loss(
        self, tmp_path: Path, monkeypatch, relora_steps: int | None, bad_loss: float
    ) -> None:
        relora_calls: list[tuple] = []
        monkeypatch.setattr(grpo_mod, "align_trainable_dtype_for_fp16", lambda *a, **kw: None)
        monkeypatch.setattr(
            grpo_mod, "manual_save_context", lambda _trainer: nullcontext()
        )
        monkeypatch.setattr(
            "soup_cli.utils.v028_features.activation_offloading_context",
            lambda *_a, **_kw: nullcontext(),
        )
        monkeypatch.setattr(
            "soup_cli.utils.peft_wiring.save_model_with_relora",
            lambda *args: relora_calls.append(args),
        )

        wrapper = _make_grpo_wrapper(
            tmp_path,
            log_history=[{"loss": 0.5, "step": 1}, {"loss": bad_loss, "step": 3}],
            relora_steps=relora_steps,
        )
        with pytest.raises(RuntimeError, match=r"non-finite training metric loss="):
            wrapper.train()

        wrapper.trainer.save_model.assert_not_called()
        assert relora_calls == []
        wrapper.tokenizer.save_pretrained.assert_not_called()
        wrapper.trainer.save_state.assert_not_called()

    @pytest.mark.parametrize(
        "trainer_cls",
        [dpo_mod.DPOTrainerWrapper, grpo_mod.GRPOTrainerWrapper],
    )
    def test_guard_precedes_regular_and_relora_save_in_source(
        self, trainer_cls: type
    ) -> None:
        source = inspect.getsource(trainer_cls.train)
        guard_idx = source.index("assert_finite_training_state")
        assert guard_idx < source.index("self.trainer.save_model")
        assert guard_idx < source.index("save_model_with_relora")
