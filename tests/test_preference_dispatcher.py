"""Tests for v0.40.0 Part B — unified preference loss dispatcher.

Adds ``task: preference`` + ``training.preference_loss`` Literal
(dpo/simpo/orpo/ipo/bco). Existing per-task forms (``task: dpo`` etc) keep
working unchanged — the unified surface is *additive*, not a breaking
collapse. Backward-compat ``resolve_preference_loss`` maps legacy task
strings to their preference_loss equivalents for callers that want a
single dispatch entry-point.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


class TestPreferenceStreaming:
    @staticmethod
    def config(loss=None, *, task="preference", **training):
        return SoupConfig(
            base="some-model",
            task=task,
            data={"train": "./data.jsonl", "format": "dpo"},
            training={
                "stream_layers": True,
                "quantization": "none",
                "batch_size": 1,
                **({"preference_loss": loss} if task == "preference" else {}),
                **training,
            },
        )

    @pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo", "ipo"])
    @pytest.mark.parametrize(
        "training",
        [{}, {"quantization": "4bit"}, {"batch_size": 2, "gradient_accumulation_steps": 2}],
    )
    def test_scalar_streaming_is_accepted(self, loss, training):
        cfg = self.config(loss, **training)
        assert cfg.task == "preference"
        assert cfg.training.preference_loss == loss
        assert cfg.training.stream_layers is True

    @pytest.mark.parametrize("loss", ["bco"])
    def test_unported_loss_is_refused_by_name(self, loss):
        with pytest.raises(ValidationError) as caught:
            self.config(loss)
        assert caught.value.errors()[0]["msg"] == (
            f"Value error, training.stream_layers cannot use "
            f"training.preference_loss={loss!r}: this loss does not stream yet. "
            "Use dpo, orpo, simpo or ipo, or set stream_layers=false."
        )

    @pytest.mark.parametrize("task", ["bco"])
    def test_legacy_unported_tasks_keep_the_generic_streaming_refusal(self, task):
        with pytest.raises(ValidationError) as caught:
            self.config(task=task)
        message = caught.value.errors()[0]["msg"]
        assert message == (
            "Value error, training.stream_layers supports task in "
            f"['dpo', 'ipo', 'kto', 'orpo', 'sft', 'simpo']; got task={task!r}."
        )
        assert "preference_loss" not in message

    def test_weighted_losses_are_refused_by_name(self):
        with pytest.raises(ValidationError) as caught:
            self.config(preference_loss_weights={"dpo": 0.5, "orpo": 0.5})
        assert caught.value.errors()[0]["msg"] == (
            "Value error, training.stream_layers cannot be combined with "
            "training.preference_loss_weights: weighted preference losses do not "
            "stream yet. Use a single training.preference_loss of dpo, orpo, "
            "simpo or ipo, or set stream_layers=false."
        )

    @pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo", "ipo"])
    @pytest.mark.parametrize(
        "training,refusal",
        [
            ({"stream_vram_probe": True}, "training.stream_vram_probe supports task='sft'"),
            ({"batch_size": "auto"}, "training.stream_layers cannot use batch_size='auto'"),
            ({"stream_loss_chunk": 16}, "training.stream_loss_chunk supports task='sft'"),
            ({"stream_recompute": "off"}, "training.stream_recompute='off' supports task='sft'"),
            ({"stream_recompute": "auto"}, "training.stream_recompute='auto'"),
            ({"quantization": "8bit"}, "training.stream_layers supports quantization"),
            ({"lora": {"r": 0}}, "training.stream_layers requires LoRA"),
            ({"lora": {"use_dora": True}}, "training.stream_layers is incompatible"),
        ],
    )
    def test_legacy_refusals_keep_the_same_messages(self, loss, training, refusal):
        with pytest.raises(ValidationError) as legacy:
            self.config(task=loss, **training)
        with pytest.raises(ValidationError) as dispatched:
            self.config(loss, **training)
        legacy_message = legacy.value.errors()[0]["msg"]
        assert refusal in legacy_message
        assert dispatched.value.errors()[0]["msg"] == legacy_message

    @pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo", "ipo"])
    def test_real_dispatcher_trains_a_streamed_cpu_model(self, tmp_path, monkeypatch, loss):
        import math
        import socket

        from safetensors.torch import load_file

        from soup_cli.trainer.preference import PreferenceTrainerWrapper
        from tests.test_v07204 import (
            _TASK_ROWS,
            _stream_cfg,
            _tiny_llama_dir,
            _wrapper_for,
            _write_tiny_tokenizer,
        )

        def refuse_network(*args, **kwargs):
            pytest.fail("The local streamed fixture must not use DNS or the network")

        monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
        monkeypatch.setattr(socket.socket, "connect", refuse_network)
        weights, _, _ = _tiny_llama_dir(tmp_path)
        _write_tiny_tokenizer(weights)
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.chdir(tmp_path)
        cfg = _stream_cfg(
            weights, tmp_path / "out", task="preference", preference_loss=loss,
            warmup_ratio=0.0,
        )
        original = cfg.model_dump()
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu", fsdp_config={"use_cpu": True})
        try:
            wrapper.setup({"train": _TASK_ROWS[loss](8)})
            inner = wrapper._inner
            assert isinstance(inner, _wrapper_for(loss))
            assert inner.config.task == loss
            assert inner.config.training.preference_loss is None
            assert inner.config.training.preference_loss_weights is None
            assert inner.config.training.stream_layers is True
            assert inner.config.training.model_dump(exclude={"preference_loss"}) == (
                cfg.training.model_dump(exclude={"preference_loss"})
            )
            runtime = inner._stream_runtime
            assert runtime is not None
            assert str(wrapper.trainer.accelerator.device) == "cpu"
            assert any(parameter.is_meta for parameter in inner.model.parameters())
            if loss == "dpo":
                assert wrapper.trainer.ref_model is None
            wrapper.trainer.args.max_steps = 1
            result = wrapper.train()
            assert result["total_steps"] == 1
            losses = [
                entry["loss"] for entry in wrapper.trainer.state.log_history if "loss" in entry
            ]
            assert losses and all(math.isfinite(value) for value in losses)
            assert runtime.pool.loads > 0
            adapter = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
            assert adapter and all(tensor.numel() > 0 for tensor in adapter.values())
            assert any(
                tensor.count_nonzero().item() > 0
                for name, tensor in adapter.items() if "lora_B" in name
            )
        finally:
            if wrapper._inner is not None:
                wrapper._inner._close_stream_runtime()
        assert cfg.model_dump() == original

# ─── Schema Tests ───────────────────────────────────────────────────────────


class TestPreferenceTaskField:
    def test_preference_task_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        assert cfg.task == "preference"
        assert cfg.training.preference_loss == "dpo"

    @pytest.mark.parametrize("loss", ["dpo", "simpo", "orpo", "ipo", "bco"])
    def test_preference_loss_each_value(self, loss):
        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": loss},
        )
        assert cfg.training.preference_loss == loss

    def test_preference_loss_unknown_rejected(self):
        with pytest.raises(ValidationError, match="preference_loss"):
            SoupConfig(
                base="some-model",
                task="preference",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss": "garbage"},
            )

    def test_preference_task_requires_preference_loss(self):
        """task=preference without preference_loss must error."""
        with pytest.raises(ValidationError, match="preference_loss"):
            SoupConfig(
                base="some-model",
                task="preference",
                data={"train": "./data.jsonl", "format": "dpo"},
            )

    def test_preference_loss_default_none_for_other_tasks(self):
        """Non-preference tasks default preference_loss=None."""
        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.training.preference_loss is None

    def test_preference_loss_set_outside_preference_task_rejected(self):
        """preference_loss is meaningful only when task=preference."""
        with pytest.raises(ValidationError, match="preference_loss"):
            SoupConfig(
                base="some-model",
                task="dpo",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss": "dpo"},
            )


# ─── Resolver ──────────────────────────────────────────────────────────────


class TestResolvePreferenceLoss:
    def test_resolver_and_schema_import_without_trainers_or_heavy_dependencies(self):
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-c", """
import sys
from soup_cli.utils.preference import resolve_preference_loss
assert 'soup_cli.config.schema' not in sys.modules
from soup_cli.config.schema import SoupConfig
assert 'soup_cli.trainer.preference' not in sys.modules
assert not {'torch', 'transformers', 'peft', 'trl', 'datasets', 'mlx'} & sys.modules.keys()
from soup_cli.trainer.preference import resolve_preference_loss as legacy_resolver
assert legacy_resolver is resolve_preference_loss
"""],
            check=True, capture_output=True, text=True,
        )

    @pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo", "ipo", "bco"])
    def test_resolve_preference_task_returns_loss(self, loss):
        from soup_cli.trainer.preference import resolve_preference_loss

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": loss},
        )
        assert resolve_preference_loss(cfg) == loss

    @pytest.mark.parametrize(
        "task,expected",
        [
            ("dpo", "dpo"),
            ("simpo", "simpo"),
            ("orpo", "orpo"),
            ("ipo", "ipo"),
            ("bco", "bco"),
        ],
    )
    def test_legacy_task_maps_to_loss(self, task, expected):
        from soup_cli.trainer.preference import resolve_preference_loss

        cfg = SoupConfig(
            base="some-model",
            task=task,
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert resolve_preference_loss(cfg) == expected

    def test_resolve_non_preference_task_returns_none(self):
        from soup_cli.trainer.preference import resolve_preference_loss

        cfg = SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl"},
        )
        assert resolve_preference_loss(cfg) is None


# ─── Wrapper Tests ─────────────────────────────────────────────────────────


class TestPreferenceTrainerWrapper:
    def test_import_exists(self):
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        assert PreferenceTrainerWrapper is not None

    @pytest.mark.parametrize(
        "loss,wrapper_path",
        [
            ("dpo", "soup_cli.trainer.dpo.DPOTrainerWrapper"),
            ("simpo", "soup_cli.trainer.simpo.SimPOTrainerWrapper"),
            ("orpo", "soup_cli.trainer.orpo.ORPOTrainerWrapper"),
            ("ipo", "soup_cli.trainer.ipo.IPOTrainerWrapper"),
            ("bco", "soup_cli.trainer.bco.BCOTrainerWrapper"),
        ],
    )
    def test_dispatcher_routes_to_correct_wrapper(self, loss, wrapper_path):
        """PreferenceTrainerWrapper.setup() must delegate to the right wrapper.

        Asserts the inner cfg sent to the wrapper has task=loss and that
        preference_loss has been cleared (defends _make_inner_cfg's contract).
        """
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": loss},
        )
        with mock_patch(wrapper_path) as mock_inner:
            mock_instance = MagicMock()
            mock_instance.train.return_value = {
                "initial_loss": 1.0, "final_loss": 0.5,
                "duration": "1m", "duration_secs": 60.0,
                "output_dir": "./out", "total_steps": 10,
            }
            mock_inner.return_value = mock_instance

            wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})
            wrapper.train()

            mock_inner.assert_called_once()
            inner_cfg = mock_inner.call_args[0][0]
            assert inner_cfg.task == loss
            assert inner_cfg.training.preference_loss is None
            mock_instance.setup.assert_called_once()
            mock_instance.train.assert_called_once()

    def test_setup_does_not_mutate_caller_cfg(self):
        """_make_inner_cfg must return a copy; cfg.task unchanged after setup."""
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        with mock_patch("soup_cli.trainer.dpo.DPOTrainerWrapper") as mock_inner:
            mock_inner.return_value = MagicMock()
            wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})
        # Caller's cfg untouched.
        assert cfg.task == "preference"
        assert cfg.training.preference_loss == "dpo"

    def test_train_before_setup_raises(self):
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
        with pytest.raises(RuntimeError, match="setup"):
            wrapper.train()

    def test_unknown_loss_raises_at_setup(self):
        """Defence-in-depth: schema gate prevents this, but if it slipped, raise."""
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        # Mutate after construction to bypass schema validation.
        cfg.training.preference_loss = "garbage"
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
        with pytest.raises(ValueError, match="preference_loss"):
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})


# ─── Train Routing ─────────────────────────────────────────────────────────


class TestPreferenceTrainRouting:
    def test_sweep_routes_to_preference_wrapper(self):
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"preference_loss": "dpo"},
        )
        fake_dataset = {"train": [{"prompt": "Q", "chosen": "A", "rejected": "B"}]}
        fake_result = {
            "initial_loss": 1.0, "final_loss": 0.5,
            "duration": "1m", "duration_secs": 60.0,
            "output_dir": "./out", "total_steps": 10,
        }
        fake_gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}

        with mock_patch(
            "soup_cli.data.loader.load_dataset", return_value=fake_dataset,
        ), mock_patch(
            "soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU"),
        ), mock_patch(
            "soup_cli.utils.gpu.get_gpu_info", return_value=fake_gpu_info,
        ), mock_patch(
            "soup_cli.experiment.tracker.ExperimentTracker",
        ) as mock_tracker_cls, mock_patch(
            "soup_cli.monitoring.display.TrainingDisplay",
        ), mock_patch(
            "soup_cli.trainer.preference.PreferenceTrainerWrapper.setup",
        ), mock_patch(
            "soup_cli.trainer.preference.PreferenceTrainerWrapper.train",
            return_value=fake_result,
        ) as mock_train:
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-pref-1"
            mock_tracker_cls.return_value = mock_tracker
            result = _run_single(cfg, {}, "pref_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-pref-1"
