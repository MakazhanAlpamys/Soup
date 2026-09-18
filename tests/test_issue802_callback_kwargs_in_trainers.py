"""Tests for Issue #802: Unify SoupTrainerCallback kwargs across all trainers
and reject unsupported monitoring on tasks without live callbacks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.monitoring.callback import SoupTrainerCallback, soup_callback_kwargs

TRAINER_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer"

EXPECTED_CALLBACK_TRAINERS = {
    "asr.py",
    "bco.py",
    "classifier.py",
    "distill.py",
    "dpo.py",
    "embedding.py",
    "grpo.py",
    "ipo.py",
    "kto.py",
    "online_dpo.py",
    "orpo.py",
    "ppo.py",
    "pretrain.py",
    "reward_model.py",
    "sft.py",
    "simpo.py",
}

UNSUPPORTED_CALLBACK_TRAINERS = {
    "mole_routing.py",
    "prm.py",
    "unlearn.py",
}


# ===========================================================================
# 1. AST and Codebase Invariants Scan
# ===========================================================================


class TestTrainerCallbackKwargsUsage:
    """Verify that all 16 trainers instantiate SoupTrainerCallback using
    **soup_callback_kwargs(...) and track self._batch_size.
    """

    def test_all_expected_trainers_exist(self) -> None:
        for filename in EXPECTED_CALLBACK_TRAINERS | UNSUPPORTED_CALLBACK_TRAINERS:
            filepath = TRAINER_DIR / filename
            assert filepath.is_file(), f"Trainer file {filename} does not exist in {TRAINER_DIR}"

    def test_ast_scan_trainer_callback_kwargs_unification(self) -> None:
        discovered_callback_trainers: set[str] = set()

        for py_file in sorted(TRAINER_DIR.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            if "SoupTrainerCallback(" in content:
                discovered_callback_trainers.add(py_file.name)

                # Must import soup_callback_kwargs
                assert "soup_callback_kwargs" in content, (
                    f"{py_file.name} references SoupTrainerCallback but lacks "
                    "soup_callback_kwargs import"
                )

                # Must pass **soup_callback_kwargs(...)
                assert "**soup_callback_kwargs(" in content, (
                    f"{py_file.name} does not pass **soup_callback_kwargs(...) to "
                    "SoupTrainerCallback"
                )

                # Must record self._batch_size in setup() or class
                assert "self._batch_size" in content, (
                    f"{py_file.name} does not track self._batch_size"
                )

        assert discovered_callback_trainers == EXPECTED_CALLBACK_TRAINERS, (
            f"Expected {EXPECTED_CALLBACK_TRAINERS}, but found {discovered_callback_trainers}"
        )

    def test_unsupported_trainers_do_not_instantiate_callback(self) -> None:
        for filename in UNSUPPORTED_CALLBACK_TRAINERS:
            content = (TRAINER_DIR / filename).read_text(encoding="utf-8")
            tree = ast.parse(content, filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_id = None
                    if isinstance(node.func, ast.Name):
                        func_id = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_id = node.func.attr
                    assert func_id != "SoupTrainerCallback", (
                        f"{filename} unexpectedly instantiates SoupTrainerCallback directly"
                    )


# ===========================================================================
# 2. soup_callback_kwargs Helper Unit Tests
# ===========================================================================


class TestSoupCallbackKwargsHelper:
    """Unit tests for soup_callback_kwargs parameter extraction."""

    def test_default_values(self) -> None:
        tcfg = TrainingConfig()
        kwargs = soup_callback_kwargs(tcfg)

        assert kwargs["loss_watchdog"] is False
        assert kwargs["loss_watchdog_threshold"] == 3.0
        assert kwargs["loss_watchdog_patience"] == 5
        assert kwargs["spike_recovery"] is False
        assert kwargs["spike_recovery_max_attempts"] == 3
        assert kwargs["spike_recovery_lr_decay"] == 0.5
        assert kwargs["grad_accum_auto_tune"] is False
        assert kwargs["grad_accum_pressure_threshold"] == pytest.approx(0.92)
        assert kwargs["grad_accum_current_steps"] == 4
        assert kwargs["grad_accum_current_batch"] == 1
        assert kwargs["eval_gate_config"] is None
        assert "output_dir" not in kwargs

        # When tcfg has no gradient_accumulation_steps attribute
        kwargs_bare = soup_callback_kwargs(SimpleNamespace())
        assert kwargs_bare["grad_accum_current_steps"] == 1

    def test_output_dir_propagation(self) -> None:
        tcfg = TrainingConfig()
        kwargs = soup_callback_kwargs(tcfg, output_dir="/models/output")
        assert kwargs["output_dir"] == "/models/output"

    def test_custom_values_mapping(self) -> None:
        tcfg = TrainingConfig(
            loss_watchdog=True,
            loss_watchdog_threshold=4.2,
            loss_watchdog_patience=10,
            loss_spike_recovery=True,
            loss_spike_recovery_max_attempts=6,
            loss_spike_recovery_lr_decay=0.25,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.82,
            gradient_accumulation_steps=4,
        )
        kwargs = soup_callback_kwargs(tcfg, batch_size=8, output_dir="/checkpoints/run1")

        assert kwargs["loss_watchdog"] is True
        assert kwargs["loss_watchdog_threshold"] == pytest.approx(4.2)
        assert kwargs["loss_watchdog_patience"] == 10
        assert kwargs["spike_recovery"] is True
        assert kwargs["spike_recovery_max_attempts"] == 6
        assert kwargs["spike_recovery_lr_decay"] == pytest.approx(0.25)
        assert kwargs["grad_accum_auto_tune"] is True
        assert kwargs["grad_accum_pressure_threshold"] == pytest.approx(0.82)
        assert kwargs["grad_accum_current_steps"] == 4
        assert kwargs["grad_accum_current_batch"] == 8
        assert kwargs["output_dir"] == "/checkpoints/run1"

    def test_batch_size_resolution_precedence_and_safety(self) -> None:
        # Explicit batch_size takes precedence
        tcfg = SimpleNamespace(batch_size=2)
        kwargs = soup_callback_kwargs(tcfg, batch_size=16)
        assert kwargs["grad_accum_current_batch"] == 16

        # Fallback to tcfg.batch_size when explicit is None
        kwargs2 = soup_callback_kwargs(tcfg, batch_size=None)
        assert kwargs2["grad_accum_current_batch"] == 2

        # Clamp batch_size <= 0 to 1
        kwargs3 = soup_callback_kwargs(tcfg, batch_size=0)
        assert kwargs3["grad_accum_current_batch"] == 1

        # Reject booleans (True should not be treated as 1 if passed as batch_size)
        kwargs4 = soup_callback_kwargs(tcfg, batch_size=True)  # type: ignore[arg-type]
        assert kwargs4["grad_accum_current_batch"] == 2

        # Handle non-integer strings or invalid types safely
        kwargs5 = soup_callback_kwargs(SimpleNamespace(), batch_size="not_a_number")  # type: ignore[arg-type]
        assert kwargs5["grad_accum_current_batch"] == 1


# ===========================================================================
# 3. SoupTrainerCallback Integration with Unpacked Kwargs
# ===========================================================================


class TestCallbackIntegrationWithKwargs:
    """Verify that SoupTrainerCallback properly initializes and behaves when
    passed kwargs generated by soup_callback_kwargs.
    """

    def test_callback_initialization_with_unpacked_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        tcfg = TrainingConfig(
            loss_watchdog=True,
            loss_watchdog_threshold=2.5,
            loss_spike_recovery=True,
            loss_spike_recovery_max_attempts=4,
            loss_spike_recovery_lr_decay=0.35,
            grad_accum_auto_tune=True,
            grad_accum_pressure_threshold=0.88,
            gradient_accumulation_steps=2,
        )
        kwargs = soup_callback_kwargs(tcfg, batch_size=4, output_dir=str(tmp_path))

        display = MagicMock()
        cb = SoupTrainerCallback(display=display, tracker=None, run_id="test_run", **kwargs)

        assert cb._watchdog_enabled is True
        assert cb._watchdog_threshold == pytest.approx(2.5)
        assert cb._spike_recovery_enabled is True
        assert cb._spike_strategy.max_attempts == 4
        assert cb._spike_strategy.lr_decay == pytest.approx(0.35)
        assert cb._grad_accum_enabled is True
        assert cb._grad_accum_monitor.threshold == pytest.approx(0.88)
        assert cb._grad_accum_current == 2
        assert cb._grad_accum_batch == 4

    def test_spike_recovery_hint_written_via_unpacked_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        tcfg = TrainingConfig(
            loss_watchdog=True,
            loss_spike_recovery=True,
            loss_spike_recovery_max_attempts=3,
            loss_spike_recovery_lr_decay=0.5,
        )
        kwargs = soup_callback_kwargs(tcfg, batch_size=2, output_dir=str(tmp_path))

        display = MagicMock()
        cb = SoupTrainerCallback(display=display, tracker=None, run_id="spike_test", **kwargs)

        args = SimpleNamespace(learning_rate=2e-4, output_dir=str(tmp_path))
        cb._write_spike_recovery_hint(args, loss=12.5)

        hint_path = tmp_path / "spike_recovery.json"
        assert hint_path.is_file()

        hint_data = json.loads(hint_path.read_text(encoding="utf-8"))
        assert hint_data["previous_lr"] == pytest.approx(2e-4)
        assert hint_data["recommended_lr"] == pytest.approx(1e-4)
        assert hint_data["should_recover"] is True
        assert hint_data["attempts"] == 1


# ===========================================================================
# 4. Schema Rejection on Unsupported Tasks
# ===========================================================================


class TestSchemaRejectionOnUnsupportedTasks:
    """Validate that tasks without live callbacks reject watchdog, spike
    recovery, and grad_accum_auto_tune at config validation time.
    """

    @pytest.mark.parametrize("task", ["prm", "moe_lora_routing", "unlearn"])
    @pytest.mark.parametrize(
        ("field", "training_payload"),
        [
            ("loss_watchdog", {"loss_watchdog": True}),
            ("loss_spike_recovery", {"loss_watchdog": True, "loss_spike_recovery": True}),
            ("grad_accum_auto_tune", {"grad_accum_auto_tune": True}),
        ],
    )
    def test_unsupported_tasks_reject_monitoring_flags(
        self, task: str, field: str, training_payload: dict
    ) -> None:
        cfg_kwargs: dict = {
            "base": "sshleifer/tiny-gpt2",
            "task": task,
            "training": dict(training_payload),
        }

        if task == "unlearn":
            cfg_kwargs["data"] = {"train": "train.jsonl", "forget_set": "forget.jsonl"}
            cfg_kwargs["training"]["unlearn_method"] = "npo"
        elif task == "prm":
            cfg_kwargs["data"] = {"train": "train.jsonl", "format": "prm"}
        elif task == "moe_lora_routing":
            cfg_kwargs["data"] = {"train": "train.jsonl"}
            cfg_kwargs["training"]["mole_task_adapters"] = ["adapter_a", "adapter_b"]

        pattern = rf"training\.{field} is not supported for task='{task}'"
        with pytest.raises(ValueError, match=pattern):
            SoupConfig(**cfg_kwargs)

    @pytest.mark.parametrize("task", ["sft", "grpo", "dpo"])
    def test_supported_tasks_accept_monitoring_flags(self, task: str) -> None:
        cfg_kwargs: dict = {
            "base": "sshleifer/tiny-gpt2",
            "task": task,
            "data": {"train": "train.jsonl"},
            "training": {
                "loss_watchdog": True,
                "loss_spike_recovery": True,
                "grad_accum_auto_tune": True,
            },
        }
        if task in ("dpo", "grpo"):
            cfg_kwargs["data"]["format"] = "chatml"

        cfg = SoupConfig(**cfg_kwargs)
        assert cfg.training.loss_watchdog is True
        assert cfg.training.loss_spike_recovery is True
        assert cfg.training.grad_accum_auto_tune is True


# ===========================================================================
# 5. Behavioural Wrapper-Path Test (Issue #802 Acceptance Criterion 2)
# ===========================================================================


class TestTrainerWrapperBehaviouralCallbackWiring:
    """Verify that wrappers genuinely wire their config into SoupTrainerCallback
    on their train() execution path rather than relying on unpinned defaults (#802).
    """

    def test_grpo_wrapper_train_wires_spike_and_grad_accum_config(
        self, tmp_path: Path
    ) -> None:
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        wrapper = object.__new__(GRPOTrainerWrapper)
        wrapper.config = SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="grpo",
            data={"train": "train.jsonl", "format": "chatml"},
            training={
                "loss_watchdog": True,
                "loss_spike_recovery": True,
                "grad_accum_auto_tune": True,
            },
        )
        wrapper._batch_size = 4
        wrapper._output_dir = str(tmp_path)
        wrapper.tokenizer = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.state.log_history = []
        wrapper.trainer = mock_trainer

        captured_callbacks: list[object] = []
        mock_trainer.add_callback.side_effect = captured_callbacks.append

        wrapper.train(display=MagicMock())

        soup_cbs = [
            cb for cb in captured_callbacks if isinstance(cb, SoupTrainerCallback)
        ]
        assert len(soup_cbs) == 1
        cb = soup_cbs[0]
        assert cb._spike_recovery_enabled is True
        assert cb._grad_accum_enabled is True
        assert cb._grad_accum_batch == 4

    def test_dpo_wrapper_train_wires_spike_and_grad_accum_config(
        self, tmp_path: Path
    ) -> None:
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        wrapper = object.__new__(DPOTrainerWrapper)
        wrapper.config = SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="dpo",
            data={"train": "train.jsonl", "format": "chatml"},
            training={
                "loss_watchdog": True,
                "loss_spike_recovery": True,
                "grad_accum_auto_tune": True,
            },
        )
        wrapper._batch_size = 8
        wrapper._output_dir = str(tmp_path)
        wrapper.tokenizer = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.state.log_history = []
        wrapper.trainer = mock_trainer

        captured_callbacks: list[object] = []
        mock_trainer.add_callback.side_effect = captured_callbacks.append

        wrapper.train(display=MagicMock())

        soup_cbs = [
            cb for cb in captured_callbacks if isinstance(cb, SoupTrainerCallback)
        ]
        assert len(soup_cbs) == 1
        cb = soup_cbs[0]
        assert cb._spike_recovery_enabled is True
        assert cb._grad_accum_enabled is True
        assert cb._grad_accum_batch == 8
