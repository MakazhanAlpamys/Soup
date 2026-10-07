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

from soup_cli.config.schema import EvalGateConfig, SoupConfig, TrainingConfig
from soup_cli.monitoring.callback import (
    SoupTrainerCallback,
    build_soup_trainer_callback,
    soup_callback_kwargs,
)

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
    """Verify that all 16 trainers use the shared callback builder."""

    def test_all_expected_trainers_exist(self) -> None:
        for filename in EXPECTED_CALLBACK_TRAINERS | UNSUPPORTED_CALLBACK_TRAINERS:
            filepath = TRAINER_DIR / filename
            assert filepath.is_file(), (
                f"Trainer file {filename} does not exist in {TRAINER_DIR}"
            )

    def test_ast_scan_trainer_callback_kwargs_unification(self) -> None:
        discovered_callback_trainers: set[str] = set()

        for py_file in sorted(TRAINER_DIR.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))

            builder_calls = []
            direct_callback_calls = []

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                func_name = ast.unparse(node.func)

                if func_name == "build_soup_trainer_callback":
                    builder_calls.append(node)
                elif (
                    func_name == "SoupTrainerCallback"
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "SoupTrainerCallback"
                    )
                ):
                    direct_callback_calls.append(node)

            # Direct callback construction is forbidden in every trainer.
            assert not direct_callback_calls, (
                f"{py_file.name} directly instantiates SoupTrainerCallback; "
                "use build_soup_trainer_callback instead"
            )

            if not builder_calls:
                continue

            discovered_callback_trainers.add(py_file.name)

            # Each supported trainer must have exactly one shared builder call.
            assert len(builder_calls) == 1, (
                f"{py_file.name} must have exactly one "
                "build_soup_trainer_callback call"
            )

            builder = builder_calls[0]
            keywords = {kw.arg: kw.value for kw in builder.keywords if kw.arg}

            assert "config" in keywords, (
                f"{py_file.name} does not pass config to "
                "build_soup_trainer_callback"
            )
            assert ast.unparse(keywords["config"]) == "self.config", (
                f"{py_file.name} must pass config=self.config"
            )

            assert "batch_size" in keywords, (
                f"{py_file.name} does not pass batch_size to "
                "build_soup_trainer_callback"
            )
            assert ast.unparse(keywords["batch_size"]) == "self._batch_size", (
                f"{py_file.name} must pass batch_size=self._batch_size"
            )

        assert discovered_callback_trainers == EXPECTED_CALLBACK_TRAINERS, (
            f"Expected {EXPECTED_CALLBACK_TRAINERS}, "
            f"but found {discovered_callback_trainers}"
        )

class TestBuildSoupTrainerCallback:
    """Tests for the shared trainer callback builder."""

    def test_passes_eval_config_to_callback(self) -> None:
        config = SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="grpo",
            data={"train": "train.jsonl", "format": "chatml"},
            eval={
                "auto_eval": True,
                "benchmarks": ["mmlu"],
                "custom_tasks": "custom.jsonl",
            },
        )
        display = MagicMock()

        callback = build_soup_trainer_callback(
            display,
            config=config,
            tracker=MagicMock(),
            run_id="test-run",
            batch_size=4,
            output_dir="/tmp/output",
        )

        assert callback.eval_config is config.eval
        assert callback.run_id == "test-run"
        assert callback.output_dir == "/tmp/output"

    def test_builder_passes_eval_gate_to_callback(self) -> None:
        config = SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="sft",
            data={"train": "train.jsonl", "format": "chatml"},
            training={
                "eval_gate": {
                    "enabled": True,
                    "suite": "evals/gate.yaml",
                }
            },
        )
        assert config.training.eval_gate is not None

        callback = build_soup_trainer_callback(MagicMock(), config=config)

        assert callback.eval_gate_config is config.training.eval_gate

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
        eval_gate = EvalGateConfig(
            enabled=False,
            suite="evals/gate.yaml",
        )
        tcfg_with_gate = TrainingConfig(eval_gate=eval_gate)
        kwargs_with_gate = soup_callback_kwargs(tcfg_with_gate)

        assert kwargs_with_gate["eval_gate_config"] is eval_gate
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
        # #1620: the monitor is built lazily on the first advisory call, so
        # construction stores the threshold instead of a GradAccumMonitor.
        assert cb._grad_accum_monitor is None
        assert cb._grad_accum_threshold == pytest.approx(0.88)
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

    def test_grpo_wrapper_train_wires_eval_config(
        self, tmp_path: Path
    ) -> None:
        from soup_cli.trainer.grpo import GRPOTrainerWrapper
        wrapper = object.__new__(GRPOTrainerWrapper)
        wrapper.config = SoupConfig(
            base="sshleifer/tiny-gpt2",
            task="grpo",
            data={"train": "train.jsonl", "format": "chatml"},
            eval={
                "auto_eval": True,
                "benchmarks": ["mmlu"],
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
        assert cb.eval_config is wrapper.config.eval
        assert cb.eval_config.auto_eval is True
        assert cb.eval_config.benchmarks == ["mmlu"]
