"""The train CLI distinguishes a checkpointed user stop from other loop exits."""

import signal
from contextlib import closing
from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.monitoring.manual_save import manual_save_context
from soup_cli.utils.run_provenance import read_run_provenance
from tests.conftest import make_cli_runner


@pytest.mark.parametrize(
    ("mode", "steps", "status"),
    [
        ("second-signal", 2, "stopped"),
        ("during-save", 1, "stopped"),
        ("ppo-loop", 2, "stopped"),
        ("configured-stop", 2, "completed"),
        ("configured-during-save", 2, "completed"),
        ("last-step", 3, "completed"),
    ],
)
def test_train_records_only_a_checkpointed_user_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, steps: int, status: str
) -> None:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("soup_cli.utils.launcher.is_in_distributed", lambda: False)
    monkeypatch.setattr("soup_cli.commands.train.detect_device", lambda **_: ("cpu", "CPU"))
    monkeypatch.setattr("soup_cli.commands.train.get_gpu_info", lambda **_: {"memory_total": "N/A"})
    Path("data.jsonl").write_text('{"instruction":"hi", "output":"hello"}\n', encoding="utf-8")
    Path("soup.yaml").write_text(
        "base: owner/base\ndata: {train: data.jsonl, format: alpaca, val_split: 0}\n"
        "training: {epochs: 1, quantization: none, batch_size: 1}\noutput: final\n",
        encoding="utf-8",
    )

    class StopScenario(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0 or (
                state.global_step == 1 and mode in {"second-signal", "ppo-loop"}
            ) or (state.global_step == 2 and mode == "last-step"):
                signal.raise_signal(signal.SIGINT)

        def on_step_end(self, args, state, control, **kwargs):
            if mode in {"configured-stop", "configured-during-save"} and state.global_step == 2:
                control.should_training_stop = True
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            if (mode == "during-save" and state.global_step == 1) or (
                mode == "configured-during-save" and state.global_step == 2
            ):
                signal.raise_signal(signal.SIGINT)

    class PatchedTrainer:
        """Use the real callback protocol without model loads or optimizer steps."""

        def __init__(self) -> None:
            self.args = TrainingArguments(
                output_dir="final", use_cpu=True, report_to="none", max_steps=3, save_steps=1000,
            )
            self.state = TrainerState(max_steps=3, save_steps=1000)
            self.control = TrainerControl()
            self.callback_handler = CallbackHandler(
                [DefaultFlowCallback(), StopScenario()], None, None, None, None,
            )

        def add_callback(self, callback: TrainerCallback) -> None:
            self.callback_handler.add_callback(callback)

        def remove_callback(self, callback: TrainerCallback) -> None:
            self.callback_handler.remove_callback(callback)

        def train(self) -> None:
            def emit(event: str) -> None:
                self.control = getattr(self.callback_handler, event)(
                    self.args, self.state, self.control,
                )

            emit("on_train_begin")
            for step in range(1, self.state.max_steps + 1):
                emit("on_step_begin")
                self.state.global_step = step
                emit("on_step_end")
                if self.control.should_save:
                    (Path("final") / f"checkpoint-{step}").mkdir(parents=True)
                    emit("on_save")
                if self.control.should_training_stop and mode != "ppo-loop":
                    break
            emit("on_train_end")

    class Wrapper:
        def setup(self, dataset: dict) -> None:
            self.trainer = PatchedTrainer()

        def train(self, **kwargs) -> dict:
            with manual_save_context(self.trainer, stop_after_save=mode == "ppo-loop"):
                self.trainer.train()
            return {
                "output_dir": "final", "initial_loss": 1.0, "final_loss": 0.5,
                "total_steps": self.trainer.state.global_step, "duration_secs": 1.0,
            }

    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", lambda *_a, **_kw: Wrapper())
    result = make_cli_runner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])
    assert result.exit_code == 0, (result.output, result.exception)
    with closing(ExperimentTracker()) as tracker:
        (run,) = tracker.list_runs()
    assert run["total_steps"] == steps
    assert run["status"] == status
    artifact_status = read_run_provenance("final").get("status")
    assert artifact_status == ("stopped" if status == "stopped" else None)
