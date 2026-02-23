"""HuggingFace Trainer callback that feeds metrics to our display and tracker."""

from __future__ import annotations

from typing import Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from soup_cli.monitoring.display import TrainingDisplay


class SoupTrainerCallback(TrainerCallback):
    """Bridges HF Trainer events to Soup's Rich live display and experiment tracker."""

    def __init__(
        self,
        display: TrainingDisplay,
        tracker: Optional[object] = None,
        run_id: str = "",
    ):
        self.display = display
        self.tracker = tracker
        self.run_id = run_id

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState,
        control: TrainerControl, **kwargs,
    ):
        self.display.start(total_steps=state.max_steps)

    def on_log(
        self, args: TrainingArguments, state: TrainerState,
        control: TrainerControl, logs=None, **kwargs,
    ):
        if logs is None:
            return

        # Try to get GPU memory
        gpu_mem = ""
        try:
            import torch

            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                gpu_mem = f"{used:.1f}/{total:.1f} GB"
        except Exception:
            pass

        step = state.global_step
        epoch = state.epoch or 0
        loss = logs.get("loss", 0.0)
        lr = logs.get("learning_rate", 0.0)
        grad_norm = logs.get("grad_norm", 0.0)
        speed = logs.get("train_steps_per_second", 0.0)

        self.display.update(
            step=step,
            epoch=epoch,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            speed=speed,
            gpu_mem=gpu_mem,
        )

        # Log to experiment tracker
        if self.tracker and self.run_id:
            self.tracker.log_metrics(
                run_id=self.run_id,
                step=step,
                epoch=epoch,
                loss=loss,
                lr=lr,
                grad_norm=grad_norm,
                speed=speed,
                gpu_mem=gpu_mem,
            )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState,
        control: TrainerControl, **kwargs,
    ):
        self.display.stop()
