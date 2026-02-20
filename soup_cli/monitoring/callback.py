"""HuggingFace Trainer callback that feeds metrics to our display."""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from soup_cli.monitoring.display import TrainingDisplay


class SoupTrainerCallback(TrainerCallback):
    """Bridges HF Trainer events to Soup's Rich live display."""

    def __init__(self, display: TrainingDisplay):
        self.display = display

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.display.start(total_steps=state.max_steps)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
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

        self.display.update(
            step=state.global_step,
            epoch=state.epoch or 0,
            loss=logs.get("loss", 0.0),
            lr=logs.get("learning_rate", 0.0),
            grad_norm=logs.get("grad_norm", 0.0),
            speed=logs.get("train_steps_per_second", 0.0),
            gpu_mem=gpu_mem,
        )

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.display.stop()
