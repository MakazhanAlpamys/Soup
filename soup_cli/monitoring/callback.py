"""HuggingFace Trainer callback that feeds metrics to our display and tracker."""

from __future__ import annotations

import logging
from typing import Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from soup_cli.monitoring.display import TrainingDisplay

logger = logging.getLogger(__name__)


class SoupTrainerCallback(TrainerCallback):
    """Bridges HF Trainer events to Soup's Rich live display and experiment tracker."""

    def __init__(
        self,
        display: TrainingDisplay,
        tracker: Optional[object] = None,
        run_id: str = "",
        eval_config: Optional[object] = None,
        output_dir: str = "",
        loss_watchdog: bool = False,
        loss_watchdog_threshold: float = 3.0,
        loss_watchdog_patience: int = 5,
    ):
        self.display = display
        self.tracker = tracker
        self.run_id = run_id
        self.eval_config = eval_config
        self.output_dir = output_dir
        # Loss watchdog state
        self._watchdog_enabled = loss_watchdog
        self._watchdog_threshold = loss_watchdog_threshold
        self._watchdog_patience = loss_watchdog_patience
        self._watchdog_counter = 0
        self._watchdog_fired = False

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
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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

        # Loss watchdog — detect loss spikes and auto-stop
        if self._watchdog_enabled and not self._watchdog_fired and "loss" in logs:
            if loss > self._watchdog_threshold:
                self._watchdog_counter += 1
                if self._watchdog_counter >= self._watchdog_patience:
                    self._watchdog_fired = True
                    # Stop Live display before printing panel
                    self.display.stop()

                    from rich.console import Console as WatchdogConsole
                    from rich.panel import Panel

                    wc = WatchdogConsole()
                    wc.print(Panel(
                        f"[bold red]Loss watchdog triggered![/]\n\n"
                        f"Loss {loss:.4f} exceeded threshold "
                        f"{self._watchdog_threshold} for "
                        f"{self._watchdog_counter} consecutive steps "
                        f"(patience={self._watchdog_patience}).\n\n"
                        f"Training will stop.",
                        title="Loss Watchdog",
                        border_style="red",
                    ))
                    control.should_training_stop = True
            else:
                self._watchdog_counter = 0

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
        self._run_auto_eval()

    def _run_auto_eval(self) -> None:
        """Run automatic evaluation if configured."""
        if self.eval_config is None:
            return
        if not getattr(self.eval_config, "auto_eval", False):
            return
        if not self.output_dir:
            return

        from rich.console import Console

        console = Console()
        console.print("\n[bold blue]Running auto-eval...[/]")

        benchmarks = getattr(self.eval_config, "benchmarks", None) or []
        custom_tasks = getattr(self.eval_config, "custom_tasks", None)

        # Run standard benchmarks
        if benchmarks:
            try:
                from soup_cli.commands.eval import benchmark
                benchmark(
                    model=self.output_dir,
                    benchmarks=",".join(benchmarks),
                    num_fewshot=None,
                    batch_size=8,
                    run_id=self.run_id,
                    device=None,
                )
            except Exception as exc:
                logger.exception("Auto-eval benchmark failed")
                console.print(f"[yellow]Auto-eval benchmark failed: {exc}[/]")

        # Run custom eval
        if custom_tasks:
            try:
                from soup_cli.commands.eval import custom
                custom(
                    tasks=custom_tasks,
                    model=self.output_dir,
                    run_id=self.run_id,
                )
            except Exception as exc:
                logger.exception("Auto-eval custom failed")
                console.print(f"[yellow]Auto-eval custom failed: {exc}[/]")
