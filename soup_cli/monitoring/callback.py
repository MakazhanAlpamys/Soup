"""HuggingFace Trainer callback that feeds metrics to our display and tracker."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from soup_cli.monitoring.display import TrainingDisplay

logger = logging.getLogger(__name__)
console = Console()


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
        eval_gate_config: Optional[object] = None,
        spike_recovery: bool = False,
        spike_recovery_max_attempts: int = 3,
        spike_recovery_lr_decay: float = 0.5,
        grad_accum_auto_tune: bool = False,
        grad_accum_pressure_threshold: float = 0.9,
        grad_accum_total_vram_gb: float = 24.0,
        grad_accum_current_steps: int = 1,
        grad_accum_current_batch: int = 1,
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
        # v0.33.0 #57 — spike-recovery hint state
        self._spike_recovery_enabled = spike_recovery
        self._spike_recovery_attempts = 0
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy
        if spike_recovery:
            self._spike_strategy = SpikeRecoveryStrategy(
                max_attempts=spike_recovery_max_attempts,
                lr_decay=spike_recovery_lr_decay,
            )
        else:
            self._spike_strategy = None
        # v0.33.0 #59 — grad-accum advisory monitor
        self._grad_accum_enabled = grad_accum_auto_tune
        self._grad_accum_current = max(1, int(grad_accum_current_steps))
        self._grad_accum_batch = max(1, int(grad_accum_current_batch))
        self._grad_accum_advised = False
        if grad_accum_auto_tune:
            from soup_cli.utils.grad_accum import GradAccumMonitor
            self._grad_accum_monitor = GradAccumMonitor(
                total_vram_gb=grad_accum_total_vram_gb,
                threshold=grad_accum_pressure_threshold,
            )
        else:
            self._grad_accum_monitor = None
        # Eval gate state (Part B of v0.26.0)
        self.eval_gate_config = eval_gate_config
        # Tests inject these; prod wiring sets them at on_train_begin time.
        self._gate_suite: Optional[object] = None
        self._gate_generate_fn: Optional[object] = None
        self._gate_baseline: Optional[dict] = None
        # Injectable for tests; default is the real implementation.
        self._gate_run_fn = None

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

                    # v0.33.0 #57 — spike recovery hint: write a recovery
                    # state file the user can resume from. We do NOT mutate
                    # optimizer state in-place (HF Trainer does not expose a
                    # safe public API for that mid-loop) but we leave a
                    # machine-readable hint so a wrapper / re-launch can
                    # resume with a decayed LR.
                    if self._spike_strategy is not None:
                        self._write_spike_recovery_hint(args, loss)

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

        # v0.33.0 #59 — grad-accum advisory (one-shot per run)
        if (
            self._grad_accum_enabled
            and not self._grad_accum_advised
            and self._grad_accum_monitor is not None
        ):
            self._maybe_advise_grad_accum()

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

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState,
        control: TrainerControl, **kwargs,
    ):
        """Run the eval gate, if configured."""
        self._run_eval_gate(state, control)

    def _run_eval_gate(self, state: TrainerState, control: TrainerControl) -> None:
        """Execute the eval gate and react per ``on_regression`` policy."""
        cfg = self.eval_gate_config
        if cfg is None or not getattr(cfg, "enabled", False):
            return
        suite = self._gate_suite
        if suite is None:
            return
        every_n = int(getattr(cfg, "every_n_epochs", 1) or 1)
        current_epoch = int(state.epoch or 0)
        if current_epoch <= 0 or current_epoch % every_n != 0:
            return

        from soup_cli.eval.gate import run_gate as _default_run_gate

        run_fn = self._gate_run_fn if self._gate_run_fn is not None else _default_run_gate

        baseline = self._gate_baseline or {}
        threshold = float(getattr(cfg, "regression_threshold", 0.05))
        generate_fn = self._gate_generate_fn or (lambda prompt: "")
        on_reg = getattr(cfg, "on_regression", "stop")
        try:
            result = run_fn(
                suite, generate_fn=generate_fn, baseline=baseline,
                regression_threshold=threshold,
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            # Structured errors — log and react per policy. 'stop' means
            # the user wants safety; treat errors as regressions for stop.
            logger.warning("eval gate failed to execute: %s", exc)
            if on_reg == "stop":
                control.should_training_stop = True
            return
        except Exception as exc:  # unexpected — fail safe under 'stop'
            logger.exception("eval gate raised unexpected error: %s", exc)
            if on_reg == "stop":
                control.should_training_stop = True
            return

        if not result.passed:
            if on_reg == "stop":
                control.should_training_stop = True
                logger.warning(
                    "eval gate FAILED (%d task(s) regressed); stopping training",
                    sum(1 for r in result.task_results if not r.passed),
                )
            elif on_reg == "warn":
                logger.warning(
                    "eval gate FAILED (%d task(s) regressed); continuing per policy",
                    sum(1 for r in result.task_results if not r.passed),
                )
            # on_reg == "continue": silent per user policy

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

    # ------------------------------------------------------------------
    # v0.33.0 #57 — spike recovery hint
    # ------------------------------------------------------------------

    def _write_spike_recovery_hint(self, args, loss: float) -> None:
        """Write a JSON recovery hint next to the run output so a wrapper
        script (or `soup train --resume`) can pick up the new LR.

        Best-effort: errors are logged but never crash training.
        """
        import json
        from pathlib import Path

        if self._spike_strategy is None:
            return
        attempts = self._spike_recovery_attempts
        try:
            new_lr = self._spike_strategy.compute_new_lr(args.learning_rate)
        except ValueError:
            return
        recover = self._spike_strategy.should_recover(attempts)
        out_dir = Path(self.output_dir or args.output_dir or ".")
        # Containment: never write outside cwd (matches v0.32.0 lr-finder /
        # autopilot / cans policy). Caller's output_dir is already validated
        # by SoupConfig but args.output_dir from raw HF TrainingArguments is
        # not — defence-in-depth.
        from soup_cli.utils.paths import is_under_cwd
        if not is_under_cwd(out_dir):
            logger.warning(
                "spike_recovery hint skipped: output_dir %s is outside cwd",
                out_dir,
            )
            return
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            hint_path = out_dir / "spike_recovery.json"
            hint_path.write_text(json.dumps({
                "attempts": attempts + 1,
                "max_attempts": self._spike_strategy.max_attempts,
                "loss_at_spike": float(loss),
                "previous_lr": float(args.learning_rate),
                "recommended_lr": float(new_lr),
                "should_recover": recover,
            }, indent=2), encoding="utf-8")
            self._spike_recovery_attempts = attempts + 1
            console.print(
                f"[yellow]Spike recovery hint written:[/] {hint_path} "
                f"(recommended_lr={new_lr:.2e}, should_recover={recover})"
            )
        except OSError as exc:
            logger.warning("Failed to write spike recovery hint: %s", exc)

    # ------------------------------------------------------------------
    # v0.33.0 #59 — grad-accum advisory (Phase 1)
    # ------------------------------------------------------------------

    def _maybe_advise_grad_accum(self) -> None:
        """Sample VRAM use; if pressure crosses threshold once, print the
        recommended (batch_size, grad_accum_steps) pair.

        Phase 1 is advisory-only. Phase 2 (live DataLoader rebuild) requires
        a small upstream TRL change tracked as a known limitation.
        """
        if self._grad_accum_advised:
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            used_gb = torch.cuda.max_memory_allocated() / (1024**3)
        except Exception:  # noqa: BLE001 — VRAM probe is best-effort
            return

        if self._grad_accum_monitor is None:
            return
        self._grad_accum_monitor.observe(used_gb)
        if not self._grad_accum_monitor.should_adjust(used_gb):
            return
        new_batch, new_accum = self._grad_accum_monitor.recommend(
            self._grad_accum_batch, self._grad_accum_current,
        )
        if new_accum == self._grad_accum_current:
            return
        self._grad_accum_advised = True
        console.print(
            f"[yellow]Grad-accum advisory:[/] VRAM pressure crossed "
            f"threshold; recommend (batch_size, grad_accum_steps) "
            f"({self._grad_accum_batch}, {self._grad_accum_current}) -> "
            f"({new_batch}, {new_accum}). "
            f"Restart training with the new pair to take effect."
        )
