"""Rich live training dashboard in the terminal."""

from typing import Any, Mapping, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from soup_cli.config.schema import SoupConfig

console = Console()


def format_gate_row(state: Optional[Mapping[str, Any]]) -> str:
    """Render the eval-gate status row for the live training panel (#36).

    Returns an empty string when ``state`` is None or empty (so the row is
    hidden when eval-gate is disabled).

    Format example::

        Gate: helpfulness 7.8 [green]✓[/] | math 0.82 [red]✗[/] (-0.06 from base) | STOP

    Pure formatter — no I/O, no side effects — so it is trivially testable
    via ``Console(file=StringIO())`` without spinning up a Live display.
    """
    if not state:
        return ""
    tasks = state.get("tasks") or []
    if not tasks:
        return ""
    parts: list[str] = []
    for task in tasks:
        name = str(task.get("name", "?"))
        score = task.get("score")
        # Explicit ``is True`` so a missing field renders the ``?`` mark
        # rather than the false-y red ✗ (e.g. tasks still pending).
        passed = task.get("passed") is True
        delta = task.get("delta")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
        mark = "[green]✓[/]" if passed else "[red]✗[/]"
        chunk = f"{name} {score_str} {mark}"
        if delta is not None and isinstance(delta, (int, float)):
            sign = "+" if delta >= 0 else ""
            chunk += f" ({sign}{delta:.2f})"
        parts.append(chunk)
    body = " | ".join(parts)
    action = state.get("action")
    suffix = ""
    if action == "stop":
        suffix = " | [bold red]STOP[/]"
    elif action == "warn":
        suffix = " | [yellow]WARN[/]"
    return f"[bold]Gate:[/] {body}{suffix}"


class TrainingDisplay:
    """Live-updating terminal dashboard for training progress."""

    def __init__(self, config: SoupConfig, device_name: str = ""):
        self.config = config
        self.device_name = device_name
        self.current_step = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.loss = 0.0
        self.lr = 0.0
        self.grad_norm = 0.0
        self.gpu_mem = ""
        self.speed = 0.0
        self._live: Optional[Live] = None

    def start(self, total_steps: int):
        """Start the live display."""
        self.total_steps = total_steps
        self._live = Live(self._render(), console=console, refresh_per_second=2)
        self._live.start()

    def update(self, step: int, epoch: float, loss: float, lr: float, **kwargs):
        """Update display with new metrics."""
        self.current_step = step
        self.current_epoch = epoch
        self.loss = loss
        self.lr = lr
        self.grad_norm = kwargs.get("grad_norm", 0.0)
        self.speed = kwargs.get("speed", 0.0)
        self.gpu_mem = kwargs.get("gpu_mem", "")

        if self._live:
            self._live.update(self._render())

    def stop(self):
        """Stop the live display. Safe to call multiple times."""
        if self._live:
            self._live.stop()
            self._live = None

    def _render(self) -> Panel:
        """Render the dashboard panel."""
        if self.total_steps > 0:
            progress_pct = self.current_step / self.total_steps * 100
        else:
            progress_pct = 0
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "#" * filled + "-" * (bar_width - filled)

        epochs = self.config.training.epochs
        epoch_str = f"Epoch {self.current_epoch:.1f}/{epochs}"
        lines = []
        lines.append(f"{epoch_str}  [{bar}] {progress_pct:.0f}%")
        lines.append(f"Step:  {self.current_step}/{self.total_steps}")
        lines.append(f"Loss:  {self.loss:.4f}    LR: {self.lr:.2e}")

        if self.speed > 0:
            lines.append(f"Speed: {self.speed:.2f} it/s")
        if self.gpu_mem:
            lines.append(f"GPU:   {self.gpu_mem}")
        if self.grad_norm > 0:
            lines.append(f"Grad:  {self.grad_norm:.4f}")

        content = "\n".join(lines)
        name = self.config.experiment_name or self.config.base
        return Panel(
            content,
            title=f"[bold green]Soup Training: {name}[/]",
            subtitle=f"[dim]{self.device_name}[/]",
            border_style="green",
        )
