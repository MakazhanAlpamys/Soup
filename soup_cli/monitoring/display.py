"""Rich live training dashboard in the terminal."""

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from soup_cli.config.schema import SoupConfig

console = Console()


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
        self._live: Live | None = None

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
        """Stop the live display."""
        if self._live:
            self._live.stop()

    def _render(self) -> Panel:
        """Render the dashboard panel."""
        progress_pct = (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = []
        lines.append(f"Epoch {self.current_epoch:.1f}/{self.config.training.epochs}  [{bar}] {progress_pct:.0f}%")
        lines.append(f"Step:  {self.current_step}/{self.total_steps}")
        lines.append(f"Loss:  {self.loss:.4f}    LR: {self.lr:.2e}")

        if self.speed > 0:
            lines.append(f"Speed: {self.speed:.2f} it/s")
        if self.gpu_mem:
            lines.append(f"GPU:   {self.gpu_mem}")
        if self.grad_norm > 0:
            lines.append(f"Grad:  {self.grad_norm:.4f}")

        content = "\n".join(lines)
        return Panel(
            content,
            title=f"[bold green]Soup Training: {self.config.experiment_name or self.config.base}[/]",
            subtitle=f"[dim]{self.device_name}[/]",
            border_style="green",
        )
