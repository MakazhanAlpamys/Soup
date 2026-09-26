"""Tests for TrainingDisplay."""

from io import StringIO

from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.monitoring.display import TrainingDisplay


def _render_to_str(panel) -> str:
    """Render a Rich Panel to a plain string for assertion."""
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True)
    console.print(panel)
    return buf.getvalue()


def _make_config():
    """Create a minimal SoupConfig for display testing."""
    return SoupConfig(
        base="test-model",
        data={"train": "./data.jsonl"},
        training={"epochs": 3},
    )


def test_display_init():
    """Display should initialize with default values."""
    display = TrainingDisplay(_make_config(), device_name="cuda")
    assert display.current_step == 0
    assert display.total_steps == 0
    assert display.loss == 0.0
    assert display.device_name == "cuda"


def test_display_update():
    """Update should store new metric values."""
    display = TrainingDisplay(_make_config())
    display.total_steps = 100

    display.update(step=50, epoch=1.5, loss=0.876, lr=1e-5, speed=3.2, gpu_mem="12/24 GB")

    assert display.current_step == 50
    assert display.current_epoch == 1.5
    assert display.loss == 0.876
    assert display.lr == 1e-5
    assert display.speed == 3.2
    assert display.gpu_mem == "12/24 GB"


def test_display_render_panel():
    """_render should produce a Panel with correct content."""
    display = TrainingDisplay(_make_config(), device_name="cuda:0")
    display.total_steps = 100
    display.update(step=62, epoch=2.0, loss=0.847, lr=1.4e-5, speed=3.2, gpu_mem="18/24 GB")

    panel = display._render()
    rendered = _render_to_str(panel)
    assert "62/100" in rendered
    assert "0.847" in rendered
    # The panel label must identify the figure as a peak, not just "GPU:" (#650).
    assert "GPU peak:" in rendered


def test_display_render_zero_steps():
    """_render with total_steps=0 should not crash (division by zero)."""
    display = TrainingDisplay(_make_config())
    display.total_steps = 0
    panel = display._render()
    assert panel is not None


def test_display_experiment_name():
    """Display should use experiment_name in panel title if set."""
    config = SoupConfig(
        base="test-model",
        data={"train": "./data.jsonl"},
        experiment_name="my-experiment",
    )
    display = TrainingDisplay(config)
    display.total_steps = 10
    panel = display._render()
    rendered = _render_to_str(panel)
    assert "my-experiment" in rendered


def test_display_start_stop():
    """Start and stop should not crash (we don't test actual terminal rendering)."""
    display = TrainingDisplay(_make_config())
    display.start(total_steps=100)
    assert display.total_steps == 100
    assert display._live is not None
    display.stop()


def test_display_update_without_live():
    """Update without calling start should not crash."""
    display = TrainingDisplay(_make_config())
    display.update(step=1, epoch=0.1, loss=2.0, lr=1e-4)
    assert display.current_step == 1
