"""Textual application for ``soup tui`` (v0.34.0 Part G).

Kept in its own module so the heavy ``textual`` import only fires when the
TUI is actually launched. The class derives from ``textual.App`` if textual
is available; otherwise importing this module raises ImportError, which
``commands/tui.py`` catches and turns into a friendly install hint.
"""

from __future__ import annotations

from typing import Any, List

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, Static
except ImportError as exc:  # pragma: no cover - guarded by commands/tui.py
    raise ImportError(
        "textual is required for `soup tui`; install with `pip install textual`"
    ) from exc


from rich.markup import escape as markup_escape

from soup_cli.utils.replay import summarise
from soup_cli.utils.run_cost import format_cost_usd


def _safe(value, fallback: str = "-") -> str:
    """Stringify a DB value with markup escaping (defends against [...] in names)."""
    if value is None or value == "":
        return fallback
    return markup_escape(str(value))


def build_runs_table_rows(runs: List[dict]) -> List[List[str]]:
    """Build the rows shown in the Runs table. Pure for testability."""
    rows: List[List[str]] = []
    for run in runs:
        cost = run.get("cost_usd")
        rows.append([
            _safe(run.get("run_id"))[:32],
            _safe(run.get("experiment_name"))[:24],
            _safe(run.get("base_model"))[:32],
            _safe(run.get("task")),
            _safe(run.get("status")),
            f"{run['final_loss']:.4f}" if run.get("final_loss") is not None else "-",
            str(run.get("total_steps") or "-"),
            format_cost_usd(cost),
        ])
    return rows


def build_run_detail(run: dict, metrics: List[dict]) -> str:
    """Render the right-hand detail panel for a selected run."""
    summary = summarise(metrics)
    lines = [
        f"Run ID: {_safe(run.get('run_id'))}",
        f"Status: {_safe(run.get('status'))}",
        f"Model:  {_safe(run.get('base_model'))}",
        f"Task:   {_safe(run.get('task'))}",
    ]
    if summary.initial_loss is not None and summary.final_loss is not None:
        lines.append(f"Loss:   {summary.initial_loss:.4f} → {summary.final_loss:.4f}")
    if summary.min_loss is not None and summary.min_loss_step is not None:
        lines.append(f"Best:   {summary.min_loss:.4f} @ step {summary.min_loss_step}")
    cost = run.get("cost_usd")
    if cost is not None:
        lines.append(f"Cost:   {format_cost_usd(cost)}")
    if run.get("duration_secs"):
        lines.append(f"Time:   {run['duration_secs']:.0f}s")
    return "\n".join(lines)


class SoupTuiApp(App):  # type: ignore[misc]
    """Two-pane dashboard: list of runs (left) + detail (right)."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    #runs-table { height: 1fr; }
    #detail { width: 40%; padding: 1 2; }
    """

    _COLUMNS = ("Run", "Name", "Model", "Task", "Status", "Loss", "Steps", "Cost")

    def __init__(self, *, refresh_secs: float = 1.0, run_limit: int = 50) -> None:
        super().__init__()
        self._refresh_secs = max(0.25, min(refresh_secs, 10.0))
        self._run_limit = max(1, min(run_limit, 1000))
        # ExperimentTracker is imported lazily — opens a SQLite connection,
        # and Textual is sometimes imported in environments without write
        # access to the default `~/.soup` dir.
        from soup_cli.experiment.tracker import ExperimentTracker

        self._tracker = ExperimentTracker()
        self._runs: List[dict] = []

    def compose(self) -> ComposeResult:  # pragma: no cover - UI plumbing
        yield Header()
        with Horizontal():
            with Vertical():
                yield DataTable(id="runs-table")
            yield Static("Select a run", id="detail")
        yield Footer()

    def on_mount(self) -> None:  # pragma: no cover - UI plumbing
        table = self.query_one("#runs-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(*self._COLUMNS)
        self._reload()
        self.set_interval(self._refresh_secs, self._reload)

    def action_refresh(self) -> None:  # pragma: no cover - UI plumbing
        self._reload()

    def _reload(self) -> None:  # pragma: no cover - thin SQLite wrapper
        self._runs = self._tracker.list_runs(limit=self._run_limit)
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        for row in build_runs_table_rows(self._runs):
            table.add_row(*row)

    def on_data_table_row_selected(self, event: Any) -> None:  # pragma: no cover
        index = getattr(event, "cursor_row", 0)
        if not (0 <= index < len(self._runs)):
            return
        run = self._runs[index]
        metrics = self._tracker.get_metrics(run["run_id"])
        detail = self.query_one("#detail", Static)
        detail.update(build_run_detail(run, metrics))
