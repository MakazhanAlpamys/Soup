"""Plotext 5.x / 6.x compatibility coverage."""

from importlib.metadata import version as distribution_version
from io import StringIO

from packaging.version import Version
from rich.console import Console

from soup_cli.utils.plotext_compat import render_histogram, render_line


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return self


class _Console:
    def __init__(self) -> None:
        self.output: list[str] = []
        self.options: list[dict] = []

    def print(self, value, **kwargs) -> None:
        self.output.append(str(value))
        self.options.append(kwargs)


class _Plotext5(_Recorder):
    def clf(self):
        return self._record("clf")

    def hist(self, *args, **kwargs):
        return self._record("hist", *args, **kwargs)

    def plot(self, *args, **kwargs):
        return self._record("plot", *args, **kwargs)

    def title(self, *args, **kwargs):
        return self._record("title", *args, **kwargs)

    def xlabel(self, *args, **kwargs):
        return self._record("xlabel", *args, **kwargs)

    def ylabel(self, *args, **kwargs):
        return self._record("ylabel", *args, **kwargs)

    def theme(self, *args, **kwargs):
        return self._record("theme", *args, **kwargs)

    def build(self):
        self._record("build")
        return "\x1b[31mplot\x1b[0m"


class _Signal(_Recorder):
    def lines(self):
        return self._record("lines")

    def label(self, *args, **kwargs):
        return self._record("label", *args, **kwargs)


class _Figure6(_Recorder):
    def clear(self):
        return self._record("clear")

    def hist(self, *args, **kwargs):
        self._record("hist", *args, **kwargs)
        return _Signal()

    def signal(self, *args, **kwargs):
        self._record("signal", *args, **kwargs)
        return _Signal()

    def draw(self, *args, **kwargs):
        return self._record("draw", *args, **kwargs)

    def title(self, *args, **kwargs):
        return self._record("title", *args, **kwargs)

    def label(self, *args, **kwargs):
        return self._record("label", *args, **kwargs)

    def theme(self, *args, **kwargs):
        return self._record("theme", *args, **kwargs)

    def build(self):
        self._record("build")
        return "\x1b[31mplot\x1b[0m"


class _Plotext6:
    def __init__(self) -> None:
        self.figure = _Figure6()


def test_plotext5_module_api_remains_supported() -> None:
    plotext = _Plotext5()
    console = _Console()
    render_histogram(
        plotext,
        [1, 2],
        console=console,
        bins=2,
        title="Histogram",
        xlabel="X",
        ylabel="Y",
        theme="dark",
    )
    render_line(
        plotext,
        [1, 2],
        [0.5, 0.25],
        console=console,
        label="loss",
        title="Loss",
        xlabel="Step",
        ylabel="Loss",
        theme="dark",
    )

    names = [name for name, _, _ in plotext.calls]
    assert names == [
        "clf",
        "hist",
        "title",
        "xlabel",
        "ylabel",
        "theme",
        "build",
        "clf",
        "plot",
        "title",
        "xlabel",
        "ylabel",
        "theme",
        "build",
    ]
    assert console.output == ["plot", "plot"]
    assert console.options == [{"soft_wrap": True}, {"soft_wrap": True}]


def test_plotext6_figure_api_is_used() -> None:
    plotext = _Plotext6()
    console = _Console()
    render_histogram(
        plotext,
        [1, 2],
        console=console,
        bins=2,
        title="Histogram",
        xlabel="X",
        ylabel="Y",
        theme="dark",
    )
    render_line(
        plotext,
        [1, 2],
        [0.5, 0.25],
        console=console,
        label="loss",
        title="Loss",
        xlabel="Step",
        ylabel="Loss",
        theme="dark",
    )

    names = [name for name, _, _ in plotext.figure.calls]
    assert names == [
        "clear",
        "hist",
        "draw",
        "title",
        "label",
        "label",
        "theme",
        "build",
        "clear",
        "signal",
        "draw",
        "title",
        "label",
        "label",
        "theme",
        "build",
    ]
    assert console.output == ["plot", "plot"]
    assert console.options == [{"soft_wrap": True}, {"soft_wrap": True}]


def test_installed_plotext_runtime_renders_histogram_and_line() -> None:
    """Exercise the installed package, not only the two contract doubles."""
    import plotext

    major = Version(distribution_version("plotext")).major
    assert major in {5, 6}
    runtime = plotext.figure if major == 6 else plotext
    assert callable(runtime.build)
    console = _Console()
    render_histogram(
        plotext,
        [1, 2, 3],
        console=console,
        bins=2,
        title="Histogram",
        xlabel="X",
        ylabel="Y",
        theme="dark",
    )
    render_line(
        plotext,
        [1, 2, 3],
        [0.5, 0.25, 0.125],
        console=console,
        label="loss",
        title="Loss",
        xlabel="Step",
        ylabel="Loss",
        theme="dark",
    )
    built = runtime.build()
    assert "Loss" in str(built)
    assert "Histogram" in console.output[0]
    assert "Loss" in console.output[1]


def test_rich_console_controls_plot_color(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    plotext = _Plotext5()
    terminal_output = StringIO()
    terminal = Console(file=terminal_output, force_terminal=True, color_system="standard")

    render_histogram(
        plotext,
        [1, 2],
        console=terminal,
        bins=2,
        title="Histogram",
        xlabel="X",
        ylabel="Y",
    )

    assert "\x1b[" in terminal_output.getvalue()


def test_rich_console_does_not_wrap_plot_rows() -> None:
    plotext = _Plotext5()
    plotext.build = lambda: "123456\nabcdef"
    redirected_output = StringIO()
    redirected = Console(file=redirected_output, width=4, color_system=None)

    render_histogram(
        plotext,
        [1, 2],
        console=redirected,
        bins=2,
        title="Histogram",
        xlabel="X",
        ylabel="Y",
    )

    assert redirected_output.getvalue().splitlines() == ["123456", "abcdef"]
