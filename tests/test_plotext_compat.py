"""Plotext 5.x / 6.x compatibility coverage."""

from importlib.metadata import version as distribution_version

from packaging.version import Version

from soup_cli.utils.plotext_compat import render_histogram, render_line


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return self


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

    def show(self):
        return self._record("show")


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

    def show(self):
        return self._record("show")


class _Plotext6:
    def __init__(self) -> None:
        self.figure = _Figure6()


def test_plotext5_module_api_remains_supported() -> None:
    plotext = _Plotext5()
    render_histogram(
        plotext,
        [1, 2],
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
        "show",
        "clf",
        "plot",
        "title",
        "xlabel",
        "ylabel",
        "theme",
        "show",
    ]


def test_plotext6_figure_api_is_used() -> None:
    plotext = _Plotext6()
    render_histogram(
        plotext,
        [1, 2],
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
        "show",
        "clear",
        "signal",
        "draw",
        "title",
        "label",
        "label",
        "theme",
        "show",
    ]


def test_installed_plotext_runtime_renders_histogram_and_line(monkeypatch) -> None:
    """Exercise the installed package, not only the two contract doubles."""
    import plotext

    major = Version(distribution_version("plotext")).major
    assert major in {5, 6}
    runtime = plotext.figure if major == 6 else plotext
    assert callable(runtime.show)
    monkeypatch.setattr(runtime, "show", lambda: None)
    render_histogram(
        plotext,
        [1, 2, 3],
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
        label="loss",
        title="Loss",
        xlabel="Step",
        ylabel="Loss",
        theme="dark",
    )
    built = runtime.build()
    assert "Loss" in str(built)
