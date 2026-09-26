"""The warn-then-refuse staging for config values Soup itself used to write (#759, #795).

A value that never took effect is refused eventually, but not at once: ``soup
autopilot`` and every run-config dump write the whole model, defaults included,
so refusing a former default breaks files Soup wrote itself. For one release the
value loads with a warning naming the release that will refuse it.

These tests cover the mechanism. Each deprecated value has its own tests beside
the field it belongs to.
"""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path

import pytest
from rich.console import Console

import soup_cli
from soup_cli.config import deprecation, loader
from soup_cli.config.deprecation import (
    DEPRECATED_VALUE_REJECTION_VERSION,
    SoupConfigDeprecationWarning,
    warn_deprecated_value,
)

ROOT = Path(__file__).resolve().parents[1]


def _major_minor(version: str) -> tuple[int, int]:
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()
    return int(major), int(minor)


@pytest.fixture
def printed(monkeypatch):
    """Capture what the loader prints, uncoloured and unwrapped."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


class TestTheWarning:
    def test_it_is_a_future_warning_that_names_the_release(self):
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            warn_deprecated_value("x: y is ignored.")
        assert issubclass(SoupConfigDeprecationWarning, FutureWarning)
        message = str(record[0].message)
        assert message.startswith("x: y is ignored.")
        assert f"v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it" in message


class TestTheDeadline:
    """A warning with no expiry is permanent, so the expiry is checked (#627's shape)."""

    def test_the_deadline_has_not_passed(self):
        """The release that reaches the deadline turns this red: flip each
        deprecated value to a refusal, then move or retire the constant."""
        assert _major_minor(soup_cli.__version__) < _major_minor(
            DEPRECATED_VALUE_REJECTION_VERSION
        ), (
            f"soup_cli is {soup_cli.__version__}, and the deprecated config values "
            f"were promised a refusal in v{DEPRECATED_VALUE_REJECTION_VERSION}"
        )

    def test_the_deadline_is_the_next_minor(self):
        """One release of notice: not zero, and not an open-ended future."""
        major, minor = _major_minor(soup_cli.__version__)
        assert _major_minor(DEPRECATED_VALUE_REJECTION_VERSION) == (major, minor + 1)

    def test_the_version_is_written_out_in_exactly_one_source_file(self):
        needle = re.compile(
            rf'"{re.escape(DEPRECATED_VALUE_REJECTION_VERSION)}"'
            rf"|v{re.escape(DEPRECATED_VALUE_REJECTION_VERSION)}\b"
        )
        holders = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "src").rglob("*.py")
            if needle.search(path.read_text(encoding="utf-8"))
        )
        assert holders == ["src/soup_cli/config/deprecation.py"], holders


class TestTheLoader:
    def _fake_config(self, *emits):
        def build(**_raw):
            for category, message in emits:
                warnings.warn(message, category)
            return "built"

        return build

    @pytest.mark.parametrize("load", ["string", "file"])
    def test_each_deprecation_prints_once_per_load(self, monkeypatch, printed, tmp_path, load):
        """Validators can run more than once per load; the user reads one line."""
        text = f"a is ignored. {deprecation.deadline_clause()}"
        monkeypatch.setattr(
            loader,
            "SoupConfig",
            self._fake_config(
                (SoupConfigDeprecationWarning, text), (SoupConfigDeprecationWarning, text)
            ),
        )
        if load == "string":
            assert loader.load_config_from_string("base: m\n") == "built"
        else:
            path = tmp_path / "soup.yaml"
            path.write_text("base: m\n", encoding="utf-8")
            assert loader.load_config(path) == "built"
        assert printed.getvalue().count("a is ignored.") == 1, printed.getvalue()
        assert "Warning:" in printed.getvalue()

    def test_it_prints_again_on_the_next_load(self, monkeypatch, printed):
        """Once per LOAD, not once per process: the default warnings filter
        would hide the second one."""
        monkeypatch.setattr(
            loader, "SoupConfig", self._fake_config((SoupConfigDeprecationWarning, "a."))
        )
        loader.load_config_from_string("base: m\n")
        loader.load_config_from_string("base: m\n")
        assert printed.getvalue().count("a.") == 2

    def test_other_warnings_pass_through_untouched(self, monkeypatch, printed):
        monkeypatch.setattr(
            loader, "SoupConfig", self._fake_config((UserWarning, "not ours"))
        )
        with pytest.warns(UserWarning, match="not ours"):
            loader.load_config_from_string("base: m\n")
        assert "not ours" not in printed.getvalue()

    def test_a_refusal_still_refuses_after_printing_the_warnings(self, monkeypatch, printed):
        def build(**_raw):
            warnings.warn("b is ignored.", SoupConfigDeprecationWarning)
            raise ValueError("refused")

        monkeypatch.setattr(loader, "SoupConfig", build)
        with pytest.raises(ValueError, match="refused"):
            loader.load_config_from_string("base: m\n")
        assert "b is ignored." in printed.getvalue()

    def test_nothing_is_printed_without_a_deprecation(self, monkeypatch, printed):
        """Control: the loader does not invent warnings."""
        monkeypatch.setattr(loader, "SoupConfig", self._fake_config())
        loader.load_config_from_string("base: m\n")
        assert printed.getvalue() == ""
