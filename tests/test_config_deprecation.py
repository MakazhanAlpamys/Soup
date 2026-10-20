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

#: The one file allowed to name the release besides ``deprecation.py``: it declares its own
#: deadline (#808). Matched on the exact relative path, so a second file of the same name
#: elsewhere in ``src/`` is scanned like any other.
STAGED_FIELDS = "src/soup_cli/config/staged_fields.py"
DEPRECATION = "src/soup_cli/config/deprecation.py"


def _major_minor(version: str) -> tuple[int, int]:
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()
    return int(major), int(minor)


def _source_files():
    """Every ``src/**/*.py`` file as ``(posix path relative to the repo, text)``."""
    for path in sorted((ROOT / "src").rglob("*.py")):
        yield path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8")


#: A deadline spelled out as a literal release number. ``was removed in v0.36.0`` is
#: history, not a promise, so past tense is left alone.
_HARDCODED_DEADLINE = (
    re.compile(r"v\d+\.\d+ will refuse"),
    re.compile(r"(?<!was )removed in (?:Soup )?v\d+\.\d+"),
)


@pytest.fixture
def printed(monkeypatch):
    """Capture what the loader prints, uncoloured and unwrapped."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


class TestTheVersionHelper:
    """``_major_minor`` decides when the deadline test turns red, so its edges are pinned."""

    def test_a_patch_release_is_its_minor(self):
        assert _major_minor("0.76.1") == (0, 76)
        assert _major_minor("0.76.0") == (0, 76)

    def test_minors_compare_as_numbers_not_as_text(self):
        """The string "0.100" sorts before "0.77"; the tuple must not."""
        assert _major_minor("0.100") == (0, 100)
        assert _major_minor("0.100") > _major_minor("0.77")
        assert _major_minor("0.100.0") > _major_minor("0.77")

    def test_a_patch_never_reaches_the_next_minor(self):
        assert _major_minor("0.76.9") < _major_minor("0.77")


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
        # STAGED_FIELDS declares its own deadline (#808), which is allowed to name the
        # same release; it is the one file exempt from this scan, by exact path.
        holders = sorted(
            relative
            for relative, text in _source_files()
            if relative != STAGED_FIELDS and needle.search(text)
        )
        assert holders == [DEPRECATION], holders

    def test_the_published_promise_is_kept(self):
        """v0.76.0 told users this version in its warnings, its ``--help`` text and its
        changelog, so moving it is a breach of that promise: it needs a changelog entry
        and a conscious edit of this test, not a quiet change of the constant."""
        assert DEPRECATED_VALUE_REJECTION_VERSION == "0.77"

    def test_deadline_clause_is_exact(self):
        """The sentence every warning ends with is built from the constant, word for word."""
        assert deprecation.deadline_clause() == (
            f"Soup v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it."
        )

    def test_no_hardcoded_refusal_version_in_src(self):
        """A deadline written as prose cannot follow the constant: this is how a warning
        keeps promising a refusal after the refusal shipped. Everything outside the two
        files that own a deadline builds the sentence from the constant."""
        offenders = sorted(
            f"{relative}: {match.group(0)!r}"
            for relative, text in _source_files()
            if relative not in (DEPRECATION, STAGED_FIELDS)
            for pattern in _HARDCODED_DEADLINE
            for match in pattern.finditer(text)
        )
        assert offenders == [], offenders

    def test_the_hardcoded_deadline_scan_sees_what_it_is_meant_to_see(self):
        """Control: each pattern matches the prose it names and ignores past tense."""
        spoken = (
            "Soup v0.76 will refuse it.",
            "set `quantization: none`; v1.2 will refuse it",
            "awq will be removed in v0.77",
            "gptq is removed in Soup v0.77",
        )
        for text in spoken:
            assert any(pattern.search(text) for pattern in _HARDCODED_DEADLINE), text
        # A fact about the past is not a promise: sft_format.py says so about v0.36.0.
        assert not any(
            pattern.search("the fallback was removed in v0.36.0.")
            for pattern in _HARDCODED_DEADLINE
        )
        # The sentence the code builds is made of the constant, not of digits.
        assert not any(
            pattern.search('f"Soup v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it."')
            for pattern in _HARDCODED_DEADLINE
        )


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
