"""The warn-then-refuse staging for config values Soup itself used to write (#759, #795).

A value that never took effect is refused eventually, but not at once: ``soup
autopilot`` and every run-config dump write the whole model, defaults included,
so refusing a former default breaks files Soup wrote itself. For one release
(v0.76) the value loaded with a warning naming the release that would refuse it;
v0.77.0 reached that release and refuses it, together with the staged fields of
#808.

These tests cover the mechanism, the warning path included (it is kept for the
next value that needs a release of notice, and reached here by monkeypatching the
switch). Each deprecated value has its own tests beside the field it belongs to.
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
from soup_cli.config.staged_fields import STAGED_FIELD_REJECTION_VERSION

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
    def test_it_is_a_future_warning_that_names_the_release(self, monkeypatch):
        """The warning path, kept for the next deprecation; this release refuses."""
        monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", "warn")
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            warn_deprecated_value("x: y is ignored.")
        assert issubclass(SoupConfigDeprecationWarning, FutureWarning)
        message = str(record[0].message)
        assert message.startswith("x: y is ignored.")
        assert f"v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it" in message

    def test_this_release_refuses_instead(self):
        """Control: with the switch as shipped, the same call is a refusal."""
        with pytest.raises(ValueError) as excinfo:
            warn_deprecated_value("x: y is ignored.", refusal="x: y is refused.")
        assert str(excinfo.value) == (
            f"x: y is refused. Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
        )


#: Each deadline beside the switch that must flip when the declared version reaches it.
_SWITCHES = (
    ("DEPRECATED_VALUE_SEVERITY", DEPRECATED_VALUE_REJECTION_VERSION),
    ("STAGED_FIELD_SEVERITY", STAGED_FIELD_REJECTION_VERSION),
)


def _switch(name: str) -> str:
    owner = deprecation if name == "DEPRECATED_VALUE_SEVERITY" else loader
    return getattr(owner, name)


class TestTheDeadline:
    """A warning with no expiry is permanent, so the expiry is checked (#627's shape)."""

    def test_the_deadline_has_passed_and_both_switches_refuse(self):
        """v0.76 warned and named v0.77; v0.77.0 reached it, so both the deprecated
        values (#759, #795, #1418) and the staged fields (#808) are refused. Setting
        either switch back to ``"warn"`` turns this red."""
        version = _major_minor(soup_cli.__version__)
        for name, deadline in _SWITCHES:
            assert version >= _major_minor(deadline), (
                f"soup_cli is {soup_cli.__version__}, before the v{deadline} deadline "
                f"{name} names; this release was meant to refuse"
            )
            assert _switch(name) == "error", (
                f"soup_cli is {soup_cli.__version__}, at or past the v{deadline} "
                f"deadline: {name} must be 'error', not {_switch(name)!r}"
            )

    @pytest.mark.parametrize(("name", "deadline"), _SWITCHES)
    def test_each_switch_refuses_exactly_from_its_deadline(self, name, deadline):
        """Both directions, as ``UNKNOWN_KEY_SEVERITY`` is pinned: a switch cannot be
        forgotten past its deadline, nor flipped before it. One commit flips both
        (ruling 2026-10-12), so both deadlines also name the same release."""
        reached = _major_minor(soup_cli.__version__) >= _major_minor(deadline)
        assert _switch(name) == ("error" if reached else "warn"), (
            f"soup_cli is {soup_cli.__version__} and {name} names v{deadline}, "
            f"but the switch is {_switch(name)!r}"
        )
        assert deadline == DEPRECATED_VALUE_REJECTION_VERSION

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

    def test_deadline_clause_is_exact(self, monkeypatch):
        """The sentence every refusal ends with is built from the constant, word for word.
        Pinned under the severity it describes, so it does not depend on the release's
        switch (``TestTheDeadline``'s first test pins that)."""
        monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", "error")
        assert deprecation.deadline_clause() == (
            f"Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
        )

    def test_deadline_clause_while_warning_is_exact(self, monkeypatch):
        """The sentence v0.76's warnings ended with, still built from the constant."""
        monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", "warn")
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


def _release(version: str) -> tuple[int, int, int]:
    """``"0.77.0"`` as ``(0, 77, 0)``; a version written without a patch reads as ``.0``."""
    major, minor, patch = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version).groups()
    return int(major), int(minor), int(patch or 0)


class TestTheExportRemovalRelease:
    """``soup export --format awq`` / ``gptq`` name the release that removed them (#1710).

    That release is a fact about the past, so it has its own constant. Built from
    :data:`DEPRECATED_VALUE_REJECTION_VERSION` it would move with the next deprecation's
    deadline, and the refusal would date the removal to a release that still shipped
    both formats."""

    def test_the_release_helper_compares_whole_releases(self):
        assert _release("0.77.0") == (0, 77, 0)
        assert _release("0.77") == (0, 77, 0)
        assert _release("0.76.9") < _release("0.77.0") < _release("0.77.1")
        assert _release("0.100.0") > _release("0.77.0")

    def test_the_removal_release_is_written_out(self):
        """The release that shipped the removal (#1951 merged before the v0.77.0 cut).
        Changing it rewrites history in every refusal: it needs a changelog entry and a
        conscious edit of this test."""
        assert deprecation.EXPORT_FORMATS_REMOVED_IN == "0.77.0"

    def test_this_version_is_at_or_past_the_removal_release(self):
        """Pinned to ``__version__`` as the deadlines above are: a version before the
        removal release would name, in the refusal, a release that has not shipped."""
        removed_in = deprecation.EXPORT_FORMATS_REMOVED_IN
        assert _release(soup_cli.__version__) >= _release(removed_in), (
            f"soup_cli is {soup_cli.__version__}, before v{removed_in}, the release the "
            "awq/gptq refusal says removed them"
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
