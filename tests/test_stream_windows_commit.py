"""Layer streaming's staging pre-flight on Windows: commit headroom, not just free RAM.

On Windows page-locked host memory is charged against the system COMMIT limit
(RAM + pagefile). An experimental Mistral Large 123B run on a 32 GB laptop was
stopped twice, once at the harness's commit floor, with pinned staging; the same
run with pageable staging passed. Soup's staging pre-flight
(``_validate_stream_staging_ram_fit``) reads free RAM only, so nothing noticed.

``utils/windows_commit.py`` adds the missing check, Windows only, and acts on it:

* ``stream_pin`` unset and pinned staging would leave commit headroom under the
  margin -> stage pageable and say so in the panel;
* ``stream_pin: true`` and the staging does not fit the commit limit -> refuse,
  naming the numbers and the fix;
* ``stream_pin: false`` -> not read, not touched;
* nothing on another OS, nothing off CUDA, and a failed read skips the check.

The numbers in the table are the ones the experimental run recorded: commit limit
54,471,286,784 B, 12.7 to 17.9 GB of headroom at the start of a run, 2,470,391,808 B
at the lowest point of the 512-token run, a 4,294,967,296 B pinned arena and a
2,324,713,472 B pageable backing (714,080,256 + 2 x 805,310,464 + 3 x 4096).

Nothing here makes a real ctypes call except ``TestTheRealReader``, which runs only
on Windows (every cell of the Windows CI fleet, and the maintainer's box).
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import re
import sys
from pathlib import Path

import pytest

from soup_cli.utils import windows_commit
from soup_cli.utils.layer_stream import TIER_DISK, TIER_RAM, StreamPlan
from soup_cli.utils.windows_commit import (
    COMMIT_HEADROOM_MARGIN_BYTES,
    CommitFit,
    CommitInfo,
    apply_commit_preflight,
    assess_commit_fit,
)

#: The real reader, taken at import time: an autouse fixture in conftest.py replaces
#: ``windows_commit.read_commit_info`` with a neutral one for every test, so a test of
#: the real wrapper has to put this back itself.
_REAL_READ_COMMIT_INFO = windows_commit.read_commit_info

_COMMIT_LIMIT = 54_471_286_784
_HEADROOM_AT_START_LOW = 12_700_000_000
_HEADROOM_AT_START_HIGH = 17_900_000_000
_HEADROOM_AT_MINIMUM = 2_470_391_808
_PINNED_ARENA = 4_294_967_296
_PAGEABLE_BACKING = 2_324_713_472
_VOCAB_SLOT = 805_310_464

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed output (Rich colours flag names on Linux CI)."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _gb(value: int) -> str:
    return f"{value / 1e9:.2f} GB"


def _info(available: int, limit: int = _COMMIT_LIMIT) -> CommitInfo:
    return CommitInfo(total_bytes=limit - available, limit_bytes=limit)


def _plan(staging_bytes: int, *, pinned: bool = True, tier: str = TIER_DISK) -> StreamPlan:
    return StreamPlan(
        arch="llama",
        tier=tier,
        n_layers=88,
        layer_bytes=714_080_256,
        store_bytes=0,
        embed_bytes=_VOCAB_SLOT,
        large_store_bytes=2 * _VOCAB_SLOT,
        large_buffer_bytes=_VOCAB_SLOT,
        buffers=2,
        buffer_bytes=0,
        pinned=pinned,
        notes=("an earlier note",),
        read_ahead=2,
        staging_bytes=staging_bytes if tier == TIER_DISK else 0,
    )


@pytest.fixture
def windows_box(monkeypatch):
    """A pretend Windows box: ``windows_box(available)`` sets the free commit it reports.

    Returns the list of reads, so a test can say the reader was (or was not) called.
    """
    reads: list[int] = []

    def set_available(available: int | None):
        def reader():
            reads.append(1)
            return None if available is None else _info(available)

        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        monkeypatch.setattr(windows_commit, "read_commit_info", reader)
        return reads

    return set_available


def _reader_must_not_run(monkeypatch) -> None:
    def boom():
        raise AssertionError("the commit reader must not be called here")

    monkeypatch.setattr(windows_commit, "read_commit_info", boom)


# ---------------------------------------------------------------------------
# 1. the pure function
# ---------------------------------------------------------------------------
class TestAssessCommitFit:
    """fits / tight / does not fit, from commit total + limit, staging and extras."""

    @pytest.mark.parametrize(
        ("headroom", "staging", "extras", "expected"),
        [
            # 17.9 GB free at the start, the 4 GiB pinned arena + one vocab slot: ~12.8 GB left.
            (_HEADROOM_AT_START_HIGH, _PINNED_ARENA, _VOCAB_SLOT, CommitFit.FITS),
            # 12.7 GB free at the start, same staging: ~7.6 GB left, under the margin.
            (_HEADROOM_AT_START_LOW, _PINNED_ARENA, _VOCAB_SLOT, CommitFit.TIGHT),
            # The pageable backing the passing runs used, on the same 12.7 GB box.
            (_HEADROOM_AT_START_LOW, _PAGEABLE_BACKING, _VOCAB_SLOT, CommitFit.FITS),
            # A box already at the run's lowest point (2.47 GB) cannot stage 3.1 GB at all.
            (_HEADROOM_AT_MINIMUM, _PAGEABLE_BACKING, _VOCAB_SLOT, CommitFit.DOES_NOT_FIT),
            # ... and with nothing resident to add it is positive but thin.
            (_HEADROOM_AT_MINIMUM, _PAGEABLE_BACKING, 0, CommitFit.TIGHT),
        ],
        ids=["start-high-pinned", "start-low-pinned", "start-low-pageable", "minimum", "thin"],
    )
    def test_the_recorded_numbers(self, headroom, staging, extras, expected):
        info = _info(headroom)
        verdict = assess_commit_fit(
            commit_total_bytes=info.total_bytes,
            commit_limit_bytes=info.limit_bytes,
            pinned_staging_bytes=staging,
            resident_extras_bytes=extras,
        )
        assert verdict.fit is expected, verdict
        assert verdict.available_bytes == headroom
        assert verdict.required_bytes == staging + extras
        assert verdict.headroom_after_bytes == headroom - staging - extras
        assert verdict.margin_bytes == COMMIT_HEADROOM_MARGIN_BYTES

    def test_the_margin_separates_the_two_recorded_starts(self):
        """The constant sits between the 7.6 GB and 12.8 GB that the two starts leave."""
        low_left = _HEADROOM_AT_START_LOW - _PINNED_ARENA - _VOCAB_SLOT
        high_left = _HEADROOM_AT_START_HIGH - _PINNED_ARENA - _VOCAB_SLOT
        assert low_left < COMMIT_HEADROOM_MARGIN_BYTES <= high_left

    @pytest.mark.parametrize(
        ("left", "expected"),
        [
            (COMMIT_HEADROOM_MARGIN_BYTES + 1, CommitFit.FITS),
            (COMMIT_HEADROOM_MARGIN_BYTES, CommitFit.FITS),
            (COMMIT_HEADROOM_MARGIN_BYTES - 1, CommitFit.TIGHT),
            (1, CommitFit.TIGHT),
            (0, CommitFit.TIGHT),
            (-1, CommitFit.DOES_NOT_FIT),
        ],
        ids=["above", "at-margin", "just-under", "one-byte", "exactly-nothing", "one-over"],
    )
    def test_the_boundaries(self, left, expected):
        staging = 3_000_000_000
        extras = 500_000_000
        available = staging + extras + left
        verdict = assess_commit_fit(
            commit_total_bytes=1_000,
            commit_limit_bytes=1_000 + available,
            pinned_staging_bytes=staging,
            resident_extras_bytes=extras,
        )
        assert verdict.headroom_after_bytes == left
        assert verdict.fit is expected

    def test_resident_extras_are_charged_not_ignored(self):
        """A mutant dropping the extras flips this from TIGHT to FITS."""
        available = COMMIT_HEADROOM_MARGIN_BYTES + 1_000_000
        common = dict(
            commit_total_bytes=0,
            commit_limit_bytes=available,
            pinned_staging_bytes=0,
        )
        assert assess_commit_fit(**common).fit is CommitFit.FITS
        assert assess_commit_fit(**common, resident_extras_bytes=2_000_000).fit is CommitFit.TIGHT

    def test_a_total_over_the_limit_leaves_nothing_not_a_negative_pool(self):
        """Commit can be transiently over the limit; that is 0 free, not -N free."""
        verdict = assess_commit_fit(
            commit_total_bytes=60_000_000_000,
            commit_limit_bytes=50_000_000_000,
            pinned_staging_bytes=1_000,
        )
        assert verdict.available_bytes == 0
        assert verdict.fit is CommitFit.DOES_NOT_FIT

    def test_the_margin_is_a_parameter(self):
        kwargs = dict(
            commit_total_bytes=0,
            commit_limit_bytes=10_000,
            pinned_staging_bytes=4_000,
        )
        assert assess_commit_fit(**kwargs, margin_bytes=6_000).fit is CommitFit.FITS
        assert assess_commit_fit(**kwargs, margin_bytes=6_001).fit is CommitFit.TIGHT

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("commit_total_bytes", -1),
            ("commit_limit_bytes", -1),
            ("pinned_staging_bytes", -1),
            ("resident_extras_bytes", -1),
            ("margin_bytes", -1),
            ("pinned_staging_bytes", True),
            ("commit_limit_bytes", 1.5),
        ],
    )
    def test_bad_inputs_name_the_argument(self, field, value):
        kwargs = dict(
            commit_total_bytes=1,
            commit_limit_bytes=10,
            pinned_staging_bytes=1,
            resident_extras_bytes=0,
            margin_bytes=0,
        )
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            assess_commit_fit(**kwargs)


# ---------------------------------------------------------------------------
# 2. reading the numbers: lazy, isolated, never fatal
# ---------------------------------------------------------------------------
class TestReadCommitInfo:
    def test_the_pages_are_multiplied_by_the_page_size(self):
        info = windows_commit._commit_info_from_pages(
            total_pages=1_000, limit_pages=13_297_000, page_size=4096
        )
        assert info == CommitInfo(total_bytes=4_096_000, limit_bytes=54_464_512_000)
        # The recorded limit is an exact multiple of the 4 KiB page.
        assert _COMMIT_LIMIT % 4096 == 0
        assert windows_commit._commit_info_from_pages(
            total_pages=0, limit_pages=_COMMIT_LIMIT // 4096, page_size=4096
        ).limit_bytes == _COMMIT_LIMIT

    def test_available_is_the_limit_less_the_total(self):
        assert _info(12_700_000_000).available_bytes == 12_700_000_000
        assert CommitInfo(total_bytes=9, limit_bytes=5).available_bytes == 0

    def test_off_windows_nothing_is_loaded_or_read(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: False)
        reached: list[int] = []

        def query():
            # Recorded, not raised: the wrapper swallows every Exception, so a raise
            # here would read as "failed read" and pass for the wrong reason.
            reached.append(1)
            return _info(_HEADROOM_AT_START_LOW)

        monkeypatch.setattr(windows_commit, "_query_commit_windows", query)
        assert _REAL_READ_COMMIT_INFO() is None
        assert reached == [], "ctypes must not be reached off Windows"

    @pytest.mark.parametrize(
        "error",
        [OSError("no psapi"), AttributeError("no windll"), ValueError("bad"), RuntimeError("x")],
        ids=lambda error: type(error).__name__,
    )
    def test_a_failed_read_is_none_with_a_debug_note(self, monkeypatch, caplog, error):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)

        def fail():
            raise error

        monkeypatch.setattr(windows_commit, "_query_commit_windows", fail)
        with caplog.at_level(logging.DEBUG, logger=windows_commit.__name__):
            assert _REAL_READ_COMMIT_INFO() is None
        notes = [rec for rec in caplog.records if rec.name == windows_commit.__name__]
        assert notes and all(rec.levelno == logging.DEBUG for rec in notes)
        assert "commit" in notes[0].getMessage().lower()

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_nonsense_limit_is_a_failed_read(self, monkeypatch, limit):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        monkeypatch.setattr(
            windows_commit,
            "_query_commit_windows",
            lambda: CommitInfo(total_bytes=5, limit_bytes=limit),
        )
        assert _REAL_READ_COMMIT_INFO() is None

    def test_a_good_read_is_returned_as_is(self, monkeypatch):
        good = _info(_HEADROOM_AT_START_LOW)
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        monkeypatch.setattr(windows_commit, "_query_commit_windows", lambda: good)
        assert _REAL_READ_COMMIT_INFO() == good

    def test_the_platform_check_follows_sys_platform(self, monkeypatch):
        for platform, expected in (("win32", True), ("linux", False), ("darwin", False)):
            monkeypatch.setattr(sys, "platform", platform)
            assert windows_commit._on_windows() is expected, platform

    def test_ctypes_is_not_imported_at_module_level(self):
        """Lazy by construction (the runtime proof is test_cli_startup_is_light)."""
        tree = ast.parse(Path(windows_commit.__file__).read_text(encoding="utf-8"))
        top_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        assert "ctypes" not in top_level
        assert "torch" not in top_level

    def test_other_tests_never_read_the_hosts_real_counters(self):
        """conftest.py neutralises the reader, so no test depends on this box's commit."""
        assert windows_commit.read_commit_info() is None


@pytest.mark.skipif(sys.platform != "win32", reason="GetPerformanceInfo exists on Windows only")
class TestTheRealReader:
    """The one real ctypes call: the structure layout is only proven by the OS."""

    def test_the_system_reports_a_sane_commit(self):
        info = windows_commit._query_commit_windows()
        assert info.limit_bytes > 0
        assert info.total_bytes > 0
        # Any box has at least a gigabyte of commit limit; page counts, not bytes, would not.
        assert info.limit_bytes >= 2**30
        assert info.limit_bytes % 4096 == 0

    def test_the_wrapper_returns_the_same_shape(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        info = _REAL_READ_COMMIT_INFO()
        assert isinstance(info, CommitInfo)
        assert info.limit_bytes >= 2**30


# ---------------------------------------------------------------------------
# 3. the decision on a plan
# ---------------------------------------------------------------------------
def _apply(plan, *, stream_pin, on_cuda=True, resident=_VOCAB_SLOT):
    said: list[str] = []
    out = apply_commit_preflight(
        plan,
        stream_pin=stream_pin,
        on_cuda=on_cuda,
        resident_bytes=resident,
        notify=said.append,
    )
    return out, said


class TestAutoFallsBackToPageable:
    def test_a_tight_headroom_stages_pageable_and_says_so_once(self, windows_box):
        windows_box(_HEADROOM_AT_START_LOW)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=None)
        assert out.pinned is False
        assert plan.pinned is True, "the plan that came in must not be mutated"
        assert out.staging_bytes == plan.staging_bytes
        assert out.tier == TIER_DISK
        added = out.notes[len(plan.notes):]
        assert out.notes[: len(plan.notes)] == plan.notes
        assert len(added) == 1, added
        line = _plain(added[0])
        assert "commit" in line.lower()
        assert "pageable" in line
        assert "training.stream_pin: true" in line
        assert _gb(_PINNED_ARENA) in line
        assert _gb(_HEADROOM_AT_START_LOW) in line
        assert _gb(COMMIT_HEADROOM_MARGIN_BYTES) in line
        assert said == [], "the panel line is the one statement; nothing else is printed"

    def test_a_staging_that_cannot_fit_also_stages_pageable(self, windows_box):
        windows_box(_HEADROOM_AT_MINIMUM)
        out, said = _apply(_plan(_PAGEABLE_BACKING), stream_pin=None)
        assert out.pinned is False
        line = _plain(out.notes[-1])
        # 2.47 GB free against 3.13 GB charged: said as 0.66 GB over, not as -0.66 GB left.
        assert "(the charge is 0.66 GB over)" in line, line
        assert "would leave none of" in line, line
        assert not re.search(r"-\d", line), line
        assert said == []

    def test_headroom_above_the_margin_keeps_the_pinned_default(self, windows_box):
        reads = windows_box(_HEADROOM_AT_START_HIGH)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=None)
        assert out is plan
        assert out.pinned is True
        assert said == []
        assert len(reads) == 1


class TestExplicitPinIsHonouredOrRefused:
    def test_a_staging_that_does_not_fit_is_refused_with_the_numbers_and_the_fix(
        self, windows_box
    ):
        windows_box(_HEADROOM_AT_MINIMUM)
        with pytest.raises(ValueError) as caught:
            _apply(_plan(_PINNED_ARENA), stream_pin=True)
        message = _plain(str(caught.value))
        assert "training.stream_pin=true" in message
        assert "commit" in message.lower()
        assert _gb(_PINNED_ARENA) in message
        assert _gb(_VOCAB_SLOT) in message
        assert _gb(_HEADROOM_AT_MINIMUM) in message
        assert "Refusing before any weight is staged" in message
        assert "training.stream_pin: false" in message
        assert "pagefile" in message
        assert "close programs" in message

    def test_a_tight_headroom_is_honoured_with_one_warning(self, windows_box):
        windows_box(_HEADROOM_AT_START_LOW)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=True)
        assert out is plan
        assert out.pinned is True
        assert len(said) == 1
        line = _plain(said[0])
        assert "training.stream_pin=true" in line
        assert "training.stream_pin: false" in line
        assert _gb(COMMIT_HEADROOM_MARGIN_BYTES) in line

    def test_a_fitting_staging_is_silent(self, windows_box):
        windows_box(_HEADROOM_AT_START_HIGH)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=True)
        assert out is plan
        assert said == []

    def test_a_refusal_is_a_value_error_like_the_ram_refusal(self, windows_box):
        """The setup surfaces both the same way, so the user sees one kind of stop."""
        windows_box(1_000)
        with pytest.raises(ValueError):
            _apply(_plan(_PINNED_ARENA), stream_pin=True)


class TestWhatIsLeftAlone:
    def test_stream_pin_false_is_unchanged_and_never_reads(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        # pinned=True on purpose: an explicit false is left alone by the check itself,
        # not only because the plan it was handed happens to be pageable already.
        plan = _plan(_PINNED_ARENA, pinned=True)
        out, said = _apply(plan, stream_pin=False)
        assert out is plan
        assert said == []

    @pytest.mark.parametrize("stream_pin", [None, True, False])
    def test_off_windows_the_reader_is_never_called(self, monkeypatch, stream_pin):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: False)
        _reader_must_not_run(monkeypatch)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=stream_pin)
        assert out is plan
        assert said == []

    @pytest.mark.parametrize("stream_pin", [None, True])
    def test_without_cuda_nothing_is_page_locked_so_nothing_is_read(
        self, monkeypatch, stream_pin
    ):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=stream_pin, on_cuda=False)
        assert out is plan
        assert said == []

    def test_a_plan_that_is_already_pageable_is_not_second_guessed(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        plan = _plan(_PINNED_ARENA, pinned=False)
        out, said = _apply(plan, stream_pin=None)
        assert out is plan
        assert said == []

    def test_the_ram_tier_has_no_staging_to_charge(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        plan = _plan(0, tier=TIER_RAM)
        out, said = _apply(plan, stream_pin=None)
        assert out is plan
        assert said == []

    @pytest.mark.parametrize("stream_pin", [None, True])
    def test_a_failed_read_skips_the_check_and_trains_on(self, windows_box, caplog, stream_pin):
        reads = windows_box(None)
        plan = _plan(_PINNED_ARENA)
        out, said = _apply(plan, stream_pin=stream_pin)
        assert out is plan
        assert out.pinned is True
        assert said == []
        assert len(reads) == 1

    def test_the_staging_is_what_the_plan_says_not_a_re_derivation(self, windows_box):
        """Same figure the RAM check is given, so the two checks cannot disagree on it."""
        windows_box(_HEADROOM_AT_START_HIGH)
        small = _plan(1_000_000)
        assert _apply(small, stream_pin=None)[0] is small
        big = dataclasses.replace(small, staging_bytes=_HEADROOM_AT_START_HIGH)
        assert _apply(big, stream_pin=None)[0].pinned is False


# ---------------------------------------------------------------------------
# 4. the real setup path: what reaches the runtime boundary
# ---------------------------------------------------------------------------
def _drive(tmp_path, monkeypatch, *, stream_pin_yaml: str = ""):
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    return _drive_wiring(
        tmp_path, monkeypatch, stream_pin_yaml=stream_pin_yaml, stream_source="disk"
    )


class TestTheSetupPath:
    def test_auto_on_a_tight_windows_box_reaches_the_runtime_as_pageable(
        self, tmp_path, monkeypatch, windows_box
    ):
        windows_box(1_000_000_000)
        captured, panel = _drive(tmp_path, monkeypatch)
        assert captured["pin"] is False
        assert captured["require_pin"] is False, "auto must not turn into a demand to page-lock"
        text = _plain(panel)
        assert text.count("Windows commit headroom") == 1, text
        assert "training.stream_pin: true" in text
        assert "pageable" in text

    def test_auto_on_a_roomy_windows_box_still_pins(self, tmp_path, monkeypatch, windows_box):
        reads = windows_box(40_000_000_000)
        captured, panel = _drive(tmp_path, monkeypatch)
        assert captured["pin"] is True
        assert "Windows commit headroom" not in _plain(panel)
        assert len(reads) == 1

    def test_explicit_pin_that_does_not_fit_stops_before_the_runtime(
        self, tmp_path, monkeypatch, windows_box
    ):
        windows_box(1_000)
        with pytest.raises(ValueError) as caught:
            _drive(tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: true\n")
        message = _plain(str(caught.value))
        assert "training.stream_pin: false" in message
        assert "commit" in message.lower()

    def test_explicit_pin_that_does_not_fit_never_builds_the_runtime(
        self, tmp_path, monkeypatch, windows_box
    ):
        windows_box(1_000)
        import soup_cli.utils.layer_stream_runtime as rt

        built = []
        monkeypatch.setattr(
            rt, "build_streamed_model", lambda **kwargs: built.append(kwargs)
        )
        with pytest.raises(ValueError):
            _drive(tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: true\n")
        assert built == []

    def test_explicit_pin_on_a_roomy_box_is_untouched(self, tmp_path, monkeypatch, windows_box):
        windows_box(40_000_000_000)
        captured, panel = _drive(tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: true\n")
        assert captured["pin"] is True
        assert captured["require_pin"] is True
        assert "Windows commit headroom" not in _plain(panel)

    def test_explicit_false_is_unchanged_and_the_reader_is_not_called(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        captured, panel = _drive(tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: false\n")
        assert captured["pin"] is False
        assert "Windows commit headroom" not in _plain(panel)

    def test_off_windows_the_default_still_pins_and_the_reader_is_not_called(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: False)
        _reader_must_not_run(monkeypatch)
        captured, panel = _drive(tmp_path, monkeypatch)
        assert captured["pin"] is True
        assert "Windows commit headroom" not in _plain(panel)

    def test_a_cpu_target_never_reads_the_commit_counters(self, tmp_path, monkeypatch):
        """Nothing is page-locked off CUDA, so there is nothing to charge to commit."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        monkeypatch.setattr(
            SFTTrainerWrapper,
            "device",
            property(lambda self: "cpu", lambda self, value: None),
            raising=False,
        )
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        captured, panel = _drive(tmp_path, monkeypatch)
        assert captured["pin"] is False
        assert "Windows commit headroom" not in _plain(panel)

    def test_the_setup_hands_over_the_pin_setting_and_the_resident_extras(
        self, tmp_path, monkeypatch
    ):
        """What the setup passes is what the check decides on."""
        import soup_cli.trainer.stream_setup  # noqa: F401  (imported by the setup itself)

        seen = {}

        def spy(plan, **kwargs):
            seen.update(kwargs)
            seen["plan"] = plan
            return plan

        monkeypatch.setattr(windows_commit, "apply_commit_preflight", spy)
        _drive(tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: true\n")
        assert seen["stream_pin"] is True
        assert seen["on_cuda"] is True
        assert seen["resident_bytes"] > 0
        assert seen["plan"].tier == TIER_DISK
        assert seen["plan"].staging_bytes > 0
        assert callable(seen["notify"])

    def test_an_unreadable_commit_does_not_stop_the_run(self, tmp_path, monkeypatch, windows_box):
        windows_box(None)
        captured, panel = _drive(tmp_path, monkeypatch)
        assert captured["pin"] is True
        assert "Windows commit headroom" not in _plain(panel)

    def test_the_ram_check_runs_first_and_is_not_replaced(
        self, tmp_path, monkeypatch, windows_box
    ):
        """The commit check is added after the free-RAM check, never instead of it."""
        import soup_cli.trainer.stream_setup as ss

        order: list[str] = []
        original = ss._validate_stream_staging_ram_fit

        def ram_check(**kwargs):
            order.append("ram")
            return original(**kwargs)

        monkeypatch.setattr(ss, "_validate_stream_staging_ram_fit", ram_check)
        reads = windows_box(1_000_000_000)
        real_reader = windows_commit.read_commit_info

        def reader():
            order.append("commit")
            return real_reader()

        monkeypatch.setattr(windows_commit, "read_commit_info", reader)
        _drive(tmp_path, monkeypatch)
        assert order == ["ram", "commit"]
        assert len(reads) == 1


class TestTheRamCheckIsUnchanged:
    """``_validate_stream_staging_ram_fit`` still reads free RAM only."""

    def test_it_never_touches_the_commit_counters(self, monkeypatch):
        from soup_cli.trainer.stream_setup import _validate_stream_staging_ram_fit

        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        _validate_stream_staging_ram_fit(
            staging_bytes=1_000_000_000, read_ahead=2, free_ram=100_000_000_000
        )
        with pytest.raises(ValueError, match="safety headroom"):
            _validate_stream_staging_ram_fit(
                staging_bytes=5_000_000_000, read_ahead=2, free_ram=6_000_000_000
            )

    def test_the_message_stays_a_free_ram_message(self):
        from soup_cli.trainer.stream_setup import _validate_stream_staging_ram_fit

        with pytest.raises(ValueError) as caught:
            _validate_stream_staging_ram_fit(
                staging_bytes=5_000_000_000, read_ahead=2, free_ram=6_000_000_000
            )
        text = _plain(str(caught.value))
        assert "free RAM" in text
        assert "commit" not in text.lower()
