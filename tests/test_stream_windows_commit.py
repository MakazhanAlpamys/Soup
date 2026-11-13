"""Layer streaming's staging pre-flight on Windows: commit headroom, not just free RAM.

On Windows every private host allocation is charged against the system COMMIT limit
(RAM + pagefile), and page-locked staging is the larger charge: it is packed into
power-of-two arenas (``plan_pinned_arenas``, #901). An experimental Mistral Large 123B
run on a 32 GB laptop was stopped twice by the harness's own commit floor with pinned
staging, and completed with pageable staging in a session that also differed in other
ways. Soup's staging pre-flight (``_validate_stream_staging_ram_fit``) reads free RAM
only, so nothing noticed.

``utils/windows_commit.py`` adds the missing check, Windows only, and acts on it:

* ``stream_pin`` unset: pinned (arena-rounded) fits -> pin; else pageable (the plain slot
  sum) fits -> stage pageable and say so in the panel; else refuse;
* ``stream_pin: true`` and the arenas cannot fit -> refuse, naming the numbers and the fix;
* ``stream_pin: false`` -> not read, not touched;
* nothing on another OS, nothing off CUDA, and a failed read skips the check.

The numbers are the ones the experimental run recorded (dense123_train34, MEMORY_COMPARISON34
and FAILURE_TRAIN01): commit limit 54,471,286,784 B; the pinned session started with
15,669,370,880 B free and was stopped at 484,179,968 B; the pageable session started with
14,439,596,032 B and bottomed at 4,078,985,216 B; the plan was 88 layers of 714,074,140 B and
two 805,306,368 B vocabulary matrices, whose regions were [714,080,256, 805,310,464,
805,310,464], which pin into 3,221,225,472 B of arenas (4,294,967,296 B at depth 2). The two
round starts, 12.7 and 17.9 GB free, are the ones the task gave.

Nothing here makes a real ctypes call except ``TestTheRealReader``, which runs only on
Windows (every cell of the Windows CI fleet, and the maintainer's box).
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.utils import windows_commit
from soup_cli.utils.layer_stream import (
    TIER_DISK,
    TIER_RAM,
    StreamPlan,
    build_stream_plan,
    render_stream_panel,
    staging_bytes_for,
)
from soup_cli.utils.windows_commit import (
    COMMIT_HEADROOM_MARGIN_BYTES,
    CommitFit,
    CommitInfo,
    apply_commit_preflight,
    assess_commit_fit,
    pinned_staging_bytes,
    staging_regions,
)

#: The real reader, taken at import time: an autouse fixture in conftest.py replaces
#: ``windows_commit.read_commit_info`` with a neutral one for every test, so a test of
#: the real wrapper has to put this back itself.
_REAL_READ_COMMIT_INFO = windows_commit.read_commit_info

_GIB = 2**30
_COMMIT_LIMIT = 54_471_286_784
_START_LOW = 12_700_000_000
_START_HIGH = 17_900_000_000
_PINNED_V1_START = 15_669_370_880
_PINNED_V1_STOP = 484_179_968
_PAGEABLE_V2_START = 14_439_596_032
_PAGEABLE_V2_MIN = 4_078_985_216
_PAGEABLE_BACKING = 2_324_713_472
_HARNESS_FLOOR = 536_870_912
_LAYER = 714_074_140
_VOCAB_MATRIX = 805_306_368
_REGIONS = [714_080_256, 805_310_464, 805_310_464]
_ARENAS_DEPTH1 = 3_221_225_472
_ARENAS_DEPTH2 = 4_294_967_296
_SLOTS_DEPTH1 = _LAYER + 2 * _VOCAB_MATRIX
_SLOTS_DEPTH2 = 2 * _LAYER + 2 * _VOCAB_MATRIX

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed output (Rich colours flag names on Linux CI)."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _gib(value: int) -> str:
    return f"{value / _GIB:.2f} GiB"


def _info(available: int, limit: int = _COMMIT_LIMIT) -> CommitInfo:
    return CommitInfo(total_bytes=limit - available, limit_bytes=limit)


def _plan(
    *,
    read_ahead: int = 1,
    pinned: bool = True,
    tier: str = TIER_DISK,
    layer_bytes: int = _LAYER,
    n_layers: int = 88,
    vocab: tuple = (_VOCAB_MATRIX, _VOCAB_MATRIX),
) -> StreamPlan:
    """A plan whose staging figure is what ``staging_bytes_for`` says of its own fields."""
    large_store = sum(vocab)
    return StreamPlan(
        arch="mistral",
        tier=tier,
        n_layers=n_layers,
        layer_bytes=layer_bytes,
        store_bytes=0,
        embed_bytes=0,
        large_store_bytes=large_store,
        large_buffer_bytes=max(vocab, default=0),
        buffers=2,
        buffer_bytes=0,
        pinned=pinned,
        notes=("an earlier note",),
        read_ahead=read_ahead,
        staging_bytes=(
            staging_bytes_for(
                read_ahead=read_ahead,
                n_layers=n_layers,
                layer_bytes=layer_bytes,
                large_store_bytes=large_store,
            )
            if tier == TIER_DISK
            else 0
        ),
    )


def _real_plan(read_ahead: int) -> StreamPlan:
    """The 123B plan through the real planner, as the setup builds it."""
    return build_stream_plan(
        arch="mistral",
        n_layers=88,
        layer_bytes=_LAYER,
        embed_bytes=0,
        available_ram_bytes=11_000_000_000,
        pinned_limit_bytes=None,
        total_ram_bytes=34_070_192_128,
        large_store_bytes=2 * _VOCAB_MATRIX,
        large_buffer_bytes=_VOCAB_MATRIX,
        read_ahead=read_ahead,
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
    """fits / tight / does not fit, from commit total + limit, a staging charge and extras."""

    @pytest.mark.parametrize(
        ("headroom", "staging", "extras", "expected"),
        [
            (_START_HIGH, _ARENAS_DEPTH2, 0, CommitFit.FITS),
            (_START_LOW, _ARENAS_DEPTH2, 0, CommitFit.TIGHT),
            (_START_HIGH, _ARENAS_DEPTH1, 0, CommitFit.FITS),
            (_START_LOW, _ARENAS_DEPTH1, 0, CommitFit.TIGHT),
            # The pinned session that was stopped: its arenas leave 12.45 GB, under 12 GiB.
            (_PINNED_V1_START, _ARENAS_DEPTH1, 0, CommitFit.TIGHT),
            # ... and the plain slot sum would have left 13.34 GB: it would have stayed pinned.
            (_PINNED_V1_START, _SLOTS_DEPTH1, 0, CommitFit.FITS),
            # The lowest point of the 512-token run: arenas cannot fit, the slots just do.
            (2_470_391_808, _ARENAS_DEPTH1, 0, CommitFit.DOES_NOT_FIT),
            (2_470_391_808, _SLOTS_DEPTH1, 0, CommitFit.TIGHT),
        ],
        ids=[
            "high-start-depth2",
            "low-start-depth2",
            "high-start-depth1",
            "low-start-depth1",
            "pinned-session-arenas",
            "pinned-session-slot-sum",
            "run-minimum-arenas",
            "run-minimum-slots",
        ],
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

    @pytest.mark.parametrize("arenas", [_ARENAS_DEPTH1, _ARENAS_DEPTH2], ids=["depth1", "depth2"])
    def test_the_margin_separates_the_two_starts_at_the_real_plan_sizes(self, arenas):
        """12.7 GB free falls under the margin, 17.9 GB clears it, at both depths."""
        low_left = _START_LOW - arenas
        high_left = _START_HIGH - arenas
        assert low_left < COMMIT_HEADROOM_MARGIN_BYTES <= high_left

    def test_the_margin_covers_the_worse_sessions_growth_and_the_floor(self):
        """The derivation in the module's comment, as arithmetic.

        Growth beyond the staging: the pageable session's was exact (it finished), the pinned
        one's a lower bound (it was stopped). The margin covers the larger plus the harness
        floor, rounded up to the next GiB and no further.
        """
        pageable_growth = (
            _PAGEABLE_V2_START - _PAGEABLE_V2_MIN - _PAGEABLE_BACKING
        )
        pinned_growth = _PINNED_V1_START - _PINNED_V1_STOP - _ARENAS_DEPTH1
        assert pageable_growth == 8_035_897_344
        assert pinned_growth == 11_963_965_440
        needed = max(pageable_growth, pinned_growth) + _HARNESS_FLOOR
        assert COMMIT_HEADROOM_MARGIN_BYTES >= needed
        assert COMMIT_HEADROOM_MARGIN_BYTES - needed < _GIB

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
# 2. what pinning really commits
# ---------------------------------------------------------------------------
class TestWhatPinningCommits:
    def test_the_regions_are_the_ones_the_run_recorded(self):
        assert staging_regions(_plan(read_ahead=1)) == _REGIONS

    def test_the_arenas_are_the_ones_the_run_recorded(self):
        assert pinned_staging_bytes(_plan(read_ahead=1)) == _ARENAS_DEPTH1
        assert pinned_staging_bytes(_plan(read_ahead=2)) == _ARENAS_DEPTH2

    def test_the_arenas_are_more_than_the_slot_sum_the_plan_reports(self):
        for depth, arenas in ((1, _ARENAS_DEPTH1), (2, _ARENAS_DEPTH2)):
            plan = _plan(read_ahead=depth)
            assert plan.staging_bytes == (_SLOTS_DEPTH1 if depth == 1 else _SLOTS_DEPTH2)
            assert pinned_staging_bytes(plan) == arenas > plan.staging_bytes

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(read_ahead=1),
            dict(read_ahead=2),
            dict(read_ahead=4),
            dict(read_ahead=2, vocab=(_VOCAB_MATRIX,)),
            dict(read_ahead=2, vocab=()),
            dict(read_ahead=3, layer_bytes=441_000_000, n_layers=80, vocab=(2_100_000_000,) * 2),
            dict(read_ahead=2, layer_bytes=1_000_000, n_layers=3, vocab=(65_536,)),
        ],
    )
    def test_pinning_never_commits_less_than_the_slots(self, kwargs):
        plan = _plan(**kwargs)
        assert pinned_staging_bytes(plan) >= plan.staging_bytes

    def test_a_tied_head_is_one_vocabulary_region(self):
        regions = staging_regions(_plan(read_ahead=1, vocab=(_VOCAB_MATRIX,)))
        assert regions == [714_080_256, 805_310_464]

    def test_no_vocabulary_matrix_is_no_region(self):
        assert staging_regions(_plan(read_ahead=2, vocab=())) == [714_080_256] * 2

    def test_the_depth_is_capped_by_the_layers_like_the_plans_own_figure(self):
        plan = _plan(read_ahead=4, n_layers=2, vocab=())
        assert staging_regions(plan) == [714_080_256] * 2
        assert plan.staging_bytes == 2 * _LAYER

    def test_the_real_planner_agrees_with_the_hand_built_plan(self):
        for depth in (1, 2):
            real = _real_plan(depth)
            assert real.tier == TIER_DISK
            assert real.pinned is True
            assert real.staging_bytes == _plan(read_ahead=depth).staging_bytes
            assert pinned_staging_bytes(real) == (_ARENAS_DEPTH1 if depth == 1 else _ARENAS_DEPTH2)


# ---------------------------------------------------------------------------
# 3. reading the numbers: lazy, isolated, never fatal
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
            return _info(_START_LOW)

        monkeypatch.setattr(windows_commit, "_query_commit_windows", query)
        assert _REAL_READ_COMMIT_INFO() is None
        assert reached == [], "ctypes must not be reached off Windows"

    @pytest.mark.parametrize(
        "error",
        [OSError("no psapi"), AttributeError("no windll"), ValueError("bad"), RuntimeError("x")],
        ids=lambda error: type(error).__name__,
    )
    def test_a_failed_read_is_none_with_an_info_note(self, monkeypatch, caplog, error):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)

        def fail():
            raise error

        monkeypatch.setattr(windows_commit, "_query_commit_windows", fail)
        with caplog.at_level(logging.INFO, logger=windows_commit.__name__):
            assert _REAL_READ_COMMIT_INFO() is None
        notes = [rec for rec in caplog.records if rec.name == windows_commit.__name__]
        assert len(notes) == 1, notes
        assert notes[0].levelno == logging.INFO
        assert "commit" in notes[0].getMessage().lower()
        assert repr(error) in notes[0].getMessage()

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_nonsense_limit_is_a_failed_read_with_an_info_note(
        self, monkeypatch, caplog, limit
    ):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        monkeypatch.setattr(
            windows_commit,
            "_query_commit_windows",
            lambda: CommitInfo(total_bytes=5, limit_bytes=limit),
        )
        with caplog.at_level(logging.INFO, logger=windows_commit.__name__):
            assert _REAL_READ_COMMIT_INFO() is None
        notes = [rec for rec in caplog.records if rec.name == windows_commit.__name__]
        assert len(notes) == 1 and notes[0].levelno == logging.INFO, notes

    def test_a_good_read_is_returned_as_is_and_says_nothing(self, monkeypatch, caplog):
        good = _info(_START_LOW)
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        monkeypatch.setattr(windows_commit, "_query_commit_windows", lambda: good)
        with caplog.at_level(logging.DEBUG, logger=windows_commit.__name__):
            assert _REAL_READ_COMMIT_INFO() == good
        assert [rec for rec in caplog.records if rec.name == windows_commit.__name__] == []

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

    def test_other_tests_never_read_the_hosts_real_counters(self, monkeypatch):
        """conftest.py neutralises the reader, on every OS.

        The Windows platform test is patched to True and the ctypes query to a recorder, so
        that a reader which had NOT been neutralised would reach the recorder on Linux and
        macOS too, not only on a Windows box.
        """
        reached: list[int] = []
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        monkeypatch.setattr(
            windows_commit,
            "_query_commit_windows",
            lambda: reached.append(1) or _info(_START_HIGH),
        )
        assert windows_commit.read_commit_info() is None
        assert reached == []


@pytest.mark.skipif(sys.platform != "win32", reason="GetPerformanceInfo exists on Windows only")
class TestTheRealReader:
    """The one real ctypes call: the structure layout is only proven by the OS."""

    def test_the_system_reports_a_sane_commit(self):
        info = windows_commit._query_commit_windows()
        assert info.total_bytes > 0
        # Any box has at least a gigabyte of commit limit; page counts, not bytes, would not.
        assert info.limit_bytes >= 2**30
        assert info.limit_bytes % 4096 == 0
        # CommitTotal and CommitLimit swapped would still pass everything above.
        assert info.limit_bytes >= info.total_bytes

    def test_it_agrees_with_global_memory_status(self):
        """``GlobalMemoryStatusEx`` reports the same limit when no job object caps the process."""
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        info = windows_commit._query_commit_windows()
        process_cap = status.ullTotalPageFile < info.limit_bytes * 0.95
        if process_cap:
            pytest.skip("a job object caps GlobalMemoryStatusEx below the system commit limit")
        # The pagefile can grow between the two calls, so the limit is compared with a margin.
        assert abs(info.limit_bytes - status.ullTotalPageFile) <= 0.05 * info.limit_bytes
        used = status.ullTotalPageFile - status.ullAvailPageFile
        assert abs(info.total_bytes - used) <= 4 * _GIB

    def test_the_wrapper_returns_the_same_shape(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        info = _REAL_READ_COMMIT_INFO()
        assert isinstance(info, CommitInfo)
        assert info.limit_bytes >= info.total_bytes


# ---------------------------------------------------------------------------
# 4. the decision on a plan
# ---------------------------------------------------------------------------
def _apply(plan, *, stream_pin, on_cuda=True, resident=0):
    said: list[str] = []
    out = apply_commit_preflight(
        plan,
        stream_pin=stream_pin,
        on_cuda=on_cuda,
        resident_bytes=resident,
        notify=said.append,
    )
    return out, said


class TestTheRealPlanMeetsTheRecordedStarts:
    """Real plan -> verdict: the 123B plan from the planner, at the recorded starts."""

    @pytest.mark.parametrize(
        ("depth", "start", "stays_pinned"),
        [
            (1, _START_HIGH, True),
            (2, _START_HIGH, True),
            (1, _START_LOW, False),
            (2, _START_LOW, False),
            # The pinned session that was stopped at the harness's floor: told to stage pageable.
            (1, _PINNED_V1_START, False),
        ],
        ids=["high-d1", "high-d2", "low-d1", "low-d2", "pinned-session-d1"],
    )
    def test_auto_keeps_pinning_or_falls_back(self, windows_box, depth, start, stays_pinned):
        windows_box(start)
        plan = _real_plan(depth)
        out, said = _apply(plan, stream_pin=None)
        assert said == []
        if stays_pinned:
            assert out is plan
            assert out.pinned is True
        else:
            assert out.pinned is False
            assert plan.pinned is True
            added = out.notes[len(plan.notes):]
            assert len(added) == 1 and "pageable" in added[0], added
            assert _gib(pinned_staging_bytes(plan)) in added[0]

    def test_the_slot_sum_alone_would_have_left_the_stopped_session_pinned(self, windows_box):
        """Why the arena size is charged: the unrounded figure clears the margin at 15.67 GB."""
        windows_box(_PINNED_V1_START)
        plan = _real_plan(1)
        unrounded = assess_commit_fit(
            commit_total_bytes=_COMMIT_LIMIT - _PINNED_V1_START,
            commit_limit_bytes=_COMMIT_LIMIT,
            pinned_staging_bytes=plan.staging_bytes,
        )
        assert unrounded.fit is CommitFit.FITS
        assert _apply(plan, stream_pin=None)[0].pinned is False

    def test_explicit_true_at_the_low_start_is_honoured_with_a_warning(self, windows_box):
        windows_box(_START_LOW)
        plan = _real_plan(1)
        out, said = _apply(plan, stream_pin=True)
        assert out is plan
        assert len(said) == 1 and "stream_pin" in said[0], said


class TestAutoFallsBackToPageable:
    def test_a_tight_headroom_stages_pageable_and_says_so_once(self, windows_box):
        windows_box(_START_LOW)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=None)
        assert out.pinned is False
        assert plan.pinned is True, "the plan that came in must not be mutated"
        assert out.staging_bytes == plan.staging_bytes
        assert out.tier == TIER_DISK
        added = out.notes[len(plan.notes):]
        assert out.notes[: len(plan.notes)] == plan.notes
        assert len(added) == 1, added
        line = _plain(added[0])
        assert line.startswith("Windows commit headroom:")
        assert "pageable" in line
        assert "training.stream_pin: true" in line
        assert _gib(_ARENAS_DEPTH2) in line
        # Both figures are said, each in its own place: the arenas pinning would commit,
        # the slots they round up and the pageable staging that replaces them.
        assert f"power-of-two arenas for {_gib(_SLOTS_DEPTH2)} of slots" in line
        assert f"Staging {_gib(_SLOTS_DEPTH2)} in pageable host memory" in line
        assert _gib(_START_LOW) in line
        assert _gib(COMMIT_HEADROOM_MARGIN_BYTES) in line
        assert "12.00 GiB" in line
        assert "GB" not in line.replace("GiB", ""), "one unit throughout: GiB"
        assert said == [], "the panel line is the one statement; nothing else is printed"

    def test_arenas_that_do_not_fit_but_pageable_slots_do_also_fall_back(self, windows_box):
        # 2.6 GB free: the 3 GiB of arenas do not fit, the 2.17 GiB of slots do.
        windows_box(2_600_000_000)
        out, said = _apply(_plan(read_ahead=1), stream_pin=None)
        assert out.pinned is False
        line = _plain(out.notes[-1])
        # 2.42 GiB free against 3.00 GiB of arenas: said as over by 0.58 GiB, not as -0.58.
        assert "leave none (over by 0.58 GiB)" in line, line
        assert not re.search(r"-\d", line), line
        assert said == []

    def test_headroom_above_the_margin_keeps_the_pinned_default(self, windows_box):
        reads = windows_box(_START_HIGH)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=None)
        assert out is plan
        assert out.pinned is True
        assert said == []
        assert len(reads) == 1

    def test_resident_extras_are_charged_to_the_decision(self, windows_box):
        windows_box(_START_HIGH)
        plan = _plan(read_ahead=1)
        assert _apply(plan, stream_pin=None, resident=0)[0] is plan
        # 14.68 GB left after the arenas; 2 GiB of extras take it under 12 GiB.
        assert _apply(plan, stream_pin=None, resident=2 * _GIB)[0].pinned is False

    def test_neither_pinned_nor_pageable_fitting_is_refused_with_the_numbers(self, windows_box):
        windows_box(2_000_000_000)
        with pytest.raises(ValueError) as caught:
            _apply(_plan(read_ahead=2), stream_pin=None)
        message = _plain(str(caught.value))
        assert "Pageable memory is committed too" in message
        assert "training.stream_pin: false does not help" in message
        assert "Refusing before any weight is staged" in message
        assert _gib(_ARENAS_DEPTH2) in message
        assert _gib(_SLOTS_DEPTH2) in message
        assert _gib(2_000_000_000) in message
        assert "close programs" in message and "pagefile" in message
        assert "training.stream_read_ahead (now 2)" in message
        assert "GB" not in message.replace("GiB", "")

    def test_at_depth_one_the_refusal_suggests_a_smaller_base(self, windows_box):
        windows_box(1_000_000_000)
        with pytest.raises(ValueError) as caught:
            _apply(_plan(read_ahead=1), stream_pin=None)
        message = _plain(str(caught.value))
        assert "smaller base" in message
        assert "stream_read_ahead" not in message


class TestExplicitPinIsHonouredOrRefused:
    def test_arenas_that_do_not_fit_are_refused_with_the_numbers_and_the_fix(self, windows_box):
        windows_box(2_600_000_000)
        with pytest.raises(ValueError) as caught:
            _apply(_plan(read_ahead=1), stream_pin=True)
        message = _plain(str(caught.value))
        assert "training.stream_pin=true" in message
        assert "Windows commit limit" in message
        assert _gib(_ARENAS_DEPTH1) in message
        assert _gib(_SLOTS_DEPTH1) in message
        assert _gib(2_600_000_000) in message
        assert "Refusing before any weight is staged" in message
        # The pageable slots fit, so `false` is a real way out and is named as one.
        assert "Set training.stream_pin: false to stage" in message
        assert "which fits" in message
        assert "pagefile" in message and "close programs" in message
        assert "GB" not in message.replace("GiB", "")

    def test_when_pageable_does_not_fit_either_false_is_not_offered(self, windows_box):
        windows_box(1_000_000_000)
        with pytest.raises(ValueError) as caught:
            _apply(_plan(read_ahead=2), stream_pin=True)
        message = _plain(str(caught.value))
        assert "Pageable memory is committed too" in message
        assert "would not fit either" in message
        assert "Set training.stream_pin: false to stage" not in message
        assert "Refusing before any weight is staged" in message

    def test_a_tight_headroom_is_honoured_with_one_warning(self, windows_box):
        windows_box(_START_LOW)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=True)
        assert out is plan
        assert out.pinned is True
        assert len(said) == 1
        line = _plain(said[0])
        assert "training.stream_pin=true" in line
        assert "training.stream_pin: false" in line
        assert _gib(COMMIT_HEADROOM_MARGIN_BYTES) in line
        assert _gib(_START_LOW) in line
        assert "GB" not in line.replace("GiB", "")

    def test_a_fitting_staging_is_silent(self, windows_box):
        windows_box(_START_HIGH)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=True)
        assert out is plan
        assert said == []

    def test_a_refusal_is_a_value_error_like_the_ram_refusal(self, windows_box):
        """The setup surfaces both the same way, so the user sees one kind of stop."""
        windows_box(1_000)
        with pytest.raises(ValueError):
            _apply(_plan(read_ahead=2), stream_pin=True)


class TestWhatIsLeftAlone:
    def test_stream_pin_false_is_unchanged_and_never_reads(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        # pinned=True on purpose: an explicit false is left alone by the check itself,
        # not only because the plan it was handed happens to be pageable already.
        plan = _plan(read_ahead=2, pinned=True)
        out, said = _apply(plan, stream_pin=False)
        assert out is plan
        assert said == []

    @pytest.mark.parametrize("stream_pin", [None, True, False])
    def test_off_windows_the_reader_is_never_called(self, monkeypatch, stream_pin):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: False)
        _reader_must_not_run(monkeypatch)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=stream_pin)
        assert out is plan
        assert said == []

    @pytest.mark.parametrize("stream_pin", [None, True])
    def test_without_cuda_nothing_is_page_locked_so_nothing_is_read(
        self, monkeypatch, stream_pin
    ):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=stream_pin, on_cuda=False)
        assert out is plan
        assert said == []

    def test_a_plan_that_is_already_pageable_is_not_second_guessed(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        plan = _plan(read_ahead=2, pinned=False)
        out, said = _apply(plan, stream_pin=None)
        assert out is plan
        assert said == []

    def test_the_ram_tier_has_no_staging_to_charge(self, monkeypatch):
        monkeypatch.setattr(windows_commit, "_on_windows", lambda: True)
        _reader_must_not_run(monkeypatch)
        plan = _plan(tier=TIER_RAM)
        out, said = _apply(plan, stream_pin=None)
        assert out is plan
        assert said == []

    @pytest.mark.parametrize("stream_pin", [None, True])
    def test_a_failed_read_skips_the_check_and_trains_on(self, windows_box, stream_pin):
        reads = windows_box(None)
        plan = _plan(read_ahead=2)
        out, said = _apply(plan, stream_pin=stream_pin)
        assert out is plan
        assert out.pinned is True
        assert said == []
        assert len(reads) == 1

    def test_a_small_plan_fits_where_a_big_one_does_not(self, windows_box):
        windows_box(_START_HIGH)
        small = _plan(read_ahead=2, layer_bytes=1_000_000, n_layers=4, vocab=(65_536,))
        assert _apply(small, stream_pin=None)[0] is small
        big = _plan(read_ahead=2, layer_bytes=3_000_000_000, n_layers=40)
        assert _apply(big, stream_pin=None)[0].pinned is False


class TestThePanelAfterTheFallback:
    @staticmethod
    def _render(plan) -> str:
        console = Console(width=200, color_system=None, record=True, force_terminal=False)
        with console.capture() as capture:
            console.print(render_stream_panel(plan))
        return _plain(capture.get())

    def test_the_fallback_panel_says_pageable_not_page_locked_when_the_box_allows(
        self, windows_box
    ):
        windows_box(_START_LOW)
        out, _ = _apply(_plan(read_ahead=2), stream_pin=None)
        text = self._render(out)
        assert "page-locked when the box allows" not in text, text
        assert "host staging read_ahead 2" in text
        assert "(pageable)" in text
        assert text.count("Windows commit headroom:") == 1

    def test_a_pinned_plan_keeps_its_line(self):
        text = self._render(_plan(read_ahead=2))
        assert "(page-locked when the box allows)" in text
        assert "(pageable)" not in text

    def test_a_stream_pin_false_plan_no_longer_claims_page_locking(self):
        text = self._render(_plan(read_ahead=2, pinned=False))
        assert "(pageable)" in text
        assert "page-locked when the box allows" not in text


# ---------------------------------------------------------------------------
# 5. the real setup path: what reaches the runtime boundary
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
        assert "page-locked when the box allows" not in text

    def test_auto_on_a_roomy_windows_box_still_pins(self, tmp_path, monkeypatch, windows_box):
        reads = windows_box(40_000_000_000)
        captured, panel = _drive(tmp_path, monkeypatch)
        assert captured["pin"] is True
        assert "Windows commit headroom" not in _plain(panel)
        assert len(reads) == 1

    def test_auto_with_no_commit_for_either_kind_stops_before_the_runtime(
        self, tmp_path, monkeypatch, windows_box
    ):
        windows_box(1_000)
        import soup_cli.utils.layer_stream_runtime as rt

        built = []
        monkeypatch.setattr(rt, "build_streamed_model", lambda **kwargs: built.append(kwargs))
        with pytest.raises(ValueError) as caught:
            _drive(tmp_path, monkeypatch)
        message = _plain(str(caught.value))
        assert "Pageable memory is committed too" in message
        assert built == []

    def test_explicit_pin_that_does_not_fit_stops_before_the_runtime(
        self, tmp_path, monkeypatch, windows_box
    ):
        windows_box(1_000)
        with pytest.raises(ValueError) as caught:
            _drive(tmp_path, monkeypatch, stream_pin_yaml="  stream_pin: true\n")
        message = _plain(str(caught.value))
        assert "commit" in message.lower()
        assert "Refusing before any weight is staged" in message

    def test_explicit_pin_that_does_not_fit_never_builds_the_runtime(
        self, tmp_path, monkeypatch, windows_box
    ):
        windows_box(1_000)
        import soup_cli.utils.layer_stream_runtime as rt

        built = []
        monkeypatch.setattr(rt, "build_streamed_model", lambda **kwargs: built.append(kwargs))
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
