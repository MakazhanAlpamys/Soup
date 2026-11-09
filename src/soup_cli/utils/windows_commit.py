"""Windows commit headroom for layer streaming's host staging.

On Windows every private host allocation is charged against the system COMMIT limit
(RAM + pagefile), pageable or page-locked, and free RAM does not show it. The disk
tier's async reader holds its staging for the whole run, and the staging pre-flight
(``trainer/stream_setup.py::_validate_stream_staging_ram_fit``) reads free RAM alone,
so a box with RAM to spare and no commit left passed it and was stopped later,
mid-run. Page-locked staging is the larger charge: it is packed into power-of-two
arenas (``plan_pinned_arenas``, #901), so the 123B plan below asks for 3 GiB where its
slots add up to 2.17 GiB.

What this module adds, all of it Windows-only and none of it a config key:

* :func:`assess_commit_fit` - a pure function: commit total / limit, a staging charge
  and the resident extras in, ``fits`` / ``tight`` / ``does not fit`` out. It is the
  one place the margin is applied.
* :func:`pinned_staging_bytes` - what pinning really commits: the arena-rounded size,
  by the same packer the reader uses.
* :func:`read_commit_info` - the numbers, read lazily through ``ctypes``. Any failure
  to read them is ``None``, logged at info: the check is skipped, training is not.
* :func:`apply_commit_preflight` - what the staging pre-flight does with the verdicts.
  ``stream_pin`` unset: pinned (arena-rounded) fits -> pin; else pageable (the plain
  slot sum) fits -> stage pageable and say so in the panel; else refuse. ``stream_pin:
  true``: refuse when the arenas cannot fit at all, warn when it is tight.
  ``stream_pin: false``: not read, not touched.

Which Windows API: ``GetPerformanceInfo`` (psapi), not ``GlobalMemoryStatusEx``.
Both report commit, but ``GlobalMemoryStatusEx`` documents ``ullTotalPageFile`` as
"the current committed memory limit for the system **or the current process,
whichever is smaller**" and ``ullAvailPageFile`` as what "the current process can
commit", so under a job object it silently becomes a per-process figure.
``GetPerformanceInfo`` returns ``CommitTotal`` and ``CommitLimit`` for the SYSTEM,
in pages, which is the quantity the OS tests when any process asks to commit and
the one the experimental harness's own guard watched (its "system commit
headroom"). The pagefile can grow when it is system-managed, so the limit is a
reading at one moment, not a ceiling.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only; this module stays import-light
    from soup_cli.utils.layer_stream import StreamPlan

logger = logging.getLogger(__name__)

#: Commit that must still be free AFTER the pinned staging (arena-rounded) and the
#: resident extras are charged, for pinning to count as ``fits``. **A chosen safety
#: margin, not a measured bound.** The pre-flight sees only the staging; the rest of a
#: run (CUDA context, dense-layer work buffers, activations spilled to host, the
#: interpreter, and whatever else the box is doing) commits on top of it.
#:
#: Derivation, from the two sessions of the experimental Mistral Large 123B run on a
#: 32 GB laptop (commit limit 54,471,286,784 B; plan: 88 layers of 714,074,140 B, two
#: 805,306,368 B vocabulary matrices, ``stream_read_ahead`` 1; its regions were
#: [714,080,256, 805,310,464, 805,310,464] and pinned they pack into 3,221,225,472 B of
#: arenas, 4,294,967,296 B at depth 2):
#:
#: * pageable, completed three updates: free commit 14,439,596,032 B at the start,
#:   4,078,985,216 B at its lowest, 2,324,713,472 B of it staging. Growth beyond the
#:   staging: 8,035,897,344 B (7.5 GiB).
#: * pinned, stopped by the harness's own 512 MiB commit floor in the first iteration:
#:   15,669,370,880 B at the start, 484,179,968 B at the stop, 3,221,225,472 B of it
#:   arenas. Growth beyond the staging, a LOWER bound because it was cut short:
#:   11,963,965,440 B (11.1 GiB).
#:
#: The margin has to cover the growth beyond the staging plus the 536,870,912 B floor
#: in the worse session: 12,500,836,352 B, rounded up to 12 GiB. With 12 GiB and the
#: real plan sizes the two starts the brief gives (12.7 GB and 17.9 GB free) come out
#: as the runs would want: at ``stream_read_ahead`` 1 (3 GiB of arenas) 12.7 GB leaves
#: 9.48 GB (``tight``, falls back to pageable) and 17.9 GB leaves 14.68 GB (``fits``);
#: at depth 2 (4 GiB) they leave 8.41 GB and 13.61 GB, the same two verdicts. The
#: recorded pinned start, 15.67 GB, leaves 12.45 GB at depth 1: ``tight`` too, so it
#: would have been told to stage pageable (charging the plain slot sum instead, 2.32
#: GB, leaves 13.34 GB and keeps it pinned: that is why the arena size is charged).
#:
#: Two caveats the records themselves make. The sessions differ in more than staging:
#: the pinned one began with 3.1 GB less free commit and commit outside the worker grew
#: by a further 4.7 GB, so the staging explains about 0.9 GB of a 3.9 GB difference in
#: growth, and the margin is sized on the worse session rather than on a controlled
#: pinned-versus-pageable pair. And it is one laptop. Being wrong costs throughput
#: (pageable staging gives up the copy overlap), not correctness, which is why a larger
#: figure is the cheaper error.
COMMIT_HEADROOM_MARGIN_BYTES = 12 * 2**30

#: ``AsyncDiskSource`` widens every staging region to sector boundaries; this is that
#: sector (``async_disk_source.SECTOR_BYTES``), repeated so the module stays import-light.
_SECTOR_BYTES = 4096


class CommitFit(str, Enum):
    """How a staging charge sits against the commit still free."""

    FITS = "fits"
    TIGHT = "tight"
    DOES_NOT_FIT = "does_not_fit"


@dataclass(frozen=True)
class CommitInfo:
    """System commit at one moment, in bytes."""

    total_bytes: int
    limit_bytes: int

    @property
    def available_bytes(self) -> int:
        """What can still be committed; transiently over the limit reads as 0."""
        return max(0, self.limit_bytes - self.total_bytes)


@dataclass(frozen=True)
class CommitVerdict:
    """The outcome of :func:`assess_commit_fit`, with the figures it was made from."""

    fit: CommitFit
    available_bytes: int
    required_bytes: int
    #: ``available - required``; NEGATIVE when the charge does not fit at all.
    headroom_after_bytes: int
    margin_bytes: int


def _check_bytes(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int number of bytes; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value}")
    return value


def assess_commit_fit(
    *,
    commit_total_bytes: int,
    commit_limit_bytes: int,
    pinned_staging_bytes: int,
    resident_extras_bytes: int = 0,
    margin_bytes: int = COMMIT_HEADROOM_MARGIN_BYTES,
) -> CommitVerdict:
    """Decide whether a staging charge leaves enough system commit.

    ``pinned_staging_bytes`` is the staging figure being judged: the arena-rounded
    size when judging pinning, the plain slot sum when judging pageable staging.
    ``available`` is ``limit - total`` (0 when the total is over the limit, which
    happens transiently while the pagefile grows). The staging and the resident
    extras are charged against it, the same two figures the free-RAM check is given.
    What is left decides the verdict:

    * ``fits`` - at least ``margin_bytes`` left;
    * ``tight`` - between 0 and ``margin_bytes`` left: it fits, with less room for
      the rest of the run than the margin asks for;
    * ``does_not_fit`` - the charge is more than is free.
    """
    total = _check_bytes("commit_total_bytes", commit_total_bytes)
    limit = _check_bytes("commit_limit_bytes", commit_limit_bytes)
    staging = _check_bytes("pinned_staging_bytes", pinned_staging_bytes)
    extras = _check_bytes("resident_extras_bytes", resident_extras_bytes)
    margin = _check_bytes("margin_bytes", margin_bytes)
    available = max(0, limit - total)
    required = staging + extras
    left = available - required
    if left < 0:
        fit = CommitFit.DOES_NOT_FIT
    elif left < margin:
        fit = CommitFit.TIGHT
    else:
        fit = CommitFit.FITS
    return CommitVerdict(
        fit=fit,
        available_bytes=available,
        required_bytes=required,
        headroom_after_bytes=left,
        margin_bytes=margin,
    )


# ---------------------------------------------------------------------------
# what pinning really commits
# ---------------------------------------------------------------------------
def _region_bytes(size: int) -> int:
    """A staging region as the reader allocates it: sector-aligned, plus one sector.

    ``AsyncDiskSource`` widens a layer's data section to sector boundaries on both
    ends. For the recorded 123B regions this reproduces them exactly: 714,074,140 ->
    714,080,256 and 805,306,368 -> 805,310,464. It is the worst case for a start that
    is not sector-aligned and at most one sector over for one that is.
    """
    return -(-int(size) // _SECTOR_BYTES) * _SECTOR_BYTES + _SECTOR_BYTES


def staging_regions(plan: "StreamPlan") -> List[int]:
    """The regions the async reader allocates for ``plan``, in allocation order.

    The decoder layers first (``min(read_ahead, n_layers)`` slots of ``layer_bytes``),
    then one region per vocabulary matrix. The plan carries the vocabulary matrices as
    their sum (``large_store_bytes``) and the largest (``large_buffer_bytes``); a
    matrix pair of equal size (embedding and untied head) splits back exactly, a tied
    model has one. This is the plan's own approximation, the one
    ``layer_stream.staging_bytes_for`` makes.
    """
    depth = max(0, min(int(plan.read_ahead), int(plan.n_layers)))
    regions = [_region_bytes(plan.layer_bytes)] * depth
    large_total = int(plan.large_store_bytes)
    if large_total > 0:
        biggest = int(plan.large_buffer_bytes)
        biggest = large_total if biggest <= 0 else min(biggest, large_total)
        regions.append(_region_bytes(biggest))
        if large_total > biggest:
            regions.append(_region_bytes(large_total - biggest))
    return regions


def pinned_staging_bytes(plan: "StreamPlan") -> int:
    """Bytes pinning commits for ``plan``: the staging packed into power-of-two arenas.

    The reader page-locks ``plan_pinned_arenas(regions).pinned_bytes`` (#901), not the
    slot sum: torch's caching host allocator rounds a pinned block up to a power of
    two, so the regions share arenas that are themselves powers of two. Pure
    arithmetic from the packer the reader calls, no torch.
    """
    from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

    regions = staging_regions(plan)
    if not regions:
        return 0
    return plan_pinned_arenas(regions, align=_SECTOR_BYTES).pinned_bytes


# ---------------------------------------------------------------------------
# reading the numbers
# ---------------------------------------------------------------------------
def _on_windows() -> bool:
    """The platform test, one seam so tests can stand in for either OS."""
    import sys

    return sys.platform == "win32"


def _commit_info_from_pages(*, total_pages: int, limit_pages: int, page_size: int) -> CommitInfo:
    """``GetPerformanceInfo`` reports commit in PAGES; the rest of Soup counts bytes."""
    return CommitInfo(total_bytes=total_pages * page_size, limit_bytes=limit_pages * page_size)


def _query_commit_windows() -> CommitInfo:
    """``GetPerformanceInfo`` through ``ctypes``. Windows only; raises on any failure."""
    import ctypes
    from ctypes import wintypes

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    # A private handle with prototypes, not the shared ``ctypes.windll``.
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetPerformanceInfo.argtypes = [
        ctypes.POINTER(PerformanceInformation),
        wintypes.DWORD,
    ]
    psapi.GetPerformanceInfo.restype = wintypes.BOOL
    info = PerformanceInformation()
    if not psapi.GetPerformanceInfo(ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "GetPerformanceInfo failed")
    return _commit_info_from_pages(
        total_pages=int(info.CommitTotal),
        limit_pages=int(info.CommitLimit),
        page_size=int(info.PageSize),
    )


def read_commit_info() -> Optional[CommitInfo]:
    """System commit now, or ``None`` when it is not Windows or could not be read.

    Never raises: this guards a pre-flight, and a pre-flight that cannot take its
    reading is skipped, not allowed to stop the training it was written to protect.
    The skip is logged at info, so a Windows user whose counters fail can see why the
    old behaviour came back.
    """
    if not _on_windows():
        return None
    try:
        info = _query_commit_windows()
    except Exception as exc:  # noqa: BLE001 - any failure here skips the check
        logger.info("Windows commit headroom could not be read (%r); skipping the check", exc)
        return None
    if info.limit_bytes <= 0:
        logger.info(
            "Windows commit limit read as %d bytes; treating it as unreadable and skipping "
            "the check",
            info.limit_bytes,
        )
        return None
    return info


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------
def _gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def _left(verdict: CommitVerdict) -> str:
    if verdict.headroom_after_bytes >= 0:
        return _gib(verdict.headroom_after_bytes)
    return f"none (over by {_gib(-verdict.headroom_after_bytes)})"


def apply_commit_preflight(
    plan: "StreamPlan",
    *,
    stream_pin: Optional[bool],
    on_cuda: bool,
    resident_bytes: int,
    notify: Callable[[str], object],
) -> "StreamPlan":
    """Charge the disk tier's host staging against Windows commit.

    Pinning is judged on the arena-rounded size (:func:`pinned_staging_bytes`), pageable
    staging on the plain slot sum (``plan.staging_bytes``). ``stream_pin`` unset:

    * pinned fits: ``plan`` itself, silently;
    * else pageable does not run out of commit: a copy with ``pinned=False`` and one
      added note, which the panel prints;
    * else ``ValueError`` - pageable memory is committed too, so staging in it does not
      rescue a box with no commit left.

    ``stream_pin: true`` is refused (``ValueError``) when the arenas cannot fit at all,
    and honoured with one warning through ``notify`` (``console.print``) when merely
    tight, because the user asked for it and the margin is a judgement.

    Nothing happens, and the commit counters are not read, when ``stream_pin`` is
    ``false`` (pageable already), the target is not CUDA (nothing is page-locked), the
    plan is already pageable, there is no staging to charge (the RAM tier), or the OS is
    not Windows.
    """
    if stream_pin is False or not on_cuda or not plan.pinned or plan.staging_bytes <= 0:
        return plan
    if not _on_windows():
        return plan
    info = read_commit_info()
    if info is None:
        return plan

    def judge(staging_bytes: int) -> CommitVerdict:
        return assess_commit_fit(
            commit_total_bytes=info.total_bytes,
            commit_limit_bytes=info.limit_bytes,
            pinned_staging_bytes=staging_bytes,
            resident_extras_bytes=resident_bytes,
        )

    pinned_bytes = pinned_staging_bytes(plan)
    pinned = judge(pinned_bytes)
    if pinned.fit is CommitFit.FITS:
        return plan
    pageable = judge(plan.staging_bytes)

    slots = _gib(plan.staging_bytes)
    extras = _gib(int(resident_bytes))
    free = _gib(pinned.available_bytes)
    margin = _gib(pinned.margin_bytes)
    committed = f"{_gib(info.total_bytes)} of {_gib(info.limit_bytes)} committed"
    what = (
        f"{_gib(pinned_bytes)} of page-locked staging (power-of-two arenas for {slots} of "
        f"slots) plus {extras} of resident extras"
    )
    pageable_fits = pageable.fit is not CommitFit.DOES_NOT_FIT
    fixes = "close programs to free commit, enlarge the Windows pagefile"
    if plan.read_ahead > 1:
        fixes += f", or lower training.stream_read_ahead (now {plan.read_ahead})"
    else:
        fixes += ", or use a smaller base"

    if stream_pin is True:
        if pinned.fit is CommitFit.DOES_NOT_FIT:
            if pageable_fits:
                way_out = (
                    f"Set training.stream_pin: false to stage {slots} in pageable memory, which "
                    f"fits, or {fixes}."
                )
            else:
                way_out = (
                    f"Pageable memory is committed too, so training.stream_pin: false would "
                    f"not fit either ({slots} against {free} free); {fixes}."
                )
            raise ValueError(
                f"training.stream_pin=true but pinned host staging does not fit the Windows "
                f"commit limit: {what} = {_gib(pinned.required_bytes)}, and only {free} of "
                f"system commit is free ({committed}). Refusing before any weight is staged. "
                f"{way_out}"
            )
        notify(
            f"[yellow]training.stream_pin=true: {what} would leave {_left(pinned)} of Windows "
            f"commit free ({free} free now), under the {margin} margin. Proceeding as asked; if "
            f"the run is stopped for lack of commit, set training.stream_pin: false.[/]"
        )
        return plan

    if not pageable_fits:
        raise ValueError(
            f"Windows commit is too low for layer streaming's host staging: {what} = "
            f"{_gib(pinned.required_bytes)}, or {slots} + {extras} = "
            f"{_gib(pageable.required_bytes)} even in pageable memory, and only {free} of "
            f"system commit is free ({committed}). Pageable memory is committed too, so "
            f"training.stream_pin: false does not help. Refusing before any weight is staged. "
            f"To continue, {fixes}."
        )
    line = (
        f"Windows commit headroom: {what} would leave {_left(pinned)} ({free} free now), "
        f"under the {margin} margin. Staging {slots} in pageable host memory instead, which "
        f"gives up the host-to-device copy overlap. training.stream_pin: true forces "
        f"page-locking; it is refused only when the arenas cannot fit the commit limit at all."
    )
    return dataclasses.replace(plan, pinned=False, notes=plan.notes + (line,))
