"""Windows commit headroom for layer streaming's page-locked host staging.

On Windows, page-locked (pinned) host memory and private mappings are charged
against the system COMMIT limit (RAM + pagefile), not only against free RAM. The
disk tier's async reader page-locks its staging for the whole run, and the staging
pre-flight (``trainer/stream_setup.py::_validate_stream_staging_ram_fit``) reads
free RAM alone, so a box with RAM to spare and no commit left passed it and was
stopped later, mid-run. An experimental Mistral Large 123B run on a 32 GB laptop
was stopped twice that way with pinned staging; the same run with pageable staging
passed. (That evidence is one box and one pair of sessions, and the reports say
other memory use differed between them. This module acts on the mechanism, which
is documented, and not on that pair alone.)

What this module adds, all of it Windows-only and none of it a config key:

* :func:`assess_commit_fit` - a pure function: commit total / limit, the planned
  pinned staging and the resident extras in, ``fits`` / ``tight`` / ``does not fit``
  out. It is the one place the margin is applied.
* :func:`read_commit_info` - the numbers, read lazily through ``ctypes``. Any
  failure to read them is ``None``, with a debug-level note: the check is skipped,
  training is not.
* :func:`apply_commit_preflight` - what the staging pre-flight does with the
  verdict. ``stream_pin`` unset and not ``fits``: stage pageable instead and say so
  in the panel. ``stream_pin: true`` and ``does not fit``: refuse, before the
  runtime stages anything. ``stream_pin: false``: not read, not touched.

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
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only; this module stays import-light
    from soup_cli.utils.layer_stream import StreamPlan

logger = logging.getLogger(__name__)

#: Commit that must still be free AFTER the pinned staging and the resident extras
#: are charged, for the staging to count as ``fits``. **A chosen safety margin, not
#: a measured bound.** The pre-flight can only see the staging; the rest of a run
#: (CUDA context, dense-layer work buffers, activations spilled to host, the
#: interpreter) commits on top of it and is not predictable from the plan. The one
#: measurement there is: a pageable 512-token Mistral Large run that passed took the
#: system's commit headroom from 12.7-17.9 GB at its start down to 2.47 GB at its
#: lowest point, 10.2-15.4 GB of growth of which about 2.3 GB was its own staging.
#: 8 GiB is the low end of the other ~8 GB, rounded up, and it sorts the two recorded
#: starts the way the runs went: 12.7 GB less a 4 GiB pinned arena and one 0.8 GB
#: vocabulary slot leaves 7.6 GB (``tight``); 17.9 GB leaves 12.8 GB (``fits``).
#: Being wrong costs throughput (pageable staging loses the copy overlap), not
#: correctness, which is why a larger figure is cheaper than a smaller one.
COMMIT_HEADROOM_MARGIN_BYTES = 8 * 2**30


class CommitFit(str, Enum):
    """How a pinned staging sits against the commit still free."""

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
    #: ``available - required``; NEGATIVE when the staging does not fit at all.
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
    """Decide whether pinned staging leaves enough system commit.

    ``available`` is ``limit - total`` (0 when the total is over the limit, which
    happens transiently while the pagefile grows). The staging and the resident
    extras are charged against it, the same two figures the free-RAM check is
    given, so the two checks cannot disagree about what is being held. What is left
    decides the verdict:

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
    reading is skipped (with a debug-level note), not allowed to stop the training
    it was written to protect.
    """
    if not _on_windows():
        return None
    try:
        info = _query_commit_windows()
    except Exception as exc:  # noqa: BLE001 - any failure here skips the check
        logger.debug("Windows commit headroom could not be read (%r); skipping the check", exc)
        return None
    if info.limit_bytes <= 0:
        logger.debug(
            "Windows commit limit read as %d bytes; treating it as unreadable and skipping "
            "the check",
            info.limit_bytes,
        )
        return None
    return info


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------
def _gb(value: int) -> str:
    return f"{value / 1e9:.2f} GB"


def apply_commit_preflight(
    plan: "StreamPlan",
    *,
    stream_pin: Optional[bool],
    on_cuda: bool,
    resident_bytes: int,
    notify: Callable[[str], object],
) -> "StreamPlan":
    """Charge the disk tier's pinned staging against Windows commit.

    Returns ``plan`` itself when nothing changes, and a copy with ``pinned=False``
    and one added note (which the panel prints) when an unset ``stream_pin`` falls
    back to pageable staging. Raises ``ValueError`` when ``stream_pin: true`` asks
    for a staging that does not fit the commit limit at all; a merely ``tight`` one
    is honoured with one warning, because the user asked for it and the margin is a
    judgement. ``notify`` is ``console.print``.

    Nothing happens, and the commit counters are not read, when ``stream_pin`` is
    ``false`` (pageable already), the target is not CUDA (nothing is page-locked),
    the plan is already pageable, there is no staging to charge (the RAM tier), or
    the OS is not Windows.
    """
    if stream_pin is False or not on_cuda or not plan.pinned or plan.staging_bytes <= 0:
        return plan
    if not _on_windows():
        return plan
    info = read_commit_info()
    if info is None:
        logger.debug("Windows commit pre-flight skipped: the commit counters were unavailable")
        return plan
    verdict = assess_commit_fit(
        commit_total_bytes=info.total_bytes,
        commit_limit_bytes=info.limit_bytes,
        pinned_staging_bytes=plan.staging_bytes,
        resident_extras_bytes=resident_bytes,
    )
    if verdict.fit is CommitFit.FITS:
        return plan

    staging = _gb(plan.staging_bytes)
    extras = _gb(int(resident_bytes))
    free = _gb(verdict.available_bytes)
    margin = _gb(verdict.margin_bytes)
    committed = f"{_gb(info.total_bytes)} of {_gb(info.limit_bytes)} committed"
    if verdict.headroom_after_bytes >= 0:
        left, over = _gb(verdict.headroom_after_bytes), ""
    else:
        left, over = "none", f" (the charge is {_gb(-verdict.headroom_after_bytes)} over)"

    if stream_pin is True:
        if verdict.fit is CommitFit.DOES_NOT_FIT:
            raise ValueError(
                f"training.stream_pin=true but pinned host staging does not fit the Windows "
                f"commit limit: {staging} of staging + {extras} of resident extras = "
                f"{_gb(verdict.required_bytes)}, and only {free} of system commit is free "
                f"({committed}). Page-locked memory is charged against commit, not just free "
                f"RAM. Refusing before any weight is staged. Set training.stream_pin: false to "
                f"stage in pageable memory, close programs to free commit, or enlarge the "
                f"Windows pagefile."
            )
        notify(
            f"[yellow]training.stream_pin=true: pinned staging ({staging} + {extras} of "
            f"resident extras) leaves {_gb(verdict.headroom_after_bytes)} of the {free} of "
            f"Windows commit still free, under the {margin} margin. Proceeding as asked; if the "
            f"run is stopped for lack of commit, set training.stream_pin: false.[/]"
        )
        return plan

    line = (
        f"Windows commit headroom: pinned staging ({staging} + {extras} of resident extras) "
        f"would leave {left} of the {free} of system commit still free{over} ({committed}), "
        f"under the {margin} margin; staging in pageable host memory instead (copies no longer "
        f"overlap compute). training.stream_pin: true forces page-locking (it is refused only "
        f"when the staging cannot fit the commit limit at all)."
    )
    return dataclasses.replace(plan, pinned=False, notes=plan.notes + (line,))
