#!/usr/bin/env python3
"""The per-drive throttle diagnostic's decision rule, as code: bins, dips, episodes, verdict.

The record is ``benchmarks/probe-rtx5070-drive-throttle.md``; its section 2 was committed
before any arm ran and this module is that section in executable form. It is pure: no torch, no
Windows API, no disk read beyond the result files it is handed, so it imports and runs anywhere
and the driver can apply it to committed files alone (``--summarize``).

Inputs per arm, from the reader's JSON (``two_drive_throttle_read.py``) and the driver's 1-s
PDH samples: each drive's spans as ``(start_s, end_s, nbytes)`` relative to the arm's reading
start ``t0_unix``, and ``[unix time, {"read_c": bytes/s, ...}]`` rows.

    python two_drive_throttle_rule.py --selftest     # synthetic arms; the rule's own tests
    python two_drive_throttle_rule.py --calibrate    # the detector on the sustained record's PDH
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# -- the rule's constants; each value's source line is quoted in the record's section 2 --------
BIN_S = 5.0  # seconds per rule bin; 1-s bins are recorded beside them
DIP_GBS = 3.0  # a bin is a dip when BOTH instruments read STRICTLY below this, GB/s per drive
MIN_EPISODE_BINS = 3  # an episode is at least this many consecutive dip bins (15 s)
LEVEL_GBS = 3.5  # the best bin of a drive in an arm must reach this, or the arm is invalid
ARM_S = 300.0  # seconds each arm reads
COVERAGE_S = 285.0  # an arm that read for less is invalid: it cannot show an onset after that
PDH_MISSING_MAX = 0.10  # share of an arm's bins that may lack a PDH value
POOL_MIN_BYTES = 16 * 10**9  # the files an arm cycles over must hold at least this per drive
FOREIGN_READ_MAX = 10**9  # bytes another process may read during one arm
CONTIG_GAP_S = 120.0  # descriptive: reads of one drive closer than this count as continuous
PDH_MIN_SAMPLES = 3  # a 5-s bin needs this many of its 1-s PDH samples

#: The sequence: round 0, then round 1 in the reverse order. (round, arm).
ROUND_PLAN: Tuple[Tuple[int, str], ...] = (
    (0, "c"),
    (0, "d"),
    (0, "both"),
    (1, "both"),
    (1, "d"),
    (1, "c"),
)
ARM_DRIVES: Dict[str, Tuple[str, ...]] = {"c": ("c",), "d": ("d",), "both": ("c", "d")}
DRIVES = ("c", "d")

Span = Tuple[float, float, int]  # (start_s, end_s, nbytes), seconds from the arm's t0


# ==========================================================================
# bins
# ==========================================================================
def spans_of(drive: Dict[str, Any]) -> List[Span]:
    """The reader's compact span arrays as ``(start_s, end_s, nbytes)`` tuples."""
    packed = drive.get("spans") or {}
    nbytes = int(packed.get("nbytes", 0))
    return [
        (float(start), float(start) + float(dur), nbytes)
        for start, dur in zip(packed.get("start_s", []), packed.get("dur_s", []))
    ]


def own_rate_bins(spans: Sequence[Span], bin_s: float, n_bins: int) -> List[float]:
    """GB/s the reader delivered in each bin; a span's bytes are spread over its duration."""
    totals = [0.0] * n_bins
    for start, end, nbytes in spans:
        width = max(end - start, 1e-9)
        first = max(int(start // bin_s), 0)
        last = min(int(end // bin_s), n_bins - 1)
        for index in range(first, last + 1):
            overlap = min(end, (index + 1) * bin_s) - max(start, index * bin_s)
            if overlap > 0:
                totals[index] += nbytes * overlap / width
    return [total / bin_s / 1e9 for total in totals]


def pdh_bins(
    rows: Sequence[Sequence[Any]],
    key: str,
    t0: float,
    bin_s: float,
    n_bins: int,
    scale: float = 1e9,
    min_samples: int = PDH_MIN_SAMPLES,
) -> List[Optional[float]]:
    """Mean of the PDH samples of ``key`` stamped in ``(t0 + k*bin_s, t0 + (k+1)*bin_s]``.

    A rate counter reports the interval that ENDS at its time stamp, so a sample belongs to the
    bin its stamp falls in. ``None`` where fewer than ``min_samples`` samples landed.
    """
    buckets: List[List[float]] = [[] for _ in range(n_bins)]
    for stamp_s, row in rows:
        if key not in row:
            continue
        index = math.ceil((stamp_s - t0) / bin_s) - 1
        if 0 <= index < n_bins:
            buckets[index].append(row[key])
    return [
        statistics.fmean(vals) / scale if len(vals) >= min_samples else None for vals in buckets
    ]


def dip_flags(
    own: Sequence[float], pdh: Sequence[Optional[float]], threshold: float = DIP_GBS
) -> List[bool]:
    """A bin is a dip when the reader's own rate AND the volume's PDH rate are below threshold."""
    return [bool(o < threshold and p is not None and p < threshold) for o, p in zip(own, pdh)]


def runs_of(flags: Sequence[bool], min_bins: int = MIN_EPISODE_BINS) -> List[Tuple[int, int]]:
    """``(first_bin, n_bins)`` of every run of at least ``min_bins`` consecutive True flags."""
    out: List[Tuple[int, int]] = []
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        end = index
        while end < len(flags) and flags[end]:
            end += 1
        if end - index >= min_bins:
            out.append((index, end - index))
        index = end
    return out


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    kept = [value for value in values if value is not None]
    return statistics.fmean(kept) if kept else None


def _r(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


# ==========================================================================
# one drive in one arm, then the arm
# ==========================================================================
def analyze_drive(
    drive: str,
    spans: Sequence[Span],
    pdh_rows: Sequence[Sequence[Any]],
    t0: float,
    n_bins: int,
) -> Dict[str, Any]:
    """The rule's per-drive inputs for one arm: bins, dip flags, episodes, and context."""
    own = own_rate_bins(spans, BIN_S, n_bins)
    pdh = pdh_bins(pdh_rows, f"read_{drive}", t0, BIN_S, n_bins)
    flags = dip_flags(own, pdh)
    write = pdh_bins(pdh_rows, f"write_{drive}", t0, BIN_S, n_bins, scale=1e6, min_samples=1)
    cpu = pdh_bins(pdh_rows, "cpu_pct", t0, BIN_S, n_bins, scale=1.0, min_samples=1)
    episodes = []
    for first, count in runs_of(flags):
        span = slice(first, first + count)
        episodes.append(
            {
                "start_s": first * BIN_S,
                "length_s": count * BIN_S,
                "start_unix": t0 + first * BIN_S,
                "min_gbs": _r(min(own[span])),
                "mean_gbs": _r(statistics.fmean(own[span])),
                "write_mbs": _r(_mean(write[span]), 2),
                "cpu_pct": _r(_mean(cpu[span]), 1),
            }
        )
    starts = [ep["start_s"] for ep in episodes]
    spacings = [b - a for a, b in zip(starts, starts[1:])]
    total_s = int(n_bins * BIN_S)
    return {
        "n_bins": n_bins,
        "own_gbs_5s": [_r(v) for v in own],
        "pdh_gbs_5s": [_r(v) for v in pdh],
        "dip_bins": flags,
        "episodes": episodes,
        "period_s": statistics.median(spacings) if spacings else None,
        "dip_share": sum(flags) / n_bins if n_bins else None,
        "best_bin_gbs": _r(max(own)) if own else None,
        "median_bin_gbs": _r(statistics.median(own)) if own else None,
        "mean_own_gbs": _r(sum(n for _s, _e, n in spans) / max(total_s, 1) / 1e9 if spans else 0),
        "mean_pdh_gbs": _r(_mean(pdh)),
        "pdh_missing_bins": sum(1 for value in pdh if value is None),
        "own_gbs_1s": [_r(v, 2) for v in own_rate_bins(spans, 1.0, total_s)],
        "pdh_gbs_1s": [
            _r(v, 2) for v in pdh_bins(pdh_rows, f"read_{drive}", t0, 1.0, total_s, min_samples=1)
        ],
    }


def analyze_arm(
    arm: str, worker: Dict[str, Any], pdh_rows: Sequence[Sequence[Any]]
) -> Dict[str, Any]:
    """Rule row 1's per-arm inputs and every drive's episodes, from the reader's JSON + PDH."""
    expected = ARM_DRIVES[arm]
    recs = worker.get("drives") or {}
    problems: List[str] = []
    if set(recs) != set(expected):
        problems.append(f"drives read {sorted(recs)}, not {list(expected)}")
    if worker.get("pinned") is not True:
        problems.append(f"pinned {worker.get('pinned')!r}, not True")
    spans = {drive: spans_of(recs[drive]) for drive in expected if drive in recs}
    ends = [sp[-1][1] for sp in spans.values() if sp]
    out: Dict[str, Any] = {"arm": arm, "t0_unix": worker.get("t0_unix"), "drives": {}}
    if len(ends) != len(expected):
        problems.append("a drive has no spans")
        out["problems"] = problems
        return out
    read_s = min(ends)
    out["read_s"] = read_s
    if read_s < COVERAGE_S:
        problems.append(f"read {read_s:.1f} s, below the {COVERAGE_S:.0f} s coverage floor")
    n_bins = int(min(float(worker.get("read_s_planned") or 0.0), read_s) // BIN_S)
    for drive in expected:
        rec = recs[drive]
        info = analyze_drive(drive, spans[drive], pdh_rows, float(worker["t0_unix"]), n_bins)
        out["drives"][drive] = info
        if rec.get("check_ok") is not True:
            problems.append(f"{drive.upper()}: byte check {rec.get('check_ok')!r}")
        if int(rec.get("pool_bytes") or 0) < POOL_MIN_BYTES:
            problems.append(
                f"{drive.upper()}: pool {rec.get('pool_bytes')!r} bytes < {POOL_MIN_BYTES}"
            )
        best = info["best_bin_gbs"]
        if best is None or best < LEVEL_GBS:
            problems.append(f"{drive.upper()}: best bin {best!r} GB/s below {LEVEL_GBS}")
        if info["pdh_missing_bins"] > PDH_MISSING_MAX * n_bins:
            problems.append(
                f"{drive.upper()}: PDH missing in {info['pdh_missing_bins']}/{n_bins} bins"
            )
    if len(expected) == 2:  # both drives: how often the other drive dipped inside an episode
        for drive, other in (("c", "d"), ("d", "c")):
            flags_other = out["drives"][other]["dip_bins"]
            for ep in out["drives"][drive]["episodes"]:
                first = int(ep["start_s"] // BIN_S)
                span = flags_other[first : first + int(ep["length_s"] // BIN_S)]
                ep["other_drive_dip_share"] = _r(sum(span) / len(span)) if span else None
    out["problems"] = problems
    return out


def annotate_continuity(arms: Dict[Tuple[int, str], Dict[str, Any]]) -> None:
    """Add ``continuous_read_s`` to every episode: how long its drive had been read, with gaps
    shorter than CONTIG_GAP_S, when the episode began. Descriptive; no row reads it."""
    for drive in DRIVES:
        intervals = []
        for pos in ROUND_PLAN:
            rec = arms.get(pos)
            if not rec or drive not in rec["drives"] or rec.get("t0_unix") is None:
                continue
            intervals.append((rec["t0_unix"], rec["t0_unix"] + rec["read_s"]))
        merged: List[List[float]] = []
        for start, end in sorted(intervals):
            if merged and start - merged[-1][1] <= CONTIG_GAP_S:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for rec in arms.values():
            if drive not in rec["drives"]:
                continue
            for ep in rec["drives"][drive]["episodes"]:
                owner = [m for m in merged if m[0] <= ep["start_unix"] <= m[1]]
                ep["continuous_read_s"] = _r(ep["start_unix"] - owner[0][0], 1) if owner else None


# ==========================================================================
# the verdict
# ==========================================================================
def _episodes(arms: Dict[Tuple[int, str], Dict[str, Any]], rnd: int, arm: str, drive: str) -> int:
    return len(arms[(rnd, arm)]["drives"][drive]["episodes"])


def drive_row(arms: Dict[Tuple[int, str], Dict[str, Any]], drive: str) -> Tuple[str, str]:
    """Rows 2-5 for one drive, first match decides: ``(row, label)``."""
    alone = [_episodes(arms, rnd, drive, drive) for rnd in (0, 1)]
    both = [_episodes(arms, rnd, "both", drive) for rnd in (0, 1)]
    if all(alone):
        return "2", "THROTTLES"
    if not any(alone) and any(both):
        return "3", "ONLY WHEN BOTH"
    if not any(alone) and not any(both):
        return "4", "STEADY"
    return "5", "INCONSISTENT"


def verdict(
    arms: Dict[Tuple[int, str], Dict[str, Any]], outcome: str
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Section 2 applied: ``([(row, result), ...], {drive: label})``."""
    rows: List[Tuple[str, str]] = []
    problems = [
        f"r{rnd} {arm}: {why}"
        for rnd, arm in ROUND_PLAN
        if (rnd, arm) in arms
        for why in arms[(rnd, arm)]["problems"]
    ]
    missing = [f"r{rnd} {arm}" for rnd, arm in ROUND_PLAN if (rnd, arm) not in arms]
    if missing:
        problems.append(f"no valid arm: {missing}")
    if outcome != "complete":
        problems.append(f"protocol: {outcome}")
    rows.append(("1 validity", "fires: " + "; ".join(problems) if problems else "does not fire"))
    if problems:
        return rows, {drive: "NO VERDICT" for drive in DRIVES}
    labels = {}
    for drive in DRIVES:
        row, label = drive_row(arms, drive)
        rows.append((f"{row} drive {drive.upper()}", label))
        labels[drive] = label
    return rows, labels


# ==========================================================================
# the detector on the sustained record's own PDH data
# ==========================================================================
def calibrate(results_dir: str) -> int:
    """Run the detector (PDH only: that record has no reader spans) on the sustained arms."""
    with open(os.path.join(results_dir, "sustain_driver.json"), encoding="utf-8") as handle:
        runs = {run["run"]: run for run in json.load(handle)["runs"] if "during_samples" in run}
    for name in ("sustain_r0_single", "sustain_r0_striped", "sustain_r1_striped"):
        with open(os.path.join(results_dir, f"{name}.json"), encoding="utf-8") as handle:
            arm = json.load(handle)
        plain = next(r for r in arm["records"] if r.get("label") == "step_plain")["records"]
        t0, t1 = plain[0]["started_unix"], plain[-1]["started_unix"] + plain[-1]["step_s"]
        n_bins = int((t1 - t0) // BIN_S)
        pdh_rows = runs[name]["during_samples"]["pdh"]
        for drive in DRIVES:
            bins = pdh_bins(pdh_rows, f"read_{drive}", t0, BIN_S, n_bins)
            if _mean(bins) is None or _mean(bins) < 0.5:
                continue
            flags = [value is not None and value < DIP_GBS for value in bins]
            found = [
                (first * BIN_S, count * BIN_S, round(min(bins[first : first + count]), 2))
                for first, count in runs_of(flags)
            ]
            print(
                f"{name} {drive.upper()}: {len(found)} episode(s) "
                f"(start s, length s, min GB/s) {found}"
            )
    return 0


# ==========================================================================
# self-test: synthetic arms, no disk, no GPU
# ==========================================================================
def _synth_spans(
    rate: Callable[[float], float], seconds: float, nbytes: int = 419430400
) -> List[Span]:
    spans, now = [], 0.0
    while now < seconds:
        end = now + nbytes / (rate(now) * 1e9)
        spans.append((now, end, nbytes))
        now = end
    return spans


def _synth_inputs(
    rates: Dict[str, Callable[[float], float]], seconds: float = 300.0, t0: float = 1e9
) -> Tuple[Dict[str, Any], List[Any]]:
    """A reader JSON and the PDH rows a driver would have sampled, for the given rate curves."""
    worker: Dict[str, Any] = {"t0_unix": t0, "read_s_planned": ARM_S, "pinned": True, "drives": {}}
    pdh = []
    for stamp in range(1, int(seconds) + 6):
        pdh.append([t0 + stamp, {f"read_{d}": rates[d](stamp - 0.5) * 1e9 for d in rates}])
    for drive, rate in rates.items():
        spans = _synth_spans(rate, seconds)
        worker["drives"][drive] = {
            "spans": {
                "start_s": [s for s, _e, _n in spans],
                "dur_s": [e - s for s, e, _n in spans],
                "nbytes": spans[0][2],
            },
            "check_ok": True,
            "pool_bytes": 40 * 419430400,
        }
    return worker, pdh


def _synth_arm(
    arm: str,
    rates: Dict[str, Callable[[float], float]],
    seconds: float = 300.0,
    t0: float = 1e9,
    tweak: Optional[Callable[[Dict[str, Any], List[Any]], None]] = None,
) -> Dict[str, Any]:
    worker, pdh = _synth_inputs(rates, seconds, t0)
    if tweak is not None:
        tweak(worker, pdh)
    return analyze_arm(arm, worker, pdh)


def _quiet(_t: float) -> float:
    return 4.0


def _dipping(start: float, length: float, low: float = 2.5) -> Callable[[float], float]:
    return lambda t: low if start <= t < start + length else 4.0


def _fake_arms(alone: Dict[str, Tuple[int, int]], both: Dict[str, Tuple[int, int]]):
    """Round arms where a drive dips (or not) in each place the flags say."""
    arms: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for position, (rnd, arm) in enumerate(ROUND_PLAN):
        flags = {d: (alone[d][rnd] if arm == d else both[d][rnd]) for d in ARM_DRIVES[arm]}
        rates = {d: _dipping(100.0, 30.0) if flags[d] else _quiet for d in ARM_DRIVES[arm]}
        arms[(rnd, arm)] = _synth_arm(arm, rates, t0=1e9 + position * 360.0)  # 60 s between arms
    return arms


def selftest() -> int:
    ok = {"c": (0, 0), "d": (0, 0)}
    # binning: bytes are conserved and pro-rated across a bin edge
    bins = own_rate_bins([(4.0, 6.0, 10**9)], 5.0, 2)
    assert abs(bins[0] - 0.1) < 1e-9 and abs(bins[1] - 0.1) < 1e-9, bins
    # pdh bins: stamp on a bin's closing edge belongs to it; a stamp at t0 itself to none
    rows = [[100.0, {"k": 9e9}]] + [[100.0 + s, {"k": 1e9 * s}] for s in range(1, 11)]
    assert pdh_bins(rows, "k", 100.0, 5.0, 2) == [3.0, 8.0], pdh_bins(rows, "k", 100.0, 5.0, 2)
    assert pdh_bins(rows[:3], "k", 100.0, 5.0, 1) == [None]
    # the threshold is strict and needs both instruments
    assert dip_flags([2.99, 3.0, 2.99], [2.99, 2.5, 3.01]) == [True, False, False]
    assert dip_flags([1.0], [None]) == [False]
    # an episode needs three bins
    assert runs_of([False, True, True, False, True, True, True]) == [(4, 3)]
    # arms: quiet, one 30-s dip, a 10-s dip that is too short
    assert _synth_arm("c", {"c": _quiet})["drives"]["c"]["episodes"] == []
    arm = _synth_arm("c", {"c": _dipping(100.0, 30.0)})
    episodes = arm["drives"]["c"]["episodes"]
    assert len(episodes) == 1 and episodes[0]["start_s"] == 100.0, episodes
    assert episodes[0]["length_s"] == 30.0 and episodes[0]["min_gbs"] < 2.6, episodes
    assert not arm["problems"], arm["problems"]
    assert _synth_arm("c", {"c": _dipping(100.0, 10.0)})["drives"]["c"]["episodes"] == []
    # PDH that does not confirm a dip in the reader's own bytes is not an episode
    worker = {"t0_unix": 1e9, "read_s_planned": ARM_S, "pinned": True, "drives": {}}
    spans = _synth_spans(_dipping(100.0, 30.0), 300.0)
    worker["drives"]["c"] = {
        "spans": {
            "start_s": [s for s, _e, _n in spans],
            "dur_s": [e - s for s, e, _n in spans],
            "nbytes": spans[0][2],
        },
        "check_ok": True,
        "pool_bytes": 40 * 419430400,
    }
    pdh = [[1e9 + s, {"read_c": 4e9}] for s in range(1, 306)]
    assert analyze_arm("c", worker, pdh)["drives"]["c"]["episodes"] == []
    # validity: a short arm and a drive that never reaches the level
    short = _synth_arm("c", {"c": _quiet}, seconds=240.0)
    assert any("coverage" in p for p in short["problems"]), short["problems"]
    slow = _synth_arm("c", {"c": lambda t: 3.2})
    assert any("best bin" in p for p in slow["problems"]), slow["problems"]

    def _unpinned(worker: Dict[str, Any], _pdh: List[Any]) -> None:
        worker["pinned"] = False

    def _bad_bytes(worker: Dict[str, Any], _pdh: List[Any]) -> None:
        worker["drives"]["c"]["check_ok"] = False

    def _small_pool(worker: Dict[str, Any], _pdh: List[Any]) -> None:
        worker["drives"]["c"]["pool_bytes"] = POOL_MIN_BYTES - 1

    def _no_pdh(_worker: Dict[str, Any], pdh: List[Any]) -> None:
        pdh.clear()

    for tweak, word in (
        (_unpinned, "pinned"),
        (_bad_bytes, "byte check"),
        (_small_pool, "pool"),
        (_no_pdh, "PDH missing"),
    ):
        bad = _synth_arm("c", {"c": _quiet}, tweak=tweak)
        assert any(word in p for p in bad["problems"]), (word, bad["problems"])
    assert _synth_arm("c", {"c": _quiet})["problems"] == []
    wrong = _synth_arm("both", {"c": _quiet})  # a BOTH arm that read only one drive
    assert any("drives read" in p for p in wrong["problems"]), wrong["problems"]
    # the four per-drive rows, and the validity row
    cases = [
        ({"c": (0, 0), "d": (1, 1)}, {"c": (0, 0), "d": (1, 1)}, ("STEADY", "THROTTLES")),
        (ok, {"c": (0, 0), "d": (0, 1)}, ("STEADY", "ONLY WHEN BOTH")),
        (ok, ok, ("STEADY", "STEADY")),
        ({"c": (1, 0), "d": (0, 0)}, ok, ("INCONSISTENT", "STEADY")),
        ({"c": (1, 1), "d": (0, 0)}, {"c": (0, 0), "d": (1, 0)}, ("THROTTLES", "ONLY WHEN BOTH")),
    ]
    for alone, both, want in cases:
        arms = _fake_arms(alone, both)
        rows, labels = verdict(arms, "complete")
        assert (labels["c"], labels["d"]) == want, (alone, both, rows)
        assert all(not rec["problems"] for rec in arms.values())
    arms = _fake_arms(ok, ok)
    assert verdict(arms, "stopped: x")[1]["c"] == "NO VERDICT"
    del arms[(1, "d")]
    assert verdict(arms, "complete")[1]["d"] == "NO VERDICT"
    arms = _fake_arms({"c": (1, 1), "d": (0, 0)}, {"c": (0, 1), "d": (0, 1)})
    annotate_continuity(arms)

    # The arms start 360 s apart and read 300 s, so 60 s of rest between them. C: is read at
    # positions 0, 2, 3, 5: 420 s idle before position 2, 60 s before 3, 420 s before 5. D: is
    # read at 1-4 with 60 s between. Every episode begins 100 s into its arm.
    def continuous(pos: Tuple[int, str], drive: str) -> List[float]:
        return [ep["continuous_read_s"] for ep in arms[pos]["drives"][drive]["episodes"]]

    assert continuous((0, "c"), "c") == [100.0], continuous((0, "c"), "c")
    assert continuous((1, "c"), "c") == [100.0], continuous((1, "c"), "c")
    assert continuous((1, "both"), "c") == [460.0], continuous((1, "both"), "c")
    assert continuous((1, "both"), "d") == [820.0], continuous((1, "both"), "d")
    # in a BOTH arm an episode records whether the other drive dipped inside it
    together = _synth_arm("both", {"c": _dipping(100.0, 30.0), "d": _dipping(100.0, 30.0)})
    assert together["drives"]["c"]["episodes"][0]["other_drive_dip_share"] == 1.0, together
    apart = _synth_arm("both", {"c": _dipping(100.0, 30.0), "d": _quiet})
    assert apart["drives"]["c"]["episodes"][0]["other_drive_dip_share"] == 0.0, apart
    assert apart["drives"]["d"]["episodes"] == [] and not apart["problems"], apart
    # the reader's loop: wraps, stops at the deadline, logs one entry per span
    from two_drive_throttle_read import read_cyclic

    clock = {"now": 0.0}
    seen: List[str] = []

    def fake_read(span: Tuple[str, int, int]) -> None:
        seen.append(span[0])
        clock["now"] += 0.25

    begin, log = read_cyclic(
        [("a", 0, 1), ("b", 0, 1), ("c", 0, 1)], fake_read, 2.0, None, lambda: clock["now"]
    )
    assert seen == ["a", "b", "c", "a", "b", "c", "a", "b"] and len(log) == 8, seen
    assert begin == 0.0 and log[0] == (0.0, 0.25, 1) and log[-1][1] == 2.0, log
    print("selftest: all assertions passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "probe-rtx5070",
            "two-drive",
        ),
    )
    args = parser.parse_args()
    if args.calibrate:
        return calibrate(args.results_dir)
    if args.selftest:
        return selftest()
    parser.error("one of --selftest / --calibrate")
    return 2


if __name__ == "__main__":
    sys.exit(main())
