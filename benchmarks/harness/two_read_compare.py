"""Series B comparator: the pre-registered between-run rule for one shape's four runs.

Run order is A1 B1 B2 A2 (A = three-read, B = two-read ``two_read_experts``), one fresh
process per run, each in its own cgroup leaf under a series parent cgroup. Every input is a
run record of schema ``two-read-cgroup-run/2`` written by the process that supervised the
run (that supervisor is not in this repository); its step JSON is
``record["expect_json"]["path"]``, used as is when absolute and resolved against the
record's ``cwd`` (which must then be absolute) when relative.

``--step-dir DIR`` re-checks runs whose files were moved off the host that wrote them (a
published record): each step JSON is then read from DIR under the base name of its
``expect_json.path``, and the record's path is not opened. The binding below is
unchanged, so a file in DIR only counts if it is the one the supervisor judged.

Each step JSON is bound to its record: the SHA-256 of its bytes must equal
``expect_json.sha256`` (the bytes the supervisor judged; the step JSON is parsed from the
bytes that were hashed) and its top-level ``supervisor_run_id`` must equal the record's
``run_id``. Roles come from argument position, so the four record files and the four step
JSON files (both by real path), the four ``run_id`` values and the four step SHA-256 values
must each be distinct: one run named twice, or a record copied under another file name,
refuses.

Verdicts and exit codes (``--out`` is written atomically in every case):

* ``REFUSED`` (2): a precondition fails (record schema, the binding and distinctness
  above, run status, verdicts, phase, arms, every run in ``--deterministic strict``,
  identical configuration / environment / numerics settings / attention and experts
  implementation / initial trainable and input-id digests / leaf cgroup read-back /
  series parent path, step counts, complete gradient digests, a finite loss at every
  step: ``float.hex`` gives one string for NaN and for each infinity, so equal non-finite
  losses would otherwise compare equal; an integer at or beyond 2**63 counts as no
  number). With
  ``--memory-max`` / ``--swap-max`` every record's leaf ``cgroup.readback`` and series
  parent ``series.readback`` must show those values; a record without ``series`` (a run
  with no limit on the series parent) refuses, since the leaf limit alone does not count
  page cache charged to the parent by earlier runs. Every reason is listed; no timing is
  reported. A step JSON that cannot be read, parsed or matched to its hash skips the
  content checks.
* ``FAIL`` (1): at some step index the losses (``float.hex``), gradient digests or dense /
  large H2D bytes differ among the four, or an A expert quantity times 2 differs from a B
  one times 3 (or is zero) for ``expert_fetches``, ``expert_bytes_read``,
  ``h2d_bytes.experts``. Every failing index and quantity is listed.
* ``PASS`` (0): otherwise. Timing is reported and never changes the verdict. A reported
  value that is not a usable number (seconds, a peak, the supervisor's memory peak) is
  shown as ``-``, never refused.

    python benchmarks/harness/two_read_compare.py \\
        --runs A1.run.json B1.run.json B2.run.json A2.run.json --out compare.json \\
        --memory-max 34359738368 --swap-max 0 [--step-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

SCHEMA = "two-read-compare/1"
RECORD_SCHEMA = "two-read-cgroup-run/2"
RUNS = ("A1", "B1", "B2", "A2")
TWO_READ = {"A1": False, "B1": True, "B2": True, "A2": False}
ARGS_IGNORED = ("two_read_experts", "out", "label")
EXPERT_QUANTITIES = ("expert_fetches", "expert_bytes_read", "h2d_bytes.experts")
EQUAL_QUANTITIES = ("h2d_bytes.dense", "h2d_bytes.large")
STEP0_DROPPED = "after guest page-cache drop; host cache not controlled"
STEP0_PLAIN = "first step (no cache drop)"
CLAIMS = (
    "random weights / shape-only for synthetic shapes; a read reduction does not imply a "
    "1.5x speedup; emulated-envelope runs are not physical 8 GB evidence"
)
EXIT = {"PASS": 0, "FAIL": 1, "REFUSED": 2}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_MISSING = object()

console = Console()


def _get(data: Any, dotted: str, default: Any = _MISSING) -> Any:
    """``data["a"]["b"]`` for ``"a.b"``; ``default`` when any level is absent."""
    for key in dotted.split("."):
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


#: Larger than any counter a run writes (bytes, fetches, steps). JSON allows integers of any
#: size; one beyond this counts as a non-number (refused, or ignored where only reported), so
#: ``float()`` and ``str()`` of these values and of their sums cannot raise.
_INT_LIMIT = 2**63


def _is_int(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and -_INT_LIMIT < value < _INT_LIMIT)


def _is_number(value: Any) -> bool:
    return isinstance(value, float) or _is_int(value)


def _shown(value: Any) -> str:
    """``repr`` of a step value for a refusal, cut to 60 characters: a JSON integer can have
    thousands of digits, and ``repr`` of one past 4,300 digits raises."""
    if isinstance(value, int) and not isinstance(value, bool) and not _is_int(value):
        return "<integer beyond 2**63>"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_run(name: str, record_path: str, refusals: List[str],
             step_dir: Optional[str] = None) -> Dict[str, Any]:
    """The record and its step JSON; problems are appended to ``refusals``.

    ``step`` stays None unless the step JSON bytes hash to the record's
    ``expect_json.sha256``; it is parsed from those same bytes. With ``step_dir`` the step
    JSON is ``step_dir/<base name of expect_json.path>``."""
    run: Dict[str, Any] = {"name": name, "record_path": os.path.abspath(record_path),
                           "record": None, "run_id": None, "step_path": None,
                           "step_sha256": None, "step": None}
    try:
        record = _read_json(record_path)
    except (OSError, ValueError) as exc:
        refusals.append(f"{name}: run record {record_path} unreadable: {exc}")
        return run
    if not isinstance(record, dict):
        refusals.append(f"{name}: run record {record_path} is not a JSON object")
        return run
    run["record"] = record
    if record.get("schema") != RECORD_SCHEMA:
        refusals.append(f"{name}: run record schema {record.get('schema')!r} != {RECORD_SCHEMA}")
    if record.get("status") != "ok":
        refusals.append(f"{name}: supervisor status {record.get('status')!r} "
                        f"(reason {record.get('reason')!r}), expected 'ok'")
    run_id = record.get("run_id")
    if isinstance(run_id, str) and run_id:
        run["run_id"] = run_id
    else:
        refusals.append(f"{name}: run record run_id {run_id!r} is not a non-empty string")
    expect_path = _get(record, "expect_json.path", None)
    if not isinstance(expect_path, str) or not expect_path:
        refusals.append(f"{name}: run record names no step JSON (expect_json.path)")
        return run
    if step_dir is not None:
        # The record was written on another host, so its path may use either separator.
        base = expect_path.replace("\\", "/").rsplit("/", 1)[-1]
        if base in ("", ".", ".."):
            refusals.append(f"{name}: expect_json.path {expect_path!r} has no file name")
            return run
        expect_path = os.path.join(step_dir, base)
    elif not os.path.isabs(expect_path):
        cwd = record.get("cwd")
        if not isinstance(cwd, str) or not os.path.isabs(cwd):
            refusals.append(f"{name}: expect_json.path {expect_path!r} is relative and record "
                            f"cwd {cwd!r} is not an absolute path")
            return run
        expect_path = os.path.join(cwd, expect_path)
    run["step_path"] = os.path.realpath(expect_path)
    if not os.path.isfile(run["step_path"]):
        refusals.append(f"{name}: step JSON {run['step_path']} does not exist")
        return run
    try:
        with open(run["step_path"], "rb") as handle:
            data = handle.read()
    except OSError as exc:
        refusals.append(f"{name}: step JSON {run['step_path']} unreadable: {exc}")
        return run
    run["step_sha256"] = hashlib.sha256(data).hexdigest()
    expected = _get(record, "expect_json.sha256", None)
    if run["step_sha256"] != expected:
        refusals.append(f"{name}: step JSON {run['step_path']} sha256 {run['step_sha256']} != "
                        f"record expect_json.sha256 {expected!r} (not the bytes the "
                        f"supervisor judged)")
        return run
    try:
        step = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        refusals.append(f"{name}: step JSON {run['step_path']} unreadable: {exc}")
        return run
    if not isinstance(step, dict):
        refusals.append(f"{name}: step JSON {run['step_path']} is not a JSON object")
        return run
    step_run_id = step.get("supervisor_run_id")
    if not isinstance(step_run_id, str) or step_run_id != run_id:
        refusals.append(f"{name}: step JSON supervisor_run_id {step_run_id!r} != record "
                        f"run_id {run_id!r}")
    run["step"] = step
    return run


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def _per_run_checks(run: Dict[str, Any], refusals: List[str]) -> None:
    name, step = run["name"], run["step"]
    status = _get(step, "run_verdict.status", None)
    if status != "ok":
        refusals.append(f"{name}: run_verdict.status {status!r}, expected 'ok'")
    memory = step.get("memory_verdict")
    if not isinstance(memory, dict):
        refusals.append(f"{name}: memory_verdict is null; timing runs must be memory-judged")
    elif memory.get("status") != "ok":
        refusals.append(f"{name}: memory_verdict.status {memory.get('status')!r} "
                        f"(reason {memory.get('reason')!r}), expected 'ok'")
    if step.get("phase") != "done":
        refusals.append(f"{name}: phase {step.get('phase')!r}, expected 'done'")
    deterministic = _get(step, "args.deterministic", None)
    if deterministic != "strict":
        refusals.append(f"{name}: args.deterministic {deterministic!r}, expected 'strict' "
                        f"(without it the SDPA backward differs run to run: the failed "
                        f"first Moonlight session in #2170)")
    arm = step.get("two_read_experts")
    if arm is not TWO_READ[name]:
        refusals.append(f"{name}: two_read_experts {arm!r}, expected {TWO_READ[name]} "
                        f"(order A1 B1 B2 A2; A = three-read, B = two-read)")
    steps = step.get("steps")
    if not isinstance(steps, list) or not steps:
        refusals.append(f"{name}: no steps")
        return
    for index, entry in enumerate(steps):
        where = f"{name} step {index}"
        if not isinstance(entry, dict):
            refusals.append(f"{where}: not an object")
            continue
        if entry.get("grads_complete") is not True:
            refusals.append(f"{where}: grads_complete {_shown(entry.get('grads_complete'))}")
        digest = entry.get("lora_grads_sha256")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            refusals.append(f"{where}: lora_grads_sha256 {_shown(digest)} is not a 64-hex digest")
        loss = entry.get("loss")
        if not _is_number(loss) or not math.isfinite(loss):
            refusals.append(f"{where}: loss {_shown(loss)} is not a finite number")
        for key in EXPERT_QUANTITIES + EQUAL_QUANTITIES:
            if not _is_int(_get(entry, key, None)):
                refusals.append(f"{where}: {key} {_shown(_get(entry, key, None))} "
                                f"is not an integer")


def _args_view(step: Dict[str, Any]) -> Any:
    args = step.get("args")
    if not isinstance(args, dict):
        return _MISSING
    return {key: value for key, value in args.items() if key not in ARGS_IGNORED}


_STEP_FIELDS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "args (without two_read_experts/out/label)": _args_view,
    "evidence": lambda s: s.get("evidence", _MISSING),
    "store_bytes": lambda s: s.get("store_bytes", _MISSING),
    "store_expert_layout": lambda s: s.get("store_expert_layout", _MISSING),
    "environment": lambda s: s.get("environment", _MISSING),
    "numerics": lambda s: s.get("numerics", _MISSING),
    "attn_implementation": lambda s: s.get("attn_implementation", _MISSING),
    "experts_implementation": lambda s: s.get("experts_implementation", _MISSING),
    "initial_trainable_sha256": lambda s: s.get("initial_trainable_sha256", _MISSING),
    "input_ids_sha256": lambda s: s.get("input_ids_sha256", _MISSING),
    "memory.mode": lambda s: _get(s, "memory.mode"),
    # The inputs only: context overhead and the fraction derived from it are measured per
    # process and differ by a few MiB between fresh processes of the same configuration.
    "envelope (target, card total, margin)": lambda s: None if s.get("envelope") is None else {
        key: s["envelope"].get(key) for key in ("target_mib", "total_mib", "margin_mib")
    },
    "nvml_device.uuid": lambda s: _get(s, "nvml_device.uuid", None),
    "memory.nvml_total_bytes": lambda s: _get(s, "memory.nvml_total_bytes", None),
}
_RECORD_FIELDS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "cgroup.readback": lambda r: _get(r, "cgroup.readback"),
    "host.hostname": lambda r: _get(r, "host.hostname"),
    "series.path": lambda r: _get(r, "series.path", None),
}


def _equal_across(label: str, values: Dict[str, Any], refusals: List[str]) -> None:
    missing = [name for name, value in values.items() if value is _MISSING]
    if missing:
        refusals.append(f"{label} missing in {', '.join(missing)}")
        return
    first = values[RUNS[0]]
    if any(value != first for value in values.values()):
        shown = "; ".join(f"{name}={json.dumps(value, sort_keys=True, default=str)}"
                          for name, value in values.items())
        refusals.append(f"{label} differs across runs: {shown}")


def _distinct(label: str, values: Dict[str, Optional[str]], refusals: List[str]) -> None:
    """Refuse every value shared by two or more runs (None was already refused at load)."""
    holders: Dict[str, List[str]] = {}
    for name, value in values.items():
        if value is not None:
            holders.setdefault(value, []).append(name)
    for value, names in holders.items():
        if len(names) > 1:
            refusals.append(f"{' and '.join(names)} share one {label} ({value}); the four "
                            f"runs must be distinct")


def distinct_refusals(runs: Dict[str, Dict[str, Any]], refusals: List[str]) -> None:
    """Roles come from argument position: the same run must not fill two of them."""
    _distinct("run record file", {n: os.path.realpath(runs[n]["record_path"]) for n in RUNS},
              refusals)
    _distinct("step JSON file", {n: runs[n]["step_path"] for n in RUNS}, refusals)
    _distinct("run_id", {n: runs[n]["run_id"] for n in RUNS}, refusals)
    _distinct("step JSON sha256", {n: runs[n]["step_sha256"] for n in RUNS}, refusals)


def _readback_refusals(name: str, where: str, readback: Any,
                       limits: List[Tuple[str, str, int]], refusals: List[str]) -> None:
    for flag, key, wanted in limits:
        got = readback.get(key) if isinstance(readback, dict) else None
        if not isinstance(got, str) or got.strip() != str(wanted):
            refusals.append(f"{name}: {where} readback {key} {got!r} != {flag} {wanted}")


def refusals_for(runs: Dict[str, Dict[str, Any]], memory_max: Optional[int],
                 swap_max: Optional[int], refusals: List[str]) -> None:
    """Append every precondition failure; ``runs`` must all have a record and a step JSON."""
    for run in runs.values():
        _per_run_checks(run, refusals)
    for label, getter in _STEP_FIELDS.items():
        _equal_across(label, {n: getter(runs[n]["step"]) for n in RUNS}, refusals)
    for label, getter in _RECORD_FIELDS.items():
        _equal_across(f"supervisor {label}", {n: getter(runs[n]["record"]) for n in RUNS},
                      refusals)
    limits = [(flag, key, wanted) for flag, key, wanted in (
        ("--memory-max", "memory.max", memory_max),
        ("--swap-max", "memory.swap.max", swap_max)) if wanted is not None]
    if limits:
        # The leaf limit only counts the run's own pages; page cache left by earlier runs is
        # charged to the series parent, so the parent must carry the limit too.
        for name in RUNS:
            record = runs[name]["record"]
            _readback_refusals(name, "cgroup", _get(record, "cgroup.readback", None), limits,
                               refusals)
            series = record.get("series")
            if isinstance(series, dict):
                _readback_refusals(name, "series", series.get("readback"), limits, refusals)
            else:
                refusals.append(f"{name}: series {series!r} is not a series parent record; "
                                f"{' / '.join(flag for flag, _, _ in limits)} require a limit "
                                f"on the series parent cgroup as well as on the run's leaf")
    counts = {n: len(runs[n]["step"].get("steps") or []) for n in RUNS
              if isinstance(runs[n]["step"].get("steps"), list)}
    if len(set(counts.values())) > 1:
        refusals.append("step counts differ: "
                        + ", ".join(f"{n}={c}" for n, c in counts.items()))


def failures_for(steps: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Every (index, quantity) where the four runs break the rule."""
    failures: List[Dict[str, Any]] = []

    def fail(index: int, quantity: str, values: Dict[str, Any], why: str) -> None:
        failures.append({"index": index, "quantity": quantity, "values": values,
                         "reason": why})

    for index in range(len(steps[RUNS[0]])):
        at = {n: steps[n][index] for n in RUNS}
        losses = {n: float.hex(float(at[n]["loss"])) for n in RUNS}
        if len(set(losses.values())) > 1:
            fail(index, "loss", losses, "float.hex(loss) differs")
        digests = {n: at[n]["lora_grads_sha256"] for n in RUNS}
        if len(set(digests.values())) > 1:
            fail(index, "lora_grads_sha256", digests, "gradient digests differ")
        for quantity in EXPERT_QUANTITIES:
            values = {n: _get(at[n], quantity) for n in RUNS}
            zero = [n for n in RUNS if values[n] == 0]
            if zero:
                fail(index, quantity, values, f"zero in {', '.join(zero)}")
            bad = [f"{a}*2={values[a] * 2} != {b}*3={values[b] * 3}"
                   for a in ("A1", "A2") for b in ("B1", "B2")
                   if values[a] * 2 != values[b] * 3]
            if bad:
                fail(index, quantity, values, "3:2 violated: " + "; ".join(bad))
        for quantity in EQUAL_QUANTITIES:
            values = {n: _get(at[n], quantity) for n in RUNS}
            if len(set(values.values())) > 1:
                fail(index, quantity, values, "differs across runs")
    return failures


def classify(ratios: List[Optional[float]]) -> str:
    if all(r is not None and r < 1 for r in ratios):
        return "two-read faster"
    if all(r is not None and r > 1 for r in ratios):
        return "two-read slower"
    return "inconclusive"


def _seconds(entry: Dict[str, Any]) -> Optional[float]:
    value = entry.get("seconds")
    return float(value) if _is_number(value) and math.isfinite(value) and value >= 0 else None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    return num / den if num is not None and den is not None and den > 0 else None


def timing_for(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Report only: unusable seconds become null ratios / ``inconclusive``, never a refusal."""
    steps = {n: runs[n]["step"]["steps"] for n in RUNS}
    rows = []
    for index in range(len(steps[RUNS[0]])):
        seconds = {n: _seconds(steps[n][index]) for n in RUNS}
        ratios = {"B1/A1": _ratio(seconds["B1"], seconds["A1"]),
                  "B2/A2": _ratio(seconds["B2"], seconds["A2"])}
        rows.append({"index": index, "seconds": seconds, "ratios": ratios,
                     "class": classify(list(ratios.values()))})
    dropped = all(runs[n]["step"].get("drop_caches") == "ok" for n in RUNS)
    step0 = dict(rows[0], label=STEP0_DROPPED if dropped else STEP0_PLAIN)
    medians: Dict[str, Optional[float]] = {}
    for name in RUNS:
        later = [row["seconds"][name] for row in rows[1:]]
        usable = bool(later) and all(value is not None for value in later)
        medians[name] = statistics.median(later) if usable else None
    return {"step0": step0, "later": rows[1:], "later_median_seconds": medians}


def _total(entries: List[Dict[str, Any]], key: str) -> Optional[int]:
    values = [_get(e, key, None) for e in entries]
    return sum(values) if all(_is_number(v) for v in values) else None


def _peak(entries: List[Dict[str, Any]], key: str) -> Optional[int]:
    values = [v for v in (e.get(key) for e in entries) if _is_number(v)]
    return max(values) if values else None


def per_run_for(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per run, each quantity in its own column (totals over steps; never summed across
    runs or quantities); peaks are maxima over steps."""
    columns = {}
    for name in RUNS:
        entries = runs[name]["step"]["steps"]
        # Untrusted record text would reach a Rich table cell; only a number is kept.
        supervisor_peak = runs[name]["record"].get("memory_peak_bytes")
        columns[name] = {
            "expert_bytes_read": _total(entries, "expert_bytes_read"),
            "ssd_read_bytes": _total(entries, "ssd_read_bytes"),
            "h2d_bytes.experts": _total(entries, "h2d_bytes.experts"),
            "h2d_bytes.dense": _total(entries, "h2d_bytes.dense"),
            "h2d_bytes.large": _total(entries, "h2d_bytes.large"),
            "expert_read_seconds": _total(entries, "expert_read_seconds"),
            "nvml_peak_process_bytes_max": _peak(entries, "nvml_peak_process_bytes"),
            "cgroup_memory_peak_bytes_max": _peak(entries, "cgroup_memory_peak_bytes"),
            "supervisor_memory_peak_bytes": (
                supervisor_peak if _is_number(supervisor_peak) else None),
        }
    return columns


def compare(record_paths: List[str], memory_max: Optional[int],
            swap_max: Optional[int], step_dir: Optional[str] = None) -> Dict[str, Any]:
    refusals: List[str] = []
    runs = {name: load_run(name, path, refusals, step_dir)
            for name, path in zip(RUNS, record_paths)}
    distinct_refusals(runs, refusals)
    if all(run["step"] is not None for run in runs.values()):
        refusals_for(runs, memory_max, swap_max, refusals)
    report: Dict[str, Any] = {
        "schema": SCHEMA, "verdict": "REFUSED", "refusals": refusals, "failures": [],
        "timing": None, "per_run": None,
        "inputs": {n: {"record": runs[n]["record_path"], "run_id": runs[n]["run_id"],
                       "step_json": runs[n]["step_path"],
                       "step_sha256": runs[n]["step_sha256"]} for n in RUNS},
        "memory_max": memory_max, "swap_max": swap_max, "claims": CLAIMS,
        "finished_utc": _utc(),
    }
    if refusals:
        return report
    report["failures"] = failures_for({n: runs[n]["step"]["steps"] for n in RUNS})
    report["verdict"] = "FAIL" if report["failures"] else "PASS"
    report["timing"] = timing_for(runs)
    report["per_run"] = per_run_for(runs)
    return report


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_report(path: str, report: Dict[str, Any]) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    os.replace(tmp, path)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render(report: Dict[str, Any]) -> None:
    style = {"PASS": "green", "FAIL": "red", "REFUSED": "yellow"}[report["verdict"]]
    console.print(f"[bold {style}]{report['verdict']}[/]  (exit {report['exit_code']})")
    for reason in report["refusals"]:
        console.print(f"  refused: {reason}", markup=False)
    for item in report["failures"]:
        console.print(f"  step {item['index']} {item['quantity']}: {item['reason']} "
                      f"{item['values']}", markup=False)
    timing = report["timing"]
    if timing is not None:
        table = Table(title="step seconds (A1 B1 B2 A2)")
        for column in ("index", *RUNS, "B1/A1", "B2/A2", "class"):
            table.add_column(column)
        for row in [timing["step0"], *timing["later"]]:
            table.add_row(str(row["index"]), *(_fmt(row["seconds"][n]) for n in RUNS),
                          _fmt(row["ratios"]["B1/A1"]), _fmt(row["ratios"]["B2/A2"]),
                          row["class"], end_section=row is timing["step0"])
        table.add_row("median >=1", *(_fmt(timing["later_median_seconds"][n]) for n in RUNS),
                      "", "", "")
        console.print(table)
        console.print(f"step 0: {timing['step0']['label']} (not pooled with later steps)")
    if report["per_run"] is not None:
        table = Table(title="per run (not summed)")
        table.add_column("quantity")
        for name in RUNS:
            table.add_column(name, justify="right")
        for key in report["per_run"][RUNS[0]]:
            table.add_row(key, *(_fmt(report["per_run"][n][key]) for n in RUNS))
        console.print(table)
    console.print(f"claims: {report['claims']}", markup=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs=4, required=True, metavar=RUNS,
                        help="four distinct supervisor run records (schema "
                             f"{RECORD_SCHEMA}) in run order A1 B1 B2 A2; each step JSON must "
                             "match the record's expect_json.sha256 and run_id")
    parser.add_argument("--out", required=True, help="compare.json path")
    parser.add_argument("--memory-max", type=int, default=None,
                        help="required memory.max read-back (bytes) of every run's leaf "
                             "cgroup and of the series parent (record series.readback)")
    parser.add_argument("--swap-max", type=int, default=None,
                        help="required memory.swap.max read-back (bytes) of every run's leaf "
                             "cgroup and of the series parent (record series.readback)")
    parser.add_argument("--step-dir", default=None, metavar="DIR",
                        help="read each step JSON from DIR by the base name of the record's "
                             "expect_json.path, for runs moved off the host that wrote them; "
                             "the sha256 and run_id checks still apply")
    args = parser.parse_args(argv)
    report = compare(args.runs, args.memory_max, args.swap_max, args.step_dir)
    report["exit_code"] = EXIT[report["verdict"]]
    write_report(args.out, report)
    render(report)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
