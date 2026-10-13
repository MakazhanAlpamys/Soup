"""Tests for the F1 pilot script (benchmarks/harness/f1_pilot.py, F1 ticket 08).

The runner and the arm watch are fakes: no test starts ``soup bench train`` or
reads a sensor. Run values are synthetic seconds per step, and no test writes
into benchmarks/results/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def pilot():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_f1_pilot_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "f1_pilot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


#: Five blocks in A, B, B, A order whose pooled sigma is 20 and whose mean of A
#: is 100, so N = ceil(7.85 * (20 / 10)^2) = 32 (the same numbers as the
#: repeats rule tests in test_f1_arm_validity.py).
PILOT = (
    (130, 70, 100, 100),
    (70, 130, 100, 100),
    (100, 100, 70, 130),
    (100, 100, 130, 70),
    (100, 100, 100, 100),
)

#: A block whose B runs are far slower; it would raise N far above 32 if it counted.
SLOW_B = (100, 600, 600, 100)


class FakeRunner:
    """Returns the next synthetic value for each run and logs the order it was asked in."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def __call__(self, arm, round, run):
        self.calls.append((arm, round, run))
        return self.values.pop(0)


class FakeWatch:
    """Stands in for ``ArmWatch``: the runs numbered in ``void_runs`` come out void."""

    def __init__(self, void_runs=()):
        self.void_runs = set(void_runs)

    def __call__(self, *, arm, round, run):
        outcome = "void" if run in self.void_runs else "ok"
        return _Watched({"arm": arm, "round": round, "run": run, "outcome": outcome})


class _Watched:
    def __init__(self, record):
        self.record = None
        self._record = record

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.record = self._record


def _flat(*blocks):
    return [value for block in blocks for value in block]


def test_pilot_runs_five_blocks_in_a_b_b_a_order(pilot):
    runner = FakeRunner(_flat(*PILOT))

    result = pilot.run_pilot(runner, FakeWatch())

    assert [arm for arm, _, _ in runner.calls] == ["A", "B", "B", "A"] * 5
    # A block is two rounds: A, B and then B, A.
    assert [round for _, round, _ in runner.calls[:8]] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert [run for _, _, run in runner.calls] == list(range(1, 21))
    assert result.repeats == pilot.Repeats(20.0, 100.0, 32)


def test_block_with_a_void_run_is_replaced(pilot):
    # Run 6 is the first B of the second block. That block is run in full,
    # kept on record, and one more block runs in its place.
    runner = FakeRunner(_flat(PILOT[0], SLOW_B, *PILOT[1:]))

    result = pilot.run_pilot(runner, FakeWatch(void_runs={6}))

    assert len(runner.calls) == 24
    assert [arm for arm, _, _ in runner.calls] == ["A", "B", "B", "A"] * 6
    assert len(result.blocks) == 6
    assert [run["outcome"] for run in result.blocks[1]] == ["ok", "void", "ok", "ok"]
    assert result.repeats.blocks_needed == 32


def test_pilot_stops_when_too_many_blocks_are_void(pilot):
    runner = FakeRunner(_flat(*PILOT * 2))

    with pytest.raises(RuntimeError, match="3 blocks run, 1 without a void run"):
        pilot.run_pilot(runner, FakeWatch(void_runs={2, 6}), max_blocks=3)


def test_report_is_labelled_and_holds_every_run_record(pilot, tmp_path):
    runner = FakeRunner(_flat(*PILOT))
    result = pilot.run_pilot(runner, FakeWatch())

    path = tmp_path / "pilot.json"
    pilot.save_report(result, path, label="synthetic", settings={"steps": 12})

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["label"] == "synthetic"
    assert report["settings"] == {"steps": 12}
    assert (report["sigma"], report["mean_a"], report["blocks_needed"]) == (20.0, 100.0, 32)
    assert report["blocks"][0][0] == {
        "arm": "A",
        "round": 1,
        "run": 1,
        "value": 130,
        "outcome": "ok",
        "record": {"arm": "A", "round": 1, "run": 1, "outcome": "ok"},
    }
