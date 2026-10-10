"""Regression coverage for issue #1414: the test-count badge step runs only upstream."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
UPSTREAM_GUARD = "github.repository == 'MakazhanAlpamys/Soup'"


def _badge_step() -> dict:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = [
        step
        for step in workflow["jobs"]["test"]["steps"]
        if step.get("name") == "Update test count badge"
    ]
    assert len(steps) == 1, "expected exactly one 'Update test count badge' step in jobs.test"
    return steps[0]


def test_badge_condition_requires_the_upstream_repository() -> None:
    condition = _badge_step()["if"]
    # The guard has to be one AND-ed term of the whole condition. An `||` anywhere would
    # let the step run whenever the guard is true, i.e. on every upstream event.
    assert "||" not in condition, condition
    terms = [term.strip() for term in condition.split("&&")]
    assert UPSTREAM_GUARD in terms, terms
    # ...and the terms it was added to must still be there. The event is the nightly
    # schedule since #1538: a push to main runs only the quick set, which has no
    # ubuntu 3.11 cell, so the nightly full run is the one that updates the badge.
    for kept in ("github.ref == 'refs/heads/main'", "github.event_name == 'schedule'"):
        assert kept in terms, terms


def test_badge_step_stays_blocking() -> None:
    # continue-on-error would hide an expired GIST_TOKEN upstream (ruling on #1414).
    assert "continue-on-error" not in _badge_step()
