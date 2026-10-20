"""The three workflows that are not required checks skip a draft pull request.

Measured 2026-10-19 over 7 days: `recipe attach`, `Recipe Validation` and `merge freshness`
started 13, 15 and 70 runs on a head whose `CI` run was a skipped draft (88 job-minutes, and a
runner start each). A draft is not merged, none of the three is a required check, and marking a
draft ready fires `ready_for_review`, which they now list so the first run still happens then.

`ci.yml` does the same for `plan` and `lint` (tests/test_ci_full_matrix_on_approval.py,
`TestADraftRunsNothing`). The contexts below are the ones that file builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.test_ci_full_matrix_on_approval import (
    _STATUS_FUNCTION,
    _Expression,
    _on_event,
    _pull_request,
    _push,
    _truthy,
)

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
#: file -> the job that runs on a pull request
PULL_REQUEST_JOBS = {
    "recipe-attach.yml": "verify",
    "recipe-validation.yml": "validate-recipes",
    "merge-freshness.yml": "gate",
}
#: The events GitHub starts a pull_request workflow on unless `types:` says otherwise.
DEFAULT_TYPES = ("opened", "synchronize", "reopened")


def _workflow(file: str) -> dict[str, Any]:
    data = yaml.safe_load((WORKFLOWS / file).read_text(encoding="utf-8"))
    assert isinstance(data, dict), file
    return data


def _triggers(file: str) -> dict[str, Any]:
    data = _workflow(file)
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return data.get("on", data.get(True))


def _starts(file: str, job: str, context: dict[str, Any]) -> bool:
    condition = str(_workflow(file)["jobs"][job].get("if", "true"))
    return _truthy(_Expression(condition, context).value())


@pytest.mark.parametrize(("file", "job"), sorted(PULL_REQUEST_JOBS.items()))
class TestADraftStartsNothing:
    @pytest.mark.parametrize("action", DEFAULT_TYPES)
    def test_a_draft_pull_request_is_skipped(self, file: str, job: str, action: str) -> None:
        assert not _starts(file, job, _pull_request(action, draft=True)), (file, action)

    @pytest.mark.parametrize("action", (*DEFAULT_TYPES, "ready_for_review"))
    def test_a_pull_request_that_is_ready_still_runs(
        self, file: str, job: str, action: str
    ) -> None:
        assert _starts(file, job, _pull_request(action, draft=False)), (file, action)

    def test_marking_a_draft_ready_starts_its_first_run(self, file: str, job: str) -> None:
        types = _triggers(file)["pull_request"]["types"]
        assert "ready_for_review" in types, types
        for default in DEFAULT_TYPES:
            assert default in types, f"{default} is a GitHub default that `types:` replaces"

    def test_the_condition_calls_no_status_function(self, file: str, job: str) -> None:
        """`always()` and friends run a job whatever its condition says about a draft."""
        assert not _STATUS_FUNCTION.search(str(_workflow(file)["jobs"][job].get("if", "")))

    def test_it_is_still_an_ordinary_pull_request_run(self, file: str, job: str) -> None:
        assert "pull_request_target" not in _triggers(file)
        assert "pull_request" in _triggers(file)


@pytest.mark.parametrize(
    ("file", "job", "context"),
    [
        pytest.param("recipe-attach.yml", "verify", _push(), id="attach-push"),
        pytest.param(
            "recipe-attach.yml", "verify", _on_event("workflow_dispatch"), id="attach-dispatch"
        ),
        pytest.param("recipe-validation.yml", "validate-recipes", _push(), id="validation-push"),
    ],
)
def test_a_push_or_a_manual_run_is_not_a_draft(
    file: str, job: str, context: dict[str, Any]
) -> None:
    assert _starts(file, job, context), file


class TestMergeFreshnessKeepsItsTwoShapes:
    """`gate` is the pull-request job and `reeval-open-prs` the push / dispatch one."""

    def test_the_gate_does_not_run_on_a_push_or_a_manual_run(self) -> None:
        for context in (_push(), _on_event("workflow_dispatch")):
            assert not _starts("merge-freshness.yml", "gate", context)

    def test_the_reevaluation_runs_on_a_push_and_a_manual_run_and_not_on_a_pull_request(
        self,
    ) -> None:
        for context in (_push(), _on_event("workflow_dispatch")):
            assert _starts("merge-freshness.yml", "reeval-open-prs", context)
        for draft in (False, True):
            context = _pull_request("synchronize", draft=draft)
            assert not _starts("merge-freshness.yml", "reeval-open-prs", context)
