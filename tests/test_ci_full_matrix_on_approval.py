"""A pull-request push runs a quick CI subset; the required cells run at approval.

Measured 2026-09-28 07:30Z: the Actions account sat at its free-plan cap of 20
concurrent jobs (8 ubuntu, 8 windows, 4 macos) with 301 jobs queued, and every
push to a pull request cost 15 jobs of ``ci.yml``. So ``plan`` picks one of
four matrices by event:

* RELEASE, every cell of the 3x3 support matrix, for pushes to ``release/**``
  -- the release checklist needs all of them green on the tagged commit;
* FULL, the same minus Windows and macOS on 3.11, for the nightly run on ``main``
  and a manual dispatch;
* APPROVAL, the four cells branch protection requires (the oldest and the newest
  Python on ubuntu, the newest on Windows and macOS), for pull requests a
  maintainer has labelled ``ci:full``;
* QUICK, ``test (ubuntu-latest, 3.12)`` alone, for a push to ``main`` and for any
  other pull-request run.

A draft pull request runs nothing at all: ``plan`` and ``lint`` carry the draft
condition, every other job needs ``plan``, and ``ready_for_review`` starts the
first run.

The merge gate survives that because of one shape, and these tests pin it: the
``test`` job's matrix is whatever the ``plan`` job outputs, so a quick run never
CREATES the other three required ``test (...)`` contexts. Branch protection
passes a skipped job and waits on a missing one, so it is the missing cells --
not the skipped smokes -- that keep the merge button locked until the approval
run has reported. APPROVAL must name exactly the required contexts: a cell it
drops that branch protection still requires would lock every merge, and a
required cell it drops silently would stop being tested before merge at all.

Nothing here talks to GitHub. The workflow is read as YAML, the ``plan`` step's
shell runs under a local bash, and the expressions GitHub would evaluate are
evaluated by a small interpreter for exactly the operators they use.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

LABEL = "ci:full"
GATE = "needs.plan.outputs.full == 'true'"
GATED_JOBS = (
    "type-check",
    "mlx-smoke",
    "torchao-contract",
    "pytorch-smoke",
    "transformers-floor",
)
#: The support matrix: what RELEASE runs, and what requires-python must match.
SUPPORT_MATRIX = {
    "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
    "python-version": ["3.10", "3.11", "3.12"],
}
#: What FULL leaves out of it. 3.11 still runs on ubuntu in every full run.
FULL_EXCLUDE = (
    {"os": "windows-latest", "python-version": "3.11"},
    {"os": "macos-latest", "python-version": "3.11"},
)
#: What APPROVAL leaves out of its two Pythons: the oldest runs on ubuntu only.
APPROVAL_EXCLUDE = (
    {"os": "windows-latest", "python-version": "3.10"},
    {"os": "macos-latest", "python-version": "3.10"},
)
QUICK_MATRIX = {"os": ["ubuntu-latest"], "python-version": ["3.12"]}
#: The four ``test`` contexts a merge waits on: what APPROVAL runs, and what branch
#: protection on ``main`` must list (8 required checks: these plus lint, mlx-smoke,
#: pytorch-smoke, transformers-floor). Read on 2026-10-07 with ``gh api
#: repos/MakazhanAlpamys/Soup/branches/main/protection``, it still listed the seven
#: cells of FULL; the three in NIGHTLY_ONLY_CONTEXTS come off it right after this
#: change merges, never before. A cell named any other way is a context nobody waits on.
REQUIRED_TEST_CONTEXTS = frozenset(
    {
        "test (ubuntu-latest, 3.10)",
        "test (ubuntu-latest, 3.12)",
        "test (windows-latest, 3.12)",
        "test (macos-latest, 3.12)",
    }
)
#: The three cells only the nightly run, a manual dispatch and ``release/**`` run. A
#: failure that exists on one of them alone shows on ``main`` the next night.
NIGHTLY_ONLY_CONTEXTS = frozenset(
    {
        "test (ubuntu-latest, 3.11)",
        "test (windows-latest, 3.10)",
        "test (macos-latest, 3.10)",
    }
)
#: The two cells only a push to ``release/**`` runs. Not required, so not a merge gate.
RELEASE_ONLY_CONTEXTS = frozenset({"test (windows-latest, 3.11)", "test (macos-latest, 3.11)"})
#: The condition on the two jobs that wait for nothing: everything but a draft pull request.
DRAFT_GATE = "github.event_name != 'pull_request' || github.event.pull_request.draft == false"
UNGATED_JOBS = ("plan", "lint")


# --- Reading the workflow ----------------------------------------------------


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(CI.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml did not parse to a mapping"
    return data


def _triggers() -> Any:
    data = _workflow()
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return data.get("on", data.get(True))


def _jobs() -> dict[str, Any]:
    return _workflow()["jobs"]


def _plan() -> dict[str, Any]:
    plan = _jobs().get("plan")
    assert isinstance(plan, dict), "ci.yml has no `plan` job"
    return plan


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _decide_step() -> dict[str, Any]:
    steps = [step for step in _plan().get("steps", []) if step.get("id") == "decide"]
    assert len(steps) == 1, "the plan job must have exactly one step with `id: decide`"
    return steps[0]


MATRIX_NAMES = ("RELEASE_MATRIX", "FULL_MATRIX", "APPROVAL_MATRIX", "QUICK_MATRIX")


def _literal(name: str) -> str:
    literal = (_plan().get("env") or {}).get(name)
    assert isinstance(literal, str), (
        f"the plan job must declare {name} once, as a JSON literal in its env"
    )
    return literal


def _matrix(name: str) -> dict[str, Any]:
    return json.loads(_literal(name))


def _cell_dicts(matrix: dict[str, Any]) -> list[dict[str, str]]:
    """The cells GitHub creates for *matrix*, each as a ``matrix`` context.

    The product of the two axes minus every cell an ``exclude`` entry matches --
    GitHub excludes on a PARTIAL match, so ``{"python-version": "3.11"}`` alone
    would drop all three 3.11 cells.
    """
    assert set(matrix) <= {"os", "python-version", "exclude"}, sorted(matrix)
    cells = []
    for runner in matrix["os"]:
        for python in matrix["python-version"]:
            cell = {"os": runner, "python-version": python}
            if any(
                all(cell.get(key) == value for key, value in rule.items())
                for rule in matrix.get("exclude", [])
            ):
                continue
            cells.append(cell)
    return cells


def _cells(matrix: dict[str, Any]) -> set[str]:
    """The ``test (...)`` contexts of *matrix*, titled with the values in key order."""
    return {f"test ({cell['os']}, {cell['python-version']})" for cell in _cell_dicts(matrix)}


def _rules(entries: Any) -> list[tuple[tuple[str, Any], ...]]:
    """Exclude entries compared as a multiset, whatever order they are listed in."""
    return sorted(tuple(sorted(entry.items())) for entry in entries)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


# --- A small evaluator for the GitHub expressions these tests need -----------
#
# It covers exactly what ci.yml's concurrency key and the plan decision use:
# string literals, true/false/null, property paths (with one `*` object
# filter), parentheses, ! == != && || and the functions format(), contains()
# and startsWith(). The semantics are GitHub's: && and || return an OPERAND,
# not a boolean; ! binds tighter than == and !=, which bind tighter than &&,
# then ||; strings compare case-insensitively; a missing property is null.
# Anything else fails loudly instead of being guessed at.

_TOKEN = re.compile(
    r"(?P<string>'(?:[^']|'')*')"
    r"|(?P<op>==|!=|&&|\|\||!|\(|\)|,)"
    r"|(?P<name>[A-Za-z_][\w-]*(?:\.(?:\*|[A-Za-z_][\w-]*))*)"
)
_TEMPLATE = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_KEYWORDS = {"true": True, "false": False, "null": None}


def _tokenize(source: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(source):
        if source[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(source, position)
        assert match, f"expression syntax this evaluator does not know: {source[position:]!r}"
        kind = match.lastgroup
        assert kind is not None
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens


def _truthy(value: Any) -> bool:
    return value not in (None, False, 0, "")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    return left == right


def _lookup(context: dict[str, Any], path: str) -> Any:
    value: Any = context
    spread = False
    for part in path.split("."):
        if part == "*":
            assert not spread, "nested object filters are not modelled"
            if isinstance(value, dict):
                value = list(value.values())
            elif not isinstance(value, list):
                value = []
            spread = True
        elif spread:
            value = [item.get(part) for item in value if isinstance(item, dict)]
        else:
            value = value.get(part) if isinstance(value, dict) else None
    return value


def _call(name: str, args: list[Any]) -> Any:
    function = name.lower()  # GitHub function names are case-insensitive
    if function == "format" and args:
        template, *values = args
        return re.sub(
            r"\{(\d+)\}", lambda match: _text(values[int(match.group(1))]), _text(template)
        )
    if function == "contains" and len(args) == 2:
        haystack, needle = args
        if isinstance(haystack, list):
            return any(_equal(item, needle) for item in haystack)
        return _text(needle).casefold() in _text(haystack).casefold()
    if function == "startswith" and len(args) == 2:
        text, prefix = args
        return _text(text).casefold().startswith(_text(prefix).casefold())
    raise AssertionError(f"{name}() is not modelled by this test's evaluator; add it to _call()")


class _Expression:
    """Recursive descent over GitHub's precedence: || < && < == != < ! < primary."""

    def __init__(self, source: str, context: dict[str, Any]) -> None:
        self._tokens = _tokenize(source)
        self._position = 0
        self._context = context

    def value(self) -> Any:
        result = self._either()
        assert self._position == len(self._tokens), f"unparsed: {self._tokens[self._position:]}"
        return result

    def _take(self, operator: str) -> bool:
        if self._position < len(self._tokens) and self._tokens[self._position] == (
            "op",
            operator,
        ):
            self._position += 1
            return True
        return False

    def _either(self) -> Any:
        result = self._both()
        while self._take("||"):
            right = self._both()
            result = result if _truthy(result) else right
        return result

    def _both(self) -> Any:
        result = self._compare()
        while self._take("&&"):
            right = self._compare()
            result = right if _truthy(result) else result
        return result

    def _compare(self) -> Any:
        result = self._unary()
        while True:
            if self._take("=="):
                result = _equal(result, self._unary())
            elif self._take("!="):
                result = not _equal(result, self._unary())
            else:
                return result

    def _unary(self) -> Any:
        if self._take("!"):
            return not _truthy(self._unary())
        return self._primary()

    def _primary(self) -> Any:
        assert self._position < len(self._tokens), "the expression ends early"
        kind, text = self._tokens[self._position]
        self._position += 1
        if kind == "string":
            return text[1:-1].replace("''", "'")
        if (kind, text) == ("op", "("):
            result = self._either()
            assert self._take(")"), "unbalanced parenthesis"
            return result
        assert kind == "name", f"unexpected {text!r}"
        if self._take("("):
            args: list[Any] = []
            if not self._take(")"):
                args.append(self._either())
                while self._take(","):
                    args.append(self._either())
                assert self._take(")"), f"unterminated call to {text}()"
            return _call(text, args)
        if text in _KEYWORDS:
            return _KEYWORDS[text]
        return _lookup(self._context, text)


def _render(template: str, context: dict[str, Any]) -> str:
    """Substitute every ``${{ ... }}`` in *template* the way the runner would."""
    return _TEMPLATE.sub(
        lambda match: _text(_Expression(match.group(1), context).value()), template
    )


def _push(
    ref: str = "refs/heads/main", *, run_id: str = "900", sha: str = "a" * 40
) -> dict[str, Any]:
    return {
        "github": {
            "workflow": "CI",
            "event_name": "push",
            "ref": ref,
            "sha": sha,
            "run_id": run_id,
            "event": {"ref": ref},
        }
    }


def _on_event(event_name: str, *, run_id: str = "800") -> dict[str, Any]:
    """A ``schedule`` or ``workflow_dispatch`` context: both run on main's tip."""
    return {
        "github": {
            "workflow": "CI",
            "event_name": event_name,
            "ref": "refs/heads/main",
            "sha": "c" * 40,
            "run_id": run_id,
            "event": {},
        }
    }


def _pull_request(
    action: str,
    *,
    labels: tuple[str, ...] = (),
    label: str | None = None,
    draft: bool = False,
    run_id: str = "100",
) -> dict[str, Any]:
    """A pull_request context shaped like GitHub's payload.

    On ``labeled``, ``event.label`` is the label just added and
    ``event.pull_request.labels`` already includes it. ``event.pull_request.draft``
    is a boolean on every pull_request event.
    """
    names = list(labels)
    event: dict[str, Any] = {"action": action, "number": 7}
    if label is not None:
        event["label"] = {"name": label}
        if label not in names:
            names.append(label)
    event["pull_request"] = {"labels": [{"name": name} for name in names], "draft": draft}
    return {
        "github": {
            "workflow": "CI",
            "event_name": "pull_request",
            "ref": "refs/pull/7/merge",
            "sha": "b" * 40,
            "run_id": run_id,
            "event": event,
        }
    }


def _group(context: dict[str, Any]) -> str:
    return _render(_workflow()["concurrency"]["group"], context)


def _full_for(context: dict[str, Any]) -> str:
    return _render(_decide_step()["env"]["FULL"], context)


def _release_for(context: dict[str, Any]) -> str:
    return _render(_decide_step()["env"]["RELEASE"], context)


def _approval_for(context: dict[str, Any]) -> str:
    expression = _decide_step()["env"].get("APPROVAL")
    assert isinstance(expression, str), "the decide step must compute APPROVAL in its env"
    return _render(expression, context)


def _runs(job: str, context: dict[str, Any]) -> bool:
    """Whether *job* starts for *context*: its job-level ``if:``, true when it has none."""
    return _truthy(_Expression(str(_jobs()[job].get("if", "true")), context).value())


#: A job-level ``if:`` that calls one of these runs even when a job it needs was skipped.
_STATUS_FUNCTION = re.compile(r"\b(?:always|cancelled|failure|success)\s*\(")


#: Which matrix each event must get: the one table both the expression tests and
#: the end-to-end run of the step's shell read.
EVENTS = [
    pytest.param(_push("refs/heads/main"), "QUICK", id="push-main"),
    pytest.param(_on_event("schedule"), "FULL", id="nightly"),
    pytest.param(_on_event("workflow_dispatch"), "FULL", id="manual-dispatch"),
    pytest.param(_push("refs/heads/release/v0.76.0"), "RELEASE", id="push-release"),
    pytest.param(_pull_request("opened"), "QUICK", id="pr-opened"),
    pytest.param(_pull_request("synchronize", labels=("bug",)), "QUICK", id="pr-push-other-label"),
    pytest.param(
        _pull_request("synchronize", labels=("bug", LABEL)), "APPROVAL", id="pr-push-ci-full"
    ),
    pytest.param(_pull_request("labeled", label=LABEL), "APPROVAL", id="adds-ci-full"),
    pytest.param(_pull_request("labeled", label="bug"), "QUICK", id="adds-other"),
    pytest.param(
        _pull_request("labeled", label="bug", labels=(LABEL,)),
        "APPROVAL",
        id="adds-other-while-ci-full",
    ),
]

#: A draft pull request, whatever starts the run and whatever it is labelled.
DRAFTS = [
    pytest.param(_pull_request("opened", draft=True), id="draft-opened"),
    pytest.param(_pull_request("synchronize", draft=True), id="draft-push"),
    pytest.param(_pull_request("reopened", draft=True), id="draft-reopened"),
    pytest.param(_pull_request("labeled", label=LABEL, draft=True), id="draft-adds-ci-full"),
    pytest.param(
        _pull_request("synchronize", labels=(LABEL,), draft=True), id="draft-push-ci-full"
    ),
]


# --- Tests -------------------------------------------------------------------


class TestTheEvaluatorHasTeeth:
    """CONTROL. The concurrency and plan tests below are only as good as this."""

    def test_and_or_return_operands_with_github_precedence(self):
        context = {"github": {"ref": "r"}}
        assert _Expression("'a' && 'b' || 'c'", context).value() == "b"
        assert _Expression("null && 'b' || 'c'", context).value() == "c"
        assert _Expression("'' || github.ref", context).value() == "r"
        assert _Expression("!('x' == 'y') && 'yes'", context).value() == "yes"

    def test_strings_compare_case_insensitively(self):
        assert _Expression("'CI:Full' == 'ci:full'", {}).value() is True
        assert _Expression("'bug' != 'ci:full'", {}).value() is True

    def test_missing_properties_are_null_and_the_filter_spreads(self):
        context = {"github": {"event": {"items": [{"name": "a"}, {"name": "B"}]}}}
        assert _Expression("github.event.nothing.here", context).value() is None
        assert _Expression("contains(github.event.items.*.name, 'b')", context).value()
        assert not _Expression("contains(github.event.none.*.name, 'b')", context).value()

    def test_templates_and_format(self):
        context = {"github": {"workflow": "CI", "run_id": "42"}}
        rendered = _render("${{ github.workflow }}-${{ format('x-{0}', github.run_id) }}", context)
        assert rendered == "CI-x-42"

    def test_starts_with_is_a_case_insensitive_prefix_test(self):
        context = {"github": {"ref": "refs/heads/Release/v1"}}
        assert _Expression("startsWith(github.ref, 'refs/heads/release/')", context).value()
        assert not _Expression("startsWith(github.ref, 'refs/heads/main')", context).value()
        assert not _Expression("startsWith(github.nothing, 'refs/')", context).value()

    def test_what_it_does_not_model_fails_instead_of_guessing(self):
        with pytest.raises(AssertionError, match="not modelled"):
            _Expression("toJSON(github)", {}).value()
        with pytest.raises(AssertionError, match="does not know"):
            _Expression("github.run_number >= 2", {}).value()


class TestTriggers:
    def test_pull_requests_run_on_these_five_actions(self):
        pull_request = _triggers()["pull_request"]
        assert pull_request["branches"] == ["main"]
        assert sorted(pull_request.get("types", [])) == sorted(
            ["opened", "synchronize", "reopened", "ready_for_review", "labeled"]
        ), "without `labeled`, adding ci:full at approval starts nothing"

    def test_there_is_no_pull_request_target(self):
        """A labelled run stays an ordinary pull_request run, with the same token and secrets
        a push to the pull request gets -- never the privileged _target variant."""
        assert "pull_request_target" not in _triggers()

    def test_pushes_to_main_and_release_branches_still_trigger(self):
        assert _triggers()["push"]["branches"] == ["main", "release/**"]

    def test_the_nightly_full_run_and_the_manual_dispatch_exist(self):
        """A push to main runs only the quick set, so the full matrix on main has to come
        from somewhere: without these two the seven-cell run would never happen on main."""
        triggers = _triggers()
        schedule = triggers.get("schedule")
        assert isinstance(schedule, list) and len(schedule) == 1, schedule
        assert str(schedule[0].get("cron", "")).count(" ") == 4, schedule
        assert "workflow_dispatch" in triggers


class TestPlanDecision:
    def test_plan_is_short_waits_for_nothing_and_exports_its_decision(self):
        plan = _plan()
        assert plan["runs-on"] == "ubuntu-latest"
        assert plan["timeout-minutes"] == 5
        assert plan.get("if") == DRAFT_GATE, "plan decides for everything but a draft"
        assert "needs" not in plan
        assert plan["outputs"] == {
            "full": "${{ steps.decide.outputs.full }}",
            "matrix": "${{ steps.decide.outputs.matrix }}",
        }

    @pytest.mark.parametrize(("context", "kind"), EVENTS)
    def test_the_decision_follows_the_event_and_the_current_labels(self, context, kind):
        """RELEASE only for a push to release/**; FULL for that, the nightly run and a
        manual dispatch; APPROVAL for a pull request whose CURRENT labels carry ci:full,
        whichever event started the run."""
        assert _release_for(context) == ("true" if kind == "RELEASE" else "false")
        assert _full_for(context) == ("true" if kind in ("RELEASE", "FULL") else "false")
        assert _approval_for(context) == ("true" if kind == "APPROVAL" else "false")


class TestTheBadgeStep:
    def test_the_test_count_badge_is_updated_by_the_nightly_run(self):
        """The badge step needs the ubuntu / 3.11 cell, which a push to main no longer runs
        (it runs ubuntu / 3.12 only), so it moved to the nightly full run. A condition still
        gated on a push would never fire again and the badge would silently go stale."""
        steps = _jobs()["test"]["steps"]
        badge = [step for step in steps if step.get("name") == "Update test count badge"]
        assert len(badge) == 1, badge
        condition = str(badge[0].get("if", ""))
        assert "github.event_name == 'schedule'" in condition, condition
        assert "github.event_name == 'push'" not in condition, condition
        assert "github.ref == 'refs/heads/main'" in condition, condition


class TestGatedJobs:
    @pytest.mark.parametrize("name", GATED_JOBS)
    def test_the_smoke_and_contract_jobs_run_only_on_a_full_run(self, name):
        job = _jobs()[name]
        assert "plan" in _needs(job), f"{name} does not need plan"
        assert job.get("if") == GATE, f"{name} is not gated on plan's decision"

    def test_lint_waits_for_nothing_and_skips_only_a_draft(self):
        lint = _jobs()["lint"]
        assert lint.get("if") == DRAFT_GATE
        assert "needs" not in lint

    def test_every_job_is_classified(self):
        """A new job must be declared quick or full here, or it lands on every PR push."""
        assert sorted(_jobs()) == sorted(["plan", "lint", "test", *GATED_JOBS])


class TestMatrix:
    def test_the_test_job_takes_its_matrix_from_plan(self):
        """A static matrix with a per-cell `if:` would SKIP the missing cells, and a skipped
        job passes branch protection; a matrix from plan never creates them."""
        test = _jobs()["test"]
        assert "plan" in _needs(test)
        assert "if" not in test
        assert test["strategy"]["fail-fast"] is False
        assert test["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.matrix) }}"
        assert test["runs-on"] == "${{ matrix.os }}"

    def test_release_is_the_support_matrix_with_nothing_excluded(self):
        release = _matrix("RELEASE_MATRIX")
        assert release == SUPPORT_MATRIX, "RELEASE must be every cell: no exclude, no include"
        assert list(release) == ["os", "python-version"], "the key order names the checks"
        assert all(isinstance(python, str) for python in release["python-version"]), (
            "a JSON number 3.10 is 3.1, which renames the check context"
        )

    def test_full_is_the_support_matrix_minus_exactly_two_cells(self):
        full = _matrix("FULL_MATRIX")
        assert list(full) == ["os", "python-version", "exclude"]
        assert full["os"] == SUPPORT_MATRIX["os"]
        assert full["python-version"] == SUPPORT_MATRIX["python-version"]
        assert _rules(full["exclude"]) == _rules(FULL_EXCLUDE)
        # Nothing else excluded, whatever the entries look like (a partial rule
        # such as {"python-version": "3.11"} would drop the ubuntu cell too).
        assert _cells(_matrix("RELEASE_MATRIX")) - _cells(full) == RELEASE_ONLY_CONTEXTS

    def test_approval_is_the_oldest_and_newest_python_with_the_oldest_on_ubuntu_only(self):
        approval = _matrix("APPROVAL_MATRIX")
        assert list(approval) == ["os", "python-version", "exclude"], (
            "the key order names the checks"
        )
        assert approval["os"] == SUPPORT_MATRIX["os"]
        pythons = SUPPORT_MATRIX["python-version"]
        assert approval["python-version"] == [
            min(pythons, key=_version),
            max(pythons, key=_version),
        ]
        assert _rules(approval["exclude"]) == _rules(APPROVAL_EXCLUDE)

    def test_approval_names_exactly_the_required_test_contexts(self):
        assert _cells(_matrix("APPROVAL_MATRIX")) == REQUIRED_TEST_CONTEXTS

    def test_full_runs_every_required_cell_and_the_three_nightly_ones(self):
        assert _cells(_matrix("FULL_MATRIX")) == REQUIRED_TEST_CONTEXTS | NIGHTLY_ONLY_CONTEXTS

    def test_release_runs_every_cell_a_smaller_matrix_leaves_out(self):
        assert _cells(_matrix("RELEASE_MATRIX")) == (
            REQUIRED_TEST_CONTEXTS | NIGHTLY_ONLY_CONTEXTS | RELEASE_ONLY_CONTEXTS
        )

    def test_the_quick_matrix_is_the_newest_python_on_ubuntu(self):
        quick = _matrix("QUICK_MATRIX")
        assert quick == QUICK_MATRIX
        assert list(quick) == ["os", "python-version"]
        names = _cells(quick)
        assert len(names) == 1 and names <= REQUIRED_TEST_CONTEXTS
        support_pythons = SUPPORT_MATRIX["python-version"]
        assert quick["python-version"] == [max(support_pythons, key=_version)]

    @pytest.mark.parametrize("name", MATRIX_NAMES)
    def test_the_matrix_literals_are_single_lines(self, name):
        """$GITHUB_OUTPUT takes one `name=value` per line; a newline would end the value."""
        assert "\n" not in _literal(name)


class TestConcurrency:
    def test_a_label_other_than_ci_full_cannot_share_the_pull_request_group(self):
        pull_request = _group(_pull_request("synchronize", labels=(LABEL,)))
        for labels in ((), (LABEL,)):
            labelled = _group(_pull_request("labeled", label="bug", labels=labels))
            assert labelled != pull_request, (
                "a run started by an unrelated label shares the pull request's group, "
                "so it cancels a full matrix already running there"
            )

    def test_each_such_run_is_alone_in_its_group(self):
        first = _group(_pull_request("labeled", label="bug", run_id="1"))
        second = _group(_pull_request("labeled", label="bug", run_id="2"))
        assert first != second

    def test_adding_ci_full_supersedes_the_quick_run(self):
        quick = _group(_pull_request("synchronize"))
        assert _group(_pull_request("labeled", label=LABEL, run_id="2")) == quick

    @pytest.mark.parametrize("action", ["opened", "synchronize", "reopened", "ready_for_review"])
    def test_every_other_pull_request_event_still_supersedes(self, action):
        earlier = _group(_pull_request("synchronize", run_id="1"))
        assert _group(_pull_request(action, run_id="2")) == earlier

    def test_a_push_is_never_in_a_pull_request_or_label_group(self):
        for ref in ("refs/heads/main", "refs/heads/release/v0.76.0"):
            push = _group(_push(ref, run_id="1"))
            assert push != _group(_pull_request("synchronize", run_id="1"))
            assert push != _group(_pull_request("labeled", label="bug", run_id="1"))

    def test_two_pushes_to_main_never_share_a_group(self):
        """Every main commit gets its own run. A shared group keeps one pending run and cancels
        the rest, which is how 22 merges in two hours left 'cancelled' on all but two."""
        first = _group(_push("refs/heads/main", run_id="1", sha="1" * 40))
        second = _group(_push("refs/heads/main", run_id="2", sha="2" * 40))
        assert first != second

    def test_the_nightly_and_manual_runs_are_not_in_the_group_of_a_push_to_the_same_commit(self):
        push = _group(_push("refs/heads/main", run_id="1", sha="c" * 40))
        assert _group(_on_event("schedule")) != push
        assert _group(_on_event("workflow_dispatch")) != push
        assert _group(_on_event("schedule")) != _group(_on_event("workflow_dispatch"))

    def test_cancellation_stays_pull_request_only(self):
        concurrency = _workflow()["concurrency"]
        assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"

    @pytest.mark.parametrize("label", [LABEL, LABEL.upper(), "bug"])
    def test_the_group_and_the_plan_agree_on_what_ci_full_is(self, label):
        """Renaming the label in one expression only would make the approval label either
        start a quick run or cancel nothing -- both silently."""
        event = _pull_request("labeled", label=label)
        joins = _group(event) == _group(_pull_request("synchronize"))
        assert joins == (_approval_for(event) == "true")


def _posix_bash() -> str | None:
    """A bash that runs a POSIX script; on Windows, Git for Windows' copy.

    On Windows ``shutil.which("bash")`` can answer ``System32\\bash.exe``, the WSL
    launcher, which does not run the script at all. Git for Windows ships a real
    bash beside git, and the Windows runners and dev boxes here all have git.
    """
    if sys.platform != "win32":
        return shutil.which("bash")
    git = shutil.which("git")
    if git is None:
        return None
    for parent in Path(git).resolve().parents:
        candidate = parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


BASH = _posix_bash()


@pytest.mark.skipif(BASH is None, reason="no POSIX bash found (on Windows: Git for Windows)")
class TestTheDecideStepScript:
    """Run the plan step's own shell, as the runner would, against a fake $GITHUB_OUTPUT."""

    def _outputs(
        self, tmp_path: Path, *, release: str, full: str, approval: str
    ) -> tuple[str, Any]:
        """Run the step with the given decision values; return (full, parsed matrix)."""
        assert BASH is not None
        output = tmp_path / "github_output"
        output.write_text("", encoding="utf-8")
        script = tmp_path / "decide.sh"
        script.write_bytes(_decide_step()["run"].encode("utf-8"))
        env = {
            **os.environ,
            **{name: _literal(name) for name in MATRIX_NAMES},
            "RELEASE": release,
            "FULL": full,
            "APPROVAL": approval,
            "GITHUB_OUTPUT": output.as_posix(),
        }
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        lines = output.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, lines
        full_key, _, full_value = lines[0].partition("=")
        matrix_key, _, matrix_value = lines[1].partition("=")
        assert (full_key, matrix_key) == ("full", "matrix"), lines
        return full_value, json.loads(matrix_value)

    @pytest.mark.parametrize(("context", "kind"), EVENTS)
    def test_each_event_gets_its_matrix(self, tmp_path, context, kind):
        """End to end: the event, through GitHub's expressions, through the shell."""
        full, matrix = self._outputs(
            tmp_path,
            release=_release_for(context),
            full=_full_for(context),
            approval=_approval_for(context),
        )
        assert matrix == _matrix(f"{kind}_MATRIX")
        assert full == ("false" if kind == "QUICK" else "true")

    @pytest.mark.parametrize(
        ("release", "full", "approval", "kind"),
        [
            ("false", "false", "false", "QUICK"),
            ("false", "", "", "QUICK"),
            ("false", "yes", "yes", "QUICK"),
            ("", "true", "false", "FULL"),
            ("yes", "true", "false", "FULL"),
            ("false", "false", "true", "APPROVAL"),
            ("yes", "", "true", "APPROVAL"),
            ("false", "true", "true", "FULL"),
        ],
    )
    def test_only_the_exact_string_true_selects(self, tmp_path, release, full, approval, kind):
        """Quick is the fail-closed answer on a pull request: fewer cells, more missing
        required contexts, a locked merge button. Of two that are set, the larger wins."""
        flag, matrix = self._outputs(tmp_path, release=release, full=full, approval=approval)
        assert matrix == _matrix(f"{kind}_MATRIX")
        assert flag == ("false" if kind == "QUICK" else "true")


class TestADraftRunsNothing:
    """A draft costs no runner: the two jobs that wait for nothing are skipped by their
    own condition, and every other job needs ``plan``."""

    @pytest.mark.parametrize("job", UNGATED_JOBS)
    @pytest.mark.parametrize("context", DRAFTS)
    def test_a_draft_starts_neither_plan_nor_lint(self, job, context):
        assert not _runs(job, context)

    @pytest.mark.parametrize("job", UNGATED_JOBS)
    @pytest.mark.parametrize(("context", "kind"), EVENTS)
    def test_everything_that_is_not_a_draft_still_starts_them(self, job, context, kind):
        assert _runs(job, context), kind

    def test_marking_it_ready_starts_the_first_run(self):
        assert "ready_for_review" in _triggers()["pull_request"].get("types", [])
        ready = _pull_request("ready_for_review")
        assert all(_runs(job, ready) for job in UNGATED_JOBS)

    def test_every_other_job_is_skipped_with_plan(self):
        """A job whose `if:` calls a status function runs even when the job it needs was
        skipped, which is how one of these would start on a draft."""
        for name, job in _jobs().items():
            if name in UNGATED_JOBS:
                continue
            assert "plan" in _needs(job), f"{name} does not need plan"
            condition = str(job.get("if", ""))
            assert not _STATUS_FUNCTION.search(condition), f"{name}: {condition}"


class TestCoverageUpload:
    def test_one_cell_of_every_gated_run_uploads_and_a_quick_run_uploads_nothing(self):
        """The upload is pinned to one cell. Pinned to a cell the approval run does not
        have, it would stop every pull request's Codecov report without a red mark."""
        steps = _jobs()["test"]["steps"]
        upload = [step for step in steps if step.get("name") == "Upload coverage to Codecov"]
        assert len(upload) == 1, upload
        condition = str(upload[0].get("if", ""))

        def uploading(name: str) -> list[dict[str, str]]:
            return [
                cell
                for cell in _cell_dicts(_matrix(name))
                if _truthy(_Expression(condition, {"matrix": cell}).value())
            ]

        for name in ("APPROVAL_MATRIX", "FULL_MATRIX", "RELEASE_MATRIX"):
            assert len(uploading(name)) == 1, (name, uploading(name))
        assert uploading("QUICK_MATRIX") == []


def _contributing() -> str:
    """CONTRIBUTING.md with its line wrapping collapsed."""
    return " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())


def test_contributing_tells_contributors_about_the_label():
    assert f"`{LABEL}`" in _contributing()


def test_contributing_says_a_draft_runs_no_ci():
    assert "A draft pull request runs no CI" in _contributing()
