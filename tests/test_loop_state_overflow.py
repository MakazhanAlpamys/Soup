"""A huge integer in ``loop.yaml`` is a named refusal, not an ``OverflowError`` traceback.

``LoopState`` checked ``canary_traffic_pct`` and ``deploy_traffic_pct`` with ``float(v)`` before
the bound test. ``loop.yaml`` is JSON, whose integers have no size limit, so a value such as
10**400 raised ``OverflowError: int too large to convert to float`` out of the validator: the
commands that read the file showed a traceback instead of the one-line "loop state invalid:
<field> must be in ..." refusal (the class of #2882). The bound is now tested on the number
itself (an ``int`` compares with a float exactly, whatever its size).

``monthly_budget_usd`` and ``spent_this_month_usd`` had no upper bound at all, and the budget
arithmetic and ``soup loop status`` convert them to ``float``: the same traceback came from there.
They are bounded now (``MAX_USD_AMOUNT``) with the same named refusal; a value that is not finite
is refused at the same place, as the budget check refused it later.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.commands.loop import app
from soup_cli.utils.loop_state import MAX_USD_AMOUNT, LoopState, read_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
HUGE = 10**400


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _state(**kwargs) -> LoopState:
    return LoopState(served_model="m", eval_suite="e", baseline="b", status="running", **kwargs)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


#: field -> the refusal a user would grep for
REFUSALS = {
    "canary_traffic_pct": "canary_traffic_pct must be in [0, 100]",
    "deploy_traffic_pct": "deploy_traffic_pct must be in (0, 100]",
    "monthly_budget_usd": "monthly_budget_usd must be in [0, 1000000000000000] or None",
    "spent_this_month_usd": "spent_this_month_usd must be in [0, 1000000000000000]",
}


@pytest.mark.parametrize("field", sorted(REFUSALS))
@pytest.mark.parametrize("value", [HUGE, -HUGE], ids=["huge", "negative-huge"])
def test_a_huge_integer_is_refused_by_name(field, value):
    with pytest.raises(ValueError, match=re.escape(REFUSALS[field])):
        _state(**{field: value})


@pytest.mark.parametrize("field", ["canary_traffic_pct", "deploy_traffic_pct"])
def test_the_bounds_are_unchanged_for_ordinary_numbers(field):
    assert getattr(_state(**{field: 100}), field) == 100  # the upper bound, an int
    assert getattr(_state(**{field: 100.0}), field) == 100.0
    assert getattr(_state(**{field: 0.5}), field) == 0.5
    for bad in (100.0000001, 101, float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            _state(**{field: bad})


@pytest.mark.parametrize("field", ["monthly_budget_usd", "spent_this_month_usd"])
def test_the_usd_amount_bounds_for_ordinary_numbers(field):
    assert MAX_USD_AMOUNT == 1e15
    for good in (0, 0.0, 0.01, 50, 1e6, 10**15, 1e15):
        assert getattr(_state(**{field: good}), field) == good
    for bad in (-1, -0.01, 10**15 + 1, 2e15, float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError, match=re.escape(REFUSALS[field])) as caught:
            _state(**{field: bad})
        assert field in str(caught.value)


def test_the_monthly_budget_may_still_be_unset():
    assert _state().monthly_budget_usd is None
    assert _state(monthly_budget_usd=None).monthly_budget_usd is None


def test_the_lower_bounds_differ_by_field():
    assert _state(canary_traffic_pct=0).canary_traffic_pct == 0  # [0, 100]
    with pytest.raises(ValueError, match="deploy_traffic_pct"):
        _state(deploy_traffic_pct=0)  # (0, 100]


@pytest.mark.parametrize("field", sorted(REFUSALS))
def test_a_loop_yaml_with_a_huge_integer_is_refused_by_name_on_read(cwd, field):
    write_state(_state(), ".soup/loop.yaml")
    data = json.loads((cwd / ".soup" / "loop.yaml").read_text(encoding="utf-8"))
    data[field] = HUGE
    (cwd / ".soup" / "loop.yaml").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        read_state(".soup/loop.yaml")


@pytest.mark.parametrize("field", sorted(REFUSALS))
def test_the_status_command_shows_one_line_for_it_not_a_traceback(cwd, field):
    write_state(_state(), ".soup/loop.yaml")
    data = json.loads((cwd / ".soup" / "loop.yaml").read_text(encoding="utf-8"))
    data[field] = HUGE
    (cwd / ".soup" / "loop.yaml").write_text(json.dumps(data), encoding="utf-8")

    result = CliRunner().invoke(app, ["status"])

    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert f"loop state invalid: state file has invalid contents: {REFUSALS[field]}" in out
    assert "OverflowError" not in out and "Traceback" not in out


@pytest.mark.parametrize("field", sorted(REFUSALS))
def test_the_watch_command_shows_one_line_for_it_not_a_traceback(cwd, field):
    write_state(_state(), ".soup/loop.yaml")
    data = json.loads((cwd / ".soup" / "loop.yaml").read_text(encoding="utf-8"))
    data[field] = HUGE
    (cwd / ".soup" / "loop.yaml").write_text(json.dumps(data), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["watch", "--foreground", "--max-iterations", "1", "--poll-interval", "1"]
    )

    out = _clean(result.output)
    assert result.exit_code == 2, (out, repr(result.exception))
    assert REFUSALS[field] in out
    assert "OverflowError" not in out and "Traceback" not in out
    assert result.exception is None or isinstance(result.exception, SystemExit)
