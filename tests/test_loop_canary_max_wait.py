"""How long a canary waits for ratings: ``--canary-max-wait`` and ``--canary-on-timeout``.

The quality verdict needs 30 rated answers on each side (tests/test_loop_quality_canary.py).
A loop whose users rate rarely would keep a canary in limbo for ever: every cycle says
UNKNOWN, `soup loop watch` starts no new iteration, and nothing improves. So the wait ends.

``soup loop init --canary-max-wait`` (default 24h; ``<int>m``, ``<int>h``, ``<int>d`` or plain
seconds; at most 30d) says how long after the rollout started the ratings may stay short.
Past it the quality verdict is INSUFFICIENT (not UNKNOWN), and ``--canary-on-timeout`` decides:

* ``errors-only`` (default): the request-error verdict alone decides, and the iteration notes
  say the promotion was unverified;
* ``rollback``: the canary is withdrawn (production untouched), even without
  ``--autoroll-on-regress``;
* ``wait``: no time limit, as before.

The evidence behind the quality verdict (counts, shares, z, p) is stored in the iteration
record's ``canary_quality`` (the ``canary.quality`` object of the iteration format).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_daemon as daemon
from soup_cli.utils.canary_router import CanaryPolicy, QualityStats, record_bucket_outcome
from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_iteration import IterationRecord, read_iteration, write_iteration
from soup_cli.utils.loop_state import (
    CANARY_ON_TIMEOUT,
    DEFAULT_CANARY_MAX_WAIT_SEC,
    MAX_CANARY_MAX_WAIT_SEC,
    LoopState,
    format_wait_duration,
    init_state,
    parse_wait_duration,
    read_state,
    write_state,
)

T0 = 1_700_000_000.0
DAY = 86400.0
STABLE = "v1"
CANARY = "v2"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def at(seconds: float) -> datetime:
    """The clock ``seconds`` after the canary started."""
    return datetime.fromtimestamp(T0 + seconds, timezone.utc)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_FEEDBACK_LOG"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def stats(canary_rated, canary_negative, stable_rated, stable_negative):
    return QualityStats(
        stable_rated=stable_rated, stable_negative=stable_negative,
        canary_rated=canary_rated, canary_negative=canary_negative,
    )


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./base")
    kwargs.setdefault("base_model", "org/base")
    kwargs.setdefault("eval_suite", "tasks.jsonl")
    kwargs.setdefault("verify", "ship")
    kwargs.setdefault("status", "running")
    return LoopState(baseline="b", **kwargs)


def _canary_state(**kwargs) -> LoopState:
    kwargs.setdefault("canary_started_ts", T0)
    kwargs.setdefault("production_name", STABLE)
    kwargs.setdefault("production_adapter", "adapters/v1")
    return _state(
        canary_active=CANARY, canary_traffic_pct=5.0, canary_rollout_id="r1",
        canary_adapter="adapters/v2", **kwargs,
    )


def _record(state, ok: int, bad: int = 0, stable=(40, 0)):
    """Request outcomes of the rollout: what the error verdict reads."""
    policy = CanaryPolicy(stable=state.stable_name(), canary=state.canary_active, traffic_pct=5.0)
    path = os.path.join(".soup", "canary-stats.json")
    for bucket, (good, failed) in {"stable": stable, "canary": (ok, bad)}.items():
        for _ in range(good):
            record_bucket_outcome(policy, bucket, True, rollout_id="r1", path=path)
        for _ in range(failed):
            record_bucket_outcome(policy, bucket, False, rollout_id="r1", path=path)


def _config(quality, **kwargs):
    kwargs.setdefault("harvest_fn", lambda s: {"pairs_harvested": 0})
    kwargs.setdefault("train_fn", lambda s, c: {"run_id": None, "skipped": True})
    kwargs.setdefault("gate_fn", lambda s, c: {"gate_verdict": "SKIPPED"})
    kwargs.setdefault("deploy_fn", lambda s, c: {"deployed": False, "canary_verdict": None})
    kwargs.setdefault("cost_fn", lambda s: 0.0)
    kwargs.setdefault("promote_fn", lambda s: {"promoted": True})
    return WatchConfig(quality_fn=quality, **kwargs)


# ---------------------------------------------------------------------------
# The duration
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("1m", 60.0),
            ("90m", 5400.0),
            ("1h", 3600.0),
            ("24h", 86400.0),
            ("36h", 129600.0),
            ("1d", 86400.0),
            ("7d", 604800.0),
            ("30d", 2592000.0),
            ("720h", 2592000.0),
            ("43200m", 2592000.0),
            ("1", 1.0),
            ("3600", 3600.0),
            ("2592000", 2592000.0),
            (" 24h ", 86400.0),
            ("24H", 86400.0),
            ("7D", 604800.0),
            ("007h", 25200.0),
        ],
    )
    def test_minutes_hours_days_or_plain_seconds(self, text, seconds):
        assert parse_wait_duration(text) == seconds

    def test_the_result_is_a_float(self):
        assert isinstance(parse_wait_duration("24h"), float)
        assert isinstance(parse_wait_duration("3600"), float)

    def test_the_cap_is_thirty_days(self):
        assert MAX_CANARY_MAX_WAIT_SEC == 30 * 86400.0
        assert DEFAULT_CANARY_MAX_WAIT_SEC == 86400.0

    @pytest.mark.parametrize("text", ["0", "0m", "0h", "0d", "000", "00h"])
    def test_zero_is_refused(self, text):
        with pytest.raises(ValueError, match="above 0"):
            parse_wait_duration(text)

    @pytest.mark.parametrize("text", ["-1", "-5m", "-24h", "-1d", "+5m", "+1"])
    def test_a_sign_is_refused(self, text):
        with pytest.raises(ValueError, match="not a valid wait"):
            parse_wait_duration(text)

    @pytest.mark.parametrize(
        "text",
        [
            "", "   ", "abc", "h", "m", "d", "24x", "24hh", "24 h", "1.5h", "1.5", "0.5d", "1e3",
            "24s", "h24", "24h30m", "0x10", "1_000", "NaN", "inf", "true", "false",
            "None", "٣h", "²h", "２４h", "24h\n7", "-", "--5m",
        ],
    )
    def test_garbage_is_refused_with_the_accepted_forms(self, text):
        with pytest.raises(ValueError, match="not a valid wait") as caught:
            parse_wait_duration(text)

        assert "<int>m" in str(caught.value) and "plain seconds" in str(caught.value)

    @pytest.mark.parametrize(
        "text", ["31d", "721h", "43201m", "2592001", "99999999d", "99999999999999999999d"]
    )
    def test_more_than_thirty_days_is_refused(self, text):
        with pytest.raises(ValueError, match="at most 30d"):
            parse_wait_duration(text)

    def test_an_absurdly_long_number_is_refused_without_building_it(self):
        with pytest.raises(ValueError, match="at most 30d"):
            parse_wait_duration("9" * 5000 + "d")

    @pytest.mark.parametrize("value", [None, True, False, 24, 24.0, 0, ["24h"], b"24h"])
    def test_anything_but_a_string_is_refused(self, value):
        with pytest.raises(ValueError, match="not a valid wait"):
            parse_wait_duration(value)

    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (86400.0, "24h"),
            (86400, "24h"),
            (172800.0, "2d"),
            (129600.0, "36h"),
            (2592000.0, "30d"),
            (3600.0, "1h"),
            (5400.0, "90m"),
            (60.0, "1m"),
            (45.0, "45s"),
            (5430.0, "5430s"),
            (1.0, "1s"),
        ],
    )
    def test_it_is_shown_in_the_largest_unit_that_divides_it(self, seconds, text):
        assert format_wait_duration(seconds) == text

    @pytest.mark.parametrize("text", ["90m", "24h", "7d", "36h", "2d", "30d", "45", "5430"])
    def test_what_is_shown_reads_back_as_the_same_wait(self, text):
        seconds = parse_wait_duration(text)
        shown = format_wait_duration(seconds)

        # seconds are shown with an s but typed without one
        typed = shown[:-1] if shown.endswith("s") else shown
        assert parse_wait_duration(typed) == seconds

    def test_the_policies(self):
        assert CANARY_ON_TIMEOUT == ("errors-only", "rollback", "wait")


# ---------------------------------------------------------------------------
# LoopState
# ---------------------------------------------------------------------------


class TestState:
    def test_the_fields_come_last_in_this_order_with_these_defaults(self):
        """Appended last when added; the version-log fields added after them are in
        tests/test_loop_writes_formats.py."""
        names = list(LoopState.__dataclass_fields__)

        start = names.index("canary_max_wait_sec")
        assert names[start:start + 2] == ["canary_max_wait_sec", "canary_on_timeout"]
        assert names[start - 1] == "canary_started_ts"
        state = _state()
        assert state.canary_max_wait_sec == 86400.0
        assert state.canary_on_timeout == "errors-only"

    @pytest.mark.parametrize("value", [1, 1.0, 0.5, 60, 86400, 86400.0, 2592000, 2592000.0])
    def test_a_wait_in_seconds_up_to_thirty_days(self, value):
        assert _state(canary_max_wait_sec=value).canary_max_wait_sec == value

    @pytest.mark.parametrize(
        "value",
        [
            True, False, 0, 0.0, -1, -86400.0, "24h", "86400", None, [1], {"s": 1},
            float("nan"), float("inf"), float("-inf"), 2592000.5, 2592001, 1e300,
        ],
    )
    def test_a_bad_wait_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="canary_max_wait_sec"):
            _state(canary_max_wait_sec=value)

    @pytest.mark.parametrize("value", ["errors-only", "rollback", "wait"])
    def test_the_three_policies(self, value):
        assert _state(canary_on_timeout=value).canary_on_timeout == value

    @pytest.mark.parametrize(
        "value",
        ["", "Rollback", "errors_only", "errors-only ", "stop", "promote", None, 1, True, ["wait"]],
    )
    def test_any_other_policy_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="canary_on_timeout"):
            _state(canary_on_timeout=value)

    def test_the_policy_refusal_lists_the_choices(self):
        with pytest.raises(ValueError, match="errors-only") as caught:
            _state(canary_on_timeout="stop")

        assert "rollback" in str(caught.value) and "wait" in str(caught.value)

    def test_they_round_trip_through_loop_yaml(self, cwd):
        state, path = init_state("m", "e", "b", path=str(cwd / "loop.json"))
        assert state.canary_max_wait_sec == 86400.0 and state.canary_on_timeout == "errors-only"
        write_state(replace(state, canary_max_wait_sec=5400.0, canary_on_timeout="rollback"), path)

        loaded = read_state(path)

        assert loaded.canary_max_wait_sec == 5400.0 and loaded.canary_on_timeout == "rollback"
        assert json.loads((cwd / "loop.json").read_text("utf-8"))["canary_max_wait_sec"] == 5400.0

    def test_init_state_takes_both(self, cwd):
        state, path = init_state(
            "m", "e", "b", canary_max_wait_sec=7200.0, canary_on_timeout="wait",
            path=str(cwd / "loop.json"),
        )

        assert (state.canary_max_wait_sec, state.canary_on_timeout) == (7200.0, "wait")
        assert read_state(path).canary_on_timeout == "wait"

    def test_init_state_refuses_a_bad_value(self, cwd):
        with pytest.raises(ValueError, match="canary_max_wait_sec"):
            init_state("m", "e", "b", canary_max_wait_sec=0, path=str(cwd / "loop.json"))
        assert not (cwd / "loop.json").exists()
        with pytest.raises(ValueError, match="canary_on_timeout"):
            init_state("m", "e", "b", canary_on_timeout="stop", path=str(cwd / "loop.json"))

    def test_a_loop_yaml_from_before_the_fields_loads_with_the_defaults(self, cwd):
        legacy = {"served_model": "m", "eval_suite": "e", "baseline": "b",
                  "canary_active": "v2", "canary_traffic_pct": 5.0, "canary_started_ts": T0}
        (cwd / "legacy.json").write_text(json.dumps(legacy), "utf-8")

        loaded = read_state(str(cwd / "legacy.json"))

        assert loaded.canary_max_wait_sec == 86400.0
        assert loaded.canary_on_timeout == "errors-only"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("canary_max_wait_sec", True),
            ("canary_max_wait_sec", 0),
            ("canary_max_wait_sec", "24h"),
            ("canary_max_wait_sec", 99999999),
            ("canary_on_timeout", "stop"),
            ("canary_on_timeout", 1),
        ],
    )
    def test_a_loop_yaml_with_a_bad_value_is_refused_naming_the_field(self, cwd, field, value):
        body = {"served_model": "m", "eval_suite": "e", "baseline": "b", field: value}
        (cwd / "bad.json").write_text(json.dumps(body), "utf-8")

        with pytest.raises(ValueError, match=f"invalid contents.*{field}"):
            read_state(str(cwd / "bad.json"))


# ---------------------------------------------------------------------------
# soup loop init / status
# ---------------------------------------------------------------------------


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output)


_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")


class TestInit:
    def test_the_defaults_are_a_day_and_errors_only(self, cwd):
        result, text = _cli(*_INIT)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state()
        assert state.canary_max_wait_sec == 86400.0
        assert state.canary_on_timeout == "errors-only"
        assert "canary_max_wait: 24h" in text
        assert "canary_on_timeout: errors-only" in text

    @pytest.mark.parametrize(
        ("flag_value", "seconds", "shown"),
        [
            ("90m", 5400.0, "90m"),
            ("12h", 43200.0, "12h"),
            ("7d", 604800.0, "7d"),
            ("30d", 2592000.0, "30d"),
            ("3600", 3600.0, "1h"),
            ("48H", 172800.0, "2d"),
        ],
    )
    def test_the_wait_is_stored_in_seconds_and_shown_back(self, cwd, flag_value, seconds, shown):
        result, text = _cli(*_INIT, "--canary-max-wait", flag_value)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().canary_max_wait_sec == seconds
        assert f"canary_max_wait: {shown}" in text

    @pytest.mark.parametrize("policy", ["errors-only", "rollback", "wait"])
    def test_each_policy_is_stored_and_shown_back(self, cwd, policy):
        result, text = _cli(*_INIT, "--canary-on-timeout", policy)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().canary_on_timeout == policy
        assert f"canary_on_timeout: {policy}" in text

    @pytest.mark.parametrize("spelling", ["ROLLBACK", " rollback ", "Rollback"])
    def test_a_policy_is_case_and_space_insensitive(self, cwd, spelling):
        result, _ = _cli(*_INIT, "--canary-on-timeout", spelling)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().canary_on_timeout == "rollback"

    def test_errors_only_may_be_written_with_an_underscore(self, cwd):
        result, _ = _cli(*_INIT, "--canary-on-timeout", "errors_only")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state().canary_on_timeout == "errors-only"

    @pytest.mark.parametrize(
        "value",
        ["0", "0m", "0h", "0d", "abc", "", "  ", "1.5h", "24s", "true", "٣h", "h"],
    )
    def test_a_bad_wait_exits_2_naming_the_flag_and_writes_nothing(self, cwd, value):
        result, text = _cli(*_INIT, "--canary-max-wait", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--canary-max-wait" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    @pytest.mark.parametrize("value", ["-1", "-5m", "-24h", "-1d"])
    def test_a_negative_wait_exits_2_naming_the_flag(self, cwd, value):
        result, text = _cli(*_INIT, f"--canary-max-wait={value}")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--canary-max-wait" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_zero_says_it_must_be_above_zero(self, cwd):
        _, text = _cli(*_INIT, "--canary-max-wait", "0h")

        assert "invalid --canary-max-wait" in text and "above 0" in text

    def test_garbage_lists_the_accepted_forms(self, cwd):
        _, text = _cli(*_INIT, "--canary-max-wait", "soon")

        assert "invalid --canary-max-wait" in text
        assert "<int>m" in text and "<int>h" in text and "<int>d" in text
        assert "plain seconds" in text

    @pytest.mark.parametrize("value", ["31d", "721h", "99999999999d"])
    def test_more_than_thirty_days_is_refused(self, cwd, value):
        result, text = _cli(*_INIT, "--canary-max-wait", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--canary-max-wait" in text and "at most 30d" in text

    def test_the_refused_text_is_printed_safely(self, cwd):
        result, text = _cli(*_INIT, "--canary-max-wait", "[bold red]x[/]\x1b]0;pwned\x07")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--canary-max-wait" in text
        assert "[bold red]x[/]" in text  # shown, not interpreted
        assert "\x1b]0;" not in result.output and "\x07" not in result.output

    @pytest.mark.parametrize("value", ["stop", "promote", "", "errors", "error-only", "off"])
    def test_a_bad_policy_exits_2_naming_the_flag_and_the_choices(self, cwd, value):
        result, text = _cli(*_INIT, "--canary-on-timeout", value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--canary-on-timeout must be one of errors-only, rollback, wait" in text
        assert not os.path.exists(os.path.join(".soup", "loop.yaml"))

    def test_both_flags_are_in_the_help(self, cwd):
        result, text = _cli("init", "--help")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--canary-max-wait" in text and "--canary-on-timeout" in text
        assert "errors-only" in text and "rollback" in text

    def test_status_shows_both(self, cwd):
        _cli(*_INIT, "--canary-max-wait", "90m", "--canary-on-timeout", "rollback")

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"canary_max_wait\W+90m", text), text
        assert re.search(r"canary_on_timeout\W+rollback", text), text

    def test_status_of_an_older_loop_yaml_shows_the_defaults(self, cwd):
        legacy = {"served_model": "./base", "eval_suite": "tasks.jsonl", "baseline": "b"}
        os.makedirs(".soup")
        with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        result, text = _cli("status")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert re.search(r"canary_max_wait\W+24h", text), text
        assert re.search(r"canary_on_timeout\W+errors-only", text), text

    def test_a_manual_canary_keeps_the_configured_wait(self, cwd):
        _cli(*_INIT, "--canary-max-wait", "2d", "--canary-on-timeout", "wait")

        result, _ = _cli("canary", "candidate", "--traffic", "10%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state()
        assert state.canary_active == "candidate"
        assert (state.canary_max_wait_sec, state.canary_on_timeout) == (172800.0, "wait")


# ---------------------------------------------------------------------------
# The daemon: when the wait ends
# ---------------------------------------------------------------------------


MAX_WAIT = DAY  # the default


class TestBeforeTheWaitEnds:
    def test_the_canary_keeps_waiting_one_second_before(self, cwd):
        state = _canary_state()
        _record(state, ok=31)
        told = []
        config = _config(
            lambda s: stats(7, 0, 40, 0),
            promote_fn=lambda s: told.append(1) or {"promoted": True},
        )

        new_state, record, decision = run_once(state, config, now=at(MAX_WAIT - 1))

        assert told == [] and decision.proceed is False
        assert new_state.canary_active == CANARY
        assert record.canary_verdict == "UNKNOWN"
        assert record.notes == (
            "canary-wait: v2 has 31 canary answers (verdict at 30) | canary errors: OK | "
            "canary quality: 7/30 canary, 40/30 stable rated (UNKNOWN)"
        )
        assert record.canary_quality["verdict"] == "UNKNOWN"
        assert record.canary_quality["timed_out"] is False
        assert record.canary_quality["decided_on"] == "errors_and_quality"

    def test_the_clock_starts_at_the_rollout_start_not_at_the_first_cycle(self, cwd):
        state = _canary_state(canary_started_ts=T0 + 3600)
        _record(state, ok=31)

        new_state, _, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                   now=at(MAX_WAIT + 3599))

        assert new_state.canary_active == CANARY  # a day after the cycle, not after the start
        new_state, _, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                   now=at(MAX_WAIT + 3600))
        assert new_state.production_name == CANARY


class TestErrorsOnly:
    def test_at_the_maximum_wait_the_error_verdict_decides_and_the_canary_is_promoted(self, cwd):
        state = _canary_state()
        _record(state, ok=31)
        told = []
        config = _config(
            lambda s: stats(7, 0, 40, 0),
            promote_fn=lambda s: told.append(s.canary_active) or {"promoted": True},
        )

        new_state, record, _ = run_once(state, config, now=at(MAX_WAIT))

        assert told == [CANARY]
        assert new_state.production_name == CANARY
        assert new_state.canary_active is None and new_state.canary_started_ts is None
        assert record.canary_verdict == "OK"
        assert (
            "promoted on errors only - quality unverified: 7/30 canary rated in 24h"
            in record.notes
        )
        assert "canary v2 promoted to production" in record.notes
        assert "canary errors: OK" in record.notes
        assert "canary quality: 7/30 canary, 40/30 stable rated (INSUFFICIENT)" in record.notes

    def test_the_evidence_says_what_was_missing(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(7, 1, 40, 2)), now=at(MAX_WAIT))

        expected = stats(7, 1, 40, 2).evidence(
            verdict="INSUFFICIENT", timed_out=True, decided_on="errors_only"
        )
        assert record.canary_quality == expected
        assert record.canary_quality["verdict"] == "INSUFFICIENT"
        assert record.canary_quality["rated"] == {"stable": 40, "canary": 7}
        assert record.canary_quality["negative"] == {"stable": 2, "canary": 1}
        assert record.canary_quality["share"] == {"stable": 0.05, "canary": 0.1429}
        assert record.canary_quality["difference"] == 0.0929
        assert record.canary_quality["timed_out"] is True
        assert record.canary_quality["decided_on"] == "errors_only"
        assert record.canary_quality["min_rated"] == 30

    def test_only_the_short_side_is_named(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(40, 0, 5, 0)), now=at(MAX_WAIT))

        assert "quality unverified: 5/30 stable rated in 24h" in record.notes

    def test_both_sides_short_are_both_named(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(7, 0, 5, 0)), now=at(MAX_WAIT))

        assert "quality unverified: 7/30 canary, 5/30 stable rated in 24h" in record.notes

    def test_the_wait_is_shown_as_configured(self, cwd):
        state = _canary_state(canary_max_wait_sec=5400.0)
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)), now=at(5400))

        assert "quality unverified: 7/30 canary rated in 90m" in record.notes

    def test_nothing_rated_at_all_is_still_unverified_not_failed(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(0, 0, 0, 0)),
                                        now=at(MAX_WAIT))

        assert new_state.production_name == CANARY
        assert "quality unverified: 0/30 canary, 0/30 stable rated in 24h" in record.notes

    def test_a_canary_the_requests_do_not_vouch_for_is_not_promoted(self, cwd):
        """Only 10 canary answers: the error verdict is itself UNKNOWN, so it keeps waiting."""
        state = _canary_state()
        _record(state, ok=10)
        told = []
        config = _config(
            lambda s: stats(7, 0, 40, 0),
            promote_fn=lambda s: told.append(1) or {"promoted": True},
        )

        new_state, record, decision = run_once(state, config, now=at(MAX_WAIT * 5))

        assert told == [] and decision.proceed is False
        assert new_state.canary_active == CANARY
        assert record.canary_verdict == "UNKNOWN"
        assert "canary errors: UNKNOWN" in record.notes
        assert "unverified" not in record.notes
        assert record.canary_quality["verdict"] == "INSUFFICIENT"

    def test_failed_requests_still_withdraw_it(self, cwd):
        state = _canary_state()
        _record(state, ok=20, bad=10)

        new_state, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                        now=at(MAX_WAIT))

        assert new_state.canary_active is None and new_state.production_name == STABLE
        assert record.rolled_back is True and record.canary_verdict == "MAJOR"
        assert "canary v2 rolled back (MAJOR)" in record.notes
        assert "unverified" not in record.notes
        assert record.canary_quality["decided_on"] == "errors_only"

    def test_failed_requests_withdraw_nothing_when_autoroll_is_off(self, cwd):
        state = _canary_state(canary_autoroll_on_regress=False)
        _record(state, ok=20, bad=10)

        new_state, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                        now=at(MAX_WAIT))

        assert new_state.canary_active == CANARY
        assert record.rolled_back is False

    def test_the_default_is_errors_only(self, cwd):
        state = _canary_state()

        assert state.canary_on_timeout == "errors-only"
        _record(state, ok=31)
        new_state, _, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)), now=at(MAX_WAIT))
        assert new_state.production_name == CANARY

    def test_a_promotion_the_server_refused_is_not_called_unverified(self, cwd):
        state = _canary_state()
        _record(state, ok=31)
        config = _config(
            lambda s: stats(7, 0, 40, 0),
            promote_fn=lambda s: {"promoted": False, "notes": "server said no"},
        )

        new_state, record, _ = run_once(state, config, now=at(MAX_WAIT))

        assert new_state.canary_active == CANARY and new_state.production_name == STABLE
        assert "canary v2 passed but was not promoted: server said no" in record.notes
        assert "promoted on errors only" not in record.notes


class TestRollback:
    def test_at_the_maximum_wait_the_canary_is_withdrawn_and_production_untouched(self, cwd):
        state = _canary_state(canary_on_timeout="rollback")
        _record(state, ok=31)
        told = []
        config = _config(
            lambda s: stats(7, 0, 40, 0),
            promote_fn=lambda s: told.append(1) or {"promoted": True},
        )

        new_state, record, _ = run_once(state, config, now=at(MAX_WAIT))

        assert told == []
        assert new_state.canary_active is None and new_state.canary_adapter is None
        assert new_state.canary_started_ts is None and new_state.canary_rollout_id is None
        assert new_state.production_name == STABLE  # production untouched
        assert new_state.production_adapter == "adapters/v1"
        assert record.rolled_back is True
        assert record.canary_verdict == "UNKNOWN"
        assert "canary v2 rolled back" in record.notes
        assert "7/30 canary rated in 24h" in record.notes
        assert "--canary-on-timeout rollback" in record.notes
        assert "promoted" not in record.notes

    def test_it_withdraws_even_without_autoroll_on_regress(self, cwd):
        state = _canary_state(canary_on_timeout="rollback", canary_autoroll_on_regress=False)
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                        now=at(MAX_WAIT))

        assert new_state.canary_active is None
        assert record.rolled_back is True

    def test_the_evidence_says_the_wait_ran_out(self, cwd):
        state = _canary_state(canary_on_timeout="rollback")
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)), now=at(MAX_WAIT))

        assert record.canary_quality["verdict"] == "INSUFFICIENT"
        assert record.canary_quality["timed_out"] is True
        assert record.canary_quality["decided_on"] == "errors_and_quality"

    def test_before_the_maximum_wait_it_still_waits(self, cwd):
        state = _canary_state(canary_on_timeout="rollback")
        _record(state, ok=31)

        new_state, record, decision = run_once(
            state, _config(lambda s: stats(7, 0, 40, 0)), now=at(MAX_WAIT - 1)
        )

        assert new_state.canary_active == CANARY
        assert record.rolled_back is False and decision.proceed is False

    def test_enough_ratings_are_judged_normally_after_the_wait(self, cwd):
        state = _canary_state(canary_on_timeout="rollback")
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(31, 3, 40, 2)),
                                        now=at(MAX_WAIT * 3))

        assert new_state.production_name == CANARY
        assert record.rolled_back is False
        assert record.canary_quality["verdict"] == "OK"
        assert record.canary_quality["timed_out"] is False

    def test_failed_requests_say_major_not_a_timeout(self, cwd):
        state = _canary_state(canary_on_timeout="rollback")
        _record(state, ok=20, bad=10)

        _, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)), now=at(MAX_WAIT))

        assert record.canary_verdict == "MAJOR"
        assert "canary v2 rolled back (MAJOR)" in record.notes


class TestWait:
    def test_it_waits_however_long_it_takes(self, cwd):
        state = _canary_state(canary_on_timeout="wait")
        _record(state, ok=31)
        told = []
        config = _config(
            lambda s: stats(7, 0, 40, 0),
            promote_fn=lambda s: told.append(1) or {"promoted": True},
        )

        for waited in (MAX_WAIT, MAX_WAIT * 30, MAX_WAIT * 3650):
            new_state, record, decision = run_once(state, config, now=at(waited))

            assert told == [] and decision.proceed is False
            assert new_state.canary_active == CANARY
            assert record.canary_verdict == "UNKNOWN"
            assert record.rolled_back is False
            assert "canary quality: 7/30 canary, 40/30 stable rated (UNKNOWN)" in record.notes
            assert record.canary_quality["verdict"] == "UNKNOWN"
            assert record.canary_quality["timed_out"] is False

    def test_a_bad_canary_is_still_withdrawn(self, cwd):
        state = _canary_state(canary_on_timeout="wait")
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(31, 12, 40, 2)),
                                        now=at(MAX_WAIT * 9))

        assert new_state.canary_active is None
        assert record.canary_verdict == "MAJOR"


class TestWhereTheLimitDoesNotApply:
    def test_enough_ratings_after_the_wait_are_judged_on_their_merits(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(31, 3, 40, 2)),
                                        now=at(MAX_WAIT * 4))

        assert new_state.production_name == CANARY
        assert "unverified" not in record.notes
        assert "(OK)" in record.notes
        assert record.canary_quality["timed_out"] is False
        assert record.canary_quality["decided_on"] == "errors_and_quality"

    def test_a_regression_found_after_the_wait_is_a_regression(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(100, 30, 100, 15)),
                                        now=at(MAX_WAIT * 4))

        assert new_state.canary_active is None and new_state.production_name == STABLE
        assert record.canary_verdict == "MAJOR"
        assert record.canary_quality["verdict"] == "MAJOR"
        assert record.canary_quality["timed_out"] is False

    def test_a_loop_without_ratings_to_read_is_judged_by_errors_as_before(self, cwd):
        for quality in (None, lambda s: None):
            state = _canary_state(canary_on_timeout="rollback")
            _record(state, ok=31)

            new_state, record, _ = run_once(state, _config(quality), now=at(MAX_WAIT * 9))

            assert new_state.production_name == CANARY
            assert "unverified" not in record.notes and "canary quality" not in record.notes
            assert record.canary_quality is None
            os.remove(os.path.join(".soup", "canary-stats.json"))

    def test_a_rollout_without_a_recorded_start_has_nothing_to_time(self, cwd):
        state = _canary_state(canary_started_ts=None)
        _record(state, ok=31)

        new_state, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                        now=at(MAX_WAIT * 9))

        assert new_state.canary_active == CANARY
        assert record.canary_verdict == "UNKNOWN"
        assert record.canary_quality["timed_out"] is False

    def test_the_wait_is_the_one_in_loop_yaml(self, cwd):
        for wait, waited, promoted in (
            (3600.0, 3599, False),
            (3600.0, 3600, True),
            (7 * DAY, 7 * DAY - 1, False),
            (7 * DAY, 7 * DAY, True),
        ):
            state = _canary_state(canary_max_wait_sec=wait)
            _record(state, ok=31)

            new_state, _, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)),
                                       now=at(waited))

            assert (new_state.production_name == CANARY) is promoted, (wait, waited)
            os.remove(os.path.join(".soup", "canary-stats.json"))


class TestTheClock:
    def test_the_clock_can_be_replaced_for_the_daemon(self, cwd, monkeypatch):
        state = _canary_state()
        _record(state, ok=31)
        monkeypatch.setattr(daemon, "_now", lambda: at(MAX_WAIT + 1))

        new_state, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)))

        assert new_state.production_name == CANARY
        assert "quality unverified" in record.notes

    def test_an_earlier_clock_means_no_timeout(self, cwd, monkeypatch):
        state = _canary_state()
        _record(state, ok=31)
        monkeypatch.setattr(daemon, "_now", lambda: at(10))

        new_state, _, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)))

        assert new_state.canary_active == CANARY

    def test_the_real_clock_is_utc_and_now(self):
        before = datetime.now(timezone.utc)

        found = daemon._now()

        assert found.tzinfo is not None and found.utcoffset().total_seconds() == 0
        assert before <= found <= datetime.now(timezone.utc)

    def test_an_explicit_now_wins(self, cwd, monkeypatch):
        state = _canary_state()
        _record(state, ok=31)
        monkeypatch.setattr(daemon, "_now", lambda: at(MAX_WAIT * 9))

        new_state, _, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)), now=at(10))

        assert new_state.canary_active == CANARY


# ---------------------------------------------------------------------------
# The evidence in the iteration record
# ---------------------------------------------------------------------------


def _iteration(**kwargs) -> IterationRecord:
    kwargs.setdefault("iteration_id", "iter-20261105T120000-1b2c3d4e")
    kwargs.setdefault("started_at", "2026-11-05T12:00:00+00:00")
    kwargs.setdefault("finished_at", "2026-11-05T12:00:01+00:00")
    kwargs.setdefault("pairs_harvested", 0)
    kwargs.setdefault("run_id", None)
    kwargs.setdefault("gate_verdict", "SKIPPED")
    kwargs.setdefault("canary_verdict", "OK")
    kwargs.setdefault("shipped", False)
    kwargs.setdefault("rolled_back", False)
    kwargs.setdefault("estimated_cost_usd", 0.0)
    return IterationRecord(**kwargs)


class TestIterationRecord:
    def test_the_evidence_field_is_last_and_defaults_to_none(self):
        names = list(IterationRecord.__dataclass_fields__)

        assert names[-1] == "canary_quality"
        assert names[-2] == "notes"
        assert _iteration().canary_quality is None

    def test_it_carries_the_evidence_dict(self):
        evidence = stats(40, 9, 40, 6).evidence()

        record = _iteration(canary_quality=evidence)

        assert record.canary_quality == evidence
        assert dict(record.to_dict())["canary_quality"] == evidence

    @pytest.mark.parametrize(
        "value",
        ["OK", ["OK"], 5, 0.5, True, False, b"x", ("a", 1), {1, 2}, {"z": float("nan")},
         {"z": float("inf")}, {"x": object()}, {1: "numeric key"}, {"x": "y" * 5000}],
    )
    def test_anything_else_is_refused_naming_the_field(self, value):
        with pytest.raises(ValueError, match="canary_quality"):
            _iteration(canary_quality=value)

    def test_it_round_trips_through_the_manifest(self, cwd):
        evidence = stats(100, 30, 100, 15).evidence()
        write_iteration(_iteration(canary_quality=evidence))

        loaded = read_iteration("iter-20261105T120000-1b2c3d4e")

        assert loaded.canary_quality == evidence

    def test_an_older_manifest_loads_without_it(self, cwd):
        record = _iteration()
        write_iteration(record)
        path = os.path.join(".soup-loops", record.iteration_id, "iteration.json")
        body = json.loads(open(path, encoding="utf-8").read())
        body.pop("canary_quality")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(body, handle)

        assert read_iteration(record.iteration_id).canary_quality is None

    def test_a_manifest_with_a_bad_value_is_refused_naming_the_field(self, cwd):
        record = _iteration()
        write_iteration(record)
        path = os.path.join(".soup-loops", record.iteration_id, "iteration.json")
        body = json.loads(open(path, encoding="utf-8").read())
        body["canary_quality"] = "OK"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(body, handle)

        with pytest.raises(ValueError, match="canary_quality"):
            read_iteration(record.iteration_id)


class TestRecordsCarryTheEvidence:
    def test_a_promotion_records_the_ok_evidence(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(31, 3, 40, 2)), now=at(60))

        assert record.canary_quality == stats(31, 3, 40, 2).evidence()
        assert record.canary_quality["verdict"] == "OK"

    def test_a_withdrawal_records_the_major_evidence(self, cwd):
        state = _canary_state()
        _record(state, ok=31)

        _, record, _ = run_once(state, _config(lambda s: stats(100, 30, 100, 15)), now=at(60))

        assert record.canary_quality == stats(100, 30, 100, 15).evidence()

    def test_a_budget_skipped_cycle_records_it_too(self, cwd):
        state = _canary_state(monthly_budget_usd=1.0)
        _record(state, ok=31)

        _, record, decision = run_once(
            state, _config(lambda s: stats(31, 3, 40, 2), cost_fn=lambda s: 5.0), now=at(60)
        )

        assert decision.proceed is False
        assert record.canary_quality["verdict"] == "OK"

    def test_a_cycle_with_no_canary_has_none(self, cwd):
        _, record, _ = run_once(_state(), _config(lambda s: stats(31, 3, 40, 2)), now=at(60))

        assert record.canary_quality is None

    def test_the_manifest_the_daemon_writes_has_it(self, cwd):
        state = _canary_state()
        _record(state, ok=31)
        _, record, _ = run_once(state, _config(lambda s: stats(7, 0, 40, 0)), now=at(MAX_WAIT))

        write_iteration(record)

        loaded = read_iteration(record.iteration_id)
        assert loaded.canary_quality["verdict"] == "INSUFFICIENT"
        assert loaded.canary_quality["rated"] == {"stable": 40, "canary": 7}
