"""``soup loop canary`` refuses an adapter whose last verdict was DON'T SHIP (review 5, F10).

The documented manual path, ``soup loop canary <adapter>``, promoted an adapter whose ``soup ship``
verdict was DON'T SHIP: the canary rule only asks that the canary is not worse than stable, and
stable had been rated down as often. Now the command refuses by name, using what the loop's own
logs say about that adapter (the latest line of the iteration that trained it, or a verdict file
``.soup-loops/verdicts/<adapter>.json``), unless ``--allow-dont-ship`` is given; the override is
recorded in ``loop.yaml`` (``canary_override``), the ``canary`` start line of ``progress.jsonl``
and, when the canary is promoted, the promotion's note and its line in ``versions.jsonl``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.commands.loop import app as loop_app
from soup_cli.utils.loop_daemon import WatchConfig, _promote
from soup_cli.utils.loop_state import LoopState, read_state, write_state

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ADAPTER = "loop-train-ab12cd34"
ITERATION = "iter-20261116T100000-aaaaaaaa"


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".soup-loops").mkdir()
    return tmp_path


def _line(iteration_id=ITERATION, *, outcome="skipped", decision="DON'T SHIP", adapter=ADAPTER,
          served_as=None, run_id=None, reason="General benchmark(s) regressed past 5.00%"):
    return {
        "schema_version": 2, "iteration_id": iteration_id, "outcome": outcome,
        "started_at": "2026-11-16T10:00:00+00:00",
        "train": {"method": "sft", "run_id": run_id or adapter, "base": "b", "root_base": "b"},
        "artifacts": {
            "adapter": f".soup-loops/adapters/{adapter}", "merged": None, "gguf": None,
            "served_as": served_as,
        },
        "verdict": {"decision": decision, "method": "ship", "reason": reason},
    }


def _write_log(cwd, lines, raw=""):
    text = "".join(json.dumps(line) + "\n" for line in lines) + raw
    (cwd / ".soup-loops" / "iterations.jsonl").write_text(text, encoding="utf-8")


def _state(**kwargs) -> LoopState:
    return LoopState(
        served_model="org/base", eval_suite="e", baseline="b", base_model="org/base",
        status="running", **kwargs,
    )


class TestKnownDontShip:
    def test_a_skipped_iteration_with_a_dont_ship_verdict_is_a_no(self, cwd):
        _write_log(cwd, [_line()])

        text = ls.known_dont_ship(ADAPTER)

        assert text == (
            f"its last verdict is DON'T SHIP (iteration {ITERATION}: "
            "General benchmark(s) regressed past 5.00%)"
        )

    def test_the_adapter_is_found_by_served_name_and_by_run_id_too(self, cwd):
        _write_log(cwd, [_line(adapter="elsewhere", served_as="named-one", run_id="run-7")])

        assert ls.known_dont_ship("named-one") is not None
        assert ls.known_dont_ship("run-7") is not None
        assert ls.known_dont_ship("elsewhere") is not None

    def test_an_iteration_that_went_on_to_a_canary_is_not_a_no(self, cwd):
        # the check found nothing to verify (canary_decides): the loop itself deployed it
        _write_log(cwd, [_line(outcome="canary")])

        assert ls.known_dont_ship(ADAPTER) is None

    @pytest.mark.parametrize("outcome", ["promoted", "rolled_back", "failed"])
    def test_other_outcomes_are_not_a_no(self, cwd, outcome):
        _write_log(cwd, [_line(outcome=outcome)])

        assert ls.known_dont_ship(ADAPTER) is None

    def test_a_ship_verdict_is_not_a_no(self, cwd):
        _write_log(cwd, [_line(decision="SHIP")])

        assert ls.known_dont_ship(ADAPTER) is None

    def test_the_latest_line_of_an_iteration_wins(self, cwd):
        _write_log(cwd, [_line(), _line(outcome="promoted")])

        assert ls.known_dont_ship(ADAPTER) is None

    def test_the_newest_iteration_that_trained_it_decides(self, cwd):
        newer = "iter-20261116T110000-bbbbbbbb"
        _write_log(cwd, [_line(decision="SHIP", outcome="canary"), _line(newer)])

        assert newer in ls.known_dont_ship(ADAPTER)

    def test_an_adapter_the_loop_never_trained_is_unknown(self, cwd):
        _write_log(cwd, [_line()])

        assert ls.known_dont_ship("manual-sft-1") is None

    def test_no_log_is_unknown(self, cwd):
        assert ls.known_dont_ship(ADAPTER) is None

    def test_a_torn_and_a_foreign_line_do_not_stop_the_lookup(self, cwd):
        _write_log(cwd, [_line()], raw='{"half"\n[1, 2]\n{"iteration_id": 5}\n')

        assert ls.known_dont_ship(ADAPTER) is not None

    def test_a_verdict_file_named_after_the_adapter_counts(self, cwd):
        verdicts = cwd / ".soup-loops" / "verdicts"
        verdicts.mkdir()
        (verdicts / "manual-sft-1.json").write_text(
            json.dumps({"decision": "DON'T SHIP", "failed_rule": "regression"}), encoding="utf-8"
        )

        assert ls.known_dont_ship("manual-sft-1") == (
            "its verdict file .soup-loops/verdicts/manual-sft-1.json says DON'T SHIP"
        )

    def test_a_ship_verdict_file_does_not(self, cwd):
        verdicts = cwd / ".soup-loops" / "verdicts"
        verdicts.mkdir()
        (verdicts / "manual-sft-1.json").write_text(
            json.dumps({"decision": "SHIP"}), encoding="utf-8"
        )

        assert ls.known_dont_ship("manual-sft-1") is None

    def test_the_log_decides_before_a_verdict_file(self, cwd):
        _write_log(cwd, [_line(outcome="canary")])
        verdicts = cwd / ".soup-loops" / "verdicts"
        verdicts.mkdir()
        (verdicts / f"{ADAPTER}.json").write_text(
            json.dumps({"decision": "DON'T SHIP"}), encoding="utf-8"
        )

        assert ls.known_dont_ship(ADAPTER) is None

    @pytest.mark.parametrize("name", ["../x", "a/b", "a\\b", "", " ", "x" * 5 + "\n", "-lead"])
    def test_a_name_that_is_not_an_adapter_name_reads_nothing(self, cwd, name):
        assert ls.known_dont_ship(name) is None


class TestTheCommand:
    def _invoke(self, *args):
        result = CliRunner().invoke(loop_app, ["canary", *args])
        return result, _plain(result.output)

    def _seed(self, cwd, **kwargs):
        write_state(_state(**kwargs), str(cwd / ".soup" / "loop.yaml"))

    def test_it_refuses_by_name_and_changes_nothing(self, cwd):
        _write_log(cwd, [_line()])
        self._seed(cwd)

        result, text = self._invoke(ADAPTER, "--traffic", "50%")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert f"refusing to canary {ADAPTER}: its last verdict is DON'T SHIP" in text
        assert f"(iteration {ITERATION}" in text
        assert "pass --allow-dont-ship to canary it anyway (the override is recorded)" in text
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active is None and state.canary_override is None
        assert not (cwd / ".soup-loops" / "progress.jsonl").exists()

    def test_an_adapter_with_no_bad_verdict_is_canaried_as_before(self, cwd):
        _write_log(cwd, [_line()])
        self._seed(cwd)

        result, text = self._invoke("manual-sft-1", "--traffic", "50%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set: manual-sft-1 @ 50.0%" in text
        assert read_state(str(cwd / ".soup" / "loop.yaml")).canary_override is None

    def test_the_override_goes_through_and_is_recorded_three_ways(self, cwd):
        _write_log(cwd, [_line()])
        self._seed(cwd)

        result, text = self._invoke(ADAPTER, "--traffic", "50%", "--allow-dont-ship")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active == ADAPTER
        assert state.canary_override.startswith(
            f"{ADAPTER} was canaried over a DON'T SHIP verdict with --allow-dont-ship ("
        )
        assert ITERATION in state.canary_override
        progress = [
            json.loads(line)
            for line in (cwd / ".soup-loops" / "progress.jsonl").read_text("utf-8").splitlines()
        ]
        start = [row for row in progress if row["stage"] == "canary" and row["event"] == "start"]
        assert len(start) == 1 and "--allow-dont-ship" in start[0]["message"]

    def test_the_override_without_a_bad_verdict_records_nothing(self, cwd):
        self._seed(cwd)

        result, _text = self._invoke("manual-sft-1", "--allow-dont-ship")

        assert result.exit_code == 0
        assert read_state(str(cwd / ".soup" / "loop.yaml")).canary_override is None

    def test_a_hostile_log_reason_prints_no_control_bytes(self, cwd):
        _write_log(cwd, [_line(reason="boom \x1b]0;PWNED\x07 done")])
        self._seed(cwd)

        result, _text = self._invoke(ADAPTER)

        assert result.exit_code == 2 and "\x1b]" not in result.output


class TestThePromotionCarriesTheOverride:
    OVERRIDE = f"{ADAPTER} was canaried over a DON'T SHIP verdict with --allow-dont-ship (x)"

    def test_the_promotion_note_and_the_state_say_so(self, cwd):
        state = _state(canary_active=ADAPTER, canary_traffic_pct=50.0,
                       canary_override=self.OVERRIDE)

        promoted, note = _promote(state, WatchConfig(poll_interval_sec=1.0))

        assert promoted.production_name == ADAPTER
        assert promoted.canary_override is None
        assert note == f"canary {ADAPTER} promoted to production; {self.OVERRIDE}; " + (
            "its adapter path is unknown, so the next verdict compares against the base"
        )

    def test_a_promotion_without_an_override_is_as_before(self, cwd):
        state = _state(canary_active=ADAPTER, canary_traffic_pct=50.0)

        _promoted, note = _promote(state, WatchConfig(poll_interval_sec=1.0))

        assert "--allow-dont-ship" not in note

    def test_rollback_clears_it(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(
            _state(canary_active=ADAPTER, canary_traffic_pct=50.0, canary_rollout_id="r1",
                   canary_override=self.OVERRIDE),
            path,
        )

        result = CliRunner().invoke(loop_app, ["rollback"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state(path).canary_active is None
        assert read_state(path).canary_override is None


class TestStateField:
    def test_it_round_trips(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(_state(canary_override="an override"), path)

        assert read_state(path).canary_override == "an override"

    @pytest.mark.parametrize("bad", ["", 5, ["x"], "a\x1bb", "a\nb"])
    def test_a_bad_value_is_refused(self, bad):
        with pytest.raises((ValueError, TypeError)):
            replace(_state(), canary_override=bad)
