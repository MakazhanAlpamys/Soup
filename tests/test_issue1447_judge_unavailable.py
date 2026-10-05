"""Regression tests for #1447: a judge that fails is unavailable, not a tie.

Before the fix the pairwise path scored every judge exception as a 0.5 tie, so
``soup ship --task-mode pairwise`` printed a measured DON'T SHIP with the judge
down; ``soup eval judge`` aborted on the first transport error and lost the
scores already collected; nothing was ever retried; and a training eval gate
called an unreachable judge a regression.

``httpx.post`` is patched throughout, so no judge server is needed, and the
retry pause is captured instead of slept.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import httpx
import pytest
from typer.testing import CliRunner

from soup_cli.eval import judge as judge_mod
from soup_cli.eval.judge import JudgeEvaluator, JudgeUnavailableError, pairwise_winrate
from tests.conftest import strip_ansi

JUDGE_BASE = "http://localhost:8000"
REQ = httpx.Request("POST", f"{JUDGE_BASE}/v1/chat/completions")
SCORED = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "scores": {"helpfulness": 4, "accuracy": 4, "safety": 4},
                        "reasoning": "ok",
                    }
                )
            }
        }
    ]
}


def _reply(status: int, body: object, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers, request=REQ)


def _content(text: str) -> httpx.Response:
    return _reply(200, {"choices": [{"message": {"content": text}}]})


def _refused(*args, **kwargs):
    raise httpx.ConnectError("connection refused", request=REQ)


def _timed_out(*args, **kwargs):
    raise httpx.ReadTimeout("read timed out", request=REQ)


@pytest.fixture
def no_sleep(monkeypatch):
    """Capture the retry pauses instead of sleeping them."""
    delays: list[float] = []
    monkeypatch.setattr(judge_mod, "_sleep", delays.append)
    return delays


@pytest.fixture
def evaluator() -> JudgeEvaluator:
    return JudgeEvaluator(provider="server", model="judge", api_base=JUDGE_BASE)


ATTEMPTS = judge_mod.JUDGE_MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Pairwise: a failure is not a tie
# ---------------------------------------------------------------------------


class TestPairwiseFailureIsNotATie:
    def test_dead_judge_raises_instead_of_scoring_half(self, evaluator, no_sleep):
        pairs = [("p", "base answer", "tuned answer")] * 5
        with mock.patch("httpx.post", side_effect=_refused) as post:
            with pytest.raises(JudgeUnavailableError) as excinfo:
                pairwise_winrate(pairs, evaluator)
        # The message names the judge, and the first pair aborts the run after
        # the bounded retries: no further pairs are attempted.
        assert JUDGE_BASE in str(excinfo.value)
        assert post.call_count == ATTEMPTS

    def test_judge_that_answers_tie_still_scores_half(self, evaluator):
        pairs = [("p", "base answer", "tuned answer")] * 3
        with mock.patch("httpx.post", return_value=_content("I cannot decide")):
            assert pairwise_winrate(pairs, evaluator) == 0.5

    def test_judge_that_picks_tuned_scores_one(self, evaluator):
        # Base is A and tuned is B in the first order; the swapped order must
        # agree, so the fake answers by position.
        def post(url, json, headers, timeout):
            body = json["messages"][0]["content"]
            winner = "B" if body.index("tuned answer") > body.index("base answer") else "A"
            return _content(f'{{"winner": "{winner}"}}')

        with mock.patch("httpx.post", side_effect=post):
            assert pairwise_winrate([("p", "base answer", "tuned answer")], evaluator) == 1.0

    def test_compare_pair_lets_the_error_propagate(self, evaluator, monkeypatch):
        def boom(prompt):
            raise JudgeUnavailableError("down", url=REQ.url)

        monkeypatch.setattr(evaluator, "_call_llm", boom)
        with pytest.raises(JudgeUnavailableError):
            evaluator.compare_pair("p", "x", "y")

    def test_online_dpo_adapter_keeps_a_down_judge_unranked(self, evaluator, no_sleep):
        # #1225's contract: the ranking trainer counts -1 pairs and stops the run
        # itself with an error naming the judge, so this one adapter must not raise.
        from soup_cli.eval.judge import make_soup_pairwise_judge

        try:
            judge = make_soup_pairwise_judge(evaluator)
        except ImportError:
            pytest.skip("installed trl has no BasePairwiseJudge")
        with mock.patch("httpx.post", side_effect=_refused) as post:
            assert judge.judge(["p"], [["first", "second"]]) == [-1]
        # the first order spent its bounded retries; the swapped order was never asked
        assert post.call_count == ATTEMPTS


# ---------------------------------------------------------------------------
# One request helper: bounded retry, Retry-After, reply-shape validation
# ---------------------------------------------------------------------------


class TestRetry:
    def test_429_then_200_is_scored_after_one_retry(self, evaluator, no_sleep):
        replies = iter(
            [_reply(429, {"error": "rate limited"}, {"Retry-After": "3"}), _reply(200, SCORED)]
        )
        with mock.patch("httpx.post", side_effect=lambda *a, **k: next(replies)) as post:
            score = evaluator.evaluate("q", "a")
        assert score.weighted_score == 4.0
        assert post.call_count == 2
        assert no_sleep == [3.0]

    def test_transport_error_then_200_is_scored(self, evaluator, no_sleep):
        replies = iter([_timed_out, _reply(200, SCORED)])

        def post(*args, **kwargs):
            item = next(replies)
            return item(*args, **kwargs) if callable(item) else item

        with mock.patch("httpx.post", side_effect=post) as post_mock:
            assert evaluator.evaluate("q", "a").weighted_score == 4.0
        assert post_mock.call_count == 2
        assert len(no_sleep) == 1

    def test_persistent_503_is_bounded_and_raises(self, evaluator, no_sleep):
        with mock.patch("httpx.post", return_value=_reply(503, {"error": "down"})) as post:
            with pytest.raises(JudgeUnavailableError, match="503"):
                evaluator.evaluate("q", "a")
        assert post.call_count == ATTEMPTS
        assert len(no_sleep) == judge_mod.JUDGE_MAX_RETRIES

    def test_backoff_without_retry_after_grows_and_is_capped(self, evaluator, no_sleep):
        with mock.patch("httpx.post", return_value=_reply(429, {"error": "rate limited"})):
            with pytest.raises(JudgeUnavailableError):
                evaluator.evaluate("q", "a")
        assert no_sleep == sorted(no_sleep)
        assert all(0 < d <= judge_mod.JUDGE_MAX_BACKOFF_SECONDS for d in no_sleep)

    def test_retry_after_is_capped(self, evaluator, no_sleep):
        with mock.patch("httpx.post", return_value=_reply(429, {}, {"Retry-After": "86400"})):
            with pytest.raises(JudgeUnavailableError):
                evaluator.evaluate("q", "a")
        assert no_sleep == [judge_mod.JUDGE_MAX_BACKOFF_SECONDS] * judge_mod.JUDGE_MAX_RETRIES

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_are_not_retried(self, evaluator, no_sleep, status):
        with mock.patch("httpx.post", return_value=_reply(status, {"error": "no"})) as post:
            with pytest.raises(JudgeUnavailableError, match=str(status)):
                evaluator.evaluate("q", "a")
        assert post.call_count == 1
        assert no_sleep == []

    @pytest.mark.parametrize(
        "body",
        [
            {"choices": []},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": None}}]},
            {"error": "no choices"},
            ["not", "a", "dict"],
        ],
        ids=["empty-choices", "no-content", "null-content", "no-choices-key", "not-a-dict"],
    )
    def test_malformed_200_reply_is_a_judge_error(self, evaluator, no_sleep, body):
        with mock.patch("httpx.post", return_value=_reply(200, body)) as post:
            with pytest.raises(JudgeUnavailableError):
                evaluator.evaluate("q", "a")
        assert post.call_count == 1

    def test_non_json_200_reply_is_a_judge_error(self, evaluator, no_sleep):
        reply = httpx.Response(200, text="<html>gateway</html>", request=REQ)
        with mock.patch("httpx.post", return_value=reply):
            with pytest.raises(JudgeUnavailableError):
                evaluator.evaluate("q", "a")

    def test_error_names_the_url_and_the_attempt_count(self, evaluator, no_sleep):
        with mock.patch("httpx.post", side_effect=_refused):
            with pytest.raises(JudgeUnavailableError) as excinfo:
                evaluator.evaluate("q", "a")
        message = str(excinfo.value)
        assert f"{JUDGE_BASE}/v1/chat/completions" in message
        assert f"{ATTEMPTS} attempt" in message
        assert "ConnectError" in message
        assert excinfo.value.url == f"{JUDGE_BASE}/v1/chat/completions"


# ---------------------------------------------------------------------------
# Review round 1: hostile Retry-After, absolute bounds, undecodable replies,
# credentials in the judge URL, pickling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["nan", "NaN", "-5", "-inf", "inf", "1e400", "abc", ""])
def test_hostile_retry_after_never_yields_a_bad_pause(evaluator, no_sleep, value):
    with mock.patch("httpx.post", return_value=_reply(429, {}, {"Retry-After": value})):
        with pytest.raises(JudgeUnavailableError):
            evaluator.evaluate("q", "a")
    assert no_sleep
    # time.sleep(nan) and time.sleep(-1) both raise ValueError
    assert all(math.isfinite(d) and 0 <= d <= judge_mod.JUDGE_MAX_BACKOFF_SECONDS for d in no_sleep)


def test_the_bounds_are_absolute_not_derived_from_themselves():
    assert judge_mod.JUDGE_MAX_RETRIES <= 5
    assert judge_mod.JUDGE_MAX_BACKOFF_SECONDS <= 60


def test_undecodable_reply_is_a_judge_error_not_a_raw_httpx_error(evaluator, no_sleep):
    def boom(*args, **kwargs):
        raise httpx.DecodingError("bad reply", request=REQ)

    with mock.patch("httpx.post", side_effect=boom) as post:
        with pytest.raises(JudgeUnavailableError, match="DecodingError"):
            evaluator.evaluate("q", "a")
    assert post.call_count == 1  # not transient: no retry
    assert no_sleep == []


@pytest.mark.parametrize(
    "base, secret",
    [
        ("https://user:hunter2@judge.example.com", "hunter2"),
        ("https://judge.example.com/?api_key=SECRET123", "SECRET123"),
    ],
)
def test_error_text_carries_no_credentials(no_sleep, base, secret):
    evaluator = JudgeEvaluator(provider="server", model="judge", api_base=base)
    with mock.patch("httpx.post", side_effect=_refused) as post:
        with pytest.raises(JudgeUnavailableError) as excinfo:
            evaluator.evaluate("q", "a")
    assert secret not in str(excinfo.value)
    assert secret not in excinfo.value.url
    assert "judge.example.com" in str(excinfo.value)
    # the request itself still went to the real URL
    assert secret in post.call_args.args[0]


def test_ipv6_judge_host_is_kept_readable():
    err = JudgeUnavailableError("down", url="http://[::1]:8000/v1/chat/completions")
    assert err.url == "http://[::1]:8000/v1/chat/completions"


def test_unavailable_error_survives_pickle():
    err = JudgeUnavailableError("down", url="http://localhost:8000/v1/chat/completions")
    clone = pickle.loads(pickle.dumps(err))
    assert str(clone) == str(err) and clone.url == err.url


# ---------------------------------------------------------------------------
# soup eval judge: a failing row is skipped and counted, the rest are saved
# ---------------------------------------------------------------------------


def _write_rows(path: Path, n: int = 3) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"prompt": f"q{i}", "response": f"a{i}"}) + "\n")


def _failing_row(prompt_marker: str, failure):
    """A fake ``httpx.post`` that fails only for the row whose prompt has the marker."""

    def post(url, json, headers, timeout):
        if prompt_marker in json["messages"][0]["content"]:
            if callable(failure):
                return failure()
            return failure
        return _reply(200, SCORED)

    return post


class TestEvalJudgeCli:
    @pytest.mark.parametrize(
        "failure",
        [
            _reply(429, {"error": "rate limited"}),
            _timed_out,
            _reply(200, {"choices": []}),
        ],
        ids=["429-after-retries", "read-timeout", "empty-choices"],
    )
    def test_one_failing_row_is_skipped_and_the_rest_scored(
        self, tmp_path, monkeypatch, no_sleep, failure
    ):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_rows(tmp_path / "t.jsonl")
        with mock.patch("httpx.post", side_effect=_failing_row("q1", failure)):
            result = CliRunner().invoke(
                app,
                ["eval", "judge", "--target", "t.jsonl", "--provider", "server"],
            )
        out = strip_ansi(result.output)
        assert result.exit_code == 0, (out, repr(result.exception))
        assert "1/3 items skipped" in out
        assert "Judge Evaluation Results" in out

    def test_failure_text_is_shown_literally_and_control_bytes_are_dropped(
        self, tmp_path, monkeypatch, no_sleep
    ):
        # The per-row warning interpolates the judge's error text; a hostile
        # transport message must neither be read as Rich markup nor reach the
        # terminal as a control sequence.
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_rows(tmp_path / "t.jsonl")

        def hostile():
            raise httpx.ConnectError("[red]boom[/] \x1b[2J gone", request=REQ)

        with mock.patch("httpx.post", side_effect=_failing_row("q1", hostile)):
            result = CliRunner().invoke(
                app,
                ["eval", "judge", "--target", "t.jsonl", "--provider", "server"],
            )
        out = strip_ansi(result.output)
        assert result.exit_code == 0, (out, repr(result.exception))
        assert "[red]boom[/]" in out
        assert "\x1b[2J" not in result.output

    def test_every_row_failing_exits_1_naming_the_judge(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_rows(tmp_path / "t.jsonl")
        with mock.patch("httpx.post", side_effect=_refused):
            result = CliRunner().invoke(
                app,
                ["eval", "judge", "--target", "t.jsonl", "--provider", "server"],
            )
        out = strip_ansi(result.output)
        assert result.exit_code == 1, (out, repr(result.exception))
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert JUDGE_BASE in out
        assert "All items failed" in out


# ---------------------------------------------------------------------------
# soup ship --task-mode pairwise: exit 1 naming the judge, never DON'T SHIP
# ---------------------------------------------------------------------------


class TestShipPairwiseCli:
    def test_dead_judge_is_a_runtime_error_not_a_verdict(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.commands import ship as ship_cmd

        monkeypatch.chdir(tmp_path)
        with (tmp_path / "tasks.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(2):
                fh.write(json.dumps({"prompt": f"task {i}", "expected": "x"}) + "\n")
        monkeypatch.setattr(
            ship_cmd,
            "_resolve_generators",
            lambda *a, **k: (lambda p: "base answer", lambda p: "tuned answer"),
        )
        with mock.patch("httpx.post", side_effect=_refused):
            result = CliRunner().invoke(
                ship_cmd.app,
                [
                    "--base",
                    "m",
                    "--tuned",
                    "t",
                    "--task-eval",
                    "tasks.jsonl",
                    "--task-mode",
                    "pairwise",
                    "--judge-model",
                    f"{JUDGE_BASE}/judge-m",
                ],
            )
        out = strip_ansi(result.output)
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "DON'T SHIP" not in out
        assert JUDGE_BASE in out
        assert "JudgeUnavailableError" in out


# ---------------------------------------------------------------------------
# Eval gate: still fails closed, and says the judge was unavailable
# ---------------------------------------------------------------------------


class TestEvalGate:
    def test_run_gate_records_the_judge_as_unavailable(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text(
            json.dumps({"prompt": "summarise"}) + "\n", encoding="utf-8"
        )
        suite = EvalSuite(
            suite="s",
            tasks=[
                GateTask(
                    type="judge",
                    name="quality",
                    threshold=0.7,
                    prompts="prompts.jsonl",
                    judge_model=f"{JUDGE_BASE}/m",
                )
            ],
        )
        with mock.patch("httpx.post", side_effect=_refused):
            result = run_gate(suite, generate_fn=lambda prompt: "x")
        assert result.passed is False
        (task,) = result.task_results
        assert task.score is None
        assert task.passed is False
        assert "judge unavailable" in task.error.lower()
        assert JUDGE_BASE in task.error

    @pytest.mark.parametrize("policy", ["stop", "warn"])
    def test_callback_log_says_unavailable_not_regressed(self, caplog, policy):
        from soup_cli.eval.gate import EvalSuite, GateResult, GateTaskResult
        from soup_cli.monitoring.callback import SoupTrainerCallback

        cb = SoupTrainerCallback(
            display=MagicMock(),
            tracker=None,
            run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True,
                every_n_epochs=1,
                on_regression=policy,
                regression_threshold=0.05,
                suite="dummy.yaml",
                baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda prompt: "x"
        error = (
            f"judge unavailable at {JUDGE_BASE}/v1/chat/completions: "
            "ConnectError: connection refused (3 attempts)"
        )
        cb._gate_run_fn = lambda suite, generate_fn, baseline, regression_threshold: GateResult(
            passed=False,
            regression=False,
            task_results=[
                GateTaskResult(
                    name="quality",
                    score=None,
                    threshold=0.7,
                    baseline=None,
                    delta=None,
                    passed=False,
                    error=error,
                )
            ],
        )
        control = MagicMock(should_training_stop=False)
        with caplog.at_level(logging.WARNING, logger="soup_cli.monitoring.callback"):
            cb.on_epoch_end(MagicMock(), MagicMock(epoch=1.0, global_step=100), control)
        assert control.should_training_stop is (policy == "stop")
        gate_lines = [
            r.getMessage() for r in caplog.records if "eval gate FAILED" in r.getMessage()
        ]
        assert len(gate_lines) == 1, caplog.text
        assert "regressed" not in gate_lines[0]
        assert "judge unavailable" in gate_lines[0]
        assert "quality" in gate_lines[0]

    def test_callback_log_still_counts_a_real_regression(self, caplog):
        from soup_cli.eval.gate import EvalSuite, GateResult, GateTaskResult
        from soup_cli.monitoring.callback import SoupTrainerCallback

        cb = SoupTrainerCallback(
            display=MagicMock(),
            tracker=None,
            run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True,
                every_n_epochs=1,
                on_regression="stop",
                regression_threshold=0.05,
                suite="dummy.yaml",
                baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda prompt: "x"
        cb._gate_run_fn = lambda suite, generate_fn, baseline, regression_threshold: GateResult(
            passed=False,
            regression=True,
            task_results=[
                GateTaskResult(
                    name="math", score=0.5, threshold=0.8, baseline=0.9, delta=-0.4, passed=False
                )
            ],
        )
        control = MagicMock(should_training_stop=False)
        with caplog.at_level(logging.WARNING, logger="soup_cli.monitoring.callback"):
            cb.on_epoch_end(MagicMock(), MagicMock(epoch=1.0, global_step=100), control)
        assert control.should_training_stop is True
        gate_lines = [
            r.getMessage() for r in caplog.records if "eval gate FAILED" in r.getMessage()
        ]
        assert gate_lines == ["eval gate FAILED (1 task(s) regressed); stopping training"]
