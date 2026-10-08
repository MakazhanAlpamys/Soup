"""Regression tests for #1522: a judge that is down stops the run, not each row.

#1484 made an unreachable judge raise ``JudgeUnavailableError`` after a bounded
retry, but every row retried on its own: 3 s of backoff per row, for the whole
dataset, against a host that never answered once. An evaluator now remembers
that its last requests all went unanswered and stops asking.

``httpx.post`` is patched throughout, so no judge server is needed, and the
retry pause is captured instead of slept.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest import mock

import httpx
import pytest
from typer.testing import CliRunner

from soup_cli.eval import judge as judge_mod
from soup_cli.eval.judge import (
    JUDGE_DOWN_AFTER,
    JudgeDownError,
    JudgeEvaluator,
    JudgeUnavailableError,
    make_judge_reward_func,
    pairwise_winrate,
)
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
ATTEMPTS = judge_mod.JUDGE_MAX_RETRIES + 1
# What one unanswered request sleeps before it gives up: 1 s, then 2 s.
ONE_REQUEST_BACKOFF = [float(2 ** attempt) for attempt in range(judge_mod.JUDGE_MAX_RETRIES)]


def _reply(status: int, body: object) -> httpx.Response:
    return httpx.Response(status, json=body, request=REQ)


def _refused(*args, **kwargs):
    raise httpx.ConnectError("connection refused", request=REQ)


def _script(*steps):
    """A fake ``httpx.post`` that plays ``steps`` in order, then refuses."""
    remaining = list(steps)

    def post(*args, **kwargs):
        step = remaining.pop(0) if remaining else _refused
        return step() if callable(step) else step

    return post


@pytest.fixture
def no_sleep(monkeypatch):
    """Capture the retry pauses instead of sleeping them."""
    delays: list[float] = []
    monkeypatch.setattr(judge_mod, "_sleep", delays.append)
    return delays


@pytest.fixture
def evaluator() -> JudgeEvaluator:
    return JudgeEvaluator(provider="server", model="judge", api_base=JUDGE_BASE)


def _judge_rows(evaluator: JudgeEvaluator, rows: int) -> list[Exception]:
    errors: list[Exception] = []
    for index in range(rows):
        try:
            evaluator.evaluate(f"q{index}", "a")
        except JudgeUnavailableError as exc:
            errors.append(exc)
    return errors


# ---------------------------------------------------------------------------
# The counter
# ---------------------------------------------------------------------------


class TestJudgeDownCounter:
    def test_a_dead_judge_costs_the_first_rows_backoff_and_no_more(self, evaluator, no_sleep):
        # The way every row-by-row caller uses it: a failed row is skipped, and
        # the run stops at the first JudgeDownError.
        skipped = 0
        with mock.patch("httpx.post", side_effect=_refused) as post:
            with pytest.raises(JudgeDownError):
                for index in range(1000):
                    try:
                        evaluator.evaluate(f"q{index}", "a")
                    except JudgeDownError:
                        raise
                    except JudgeUnavailableError:
                        skipped += 1

        assert skipped == JUDGE_DOWN_AFTER - 1
        assert post.call_count == JUDGE_DOWN_AFTER * ATTEMPTS
        assert no_sleep == ONE_REQUEST_BACKOFF * JUDGE_DOWN_AFTER

    def test_the_error_names_the_redacted_url_and_the_run_of_failures(self, no_sleep):
        evaluator = JudgeEvaluator(
            provider="server", model="judge", api_base="https://user:hunter2@judge.example.com"
        )
        with mock.patch("httpx.post", side_effect=_refused):
            errors = _judge_rows(evaluator, JUDGE_DOWN_AFTER)

        down = errors[-1]
        assert isinstance(down, JudgeDownError)
        assert "https://judge.example.com/v1/chat/completions" in str(down)
        assert "hunter2" not in str(down)
        assert f"{JUDGE_DOWN_AFTER} requests in a row" in str(down)
        assert "ConnectError" in str(down)

    def test_one_reply_resets_the_count(self, evaluator, no_sleep):
        unanswered = [_refused] * ATTEMPTS
        steps = [*unanswered, *unanswered, _reply(200, SCORED), *unanswered, *unanswered]
        with mock.patch("httpx.post", side_effect=_script(*steps)) as post:
            errors = []
            scored = 0
            for index in range(5):
                try:
                    evaluator.evaluate(f"q{index}", "a")
                    scored += 1
                except JudgeUnavailableError as exc:
                    errors.append(exc)

        assert scored == 1
        assert len(errors) == 4
        assert not any(isinstance(exc, JudgeDownError) for exc in errors)
        assert post.call_count == len(steps)

    @pytest.mark.parametrize(
        "answer",
        [_reply(400, {"error": "prompt too long"}), _reply(200, {"choices": []})],
        ids=["4xx", "unusable-200"],
    )
    def test_a_judge_that_answers_badly_is_not_down(self, evaluator, no_sleep, answer):
        with mock.patch("httpx.post", return_value=answer) as post:
            errors = _judge_rows(evaluator, 10)

        assert len(errors) == 10
        assert not any(isinstance(exc, JudgeDownError) for exc in errors)
        assert post.call_count == 10
        assert no_sleep == []

    def test_a_bad_answer_between_outages_resets_the_count(self, evaluator, no_sleep):
        unanswered = [_refused] * ATTEMPTS
        steps = [*unanswered, *unanswered, _reply(400, {"error": "no"}), *unanswered]
        with mock.patch("httpx.post", side_effect=_script(*steps)):
            errors = _judge_rows(evaluator, 4)

        assert len(errors) == 4
        assert not any(isinstance(exc, JudgeDownError) for exc in errors)

    def test_pairwise_and_pointwise_calls_share_the_count(self, evaluator, no_sleep):
        with mock.patch("httpx.post", side_effect=_refused):
            for _ in range(JUDGE_DOWN_AFTER - 1):
                with pytest.raises(JudgeUnavailableError) as excinfo:
                    evaluator.evaluate("q", "a")
                assert not isinstance(excinfo.value, JudgeDownError)
            with pytest.raises(JudgeDownError):
                evaluator.compare_pair("q", "a", "b")

    def test_each_evaluator_counts_its_own_judge(self, no_sleep):
        dead = JudgeEvaluator(provider="server", model="judge", api_base=JUDGE_BASE)
        other = JudgeEvaluator(provider="server", model="judge", api_base="http://localhost:9000")
        with mock.patch("httpx.post", side_effect=_refused):
            _judge_rows(dead, JUDGE_DOWN_AFTER)
        with mock.patch("httpx.post", return_value=_reply(200, SCORED)):
            assert other.evaluate("q", "a").weighted_score == 4.0

    def test_a_judge_that_stays_down_is_still_asked(self, evaluator, no_sleep):
        # No cooldown: a caller that carries on gets a real request each time,
        # so the first reply after an outage is seen.
        with mock.patch("httpx.post", side_effect=_refused) as post:
            errors = _judge_rows(evaluator, JUDGE_DOWN_AFTER + 2)

        assert post.call_count == (JUDGE_DOWN_AFTER + 2) * ATTEMPTS
        assert all(isinstance(exc, JudgeDownError) for exc in errors[JUDGE_DOWN_AFTER - 1:])
        assert f"{JUDGE_DOWN_AFTER + 2} requests in a row" in str(errors[-1])

    def test_down_error_is_an_unavailable_error_and_survives_pickle(self):
        err = JudgeDownError("3 requests in a row failed", url=f"{JUDGE_BASE}/v1/chat/completions")
        told = err.for_rows(7, 10, "pairs")
        assert isinstance(told, JudgeDownError) and isinstance(told, JudgeUnavailableError)
        assert "7 of 10 pairs not judged" in str(told)
        clone = pickle.loads(pickle.dumps(told))
        assert type(clone) is JudgeDownError
        assert str(clone) == str(told) and clone.url == told.url


# ---------------------------------------------------------------------------
# Online DPO: a failure is a score, so a short outage must not outlast itself
# ---------------------------------------------------------------------------


def test_a_short_outage_does_not_leave_a_judge_ranked_run_unranked(evaluator, no_sleep):
    ok = _reply(200, SCORED)
    # three requests in a row are refused, then the judge answers every request
    steps = [*[_refused] * (JUDGE_DOWN_AFTER * ATTEMPTS), *[ok] * 20]
    reward = make_judge_reward_func(evaluator)
    with mock.patch("httpx.post", side_effect=_script(*steps)):
        scores = reward(["q"] * 10, ["a"] * 10)
    assert scores[:3] == [0.0, 0.0, 0.0]
    assert scores[3:] == [4.0] * 7, scores


# ---------------------------------------------------------------------------
# soup eval judge
# ---------------------------------------------------------------------------


def _write_rows(path: Path, n: int) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"prompt": f"q{i}", "response": f"a{i}"}) + "\n")


class TestEvalJudgeCli:
    def test_a_dead_judge_stops_the_run_with_one_message(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_rows(tmp_path / "t.jsonl", 1000)
        with mock.patch("httpx.post", side_effect=_refused) as post:
            result = CliRunner().invoke(
                app, ["eval", "judge", "--target", "t.jsonl", "--provider", "server"],
            )
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, (out, repr(result.exception))
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert post.call_count == JUDGE_DOWN_AFTER * ATTEMPTS
        assert no_sleep == ONE_REQUEST_BACKOFF * JUDGE_DOWN_AFTER
        assert JUDGE_BASE in out
        assert "1000 of 1000 items not judged" in out
        # The rows before the stop still warn; the 997 after it print nothing.
        assert out.count("Warning: judge failed") == JUDGE_DOWN_AFTER - 1

    def test_scores_from_before_the_outage_are_shown_and_the_run_still_fails(
        self, tmp_path, monkeypatch, no_sleep
    ):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_rows(tmp_path / "t.jsonl", 50)
        steps = [_reply(200, SCORED)] * 5
        with mock.patch("httpx.post", side_effect=_script(*steps)):
            result = CliRunner().invoke(
                app, ["eval", "judge", "--target", "t.jsonl", "--provider", "server"],
            )
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "45 of 50 items not judged" in out
        assert "Judge Evaluation Results" in out


# ---------------------------------------------------------------------------
# soup data from-traces --judge (the pair filter)
# ---------------------------------------------------------------------------


class TestPairFilter:
    def _pairs(self, n: int):
        from soup_cli.data.traces.pair_builder import PreferencePair

        return [
            PreferencePair(prompt=f"q{i}", chosen="good", rejected="bad", source="t")
            for i in range(n)
        ]

    def test_a_dead_judge_stops_the_filter(self, evaluator, no_sleep):
        from soup_cli.data.traces.quality import judge_filter_pairs

        with mock.patch("httpx.post", side_effect=_refused) as post:
            with pytest.raises(JudgeDownError) as excinfo:
                judge_filter_pairs(self._pairs(1000), judge=evaluator)

        assert post.call_count == JUDGE_DOWN_AFTER * ATTEMPTS
        assert no_sleep == ONE_REQUEST_BACKOFF * JUDGE_DOWN_AFTER
        assert JUDGE_BASE in str(excinfo.value)
        assert "1000 of 1000 pairs not judged" in str(excinfo.value)

    def test_the_count_leaves_out_the_pairs_already_judged(self, evaluator, no_sleep):
        from soup_cli.data.traces.quality import judge_filter_pairs

        def scored(value):
            content = json.dumps({
                "scores": {"helpfulness": value, "accuracy": value, "safety": value},
                "reasoning": "ok",
            })
            return _reply(200, {"choices": [{"message": {"content": content}}]})

        # Pair 1 is kept (chosen 5, rejected 1), pairs 2-4 are dropped (a tie
        # is below the 0.7 default), then nothing answers.
        steps = [scored(5), scored(1)] + [_reply(200, SCORED)] * 6
        with mock.patch("httpx.post", side_effect=_script(*steps)):
            with pytest.raises(JudgeDownError) as excinfo:
                judge_filter_pairs(self._pairs(10), judge=evaluator)

        assert "6 of 10 pairs not judged" in str(excinfo.value)

    def test_one_failing_pair_is_still_only_counted(self, evaluator, no_sleep):
        from soup_cli.data.traces.quality import judge_filter_pairs

        steps = [*[_refused] * ATTEMPTS, *[_reply(200, SCORED)] * 4]
        with mock.patch("httpx.post", side_effect=_script(*steps)):
            kept, report = judge_filter_pairs(self._pairs(3), judge=evaluator)

        assert report.errors == 1
        assert report.kept + report.dropped == 2

    def test_from_traces_exits_1_naming_the_judge(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        with (tmp_path / "traces.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(20):
                for text, score in (("good", 1), ("bad", 0)):
                    fh.write(json.dumps({
                        "inputs": {"input": f"q{i}"},
                        "outputs": {"output": text},
                        "feedback": [{"key": "thumbs", "score": score}],
                    }) + "\n")
        with mock.patch("httpx.post", side_effect=_refused) as post:
            result = CliRunner().invoke(
                app,
                ["data", "from-traces", "--logs", "traces.jsonl", "--format", "langchain",
                 "--signal", "thumbs_up", "--output", "prefs.jsonl", "--judge",
                 "--judge-provider", "server", "--judge-api-base", JUDGE_BASE],
            )
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, (out, repr(result.exception))
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert post.call_count == JUDGE_DOWN_AFTER * ATTEMPTS
        assert JUDGE_BASE in out
        assert "20 of 20 pairs not judged" in out
        assert not (tmp_path / "prefs.jsonl").exists()


# ---------------------------------------------------------------------------
# soup ship --task-mode pairwise, and best-of-n
# ---------------------------------------------------------------------------


class TestPairwiseWinrate:
    def test_the_error_says_how_many_pairs_were_not_judged(self, evaluator, no_sleep):
        pairs = [(f"p{i}", "base answer", "tuned answer") for i in range(8)]
        # Three pairs judged (two calls each, tuned wins), then nothing answers.
        steps = []
        for _ in range(3):
            steps += [
                _reply(200, {"choices": [{"message": {"content": '{"winner": "B"}'}}]}),
                _reply(200, {"choices": [{"message": {"content": '{"winner": "A"}'}}]}),
            ]
        with mock.patch("httpx.post", side_effect=_script(*steps)):
            with pytest.raises(JudgeUnavailableError) as excinfo:
                pairwise_winrate(pairs, evaluator)

        assert JUDGE_BASE in str(excinfo.value)
        assert "5 of 8 pairs not judged" in str(excinfo.value)

    def test_ship_names_the_judge_and_the_pairs_lost(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.commands import ship as ship_cmd

        monkeypatch.chdir(tmp_path)
        with (tmp_path / "tasks.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(4):
                fh.write(json.dumps({"prompt": f"task {i}", "expected": "x"}) + "\n")
        monkeypatch.setattr(
            ship_cmd, "_resolve_generators",
            lambda *a, **k: (lambda p: "base answer", lambda p: "tuned answer"),
        )
        with mock.patch("httpx.post", side_effect=_refused):
            result = CliRunner().invoke(
                ship_cmd.app,
                ["--base", "m", "--tuned", "t", "--task-eval", "tasks.jsonl",
                 "--task-mode", "pairwise", "--judge-model", f"{JUDGE_BASE}/judge-m"],
            )
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "DON'T SHIP" not in out
        assert JUDGE_BASE in out
        assert "4 of 4 pairs not judged" in out


class TestBestOfNCli:
    def test_the_stop_names_the_judge_and_the_prompts_lost(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.commands.data import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text(
            "".join(json.dumps({"prompt": f"prompt {i}"}) + "\n" for i in range(4)),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "soup_cli.utils.magpie.make_magpie_generate_fn",
            lambda *a, **k: (lambda prompt: "a candidate"),
        )
        with mock.patch("httpx.post", side_effect=_refused):
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--provider", "ollama", "--model", "sampler",
                 "--prompts", "prompts.jsonl", "--n", "3", "--judge", "ollama://judge",
                 "--output", "sft.jsonl"],
            )
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "http://localhost:11434/v1/chat/completions" in out
        assert "4 of 4 prompts not judged" in out
        assert "--resume" in out

    def test_the_count_leaves_out_the_prompts_already_done(self, tmp_path, monkeypatch, no_sleep):
        from soup_cli.commands.data import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text(
            "".join(json.dumps({"prompt": f"prompt {i}"}) + "\n" for i in range(4)),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "soup_cli.utils.magpie.make_magpie_generate_fn",
            lambda *a, **k: (lambda prompt: "a candidate"),
        )
        # The first prompt's three candidates are scored, then nothing answers.
        steps = [_reply(200, SCORED)] * 3
        with mock.patch("httpx.post", side_effect=_script(*steps)):
            result = CliRunner().invoke(
                app,
                ["best-of-n", "--provider", "ollama", "--model", "sampler",
                 "--prompts", "prompts.jsonl", "--n", "3", "--judge", "ollama://judge",
                 "--output", "sft.jsonl"],
            )
        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, (out, repr(result.exception))
        assert "3 of 4 prompts not judged" in out
        assert "stopped after 1/4 prompts" in out
