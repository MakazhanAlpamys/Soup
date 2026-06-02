"""Tests for v0.71.5 — Ingest / data / prompt / drift patch.

Closes (6 of 7): #164, #163, #207, #205, #149, #157.
Deferred: #204 (live SaaS pull adapters — external-account-gated, infra-blocked).

One test file per release (project convention). Grouped by issue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ===========================================================================
# #164 — extend get_metric_series to the eval_results table
# ===========================================================================


class TestGetMetricSeriesEvalResults:
    """`tracker.get_metric_series` falls back to eval_results for benchmarks."""

    def _tracker(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        return ExperimentTracker(db_path=tmp_path / "exp.db")

    def test_eval_results_series_returned(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.save_eval_result("m1", "task_accuracy", 0.81, {}, run_id="run-1")
        tracker.save_eval_result("m1", "task_accuracy", 0.79, {}, run_id="run-1")
        series = tracker.get_metric_series("run-1", "task_accuracy")
        assert series == [0.81, 0.79]

    def test_eval_results_filtered_by_benchmark(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.save_eval_result("m1", "task_accuracy", 0.81, {}, run_id="run-1")
        tracker.save_eval_result("m1", "refusal_rate", 0.02, {}, run_id="run-1")
        assert tracker.get_metric_series("run-1", "refusal_rate") == [0.02]

    def test_eval_results_filtered_by_run_id(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.save_eval_result("m1", "task_accuracy", 0.81, {}, run_id="run-1")
        tracker.save_eval_result("m2", "task_accuracy", 0.55, {}, run_id="run-2")
        assert tracker.get_metric_series("run-1", "task_accuracy") == [0.81]
        assert tracker.get_metric_series("run-2", "task_accuracy") == [0.55]

    def test_per_step_column_still_uses_metrics_table(self, tmp_path):
        # No regression: known training-loop columns read from `metrics`.
        tracker = self._tracker(tmp_path)
        tracker.log_metrics("run-1", step=1, loss=2.0)
        tracker.log_metrics("run-1", step=2, loss=1.5)
        # Also write a (different) eval result to prove we DON'T read it for loss.
        tracker.save_eval_result("m1", "loss", 99.0, {}, run_id="run-1")
        assert tracker.get_metric_series("run-1", "loss") == [2.0, 1.5]

    def test_missing_metric_returns_empty_list(self, tmp_path):
        tracker = self._tracker(tmp_path)
        assert tracker.get_metric_series("nope", "task_accuracy") == []

    def test_unknown_benchmark_returns_empty(self, tmp_path):
        tracker = self._tracker(tmp_path)
        tracker.save_eval_result("m1", "task_accuracy", 0.81, {}, run_id="run-1")
        assert tracker.get_metric_series("run-1", "bleu") == []

    def test_eval_series_ordered_by_id(self, tmp_path):
        # Deterministic pairing for the paired-bootstrap CI: the series is
        # ordered by insertion id, independent of the created_at timestamp.
        # (The `value is None` / non-numeric branches in _eval_score_series
        # are defensive-only — `eval_results.score` is REAL NOT NULL, so a
        # NULL / non-numeric cell cannot be inserted; no test can construct
        # one without violating the schema.)
        tracker = self._tracker(tmp_path)
        conn = tracker._get_conn()
        for score, ts in ((0.1, "2026-01-09"), (0.2, "2026-01-01"), (0.3, "2026-01-05")):
            conn.execute(
                "INSERT INTO eval_results (run_id, model_path, benchmark, score, "
                "details_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("run-1", "m", "task_accuracy", score, "{}", ts),
            )
        conn.commit()
        assert tracker.get_metric_series("run-1", "task_accuracy") == [0.1, 0.2, 0.3]

    def test_empty_run_id_rejected(self, tmp_path):
        tracker = self._tracker(tmp_path)
        with pytest.raises(ValueError, match="non-empty string"):
            tracker.get_metric_series("", "task_accuracy")

    def test_empty_metric_rejected(self, tmp_path):
        tracker = self._tracker(tmp_path)
        with pytest.raises(ValueError, match="non-empty string"):
            tracker.get_metric_series("run-1", "")

    @pytest.mark.parametrize("bad", [True, 123, None])
    def test_non_string_run_id_rejected(self, tmp_path, bad):
        tracker = self._tracker(tmp_path)
        with pytest.raises(ValueError):
            tracker.get_metric_series(bad, "task_accuracy")

    @pytest.mark.parametrize("bad", [True, 123, None])
    def test_non_string_metric_rejected(self, tmp_path, bad):
        tracker = self._tracker(tmp_path)
        with pytest.raises(ValueError):
            tracker.get_metric_series("run-1", bad)


# ===========================================================================
# #163 — bias build_verdict from advise_history outcomes
# ===========================================================================


class _FakeHistoryEntry:
    """Duck-typed HistoryEntry stand-in for build_verdict bias tests."""

    def __init__(self, *, choice, task_category="factual_lookup", accepted=True,
                 outcome=0.5, project="proj-a"):
        self.choice = choice
        self.task_category = task_category
        self.accepted = accepted
        self.outcome = outcome
        self.project = project


class TestBuildVerdictHistoryBias:
    def _profile(self, **over):
        from soup_cli.utils.advise import DatasetProfile

        base = dict(
            row_count=2000,
            avg_input_chars=120.0,
            avg_output_chars=80.0,
            type_token_diversity=0.6,
            label_variance=0.7,
            has_chosen_rejected=False,
            has_reasoning_traces=False,
        )
        base.update(over)
        return DatasetProfile(**base)

    def test_summarise_empty_and_none(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        assert _summarise_history_outcomes(None) == {}
        assert _summarise_history_outcomes([]) == {}

    def test_summarise_filters_by_project(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        hist = [
            _FakeHistoryEntry(choice="SFT", outcome=0.5, project="proj-a"),
            _FakeHistoryEntry(choice="SFT", outcome=0.5, project="proj-b"),
        ]
        out = _summarise_history_outcomes(hist, project="proj-a")
        assert out["SFT"] == (0.5, 1)  # only proj-a counted

    def test_summarise_skips_unaccepted_and_none_outcome(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        hist = [
            _FakeHistoryEntry(choice="SFT", outcome=0.5, accepted=False),
            _FakeHistoryEntry(choice="SFT", outcome=None, accepted=True),
            _FakeHistoryEntry(choice="SFT", outcome=0.4, accepted=True),
        ]
        out = _summarise_history_outcomes(hist)
        assert out["SFT"] == (0.4, 1)

    def test_summarise_skips_bool_outcome(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        hist = [_FakeHistoryEntry(choice="SFT", outcome=True)]
        assert _summarise_history_outcomes(hist) == {}

    def test_summarise_non_sequence_rejected(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        with pytest.raises(TypeError):
            _summarise_history_outcomes(123)  # type: ignore[arg-type]

    def test_no_history_identical_to_base(self):
        from soup_cli.utils.advise import build_verdict

        profile = self._profile()  # factual_lookup + high variance → RAG
        base = build_verdict(profile, "factual_lookup")
        with_none = build_verdict(profile, "factual_lookup", history=None)
        assert base.choice == with_none.choice == "RAG"
        assert base.confidence == with_none.confidence

    def test_sft_precedents_flip_marginal_rag_to_sft(self):
        from soup_cli.utils.advise import build_verdict

        profile = self._profile()  # would be RAG by default
        hist = [
            _FakeHistoryEntry(choice="SFT", outcome=0.4, project="proj-a")
            for _ in range(3)
        ]
        verdict = build_verdict(
            profile, "factual_lookup", history=hist, project="proj-a"
        )
        assert verdict.choice == "SFT"
        assert "precedent" in verdict.reason.lower()

    def test_sft_precedents_other_project_do_not_flip(self):
        from soup_cli.utils.advise import build_verdict

        profile = self._profile()
        hist = [
            _FakeHistoryEntry(choice="SFT", outcome=0.4, project="other")
            for _ in range(3)
        ]
        verdict = build_verdict(
            profile, "factual_lookup", history=hist, project="proj-a"
        )
        assert verdict.choice == "RAG"  # unchanged — wrong project

    def test_fewer_than_three_sft_precedents_no_flip(self):
        from soup_cli.utils.advise import build_verdict

        profile = self._profile()
        hist = [
            _FakeHistoryEntry(choice="SFT", outcome=0.4, project="proj-a")
            for _ in range(2)
        ]
        verdict = build_verdict(
            profile, "factual_lookup", history=hist, project="proj-a"
        )
        assert verdict.choice == "RAG"

    def test_negative_grpo_precedents_suppress_grpo(self):
        from soup_cli.utils.advise import build_verdict

        profile = self._profile(
            row_count=800, has_reasoning_traces=True, label_variance=0.3,
        )
        # Default (no history) → GRPO.
        assert build_verdict(profile, "reasoning").choice == "GRPO"
        hist = [
            _FakeHistoryEntry(choice="GRPO", outcome=-0.2, project="proj-a")
            for _ in range(3)
        ]
        verdict = build_verdict(
            profile, "reasoning", history=hist, project="proj-a"
        )
        assert verdict.choice == "SFT"
        assert "precedent" in verdict.reason.lower()

    def test_encouraged_choice_nudges_confidence(self):
        from soup_cli.utils.advise import build_verdict

        # SFT task (not factual_lookup) so base is SFT; encourage SFT.
        profile = self._profile(label_variance=0.3)
        base = build_verdict(profile, "style_shaping")
        assert base.choice == "SFT"
        hist = [
            _FakeHistoryEntry(choice="SFT", outcome=0.5, project="proj-a")
            for _ in range(3)
        ]
        biased = build_verdict(
            profile, "style_shaping", history=hist, project="proj-a"
        )
        assert biased.choice == "SFT"
        # Exact nudge (+0.05, clamped at 0.95) — a zero-nudge bug would fail.
        assert biased.confidence == pytest.approx(min(0.95, base.confidence + 0.05))

    def test_summarise_outcome_out_of_range_skipped(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        hist = [_FakeHistoryEntry(choice="SFT", outcome=2.0)]
        assert _summarise_history_outcomes(hist) == {}

    def test_summarise_accepted_must_be_true_not_truthy(self):
        from soup_cli.utils.advise import _summarise_history_outcomes

        # accepted=1 (int) is NOT True → rejected (the guard is `is not True`).
        hist = [_FakeHistoryEntry(choice="SFT", accepted=1, outcome=0.5)]
        assert _summarise_history_outcomes(hist) == {}

    def test_is_encouraged_threshold_boundaries(self):
        from soup_cli.utils.advise import _is_encouraged

        # >= 0.3 over >= 3 precedents.
        assert _is_encouraged({"SFT": (0.3, 3)}, "SFT") is True
        assert _is_encouraged({"SFT": (0.29, 3)}, "SFT") is False
        assert _is_encouraged({"SFT": (0.5, 2)}, "SFT") is False  # count < 3
        assert _is_encouraged({}, "SFT") is False

    def test_is_discouraged_threshold_boundaries(self):
        from soup_cli.utils.advise import _is_discouraged

        # strict < 0.0 over >= 3 precedents.
        assert _is_discouraged({"GRPO": (0.0, 3)}, "GRPO") is False
        assert _is_discouraged({"GRPO": (-0.01, 3)}, "GRPO") is True
        assert _is_discouraged({"GRPO": (-0.5, 2)}, "GRPO") is False  # count < 3

    def test_real_history_entry_duck_types(self):
        from soup_cli.utils.advise import _summarise_history_outcomes
        from soup_cli.utils.advise_history import HistoryEntry

        entry = HistoryEntry(
            timestamp="2026-01-01T00:00:00+00:00",
            project="proj-a",
            choice="SFT",
            task_category="style_shaping",
            confidence=0.8,
            reason="r",
            reverse_when="w",
            accepted=True,
            outcome=0.5,
            notes="",
        )
        out = _summarise_history_outcomes([entry], project="proj-a")
        assert out["SFT"] == (0.5, 1)


# ===========================================================================
# #207 — shared utils/webhooks.py + --slack-url/--discord-url on 4 commands
# ===========================================================================


class TestSharedWebhooks:
    def test_webhooks_module_exports(self):
        from soup_cli.utils import webhooks

        assert hasattr(webhooks, "validate_webhook_url")
        assert hasattr(webhooks, "post_webhook")
        assert hasattr(webhooks, "send_webhooks")

    def test_drift_alarm_reexports_same_objects(self):
        from soup_cli.utils import drift_alarm, webhooks

        assert drift_alarm.validate_webhook_url is webhooks.validate_webhook_url
        assert drift_alarm.post_webhook is webhooks.post_webhook

    def test_validate_webhook_url_https_ok(self):
        from soup_cli.utils.webhooks import validate_webhook_url

        assert validate_webhook_url("https://hooks.slack.com/x") is not None

    def test_validate_webhook_url_rejects_rfc1918(self):
        from soup_cli.utils.webhooks import validate_webhook_url

        with pytest.raises(ValueError):
            validate_webhook_url("http://10.0.0.5/hook")

    def test_validate_webhook_url_rejects_loopback_only_http(self):
        from soup_cli.utils.webhooks import validate_webhook_url

        assert validate_webhook_url("http://127.0.0.1:9000/h") is not None
        with pytest.raises(ValueError):
            validate_webhook_url("http://example.com/h")  # remote http

    @pytest.mark.parametrize("bad", [True, 123, None])
    def test_validate_webhook_url_type_rejection(self, bad):
        from soup_cli.utils.webhooks import validate_webhook_url

        with pytest.raises(TypeError):
            validate_webhook_url(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "ftp://example.com/h",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "http://0.0.0.0/h",
            "http://169.254.169.254/latest",  # link-local cloud metadata
            "https://host/h\x00",  # null byte
            "https://host/h\nX",  # control char
            "https://" + "a" * 5000,  # oversize
        ],
    )
    def test_validate_webhook_url_value_rejection_matrix(self, bad):
        # Re-prove the SSRF gate against the NEW module path (the validator
        # moved out of drift_alarm in #207 — don't rely on the re-export).
        from soup_cli.utils.webhooks import validate_webhook_url

        with pytest.raises(ValueError):
            validate_webhook_url(bad)

    def test_send_webhooks_posts_both(self, monkeypatch):
        from soup_cli.utils import webhooks

        calls = []
        monkeypatch.setattr(
            webhooks, "post_webhook",
            lambda **kw: (calls.append(kw), True)[1],
        )
        results = webhooks.send_webhooks(
            {"k": 1},
            slack_url="https://hooks.slack.com/x",
            discord_url="https://discord.com/api/webhooks/x",
        )
        assert results == [("slack", True), ("discord", True)]
        assert len(calls) == 2
        assert calls[0]["payload"] == {"k": 1}

    def test_send_webhooks_skips_none(self, monkeypatch):
        from soup_cli.utils import webhooks

        calls = []
        monkeypatch.setattr(
            webhooks, "post_webhook",
            lambda **kw: (calls.append(kw), True)[1],
        )
        results = webhooks.send_webhooks({"k": 1}, slack_url=None, discord_url=None)
        assert results == []
        assert calls == []

    def test_send_webhooks_swallows_failure(self, monkeypatch):
        from soup_cli.utils import webhooks

        monkeypatch.setattr(webhooks, "post_webhook", lambda **kw: False)
        results = webhooks.send_webhooks(
            {"k": 1}, slack_url="https://hooks.slack.com/x"
        )
        assert results == [("slack", False)]

    # --- CLI flag plumbing on each of the 4 commands ----------------------

    @pytest.mark.parametrize(
        "argv",
        [
            ["ingest", "--help"],
            ["prune-prompt", "--help"],
            ["ab", "--help"],
            ["data", "active-sample", "--help"],
        ],
    )
    def test_webhook_flags_in_help(self, argv):
        import re as _re

        from soup_cli.cli import app

        result = runner.invoke(app, argv)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        clean = _re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        clean = clean.replace("\n", " ")
        clean = _re.sub(r"\s+", " ", clean)
        assert "--slack-url" in clean
        assert "--discord-url" in clean

    def test_ingest_rejects_bad_webhook(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs.jsonl").write_text(
            '{"input": "hi", "output": "yo"}\n', encoding="utf-8"
        )
        result = runner.invoke(
            app,
            ["ingest", "--source", "langfuse", "--logs", "logs.jsonl",
             "--slack-url", "http://10.0.0.1/h"],
        )
        assert result.exit_code == 2

    def test_bad_discord_url_labels_correct_flag(self, tmp_path, monkeypatch):
        # The second loop iteration in validate_webhook_flags must label
        # --discord-url (guards a copy-paste bug printing --slack-url twice).
        import re as _re

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs.jsonl").write_text(
            '{"input": "hi", "output": "yo"}\n', encoding="utf-8"
        )
        result = runner.invoke(
            app,
            ["ingest", "--source", "langfuse", "--logs", "logs.jsonl",
             "--discord-url", "http://10.0.0.1/h"],
        )
        assert result.exit_code == 2
        clean = _re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")
        assert "--discord-url" in clean

    def test_emit_webhooks_early_return_no_urls(self):
        # Direct unit test of the CLI helper (only exercised via 4 CLIs).
        from rich.console import Console

        from soup_cli.commands._webhook_cli import emit_webhooks

        # Both None → no-op, no exception.
        emit_webhooks(None, None, payload={"k": 1}, console=Console())

    def test_ingest_posts_payload_on_success(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import webhooks

        captured = []
        monkeypatch.setattr(
            webhooks, "post_webhook",
            lambda **kw: (captured.append(kw), True)[1],
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "logs.jsonl").write_text(
            '{"input": "hi", "output": "yo"}\n', encoding="utf-8"
        )
        result = runner.invoke(
            app,
            ["ingest", "--source", "langfuse", "--logs", "logs.jsonl",
             "--slack-url", "https://hooks.slack.com/x"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert payload["source"] == "langfuse"
        assert payload["traces_written"] == 1
        assert "auth_env_set" in payload

    def test_prune_prompt_posts_payload(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import webhooks

        captured = []
        monkeypatch.setattr(
            webhooks, "post_webhook",
            lambda **kw: (captured.append(kw), True)[1],
        )
        monkeypatch.chdir(tmp_path)
        rows = "".join(
            '{"prompt": "SYS PREAMBLE. ask %d", "output": "a"}\n' % i
            for i in range(5)
        )
        (tmp_path / "in.jsonl").write_text(rows, encoding="utf-8")
        result = runner.invoke(
            app,
            ["prune-prompt", "--input", "in.jsonl", "--output", "out.jsonl",
             "--min-frequency", "0.9",
             "--discord-url", "https://discord.com/api/webhooks/x"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert "rows_pruned" in payload
        assert "prefix_chars" in payload

    def test_ab_posts_only_on_decision(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import webhooks

        captured = []
        monkeypatch.setattr(
            webhooks, "post_webhook",
            lambda **kw: (captured.append(kw), True)[1],
        )
        monkeypatch.chdir(tmp_path)
        # Two-row "continue" case → no webhook fired.
        (tmp_path / "ab.jsonl").write_text(
            '{"arm": "control", "latency": 1.0}\n'
            '{"arm": "treatment", "latency": 1.0}\n',
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["ab", "--input", "ab.jsonl", "--metric", "latency",
             "--slack-url", "https://hooks.slack.com/x"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # decision == continue → no webhook
        assert captured == []

    def test_active_sample_posts_payload(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import webhooks

        captured = []
        monkeypatch.setattr(
            webhooks, "post_webhook",
            lambda **kw: (captured.append(kw), True)[1],
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "traces.jsonl").write_text(
            '{"prompt": "a", "rm_score": 0.5}\n'
            '{"prompt": "b", "rm_score": 0.9}\n',
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["data", "active-sample", "--input", "traces.jsonl",
             "--output", "sel.jsonl", "--budget", "1",
             "--slack-url", "https://hooks.slack.com/x"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert "rows_selected" in payload
        assert "mean_uncertainty" in payload


# ===========================================================================
# #205 — tokenizer-aware prefix detection for prune-prompt
# ===========================================================================


class _FakeWordTokenizer:
    """Whitespace tokenizer with a stable str->id vocab (offline test seam).

    Mimics the `transformers` tokenizer surface: ``encode(text,
    add_special_tokens=...)`` -> list[int]; ``decode(ids)`` -> str.
    """

    def __init__(self):
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._stoi:
            idx = len(self._stoi)
            self._stoi[tok] = idx
            self._itos[idx] = tok
        return self._stoi[tok]

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return [self._id(t) for t in text.split(" ") if t != ""]

    def decode(self, ids):
        return " ".join(self._itos[i] for i in ids)


class TestPrunePromptTokenizer:
    def test_detect_common_prefix_tokens_happy(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        rows = [
            [1, 2, 3, 4],
            [1, 2, 3, 9],
            [1, 2, 3, 7],
        ]
        assert detect_common_prefix_tokens(rows, min_frequency=1.0) == [1, 2, 3]

    def test_detect_common_prefix_tokens_partial_majority(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        rows = [
            [1, 2, 3],
            [1, 2, 3],
            [9, 9, 9],
        ]
        # 2/3 share [1,2,3] → at 0.6 threshold, returns it.
        assert detect_common_prefix_tokens(rows, min_frequency=0.6) == [1, 2, 3]
        # at 0.9 threshold, only [9..]? no shared prefix across all → []
        assert detect_common_prefix_tokens(rows, min_frequency=0.9) == []

    def test_detect_common_prefix_tokens_empty(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        assert detect_common_prefix_tokens([], min_frequency=1.0) == []

    def test_detect_common_prefix_tokens_single_row(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        assert detect_common_prefix_tokens([[1, 2]], min_frequency=1.0) == [1, 2]
        assert detect_common_prefix_tokens([[1, 2]], min_frequency=0.5) == []

    def test_detect_common_prefix_tokens_invalid_min_frequency(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        with pytest.raises(ValueError):
            detect_common_prefix_tokens([[1]], min_frequency=1.5)

    def test_detect_common_prefix_tokens_min_freq_lower_and_nan(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        with pytest.raises(ValueError):
            detect_common_prefix_tokens([[1]], min_frequency=-0.1)
        with pytest.raises(ValueError):
            detect_common_prefix_tokens([[1]], min_frequency=float("nan"))

    def test_detect_common_prefix_tokens_non_iterable_rejected(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        with pytest.raises(TypeError):
            detect_common_prefix_tokens(123, min_frequency=1.0)  # type: ignore[arg-type]

    def test_detect_common_prefix_tokens_non_iterable_row_rejected(self):
        from soup_cli.utils.prune_prompt import detect_common_prefix_tokens

        with pytest.raises(TypeError):
            detect_common_prefix_tokens([123, [1, 2]], min_frequency=1.0)  # type: ignore[list-item]

    def test_prune_traces_with_fake_tokenizer(self, tmp_path, monkeypatch):
        from soup_cli.utils.prune_prompt import prune_traces

        monkeypatch.chdir(tmp_path)
        # The varying token (ask0/ask1/...) carries no leading space so the
        # shared prefix is exactly the 3-token preamble.
        rows = "".join(
            '{"prompt": "SYS PREAMBLE HERE ask%d", "output": "a"}\n' % i
            for i in range(5)
        )
        (tmp_path / "in.jsonl").write_text(rows, encoding="utf-8")
        report = prune_traces(
            "in.jsonl",
            output_path="out.jsonl",
            min_frequency=0.9,
            tokenizer=_FakeWordTokenizer(),
        )
        assert report.rows_pruned == 5
        assert report.prefix == "SYS PREAMBLE HERE"
        # Output prompts no longer carry the preamble tokens.
        import json as _json

        lines = [
            _json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert all(not row["prompt"].startswith("SYS PREAMBLE") for row in lines)
        assert lines[0]["prompt"] == "ask0"

    def test_prune_traces_tokenizer_multibyte_safe(self, tmp_path, monkeypatch):
        # A shared prefix containing a multi-byte token is stripped on a
        # token boundary — never mid-code-point.
        from soup_cli.utils.prune_prompt import prune_traces

        monkeypatch.chdir(tmp_path)
        rows = "".join(
            '{"prompt": "café ☕ menu item %d", "output": "x"}\n' % i
            for i in range(4)
        )
        (tmp_path / "in.jsonl").write_text(rows, encoding="utf-8")
        report = prune_traces(
            "in.jsonl",
            output_path="out.jsonl",
            min_frequency=0.9,
            tokenizer=_FakeWordTokenizer(),
        )
        assert "café" in report.prefix and "☕" in report.prefix
        assert report.rows_pruned == 4

    def test_prune_traces_string_tokenizer_lazy_loads(self, tmp_path, monkeypatch):
        # `tokenizer` as a string → lazy AutoTokenizer.from_pretrained.
        import types

        from soup_cli.utils import prune_prompt as pp

        monkeypatch.chdir(tmp_path)
        loaded = {}

        def _fake_from_pretrained(name, **kw):  # noqa: ARG001
            loaded["name"] = name
            return _FakeWordTokenizer()

        fake_auto = types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
        fake_transformers = types.SimpleNamespace(AutoTokenizer=fake_auto)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        rows = "".join(
            '{"prompt": "SYS ask %d", "output": "a"}\n' % i for i in range(3)
        )
        (tmp_path / "in.jsonl").write_text(rows, encoding="utf-8")
        report = pp.prune_traces(
            "in.jsonl",
            output_path="out.jsonl",
            min_frequency=0.9,
            tokenizer="my/model",
        )
        assert loaded["name"] == "my/model"
        assert report.rows_pruned == 3

    def test_prune_traces_friendly_error_on_missing_tokenizer(self, tmp_path, monkeypatch):
        import types

        from soup_cli.utils import prune_prompt as pp

        monkeypatch.chdir(tmp_path)

        def _boom(name, **kw):  # noqa: ARG001
            raise OSError("no such model")

        fake_auto = types.SimpleNamespace(from_pretrained=_boom)
        monkeypatch.setitem(
            sys.modules, "transformers",
            types.SimpleNamespace(AutoTokenizer=fake_auto),
        )
        (tmp_path / "in.jsonl").write_text(
            '{"prompt": "a", "output": "b"}\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="could not load tokenizer"):
            pp.prune_traces(
                "in.jsonl",
                output_path="out.jsonl",
                tokenizer="bad/model",
            )

    def test_char_level_unchanged_when_no_tokenizer(self, tmp_path, monkeypatch):
        from soup_cli.utils.prune_prompt import prune_traces

        monkeypatch.chdir(tmp_path)
        rows = "".join(
            '{"prompt": "PREFIX %d", "output": "a"}\n' % i for i in range(4)
        )
        (tmp_path / "in.jsonl").write_text(rows, encoding="utf-8")
        report = prune_traces(
            "in.jsonl",
            output_path="out.jsonl",
            min_frequency=0.9,
        )
        assert report.prefix.startswith("PREFIX ")
        assert report.rows_pruned == 4

    def test_no_top_level_transformers_import(self):
        # Lazy import — `soup prune-prompt --help` must not pull transformers.
        # Anchor on __file__ (not cwd) so the guard survives another test's
        # monkeypatch.chdir (v0.58.0 source-grep precedent).
        repo_root = Path(__file__).resolve().parent.parent
        src = (
            repo_root / "src" / "soup_cli" / "utils" / "prune_prompt.py"
        ).read_text(encoding="utf-8")
        assert "\nimport transformers" not in src
        assert "\nfrom transformers" not in src


# ===========================================================================
# #149 — DynamicCurriculumCallback bucket selection by curriculum_metric
# ===========================================================================


class _FakeState:
    def __init__(self, global_step):
        self.global_step = global_step


class TestPercentileBucket:
    def test_max_value_top_bucket(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        window = [0.1, 0.2, 0.3, 0.4]
        assert percentile_bucket(0.9, window, 4) == 3

    def test_min_value_bucket_zero(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        window = [0.5, 0.6, 0.7, 0.8]
        assert percentile_bucket(0.1, window, 4) == 0

    def test_mid_value_mid_bucket(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        window = [0.0, 1.0, 2.0, 3.0]
        # value 1.5 → 2/4 le → rank 0.5 → bucket 2
        assert percentile_bucket(1.5, window, 4) == 2

    def test_single_bucket(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        assert percentile_bucket(5.0, [1.0, 2.0], 1) == 0

    def test_empty_window_bucket_zero(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        assert percentile_bucket(5.0, [], 4) == 0

    def test_bool_value_rejected(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        with pytest.raises(ValueError):
            percentile_bucket(True, [1.0], 4)

    def test_non_finite_value_rejected(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        with pytest.raises(ValueError):
            percentile_bucket(float("nan"), [1.0], 4)

    def test_bool_num_buckets_rejected(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        with pytest.raises(ValueError):
            percentile_bucket(1.0, [1.0], True)

    @pytest.mark.parametrize("nb", [0, 21])
    def test_num_buckets_out_of_bounds_rejected(self, nb):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        with pytest.raises(ValueError):
            percentile_bucket(1.0, [1.0], nb)

    def test_equal_to_all_window_values_top_bucket(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        # value == every member → le == len → rank 1.0 → clamped to top.
        # Pins the `<=` (not `<`) in the rank tally.
        assert percentile_bucket(0.5, [0.5, 0.5], 4) == 3

    def test_consistently_high_value_stable_top_bucket(self):
        from soup_cli.utils.curriculum_dynamic import percentile_bucket

        window = [0.1, 0.2, 0.3, 0.4, 0.5]
        # Two independent appearances of a high loss → same top bucket.
        b1 = percentile_bucket(9.0, window, 4)
        b2 = percentile_bucket(9.0, window + [0.15, 0.25], 4)
        assert b1 == b2 == 3


class TestCurriculumCallbackMetric:
    def _callback(self, tmp_path, metric):
        from soup_cli.monitoring.curriculum_callback import (
            DynamicCurriculumCallback,
        )
        from soup_cli.utils.curriculum_dynamic import DynamicCurriculumPolicy

        policy = DynamicCurriculumPolicy(num_buckets=4, recompute_every_n_steps=10)
        return DynamicCurriculumCallback(
            policy=policy, output_dir=str(tmp_path), curriculum_metric=metric
        )

    def test_metric_validated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            self._callback(tmp_path, "bogus")

    def test_metric_property(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = self._callback(tmp_path, "loss")
        assert cb.curriculum_metric == "loss"

    def test_loss_metric_routes_high_to_top_bucket(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = self._callback(tmp_path, "loss")
        # Warm the window with low losses (round-robin during warm-up).
        for step in range(1, 9):
            cb.on_log(None, _FakeState(step), None, logs={"loss": 0.1 * step})
        # Now a clearly-high loss → top bucket (index 3).
        cb.on_log(None, _FakeState(50), None, logs={"loss": 99.0})
        # The high-loss sample must be in the top bucket.
        assert 3 in cb._stats
        assert cb._stats[3]["num_samples"] >= 1.0

    def test_length_metric_falls_back_to_round_robin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = self._callback(tmp_path, "length")
        # No length signal in logs → round-robin (step % num_buckets).
        cb.on_log(None, _FakeState(2), None, logs={"loss": 99.0})
        # step 2 % 4 buckets == 2.
        assert 2 in cb._stats

    def test_default_metric_is_length_round_robin(self, tmp_path, monkeypatch):
        from soup_cli.monitoring.curriculum_callback import (
            DynamicCurriculumCallback,
        )
        from soup_cli.utils.curriculum_dynamic import DynamicCurriculumPolicy

        monkeypatch.chdir(tmp_path)
        policy = DynamicCurriculumPolicy(num_buckets=4, recompute_every_n_steps=10)
        cb = DynamicCurriculumCallback(policy=policy, output_dir=str(tmp_path))
        assert cb.curriculum_metric == "length"
        cb.on_log(None, _FakeState(3), None, logs={"loss": 1.0})
        assert 3 in cb._stats  # round-robin: step 3 % 4

    def test_perplexity_metric_buckets_like_loss(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cb = self._callback(tmp_path, "perplexity")
        for step in range(1, 9):
            cb.on_log(None, _FakeState(step), None, logs={"loss": 0.1 * step})
        cb.on_log(None, _FakeState(50), None, logs={"loss": 20.0})
        assert 3 in cb._stats

    def test_attach_threads_curriculum_metric(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from types import SimpleNamespace

        from soup_cli.utils.peft_wiring import attach_curriculum_callback

        captured = {}

        class _FakeTrainer:
            def add_callback(self, cb):
                captured["cb"] = cb

        tcfg = SimpleNamespace(
            curriculum_dynamic=True,
            curriculum_buckets=4,
            curriculum_metric="loss",
            curriculum_dynamic_recompute_steps=10,
            curriculum_dynamic_floor=0.05,
            curriculum_dynamic_temperature=1.0,
        )
        ok = attach_curriculum_callback(_FakeTrainer(), tcfg, str(tmp_path))
        assert ok is True
        assert captured["cb"].curriculum_metric == "loss"


# ===========================================================================
# #157 — extend --hub to data push / data forge
# ===========================================================================


class TestDataPushHub:
    def test_hub_flag_in_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "push", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        import re as _re

        clean = _re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")
        assert "--hub" in clean

    def test_invalid_hub_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "ds.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            ["data", "push", "--input", "ds.jsonl",
             "--hf-dataset", "user/ds", "--hub", "bogus"],
        )
        assert result.exit_code == 2

    def test_modelscope_routes_via_upload_repo(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        captured = {}

        def _fake_upload(hub, repo_id, **kw):
            captured["hub"] = hub
            captured["repo_id"] = repo_id
            captured.update(kw)
            # The staging dir is cleaned up after this returns, so check the
            # file is present at call time (proves the JSONL was staged).
            captured["staged_file_present"] = (
                Path(kw["folder_path"]) / "ds.jsonl"
            ).exists()
            # upload_repo enforces cwd-containment on folder_path; the staging
            # dir MUST be under cwd (regression guard for the system-tempdir
            # bug found in the v0.71.5 step-6 smoke).
            from soup_cli.utils.paths import is_under_cwd

            captured["folder_under_cwd"] = is_under_cwd(kw["folder_path"])

        monkeypatch.setattr(hubs, "upload_repo", _fake_upload)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ds.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            ["data", "push", "--input", "ds.jsonl",
             "--hf-dataset", "user/ds", "--hub", "modelscope"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert captured["hub"] == "modelscope"
        assert captured["repo_id"] == "user/ds"
        assert captured["repo_type"] == "dataset"
        assert captured["staged_file_present"] is True
        assert captured["folder_under_cwd"] is True

    def test_modelers_missing_sdk_friendly_error(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        def _boom(hub, repo_id, **kw):  # noqa: ARG001
            raise ImportError("openmind_hub is not installed. pip install ...")

        monkeypatch.setattr(hubs, "upload_repo", _boom)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ds.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            ["data", "push", "--input", "ds.jsonl",
             "--hf-dataset", "user/ds", "--hub", "modelers"],
        )
        assert result.exit_code == 1
        assert "openmind_hub" in result.output

    def test_generic_upload_error_exit_1(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        def _boom(hub, repo_id, **kw):  # noqa: ARG001
            raise RuntimeError("network down")

        monkeypatch.setattr(hubs, "upload_repo", _boom)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ds.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            ["data", "push", "--input", "ds.jsonl",
             "--hf-dataset", "user/ds", "--hub", "modelscope"],
        )
        assert result.exit_code == 1
        assert "Upload failed" in result.output

    def test_hf_default_unchanged_no_token(self, tmp_path, monkeypatch):
        # Default --hub hf with no token → existing "no token" error (regression).
        from soup_cli.cli import app
        from soup_cli.utils import hf as _hf

        monkeypatch.setattr(_hf, "resolve_token", lambda: None)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ds.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            ["data", "push", "--input", "ds.jsonl", "--hf-dataset", "user/ds"],
        )
        assert result.exit_code == 1
        assert "token" in result.output.lower()


class TestDataForgeHub:
    def _docs(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text(
            "Paragraph one with content.\n\nParagraph two here.\n",
            encoding="utf-8",
        )
        return docs

    def test_hub_flag_in_help(self):
        from soup_cli.cli import app

        result = runner.invoke(app, ["data", "forge", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        import re as _re

        clean = _re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\n", " ")
        assert "--hub" in clean

    def test_invalid_hub_rejected(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        self._docs(tmp_path)
        result = runner.invoke(
            app,
            ["data", "forge", "--docs", "docs", "--hub", "bogus",
             "--teacher", "owner/repo"],
        )
        assert result.exit_code == 2

    def test_non_hf_prefetches_teacher(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        captured = {}

        def _fake_prefetch(base, hub, **kw):  # noqa: ARG001
            captured["base"] = base
            captured["hub"] = hub
            return str(tmp_path / "cache" / "teacher")

        monkeypatch.setattr(hubs, "prefetch_model_from_hub", _fake_prefetch)
        monkeypatch.chdir(tmp_path)
        self._docs(tmp_path)
        result = runner.invoke(
            app,
            ["data", "forge", "--docs", "docs", "--hub", "modelers",
             "--teacher", "owner/teacher-model", "--target-rows", "2"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert captured["base"] == "owner/teacher-model"
        assert captured["hub"] == "modelers"

    def test_hf_does_not_prefetch(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        called = {"n": 0}

        def _fake_prefetch(base, hub, **kw):  # noqa: ARG001
            called["n"] += 1
            return "x"

        monkeypatch.setattr(hubs, "prefetch_model_from_hub", _fake_prefetch)
        monkeypatch.chdir(tmp_path)
        self._docs(tmp_path)
        result = runner.invoke(
            app,
            ["data", "forge", "--docs", "docs", "--target-rows", "2",
             "--teacher", "owner/teacher-model"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert called["n"] == 0  # --hub hf default → no prefetch

    def test_non_hf_bare_teacher_warns_no_prefetch(self, tmp_path, monkeypatch):
        # Non-HF hub but teacher lacks owner/name → warn, do NOT prefetch
        # (code-review MEDIUM fix — no silent no-op of --hub).
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        called = {"n": 0}
        monkeypatch.setattr(
            hubs, "prefetch_model_from_hub",
            lambda *a, **k: (called.__setitem__("n", called["n"] + 1), "x")[1],
        )
        monkeypatch.chdir(tmp_path)
        self._docs(tmp_path)
        result = runner.invoke(
            app,
            ["data", "forge", "--docs", "docs", "--hub", "modelers",
             "--teacher", "barename", "--target-rows", "2"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert called["n"] == 0
        assert "ignored" in result.output.lower()

    def test_non_hf_prefetch_import_error_exit_1(self, tmp_path, monkeypatch):
        from soup_cli.cli import app
        from soup_cli.utils import hubs

        def _boom(base, hub, **kw):  # noqa: ARG001
            raise ImportError("openmind_hub is not installed")

        monkeypatch.setattr(hubs, "prefetch_model_from_hub", _boom)
        monkeypatch.chdir(tmp_path)
        self._docs(tmp_path)
        result = runner.invoke(
            app,
            ["data", "forge", "--docs", "docs", "--hub", "modelers",
             "--teacher", "owner/teacher", "--target-rows", "2"],
        )
        assert result.exit_code == 1
        assert "openmind_hub" in result.output
