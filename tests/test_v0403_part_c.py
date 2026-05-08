"""Tests for v0.40.3 Part C — #33 (a) data from-traces --judge + (b) serve --trace-log."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.data.traces.pair_builder import PreferencePair
from soup_cli.data.traces.quality import (
    DEFAULT_MIN_CONFIDENCE,
    JudgeFilterReport,
    judge_filter_pairs,
)
from soup_cli.monitoring.trace_logger import (
    DEFAULT_CAP_MB,
    TraceLogWriter,
)

# ---------------------------------------------------------------------------
# Part C-a — judge filter
# ---------------------------------------------------------------------------


_CRASH = object()


def _pair(prompt="p", chosen="c", rejected="r"):
    return PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected, source="t")


class _FakeJudge:
    """Mock judge that returns scripted weighted_score values."""

    def __init__(self, scores, scale=(1, 5)):
        self._scores = list(scores)
        self.rubric = {"scale": {"min": scale[0], "max": scale[1]}}
        self.calls = 0

    def evaluate(self, prompt, response, category="default"):
        if self.calls >= len(self._scores):
            raise RuntimeError("exhausted scripted scores")
        s = self._scores[self.calls]
        self.calls += 1
        if s is _CRASH:
            raise RuntimeError("backend down")
        return SimpleNamespace(weighted_score=s)


class TestJudgeFilterPairs:
    def test_default_threshold_is_07(self):
        assert DEFAULT_MIN_CONFIDENCE == 0.7

    def test_kept_when_chosen_clearly_wins(self):
        # scores: chosen=5, rejected=1  → norm: 1.0 vs 0.0 → diff 1.0 >= 0.7
        judge = _FakeJudge([5, 1])
        kept, report = judge_filter_pairs([_pair()], judge=judge)
        assert len(kept) == 1
        assert report.kept == 1 and report.dropped == 0 and report.errors == 0

    def test_dropped_when_diff_below_threshold(self):
        # 3 vs 2  → 0.5 vs 0.25 → diff 0.25 < 0.7
        judge = _FakeJudge([3, 2])
        kept, report = judge_filter_pairs([_pair()], judge=judge)
        assert kept == []
        assert report.dropped == 1 and report.kept == 0

    def test_dropped_when_rejected_wins(self):
        judge = _FakeJudge([1, 5])
        kept, report = judge_filter_pairs([_pair()], judge=judge)
        assert kept == []
        assert report.dropped == 1

    def test_judge_exception_counts_as_error(self):
        judge = _FakeJudge([_CRASH, _CRASH])
        kept, report = judge_filter_pairs([_pair()], judge=judge)
        assert kept == []
        assert report.errors == 1
        assert report.dropped == 0
        assert report.kept == 0

    def test_lower_threshold_keeps_borderline(self):
        judge = _FakeJudge([3, 2])  # diff 0.25
        kept, report = judge_filter_pairs(
            [_pair()], judge=judge, min_confidence=0.2,
        )
        assert len(kept) == 1
        assert report.kept == 1

    def test_threshold_zero_keeps_ties(self):
        judge = _FakeJudge([3, 3])
        kept, _ = judge_filter_pairs([_pair()], judge=judge, min_confidence=0.0)
        assert len(kept) == 1

    def test_rejects_non_numeric_threshold(self):
        with pytest.raises(TypeError):
            judge_filter_pairs([_pair()], judge=_FakeJudge([5, 1]),
                                min_confidence="0.7")  # type: ignore[arg-type]

    def test_rejects_bool_threshold(self):
        with pytest.raises(TypeError):
            judge_filter_pairs([_pair()], judge=_FakeJudge([5, 1]),
                                min_confidence=True)  # type: ignore[arg-type]

    def test_rejects_threshold_above_one(self):
        with pytest.raises(ValueError):
            judge_filter_pairs([_pair()], judge=_FakeJudge([5, 1]),
                                min_confidence=1.5)

    def test_rejects_threshold_below_zero(self):
        with pytest.raises(ValueError):
            judge_filter_pairs([_pair()], judge=_FakeJudge([5, 1]),
                                min_confidence=-0.1)

    def test_rejects_nan_threshold(self):
        with pytest.raises(ValueError):
            judge_filter_pairs([_pair()], judge=_FakeJudge([5, 1]),
                                min_confidence=float("nan"))

    def test_too_many_pairs_rejected(self):
        too_many = [_pair() for _ in range(100_001)]
        judge = _FakeJudge([])
        with pytest.raises(ValueError):
            judge_filter_pairs(too_many, judge=judge)

    def test_empty_pairs_returns_empty(self):
        judge = _FakeJudge([])
        kept, report = judge_filter_pairs([], judge=judge)
        assert kept == []
        assert report == JudgeFilterReport(kept=0, dropped=0, errors=0)

    def test_uses_judge_rubric_scale(self):
        # 1-10 scale: chosen=10 rejected=2 → norm 1.0 vs 0.111
        judge = _FakeJudge([10, 2], scale=(1, 10))
        kept, _ = judge_filter_pairs([_pair()], judge=judge)
        assert len(kept) == 1

    def test_report_is_frozen_dataclass(self):
        import dataclasses

        report = JudgeFilterReport(kept=1, dropped=2, errors=3)
        # Python 3.11+: dataclasses.FrozenInstanceError; older versions raise
        # AttributeError. Accept either explicitly — never the broad Exception.
        frozen_error = getattr(dataclasses, "FrozenInstanceError", AttributeError)
        with pytest.raises(frozen_error):
            report.kept = 5  # type: ignore[misc]

    def test_degenerate_scale_drops_all_at_positive_threshold(self):
        # scale_max == scale_min → both normalised to 0 → diff=0 → dropped at >0.
        judge = _FakeJudge([3, 3], scale=(5, 5))
        kept, report = judge_filter_pairs(
            [_pair()], judge=judge, min_confidence=0.1,
        )
        assert kept == [] and report.dropped == 1

    def test_degenerate_scale_keeps_all_at_zero_threshold(self):
        judge = _FakeJudge([3, 3], scale=(5, 5))
        kept, _ = judge_filter_pairs(
            [_pair()], judge=judge, min_confidence=0.0,
        )
        assert len(kept) == 1

    def test_lazy_materialisation_does_not_buffer_full_generator(self):
        # Build a generator that explodes on the (_MAX_BATCH+2)th item.
        # Lazy slicing should hit the cap raise WITHOUT advancing past _MAX_BATCH+1.
        def _explode_after(n):
            for i in range(n):
                yield _pair(prompt=f"q{i}")
            raise RuntimeError("generator must not have been advanced this far")

        with pytest.raises(ValueError):
            judge_filter_pairs(
                _explode_after(100_001), judge=_FakeJudge([5, 1] * 100_000),
            )


class TestFromTracesJudgeCli:
    """End-to-end CLI: feed tiny LangChain JSONL, --judge enabled with mocked
    evaluator, expect filtered output file."""

    def _write_traces(self, tmp_path):
        log = tmp_path / "traces.jsonl"
        events = [
            {"prompt": "q1", "output": "a1-good", "signal": "thumbs_up"},
            {"prompt": "q1", "output": "a1-bad", "signal": "thumbs_down"},
        ]
        with log.open("w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        return log

    def test_judge_flag_filters_pairs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        traces = self._write_traces(tmp_path)
        out = tmp_path / "prefs.jsonl"

        # Patch JudgeEvaluator + the filter to deterministic mock.

        class FakeJudgeCls:
            def __init__(self, **kw):
                self.rubric = {"scale": {"min": 1, "max": 5}}

            def evaluate(self, prompt, response, category="default"):
                # chosen wins by a large margin so kept.
                return SimpleNamespace(
                    weighted_score=5.0 if "good" in response else 1.0,
                )

        monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", FakeJudgeCls)

        runner = CliRunner()
        result = runner.invoke(
            soup_app,
            [
                "data", "from-traces",
                "--logs", str(traces),
                "--format", "langchain",
                "--signal", "thumbs_up",
                "--output", str(out),
                "--judge",
                "--min-confidence", "0.5",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.exists()
        assert "Judge filter" in result.output

    def test_judge_help_text(self):
        runner = CliRunner()
        result = runner.invoke(soup_app, ["data", "from-traces", "--help"])
        assert result.exit_code == 0
        # Narrow terminals on CI may break a long flag across two lines
        # (e.g. "--judge\n-provider"). Strip ANSI + whitespace before match.
        cleaned = re.sub(r"(\x1b\[[0-9;]*m|\s)+", "", result.output)
        assert "--judge" in cleaned
        assert "--min-confidence" in cleaned

    def test_judge_provider_invalid_rejected_early(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        traces = self._write_traces(tmp_path)
        out = tmp_path / "prefs.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            soup_app,
            [
                "data", "from-traces",
                "--logs", str(traces),
                "--format", "langchain",
                "--signal", "thumbs_up",
                "--output", str(out),
                "--judge",
                "--judge-provider", "evilcorp",
            ],
        )
        assert result.exit_code == 1
        assert "evilcorp" in result.output
        assert "invalid" in result.output.lower()


# ---------------------------------------------------------------------------
# Part C-b — TraceLogWriter
# ---------------------------------------------------------------------------


class TestTraceLogWriter:
    def test_default_cap_is_100(self):
        assert DEFAULT_CAP_MB == 100

    def test_records_jsonl_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "trace.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="hello", response="world", latency_ms=12.5, tokens=3,
        )
        contents = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(contents) == 1
        record = json.loads(contents[0])
        assert record["prompt"] == "hello"
        assert record["response"] == "world"
        assert record["latency_ms"] == 12.5
        assert record["tokens"] == 3
        assert "ts" in record

    def test_appends_multiple_lines(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        writer = TraceLogWriter(str(tmp_path / "log.jsonl"))
        for i in range(5):
            writer.record(prompt=f"p{i}", response="r", latency_ms=1.0, tokens=1)
        lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Use an absolute path outside cwd.
        outside = Path(tmp_path).parent / "evil.jsonl"
        with pytest.raises(ValueError, match="cwd"):
            TraceLogWriter(str(outside))

    def test_null_byte_path_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            TraceLogWriter("a\x00b.jsonl")

    def test_empty_path_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            TraceLogWriter("")

    def test_cap_below_min_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            TraceLogWriter(str(tmp_path / "x.jsonl"), cap_mb=0)

    def test_cap_above_max_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            TraceLogWriter(str(tmp_path / "x.jsonl"), cap_mb=10_001)

    def test_bool_cap_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            TraceLogWriter(str(tmp_path / "x.jsonl"), cap_mb=True)  # type: ignore[arg-type]

    def test_non_string_path_rejected(self):
        with pytest.raises(ValueError):
            TraceLogWriter(123)  # type: ignore[arg-type]

    def test_rotation_when_cap_exceeded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path), cap_mb=1)
        # Write a single very large response so the next call rotates.
        big = "x" * (1024 * 1024)  # 1 MB string
        writer.record(prompt="p", response=big, latency_ms=1.0, tokens=1)
        # Next write triggers rotation: file size > 1MB cap.
        writer.record(prompt="p2", response="r2", latency_ms=1.0, tokens=1)
        backup = path.with_suffix(path.suffix + ".1")
        assert backup.exists()
        # The new active file should contain the second record only.
        active = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(active) == 1
        assert json.loads(active[0])["prompt"] == "p2"
        # The backup file must contain the original large record.
        backup_lines = backup.read_text(encoding="utf-8").strip().splitlines()
        assert len(backup_lines) == 1
        assert json.loads(backup_lines[0])["prompt"] == "p"
        # A third small record must append to the (small) new active file
        # without triggering another rotation.
        writer.record(prompt="p3", response="r3", latency_ms=1.0, tokens=1)
        final = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(final) == 2
        assert json.loads(final[1])["prompt"] == "p3"

    def test_secret_redaction_in_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="auth: Bearer abc123def456ghi789",
            response="ok",
            latency_ms=1.0,
            tokens=1,
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert "<redacted>" in rec["prompt"]
        assert "abc123def456ghi789" not in rec["prompt"]

    def test_secret_redaction_in_response(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="ok",
            response="here is your token: hf_abcdef0123456789",
            latency_ms=1.0,
            tokens=1,
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert "<redacted>" in rec["response"]
        assert "hf_abcdef0123456789" not in rec["response"]

    def test_secret_redaction_preserves_trailing_period(self, tmp_path, monkeypatch):
        """The Bearer body excludes `.` so an end-of-sentence period survives."""
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="Use Bearer abc12345xyz.",
            response="ok",
            latency_ms=1.0,
            tokens=1,
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert "<redacted>" in rec["prompt"]
        # Trailing period preserved (not consumed by greedy match).
        assert rec["prompt"].endswith(".")

    def test_secret_redaction_in_extra_dict(self, tmp_path, monkeypatch):
        """Caller-supplied `extra` values are redacted recursively."""
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="ok", response="ok", latency_ms=1.0, tokens=1,
            extra={"system_prompt": "Bearer abcdefghij1234567890"},
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert "<redacted>" in rec["system_prompt"]
        assert "abcdefghij1234567890" not in rec["system_prompt"]

    def test_secret_redaction_redacts_sk_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="key sk-abcdefghijklmnopqrstuvwxyz",
            response="ok",
            latency_ms=1.0,
            tokens=1,
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert "<redacted>" in rec["prompt"]
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in rec["prompt"]

    def test_rotation_refuses_symlink_backup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        # Pre-place a symlink at <path>.1 → some target outside the log
        outside = tmp_path / "decoy.txt"
        outside.write_text("decoy", encoding="utf-8")
        backup = path.with_suffix(path.suffix + ".1")
        try:
            backup.symlink_to(outside)
        except (NotImplementedError, OSError):
            pytest.skip("symlink creation not supported on this platform/user")

        writer = TraceLogWriter(str(path), cap_mb=1)
        big = "x" * (1024 * 1024)
        writer.record(prompt="p", response=big, latency_ms=1.0, tokens=1)
        # Trigger rotation; symlink must be rejected → no rename happens.
        writer.record(prompt="p2", response="r2", latency_ms=1.0, tokens=1)
        # Symlink remains pointing at the decoy; decoy contents preserved.
        assert outside.read_text(encoding="utf-8") == "decoy"

    def test_unserialisable_entry_dropped_silently(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        # default=str fallback handles most types; force the JSON encoder to
        # raise by injecting a "bad" extra value that breaks even repr.
        # Simpler: just verify the call doesn't raise on weird shapes.
        writer.record(
            prompt="p", response="r", latency_ms=float("nan"), tokens=1,
        )
        # The file may or may not contain a line — important: no raise.
        assert path.exists()

    def test_path_property(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        writer = TraceLogWriter(str(tmp_path / "log.jsonl"))
        assert writer.path.name == "log.jsonl"

    def test_cap_bytes_property(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        writer = TraceLogWriter(str(tmp_path / "log.jsonl"), cap_mb=5)
        assert writer.cap_bytes == 5 * 1024 * 1024

    def test_extra_fields_merged(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="p", response="r", latency_ms=1.0, tokens=1,
            extra={"adapter": "chat", "model": "qwen"},
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert rec["adapter"] == "chat"
        assert rec["model"] == "qwen"

    def test_extra_does_not_override_core_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))
        writer.record(
            prompt="real", response="r", latency_ms=1.0, tokens=1,
            extra={"prompt": "spoofed"},
        )
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        assert rec["prompt"] == "real"

    def test_thread_safety_basic(self, tmp_path, monkeypatch):
        # Best-effort: spawn N threads each writing once, expect N lines.
        import threading

        monkeypatch.chdir(tmp_path)
        path = tmp_path / "log.jsonl"
        writer = TraceLogWriter(str(path))

        def _worker(i):
            writer.record(prompt=f"p{i}", response="r", latency_ms=1.0, tokens=1)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 20


_FASTAPI_AVAILABLE = True
try:
    import fastapi  # noqa: F401
except ImportError:
    _FASTAPI_AVAILABLE = False


class TestServeTraceLogWiring:
    def test_serve_help_lists_trace_log(self):
        runner = CliRunner()
        result = runner.invoke(soup_app, ["serve", "--help"])
        assert result.exit_code == 0
        # Narrow terminals on CI may break "--trace-log" across two lines
        # (e.g. "--trace\n-log"). Strip whitespace before match.
        cleaned = re.sub(r"(\x1b\[[0-9;]*m|\s)+", "", result.output)
        assert "--trace-log" in cleaned

    def test_serve_help_lists_trace_log_cap_mb(self):
        runner = CliRunner()
        result = runner.invoke(soup_app, ["serve", "--help"])
        assert result.exit_code == 0
        cleaned = re.sub(r"(\x1b\[[0-9;]*m|\s)+", "", result.output)
        assert "--trace-log-cap-mb" in cleaned

    @pytest.mark.skipif(
        not _FASTAPI_AVAILABLE,
        reason="fastapi not installed (only required for `serve` extras)",
    )
    def test_create_app_accepts_trace_log_writer(self, tmp_path, monkeypatch):
        # Not building a real model — just assert the parameter is accepted
        # and stored on app.state.
        from soup_cli.commands.serve import _create_app

        monkeypatch.chdir(tmp_path)
        writer = TraceLogWriter(str(tmp_path / "log.jsonl"))

        tokenizer = MagicMock()
        model_obj = MagicMock()
        app = _create_app(
            model_obj=model_obj,
            tokenizer=tokenizer,
            device="cpu",
            model_name="test",
            max_tokens_default=64,
            trace_log_writer=writer,
        )
        assert app.state.trace_log_writer is writer

    @pytest.mark.skipif(
        not _FASTAPI_AVAILABLE,
        reason="fastapi not installed (only required for `serve` extras)",
    )
    def test_create_app_default_writer_is_none(self, tmp_path, monkeypatch):
        from soup_cli.commands.serve import _create_app

        monkeypatch.chdir(tmp_path)
        tokenizer = MagicMock()
        model_obj = MagicMock()
        app = _create_app(
            model_obj=model_obj,
            tokenizer=tokenizer,
            device="cpu",
            model_name="test",
            max_tokens_default=64,
        )
        assert app.state.trace_log_writer is None
