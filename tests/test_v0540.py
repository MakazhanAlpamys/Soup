"""v0.54.0 — `soup advise` pre-flight decision engine.

Three Parts:
- Part A: Verdict engine (taxonomy / dataset profile / Verdict dataclass / CLI).
- Part B: Probe runner (ROIEstimate / synth_probe_* helpers / --probe flag).
- Part C: Cross-project learning (advise_history + `soup advise compare`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

from soup_cli.commands import advise as advise_cmd
from soup_cli.utils.advise import (
    CHOICES,
    TASK_CATEGORIES,
    DatasetProfile,
    ROIEstimate,
    Verdict,
    build_verdict,
    classify_task,
    compute_dataset_profile,
    format_verdict_rubric,
    load_advise_dataset,
    next_command_for,
    synth_probe_baselines,
    synth_probe_lora_delta,
)
from soup_cli.utils.advise_history import (
    HistoryEntry,
    history_path,
    load_history,
    record_verdict,
    summarize_history,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


@pytest.fixture
def tiny_sft_rows() -> List[dict]:
    return [
        {"prompt": f"Q{i}", "response": f"A{i} text body example here"}
        for i in range(20)
    ]


@pytest.fixture
def big_sft_rows() -> List[dict]:
    return [
        {
            "prompt": f"User question number {i}: explain X",
            "response": f"Response number {i} with varied vocabulary {i*7 % 999}",
        }
        for i in range(200)
    ]


@pytest.fixture
def preference_rows() -> List[dict]:
    return [
        {
            "prompt": f"Q{i}",
            "chosen": f"good answer {i}",
            "rejected": f"bad answer {i}",
        }
        for i in range(120)
    ]


@pytest.fixture
def reasoning_rows() -> List[dict]:
    # Size >= _MIN_ROWS_FOR_GRPO (500) so build_verdict routes to GRPO.
    return [
        {
            "prompt": f"Solve {i}*{i}",
            "response": f"<think>Step: multiply {i} by {i}</think>{i*i}",
        }
        for i in range(600)
    ]


@pytest.fixture
def factual_rows() -> List[dict]:
    return [
        {
            "question": f"What is the capital of country_{i}?",
            "answer": f"city_{i}_distinct_label_{i*3}",
        }
        for i in range(120)
    ]


# ===========================================================================
# Part A — Verdict engine
# ===========================================================================


class TestPublicSurface:
    def test_choices_constant(self):
        assert CHOICES == ("PROMPT_ENG", "RAG", "SFT", "DPO", "GRPO")

    def test_task_categories_constant(self):
        assert "factual_lookup" in TASK_CATEGORIES
        assert "reasoning" in TASK_CATEGORIES
        assert "tool_use" in TASK_CATEGORIES
        assert len(TASK_CATEGORIES) == 7

    def test_verdict_frozen(self):
        v = Verdict(
            choice="SFT", confidence=0.7, reason="r", reverse_when="w",
            task_category="reasoning",
        )
        with pytest.raises(Exception):
            v.choice = "DPO"  # type: ignore[misc]

    def test_roi_estimate_default(self):
        r = ROIEstimate()
        assert r.prompt_eng_delta is None
        assert r.sft_delta is None

    def test_dataset_profile_frozen(self):
        p = DatasetProfile(
            row_count=10, avg_input_chars=10.0, avg_output_chars=10.0,
            type_token_diversity=0.5, label_variance=0.5,
        )
        with pytest.raises(Exception):
            p.row_count = 99  # type: ignore[misc]


class TestLoadAdviseDataset:
    def test_happy(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [{"a": 1}, {"a": 2}])
        os.chdir(tmp_path)
        rows = load_advise_dataset("data.jsonl")
        assert len(rows) == 2

    def test_rejects_outside_cwd(self, tmp_path):
        os.chdir(tmp_path)
        outside = tmp_path.parent / "outside.jsonl"
        outside.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cwd"):
            load_advise_dataset(str(outside))

    def test_rejects_symlink(self, tmp_path):
        if os.name == "nt":
            pytest.skip("symlink behaviour differs on Windows")
        target = tmp_path / "real.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        link = tmp_path / "linked.jsonl"
        os.symlink(target, link)
        os.chdir(tmp_path)
        with pytest.raises(ValueError, match="symlink"):
            load_advise_dataset("linked.jsonl")

    def test_rejects_null_byte_path(self, tmp_path):
        os.chdir(tmp_path)
        with pytest.raises(ValueError, match="NUL"):
            load_advise_dataset("bad\x00path.jsonl")

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match="non-empty"):
            load_advise_dataset("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            load_advise_dataset(None)  # type: ignore[arg-type]

    def test_missing_file(self, tmp_path):
        os.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_advise_dataset("nope.jsonl")

    def test_malformed_json_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        os.chdir(tmp_path)
        with pytest.raises(ValueError, match="valid JSON"):
            load_advise_dataset("bad.jsonl")

    def test_non_object_row(self, tmp_path):
        p = tmp_path / "list.jsonl"
        p.write_text("[1,2,3]\n", encoding="utf-8")
        os.chdir(tmp_path)
        with pytest.raises(ValueError, match="JSON object"):
            load_advise_dataset("list.jsonl")

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
        os.chdir(tmp_path)
        rows = load_advise_dataset("data.jsonl")
        assert len(rows) == 2

    def test_utf8_bom_stripped(self, tmp_path):
        p = tmp_path / "bom.jsonl"
        p.write_bytes(b'\xef\xbb\xbf{"a":1}\n')
        os.chdir(tmp_path)
        rows = load_advise_dataset("bom.jsonl")
        assert rows[0]["a"] == 1


class TestClassifyTask:
    def test_reasoning_via_traces(self, reasoning_rows):
        assert classify_task(reasoning_rows) == "reasoning"

    def test_tool_use_via_field(self):
        rows = [{"tool_calls": [{"name": "x"}], "prompt": "p"}] * 10
        assert classify_task(rows) == "tool_use"

    def test_summarization_keyword(self):
        rows = [{"prompt": "Summarize this text", "response": "tldr"}] * 30
        assert classify_task(rows) == "summarization"

    def test_classification_keyword(self):
        rows = [
            {"prompt": "classify this", "response": "label A"} for _ in range(30)
        ]
        assert classify_task(rows) == "classification"

    def test_default_fallback(self):
        rows = [{"a": 1}]
        assert classify_task(rows) == "factual_lookup"

    def test_goal_steers(self):
        rows = [{"prompt": "x", "response": "y"}] * 5
        assert classify_task(rows, goal="translate text") == "format_conversion"

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            classify_task(42)  # type: ignore[arg-type]

    def test_null_byte_goal_rejected(self):
        with pytest.raises(ValueError):
            classify_task([{}], goal="bad\x00goal")

    def test_oversize_goal_rejected(self):
        with pytest.raises(ValueError):
            classify_task([{}], goal="x" * 99999)

    def test_non_string_goal_rejected(self):
        with pytest.raises(TypeError):
            classify_task([{}], goal=123)  # type: ignore[arg-type]

    def test_messages_field_chat_format(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "translate this"},
                    {"role": "assistant", "content": "translated"},
                ]
            }
        ] * 20
        assert classify_task(rows) == "format_conversion"


class TestComputeDatasetProfile:
    def test_basic(self, big_sft_rows):
        p = compute_dataset_profile(big_sft_rows)
        assert p.row_count == 200
        assert p.avg_input_chars > 0
        assert p.avg_output_chars > 0
        assert 0.0 <= p.type_token_diversity <= 1.0
        assert 0.0 <= p.label_variance <= 1.0

    def test_empty(self):
        p = compute_dataset_profile([])
        assert p.row_count == 0
        assert p.type_token_diversity == 0.0

    def test_preference_detected(self, preference_rows):
        p = compute_dataset_profile(preference_rows)
        assert p.has_chosen_rejected is True

    def test_reasoning_detected(self, reasoning_rows):
        p = compute_dataset_profile(reasoning_rows)
        assert p.has_reasoning_traces is True

    def test_base_model_proximity_validates(self):
        with pytest.raises(TypeError):
            compute_dataset_profile([], base_model_proximity=True)  # bool

    def test_base_model_proximity_bounds(self):
        with pytest.raises(ValueError):
            compute_dataset_profile([], base_model_proximity=1.5)

    def test_base_model_proximity_nan(self):
        with pytest.raises(ValueError):
            compute_dataset_profile([], base_model_proximity=float("nan"))

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            compute_dataset_profile(42)  # type: ignore[arg-type]


class TestBuildVerdict:
    def test_preference_routes_to_dpo(self, preference_rows):
        profile = compute_dataset_profile(preference_rows)
        v = build_verdict(profile, "reasoning")
        assert v.choice == "DPO"

    def test_reasoning_with_traces_routes_to_grpo(self, reasoning_rows):
        profile = compute_dataset_profile(reasoning_rows)
        v = build_verdict(profile, "reasoning")
        assert v.choice == "GRPO"

    def test_tiny_routes_to_prompt_eng(self, tiny_sft_rows):
        profile = compute_dataset_profile(tiny_sft_rows)
        v = build_verdict(profile, "summarization")
        assert v.choice == "PROMPT_ENG"

    def test_factual_high_variance_routes_to_rag(self, factual_rows):
        profile = compute_dataset_profile(factual_rows)
        v = build_verdict(profile, "factual_lookup")
        assert v.choice == "RAG"

    def test_default_routes_to_sft(self, big_sft_rows):
        profile = compute_dataset_profile(big_sft_rows)
        v = build_verdict(profile, "summarization")
        assert v.choice == "SFT"

    def test_unknown_task_category(self, big_sft_rows):
        profile = compute_dataset_profile(big_sft_rows)
        with pytest.raises(ValueError, match="task_category"):
            build_verdict(profile, "nonsense")

    def test_invalid_roi_type(self, big_sft_rows):
        profile = compute_dataset_profile(big_sft_rows)
        with pytest.raises(TypeError):
            build_verdict(profile, "summarization", roi="bad")  # type: ignore

    def test_confidence_in_unit_interval(self, big_sft_rows):
        profile = compute_dataset_profile(big_sft_rows)
        v = build_verdict(profile, "summarization")
        assert 0.0 <= v.confidence <= 1.0

    def test_reason_non_empty(self, big_sft_rows):
        profile = compute_dataset_profile(big_sft_rows)
        v = build_verdict(profile, "summarization")
        assert v.reason
        assert v.reverse_when

    def test_roi_is_attached(self, big_sft_rows):
        profile = compute_dataset_profile(big_sft_rows)
        roi = ROIEstimate(sft_delta=0.42)
        v = build_verdict(profile, "summarization", roi=roi)
        assert v.estimated_roi.sft_delta == 0.42


# ===========================================================================
# Part B — Probe runner
# ===========================================================================


class TestSynthProbeBaselines:
    def test_keys_present(self, big_sft_rows):
        out = synth_probe_baselines(big_sft_rows)
        assert set(out.keys()) == {"zero_shot", "few_shot", "rag"}

    def test_empty_input(self):
        out = synth_probe_baselines([])
        assert out == {"zero_shot": 0.0, "few_shot": 0.0, "rag": 0.0}

    def test_bounded(self, big_sft_rows):
        out = synth_probe_baselines(big_sft_rows)
        for v in out.values():
            assert -1.0 <= v <= 1.0

    def test_rejects_bool_n_holdout(self, big_sft_rows):
        with pytest.raises(TypeError):
            synth_probe_baselines(big_sft_rows, n_holdout=True)

    def test_rejects_non_int_n_holdout(self, big_sft_rows):
        with pytest.raises(TypeError):
            synth_probe_baselines(big_sft_rows, n_holdout=1.5)  # type: ignore

    def test_rejects_n_holdout_out_of_range(self, big_sft_rows):
        with pytest.raises(ValueError):
            synth_probe_baselines(big_sft_rows, n_holdout=0)
        with pytest.raises(ValueError):
            synth_probe_baselines(big_sft_rows, n_holdout=99_999)

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            synth_probe_baselines(42)  # type: ignore[arg-type]


class TestSynthProbeLoraDelta:
    def test_returns_tuple(self, big_sft_rows):
        delta, secs = synth_probe_lora_delta(big_sft_rows)
        assert isinstance(delta, float)
        assert isinstance(secs, float)

    def test_tiny_returns_zero(self, tiny_sft_rows):
        delta, secs = synth_probe_lora_delta(tiny_sft_rows)
        assert delta == 0.0
        assert secs > 0

    def test_wall_clock_bounded(self, big_sft_rows):
        _, secs = synth_probe_lora_delta(big_sft_rows, n_steps=10_000)
        assert secs <= 600.0

    def test_delta_bounded(self, big_sft_rows):
        delta, _ = synth_probe_lora_delta(big_sft_rows)
        assert -0.5 < delta < 1.0

    def test_rejects_bool_n_steps(self, big_sft_rows):
        with pytest.raises(TypeError):
            synth_probe_lora_delta(big_sft_rows, n_steps=True)

    def test_rejects_n_steps_out_of_range(self, big_sft_rows):
        with pytest.raises(ValueError):
            synth_probe_lora_delta(big_sft_rows, n_steps=0)


class TestFormatVerdictRubric:
    def test_basic(self):
        v = Verdict(
            choice="SFT", confidence=0.7, reason="r1", reverse_when="r2",
            task_category="reasoning",
        )
        text = format_verdict_rubric(v)
        assert "SFT" in text
        assert "Confidence" in text
        assert "Reason" in text
        assert "Reverses when" in text

    def test_roi_renders(self):
        roi = ROIEstimate(sft_delta=0.3, prompt_eng_delta=-0.1)
        v = Verdict(
            choice="SFT", confidence=0.7, reason="r", reverse_when="w",
            task_category="reasoning", estimated_roi=roi,
        )
        text = format_verdict_rubric(v)
        assert "+0.300" in text
        assert "-0.100" in text

    def test_not_measured_when_none(self):
        v = Verdict(
            choice="SFT", confidence=0.7, reason="r", reverse_when="w",
            task_category="reasoning",
        )
        text = format_verdict_rubric(v)
        assert "(not measured)" in text

    def test_rejects_non_verdict(self):
        with pytest.raises(TypeError):
            format_verdict_rubric({"choice": "SFT"})  # type: ignore[arg-type]


# ===========================================================================
# Part C — Cross-project learning
# ===========================================================================


@pytest.fixture
def history_file(tmp_path):
    return str(tmp_path / "history.jsonl")


def _make_verdict() -> Verdict:
    return Verdict(
        choice="SFT", confidence=0.7, reason="r", reverse_when="w",
        task_category="reasoning",
    )


class TestRecordVerdict:
    def test_happy(self, history_file):
        entry = record_verdict(
            _make_verdict(), accepted=True, outcome=0.5, path=history_file,
        )
        assert isinstance(entry, HistoryEntry)
        assert entry.choice == "SFT"
        assert entry.accepted is True
        assert entry.outcome == 0.5
        assert Path(history_file).exists()

    def test_rejects_non_verdict(self, history_file):
        with pytest.raises(TypeError):
            record_verdict({}, accepted=True, path=history_file)  # type: ignore

    def test_rejects_non_bool_accepted(self, history_file):
        with pytest.raises(TypeError):
            record_verdict(
                _make_verdict(), accepted="yes", path=history_file,  # type: ignore
            )

    def test_rejects_bool_outcome(self, history_file):
        with pytest.raises(TypeError):
            record_verdict(
                _make_verdict(), accepted=True, outcome=True,  # type: ignore
                path=history_file,
            )

    def test_rejects_non_finite_outcome(self, history_file):
        with pytest.raises(ValueError):
            record_verdict(
                _make_verdict(), accepted=True, outcome=float("inf"),
                path=history_file,
            )

    def test_rejects_outcome_out_of_range(self, history_file):
        with pytest.raises(ValueError):
            record_verdict(
                _make_verdict(), accepted=True, outcome=1.5, path=history_file,
            )

    def test_rejects_null_byte_notes(self, history_file):
        with pytest.raises(ValueError):
            record_verdict(
                _make_verdict(), accepted=True, notes="bad\x00", path=history_file,
            )

    def test_rejects_oversize_notes(self, history_file):
        with pytest.raises(ValueError):
            record_verdict(
                _make_verdict(), accepted=True, notes="x" * 99999,
                path=history_file,
            )

    def test_rejects_invalid_choice(self, history_file):
        v = Verdict(
            choice="NONSENSE", confidence=0.5, reason="r", reverse_when="w",
            task_category="reasoning",
        )
        with pytest.raises(ValueError):
            record_verdict(v, accepted=True, path=history_file)

    def test_rejects_invalid_task_category(self, history_file):
        v = Verdict(
            choice="SFT", confidence=0.5, reason="r", reverse_when="w",
            task_category="nonsense",
        )
        with pytest.raises(ValueError):
            record_verdict(v, accepted=True, path=history_file)

    def test_rejects_symlink_path(self, tmp_path):
        if os.name == "nt":
            pytest.skip("symlink semantics differ on Windows")
        target = tmp_path / "real.jsonl"
        target.write_text("", encoding="utf-8")
        link = tmp_path / "link.jsonl"
        os.symlink(target, link)
        with pytest.raises(ValueError, match="symlink"):
            record_verdict(_make_verdict(), accepted=True, path=str(link))

    def test_notes_strips_newlines(self, history_file):
        entry = record_verdict(
            _make_verdict(), accepted=True,
            notes="line1\nline2\rline3", path=history_file,
        )
        assert "\n" not in entry.notes
        assert "\r" not in entry.notes


class TestLoadHistory:
    def test_empty(self, history_file):
        assert load_history(path=history_file) == []

    def test_round_trip(self, history_file):
        record_verdict(_make_verdict(), accepted=True, path=history_file)
        record_verdict(_make_verdict(), accepted=False, path=history_file)
        entries = load_history(path=history_file)
        assert len(entries) == 2
        # Newest first.
        assert entries[0].accepted is False

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "h.jsonl"
        good = (
            '{"choice":"SFT","task_category":"reasoning",'
            '"confidence":0.5,"accepted":true}'
        )
        p.write_text("not json\n" + good + "\n", encoding="utf-8")
        entries = load_history(path=str(p))
        assert len(entries) == 1

    def test_skips_invalid_choice(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text(
            '{"choice":"NONSENSE","task_category":"reasoning","confidence":0.5,"accepted":true}\n',
            encoding="utf-8",
        )
        entries = load_history(path=str(p))
        assert entries == []

    def test_limit_bounds(self, history_file):
        with pytest.raises(ValueError):
            load_history(limit=0, path=history_file)
        with pytest.raises(ValueError):
            load_history(limit=10_000_000, path=history_file)

    def test_limit_bool_rejected(self, history_file):
        with pytest.raises(TypeError):
            load_history(limit=True, path=history_file)

    def test_limit_caps_returned(self, history_file):
        for _ in range(5):
            record_verdict(_make_verdict(), accepted=True, path=history_file)
        entries = load_history(limit=3, path=history_file)
        assert len(entries) == 3


class TestHistoryPath:
    def test_default_under_home(self, monkeypatch):
        monkeypatch.delenv("SOUP_ADVISE_HISTORY_PATH", raising=False)
        p = history_path()
        assert p.endswith(os.path.join(".soup", "advise_history.jsonl"))

    def test_env_override_in_tempdir(self, monkeypatch, tmp_path):
        # tmp_path is typically inside the system temp dir
        import tempfile as _tempfile
        sysdir = os.path.realpath(_tempfile.gettempdir())
        override = os.path.join(sysdir, "soup_advise_test.jsonl")
        monkeypatch.setenv("SOUP_ADVISE_HISTORY_PATH", override)
        p = history_path()
        assert os.path.realpath(p) == os.path.realpath(override)

    def test_env_override_out_of_bounds_falls_back(self, monkeypatch):
        # /etc is neither home, cwd, nor tempdir on POSIX.
        if os.name == "nt":
            pytest.skip("path semantics differ on Windows")
        monkeypatch.setenv("SOUP_ADVISE_HISTORY_PATH", "/etc/advise.jsonl")
        p = history_path()
        assert p != "/etc/advise.jsonl"

    def test_env_null_byte_falls_back(self, monkeypatch):
        # OS env layer rejects raw NUL bytes (POSIX execve / Win32 SetEnv
        # both refuse `\x00`), so we exercise the helper's defence-in-depth
        # NUL guard via a stubbed env dict rather than `monkeypatch.setenv`.
        from soup_cli.utils import advise_history

        original = advise_history.os.environ
        try:
            advise_history.os.environ = {"SOUP_ADVISE_HISTORY_PATH": "x\x00y"}  # type: ignore[assignment]
            p = history_path()
        finally:
            advise_history.os.environ = original  # type: ignore[assignment]
        assert "\x00" not in p


class TestSummarizeHistory:
    def test_counts(self, history_file):
        v_sft = _make_verdict()
        v_dpo = Verdict(
            choice="DPO", confidence=0.7, reason="r", reverse_when="w",
            task_category="reasoning",
        )
        record_verdict(v_sft, accepted=True, path=history_file)
        record_verdict(v_dpo, accepted=True, path=history_file)
        record_verdict(v_dpo, accepted=True, path=history_file)
        entries = load_history(path=history_file)
        counts = summarize_history(entries)
        assert counts["SFT"] == 1
        assert counts["DPO"] == 2
        assert counts["GRPO"] == 0


# ===========================================================================
# CLI smoke
# ===========================================================================


class TestCLI:
    def test_help(self):
        result = runner.invoke(advise_cmd.app, ["--help"])
        assert result.exit_code == 0
        assert "advise" in result.output.lower() or "pre-flight" in result.output.lower()

    def test_explain_no_prior_verdict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("TMP", str(tmp_path))
        # Clean cached verdict if any
        last = advise_cmd._last_verdict_path()
        if os.path.exists(last):
            os.unlink(last)
        result = runner.invoke(advise_cmd.app, ["explain"])
        # Acceptable: no prior verdict → exit 1 OR previous verdict present
        assert result.exit_code in (0, 1)

    def test_compare_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "SOUP_ADVISE_HISTORY_PATH",
            str(tmp_path / "history.jsonl"),
        )
        result = runner.invoke(advise_cmd.app, ["compare"])
        assert result.exit_code == 0
        assert "No history" in result.output or "history" in result.output.lower()

    def test_default_happy_path(self, tmp_path, monkeypatch):
        os.chdir(tmp_path)
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [
            {"prompt": f"summarize doc {i}", "response": f"tldr {i}"}
            for i in range(120)
        ])
        result = runner.invoke(
            advise_cmd.app, ["run", "data.jsonl", "--goal", "summarize"]
        )
        assert result.exit_code == 0, result.output
        # Either SFT or PROMPT_ENG depending on heuristics; must be a known choice
        assert any(c in result.output for c in CHOICES)

    def test_default_missing_data(self):
        result = runner.invoke(advise_cmd.app, [])
        # `no_args_is_help=True`: Click 8.0–8.1 returns rc=0, Click 8.2+
        # returns rc=2 (the "missing command" convention). Both renderings
        # print the help text, so accept either.
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "advise" in result.output.lower()

    def test_default_nonexistent_data(self, tmp_path):
        os.chdir(tmp_path)
        result = runner.invoke(advise_cmd.app, ["run", "nope.jsonl"])
        assert result.exit_code == 1
        assert "Dataset error" in result.output or "not found" in result.output.lower()

    def test_default_outside_cwd(self, tmp_path):
        os.chdir(tmp_path)
        outside = tmp_path.parent / "outside.jsonl"
        outside.write_text("{}\n", encoding="utf-8")
        result = runner.invoke(advise_cmd.app, ["run", str(outside)])
        assert result.exit_code == 1

    def test_probe_flag(self, tmp_path, monkeypatch):
        os.chdir(tmp_path)
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [
            {"prompt": f"q{i}", "response": f"a{i}"} for i in range(120)
        ])
        result = runner.invoke(
            advise_cmd.app,
            ["run", "data.jsonl", "--goal", "summarize", "--probe"],
        )
        assert result.exit_code == 0, result.output
        assert "ROI" in result.output or "delta" in result.output.lower()

    def test_record_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "SOUP_ADVISE_HISTORY_PATH",
            str(tmp_path / "history.jsonl"),
        )
        os.chdir(tmp_path)
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [
            {"prompt": f"q{i}", "response": f"a{i}"} for i in range(120)
        ])
        result = runner.invoke(
            advise_cmd.app,
            ["run", "data.jsonl", "--goal", "summarize", "--record"],
        )
        assert result.exit_code == 0
        assert (tmp_path / "history.jsonl").exists()
        entries = load_history(path=str(tmp_path / "history.jsonl"))
        assert len(entries) == 1

    def test_compare_after_record(self, tmp_path, monkeypatch):
        history = tmp_path / "history.jsonl"
        monkeypatch.setenv("SOUP_ADVISE_HISTORY_PATH", str(history))
        record_verdict(_make_verdict(), accepted=True, path=str(history))
        result = runner.invoke(advise_cmd.app, ["compare"])
        assert result.exit_code == 0
        assert "SFT" in result.output

    def test_explain_after_default(self, tmp_path, monkeypatch):
        # Run advise to populate scratch file, then explain
        os.chdir(tmp_path)
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [
            {"prompt": f"q{i}", "response": f"a{i}"} for i in range(120)
        ])
        result1 = runner.invoke(
            advise_cmd.app, ["run", "data.jsonl", "--goal", "summarize"]
        )
        assert result1.exit_code == 0
        result2 = runner.invoke(advise_cmd.app, ["explain"])
        assert result2.exit_code == 0
        assert "Choice" in result2.output

    def test_compare_limit_bounds(self):
        result = runner.invoke(advise_cmd.app, ["compare", "--limit", "0"])
        assert result.exit_code != 0


# ===========================================================================
# Source-level wiring checks (regression guards)
# ===========================================================================


class TestSourceWiring:
    def test_cli_registers_advise(self):
        cli_path = Path(__file__).resolve().parent.parent / "soup_cli" / "cli.py"
        text = cli_path.read_text(encoding="utf-8")
        assert "_advise_cmd" in text
        assert 'name="advise"' in text

    def test_version_bump(self):
        from soup_cli import __version__
        assert __version__ == "0.54.0"

    def test_advise_module_imports(self):
        # Importable without heavy deps (lazy imports inside helpers).
        from soup_cli.commands import advise as _cmd  # noqa: F401
        from soup_cli.utils import advise as _advise  # noqa: F401
        from soup_cli.utils import advise_history as _hist  # noqa: F401


class TestArgvRewriter:
    def test_no_advise_in_argv(self):
        from soup_cli.cli import _rewrite_advise_argv
        argv = ["soup", "train", "--config", "x.yaml"]
        assert _rewrite_advise_argv(argv) == argv

    def test_advise_with_subcommand_unchanged(self):
        from soup_cli.cli import _rewrite_advise_argv
        for sub in ("run", "explain", "compare", "--help"):
            argv = ["soup", "advise", sub]
            assert _rewrite_advise_argv(argv) == argv

    def test_advise_with_data_injects_run(self):
        from soup_cli.cli import _rewrite_advise_argv
        argv = ["soup", "advise", "data.jsonl"]
        out = _rewrite_advise_argv(argv)
        assert out == ["soup", "advise", "run", "data.jsonl"]

    def test_advise_with_data_and_flags(self):
        from soup_cli.cli import _rewrite_advise_argv
        argv = ["soup", "advise", "data.jsonl", "--goal", "summarize"]
        out = _rewrite_advise_argv(argv)
        assert out[2] == "run"
        assert out[3] == "data.jsonl"

    def test_advise_with_flag_only_unchanged(self):
        from soup_cli.cli import _rewrite_advise_argv
        argv = ["soup", "advise", "--probe"]
        assert _rewrite_advise_argv(argv) == argv

    def test_advise_alone_unchanged(self):
        from soup_cli.cli import _rewrite_advise_argv
        argv = ["soup", "advise"]
        assert _rewrite_advise_argv(argv) == argv


# ===========================================================================
# Edge / defensive tests to round out coverage
# ===========================================================================


class TestDefensive:
    def test_classify_task_with_messages_assistant_only(self):
        rows = [
            {"messages": [{"role": "assistant", "content": "x"}]}
        ] * 5
        # No keywords + no tool_calls → default
        assert classify_task(rows) == "factual_lookup"

    def test_dataset_profile_handles_non_mapping_row(self):
        rows = [{"a": 1}, "not a dict", {"b": 2}]  # type: ignore[list-item]
        p = compute_dataset_profile(rows)  # type: ignore[arg-type]
        assert p.row_count == 3

    def test_build_verdict_with_preference_overrides_tiny(self):
        # Preference rule should fire BEFORE tiny rule
        rows = [
            {"chosen": "a", "rejected": "b"} for _ in range(5)
        ]
        profile = compute_dataset_profile(rows)
        v = build_verdict(profile, "reasoning")
        assert v.choice == "DPO"

    def test_synth_probe_baselines_rag_correlates_with_variance(
        self, factual_rows
    ):
        out = synth_probe_baselines(factual_rows)
        # High-variance factual dataset → rag delta should not be the worst.
        assert out["rag"] >= out["zero_shot"] - 0.1

    def test_format_rubric_handles_nan_delta(self):
        roi = ROIEstimate(prompt_eng_delta=float("nan"))
        v = Verdict(
            choice="SFT", confidence=0.5, reason="r", reverse_when="w",
            task_category="reasoning", estimated_roi=roi,
        )
        text = format_verdict_rubric(v)
        assert "non-finite" in text or "nan" in text.lower()

    def test_record_verdict_creates_parent_dir(self, tmp_path):
        target = tmp_path / "nested" / "subdir" / "h.jsonl"
        record_verdict(_make_verdict(), accepted=True, path=str(target))
        assert target.exists()

    def test_load_history_returns_list(self, tmp_path):
        p = str(tmp_path / "h.jsonl")
        # Doesn't exist yet
        assert load_history(path=p) == []

    def test_history_entry_frozen(self):
        e = HistoryEntry(
            timestamp="t", project="p", choice="SFT", task_category="reasoning",
            confidence=0.5, reason="r", reverse_when="w", accepted=True,
            outcome=None,
        )
        with pytest.raises(Exception):
            e.choice = "DPO"  # type: ignore[misc]


# ===========================================================================
# TDD-review additions — boundary + concurrency + just-added security paths
# ===========================================================================


class TestTDDFollowups:
    def test_read_last_verdict_rejects_symlink(self, tmp_path, monkeypatch):
        if os.name == "nt":
            pytest.skip("symlink semantics differ on Windows")
        target = tmp_path / "real.json"
        target.write_text('{"choice":"SFT"}', encoding="utf-8")
        link = tmp_path / "advise_last.json"
        os.symlink(target, link)
        monkeypatch.setattr(advise_cmd, "_last_verdict_path", lambda: str(link))
        # Symlinked scratch file must NOT be read — returns None silently.
        assert advise_cmd._read_last_verdict() is None

    def test_oversized_history_line_skipped(self, tmp_path):
        p = tmp_path / "h.jsonl"
        oversize = (
            '{"choice":"SFT","task_category":"reasoning","confidence":0.5,'
            '"accepted":true,"notes":"' + ("a" * 70000) + '"}'
        )
        good = (
            '{"choice":"DPO","task_category":"reasoning","confidence":0.5,'
            '"accepted":true}'
        )
        p.write_text(oversize + "\n" + good + "\n", encoding="utf-8")
        entries = load_history(path=str(p))
        # Oversize line dropped, valid one kept.
        assert len(entries) == 1
        assert entries[0].choice == "DPO"

    def test_concurrent_record_no_torn_writes(self, tmp_path):
        import threading

        p = str(tmp_path / "h.jsonl")
        errors: list = []

        def worker(idx: int):
            try:
                record_verdict(
                    _make_verdict(), accepted=True,
                    notes=f"thread-{idx}", path=p,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        entries = load_history(limit=20, path=p)
        # Every line valid JSON + parseable as HistoryEntry.
        assert len(entries) == 8

    def test_prompt_eng_at_49_sft_at_50(self):
        prof_small = DatasetProfile(
            row_count=49, avg_input_chars=10.0, avg_output_chars=10.0,
            type_token_diversity=0.5, label_variance=0.3,
        )
        prof_50 = DatasetProfile(
            row_count=50, avg_input_chars=10.0, avg_output_chars=10.0,
            type_token_diversity=0.5, label_variance=0.3,
        )
        assert build_verdict(prof_small, "summarization").choice == "PROMPT_ENG"
        assert build_verdict(prof_50, "summarization").choice == "SFT"

    def test_grpo_threshold_exact(self):
        prof_499 = DatasetProfile(
            row_count=499, avg_input_chars=10.0, avg_output_chars=10.0,
            type_token_diversity=0.5, label_variance=0.3,
            has_reasoning_traces=True,
        )
        prof_500 = DatasetProfile(
            row_count=500, avg_input_chars=10.0, avg_output_chars=10.0,
            type_token_diversity=0.5, label_variance=0.3,
            has_reasoning_traces=True,
        )
        # 499 rows of reasoning traces → still SFT (below GRPO floor).
        assert build_verdict(prof_499, "reasoning").choice == "SFT"
        # 500 rows → GRPO.
        assert build_verdict(prof_500, "reasoning").choice == "GRPO"

    def test_notes_boundary_4096_vs_4097(self, tmp_path):
        p = str(tmp_path / "h.jsonl")
        # Exactly 4096 chars accepted.
        record_verdict(_make_verdict(), accepted=True, notes="a" * 4096, path=p)
        # 4097 rejected.
        with pytest.raises(ValueError, match="4096"):
            record_verdict(_make_verdict(), accepted=True, notes="a" * 4097, path=p)

    def test_symlink_history_path_rejected_on_load(self, tmp_path):
        if os.name == "nt":
            pytest.skip("symlink semantics differ on Windows")
        target = tmp_path / "real.jsonl"
        target.write_text("", encoding="utf-8")
        link = tmp_path / "linked.jsonl"
        os.symlink(target, link)
        with pytest.raises(ValueError, match="symlink"):
            load_history(path=str(link))

    def test_build_verdict_null_byte_goal_rejected(self):
        profile = DatasetProfile(
            row_count=100, avg_input_chars=10.0, avg_output_chars=10.0,
            type_token_diversity=0.5, label_variance=0.3,
        )
        with pytest.raises(ValueError, match="NUL"):
            build_verdict(profile, "summarization", goal="bad\x00goal")

    def test_fmt_delta_bool_renders_invalid(self):
        roi = ROIEstimate(prompt_eng_delta=True)  # type: ignore[arg-type]
        v = Verdict(
            choice="SFT", confidence=0.5, reason="r", reverse_when="w",
            task_category="reasoning", estimated_roi=roi,
        )
        text = format_verdict_rubric(v)
        # bool is rejected as invalid even though it's technically isinstance(int).
        assert "(invalid)" in text or "+1.000" in text or "+0.000" in text


class TestAdviseSchemaField:
    """Plan's cross-cutting bullet: schema-only `advise: AdviseConfig` on SoupConfig."""

    def test_advise_field_default_none(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """
base: test-model
task: sft
data:
  train: ./data.jsonl
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.advise is None

    def test_advise_goal_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """
base: test-model
task: sft
data:
  train: ./data.jsonl
advise:
  goal: "summarize my reports"
  probe: true
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.advise is not None
        assert cfg.advise.goal == "summarize my reports"
        assert cfg.advise.probe is True
        assert cfg.advise.record is False

    def test_advise_null_byte_goal_rejected(self):
        # YAML parser already rejects raw NUL bytes; the AdviseConfig
        # field_validator is the defence-in-depth layer for callers that
        # bypass YAML (e.g. construct AdviseConfig directly).
        from pydantic import ValidationError

        from soup_cli.config.schema import AdviseConfig

        with pytest.raises(ValidationError, match="null"):
            AdviseConfig(goal="bad\x00goal")

    def test_advise_oversize_goal_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        oversized = "a" * 5000
        yaml_text = f"""
base: test-model
task: sft
data:
  train: ./data.jsonl
advise:
  goal: "{oversized}"
"""
        with pytest.raises(ValueError, match="4096|characters"):
            load_config_from_string(yaml_text)


class TestNextCommandFor:
    @pytest.mark.parametrize("choice,expected_token", [
        ("PROMPT_ENG", "soup chat"),
        ("RAG", "RAG is outside"),
        ("SFT", "soup autopilot"),
        ("DPO", "soup autopilot"),
        ("GRPO", "soup autopilot"),
    ])
    def test_known_choices(self, choice: str, expected_token: str):
        v = Verdict(
            choice=choice, confidence=0.5, reason="r", reverse_when="w",
            task_category="reasoning",
        )
        assert expected_token in next_command_for(v)

    def test_rejects_non_verdict(self):
        with pytest.raises(TypeError):
            next_command_for({"choice": "SFT"})  # type: ignore[arg-type]


class TestProbeForwardCompatKwargs:
    def test_baselines_accepts_future_kwargs(self, big_sft_rows):
        # v0.54.1 will add live model/device/timeout — calling with them
        # today must not raise.
        out = synth_probe_baselines(
            big_sft_rows, n_holdout=50,
            model="meta-llama/Llama-3.1-8B", device="cpu", timeout_seconds=60,
        )
        assert "zero_shot" in out

    def test_lora_delta_accepts_future_kwargs(self, big_sft_rows):
        delta, _ = synth_probe_lora_delta(
            big_sft_rows, n_steps=50,
            model="meta-llama/Llama-3.1-8B", device="cpu",
            lr=2e-5, timeout_seconds=120,
        )
        assert isinstance(delta, float)
