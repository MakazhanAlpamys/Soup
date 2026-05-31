"""v0.65.0 review-fix follow-ups (boundary + dedup + new-helper coverage).

Closes review L5 boundary tests + HIGH H1-H4 + MEDIUM M5/M6/M7/M2 fixes:

- H1 — behavior_battery fixture symlink + size cap + Traversable / op
- H2 — checklist_dsl shared `enforce_under_cwd_and_no_symlink` helper
- H3 — irt streaming + shared TOCTOU helper
- H4 — word-boundary agreement (rejects ``"safe"`` in ``"unsafe"``;
       accepts ``"safe."``)
- M2 — checklist DSL word-boundary MFT/DIR matching
- M4 — parse_checklist_spec named-test error for non-list prompts
- M5 — `_validate_run_id` rejection at CLI boundary
- M6 — evidence file 16 MiB cap
- L1-L6 — boundary tests
"""
from __future__ import annotations

import json
import os
import platform

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.commands.eval import app
from soup_cli.eval.calibrate import (
    PairwiseJudgement,
    conformal_threshold,
    fit_position_bias,
)
from soup_cli.utils.behavior_battery import (
    _agreement_rate,
    classify_behavior_score,
    compute_behavior_diff,
)
from soup_cli.utils.checklist_dsl import (
    CheckListSpec,
    CheckListTest,
    parse_checklist_spec,
    run_checklist_spec,
)
from soup_cli.utils.irt import (
    ItemDifficulty,
    pick_irt_subset,
)

# ─── H4: word-boundary agreement (review fix) ───


class TestWordBoundaryAgreement:
    def test_safe_does_not_match_unsafe(self):
        # "safe" must NOT match "unsafe" — earlier whitespace-tokenised
        # version would have failed (since tokens are ["unsafe"]).
        responses = ["unsafe"] * 5
        oracle = ["safe"] * 5
        rate = _agreement_rate(responses, oracle)
        assert rate == 0.0

    def test_safe_matches_safe_with_period(self):
        # H4 fix: "safe." must match oracle "safe" — earlier whitespace
        # tokeniser would have failed (since tokens are ["safe."]).
        responses = ["safe."] * 5
        oracle = ["safe"] * 5
        rate = _agreement_rate(responses, oracle)
        assert rate == 1.0

    def test_safe_matches_in_sentence(self):
        responses = ["the answer is safe enough"]
        oracle = ["safe"]
        rate = _agreement_rate(responses, oracle)
        assert rate == 1.0

    def test_regression_detected_word_boundary(self):
        # Combined H4 fix: post-responses say "unsafe" → oracle "safe" must
        # report MAJOR (was MISSING when substring match always passed).
        r = compute_behavior_diff(
            run_id="r", battery="xstest",
            pre_responses=["safe answer"] * 10,
            post_responses=["unsafe answer"] * 10,
            oracle=["safe"] * 10,
        )
        assert r.overall == "MAJOR"
        assert r.delta < -0.5


# ─── M2: checklist DSL word-boundary MFT/DIR ───


class TestChecklistWordBoundary:
    def test_mft_word_boundary_rejects_substring(self):
        t = CheckListTest(
            name="sand-test", kind="mft",
            prompts=("Some prompt",), expected=("and",),
        )
        spec = CheckListSpec(tests=(t,))
        # "sand" should NOT match "and" — was passing under substring.
        report = run_checklist_spec(spec, evidence={
            "sand-test": ["I see sand on the beach."],
        })
        assert report.results[0].verdict == "MAJOR"
        assert report.results[0].passed == 0

    def test_mft_word_boundary_accepts_word(self):
        t = CheckListTest(
            name="and-test", kind="mft",
            prompts=("Some prompt",), expected=("and",),
        )
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence={
            "and-test": ["I see and you do too."],
        })
        assert report.results[0].verdict == "OK"
        assert report.results[0].passed == 1

    def test_mft_word_boundary_with_punctuation(self):
        t = CheckListTest(
            name="and-test", kind="mft",
            prompts=("Some prompt",), expected=("yes",),
        )
        spec = CheckListSpec(tests=(t,))
        report = run_checklist_spec(spec, evidence={
            "and-test": ["yes, of course"],
        })
        assert report.results[0].verdict == "OK"


# ─── M4: parse_checklist_spec named-index error ───


class TestParseChecklistNamedError:
    def test_non_list_prompts_names_index(self):
        with pytest.raises(ValueError, match=r"tests\[0\]\.prompts"):
            parse_checklist_spec({
                "tests": [{"name": "t", "kind": "mft",
                           "prompts": "not a list", "expected": ["a"]}]
            })

    def test_non_list_expected_names_index(self):
        with pytest.raises(ValueError, match=r"tests\[1\]\.expected"):
            parse_checklist_spec({
                "tests": [
                    {"name": "t0", "kind": "mft",
                     "prompts": ["p"], "expected": ["a"]},
                    {"name": "t1", "kind": "mft",
                     "prompts": ["q"], "expected": "not a list"},
                ]
            })


# ─── M5: CLI run_id validation ───


class TestCliRunIdValidation:
    def test_behavior_rejects_null_byte_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "behavior", "evil\x00", "--battery", "xstest",
        ])
        assert result.exit_code != 0

    def test_capability_rejects_empty_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "capability", "", "--suite", "fast",
        ])
        assert result.exit_code != 0

    def test_behavior_rejects_oversize_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "behavior", "a" * 300, "--battery", "xstest",
        ])
        assert result.exit_code != 0


# ─── M6: evidence file 16 MiB cap ───


class TestEvidenceCap:
    @pytest.mark.skipif(
        platform.system() == "Windows" and not os.environ.get("CI"),
        reason="Slow to create 17 MiB file on Windows local",
    )
    def test_behavior_evidence_oversize_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        big = tmp_path / "huge.json"
        # Write 17 MiB of JSON (just one big string field — valid JSON but
        # over the 16 MiB cap).
        payload = '{"junk": "' + "a" * (17 * 1024 * 1024) + '"}'
        big.write_text(payload, encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(app, [
            "behavior", "r1", "--battery", "xstest",
            "--evidence", str(big),
        ])
        assert result.exit_code != 0


# ─── L5: boundary tests ───


class TestBoundaries:
    def test_classify_behavior_score_zero(self):
        assert classify_behavior_score(0.0) == "MAJOR"

    def test_classify_behavior_score_one(self):
        assert classify_behavior_score(1.0) == "OK"

    def test_classify_behavior_score_exact_85(self):
        assert classify_behavior_score(0.85) == "OK"

    def test_classify_behavior_score_just_below_85(self):
        assert classify_behavior_score(0.849999) == "MINOR"

    def test_classify_behavior_score_exact_60(self):
        assert classify_behavior_score(0.60) == "MINOR"

    def test_classify_behavior_score_just_below_60(self):
        assert classify_behavior_score(0.599999) == "MAJOR"

    def test_fit_position_bias_all_ties_returns_zero(self):
        judgements = [
            PairwiseJudgement(
                prompt_id=f"p{i}", first_winner="tie",
                second_winner="tie", oracle="tie",
            )
            for i in range(10)
        ]
        # Even with all ties counted in `total`, signed_sum=0 → bias=0.
        bias = fit_position_bias(judgements)
        assert bias == 0.0

    def test_conformal_threshold_extremes(self):
        # alpha=0.0 → min
        scores = [0.0, 1.0]
        assert conformal_threshold(scores, alpha=0.0) == 0.0
        # alpha=1.0 → max
        assert conformal_threshold(scores, alpha=1.0) == 1.0

    def test_compute_behavior_diff_delta_at_minor_threshold(self):
        # Pre value 1.0, post value 0.89 → delta = -0.11 → just past the
        # -0.10 OK→MINOR boundary.
        r = compute_behavior_diff(
            run_id="r", battery="xstest",
            pre_responses=["safe"] * 100,
            post_responses=(["safe"] * 89) + (["unsafe"] * 11),
            oracle=["safe"] * 100,
        )
        assert r.delta < -0.10
        # Post value 0.89 → MINOR; combined with regression → not necessarily
        # MAJOR yet.
        assert r.overall in ("MINOR", "MAJOR")

    def test_pick_irt_subset_total_one_tiny_floors_at_one(self):
        difficulty = (
            ItemDifficulty(item_id="q1", difficulty=0.0, info=1.0),
        )
        plan = pick_irt_subset(difficulty, size="tiny")
        assert len(plan.item_ids) == 1
        assert plan.cost_ratio == 1.0

    def test_pick_irt_subset_total_three_tiny_keeps_at_least_one(self):
        difficulty = tuple(
            ItemDifficulty(item_id=f"q{i}", difficulty=0.0, info=1.0)
            for i in range(3)
        )
        plan = pick_irt_subset(difficulty, size="tiny")
        # 10% of 3 = 0.3 → max(1, round(0.3)) = 1.
        assert len(plan.item_ids) == 1


# ─── L6: dedup helper actually used ───


class TestDeduplicatedWriteHelper:
    def test_capability_uses_atomic_write(self, tmp_path, monkeypatch):
        # Smoke test the L6 dedup helper is used by capability output.
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "cap.json"
        runner = CliRunner()
        result = runner.invoke(app, [
            "capability", "test", "--suite", "fast",
            "--output", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert data["suite"] == "fast"

    def test_irt_uses_atomic_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "r.jsonl"
        p.write_text("\n".join(
            json.dumps({"item_id": f"q{i % 5}", "correct": i % 2 == 0})
            for i in range(50)
        ))
        out = tmp_path / "plan.json"
        runner = CliRunner()
        result = runner.invoke(app, [
            "irt-subset", str(p), "--size", "small", "--output", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert data["size"] == "small"

    def test_checklist_uses_atomic_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.safe_dump({
            "tests": [{"name": "t1", "kind": "mft",
                       "prompts": ["p"], "expected": ["a"]}]
        }))
        out = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(app, [
            "checklist", str(p), "--output", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert "overall" in data


# ─── Source-grep regression guards ───


class TestSourceWiring:
    def test_behavior_battery_uses_traversable_path(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "behavior_battery.py"
        )
        text = src.read_text(encoding="utf-8")
        # H1 fix — must use as_file + Traversable / op, not os.path.join.
        assert "as_file" in text
        # The OLD bad pattern (os.path.join on stringified pkg_root) is gone.
        assert "os.path.join(str(pkg_root)" not in text

    def test_checklist_uses_shared_helper(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "checklist_dsl.py"
        )
        text = src.read_text(encoding="utf-8")
        # H2 fix — must use the shared helper.
        assert "enforce_under_cwd_and_no_symlink" in text

    def test_irt_uses_shared_helper(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "irt.py"
        )
        text = src.read_text(encoding="utf-8")
        # H3 fix — must use the shared helper.
        assert "enforce_under_cwd_and_no_symlink" in text
        # Must stream via .open(), not .read_text() (the old pattern that
        # materialised 256 MiB into RAM).
        assert ".read_text(encoding=\"utf-8\")" not in text

    def test_cli_uses_dedup_helpers(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "commands" / "_eval_v0650.py"
        )
        text = src.read_text(encoding="utf-8")
        # L6 dedup — _write_json_output + _read_evidence_json helpers.
        assert "_write_json_output" in text
        assert "_read_evidence_json" in text
        # M5 — _validate_run_id is the gate.
        assert "_validate_run_id" in text


# ─── H-NEW-1: O_NOFOLLOW + fstat (wave-2 review fix) ───


class TestONofollowWiring:
    def test_checklist_uses_o_nofollow(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "checklist_dsl.py"
        )
        text = src.read_text(encoding="utf-8")
        # H-NEW-1 fix: must use O_NOFOLLOW + fstat (no more double-lstat).
        assert "O_NOFOLLOW" in text
        assert "os.fstat(fd)" in text

    def test_irt_uses_o_nofollow(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "irt.py"
        )
        text = src.read_text(encoding="utf-8")
        assert "O_NOFOLLOW" in text
        assert "os.fstat(fd)" in text

    def test_eval_v0650_uses_o_nofollow(self):
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "commands" / "_eval_v0650.py"
        )
        text = src.read_text(encoding="utf-8")
        # H-NEW-2 fix.
        assert "O_NOFOLLOW" in text
        assert "os.fstat(fd)" in text


# ─── M-NEW-2: irt _MAX_ROWS bounds TOTAL iteration ───


class TestIrtRowsCap:
    def test_malformed_rows_count_toward_cap(self, tmp_path, monkeypatch):
        # Lots of malformed lines (>1M) — must still exit via the cap,
        # NOT stream to completion (review M-NEW-2 fix).
        from soup_cli.utils.irt import _MAX_ROWS, load_response_rows

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "huge.jsonl"
        # Write _MAX_ROWS + 100 malformed lines. Use binary write to keep
        # the file small enough not to trip the 256 MiB cap.
        lines = b"not json\n" * (_MAX_ROWS + 100)
        p.write_bytes(lines)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="cap"):
            load_response_rows(str(p))


# ─── M-NEW-3: INV empty-string normalisation fix ───


class TestInvEmptyStringRejected:
    def test_inv_all_whitespace_responses_rejected(self):
        from soup_cli.utils.checklist_dsl import _inv_pass

        # All responses are whitespace → normalise to {""} (len=1) — the
        # OLD code returned True (silent INV pass). Now must return False.
        assert _inv_pass(["", "   ", " \t "], expected=()) is False


# ─── L-NEW-3: load_response_rows skip-count WARNING is emitted ───


class TestLoadResponseRowsWarnsOnSkip:
    def test_warning_emitted_on_malformed(self, tmp_path, monkeypatch, caplog):
        import logging

        from soup_cli.utils.irt import load_response_rows

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "responses.jsonl"
        p.write_text(
            '{"item_id": "q1", "correct": true}\n'
            'not json\n'
            '{"item_id": "q2", "correct": false}\n'
        )
        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.irt"):
            rows = load_response_rows(str(p))
        assert len(rows) == 2
        assert any("skipped" in rec.message.lower() for rec in caplog.records)

    def test_no_warning_when_clean(self, tmp_path, monkeypatch, caplog):
        import logging

        from soup_cli.utils.irt import load_response_rows

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "responses.jsonl"
        p.write_text(
            '{"item_id": "q1", "correct": true}\n'
            '{"item_id": "q2", "correct": false}\n'
        )
        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.irt"):
            rows = load_response_rows(str(p))
        assert len(rows) == 2
        # No skip messages.
        assert not any(
            "skipped" in rec.message.lower() for rec in caplog.records
        )
