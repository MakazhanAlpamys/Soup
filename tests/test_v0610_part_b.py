"""Tests for v0.61.0 Part B — Unlearning eval suite (TOFU / MUSE / WMDP).

Coverage:
- ``UnlearnMetric`` / ``UnlearnReport`` frozen dataclasses.
- ``classify_unlearn_score`` OK / MINOR / MAJOR thresholds.
- ``compute_forget_quality`` / ``compute_model_utility`` / ``compute_priv_leak`` kernels.
- ``run_unlearn_eval`` orchestrator.
- ``BENCHMARKS`` allowlist + closed name validation.
- ``soup eval unlearning`` CLI smoke.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

# ---------- Module surface ----------


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.unlearning_eval import (
            BENCHMARKS,
            VERDICTS,
            UnlearnMetric,
            UnlearnReport,
            classify_unlearn_score,
            compute_forget_quality,
            compute_model_utility,
            compute_priv_leak,
            run_unlearn_eval,
            validate_benchmark_name,
        )
        assert callable(classify_unlearn_score)
        assert callable(compute_forget_quality)
        assert callable(compute_model_utility)
        assert callable(compute_priv_leak)
        assert callable(run_unlearn_eval)
        assert callable(validate_benchmark_name)
        assert dataclasses.is_dataclass(UnlearnMetric)
        assert dataclasses.is_dataclass(UnlearnReport)
        assert isinstance(BENCHMARKS, frozenset)
        assert isinstance(VERDICTS, tuple)

    def test_benchmarks_exact(self):
        from soup_cli.utils.unlearning_eval import BENCHMARKS

        assert BENCHMARKS == frozenset({"tofu", "muse", "wmdp"})

    def test_verdicts_exact(self):
        from soup_cli.utils.unlearning_eval import VERDICTS

        assert VERDICTS == ("OK", "MINOR", "MAJOR")


# ---------- classify_unlearn_score ----------


class TestClassifyUnlearnScore:
    def test_ok_boundary(self):
        from soup_cli.utils.unlearning_eval import classify_unlearn_score

        assert classify_unlearn_score(0.85) == "OK"
        assert classify_unlearn_score(1.0) == "OK"

    def test_minor_band(self):
        from soup_cli.utils.unlearning_eval import classify_unlearn_score

        assert classify_unlearn_score(0.60) == "MINOR"
        assert classify_unlearn_score(0.84) == "MINOR"

    def test_major_band(self):
        from soup_cli.utils.unlearning_eval import classify_unlearn_score

        assert classify_unlearn_score(0.0) == "MAJOR"
        assert classify_unlearn_score(0.59) == "MAJOR"

    def test_bool_rejected(self):
        from soup_cli.utils.unlearning_eval import classify_unlearn_score

        with pytest.raises(TypeError):
            classify_unlearn_score(True)

    def test_non_finite_rejected(self):
        from soup_cli.utils.unlearning_eval import classify_unlearn_score

        with pytest.raises(ValueError):
            classify_unlearn_score(float("nan"))

        with pytest.raises(ValueError):
            classify_unlearn_score(float("inf"))

    def test_out_of_range_rejected(self):
        from soup_cli.utils.unlearning_eval import classify_unlearn_score

        with pytest.raises(ValueError):
            classify_unlearn_score(-0.1)

        with pytest.raises(ValueError):
            classify_unlearn_score(1.1)


# ---------- Benchmark name validation ----------


class TestValidateBenchmarkName:
    def test_happy_path(self):
        from soup_cli.utils.unlearning_eval import validate_benchmark_name

        assert validate_benchmark_name("tofu") == "tofu"
        assert validate_benchmark_name("MUSE") == "muse"
        assert validate_benchmark_name("WMDP") == "wmdp"

    def test_unknown_rejected(self):
        from soup_cli.utils.unlearning_eval import validate_benchmark_name

        with pytest.raises(ValueError, match="unknown"):
            validate_benchmark_name("zzz")

    def test_bool_rejected(self):
        from soup_cli.utils.unlearning_eval import validate_benchmark_name

        with pytest.raises(TypeError):
            validate_benchmark_name(True)

    def test_null_byte_rejected(self):
        from soup_cli.utils.unlearning_eval import validate_benchmark_name

        with pytest.raises(ValueError):
            validate_benchmark_name("tofu\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.unlearning_eval import validate_benchmark_name

        with pytest.raises(ValueError):
            validate_benchmark_name("a" * 100)

    def test_empty_rejected(self):
        from soup_cli.utils.unlearning_eval import validate_benchmark_name

        with pytest.raises(ValueError):
            validate_benchmark_name("")


# ---------- Metric kernels ----------


class TestComputeForgetQuality:
    def test_perfect_forget(self):
        # If post-unlearn loss on forget set is HIGH and pre was LOW,
        # forget quality = 1.0.
        from soup_cli.utils.unlearning_eval import compute_forget_quality

        score = compute_forget_quality(pre_loss=0.5, post_loss=5.0)
        assert score == 1.0

    def test_no_forget(self):
        # If post-loss == pre-loss, quality is 0.
        from soup_cli.utils.unlearning_eval import compute_forget_quality

        score = compute_forget_quality(pre_loss=2.0, post_loss=2.0)
        assert score == 0.0

    def test_partial_forget(self):
        from soup_cli.utils.unlearning_eval import compute_forget_quality

        score = compute_forget_quality(pre_loss=1.0, post_loss=2.0)
        assert 0.0 < score < 1.0

    def test_bool_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_forget_quality

        with pytest.raises(TypeError):
            compute_forget_quality(pre_loss=True, post_loss=2.0)

        with pytest.raises(TypeError):
            compute_forget_quality(pre_loss=1.0, post_loss=True)

    def test_non_finite_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_forget_quality

        with pytest.raises(ValueError):
            compute_forget_quality(pre_loss=float("nan"), post_loss=1.0)

    def test_negative_loss_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_forget_quality

        with pytest.raises(ValueError):
            compute_forget_quality(pre_loss=-0.5, post_loss=1.0)


class TestComputeModelUtility:
    def test_perfect_utility(self):
        # If retain accuracy is preserved (post == pre), utility = 1.0.
        from soup_cli.utils.unlearning_eval import compute_model_utility

        score = compute_model_utility(pre_acc=0.8, post_acc=0.8)
        assert score == 1.0

    def test_no_utility(self):
        # If retain accuracy drops to 0, utility = 0.
        from soup_cli.utils.unlearning_eval import compute_model_utility

        score = compute_model_utility(pre_acc=0.8, post_acc=0.0)
        assert score == 0.0

    def test_partial_drop(self):
        from soup_cli.utils.unlearning_eval import compute_model_utility

        score = compute_model_utility(pre_acc=0.8, post_acc=0.6)
        assert 0.0 < score < 1.0

    def test_bool_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_model_utility

        with pytest.raises(TypeError):
            compute_model_utility(pre_acc=True, post_acc=0.5)

    def test_out_of_range_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_model_utility

        with pytest.raises(ValueError):
            compute_model_utility(pre_acc=1.5, post_acc=0.5)

        with pytest.raises(ValueError):
            compute_model_utility(pre_acc=0.5, post_acc=-0.1)


class TestComputePrivLeak:
    def test_no_leak(self):
        # Membership-inference AUC ≈ 0.5 → no leak.
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        score = compute_priv_leak(mia_auc=0.5)
        assert score >= 0.95  # very high "privacy preserved"

    def test_full_leak(self):
        # MIA AUC = 1.0 → adversary can perfectly distinguish forget vs holdout.
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        score = compute_priv_leak(mia_auc=1.0)
        assert score == 0.0

    def test_below_random(self):
        # AUC < 0.5 is still leak (adversary can invert).
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        score = compute_priv_leak(mia_auc=0.0)
        assert score == 0.0

    def test_bool_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        with pytest.raises(TypeError):
            compute_priv_leak(mia_auc=True)

    def test_out_of_range_rejected(self):
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        with pytest.raises(ValueError):
            compute_priv_leak(mia_auc=1.5)

    def test_boundary_zero_accepted(self):
        """Review L3 — exact lower boundary."""
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        # AUC=0.0 is in [0, 1] but reads as max distinguishable inverse.
        assert compute_priv_leak(mia_auc=0.0) == 0.0

    def test_boundary_one_accepted(self):
        """Review L3 — exact upper boundary."""
        from soup_cli.utils.unlearning_eval import compute_priv_leak

        assert compute_priv_leak(mia_auc=1.0) == 0.0


# ---------- UnlearnMetric / UnlearnReport ----------


class TestUnlearnMetric:
    def test_construct(self):
        from soup_cli.utils.unlearning_eval import UnlearnMetric

        m = UnlearnMetric(
            name="forget_quality",
            score=0.9,
            verdict="OK",
            evidence="pre=0.5, post=5.0",
        )
        assert m.name == "forget_quality"
        assert m.score == 0.9
        assert m.verdict == "OK"

    def test_frozen(self):
        from soup_cli.utils.unlearning_eval import UnlearnMetric

        m = UnlearnMetric(
            name="forget_quality",
            score=0.9,
            verdict="OK",
            evidence="",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.score = 0.5  # type: ignore

    def test_verdict_must_match_score(self):
        from soup_cli.utils.unlearning_eval import UnlearnMetric

        with pytest.raises(ValueError, match="disagrees"):
            UnlearnMetric(
                name="forget_quality",
                score=0.3,
                verdict="OK",
                evidence="",
            )


class TestUnlearnReport:
    def test_construct(self):
        from soup_cli.utils.unlearning_eval import UnlearnMetric, UnlearnReport

        report = UnlearnReport(
            run_id="test-run",
            benchmark="tofu",
            metrics=(
                UnlearnMetric(name="forget_quality", score=0.9, verdict="OK", evidence=""),
                UnlearnMetric(name="model_utility", score=0.95, verdict="OK", evidence=""),
                UnlearnMetric(name="priv_leak", score=0.9, verdict="OK", evidence=""),
            ),
            overall="OK",
            soup_version="0.61.0",
        )
        assert report.run_id == "test-run"
        assert report.benchmark == "tofu"
        assert len(report.metrics) == 3
        assert report.overall == "OK"

    def test_frozen(self):
        from soup_cli.utils.unlearning_eval import UnlearnMetric, UnlearnReport

        report = UnlearnReport(
            run_id="r",
            benchmark="tofu",
            metrics=(
                UnlearnMetric(name="forget_quality", score=0.9, verdict="OK", evidence=""),
            ),
            overall="OK",
            soup_version="0.61.0",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.overall = "MAJOR"  # type: ignore

    def test_to_dict(self):
        from soup_cli.utils.unlearning_eval import UnlearnMetric, UnlearnReport

        report = UnlearnReport(
            run_id="r",
            benchmark="tofu",
            metrics=(
                UnlearnMetric(name="forget_quality", score=0.9, verdict="OK", evidence="e"),
            ),
            overall="OK",
            soup_version="0.61.0",
        )
        d = report.to_dict()
        assert d["run_id"] == "r"
        assert d["benchmark"] == "tofu"
        assert d["overall"] == "OK"
        assert len(d["metrics"]) == 1
        # Round-trip via json
        s = json.dumps(d)
        assert json.loads(s) == d


# ---------- run_unlearn_eval ----------


class TestRunUnlearnEval:
    def test_happy_path(self):
        from soup_cli.utils.unlearning_eval import run_unlearn_eval

        report = run_unlearn_eval(
            run_id="test-run",
            benchmark="tofu",
            evidence={
                "forget_quality": {"pre_loss": 0.5, "post_loss": 5.0},
                "model_utility": {"pre_acc": 0.8, "post_acc": 0.78},
                "priv_leak": {"mia_auc": 0.51},
            },
        )
        assert report.run_id == "test-run"
        assert report.benchmark == "tofu"
        assert report.overall in ("OK", "MINOR", "MAJOR")

    def test_unknown_benchmark_rejected(self):
        from soup_cli.utils.unlearning_eval import run_unlearn_eval

        with pytest.raises(ValueError, match="unknown"):
            run_unlearn_eval(run_id="r", benchmark="zzz", evidence={})

    def test_missing_evidence_neutral(self):
        # Missing-evidence policy: neutral OK score (matches v0.56.0
        # diagnose-runner neutral_score policy).
        from soup_cli.utils.unlearning_eval import run_unlearn_eval

        report = run_unlearn_eval(run_id="r", benchmark="tofu", evidence={})
        # All metrics neutral OK
        assert report.overall == "OK"

    def test_overall_worst_case(self):
        from soup_cli.utils.unlearning_eval import run_unlearn_eval

        report = run_unlearn_eval(
            run_id="r",
            benchmark="tofu",
            evidence={
                "forget_quality": {"pre_loss": 1.0, "post_loss": 1.0},  # MAJOR
                "model_utility": {"pre_acc": 0.8, "post_acc": 0.8},  # OK
                "priv_leak": {"mia_auc": 0.5},  # OK
            },
        )
        assert report.overall == "MAJOR"

    def test_invalid_evidence_raises_loudly(self):
        """Review HIGH H3 — present-but-invalid evidence must raise
        instead of silently scoring OK."""
        from soup_cli.utils.unlearning_eval import run_unlearn_eval

        with pytest.raises(ValueError):
            run_unlearn_eval(
                run_id="r",
                benchmark="tofu",
                evidence={
                    "forget_quality": {"pre_loss": -1.0, "post_loss": 1.0},
                },
            )

    def test_partial_evidence_neutral_on_missing(self):
        """Review HIGH H3 — missing keys still produce neutral OK."""
        from soup_cli.utils.unlearning_eval import run_unlearn_eval

        report = run_unlearn_eval(
            run_id="r",
            benchmark="tofu",
            evidence={
                "forget_quality": {"pre_loss": 0.5, "post_loss": 5.0},
                # model_utility + priv_leak missing -> neutral
            },
        )
        assert report.overall == "OK"


# ---------- CLI ----------


class TestCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["eval", "unlearning", "--help"])
        assert result.exit_code == 0, result.output

    def test_unknown_benchmark_rejected(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["eval", "unlearning", "test-run", "--benchmark", "zzz"])
        assert result.exit_code != 0
        assert "unknown" in result.output.lower() or "invalid" in result.output.lower()

    def test_neutral_run(self, tmp_path):
        # Without evidence, every metric is neutral OK.
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            out = Path(fs) / "report.json"
            result = runner.invoke(app, [
                "eval", "unlearning", "test-run",
                "--benchmark", "tofu",
                "--output", str(out),
            ])
            assert result.exit_code == 0, result.output
            assert out.exists()
            data = json.loads(out.read_text())
            assert data["benchmark"] == "tofu"
            assert data["overall"] == "OK"

    def test_outside_cwd_output_rejected(self, tmp_path):
        """Review L8 — use tmp_path.parent so path is deterministically
        outside the isolated_filesystem on every platform."""
        runner = CliRunner()
        # The runner's isolated_filesystem cd's into a fresh subdir under
        # tmp_path; pointing --output at tmp_path itself is reliably
        # outside the new cwd regardless of OS or symlink layout.
        outside_target = str(tmp_path / "evil.json")
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [
                "eval", "unlearning", "test-run",
                "--benchmark", "tofu",
                "--output", outside_target,
            ])
            assert result.exit_code != 0


# ---------- Fixtures ----------


class TestLoadEvidenceFile:
    """v0.71.1 — cover the `soup eval unlearning --evidence` loader."""

    def test_happy_returns_dict(self, tmp_path, monkeypatch):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        monkeypatch.chdir(tmp_path)
        (tmp_path / "ev.json").write_text(
            json.dumps({"forget_quality": {"pre_loss": 1.0, "post_loss": 2.0}}),
            encoding="utf-8",
        )
        data = load_evidence_file("ev.json")
        assert isinstance(data, dict)
        assert "forget_quality" in data

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_evidence_file("nope.json")

    def test_non_string_path_rejected(self):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        with pytest.raises(ValueError):
            load_evidence_file(123)  # type: ignore[arg-type]

    def test_empty_path_rejected(self):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        with pytest.raises(ValueError):
            load_evidence_file("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        with pytest.raises(ValueError, match="null"):
            load_evidence_file("a\x00b.json")

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "ev.json").write_text("{}", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError, match="cwd"):
            load_evidence_file(str(outside / "ev.json"))

    def test_non_dict_root_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        monkeypatch.chdir(tmp_path)
        (tmp_path / "arr.json").write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_evidence_file("arr.json")

    @pytest.mark.skipif(
        sys.platform == "win32", reason="symlink creation needs admin on Windows"
    )
    def test_symlink_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.unlearning_eval import load_evidence_file

        monkeypatch.chdir(tmp_path)
        (tmp_path / "real.json").write_text("{}", encoding="utf-8")
        link = tmp_path / "link.json"
        os.symlink(tmp_path / "real.json", link)
        with pytest.raises(ValueError, match="symlink"):
            load_evidence_file("link.json")


class TestFixtures:
    def test_tofu_fixture_exists(self):
        from soup_cli.utils.unlearning_eval import get_fixture_path

        # TOFU should be bundled.
        p = get_fixture_path("tofu")
        assert p is not None

    def test_unknown_fixture_returns_none(self):
        from soup_cli.utils.unlearning_eval import get_fixture_path

        assert get_fixture_path("zzz") is None

    # v0.71.1 #195 — MUSE + WMDP bundled mini-fixtures.
    def test_muse_fixture_exists(self):
        from soup_cli.utils.unlearning_eval import get_fixture_path

        p = get_fixture_path("muse")
        assert p is not None
        assert p.is_file()

    def test_wmdp_fixture_exists(self):
        from soup_cli.utils.unlearning_eval import get_fixture_path

        p = get_fixture_path("wmdp")
        assert p is not None
        assert p.is_file()

    def test_all_benchmarks_resolve_a_fixture(self):
        from soup_cli.utils.unlearning_eval import BENCHMARKS, get_fixture_path

        for bench in BENCHMARKS:
            assert get_fixture_path(bench) is not None, bench

    def test_muse_fixture_is_valid_jsonl(self):
        from soup_cli.utils.unlearning_eval import get_fixture_path

        p = get_fixture_path("muse")
        assert p is not None
        lines = p.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        assert len(rows) >= 4
        # Each row carries a prompt/response pair + a forget/retain split.
        splits = {row["split"] for row in rows}
        assert "forget" in splits
        assert "retain" in splits
        for row in rows:
            assert isinstance(row["prompt"], str) and row["prompt"]
            assert isinstance(row["response"], str) and row["response"]

    def test_wmdp_fixture_is_valid_jsonl(self):
        from soup_cli.utils.unlearning_eval import get_fixture_path

        p = get_fixture_path("wmdp")
        assert p is not None
        lines = p.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        assert len(rows) >= 4
        splits = {row["split"] for row in rows}
        assert "forget" in splits
        assert "retain" in splits
        # WMDP rows are multiple-choice hazardous-knowledge probes.
        for row in rows:
            assert isinstance(row["prompt"], str) and row["prompt"]
            assert isinstance(row["response"], str) and row["response"]
        # Redaction-policy invariant (v0.71.1 #195): every forget-set row is a
        # REFUSED placeholder with its hazardous content redacted — Soup never
        # ships verbatim WMDP probes.
        forget_rows = [row for row in rows if row["split"] == "forget"]
        assert forget_rows
        for row in forget_rows:
            assert row["response"].startswith("REFUSED")
            assert "[redacted]" in row["prompt"]

    def test_metadata_fixtures_are_filenames_not_paths(self):
        from soup_cli.utils.unlearning_eval import _BENCHMARK_METADATA

        for name, meta in _BENCHMARK_METADATA.items():
            fixture = meta["fixture"]
            assert fixture, f"{name} fixture must be bundled (non-empty)"
            assert "/" not in fixture and "\\" not in fixture
