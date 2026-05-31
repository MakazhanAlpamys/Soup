"""v0.69.0 Part E — Brain-rot detector (arXiv 2510.13928)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import brain_rot


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Verdict allowlist
# -----------------------------------------------------------------------------


class TestBrainRotVerdicts:
    def test_exact(self) -> None:
        assert brain_rot.BRAIN_ROT_VERDICTS == ("OK", "MINOR", "MAJOR")

    def test_immutable(self) -> None:
        # Tuples reject item assignment with TypeError.
        with pytest.raises(TypeError):
            brain_rot.BRAIN_ROT_VERDICTS[0] = "EVIL"  # type: ignore[index]


class TestClassifyBoundaries:
    """TDD review H1 — exact-threshold boundary tests at ± epsilon."""

    def test_just_below_ok_is_minor(self) -> None:
        assert brain_rot.classify_brain_rot(0.8499999) == "MINOR"

    def test_just_below_minor_is_major(self) -> None:
        assert brain_rot.classify_brain_rot(0.5999999) == "MAJOR"

    def test_exact_minor_threshold(self) -> None:
        assert brain_rot.classify_brain_rot(0.60) == "MINOR"

    def test_exact_ok_threshold(self) -> None:
        assert brain_rot.classify_brain_rot(0.85) == "OK"


# -----------------------------------------------------------------------------
# Classify
# -----------------------------------------------------------------------------


class TestClassifyBrainRot:
    def test_high_score_is_ok(self) -> None:
        # High score = healthy data, low brain-rot.
        assert brain_rot.classify_brain_rot(0.9) == "OK"
        assert brain_rot.classify_brain_rot(0.85) == "OK"

    def test_mid_score_is_minor(self) -> None:
        assert brain_rot.classify_brain_rot(0.70) == "MINOR"
        assert brain_rot.classify_brain_rot(0.6) == "MINOR"

    def test_low_score_is_major(self) -> None:
        assert brain_rot.classify_brain_rot(0.59) == "MAJOR"
        assert brain_rot.classify_brain_rot(0.0) == "MAJOR"

    def test_boundary_thresholds(self) -> None:
        # Match v0.26 / v0.56 / v0.65 thresholds: >= 0.85 OK / >= 0.60 MINOR.
        assert brain_rot.classify_brain_rot(0.849) == "MINOR"
        assert brain_rot.classify_brain_rot(0.599) == "MAJOR"
        assert brain_rot.classify_brain_rot(1.0) == "OK"

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.classify_brain_rot(True)

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.classify_brain_rot(float("nan"))

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.classify_brain_rot(float("inf"))

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.classify_brain_rot(1.5)
        with pytest.raises(ValueError):
            brain_rot.classify_brain_rot(-0.1)

    def test_non_number(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.classify_brain_rot("0.5")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Triviality + popularity heuristics
# -----------------------------------------------------------------------------


class TestScoreTriviality:
    def test_substantive_low(self) -> None:
        # Long, varied, substantive text → low triviality.
        text = (
            "The mitochondrion is the powerhouse of the cell because it "
            "converts nutrients into ATP through oxidative phosphorylation."
        )
        score = brain_rot.score_triviality(text)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_trivial_high(self) -> None:
        # Short, repetitive, exclamation-heavy → high triviality.
        text = "lol!!!! omg!!!! lol omg!!! lol!!!"
        score = brain_rot.score_triviality(text)
        assert score > 0.5

    def test_empty_returns_one(self) -> None:
        assert brain_rot.score_triviality("") == 1.0

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.score_triviality(42)  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.score_triviality(True)


class TestScorePopularitySignal:
    def test_substantive_low(self) -> None:
        text = "Detailed scientific explanation of photosynthesis cycles."
        score = brain_rot.score_popularity_signal(text)
        assert score < 0.5

    def test_slop_high(self) -> None:
        text = "click here for the top 10 you won't believe what happened next"
        score = brain_rot.score_popularity_signal(text)
        assert score > 0.5

    def test_empty_zero(self) -> None:
        assert brain_rot.score_popularity_signal("") == 0.0

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.score_popularity_signal(42)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Per-row + dataset scoring
# -----------------------------------------------------------------------------


class TestScoreRowBrainRot:
    def test_substantive(self) -> None:
        row = {"text": "Long substantive paragraph with diverse vocabulary."}
        score = brain_rot.score_row_brain_rot(row)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_slop(self) -> None:
        row = {"text": "lol!!!! omg!!! top 10 you won't believe"}
        score = brain_rot.score_row_brain_rot(row)
        assert score < 0.5

    def test_missing_text_returns_zero(self) -> None:
        # No text signal → score 0.0 (cannot judge → treat as MAJOR).
        assert brain_rot.score_row_brain_rot({}) == 0.0

    def test_non_dict_rejected(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.score_row_brain_rot("not a dict")  # type: ignore[arg-type]


class TestBrainRotReport:
    def test_frozen(self) -> None:
        report = brain_rot.BrainRotReport(
            num_rows=10,
            mean_score=0.5,
            num_major=2,
            num_minor=3,
            num_ok=5,
            overall_verdict="MINOR",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.mean_score = 0.9  # type: ignore[misc]

    def test_validates_counts(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.BrainRotReport(
                num_rows=-1,
                mean_score=0.5,
                num_major=0,
                num_minor=0,
                num_ok=0,
                overall_verdict="OK",
            )

    def test_validates_verdict(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.BrainRotReport(
                num_rows=1,
                mean_score=0.5,
                num_major=0,
                num_minor=0,
                num_ok=1,
                overall_verdict="BOGUS",
            )

    def test_mean_score_finite(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.BrainRotReport(
                num_rows=1,
                mean_score=float("nan"),
                num_major=0,
                num_minor=0,
                num_ok=1,
                overall_verdict="OK",
            )

    def test_mean_score_range(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.BrainRotReport(
                num_rows=1,
                mean_score=1.5,
                num_major=0,
                num_minor=0,
                num_ok=1,
                overall_verdict="OK",
            )


class TestScoreDatasetBrainRot:
    def test_clean_dataset(self) -> None:
        rows = [
            {"text": "Detailed scientific explanation of photosynthesis."},
            {"text": "Comprehensive overview of cellular respiration cycles."},
        ]
        report = brain_rot.score_dataset_brain_rot(rows)
        assert report.num_rows == 2
        assert math.isfinite(report.mean_score)
        assert report.overall_verdict in brain_rot.BRAIN_ROT_VERDICTS

    def test_sloppy_dataset(self) -> None:
        rows = [{"text": "lol!!!! omg top 10"} for _ in range(5)]
        report = brain_rot.score_dataset_brain_rot(rows)
        # Slop-heavy → MAJOR or MINOR
        assert report.overall_verdict in ("MAJOR", "MINOR")

    def test_empty(self) -> None:
        report = brain_rot.score_dataset_brain_rot([])
        assert report.num_rows == 0
        assert report.mean_score == 0.0
        assert report.overall_verdict == "MAJOR"

    def test_non_iterable(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.score_dataset_brain_rot(42)  # type: ignore[arg-type]

    def test_non_dict_row_skipped(self) -> None:
        rows = [
            {"text": "good substantive content"},
            "not a dict",
            {"text": "more good content"},
        ]
        report = brain_rot.score_dataset_brain_rot(rows)
        # Only 2 dict rows count.
        assert report.num_rows == 2

    def test_overall_minor_band(self) -> None:
        # Custom mix to land in MINOR band.
        rows = (
            [{"text": "lol!!! omg!!!"} for _ in range(3)]
            + [
                {"text": "Detailed scientific overview of cellular respiration."}
                for _ in range(2)
            ]
        )
        report = brain_rot.score_dataset_brain_rot(rows)
        assert math.isfinite(report.mean_score)


# -----------------------------------------------------------------------------
# refuse_if_rotten
# -----------------------------------------------------------------------------


class TestRefuseIfRotten:
    def test_clean_passes(self) -> None:
        rows = [
            {"text": "Detailed scientific overview of cellular biology."}
            for _ in range(3)
        ]
        # Should not raise.
        brain_rot.refuse_if_rotten(rows, max_major_fraction=0.5)

    def test_too_rotten_raises(self) -> None:
        rows = [{"text": "lol!!! omg!!!"} for _ in range(5)]
        with pytest.raises(ValueError, match="brain.?rot"):
            brain_rot.refuse_if_rotten(rows, max_major_fraction=0.1)

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            brain_rot.refuse_if_rotten([], max_major_fraction=1.5)

    def test_bool_threshold(self) -> None:
        with pytest.raises(TypeError):
            brain_rot.refuse_if_rotten([], max_major_fraction=True)


# -----------------------------------------------------------------------------
# CLI: `soup data brain-rot`
# -----------------------------------------------------------------------------


class TestBrainRotCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["data", "brain-rot", "--help"])
        assert result.exit_code == 0, result.output
        assert "brain" in result.output.lower()

    def test_clean(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = _write(
            tmp_path / "clean.jsonl",
            "\n".join(
                json.dumps({"text": "Detailed scientific exposition number " + str(i)})
                for i in range(5)
            )
            + "\n",
        )
        runner = CliRunner()
        result = runner.invoke(app, ["data", "brain-rot", str(path)])
        assert result.exit_code == 0, result.output

    def test_sloppy_exits_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = _write(
            tmp_path / "slop.jsonl",
            "\n".join(json.dumps({"text": "lol!!! omg!!!"}) for _ in range(10))
            + "\n",
        )
        runner = CliRunner()
        result = runner.invoke(
            app, ["data", "brain-rot", str(path), "--strict"]
        )
        assert result.exit_code == 3

    def test_missing_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["data", "brain-rot", "nope.jsonl"])
        assert result.exit_code != 0

    def test_outside_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "d.jsonl", json.dumps({"text": "x"}) + "\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        runner = CliRunner()
        result = runner.invoke(
            app, ["data", "brain-rot", str(outside / "d.jsonl")]
        )
        assert result.exit_code != 0

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = _write(
            tmp_path / "real.jsonl", json.dumps({"text": "x"}) + "\n"
        )
        link = tmp_path / "link.jsonl"
        os.symlink(str(target), str(link))
        runner = CliRunner()
        result = runner.invoke(app, ["data", "brain-rot", str(link)])
        assert result.exit_code != 0


# -----------------------------------------------------------------------------
# Source wiring
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_heavy_imports(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "brain_rot.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport sentence_transformers",
        ):
            assert forbidden not in src

    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        major_minor = tuple(int(x) for x in __version__.split(".")[:2])
        assert major_minor >= (0, 69)
