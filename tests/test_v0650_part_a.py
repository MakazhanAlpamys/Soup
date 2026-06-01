"""v0.65.0 Part A — Judge calibration tests (TDD-first).

Covers SCOPE/CJE-style bidirectional pairwise judging, position-bias offset
fitting, and conformal abstention thresholds. Refusal to use uncalibrated
judges in production scoring is wired through a runtime gate.
"""
from __future__ import annotations

import math

import pytest

from soup_cli.eval.calibrate import (
    JudgeCalibrationReport,
    PairwiseJudgement,
    classify_kl_delta,
    conformal_threshold,
    ensure_judge_calibrated,
    fit_position_bias,
    kl_divergence,
    load_judge_calibration,
    run_pairwise_calibration,
    write_judge_calibration,
)

# ─── PairwiseJudgement frozen dataclass ───


class TestPairwiseJudgement:
    def test_frozen(self):
        j = PairwiseJudgement(
            prompt_id="p1", first_winner="a", second_winner="a", oracle="a",
        )
        with pytest.raises(Exception):
            j.first_winner = "b"  # type: ignore[misc]

    def test_invalid_winner_first(self):
        with pytest.raises(ValueError, match="first_winner"):
            PairwiseJudgement(
                prompt_id="p", first_winner="X", second_winner="a", oracle="a",
            )

    def test_invalid_winner_second(self):
        with pytest.raises(ValueError, match="second_winner"):
            PairwiseJudgement(
                prompt_id="p", first_winner="a", second_winner="Z", oracle="a",
            )

    def test_invalid_oracle(self):
        with pytest.raises(ValueError, match="oracle"):
            PairwiseJudgement(
                prompt_id="p", first_winner="a", second_winner="b", oracle="x",
            )

    def test_null_byte_prompt_id(self):
        with pytest.raises(ValueError, match="null"):
            PairwiseJudgement(
                prompt_id="p\x00", first_winner="a", second_winner="b", oracle="a",
            )

    def test_empty_prompt_id(self):
        with pytest.raises(ValueError, match="prompt_id"):
            PairwiseJudgement(
                prompt_id="", first_winner="a", second_winner="a", oracle="a",
            )

    def test_accepts_tie(self):
        j = PairwiseJudgement(
            prompt_id="p", first_winner="tie", second_winner="tie", oracle="tie",
        )
        assert j.first_winner == "tie"


# ─── fit_position_bias ───


class TestFitPositionBias:
    def test_no_bias_when_consistent(self):
        judgements = [
            PairwiseJudgement(prompt_id=f"p{i}", first_winner="a",
                              second_winner="a", oracle="a")
            for i in range(10)
        ]
        bias = fit_position_bias(judgements)
        assert bias == 0.0

    def test_position_bias_when_first_always_wins(self):
        # Judge always picks the FIRST slot regardless of swap:
        #   arrangement 1: a, b -> judge picks "a" (first slot)
        #   arrangement 2 (swapped): b, a -> judge picks "b" (first slot)
        judgements = []
        for i in range(10):
            oracle = "a" if i % 2 == 0 else "b"
            judgements.append(PairwiseJudgement(
                prompt_id=f"p{i}",
                first_winner="a",
                second_winner="b",
                oracle=oracle,
            ))
        bias = fit_position_bias(judgements)
        # judge always picks the first slot → strong positive position bias
        assert bias > 0.5

    def test_empty_judgements(self):
        with pytest.raises(ValueError, match="judgements"):
            fit_position_bias([])

    def test_non_iterable(self):
        with pytest.raises(TypeError):
            fit_position_bias(42)  # type: ignore[arg-type]

    def test_returns_finite(self):
        judgements = [
            PairwiseJudgement(prompt_id=f"p{i}",
                              first_winner="a", second_winner="b", oracle="a")
            for i in range(5)
        ]
        bias = fit_position_bias(judgements)
        assert math.isfinite(bias)

    def test_bias_in_range(self):
        # Position-bias coefficient should be in [-1, 1].
        judgements = [
            PairwiseJudgement(prompt_id=f"p{i}",
                              first_winner="a", second_winner="b", oracle="a")
            for i in range(10)
        ]
        bias = fit_position_bias(judgements)
        assert -1.0 <= bias <= 1.0


# ─── conformal_threshold ───


class TestConformalThreshold:
    def test_basic(self):
        # 100 scores 0.0..0.99; alpha=0.1 means we keep top 90%, so threshold
        # is the 10th percentile.
        scores = [i / 100.0 for i in range(100)]
        t = conformal_threshold(scores, alpha=0.1)
        # Should be roughly 0.10 (10th percentile).
        assert 0.05 <= t <= 0.15

    def test_alpha_zero_keeps_all(self):
        scores = [0.5, 0.6, 0.7]
        t = conformal_threshold(scores, alpha=0.0)
        assert t == min(scores)

    def test_alpha_one_keeps_none(self):
        scores = [0.5, 0.6, 0.7]
        t = conformal_threshold(scores, alpha=1.0)
        assert t == max(scores)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            conformal_threshold([0.5], alpha=-0.1)

    def test_invalid_alpha_above_one(self):
        with pytest.raises(ValueError, match="alpha"):
            conformal_threshold([0.5], alpha=1.1)

    def test_invalid_alpha_nan(self):
        with pytest.raises(ValueError, match="alpha"):
            conformal_threshold([0.5], alpha=float("nan"))

    def test_invalid_alpha_bool(self):
        with pytest.raises(ValueError, match="alpha"):
            conformal_threshold([0.5], alpha=True)

    def test_empty_scores(self):
        with pytest.raises(ValueError, match="scores"):
            conformal_threshold([], alpha=0.1)

    def test_non_finite_score(self):
        with pytest.raises(ValueError, match="finite"):
            conformal_threshold([0.5, float("inf")], alpha=0.1)

    def test_score_out_of_range_high(self):
        with pytest.raises(ValueError, match="range"):
            conformal_threshold([0.5, 1.5], alpha=0.1)

    def test_score_out_of_range_low(self):
        with pytest.raises(ValueError, match="range"):
            conformal_threshold([-0.1, 0.5], alpha=0.1)


# ─── JudgeCalibrationReport ───


class TestJudgeCalibrationReport:
    def test_frozen(self):
        r = JudgeCalibrationReport(
            position_bias=0.05,
            conformal_threshold=0.3,
            agreement_rate=0.8,
            num_pairs=10,
            calibrated=True,
        )
        with pytest.raises(Exception):
            r.position_bias = 0.0  # type: ignore[misc]

    def test_invalid_bias(self):
        with pytest.raises(ValueError, match="position_bias"):
            JudgeCalibrationReport(
                position_bias=2.0,
                conformal_threshold=0.3,
                agreement_rate=0.8,
                num_pairs=10,
                calibrated=True,
            )

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="conformal_threshold"):
            JudgeCalibrationReport(
                position_bias=0.0,
                conformal_threshold=2.0,
                agreement_rate=0.8,
                num_pairs=10,
                calibrated=True,
            )

    def test_invalid_agreement(self):
        with pytest.raises(ValueError, match="agreement_rate"):
            JudgeCalibrationReport(
                position_bias=0.0,
                conformal_threshold=0.3,
                agreement_rate=1.5,
                num_pairs=10,
                calibrated=True,
            )

    def test_invalid_num_pairs(self):
        with pytest.raises(ValueError, match="num_pairs"):
            JudgeCalibrationReport(
                position_bias=0.0,
                conformal_threshold=0.3,
                agreement_rate=0.8,
                num_pairs=-1,
                calibrated=True,
            )

    def test_bool_num_pairs(self):
        with pytest.raises(ValueError, match="num_pairs"):
            JudgeCalibrationReport(
                position_bias=0.0,
                conformal_threshold=0.3,
                agreement_rate=0.8,
                num_pairs=True,  # type: ignore[arg-type]
                calibrated=True,
            )

    def test_non_bool_calibrated(self):
        with pytest.raises(ValueError, match="calibrated"):
            JudgeCalibrationReport(
                position_bias=0.0,
                conformal_threshold=0.3,
                agreement_rate=0.8,
                num_pairs=10,
                calibrated="yes",  # type: ignore[arg-type]
            )


# ─── run_pairwise_calibration ───


class TestRunPairwiseCalibration:
    def test_perfect_calibration(self):
        judgements = [
            PairwiseJudgement(
                prompt_id=f"p{i}", first_winner="a",
                second_winner="a", oracle="a",
            ) for i in range(10)
        ]
        scores = [0.9] * 10
        report = run_pairwise_calibration(judgements, scores=scores, alpha=0.1)
        assert report.agreement_rate == 1.0
        assert report.calibrated is True
        assert report.num_pairs == 10

    def test_returns_report(self):
        judgements = [
            PairwiseJudgement(
                prompt_id=f"p{i}", first_winner="a",
                second_winner="a", oracle="a",
            ) for i in range(20)
        ]
        scores = [0.5 + i * 0.02 for i in range(20)]
        r = run_pairwise_calibration(judgements, scores=scores, alpha=0.1)
        assert isinstance(r, JudgeCalibrationReport)

    def test_length_mismatch(self):
        judgements = [
            PairwiseJudgement(
                prompt_id="p", first_winner="a",
                second_winner="a", oracle="a",
            )
        ]
        with pytest.raises(ValueError, match="length"):
            run_pairwise_calibration(judgements, scores=[0.5, 0.6], alpha=0.1)

    def test_too_few_pairs(self):
        # Need a minimum sample to compute conformal threshold meaningfully.
        with pytest.raises(ValueError, match="pairs"):
            run_pairwise_calibration([], scores=[], alpha=0.1)

    def test_too_many_pairs(self):
        judgements = [
            PairwiseJudgement(
                prompt_id=f"p{i}", first_winner="a",
                second_winner="a", oracle="a",
            ) for i in range(50_001)
        ]
        scores = [0.5] * 50_001
        with pytest.raises(ValueError, match="cap"):
            run_pairwise_calibration(judgements, scores=scores, alpha=0.1)


# ─── ensure_judge_calibrated production gate ───


class TestEnsureJudgeCalibrated:
    def test_passes_calibrated(self):
        r = JudgeCalibrationReport(
            position_bias=0.05, conformal_threshold=0.3,
            agreement_rate=0.85, num_pairs=20, calibrated=True,
        )
        # Should not raise.
        ensure_judge_calibrated(r)

    def test_refuses_uncalibrated(self):
        r = JudgeCalibrationReport(
            position_bias=0.05, conformal_threshold=0.3,
            agreement_rate=0.85, num_pairs=20, calibrated=False,
        )
        with pytest.raises(RuntimeError, match="calibrat"):
            ensure_judge_calibrated(r)

    def test_refuses_none(self):
        with pytest.raises(RuntimeError, match="calibrat"):
            ensure_judge_calibrated(None)

    def test_refuses_low_agreement(self):
        r = JudgeCalibrationReport(
            position_bias=0.0, conformal_threshold=0.3,
            agreement_rate=0.5, num_pairs=20, calibrated=True,
        )
        with pytest.raises(RuntimeError, match="agreement"):
            ensure_judge_calibrated(r, min_agreement=0.7)

    def test_refuses_high_bias(self):
        r = JudgeCalibrationReport(
            position_bias=0.4, conformal_threshold=0.3,
            agreement_rate=0.9, num_pairs=20, calibrated=True,
        )
        with pytest.raises(RuntimeError, match="bias"):
            ensure_judge_calibrated(r, max_bias=0.2)

    def test_non_report_type(self):
        with pytest.raises(TypeError):
            ensure_judge_calibrated("calibrated")  # type: ignore[arg-type]


# ─── back-compat: existing kl_divergence + classify_kl_delta still work ───


class TestBackCompat:
    def test_kl_divergence_still_works(self):
        assert kl_divergence([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_classify_kl_delta_still_works(self):
        assert classify_kl_delta(0.0) == "OK"
        assert classify_kl_delta(0.1) == "MINOR"
        assert classify_kl_delta(0.3) == "MAJOR"


# ─── No heavy top-level imports ───


class TestSourceWiring:
    def test_no_heavy_imports(self):
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "eval" / "calibrate.py"
        text = src.read_text(encoding="utf-8")
        # Should not import torch, transformers, peft at module scope.
        forbidden_imports = (
            "import torch\n",
            "import transformers\n",
            "from torch",
            "from transformers",
        )
        for forbidden in forbidden_imports:
            assert forbidden not in text, f"Found heavy top-level import: {forbidden!r}"


# ─── v0.71.1 #214 — judge_calibration registry artifact persistence ───


class TestJudgeCalibrationPersistence:
    def _report(self, calibrated: bool = True) -> JudgeCalibrationReport:
        return JudgeCalibrationReport(
            position_bias=0.05,
            conformal_threshold=0.3,
            agreement_rate=0.8,
            num_pairs=12,
            calibrated=calibrated,
        )

    def test_to_dict_round_trips_fields(self):
        r = self._report()
        d = r.to_dict()
        assert d == {
            "position_bias": 0.05,
            "conformal_threshold": 0.3,
            "agreement_rate": 0.8,
            "num_pairs": 12,
            "calibrated": True,
        }

    def test_write_then_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = self._report()
        out = write_judge_calibration(r, "calib.json")
        assert out.exists()
        loaded = load_judge_calibration(str(out))
        assert isinstance(loaded, JudgeCalibrationReport)
        assert loaded == r

    def test_write_outside_cwd_rejected(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside"
        outside.mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError, match="cwd"):
            write_judge_calibration(self._report(), str(outside / "calib.json"))

    def test_write_rejects_non_report(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            write_judge_calibration({"calibrated": True}, "calib.json")  # type: ignore[arg-type]

    def test_load_re_validates_corrupt_report(self, tmp_path, monkeypatch):
        # An out-of-range field on disk must be rejected on load (the
        # frozen dataclass __post_init__ is the production-gate safety net).
        import json

        monkeypatch.chdir(tmp_path)
        (tmp_path / "bad.json").write_text(
            json.dumps(
                {
                    "position_bias": 2.0,  # out of [-1, 1]
                    "conformal_threshold": 0.3,
                    "agreement_rate": 0.8,
                    "num_pairs": 12,
                    "calibrated": True,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="position_bias"):
            load_judge_calibration("bad.json")

    def test_load_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_judge_calibration("nope.json")

    def test_registry_kind_registered(self):
        from soup_cli.registry.store import _VALID_KINDS

        assert "judge_calibration" in _VALID_KINDS
