"""Tests for v0.43.0 Part B — Eval metrics."""
from __future__ import annotations

import pytest

from soup_cli.eval.arena import (
    DEFAULT_BASE_RATING,
    Tournament,
    expected_score,
    update_elo,
)
from soup_cli.eval.benchmarks_v0_43 import (
    NEW_BENCHMARKS_V0_43,
    benchmark_metadata,
    is_v0_43_benchmark,
    lm_eval_task_for,
)
from soup_cli.eval.calibrate import (
    CalibrationReport,
    classify_kl_delta,
    kl_divergence,
    run_calibration,
)
from soup_cli.utils.nlg_metrics import (
    NLG_METRICS,
    bleu_score,
    compute_nlg_metric,
    effective_tokens_per_second,
    rouge_l_score,
    rouge_n_score,
)

# ----------------- BLEU / ROUGE -----------------

class TestBleuScore:
    def test_perfect_match(self):
        score = bleu_score(["the cat sat on the mat"], ["the cat sat on the mat"])
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_zero_overlap_no_smoothing(self):
        # Without smoothing, any zero n-gram precision -> BLEU 0.
        score = bleu_score(
            ["alpha beta gamma delta"],
            ["one two three four"],
            smooth=False,
        )
        assert score == 0.0

    def test_zero_overlap_smoothed(self):
        # With smoothing (the default), Chen & Cherry assigns small mass
        # to each zero bucket, so the score is small but positive.
        score = bleu_score(["alpha beta gamma delta"], ["one two three four"])
        assert 0.0 < score < 0.2

    def test_brevity_penalty_one_when_pred_longer(self):
        # Prediction longer than reference: BP = 1.0 (no penalty).
        # Score reflects modified precision only.
        # pred unigrams the=2,cat=1,sat=1,on=1,the=2,mat=1 (6 total);
        # ref the=1,cat=1,sat=1.
        # min-clipped overlap = the(min(2,1)) + cat + sat = 3, total=6 → 0.5
        score = bleu_score(
            ["the cat sat on the mat"], ["the cat sat"], max_n=1
        )
        assert score == pytest.approx(0.5)

    def test_partial_overlap(self):
        # BLEU-2 partial: shorter prediction with all 4 unigram + 3 bigram
        # overlaps — guarantees nonzero standard BLEU.
        score = bleu_score(
            ["the quick brown fox"],
            ["the quick brown fox jumped"],
            max_n=2,
        )
        assert 0.0 < score < 1.0

    def test_empty_corpus(self):
        assert bleu_score([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            bleu_score(["a"], ["a", "b"])

    def test_invalid_max_n(self):
        with pytest.raises(ValueError):
            bleu_score(["a"], ["a"], max_n=0)
        with pytest.raises(ValueError):
            bleu_score(["a"], ["a"], max_n=10)
        with pytest.raises(ValueError):
            bleu_score(["a"], ["a"], max_n=True)  # type: ignore[arg-type]

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError, match="null"):
            bleu_score(["a\x00"], ["a"])


class TestRougeNScore:
    def test_perfect_match(self):
        score = rouge_n_score(["alpha beta gamma"], ["alpha beta gamma"])
        assert score == pytest.approx(1.0)

    def test_zero_overlap(self):
        score = rouge_n_score(["alpha beta"], ["one two"])
        assert score == 0.0

    def test_n_2(self):
        score = rouge_n_score(["the cat sat"], ["the cat sat"], n=2)
        assert score == pytest.approx(1.0)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            rouge_n_score(["a"], ["a"], n=0)
        with pytest.raises(ValueError):
            rouge_n_score(["a"], ["a"], n=True)  # type: ignore[arg-type]

    def test_too_short_for_n(self):
        # Single-token strings have no bigrams.
        score = rouge_n_score(["a"], ["a"], n=2)
        assert score == 0.0

    def test_length_mismatch_message(self):
        with pytest.raises(ValueError, match="same length"):
            rouge_n_score(["a"], ["a", "b"])


class TestRougeLScore:
    def test_perfect_match(self):
        score = rouge_l_score(["alpha beta gamma"], ["alpha beta gamma"])
        assert score == pytest.approx(1.0)

    def test_lcs_partial(self):
        score = rouge_l_score(
            ["the quick brown fox"],
            ["a quick brown dog"],
        )
        # LCS = "quick brown" (2 tokens)
        # P = 2/4, R = 2/4 (4 tokens each), F1 = 0.5
        assert score == pytest.approx(0.5)

    def test_zero_overlap(self):
        score = rouge_l_score(["alpha"], ["beta"])
        assert score == 0.0

    def test_empty(self):
        assert rouge_l_score([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            rouge_l_score(["a"], [])


class TestComputeNlgMetric:
    @pytest.mark.parametrize("metric", list(NLG_METRICS))
    def test_dispatch(self, metric):
        score = compute_nlg_metric(metric, ["a b c"], ["a b c"])
        assert 0.0 <= score <= 1.0

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown nlg metric"):
            compute_nlg_metric("meteor", ["a"], ["a"])

    def test_non_string(self):
        with pytest.raises(ValueError):
            compute_nlg_metric(None, ["a"], ["a"])  # type: ignore[arg-type]

    def test_case_insensitive(self):
        assert compute_nlg_metric("BLEU", ["a"], ["a"]) >= 0.0


# ----------------- effective_tokens_per_second -----------------

class TestEffectiveTokensPerSecond:
    def test_happy(self):
        assert effective_tokens_per_second(
            unmasked_tokens=10000, wall_clock_seconds=10.0
        ) == 1000.0

    def test_zero_wall_clock_returns_none(self):
        assert effective_tokens_per_second(
            unmasked_tokens=100, wall_clock_seconds=0.0
        ) is None

    def test_negative_wall_clock_returns_none(self):
        assert effective_tokens_per_second(
            unmasked_tokens=100, wall_clock_seconds=-1.0
        ) is None

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValueError):
            effective_tokens_per_second(
                unmasked_tokens=-1, wall_clock_seconds=1.0
            )

    def test_bool_tokens_rejected(self):
        with pytest.raises(ValueError):
            effective_tokens_per_second(
                unmasked_tokens=True, wall_clock_seconds=1.0  # type: ignore[arg-type]
            )

    def test_bool_wall_clock_rejected(self):
        with pytest.raises(ValueError):
            effective_tokens_per_second(
                unmasked_tokens=100, wall_clock_seconds=True  # type: ignore[arg-type]
            )

    def test_nonfinite_wall_clock_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            effective_tokens_per_second(
                unmasked_tokens=100, wall_clock_seconds=float("inf")
            )


# ----------------- KL Calibration -----------------

class TestKlDivergence:
    def test_identical_distributions(self):
        kl = kl_divergence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert kl == pytest.approx(0.0, abs=1e-9)

    def test_different_distributions_positive(self):
        kl = kl_divergence([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert kl > 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            kl_divergence([1.0, 2.0], [1.0])

    def test_empty(self):
        with pytest.raises(ValueError):
            kl_divergence([], [])

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            kl_divergence([1.0, float("inf")], [1.0, 2.0])

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            kl_divergence([True, False], [1.0, 2.0])  # type: ignore[list-item]


class TestClassifyKlDelta:
    @pytest.mark.parametrize(
        "kl,status",
        [
            (0.0, "OK"),
            (0.04, "OK"),
            (0.05, "MINOR"),
            (0.10, "MINOR"),
            (0.19, "MINOR"),
            (0.20, "MAJOR"),
            (1.0, "MAJOR"),
        ],
    )
    def test_thresholds(self, kl, status):
        assert classify_kl_delta(kl) == status

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            classify_kl_delta(-0.01)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            classify_kl_delta(float("nan"))

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            classify_kl_delta(True)  # type: ignore[arg-type]


class TestRunCalibration:
    def test_perfect_match_ok(self):
        baseline = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        quant = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        report = run_calibration(baseline, quant)
        assert isinstance(report, CalibrationReport)
        assert report.mean_kl == pytest.approx(0.0, abs=1e-9)
        assert report.delta_status == "OK"
        assert report.num_prompts == 2

    def test_diverged_quant_major(self):
        baseline = [[10.0, 0.0, 0.0]]
        quant = [[0.0, 0.0, 10.0]]
        report = run_calibration(baseline, quant)
        # Large divergence — guaranteed MAJOR.
        assert report.delta_status == "MAJOR"

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            run_calibration([[1.0]], [])

    def test_empty(self):
        with pytest.raises(ValueError):
            run_calibration([], [])

    def test_too_many_prompts(self):
        big = [[1.0]] * 10_001
        with pytest.raises(ValueError, match="too many"):
            run_calibration(big, big)

    def test_report_frozen(self):
        from dataclasses import FrozenInstanceError

        report = run_calibration([[1.0, 2.0]], [[1.0, 2.0]])
        with pytest.raises(FrozenInstanceError):
            report.mean_kl = 0.5  # type: ignore[misc]


# ----------------- Arena -----------------

class TestExpectedScore:
    def test_equal_ratings_50_50(self):
        assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)

    def test_higher_rated_favored(self):
        assert expected_score(1700.0, 1500.0) > 0.5

    def test_lower_rated_disfavored(self):
        assert expected_score(1300.0, 1500.0) < 0.5

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            expected_score(float("nan"), 1500.0)

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            expected_score(True, 1500.0)  # type: ignore[arg-type]


class TestUpdateElo:
    def test_winner_gains(self):
        new_a, new_b = update_elo(1500.0, 1500.0, score_a=1.0)
        assert new_a > 1500.0
        assert new_b < 1500.0

    def test_draw_no_change_at_equal(self):
        new_a, new_b = update_elo(1500.0, 1500.0, score_a=0.5)
        assert new_a == pytest.approx(1500.0)
        assert new_b == pytest.approx(1500.0)

    def test_zero_sum(self):
        new_a, new_b = update_elo(1500.0, 1700.0, score_a=1.0)
        # Symmetric Elo: total rating mass conserved.
        assert (new_a - 1500.0) == pytest.approx(-(new_b - 1700.0), abs=1e-6)

    def test_invalid_score_a(self):
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, score_a=-0.1)
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, score_a=1.1)
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, score_a=True)  # type: ignore[arg-type]

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, score_a=1.0, k=0)
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, score_a=1.0, k=-32)


class TestTournament:
    def test_register_and_record(self):
        t = Tournament()
        t.register("alpha")
        t.register("beta")
        t.record("alpha", "beta", winner="a")
        assert t.ratings["alpha"] > DEFAULT_BASE_RATING
        assert t.ratings["beta"] < DEFAULT_BASE_RATING

    def test_register_idempotent(self):
        t = Tournament()
        t.register("alpha")
        t.register("alpha")
        assert t.ratings["alpha"] == DEFAULT_BASE_RATING

    def test_ratings_immutable_view(self):
        t = Tournament()
        t.register("alpha")
        # Ratings property must not allow mutation through the returned view.
        with pytest.raises(TypeError):
            t.ratings["alpha"] = 9999.0  # type: ignore[index]

    def test_implicit_register_on_record(self):
        t = Tournament()
        t.record("alpha", "beta", winner="draw")
        assert "alpha" in t.ratings
        assert "beta" in t.ratings

    def test_self_play_rejected(self):
        t = Tournament()
        with pytest.raises(ValueError, match="must differ"):
            t.record("alpha", "alpha", winner="a")

    def test_invalid_winner(self):
        t = Tournament()
        with pytest.raises(ValueError, match="winner"):
            t.record("a", "b", winner="c")

    def test_empty_name_rejected(self):
        t = Tournament()
        with pytest.raises(ValueError):
            t.register("")

    def test_null_byte_name_rejected(self):
        t = Tournament()
        with pytest.raises(ValueError, match="null"):
            t.register("foo\x00")

    def test_oversize_name_rejected(self):
        t = Tournament()
        with pytest.raises(ValueError):
            t.register("a" * 256)

    def test_rich_markup_metacharacter_rejected(self):
        t = Tournament()
        with pytest.raises(ValueError, match="markup"):
            t.register("[red]evil[/red]")
        with pytest.raises(ValueError, match="markup"):
            t.register("foo]bar")

    def test_model_cap_exceeded(self):
        from soup_cli.eval.arena import _MAX_MODELS
        t = Tournament()
        for i in range(_MAX_MODELS):
            t.register(f"model_{i}")
        with pytest.raises(ValueError, match="model cap"):
            t.register("one_too_many")

    def test_invalid_k_nan(self):
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, score_a=1.0, k=float("nan"))

    def test_invalid_base_rating(self):
        with pytest.raises(ValueError):
            Tournament(base_rating=float("nan"))

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            Tournament(k=0)

    def test_leaderboard_sorted(self):
        t = Tournament()
        t.record("alpha", "beta", winner="a")
        t.record("alpha", "gamma", winner="a")
        board = t.leaderboard()
        assert board[0]["model"] == "alpha"
        assert board[0]["wins"] == 2
        assert board[0]["losses"] == 0
        # Highest rating first.
        for i in range(len(board) - 1):
            assert board[i]["rating"] >= board[i + 1]["rating"]

    def test_draw_records(self):
        t = Tournament()
        t.record("alpha", "beta", winner="draw")
        for name in ("alpha", "beta"):
            row = next(r for r in t.leaderboard() if r["model"] == name)
            assert row["draws"] == 1
            assert row["wins"] == 0
            assert row["losses"] == 0


# ----------------- Benchmarks v0.43 -----------------

class TestBenchmarksV043:
    @pytest.mark.parametrize("name", list(NEW_BENCHMARKS_V0_43))
    def test_recognised(self, name):
        assert is_v0_43_benchmark(name) is True
        meta = benchmark_metadata(name)
        assert meta is not None
        assert "description" in meta

    def test_case_insensitive(self):
        assert is_v0_43_benchmark("CEval") is True

    def test_unknown_returns_false(self):
        assert is_v0_43_benchmark("mmlu") is False
        assert is_v0_43_benchmark("garbage") is False

    def test_non_string_returns_false(self):
        assert is_v0_43_benchmark(None) is False  # type: ignore[arg-type]
        assert is_v0_43_benchmark(123) is False  # type: ignore[arg-type]

    def test_metadata_immutable(self):
        meta = benchmark_metadata("ceval")
        with pytest.raises(TypeError):
            meta["description"] = "evil"  # type: ignore[index]

    def test_lm_eval_task(self):
        assert lm_eval_task_for("ceval") == "ceval-valid"
        assert lm_eval_task_for("cmmlu") == "cmmlu"
        # Aider Polyglot has no lm-eval mapping.
        assert lm_eval_task_for("aider_polyglot") is None
        assert lm_eval_task_for("garbage") is None
