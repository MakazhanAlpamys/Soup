"""#2618: ranking metrics for task: reranker / task: cross_encoder.

Covers the pure metric code (hand-computed values, ties, skipped queries), the
score rule per head shape, query grouping (including a query split between the
training and validation rows), query_id validation at load, the trainer wiring,
and one real evaluation on a tiny local model.
"""

from __future__ import annotations

import io
import math
from typing import Any

import pytest
from rich.console import Console

from soup_cli.utils.classifier import (
    ranking_metrics,
    ranking_query_groups,
    ranking_relevance,
    ranking_score_skip_reason,
    ranking_scores,
    validate_query_id,
)


def _tiny_base() -> str:
    from tests._tiny_hf_models import tiny_model_dir

    return tiny_model_dir("llama")


def _config(
    task: str,
    num_labels: int,
    kind: str = "single_label",
    base: str = "x/y",
    batch_size: int = 2,
):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": task,
                "data": {"train": "unused.jsonl", "val_split": 0.0, "max_length": 64},
                "training": {
                    "num_labels": num_labels,
                    "classifier_kind": kind,
                    "quantization": "none",
                    "epochs": 1,
                    "batch_size": batch_size,
                },
            }
        )
    )


@pytest.fixture
def captured_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    import soup_cli.trainer.classifier as trainer_mod

    buffer = io.StringIO()
    monkeypatch.setattr(trainer_mod, "console", Console(file=buffer, width=200))
    return buffer


class _EvalPred:
    def __init__(self, predictions: Any, label_ids: Any) -> None:
        self.predictions = predictions
        self.label_ids = label_ids


# --------------------------------------------------------------------------- metrics


class TestRankingMetrics:
    def test_hand_computed_values(self) -> None:
        # q1: the positive ranks 1st. q2: the positive ranks 3rd of 3.
        scores = [0.9, 0.1, 0.2, 0.5, 0.8, 0.7]
        relevance = [1, 0, 1, 0, 0, 0]
        groups = ["q1", "q1", "q2", "q2", "q2", "q1"]
        m = ranking_metrics(scores, relevance, groups)

        assert m["ranking_queries"] == 2
        assert m["ranking_queries_no_positive"] == 0
        assert m["mrr@10"] == pytest.approx((1 + 1 / 3) / 2)
        assert m["recall@1"] == pytest.approx(0.5)
        assert m["recall@5"] == pytest.approx(1.0)
        assert m["recall@10"] == pytest.approx(1.0)
        # q1 nDCG = 1; q2 nDCG = (1/log2(4)) / 1 = 0.5
        assert m["ndcg@10"] == pytest.approx(0.75)

    def test_query_without_positive_is_skipped_and_counted(self) -> None:
        m = ranking_metrics([0.9, 0.1, 0.5], [1, 0, 0], ["q1", "q1", "q2"])
        assert m["ranking_queries"] == 1
        assert m["ranking_queries_no_positive"] == 1
        assert m["mrr@10"] == pytest.approx(1.0)

    def test_ties_are_broken_by_row_order(self) -> None:
        # Equal scores: the earlier row ranks first, so the result is deterministic.
        first = ranking_metrics([0.5, 0.5], [0, 1], ["q", "q"])
        assert first["mrr@10"] == pytest.approx(0.5)
        second = ranking_metrics([0.5, 0.5], [1, 0], ["q", "q"])
        assert second["mrr@10"] == pytest.approx(1.0)

    def test_several_positives(self) -> None:
        # Order by score: rel, non, rel, non -> recall@1 = 1/2, recall@5 = 1
        m = ranking_metrics([4, 3, 2, 1], [1, 0, 1, 0], ["q"] * 4)
        assert m["recall@1"] == pytest.approx(0.5)
        assert m["recall@5"] == pytest.approx(1.0)
        ideal = 1 + 1 / math.log2(3)
        assert m["ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)

    def test_positive_below_cutoff_scores_zero_mrr(self) -> None:
        scores = list(range(12, 0, -1))  # row 0 ranks first
        relevance = [0] * 11 + [1]       # the only positive ranks 12th
        m = ranking_metrics(scores, relevance, ["q"] * 12)
        assert m["mrr@10"] == 0.0
        assert m["recall@10"] == 0.0
        assert m["ndcg@10"] == 0.0

    def test_rows_with_no_group_are_left_out(self) -> None:
        m = ranking_metrics([0.1, 0.9], [1, 0], ["q", None])
        assert m["ranking_queries"] == 1
        assert m["mrr@10"] == pytest.approx(1.0)

    def test_non_finite_score_ranks_last(self) -> None:
        m = ranking_metrics([float("nan"), 0.1], [0, 1], ["q", "q"])
        assert m["mrr@10"] == pytest.approx(1.0)

    def test_no_scorable_query_reports_counts_only(self) -> None:
        m = ranking_metrics([0.1], [0], ["q"])
        assert m == {"ranking_queries": 0.0, "ranking_queries_no_positive": 1.0}

    def test_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            ranking_metrics([0.1, 0.2], [1], ["q", "q"])


# --------------------------------------------------------------------------- score rule


class TestScoreRule:
    def test_two_logits_rank_by_difference_not_positive_logit(self) -> None:
        # Doc A: positive logit 1, difference 6. Doc B: positive logit 3, difference 1.
        # By the positive logit alone B > A; by the difference (what cross-entropy
        # trains) A > B.
        logits = [[-5.0, 1.0], [2.0, 3.0]]
        assert ranking_scores(logits, 2, False) == [6.0, 1.0]
        m = ranking_metrics(ranking_scores(logits, 2, False), [1, 0], ["q", "q"])
        assert m["mrr@10"] == pytest.approx(1.0)

    def test_one_logit_with_multi_label(self) -> None:
        assert ranking_score_skip_reason(1, True) is None
        assert ranking_scores([[0.3], [-1.2]], 1, True) == [0.3, -1.2]
        assert ranking_relevance([[1.0], [0.0]], 1, True) == [1, 0]

    def test_relevance_for_two_labels(self) -> None:
        assert ranking_relevance([1, 0, 1], 2, False) == [1, 0, 1]

    @pytest.mark.parametrize(
        "num_labels,multi_label,needle",
        [
            (1, False, "classifier_kind: multi_label"),
            (2, True, "no single positive class"),
            (3, False, "num_labels: 3"),
        ],
    )
    def test_other_heads_are_skipped_with_a_reason(
        self, num_labels: int, multi_label: bool, needle: str
    ) -> None:
        reason = ranking_score_skip_reason(num_labels, multi_label)
        assert reason is not None and needle in reason
        with pytest.raises(ValueError):
            ranking_scores([[0.0] * num_labels], num_labels, multi_label)


# --------------------------------------------------------------------------- grouping


class TestQueryGroups:
    def test_query_split_between_training_and_validation_is_left_out(self) -> None:
        train = [{"query_id": "q1"}, {"query_id": "q2"}]
        val = [
            {"query_id": "q1"},  # q1 also has rows in training: incomplete
            {"query_id": "q3"},
            {"query_id": "q3"},
        ]
        groups, counts = ranking_query_groups(train, val)
        assert groups == [None, "q3", "q3"]
        assert counts == {"rows_without_query_id": 0, "queries_split": 1}

    def test_rows_without_query_id(self) -> None:
        groups, counts = ranking_query_groups([], [{"text": "a"}, {"query_id": 7}])
        assert groups == [None, 7]
        assert counts["rows_without_query_id"] == 1

    def test_int_and_str_ids_are_different_queries(self) -> None:
        groups, counts = ranking_query_groups([{"query_id": 1}], [{"query_id": "1"}])
        assert groups == ["1"]
        assert counts["queries_split"] == 0

    @pytest.mark.parametrize("bad", [True, [1], {"a": 1}, 1.5])
    def test_bad_query_id_types(self, bad: object) -> None:
        with pytest.raises(TypeError):
            validate_query_id(bad)

    @pytest.mark.parametrize("bad", ["", "a\x00b", "x" * 513])
    def test_bad_query_id_strings(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_query_id(bad)


# --------------------------------------------------------------------------- load-time validation


class TestQueryIdValidationAtLoad:
    @pytest.mark.parametrize("bad", [True, [1], {"id": 1}])
    def test_bad_query_id_is_refused_with_its_row(self, bad: object) -> None:
        from soup_cli.trainer.classifier import validate_classification_dataset

        cfg = _config("reranker", 2)
        dataset = {
            "train": [
                {"query_id": "q1", "text": "a", "label": 1},
                {"query_id": bad, "text": "b", "label": 0},
            ]
        }
        with pytest.raises(TypeError, match="train row 1: query_id"):
            validate_classification_dataset(cfg, dataset)

    def test_classifier_task_ignores_query_id(self) -> None:
        from soup_cli.trainer.classifier import validate_classification_dataset

        cfg = _config("classifier", 2)
        validate_classification_dataset(
            cfg, {"train": [{"query_id": True, "text": "a", "label": 1}]}
        )


# --------------------------------------------------------------------------- wiring


def _ranked_dataset() -> dict:
    train = [
        {"query_id": "q1", "text_a": "query one", "text_b": "doc", "text": "q d", "label": 1},
        {"query_id": "q2", "text_a": "query two", "text_b": "doc", "text": "q d", "label": 0},
    ]
    val = [
        {"query_id": "q1", "text_a": "query one", "text_b": "doc b", "text": "q d", "label": 0},
        {"query_id": "q3", "text_a": "query three", "text_b": "a", "text": "q a", "label": 1},
        {"query_id": "q3", "text_a": "query three", "text_b": "b", "text": "q b", "label": 0},
    ]
    return {"train": train, "val": val}


class TestWiring:
    @pytest.mark.parametrize("task", ["reranker", "cross_encoder"])
    def test_ranking_tasks_get_compute_metrics(
        self, task: str, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        compute = build_ranking_compute_metrics(_config(task, 2), _ranked_dataset())
        assert compute is not None
        assert "1 query has rows in both training and validation" in captured_console.getvalue()

        # Row 0 (q1) is left out; q3's positive ranks first.
        metrics = compute(_EvalPred([[0.0, 5.0], [0.0, 2.0], [0.0, 1.0]], [0, 1, 0]))
        assert set(metrics) == {
            "mrr@10", "recall@1", "recall@5", "recall@10", "ndcg@10",
            "ranking_queries", "ranking_queries_no_positive", "ranking_queries_split",
        }
        assert metrics["mrr@10"] == pytest.approx(1.0)
        assert metrics["ranking_queries"] == 1
        assert metrics["ranking_queries_split"] == 1

    def test_classifier_task_gets_none(self, captured_console: io.StringIO) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        assert build_ranking_compute_metrics(_config("classifier", 2), _ranked_dataset()) is None
        assert captured_console.getvalue() == ""

    def test_no_validation_rows_gets_none_silently(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        data = {"train": _ranked_dataset()["train"], "val": []}
        assert build_ranking_compute_metrics(_config("reranker", 2), data) is None
        assert captured_console.getvalue() == ""

    def test_validation_without_query_id_is_skipped_with_a_line(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        data = {
            "train": [{"text": "a", "label": 1}],
            "val": [{"text": "b", "label": 0}, {"text": "c", "label": 1}],
        }
        assert build_ranking_compute_metrics(_config("reranker", 2), data) is None
        assert "validation rows carry no 'query_id'" in captured_console.getvalue()

    @pytest.mark.parametrize(
        "num_labels,kind",
        [(1, "single_label"), (2, "multi_label"), (3, "single_label")],
    )
    def test_heads_without_a_score_rule_are_skipped_with_a_line(
        self, num_labels: int, kind: str, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        cfg = _config("reranker", num_labels, kind)
        assert build_ranking_compute_metrics(cfg, _ranked_dataset()) is None
        assert "Ranking metrics skipped" in captured_console.getvalue()

    def test_one_logit_multi_label_is_wired(self, captured_console: io.StringIO) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        compute = build_ranking_compute_metrics(
            _config("reranker", 1, "multi_label"), _ranked_dataset()
        )
        assert compute is not None
        metrics = compute(_EvalPred([[9.0], [0.4], [-0.3]], [[0.0], [1.0], [0.0]]))
        assert metrics["mrr@10"] == pytest.approx(1.0)

    def test_every_query_split_is_skipped_with_a_line(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_ranking_compute_metrics

        data = {
            "train": [{"query_id": "q1", "text": "a", "label": 1}],
            "val": [{"query_id": "q1", "text": "b", "label": 0}],
        }
        assert build_ranking_compute_metrics(_config("reranker", 2), data) is None
        assert "no query has all of its rows" in captured_console.getvalue()

    def test_recorded_validation_number_stays_val_loss(self) -> None:
        from soup_cli.utils.eval_schedule import DEFAULT_VALIDATION_METRIC, validation_metric

        for task in ("reranker", "cross_encoder"):
            assert validation_metric(task) is DEFAULT_VALIDATION_METRIC


# --------------------------------------------------------------------------- real evaluation


class TestRealEvaluation:
    """One real ``Trainer.evaluate()`` on the tiny local Llama.

    Batch size 1: the tiny Llama config has no ``pad_token_id``, and a
    sequence-classification head refuses batches larger than 1 without one.
    That is the same on ``main`` for every classifier-family task and is not
    part of #2618.
    """
    def test_reranker_evaluation_logs_ranking_metrics_beside_eval_loss(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        cfg = _config("reranker", 2, base=_tiny_base(), batch_size=1)
        wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
        wrapper.setup(_ranked_dataset())
        assert wrapper.trainer.compute_metrics is not None

        metrics = wrapper.trainer.evaluate()
        assert "eval_loss" in metrics
        for key in ("eval_mrr@10", "eval_recall@1", "eval_recall@5",
                    "eval_recall@10", "eval_ndcg@10"):
            assert 0.0 <= metrics[key] <= 1.0, key
        assert metrics["eval_ranking_queries"] == 1
        assert metrics["eval_ranking_queries_split"] == 1

    def test_classifier_evaluation_is_unchanged(self, captured_console: io.StringIO) -> None:
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        cfg = _config("classifier", 2, base=_tiny_base(), batch_size=1)
        wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
        wrapper.setup(_ranked_dataset())
        assert wrapper.trainer.compute_metrics is None
        metrics = wrapper.trainer.evaluate()
        assert "eval_loss" in metrics
        assert not any("mrr" in key for key in metrics)
