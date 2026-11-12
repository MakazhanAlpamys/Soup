"""#2638: accuracy and F1 for task: classifier.

Covers the pure metric code (hand-computed single-label and multi-label values,
a class left out of the macro average, the logit-0 threshold), the skipped
num_labels: 1 single-label head, the trainer wiring, and one real evaluation
on a tiny local model after a training step.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from soup_cli.utils.classifier import classification_metrics, classification_skip_reason


def _tiny_base() -> str:
    from tests._tiny_hf_models import tiny_model_dir

    return tiny_model_dir("llama")


def _config(
    task: str,
    num_labels: int,
    kind: str = "single_label",
    base: str = "x/y",
    batch_size: int = 2,
    output: str = "./output",
):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": task,
                "output": output,
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


def _one_hot_logits(predicted: list[int], num_labels: int) -> list[list[float]]:
    return [[2.0 if c == p else 0.0 for c in range(num_labels)] for p in predicted]


# --------------------------------------------------------------------------- single-label


class TestSingleLabel:
    def test_hand_computed_values_with_a_class_left_out(self) -> None:
        # num_labels 4; class 3 has no true and no predicted row.
        labels = [0, 0, 1, 1, 2, 2]
        predicted = [0, 1, 1, 1, 2, 0]
        metrics = classification_metrics(_one_hot_logits(predicted, 4), labels, 4, False)

        assert set(metrics) == {"accuracy", "f1_macro"}
        assert metrics["accuracy"] == pytest.approx(4 / 6)
        # class 0: tp 1, fp 1, fn 1 -> 0.5; class 1: tp 2, fp 1 -> 0.8;
        # class 2: tp 1, fn 1 -> 2/3; class 3 left out (not counted as 0).
        assert metrics["f1_macro"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)

    def test_argmax_takes_the_first_index_on_a_tie(self) -> None:
        metrics = classification_metrics([[1.0, 1.0, 0.0]], [0], 3, False)
        assert metrics["accuracy"] == 1.0

    def test_nan_logit_is_never_the_argmax(self) -> None:
        metrics = classification_metrics([[math.nan, 1.0, 0.0]], [1], 3, False)
        assert metrics["accuracy"] == 1.0

    def test_numpy_inputs(self) -> None:
        np = pytest.importorskip("numpy")

        metrics = classification_metrics(
            np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32), np.array([1, 1]), 2, False
        )
        assert metrics["accuracy"] == pytest.approx(0.5)
        # class 0: fp 1 -> 0.0; class 1: tp 1, fn 1 -> 2/3.
        assert metrics["f1_macro"] == pytest.approx((0.0 + 2 / 3) / 2)

    def test_label_out_of_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            classification_metrics([[0.0, 1.0]], [2], 2, False)


# --------------------------------------------------------------------------- multi-label


class TestMultiLabel:
    def test_hand_computed_values_with_threshold_at_logit_zero(self) -> None:
        logits = [
            [2.0, -1.0, 0.0],   # predicted {0}; logit 0 is not predicted -> exact
            [1.0, 1.0, -1.0],   # predicted {0, 1}, true {0, 2}
            [-1.0, 3.0, 0.5],   # predicted {1, 2} -> exact
            [-2.0, -2.0, -2.0],  # nothing predicted, nothing true -> exact
        ]
        labels = [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        metrics = classification_metrics(logits, labels, 3, True)

        assert set(metrics) == {"subset_accuracy", "f1_micro", "f1_macro"}
        assert metrics["subset_accuracy"] == pytest.approx(3 / 4)
        # tp 4, fp 1, fn 1 overall.
        assert metrics["f1_micro"] == pytest.approx(8 / 10)
        # label 0: 1.0; label 1: tp 1, fp 1 -> 2/3; label 2: tp 1, fn 1 -> 2/3.
        assert metrics["f1_macro"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)

    def test_logit_exactly_zero_on_a_true_label_is_a_miss(self) -> None:
        metrics = classification_metrics([[0.0]], [[1.0]], 1, True)
        assert metrics == {"subset_accuracy": 0.0, "f1_micro": 0.0, "f1_macro": 0.0}

    def test_label_with_no_true_and_no_predicted_row_is_left_out(self) -> None:
        metrics = classification_metrics([[1.0, -1.0], [-1.0, -1.0]], [[1, 0], [0, 0]], 2, True)
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_micro"] == 1.0

    def test_nothing_true_and_nothing_predicted_has_no_f1(self) -> None:
        assert classification_metrics([[-1.0]], [[0.0]], 1, True) == {"subset_accuracy": 1.0}

    def test_one_label_multi_label_is_measured(self) -> None:
        assert classification_skip_reason(1, True) is None
        metrics = classification_metrics([[0.3], [-0.2], [0.1]], [[1.0], [1.0], [0.0]], 1, True)
        assert metrics["subset_accuracy"] == pytest.approx(1 / 3)
        # tp 1, fp 1, fn 1.
        assert metrics["f1_micro"] == pytest.approx(0.5)
        assert metrics["f1_macro"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- shape checks


class TestShapes:
    def test_one_label_single_label_is_skipped_with_a_reason(self) -> None:
        reason = classification_skip_reason(1, False)
        assert reason is not None and "only label 0" in reason
        with pytest.raises(ValueError, match="only label 0"):
            classification_metrics([[0.0]], [0], 1, False)

    @pytest.mark.parametrize("num_labels,multi_label", [(2, False), (5, False), (2, True)])
    def test_other_heads_are_measured(self, num_labels: int, multi_label: bool) -> None:
        assert classification_skip_reason(num_labels, multi_label) is None

    def test_row_count_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2 rows of logits for 1 labels"):
            classification_metrics([[0.0, 1.0], [1.0, 0.0]], [1], 2, False)

    def test_wrong_logit_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected 3 logits per row, got 2"):
            classification_metrics([[0.0, 1.0]], [1], 3, False)

    def test_no_rows_give_an_empty_result(self) -> None:
        assert classification_metrics([], [], 3, False) == {}


# --------------------------------------------------------------------------- wiring


def _labelled_dataset() -> dict:
    train = [{"text": f"train row {i}", "label": i % 3} for i in range(4)]
    val = [{"text": f"val row {i}", "label": i % 3} for i in range(4)]
    return {"train": train, "val": val}


def _multi_label_dataset() -> dict:
    train = [{"text": "a b", "label": [0, 1]}, {"text": "c", "label": [2]}]
    val = [{"text": "a", "label": [0]}, {"text": "b c", "label": [1, 2]}]
    return {"train": train, "val": val}


class TestWiring:
    def test_classifier_with_validation_rows_gets_compute_metrics(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_classification_compute_metrics

        compute = build_classification_compute_metrics(
            _config("classifier", 3), _labelled_dataset()
        )
        assert compute is not None
        metrics = compute(_EvalPred(_one_hot_logits([0, 1, 1], 3), [0, 1, 2]))
        assert metrics["accuracy"] == pytest.approx(2 / 3)
        assert captured_console.getvalue() == ""

    def test_tuple_predictions_are_unwrapped(self) -> None:
        np = pytest.importorskip("numpy")
        from soup_cli.trainer.classifier import build_classification_compute_metrics

        compute = build_classification_compute_metrics(
            _config("classifier", 2), _labelled_dataset()
        )
        assert compute is not None
        logits = np.array([[0.0, 1.0]])
        metrics = compute(_EvalPred((logits, np.zeros((1, 4))), np.array([1])))
        assert metrics["accuracy"] == 1.0

    def test_no_validation_rows_gets_none_silently(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_classification_compute_metrics

        data = {"train": _labelled_dataset()["train"], "val": []}
        assert build_classification_compute_metrics(_config("classifier", 3), data) is None
        assert captured_console.getvalue() == ""

    def test_one_label_single_label_is_skipped_with_a_line(
        self, captured_console: io.StringIO
    ) -> None:
        from soup_cli.trainer.classifier import build_classification_compute_metrics

        data = {"train": [{"text": "a", "label": 0}], "val": [{"text": "b", "label": 0}]}
        assert build_classification_compute_metrics(_config("classifier", 1), data) is None
        assert "Classification metrics skipped" in captured_console.getvalue()

    def test_multi_label_is_wired(self, captured_console: io.StringIO) -> None:
        from soup_cli.trainer.classifier import build_classification_compute_metrics

        compute = build_classification_compute_metrics(
            _config("classifier", 3, "multi_label"), _multi_label_dataset()
        )
        assert compute is not None
        metrics = compute(
            _EvalPred([[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]], [[1, 0, 0], [0, 1, 1]])
        )
        assert metrics["subset_accuracy"] == 1.0

    @pytest.mark.parametrize("task", ["reranker", "cross_encoder"])
    def test_ranking_tasks_get_none(self, task: str, captured_console: io.StringIO) -> None:
        from soup_cli.trainer.classifier import build_classification_compute_metrics

        assert build_classification_compute_metrics(_config(task, 2), _labelled_dataset()) is None
        assert captured_console.getvalue() == ""

    def test_recorded_validation_number_stays_val_loss(self) -> None:
        from soup_cli.utils.eval_schedule import DEFAULT_VALIDATION_METRIC, validation_metric

        assert validation_metric("classifier") is DEFAULT_VALIDATION_METRIC
        assert DEFAULT_VALIDATION_METRIC.log_key == "eval_loss"


# --------------------------------------------------------------------------- real evaluation


class TestRealEvaluation:
    """Real ``Trainer.evaluate()`` runs on the tiny local Llama after one training step.

    Training first sets ``pad_token_id`` from the tokenizer, so these runs can
    use batch size 2, as every Soup run evaluates after it trains.
    """

    @staticmethod
    def _train_one_step(cfg: Any, dataset: dict) -> Any:
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
        wrapper.setup(dataset)
        wrapper.trainer.args.max_steps = 1
        wrapper.trainer.args.save_strategy = "no"
        wrapper.trainer.train()
        return wrapper.trainer

    def test_single_label_evaluation_logs_accuracy_and_f1_beside_eval_loss(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        cfg = _config("classifier", 3, base=_tiny_base(), output=str(tmp_path))
        trainer = self._train_one_step(cfg, _labelled_dataset())

        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert 0.0 <= metrics["eval_accuracy"] <= 1.0
        assert 0.0 <= metrics["eval_f1_macro"] <= 1.0
        assert not any("mrr" in key for key in metrics)

    def test_multi_label_evaluation_logs_subset_accuracy_and_f1(
        self, tmp_path: Path, captured_console: io.StringIO
    ) -> None:
        cfg = _config("classifier", 3, "multi_label", base=_tiny_base(), output=str(tmp_path))
        trainer = self._train_one_step(cfg, _multi_label_dataset())

        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert 0.0 <= metrics["eval_subset_accuracy"] <= 1.0
        # A tiny random head may predict nothing for any label, which leaves
        # the F1 keys out; when present they are in range.
        for key in ("eval_f1_micro", "eval_f1_macro"):
            if key in metrics:
                assert 0.0 <= metrics[key] <= 1.0, key

    def test_classifier_without_validation_rows_has_no_compute_metrics(
        self, tmp_path: Path
    ) -> None:
        from soup_cli.trainer.classifier import ClassifierTrainerWrapper

        cfg = _config("classifier", 3, base=_tiny_base(), output=str(tmp_path))
        wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": _labelled_dataset()["train"], "val": []})
        assert wrapper.trainer.compute_metrics is None
