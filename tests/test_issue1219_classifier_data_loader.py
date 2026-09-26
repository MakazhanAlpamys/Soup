"""Tests for Issue #1219: classifier/reranker/cross_encoder data loading and validation.

Verifies:
1. Source column retention via shared helper `task_preserves_source_columns`.
2. Format detection and conversion for `cross_encoder` (both text_a/text_b and question/answer).
3. Upfront label and shape validation with specific row indices on --dry-run and at load.
4. All variants: string labels, integer labels, multi-label lists, chatml with label,
   reranker, cross_encoder (with both auto and explicit formats).
5. Documented example end-to-end training through `soup train`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.data.formats import (
    FORMAT_SIGNATURES,
    VALID_FORMATS,
    detect_format,
    format_to_messages,
)
from soup_cli.data.loader import (
    PRESERVE_SOURCE_TASKS,
    load_dataset,
    task_preserves_source_columns,
)
from soup_cli.trainer.classifier import (
    ClassifierTrainerWrapper,
    validate_classification_dataset,
)
from tests._windows_ci import skip_on_windows_ci
from tests.conftest import strip_ansi


def _strip_ansi(text: str) -> str:
    """Strip ANSI color codes and collapse whitespace for resilient assertions."""
    plain = strip_ansi(text)
    return " ".join(plain.split())


class TestTaskPreservesSourceColumns:
    """One helper decides which tasks keep source columns; train and sweep must agree."""

    def test_expected_tasks_in_preserve_set(self) -> None:
        assert PRESERVE_SOURCE_TASKS == {
            "grpo",
            "classifier",
            "reranker",
            "cross_encoder",
        }

    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            ("grpo", True),
            ("classifier", True),
            ("reranker", True),
            ("cross_encoder", True),
            ("sft", False),
            ("dpo", False),
            ("kto", False),
            ("orpo", False),
            ("reward_model", False),
            ("pretrain", False),
            ("embedding", False),
            ("distill", False),
            ("unlearn", False),
        ],
    )
    def test_helper_predicate_values(self, task: str, expected: bool) -> None:
        assert task_preserves_source_columns(task) is expected

    def test_train_and_sweep_use_same_helper_ast(self) -> None:
        """Audit AST of train.py and sweep.py: both must call task_preserves_source_columns."""
        train_py = Path("src/soup_cli/commands/train.py").read_text(encoding="utf-8")
        sweep_py = Path("src/soup_cli/commands/sweep.py").read_text(encoding="utf-8")

        # Neither file should have the old hardcoded cfg.task == "grpo"
        assert 'preserve_source_columns=cfg.task == "grpo"' not in train_py
        assert 'preserve_source_columns=cfg.task == "grpo"' not in sweep_py

        # Both must call task_preserves_source_columns
        assert "task_preserves_source_columns(cfg.task)" in train_py
        assert "task_preserves_source_columns(cfg.task)" in sweep_py

        # Both must import task_preserves_source_columns from soup_cli.data.loader
        train_ast = ast.parse(train_py)
        sweep_ast = ast.parse(sweep_py)

        def finds_import(tree: ast.AST) -> bool:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "soup_cli.data.loader":
                    for name in node.names:
                        if name.name == "task_preserves_source_columns":
                            return True
            return False

        assert finds_import(train_ast), "train.py must import task_preserves_source_columns"
        assert finds_import(sweep_ast), "sweep.py must import task_preserves_source_columns"


class TestCrossEncoderFormat:
    """Format detection and conversion for paired sequence classification."""

    def test_signatures_and_valid_formats(self) -> None:
        assert "cross_encoder" in FORMAT_SIGNATURES
        assert FORMAT_SIGNATURES["cross_encoder"] == {"text_a", "text_b"}
        assert "cross_encoder" in VALID_FORMATS

    def test_detect_format_text_a_text_b(self) -> None:
        rows = [{"text_a": "query", "text_b": "passage", "label": 1}]
        assert detect_format(rows) == "cross_encoder"

    def test_detect_format_question_answer(self) -> None:
        rows = [{"question": "what is this?", "answer": "it is a cat", "label": 0}]
        assert detect_format(rows) == "cross_encoder"

    def test_conversion_preserves_pairs_and_label(self) -> None:
        row_ab = {"text_a": "first", "text_b": "second", "label": "relevant", "meta": 123}
        out_ab = format_to_messages(row_ab, "cross_encoder")
        assert out_ab is not None
        assert out_ab["text_a"] == "first"
        assert out_ab["text_b"] == "second"
        assert out_ab["label"] == "relevant"

        row_qa = {"question": "q?", "answer": "a!", "label": 1}
        out_qa = format_to_messages(row_qa, "cross_encoder")
        assert out_qa is not None
        assert out_qa["question"] == "q?"
        assert out_qa["answer"] == "a!"
        assert out_qa["label"] == 1

    def test_conversion_rejects_non_string_pairs(self) -> None:
        assert format_to_messages({"text_a": 123, "text_b": "good"}, "cross_encoder") is None
        bad_qa = {"question": "q", "answer": ["not", "string"]}
        assert format_to_messages(bad_qa, "cross_encoder") is None


class TestLabelValidationUpfront:
    """Missing or invalid labels must fail at load and on --dry-run with row index."""

    def test_validate_missing_label_names_row_index(self) -> None:
        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: classifier
            training:
              num_labels: 2
            data:
              train: ./fake.jsonl
            """
        )
        dataset = {
            "train": [
                {"text": "good message", "label": 1},
                {"text": "bad message missing label"},
                {"text": "another good message", "label": 0},
            ]
        }
        with pytest.raises(ValueError, match=r"train row 1: missing required 'label' field"):
            validate_classification_dataset(cfg, dataset)

    def test_validate_invalid_label_type_names_row_index(self) -> None:
        cfg = load_config_from_string(
            """
            base: sshleifer/tiny-gpt2
            task: classifier
            training:
              num_labels: 2
            data:
              train: ./fake.jsonl
            """
        )
        dataset = {
            "train": [
                {"text": "message", "label": True},
            ]
        }
        with pytest.raises(ValueError, match=r"train row 0: label must not be bool"):
            validate_classification_dataset(cfg, dataset)

    def test_dry_run_refuses_missing_label_with_row_index(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            json.dumps({"text": "sample 0", "label": "ham"}) + "\n"
            + json.dumps({"text": "sample 1"}) + "\n"
        )
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(
            f"""
            base: sshleifer/tiny-gpt2
            task: classifier
            data:
              train: {data_file.as_posix()}
              max_length: 64
              val_split: 0.0
            training:
              num_labels: 2
              label_names: [ham, spam]
            """
        )
        runner = CliRunner()
        res = runner.invoke(app, ["train", "--config", str(cfg_file), "--dry-run"])
        assert res.exit_code == 1
        clean = _strip_ansi(res.output)
        assert "train row 1" in clean
        assert "missing required 'label' field" in clean

    def test_train_load_refuses_missing_label_before_model_loads(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            json.dumps({"text": "sample 0"}) + "\n"
        )
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(
            f"""
            base: invalid-model-that-would-fail-if-loaded
            task: classifier
            data:
              train: {data_file.as_posix()}
              max_length: 64
              val_split: 0.0
            training:
              num_labels: 2
              label_names: [ham, spam]
            """
        )
        runner = CliRunner()
        res = runner.invoke(app, ["train", "--config", str(cfg_file), "--yes"])
        assert res.exit_code == 1
        clean = _strip_ansi(res.output)
        assert "train row 0" in clean
        assert "missing required 'label' field" in clean
        assert "Setting up model + trainer" not in clean


class TestDatasetVariantsLoading:
    """Verify all variants preserve source columns and setup succeeds."""

    @pytest.mark.parametrize(
        ("fmt", "rows", "task", "training_dict", "expected_labels"),
        [
            # String labels with label_names (auto format)
            (
                "auto",
                [{"text": f"text {i}", "label": "spam" if i % 2 else "ham"} for i in range(4)],
                "classifier",
                {"num_labels": 2, "label_names": ["ham", "spam"]},
                [0, 1, 0, 1],
            ),
            # String labels with explicit plaintext format
            (
                "plaintext",
                [{"text": f"text {i}", "label": "spam" if i % 2 else "ham"} for i in range(4)],
                "classifier",
                {"num_labels": 2, "label_names": ["ham", "spam"]},
                [0, 1, 0, 1],
            ),
            # Integer labels (auto format)
            (
                "auto",
                [{"text": f"text {i}", "label": i % 3} for i in range(6)],
                "classifier",
                {"num_labels": 3},
                [0, 1, 2, 0, 1, 2],
            ),
            # Integer labels with explicit plaintext format
            (
                "plaintext",
                [{"text": f"text {i}", "label": i % 3} for i in range(6)],
                "classifier",
                {"num_labels": 3},
                [0, 1, 2, 0, 1, 2],
            ),
            # Multi-label lists (auto format)
            (
                "auto",
                [{"text": f"text {i}", "label": [0, 2]} for i in range(3)],
                "classifier",
                {"num_labels": 3, "classifier_kind": "multi_label"},
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            ),
            # Multi-label lists with label names
            (
                "plaintext",
                [{"text": f"text {i}", "label": ["ham", "promo"]} for i in range(3)],
                "classifier",
                {
                    "num_labels": 3,
                    "classifier_kind": "multi_label",
                    "label_names": ["ham", "spam", "promo"],
                },
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            ),
            # ChatML rows carrying label (auto format)
            (
                "auto",
                [
                    {
                        "messages": [{"role": "user", "content": f"hello {i}"}],
                        "label": "spam" if i % 2 else "ham",
                    }
                    for i in range(4)
                ],
                "classifier",
                {"num_labels": 2, "label_names": ["ham", "spam"]},
                [0, 1, 0, 1],
            ),
            # ChatML rows carrying label (explicit chatml format)
            (
                "chatml",
                [
                    {
                        "messages": [{"role": "user", "content": f"hello {i}"}],
                        "label": "spam" if i % 2 else "ham",
                    }
                    for i in range(4)
                ],
                "classifier",
                {"num_labels": 2, "label_names": ["ham", "spam"]},
                [0, 1, 0, 1],
            ),
            # task: reranker (auto format)
            (
                "auto",
                [{"text": f"query doc {i}", "label": "relevant"} for i in range(3)],
                "reranker",
                {"num_labels": 1, "label_names": ["relevant"]},
                [0, 0, 0],
            ),
            # task: reranker (explicit plaintext format)
            (
                "plaintext",
                [{"text": f"query doc {i}", "label": "relevant"} for i in range(3)],
                "reranker",
                {"num_labels": 1, "label_names": ["relevant"]},
                [0, 0, 0],
            ),
            # task: cross_encoder with text_a and text_b (auto format)
            (
                "auto",
                [{"text_a": f"q {i}", "text_b": f"doc {i}", "label": i % 2} for i in range(4)],
                "cross_encoder",
                {"num_labels": 2},
                [0, 1, 0, 1],
            ),
            # task: cross_encoder with text_a and text_b (explicit cross_encoder format)
            (
                "cross_encoder",
                [{"text_a": f"q {i}", "text_b": f"doc {i}", "label": i % 2} for i in range(4)],
                "cross_encoder",
                {"num_labels": 2},
                [0, 1, 0, 1],
            ),
            # task: cross_encoder with question and answer (auto format)
            (
                "auto",
                [{"question": f"q {i}", "answer": f"ans {i}", "label": 1} for i in range(3)],
                "cross_encoder",
                {"num_labels": 2},
                [1, 1, 1],
            ),
            # task: cross_encoder with question and answer (explicit cross_encoder format)
            (
                "cross_encoder",
                [{"question": f"q {i}", "answer": f"ans {i}", "label": 1} for i in range(3)],
                "cross_encoder",
                {"num_labels": 2},
                [1, 1, 1],
            ),
        ],
    )
    def test_load_dataset_retains_columns_and_validates(
        self,
        tmp_path: Path,
        fmt: str,
        rows: list[dict[str, Any]],
        task: str,
        training_dict: dict[str, Any],
        expected_labels: list[Any],
    ) -> None:
        import yaml

        data_file = tmp_path / f"test_{task}_{fmt}.jsonl"
        data_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        cfg_obj = {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {
                "train": data_file.as_posix(),
                "format": fmt,
                "val_split": 0.0,
                "max_length": 64,
            },
            "training": training_dict,
        }
        cfg = load_config_from_string(yaml.safe_dump(cfg_obj))

        # 1. load_dataset with task_preserves_source_columns
        dataset = load_dataset(
            cfg.data,
            preserve_source_columns=task_preserves_source_columns(cfg.task),
        )
        assert len(dataset["train"]) == len(rows)
        for row in dataset["train"]:
            assert "label" in row

        # 2. validate_classification_dataset passes
        validate_classification_dataset(cfg, dataset)

        # 3. Setup wrapper
        wrapper = ClassifierTrainerWrapper(cfg, device="cpu")
        wrapper.setup(dataset)
        assert wrapper.train_dataset is not None
        assert wrapper.train_dataset["labels"] == expected_labels


class TestDocumentedExampleTraining:
    """End-to-end training of the documented classifier example."""

    @skip_on_windows_ci
    def test_documented_classifier_example_trains_and_records_labels(
        self, tmp_path: Path
    ) -> None:
        labels = ["ham", "spam", "promo"]
        data_path = tmp_path / "labelled.jsonl"
        data_path.write_text(
            "\n".join(
                json.dumps({"text": f"message {i}", "label": labels[i % 3]})
                for i in range(6)
            )
            + "\n"
        )
        out_dir = tmp_path / "out"
        cfg_file = tmp_path / "cls.yaml"
        cfg_file.write_text(
            f"""
            base: hf-internal-testing/tiny-random-LlamaForCausalLM
            task: classifier
            data:
              train: {data_path.as_posix()}
              max_length: 64
              val_split: 0.0
            training:
              num_labels: 3
              label_names: [ham, spam, promo]
              quantization: none
              epochs: 1
              batch_size: 2
            output: {out_dir.as_posix()}
            """
        )
        runner = CliRunner()
        res = runner.invoke(app, ["train", "--config", str(cfg_file), "--yes"])
        clean = _strip_ansi(res.output)
        assert res.exit_code == 0, f"Command failed: {res.output}"
        assert "Auto-detected format: plaintext" in clean
        assert "Loaded: 6 train samples" in clean
        assert "Training Complete!" in clean

        # Verify saved model config recorded the labels
        saved_config_path = out_dir / "config.json"
        assert saved_config_path.exists(), "config.json should be saved in output directory"
        saved_cfg = json.loads(saved_config_path.read_text(encoding="utf-8"))
        assert saved_cfg.get("id2label") == {"0": "ham", "1": "spam", "2": "promo"}
        assert saved_cfg.get("label2id") == {"ham": 0, "spam": 1, "promo": 2}
