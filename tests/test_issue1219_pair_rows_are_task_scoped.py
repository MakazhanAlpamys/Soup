import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.data.formats import detect_format
from tests.conftest import strip_ansi

BASE = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def test_question_answer_rows_are_not_detected_globally():
    """GSM8K's row shape. Only task: cross_encoder reads it as a pair (#1219)."""
    with pytest.raises(ValueError, match="Cannot detect format"):
        detect_format([{"question": "What is 2+2?", "answer": "4"}])


@pytest.mark.parametrize("task", ["sft", "grpo"])
def test_question_answer_rows_are_not_ready_to_train_off_cross_encoder(
    tmp_path, monkeypatch, task
):
    monkeypatch.chdir(tmp_path)
    rows = [{"question": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(4)]
    (tmp_path / "qa.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    reward = "  reward_fn: accuracy\n" if task == "grpo" else ""
    (tmp_path / "soup.yaml").write_text(
        f"base: {BASE}\ntask: {task}\ndata:\n  train: qa.jsonl\n  val_split: 0.0\n"
        f"training:\n  quantization: none\n{reward}"
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    assert result.exit_code != 0, _plain(result.output)
    assert "Ready to train" not in _plain(result.output)


def test_pretrain_text_rows_with_question_answer_columns_stay_plaintext():
    rows = [{"text": "doc", "question": "q", "answer": "a"}]
    assert detect_format(rows) == "plaintext"


def test_cross_encoder_format_requires_the_cross_encoder_task():
    with pytest.raises(ValueError, match="requires task='cross_encoder'"):
        load_config_from_string(
            "base: a/b\ntask: sft\ndata:\n  train: x.jsonl\n  format: cross_encoder\n"
        )


def test_validate_names_a_val_row_missing_its_label():
    """The val split is checked too: #1313 evaluates it, so a bad val row would
    otherwise fail only after the model loads."""
    from soup_cli.trainer.classifier import validate_classification_dataset

    cfg = load_config_from_string(
        "base: a/b\ntask: classifier\ntraining:\n  num_labels: 2\ndata:\n  train: x.jsonl\n"
    )
    dataset = {
        "train": [{"text": "a", "label": 0}],
        "val": [{"text": "b", "label": 1}, {"text": "c"}],
    }
    with pytest.raises(ValueError, match=r"val row 1: missing required 'label' field"):
        validate_classification_dataset(cfg, dataset)


@pytest.mark.parametrize("pair", [("text_a", "text_b"), ("question", "answer")])
def test_cross_encoder_auto_reads_both_pair_shapes_through_soup_train(
    tmp_path, monkeypatch, pair
):
    monkeypatch.chdir(tmp_path)
    left, right = pair
    rows = [{left: f"q {i}", right: f"doc {i}", "label": i % 2} for i in range(4)]
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "soup.yaml").write_text(
        f"base: {BASE}\ntask: cross_encoder\ndata:\n  train: pairs.jsonl\n  val_split: 0.0\n"
        "training:\n  num_labels: 2\n  quantization: none\n"
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    output = _plain(result.output)
    assert result.exit_code == 0, (output, repr(result.exception))
    assert "Data OK: 4 train samples" in output


@pytest.mark.parametrize("pair", [("text_a", "text_b"), ("question", "answer")])
def test_real_run_reads_pair_rows_the_way_the_dry_run_does(tmp_path, monkeypatch, pair):
    """The real-run load resolves ``format: auto`` for cross_encoder too, so it
    reaches the row check instead of stopping at format detection (#1219)."""
    monkeypatch.chdir(tmp_path)
    left, right = pair
    rows = [{left: f"q {i}", right: f"doc {i}", "label": i % 2} for i in range(3)]
    rows.append({left: "q 3", right: "doc 3"})  # no label: the row check must name it
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "soup.yaml").write_text(
        "base: invalid-model-that-would-fail-if-loaded\ntask: cross_encoder\n"
        "data:\n  train: pairs.jsonl\n  val_split: 0.0\n"
        "training:\n  num_labels: 2\n  quantization: none\n"
    )
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--yes"])
    output = _plain(result.output)
    assert result.exit_code == 1, (output, repr(result.exception))
    assert "train row 3: missing required 'label' field" in output
