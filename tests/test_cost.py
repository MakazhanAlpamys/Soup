"""Tests for soup cost command."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


def test_cost_with_config_file(tmp_path):
    """Test basic cost table output."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  epochs: 3\n"
        "  batch_size: 4\n"
        "  quantization: 4bit\n"
        "  lora:\n"
        "    r: 64\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "Training Cost Estimate" in result.output
    assert "Provider" in result.output
    assert "RunPod" in result.output


def test_cost_json_output(tmp_path):
    """Test JSON output for automation."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "total_cost" in data[0]
    assert "provider" in data[0]


def test_cost_with_gpu_filter(tmp_path):
    """Test filtering by specific GPU."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file), "--gpu", "H100"])
    assert result.exit_code == 0
    assert "H100" in result.output
    # Filtering to H100 must exclude other GPUs from the pricing table
    assert "RTX 4090" not in result.output


def test_cost_with_unknown_gpu(tmp_path):
    """Test filtering by unknown GPU."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file), "--gpu", "nonexistent"])
    assert result.exit_code == 1
    assert "No matching GPUs found" in result.output


def test_cost_missing_config():
    """Test cost fails gracefully when config doesn't exist."""
    result = runner.invoke(app, ["cost", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0


def test_cost_warns_when_dataset_unreadable(tmp_path):
    """When dataset cannot be read, user should see a fallback warning."""
    config_file = tmp_path / "soup.yaml"
    # train path points nowhere, so the loader cannot read it
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/missing_train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "Could not read training dataset" in result.output



@pytest.mark.parametrize(
    "num_examples", [None, 100], ids=["missing-splits", "known-split"]
)
@pytest.mark.parametrize(
    "has_validation", [False, True], ids=["no-validation", "has-validation"]
)
@pytest.mark.parametrize(
    "dataset_id", ["example/training-data", "teknium/OpenHermes-2.5"]
)
def test_cost_with_hub_split_metadata(
    tmp_path, monkeypatch, num_examples, has_validation, dataset_id
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "100")

    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        f"base: tiny\n"
        f"task: sft\n"
        f"data:\n"
        f"  train: {dataset_id}\n"
        f"  val_split: 0.1\n",
        encoding="utf-8",
    )

    splits = None
    if num_examples is not None:
        splits = {"train": SimpleNamespace(num_examples=num_examples)}
        if has_validation:
            splits["validation"] = SimpleNamespace(num_examples=20)

    requested = []

    def load_dataset_builder(path):
        requested.append(path)
        return SimpleNamespace(info=SimpleNamespace(splits=splits))

    datasets = ModuleType("datasets")
    datasets.load_dataset_builder = load_dataset_builder
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    estimated_sizes = []

    def capture_training_time(dataset_size, epochs, samples_per_sec):
        estimated_sizes.append(dataset_size)
        return 1.0

    monkeypatch.setattr(
        "soup_cli.commands.cost.estimate_training_time",
        capture_training_time,
    )

    result = runner.invoke(
        app, ["cost", "--config", str(config_file), "--gpu", "H100"]
    )

    assert result.exit_code == 0, result.exception
    assert requested == [dataset_id]

    expected_size = (
        10_000
        if num_examples is None
        else 100
        if has_validation
        else 90
    )
    assert estimated_sizes
    assert set(estimated_sizes) == {expected_size}

    text = " ".join(strip_ansi(result.stdout).split())
    assert "Training Cost Estimate" in text
    assert ("using default of 10,000 examples" in text) == (num_examples is None)



@pytest.mark.parametrize(
    "path, exists",
    [("missing.CSV", False), ("present.tsv", True), ("s3://bucket/dataset", False)],
)
def test_cost_does_not_query_hub_for_non_hub_inputs(tmp_path, monkeypatch, path, exists):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "100")
    if exists:
        (tmp_path / path).write_text("text\texample\n", encoding="utf-8")
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(f"base: tiny\ndata:\n  train: {path}\n", encoding="utf-8")

    def unexpected_hub_lookup(path):
        pytest.fail(f"non-Hub input reached the Hub loader: {path}")

    datasets = ModuleType("datasets")
    datasets.load_dataset_builder = unexpected_hub_lookup
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    result = runner.invoke(app, ["cost", "--config", str(config_file)])

    assert result.exit_code == 0, result.exception
    text = " ".join(strip_ansi(result.stdout).split())
    assert "using default of 10,000 examples" in text


def test_cost_shows_variance_disclaimer(tmp_path):
    """Table output must include a variance disclaimer for table consumers."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "estimates are approximate" in result.output
