"""Smoke test for the full training pipeline.

Uses a tiny model (sshleifer/tiny-gpt2) to verify the entire flow:
config → data loading → model setup → training → output.

Marked as slow — skipped by default, run with: pytest -m smoke
"""

import json
from pathlib import Path

import pytest

# Mark all tests in this module as "smoke" so they can be skipped by default
pytestmark = pytest.mark.smoke


@pytest.fixture
def tiny_train_data(tmp_path: Path) -> Path:
    """Create a tiny ChatML-format JSONL file for smoke testing."""
    data_path = tmp_path / "train.jsonl"
    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "Hello!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "Blue"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Name a fruit"},
                {"role": "assistant", "content": "Apple"},
            ]
        },
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return data_path


@pytest.fixture
def tiny_dpo_data(tmp_path: Path) -> Path:
    """Create a tiny DPO preference JSONL file for smoke testing."""
    data_path = tmp_path / "dpo_train.jsonl"
    samples = [
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "I don't know"},
        {"prompt": "Say hello", "chosen": "Hello!", "rejected": "Go away"},
        {"prompt": "What color is sky?", "chosen": "Blue", "rejected": "Green"},
        {"prompt": "Name a fruit", "chosen": "Apple", "rejected": "Chair"},
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return data_path


@pytest.fixture
def sft_config_yaml(tmp_path: Path, tiny_train_data: Path) -> Path:
    """Create a minimal SFT config for smoke testing with tiny-gpt2."""
    config_path = tmp_path / "soup.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        f"""base: sshleifer/tiny-gpt2
task: sft

data:
  train: {tiny_train_data}
  format: chatml
  val_split: 0.0
  max_length: 128

training:
  epochs: 1
  lr: 5e-4
  batch_size: 2
  gradient_accumulation_steps: 1
  lora:
    r: 4
    alpha: 8
  quantization: none
  save_steps: 999
  logging_steps: 1

output: {output_dir}
""",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def dpo_config_yaml(tmp_path: Path, tiny_dpo_data: Path) -> Path:
    """Create a minimal DPO config for smoke testing with tiny-gpt2."""
    config_path = tmp_path / "soup_dpo.yaml"
    output_dir = tmp_path / "output_dpo"
    config_path.write_text(
        f"""base: sshleifer/tiny-gpt2
task: dpo

data:
  train: {tiny_dpo_data}
  format: dpo
  val_split: 0.0
  max_length: 128

training:
  epochs: 1
  lr: 5e-4
  batch_size: 2
  gradient_accumulation_steps: 1
  dpo_beta: 0.1
  lora:
    r: 4
    alpha: 8
  quantization: none
  save_steps: 999
  logging_steps: 1

output: {output_dir}
""",
        encoding="utf-8",
    )
    return config_path


def test_sft_smoke(sft_config_yaml: Path):
    """Full SFT training pipeline smoke test with tiny-gpt2."""
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.gpu import detect_device

    # 1. Load config
    cfg = load_config(sft_config_yaml)
    assert cfg.task == "sft"
    assert cfg.base == "sshleifer/tiny-gpt2"

    # 2. Detect device
    device, device_name = detect_device()

    # 3. Load data
    dataset = load_dataset(cfg.data)
    assert len(dataset["train"]) >= 3

    # 4. Setup trainer
    trainer = SFTTrainerWrapper(cfg, device=device)
    trainer.setup(dataset)
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.trainer is not None

    # 5. Train
    result = trainer.train()
    assert "final_loss" in result
    assert "output_dir" in result
    assert result["total_steps"] > 0

    # 6. Check output
    output_dir = Path(result["output_dir"])
    assert output_dir.exists()
    assert (output_dir / "adapter_config.json").exists()


def test_dpo_smoke(dpo_config_yaml: Path):
    """Full DPO training pipeline smoke test with tiny-gpt2."""
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from soup_cli.utils.gpu import detect_device

    # 1. Load config
    cfg = load_config(dpo_config_yaml)
    assert cfg.task == "dpo"

    # 2. Detect device
    device, device_name = detect_device()

    # 3. Load data
    dataset = load_dataset(cfg.data)
    assert len(dataset["train"]) >= 3

    # 4. Setup trainer
    trainer = DPOTrainerWrapper(cfg, device=device)
    trainer.setup(dataset)
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.trainer is not None

    # 5. Train
    result = trainer.train()
    assert "final_loss" in result
    assert "output_dir" in result
    assert result["total_steps"] > 0

    # 6. Check output
    output_dir = Path(result["output_dir"])
    assert output_dir.exists()
    assert (output_dir / "adapter_config.json").exists()
