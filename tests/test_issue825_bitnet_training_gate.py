import pytest

from soup_cli.config.loader import load_config_from_string


@pytest.mark.parametrize("task", ["sft", "pretrain", "dpo"])
def test_bitnet_training_is_rejected_at_config_load(task: str) -> None:
    yaml = (
        "base: tiiuae/Falcon-E-1B-Instruct\n"
        f"task: {task}\n"
        "data: {train: ./d.jsonl}\n"
        "training: {quantization: bitnet_1.58}\n"
    )

    with pytest.raises(ValueError) as exc_info:
        load_config_from_string(yaml)

    message = str(exc_info.value)
    assert "BitNet 1.58 training is not implemented yet" in message
    assert "soup export --format tq1_0" in message
