from types import SimpleNamespace
import pytest

from soup_cli.config.schema import TrainingConfig
from soup_cli.utils.config_bounds import ROLLOUT_STREAM_TASKS, SUPPORTED_STREAM_TASKS
from soup_cli.utils.layer_stream import stream_arch_of


def test_stream_layers_description_scope():
    desc = TrainingConfig.model_fields["stream_layers"].description

    # Ensure all tasks are dynamically included
    all_tasks = set(SUPPORTED_STREAM_TASKS) | set(ROLLOUT_STREAM_TASKS)
    for task in all_tasks:
        assert task in desc

    # Ensure 4bit and NVMe disk tier are mentioned
    assert "4bit" in desc
    assert "NVMe disk tier" in desc

    # Ensure obsolete restrictions are removed
    assert "quantization=none only" not in desc
    assert "batch_size 1" not in desc
    assert "no gradient accumulation" not in desc


def test_stream_arch_of_runtime_message():
    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(SimpleNamespace(model_type="falcon"))

    err_msg = str(excinfo.value)
    assert "falcon" in err_msg
    assert "Supported:" in err_msg
    # Obsolete milestone tag must not be present
    assert "v0.72.3" not in err_msg