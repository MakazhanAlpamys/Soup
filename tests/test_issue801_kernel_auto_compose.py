"""Regression tests for issue #801's non-functional kernel selector."""

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def test_kernel_auto_compose_refuses_false_benchmark_claim() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SoupConfig(
            base="test/model",
            data={"train": "./data.jsonl"},
            training={"kernel_auto_compose": True},
        )

    message = str(exc_info.value)
    assert "same already-loaded model" in message
    assert "does not apply the selected flags" in message
    assert "use_liger" in message
    assert "use_flash_attn" in message


@pytest.mark.parametrize("value", [False, None])
def test_kernel_auto_compose_default_and_false_remain_valid(value: bool | None) -> None:
    training = {} if value is None else {"kernel_auto_compose": value}
    cfg = SoupConfig(
        base="test/model",
        data={"train": "./data.jsonl"},
        training=training,
    )

    assert cfg.training.kernel_auto_compose is False
