"""Issue #2409 -- soup ship's live eval loads the dtype training would pick on that card."""

from __future__ import annotations

import pytest

from soup_cli.utils.ship_verdict import (
    NUMERICS_FAMILY_FULL,
    numerics_family,
    parse_numerics,
)

pytestmark = pytest.mark.unit


def _flags(monkeypatch, *, bf16: bool, fp16: bool) -> None:
    monkeypatch.setattr(
        "soup_cli.utils.gpu.bf16_fp16_flags",
        lambda device, **kwargs: (bf16, fp16),
    )


def test_cuda_card_with_bf16_stamps_bfloat16(monkeypatch):
    from soup_cli.commands.ship import _live_eval_numerics

    _flags(monkeypatch, bf16=True, fp16=False)

    assert _live_eval_numerics(None, "cuda") == "bfloat16"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_cuda_card_without_bf16_stamps_float16(monkeypatch, device):
    from soup_cli.commands.ship import _live_eval_numerics

    _flags(monkeypatch, bf16=False, fp16=True)

    assert _live_eval_numerics(None, device) == "float16"


def test_cpu_still_stamps_float32(monkeypatch):
    from soup_cli.commands.ship import _live_eval_numerics

    _flags(monkeypatch, bf16=False, fp16=True)

    assert _live_eval_numerics(None, "cpu") == "float32"


def test_quantized_load_still_stamps_its_format(monkeypatch):
    from soup_cli.commands.ship import _live_eval_numerics

    _flags(monkeypatch, bf16=False, fp16=True)

    assert _live_eval_numerics("4bit", "cuda") == "4bit"


def test_float16_is_an_accepted_full_precision_stamp():
    assert parse_numerics("float16") == "float16"
    assert numerics_family("float16") == NUMERICS_FAMILY_FULL


def test_the_dtype_that_reaches_the_load_is_the_stamp(monkeypatch):
    """The console line, the verdict and the model load must name one dtype."""
    from soup_cli.commands import ship
    from soup_cli.utils import live_eval

    _flags(monkeypatch, bf16=False, fp16=True)
    seen = []

    def fake_make_generator(base, **kwargs):
        seen.append(kwargs["dtype"])
        return lambda prompt: ""

    monkeypatch.setattr(live_eval, "make_generator", fake_make_generator)

    ship._resolve_generators("base", "tuned", None, "cuda", quantization=None)

    assert seen and set(seen) == {"float16"}
    assert ship._live_eval_numerics(None, "cuda") == "float16"
