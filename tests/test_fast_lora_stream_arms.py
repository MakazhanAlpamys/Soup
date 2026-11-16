"""#792 cause work: one arm per candidate operation on the streamed `4bit` row.

The streamed `4bit` row reads above 2 on three names on a native-bf16 card, and
the readings agree to every digit across three cards and two heads.  What is
missing is the cause, so this file measures it: each arm replaces exactly one
operation of the fast path with PEFT's, and the row is measured again against
the float64 oracle the row already builds.

The arms are patches, so no kernel changes are needed here.  A fix for the
cause is a separate claim; this file only measures.

Arms
----
base_weight_dtype
    The fast path multiplies the base with the weight dequantised into the
    compute dtype (fp16).  The streamed runtime and the oracle use the
    activation dtype (bf16), and bitsandbytes' ``MatMul4Bit.backward`` casts
    the dequantised weight to the grad dtype.  This arm drops the cast, so the
    fast arm multiplies with the weight the references multiply with.

scaling_order
    The fast path folds the scaling into the addition
    (``torch.add(out, term, alpha=scaling)``); PEFT scales the LoRA term and
    then adds it.  This arm writes PEFT's order.

per_projection_a
    The fast path multiplies the input by the stacked LoRA A matrices in one
    GEMM and slices the result; PEFT multiplies per projection.  This arm
    splits the stacked A into per-projection GEMMs.

Controls
--------
* no arm reproduces the row's own numbers (the recorded ones on a native-bf16
  card, this card's own numbers elsewhere);
* every arm on gives the PEFT error on the three names (a ratio near 1);
* two runs of the same configuration agree.

The row needs CUDA; the dequantisation check runs on the CPU.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

#: The three names the streamed `4bit` row reads above 2 on a native-bf16 card.
THE_THREE_NAMES = (
    "layers.3.mlp.up_proj.lora_B",
    "layers.0.self_attn.q_proj.lora_A",
    "layers.1.mlp.gate_proj.lora_B",
)

#: Every arm the harness can switch on.
ARMS = ("base_weight_dtype", "scaling_order", "per_projection_a")


def _deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    bnb = pytest.importorskip("bitsandbytes")
    return torch, bnb


# ─── the arms ───────────────────────────────────────────────────────────────


def _dense_weight_without_the_cast(weight, qmeta, qparts, dtype):
    """``_dense_weight`` that keeps the weight's own dtype.

    ``dequantize_4bit`` returns the weight in the dtype it was quantised from,
    which is the activation dtype for a bf16 model.  The fast path casts it to
    the compute dtype instead; this arm does not.
    """
    if qmeta is None:
        return weight
    from bitsandbytes.functional import dequantize_4bit

    from soup_cli.utils.fast_lora import _rebuild_quant_state

    state = _rebuild_quant_state(qmeta, qparts)
    return dequantize_4bit(weight, state)


def _scaled_lora_add_peft_order(out, hidden, lora_b, scaling):
    """PEFT's order: scale the LoRA term, then add it."""
    import torch

    term = torch.matmul(hidden, lora_b.t()) * float(scaling)
    promoted = torch.promote_types(out.dtype, term.dtype)
    return torch.add(out.to(promoted), term.to(promoted)).to(out.dtype)


def _per_projection_linear(original):
    """``F.linear`` that splits a stacked LoRA A into per-projection GEMMs.

    The adapters are fp32 while the base weights are bf16, and the stacked A
    holds one block of ``r`` rows per projection, so an fp32 weight whose row
    count is a multiple of the rank is the stacked A.  Splitting it changes
    the GEMM shape and nothing else.
    """
    import torch

    def linear(input, weight, bias=None):
        if bias is None and weight.dtype == torch.float32 and weight.shape[0] % 4 == 0:
            blocks = torch.split(weight, 4, dim=0)
            return torch.cat([original(input, block) for block in blocks], dim=-1)
        return original(input, weight, bias)

    return linear


@contextlib.contextmanager
def _arms_enabled(arms):
    """Enable the named arms for the duration of the block."""
    import torch

    from soup_cli.utils import fast_lora, fast_lora_mlp, fast_lora_qkv

    modules = (fast_lora, fast_lora_mlp, fast_lora_qkv)
    saved: list[tuple[Any, str, Any]] = []
    try:
        if "base_weight_dtype" in arms:
            for module in modules:
                saved.append((module, "_dense_weight", module._dense_weight))
                module._dense_weight = _dense_weight_without_the_cast
        if "scaling_order" in arms:
            for module in modules:
                saved.append((module, "_scaled_lora_add", module._scaled_lora_add))
                module._scaled_lora_add = _scaled_lora_add_peft_order
        if "per_projection_a" in arms:
            saved.append((torch.nn.functional, "linear", torch.nn.functional.linear))
            torch.nn.functional.linear = _per_projection_linear(torch.nn.functional.linear)
        yield
    finally:
        for module, name, old in saved:
            setattr(module, name, old)


# ─── the CPU dequantisation check ───────────────────────────────────────────


def test_the_rebuilt_quant_state_dequantises_to_the_same_weight(tmp_path):
    """The streamed route's rebuild is faithful: the same weight, bit for bit.

    The parts are split, saved to disk, loaded back and rebuilt exactly as the
    streamed route does, and the dequantised weight is compared with the one
    the original state gives.  The weight is quantised on CUDA when one is
    there and dequantised on the CPU, so this shows the rebuild is faithful on
    this device; the same comparison on the card belongs to the red-row report.
    """
    torch, bnb = _deps()
    from bitsandbytes.functional import dequantize_4bit

    from soup_cli.utils.fast_lora import _quant_state_parts, _rebuild_quant_state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    weight = torch.randn(64, 64, dtype=torch.float16, device=device)
    quantised, state = bnb.functional.quantize_4bit(weight, quant_type="nf4")
    original = dequantize_4bit(quantised, state)

    parts, meta = _quant_state_parts(state)
    for index, tensor in enumerate(parts):
        torch.save(tensor.cpu(), tmp_path / f"part_{index}.pt")
    loaded = [
        torch.load(tmp_path / f"part_{index}.pt", weights_only=True)
        for index in range(len(parts))
    ]
    rebuilt = _rebuild_quant_state(meta, loaded)

    rebuilt_weight = dequantize_4bit(quantised.cpu(), rebuilt)
    assert torch.equal(rebuilt_weight, original.cpu())


# ─── the row, with arms ─────────────────────────────────────────────────────


def _ratio_for(ratios: dict, name: str) -> float | None:
    """The ratio of the measured name that ends with ``name``."""
    for key, value in ratios.items():
        if key.endswith(name) or f".{name}." in key:
            return value
    return None


def _row_report(tmp_path, monkeypatch, capsys, arms) -> dict:
    """Run the streamed `4bit` row with the arms on; return the printed report.

    The row prints its whole ``#792 {...}`` line before it raises on a name
    above the criterion, so the report is read from the output either way.
    """
    from tests.test_fast_lora_flag import _run_streamed_row

    capsys.readouterr()  # drop an earlier report before this clean run
    tmp_path.mkdir(parents=True, exist_ok=True)
    with _arms_enabled(arms):
        try:
            _run_streamed_row(tmp_path, monkeypatch, "4bit")
        except AssertionError:
            pass  # the row is red; the report was printed before the raise
    printed = capsys.readouterr().out
    for line in printed.splitlines():
        if line.startswith("#792 {"):
            return json.loads(line[len("#792 "):])
    raise AssertionError(f"the row printed no report:\n{printed[-2000:]}")


@pytest.mark.gpu
def test_the_arms_move_the_three_names(tmp_path, monkeypatch, capsys):
    """Each arm, and the controls, on the streamed `4bit` row.

    The row is red on a native-bf16 card and green on this one, so the
    baseline is whatever this card prints and the arms are read against it.
    """
    baseline = _row_report(tmp_path / "baseline", monkeypatch, capsys, ())
    armed = {
        arm: _row_report(tmp_path / arm, monkeypatch, capsys, (arm,)) for arm in ARMS
    }
    every = _row_report(tmp_path / "every", monkeypatch, capsys, ARMS)
    again = _row_report(tmp_path / "again", monkeypatch, capsys, ())

    print("#792 arms " + json.dumps({
        "baseline": {name: _ratio_for(baseline["ratios"], name) for name in THE_THREE_NAMES},
        "arms": {
            arm: {name: _ratio_for(report["ratios"], name) for name in THE_THREE_NAMES}
            for arm, report in armed.items()
        },
        "every_arm": {
            name: _ratio_for(every["ratios"], name) for name in THE_THREE_NAMES
        },
        "logits": {
            "baseline": baseline["ratios"].get("logits"),
            "every_arm": every["ratios"].get("logits"),
        },
    }))

    # Control: the same configuration twice gives the same numbers.
    assert again["ratios"] == baseline["ratios"]
