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
base_matmul_dtype_forward_and_backward
    The fast path runs each 4-bit base matmul in bitsandbytes' compute dtype
    (fp16): the input is cast there, the weight is dequantised there, and the
    same choice is made in the backward.  The streamed runtime and the row's
    own oracle multiply with the weight in the activation dtype (bf16).  This
    arm changes that choice everywhere it is read, so it covers the forward
    and the backward together.

base_weight_dtype
    The same choice seen from the weight's side: the dequantised weight of a
    quantised projection keeps its own dtype instead of the requested one.
    A dense (not quantised) projection keeps the real behaviour.

scaling_order
    The fast path folds the scaling into the addition
    (``torch.add(out, term, alpha=scaling)``); PEFT scales the LoRA term and
    then adds it.  This arm writes PEFT's order.

per_projection_a
    The fast path multiplies the input by the stacked LoRA A matrices in one
    GEMM and slices the result; PEFT multiplies per projection.  This arm
    splits the stacked A into per-projection GEMMs.  Only the stacked A is
    split: it is an fp32 weight with few rows, where the reference's own fp32
    linears have many.

Controls
--------
* two runs of the same configuration give the same numbers;
* every armed run reached its replacement (each arm counts its calls);
* on a card that runs the fixture at bf16 -- where the row is red -- no arm
  gives the recorded numbers and every arm brings the three names to a ratio
  near 1 (at most 1.5 here).

The report also prints the dtype each base GEMM asked for, so a card owner can
see which dtypes were in play.

The row needs CUDA; the dequantisation check runs on the CPU.
"""

from __future__ import annotations

import contextlib
import json

import pytest

#: The three names the streamed `4bit` row reads above 2 on a native-bf16 card.
THE_THREE_NAMES = (
    "layers.3.mlp.up_proj.lora_B",
    "layers.0.self_attn.q_proj.lora_A",
    "layers.1.mlp.gate_proj.lora_B",
)

#: The ratios the same row is recorded with on a native-bf16 card.
RECORDED_RATIOS = {
    "layers.3.mlp.up_proj.lora_B": 3.038320,
    "layers.0.self_attn.q_proj.lora_A": 2.311347,
    "layers.1.mlp.gate_proj.lora_B": 2.007616,
}

#: How near 1 "every arm gives the PEFT error" is taken to mean here.
NEAR_ONE = 1.5

#: Every arm the harness can switch on.
ARMS = (
    "base_matmul_dtype_forward_and_backward",
    "base_weight_dtype",
    "scaling_order",
    "per_projection_a",
)


def _deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    bnb = pytest.importorskip("bitsandbytes")
    return torch, bnb


# ─── the arms ───────────────────────────────────────────────────────────────


def _dense_weight_without_the_cast(weight, qmeta, qparts, dtype, calls):
    """``_dense_weight`` that keeps a quantised weight's own dtype.

    ``dequantize_4bit`` returns the weight in the dtype it was quantised from,
    which is the compute dtype for weights quantised from it, so this arm only
    moves a weight whose own dtype differs from the requested one.  A dense
    projection is not quantised and keeps the real behaviour.
    """
    calls.append("base_weight_dtype")
    if qmeta is None:
        from soup_cli.utils.fast_lora import _as_dtype

        return _as_dtype(weight, dtype)
    from bitsandbytes.functional import dequantize_4bit

    from soup_cli.utils.fast_lora import _rebuild_quant_state

    state = _rebuild_quant_state(qmeta, qparts)
    return dequantize_4bit(weight, state)


def _scaled_lora_add_peft_order(out, hidden, lora_b, scaling, calls):
    """PEFT's order: scale the LoRA term, then add it."""
    import torch

    calls.append("scaling_order")
    term = torch.matmul(hidden, lora_b.t()) * float(scaling)
    promoted = torch.promote_types(out.dtype, term.dtype)
    return torch.add(out.to(promoted), term.to(promoted)).to(out.dtype)


def _per_projection_linear(original, calls, max_rows):
    """``F.linear`` that splits a stacked LoRA A into per-projection GEMMs.

    The stacked A holds ``r`` rows per projection, so its row count is small
    (three projections of rank four are twelve rows here), while the base
    weights and the LoRA B matrices have the projection's output width in
    their rows.  Splitting on the row count alone therefore leaves the
    reference's own linears alone.
    """
    import torch

    def linear(input, weight, bias=None):
        if (
            bias is None
            and weight.shape[0] <= max_rows
            and weight.shape[0] % 4 == 0
        ):
            calls.append("per_projection_a")
            blocks = torch.split(weight, 4, dim=0)
            return torch.cat([original(input, block) for block in blocks], dim=-1)
        return original(input, weight, bias)

    return linear


def _in_activation_dtype(meta, fallback, calls):
    """The base matmul in the activation dtype, not the compute one."""
    calls.append("base_matmul_dtype_forward_and_backward")
    return fallback


@contextlib.contextmanager
def _arms_enabled(arms, monkeypatch, calls: list[str], trace: list[str] | None = None):
    """Enable the named arms for the duration of the block."""
    import torch

    from soup_cli.utils import fast_lora, fast_lora_mlp, fast_lora_qkv

    modules = (fast_lora, fast_lora_mlp, fast_lora_qkv)
    if "base_matmul_dtype_forward_and_backward" in arms:
        for module in modules:
            monkeypatch.setattr(
                module,
                "_projection_compute_dtype",
                lambda meta, fallback, _calls=calls: _in_activation_dtype(
                    meta, fallback, _calls
                ),
            )
    if "base_weight_dtype" in arms:
        for module in modules:
            monkeypatch.setattr(
                module,
                "_dense_weight",
                lambda weight, qmeta, qparts, dtype, _calls=calls, _m=module: (
                    _dense_weight_without_the_cast(
                        weight, qmeta, qparts, dtype, _calls
                    )
                ),
            )
    if "scaling_order" in arms:
        for module in modules:
            monkeypatch.setattr(
                module,
                "_scaled_lora_add",
                lambda out, hidden, lora_b, scaling, _calls=calls: (
                    _scaled_lora_add_peft_order(out, hidden, lora_b, scaling, _calls)
                ),
            )
    if "per_projection_a" in arms:
        monkeypatch.setattr(
            torch.nn.functional,
            "linear",
            _per_projection_linear(torch.nn.functional.linear, calls, max_rows=16),
        )
    if trace is not None:
        for module in modules:
            original = module._dense_weight

            def wrapper(*args, _original=original, _name=module.__name__, **kwargs):
                weight = args[0] if args else None
                dtype = args[3] if len(args) > 3 else kwargs.get("dtype")
                trace.append(
                    f"{_name.rsplit('.', 1)[-1]}._dense_weight("
                    f"weight={getattr(weight, 'dtype', None)}, dtype={dtype})"
                )
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, "_dense_weight", wrapper)
    yield


# ─── the CPU dequantisation check ───────────────────────────────────────────


def test_the_rebuilt_quant_state_dequantises_to_the_same_weight(tmp_path):
    """The streamed route's rebuild is faithful: the same weight, bit for bit.

    A quant state is built by hand, its parts are split, saved to disk, loaded
    back and rebuilt exactly as the streamed route does, and the dequantised
    weight is compared with the one the original state gives.  Both
    dequantisations run on the CPU, through bitsandbytes' CPU path, so this
    shows the rebuild is faithful on this device; the same comparison on the
    card belongs to the red-row report.
    """
    torch, bnb = _deps()
    from bitsandbytes.functional import QuantState, dequantize_4bit

    from soup_cli.utils.fast_lora import _quant_state_parts, _rebuild_quant_state

    torch.manual_seed(0)
    size, blocks = 64, 64
    code = torch.randint(0, 255, (size * size // 2,), dtype=torch.uint8)
    absmax = torch.rand(size * size // blocks, dtype=torch.float32) + 0.5
    state = QuantState(
        absmax=absmax,
        shape=torch.Size((size, size)),
        dtype=torch.float16,
        blocksize=blocks,
        code=code,
        quant_type="nf4",
    )
    original = dequantize_4bit(code, state)

    parts, meta = _quant_state_parts(state)
    for index, tensor in enumerate(parts):
        torch.save(tensor, tmp_path / f"part_{index}.pt")
    loaded = [
        torch.load(tmp_path / f"part_{index}.pt", weights_only=True)
        for index in range(len(parts))
    ]
    rebuilt = _rebuild_quant_state(meta, loaded)

    rebuilt_weight = dequantize_4bit(code, rebuilt)
    assert torch.equal(rebuilt_weight, original)
    assert original.device.type == "cpu"


# ─── the row, with arms ─────────────────────────────────────────────────────


def _ratio_for(ratios: dict, name: str) -> float | None:
    """The ratio of the measured name that ends with ``name``."""
    for key, value in ratios.items():
        if f".{name}." in key:
            return value
    return None


def _row_report(tmp_path, monkeypatch, capsys, arms) -> tuple[dict, list[str], list[str]]:
    """Run the streamed `4bit` row with the arms on; return the report.

    The row prints its whole ``#792 {...}`` line before it raises on a name
    above the criterion, so the report is read from the output either way.
    Only that one failure is accepted: the row's other assertions (the oracle
    drifting away from the arms' weights, a name not measured) mean the
    numbers are meaningless and must not pass as a reading.
    """
    from tests.test_fast_lora_flag import _run_streamed_row

    capsys.readouterr()  # drop an earlier report before this clean run
    tmp_path.mkdir(parents=True, exist_ok=True)
    trace: list[str] = []
    calls: list[str] = []
    with _arms_enabled(arms, monkeypatch, calls, trace):
        try:
            _run_streamed_row(tmp_path, monkeypatch, "4bit")
        except AssertionError as exc:
            message = str(exc)
            if "#792 criterion violated" not in message or "drifted" in message:
                raise
    printed = capsys.readouterr().out
    for line in printed.splitlines():
        if line.startswith("#792 {"):
            return json.loads(line[len("#792 "):]), trace, calls
    raise AssertionError(f"the row printed no report:\n{printed[-2000:]}")


@pytest.mark.gpu
def test_the_arms_move_the_three_names(tmp_path, monkeypatch, capsys):
    """Each arm, and the controls, on the streamed `4bit` row.

    The row is red on a card that runs the fixture at bf16 and green elsewhere,
    so the two red-row controls are asserted only where the row is red, and the
    controls that hold everywhere (the same configuration twice, and every arm
    reached) are asserted always.
    """
    report, trace, _ = _row_report(tmp_path / "baseline", monkeypatch, capsys, ())
    baseline = report["ratios"]
    armed = {}
    arm_calls = {}
    for arm in ARMS:
        arm_report, _, calls = _row_report(tmp_path / arm, monkeypatch, capsys, (arm,))
        armed[arm] = arm_report["ratios"]
        arm_calls[arm] = calls
    every_report, _, every_calls = _row_report(tmp_path / "every", monkeypatch, capsys, ARMS)
    again_report, _, _ = _row_report(tmp_path / "again", monkeypatch, capsys, ())

    def names(ratios: dict) -> dict:
        return {name: _ratio_for(ratios, name) for name in THE_THREE_NAMES}

    measured = {
        "baseline": names(baseline),
        "arms": {arm: names(ratios) for arm, ratios in armed.items()},
        "every_arm": names(every_report["ratios"]),
        "logits": {
            "baseline": baseline.get("logits"),
            "every_arm": every_report["ratios"].get("logits"),
        },
        "base_gemm_dtypes": sorted(set(trace)),
        "arm_calls": {arm: calls.count(arm) for arm, calls in arm_calls.items()},
        "every_arm_calls": {arm: every_calls.count(arm) for arm in ARMS},
    }
    with capsys.disabled():
        print("#792 arms " + json.dumps(measured))

    # Control: every arm was reached, on any card.
    for arm in ARMS:
        assert arm_calls[arm].count(arm) > 0, arm
        assert every_calls.count(arm) > 0, arm

    # Control: the same configuration twice gives the same numbers.
    assert again_report["ratios"] == baseline

    # The red-row controls, where the row is red: the recorded numbers with no
    # arm, and the PEFT error with every arm.
    runs_at_bf16 = any("dtype=torch.bfloat16" in line for line in trace)
    if runs_at_bf16:
        for name, recorded in RECORDED_RATIOS.items():
            assert _ratio_for(baseline, name) == pytest.approx(recorded, abs=1e-5), name
        for name in THE_THREE_NAMES:
            assert _ratio_for(every_report["ratios"], name) <= NEAR_ONE, name
