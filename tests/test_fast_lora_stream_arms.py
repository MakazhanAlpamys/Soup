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

scaling_order
    The fast path folds the scaling into the addition
    (``torch.add(out, term, alpha=scaling)``); PEFT scales the LoRA term and
    then adds it.  This arm writes PEFT's order.

per_projection_a
    The fast path multiplies the input by the stacked LoRA A matrices in one
    GEMM and slices the result; PEFT multiplies per projection.  This arm
    splits the stacked A into per-projection GEMMs.  Only the stacked A is
    split: it holds ``r`` rows per projection, so its row count is small,
    while the base weights and the LoRA B matrices have the projection's
    output width in their rows.

single_projection_forward_dtype
    The single-projection ``o_proj`` forward reads ``quant_state.compute_dtype``
    directly and dequantises the base weight itself, so the existing dtype arm
    (which patches ``_projection_compute_dtype``) never reaches that forward.
    This arm overrides that one read with the activation dtype for the duration
    of the forward.  The forward then records the choice in the saved quant
    metadata, so the backward follows it as well.

mlp_backward_scaling_order
    The fused SwiGLU backward scales each LoRA gradient *after* its GEMM
    (``(grad @ h) * s``); PEFT scales the incoming gradient before the GEMM.
    This arm builds the MLP Function's backward from the PRODUCTION SOURCE with
    exactly those six post-GEMM scalings moved before their GEMMs.  It fails
    closed if the six sites are not found, so a kernel change cannot turn this
    arm into a silent no-op.  Nothing is copied by hand into this file.

    LIMIT: this fixture has r=4 and alpha=8, so the scaling is 2.0.  Multiplying
    by 2 is exact and commutes with a GEMM's rounding: ``(grad @ h) * s`` and
    ``(grad * s) @ h`` are the same bits.  "Reached, nothing moved" here rules
    out only the PLACEMENT of the scaling for this fixture; it cannot distinguish
    anything else.  The arm is informative at a scaling that is not a power of
    two (alpha=6 gives 1.5), which would be a separately labelled configuration
    whose baseline is not the recorded row.

Controls
--------
* two runs of the same configuration give the same numbers;
* every armed run reached its replacement (each arm counts its calls);
* on a card that runs the fixture at bf16 -- where the row is red -- with no
  arm the baseline gives the recorded numbers.

The "all arms give the PEFT error" premise is FALSE on a native-bf16 card: the
arms were reached there and did not move any name.  The all-arms ratios are
therefore printed as a RESULT (with whether the premise held), not asserted --
a harness control must not claim what the host did not demonstrate.

The report also prints the dtype each base GEMM asked for, including the
single-projection forward that the old trace omitted, so a card owner can see
which dtypes were in play.

The row needs CUDA; the dequantisation check runs on the CPU.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import textwrap

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
    "scaling_order",
    "per_projection_a",
    "single_projection_forward_dtype",
    "mlp_backward_scaling_order",
)

#: The production MLP backward's post-GEMM LoRA scalings.  Each is replaced by
#: a pre-GEMM scale of the SAME incoming gradient; the count is asserted so a
#: source change fails closed instead of silently dropping the arm.
_MLP_SCALING_SITES = (
    ("_flatten(_as_dtype(grad_y, hd.dtype)).t() @ _flatten(hd) * sd",
     "_flatten(_as_dtype(grad_y, hd.dtype) * sd).t() @ _flatten(hd)"),
    ("torch.matmul(_as_dtype(grad_y, bdl.dtype), bdl) * sd",
     "torch.matmul(_as_dtype(grad_y, bdl.dtype) * sd, bdl)"),
    ("_flatten(_as_dtype(grad_g, hg.dtype)).t() @ _flatten(hg) * sg",
     "_flatten(_as_dtype(grad_g, hg.dtype) * sg).t() @ _flatten(hg)"),
    ("torch.matmul(_as_dtype(grad_g, bgl.dtype), bgl) * sg",
     "torch.matmul(_as_dtype(grad_g, bgl.dtype) * sg, bgl)"),
    ("_flatten(_as_dtype(grad_u, hu.dtype)).t() @ _flatten(hu) * su",
     "_flatten(_as_dtype(grad_u, hu.dtype) * su).t() @ _flatten(hu)"),
    ("torch.matmul(_as_dtype(grad_u, bul.dtype), bul) * su",
     "torch.matmul(_as_dtype(grad_u, bul.dtype) * su, bul)"),
)


def _deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    bnb = pytest.importorskip("bitsandbytes")
    return torch, bnb


# ─── the arms ───────────────────────────────────────────────────────────────


def _scaled_lora_add_peft_order(out, hidden, lora_b, scaling, calls):
    """PEFT's order: scale the LoRA term, then add it."""
    import torch

    calls.append("scaling_order")
    term = torch.matmul(hidden, lora_b.t()) * float(scaling)
    promoted = torch.promote_types(out.dtype, term.dtype)
    return torch.add(out.to(promoted), term.to(promoted)).to(out.dtype)


def _per_projection_linear(original, calls, rows, max_rows):
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
            and 4 < weight.shape[0] <= max_rows
            and weight.shape[0] % 4 == 0
        ):
            calls.append("per_projection_a")
            rows.append(weight.shape[0])
            blocks = torch.split(weight, 4, dim=0)
            return torch.cat([original(input, block) for block in blocks], dim=-1)
        return original(input, weight, bias)

    return linear


def _in_activation_dtype(meta, fallback, calls):
    """The base matmul in the activation dtype, not the compute one."""
    calls.append("base_matmul_dtype_forward_and_backward")
    return fallback


def _move_mlp_scaling_before_gemm(source: str) -> tuple[str, int]:
    """Move each production post-GEMM LoRA scaling before its GEMM.

    Returns the edited source and the number of sites changed.  The caller
    requires exactly ``len(_MLP_SCALING_SITES)``; a different count means the
    production kernel changed and this arm must not be trusted.
    """
    edited = source
    count = 0
    for old, new in _MLP_SCALING_SITES:
        found = edited.count(old)
        count += found
        edited = edited.replace(old, new)
    return edited, count


def _mlp_backward_peft_order(fast_lora_mlp, calls):
    """Build the MLP Function's backward from its own source, with PEFT's order.

    The six scaling sites are inline in the production backward, so there is no
    helper to patch.  Reading and editing the installed source at arm-enable
    time keeps this a reading of the production path: no backward is written by
    hand here, and a source change that removes a site fails closed.
    """
    import torch

    function = fast_lora_mlp._mlp_function()
    original = inspect.unwrap(function.backward)
    source = textwrap.dedent(inspect.getsource(original))
    lines = source.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.lstrip().startswith("def backward")
    )
    body = "\n".join(lines[start:])
    edited, count = _move_mlp_scaling_before_gemm(body)
    expected = len(_MLP_SCALING_SITES)
    if count != expected:
        raise AssertionError(
            f"#792 MLP arm found {count} of {expected} scaling sites in the "
            "production backward; refusing to install an untrusted arm"
        )
    # The production backward closes over torch (imported inside _mlp_function),
    # so the exec namespace needs those cells as well as the module globals.
    namespace = dict(original.__globals__)
    for name, cell in zip(original.__code__.co_freevars, original.__closure__ or ()):
        namespace[name] = cell.cell_contents
    exec(compile(edited, "<#792 mlp backward arm>", "exec"), namespace)
    new_backward = namespace["backward"]

    def armed_backward(ctx, grad_y):
        calls.append("mlp_backward_scaling_order")
        return new_backward(ctx, grad_y)

    return function, staticmethod(
        torch.autograd.function.once_differentiable(armed_backward)
    )


@contextlib.contextmanager
def _arms_enabled(
    arms, monkeypatch, calls: list[str], trace: list[str] | None = None,
    rows: list[int] | None = None,
):
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
            _per_projection_linear(
                torch.nn.functional.linear, calls, rows if rows is not None else [], max_rows=16
            ),
        )
    if trace is not None or "single_projection_forward_dtype" in arms:
        # The single-projection forward reads quant_state.compute_dtype itself,
        # so the dtype arm above never reaches it.  Trace the read, and for the
        # arm override it with the activation dtype for the forward only.
        function = fast_lora._single_projection_function()
        original_forward = function.forward

        def forward_with_trace(
            ctx, x, weight, bias, lora_a, lora_b, scaling, quant_state,
            dropout_p, dropout_training,
        ):
            compute_dtype = getattr(quant_state, "compute_dtype", None)
            if trace is not None:
                trace.append(
                    "fast_lora._single_projection_forward("
                    f"compute_dtype={compute_dtype}, activation_dtype={x.dtype})"
                )
            if "single_projection_forward_dtype" in arms and quant_state is not None:
                calls.append("single_projection_forward_dtype")
                saved = getattr(quant_state, "compute_dtype", None)
                quant_state.compute_dtype = x.dtype
                try:
                    return original_forward(
                        ctx, x, weight, bias, lora_a, lora_b, scaling, quant_state,
                        dropout_p, dropout_training,
                    )
                finally:
                    quant_state.compute_dtype = saved
            return original_forward(
                ctx, x, weight, bias, lora_a, lora_b, scaling, quant_state,
                dropout_p, dropout_training,
            )

        monkeypatch.setattr(function, "forward", staticmethod(forward_with_trace))
    if "mlp_backward_scaling_order" in arms:
        function, armed_backward = _mlp_backward_peft_order(fast_lora_mlp, calls)
        monkeypatch.setattr(function, "backward", armed_backward)
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


# ─── CPU checks of the arms themselves ──────────────────────────────────────


def test_mlp_arm_source_edit_moves_exactly_the_six_scaling_sites():
    source = "def backward(ctx, grad_y):\n" + "\n".join(
        f"    {old}" for old, _ in _MLP_SCALING_SITES
    )
    edited, count = _move_mlp_scaling_before_gemm(source)
    assert count == len(_MLP_SCALING_SITES)
    for old, new in _MLP_SCALING_SITES:
        assert old not in edited
        assert new in edited


def test_mlp_arm_fails_closed_when_the_kernel_changed():
    source = "def backward(ctx, grad_y):\n    " + _MLP_SCALING_SITES[0][0]
    _, count = _move_mlp_scaling_before_gemm(source)
    assert count == 1 != len(_MLP_SCALING_SITES)


def test_mlp_arm_fails_closed_at_install_time(monkeypatch):
    """Deleting the raise must be caught: a missing site cannot become a no-op."""
    import sys

    pytest.importorskip("torch")
    from soup_cli.utils import fast_lora_mlp

    original = tuple(_MLP_SCALING_SITES)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_MLP_SCALING_SITES",
        original + (("a site that cannot occur", "replacement"),),
    )
    with pytest.raises(AssertionError, match="found 6 of 7"):
        _mlp_backward_peft_order(fast_lora_mlp, [])


def test_mlp_arm_builds_a_backward_from_the_installed_source():
    pytest.importorskip("torch")
    from soup_cli.utils import fast_lora_mlp

    calls: list[str] = []
    function, armed = _mlp_backward_peft_order(fast_lora_mlp, calls)
    assert function is fast_lora_mlp._mlp_function()
    assert callable(armed)
    assert calls == []  # counting happens on call, not at build time


def test_all_arms_result_reports_inert_arms_instead_of_asserting():
    baseline = {name: 3.0 for name in THE_THREE_NAMES}
    armed = {arm: dict(baseline) for arm in ARMS}
    inert = _all_arms_result(baseline, armed, dict(baseline))
    assert inert["premise_holds"] is False
    assert inert["near_one"] is False
    assert inert["verdict"] == "not_applicable_arms_inert"

    moved = {arm: dict(baseline) for arm in ARMS}
    moved["scaling_order"] = {name: 1.1 for name in THE_THREE_NAMES}
    near = _all_arms_result(baseline, moved, {name: 1.1 for name in THE_THREE_NAMES})
    assert near["premise_holds"] is True
    assert near["verdict"] == "near_one"

    far = _all_arms_result(baseline, moved, dict(baseline))
    assert far["premise_holds"] is True
    assert far["verdict"] == "not_near_one"


# ─── the row, with arms ─────────────────────────────────────────────────────


def _ratio_for(ratios: dict, name: str) -> float | None:
    """The ratio of the measured name that ends with ``name``."""
    for key, value in ratios.items():
        if f".{name}." in key:
            return value
    return None


def _all_arms_result(baseline: dict, armed: dict, every: dict) -> dict:
    """The all-arms control as a RESULT, not an assertion.

    Its old premise ("all arms bring the three names near 1") was never shown
    on a native-bf16 card, where the arms are reached but inert.  Report what
    happened: whether any arm moved a name, the every-arm ratios, and the
    near-one bound -- without turning a true observation into a failure.
    """
    premise = any(
        armed[arm][name] != baseline[name] for arm in armed for name in baseline
    )
    near = all(every[name] <= NEAR_ONE for name in baseline)
    return {
        "premise_holds": premise,
        "near_one": near,
        "near_one_bound": NEAR_ONE,
        "ratios": dict(every),
        "verdict": (
            "not_applicable_arms_inert"
            if not premise
            else ("near_one" if near else "not_near_one")
        ),
    }


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
    rows: list[int] = []
    with monkeypatch.context() as patch:
        with _arms_enabled(arms, patch, calls, trace, rows):
            try:
                _run_streamed_row(tmp_path, patch, "4bit")
            except AssertionError as exc:
                message = str(exc)
                if "#792 criterion violated" not in message or "drifted" in message:
                    raise
    printed = capsys.readouterr().out
    for line in printed.splitlines():
        if line.startswith("#792 {"):
            return json.loads(line[len("#792 "):]), trace, calls, rows
    raise AssertionError(f"the row printed no report:\n{printed[-2000:]}")


@pytest.mark.gpu
def test_the_arms_move_the_three_names(tmp_path, monkeypatch, capsys):
    """Each arm, and the controls, on the streamed `4bit` row.

    The row is red on a card that runs the fixture at bf16 and green elsewhere,
    so the two red-row controls are asserted only where the row is red, and the
    controls that hold everywhere (the same configuration twice, and every arm
    reached) are asserted always.
    """
    report, trace, _, _ = _row_report(tmp_path / "baseline", monkeypatch, capsys, ())
    baseline = report["ratios"]
    armed = {}
    arm_calls = {}
    per_projection_rows: list[int] = []
    for arm in ARMS:
        arm_report, _, calls, rows = _row_report(
            tmp_path / arm, monkeypatch, capsys, (arm,)
        )
        armed[arm] = arm_report["ratios"]
        arm_calls[arm] = calls
        if arm == "per_projection_a":
            per_projection_rows = rows
    every_report, _, every_calls, every_rows = _row_report(
        tmp_path / "every", monkeypatch, capsys, ARMS
    )
    again_report, _, _, _ = _row_report(tmp_path / "again", monkeypatch, capsys, ())

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
        "split_rows": sorted(set(per_projection_rows)),
        "every_split_rows": sorted(set(every_rows)),
    }
    with capsys.disabled():
        print("#792 arms " + json.dumps(measured))

    # Control: every arm was reached, on any card.
    for arm in ARMS:
        assert arm_calls[arm].count(arm) > 0, arm
        assert every_calls.count(arm) > 0, arm

    # The stacked A is 12 rows for q/k/v and 8 for gate/up: pin that the
    # split actually saw those two shapes, not just one-block rank-4 A's.
    assert 12 in per_projection_rows and 8 in per_projection_rows
    assert 12 in every_rows and 8 in every_rows

    # Control: the single-projection forward is traced too.  The old trace
    # listed only the MLP/QKV _dense_weight calls, which is why the o_proj
    # forward's dtype choice was invisible on the red card.
    assert any("fast_lora._single_projection_forward(" in line for line in trace)

    # Control: the same configuration twice gives the same numbers.
    assert again_report["ratios"] == baseline

    # The red-row baseline control: on a card running the fixture at bf16 the
    # row must be the recorded one.  This stays an assertion -- a different row
    # would make every reading below meaningless.
    # Key on the _dense_weight lines only.  The new single-projection trace
    # also contains "dtype=torch.bfloat16" and must not widen this gate.
    runs_at_bf16 = any(
        "_dense_weight(" in line and "dtype=torch.bfloat16" in line for line in trace
    )
    if runs_at_bf16:
        for name, recorded in RECORDED_RATIOS.items():
            assert _ratio_for(baseline, name) == pytest.approx(recorded, abs=1e-5), name

    # The old all-arms control asserted that all arms bring the three names
    # near 1.  That premise was not demonstrated on a native-bf16 card (the
    # arms were reached and inert), so print the result instead of claiming it.
    all_arms = _all_arms_result(
        names(baseline),
        {arm: names(ratios) for arm, ratios in armed.items()},
        names(every_report["ratios"]),
    )
    with capsys.disabled():
        print("#792 all-arms " + json.dumps(all_arms))
