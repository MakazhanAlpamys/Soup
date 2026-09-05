"""v0.72.3 — Layer streaming: breadth.

Six items, each with its own gate because each fails silently in its own way.
Gate numbers and method live in ``.claude/v0723-gate-results.md``.

The inherited standard for every item is unchanged from v0.72.0: **a streamed run
must be bit-exact against the resident run of the same numerics**. What changes
per item is the *reference*, not the standard. Gradient accumulation is the one
exception the brief names — its gate is a measured I/O cost, not an equality.
"""

import pathlib

import pytest

from soup_cli.utils import layer_stream, layer_stream_runtime

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _cuda() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a [train] extra
        return False
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")


# ==========================================================================
# item 2 — pre-flight VRAM budget (batch- and vocab-aware)
# ==========================================================================
#: The GATE 2 measurement grid, verbatim. Each row is a real streamed
#: forward+backward+step on this box with ``torch.cuda.max_memory_allocated()``
#: recorded. These are the numbers the estimator is accountable to; a change
#: that stops reproducing them is a regression in the estimator, not in the test.
_SMOL = dict(pool=14160384, extras=56624256, adapter=921600, vocab=49152, hidden=576,
             intermediate=1536, n_layers=30)
_QWEN = dict(pool=59649536, extras=272271104, adapter=1081344, vocab=151936, hidden=896,
             intermediate=4864, n_layers=24)

MEASURED_VRAM_GRID = [
    dict(label="SmolLM2-135M B1 S256", batch=1, seq=256, peak=284555264, **_SMOL),
    dict(label="SmolLM2-135M B1 S512", batch=1, seq=512, peak=470879232, **_SMOL),
    dict(label="SmolLM2-135M B2 S512", batch=2, seq=512, peak=842736640, **_SMOL),
    dict(label="SmolLM2-135M B4 S512", batch=4, seq=512, peak=1584485376, **_SMOL),
    dict(label="SmolLM2-135M B8 S512", batch=8, seq=512, peak=3068638208, **_SMOL),
    dict(label="Qwen2.5-0.5B B1 S256", batch=1, seq=256, peak=919886336, **_QWEN),
    dict(label="Qwen2.5-0.5B B1 S512", batch=1, seq=512, peak=1476423168, **_QWEN),
    dict(label="Qwen2.5-0.5B B2 S512", batch=2, seq=512, peak=2592245248, **_QWEN),
    dict(label="Qwen2.5-0.5B B4 S512", batch=4, seq=512, peak=4818908672, **_QWEN),
    dict(label="Qwen2.5-0.5B B8 S512", batch=8, seq=512, peak=9266992640, **_QWEN),
]

# ---------------------------------------------------------------------------
# #395 — the SECOND stack. Deliberately NOT part of MEASURED_VRAM_GRID above:
# that grid carries a <1% accuracy claim fitted on the RTX 3050 / Windows /
# torch 2.5.1 / transformers 4.57.6 box, and these rows are ~16% off it. Merging
# them would destroy a real claim rather than widen it.
#
# What they DO carry is the direction property at sequence lengths the first
# grid never reached, which is the hole #395 names. Measured on an A10G 23 GB /
# Ubuntu 22.04 / torch 2.13.0+cu130 / transformers 5.16.1 / trl 0.29.1, real
# `soup train` setup + one step, bf16, batch 1, quantization none, LoRA r=8.
# Full record: benchmarks/gate-395-second-stack-vram.md
SECOND_STACK_VRAM_GRID = [
    dict(label="A10G SmolLM2-135M B1 S2048", batch=1, seq=2048, peak=1365200000, **_SMOL),
    dict(label="A10G SmolLM2-135M B1 S3072", batch=1, seq=3072, peak=2016700000, **_SMOL),
    dict(label="A10G SmolLM2-135M B1 S4096", batch=1, seq=4096, peak=2657800000, **_SMOL),
    dict(label="A10G SmolLM2-135M B1 S4352", batch=1, seq=4352, peak=2818400000, **_SMOL),
    dict(label="A10G SmolLM2-135M B1 S5120", batch=1, seq=5120, peak=3303000000, **_SMOL),
    dict(label="A10G SmolLM2-135M B1 S6144", batch=1, seq=6144, peak=3943000000, **_SMOL),
    dict(label="A10G Qwen2.5-0.5B B1 S2048", batch=1, seq=2048, peak=4177000000, **_QWEN),
    dict(label="A10G Qwen2.5-0.5B B1 S4096", batch=1, seq=4096, peak=8012300000, **_QWEN),
    dict(label="A10G Qwen2.5-0.5B B1 S5120", batch=1, seq=5120, peak=9929400000, **_QWEN),
    dict(label="A10G Qwen2.5-0.5B B1 S6144", batch=1, seq=6144, peak=11848100000, **_QWEN),
]

# Accounting control for #324. This is deliberately separate from the measured
# accuracy grid above: its peak is the first real row plus one additive
# vocabulary slot, so it guards composition without pretending to be another
# hardware measurement.
_LARGE_SLOT_ACCOUNTING_ROW = {
    **MEASURED_VRAM_GRID[0],
    "label": "SmolLM2-135M B1 S256 + 16 MiB vocabulary-slot control",
    "large_layer_bytes": 16 * 1024 * 1024,
    "peak": MEASURED_VRAM_GRID[0]["peak"] + 16 * 1024 * 1024,
}


def _predict(row):
    from soup_cli.utils.layer_stream import estimate_stream_peak_vram

    return estimate_stream_peak_vram(
        layer_bytes=row["pool"] // 2,
        buffers=2,
        extras_bytes=row["extras"],
        adapter_params=row["adapter"],
        vocab_size=row["vocab"],
        hidden_size=row["hidden"],
        intermediate_size=row["intermediate"],
        n_layers=row["n_layers"],
        seq_len=row["seq"],
        batch_size=row["batch"],
        large_layer_bytes=row.get("large_layer_bytes", 0),
    )


class TestSecondStackDirectionProperty:
    """#395 — the direction property at seq 2048..6144, on a second GPU/stack.

    The first grid's evidence stops at seq 512, which is the hole #395 names.
    These rows extend the sequence axis by a factor of 12 on two models whose
    vocabularies differ 3.1x, and they are the only evidence in this file taken
    on hardware other than the RTX 3050.

    They assert DIRECTION, not accuracy. The formula over-predicts by ~16% here
    (see test_second_stack_gap_is_the_logits_term_alone for why that is one
    named term rather than drift), so folding them into the <1% accuracy grid
    would break a claim that is true on the stack it was fitted to.
    """

    @pytest.mark.parametrize(
        "row", SECOND_STACK_VRAM_GRID, ids=lambda r: r["label"]
    )
    def test_never_under_predicts_on_the_second_stack(self, row):
        assert _predict(row) >= row["peak"], row["label"]

    @pytest.mark.parametrize(
        "row", SECOND_STACK_VRAM_GRID, ids=lambda r: r["label"]
    )
    def test_sequence_axis_is_actually_exercised(self, row):
        """The rows are only worth anything if they are past the first grid's
        ceiling — a regression that quietly shortened them would leave this file
        asserting the same seq<=512 regime twice."""
        assert row["seq"] >= 2048

    def test_second_stack_has_no_retained_logits_copy(self):
        """The #395 finding, stated as the arithmetic that forces it.

        The shipped 14 is ``LOGITS_LOSS_BYTES_PER_ELEMENT`` (12, measured at
        12.000000 with zero spread on this stack too) plus 2 for one further
        bf16 logits-shaped tensor whose retention #327 has not explained.

        If that copy were retained on the A10G, the whole real peak would have
        to exceed the logits term at 14 alone. On every row it does not, which
        leaves no room for the non-logits terms let alone the copy. That is a
        constraint on any explanation of #327's retention, and it is why this
        grid over-predicts rather than under-predicting.

        An earlier revision asserted a back-solved 11.83 B/element here. That
        was withdrawn: it attributed the whole discrepancy to the logits
        multiplier, when the multiplier is pinned at 12 on both stacks and the
        gap is the absent copy plus a ~15% over-estimate of the non-logits
        terms (benchmarks/gate-395-second-stack-vram.md).
        """
        from soup_cli.utils.layer_stream import (
            LOGITS_BYTES_PER_ELEMENT,
            estimate_logits_bytes,
        )

        for row in SECOND_STACK_VRAM_GRID:
            logits_at_shipped = estimate_logits_bytes(
                vocab_size=row["vocab"], seq_len=row["seq"], batch_size=row["batch"]
            )
            assert logits_at_shipped > row["peak"], (
                f"{row['label']}: the logits term at "
                f"{LOGITS_BYTES_PER_ELEMENT} B/element is "
                f"{logits_at_shipped} but the whole measured peak is "
                f"{row['peak']} - the retained bf16 copy would fit, so this "
                f"row no longer supports the finding"
            )

    def test_the_non_logits_over_estimate_is_flat(self):
        """Holding the MEASURED 12 fixed, the residual is a fixed fraction.

        Flat across a 3.1x vocabulary contrast and a 3x sequence range argues
        for a fixed-fraction over-estimate rather than a term that grows with
        either. If this spread widened, the decomposition in the record would
        no longer hold.
        """
        from soup_cli.utils.layer_stream import (
            LOGITS_LOSS_BYTES_PER_ELEMENT,
            estimate_logits_bytes,
        )

        ratios = []
        for row in SECOND_STACK_VRAM_GRID:
            elements = row["seq"] * row["vocab"] * row["batch"]
            logits_at_shipped = estimate_logits_bytes(
                vocab_size=row["vocab"], seq_len=row["seq"], batch_size=row["batch"]
            )
            non_logits_modelled = _predict(row) - logits_at_shipped
            non_logits_true = row["peak"] - LOGITS_LOSS_BYTES_PER_ELEMENT * elements
            assert non_logits_true > 0, row["label"]
            ratios.append(non_logits_modelled / non_logits_true)

        mean = sum(ratios) / len(ratios)
        assert 1.10 < mean < 1.20, f"non-logits over-estimate moved: {mean}"
        assert max(ratios) - min(ratios) < 0.10, f"no longer flat: {ratios}"


class TestLogitsBytesIsMeasuredNotDerived:
    """The v0.72.0 estimate charged 6 bytes per logit element (bf16 + fp32
    upcast). GATE 2 measured 14 — transformers' ForCausalLMLoss holds the bf16
    logits, the fp32 upcast, log_softmax's fp32 output and the fp32 gradient
    live at once. Under-predicting this term by 2.33x is a ~5 GB error on a
    152k-vocab model at batch 8, which is the whole reason a fit gate exists."""

    def test_charges_the_measured_fourteen_bytes_per_element(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        got = estimate_logits_bytes(vocab_size=151936, seq_len=512, batch_size=1)
        assert got == 512 * 151936 * 14

    def test_is_not_the_old_first_principles_six(self):
        """Control: without this the test above passes for any constant."""
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        got = estimate_logits_bytes(vocab_size=1000, seq_len=8, batch_size=1)
        assert got != 8 * 1000 * 6

    def test_scales_with_batch(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        one = estimate_logits_bytes(vocab_size=1000, seq_len=8, batch_size=1)
        four = estimate_logits_bytes(vocab_size=1000, seq_len=8, batch_size=4)
        assert four == 4 * one

    def test_without_the_loss_only_the_bf16_logits_are_live(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        got = estimate_logits_bytes(
            vocab_size=1000, seq_len=10, batch_size=1, upcast_fp32=False
        )
        assert got == 10 * 1000 * 2

    def test_rejects_non_positive_dimensions(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        with pytest.raises(ValueError, match="positive"):
            estimate_logits_bytes(vocab_size=0, seq_len=8, batch_size=1)
        with pytest.raises(ValueError, match="positive"):
            estimate_logits_bytes(vocab_size=8, seq_len=8, batch_size=0)


class TestActivationBytes:
    def test_boundary_term_scales_with_layers_but_transient_does_not(self):
        """The checkpoint boundary save is one copy PER LAYER; the recompute
        transient is one layer's worth regardless of depth. That separation is
        the entire memory argument for streaming, so it is pinned."""
        from soup_cli.utils.layer_stream import estimate_activation_bytes

        kw = dict(hidden_size=576, intermediate_size=1536, seq_len=512, batch_size=1)
        ten = estimate_activation_bytes(n_layers=10, **kw)
        twenty = estimate_activation_bytes(n_layers=20, **kw)
        # Only the 2*n_layers*hidden term moved.
        assert twenty - ten == 10 * 2 * 576 * 512

    def test_scales_linearly_in_batch_and_seq(self):
        from soup_cli.utils.layer_stream import estimate_activation_bytes

        kw = dict(hidden_size=64, intermediate_size=128, n_layers=4)
        base = estimate_activation_bytes(seq_len=100, batch_size=1, **kw)
        assert estimate_activation_bytes(seq_len=100, batch_size=4, **kw) == 4 * base
        assert estimate_activation_bytes(seq_len=400, batch_size=1, **kw) == 4 * base


class TestPeakVramReproducesTheMeasuredGrid:
    """GATE 2: worst absolute error 0.85% over 10 real runs spanning two models,
    a 3.1x vocab contrast, batch 1..8 and two sequence lengths."""

    @pytest.mark.parametrize("row", MEASURED_VRAM_GRID, ids=lambda r: r["label"])
    def test_within_one_percent_of_measured(self, row):
        measured = row["peak"]
        predicted = _predict(row)
        err = abs(predicted - measured) / measured
        assert err < 0.01, f"predicted {predicted} vs measured {measured} ({err:.2%})"

    @pytest.mark.parametrize(
        "row", MEASURED_VRAM_GRID + [_LARGE_SLOT_ACCOUNTING_ROW], ids=lambda r: r["label"]
    )
    def test_never_under_predicts(self, row):
        """The only safe direction for a gate that refuses configs. An estimator
        that is accurate on average but sometimes low still OOMs users.

        SCOPE, narrowed in v0.73.1 (#349): every row of this grid is at seq 256
        or 512. The grid varies BATCH (1..8), so this pins "never under-predicts
        as batch grows" and nothing about sequence length — a control only covers
        the variable it varies. Measured later on the same box, the property
        fails as seq grows: 0.934x the real peak at seq 5120 and 0.787x at 6144
        — against the probe the same formula reads 0.992x at seq 4096 and
        0.830x at 5120, because the probe runs 12.5-14.3% above the real step,
        and the measurement is deterministic (repeats at a fixed shape return
        bit-identical peaks, #395). Read this as a bound on the regime below,
        not as the global guarantee the phrase suggests;
        `training.stream_vram_probe` exists because no fitted formula can carry
        that guarantee everywhere.

        #395 KEPT this assertion rather than removing it, and that is forced
        rather than preferred. Criteria 2 and 3 of #395 are jointly
        unsatisfiable on this grid: criterion 3 asks to drop the seq<=512 scope,
        but `test_never_under_predicts` is parametrized over a grid fitted on
        the RTX 3050, and that stack demonstrably under-predicts at seq >= 5120
        (the real-peak series above). Separating the grids is the only
        construction that satisfies both. On the second stack — A10G / Linux /
        torch 2.13 / transformers 5.16.1 — the property HOLDS to seq 6144
        against real peaks, a flat 1.16x over-prediction
        (SECOND_STACK_VRAM_GRID), same denominator as the 0.934x/0.787x series.
        Two stacks disagreeing about one shape is the argument FOR pinning
        scope per-grid: dropping the assertion because one stack is safe would
        assert globally what neither stack can carry alone.
        """
        assert row["seq"] <= 512, (
            "this grid's evidence is seq<=512; a longer row added here would "
            "silently widen a claim the measurements do not support (#349)"
        )
        assert _predict(row) >= row["peak"], row["label"]

    def test_logits_term_dominates_at_large_batch(self):
        """The finding that makes batch budgeting the estimator rather than a
        refinement of it: at Qwen2.5-0.5B B8 S512 the logits tensor is 146x the
        entire buffer pool. A weights-and-buffers pre-flight green-lights this."""
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        row = MEASURED_VRAM_GRID[-1]
        logits = estimate_logits_bytes(
            vocab_size=row["vocab"], seq_len=row["seq"], batch_size=row["batch"]
        )
        assert logits > 100 * row["pool"]


class TestEstimateAdapterParams:
    def test_counts_two_matrices_per_target_per_layer(self):
        """LoRA adds an A and a B per targeted module. Deliberately biased HIGH
        (it assumes hidden x hidden), which is the safe direction for a budget."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        class _T:
            class lora:  # noqa: N801 — mirrors the config attribute name
                r = 16
                target_modules = ["q_proj", "v_proj"]

        class _C:
            hidden_size = 4096
            num_hidden_layers = 32

        got = SFTTrainerWrapper._estimate_adapter_params(None, _T(), _C())
        assert got == 32 * 2 * 2 * 16 * 4096

    def test_auto_target_modules_assume_four(self):
        """`target_modules: auto` is a string, not a list — it must not be
        len()'d into 4 by accident of the word's length."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        class _T:
            class lora:  # noqa: N801
                r = 8
                target_modules = "auto"

        class _C:
            hidden_size = 64
            num_hidden_layers = 2

        assert SFTTrainerWrapper._estimate_adapter_params(None, _T(), _C()) == (
            2 * 4 * 2 * 8 * 64
        )

    def test_moe_lora_auto_counts_expert_instances_not_pattern_names(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        class _T:
            moe_lora = True

            class lora:  # noqa: N801
                r = 8
                target_modules = "auto"

        class _C:
            hidden_size = 64
            num_hidden_layers = 2
            num_experts = 16

        assert SFTTrainerWrapper._estimate_adapter_params(None, _T(), _C()) == (
            2 * (4 + 3 * 16) * 2 * 8 * 64
        )

    def test_moe_lora_expands_explicit_expert_patterns_per_expert(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        class _T:
            moe_lora = True

            class lora:  # noqa: N801
                r = 4
                target_modules = ["q_proj", "gate_proj", "up_proj", "down_proj"]

        class _C:
            hidden_size = 32
            num_hidden_layers = 3
            num_experts = 8

        assert SFTTrainerWrapper._estimate_adapter_params(None, _T(), _C()) == (
            3 * (1 + 3 * 8) * 2 * 4 * 32
        )


class TestUntiedEmbeddingsAreBudgeted:
    """8B has untied embed + lm_head, so two large matrices go resident. The
    brief: budget them or the card OOMs on a model the planner said would fit."""

    def test_extras_are_charged_to_vram(self):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        kw = dict(
            layer_bytes=1000, buffers=2, adapter_params=0, vocab_size=100,
            hidden_size=8, intermediate_size=16, n_layers=2, seq_len=4, batch_size=1,
        )
        tied = estimate_stream_peak_vram(extras_bytes=1_000_000, **kw)
        untied = estimate_stream_peak_vram(extras_bytes=2_000_000, **kw)
        assert untied - tied == 1_000_000

    def test_streaming_large_layers_reclaims_exactly_one_untied_matrix(self):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        kw = dict(
            layer_bytes=1000,
            buffers=2,
            adapter_params=0,
            vocab_size=100,
            hidden_size=8,
            intermediate_size=16,
            n_layers=2,
            seq_len=4,
            batch_size=1,
        )
        old_untied = estimate_stream_peak_vram(extras_bytes=2_000_000, **kw)
        streamed_untied = estimate_stream_peak_vram(
            extras_bytes=0, large_layer_bytes=1_000_000, **kw
        )
        old_tied = estimate_stream_peak_vram(extras_bytes=1_000_000, **kw)
        streamed_tied = estimate_stream_peak_vram(
            extras_bytes=0, large_layer_bytes=1_000_000, **kw
        )

        assert old_untied - streamed_untied == 1_000_000
        assert old_tied == streamed_tied

    def test_published_8b_nf4_row_is_bracketed_on_the_safe_side(self):
        """Independent check — nothing in the formula was fitted to this row.
        Llama-3.1-8B NF4, untied embeddings, measured 3.32 GB in v0.72.2."""
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        predicted = estimate_stream_peak_vram(
            layer_bytes=int(3.60e9 / 32),
            buffers=2,
            extras_bytes=2 * 128256 * 4096 * 2,
            adapter_params=32 * 2 * (4096 * 16 + 16 * 4096),
            vocab_size=128256,
            hidden_size=4096,
            intermediate_size=14336,
            n_layers=32,
            seq_len=512,
            batch_size=1,
        )
        measured = 3.32e9
        assert measured <= predicted <= measured * 1.15, predicted


class TestStreamFitDecision:
    def test_refuses_when_demand_exceeds_available(self):
        from soup_cli.utils.layer_stream import decide_stream_fit

        fit = decide_stream_fit(predicted_bytes=5_000_000_000, available_bytes=3_445_000_000)
        assert not fit.fits
        assert "GB" in fit.reason

    def test_accepts_when_it_fits(self):
        from soup_cli.utils.layer_stream import decide_stream_fit

        fit = decide_stream_fit(predicted_bytes=1_000_000_000, available_bytes=3_445_000_000)
        assert fit.fits

    def test_the_flagship_8b_config_is_not_refused(self):
        """Regression guard with teeth: the measured 8B NF4 peak (3.32 GB) sat
        under this box's measured 3.445 GB of allocator-visible VRAM. An
        over-conservative reserve -- e.g. charging the plan's 1 GB workspace on
        top -- would refuse precisely the run this feature exists to enable."""
        from soup_cli.utils.layer_stream import decide_stream_fit

        assert decide_stream_fit(
            predicted_bytes=int(3.32e9), available_bytes=int(3.445e9)
        ).fits

    def test_the_measured_spill_config_is_refused(self):
        """Qwen2.5-0.5B B4 S512 demanded 4.82 GB on a 4.29 GB card. Windows did
        NOT raise -- WDDM spilled to host memory and the run merely became very
        slow -- so the estimator is the only thing standing between the user and
        a silent 10x slowdown."""
        from soup_cli.utils.layer_stream import decide_stream_fit

        assert not decide_stream_fit(
            predicted_bytes=int(4.82e9), available_bytes=int(3.445e9)
        ).fits

    def test_reason_names_the_knobs_the_user_can_actually_turn(self):
        from soup_cli.utils.layer_stream import decide_stream_fit

        reason = decide_stream_fit(
            predicted_bytes=5_000_000_000, available_bytes=1_000_000_000
        ).reason
        assert "batch_size" in reason and "max_length" in reason


class TestResolveAvailableVramBytes:
    """training.stream_vram_override (#347): the driver's mem_get_info() is a
    device-level query and cannot see a per-process cap, so the override must
    fully REPLACE the measured figure, not adjust it."""

    def test_no_override_uses_the_measured_figure(self):
        from soup_cli.utils.layer_stream import resolve_available_vram_bytes

        got = resolve_available_vram_bytes(measured_bytes=3_445_000_000, override_bytes=None)
        assert got == 3_445_000_000

    def test_override_replaces_a_larger_measured_figure(self):
        """Raising the budget: let a documented over-prediction through even
        though the driver reports plenty of headroom."""
        from soup_cli.utils.layer_stream import resolve_available_vram_bytes

        got = resolve_available_vram_bytes(
            measured_bytes=16_000_000_000, override_bytes=3_541_000_000
        )
        assert got == 3_541_000_000

    def test_override_replaces_a_smaller_measured_figure(self):
        """Lowering the budget below what the driver reports: the Colab/Kaggle
        case from #347's follow-up comment, where set_per_process_memory_fraction
        caps the process but mem_get_info() still reports the whole card."""
        from soup_cli.utils.layer_stream import resolve_available_vram_bytes

        got = resolve_available_vram_bytes(
            measured_bytes=16_000_000_000, override_bytes=4_000_000_000
        )
        assert got == 4_000_000_000

    def test_override_zero_is_honoured_not_treated_as_absent(self):
        """0 is a legitimate override ("assume nothing is free, refuse
        everything") and the value someone sets first to confirm the flag is
        wired at all, expecting a refusal. The `is None` check in
        resolve_available_vram_bytes is correct, but nothing previously
        exercised it at the resolver: a `override_bytes or measured_bytes`
        mutation (falsy-0 falls back to the driver figure) still passed the
        full suite (review on #386, blocking item). This pins the resolver
        itself, not just that 0 survives config load."""
        from soup_cli.utils.layer_stream import resolve_available_vram_bytes

        got = resolve_available_vram_bytes(measured_bytes=15_360_000_000, override_bytes=0)
        assert got == 0

    def test_override_below_real_free_vram_makes_a_fitting_config_refused(self):
        """The acceptance test #347's follow-up comment asks for verbatim: an
        override set below the real free VRAM must turn an otherwise-fitting
        config into a refusal, because that is the assertion nothing can fake."""
        from soup_cli.utils.layer_stream import decide_stream_fit, resolve_available_vram_bytes

        predicted = 5_000_000_000
        measured_free = 16_000_000_000  # plenty, would normally fit
        assert decide_stream_fit(predicted_bytes=predicted, available_bytes=measured_free).fits

        capped = resolve_available_vram_bytes(
            measured_bytes=measured_free, override_bytes=4_000_000_000
        )
        assert not decide_stream_fit(predicted_bytes=predicted, available_bytes=capped).fits


class TestThroughputForecast:
    def test_ceiling_uses_c_equals_six(self):
        from soup_cli.utils.layer_stream import forecast_stream_throughput

        got = forecast_stream_throughput(
            params=1_000_000_000, effective_tflops=6.0, tokens_per_epoch=0
        )
        assert got.tokens_per_sec_ceiling == pytest.approx(1000.0)

    def test_epoch_seconds_is_a_floor_from_the_ceiling(self):
        from soup_cli.utils.layer_stream import forecast_stream_throughput

        got = forecast_stream_throughput(
            params=1_000_000_000, effective_tflops=6.0, tokens_per_epoch=10_000
        )
        assert got.epoch_seconds_floor == pytest.approx(10.0)

    def test_is_labelled_a_ceiling_with_the_measured_observed_fraction(self):
        """Honesty: real streamed training landed at 68%-100% of the measured
        GEMM ceiling on the dev box, so a bare tok/s number would over-promise
        by up to 1.5x. The forecast must carry the bracket."""
        from soup_cli.utils.layer_stream import (
            MEASURED_CEILING_FRACTION,
            forecast_stream_throughput,
        )

        low, high = MEASURED_CEILING_FRACTION
        assert 0.0 < low < high <= 1.0
        got = forecast_stream_throughput(
            params=1_000_000_000, effective_tflops=6.0, tokens_per_epoch=10_000
        )
        assert got.tokens_per_sec_low == pytest.approx(1000.0 * low)

    def test_rejects_a_non_finite_tflops(self):
        from soup_cli.utils.layer_stream import forecast_stream_throughput

        with pytest.raises(ValueError, match="finite"):
            forecast_stream_throughput(
                params=1, effective_tflops=float("inf"), tokens_per_epoch=1
            )


#: Upper plausibility bound for the measured GEMM ceiling, in TFLOPS.
#:
#: The probe times a dense bf16 matmul, so it is bounded above by the card's
#: dense bf16 tensor-core peak. Vendor dense bf16 peaks, for scale: RTX 3050
#: Laptop ~18, A100 SXM 312, H100 SXM ~989, B200 ~2250. Measured *through this
#: probe*: 6.75 TFLOPS on the RTX 3050 dev box at 952 MHz, and 786.5 TFLOPS on
#: an H100 at 1980 MHz (`benchmarks/gate-h100-validation.md`, FINDING 3).
#:
#: The bound that belongs here is one that catches a BROKEN PROBE -- a unit slip
#: (ms read as seconds) or a wrong FLOP count, both wrong by three orders of
#: magnitude -- without encoding one generation of hardware. The 200.0 this
#: replaces was written when the only card this project had ever run on was that
#: laptop; it then failed an H100 at a correct 786.5, which made the shipped
#: suite un-greenable on exactly the machines this feature gets audited on.
#: 10_000 is >4x the fastest shipping accelerator and still ~100x below what any
#: unit error would report.
_MAX_PLAUSIBLE_GEMM_TFLOPS = 10_000.0


@requires_cuda
class TestMeasuredGemmCeiling:
    def test_returns_a_plausible_tflops_and_the_clock_it_was_taken_at(self):
        """A fraction-of-ceiling quoted without a stated clock is meaningless --
        this box's boost clock moved 442..952 MHz inside a single gate run."""
        import math

        from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops

        got = measure_gemm_tflops(device="cuda")
        assert got is not None
        assert math.isfinite(got.tflops)
        assert 0.5 < got.tflops < _MAX_PLAUSIBLE_GEMM_TFLOPS, (
            f"{got.tflops} TFLOPS is not a rate any shipping GPU produces -- "
            "suspect the probe's timing or FLOP count, not the card"
        )
        assert got.sm_clock_mhz is None or 100 <= got.sm_clock_mhz <= 4000
        # the shape is reported so a fraction-of-ceiling can be shape-matched
        assert got.size == 4096
        assert got.dtype in {"float16", "bfloat16"}

    def test_the_reported_rate_matches_an_independently_timed_matmul(self):
        """The plausibility band above cannot be tight without pinning a
        hardware generation, so the check that actually has teeth is a
        hardware-INDEPENDENT one: time the same matmul here and require the
        probe to agree to within an order of magnitude. A unit slip or a wrong
        FLOP count is off by ~1000x and fails this on any card; a cold clock, a
        busy GPU or best-of-N vs a single sample are worth at most a few x
        (measured spread on the dev box: 38% within a session, 1.9x across
        sessions at the same reported clock)."""
        import torch

        from soup_cli.utils.layer_stream import resolve_stream_dtype
        from soup_cli.utils.layer_stream_runtime import _GEMM_SIZE, measure_gemm_tflops

        got = measure_gemm_tflops(device="cuda")
        assert got is not None

        size, iters = _GEMM_SIZE, 8
        dtype_name = resolve_stream_dtype("cuda")
        dtype = getattr(torch, dtype_name)
        left = torch.randn(size, size, device="cuda", dtype=dtype)
        right = torch.randn(size, size, device="cuda", dtype=dtype)
        try:
            for _ in range(3):  # warm up: the first matmul pays kernel selection
                left @ right
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                left @ right
            end.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
        finally:
            del left, right
            torch.cuda.empty_cache()

        assert seconds > 0
        independent = 2.0 * (size**3) * iters / seconds / 1e12
        assert 0.1 * independent <= got.tflops <= 10.0 * independent, (
            f"probe reported {got.tflops} TFLOPS where an independent timing of "
            f"the same {size}^3 {dtype_name} matmul gives {independent}"
        )

    def test_returns_none_on_cpu_rather_than_inventing_a_number(self):
        from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops

        assert measure_gemm_tflops(device="cpu") is None

    def test_takes_the_best_repeat_not_the_first(self):
        """A ceiling's noise is ONE-SIDED — contention, a cold clock and thermal
        throttling only ever make an achievable rate look slower. Taking the best
        repeat must return max(samples) and include all repeat measurements."""
        from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops

        got = measure_gemm_tflops(device="cuda", reps=4)
        assert got is not None
        assert len(got.samples) == 4
        assert got.tflops == max(got.samples)
        assert got.tflops >= got.samples[0]
        assert got.tflops >= max(got.samples[1:])


class TestIssue444BestRepeatSelection:
    """Regression tests for #444: deterministic repeat selection in measure_gemm_tflops."""

    def test_gemm_ceiling_defaults_samples_to_empty_tuple(self) -> None:
        from soup_cli.utils.layer_stream_runtime import GemmCeiling

        ceiling = GemmCeiling(
            tflops=10.5,
            sm_clock_mhz=1200,
            size=4096,
            dtype="bfloat16",
        )
        assert ceiling.tflops == 10.5
        assert ceiling.sm_clock_mhz == 1200
        assert ceiling.size == 4096
        assert ceiling.dtype == "bfloat16"
        assert ceiling.samples == ()

    def test_selection_picks_maximum_repeat_not_first_or_last(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthetic benchmark proving best-of-N selects max(samples) deterministically."""
        import sys

        from soup_cli.utils import layer_stream_runtime

        elapsed_sequence = [100.0, 20.0, 50.0, 40.0]
        call_idx = 0

        class _MockEvent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def record(self) -> None:
                pass

            def elapsed_time(self, other: object) -> float:
                nonlocal call_idx
                val = elapsed_sequence[call_idx % len(elapsed_sequence)]
                call_idx += 1
                return val

        class _MockCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def synchronize() -> None:
                pass

            @staticmethod
            def empty_cache() -> None:
                pass

            Event = _MockEvent
            OutOfMemoryError = RuntimeError

        class _MockTensor:
            def __matmul__(self, other: object) -> "_MockTensor":
                return self

        class _MockTorch:
            bfloat16 = "bfloat16"
            float16 = "float16"
            cuda = _MockCuda

            @staticmethod
            def randn(*args: object, **kwargs: object) -> _MockTensor:
                return _MockTensor()

        monkeypatch.setitem(sys.modules, "torch", _MockTorch)
        monkeypatch.setattr(layer_stream_runtime, "sm_clock_mhz", lambda: 1500)
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream,
            "resolve_stream_dtype",
            lambda device: "bfloat16",
        )

        res = layer_stream_runtime.measure_gemm_tflops(
            device="cuda", iters=8, reps=4, size=4096
        )
        assert res is not None
        assert len(res.samples) == 4
        # Repeat index 1 (20ms) is 5x faster than repeat index 0 (100ms)
        assert res.samples[1] > res.samples[0]
        assert res.tflops == max(res.samples)
        assert res.tflops == res.samples[1]
        assert res.tflops != res.samples[0]
        assert res.tflops != res.samples[-1]

    @pytest.mark.parametrize("resolved_dtype", ["float16", "bfloat16"])
    def test_gemm_uses_resolved_stream_dtype(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resolved_dtype: str,
    ) -> None:
        """The GEMM probe must use the card-resolved stream dtype."""
        import sys

        from soup_cli.utils import layer_stream, layer_stream_runtime

        allocated_dtypes: list[object] = []

        class _MockEvent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def record(self) -> None:
                pass

            def elapsed_time(self, other: object) -> float:
                return 100.0

        class _MockCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def synchronize() -> None:
                pass

            @staticmethod
            def empty_cache() -> None:
                pass

            Event = _MockEvent
            OutOfMemoryError = RuntimeError

        class _MockTensor:
            def __matmul__(self, other: object) -> "_MockTensor":
                return self

        class _MockTorch:
            bfloat16 = "bfloat16"
            float16 = "float16"
            cuda = _MockCuda

            @staticmethod
            def randn(*args: object, **kwargs: object) -> _MockTensor:
                allocated_dtypes.append(kwargs["dtype"])
                return _MockTensor()

        monkeypatch.setitem(sys.modules, "torch", _MockTorch)
        monkeypatch.setattr(
            layer_stream,
            "resolve_stream_dtype",
            lambda device: resolved_dtype,
        )
        monkeypatch.setattr(layer_stream_runtime, "sm_clock_mhz", lambda: 1500)

        res = layer_stream_runtime.measure_gemm_tflops(
            device="cuda",
            iters=1,
            reps=1,
            size=4,
        )

        assert res is not None
        assert res.dtype == resolved_dtype
        assert allocated_dtypes == [resolved_dtype, resolved_dtype]

    def test_zero_elapsed_timing_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degenerate timing (seconds <= 0) must return None without raising."""
        import sys

        from soup_cli.utils import layer_stream_runtime

        class _MockEvent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def record(self) -> None:
                pass

            def elapsed_time(self, other: object) -> float:
                return 0.0

        class _MockCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def synchronize() -> None:
                pass

            @staticmethod
            def empty_cache() -> None:
                pass

            Event = _MockEvent
            OutOfMemoryError = RuntimeError

        class _MockTensor:
            def __matmul__(self, other: object) -> "_MockTensor":
                return self

        class _MockTorch:
            bfloat16 = "bfloat16"
            cuda = _MockCuda

            @staticmethod
            def randn(*args: object, **kwargs: object) -> _MockTensor:
                return _MockTensor()

        monkeypatch.setitem(sys.modules, "torch", _MockTorch)
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream,
            "resolve_stream_dtype",
            lambda device: "bfloat16",
        )
        got = layer_stream_runtime.measure_gemm_tflops(device="cuda", reps=4)
        assert got is None

    def test_zero_iters_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degenerate iters=0 must yield non-positive rates and return None."""
        import sys

        from soup_cli.utils import layer_stream_runtime

        class _MockEvent:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def record(self) -> None:
                pass

            def elapsed_time(self, other: object) -> float:
                return 10.0

        class _MockCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def synchronize() -> None:
                pass

            @staticmethod
            def empty_cache() -> None:
                pass

            Event = _MockEvent
            OutOfMemoryError = RuntimeError

        class _MockTensor:
            def __matmul__(self, other: object) -> "_MockTensor":
                return self

        class _MockTorch:
            bfloat16 = "bfloat16"
            cuda = _MockCuda

            @staticmethod
            def randn(*args: object, **kwargs: object) -> _MockTensor:
                return _MockTensor()

        monkeypatch.setitem(sys.modules, "torch", _MockTorch)
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream,
            "resolve_stream_dtype",
            lambda device: "bfloat16",
        )
        got = layer_stream_runtime.measure_gemm_tflops(device="cuda", iters=0, reps=4)
        assert got is None

class TestIssue617PanelDtype:
    """Regression coverage for the user-visible GEMM dtype in the stream panel."""

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_stream_budget_panel_reports_measured_gemm_dtype(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dtype: str,
    ) -> None:
        from types import SimpleNamespace

        from soup_cli.trainer import stream_setup

        class _Setup(stream_setup.StreamingSetupMixin):
            _STREAM_ROWS_PER_EXAMPLE = 1

            @staticmethod
            def _stream_shape_config(model_config):
                return model_config

            @staticmethod
            def _stream_intermediate_size(model_config):
                return 128

            @staticmethod
            def _estimate_adapter_params(tcfg, model_config):
                return 0

        setup = object.__new__(_Setup)
        setup.device = "cuda"

        cfg = SimpleNamespace(
            data=SimpleNamespace(max_length=16),
        )
        tcfg = SimpleNamespace(
            batch_size=1,
            stream_buffers=2,
            stream_vram_probe=False,
            stream_vram_override=None,
            gradient_accumulation_steps=1,
        )
        model_config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
        )
        index = SimpleNamespace(
            total_params=1_000_000,
            n_layers=2,
        )

        monkeypatch.setattr(
            stream_setup,
            "console",
            SimpleNamespace(print=lambda *args, **kwargs: None),
        )

        monkeypatch.setattr(
            layer_stream_runtime,
            "measure_gemm_tflops",
            lambda device: SimpleNamespace(
                tflops=6.75,
                sm_clock_mhz=862,
                dtype=dtype,
            ),
        )

        monkeypatch.setattr(
            layer_stream,
            "resolve_available_vram_bytes",
            lambda measured_bytes, override_bytes=None: 4_000_000_000,
        )

        monkeypatch.setattr(
            layer_stream,
            "decide_stream_fit",
            lambda predicted_bytes, available_bytes: SimpleNamespace(
                fits=True,
                reason="fits",
            ),
        )

        import torch

        monkeypatch.setattr(
            torch.cuda,
            "mem_get_info",
            lambda: (4_000_000_000, 8_000_000_000),
        )

        lines, plan = setup._stream_budget_lines(
            cfg,
            tcfg,
            model_config=model_config,
            layer_bytes=1_000_000,
            embed_bytes=1_000_000,
            index=index,
            on_cuda=True,
            large_layer_bytes=0,
        )

        assert plan is None

        panel_text = "\n".join(lines)
        assert (
            f"(from 6.75 TFLOPS measured on this card now "
            f"using {dtype} @ 862 MHz)"
        ) in panel_text


class TestBatchSizeIsSupported:
    """The shipped v0.72.2 refusal said "larger batches land in v0.72.3", so
    shipping v0.72.3 without lifting it makes that message a lie. Batch is also
    where streaming pays off: one weight read amortised over more tokens."""

    def test_batch_size_above_one_is_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_stream_yaml(batch_size=4))
        assert cfg.training.batch_size == 4
        assert cfg.training.stream_layers is True

    def test_auto_batch_size_is_still_refused(self):
        """"auto" resolves by OOM-probing a resident model, which streaming does
        not have -- it would probe a model that never loads."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="batch_size"):
            load_config_from_string(_stream_yaml(batch_size='"auto"'))


# ==========================================================================
# item 1 — the architecture allowlist
# ==========================================================================
class TestArchAllowlist:
    """GATE 1 proved each family bit-exact against a resident run; this pins the
    allowlist itself, which is what decides whether that path is reachable."""

    @pytest.mark.parametrize(
        "family",
        ["llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3_text",
         "phi", "phi3"],
    )
    def test_gated_families_are_accepted(self, family):
        from soup_cli.utils.layer_stream import stream_arch_of

        class _Cfg:
            model_type = family

        assert stream_arch_of(_Cfg()) == family

    @pytest.mark.parametrize("family", ["gpt2", "gemma3", "falcon", "mixtral"])
    def test_ungated_families_are_refused_naming_the_allowlist(self, family):
        """`gemma3` is the one that matters: a real google/gemma-3-* reports it
        for the VISION wrapper, and streaming a multimodal model as a causal LM
        is the silent mis-train the allowlist exists to prevent."""
        from soup_cli.utils.layer_stream import stream_arch_of

        class _Cfg:
            model_type = family

        with pytest.raises(ValueError, match="Supported"):
            stream_arch_of(_Cfg())


# ==========================================================================
# item 5 — disk-kind detection (the `soup doctor` rider + the tier gate)
# ==========================================================================
class TestDiskKindDetection:
    """v0.72.0 shipped a `choose_tier` that refuses non-NVMe, wired to a
    HARDCODED ``disk_kind="nvme"`` in the trainer — a guard connected to a
    constant, which can never fire."""

    def test_reports_one_of_the_known_kinds(self):
        from soup_cli.utils.layer_stream import DISK_KINDS, detect_disk_kind

        assert detect_disk_kind(".") in DISK_KINDS

    def test_result_is_cached_per_volume(self, monkeypatch):
        """The probe costs ~9 s on Windows (measured), so it must not repeat.

        Asserted by counting probe calls rather than by timing: a wall-clock
        threshold would itself pay the 9 s on a cold cache and would flake on a
        loaded CI runner."""
        import soup_cli.utils.layer_stream as ls

        calls = []
        monkeypatch.setattr(ls, "_DISK_KIND_CACHE", {})
        monkeypatch.setattr(
            ls, "_probe_disk_kind", lambda p: calls.append(p) or ls.DiskClassification("nvme")
        )

        assert ls.detect_disk_kind(".") == "nvme"
        assert ls.detect_disk_kind(".") == "nvme"
        assert len(calls) == 1, f"probed {len(calls)} times, expected 1"

    def test_unknown_is_refused_rather_than_assumed_fast(self):
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError, match="NVMe"):
            choose_tier(100, 10, "unknown")

    @pytest.mark.parametrize("kind", ["hdd", "ssd", "unknown"])
    def test_only_nvme_earns_the_disk_tier(self, kind):
        """A SATA SSD is refused too — that is v0.72.0's documented policy and
        this release does not quietly widen it."""
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError, match="NVMe"):
            choose_tier(100, 10, kind)

    def test_windows_bus_type_beats_media_type(self):
        """An NVMe drive reports MediaType 'SSD' and BusType 'NVMe' (measured on
        this box), so keying on MediaType alone would refuse every NVMe disk."""
        from soup_cli.utils.layer_stream import _windows_kind

        assert _windows_kind({"MediaType": "SSD", "BusType": "NVMe"}) == "nvme"
        assert _windows_kind({"MediaType": "SSD", "BusType": "SATA"}) == "ssd"
        assert _windows_kind({"MediaType": "HDD", "BusType": "SATA"}) == "hdd"
        assert _windows_kind({}) == "unknown"

    def test_numeric_media_type_codes_map_to_the_documented_enum(self):
        """`ConvertTo-Json` emits the integer rather than the friendly string on
        some systems. MSFT_PhysicalDisk.MediaType ValueMap is {0 Unspecified,
        3 HDD, 4 SSD, 5 SCM} — verified against this box, whose NVMe reports raw
        value 4. The first cut had 3 and 4 SWAPPED, which would have reported a
        spinning disk as an SSD."""
        from soup_cli.utils.layer_stream import _windows_kind

        assert _windows_kind({"MediaType": "3", "BusType": "SATA"}) == "hdd"
        assert _windows_kind({"MediaType": "4", "BusType": "SATA"}) == "ssd"
        assert _windows_kind({"MediaType": "0", "BusType": "SATA"}) == "unknown"
        assert _windows_kind({"MediaType": "5", "BusType": "SATA"}) == "unknown"


class TestDiskKindMeasuredFallback:
    """#365 — a virtio disk reports ``rotational=1`` with no media hint, so the
    flag alone refused a genuinely NVMe-backed cloud disk the overflow tier. When
    the flag is unreliable, classify on a measured sequential read instead; the
    HDD refusal (160 seeks/step, plan P11) must survive as a control."""

    def test_throughput_at_or_above_the_floor_is_nvme_class(self):
        from soup_cli.utils.layer_stream import (
            NVME_TIER_MIN_BYTES_PER_S,
            _classify_measured_read,
        )

        assert _classify_measured_read(NVME_TIER_MIN_BYTES_PER_S) == "nvme"
        assert _classify_measured_read(1.5e9) == "nvme"  # the reported virtio disk

    def test_slow_or_unmeasurable_stays_hdd(self):
        from soup_cli.utils.layer_stream import (
            NVME_TIER_MIN_BYTES_PER_S,
            _classify_measured_read,
        )

        assert _classify_measured_read(NVME_TIER_MIN_BYTES_PER_S - 1) == "hdd"
        assert _classify_measured_read(150e6) == "hdd"  # a real spinning disk
        assert _classify_measured_read(None) == "hdd"  # can't tell -> refuse

    @staticmethod
    def _fake_linux(monkeypatch, *, devices, rotational, measured_bps):
        """Simulate a Linux ``/sys/block`` layout so the real probe branch runs."""
        import builtins
        import io
        import os
        import platform
        import sys

        import pytest

        # detect_disk_kind is a Linux /sys/block feature (it returns "unknown"
        # elsewhere by design). This helper hardcodes forward-slash /sys paths,
        # which os.path.join mangles on Windows, so the simulation is meaningful
        # only on Linux — where the real CI cells and local runs exercise it.
        if sys.platform != "linux":
            pytest.skip("simulates a Linux /sys/block layout; disk-tier detection is Linux-only")

        import soup_cli.utils.layer_stream as ls

        real_listdir = os.listdir
        real_open = builtins.open
        real_exists = os.path.exists

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            os,
            "listdir",
            lambda p: list(devices) if str(p) == "/sys/block" else real_listdir(p),
        )
        monkeypatch.setattr(
            os.path,
            "exists",
            lambda p: True if "/queue/rotational" in str(p) else real_exists(p),
        )

        def fake_open(p, *a, **k):
            if "/queue/rotational" in str(p):
                return io.StringIO(f"{rotational}\n")
            return real_open(p, *a, **k)

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setattr(ls, "_measure_seq_read_bytes_per_s", lambda _p: measured_bps)
        monkeypatch.setattr(ls, "_DISK_KIND_CACHE", {})

    def test_virtio_fast_disk_now_earns_the_disk_tier(self, monkeypatch):
        """Criterion 1: no NVMe device, ``rotational=1``, fast measured read."""
        import soup_cli.utils.layer_stream as ls

        self._fake_linux(monkeypatch, devices=["vda"], rotational=1, measured_bps=1.5e9)
        assert ls.detect_disk_kind("/data") == "nvme"
        # ...and choose_tier accepts it where the pre-fix "hdd" would have raised.
        assert ls.choose_tier(1000, 10, ls.detect_disk_kind("/data")) == ls.TIER_DISK

    def test_a_genuinely_slow_device_is_still_refused(self, monkeypatch):
        """Criterion 2 (the control): the fix cannot be satisfied by 'always allow'."""
        import soup_cli.utils.layer_stream as ls

        self._fake_linux(monkeypatch, devices=["vda"], rotational=1, measured_bps=150e6)
        assert ls.detect_disk_kind("/data") == "hdd"
        with pytest.raises(ValueError, match="NVMe"):
            ls.choose_tier(1000, 10, ls.detect_disk_kind("/data"))

    def test_rotational_zero_is_authoritative_and_never_measures(self, monkeypatch):
        """A device that declares itself solid state is trusted — no probe cost."""
        import soup_cli.utils.layer_stream as ls

        self._fake_linux(monkeypatch, devices=["sda"], rotational=0, measured_bps=None)
        calls = []
        monkeypatch.setattr(
            ls, "_measure_seq_read_bytes_per_s", lambda _p: calls.append(1) or 9e9
        )
        assert ls.detect_disk_kind("/data") == "ssd"
        assert calls == [], "measured probe ran on an authoritative rotational=0"

    def test_measurement_is_bounded_and_best_effort(self, monkeypatch):
        """Criterion 4: no O_DIRECT (or any failure) degrades to None -> refused,
        never an unbounded or crashing probe."""
        import os

        import soup_cli.utils.layer_stream as ls

        monkeypatch.delattr(os, "O_DIRECT", raising=False)
        assert ls._measure_seq_read_bytes_per_s(".") is None

    def test_refusal_cites_the_measured_rate(self):
        """#411 review: a measurement-driven refusal names the rate that earned
        it, not just the opaque ``'hdd'`` verdict. CPU-only — the rate travels
        with the verdict in a ``DiskClassification``, so no /sys simulation is
        needed to exercise the note (it runs on all CI cells, not just Linux)."""
        import soup_cli.utils.layer_stream as ls

        # A verdict DERIVED from a measurement carries the rate that produced it.
        slow = ls.DiskClassification("hdd", measured_bps=150e6)
        with pytest.raises(ValueError) as excinfo:
            ls.choose_tier(1000, 10, slow)
        message = str(excinfo.value)
        assert "measured 0.15 GB/s" in message
        assert "1.0 GB/s NVMe floor" in message

    def test_override_verdict_does_not_cite_a_probe_rate(self):
        """#411 re-review (blocker 2): the bug was a module global that let a
        refusal cite a rate ABOVE the floor as the reason a disk fell UNDER it —
        e.g. stream_disk_kind=hdd on a fast virtio disk. The rate now travels
        with the verdict, and an override verdict carries none, so the refusal
        structurally cannot cite a reading it did not produce. CPU-only."""
        import soup_cli.utils.layer_stream as ls

        # kind='hdd' from an override; measured_bps=None because the verdict is
        # the user's, not the probe's (see resolve_disk_kind).
        overridden = ls.DiskClassification("hdd", measured_bps=None)
        with pytest.raises(ValueError) as excinfo:
            ls.choose_tier(1000, 10, overridden)
        assert "measured" not in str(excinfo.value)

    def test_override_returns_a_measureless_classification(self, monkeypatch):
        """resolve_disk_kind with an override must strip any probe rate, even
        when detection measured a fast disk underneath (the #411 re-review repro:
        detect 2.0 GB/s, override to hdd — the 2.0 must not survive)."""
        import soup_cli.utils.layer_stream as ls

        monkeypatch.setattr(
            ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("nvme", 2.0e9)
        )
        result = ls.resolve_disk_kind("/data", "hdd", notify=lambda _m: None)
        assert result.kind == "hdd"
        assert result.measured_bps is None

    def test_the_read_is_repeated_and_the_best_sample_wins(self, monkeypatch, tmp_path):
        """#411 review: the single-sample threshold is the weakness — repeat the
        read and keep the fastest so a lone cold sample cannot refuse a fast disk.

        The filesystem I/O is mocked so the timing is deterministic regardless of
        whether the test host's temp dir actually supports O_DIRECT."""
        import os
        import tempfile
        import time

        import soup_cli.utils.layer_stream as ls

        if getattr(os, "O_DIRECT", None) is None:
            pytest.skip("O_DIRECT is Linux-only; the probe returns None elsewhere")

        scratch = str(tmp_path / "scratch")
        monkeypatch.setattr(tempfile, "mkstemp", lambda **_k: (123, scratch))
        monkeypatch.setattr(os, "write", lambda _fd, b: len(b))
        monkeypatch.setattr(os, "fsync", lambda _fd: None)
        monkeypatch.setattr(os, "close", lambda _fd: None)
        monkeypatch.setattr(os, "open", lambda _p, _flags: 456)
        monkeypatch.setattr(os, "lseek", lambda _fd, _off, _whence: 0)
        monkeypatch.setattr(os, "unlink", lambda _p: None)
        readv_calls = []

        def fake_readv(_fd, _bufs):
            readv_calls.append(1)
            return ls._MEASURE_READ_BYTES

        monkeypatch.setattr(os, "readv", fake_readv)
        # Three reads with elapsed 2s, 1s, 4s -> the 1s sample is the fastest.
        clock = iter([0.0, 2.0, 0.0, 1.0, 0.0, 4.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(clock))

        best = ls._measure_seq_read_bytes_per_s(str(tmp_path))
        assert len(readv_calls) == ls._MEASURE_READ_SAMPLES
        assert best == ls._MEASURE_READ_BYTES / 1.0  # bytes / fastest elapsed (1s)

    def test_probe_writes_into_the_target_volume_not_its_parent(
        self, monkeypatch, tmp_path
    ):
        """The caller passes shard_dir (a directory); the scratch probe must land
        INSIDE it, not its parent — else a mount point's throughput is measured
        on the wrong filesystem."""
        import os
        import tempfile

        import soup_cli.utils.layer_stream as ls

        if not hasattr(os, "O_DIRECT"):
            pytest.skip("O_DIRECT is Linux-only")

        target = tmp_path / "shards"
        target.mkdir()
        seen = {}
        real_mkstemp = tempfile.mkstemp

        def spy_mkstemp(*a, **k):
            seen["dir"] = k.get("dir")
            return real_mkstemp(*a, **k)

        monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)
        ls._measure_seq_read_bytes_per_s(str(target))  # O_DIRECT may fail on tmpfs; fine
        assert seen.get("dir") == str(target)

    def test_override_wins_and_reports_what_it_overrode(self, monkeypatch):
        """Criterion 3: the override is used AND the notice names both values."""
        import soup_cli.utils.layer_stream as ls

        monkeypatch.setattr(ls, "detect_disk_kind", lambda *_a, **_k: "hdd")
        notes = []
        assert ls.resolve_disk_kind("/data", "nvme", notify=notes.append).kind == "nvme"
        assert notes and "nvme" in notes[0] and "hdd" in notes[0]

    def test_no_override_returns_the_detected_kind_silently(self, monkeypatch):
        import soup_cli.utils.layer_stream as ls

        monkeypatch.setattr(
            ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("ssd")
        )
        notes = []
        assert ls.resolve_disk_kind("/data", None, notify=notes.append).kind == "ssd"
        assert notes == []

    def test_stream_disk_kind_is_a_footgun_without_stream_layers(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="stream_disk_kind"):
            load_config_from_string(
                yaml.safe_dump(_stream_disk_kind_config(stream_layers=False))
            )

    def test_stream_disk_kind_is_accepted_while_streaming(self):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            yaml.safe_dump(_stream_disk_kind_config(stream_layers=True))
        )
        assert cfg.training.stream_disk_kind == "nvme"


def _stream_disk_kind_config(*, stream_layers):
    return {
        "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
        "training": {
            "stream_layers": stream_layers,
            "stream_disk_kind": "nvme",
            "quantization": "none",
            "batch_size": 1,
            "epochs": 1,
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        },
    }


class TestDoctorReportsTheDiskKind:
    """The rider's CLI half. Each branch of the verdict table is exercised —
    including the exception path, because a diagnostic that crashes the report
    is worse than one that says "unknown"."""

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("nvme", "NVMe"),
            ("ssd", "SATA SSD"),
            ("hdd", "HDD"),
            ("unknown", "Unknown"),
        ],
    )
    def test_each_media_type_gets_its_own_verdict(self, monkeypatch, kind, expected):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.setattr(
            "soup_cli.utils.layer_stream.detect_disk_kind", lambda *_a, **_k: kind
        )
        result = CliRunner().invoke(app, ["doctor", "--disk"])
        plain = _strip_ansi(result.output)
        assert "Disk type" in plain
        assert expected in plain, plain

    def test_the_probe_is_opt_in_so_doctor_stays_fast(self, monkeypatch):
        """Matching this command's own --nccl convention for expensive probes.
        Measured: the Windows query costs ~9 s cold and ~2.4 s warm (17.6 s ->
        20.0 s on `soup doctor`), paid by every user including the majority who
        never touch layer streaming. A streaming run probes lazily on its own,
        so defaulting this off loses nothing."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        calls = []
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream.detect_disk_kind",
            lambda *_a, **_k: calls.append(1) or "nvme",
        )
        plain = _strip_ansi(CliRunner().invoke(app, ["doctor"]).output)
        assert calls == [], "the expensive probe ran without --disk"
        assert "Disk type" not in plain

    def test_a_failing_probe_degrades_to_unknown_instead_of_crashing(
        self, monkeypatch
    ):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        def _boom(*_a, **_k):
            raise OSError("no storage subsystem")

        monkeypatch.setattr("soup_cli.utils.layer_stream.detect_disk_kind", _boom)
        result = CliRunner().invoke(app, ["doctor", "--disk"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Unknown" in _strip_ansi(result.output)


def _strip_ansi(text):
    """Rich wraps and colourises; assertions must survive both."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text).replace("\n", " ")


#: #622 repro values: the reported 32B NF4 store fits the dynamic free-RAM
#: headroom (16.10 GB < 27 GB * 0.70) but exceeds the physical-RAM ceiling
#: once resident extras are included (16.10 GB + 3.114 GB >= 30 GB * 0.55).
_ISSUE622_STORE_BYTES = 16_100_000_000
_ISSUE622_RESIDENT_BYTES = 3_114_000_000
_ISSUE622_SMALL_STORE_BYTES = 8_000_000_000
_ISSUE622_FREE_RAM_BYTES = 27_000_000_000
_ISSUE622_TOTAL_RAM_BYTES = 30_000_000_000
_ISSUE622_REVIEW_TIGHT_FREE_RAM_BYTES = 10_000_000_000
_ISSUE622_REVIEW_TOTAL_RAM_BYTES = 100_000_000_000
_ISSUE622_REVIEW_STORE_BYTES = 5_000_000_000
_ISSUE622_REVIEW_RESIDENT_BYTES = 3_000_000_000
_ISSUE622_LAYERS = 64
_ISSUE622_PSUTIL_TOTAL_BYTES = 123_456_789
_ISSUE622_SCHEMA_PHYSICAL_TEXT = "physical RAM ceiling"
_ISSUE622_SCHEMA_RESIDENT_TEXT = "resident extras"


class TestTierProbeIsLazy:
    def test_ram_tier_never_pays_for_the_probe(self):
        """The measured reason this matters: the probe is ~9 s on Windows and
        the answer is irrelevant whenever the base fits in RAM."""
        from soup_cli.utils.layer_stream import TIER_RAM, choose_tier

        calls = []

        def probe():
            calls.append(1)
            return "nvme"

        assert choose_tier(10, 1000, probe) == TIER_RAM
        assert calls == [], "the disk probe ran on a RAM-tier decision"

    def test_disk_tier_does_pay_for_it(self):
        from soup_cli.utils.layer_stream import TIER_DISK, choose_tier

        calls = []

        def probe():
            calls.append(1)
            return "nvme"

        assert choose_tier(1000, 10, probe) == TIER_DISK
        assert calls == [1]


class TestPhysicalRamBudget:
    """#622: the RAM tier needs an absolute physical-host budget too.

    The reported 32B NF4 run had plenty of ``MemAvailable`` for the old dynamic
    check, but the pinned resident store plus extras consumed too much of the
    physical host while 18 GB of safetensors shards were being read.
    """

    def test_auto_falls_to_disk_when_physical_ram_budget_is_exceeded(self):
        from soup_cli.utils.layer_stream import TIER_DISK, choose_tier

        assert (
            choose_tier(
                _ISSUE622_STORE_BYTES,
                _ISSUE622_FREE_RAM_BYTES,
                "nvme",
                resident_bytes=_ISSUE622_RESIDENT_BYTES,
                total_ram_bytes=_ISSUE622_TOTAL_RAM_BYTES,
            )
            == TIER_DISK
        )

    def test_physical_ceiling_refusal_names_why_disk_was_needed(self):
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError, match="store plus resident extras"):
            choose_tier(
                _ISSUE622_STORE_BYTES,
                _ISSUE622_FREE_RAM_BYTES,
                "hdd",
                resident_bytes=_ISSUE622_RESIDENT_BYTES,
                total_ram_bytes=_ISSUE622_TOTAL_RAM_BYTES,
            )

    def test_physical_ram_budget_keeps_ram_when_store_is_small(self):
        from soup_cli.utils.layer_stream import TIER_RAM, choose_tier

        probes = []

        def probe():
            probes.append("probed")
            return "nvme"

        assert (
            choose_tier(
                _ISSUE622_SMALL_STORE_BYTES,
                _ISSUE622_FREE_RAM_BYTES,
                probe,
                total_ram_bytes=_ISSUE622_TOTAL_RAM_BYTES,
            )
            == TIER_RAM
        )
        assert probes == []

    def test_free_ram_budget_counts_resident_extras(self):
        from soup_cli.utils.layer_stream import TIER_DISK, choose_tier

        assert (
            choose_tier(
                _ISSUE622_REVIEW_STORE_BYTES,
                _ISSUE622_REVIEW_TIGHT_FREE_RAM_BYTES,
                "nvme",
                resident_bytes=_ISSUE622_REVIEW_RESIDENT_BYTES,
                total_ram_bytes=_ISSUE622_REVIEW_TOTAL_RAM_BYTES,
            )
            == TIER_DISK
        )

    def test_free_ram_fallback_note_names_resident_extras(self):
        from soup_cli.utils.layer_stream import TIER_DISK, build_stream_plan

        plan = build_stream_plan(
            arch="qwen2",
            n_layers=_ISSUE622_LAYERS,
            layer_bytes=_ISSUE622_REVIEW_STORE_BYTES // _ISSUE622_LAYERS,
            embed_bytes=_ISSUE622_REVIEW_RESIDENT_BYTES,
            store_bytes=_ISSUE622_REVIEW_STORE_BYTES,
            available_ram_bytes=_ISSUE622_REVIEW_TIGHT_FREE_RAM_BYTES,
            total_ram_bytes=_ISSUE622_REVIEW_TOTAL_RAM_BYTES,
            pinned_limit_bytes=None,
            disk_kind="nvme",
        )

        assert plan.tier == TIER_DISK
        joined = " ".join(plan.notes)
        assert "resident extras" in joined
        assert "free-RAM" in joined

    def test_auto_disk_fallback_names_the_physical_ram_ceiling(self):
        from soup_cli.utils.layer_stream import TIER_DISK, build_stream_plan

        plan = build_stream_plan(
            arch="qwen2",
            n_layers=_ISSUE622_LAYERS,
            layer_bytes=_ISSUE622_STORE_BYTES // _ISSUE622_LAYERS,
            embed_bytes=_ISSUE622_RESIDENT_BYTES,
            store_bytes=_ISSUE622_STORE_BYTES,
            available_ram_bytes=_ISSUE622_FREE_RAM_BYTES,
            total_ram_bytes=_ISSUE622_TOTAL_RAM_BYTES,
            pinned_limit_bytes=None,
            disk_kind="nvme",
        )

        assert plan.tier == TIER_DISK
        joined = " ".join(plan.notes)
        assert "physical RAM" in joined
        assert "55%" in joined
        assert "stream_source='ram'" in joined

    def test_forced_ram_refuses_when_physical_ram_budget_is_exceeded(self):
        from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

        with pytest.raises(ValueError, match="physical RAM"):
            _validate_qwen4_ngram_ram_fit(
                stream_source="ram",
                ngram_source="disk",
                required_ram=_ISSUE622_STORE_BYTES,
                free_ram=_ISSUE622_FREE_RAM_BYTES,
                resident_ram=_ISSUE622_RESIDENT_BYTES,
                total_ram=_ISSUE622_TOTAL_RAM_BYTES,
            )

    def test_forced_ram_free_budget_counts_resident_extras(self):
        from soup_cli.trainer.stream_setup import _validate_qwen4_ngram_ram_fit

        with pytest.raises(ValueError, match="resident extras"):
            _validate_qwen4_ngram_ram_fit(
                stream_source="ram",
                ngram_source="disk",
                required_ram=_ISSUE622_REVIEW_STORE_BYTES,
                free_ram=_ISSUE622_REVIEW_TIGHT_FREE_RAM_BYTES,
                resident_ram=_ISSUE622_REVIEW_RESIDENT_BYTES,
                total_ram=_ISSUE622_REVIEW_TOTAL_RAM_BYTES,
            )

    def test_total_ram_bytes_reports_psutil_total(self, monkeypatch):
        import sys
        import types

        from soup_cli.utils.layer_stream import total_ram_bytes

        fake_psutil = types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(total=_ISSUE622_PSUTIL_TOTAL_BYTES)
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        assert total_ram_bytes() == _ISSUE622_PSUTIL_TOTAL_BYTES

    def test_total_ram_bytes_returns_none_when_psutil_is_missing(self, monkeypatch):
        import builtins
        import sys

        from soup_cli.utils.layer_stream import total_ram_bytes

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "psutil", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert total_ram_bytes() is None

    def test_total_ram_bytes_returns_none_when_psutil_cannot_report(self, monkeypatch):
        import sys
        import types

        from soup_cli.utils.layer_stream import total_ram_bytes

        def no_memory():
            raise OSError("host probe failed")

        monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(virtual_memory=no_memory))

        assert total_ram_bytes() is None

    def test_schema_descriptions_name_the_physical_ram_ceiling(self):
        from soup_cli.config.schema import TrainingConfig

        stream_source = TrainingConfig.model_fields["stream_source"].description
        stream_ngram_source = TrainingConfig.model_fields["stream_ngram_source"].description

        assert _ISSUE622_SCHEMA_PHYSICAL_TEXT in stream_source
        assert _ISSUE622_SCHEMA_RESIDENT_TEXT in stream_source
        assert _ISSUE622_SCHEMA_PHYSICAL_TEXT in stream_ngram_source
        assert _ISSUE622_SCHEMA_RESIDENT_TEXT in stream_ngram_source


# ==========================================================================
# item 6 — the disk overflow tier
# ==========================================================================
class TestDiskTier:
    """GATE 6. The strongest reference is the RAM tier, not the resident model:
    both stream through the same pool, prefetcher and layer wrapper and differ
    ONLY in where ``get(idx, name)`` reads from, so any difference is
    attributable to the source. The resident comparison is kept as well, so the
    disk tier is anchored to ground truth and not merely to its sibling."""

    def test_disk_tier_is_bit_exact_against_the_ram_tier(self, tmp_path):
        import torch

        disk, _, _ = _tiny_stream(tmp_path, name="d1", tier="disk")
        ram, _, _ = _tiny_stream(tmp_path, name="r1", tier="ram")
        _randomise_lora_b(disk)
        _sync_lora(disk, ram)

        ids = torch.randint(0, 64, (1, 12))
        disk.eval()
        ram.eval()
        with torch.no_grad():
            got, want = disk(input_ids=ids).logits, ram(input_ids=ids).logits
        assert torch.equal(got, want), float((got - want).abs().max())

    def test_disk_tier_is_bit_exact_against_resident(self, tmp_path):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM

        disk, _, weights = _tiny_stream(tmp_path, name="d1", tier="disk")
        _randomise_lora_b(disk)
        resident = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32)
        ref = get_peft_model(
            resident,
            LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
            ),
        )
        _sync_lora(disk, ref)

        ids = torch.randint(0, 64, (1, 12))
        disk.eval()
        ref.eval()
        with torch.no_grad():
            got, want = disk(input_ids=ids).logits, ref(input_ids=ids).logits
        assert torch.equal(got, want), float((got - want).abs().max())

    def test_layer_zero_adapter_gradient_is_non_zero(self, tmp_path):
        """plan P2 — a severed graph still lowers the loss, so nothing else
        catches it. Re-checked per tier, not inherited."""
        import torch

        disk, _, _ = _tiny_stream(tmp_path, name="d1", tier="disk")
        ids = torch.randint(0, 64, (1, 12))
        disk.train()
        disk(input_ids=ids, labels=ids).loss.backward()
        grads = [
            float(p.grad.abs().sum())
            for n, p in disk.named_parameters()
            if "lora_" in n and ".layers.0." in n and p.grad is not None
        ]
        assert grads and sum(grads) > 0.0

    def test_nothing_is_held_resident(self, tmp_path):
        """The point of the tier. ``store_bytes`` is what the RAM tier would
        have pinned; on disk it must be zero, with the size reported separately
        so the operator still sees what the model costs."""
        disk, runtime, _ = _tiny_stream(tmp_path, name="d1", tier="disk")
        stats = runtime.stats()
        assert stats["tier"] == "disk"
        assert stats["store_bytes"] == 0
        assert stats["disk_bytes"] > 0

    def test_disk_and_ram_accounting_agree(self, tmp_path):
        """A disk tier that reported a different model size than the RAM tier
        would mean one of the two is mis-measuring the shards."""
        _, disk_rt, _ = _tiny_stream(tmp_path, name="d1", tier="disk")
        _, ram_rt, _ = _tiny_stream(tmp_path, name="r1", tier="ram")
        assert disk_rt.stats()["disk_bytes"] == ram_rt.stats()["store_bytes"]

    def test_disk_tier_is_bit_exact_against_the_ram_tier_under_nf4(self, tmp_path):
        """The schema permits `quantization: 4bit` together with
        `stream_source: disk`, and the two paths differ: RamSource pre-allocates
        at the per-tensor spec dtype and copies in, while DiskSource hands back
        whatever the shard holds. Under NF4 a layer is MIXED uint8/float32, so
        this combination is exactly where a dtype assumption would bite."""
        import torch

        disk, _, _ = _tiny_stream(tmp_path, name="dq", tier="disk", quant="nf4")
        ram, _, _ = _tiny_stream(tmp_path, name="rq", tier="ram", quant="nf4")
        _randomise_lora_b(disk)
        _sync_lora(disk, ram)

        ids = torch.randint(0, 64, (1, 12))
        disk.eval()
        ram.eval()
        with torch.no_grad():
            got, want = disk(input_ids=ids).logits, ram(input_ids=ids).logits
        assert torch.equal(got, want), float((got - want).abs().max())

    def test_a_missing_shard_closes_the_handles_already_opened(self, tmp_path):
        """DiskSource opens one handle per layer through an ExitStack precisely
        so a failure partway through does not strand the earlier ones."""
        import os

        from soup_cli.utils.layer_shard import layer_shard_path
        from soup_cli.utils.layer_stream_runtime import DiskSource, RamSource

        _tiny_stream(tmp_path, name="d1", tier="ram")
        shards = str(tmp_path / "d1")
        spec = RamSource.spec_from_shard(shards)
        os.remove(layer_shard_path(shards, 2))
        with pytest.raises((OSError, FileNotFoundError, Exception)) as excinfo:
            DiskSource(shards, 3, spec)
        assert "layer_002" in str(excinfo.value) or "No such" in str(excinfo.value)

    def test_runtime_close_releases_the_shard_handles(self, tmp_path):
        """The disk tier holds ONE open handle per decoder layer — 80+ on a large
        model. Without an explicit release they live until the process exits and
        leak across back-to-back runs in one process (`soup sweep`, the web UI)."""
        from soup_cli.utils.layer_stream_runtime import DiskSource

        _, runtime, _ = _tiny_stream(tmp_path, name="d1", tier="disk")
        assert isinstance(runtime.source, DiskSource)
        assert runtime.source._handles, "no shard handles were opened at all"
        runtime.close()
        assert runtime.source._handles == []
        assert runtime.hook is None, "the prefetch hook was left attached"
        runtime.close()  # idempotent

    def test_ram_tier_close_is_a_no_op(self, tmp_path):
        """`close()` is called unconditionally by the trainer, so the RAM tier —
        which owns no handles — must tolerate it."""
        _, runtime, _ = _tiny_stream(tmp_path, name="r1", tier="ram")
        runtime.close()
        runtime.close()


#: Free RAM to report so the tiny base comfortably takes the RAM tier.
_RAM_TIER_FREE_BYTES = 10_000_000_000
#: Render width for the captured pre-flight. The plan notes arrive inside a
#: `rich.panel.Panel`; too narrow a console wraps the measured figures across a
#: line break and the assertions below miss text that IS on screen.
_PANEL_CAPTURE_WIDTH = 200
#: The measured page-locking gain (`PIN_THROUGHPUT_GAIN_REAL`, Qwen2.5-32B NF4)
#: as it appears on screen. Written as the literal the user reads rather than
#: imported, so that changing the production constant fails these tests too.
_PIN_GAIN_REAL_TEXT = "6.56"
#: The forced-pageable note must be identifiable as such, not merely contain the
#: number: some unrelated line carrying "6.56" would otherwise satisfy the check.
_FORCED_PAGEABLE_TEXT = "training.stream_pin=false"
#: Free/total VRAM to report from a stubbed `torch.cuda.mem_get_info()` so the
#: pre-flight fit check passes on a box with no GPU. Generous on purpose: the
#: point of these cases is the require_pin wiring, not the VRAM budget.
_SIMULATED_FREE_VRAM_BYTES = 16_000_000_000


class TestAutoTierFallback:
    """`stream_source: auto` (the default) takes RAM when the base fits and
    falls back to the NVMe disk tier when it does not. Driven through the real
    trainer, because the tier is decided from actual shard sizes after sharding
    — a unit test of `choose_tier` alone would not catch a regression that stops
    threading the decision into `build_streamed_model`."""

    def _run(
        self, tmp_path, monkeypatch, *, free_ram, stream_source,
        disk_kind="nvme", device="cpu", extra_training_yaml="",
    ):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper

        _tiny_stream(tmp_path)  # writes a real tiny checkpoint at tmp_path/model
        weights = str(tmp_path / "model")
        _write_tiny_tokenizer(weights)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            "soup_cli.utils.spectrum_scan.resolve_model_weights", lambda *_a, **_k: weights
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream.free_ram_bytes", lambda: free_ram
        )
        # Pinned rather than probed: the real media type differs between this
        # box (NVMe) and a CI runner (often "unknown"), and an
        # environment-dependent tier would make these flaky rather than wrong.
        # Patch classify_disk_kind (what resolve_disk_kind and detect_disk_kind
        # both route through) so the pin reaches the streaming setup's lambda.
        import soup_cli.utils.layer_stream as _ls

        monkeypatch.setattr(
            _ls, "classify_disk_kind", lambda *_a, **_k: _ls.DiskClassification(disk_kind)
        )
        cfg = load_config_from_string(
            f"base: {weights}\ntask: sft\nbackend: transformers\nmodality: text\n"
            "data:\n  train: data.jsonl\n  format: alpaca\n"
            "training:\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
            f"  quantization: none\n  stream_layers: true\n"
            f"  stream_source: {stream_source}\n{extra_training_yaml}"
            "  lora:\n    r: 4\n    target_modules: [q_proj, v_proj]\n"
        )
        wrapper = SFTTrainerWrapper(cfg)
        wrapper.device = device
        wrapper._setup_streaming_transformers(cfg, cfg.training)
        return wrapper

    @staticmethod
    def _stub_build_streamed_model(monkeypatch, captured, *, tier="ram"):
        """Intercept build_streamed_model so a CPU-only test can assert what the
        pre-flight threaded into the runtime, without building a real model.

        Patched on the SOURCE module because stream_setup imports it locally."""
        from unittest.mock import MagicMock

        import soup_cli.utils.layer_stream_runtime as rt

        def fake_build(**kwargs):
            captured.update(kwargs)
            runtime = MagicMock()
            runtime.tier = tier
            runtime.stats.return_value = {
                "tier": tier,
                "store_bytes": 1_000_000_000,
                "disk_bytes": 1_000_000_000,
                "pinned": bool(kwargs.get("require_pin") or kwargs.get("pin")),
                "buffers": 2,
                "buffer_bytes": 4_000_000,
                "n_layers": 2,
            }
            return MagicMock(), runtime

        monkeypatch.setattr(rt, "build_streamed_model", fake_build)

    def test_stream_pin_threads_require_pin_into_the_runtime_and_announces_cpu(
        self, tmp_path, monkeypatch
    ):
        """#366 re-review blocker 2: prove the key REACHES the runtime call —
        `captured` fails with KeyError if the require_pin= argument at
        stream_setup.py is deleted, which is the mutation that previously left
        the suite green. And blocker 1: on CPU pinning is inapplicable (no CUDA
        device), so an explicit stream_pin=true is ANNOUNCED, not silently
        dropped. CPU-only — build_streamed_model is stubbed."""
        import soup_cli.trainer.stream_setup as ss

        messages = []

        class _RecordingConsole:
            def print(self, *args, **_kwargs):
                messages.append(" ".join(str(a) for a in args))

        monkeypatch.setattr(ss, "console", _RecordingConsole())
        captured = {}
        self._stub_build_streamed_model(monkeypatch, captured)
        self._run(
            tmp_path, monkeypatch, free_ram=10_000_000_000, stream_source="auto",
            device="cpu", extra_training_yaml="  stream_pin: true\n",
        )
        # The runtime call received require_pin (blocker 2) — False here because
        # it is gated on a real CUDA device, which is the correct value on CPU.
        assert captured["require_pin"] is False
        # The explicit request was announced, not dropped (blocker 1).
        assert any("no CUDA device" in m for m in messages), messages

    def _run_capture(self, tmp_path, monkeypatch, extra_training_yaml):
        """Drive the real pre-flight and return what a user would SEE.

        The plan notes reach the user inside a ``rich.panel.Panel``, so a
        recording console that merely stringifies its arguments captures the
        Panel's object repr and sees none of the note text — a false negative.
        It has to render through a real ``Console`` writing to a buffer, wide
        enough that the measured figures are not broken across a wrap."""
        import io

        from rich.console import Console

        import soup_cli.trainer.stream_setup as ss

        buffer = io.StringIO()
        monkeypatch.setattr(
            ss, "console", Console(file=buffer, width=_PANEL_CAPTURE_WIDTH)
        )
        self._stub_build_streamed_model(monkeypatch, {})
        self._run(
            tmp_path, monkeypatch, free_ram=_RAM_TIER_FREE_BYTES,
            stream_source="auto", device="cpu",
            extra_training_yaml=extra_training_yaml,
        )
        return buffer.getvalue()

    def test_stream_pin_false_makes_the_preflight_state_the_cost(
        self, tmp_path, monkeypatch
    ):
        """#366 round-3 C1: the VALUE of training.stream_pin must reach the plan,
        not merely the keyword. Hardcoding `stream_pin=None` (or True/False) at
        the build_stream_plan call left the whole suite green, because every
        existing test called build_stream_plan directly and nothing crossed the
        wiring. Driven end-to-end through the real pre-flight."""
        out = self._run_capture(tmp_path, monkeypatch, "  stream_pin: false\n")
        assert _FORCED_PAGEABLE_TEXT in out, out
        assert _PIN_GAIN_REAL_TEXT in out, out

    def test_unset_stream_pin_does_not_state_a_forced_pageable_cost(
        self, tmp_path, monkeypatch
    ):
        """The control that makes the test above discriminating: without it a
        mutant that ALWAYS emitted the forced-pageable note would pass."""
        out = self._run_capture(tmp_path, monkeypatch, "")
        assert _FORCED_PAGEABLE_TEXT not in out, out
        assert _PIN_GAIN_REAL_TEXT not in out, out

    def _run_on_simulated_cuda(self, tmp_path, monkeypatch, extra_training_yaml):
        """Drive the pre-flight down its `on_cuda` branch WITHOUT a GPU.

        `on_cuda` is `str(self.device).startswith("cuda")` — a plain string
        check — so the branch itself needs no device. Only two things behind it
        do: the free-VRAM measurement and the allocator hint, both stubbed here.
        `build_streamed_model` is stubbed too, so nothing ever reaches a kernel.

        Written this way on purpose: gated behind @requires_cuda these two cases
        would skip on CI, which is exactly where the mutation they exist to kill
        needs to die."""
        import torch

        import soup_cli.utils.layer_stream_runtime as rt

        monkeypatch.setattr(
            torch.cuda, "mem_get_info",
            lambda *_a, **_k: (_SIMULATED_FREE_VRAM_BYTES, _SIMULATED_FREE_VRAM_BYTES),
        )
        # Patched on the SOURCE module, like build_streamed_model above:
        # stream_setup imports it inside the function.
        monkeypatch.setattr(
            rt, "expandable_segments_status", lambda *_a, **_k: (True, "")
        )
        captured = {}
        self._stub_build_streamed_model(monkeypatch, captured)
        self._run(
            tmp_path, monkeypatch, free_ram=_RAM_TIER_FREE_BYTES,
            stream_source="auto", device="cuda",
            extra_training_yaml=extra_training_yaml,
        )
        return captured

    def test_stream_pin_true_reaches_the_runtime_as_true_on_cuda(
        self, tmp_path, monkeypatch
    ):
        """#366 round-3 C3: on CPU `(stream_pin is True) and on_cuda` is False
        whatever stream_pin holds, so a CPU assertion of `is False` is satisfied
        identically by the real expression and by a hardcoded constant — it
        cannot discriminate. This is the case that pins the other half of that
        conjunction, and therefore the only one that fails when `require_pin` is
        hardcoded to False."""
        captured = self._run_on_simulated_cuda(
            tmp_path, monkeypatch, "  stream_pin: true\n"
        )
        assert captured["require_pin"] is True

    def test_unset_stream_pin_reaches_the_runtime_as_false_on_cuda(
        self, tmp_path, monkeypatch
    ):
        """Control for the case above — otherwise a hardcoded `True` would pass
        and the pair would prove nothing about the value."""
        captured = self._run_on_simulated_cuda(tmp_path, monkeypatch, "")
        assert captured["require_pin"] is False

    def test_auto_uses_ram_when_the_base_fits(self, tmp_path, monkeypatch):
        wrapper = self._run(
            tmp_path, monkeypatch, free_ram=10_000_000_000, stream_source="auto"
        )
        assert wrapper._stream_runtime.tier == "ram"

    def test_auto_falls_back_to_disk_when_it_does_not(self, tmp_path, monkeypatch):
        wrapper = self._run(tmp_path, monkeypatch, free_ram=1000, stream_source="auto")
        assert wrapper._stream_runtime.tier == "disk"
        assert wrapper._stream_runtime.stats()["store_bytes"] == 0

    def test_ram_refuses_instead_of_falling_back(self, tmp_path, monkeypatch):
        """`auto` trades speed to complete the run; `ram` is how an operator says
        they would rather be told no."""
        with pytest.raises(ValueError, match="stream_source='ram'"):
            self._run(tmp_path, monkeypatch, free_ram=1000, stream_source="ram")

    def test_ram_refusal_does_not_depend_on_the_disk_kind(self, tmp_path, monkeypatch):
        """A `ram`-only run is refused for the same reason whatever the disk is,
        so it must not pay the ~9 s probe NOR surface `choose_tier`'s generic
        "needs NVMe" message instead of the actionable one."""
        probed = []

        def _probe(*_a, **_k):
            probed.append(1)
            return "hdd"

        monkeypatch.setattr("soup_cli.utils.layer_stream.detect_disk_kind", _probe)
        with pytest.raises(ValueError, match="stream_source='ram'"):
            self._run(
                tmp_path, monkeypatch, free_ram=1000, stream_source="ram",
                disk_kind="hdd",
            )
        assert probed == [], "the disk probe ran for a decision it cannot change"

    def test_a_non_nvme_disk_is_refused_not_thrashed(self, tmp_path, monkeypatch):
        """plan P11 — on a spinning disk each step costs two seeks per layer, so
        the fallback must refuse rather than silently degrade."""
        with pytest.raises(ValueError, match="NVMe"):
            self._run(
                tmp_path, monkeypatch, free_ram=1000, stream_source="auto",
                disk_kind="hdd",
            )

    @requires_cuda
    def test_an_over_budget_config_is_refused_by_the_trainer_not_just_the_math(
        self, tmp_path, monkeypatch
    ):
        """`decide_stream_fit` is unit-tested against the measured grid, but the
        refusal only protects anyone if `_setup_streaming_transformers` actually
        consults it and STOPS. Driven with free VRAM mocked to a sliver."""
        import torch

        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *_a, **_k: (1_000_000, 4_000_000_000)
        )
        with pytest.raises(ValueError, match="predicted to need"):
            self._run(
                tmp_path, monkeypatch, free_ram=10_000_000_000,
                stream_source="auto", device="cuda",
            )

    @requires_cuda
    def test_stream_vram_override_below_measured_free_refuses_a_fitting_config(
        self, tmp_path, monkeypatch
    ):
        """#347: mem_get_info() is device-level and cannot see a per-process cap
        (a Colab/Kaggle set_per_process_memory_fraction, a MIG slice, a shared
        card), so it reports plenty of free VRAM here on purpose. The override
        must still make the pre-flight refuse, because on the real hardware this
        models that is the whole point."""
        import torch

        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *_a, **_k: (16_000_000_000, 16_000_000_000)
        )
        with pytest.raises(ValueError, match="predicted to need"):
            self._run(
                tmp_path, monkeypatch, free_ram=10_000_000_000,
                stream_source="auto", device="cuda",
                extra_training_yaml="  stream_vram_override: 1000000\n",
            )

    @requires_cuda
    def test_stream_vram_override_above_measured_free_lets_a_refused_config_through(
        self, tmp_path, monkeypatch
    ):
        """The other direction: a documented over-prediction that would
        otherwise refuse a known-safe config now proceeds."""
        import torch

        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *_a, **_k: (1_000_000, 4_000_000_000)
        )
        wrapper = self._run(
            tmp_path, monkeypatch, free_ram=10_000_000_000,
            stream_source="auto", device="cuda",
            extra_training_yaml="  stream_vram_override: 16000000000\n",
        )
        assert wrapper._stream_runtime is not None

    @requires_cuda
    def test_the_measured_probe_actually_runs_from_setup(self, tmp_path, monkeypatch):
        """v0.73.1 (#349): `_run_stream_vram_probe` is unit-tested by calling it
        directly, which proves the method and NOT that anything calls it. This
        drives the real `_setup_streaming_transformers`, so a probe that were
        wired up but never invoked would fail here.

        It also pins the shape: the probe must be asked about the SAME rows and
        seq the formula budgeted, or the two numbers printed side by side are
        about different runs.
        """
        import torch

        seen = {}

        def _fake_probe(model, *, rows, seq_len, vocab_size, device):
            from soup_cli.utils.layer_stream_runtime import StepPeak

            seen.update(rows=rows, seq_len=seq_len, vocab_size=vocab_size)
            return StepPeak(
                peak_bytes=1_000, reserved_bytes=1_000, seconds=0.01,
                rows=rows, seq_len=seq_len,
            )

        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.measure_step_peak_bytes", _fake_probe
        )
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *_a, **_k: (4_000_000_000, 4_000_000_000)
        )
        wrapper = self._run(
            tmp_path, monkeypatch, free_ram=10_000_000_000,
            stream_source="auto", device="cuda",
            extra_training_yaml="  stream_vram_probe: true\n",
        )
        assert seen, "setup() never invoked the measured VRAM probe"
        assert seen["rows"] == 1, seen
        assert seen["seq_len"] == wrapper.config.data.max_length, seen
        assert seen["vocab_size"] > 0, seen

    @requires_cuda
    def test_a_measured_miss_stops_setup_from_completing(self, tmp_path, monkeypatch):
        """Control for the test above: it asserts the probe is CALLED, which a
        wiring that ignored the answer would also satisfy."""
        import torch

        from soup_cli.utils.layer_stream_runtime import StepPeak

        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.measure_step_peak_bytes",
            lambda *_a, **kw: StepPeak(
                peak_bytes=9_000_000_000, reserved_bytes=9_000_000_000,
                seconds=0.01, rows=kw["rows"], seq_len=kw["seq_len"],
            ),
        )
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *_a, **_k: (4_000_000_000, 4_000_000_000)
        )
        with pytest.raises(ValueError, match="MEASURED"):
            self._run(
                tmp_path, monkeypatch, free_ram=10_000_000_000,
                stream_source="auto", device="cuda",
                extra_training_yaml="  stream_vram_probe: true\n",
            )

    def test_the_fallback_says_the_cost_is_unmeasured(self, tmp_path, monkeypatch):
        """A silent fallback to a slower path is the failure mode this project
        keeps calling out; the note must not overstate what was measured."""
        from soup_cli.utils.layer_stream import build_stream_plan

        plan = build_stream_plan(
            arch="llama", n_layers=2, layer_bytes=1000, embed_bytes=0,
            available_ram_bytes=10, pinned_limit_bytes=None, disk_kind="nvme",
        )
        joined = " ".join(plan.notes)
        assert "unmeasured" in joined and "stream_source='ram'" in joined


def _write_tiny_tokenizer(weights_dir):
    """A minimal tokenizer so the trainer's AutoTokenizer load succeeds."""
    import json as _json
    import os

    vocab = {f"<{i}>": i for i in range(64)}
    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<0>"},
    }
    with open(os.path.join(weights_dir, "tokenizer.json"), "w", encoding="utf-8") as fh:
        _json.dump(payload, fh)
    with open(
        os.path.join(weights_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as fh:
        _json.dump(
            {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<0>",
             "eos_token": "<0>", "pad_token": "<0>"}, fh
        )


class TestRequirePinSurvivesEveryHop:
    """#366 round-3 C2: `require_pin` crosses two hops between the pre-flight and
    the source constructor — `build_streamed_model -> install_streaming` and
    `install_streaming -> _build_source`. Hardcoding `require_pin=False` at
    EITHER left the whole suite green, because the only refusal test called
    `_build_source` directly and so jumped over both. These drive the real chain
    over the existing tiny-checkpoint harness. CPU-only."""

    @staticmethod
    def _fail_pinning(monkeypatch):
        """Make page-locking fail the way a box out of pinnable RAM does.

        Subclasses the REAL RamSource rather than replacing it with a stub: the
        pageable construction has to actually work (the control below streams
        through it), and `install_streaming` calls `RamSource.spec_from_shard`
        before ever building a source."""
        import soup_cli.utils.layer_stream_runtime as rt

        class _FailsWhenPinned(rt.RamSource):
            def __init__(
                self, shard_dir, n_layers, spec, *, pin=True, shard_paths=None
            ):
                if pin:
                    raise RuntimeError("CUDA error: cannot allocate pinned memory")
                super().__init__(
                    shard_dir,
                    n_layers,
                    spec,
                    pin=False,
                    shard_paths=shard_paths,
                )

        monkeypatch.setattr(rt, "RamSource", _FailsWhenPinned)

    def test_forced_pin_refuses_through_the_whole_real_chain(
        self, tmp_path, monkeypatch
    ):
        self._fail_pinning(monkeypatch)
        with pytest.raises(RuntimeError, match="stream_pin"):
            _tiny_stream(tmp_path, pin=True, require_pin=True)

    def test_the_same_failure_falls_back_silently_without_the_request(
        self, tmp_path, monkeypatch
    ):
        """The control that makes the case above discriminating: the identical
        page-lock failure must still fall back when nobody asked for the pin, so
        the refusal is attributable to the flag and not to the failure."""
        self._fail_pinning(monkeypatch)
        _model, runtime, _weights = _tiny_stream(tmp_path, pin=True)
        assert runtime.stats()["pinned"] is False


class TestDiskTierConfig:
    def test_stream_source_disk_is_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_stream_yaml(stream_source="disk"))
        assert cfg.training.stream_source == "disk"

    def test_stream_source_auto_is_still_the_default(self):
        from soup_cli.config.loader import load_config_from_string

        assert load_config_from_string(_stream_yaml()).training.stream_source == "auto"


# ==========================================================================
# item 3 — gradient accumulation
# ==========================================================================
class TestGradientAccumulationIsSupported:
    def test_accumulation_above_one_is_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_stream_yaml(gradient_accumulation_steps=4))
        assert cfg.training.gradient_accumulation_steps == 4

    def test_batch_and_accumulation_compose(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            _stream_yaml(batch_size=2, gradient_accumulation_steps=2)
        )
        assert (cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (2, 2)


class TestAccumulationIsPerTokenIoNeutral:
    """The measured invariant behind the whole item, executed rather than
    asserted from a docstring: `accum=N` reads the base N times AND processes N
    times the tokens, so **layer reads per token do not move**. GATE 3 measured
    175.78 loads/1k tokens constant across accum 1/2/4."""

    def test_layer_reads_per_token_are_unchanged_by_accumulation(self, tmp_path):
        import torch

        def reads_per_token(accum, micro_batches):
            model, runtime, _ = _tiny_stream(tmp_path, name=f"a{accum}", seed=3)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4
            )
            torch.manual_seed(0)
            ids = torch.randint(0, 64, (1, 12))
            model.train()
            before = runtime.pool.loads
            for _ in range(micro_batches // accum):
                for _ in range(accum):
                    (model(input_ids=ids, labels=ids).loss / accum).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            tokens = micro_batches * 12
            return (runtime.pool.loads - before) / tokens

        # Same token budget both ways: 8 micro-batches either as 8 steps of 1 or
        # 2 steps of 4.
        assert reads_per_token(1, 8) == pytest.approx(reads_per_token(4, 8))

    def test_a_bigger_batch_really_does_amortise_the_read(self, tmp_path):
        """The other half, and the reason the advisory says to raise batch
        first: one weight read covering more tokens IS fewer reads per token.
        Without this the test above is consistent with reads never changing."""
        import torch

        def reads_per_token(batch):
            model, runtime, _ = _tiny_stream(tmp_path, name=f"b{batch}", seed=3)
            torch.manual_seed(0)
            ids = torch.randint(0, 64, (batch, 12))
            model.train()
            before = runtime.pool.loads
            model(input_ids=ids, labels=ids).loss.backward()
            return (runtime.pool.loads - before) / (batch * 12)

        assert reads_per_token(4) < reads_per_token(1)


class TestAccumulationAdvice:
    """GATE 3 measured the thing users will get wrong. Per TOKEN accumulation is
    I/O-neutral (layer loads per 1k tokens constant at 175.78 across accum 1/2/4),
    so the cost is opportunity cost: at the same effective batch, raising
    batch_size was 2.52x faster because one weight read covers 4x the tokens."""

    def test_advises_raising_batch_when_accumulating_at_batch_one(self):
        from soup_cli.utils.layer_stream import accumulation_advice

        note = accumulation_advice(batch_size=1, accum=4)
        assert note is not None
        assert "batch_size" in note

    def test_quotes_the_measured_ratio_not_a_guess(self):
        from soup_cli.utils.layer_stream import (
            ACCUM_VS_BATCH_SPEEDUP,
            accumulation_advice,
        )

        assert ACCUM_VS_BATCH_SPEEDUP == pytest.approx(2.5, abs=0.1)
        assert f"{ACCUM_VS_BATCH_SPEEDUP:.1f}x" in accumulation_advice(
            batch_size=1, accum=2
        )

    def test_says_accumulation_holds_vram_flat(self):
        """Its real value under streaming: effective batch at constant VRAM
        (peak moved 0.842 -> 0.846 GB across accum 1->4)."""
        note = _advice(batch_size=1, accum=4)
        assert "VRAM" in note

    def test_silent_when_not_accumulating(self):
        assert _advice(batch_size=4, accum=1) is None

    def test_rejects_nonsense_counts(self):
        from soup_cli.utils.layer_stream import accumulation_advice

        with pytest.raises(ValueError, match="positive"):
            accumulation_advice(batch_size=0, accum=1)


def _advice(**kw):
    from soup_cli.utils.layer_stream import accumulation_advice

    return accumulation_advice(**kw)


# ==========================================================================
# item 4 — checkpoint / resume
# ==========================================================================
def _tiny_stream(
    tmp_path, name="shards", seed=3, n_layers=3, device="cpu", tier="ram",
    quant="none", pin=False, require_pin=False,
):
    """A streamed model over a real (tiny) on-disk Llama checkpoint."""
    import torch
    from peft import LoraConfig, TaskType
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, LlamaConfig

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = tmp_path / "model"
    if not weights.exists():
        torch.manual_seed(7)
        # hidden_size 64, not 32, and that is load-bearing for the NF4 tests:
        # bitsandbytes' CPU 4-bit forward reshapes absmax to
        # [rows, blocks_per_row], and at hidden 32 there are only 16 absmax
        # blocks for 32 rows, so blocks_per_row floors to ZERO and it raises
        # "shape '[32, 0]' is invalid for input of size 16". A CUDA build never
        # calls that path, so it is invisible on a GPU box and fails on every
        # CPU-only CI runner. Do not shrink this back.
        config = LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=64,
            num_hidden_layers=n_layers, num_attention_heads=4,
            num_key_value_heads=2, tie_word_embeddings=True,
            max_position_embeddings=128,
        )
        model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
        weights.mkdir(parents=True, exist_ok=True)
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        state.pop("lm_head.weight", None)
        save_file(state, str(weights / "model.safetensors"))
        config.save_pretrained(str(weights))
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
    )
    shards = str(tmp_path / name)
    suffixes = ()
    if quant == "nf4":
        from soup_cli.utils.layer_stream_runtime import (
            build_meta_skeleton,
            quantised_layer_suffixes,
        )

        probe = build_meta_skeleton(str(weights), dtype="float32", quant="nf4")
        suffixes = quantised_layer_suffixes(probe)
        del probe
    index = shard_checkpoint(
        str(weights), shards, dtype="float32", arch="llama", quant=quant,
        quant_suffixes=suffixes, quant_device="cpu",
    )
    model, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=shards, index=index, lora_config=lora,
        device=device, dtype="float32", buffers=2, pin=pin, seed=seed,
        tier=tier, quant=quant, require_pin=require_pin,
    )
    return model, runtime, str(weights)


def _randomise_lora_b(model, seed=11):
    """PEFT initialises lora_B to ZERO, so an adapter that fails to load is
    byte-identical to a freshly built one and every assertion below would pass
    vacuously. Give B real values first."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
                count += 1
    assert count, "no lora_B parameters — the adapter never attached"
    return count


def _sync_lora(src, dst):
    """Copy adapter weights src -> dst, normalising the streaming wrapper's
    `.inner.` segment so a streamed and a non-streamed model can be compared."""
    import torch

    s = {k.replace(".inner.", "."): v for k, v in src.state_dict().items() if "lora_" in k}
    d = {k.replace(".inner.", "."): v for k, v in dst.state_dict().items() if "lora_" in k}
    assert s and set(s) == set(d), (sorted(s)[:2], sorted(d)[:2])
    with torch.no_grad():
        for key, value in s.items():
            d[key].copy_(value)
    return len(s)


def _load_adapter_cpu(model, path):
    """Exercise the key-redirection MECHANISM without PEFT's device dispatch.

    ``PeftModel.load_adapter`` additionally re-dispatches the model when
    ``hf_device_map`` mentions "cpu", which rewrites the map ``install_streaming``
    relies on and moves everything to CUDA. That only happens on a CPU-built
    streamed model -- a configuration with no reason to exist, since streaming
    exists to bound VRAM. Production (CUDA) is covered by the CUDA test below,
    which drives the real ``load_adapter``.

    ``set_peft_model_state_dict`` is what ``load_adapter`` calls internally to
    place the weights, so this is the same load path minus the dispatch.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(pathlib.Path(path) / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, saved, adapter_name="default")
    return saved


def _train(model, batches, lr=5e-2):
    import torch

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    model.train()
    for ids in batches:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
    return losses


def _landed(model, saved):
    """How many saved tensors are present in the live model, by name AND value."""
    import torch

    norm = {
        k.replace("base_model.model.", "").replace(".weight", ""): v
        for k, v in saved.items()
    }
    count = 0
    for name, param in model.named_parameters():
        if "lora_" not in name or param.is_meta:
            continue
        key = (
            name.replace("base_model.model.", "")
            .replace(".inner.", ".")
            .replace(".default.", ".")
            .replace(".weight", "")
        )
        if key in norm and torch.equal(
            param.detach().cpu().float(), norm[key].cpu().float()
        ):
            count += 1
    return count


class TestResumeLoadsIntoAStreamedModel:
    """v0.72.1 made the SAVE direction canonical and deliberately left LOAD
    unsupported: ``state_dict()`` delegates at the wrapper's own prefix while
    ``named_parameters()`` still carries ``.inner.``. Measured before this fix --
    **0 of 12** tensors landed, PEFT emitted only a UserWarning, and the resumed
    loss curve was byte-identical to a from-scratch one."""

    def test_every_saved_tensor_lands_by_name_and_value(self, tmp_path):
        model, _, _ = _tiny_stream(tmp_path)
        _randomise_lora_b(model)
        ckpt = tmp_path / "ckpt"
        model.save_pretrained(str(ckpt))

        fresh, _, _ = _tiny_stream(tmp_path, name="shards2", seed=99)
        saved = _load_adapter_cpu(fresh, ckpt)
        assert saved and not [k for k in saved if ".inner." in k]
        assert _landed(fresh, saved) == len(saved)

    def test_the_assertion_would_catch_a_dropped_adapter(self, tmp_path):
        """Control. Without it the test above passes for any loader that leaves
        matching values in place -- 0-of-N loading raises nothing."""
        from safetensors.torch import load_file, save_file

        model, _, _ = _tiny_stream(tmp_path)
        _randomise_lora_b(model)
        ckpt = tmp_path / "ckpt"
        model.save_pretrained(str(ckpt))
        saved = load_file(str(ckpt / "adapter_model.safetensors"))

        # Re-mangle the keys back to a shape nothing can match, in a SEPARATE
        # dir (rewriting in place fails on Windows -- the file is still mmapped).
        broken = tmp_path / "broken"
        broken.mkdir()
        save_file(
            {k.replace(".layers.", ".layers.INVALID."): v for k, v in saved.items()},
            str(broken / "adapter_model.safetensors"),
        )
        (broken / "adapter_config.json").write_text(
            (ckpt / "adapter_config.json").read_text(encoding="utf-8"), encoding="utf-8"
        )

        fresh, _, _ = _tiny_stream(tmp_path, name="shards2", seed=99)
        _load_adapter_cpu(fresh, broken)
        assert _landed(fresh, saved) == 0, "a mangled checkpoint appeared to load"

    def test_a_strict_load_does_not_report_phantom_unexpected_keys(self, tmp_path):
        """The redirect MOVES the canonical key rather than copying it.

        Leaving the original behind makes the wrapper's own strict scan flag it
        `unexpected` — `self_attn` is not one of its children, only `inner` is —
        so `strict=True` raised "Unexpected key(s)" on a load that had actually
        succeeded. PEFT always passes `strict=False` and ignores the list, so it
        was inert in practice; it was still a landmine for any direct caller.
        """
        model, _, _ = _tiny_stream(tmp_path)
        _randomise_lora_b(model)
        canonical = {
            k.replace(".inner.", "."): v.clone() for k, v in model.state_dict().items()
        }
        result = model.load_state_dict(canonical, strict=False)
        assert list(result.unexpected_keys) == [], result.unexpected_keys

    def test_a_doubly_spelled_checkpoint_is_refused(self, tmp_path):
        """A file carrying BOTH spellings of one weight is malformed. Silently
        keeping one would load a tensor the file does not unambiguously name."""
        import re

        model, _, _ = _tiny_stream(tmp_path)
        # state_dict() is canonical, so the second spelling has to be added:
        # insert `.inner.` after the layer index, which is exactly the shape a
        # v0.72.0-era file used.
        sd = dict(model.state_dict())
        both = dict(sd)
        added = 0
        for key, value in sd.items():
            mangled = re.sub(r"(\.layers\.\d+\.)", r"\1inner.", key, count=1)
            if mangled != key:
                both[mangled] = value.clone()
                added += 1
        assert added, "no layer keys found — the collision was never constructed"
        with pytest.raises(ValueError, match="malformed"):
            model.load_state_dict(both, strict=False)

    def test_subprocess_tools_are_resolved_to_absolute_paths(self):
        """CWE-427: Windows CreateProcess searches the CURRENT DIRECTORY before
        PATH, so a bare "powershell" run from a cloned project would execute an
        attacker-planted binary sitting in that checkout."""
        import os

        from soup_cli.utils.layer_stream import _resolve_tool

        assert _resolve_tool("__soup_no_such_tool__") is None
        found = _resolve_tool("__soup_no_such_tool__", __file__)
        assert found == __file__, "a real fallback path should be accepted"
        resolved = _resolve_tool("python") or _resolve_tool("python3")
        if resolved is not None:
            assert os.path.isabs(resolved)

    def test_named_parameters_still_carries_inner(self, tmp_path):
        """The fix is load-side only, by design: it redirects keys at load time
        rather than re-parenting the module tree, so v0.72.0's bit-exactness
        gates stay valid without being re-run."""
        model, _, _ = _tiny_stream(tmp_path)
        assert any(".inner." in n for n, _ in model.named_parameters())


@requires_cuda
class TestResumeOnTheProductionPath:
    """The brief's gate, driven through the real ``load_adapter`` on CUDA --
    which is what ``Trainer._load_from_checkpoint`` calls for a PEFT model."""

    def test_resume_lands_everything_and_continues_the_loss_curve(self, tmp_path):
        import torch

        model, _, _ = _tiny_stream(tmp_path, device="cuda")
        _randomise_lora_b(model)
        torch.manual_seed(0)
        batches = [torch.randint(0, 64, (1, 12), device="cuda") for _ in range(12)]
        _train(model, batches[:6])
        ckpt = tmp_path / "ckpt"
        model.save_pretrained(str(ckpt))
        from safetensors.torch import load_file

        saved = load_file(str(ckpt / "adapter_model.safetensors"))

        resumed, _, _ = _tiny_stream(tmp_path, name="shards2", seed=99, device="cuda")
        before_map = dict(resumed.hf_device_map)
        resumed.load_adapter(str(ckpt), "default", is_trainable=True)

        assert _landed(resumed, saved) == len(saved)
        assert dict(resumed.hf_device_map) == before_map, (
            "hf_device_map was rewritten -- Trainer._move_model_to_device relies "
            "on it to skip .to() on meta weights"
        )
        assert sum(
            1 for n, p in resumed.named_parameters()
            if p.is_meta and ".layers." in n and "lora_" not in n
        ) > 0, "the base was materialised -- that is not streaming any more"

        after_resume = _train(resumed, batches[6:])
        scratch, _, _ = _tiny_stream(tmp_path, name="shards3", seed=99, device="cuda")
        after_scratch = _train(scratch, batches[6:])
        assert after_resume != after_scratch, (
            "resuming reproduced the from-scratch curve exactly -- the checkpoint "
            "contributed nothing"
        )


class TestResumeRevalidatesTheShardCache:
    def test_a_changed_source_checkpoint_changes_the_fingerprint(self, tmp_path):
        """The silent failure the brief names: resuming against a stale shard
        cache streams the WRONG weights with no error."""
        from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint

        _, _, weights = _tiny_stream(tmp_path)
        before = read_shard_index(str(tmp_path / "shards")).source_fingerprint
        (tmp_path / "model" / "model.safetensors").touch()
        after = shard_checkpoint(
            weights, str(tmp_path / "shards"), dtype="float32", arch="llama"
        ).source_fingerprint
        assert before and after and before != after


class TestResumeFlagsAreAccepted:
    """Asserted behaviourally, following v0.72.1: a source grep would break on a
    harmless refactor and pass on a guard moved into dead code.

    **Scope of this class, stated so it does not overclaim:** it proves only
    that the streaming-specific *refusal* is gone — the run proceeds past it and
    fails later, for an unrelated reason. It does NOT prove the redirect hook
    works; that is `TestResumeLoadsIntoAStreamedModel` (CPU, mechanism) and
    `TestResumeOnTheProductionPath` (CUDA, real `load_adapter`). A full
    `soup train --resume` end-to-end is not runnable on a box with torch < 2.6,
    where `transformers.check_torch_load_is_safe` refuses EVERY resume,
    streaming or not (CVE-2025-32434).
    """

    @pytest.mark.parametrize("flag", [["--resume", "ckpt"], ["--hf-resume"]])
    def test_streaming_no_longer_refuses_the_resume_flags(
        self, tmp_path, monkeypatch, flag
    ):
        import json as _json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        _tiny_stream(tmp_path)  # writes a real tiny checkpoint at tmp_path/model
        data = tmp_path / "data.jsonl"
        data.write_text('{"text": "hello world"}\n', encoding="utf-8")
        config = tmp_path / "soup.yaml"
        config.write_text(
            f"base: {tmp_path / 'model'}\n"
            "task: sft\n"
            f"data:\n  train: {_json.dumps(str(data))}\n  format: plaintext\n"
            "training:\n"
            "  stream_layers: true\n  batch_size: 1\n  quantization: none\n"
            "  gradient_accumulation_steps: 1\n  epochs: 1\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        argv = ["train", "--config", str(config), "--yes", *flag]
        if flag == ["--hf-resume"]:
            argv += ["--push-as", "someone/somewhere"]
        result = CliRunner().invoke(app, argv)
        assert "are not supported with" not in result.output, result.output
        assert "lands in v0.72.3" not in result.output, result.output
        # Positive half: the run reached the resume machinery rather than being
        # turned away at the streaming gate. Without this, deleting the flag
        # handling entirely would also satisfy the two assertions above.
        assert "resum" in _strip_ansi(result.output).lower(), result.output


def _stream_yaml(**over):
    fields = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "stream_layers": "true",
        "quantization": "none",
        "stream_source": "auto",
    }
    fields.update(over)
    return f"""
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: data.jsonl
training:
  batch_size: {fields["batch_size"]}
  gradient_accumulation_steps: {fields["gradient_accumulation_steps"]}
  quantization: {fields["quantization"]}
  stream_layers: {fields["stream_layers"]}
  stream_source: {fields["stream_source"]}
  lora:
    r: 16
"""
