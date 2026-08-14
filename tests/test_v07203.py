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
    )


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

    @pytest.mark.parametrize("row", MEASURED_VRAM_GRID, ids=lambda r: r["label"])
    def test_never_under_predicts(self, row):
        """The only safe direction for a gate that refuses configs. An estimator
        that is accurate on average but sometimes low still OOMs users.

        SCOPE, narrowed in v0.73.1 (#349): every row of this grid is at seq 256
        or 512. The grid varies BATCH (1..8), so this pins "never under-predicts
        as batch grows" and nothing about sequence length — a control only covers
        the variable it varies. Measured later on the same box, the property
        fails as seq grows: 0.992x the real peak at seq 4096 and 0.830x at 5120,
        deterministically. Read this as a bound on the regime below, not as the
        global guarantee the phrase suggests; `training.stream_vram_probe`
        exists because no fitted formula can carry that guarantee everywhere.
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

        from soup_cli.utils.layer_stream_runtime import _GEMM_SIZE, measure_gemm_tflops

        got = measure_gemm_tflops(device="cuda")
        assert got is not None

        size, iters = _GEMM_SIZE, 8
        left = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
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
            f"the same {size}^3 bf16 matmul gives {independent}"
        )

    def test_returns_none_on_cpu_rather_than_inventing_a_number(self):
        from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops

        assert measure_gemm_tflops(device="cpu") is None

    def test_takes_the_best_repeat_not_the_first(self):
        """A ceiling's noise is ONE-SIDED — contention, a cold clock and thermal
        throttling only ever make an achievable rate look slower. Measured: at
        2048 the probe spread 38% across five repeats and ramped monotonically
        upward, which is how a first-repeat probe reported 3.54 TFLOPS in one
        session and 6.75 in another at the same reported clock."""
        from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops

        one = measure_gemm_tflops(device="cuda", reps=1)
        many = measure_gemm_tflops(device="cuda", reps=4)
        assert one is not None and many is not None
        # best-of-4 can only be >= a single sample, up to run-to-run noise
        assert many.tflops >= one.tflops * 0.9


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
        monkeypatch.setattr(ls, "_probe_disk_kind", lambda p: calls.append(p) or "nvme")

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
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream.detect_disk_kind", lambda *_a, **_k: disk_kind
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
    quant="none",
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
        device=device, dtype="float32", buffers=2, pin=False, seed=seed,
        tier=tier, quant=quant,
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
