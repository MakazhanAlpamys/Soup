"""v0.73.1 #349 — decide the streaming VRAM fit on a MEASUREMENT, not a prediction.

The pre-flight predicts peak VRAM from a formula fitted to 10 real runs, and its
documented contract is that it NEVER under-predicts. Measured through the real
``soup train`` on an RTX 3050 Laptop (4 GB, Windows, torch 2.5.1), SmolLM2-135M
streamed in bf16 at batch 1, that contract holds at short sequence and then
breaks::

    seq    predicted    real peak    pred/real
    4352      3.282 GB     3.036 GB      1.081   over-predicts, safe
    5120      3.844 GB     4.118 GB      0.934   UNDER-predicts
    6144      4.590 GB     5.830 GB      0.787   UNDER by 21%

That is the failure direction the pre-flight exists to prevent, and the one that
does not announce itself: on Linux it is an OOM, on Windows/WDDM it is a silent
spill to host memory. A formula cannot see a term it does not model. A
measurement can, which is the whole of #349.

The MECHANISM is deliberately NOT claimed. ``seq**2`` from the attention score
matrix is the obvious candidate and the numbers do not settle it, so what makes
the formula diverge past ~4700 tokens is recorded as unidentified rather than
guessed at. Two readings were tried and withdrawn during this work — first "the
preference losses are over-budgeted ~8.8x" (it is 1.15x at the budgeted shape,
and the earlier figure came from rows that realised 142 of a budgeted 2048
tokens), then "the run silently spilled" (``num_alloc_retries`` was 0 on every
shape measured). Both are noted here because the tempting inference was wrong
twice in the same investigation.

The probe is not the same instrument as the training step and does not pretend
to be: measured standalone it reads 3.471 GB where the real run peaks at 3.036
(seq 4352) and 4.633 against 4.118 (seq 5120), i.e. **+13-14%, consistently
conservative**. For a gate that is the safe direction, and it is why the probe
crosses into refusal slightly before the training step would.

The gate quantity is ``max_memory_allocated``, NOT ``max_memory_reserved``, and
that choice has a counterexample behind it rather than an intuition: the
project's own flagship configuration (Llama-3.1-8B NF4, batch 1 x seq 512)
measured 3.4116 GB allocated against 3.6973 GB reserved with only 3.45 GB free,
and it runs — the caching allocator holds freed blocks it returns under
pressure, and ``num_alloc_retries`` was 0 on every shape measured here. Gating on
reserved would refuse the headline config of the feature it is protecting.
"""

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.layer_stream import decide_measured_fit, decide_stream_fit

GB = 1_000_000_000


def _cfg(**training):
    body = "\n".join(f"  {k}: {v}" for k, v in training.items())
    return (
        "base: HuggingFaceTB/SmolLM2-135M\n"
        "task: sft\n"
        "data:\n"
        "  train: d.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 1\n"
        "  lora:\n"
        "    r: 8\n" + body + "\n"
    )


class TestTheMeasuredGateOutranksThePrediction:
    """The measurement decides; the prediction is printed beside it."""

    def test_a_measured_peak_over_available_is_refused_even_when_predicted_to_fit(self):
        """The reproduced defect: seq 5120 predicted 3.844 GB and used 4.633 GB.

        The prediction passes its own gate and the run still does not fit. Only
        the measurement sees it, so only the measurement can refuse it.
        """
        assert decide_stream_fit(
            predicted_bytes=int(3.844 * GB), available_bytes=int(4.0 * GB)
        ).fits, "control: the PREDICTION accepts this shape, which is the whole problem"

        fit = decide_measured_fit(
            measured_bytes=int(4.633 * GB),
            predicted_bytes=int(3.844 * GB),
            available_bytes=int(4.0 * GB),
        )
        assert not fit.fits
        assert "measured" in fit.reason.lower()
        # The divergence has to be visible or the user cannot tell that the
        # formula was wrong rather than their config being too big.
        assert "3.84" in fit.reason and "4.63" in fit.reason

    def test_a_measured_peak_that_fits_unblocks_a_config_the_prediction_refused(self):
        """The other direction: an over-prediction must not refuse a run that fits."""
        assert not decide_stream_fit(
            predicted_bytes=int(3.6 * GB), available_bytes=int(3.45 * GB)
        ).fits, "control: the PREDICTION refuses this shape"

        fit = decide_measured_fit(
            measured_bytes=int(2.1 * GB),
            predicted_bytes=int(3.6 * GB),
            available_bytes=int(3.45 * GB),
        )
        assert fit.fits
        assert "2.10" in fit.reason

    def test_the_boundary_is_inclusive_so_exactly_available_still_fits(self):
        fit = decide_measured_fit(
            measured_bytes=int(3.0 * GB),
            predicted_bytes=int(3.0 * GB),
            available_bytes=int(3.0 * GB),
        )
        assert fit.fits

    def test_measured_bytes_is_what_vramfit_reports_not_the_prediction(self):
        """A VramFit whose predicted_bytes held the measurement would make the
        panel print one number twice and hide exactly the divergence #349 exists
        to surface."""
        fit = decide_measured_fit(
            measured_bytes=7 * GB, predicted_bytes=3 * GB, available_bytes=4 * GB
        )
        assert fit.predicted_bytes == 3 * GB
        assert fit.measured_bytes == 7 * GB
        assert fit.available_bytes == 4 * GB


class TestTheProbeFailsSafeRatherThanFailingOpen:
    """An instrument that breaks must not remove the gate it was added beside."""

    def test_the_probe_returns_none_without_cuda_instead_of_raising(self):
        """A pre-flight that dies because its own instrument failed is worse than
        one that falls back to the shipped prediction — the same contract
        measure_logits_loss_bytes_per_element already keeps."""
        import torch

        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        if torch.cuda.is_available():
            pytest.skip("asserts the no-CUDA fallback; this box has CUDA")
        assert measure_step_peak_bytes(object(), rows=1, seq_len=8, vocab_size=32) is None

    def test_step_peak_carries_the_oom_flag_so_a_refusal_is_distinguishable(self):
        """An OOM is a measurement RESULT ("does not fit"), not an instrument
        failure ("cannot tell"). Collapsing them into None would turn a refusal
        into a silent fallback to the prediction that just accepted the run."""
        from soup_cli.utils.layer_stream_runtime import StepPeak

        peak = StepPeak(
            peak_bytes=0, reserved_bytes=0, seconds=1.0, rows=1, seq_len=4096, oom=True
        )
        assert peak.oom is True


class TestTheProbeIsActuallyWiredIntoSetup:
    """The pure decision passing proves nothing about `setup()` calling it."""

    @staticmethod
    def _wrapper(monkeypatch, peak):
        """A StreamingSetupMixin whose probe returns `peak`, with a close spy."""
        from soup_cli.trainer import stream_setup

        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.measure_step_peak_bytes",
            lambda *a, **k: peak,
        )

        class _Wrapper(stream_setup.StreamingSetupMixin):
            device = "cuda"

            def __init__(self):
                self.closed = 0
                self._stream_runtime = None

            def _close_stream_runtime(self):
                self.closed += 1

        return _Wrapper()

    @staticmethod
    def _plan(predicted, available):
        from soup_cli.trainer.stream_setup import _ProbePlan

        return _ProbePlan(
            rows=1,
            seq_len=4096,
            vocab_size=32,
            predicted_bytes=predicted,
            available_bytes=available,
        )

    def _peak(self, alloc, *, oom=False):
        from soup_cli.utils.layer_stream_runtime import StepPeak

        return StepPeak(
            peak_bytes=alloc,
            reserved_bytes=alloc,
            seconds=1.0,
            rows=1,
            seq_len=4096,
            oom=oom,
        )

    def test_a_measured_miss_refuses_a_run_the_prediction_accepted(self, monkeypatch):
        """The reproduced defect, driven through the wiring rather than the maths."""
        wrapper = self._wrapper(monkeypatch, self._peak(int(4.633 * GB)))
        with pytest.raises(ValueError) as exc:
            wrapper._run_stream_vram_probe(object(), self._plan(int(3.844 * GB), 4 * GB))
        assert "MEASURED" in str(exc.value)
        assert wrapper.closed == 1, "the pinned RAM store must be released before raising"

    def test_a_measured_hit_proceeds_and_releases_nothing(self, monkeypatch):
        wrapper = self._wrapper(monkeypatch, self._peak(int(2.1 * GB)))
        wrapper._run_stream_vram_probe(object(), self._plan(int(3.6 * GB), int(3.45 * GB)))
        assert wrapper.closed == 0

    def test_a_probe_oom_is_a_refusal_not_a_fallback(self, monkeypatch):
        wrapper = self._wrapper(monkeypatch, self._peak(0, oom=True))
        with pytest.raises(ValueError) as exc:
            wrapper._run_stream_vram_probe(object(), self._plan(int(3.0 * GB), 4 * GB))
        assert "ran out of VRAM" in str(exc.value)
        assert wrapper.closed == 1

    def test_a_probe_that_raised_mid_step_refuses_rather_than_falling_back(
        self, monkeypatch
    ):
        """`failed` is not `None`. The probe ran a real CUDA op and it raised,
        which can leave the context poisoned — so "the arithmetic was happy" is
        not a reason to keep driving the device. A fall-back here would be a
        silent pass on a prediction that never got checked."""
        from soup_cli.utils.layer_stream_runtime import StepPeak

        broke = StepPeak(
            peak_bytes=0,
            reserved_bytes=0,
            seconds=0.1,
            rows=1,
            seq_len=4096,
            failed=True,
            error="RuntimeError",
        )
        wrapper = self._wrapper(monkeypatch, broke)
        with pytest.raises(ValueError) as exc:
            # The PREDICTION fits here, so a fall-back would have proceeded.
            wrapper._run_stream_vram_probe(object(), self._plan(int(1.0 * GB), 4 * GB))
        assert "RuntimeError" in str(exc.value)
        assert wrapper.closed == 1

    def test_a_failed_probe_still_honours_a_prediction_that_refused(self, monkeypatch):
        """Fail-safe: a probe that could not run must not be a free pass through
        a gate the formula had already closed."""
        wrapper = self._wrapper(monkeypatch, None)
        with pytest.raises(ValueError) as exc:
            wrapper._run_stream_vram_probe(object(), self._plan(int(5.0 * GB), 4 * GB))
        assert "failed to run" in str(exc.value)
        assert wrapper.closed == 1

    def test_an_unexpected_exception_still_releases_the_pinned_ram_store(
        self, monkeypatch
    ):
        """`measure_step_peak_bytes` validates its arguments and raises BEFORE
        its own handler exists. Unreachable from the current caller, but the
        docstring promises the runtime is released before anything propagates,
        and a promise that holds only for today's paths breaks on the next one."""

        def _boom(*_a, **_k):
            raise ValueError("rows/seq_len/vocab_size must all be >= 1")

        wrapper = self._wrapper(monkeypatch, None)
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.measure_step_peak_bytes", _boom
        )
        with pytest.raises(ValueError, match="must all be >= 1"):
            wrapper._run_stream_vram_probe(object(), self._plan(GB, 4 * GB))
        assert wrapper.closed == 1

    def test_a_failed_probe_does_not_invent_a_refusal_when_the_formula_was_happy(
        self, monkeypatch
    ):
        """Control for the test above — otherwise 'refuses on None' would pass
        for an implementation that refuses on None unconditionally."""
        wrapper = self._wrapper(monkeypatch, None)
        wrapper._run_stream_vram_probe(object(), self._plan(int(1.0 * GB), 4 * GB))
        assert wrapper.closed == 0


class TestTheDeferralHasACeiling:
    """The probe corrects a margin, not an order of magnitude."""

    @staticmethod
    def _budget(probe, predicted_gb, free_gb, *, on_cuda=False, monkeypatch=None):
        """Drive the REAL `_stream_budget_lines`.

        `on_cuda=True` is where the fit decision (and so the ceiling) lives, so
        the device queries are patched rather than skipped: patching them keeps
        the code path under test identical on this box and on a GPU-less runner,
        where skipping would leave the ceiling covered on neither.
        """
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        if on_cuda:
            import torch

            monkeypatch.setattr(
                torch.cuda, "mem_get_info", lambda *_a, **_k: (int(free_gb * GB), 0)
            )
            monkeypatch.setattr(
                "soup_cli.utils.layer_stream_runtime.measure_gemm_tflops",
                lambda *_a, **_k: None,
            )

        class _Lora:
            r = 8
            target_modules = ["q_proj", "v_proj"]

        class _Train:
            batch_size = 1
            stream_buffers = 2
            stream_vram_probe = probe
            stream_vram_override = int(free_gb * GB)
            gradient_accumulation_steps = 1
            lora = _Lora()

        class _Model:
            # Sized so the logits term alone lands on `predicted_gb`.
            vocab_size = int(predicted_gb * GB / (14 * 1024))
            hidden_size = 8
            intermediate_size = 8

        class _Index:
            n_layers = 1
            total_params = 0

        class _Data:
            max_length = 1024

        class _Cfg:
            data = _Data()

        mixin = StreamingSetupMixin()
        mixin.device = "cuda" if on_cuda else "cpu"
        return mixin._stream_budget_lines(
            _Cfg(),
            _Train(),
            model_config=_Model(),
            layer_bytes=1000,
            embed_bytes=0,
            index=_Index(),
            on_cuda=on_cuda,
        )

    def test_far_over_budget_is_refused_outright_even_with_the_probe_on(self, monkeypatch):
        """Deferring here would trade a free arithmetic refusal for minutes of
        sharding plus a real allocation at that shape, driven by a soup.yaml
        whose author need not be whoever runs it."""
        with pytest.raises(ValueError) as exc:
            self._budget(True, 8.0, 1.0, on_cuda=True, monkeypatch=monkeypatch)
        assert "cannot overrule" in str(exc.value)

    def test_just_inside_the_ceiling_still_defers_to_the_measurement(self, monkeypatch):
        """Control: a ceiling that refused everything would delete the feature.
        3x over is inside the 4x ceiling, so the probe still gets to speak."""
        lines, plan = self._budget(True, 3.0, 1.0, on_cuda=True, monkeypatch=monkeypatch)
        assert plan is not None

    def test_without_the_probe_the_ceiling_changes_nothing(self, monkeypatch):
        """Control: over-budget without the probe still raises the ORIGINAL
        prediction message, not the new ceiling one."""
        with pytest.raises(ValueError) as exc:
            self._budget(False, 8.0, 1.0, on_cuda=True, monkeypatch=monkeypatch)
        assert "predicted to need" in str(exc.value)
        assert "cannot overrule" not in str(exc.value)

    def test_off_cuda_with_the_probe_on_says_the_gate_is_inactive(self, capsys):
        """A silently inactive gate reads exactly like an active one."""
        self._budget(True, 1.0, 8.0)
        assert "not on CUDA" in capsys.readouterr().out

    def test_off_cuda_without_the_probe_says_nothing_about_it(self, capsys):
        """Control: the message must be about the flag, not printed always."""
        self._budget(False, 1.0, 8.0)
        assert "not on CUDA" not in capsys.readouterr().out


class TestTheFormulaStillProducesTheNumbersThisIssueWasFiledOn:
    """The empirical claim behind #349, asserted against the REAL estimator.

    The fit tests above feed hand-typed literals into `decide_measured_fit`, so
    they would keep passing if `estimate_stream_peak_vram` changed underneath
    them. These pin the formula's own output at the two shapes that were
    measured on the box, so a future change to it surfaces here rather than
    silently invalidating the record in `benchmarks/`.
    """

    #: SmolLM2-135M streamed bf16, batch 1, buffers 2, r=8 — the configuration
    #: measured in benchmarks/gate-v0.73.1-measured-vram-fit.md.
    _GEOM = dict(
        layer_bytes=7_100_000,
        buffers=2,
        extras_bytes=56_600_000,
        adapter_params=1_000_000,
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        n_layers=30,
        batch_size=1,
    )

    @pytest.mark.parametrize(
        "seq, predicted_gb, measured_gb",
        [
            (4352, 3.28, 3.036),  # formula 1.081x the real peak — safe
            (5120, 3.84, 4.118),  # formula 0.934x — UNDER-predicts
            (6144, 4.59, 5.830),  # formula 0.787x — under by 21%
        ],
    )
    def test_the_formula_reproduces_its_published_prediction(
        self, seq, predicted_gb, measured_gb
    ):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        got = estimate_stream_peak_vram(seq_len=seq, **self._GEOM) / GB
        assert got == pytest.approx(predicted_gb, abs=0.02), (
            f"seq {seq}: formula now says {got:.3f} GB where the measured record "
            f"was taken against {predicted_gb} GB"
        )

    def test_the_record_is_an_under_prediction_at_the_long_shapes(self):
        """Guards the direction, not just the digits: if a future formula stopped
        under-predicting here, `stream_vram_probe`'s justification changes and
        this should be the thing that says so."""
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        assert estimate_stream_peak_vram(seq_len=6144, **self._GEOM) < 5.830 * GB
        assert estimate_stream_peak_vram(seq_len=4352, **self._GEOM) > 3.036 * GB


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")


class TestTheProbeItselfOnRealHardware:
    """`measure_step_peak_bytes` against a real model on a real device.

    Everything above stubs the probe out. These are the only tests that run the
    instrument, and without them "the probe returns a StepPeak" is asserted only
    against a StepPeak the test constructed itself.
    """

    @staticmethod
    def _toy():
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        return LlamaForCausalLM(cfg).to("cuda").to(torch.float32)

    @requires_cuda
    def test_it_returns_a_peak_that_grows_with_the_shape(self):
        """A probe returning a constant would satisfy every stubbed test above
        and measure nothing, so the discriminating property is that the number
        MOVES with the shape it was asked about."""
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        model = self._toy()
        small = measure_step_peak_bytes(model, rows=1, seq_len=16, vocab_size=64)
        large = measure_step_peak_bytes(model, rows=1, seq_len=256, vocab_size=64)
        assert small is not None and large is not None
        assert not small.oom and not small.failed
        assert large.peak_bytes > small.peak_bytes
        assert large.seq_len == 256 and large.rows == 1
        assert large.seconds > 0

    @requires_cuda
    def test_it_leaves_no_gradients_behind(self):
        """The probe backprops through random token ids. A gradient surviving
        into the first optimizer step would train on noise, once, silently, and
        only when the flag is on."""
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        model = self._toy()
        assert measure_step_peak_bytes(model, rows=1, seq_len=16, vocab_size=64)
        assert all(p.grad is None for p in model.parameters())

    @requires_cuda
    def test_a_nonsense_shape_is_rejected_rather_than_measured(self):
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        with pytest.raises(ValueError, match="must all be >= 1"):
            measure_step_peak_bytes(self._toy(), rows=0, seq_len=16, vocab_size=64)

    @requires_cuda
    def test_a_model_whose_parameters_break_does_not_lose_the_measurement(self, caplog):
        """`_zero_probe_grads` must never mask the result the caller came for —
        but it must not go quiet either, or the gradient corruption it exists to
        prevent would happen with nothing to find it by."""
        import torch

        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        real = self._toy()

        class _BadParams:
            def __init__(self, inner):
                self.inner = inner
                self.config = inner.config

            def __call__(self, **kw):
                return self.inner(**kw)

            def parameters(self):
                raise RuntimeError("no parameters for you")

        with caplog.at_level("WARNING"):
            peak = measure_step_peak_bytes(
                _BadParams(real), rows=1, seq_len=16, vocab_size=64
            )
        assert peak is not None and not peak.failed, "the measurement must survive"
        assert any("clear the VRAM probe's gradients" in r.message for r in caplog.records)
        del real
        torch.cuda.empty_cache()

    @requires_cuda
    def test_a_model_that_raises_reports_failed_not_none(self):
        """`failed` and `None` mean different things to the caller: one refuses,
        the other falls back to the prediction."""
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        class _Broken:
            def __call__(self, **_kw):
                raise RuntimeError("device-side assert triggered")

            def parameters(self):
                return iter(())

        peak = measure_step_peak_bytes(_Broken(), rows=1, seq_len=8, vocab_size=64)
        assert peak is not None, "an attempted-and-broken probe is not 'never attempted'"
        assert peak.failed is True
        assert peak.error == "RuntimeError"


class TestTheSchemaGate:
    def test_stream_vram_probe_parses_when_streaming_is_on(self):
        cfg = load_config_from_string(
            _cfg(stream_layers="true", stream_vram_probe="true", quantization="none")
        )
        assert cfg.training.stream_vram_probe is True

    def test_stream_vram_probe_defaults_off(self):
        cfg = load_config_from_string(_cfg(stream_layers="true", quantization="none"))
        assert cfg.training.stream_vram_probe is False

    def test_stream_vram_probe_is_refused_on_a_preference_task(self):
        """The probe runs a causal-LM step. That IS the SFT step, but a
        preference loss concatenates chosen+rejected and reduces logits to
        per-token log-probs, so it is a different computation and the probe's
        agreement with it is not established.

        Measured at ONE matching shape (SmolLM2-135M, batch 2 x seq 2048) the
        probe reads 6.02 GB against the real DPO step's 5.30 GB — +13.5%, the
        same SAFE direction it shows for SFT. So this is not "the probe is known
        to be unsafe here"; it is "one point is not a validation", and if the
        sign ever flips the failure is a gate waving through over-budget runs."""
        body = _cfg(
            stream_layers="true", stream_vram_probe="true", quantization="none"
        ).replace("task: sft", "task: dpo")
        with pytest.raises(Exception) as exc:
            load_config_from_string(body)
        assert "stream_vram_probe" in str(exc.value)
        assert "sft" in str(exc.value)

    def test_streaming_still_accepts_a_preference_task_without_the_probe(self):
        """Control: the refusal above must be about the PROBE, not re-refusing
        the preference streaming that v0.72.4 shipped."""
        body = _cfg(stream_layers="true", quantization="none").replace(
            "task: sft", "task: dpo"
        )
        cfg = load_config_from_string(body)
        assert cfg.training.stream_layers is True

    def test_batch_size_below_one_is_refused(self):
        """Pre-v0.73.1 `batch_size: -4` parsed and reached the VRAM arithmetic,
        which multiplies by it. Surfaced by the #349 security review."""
        for bad in (0, -4):
            with pytest.raises(Exception) as exc:
                load_config_from_string(_cfg(batch_size=bad).replace("batch_size: 1\n", ""))
            assert "batch_size" in str(exc.value)

    def test_batch_size_auto_and_positive_ints_still_parse(self):
        """Control: a bound that rejected the working values would be worse than
        the hole it closed."""
        assert load_config_from_string(_cfg(quantization="none")).training.batch_size == 1
        body = _cfg(quantization="none").replace("batch_size: 1", "batch_size: auto")
        assert load_config_from_string(body).training.batch_size == "auto"

    def test_stream_vram_probe_while_streaming_is_off_is_refused(self):
        """Mirrors the stream_source / stream_buffers / stream_vram_override
        footgun gate: setting it almost certainly means stream_layers was
        forgotten, and silently ignoring it would leave the user believing a
        measured gate was protecting the run."""
        with pytest.raises(Exception) as exc:
            load_config_from_string(_cfg(stream_layers="false", stream_vram_probe="true"))
        assert "stream_vram_probe" in str(exc.value)
        assert "stream_layers" in str(exc.value)
