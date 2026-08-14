"""Issue #327 — is ``LOGITS_BYTES_PER_ELEMENT`` stack-dependent, and can the
estimate be made stack-independent instead of guessed?

The finding these tests encode: **no, not by measuring.** The measurable half —
the loss path's own arithmetic — is already stack-independent, measured at
exactly 12.000000 bytes per logit element with zero spread under torch 2.5.1 AND
2.13.0 and under trl 0.19.1 AND 0.26.2 (H100, 3 repeats per cell). What differs
between the two published grids is a *retention*: one further bf16 logits-shaped
tensor still referenced when the loss backward peaks, worth exactly +2.0000 bytes
per element in a single-variable control. A synthetic probe cannot observe that —
it sees only its own reference — so the constant stays at the conservative 14 and
the calibration hook may only ever raise it.

The counterfactual is pinned here too: adopting the H100-fitted 12.311 would
under-predict **10 of the 10** v0.72.3 rows by up to 10.49%. Under-prediction is
the failure that does not raise on Windows/WDDM — it silently spills to host
memory — so a test that only checks accuracy would have let that through.
"""

import pytest


def _cuda() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a [train] extra
        return False
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")


#: The v0.72.3 GATE 2 grid, re-stated here so this file can pin the
#: counterfactual without importing (or being able to perturb)
#: ``tests/test_v07203.py``, which is the estimator's standing guard.
_SMOL = dict(pool=14160384, extras=56624256, adapter=921600, vocab=49152, hidden=576,
             intermediate=1536, n_layers=30)
_QWEN = dict(pool=59649536, extras=272271104, adapter=1081344, vocab=151936, hidden=896,
             intermediate=4864, n_layers=24)

_GRID = [
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

#: The slope the H100 grid in issue #327 fitted, and the one a real trl training
#: step measures on that box. Neither is a candidate replacement — see below.
_H100_FITTED_SLOPE = 12.311


def _predict_at(row, bytes_per_element):
    """The shipped formula with the logits slope swapped out, nothing else."""
    tokens = row["batch"] * row["seq"]
    activations = 2 * row["n_layers"] * row["hidden"] + 4 * (row["hidden"] + row["intermediate"])
    return (
        row["pool"]
        + row["extras"]
        + row["adapter"] * 16
        + 13_500_000
        + tokens * activations
        + int(row["vocab"] * tokens * bytes_per_element)
    )


class TestTheConstantIsTheSumOfTwoMeasuredTerms:
    """v0.72.3 shipped 14 as one fitted number with a decomposition that a
    stage-by-stage measurement does not support. The total was right; the story
    was not, and a wrong story is what made #327 look preference-specific."""

    def test_the_two_terms_sum_to_the_shipped_constant(self):
        from soup_cli.utils.layer_stream import (
            LOGITS_BYTES_PER_ELEMENT,
            LOGITS_LOSS_BYTES_PER_ELEMENT,
            LOGITS_RETENTION_BYTES_PER_ELEMENT,
        )

        assert LOGITS_LOSS_BYTES_PER_ELEMENT == 12
        assert LOGITS_RETENTION_BYTES_PER_ELEMENT == 2
        assert (
            LOGITS_LOSS_BYTES_PER_ELEMENT + LOGITS_RETENTION_BYTES_PER_ELEMENT
            == LOGITS_BYTES_PER_ELEMENT
        )

    def test_the_shipped_value_did_not_move(self):
        """Control: the split is a decomposition, not a re-fit. Every downstream
        prediction must be byte-identical to what v0.72.3 shipped."""
        from soup_cli.utils.layer_stream import LOGITS_BYTES_PER_ELEMENT, estimate_logits_bytes

        assert LOGITS_BYTES_PER_ELEMENT == 14
        assert estimate_logits_bytes(vocab_size=151936, seq_len=512, batch_size=1) == (
            512 * 151936 * 14
        )


class TestAdoptingTheH100SlopeWouldUnderPredict:
    """Why #327's proposed fix is refused. The issue fits 12.311 on one stack and
    says so honestly; these tests put a number on what swapping it would cost
    against the other stack's ten real measurements."""

    @pytest.mark.parametrize("row", _GRID, ids=lambda r: r["label"])
    def test_every_row_would_be_under_predicted(self, row):
        assert _predict_at(row, _H100_FITTED_SLOPE) < row["peak"], row["label"]

    def test_the_worst_under_prediction_is_over_ten_percent(self):
        worst = min(
            (_predict_at(r, _H100_FITTED_SLOPE) - r["peak"]) / r["peak"] for r in _GRID
        )
        assert worst < -0.10, f"worst under-prediction only {worst:.2%}"

    @pytest.mark.parametrize("row", _GRID, ids=lambda r: r["label"])
    def test_the_shipped_constant_is_the_control(self, row):
        """Without this the tests above are consistent with the grid being
        unpredictable by anything, rather than with 14 being the right slope."""
        predicted = _predict_at(row, 14)
        assert predicted >= row["peak"], row["label"]
        assert (predicted - row["peak"]) / row["peak"] < 0.01, row["label"]

    def test_the_loss_arithmetic_alone_is_not_a_candidate_either(self):
        """12 is what the loss path measurably costs, and it is still not the
        right budget — which is the whole argument for keeping the retention."""
        from soup_cli.utils.layer_stream import LOGITS_LOSS_BYTES_PER_ELEMENT

        assert all(
            _predict_at(r, LOGITS_LOSS_BYTES_PER_ELEMENT) < r["peak"] for r in _GRID
        )


class TestTheOverrideCanOnlyRaiseTheBudget:
    """The estimator is allowed to be wrong high. It is never allowed to be wrong
    low, so a measurement that comes back below the shipped constant is treated
    as a probe that could not see the retention, not as a cheaper stack."""

    def test_a_lower_reading_is_ignored(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        kw = dict(vocab_size=1000, seq_len=8, batch_size=1)
        assert estimate_logits_bytes(bytes_per_element=12.311, **kw) == 8 * 1000 * 14
        assert estimate_logits_bytes(bytes_per_element=0.0, **kw) == 8 * 1000 * 14

    def test_a_higher_reading_is_adopted(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        kw = dict(vocab_size=1000, seq_len=8, batch_size=1)
        assert estimate_logits_bytes(bytes_per_element=18.0, **kw) == 8 * 1000 * 18

    def test_a_fractional_reading_rounds_up_not_down(self):
        """Rounding down is an under-prediction, however small."""
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        got = estimate_logits_bytes(
            vocab_size=3, seq_len=1, batch_size=1, bytes_per_element=14.5
        )
        assert got == 44  # ceil(3 * 14.5) == 44, not 43

    def test_a_non_finite_reading_is_refused(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        with pytest.raises(ValueError, match="finite"):
            estimate_logits_bytes(
                vocab_size=10, seq_len=1, batch_size=1, bytes_per_element=float("nan")
            )

    def test_peak_vram_threads_the_override_and_floors_it(self):
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        kw = dict(
            layer_bytes=1000, buffers=2, extras_bytes=0, adapter_params=0,
            vocab_size=1024, hidden_size=8, intermediate_size=16, n_layers=2,
            seq_len=4, batch_size=1,
        )
        base = estimate_stream_peak_vram(**kw)
        assert estimate_stream_peak_vram(logits_bytes_per_element=12.311, **kw) == base
        raised = estimate_stream_peak_vram(logits_bytes_per_element=16.0, **kw)
        assert raised - base == 1024 * 4 * 2

    def test_calibration_falls_back_to_the_constant_without_cuda(self, monkeypatch):
        """A pre-flight that dies because its own instrument failed is worse than
        one that falls back. Pinned by making the measurement return None."""
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream, "measure_logits_loss_bytes_per_element", lambda **kw: None
        )
        assert layer_stream.calibrated_logits_bytes_per_element() == 14.0

    def test_calibration_adds_the_retention_to_whatever_it_measured(self, monkeypatch):
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream, "measure_logits_loss_bytes_per_element", lambda **kw: 16.0
        )
        assert layer_stream.calibrated_logits_bytes_per_element() == 18.0

    def test_calibration_never_returns_below_the_shipped_constant(self, monkeypatch):
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(
            layer_stream, "measure_logits_loss_bytes_per_element", lambda **kw: 4.0
        )
        assert layer_stream.calibrated_logits_bytes_per_element() == 14.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(vocab_size=0),
            dict(tokens=(0, 16)),
            dict(tokens=(2048, 1024)),
            dict(tokens=(512, 512)),
        ],
    )
    def test_the_probe_refuses_a_degenerate_request(self, kwargs):
        """A zero or non-increasing token pair divides by zero and would report a
        slope of infinity or NaN — a budget, not an exception."""
        from soup_cli.utils.layer_stream import measure_logits_loss_bytes_per_element

        with pytest.raises(ValueError):
            measure_logits_loss_bytes_per_element(**kwargs)


class TestTheModuleStaysOnTheLightCliPath:
    def test_importing_layer_stream_does_not_import_torch(self):
        """The probe's torch import is inside the function. A lazy import that is
        called at module scope is not lazy, which this repo has shipped before."""
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; import soup_cli.utils.layer_stream as m; "
             "print('torch' in sys.modules)"],
            capture_output=True, text=True, check=True,
        )
        assert out.stdout.strip() == "False", out.stdout + out.stderr


@requires_cuda
class TestTheStackIsRemeasuredWhereverTheSuiteRuns:
    """The point of the calibration hook. v0.72.3's 14 was measured once on one
    box and frozen; these run the instrument against whatever stack is executing
    the suite, so a future torch or transformers whose loss path costs MORE fails
    loudly here instead of silently OOMing a user."""

    def test_the_measured_loss_path_does_not_exceed_the_shipped_budget(self):
        from soup_cli.utils.layer_stream import (
            LOGITS_BYTES_PER_ELEMENT,
            measure_logits_loss_bytes_per_element,
        )

        measured = measure_logits_loss_bytes_per_element()
        assert measured is not None
        assert measured + 2 <= LOGITS_BYTES_PER_ELEMENT, (
            f"this stack's loss path costs {measured} bytes per logit element; "
            f"the shipped budget of {LOGITS_BYTES_PER_ELEMENT} no longer covers it"
        )

    def test_the_measurement_is_the_documented_twelve(self):
        """Measured 12.000000, spread 0.00e+00, on torch 2.5.1 and 2.13.0 alike.
        A tolerance is allowed for allocator rounding at small probe sizes; the
        assertion that matters is the one above."""
        from soup_cli.utils.layer_stream import measure_logits_loss_bytes_per_element

        measured = measure_logits_loss_bytes_per_element()
        assert 11.5 <= measured <= 12.5, measured

    def test_the_measurement_is_reproducible(self):
        """Three repeats. ``max_memory_allocated`` is deterministic for a fixed
        shape, so a spread here means the probe is measuring something else."""
        from soup_cli.utils.layer_stream import measure_logits_loss_bytes_per_element

        runs = [measure_logits_loss_bytes_per_element() for _ in range(3)]
        assert max(runs) - min(runs) == 0.0, runs

    def test_calibration_agrees_with_the_shipped_constant_on_this_stack(self):
        from soup_cli.utils.layer_stream import calibrated_logits_bytes_per_element

        assert calibrated_logits_bytes_per_element() == 14.0

    def test_retaining_the_output_costs_exactly_the_retention_term(self):
        """The single-variable control behind ``LOGITS_RETENTION_BYTES_PER_ELEMENT``:
        hold the model output object across ``backward()`` instead of letting it
        die with the local, change nothing else, and the peak moves by exactly
        one bf16 copy of the logits. This is the whole of the #327 gap."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        from soup_cli.utils.layer_stream import LOGITS_RETENTION_BYTES_PER_ELEMENT

        vocab, tokens = 32768, 512
        cfg = AutoConfig.for_model(
            "llama", vocab_size=vocab, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            tie_word_embeddings=False,
        )
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).cuda()
        model.gradient_checkpointing_enable()
        model.train()
        ids = torch.randint(0, vocab, (1, tokens), device="cuda")

        def step(hold):
            def compute_loss():
                outputs = model(input_ids=ids, labels=ids)
                return (outputs.loss, outputs) if hold else (outputs.loss, None)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            loss, kept = compute_loss()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            del kept, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return peak

        step(False)  # warm-up: the first step on a fresh model pays allocator growth
        free = step(False)
        held = step(True)
        delta = (held - free) / (vocab * tokens)
        assert delta == pytest.approx(LOGITS_RETENTION_BYTES_PER_ELEMENT, abs=0.05), delta
