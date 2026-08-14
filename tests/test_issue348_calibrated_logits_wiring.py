"""#348 — ``calibrated_logits_bytes_per_element()`` has no caller.

#327 added the calibration hook (and the probe underneath it) as the guard against
a stack whose loss path grows a fourth fp32 buffer, which would otherwise
under-budget the VRAM pre-flight by 12.5% with nothing to catch it. But
``_stream_budget_lines`` called ``estimate_stream_peak_vram`` without
``logits_bytes_per_element=``, so the parameter was always ``None`` and the
calibration was exercised only by ``tests/test_issue327_logits_estimate.py``, never
by a real run. These tests assert the wiring, not the arithmetic (already pinned in
the #327 suite): the parameter must actually be reachable from the pre-flight.
"""

from soup_cli.trainer.stream_setup import StreamingSetupMixin


class _Index:
    n_layers = 4


class _ModelConfig:
    # A large vocab so a 14 -> 40 bytes-per-element swing moves the displayed
    # GB figure at 2-decimal precision, not just the underlying byte count.
    vocab_size = 2_000_000
    hidden_size = 8
    intermediate_size = 16
    num_hidden_layers = 4


class _Lora:
    r = 8
    target_modules = ["q_proj", "v_proj"]


class _TrainConfig:
    batch_size = 1
    stream_buffers = 2
    # v0.73.1 (#349): `_stream_budget_lines` reads this on every path, including
    # the `on_cuda=False` one this file exercises. Left off, so this file keeps
    # asserting the #348 calibration wiring and nothing else.
    stream_vram_probe = False
    lora = _Lora()


class _Data:
    max_length = 4


class _Config:
    data = _Data()


def _budget_lines():
    """``on_cuda=False`` exercises the panel without needing a GPU: the fit
    decision is skipped, but the calibrated budget and its display line are
    built either way (a CPU dry run reports the same peak VRAM figure a CUDA
    run would refuse or accept against).

    v0.73.1 (#349): ``_stream_budget_lines`` now returns ``(lines, probe_plan)``.
    Only the lines are this file's subject, so the plan is dropped HERE rather
    than at every call site — indexing ``[0]`` at the call sites would silently
    start meaning "the whole lines tuple" instead of "the first line" and the
    assertions would go on passing against the wrong object.
    """
    lines, _probe_plan = StreamingSetupMixin()._stream_budget_lines(
        _Config(),
        _TrainConfig(),
        model_config=_ModelConfig(),
        layer_bytes=1000,
        embed_bytes=0,
        index=_Index(),
        on_cuda=False,
    )
    return lines


def _peak_gb(line: str) -> str:
    """Isolate the peak-VRAM figure from the trailing ``(logits ... GB)`` aside,
    so a change confined to the logits display cannot satisfy an assertion about
    the peak (the budget that actually refuses a run)."""
    return line.split("~", 1)[1].split(" GB", 1)[0]


class TestCalibrationIsForwardedToTheBudget:
    def test_a_high_calibration_raises_the_predicted_peak(self, monkeypatch):
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
        baseline = _budget_lines()[0]

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 40.0)
        raised = _budget_lines()[0]

        assert _peak_gb(baseline) != _peak_gb(raised)

    def test_a_high_calibration_also_raises_the_displayed_logits_figure(self, monkeypatch):
        """The panel's logits figure is a separate claim from the peak above,
        and moving together is not itself proof either one is wired correctly."""
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
        baseline = _budget_lines()[0]

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 40.0)
        raised = _budget_lines()[0]

        assert baseline.split("(logits ", 1)[1] != raised.split("(logits ", 1)[1]

    def test_a_below_constant_calibration_does_not_lower_the_peak(self, monkeypatch):
        """calibrated_logits_bytes_per_element() itself floors at the shipped
        constant, so this also guards a future caller that bypasses the floor."""
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
        at_constant = _budget_lines()

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 4.0)
        below = _budget_lines()

        assert at_constant == below

    def test_a_none_probe_leaves_the_peak_byte_identical_to_today(self, monkeypatch):
        """No CUDA: measure_logits_loss_bytes_per_element() returns None and
        calibrated_logits_bytes_per_element() falls back to LOGITS_BYTES_PER_ELEMENT,
        so a machine without a GPU sees exactly today's number."""
        from soup_cli.utils import layer_stream
        from soup_cli.utils.layer_stream import estimate_stream_peak_vram

        monkeypatch.setattr(
            layer_stream, "measure_logits_loss_bytes_per_element", lambda **kw: None
        )
        (line,) = _budget_lines()

        direct = estimate_stream_peak_vram(
            layer_bytes=1000,
            buffers=2,
            extras_bytes=0,
            adapter_params=4 * 2 * 2 * 8 * 8,
            vocab_size=2_000_000,
            hidden_size=8,
            intermediate_size=16,
            n_layers=4,
            seq_len=4,
            batch_size=1,
        )
        assert f"peak VRAM    ~{direct / 1e9:.2f} GB" in line


class TestThePanelReportsWhenCalibrationDiverges:
    def test_no_extra_line_at_the_shipped_constant(self, monkeypatch):
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
        assert len(_budget_lines()) == 1

    def test_an_extra_line_names_both_numbers_above_the_constant(self, monkeypatch):
        from soup_cli.utils import layer_stream

        monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 18.0)
        lines = _budget_lines()
        assert len(lines) == 2
        assert "18.000" in lines[1]
        assert "14" in lines[1]
