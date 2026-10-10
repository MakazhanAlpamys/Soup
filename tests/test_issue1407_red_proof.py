"""RED proof for #1407: the gap, stated through the pre-fix PUBLIC entry point.

Every name used here exists on ``main``. That is deliberate: a RED that fails on a
newly-added symbol proves only that the symbol is new, not that the estimate was
wrong. So this file drives ``_stream_budget_lines`` -- the pre-flight that actually
refuses runs -- and monkeypatches the device probe with ``raising=False``, which
means against the unfixed module there is nothing to override and the prediction
stays at whatever the formula always said. It reaches the assertion and fails
there, naming the under-prediction.

Checked by reverting only ``src/``::

    git checkout <base> -- src/soup_cli/
    PYTHONPATH=src pytest tests/test_issue1407_red_proof.py -q --no-cov
    # 1 failed, 1 passed -- the failure is on the assertion, not an AttributeError

This is also the only file that covers the WIRING, and the wiring is where the
surviving mutations live: passing ``cublas_workspace_bytes=0`` at the call site, or
skipping the panel line, leaves every arithmetic test in the sibling file passing
(408 passed, 41 skipped with either applied). Both fail the first test here.

The arithmetic lives in ``tests/test_issue1407_cublas_workspace.py`` instead of
here, and that split is deliberate. Those tests name the charge explicitly, because
the parameter's default is ``0``: the pre-flight resolves the charge from the run's
own device and passes it, since a default that probed the live card would bill a
CPU-side caller on a machine that merely has a card visible
(``torch.cuda.get_device_properties(None)`` reads the CURRENT card). A test here
that passed the new keyword would fail on ``main`` with a ``TypeError`` rather than
an assertion, which is an environment error dressed up as a RED.
"""

from types import SimpleNamespace

from soup_cli.utils import layer_stream
from soup_cli.utils.layer_stream import STREAM_FIXED_SLACK_BYTES

GB, MIB = 1_000_000_000, 1024 * 1024

#: The 2-layer streamed fixture of ``tests/test_v07204.py`` at 2 rows x seq 64,
#: built through the real ``setup()``. The numbers are the issue's: this shape
#: predicted 13,934,816 B against a measured peak of 67,734,016 B on an RTX 5070
#: (cc 12.0), so the fixture reproduces the first number exactly.
LAYER_BYTES = 94_528
INTERMEDIATE_SIZE = 128

#: The variable PyTorch reads, by name. Read literally rather than through a
#: constant so this file imports nothing the unfixed module lacks.
ENV_NAME = "CUBLAS_WORKSPACE_CONFIG"


class _FakeCuda:
    """The two CUDA facts the CUDA branch of ``_stream_budget_lines`` reads."""

    @staticmethod
    def mem_get_info():
        return (8 * GB, 8 * GB)


def _fake_torch(monkeypatch):
    """A ``torch`` module carrying only what this path touches.

    The pre-flight's CUDA branch is worth driving without the ML stack: torch is a
    multi-gigabyte install, and this test needs exactly one call from it. Injecting
    the module keeps the wiring proof runnable in the minimal dev environment (and
    on any CPU-only checkout), instead of being skipped for a dependency whose only
    role here is to report free VRAM.
    """
    import sys
    from types import ModuleType

    stub = ModuleType("torch")
    stub.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", stub)


def _budget(monkeypatch, *, on_cuda):
    """Drive the real ``_stream_budget_lines`` and return its panel and prediction.

    torch is stubbed only for the CUDA path, which is the only one that reaches it:
    ``_stream_budget_lines`` returns before ``import torch`` when ``on_cuda`` is
    False. The off-CUDA control test therefore needs no stub at all.
    """
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    monkeypatch.delenv(ENV_NAME, raising=False)
    monkeypatch.setattr(  # a compute-capability-12 card, as in the issue
        layer_stream, "_device_compute_capability_major", lambda device=None: 12, raising=False
    )
    monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_gemm_tflops", lambda *_a, **_k: None
    )
    if on_cuda:
        _fake_torch(monkeypatch)
    seen = []
    real = layer_stream.estimate_stream_peak_vram

    def spy(**kwargs):
        seen.append(real(**kwargs))
        return seen[-1]

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", spy)
    tcfg = SimpleNamespace(
        batch_size=1,
        stream_buffers=2,
        stream_vram_probe=False,
        stream_vram_override=None,
        gradient_accumulation_steps=1,
        lora=SimpleNamespace(r=8, target_modules=["q_proj", "v_proj"]),
    )
    mixin = StreamingSetupMixin()
    if on_cuda:  # off CUDA there is no device attribute, as the #348 harness drives it
        mixin.device = "cuda"
    lines, _plan = mixin._stream_budget_lines(
        SimpleNamespace(data=SimpleNamespace(max_length=64)),
        tcfg,
        model_config=SimpleNamespace(
            vocab_size=64, hidden_size=64, intermediate_size=INTERMEDIATE_SIZE
        ),
        layer_bytes=LAYER_BYTES,
        embed_bytes=0,
        index=SimpleNamespace(n_layers=2, total_params=0),
        on_cuda=on_cuda,
    )
    return lines, seen[-1]


def test_a_cuda_run_charges_both_workspaces_and_says_so(monkeypatch):
    """The pre-flight, not the formula: a charge that never reaches the caller
    refuses no run. On ``main`` this predicts 13,934,816 B against a real peak of
    67,734,016 B and fails here -- and it fails under either wiring mutation the
    sibling file's arithmetic cannot see (a ``0`` at the call site, or no panel
    line)."""
    lines, predicted = _budget(monkeypatch, on_cuda=True)
    assert predicted >= STREAM_FIXED_SLACK_BYTES + 64 * MIB, predicted
    assert any("cuBLAS" in line for line in lines), lines


def test_off_cuda_nothing_is_charged_even_with_a_card_visible(monkeypatch):
    """CONTROL for the test above, and the reason the pre-flight gates on ``on_cuda``
    rather than on the probe. A CPU run must not be billed for a workspace it never
    allocates, even on a machine that has a card: ``get_device_properties(None)``
    reads the CURRENT card, so a pre-flight that charged on the probe alone would
    refuse CPU runs over GPU memory they never touch. Passes on ``main`` too -- it
    is the assertion that stops the fix from over-charging."""
    lines, predicted = _budget(monkeypatch, on_cuda=False)
    assert predicted < STREAM_FIXED_SLACK_BYTES + 64 * MIB, predicted
    assert not any("cuBLAS" in line for line in lines), lines
