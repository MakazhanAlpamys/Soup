"""RED proof for #1407: the gap, stated through the pre-fix PUBLIC entry point.

Every name used here exists on ``main``. That is deliberate: a RED that fails on a
newly-added symbol proves only that the symbol is new, not that the estimate was
wrong. So this file monkeypatches the device probe with ``raising=False`` and
reads the environment variable by its literal name, which means against the
unfixed formula it reaches the assertion below and fails there -- naming the
under-prediction -- and against the fixed one it passes.

Checked by reverting only ``src/``::

    git checkout <base> -- src/soup_cli/
    PYTHONPATH=src pytest tests/test_issue1407_red_proof.py -q --no-cov
    # 3 failed -- all three on the assertion, none on an AttributeError
"""

from soup_cli.utils import layer_stream
from soup_cli.utils.layer_stream import estimate_stream_peak_vram

#: The 2-layer streamed fixture of ``tests/test_v07204.py`` at 2 rows x seq 64,
#: built through the real ``setup()``.
_FIXTURE = dict(
    layer_bytes=14_160_384 // 2,
    buffers=2,
    extras_bytes=0,
    adapter_params=0,
    vocab_size=64,
    hidden_size=64,
    intermediate_size=160,
    n_layers=2,
    seq_len=64,
    batch_size=2,
)

#: Compute capability major of the RTX 5070 Laptop the peaks below were measured
#: on. Injected because the charge is a function of the device and CI has no GPU:
#: a budget that cannot name its device cannot size its workspace.
CC_BLACKWELL = 12

#: Measured on that card (8151 MiB, driver 616.92, Windows 11, torch 2.14.0+cu130,
#: transformers 5.17.0, trl 0.29.1), one DPO step, forward and backward.
MEASURED_PEAK = 67_734_016

#: The same step with ``CUBLAS_WORKSPACE_CONFIG=:4096:1``, shrinking each
#: workspace from 32 MiB to 4 MiB.
MEASURED_PEAK_4MIB = 9_013_760

#: Difference between the two readings: 58,720,256 B = 2 x (32 - 4) MiB. This is
#: what the term is derived from, and on the reporter's card it was the whole miss.
MEASURED_WORKSPACE_DELTA = 58_720_256

#: The variable PyTorch reads, by name. Read literally rather than through a
#: constant so this file imports nothing the unfixed module lacks.
ENV_NAME = "CUBLAS_WORKSPACE_CONFIG"


def _pretend_blackwell(monkeypatch):
    """Make the formula believe the process is on an RTX 5070.

    ``raising=False`` because the probe itself is part of the fix: against the
    unfixed module there is nothing to override, and the prediction stays at
    whatever the formula always said -- which is the point.
    """
    monkeypatch.delenv(ENV_NAME, raising=False)
    monkeypatch.setattr(
        layer_stream, "_device_compute_capability_major", lambda device: CC_BLACKWELL, raising=False
    )


def test_the_pre_flight_does_not_under_predict_the_reported_peak(monkeypatch):
    _pretend_blackwell(monkeypatch)
    predicted = estimate_stream_peak_vram(**_FIXTURE)
    assert predicted >= MEASURED_PEAK, (
        f"predicted {predicted} is below the measured {MEASURED_PEAK} peak "
        f"({predicted / MEASURED_PEAK:.2f}x) -- the formula charges nothing for "
        f"the two cuBLAS workspaces a streamed step holds"
    )


def test_the_charge_tracks_the_workspace_the_operator_asks_for(monkeypatch):
    """PyTorch reads ``CUBLAS_WORKSPACE_CONFIG`` once per process for the same
    allocation, so the estimate must move by exactly what the allocator moved:
    58,720,256 B, i.e. 2 x (32 - 4) MiB."""
    _pretend_blackwell(monkeypatch)
    default = estimate_stream_peak_vram(**_FIXTURE)

    monkeypatch.setenv(ENV_NAME, ":4096:1")
    shrunk = estimate_stream_peak_vram(**_FIXTURE)

    assert default - shrunk == MEASURED_WORKSPACE_DELTA, (
        f"the prediction moved {default - shrunk} bytes when the workspace went "
        f"from 32 MiB to 4 MiB; the allocator moved {MEASURED_WORKSPACE_DELTA}"
    )


def test_it_still_covers_the_step_with_the_workspace_removed(monkeypatch):
    """``CUBLAS_WORKSPACE_CONFIG=:0:0`` removes the workspaces outright. The fix must
    not depend on them being large."""
    _pretend_blackwell(monkeypatch)
    monkeypatch.setenv(ENV_NAME, ":0:0")
    assert estimate_stream_peak_vram(**_FIXTURE) >= MEASURED_PEAK_4MIB
