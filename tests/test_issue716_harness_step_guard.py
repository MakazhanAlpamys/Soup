"""#716: the MLX benchmark harness refuses a run whose step counts it did not pin.

The harness resolves iterations and optimizer updates from the LOADED config and
exits non-zero when either differs from ``rows * epochs``. These tests pin two
things the bridge suite only covers incidentally:

- ``resolve_step_counts`` agrees with what ``MLXSFTTrainerWrapper`` actually
  hands mlx-lm, so the abort message states numbers the wrapper really uses;
- the abort path fires through the real ``main()`` when a pin is removed, so
  deleting the guard (while keeping the pins) no longer goes unnoticed.
"""

from __future__ import annotations

import pytest

from tests import test_issue23_mlx_harness_bridge as _bridge
from tests.test_issue684_mlx_grad_accumulation import _install_fake_mlx, _mlx_wrapper

# The bridge suite's fixture: real wrapper, real Rich/SQLite bridge, fake MLX.
harness_run = _bridge.harness_run


def _load_harness():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "benchmarks" / "harness" / "mlx_sft_smoke.py"
    spec = importlib.util.spec_from_file_location("mlx_sft_smoke_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# (rows, epochs, batch_size, accum) -> (iterations, optimizer updates). The first
# four were read back from the real wrapper's TrainingArgs in #727's review.
_CASES = [
    ((48, 1, 1, 1), (48, 48)),
    ((48, 1, 1, 4), (48, 12)),  # nothing quartered: 48 iterations, 12 updates
    ((50, 1, 1, 4), (48, 12)),  # rounded down, two rows dropped
    ((10, 1, 1, 4), (8, 2)),
    ((10, 1, 4, 1), (3, 3)),  # ceil, not floor
    ((50, 2, 1, 4), (100, 25)),
]


@pytest.mark.parametrize(("inputs", "expected"), _CASES)
def test_resolve_step_counts_matches_the_recorded_wrapper_arithmetic(inputs, expected):
    assert _load_harness().resolve_step_counts(*inputs) == expected


@pytest.mark.parametrize(("inputs", "expected"), _CASES)
def test_resolve_step_counts_matches_what_the_wrapper_hands_mlx_lm(
    tmp_path, monkeypatch, inputs, expected,
):
    rows, epochs, batch_size, accum = inputs
    captured = _install_fake_mlx(monkeypatch)
    wrapper = _mlx_wrapper(
        tmp_path, train_row_count=rows, epochs=epochs, lr=1e-4,
        batch_size=batch_size, gradient_accumulation_steps=accum,
    )

    wrapper.train()

    args = captured[0]["args"]
    real = (args.iters, args.iters // args.grad_accumulation_steps)
    assert real == expected
    assert _load_harness().resolve_step_counts(*inputs) == real


def _mutate_harness_yaml(monkeypatch, old: str, new: str) -> None:
    import soup_cli.config.loader as loader

    original = loader.load_config_from_string

    def mutated(text, *args, **kwargs):
        assert text.count(old) == 1, f"harness YAML no longer contains {old!r}"
        return original(text.replace(old, new), *args, **kwargs)

    monkeypatch.setattr(loader, "load_config_from_string", mutated)


def _run_expecting_abort(state, monkeypatch, rows: int) -> str:
    import sys

    monkeypatch.setattr(sys, "argv", [sys.argv[0], state.harness.DEFAULT_MODEL, str(rows), "1"])
    with pytest.raises(SystemExit) as excinfo:
        state.harness.main()
    assert state.displays == [], "the guard must stop the run before training starts"
    message = excinfo.value.code
    assert isinstance(message, str), "sys.exit(str) exits 1 with the message on stderr"
    return message


def test_deleting_the_accumulation_pin_aborts_with_the_real_counts(harness_run, monkeypatch):
    _mutate_harness_yaml(monkeypatch, "  gradient_accumulation_steps: 1\n", "")

    message = _run_expecting_abort(harness_run, monkeypatch, rows=48)

    assert "to 48 iterations and 12 optimizer updates, dropping 0 row(s)" in message
    assert "gradient_accumulation_steps=4" in message


def test_the_coincidence_breaker_reports_the_dropped_rows(harness_run, monkeypatch):
    _mutate_harness_yaml(monkeypatch, "  gradient_accumulation_steps: 1\n", "")

    message = _run_expecting_abort(harness_run, monkeypatch, rows=50)

    assert "to 48 iterations and 12 optimizer updates, dropping 2 row(s)" in message


def test_moving_batch_size_aborts(harness_run, monkeypatch):
    _mutate_harness_yaml(monkeypatch, "  batch_size: 1\n", "  batch_size: 4\n")

    message = _run_expecting_abort(harness_run, monkeypatch, rows=10)

    assert "batch_size=4" in message
    assert "to 3 iterations and 3 optimizer updates" in message


def test_the_shipped_pins_pass_and_print_both_counts(harness_run, capsys):
    assert harness_run.harness.main() == 0

    assert "steps         : 10 iterations / 10 optimizer updates" in capsys.readouterr().out
