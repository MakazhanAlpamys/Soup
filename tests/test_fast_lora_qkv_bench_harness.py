"""benchmarks/harness/fast_lora_qkv_bench.py, the #838 Q/K/V timing harness, without a GPU.

The measurement needs a CUDA card. What can be pinned here: the self-test (the protocol at
tiny shapes on the CPU, reading no figure), the path proof and the agreement check that
void a process, the alternated arm order, the verdict rule, the exit status and the
arguments the harness refuses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

_HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "fast_lora_qkv_bench.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("fast_lora_qkv_bench", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()
TINY = harness.Shapes(hidden=32, q_out=32, kv_out=16, tokens=8)


def _proc(fast, plain, **extra):
    return harness.ProcessResult(cell="x", fast_ms=fast, plain_ms=plain, **extra)


def test_the_self_test_passes(capsys):
    assert harness.run_self_test() == harness.EXIT_OK

    out = capsys.readouterr().out
    assert "0 failed" in out  # ansi-ok: print() output, no Rich console
    assert "FAIL" not in out


def test_the_arms_alternate_and_run_equally_often():
    order = harness.schedule(4)

    assert order == ["fast", "plain", "plain", "fast", "fast", "plain", "plain", "fast"]
    assert order.count("fast") == order.count("plain") == 4


@pytest.mark.parametrize("targets", ["qkv", "qv"])
def test_a_cell_takes_the_fast_path_on_every_iteration(targets):
    result = harness.measure(
        harness.Cell(2, "bf16", targets), TINY, "cpu", warmup=1, iters=3, tolerance=2e-2
    )

    assert result.void is None and not result.oom
    assert result.patched == 1
    assert result.fast_path_ok == result.plain_path_ok == 3
    assert result.agreement <= 2e-2
    assert result.order == harness.schedule(3)


def test_k_proj_is_base_only_in_a_qv_cell():
    model, attn, _, _ = harness.build(harness.Cell(2, "bf16", "qv"), TINY, "cpu", 0)

    assert not hasattr(attn.k_proj, "lora_A")
    assert hasattr(attn.q_proj, "lora_A") and hasattr(attn.v_proj, "lora_A")
    assert all(p.requires_grad == ("lora_" in n) for n, p in model.named_parameters())


def test_every_lora_b_is_non_zero_so_every_gradient_is():
    model, _, _, _ = harness.build(harness.Cell(2, "bf16", "qkv"), TINY, "cpu", 0)

    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert len(lora_b) == 3 and all(p.abs().sum() > 0 for p in lora_b)


def test_a_patch_that_applies_nothing_voids_the_process():
    result = harness.measure(
        harness.Cell(2, "bf16", "qkv"),
        TINY,
        "cpu",
        warmup=1,
        iters=2,
        tolerance=2e-2,
        patch=lambda model: 0,
    )

    assert result.void == "patch_fast_lora_qkv returned 0, not 1"
    assert result.fast_ms is None


def test_a_patch_count_that_lies_voids_the_process():
    """Patches nothing but reports 1: the arms agree, so only the path proof catches it."""
    result = harness.measure(
        harness.Cell(2, "bf16", "qkv"),
        TINY,
        "cpu",
        warmup=1,
        iters=2,
        tolerance=2e-2,
        patch=lambda model: 1,
    )

    assert result.agreement == 0.0
    assert result.void is not None and "path proof" in result.void
    assert result.fast_path_ok == 0 and result.plain_path_ok == 2


def test_a_patched_module_in_the_plain_arm_voids_the_process(monkeypatch):
    """The plain arm must run unpatched: an unpatch that does nothing is caught."""
    monkeypatch.setattr("soup_cli.utils.fast_lora_qkv.unpatch_fast_lora_qkv", lambda model: None)
    result = harness.measure(
        harness.Cell(2, "bf16", "qkv"), TINY, "cpu", warmup=1, iters=2, tolerance=2e-2
    )

    assert result.void is not None and "path proof" in result.void
    assert result.plain_path_ok == 0


def test_arms_that_disagree_before_timing_void_the_process(monkeypatch):
    monkeypatch.setattr(harness, "relative_error", lambda actual, expected: 0.5)

    result = harness.measure(
        harness.Cell(2, "bf16", "qkv"), TINY, "cpu", warmup=1, iters=2, tolerance=2e-2
    )

    assert result.void is not None and "disagree" in result.void
    assert result.fast_ms is None


@pytest.mark.parametrize(
    ("results", "wanted"),
    [
        ([_proc(9.0, 10.0), _proc(9.1, 10.1), _proc(9.2, 10.0)], "faster"),
        ([_proc(11.0, 10.0), _proc(11.1, 10.1), _proc(11.2, 10.0)], "slower"),
        # Ratios on both sides of 1.00.
        ([_proc(9.0, 10.0), _proc(10.5, 10.0), _proc(9.5, 10.0)], "no measurable difference"),
        # One side, but the gap (0.05) is inside the fast arm's spread (0.95).
        ([_proc(9.9, 10.0), _proc(9.0, 10.0), _proc(9.95, 10.0)], "no measurable difference"),
        ([_proc(9.0, 10.0, void="path"), _proc(9.0, 10.0), _proc(9.0, 10.0)], "VOID"),
        ([_proc(9.0, 10.0), harness.ProcessResult(cell="x", oom=True)], "OOM"),
    ],
)
def test_the_verdict_rule(results, wanted):
    assert harness.cell_verdict(results) == wanted


def test_void_outranks_oom_in_the_exit_status():
    assert harness.exit_status(["faster", "no measurable difference"]) == harness.EXIT_OK
    assert harness.exit_status(["slower", "OOM"]) == harness.EXIT_OOM
    assert harness.exit_status(["OOM", "VOID"]) == harness.EXIT_VOID


def test_a_void_process_row_says_why():
    row = harness.process_row(2, harness.ProcessResult(cell="x", void="patch returned 0"))

    assert row.split() == ["2", "VOID:", "patch", "returned", "0"]


def test_a_cell_round_trips_through_its_child_argument():
    cell = harness.Cell(64, "nf4", "qv")

    assert harness.Cell.decode(cell.encode()) == cell
    assert cell.name == "r64 nf4 qv"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--ranks", "16,x"], "--ranks must be comma-separated integers"),
        (["--ranks", "0"], "every --ranks entry must be positive"),
        (["--dtypes", "fp8"], "--dtypes takes bf16, nf4"),
        (["--targets", "qk"], "--targets takes qkv, qv"),
        (["--iters", "0"], "must be positive"),
        (["--tolerance", "1.5"], "--tolerance must be between 0 and 1"),
        (["--cell", "16:bf16:qkv"], "--cell needs --cell-out"),
    ],
)
def test_bad_arguments_exit_2(argv, message, capsys):
    assert harness.main(argv) == harness.EXIT_USAGE
    assert message in capsys.readouterr().out


def test_no_cuda_exits_2_before_anything_runs(monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert harness.main([]) == harness.EXIT_USAGE
    assert "CUDA required" in capsys.readouterr().out
