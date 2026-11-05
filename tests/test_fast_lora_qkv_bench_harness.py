"""benchmarks/harness/fast_lora_qkv_bench.py, the #838 Q/K/V timing harness, without a GPU.

The measurement needs a CUDA card. What can be pinned here: the self-test (the protocol at
tiny shapes on the CPU, reading no figure), the path proof and the agreement check that
void a process, the alternated arm order, the verdict rule, the exit status and the
arguments the harness refuses.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
        # Gap 1.0 beats the spread 0.6, but the ratios (0.90, 0.90, 1.01) sit on both sides.
        ([_proc(9.0, 10.0), _proc(9.0, 10.0), _proc(9.6, 9.5)], "no measurable difference"),
        # Every ratio above 1.00, but the gap 0.2 is inside the fast arm's spread 1.4.
        ([_proc(10.1, 10.0), _proc(11.5, 10.0), _proc(10.2, 10.0)], "no measurable difference"),
        # Gap 1.0 beats the fast arm's spread 0.1 but not the plain arm's 1.3.
        ([_proc(9.0, 10.0), _proc(9.1, 9.2), _proc(9.0, 10.5)], "no measurable difference"),
        (
            [_proc(9.0, 10.0), harness.ProcessResult(cell="x", crashed="exit 1"), _proc(9.0, 10.0)],
            "CRASHED",
        ),
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
    assert harness.exit_status(["VOID", "CRASHED", "OOM"]) == harness.EXIT_CRASHED


# --- the path proof, one half at a time -------------------------------------------


class _FastLoraQKVBackward:
    """Stands in for the fused backward: the proof reads only the class name."""


class _OtherBackward:
    pass


def _outputs(grad_fn_type):
    return (SimpleNamespace(grad_fn=grad_fn_type()),)


def _attn(hits=None, patched=False):
    attn = SimpleNamespace()
    if hits is not None:
        setattr(attn, harness._HITS_MARKER, (hits, ()))
    if patched:
        setattr(attn, harness._PATCH_MARKER, True)
    return attn


@pytest.mark.parametrize(
    ("hits", "grad_fn", "wanted"),
    [
        (2, _FastLoraQKVBackward, True),
        # q took the fused path, but K and V missed the cache and recomputed.
        (1, _FastLoraQKVBackward, False),
        (0, _FastLoraQKVBackward, False),
        # Both cache hits, but the output did not come from the fused backward.
        (2, _OtherBackward, False),
        (None, _OtherBackward, False),
    ],
)
def test_the_fast_path_needs_both_cache_hits_and_the_fused_backward(hits, grad_fn, wanted):
    assert harness.took_fast_path(_attn(hits=hits), _outputs(grad_fn)) is wanted


@pytest.mark.parametrize(
    ("patched", "grad_fn", "wanted"),
    [
        (False, _OtherBackward, True),
        # The patch marker is still on the module.
        (True, _OtherBackward, False),
        # No marker, but the output came from the fused backward.
        (False, _FastLoraQKVBackward, False),
    ],
)
def test_an_unpatched_run_needs_no_marker_and_no_fused_backward(patched, grad_fn, wanted):
    assert harness.ran_unpatched(_attn(patched=patched), _outputs(grad_fn)) is wanted


def test_one_fast_iteration_off_the_path_voids_the_process(monkeypatch):
    real = harness.took_fast_path
    calls = []

    def one_miss(attn, outputs):
        calls.append(None)
        # Call 1 is the warm-up's fast iteration; call 3 is the second timed one.
        return False if len(calls) == 3 else real(attn, outputs)

    monkeypatch.setattr(harness, "took_fast_path", one_miss)
    result = harness.measure(
        harness.Cell(2, "bf16", "qkv"), TINY, "cpu", warmup=1, iters=3, tolerance=2e-2
    )

    assert result.fast_path_ok == 2 and result.plain_path_ok == 3
    assert result.void is not None
    assert "2/3 fast iterations took the fast path" in result.void


def test_arms_that_disagree_only_on_a_lora_gradient_void_the_process(monkeypatch):
    """Outputs and dX agree; the LoRA gradients (their shapes carry the rank, 2) do not."""
    monkeypatch.setattr(
        harness,
        "relative_error",
        lambda actual, expected: 0.5 if 2 in tuple(actual.shape) else 0.0,
    )

    result = harness.measure(
        harness.Cell(2, "bf16", "qkv"), TINY, "cpu", warmup=1, iters=2, tolerance=2e-2
    )

    assert result.void is not None and "disagree" in result.void
    assert result.fast_ms is None


# --- a crashed process --------------------------------------------------------------


def _child_args():
    return argparse.Namespace(warmup=1, iters=2, tolerance=2e-2, child_timeout=60.0)


def _fake_child(monkeypatch, returncode, write_result):
    def run(argv, check, timeout):
        if write_result:
            out = Path(argv[argv.index("--cell-out") + 1])
            out.write_text(json.dumps({"cell": "2:bf16:qkv", "fast_ms": 1.0, "plain_ms": 2.0}))
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(harness.subprocess, "run", run)


@pytest.mark.parametrize(
    ("returncode", "write_result", "crashed"),
    [
        (1, False, "exit 1"),
        # A result file does not make a non-zero exit clean.
        (1, True, "exit 1"),
        (0, False, "exit 0 without a result"),
    ],
)
def test_a_child_that_fails_is_a_crashed_row(
    monkeypatch, tmp_path, returncode, write_result, crashed
):
    _fake_child(monkeypatch, returncode, write_result)

    result = harness.run_cell_process(
        harness.Cell(2, "bf16", "qkv"), TINY, _child_args(), "cpu", tmp_path, 1
    )

    assert result.crashed == crashed
    assert result.fast_ms is None
    assert harness.process_row(1, result).split()[:2] == ["1", "CRASHED:"]


def test_a_child_that_hangs_is_a_crashed_row(monkeypatch, tmp_path):
    def run(argv, check, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(harness.subprocess, "run", run)

    result = harness.run_cell_process(
        harness.Cell(2, "bf16", "qkv"), TINY, _child_args(), "cpu", tmp_path, 1
    )

    assert result.crashed == "no result after 60 s"


def test_the_sweep_goes_on_after_a_crashed_cell(monkeypatch, capsys):
    crashing = harness.Cell(16, "nf4", "qkv")
    healthy = harness.Cell(16, "bf16", "qkv")

    def run_cell_process(cell, shapes, args, device, work_dir, index):
        if cell == crashing:
            return harness.ProcessResult(cell=cell.encode(), crashed="exit 1")
        return _proc(
            9.0 + index / 100,
            10.0,
            patched=1,
            agreement=0.0,
            iterations=2,
            fast_path_ok=2,
            plain_path_ok=2,
        )

    monkeypatch.setattr(harness, "machine_lines", lambda: ["# machine"])
    monkeypatch.setattr(harness, "run_cell_process", run_cell_process)
    args = harness.parse_args(["--processes", "3"])

    status = harness.sweep([crashing, healthy], TINY, args)

    out = capsys.readouterr().out
    assert status == harness.EXIT_CRASHED
    assert out.count("CRASHED: exit 1") == 3  # ansi-ok: print() output, no Rich console
    assert "  verdict: CRASHED" in out
    assert "r16 bf16 qkv" in out and "  verdict: faster" in out


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
        (["--processes", "1"], "--processes must be at least 3"),
        (["--processes", "2"], "--processes must be at least 3"),
        (["--child-timeout", "0"], "--child-timeout must be positive"),
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
