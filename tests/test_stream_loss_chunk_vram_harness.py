"""benchmarks/harness/stream_loss_chunk_vram.py, the #1693 peak-VRAM harness, without a GPU.

The measurement needs a CUDA card. What can be pinned here: the self-test (the real
setup and one step per arm on a tiny local Llama, no peak read), one arm run in a
process of its own, the configs of both arms, the arguments the harness refuses, and
the verdicts and exit status a row gets.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

_HARNESS = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "stream_loss_chunk_vram.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("stream_loss_chunk_vram", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


@pytest.fixture(autouse=True)
def restore_alloc_conf(monkeypatch):
    """An arm sets PYTORCH_CUDA_ALLOC_CONF for real on Linux; put it back after each test."""
    import os

    before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    yield
    if before is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = before


def _arm(**fields):
    values = dict(
        seq=1024,
        chunk=128,
        realised=1024,
        predicted=3_500_000_000,
        preflight={},
        free=4_000_000_000,
        peak=3_463_000_000,
        oom=False,
        hooked=True,
        loss=1.0,
    )
    values.update(fields)
    return harness.Arm(**values)


def test_the_self_test_passes(capsys):
    assert harness.run_self_test() == harness.EXIT_OK

    out = capsys.readouterr().out
    assert "self-test: 13 passed, 0 failed" in out  # ansi-ok: print() output, no Rich console
    assert "FAIL" not in out


def test_each_arm_is_torn_down(monkeypatch, capsys):
    """On a card a hook or streamed runtime left behind would sit in the next arm's peak."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    calls = {"hook": 0, "runtime": 0}
    remove_hook = SFTTrainerWrapper._remove_stream_loss_chunk_hook
    close_runtime = SFTTrainerWrapper._close_stream_runtime

    def counting_remove_hook(self):
        calls["hook"] += 1
        return remove_hook(self)

    def counting_close_runtime(self):
        calls["runtime"] += 1
        return close_runtime(self)

    monkeypatch.setattr(SFTTrainerWrapper, "_remove_stream_loss_chunk_hook", counting_remove_hook)
    monkeypatch.setattr(SFTTrainerWrapper, "_close_stream_runtime", counting_close_runtime)

    assert harness.run_self_test() == harness.EXIT_OK
    capsys.readouterr()
    assert calls == {"hook": 2, "runtime": 2}, "both teardowns, once per arm"


@pytest.mark.parametrize("chunk", [None, 128])
@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_both_arms_are_configs_that_load(tmp_path, chunk, quantization):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    doc = harness.config_doc("test/model", 1024, chunk, quantization, tmp_path)
    cfg = load_config_from_string(yaml.safe_dump(doc))

    assert cfg.task == "sft"
    assert cfg.training.stream_layers is True
    assert cfg.training.stream_loss_chunk == chunk
    assert cfg.data.max_length == 1024 and cfg.training.quantization == quantization
    assert cfg.training.batch_size == 1 and cfg.training.stream_buffers == 2
    assert cfg.training.rewind_log is True  # the default trainer, as a user runs it


def test_the_rows_overshoot_the_length_they_are_cut_to():
    rows = harness.rows(512)

    assert len(rows) == harness.ROWS_PER_ARM
    assert len(rows[0]["messages"][1]["content"].split()) == harness.OVERSHOOT * 512


def test_a_row_at_or_above_its_peak_is_ok():
    row = harness.table_row(_arm())

    assert "1.0107" in row and row.endswith("| ok")
    assert harness.exit_status([_arm()]) == harness.EXIT_OK


def test_a_row_below_its_peak_under_predicts_and_exits_1():
    short = _arm(predicted=3_400_000_000)

    assert "0.9818" in harness.table_row(short)
    assert harness.verdict(short) == harness.VERDICT_UNDER
    assert harness.exit_status([_arm(), short]) == harness.EXIT_UNDER


def test_an_out_of_memory_arm_the_preflight_would_refuse_is_a_row_not_a_failure():
    refused = _arm(
        chunk=None, seq=2048, realised=2048, predicted=6_000_000_000, peak=None, oom=True
    )

    assert harness.table_row(refused).split() == [
        "2048", "2048", "off", "|", "6.000GB", "4.000GB", "OOM", "-", "|", "OOM,", "refused",
    ]
    assert harness.exit_status([refused]) == harness.EXIT_OK


def test_an_out_of_memory_arm_inside_the_predicted_fit_under_predicts():
    inside = _arm(predicted=3_900_000_000, peak=None, oom=True)

    assert harness.verdict(inside) == harness.VERDICT_UNDER
    assert harness.exit_status([inside]) == harness.EXIT_UNDER


def test_a_row_at_another_length_is_void_and_outranks_an_under_prediction():
    void = _arm(realised=1000)

    assert harness.verdict(void) == harness.VERDICT_VOID
    assert harness.exit_status([void, _arm(predicted=1)]) == harness.EXIT_VOID


def test_the_free_vram_recorded_is_the_measured_reading_not_the_override():
    from soup_cli.utils import layer_stream

    with harness.recording_preflight() as seen:
        # What the pre-flight calls on CUDA, with the harness's override set.
        available = layer_stream.resolve_available_vram_bytes(
            measured_bytes=4_000_000_000, override_bytes=harness.ASSUME_FREE_BYTES
        )

    assert available == harness.ASSUME_FREE_BYTES  # the pre-flight itself is unchanged
    assert seen["free"] == 4_000_000_000


def test_the_preflight_functions_are_put_back_when_the_setup_raises():
    from soup_cli.utils import layer_stream

    estimate = layer_stream.estimate_stream_peak_vram
    resolve = layer_stream.resolve_available_vram_bytes
    with pytest.raises(RuntimeError):
        with harness.recording_preflight():
            assert layer_stream.estimate_stream_peak_vram is not estimate
            raise RuntimeError("setup failed")

    assert layer_stream.estimate_stream_peak_vram is estimate
    assert layer_stream.resolve_available_vram_bytes is resolve


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "--base is required"),
        (["--base", "m", "--chunk", "15"], "between 16 and 8192"),
        (["--base", "m", "--chunk", "8193"], "between 16 and 8192"),
        (["--base", "m", "--seqs", "512,x"], "comma-separated integers"),
        (["--base", "m", "--seqs", "512,0"], "must be positive"),
        (["--base", "m", "--chunk", "16"], "CUDA required"),
        (["--base", "m", "--arm", "512:x", "--arm-out", "r.json"], "--arm must be"),
        (["--base", "m", "--arm", "512", "--arm-out", "r.json"], "--arm must be"),
        (["--base", "m", "--arm", "512:off"], "--arm needs --arm-out"),
    ],
)
def test_refused_arguments_exit_2_before_anything_is_built(monkeypatch, capsys, argv, message):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(harness, "sweep", lambda *args: pytest.fail("sweep must not run"))

    assert harness.main(argv) == harness.EXIT_USAGE
    assert message in capsys.readouterr().out


def _sweep_with_fake_processes(monkeypatch, returncode=0, write_result=True):
    """Run ``sweep`` with each arm's process replaced by a call to ``main`` in this one.

    The child side is the harness's own ``main``, so the argv and the result file are
    the ones a real child gets; only ``run_one`` is faked, and only inside a "child".
    """
    import torch

    def no_cuda_here():
        pytest.fail("the sweep's own process must not initialise CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda_here)
    # The fake children run in this process: keep them off a card if one is visible.
    monkeypatch.setattr(harness, "trainer_device", lambda: "cpu")
    state = {"in_child": False, "launched": [], "ran": [], "events": []}

    def fake_run_one(base, seq, chunk, quantization, device, work_dir, measure=True):
        if not state["in_child"]:
            pytest.fail("no arm may run in the sweep's own process")
        state["ran"].append((base, seq, chunk, quantization))
        return _arm(seq=seq, chunk=chunk, realised=seq)

    flush = sys.stdout.flush
    monkeypatch.setattr(sys.stdout, "flush", lambda: (state["events"].append("flush"), flush()))

    def fake_run(argv, **kwargs):
        state["events"].append("run")
        state["launched"].append(argv)
        if not write_result:
            return subprocess.CompletedProcess(argv, returncode)
        state["in_child"] = True
        try:
            child_status = harness.main(argv[2:])
        finally:
            state["in_child"] = False
        return subprocess.CompletedProcess(argv, returncode or child_status)

    monkeypatch.setattr(harness, "run_one", fake_run_one)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return state


def test_each_measured_arm_runs_in_a_process_of_its_own(monkeypatch, capsys):
    """In one process only the first arm got the RAM tier; the rest fell back to disk (#1693)."""
    state = _sweep_with_fake_processes(monkeypatch)

    assert harness.sweep("test/model", [512, 1024], 128, "4bit") == harness.EXIT_OK

    assert [argv[:2] for argv in state["launched"]] == [[sys.executable, str(_HARNESS)]] * 4
    # Every flag the sweep was given reaches every arm: a dropped --quantization would
    # measure the child's default and print a table that looks normal.
    assert state["ran"] == [
        ("test/model", 512, None, "4bit"),
        ("test/model", 512, 128, "4bit"),
        ("test/model", 1024, None, "4bit"),
        ("test/model", 1024, 128, "4bit"),
    ]
    lines = capsys.readouterr().out.splitlines()
    rows = [i for i, line in enumerate(lines) if line.endswith("| ok")]
    assert [lines[i].split()[:3] for i in rows] == [
        ["512", "512", "off"],
        ["512", "512", "128"],
        ["1024", "1024", "off"],
        ["1024", "1024", "128"],
    ]
    # One header, before the first row; one device line, from the first arm's result.
    assert [i for i, line in enumerate(lines) if line == harness.HEADER] == [rows[0] - 1]
    assert [line for line in lines if line.startswith("# device ")] == ["# device cpu"]


def test_the_sweep_flushes_its_output_before_each_arm_starts(monkeypatch, capsys):
    """A child writes straight to the same stdout, so unflushed lines would follow it."""
    state = _sweep_with_fake_processes(monkeypatch)

    harness.sweep("test/model", [512], 128, "none")

    runs = [i for i, event in enumerate(state["events"]) if event == "run"]
    assert len(runs) == 2 and all(i > 0 and state["events"][i - 1] == "flush" for i in runs)


@pytest.mark.parametrize(("returncode", "write_result"), [(1, True), (0, False)])
def test_an_arm_process_that_fails_stops_the_sweep(monkeypatch, capsys, returncode, write_result):
    _sweep_with_fake_processes(monkeypatch, returncode=returncode, write_result=write_result)

    with pytest.raises(
        RuntimeError, match=f"seq 512, chunk off: the arm's process exited {returncode}"
    ):
        harness.sweep("test/model", [512], 128, "none")


def test_an_arm_runs_end_to_end_in_a_real_child_process(tmp_path, monkeypatch):
    base = harness._tiny_llama(tmp_path / "model")
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")  # no peak read, card or not

    seq = harness._SELF_TEST_SEQ
    arm, device = harness.run_arm_process(base, seq, 16, "none", tmp_path / "arm")

    assert arm.seq == arm.realised == seq
    assert arm.chunk == 16 and arm.preflight["loss_chunk"] == 16 and arm.hooked
    assert arm.predicted > 0 and arm.peak is None and not arm.oom  # no peak off CUDA
    assert device in ("cpu", "mps")


def test_an_arm_process_reads_the_device_only_after_its_setup(monkeypatch, tmp_path):
    """Soup sets the allocator hint during setup, and only if nothing has initialised CUDA."""
    import types

    import torch

    order = []

    def fake_run_one(base, seq, chunk, quantization, device, work_dir, measure=True):
        order.append(("arm", device))
        return _arm(seq=seq, chunk=chunk, realised=seq)

    def fake_properties(index):
        order.append(("device", index))
        return types.SimpleNamespace(name="Fake GPU", total_memory=6_000_000_000)

    monkeypatch.setattr(harness, "trainer_device", lambda: "cuda")
    monkeypatch.setattr(harness, "run_one", fake_run_one)
    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_properties)
    out = tmp_path / "arm.json"

    assert harness.main(["--base", "m", "--arm", "512:off", "--arm-out", str(out)]) == 0

    assert order == [("arm", "cuda"), ("device", 0)]
    assert '"device": "Fake GPU | 6.000 GB"' in out.read_text(encoding="utf-8")


def test_an_arm_process_sets_the_allocator_hint_as_soup_train_does(monkeypatch, tmp_path):
    """``soup train`` sets PYTORCH_CUDA_ALLOC_CONF before CUDA starts (#2149); so must an arm."""
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    seen = []

    def fake_run_one(base, seq, chunk, quantization, device, work_dir, measure=True):
        seen.append(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        return _arm(seq=seq, chunk=chunk, realised=seq)

    monkeypatch.setattr(harness, "trainer_device", lambda: "cpu")
    monkeypatch.setattr(harness, "run_one", fake_run_one)
    out = tmp_path / "arm.json"

    assert harness.main(["--base", "m", "--arm", "512:off", "--arm-out", str(out)]) == 0
    assert seen == ["expandable_segments:True"]
