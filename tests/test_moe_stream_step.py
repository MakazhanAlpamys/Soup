"""benchmarks/harness/moe_stream_step.py: the loss window and the per-step I/O counters.

``--loss-tokens N`` computes the loss on the last N positions only, so a long step does
not hold vocabulary-sized logits for every position (seq-scaling probe,
``benchmarks/probe-rtx5070-giant-moe-seq-scaling.md``). The per-step ``os_io`` record
is what that probe uses to show where the bytes came from.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

CUDA = pytest.mark.gpu(reason="NF4 dequantisation needs bitsandbytes on CUDA")


def _harness(name):
    harness = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location(name, harness / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(rows, seq):
    numpy = pytest.importorskip("numpy")
    return numpy.arange(rows * seq).reshape(rows, seq)


# -- --loss-tokens ----------------------------------------------------------


def test_loss_tokens_defaults_to_the_full_loss():
    step = _harness("moe_stream_step")
    args = step._args(["--store", "s", "--out", "o"])
    assert args.loss_tokens == 0
    ids = _ids(1, 8)
    labels, extra = step._loss_inputs(ids, args.loss_tokens)
    assert labels is ids
    assert extra == {}


@pytest.mark.parametrize("window", [8, 9, 4096])
def test_a_window_at_or_past_the_sequence_is_the_full_loss(window):
    step = _harness("moe_stream_step")
    ids = _ids(2, 8)
    labels, extra = step._loss_inputs(ids, window)
    assert labels is ids
    assert extra == {}


def test_a_short_window_keeps_the_last_positions_of_every_row():
    step = _harness("moe_stream_step")
    ids = _ids(2, 8)
    labels, extra = step._loss_inputs(ids, 3)
    assert labels.tolist() == [[5, 6, 7], [13, 14, 15]]
    assert extra == {"logits_to_keep": 3}


@pytest.mark.parametrize("value", ["-1", "1.5", "many"])
def test_loss_tokens_refuses_a_value_that_is_not_a_count(value, capsys):
    step = _harness("moe_stream_step")
    with pytest.raises(SystemExit):
        step._args(["--store", "s", "--out", "o", "--loss-tokens", value])
    assert "--loss-tokens must be a whole number >= 0" in capsys.readouterr().err


def test_forward_loss_passes_the_window_to_the_model():
    step = _harness("moe_stream_step")
    seen = {}

    def model(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(loss="L")

    ids = _ids(1, 8)
    loss, positions = step._forward_loss(model, ids, 2)
    assert loss == "L"
    assert positions == 2
    assert seen["input_ids"] is ids
    assert seen["labels"].tolist() == [[6, 7]]
    assert seen["logits_to_keep"] == 2

    seen.clear()
    loss, positions = step._forward_loss(model, ids, 0)
    assert positions == 8
    assert seen["labels"] is ids
    assert "logits_to_keep" not in seen


def test_the_window_is_the_next_token_loss_of_the_last_positions():
    """Through a real (tiny, CPU) DeepSeek-V3: the windowed loss is the cross entropy of
    the last N logits against the ids one position later, and the gradient still
    reaches a position outside the window."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    step = _harness("moe_stream_step")
    synth = _harness("synth_moe_store")
    config = transformers.DeepseekV3Config(**synth.config_dict(synth.SHAPES["tiny"]))
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).float().eval()
    seq, window = 24, 6
    ids = torch.arange(seq).reshape(1, seq) * 5 % config.vocab_size

    loss, positions = step._forward_loss(model, ids, window)
    assert positions == window
    assert math.isfinite(float(loss.detach()))
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, -window:, :]
    expected = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, seq - window + 1 :])
    assert float(loss.detach()) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    loss.backward()
    grad = model.get_input_embeddings().weight.grad
    first = int(ids[0, 0])
    assert first not in ids[0, seq - window :].tolist()
    assert float(grad[first].abs().sum()) > 0.0


# -- per-step I/O counters --------------------------------------------------


class _FakePsutil(types.ModuleType):
    class Error(Exception):
        pass

    def __init__(self, process_bytes, disks):
        super().__init__("psutil")
        self._process_bytes = process_bytes
        self._disks = disks

    def Process(self):  # noqa: N802 - psutil's name
        outer = self

        class _Process:
            def io_counters(self):
                return types.SimpleNamespace(read_bytes=outer._process_bytes)

        return _Process()

    def disk_io_counters(self, perdisk=False):
        assert perdisk is True
        return {
            name: types.SimpleNamespace(read_bytes=value) for name, value in self._disks.items()
        }


def test_io_snapshot_reads_the_process_and_every_drive(monkeypatch):
    step = _harness("moe_stream_step")
    fake = _FakePsutil(1_000, {"PhysicalDrive0": 10, "PhysicalDrive1": 20})
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert step._io_snapshot() == {
        "process_read_bytes": 1_000,
        "disk_read_bytes": {"PhysicalDrive0": 10, "PhysicalDrive1": 20},
    }


def test_io_snapshot_without_psutil_says_none(monkeypatch):
    step = _harness("moe_stream_step")
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert step._io_snapshot() == {"process_read_bytes": None, "disk_read_bytes": None}


def test_io_snapshot_survives_counters_the_platform_refuses(monkeypatch):
    step = _harness("moe_stream_step")
    fake = _FakePsutil(0, {})

    def refuse(perdisk=False):
        raise RuntimeError("couldn't find any physical disk")

    def no_process():
        raise fake.Error("access denied")

    fake.disk_io_counters = refuse
    fake.Process = no_process
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert step._io_snapshot() == {"process_read_bytes": None, "disk_read_bytes": None}


def test_io_delta_is_the_bytes_read_between_two_snapshots():
    step = _harness("moe_stream_step")
    before = {"process_read_bytes": 100, "disk_read_bytes": {"PhysicalDrive0": 5, "D1": 7}}
    after = {"process_read_bytes": 1_600, "disk_read_bytes": {"PhysicalDrive0": 5, "D1": 1_507}}
    assert step._io_delta(before, after) == {
        "process_read_bytes": 1_500,
        "disk_read_bytes": {"PhysicalDrive0": 0, "D1": 1_500},
    }


def test_io_delta_drops_a_drive_seen_only_once_and_keeps_none():
    step = _harness("moe_stream_step")
    before = {"process_read_bytes": None, "disk_read_bytes": {"A": 1}}
    after = {"process_read_bytes": 9, "disk_read_bytes": {"A": 4, "B": 50}}
    assert step._io_delta(before, after) == {
        "process_read_bytes": None,
        "disk_read_bytes": {"A": 3},
    }
    none = {"process_read_bytes": None, "disk_read_bytes": None}
    assert step._io_delta(none, after) == {"process_read_bytes": None, "disk_read_bytes": None}


# -- the wiring, end to end (GPU only) --------------------------------------


@CUDA
def test_one_windowed_step_records_the_window_and_the_io(tmp_path):
    pytest.importorskip("bitsandbytes")
    store = str(tmp_path / "store")
    _harness("synth_moe_store").write_store(store, "tiny", log=lambda _line: None)
    out = tmp_path / "step.json"
    assert (
        _harness("moe_stream_step").main(
            [
                "--store",
                store,
                "--out",
                str(out),
                "--seq",
                "16",
                "--steps",
                "1",
                "--no-pin",
                "--route",
                "uniform",
                "--loss-tokens",
                "4",
            ]
        )
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["args"]["loss_tokens"] == 4
    (entry,) = report["steps"]
    assert entry["loss_finite"] and entry["grads_finite"]
    assert entry["loss_positions"] == 4
    assert entry["os_io"]["process_read_bytes"] >= entry["expert_bytes_read"] > 0
