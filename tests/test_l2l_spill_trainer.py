"""The activation spill through the real HF trainer loop (CPU).

Spill is forced only by the real budget path: ``layer_stream.free_ram_bytes`` is
monkeypatched and the cgroup roots point at an empty tree. The fixture is deeper than
``tests/test_l2l_probe_harness.py::build_streamed``: with ring 4 the rings are 8 buffers, so
spill pays only when ``n_layers x k x chunk > 8 x (slot + 4096)``. Eight layers of hidden 64
and 64-token rows give a 16 KiB float32 chunk, where it does, without unused capacity.
"""

from __future__ import annotations

import errno
import io
import os
import time
from typing import Any, List, NamedTuple

import pytest

from tests.test_l2l_probe_harness import SEQ, _load
from tests.test_l2l_trainer_labels import _MaskedRows

N_LAYERS = 8
#: Rows long enough for spill to save RAM after charging the rings.
SPILL_SEQ = 64
#: Free RAM that leaves a budget of 240,000 bytes (below 1.79 GB the budget is what the 512 MiB
#: reserve leaves). A chunk is 64 x 64 x 4 = 16 KiB and the rings 8 x 20 KiB = 160 KiB, so 6
#: of 8 layers spill at k=2 (all resident: 256 KiB) and 7 at k=4 (all resident: 512 KiB).
SMALL = (512 << 20) + 240_000
PLENTY = 10**12
PLAN_KEYS = {
    "spill_layers", "n_layers", "k", "chunk_bytes", "mb_bytes", "slot_bytes", "ring", "pin",
    "host_bytes",
    "all_resident_bytes", "budget_bytes", "available_bytes", "limiting_level",
    "limiting_numbers", "spill_dir", "spill_path", "spill_file_bytes", "explicit_dir",
}


def build_deep(tmp_path, n_layers=N_LAYERS):
    """``build_streamed`` with more layers; the caller closes ``runtime``."""
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07204 import _tiny_llama_dir

    weights, _model, _config = _tiny_llama_dir(tmp_path, n_layers=n_layers, tie=False)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM,
    )
    return build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora, device="cpu",
        dtype="float32", buffers=2, pin=False, seed=3,
    )


@pytest.fixture()
def deep(tmp_path):
    model, runtime = build_deep(tmp_path)
    yield model, runtime
    runtime.close()


@pytest.fixture()
def ram(tmp_path, monkeypatch):
    """Sets the free RAM the budget sees; no cgroup limits, no mountinfo."""
    from soup_cli.utils import l2l_spill

    empty = tmp_path / "no-cgroup"
    empty.mkdir()
    for name in ("PROC_ROOT", "CGROUP_V2_ROOT", "CGROUP_V1_ROOT"):
        monkeypatch.setattr(l2l_spill, name, str(empty))
    probe = tmp_path / "probe"
    probe.mkdir()
    problem = l2l_spill.filesystem_refusal(os.path.realpath(probe))
    problem = problem or l2l_spill._round_trip(str(probe))
    if problem:
        pytest.skip(f"the temp directory cannot hold an unbuffered spill: {problem}")

    def set_free(nbytes):
        monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes", lambda: nbytes)

    set_free(PLENTY)
    return set_free


class Run(NamedTuple):
    logs: List[dict]
    grads: List[dict]
    stores: List[Any]
    held: List[int]
    plans: List[dict]
    state: Any
    trainer: Any


def _held_bytes(store) -> int:
    """Host bytes the store's own allocations hold, each at its real storage size."""
    held = sum(t.untyped_storage().nbytes() for t in store._resident.values())
    if store.spill_layers:
        for ring in (store._write_ring, store._read_ring):
            deadline = time.time() + 5
            while ring._free.qsize() < 4 and time.time() < deadline:
                time.sleep(0.005)
            held += sum(buf.untyped_storage().nbytes() for buf, _ in list(ring._free.queue))
    return held


def _allocation_refs(store):
    """Weak references to each owned host allocation, including staging-buffer bases."""
    import weakref

    tensors = list(store._resident.values())
    if store.spill_layers:
        for ring in (store._write_ring, store._read_ring):
            tensors.extend(
                buf._base if buf._base is not None else buf
                for buf, _event in list(ring._free.queue)
            )
    return [(weakref.ref(tensor), tensor.untyped_storage().nbytes()) for tensor in tensors]


def _run(model, runtime, tmp_path, data, *, k, name, free, set_free,
         spill_dir=None, resume=None, callbacks=(), sequential=False, **overrides) -> Run:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads, stores, held, plans = [], [], [], []
    holder = {}

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append({
                n: p.grad.detach().clone()
                for n, p in model.named_parameters() if "lora_" in n and p.grad is not None
            })

        def on_step_end(self, args, state, control, **kw):
            store = holder["state"].store
            stores.append(store)
            held.append(_held_bytes(store))
            plans.append(dict(holder["state"].spill_plan))

    options = dict(
        output_dir=str(tmp_path / name), per_device_train_batch_size=1,
        gradient_accumulation_steps=k, learning_rate=0.5, optim="sgd",
        lr_scheduler_type="constant", num_train_epochs=1, logging_steps=1, save_strategy="no",
        report_to=[], use_cpu=True, seed=1, data_seed=1, max_grad_norm=0.0, disable_tqdm=True,
    )
    options.update(overrides)

    class SequentialTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return torch.utils.data.SequentialSampler(dataset)

    base_cls = SequentialTrainer if sequential else Trainer
    trainer = make_l2l_trainer_class(base_cls)(
        model=model, args=TrainingArguments(**options), train_dataset=data,
        callbacks=[Recorder(), *callbacks],
    )
    state = attach_l2l_state(
        trainer, runtime, dtype=torch.float32, spill_dir=spill_dir
    )
    holder["state"] = state
    set_free(free)
    trainer.train(resume_from_checkpoint=resume)
    logs = [row for row in trainer.state.log_history if "loss" in row]
    return Run(logs, grads, stores, held, plans, state, trainer)


def _pair(deep, tmp_path, set_free, data, k, **overrides):
    """The same rows from the same adapters, without and with spill."""
    harness = _load("l2l_schedule")
    model, runtime = deep
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    resident = _run(model, runtime, tmp_path, data, k=k, name="resident", free=PLENTY,
                    set_free=set_free, **overrides)
    harness.restore_adapters(model, start)
    spilled = _run(model, runtime, tmp_path, data, k=k, name="spilled", free=SMALL,
                   set_free=set_free, **overrides)
    return resident, spilled


def _assert_equal_runs(resident, spilled, windows):
    import torch

    assert len(resident.grads) == len(spilled.grads) == windows
    for ours, theirs in zip(spilled.grads, resident.grads):
        assert ours.keys() == theirs.keys()
        assert any(bool(g.abs().sum() > 0) for g in ours.values())
        for name in theirs:
            assert torch.equal(ours[name], theirs[name]), name
    # a resumed trainer's history starts with the checkpoint's rows: compare this run's own
    resident_logs, spilled_logs = resident.logs[-windows:], spilled.logs[-windows:]
    assert [row["loss"] for row in spilled_logs] == [row["loss"] for row in resident_logs]
    # the spill really happened, and the twin really did not
    assert all(row["l2l_spill_layers"] == 0 for row in resident_logs)
    layers = [plan["spill_layers"] for plan in spilled.plans]
    assert len(layers) == windows and all(0 < count < N_LAYERS for count in layers)
    assert [row["l2l_spill_layers"] for row in spilled_logs] == layers
    assert all(store.stats.bytes_written > 0 and store.stats.bytes_read > 0
               for store in spilled.stores)
    assert all(store.stats.bytes_written == 0 for store in resident.stores)


def _masked(lengths):
    return _MaskedRows(lengths, [[(0, i % 3)] if i % 3 else [] for i in range(len(lengths))])


# --------------------------------------------------------------------------
# equality
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k,layers", [(2, 6), (4, 7)])
def test_spill_gives_the_losses_and_gradients_of_the_resident_store(
    deep, tmp_path, ram, k, layers
):
    resident, spilled = _pair(deep, tmp_path, ram, _masked([SPILL_SEQ] * 8), k)
    _assert_equal_runs(resident, spilled, 8 // k)
    assert spilled.state.spill_plan["spill_layers"] == layers


@pytest.mark.parametrize("k", [2, 4])
def test_rows_of_different_lengths_spill_and_stay_equal(deep, tmp_path, ram, k):
    # Every window exceeds the RAM budget, but each position keeps its own capacity.
    lengths = [61, 64, 60, 63, 62, 61, 64, 60]
    resident, spilled = _pair(deep, tmp_path, ram, _masked(lengths), k)
    _assert_equal_runs(resident, spilled, 8 // k)
    assert any(len(set(plan["mb_bytes"])) > 1 for plan in spilled.plans)
    assert all(held == plan["host_bytes"] <= plan["budget_bytes"]
               for held, plan in zip(spilled.held, spilled.plans))


def test_a_short_last_window_spills_and_stays_equal(deep, tmp_path, ram):
    resident, spilled = _pair(deep, tmp_path, ram, _masked([SPILL_SEQ] * 7), 3)
    _assert_equal_runs(resident, spilled, 3)  # windows 3, 3, 1


def test_resume_with_spill_equals_resume_without(deep, tmp_path, ram, monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    harness = _load("l2l_schedule")
    model, runtime = deep
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    data = _masked([SPILL_SEQ] * 7)
    options = dict(k=3, num_train_epochs=2, lr_scheduler_type="linear", warmup_steps=1,
                   set_free=ram)
    _run(model, runtime, tmp_path, data, name="full", free=PLENTY, save_strategy="steps",
         save_steps=1, **options)
    checkpoint = str(tmp_path / "full" / "checkpoint-3")
    runs = {}
    for name, free in (("resident", PLENTY), ("spilled", SMALL)):
        harness.restore_adapters(model, start)
        runs[name] = _run(model, runtime, tmp_path, data, name=name, free=free,
                          resume=checkpoint, **options)
    assert runs["spilled"].trainer.state.global_step == 6
    _assert_equal_runs(runs["resident"], runs["spilled"], 3)


# --------------------------------------------------------------------------
# one store and one file per run, within the plan
# --------------------------------------------------------------------------
def test_one_store_and_one_file_serve_windows_of_every_length(deep, tmp_path, ram, monkeypatch):
    from soup_cli.utils import l2l_activations, l2l_spill

    created = []

    class Spy(l2l_activations.DirectSpillFile):
        def __init__(self, path, size, **kw):
            if os.path.basename(path).startswith(l2l_spill.SPILL_FILE_PREFIX):
                created.append((path, size))
            super().__init__(path, size, **kw)

    monkeypatch.setattr(l2l_activations, "DirectSpillFile", Spy)
    buffer = io.StringIO()
    from rich.console import Console

    monkeypatch.setattr(l2l_spill, "console", Console(file=buffer, width=80))
    model, runtime = deep
    _load("l2l_schedule").make_non_vacuous(model)
    # The first full window establishes both capacities; later rows and the last window
    # fit them without an up-front dataset maximum or padding the rows.
    lengths = [64, 64, 3, 8, 60, 9, 4, 33, 5]
    run = _run(model, runtime, tmp_path, _masked(lengths), k=2, name="spilled", free=SMALL,
               set_free=ram, num_train_epochs=2, sequential=True)

    plan = run.state.spill_plan
    assert PLAN_KEYS <= set(plan)
    assert plan["spill_layers"] == 6 and plan["k"] == 2 and plan["ring"] == 4
    assert plan["chunk_bytes"] == SPILL_SEQ * 64 * 4
    assert plan["mb_bytes"] == [SPILL_SEQ * 64 * 4] * 2
    assert plan["limiting_level"] == "psutil"
    assert plan["limiting_numbers"] == {"psutil_available": SMALL}
    assert plan["explicit_dir"] is False
    assert plan["spill_dir"] == os.path.realpath(tmp_path / "spilled" / "_l2l_spill")
    # one file, of the planned size, for ten windows of different lengths
    assert created == [(plan["spill_path"], plan["spill_file_bytes"])]
    assert len(run.stores) == 10 and len({id(store) for store in run.stores}) == 1
    assert set(run.state.calls_by_shape) == {(1, length) for length in lengths}
    # the store's real allocations are the plan's, within the budget, at every step
    assert run.held == [plan["host_bytes"]] * 10
    assert plan["host_bytes"] <= plan["budget_bytes"]
    # removed at the end, and the state no longer holds the store
    assert run.state.store is None
    assert not os.path.exists(plan["spill_path"])
    # one console line before the first step: layers, GB, file, budget, limiting level
    lines = [line for line in buffer.getvalue().splitlines() if "spilling" in line]
    assert len(lines) == 1
    line = lines[0]
    assert "6 of 8 layers" in line and " GB" in line and plan["spill_path"] in line
    assert "RAM budget" in line and "psutil available" in line


def test_activations_that_fit_change_nothing(deep, tmp_path, ram, monkeypatch):
    from soup_cli.utils import l2l_spill

    def forbidden(*a, **kw):
        raise AssertionError("the spill directory was checked")

    monkeypatch.setattr(l2l_spill, "prepare_spill_dir", forbidden)
    model, runtime = deep
    _load("l2l_schedule").make_non_vacuous(model)
    run = _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="fits", free=PLENTY,
               set_free=ram)
    assert [row["l2l_spill_layers"] for row in run.logs] == [0, 0]
    assert run.state.spill_plan["spill_path"] is None
    assert not (tmp_path / "fits" / "_l2l_spill").exists()


def test_a_budget_below_every_layer_spilled_stops_before_allocating(
    deep, tmp_path, ram, monkeypatch
):
    from soup_cli.utils import l2l_activations

    def forbidden(*a, **kw):
        raise AssertionError("the store was allocated")

    monkeypatch.setattr(l2l_activations, "ActivationStore", forbidden)
    model, runtime = deep
    with pytest.raises(ValueError) as caught:
        _run(model, runtime, tmp_path, _masked([SEQ] * 4), k=2, name="short", free=1000,
             set_free=ram)
    message = str(caught.value)
    for fix in ("gradient_accumulation_steps", "data.max_length", "batch_size", "free RAM"):
        assert fix in message


# --------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------
def _raising_at_step(exc):
    from transformers import TrainerCallback

    seen = {}

    class Boom(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            path = seen["state"].spill_plan["spill_path"]
            seen["existed"] = os.path.exists(path)
            raise exc

    return Boom(), seen


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_an_exception_or_ctrl_c_removes_the_spill_file(deep, tmp_path, ram, monkeypatch, exc):
    from soup_cli.trainer import l2l_trainer
    from soup_cli.utils.l2l_spill import spill_files

    callback, seen = _raising_at_step(exc)
    real_attach = l2l_trainer.attach_l2l_state

    def attach(*a, **kw):
        seen["state"] = real_attach(*a, **kw)
        return seen["state"]

    monkeypatch.setattr(l2l_trainer, "attach_l2l_state", attach)
    model, runtime = deep
    with pytest.raises(type(exc)):
        _run(model, runtime, tmp_path, _masked([SPILL_SEQ] * 4), k=2, name="boom", free=SMALL,
             set_free=ram, callbacks=[callback])
    assert seen["existed"] is True
    assert seen["state"].store is None
    assert spill_files(str(tmp_path / "boom" / "_l2l_spill")) == []


def test_a_full_drive_mid_step_names_size_and_directory_and_removes_the_file(
    deep, tmp_path, ram, monkeypatch
):
    from soup_cli.utils import l2l_activations
    from soup_cli.utils.l2l_spill import SPILL_FILE_PREFIX, SpillDiskFullError, spill_files

    real_write = l2l_activations.DirectSpillFile.write

    def write(self, offset, buf):
        if os.path.basename(self.path).startswith(SPILL_FILE_PREFIX):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(self, offset, buf)

    monkeypatch.setattr(l2l_activations.DirectSpillFile, "write", write)
    model, runtime = deep
    with pytest.raises(SpillDiskFullError) as caught:
        _run(model, runtime, tmp_path, _masked([SPILL_SEQ] * 4), k=2, name="full", free=SMALL,
             set_free=ram)
    directory = os.path.realpath(tmp_path / "full" / "_l2l_spill")
    message = str(caught.value)
    file_bytes = 6 * 2 * SPILL_SEQ * 64 * 4
    assert directory in message and f"{file_bytes:,} bytes" in message
    assert spill_files(directory) == []


def test_train_sweeps_files_of_dead_runs_first(deep, tmp_path, ram):
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    directory = tmp_path / "swept" / "_l2l_spill"
    directory.mkdir(parents=True)
    stale = directory / f"soup-l2l-spill-{proc.pid}-deadbeef.bin"
    stale.write_bytes(b"x" * 4096)
    mine = directory / f"soup-l2l-spill-{os.getpid()}-0badf00d.bin"
    mine.write_bytes(b"x")
    model, runtime = deep
    run = _run(model, runtime, tmp_path, _masked([SEQ] * 2), k=2, name="swept", free=PLENTY,
               set_free=ram)
    assert run.state.stale_removed == [os.path.realpath(stale)]
    assert not stale.exists() and mine.exists()


# --------------------------------------------------------------------------
# the store's capacity and the spill directory setting
# --------------------------------------------------------------------------
def _stepper(state, model, seq):
    from soup_cli.trainer.l2l_trainer import _l2l_step_for

    return _l2l_step_for(state, model, _ids([seq]))[0]


def _ids(lengths):
    import torch

    return [torch.zeros((1, length), dtype=torch.long) for length in lengths]


def test_only_a_larger_position_rebuilds_the_store_and_makes_a_second_file(
    deep, tmp_path, ram, monkeypatch
):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_spill

    opened = []
    real_open = l2l_spill.open_spill_file
    monkeypatch.setattr(
        l2l_spill, "open_spill_file", lambda path, size: opened.append(path) or real_open(
            path, size)
    )
    # The 400,000-byte budget spills 5 layers at four 64-token positions. Growing only
    # position 0 to 128 tokens needs larger rings, but leaves the other capacities alone.
    ram((512 << 20) + 400_000)
    model, runtime = deep
    state = L2LState(runtime=runtime, dtype=torch.float32, output_dir=str(tmp_path / "out"))
    try:
        first = _store_for(state, _stepper(state, model, 64), _ids([64] * 4), "cpu")
        assert first.mb_shapes == [(64 * 64,)] * 4 and first.spill_layers == 5
        assert _store_for(state, _stepper(state, model, 8), _ids([8] * 2), "cpu") is first
        bigger = _store_for(state, _stepper(state, model, 128), _ids([128, 64, 64, 64]), "cpu")
        assert bigger is not first and bigger.mb_shapes == [(128 * 64,)] + [(64 * 64,)] * 3
        assert bigger.spill_layers == 7
        plan = state.spill_plan
        assert plan["mb_bytes"] == [32768, 16384, 16384, 16384]
        assert _held_bytes(bigger) == plan["host_bytes"] <= plan["budget_bytes"]
        assert len(opened) == 2
        assert not os.path.exists(opened[0]) and os.path.exists(opened[1])
    finally:
        state.close()
    assert not os.path.exists(opened[1])


@pytest.mark.parametrize("budget_bytes,first_spill,bigger_spill",
                         [(400_000, 5, 8), (600_000, 0, 6)])
def test_growth_releases_old_allocations_and_store_before_reading_ram_budget(
    tmp_path, ram, monkeypatch, budget_bytes, first_spill, bigger_spill
):
    import weakref
    from types import SimpleNamespace

    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_spill

    ram((512 << 20) + budget_bytes)
    stepper = SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=64)), layers=range(N_LAYERS)
    )
    state = L2LState(runtime=SimpleNamespace(tier="ram"), dtype=torch.float32,
                     output_dir=str(tmp_path / "growth"))
    allocations, observations = [], []
    old_store = None
    real_budget = l2l_spill.ram_budget

    def live_bytes():
        return sum(nbytes for ref, nbytes in allocations if ref() is not None)

    def budget():
        observations.append((old_store is not None and old_store() is not None, live_bytes()))
        return real_budget()

    # Unlike a constant free-RAM mock, still-live old allocations reduce the new budget.
    monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes",
                        lambda: (512 << 20) + budget_bytes - live_bytes())
    monkeypatch.setattr(l2l_spill, "ram_budget", budget)
    try:
        first = _store_for(state, stepper, _ids([64] * 4), "cpu")
        assert first.spill_layers == first_spill
        old_store = weakref.ref(first)
        allocations = _allocation_refs(first)
        assert live_bytes() > 0
        assert _store_for(state, stepper, _ids([8] * 2), "cpu") is first
        del first

        bigger = _store_for(state, stepper, _ids([128] * 4), "cpu")
        assert bigger.mb_shapes == [(128 * 64,)] * 4 and bigger.spill_layers == bigger_spill
        assert observations == [(False, 0), (False, 0)]
        assert state.spill_plan["budget_bytes"] == budget_bytes
        assert _store_for(state, stepper, _ids([16]), "cpu") is bigger
        assert len(observations) == 2
    finally:
        state.close()


@pytest.mark.parametrize("public_cache_api", [True, False])
def test_cuda_growth_releases_idle_pinned_cache_before_rebudget_or_warns_without_public_api(
    tmp_path, ram, monkeypatch, caplog, public_cache_api
):
    import weakref
    from types import SimpleNamespace

    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_activations, l2l_spill

    ram((512 << 20) + 600_000)
    stepper = SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=64)), layers=range(N_LAYERS)
    )
    state = L2LState(runtime=SimpleNamespace(tier="ram"), dtype=torch.float32,
                     output_dir=str(tmp_path / "cuda-growth"))
    real_store = l2l_activations.ActivationStore

    def cpu_store(**kwargs):
        # Keep real allocation/close behavior, without requiring CUDA in this planner test.
        return real_store(**{**kwargs, "device": "cpu", "pin": False})

    monkeypatch.setattr(l2l_activations, "ActivationStore", cpu_store)
    allocations, observations, releases = [], [], []
    old_store = None
    idle_cache = 0
    real_budget = l2l_spill.ram_budget

    def empty_host_cache():
        nonlocal idle_cache
        assert old_store is not None and old_store() is None
        assert all(ref() is None for ref, _nbytes in allocations)
        releases.append(idle_cache)
        idle_cache = 0

    def budget():
        observations.append(idle_cache)
        return real_budget()

    accelerator = SimpleNamespace()
    if public_cache_api:
        accelerator.empty_host_cache = empty_host_cache
    monkeypatch.setattr(torch, "accelerator", accelerator, raising=False)
    monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes",
                        lambda: (512 << 20) + 600_000 - idle_cache)
    monkeypatch.setattr(l2l_spill, "ram_budget", budget)
    try:
        first = _store_for(state, stepper, _ids([64] * 4), "cuda:0")
        assert first.spill_layers == 0
        old_store = weakref.ref(first)
        allocations = _allocation_refs(first)
        # CUDA's 32 resident chunks are 16 KiB each; deletion moves them to its cache.
        idle_cache = 524_288
        assert _store_for(state, stepper, _ids([8] * 2), "cuda:0") is first
        assert releases == [] and observations == [0]
        del first

        if public_cache_api:
            bigger = _store_for(state, stepper, _ids([128] * 4), "cuda:0")
            assert bigger.mb_shapes == [(128 * 64,)] * 4 and bigger.spill_layers == 8
            assert observations == [0, 0] and releases == [524_288]
            assert _store_for(state, stepper, _ids([16]), "cuda:0") is bigger
            assert len(observations) == 2 and len(releases) == 1
        else:
            with pytest.raises(l2l_spill.SpillRefusedError, match="every layer spilled"):
                _store_for(state, stepper, _ids([128] * 4), "cuda:0")
            assert old_store() is None and all(ref() is None for ref, _nbytes in allocations)
            assert observations == [0, 524_288] and releases == []
            assert "empty_host_cache" in caplog.text and "remain charged" in caplog.text
    finally:
        state.close()


def test_cuda_windows_that_fit_keep_one_plan_without_flushing_pinned_cache(
    tmp_path, ram, monkeypatch
):
    from types import SimpleNamespace

    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_activations, l2l_spill

    ram(PLENTY)
    real_store = l2l_activations.ActivationStore
    real_budget = l2l_spill.ram_budget
    budgets = []

    def forbidden():
        raise AssertionError("a fixed-capacity store flushed the pinned cache")

    def budget():
        budgets.append(real_budget())
        return budgets[-1]

    monkeypatch.setattr(torch, "accelerator", SimpleNamespace(empty_host_cache=forbidden),
                        raising=False)
    monkeypatch.setattr(l2l_activations, "ActivationStore",
                        lambda **kw: real_store(**{**kw, "device": "cpu", "pin": False}))
    monkeypatch.setattr(l2l_spill, "ram_budget", budget)
    stepper = SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=64)), layers=range(N_LAYERS)
    )
    state = L2LState(runtime=SimpleNamespace(tier="ram"), dtype=torch.float32,
                     output_dir=str(tmp_path / "known"))
    try:
        window = [torch.zeros((2, 128), dtype=torch.long)] * 4
        store = _store_for(state, stepper, window, "cuda:0")
        assert store.mb_shapes == [(2 * 128 * 64,)] * 4 and store.k == 4
        plan = state.spill_plan
        for smaller in (_ids([64] * 2), window, _ids([16])):
            assert _store_for(state, stepper, smaller, "cuda:0") is store
            assert state.spill_plan is plan
        assert len(budgets) == 1
        assert plan["spill_layers"] == 0 and plan["pin"] is True
    finally:
        state.close()


def test_an_explicit_spill_dir_is_used(deep, tmp_path, ram):
    model, runtime = deep
    _load("l2l_schedule").make_non_vacuous(model)
    target = tmp_path / "elsewhere" / "spill"
    run = _run(model, runtime, tmp_path, _masked([SPILL_SEQ] * 4), k=2, name="explicit", free=SMALL,
               set_free=ram, spill_dir=str(target))
    plan = run.state.spill_plan
    assert plan["explicit_dir"] is True
    assert plan["spill_dir"] == os.path.realpath(target)
    assert os.path.dirname(plan["spill_path"]) == os.path.realpath(target)
    assert not (tmp_path / "explicit" / "_l2l_spill").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_the_disk_tier_keeps_the_default_dir_off_the_weights_disk(
    deep, tmp_path, ram, monkeypatch, explicit
):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill, "physical_disks", lambda path: (frozenset({"sda"}), "sda"))
    lines = []
    monkeypatch.setattr(l2l_spill, "say", lambda line, notify=None: lines.append(line))
    ram(SMALL)
    model, runtime = deep
    monkeypatch.setattr(runtime, "tier", "disk")
    state = L2LState(
        runtime=runtime, dtype=torch.float32, output_dir=str(tmp_path / "out"),
        spill_dir=str(tmp_path / "mine") if explicit else None,
        weight_dirs=(str(tmp_path / "shards"),),
    )
    try:
        if not explicit:
            with pytest.raises(l2l_spill.SpillRefusedError, match="training.l2l_spill_dir"):
                _store_for(state, _stepper(state, model, 64), _ids([64] * 2), "cpu")
            return
        store = _store_for(state, _stepper(state, model, 64), _ids([64] * 2), "cpu")
        assert store.spill_layers > 0
        assert any("13% slower" in line for line in lines)
    finally:
        state.close()
