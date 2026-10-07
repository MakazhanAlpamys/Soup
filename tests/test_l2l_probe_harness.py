"""CPU tests for the L2L step-0 probe harness (benchmarks/harness/l2l_*.py).

The harness is benchmark code, not shipped; these tests pin the parts a GPU run
cannot check by itself: the schedule's bit-exactness against gradient
accumulation, the activation store's byte round trip and spill ordering, the
suspend watch, and the rule's verdict logic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest


def _psutil_has_io_counters() -> bool:
    """macOS psutil has no per-process io_counters; the two tests that read it skip there."""
    try:
        import psutil
    except ImportError:
        return False
    return hasattr(psutil.Process(), "io_counters")


_NEEDS_IO_COUNTERS = pytest.mark.skipif(
    not _psutil_has_io_counters(),
    reason="psutil exposes no per-process io_counters on this platform",
)

HARNESS = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"


def _load(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# l2l_activations
# --------------------------------------------------------------------------
@pytest.fixture()
def acts():
    return _load("l2l_activations")


def _chunk(seed: int, shape=(1, 8, 128)):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen)


def _store(acts, tmp_path, *, n_layers=4, k=3, spill_layers=0, backend=None, ring=8):
    import torch

    factory = None
    if spill_layers:
        made = backend or acts.BufferedSpillFile

        def factory(size):
            return made(str(tmp_path / "spill.bin"), size)

    return acts.ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, 8, 128),
        dtype=torch.float32,
        device="cpu",
        pin=False,
        spill_layers=spill_layers,
        spill_factory=factory,
        ring=ring,
    )


def _forward_puts(store, n_layers, k):
    for layer in range(n_layers):
        for mb in range(k):
            store.put(layer, mb, _chunk(layer * 100 + mb))


def test_aligned_buffer_is_sector_aligned_and_exact(acts):
    buf = acts.aligned_buffer(3 * 4096, pin=False)
    assert buf.data_ptr() % 4096 == 0
    assert buf.numel() == 3 * 4096


def test_resident_round_trip_returns_a_copy(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path)
    _forward_puts(store, 4, 3)
    store.begin_backward()
    for layer in range(3, -1, -1):
        for mb in range(3):
            got = store.get(layer, mb)
            assert torch.equal(got, _chunk(layer * 100 + mb))
            got.zero_()  # a caller scribbling on its tensor must not reach the store
            assert torch.equal(store.get(layer, mb), _chunk(layer * 100 + mb))
    store.end_step()
    store.close()


def test_spill_round_trip_picks_the_lowest_layers(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path, spill_layers=2)
    assert [store.is_spilled(layer) for layer in range(4)] == [True, True, False, False]
    assert store.stats.resident_chunks == 2 * 3
    assert store.stats.spilled_chunks == 2 * 3
    _forward_puts(store, 4, 3)
    store.begin_backward()
    for layer in range(3, -1, -1):
        for mb in range(3):
            assert torch.equal(store.get(layer, mb), _chunk(layer * 100 + mb))
    store.end_step()
    assert store.stats.bytes_written == 2 * 3 * store.chunk_bytes
    assert store.stats.bytes_read == 2 * 3 * store.chunk_bytes
    store.close()
    assert not (tmp_path / "spill.bin").exists()


def test_spill_survives_two_steps_with_new_values(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path, spill_layers=2)
    for step in range(2):
        for layer in range(4):
            for mb in range(3):
                store.put(layer, mb, _chunk(step * 1000 + layer * 100 + mb))
        store.begin_backward()
        for layer in range(3, -1, -1):
            for mb in range(3):
                assert torch.equal(store.get(layer, mb), _chunk(step * 1000 + layer * 100 + mb))
        store.end_step()
    store.close()


def test_get_without_begin_backward_reads_on_demand(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path, spill_layers=1)
    _forward_puts(store, 4, 3)
    assert torch.equal(store.get(0, 2), _chunk(2))
    store.close()


def test_a_ring_of_one_back_pressures_instead_of_overwriting(acts, tmp_path):
    import torch

    class SlowFile(acts.BufferedSpillFile):
        def write(self, offset, buf):
            time.sleep(0.02)  # the staging buffer must not be refilled meanwhile
            super().write(offset, buf)

    store = _store(acts, tmp_path, spill_layers=3, backend=SlowFile, ring=1)
    _forward_puts(store, 4, 3)
    store.begin_backward()
    for layer in range(3, -1, -1):
        for mb in range(3):
            assert torch.equal(store.get(layer, mb), _chunk(layer * 100 + mb))
    store.end_step()
    assert store.stats.put_wait_s > 0.0
    store.close()


def test_spill_write_failure_surfaces_on_the_next_call(acts, tmp_path):
    class BrokenFile(acts.BufferedSpillFile):
        def write(self, offset, buf):
            raise OSError("disk full")

    store = _store(acts, tmp_path, spill_layers=2, backend=BrokenFile)
    store.put(0, 0, _chunk(0))
    with pytest.raises(RuntimeError, match="spill.bin"):
        deadline = time.time() + 5
        while time.time() < deadline:
            store.put(0, 1, _chunk(1))
            time.sleep(0.01)
    store.close()


def test_end_step_refuses_unconsumed_spilled_chunks(acts, tmp_path):
    store = _store(acts, tmp_path, spill_layers=2)
    _forward_puts(store, 4, 3)
    store.begin_backward()
    store.get(3, 0)
    with pytest.raises(RuntimeError, match="not consumed"):
        store.end_step()
    store.close()


def test_spill_layers_without_a_factory_is_refused(acts):
    import torch

    with pytest.raises(ValueError, match="spill_factory"):
        acts.ActivationStore(
            n_layers=4,
            k=2,
            chunk_shape=(1, 8, 128),
            dtype=torch.float32,
            device="cpu",
            pin=False,
            spill_layers=1,
        )


def test_direct_spill_refuses_an_unaligned_size(acts, tmp_path):
    with pytest.raises(ValueError, match="multiple of 4096"):
        acts.DirectSpillFile(str(tmp_path / "x.bin"), 4096 + 512)


# --------------------------------------------------------------------------
# l2l_box
# --------------------------------------------------------------------------
@pytest.fixture()
def box():
    return _load("l2l_box")


class _FakeClock:
    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


def test_steady_samples_are_not_void(box):
    watch = box.SuspendWatch(clock=_FakeClock([0, 2, 4, 6]), power=lambda: True)
    for _ in range(4):
        watch.sample()
    report = watch.report()
    assert report["samples"] == 4
    assert report["max_gap_s"] == pytest.approx(2.0)
    assert report["void"] is False


def test_gap_over_limit_voids(box):
    watch = box.SuspendWatch(clock=_FakeClock([0, 2, 33]), power=lambda: True)
    for _ in range(3):
        watch.sample()
    assert watch.report()["max_gap_s"] == pytest.approx(31.0)
    assert watch.report()["void"] is True


def test_battery_sample_voids(box):
    power = iter([True, False, True])
    watch = box.SuspendWatch(clock=_FakeClock([0, 2, 4]), power=lambda: next(power))
    for _ in range(3):
        watch.sample()
    assert watch.report()["battery_samples"] == 1
    assert watch.report()["void"] is True


def test_unknown_power_is_recorded_not_void(box):
    watch = box.SuspendWatch(clock=_FakeClock([0, 2]), power=lambda: None)
    watch.sample()
    watch.sample()
    assert watch.report()["unknown_power_samples"] == 2
    assert watch.report()["void"] is False


def test_thread_start_stop_takes_samples(box):
    watch = box.SuspendWatch(interval=0.01, power=lambda: True)
    watch.start()
    time.sleep(0.1)
    report = watch.stop()
    assert report["samples"] >= 3
    assert not any(t.name == "l2l-suspend-watch" for t in threading.enumerate())


def test_box_stamp_has_the_keys(box):
    stamp = box.box_stamp()
    for key in (
        "unix",
        "avail_phys_gb",
        "commit_avail_gb",
        "ac",
        "gpu_pids",
        "gpu_mem_used_mib",
        "python_processes",
    ):
        assert key in stamp


def test_foreign_readers_flags_heavy_other_processes(box):
    before = {
        10: {"name": "a.exe", "read": 0},
        11: {"name": "b.exe", "read": 5},
        99: {"name": "python.exe", "read": 0},
    }
    after = {
        10: {"name": "a.exe", "read": 2_000_000_000},
        11: {"name": "b.exe", "read": 6},
        99: {"name": "python.exe", "read": 3_000_000_000},
        12: {"name": "SearchIndexer.exe", "read": 1_500_000_000},
    }
    heavy = box.foreign_readers(before, after, own_pids={99}, threshold=1_000_000_000)
    assert [(row["pid"], row["name"]) for row in heavy] == [
        (10, "a.exe"),
        (12, "SearchIndexer.exe"),
    ]
    assert heavy[0]["read_bytes"] == 2_000_000_000


def test_foreign_readers_is_none_without_snapshots(box):
    assert box.foreign_readers(None, {}, own_pids=set()) is None


@_NEEDS_IO_COUNTERS
def test_process_read_bytes_sees_this_process(box):
    import os

    snapshot = box.process_read_bytes()
    if snapshot is None:
        pytest.skip("psutil is not installed")
    assert os.getpid() in snapshot
    assert snapshot[os.getpid()]["read"] >= 0


# --------------------------------------------------------------------------
# l2l_schedule — bit-exact against gradient accumulation, on the SHIPPED runtime
# --------------------------------------------------------------------------
SEQ = 8


@pytest.fixture()
def sched():
    return _load("l2l_schedule")


@pytest.fixture()
def streamed(tmp_path):
    """A tiny untied Llama through shard_checkpoint + build_streamed_model on CPU.

    Untied, so the embed/head share the streamed large slot exactly as the 70B
    fixture's do; three layers, so the zig-zag turnaround and the backward
    prefetch both run.
    """
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07204 import _tiny_llama_dir

    weights, _model, _config = _tiny_llama_dir(tmp_path, n_layers=3, tie=False)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=lora,
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=3,
    )
    yield model, runtime
    runtime.close()


def _micro_batches(k, vocab=64, seed=17):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (1, SEQ), generator=gen) for _ in range(k)]


def _cpu_store(acts, k, n_layers=3, hidden=64):
    import torch

    return acts.ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, SEQ, hidden),
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_l2l_equals_gradient_accumulation_bit_for_bit(sched, acts, streamed, k):
    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = _micro_batches(k)
    sched.zero_grads(model)
    ga_losses = sched.ga_step(model, mbs)
    ga = sched.collect_grads(model)

    call = sched.capture_layer_call(model, mbs[0])
    step = sched.L2LStep(model, runtime, call)
    sched.zero_grads(model)
    l2l_losses = step.run(mbs, _cpu_store(acts, k))
    l2l = sched.collect_grads(model)

    result = sched.compare(ga, l2l)
    assert result["tensors"] == 3 * 4 * 2  # 3 layers x q/k/v/o x A/B
    assert result["all_zero"] == []
    assert result["equal"], result
    assert sched.losses_equal(ga_losses, l2l_losses)


def test_the_divisor_matters(sched, acts, streamed):
    """Mutation guard: dividing by one micro-batch's tokens instead of all k must differ."""
    model, runtime = streamed
    sched.make_non_vacuous(model)
    mbs = _micro_batches(2)
    sched.zero_grads(model)
    sched.ga_step(model, mbs)
    ga = sched.collect_grads(model)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    sched.zero_grads(model)
    step.run(mbs, _cpu_store(acts, 2), n_items=SEQ - 1)
    assert not sched.compare(ga, sched.collect_grads(model))["equal"]


def test_losses_are_compared_per_position(sched, acts, streamed):
    """Mutation guard: the same micro-batches in another order must not pass."""
    model, runtime = streamed
    mbs = _micro_batches(2)
    sched.zero_grads(model)
    ga_losses = sched.ga_step(model, mbs)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    sched.zero_grads(model)
    swapped = step.run(list(reversed(mbs)), _cpu_store(acts, 2))
    assert not sched.losses_equal(ga_losses, swapped)


def test_one_prime_and_the_loads_of_one_micro_batch(sched, acts, streamed):
    """`prime` loads layer 0 unconditionally and `advance` skips owned slots, so a
    step's load count depends on what the previous walk left in the slots. Both
    measured steps therefore follow the same forward-only walk."""
    model, runtime = streamed
    mbs = _micro_batches(3)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    sched.zero_grads(model)
    loads = runtime.pool.loads
    sched.ga_step(model, mbs[:1])
    ga_loads = runtime.pool.loads - loads
    sched.capture_layer_call(model, mbs[0])
    sched.zero_grads(model)
    loads, primes = runtime.pool.loads, runtime.prefetcher.primes
    step.run(mbs, _cpu_store(acts, 3))
    assert runtime.prefetcher.primes - primes == 1
    assert runtime.pool.loads - loads == ga_loads


def test_checkpoint_flags_are_restored(sched, acts, streamed):
    model, runtime = streamed
    mbs = _micro_batches(2)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    step.run(mbs, _cpu_store(acts, 2))
    assert all(layer.use_checkpoint for layer in step.layers)


def test_capture_drops_num_items_and_keeps_rotary(sched, streamed):
    model, _runtime = streamed
    call = sched.capture_layer_call(model, _micro_batches(1)[0])
    assert "num_items_in_batch" not in call.kwargs
    assert call.kwargs.get("position_embeddings") is not None


def test_run_refuses_more_micro_batches_than_the_store_holds(sched, acts, streamed):
    model, runtime = streamed
    mbs = _micro_batches(3)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    with pytest.raises(ValueError, match="3 micro-batches.*holds 2"):
        step.run(mbs, _cpu_store(acts, 2))
    with pytest.raises(ValueError, match="at least one"):
        step.run([], _cpu_store(acts, 2))


def test_missing_prime_hook_is_refused(sched, streamed):
    model, runtime = streamed
    runtime.hook.remove()
    with pytest.raises(RuntimeError, match="prime hook"):
        sched.L2LStep(model, runtime, sched.LayerCall(args=(), kwargs={}))


def test_snapshot_restore_round_trip(sched, streamed):
    import torch

    model, _runtime = streamed
    snap = sched.snapshot_adapters(model)
    sched.make_non_vacuous(model, seed=99)
    sched.restore_adapters(model, snap)
    for name, param in sched.lora_parameters(model).items():
        assert torch.equal(param.detach(), snap[name])
        assert param.grad is None


# --------------------------------------------------------------------------
# l2l_rule
# --------------------------------------------------------------------------
@pytest.fixture()
def rule():
    return _load("l2l_rule")


def _cmp(equal=True, zero=()):
    return {
        "tensors": 4,
        "unequal": [] if equal else ["x"],
        "max_abs": 0.0 if equal else 1e-3,
        "worst": [],
        "all_zero": list(zero),
        "equal": equal,
    }


def _corr(fixture, *, a_vs_a=True, l2l=True, det=False, error=None):
    rec = {
        "kind": "correctness",
        "fixture": fixture,
        "k": 2,
        "deterministic": det,
        "a_vs_a": _cmp(a_vs_a),
        "a_vs_a_losses_equal": a_vs_a,
        "l2l": _cmp(l2l),
        "l2l_losses_equal": l2l,
    }
    if error:
        rec["deterministic_error"] = error
    return {"meta": {}, "records": [rec]}


def _arm(
    mode, arm, k, rnd, step_s, *, peak=4.4, void=False, variant=None, skipped=None, loads=157.0
):
    return {
        "kind": "arm",
        "mode": mode,
        "arm": arm,
        "k": k,
        "round": rnd,
        "variant": variant,
        "step_s_mean": step_s,
        "tokens_per_step": 512 * k,
        "peak_alloc_gb": peak,
        "layer_loads_per_step": loads,
        "large_loads_per_step": 2.0,
        "bytes_moved_per_step": loads * 441e6 + 2 * 525e6,
        "suspend": {"void": void},
        "store": None,
        "skipped": skipped,
    }


def _timing(**over):
    """Predicted-shape numbers that pass every gate."""
    times = {
        ("plain", "A", 1): 17.1,
        ("plain", "B", 1): 6.43,
        ("l2l", "A", 1): 17.2,
        ("l2l", "B", 1): 6.5,
        ("l2l", "A", 16): 104.0,
        ("l2l", "B", 16): 103.0,
        ("l2l", "C", 4): 26.0,
        ("l2l", "D", 4): 20.0,
        ("l2l", "A", 4): 26.5,
        ("l2l", "B", 4): 26.0,
    }
    times.update(over.pop("times", {}))
    records = []
    for (mode, arm, k), step_s in times.items():
        for rnd in (0, 1):
            records.append(
                _arm(mode, arm, k, rnd, step_s, peak=over.get("peak16", 4.6) if k == 16 else 4.4)
            )
    for patch in over.pop("patch", []):
        patch(records)
    meta = {"direct_io": over.get("direct_io", True), "v3": {"ok": over.get("v3", True)}}
    return {"meta": meta, "records": records}


def _write(tmp_path, timing=None, f1=None, f2=None, f1_det=None, spill=None):
    files = {
        "timing.json": timing or _timing(),
        "correctness_f1.json": f1 or _corr("F1"),
        "correctness_f2.json": f2 or _corr("F2"),
    }
    if f1_det:
        files["correctness_f1_det.json"] = f1_det
    if spill:
        files["spill.json"] = spill
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path)


def test_all_gates_pass(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path))
    assert [out["gates"][g]["status"] for g in ("G1", "G2", "G3", "G4")] == ["pass"] * 4
    assert out["verdict"] == "BUILD STEP 1"


def test_g2_fails_when_the_read_does_not_hide(rule, tmp_path):
    timing = _timing(times={("l2l", "A", 16): 140.0})  # T_A = 0.74 x T_B
    out = rule.evaluate(_write(tmp_path, timing=timing))
    assert out["gates"]["G2"]["status"] == "fail"
    assert out["verdict"].startswith("STOP")


def test_g4_fails_when_vram_grows(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path, timing=_timing(peak16=5.0)))
    assert out["gates"]["G4"]["status"] == "fail"


def test_round_spread_over_5pct_removes_the_gates_using_that_arm(rule, tmp_path):
    def widen(records):
        for rec in records:
            if (rec["mode"], rec["arm"], rec["k"], rec["round"]) == ("l2l", "A", 16, 1):
                rec["step_s_mean"] = 104.0 * 1.12

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[widen])))
    for gate in ("G2", "G3", "G4"):
        assert out["gates"][gate]["status"] == "no verdict", gate
    assert out["verdict"].startswith("NO VERDICT")


def test_suspend_void_on_b16_removes_g2_only(rule, tmp_path):
    def void(records):
        for rec in records:
            if (rec["mode"], rec["arm"], rec["k"]) == ("l2l", "B", 16):
                rec["suspend"]["void"] = True

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[void])))
    assert out["gates"]["G2"]["status"] == "no verdict"
    assert out["gates"]["G3"]["status"] == "pass"
    assert out["gates"]["G4"]["status"] == "pass"


def test_v2_judges_the_work_not_the_time(rule, tmp_path):
    """L2L at k=1 skips the checkpoint machinery, so it may be faster than the
    shipped step with the same reads and the same arithmetic (smoke, SmolLM2:
    10-20%). V2 compares loads and bytes per step; the time ratio is I6."""
    out = rule.evaluate(_write(tmp_path, timing=_timing(times={("l2l", "B", 1): 5.5})))
    assert out["validity"]["V2"]["ok"] is True
    assert out["info"]["I6"]["b"] == pytest.approx(5.5 / 6.43)
    assert out["verdict"] == "BUILD STEP 1"


def test_v2_fails_when_l2l_k1_loads_differ_from_the_shipped_step(rule, tmp_path):
    def fewer(records):
        for rec in records:
            if (rec["mode"], rec["k"]) == ("l2l", 1):
                rec["layer_loads_per_step"] = 80.0

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[fewer])))
    assert out["validity"]["V2"]["ok"] is False
    for gate in ("G2", "G3", "G4"):
        assert out["gates"][gate]["status"] == "no verdict"


def test_direct_io_off_removes_every_timing_gate(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path, timing=_timing(direct_io=False)))
    assert out["validity"]["V1"]["ok"] is False
    assert out["gates"]["G2"]["status"] == "no verdict"


def test_skipped_k16_removes_g2_to_g4(rule, tmp_path):
    def skip(records):
        for rec in records:
            if rec["k"] == 16:
                rec["skipped"] = "V3: free 12.0 GB < need 10.7 + 4.0 GB"
                rec["step_s_mean"] = None

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[skip])))
    for gate in ("G2", "G3", "G4"):
        assert out["gates"][gate]["status"] == "no verdict"


def test_g1_uses_the_deterministic_rerun_when_a_vs_a_is_inexact(rule, tmp_path):
    out = rule.evaluate(
        _write(tmp_path, f1=_corr("F1", a_vs_a=False, l2l=False), f1_det=_corr("F1", det=True))
    )
    assert out["gates"]["G1"]["status"] == "pass"


def test_g1_no_verdict_when_even_deterministic_is_inexact(rule, tmp_path):
    out = rule.evaluate(
        _write(tmp_path, f1=_corr("F1", a_vs_a=False), f1_det=_corr("F1", a_vs_a=False, det=True))
    )
    assert out["gates"]["G1"]["status"] == "no verdict"


def test_g1_fails_when_l2l_differs_with_exact_a_vs_a(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path, f2=_corr("F2", l2l=False)))
    assert out["gates"]["G1"]["status"] == "fail"


def test_g1_fails_on_an_all_zero_gradient(rule, tmp_path):
    f1 = _corr("F1")
    f1["records"][0]["a_vs_a"]["all_zero"] = ["lora_A.x"]
    out = rule.evaluate(_write(tmp_path, f1=f1))
    assert out["gates"]["G1"]["status"] == "fail"


def test_spill_ratio_is_informational(rule, tmp_path):
    spill = {
        "meta": {"direct_io": True},
        "records": [
            _arm("l2l", "A", 16, 0, 104.0, variant="pinned"),
            _arm("l2l", "A", 16, 1, 104.0, variant="pinned"),
            _arm("l2l", "A", 16, 0, 106.0, variant="spill"),
            _arm("l2l", "A", 16, 1, 106.0, variant="spill"),
        ],
    }
    out = rule.evaluate(_write(tmp_path, spill=spill))
    assert out["info"]["I4"]["ratio"] == pytest.approx(104.0 / 106.0)
    assert out["verdict"] == "BUILD STEP 1"


def test_foreign_reader_on_a16_removes_g2_to_g4(rule, tmp_path):
    def heavy(records):
        for rec in records:
            if (rec["mode"], rec["arm"], rec["k"], rec["round"]) == ("l2l", "A", 16, 0):
                rec["foreign_readers"] = [
                    {"pid": 4, "name": "SearchIndexer.exe", "read_bytes": 4_940_000_000}
                ]

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[heavy])))
    for gate in ("G2", "G3", "G4"):
        assert out["gates"][gate]["status"] == "no verdict", gate
    assert "SearchIndexer.exe" in out["gates"]["G2"]["detail"]


def test_unknown_foreign_readers_do_not_void(rule, tmp_path):
    def unknown(records):
        for rec in records:
            rec["foreign_readers"] = None

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[unknown])))
    assert out["verdict"] == "BUILD STEP 1"


# --------------------------------------------------------------------------
# l2l_probe — pure planning helpers (the GPU paths run in Task 7)
# --------------------------------------------------------------------------
@pytest.fixture()
def probe():
    return _load("l2l_probe")


def test_timing_plan_interleaves_and_alternates(probe):
    plan = probe.plan_timing_arms([1, 2, 16], rounds=2, cd_k=4)
    r0 = [row for row in plan if row[0] == 0]
    r1 = [row for row in plan if row[0] == 1]
    assert r0[:2] == [(0, "plain", "A", 1), (0, "plain", "B", 1)]
    assert r1[:2] == [(1, "plain", "B", 1), (1, "plain", "A", 1)]
    assert [row[3] for row in r0 if row[1] == "l2l" and row[2] in "AB"] == [1, 1, 2, 2, 16, 16]
    assert [row[3] for row in r1 if row[1] == "l2l" and row[2] in "AB"] == [16, 16, 2, 2, 1, 1]
    assert [row[2] for row in r0 if row[1] == "l2l" and row[3] == 2] == ["A", "B"]
    assert [row[2] for row in r1 if row[1] == "l2l" and row[3] == 2] == ["B", "A"]
    assert r0[-2:] == [(0, "l2l", "C", 4), (0, "l2l", "D", 4)]
    assert r1[-2:] == [(1, "l2l", "D", 4), (1, "l2l", "C", 4)]


def test_spill_plan_alternates(probe):
    assert probe.plan_spill_arms(2) == [(0, "pinned"), (0, "spill"), (1, "spill"), (1, "pinned")]


def test_spill_layers_for_the_70b_shape(probe):
    assert probe.spill_layers_for(80, 16, 8) == 40
    assert probe.spill_layers_for(80, 8, 8) == 0
    with pytest.raises(ValueError):
        probe.spill_layers_for(80, 16, 0)


def test_store_allocation_failure_becomes_a_skipped_arm(probe, monkeypatch):
    import torch

    calls = []

    def boom(**_kwargs):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(probe.l2l_activations, "ActivationStore", boom)
    monkeypatch.setattr(probe, "_recover", lambda: calls.append("recovered"))
    store, reason = probe.allocate_store(
        n_layers=4, k=2, chunk_shape=(1, 8, 64), dtype=torch.float32, device="cpu", pin=False
    )
    assert store is None
    assert "out of memory" in reason
    assert calls == ["recovered"]  # drain the stale error AND release the cached pinned blocks


def test_observe_attaches_stamps_watch_and_foreign_readers(probe, monkeypatch):
    snaps = iter(
        [
            {1: {"name": "a.exe", "read": 0}},
            {1: {"name": "a.exe", "read": 2_000_000_000}},
        ]
    )
    monkeypatch.setattr(probe.l2l_box, "process_read_bytes", lambda: next(snaps))
    monkeypatch.setattr(probe.l2l_box, "box_stamp", lambda: {"unix": 0.0})
    result = probe.observe(lambda: {"step_s_mean": 1.0}, interval=0.01)
    assert result["step_s_mean"] == 1.0
    assert result["suspend"]["samples"] >= 2
    assert result["box_before"] == {"unix": 0.0} and result["box_after"] == {"unix": 0.0}
    assert result["foreign_readers"] == [{"pid": 1, "name": "a.exe", "read_bytes": 2_000_000_000}]


@_NEEDS_IO_COUNTERS
def test_observe_stops_the_watch_when_the_body_raises(probe):
    with pytest.raises(ValueError, match="boom"):
        probe.observe(lambda: (_ for _ in ()).throw(ValueError("boom")), interval=0.01)
    assert not any(t.name == "l2l-suspend-watch" for t in threading.enumerate())


def test_fits_in_ram_keeps_the_margin(probe):
    gib8 = 8 * 1024**3
    assert probe.fits_in_ram(gib8, free_gb=8.59 + 4.0 + 0.01) is True
    assert probe.fits_in_ram(gib8, free_gb=8.59 + 3.9) is False
    assert probe.fits_in_ram(gib8, free_gb=None) is False


# --------------------------------------------------------------------------
# final-review fix pass (2026-10-02)
# --------------------------------------------------------------------------
def test_v2_ignores_timing_validity_of_the_k1_arms(rule, tmp_path):
    """I1: V2 is a loads/bytes identity. A 10% round spread on the shipped step
    voids that arm's time (V4), but no gate reads P1's time, so G2-G4 stand."""

    def spread(records):
        for rec in records:
            if (rec["mode"], rec["arm"], rec["round"]) == ("plain", "A", 1):
                rec["step_s_mean"] = 17.1 * 1.10

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[spread])))
    assert out["validity"]["V2"]["ok"] is True
    assert out["verdict"] == "BUILD STEP 1"


def test_v2_fails_when_the_shipped_step_never_ran(rule, tmp_path):
    def skip(records):
        for rec in records:
            if rec["mode"] == "plain":
                rec["skipped"] = "never ran"
                rec["step_s_mean"] = None

    out = rule.evaluate(_write(tmp_path, timing=_timing(patch=[skip])))
    assert out["validity"]["V2"]["ok"] is False
    assert out["gates"]["G2"]["status"] == "no verdict"


def _rec_file(*records):
    return {"meta": {}, "records": list(records)}


def test_g1_no_verdict_when_the_deterministic_rerun_raised(rule, tmp_path):
    """I2: the probe writes no a_vs_a when deterministic mode raises."""
    det = _rec_file(
        {
            "kind": "correctness",
            "fixture": "F1",
            "k": 4,
            "deterministic": True,
            "deterministic_error": "an op does not have a deterministic implementation",
        }
    )
    out = rule.evaluate(_write(tmp_path, f1=_corr("F1", a_vs_a=False), f1_det=det))
    assert out["gates"]["G1"]["status"] == "no verdict"
    assert "deterministic" in out["gates"]["G1"]["detail"]


def test_g1_no_verdict_on_an_empty_correctness_file(rule, tmp_path):
    """I2: a block that crashed leaves Sink's first flush, records == []."""
    out = rule.evaluate(_write(tmp_path, f2=_rec_file()))
    assert out["gates"]["G1"]["status"] == "no verdict"


def test_g1_no_verdict_on_a_skipped_correctness_record(rule, tmp_path):
    reason = "activation store allocation failed: CUDA error: out of memory"
    skipped = _rec_file(
        {"kind": "correctness", "fixture": "F2", "k": 2, "deterministic": False, "skipped": reason}
    )
    out = rule.evaluate(_write(tmp_path, f2=skipped))
    assert out["gates"]["G1"]["status"] == "no verdict"
    assert "skipped" in out["gates"]["G1"]["detail"]


def test_start_refusal_names_the_failing_row(probe):
    """I4: a block whose V1/V3 already fail at its start cannot reach a verdict."""
    v3_bad = {"ok": False, "ram_ok": False, "gpu_ok": True, "free_gb": 9.0, "need_gb": 10.7}
    assert probe.start_refusal("timing", v3_bad, True, "disk").startswith("V3")
    assert "V1" in probe.start_refusal("spill", {"ok": True}, False, "disk")
    assert probe.start_refusal("timing", {"ok": True}, True, "disk") is None
    assert probe.start_refusal("timing", {"ok": True}, None, "ram") is None
    assert probe.start_refusal("correctness", {"ok": False}, False, "disk") is None


def test_close_survives_a_spill_file_held_by_another_process(acts, tmp_path, monkeypatch):
    """M2: an indexer or scanner holding the file must not crash the block."""
    store = _store(acts, tmp_path, spill_layers=1)

    def held(path):
        raise PermissionError(13, "The process cannot access the file", path)

    monkeypatch.setattr(acts.os, "remove", held)
    store.close()
    assert store.stats.spill_left_behind == str(tmp_path / "spill.bin")


def test_release_arm_state_drops_the_zero_weight_cache_after_c_and_d(probe):
    """M3: the nodequant arms cache a zero weight per shape on the GPU (~1.1 GB on
    the 70B shape); later arms must not carry it."""

    class Inst:
        def __init__(self):
            self._zero_cache = {(1, 2): object()}

    for arm, emptied in (("A", False), ("B", False), ("C", True), ("D", True)):
        inst = Inst()
        probe.release_arm_state(inst, arm)
        assert (not inst._zero_cache) is emptied, arm


def test_fits_in_ram_also_needs_commit(probe):
    """M4: pinned memory is charged against Windows commit as well."""
    gib8 = 8 * 1024**3
    assert probe.fits_in_ram(gib8, free_gb=20.0, commit_gb=10.0) is False
    assert probe.fits_in_ram(gib8, free_gb=20.0, commit_gb=13.0) is True
    assert probe.fits_in_ram(gib8, free_gb=20.0) is True


def test_the_l2l_step_calls_each_layer_twice_per_micro_batch(sched, acts, streamed):
    """M9: one no-grad forward and one forward under grad per (layer, micro-batch).
    The wrapper's checkpoint left on would add a recompute: three calls, not two."""
    model, runtime = streamed
    mbs = _micro_batches(2)
    step = sched.L2LStep(model, runtime, sched.capture_layer_call(model, mbs[0]))
    calls = []
    original = runtime.prefetcher.advance

    def counting(idx):
        calls.append(idx)
        return original(idx)

    runtime.prefetcher.advance = counting
    try:
        step.run(mbs, _cpu_store(acts, 2))
    finally:
        runtime.prefetcher.advance = original
    assert len(calls) == 2 * 2 * 3
