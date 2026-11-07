"""Adapter-gradient averaging for multi-node layer streaming (design note, section 6).

``AdapterGradSync`` (``soup_cli/utils/adapter_grad_sync.py``) is the one piece a
streamed model needs to be data-parallel: its base weights sit on ``meta``, so
torch's DDP cannot wrap it, and on CPU accelerate never wraps at all. These tests
pin the sync and the two library facts the design leans on, without a GPU.

What each group pins:

* ``TestAccelerateBypass`` (design T3e, row 5): ``Accelerator.multi_device`` excludes
  ``MULTI_CPU``, so a CPU run gets no gradient sync from accelerate; and a model
  flagged ``_is_accelerate_prepared`` comes back from ``prepare_model`` as the same
  object while an unflagged one does not. That flag is the only thing keeping a GPU
  run from a DDP wrap and a CPU run cannot catch it, so the capability is tested here.
* ``TestHarnessSeesAMissingSync`` (T0): two real gloo processes, no sync, rank-specific
  data: the adapters start equal and end different. It documents row 5 and proves the
  harness would notice a sync that is missing.
* ``TestSyncedRanks`` (T3a/T3b/T8): the same run with the sync. The adapters end
  BITWISE equal on both ranks, the synced gradient equals the mean of the ranks'
  gradients (so a SUM without the divide fails), and the reported ``bytes`` is exactly
  ``4 * n_trainable``.
* ``TestChunking`` / ``TestOneRankHasNoGradient`` (T1d, T3d): a tiny ``chunk_bytes``
  gives the same bits as one chunk; a parameter whose ``grad`` is ``None`` on one rank
  is zero-filled, so the layout stays equal and nothing hangs.
* ``TestFlatten``, ``TestInProcessWorld``, ``TestConstruction``, ``TestNoOp``: the
  layout, the chunk boundaries, the divide by the world size, and the refusals, in
  process with a faked ``all_reduce`` (they run where gloo is missing too).

The two-process tests need torch, transformers, peft, safetensors and gloo and skip
without them. Each rank is a child process of ``_stream_adapter_sync_worker.py`` with
its own shard cache and ``CUDA_VISIBLE_DEVICES=-1``.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from soup_cli.utils.adapter_grad_sync import (  # noqa: E402
    DEFAULT_CHUNK_BYTES,
    AdapterGradSync,
    SyncStats,
    flatten_grads,
    unflatten_grads,
    validate_chunk_bytes,
)

_WORKER = Path(__file__).resolve().with_name("_stream_adapter_sync_worker.py")
_STEPS = 3
_TIMEOUT_SECONDS = 180
_SMALL_CHUNK_BYTES = 1024


# --------------------------------------------------------------------------
# tiny in-process fixtures
# --------------------------------------------------------------------------


def _two_linears():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))


def _give_grads(module, scale=0.5):
    """Distinct, exactly representable gradients: element i of the flat vector is i*scale."""
    offset = 0
    for param in module.parameters():
        count = param.numel()
        values = torch.arange(offset, offset + count, dtype=param.dtype) * scale
        param.grad = values.reshape(param.shape).clone()
        offset += count
    return offset


def _flat(module):
    return torch.cat([param.grad.reshape(-1) for param in module.parameters()])


# --------------------------------------------------------------------------
# validation and construction
# --------------------------------------------------------------------------


class TestChunkBytesValidation:
    @pytest.mark.parametrize("bad", [0, -1, 1, 3])
    def test_less_than_one_fp32_element_is_refused(self, bad):
        with pytest.raises(ValueError, match=r"chunk_bytes must be at least 4 .*got " + str(bad)):
            validate_chunk_bytes(bad)

    @pytest.mark.parametrize("bad", [True, False, 1.5, "8", None])
    def test_a_non_int_is_refused_and_bool_is_not_an_int(self, bad):
        with pytest.raises(ValueError, match=r"chunk_bytes must be an int number of bytes"):
            validate_chunk_bytes(bad)

    def test_constructor_applies_the_same_check(self):
        with pytest.raises(ValueError, match=r"chunk_bytes must be at least 4"):
            AdapterGradSync(_two_linears(), chunk_bytes=0)

    def test_the_smallest_chunk_and_the_default_are_accepted(self):
        assert validate_chunk_bytes(4) == 4
        assert validate_chunk_bytes(DEFAULT_CHUNK_BYTES) == 64 * 1024 * 1024


class TestConstruction:
    def test_collects_trainable_parameters_in_named_parameters_order(self):
        module = _two_linears()
        module[0].bias.requires_grad_(False)
        sync = AdapterGradSync(module)
        assert sync.n_params == 3
        assert sync.n_elements == 12 + 8 + 2
        assert sync.payload_bytes == 4 * 22
        assert [name for name, _p in sync._params] == ["0.weight", "1.weight", "1.bias"]

    def test_accepts_an_iterable_of_named_parameters(self):
        module = _two_linears()
        sync = AdapterGradSync(iter(module.named_parameters()))
        assert sync.n_params == 4

    def test_mixed_dtypes_are_refused_naming_both_parameters(self):
        module = _two_linears()
        module[1].to(torch.bfloat16)
        with pytest.raises(
            ValueError,
            match=r"one dtype across the trainable parameters, but '0\.weight' is "
            r"torch\.float32 and '1\.weight' is torch\.bfloat16",
        ):
            AdapterGradSync(module)

    def test_a_frozen_parameter_of_another_dtype_is_not_a_mixed_dtype(self):
        module = _two_linears()
        module[1].to(torch.bfloat16)
        for param in module[1].parameters():
            param.requires_grad_(False)
        assert AdapterGradSync(module).n_params == 2

    def test_no_trainable_parameter_is_refused(self):
        module = _two_linears()
        module.requires_grad_(False)
        with pytest.raises(ValueError, match=r"found no trainable parameters"):
            AdapterGradSync(module)

    def test_meta_parameters_are_refused(self):
        module = torch.nn.Linear(2, 2, device="meta")
        with pytest.raises(ValueError, match=r"cannot synchronise parameters on the meta device"):
            AdapterGradSync(module)

    def test_parameters_on_two_devices_are_refused(self):
        named = [
            ("real", torch.nn.Parameter(torch.zeros(2))),
            ("ghost", torch.nn.Parameter(torch.zeros(2, device="meta"))),
        ]
        with pytest.raises(ValueError, match=r"'real' is on cpu and 'ghost' is on meta"):
            AdapterGradSync(named)


# --------------------------------------------------------------------------
# flatten / unflatten (the layout every rank must agree on)
# --------------------------------------------------------------------------


class TestFlatten:
    def test_layout_is_the_gradients_in_named_parameters_order(self):
        module = _two_linears()
        _give_grads(module)
        params = list(module.named_parameters())
        flat = flatten_grads(params)
        assert flat.dtype == torch.float32
        assert flat.tolist() == [index * 0.5 for index in range(26)]

    def test_a_none_grad_is_written_as_zeros_over_whatever_the_buffer_held(self):
        """Mutation target (T3d): without the zero-fill the slot keeps the old bytes."""
        module = _two_linears()
        _give_grads(module)
        module[0].bias.grad = None  # elements 12..15 of the flat vector
        poisoned = torch.full((26,), float("nan"))
        flat = flatten_grads(list(module.named_parameters()), out=poisoned)
        assert flat is poisoned
        assert flat[12:16].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert not torch.isnan(flat).any(), "a slot was left out of the buffer"
        assert flat[:12].tolist() == [index * 0.5 for index in range(12)]
        assert flat[16:].tolist() == [index * 0.5 for index in range(16, 26)]

    def test_every_rank_gets_the_same_length_whichever_grads_are_missing(self):
        module = _two_linears()
        params = list(module.named_parameters())
        lengths = set()
        for missing in (set(), {0}, {1, 3}, {0, 1, 2, 3}):
            for index, (_name, param) in enumerate(params):
                param.grad = None if index in missing else torch.ones_like(param)
            lengths.add(flatten_grads(params).numel())
        assert lengths == {26}

    def test_a_buffer_of_the_wrong_size_or_dtype_is_refused(self):
        module = _two_linears()
        _give_grads(module)
        params = list(module.named_parameters())
        with pytest.raises(ValueError, match=r"flatten buffer must be 26 fp32 elements, got 25"):
            flatten_grads(params, out=torch.zeros(25))
        with pytest.raises(ValueError, match=r"flatten buffer must be 26 fp32 elements"):
            flatten_grads(params, out=torch.zeros(26, dtype=torch.float64))

    def test_a_gradient_of_the_wrong_size_is_refused_by_name(self):
        # ``Parameter.grad = ...`` itself refuses a grad of another shape, so a stand-in
        # plays the exotic parameter (e.g. a sparse gradient) that could reach the check.
        odd = SimpleNamespace(device=torch.device("cpu"), numel=lambda: 8, grad=torch.zeros(3))
        with pytest.raises(
            ValueError, match=r"gradient of 'odd' has 3 elements, the parameter has 8"
        ):
            flatten_grads([("odd", odd)])

    def test_unflatten_writes_in_place_and_creates_missing_grads_without_aliasing(self):
        module = _two_linears()
        _give_grads(module)
        module[0].bias.grad = None
        params = list(module.named_parameters())
        kept = module[0].weight.grad
        kept_ptr = kept.data_ptr()
        flat = torch.arange(26, dtype=torch.float32) + 100.0
        unflatten_grads(params, flat)
        assert module[0].weight.grad is kept and kept.data_ptr() == kept_ptr
        assert module[0].bias.grad is not None
        assert module[0].bias.grad.shape == module[0].bias.shape
        assert module[0].bias.grad.tolist() == [112.0, 113.0, 114.0, 115.0]
        flat.zero_()
        assert module[0].bias.grad.tolist() == [112.0, 113.0, 114.0, 115.0], "grad aliases flat"
        assert kept.reshape(-1).tolist() == [100.0 + index for index in range(12)]

    def test_a_flat_buffer_of_the_wrong_size_is_refused_on_the_way_back(self):
        module = _two_linears()
        with pytest.raises(ValueError, match=r"flat gradient buffer has 5 elements, expected 26"):
            unflatten_grads(list(module.named_parameters()), torch.zeros(5))

    def test_round_trip_is_exact_for_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            module = _two_linears().to(dtype)
            _give_grads(module)
            before = [param.grad.clone() for param in module.parameters()]
            params = list(module.named_parameters())
            unflatten_grads(params, flatten_grads(params))
            for old, param in zip(before, module.parameters()):
                assert param.grad.dtype == dtype
                assert torch.equal(param.grad, old)


# --------------------------------------------------------------------------
# no group -> nothing happens
# --------------------------------------------------------------------------


class TestNoOp:
    def test_without_a_process_group_the_sync_does_nothing(self):
        import torch.distributed as dist

        assert not dist.is_initialized(), "another test left a process group behind"
        module = _two_linears()
        _give_grads(module)
        before = [param.grad.clone() for param in module.parameters()]
        identities = [id(param.grad) for param in module.parameters()]
        stats = AdapterGradSync(module).sync()
        assert stats == SyncStats(bytes=0, seconds=0.0, n_params=0, n_chunks=0)
        assert [id(param.grad) for param in module.parameters()] == identities
        for old, param in zip(before, module.parameters()):
            assert torch.equal(param.grad, old)

    def test_a_world_of_one_does_nothing_and_leaves_missing_grads_missing(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("installed torch has no CPU Gloo backend")
        module = _two_linears()
        _give_grads(module)
        module[0].bias.grad = None
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1
        )
        try:
            stats = AdapterGradSync(module).sync()
        finally:
            dist.destroy_process_group()
        assert stats == SyncStats()
        assert module[0].bias.grad is None
        assert module[0].weight.grad.reshape(-1).tolist() == [index * 0.5 for index in range(12)]

    def test_the_stats_are_frozen(self):
        with pytest.raises(AttributeError):
            SyncStats().bytes = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# the sync itself, in process, against a faked all_reduce
# --------------------------------------------------------------------------


def _fake_world(monkeypatch, peer_flat, world_size):
    """Pretend ``world_size`` ranks: ``all_reduce`` adds ``peer_flat`` (the others' sum)."""
    import torch.distributed as dist

    calls = []
    state = {"offset": 0}

    def all_reduce(tensor, op=None, group=None):
        assert op == dist.ReduceOp.SUM, f"the mean must be SUM then divide, got {op}"
        count = tensor.numel()
        start = state["offset"] % peer_flat.numel()  # one sync is one pass over peer_flat
        tensor.add_(peer_flat[start : start + count])
        state["offset"] += count
        calls.append(count)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    return calls


class TestInProcessWorld:
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_the_result_is_the_mean_over_the_ranks_not_the_sum(self, monkeypatch, world_size):
        module = _two_linears()
        _give_grads(module)
        own = _flat(module).clone()
        peer = torch.arange(26, dtype=torch.float32) * 0.25
        _fake_world(monkeypatch, peer, world_size)
        stats = AdapterGradSync(module).sync()
        assert torch.equal(_flat(module), (own + peer) / world_size)
        assert stats.bytes == 4 * 26
        assert stats.n_params == 4
        assert stats.n_chunks == 1
        assert stats.seconds > 0

    def test_chunks_are_at_most_chunk_bytes_and_cover_everything_once(self, monkeypatch):
        module = _two_linears()
        _give_grads(module)
        own = _flat(module).clone()
        peer = torch.arange(26, dtype=torch.float32)
        calls = _fake_world(monkeypatch, peer, 2)
        stats = AdapterGradSync(module, chunk_bytes=40).sync()  # 10 fp32 elements a chunk
        assert calls == [10, 10, 6]
        assert stats.n_chunks == 3
        assert stats.bytes == 4 * 26, "bytes is the payload, not the sum of the chunks' padding"
        assert torch.equal(_flat(module), (own + peer) / 2)

    def test_a_chunk_size_that_is_not_a_multiple_of_four_rounds_down(self, monkeypatch):
        module = _two_linears()
        _give_grads(module)
        calls = _fake_world(monkeypatch, torch.zeros(26), 2)
        AdapterGradSync(module, chunk_bytes=43).sync()
        assert max(calls) == 10 and sum(calls) == 26

    def test_a_missing_grad_is_contributed_as_zeros_and_comes_back_averaged(self, monkeypatch):
        module = _two_linears()
        _give_grads(module)
        module[0].bias.grad = None  # elements 12..15
        peer = torch.full((26,), 8.0)
        _fake_world(monkeypatch, peer, 2)
        AdapterGradSync(module).sync()
        assert module[0].bias.grad.tolist() == [4.0, 4.0, 4.0, 4.0]
        assert module[0].weight.grad.reshape(-1).tolist() == [
            (index * 0.5 + 8.0) / 2 for index in range(12)
        ]

    def test_repeated_syncs_reuse_the_same_grad_tensors(self, monkeypatch):
        module = _two_linears()
        _give_grads(module)
        _fake_world(monkeypatch, torch.zeros(26), 2)
        sync = AdapterGradSync(module)
        identities = [id(param.grad) for param in module.parameters()]
        sync.sync()
        sync.sync()
        assert [id(param.grad) for param in module.parameters()] == identities


# --------------------------------------------------------------------------
# accelerate: T3e and design row 5
# --------------------------------------------------------------------------


@pytest.fixture
def accelerate_cpu(monkeypatch):
    pytest.importorskip("accelerate")
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR"):
        monkeypatch.delenv(name, raising=False)
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    try:
        yield Accelerator(cpu=True)
    finally:
        AcceleratorState._reset_state(True)
        GradientState._reset_state()


class TestAccelerateBypass:
    def test_multi_cpu_gets_no_ddp_so_no_gradient_sync_from_accelerate(self):
        """Design row 5: a CPU 2-process run trains two unsynchronised models."""
        pytest.importorskip("accelerate")
        from accelerate import Accelerator
        from accelerate.utils import DistributedType

        multi_device = Accelerator.multi_device.fget
        cpu = SimpleNamespace(use_distributed=True, distributed_type=DistributedType.MULTI_CPU)
        gpu = SimpleNamespace(use_distributed=True, distributed_type=DistributedType.MULTI_GPU)
        single = SimpleNamespace(use_distributed=False, distributed_type=DistributedType.MULTI_GPU)
        assert multi_device(cpu) is False
        assert multi_device(gpu) is True
        assert multi_device(single) is False

    def test_a_flagged_model_comes_back_unwrapped_and_an_unflagged_one_does_not(
        self, accelerate_cpu, monkeypatch
    ):
        from accelerate import Accelerator

        monkeypatch.setattr(Accelerator, "multi_device", property(lambda self: True))
        assert accelerate_cpu.multi_device is True

        flagged = torch.nn.Linear(2, 2)
        flagged._is_accelerate_prepared = True
        returned = accelerate_cpu.prepare_model(flagged)
        assert returned is flagged
        assert not isinstance(returned, torch.nn.parallel.DistributedDataParallel)

        # The unflagged control goes on to build the DDP wrapper; with no process group
        # that raises (torch 2.14), which is the proof the flag alone skipped it.
        control = torch.nn.Linear(2, 2)
        with pytest.raises(ValueError, match=r"Default process group has not been initialized"):
            accelerate_cpu.prepare_model(control)


# --------------------------------------------------------------------------
# two real gloo processes
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "<no output>"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _run_pair(out_dir: Path, mode: str, *extra: str):
    """Run rank 0 and rank 1 as child processes; return their result rows and tensors."""
    distributed = pytest.importorskip("torch.distributed")
    if not distributed.is_gloo_available():
        pytest.skip("installed torch has no CPU Gloo backend")
    if sys.platform == "darwin":
        # Same reason as tests/test_distributed_base_revision.py: two-process
        # rendezvous can time out on loaded macOS runners.
        pytest.skip("two-process Gloo runs are skipped on macOS (runner timeouts)")
    for module in ("transformers", "peft", "safetensors"):
        pytest.importorskip(module)
    from safetensors.torch import load_file

    import soup_cli

    source_root = str(Path(soup_cli.__file__).resolve().parent.parent)
    env = os.environ.copy()
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)  # a launcher variable left in the shell must not reach the ranks
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONPATH=os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH", "")])),
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="1",
    )
    port = _free_port()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = [out_dir / f"rank{rank}.log" for rank in range(2)]
    handles, processes = [], []
    failure = None
    try:
        for rank in range(2):
            handle = logs[rank].open("w", encoding="utf-8")
            handles.append(handle)
            command = [sys.executable, str(_WORKER)]
            for flag, value in (
                ("--rank", rank),
                ("--world-size", 2),
                ("--port", port),
                ("--mode", mode),
                ("--steps", _STEPS),
                ("--out-dir", out_dir),
            ):
                command += [flag, str(value)]
            command += extra
            processes.append(
                subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while failure is None and any(process.poll() is None for process in processes):
            for rank, process in enumerate(processes):
                if process.poll() not in (None, 0):
                    failure = f"rank {rank} exited with {process.poll()}"
            if failure is None and time.monotonic() > deadline:
                failure = f"still running after {_TIMEOUT_SECONDS} s (a hang)"
            time.sleep(0.2)
        for rank, process in enumerate(processes):
            if failure is None and process.returncode != 0:
                failure = f"rank {rank} exited with {process.returncode}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        for handle in handles:
            handle.close()
    if failure is not None:
        report = "\n".join(
            f"--- rank {rank} (last 40 lines) ---\n{_tail(logs[rank])}" for rank in range(2)
        )
        pytest.fail(f"{mode} run failed: {failure}\n{report}", pytrace=False)
    rows = [json.loads((out_dir / f"rank{rank}.json").read_text("utf-8")) for rank in range(2)]
    tensors = [load_file(str(out_dir / f"rank{rank}.safetensors")) for rank in range(2)]
    return SimpleNamespace(rows=rows, tensors=tensors)


@pytest.fixture(scope="module")
def pair_without_sync(tmp_path_factory):
    return _run_pair(tmp_path_factory.mktemp("no-sync"), "none")


@pytest.fixture(scope="module")
def pair_synced(tmp_path_factory):
    return _run_pair(tmp_path_factory.mktemp("synced"), "sync")


@pytest.fixture(scope="module")
def pair_chunked(tmp_path_factory):
    return _run_pair(
        tmp_path_factory.mktemp("chunked"), "sync", "--chunk-bytes", str(_SMALL_CHUNK_BYTES)
    )


@pytest.fixture(scope="module")
def pair_one_rank_without_a_grad(tmp_path_factory):
    return _run_pair(tmp_path_factory.mktemp("unused"), "sync", "--unused-on-rank1")


def _tensors_equal(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


class TestHarnessSeesAMissingSync:
    def test_without_the_sync_equal_ranks_drift_apart(self, pair_without_sync):
        """T0: the harness can tell a missing sync from a working one (design row 5)."""
        zero, one = pair_without_sync.rows
        assert zero["n_trainable"] == one["n_trainable"] > 0
        assert zero["initial_sha"] == one["initial_sha"], "the ranks did not start equal"
        assert zero["final_sha"] != zero["initial_sha"], "vacuous: the adapters never moved"
        assert zero["final_sha"] != one["final_sha"], "ranks stayed equal with no sync at all"
        assert zero["losses"][0] != one["losses"][0], "the ranks saw the same rows"
        left, right = pair_without_sync.tensors
        assert left.keys() == right.keys()
        assert not _tensors_equal(left, right)


class TestSyncedRanks:
    def test_the_adapters_end_bitwise_equal_on_both_ranks(self, pair_synced):
        zero, one = pair_synced.rows
        assert zero["initial_sha"] == one["initial_sha"], "the ranks did not start equal"
        assert zero["final_sha"] != zero["initial_sha"], "vacuous: the adapters never moved"
        assert zero["final_sha"] == one["final_sha"]
        assert _tensors_equal(*pair_synced.tensors)

    def test_the_synced_gradient_is_the_mean_of_the_ranks_gradients(self, pair_synced):
        """Mutation target (T3b): SUM without the divide makes every step's check False."""
        for row in pair_synced.rows:
            assert row["grads_nonzero"], "vacuous: a rank's gradient was all zeros"
            assert row["ranks_differ_every_step"], "vacuous: both ranks computed one gradient"
            assert row["matches_mean"]

    def test_the_wire_accounting_is_exact(self, pair_synced):
        """T8: ``bytes`` is 4 * n_trainable (one rank's payload), in one chunk, taking time."""
        for row in pair_synced.rows:
            stats = row["stats"]
            assert row["n_params"] == 8  # 2 layers x (q_proj, v_proj) x (lora_A, lora_B)
            assert stats["n_params"] == 8
            assert stats["bytes"] == 4 * row["n_trainable"] > 0
            assert stats["n_chunks"] == 1
            assert stats["seconds"] > 0


class TestChunking:
    def test_several_chunks_give_the_bits_one_chunk_gives(self, pair_chunked, pair_synced):
        zero, one = pair_chunked.rows
        expected_chunks = math.ceil(4 * zero["n_trainable"] / _SMALL_CHUNK_BYTES)
        assert expected_chunks > 1, "vacuous: the tiny model fits in one chunk"
        for row in pair_chunked.rows:
            assert row["stats"]["n_chunks"] == expected_chunks
            assert row["stats"]["bytes"] == 4 * row["n_trainable"]
            assert row["matches_mean"]
        assert zero["final_sha"] == one["final_sha"]
        assert zero["final_sha"] == pair_synced.rows[0]["final_sha"]
        assert _tensors_equal(pair_chunked.tensors[0], pair_synced.tensors[0])


class TestOneRankHasNoGradient:
    def test_a_missing_grad_on_one_rank_keeps_the_layout_and_does_not_hang(
        self, pair_one_rank_without_a_grad
    ):
        """T1d, mutation target (T3d): the dropped slot must still be sent, as zeros."""
        zero, one = pair_one_rank_without_a_grad.rows
        assert zero["final_sha"] == one["final_sha"]
        assert _tensors_equal(*pair_one_rank_without_a_grad.tensors)
        for row in (zero, one):
            assert row["matches_mean"]
            assert row["stats"]["bytes"] == 4 * row["n_trainable"]
