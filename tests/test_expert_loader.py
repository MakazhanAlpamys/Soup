"""Native reader selection, packed shard parity and lease lifetime."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import expert_loader
from tests.test_stream_expert_layout import _cfg, _deepseek_cfg

FIELD = "training.stream_expert_loader"
NATIVE = "cpp_async_2slot"
OPTION = f"{FIELD}='{NATIVE}'"


def test_default_loader_never_probes_the_extension(monkeypatch):
    def unexpected():
        pytest.fail("the default loader probed native dependencies")

    monkeypatch.setattr(expert_loader, "_load_extension", unexpected)
    assert load_config_from_string(_cfg()).training.stream_expert_loader == "async_lease"
    assert expert_loader.resolve_expert_loader(device="cpu") == "async_lease"


def test_config_accepts_the_native_loader_for_per_expert_streaming():
    cfg = load_config_from_string(
        _cfg(stream_layers="true", stream_expert_layout="per_expert", stream_expert_loader=NATIVE)
    )
    assert cfg.training.stream_expert_loader == NATIVE


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, f"{FIELD} set but stream_layers is false"),
        (
            {"stream_layers": "true"},
            f"{OPTION} requires training.stream_expert_layout='per_expert'",
        ),
    ],
)
def test_config_refuses_an_unused_native_loader(options, message):
    with pytest.raises(ValueError) as caught:
        load_config_from_string(_cfg(stream_expert_loader=NATIVE, **options))
    assert message in str(caught.value)


def test_config_refuses_an_unknown_loader():
    with pytest.raises(ValueError) as caught:
        load_config_from_string(_cfg(stream_expert_loader="native"))
    assert "stream_expert_loader" in str(caught.value)
    assert "'async_lease' or 'cpp_async_2slot'" in str(caught.value)


@pytest.mark.parametrize(("device", "available"), [("cpu", True), ("cuda:0", False)])
def test_native_refusal_names_cuda_before_opening_the_extension(monkeypatch, device, available):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(
        expert_loader, "_load_extension", lambda: pytest.fail("extension was opened")
    )
    with pytest.raises(expert_loader.CppAsyncLoaderUnavailableError) as caught:
        expert_loader.resolve_expert_loader(NATIVE, device=device)
    assert str(caught.value) == f"{OPTION} requires CUDA on the target device; got {device!r}"


@pytest.mark.parametrize("failure", [ModuleNotFoundError("module absent"), OSError("ABI mismatch")])
def test_native_refusal_names_the_extension_and_build_command(monkeypatch, failure):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def broken_module():
        raise failure

    monkeypatch.setattr(expert_loader, "_load_extension", broken_module)
    with pytest.raises(expert_loader.CppAsyncLoaderUnavailableError) as caught:
        expert_loader.resolve_expert_loader(NATIVE, device="cuda")
    message = str(caught.value)
    assert f"{OPTION} requires the compiled extension _native_expert_stream_async" in message
    assert "python -m soup_cli.utils.native_expert_reader.build_ext build_ext" in message
    assert str(failure) in message


@pytest.mark.parametrize("entry", ["build_streamed_model", "install_streaming"])
def test_runtime_refuses_cpu_before_accessing_the_model(entry):
    from soup_cli.utils import layer_stream_runtime as runtime

    pytest.importorskip("torch")
    func = getattr(runtime, entry)
    model_arg = {"model_id": "unused", "dtype": "float32"} if entry.startswith("build") else {
        "model": None,
    }
    with pytest.raises(expert_loader.CppAsyncLoaderUnavailableError) as caught:
        func(
            **model_arg,
            shard_dir="unused",
            index=SimpleNamespace(expert_layout="per_expert"),
            device="cpu",
            expert_loader=NATIVE,
        )
    assert str(caught.value) == f"{OPTION} requires CUDA on the target device; got 'cpu'"


def test_runtime_refuses_native_and_python_prefetch_before_setup():
    from soup_cli.utils.layer_stream_runtime import install_streaming

    with pytest.raises(ValueError) as caught:
        install_streaming(
            None,
            shard_dir="unused",
            index=SimpleNamespace(expert_layout="per_expert"),
            expert_loader=NATIVE,
            expert_prefetch=True,
        )
    assert str(caught.value) == f"{OPTION} cannot be combined with expert_prefetch=True"


def test_runtime_refuses_native_on_a_fused_cache_before_setup():
    from soup_cli.utils.layer_stream_runtime import install_streaming

    with pytest.raises(ValueError) as caught:
        install_streaming(
            None, shard_dir="unused", index=SimpleNamespace(expert_layout=""), expert_loader=NATIVE,
        )
    assert str(caught.value) == f"{OPTION} needs a cache sharded with expert_layout='per_expert'"


def test_training_refuses_native_before_config_tokenizer_or_shard_access(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    calls = {}
    with pytest.raises(expert_loader.CppAsyncLoaderUnavailableError) as caught:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(),
            load_calls=calls,
            tcfg_overrides={"stream_expert_layout": "per_expert", "stream_expert_loader": NATIVE},
        )
    assert str(caught.value) == f"{OPTION} requires CUDA on the target device; got 'cpu'"
    assert calls == {}


def test_training_forwards_native_to_model_build_and_charges_the_pool(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    monkeypatch.setattr(expert_loader, "resolve_expert_loader", lambda loader, **kwargs: loader)
    monkeypatch.setattr(
        expert_loader, "native_expert_buffer_requirements", lambda *a, **k: (80, 64),
    )
    calls = {}
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(), load_calls=calls,
        tcfg_overrides={"stream_expert_layout": "per_expert", "stream_expert_loader": NATIVE},
    )
    assert calls["build"][0]["expert_loader"] == NATIVE
    assert calls["budget"][0]["expert_pool_bytes"] == 80


def test_training_refuses_when_native_host_buffers_exceed_available_ram(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    monkeypatch.setattr(expert_loader, "resolve_expert_loader", lambda loader, **kwargs: loader)
    monkeypatch.setattr(
        expert_loader, "native_expert_buffer_requirements", lambda *a, **k: (80, 200),
    )
    with pytest.raises(ValueError) as caught:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(), free_ram=100,
            tcfg_overrides={"stream_expert_layout": "per_expert", "stream_expert_loader": NATIVE},
        )
    assert str(caught.value) == (
        f"{OPTION} needs 200 bytes of host staging, but only 100 bytes of RAM are available"
    )


class ThreadedPackedStream:
    """Read the registered byte ranges in a worker and copy into the consumer's buffer."""

    def __init__(self, plan):
        self.plan = plan
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.closed = False
        self.fail = False
        self.wrong_expert = False
        self.finishes = 0

    def start_sequence(self, layer, experts):
        if self.pending is not None:
            raise RuntimeError("active sequence")
        entry = self.plan.layers[layer]
        self.pending = iter([self.worker.submit(self._read, entry, expert) for expert in experts])

    def _read(self, entry, expert):
        if self.fail:
            raise OSError("injected shard read failure")
        block = bytearray(self.plan.packed_bytes)
        with open(entry.reader.path, "rb") as shard:
            for start, end, offset in entry.ranges[expert]:
                shard.seek(start)
                block[offset : offset + end - start] = shard.read(end - start)
        return expert, block

    def consume_next(self, destination):
        import ctypes

        expert, block = next(self.pending).result()
        ctypes.memmove(destination, bytes(block), len(block))
        return expert + int(self.wrong_expert)

    def finish_sequence(self):
        for future in self.pending:
            future.cancel()
            if not future.cancelled():
                future.exception()
        self.pending = None
        self.finishes += 1

    def close(self):
        assert self.pending is None
        assert not self.closed
        self.closed = True
        self.worker.shutdown(wait=True)


@pytest.fixture
def reader_factory(tmp_path):
    pytest.importorskip("torch")
    from tests.test_expert_buffer_pool import _experts, _store

    def create(nf4=False, folder="reader", nested=True):
        path = tmp_path / folder
        path.mkdir()
        return _store(path, _experts(count=4), nf4=nf4, nested=nested, with_pool=False)

    return create


@pytest.mark.parametrize("nf4", [False, True])
@pytest.mark.parametrize("compiled", [False, True], ids=["threaded", "compiled"])
def test_stream_and_pool_match_python_stored_bytes_and_restored_weights(
    reader_factory, nf4, compiled
):
    import torch

    from soup_cli.utils.expert_shard import aligned_staging
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from tests.test_native_expert_reader_port import extension_or_skip

    if compiled:
        extension_or_skip()
    reader, _, codes = reader_factory(nf4)
    _, staging = aligned_staging(reader.max_span)
    stream = (
        expert_loader.open_native_stream({8: reader})
        if compiled
        else make_fake_stream({8: reader})
    )
    slot = torch.empty(stream.plan.packed_bytes, dtype=torch.uint8)
    plan = plan_expert_buffer(reader, dtype="float32")
    options = dict(
        device="cpu", dtype="float32", weight_budget_bytes=plan.weight_bytes, codes=codes,
    )
    pool = ExpertBufferPool(reader, stream=stream, **options)
    baseline = ExpertBufferPool(reader, **options)
    try:
        for expert in (2, 0, 3, 0):
            expected = reader.read_host(expert, staging)
            assert stream.read_into(reader, expert, slot) == sum(
                tensor.numel() * tensor.element_size() for tensor in expected.values()
            )
            got = stream.host_views(reader, expert, slot)
            assert list(got) == list(expected)
            assert all(torch.equal(got[name], expected[name]) for name in expected)
            with baseline.lease(reader, expert) as want, pool.lease(reader, expert) as actual:
                assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(want, actual, strict=True))
        assert pool.fetches == baseline.fetches == 4
        assert pool.logical_bytes_read == baseline.logical_bytes_read > 0
    finally:
        pool.close()
        baseline.close()
        reader.close()


def make_fake_stream(readers):
    plan = expert_loader.plan_packed_stream(readers)
    return expert_loader.CppPackedStream(ThreadedPackedStream(plan), plan)


def test_partial_sequence_cancels_and_releases_the_worker(reader_factory):
    import torch

    reader, _, _ = reader_factory()
    stream = make_fake_stream({9: reader})
    slot = torch.empty(stream.plan.packed_bytes, dtype=torch.uint8)
    try:
        sequence = stream.sequence(reader, [3, 1, 2], slot)
        assert next(sequence) == 3
        with pytest.raises(RuntimeError, match="^active sequence$"):
            stream.read_into(reader, 0, slot)
        sequence.close()
        assert list(stream.sequence(reader, [2, 0], slot)) == [2, 0]
        assert stream.handle.finishes == 2
    finally:
        stream.close()
    assert stream.handle.closed


def test_native_pool_failure_releases_lease_and_counts_only_success(reader_factory, monkeypatch):
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer

    reader, _, codes = reader_factory(nf4=True)
    stream = make_fake_stream({0: reader})
    plan = plan_expert_buffer(reader, dtype="float32")
    pool = ExpertBufferPool(
        reader, stream=stream, device="cpu", dtype="float32", codes=codes,
        weight_budget_bytes=plan.weight_bytes,
    )
    pool.phase_read_tracking = True
    try:
        stream.handle.fail = True
        with pytest.raises(OSError, match="^injected shard read failure$"), pool.lease(reader, 0):
            pytest.fail("a failed lease exposed weights")
        stream.handle.fail = False
        stream.handle.wrong_expert = True
        with pytest.raises(RuntimeError, match="returned expert 1 where the sequence expects 0"):
            with pool.lease(reader, 0):
                pytest.fail("a wrong expert exposed weights")
        stream.handle.wrong_expert = False

        def fail_restore(*args):
            raise RuntimeError("injected restore failure")

        with monkeypatch.context() as patch:
            patch.setattr(pool, "_dequantize", fail_restore)
            with pytest.raises(RuntimeError, match="^injected restore failure$"):
                with pool.lease(reader, 0, phase="backward_dX", layer_idx=1):
                    pytest.fail("a failed restore exposed weights")
        assert pool.fetches == pool.last_lease_bytes == pool.logical_bytes_read == 0
        assert pool.phase_read_snapshot()[1]["backward_dX"] == {
            "expert_fetches": 0, "expert_bytes_read": 0,
        }
        with pool.lease(reader, 0):
            with pytest.raises(RuntimeError, match="cannot close an expert pool with an active"):
                pool.close()
        assert pool.fetches == 1 and pool.logical_bytes_read > 0
        assert stream.handle.finishes == 4
    finally:
        pool.close()
    pool.close()
    assert stream.handle.closed


def test_native_registration_rejects_a_file_replaced_since_its_header(
    reader_factory, monkeypatch
):
    reader, _, _ = reader_factory()
    replacement, _, _ = reader_factory(folder="replacement")
    Path(replacement.path).replace(reader.path)
    module = SimpleNamespace(PackedStreamAsync=lambda *args: pytest.fail("registered stale bytes"))
    monkeypatch.setattr(expert_loader, "_load_extension", lambda: module)
    with pytest.raises(ValueError, match="changed on disk since its header was read"):
        expert_loader.open_native_stream({0: reader})


def test_native_pool_forward_trace_uses_the_successful_lease_bytes(reader_factory):
    import torch
    from torch.nn.functional import silu

    from soup_cli.utils.expert_stream import (
        ExpertBufferPool,
        plan_expert_buffer,
        streamed_experts_class,
    )

    reader, _, codes = reader_factory()
    stream = make_fake_stream({0: reader})
    plan = plan_expert_buffer(reader, dtype="float32")
    pool = ExpertBufferPool(
        reader, stream=stream, device="cpu", dtype="float32", codes=codes,
        weight_budget_bytes=plan.weight_bytes,
    )
    try:
        module = streamed_experts_class()(
            reader, pool, silu, layer_idx=0, grouped_expert_routes=True,
        )
        hidden = torch.ones(2, reader.shapes["down"][0])
        module(hidden, torch.tensor([[0], [1]]), torch.ones(2, 1))
        assert reader.bytes_read == 0
        traced = sum(row["logical_bytes_read"] for row in module.last_trace)
        assert traced == pool.logical_bytes_read > 0
        stream.handle.fail = True
        with pytest.raises(OSError, match="^injected shard read failure$"):
            module(hidden, torch.tensor([[0], [1]]), torch.ones(2, 1))
        assert len(module.last_trace) == 1
        assert module.last_trace[0]["status"] == "error"
        assert module.last_trace[0]["logical_bytes_read"] == 0
        assert pool.logical_bytes_read == traced
    finally:
        pool.close()


def test_native_install_closes_the_stream_if_pool_setup_fails(reader_factory, monkeypatch):
    from soup_cli.utils import expert_stream, layer_shard, layer_stream_runtime

    reader, _, _ = reader_factory()
    stream = make_fake_stream({0: reader})
    closed_readers = []

    def pool_failure(*args, **kwargs):
        assert kwargs["stream"] is stream
        raise RuntimeError("injected expert pool setup failure")

    monkeypatch.setattr(expert_stream, "ExpertShardReader", lambda *args, **kwargs: reader)
    monkeypatch.setattr(reader, "close", lambda: closed_readers.append(reader))
    monkeypatch.setattr(layer_shard, "expert_paths", lambda *args: {0: reader.path})
    monkeypatch.setattr(expert_loader, "open_native_stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(expert_stream, "ExpertBufferPool", pool_failure)
    index = SimpleNamespace(
        expert_layout="per_expert", expert_count=reader.count, expert_shapes=reader.shapes,
        expert_quant_specs={}, dtype="float32",
    )
    with pytest.raises(RuntimeError, match="^injected expert pool setup failure$"):
        layer_stream_runtime._install_expert_streaming(
            [], "unused", index, device="cuda", codes={}, expert_loader=NATIVE,
        )
    assert stream.handle.closed
    assert closed_readers == [reader]


def test_native_budget_counts_the_ring_workers_and_cpu_destination(reader_factory):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    reader, _, _ = reader_factory()
    stream = make_fake_stream({0: reader})
    try:
        native = expert_loader.plan_native_expert_buffer(
            reader, stream.plan, dtype="float32", transfer="sync",
        )
        python = plan_expert_buffer(reader, dtype="float32")
        block = stream.plan.packed_bytes
        scratch = stream.plan.layers[0].staging_bytes
        assert native.host_staging_bytes == 3 * block + 2 * scratch
        assert native.weight_bytes == python.weight_bytes
    finally:
        stream.close()


def test_failed_layer_install_closes_an_already_installed_native_pool(reader_factory, monkeypatch):
    from soup_cli.utils import layer_stream_runtime as runtime

    reader, _, _ = reader_factory()
    stream = make_fake_stream({0: reader})
    owner = SimpleNamespace(layers=[object()])
    monkeypatch.setattr(runtime, "_validate_expert_loader", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "decoder_owner", lambda *a: owner)
    monkeypatch.setattr(runtime, "_require_stripe_folders", lambda *a: None)
    monkeypatch.setattr(runtime, "_install_expert_streaming", lambda *a, **k: (stream, (reader,)))
    monkeypatch.setattr(runtime, "_layer_name_map", lambda layer: {})
    with pytest.raises(RuntimeError, match="no meta decoder weights found"):
        runtime.install_streaming(
            None, shard_dir="unused", index=SimpleNamespace(n_layers=1), expert_loader=NATIVE,
        )
    assert stream.handle.closed


@pytest.mark.gpu
@pytest.mark.parametrize("nested", [False, True])
def test_compiled_reader_feeds_the_existing_cuda_nf4_restore(reader_factory, monkeypatch, nested):
    import torch

    from soup_cli.utils.expert_stream import (
        ExpertBufferPool,
        expert_allocation_budget,
        plan_expert_buffer,
    )
    from tests.test_native_expert_reader_port import extension_or_skip

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("bitsandbytes")
    extension_or_skip()
    reader, _, codes = reader_factory(nf4=True, nested=nested)
    baseline_reader, _, _ = reader_factory(nf4=True, folder="baseline", nested=nested)
    plan = plan_expert_buffer(reader, dtype="float32", transfer="packed")
    options = dict(
        device="cuda", dtype="float32", transfer="packed", codes=codes,
        weight_budget_bytes=expert_allocation_budget(plan, device="cuda"),
    )
    stream = expert_loader.open_native_stream({0: reader})
    with monkeypatch.context() as patch:
        patch.setattr(reader, "read_host", lambda *a: pytest.fail("used the Python reader"))
        baseline = ExpertBufferPool(baseline_reader, **options)
        pool = ExpertBufferPool(reader, stream=stream, **options)
        try:
            for expert in (1, 0, 3):
                with baseline.lease(baseline_reader, expert) as want:
                    with pool.lease(reader, expert) as actual:
                        assert all(
                            torch.equal(lhs, rhs) for lhs, rhs in zip(want, actual, strict=True)
                        )
            assert pool.h2d_copies == pool.fetches == 3
            assert pool.logical_bytes_read == baseline.logical_bytes_read > 0
        finally:
            pool.close()
            baseline.close()
            baseline_reader.close()
            reader.close()
