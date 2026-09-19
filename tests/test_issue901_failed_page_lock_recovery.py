"""#901 — a failed page-lock left a stale CUDA error that killed the next kernel launch.

The report: Qwen2.5-14B, ``stream_layers: true``, NF4, on an 8 GB card with 32 GB
of RAM. The pre-flight predicted a 3.39 GB peak against 7.41 GB free, the run
printed "could not page-lock the base ... falling back to a PAGEABLE RAM store",
and then died with ``AcceleratorError: CUDA error: out of memory`` inside
``SFTTrainer.__init__`` — or, with ``training.stream_vram_probe`` on, inside the
probe's first forward. Lowering ``max_length`` changed nothing.

Reproduced on the dev box (RTX 5070 Laptop 8 GB, Windows 11, torch 2.14.0+cu130)
with a synthetic Qwen2.5-14B-shaped checkpoint, then reduced to pure torch:

* ``torch.empty(N, pin_memory=True)`` for N past what ``cuMemHostAlloc`` will
  give raises ``AcceleratorError("CUDA error: out of memory")`` in 0.1 s.
* After that, ``cudaMalloc``, ``synchronize`` and a host-to-device ``memcpy``
  all succeed — but the FIRST KERNEL LAUNCH raises the same "out of memory"
  with 7.3 GB of VRAM free, and the second launch works. The failed host
  allocation leaves the runtime's per-thread "last error" set, nothing on the
  fallback path clears it, and the launch check of the next kernel reads it.
  In the report that next kernel was ``param.data.to(torch.bfloat16)``; in the
  probe it was the forward.

So the pageable fallback was dead on arrival: it announced itself and then
handed the run a poisoned context. Memory was never the problem.

There is no ``cudaGetLastError`` binding in ``torch.cuda.cudart()``, so the
drain launches one trivial kernel and lets its launch check consume the stale
error, then launches a second to prove the context is healthy. The failed
attempt's page-locked blocks also stay in torch's caching host allocator until
released, so the recovery empties that cache too: on the 14B store that is
gigabytes of dead pinned memory otherwise held for the whole run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


#: Far beyond any box's RAM, so ``cuMemHostAlloc`` refuses it at once without
#: touching memory — the fastest way to leave the stale error behind.
_IMPOSSIBLE_PIN_BYTES = 2**40


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, msg):
        self.printed.append(str(msg))


class TestDrainStaleCudaError:
    """The drain: consume a stale error with one launch, prove health with a second."""

    @staticmethod
    def _launch_with(outcomes):
        calls = []

        def launch():
            calls.append(len(calls))
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        return launch, calls

    def test_a_stale_out_of_memory_is_consumed_and_the_second_launch_proves_the_context(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with([RuntimeError("CUDA error: out of memory"), None])
        assert drain_stale_cuda_error("cuda", launch=launch) is True
        assert len(calls) == 2

    def test_a_healthy_context_launches_once_and_reports_nothing_drained(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with([None])
        assert drain_stale_cuda_error("cuda", launch=launch) is False
        assert len(calls) == 1

    def test_two_failures_in_a_row_propagate_as_a_genuinely_broken_context(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with(
            [RuntimeError("CUDA error: out of memory"), RuntimeError("CUDA error: out of memory")]
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            drain_stale_cuda_error("cuda", launch=launch)
        assert len(calls) == 2

    def test_an_error_that_is_not_out_of_memory_is_never_swallowed(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        launch, calls = self._launch_with(
            [RuntimeError("CUDA error: an illegal memory access was encountered"), None]
        )
        with pytest.raises(RuntimeError, match="illegal memory access"):
            drain_stale_cuda_error("cuda", launch=launch)
        assert len(calls) == 1

    def test_without_a_cuda_device_there_is_nothing_to_drain(self, monkeypatch):
        """No context, no stale error — and no attempt to launch on a device
        that is not there, which would itself raise on a CPU-only box."""
        import torch

        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert drain_stale_cuda_error("cuda") is False


class TestRecoveryHelpersOffTheHappyPath:
    """The two helpers around the drain, on the paths CI can reach (security
    review, 2026-09-15): without a CUDA device the recovery is a quiet no-op,
    and a private host-cache call that raises is reported as nothing released
    rather than replacing the page-lock error with one about the recovery."""

    def test_without_a_cuda_device_the_recovery_is_a_quiet_no_op(self, monkeypatch):
        import torch

        from soup_cli.utils.layer_stream_runtime import recover_from_failed_page_lock

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        console = _Console()
        assert recover_from_failed_page_lock(device="cuda", console=console) is False
        assert console.printed == []

    def test_a_raising_host_cache_call_reports_nothing_released(self, monkeypatch):
        import torch

        from soup_cli.utils.layer_stream_runtime import release_cached_pinned_memory

        def boom():
            raise RuntimeError("private API moved")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch._C, "_host_emptyCache", boom, raising=False)
        monkeypatch.setattr(torch.cuda, "host_memory_stats", lambda: {}, raising=False)
        assert release_cached_pinned_memory() == 0

    def test_what_the_recovery_did_is_said_once_on_the_console(self, monkeypatch):
        """Both halves reported in one line, and the return value is the drain
        (the half that decides whether the next kernel launch lives)."""
        import soup_cli.utils.layer_stream_runtime as rt

        monkeypatch.setattr(rt, "drain_stale_cuda_error", lambda device: True)
        monkeypatch.setattr(rt, "release_cached_pinned_memory", lambda: 2_000_000_000)
        console = _Console()
        assert rt.recover_from_failed_page_lock(device="cuda", console=console) is True
        assert len(console.printed) == 1
        assert "cleared the stale CUDA out-of-memory error" in console.printed[0]
        assert "released 2.00 GB of page-locked memory" in console.printed[0]

    def test_without_a_console_the_recovery_reports_through_the_logger(self, monkeypatch, caplog):
        import logging

        import soup_cli.utils.layer_stream_runtime as rt

        monkeypatch.setattr(rt, "drain_stale_cuda_error", lambda device: False)
        monkeypatch.setattr(rt, "release_cached_pinned_memory", lambda: 512 * 2**20)
        with caplog.at_level(logging.INFO, logger=rt.logger.name):
            assert rt.recover_from_failed_page_lock(device="cuda", console=None) is False
        assert "released 0.54 GB of page-locked memory" in caplog.text
        assert "cleared" not in caplog.text


class TestTheFallbackRecoversBeforeBuildingThePageableStore:
    """``_build_source``: the recovery runs BETWEEN the failed pinned constructor
    and the pageable one, on both tiers. After the pageable store is built the
    next CUDA op is the first kernel of the run, so recovering after it would be
    exactly as late as never."""

    def test_ram_tier_recovers_before_the_pageable_store_is_built(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import shard_checkpoint
        from tests.test_v07200 import _tiny_llama_dir

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)
        events = []
        real = rt.RamSource

        class _FailsWhenPinned(real):
            def __init__(self, shard_dir, n_layers, spec, *, pin=True):
                events.append(("ramsource", pin))
                if pin:
                    raise RuntimeError("CUDA error: out of memory")
                super().__init__(shard_dir, n_layers, spec, pin=False)

        monkeypatch.setattr(rt, "RamSource", _FailsWhenPinned)
        console = _Console()
        monkeypatch.setattr(
            rt,
            "recover_from_failed_page_lock",
            lambda **kwargs: events.append(("recover", kwargs.get("console") is console)) or True,
        )
        source, pinned = rt._build_source(shards, index.n_layers, spec, True, console)
        assert events == [("ramsource", True), ("recover", True), ("ramsource", False)]
        assert pinned is False
        assert source.nbytes > 0

    def test_disk_tier_recovers_before_the_pageable_staging_is_built(self, tmp_path, monkeypatch):
        import soup_cli.utils.async_disk_source as ads
        import soup_cli.utils.layer_stream_runtime as rt
        from tests.test_issue971_async_disk_source import N_LAYERS, _shards, _spec

        shard_dir = _shards(tmp_path)
        events = []
        real = ads.AsyncDiskSource

        class _FailsWhenPinned(real):
            def __init__(self, *args, pin=True, **kwargs):
                events.append(("asyncsource", pin))
                if pin:
                    raise RuntimeError("CUDA error: out of memory")
                super().__init__(*args, pin=False, **kwargs)

        monkeypatch.setattr(ads, "AsyncDiskSource", _FailsWhenPinned)
        console = _Console()
        monkeypatch.setattr(
            rt,
            "recover_from_failed_page_lock",
            lambda **kwargs: events.append(("recover", kwargs.get("console") is console)) or True,
        )
        source, pinned = rt._build_source(
            shard_dir, N_LAYERS, _spec(shard_dir), True, console, "disk", read_ahead=2
        )
        try:
            assert events == [("asyncsource", True), ("recover", True), ("asyncsource", False)]
            assert pinned is False
        finally:
            source.close()

    def test_a_refused_pin_under_stream_pin_true_still_drains_before_raising(
        self, tmp_path, monkeypatch
    ):
        """``require_pin`` raises instead of falling back — no pageable store is
        built — but the stale error outlives the exception, so the recovery runs
        before the refusal (python review, 2026-09-15): a caller that catches
        the refusal must not inherit a poisoned context."""
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import shard_checkpoint
        from tests.test_v07200 import _tiny_llama_dir

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)
        events = []

        class _FailsWhenPinned:
            def __init__(self, shard_dir, n_layers, spec, *, pin=True, **kwargs):
                events.append(("ramsource", pin))
                raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(rt, "RamSource", _FailsWhenPinned)
        monkeypatch.setattr(
            rt,
            "recover_from_failed_page_lock",
            lambda **kwargs: events.append("recover") or True,
        )
        with pytest.raises(RuntimeError, match="stream_pin=true"):
            rt._build_source(shards, index.n_layers, spec, True, _Console(), require_pin=True)
        assert events == [("ramsource", True), "recover"]


@pytest.mark.gpu
class TestOnRealHardware:
    """The mechanism itself, on a real device. CI has no GPU, so these run on
    dev boxes; the first one is a characterisation of torch/CUDA behaviour and
    is what tells us when the drain stops being necessary."""

    @staticmethod
    def _fail_a_page_lock():
        import torch

        with pytest.raises(RuntimeError):
            torch.empty(_IMPOSSIBLE_PIN_BYTES, dtype=torch.uint8, pin_memory=True)

    @staticmethod
    def _launch():
        import torch

        torch.ones(1, device="cuda")
        torch.cuda.synchronize()

    def test_a_failed_page_lock_poisons_the_next_launch_and_only_the_next(self):
        """The upstream behaviour this fix exists for. If this test ever FAILS
        because the first launch after the failed page-lock succeeds, torch
        or the driver has started clearing the error itself and the drain can
        be retired."""
        self._fail_a_page_lock()
        with pytest.raises(RuntimeError, match="out of memory"):
            self._launch()
        self._launch()

    def test_the_drain_makes_the_first_launch_after_a_failed_page_lock_succeed(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        self._fail_a_page_lock()
        assert drain_stale_cuda_error("cuda") is True
        self._launch()

    def test_the_drain_on_a_healthy_context_is_a_no_op(self):
        from soup_cli.utils.layer_stream_runtime import drain_stale_cuda_error

        self._launch()
        assert drain_stale_cuda_error("cuda") is False
        self._launch()

    def test_the_pageable_fallback_leaves_a_usable_context(self, tmp_path, monkeypatch):
        """The whole chain, with a REAL failed page-lock behind the fallback.

        ``torch.empty(pin_memory=True)`` is wrapped so the pinned store's first
        allocation is an impossible one: the genuine driver failure, the
        genuine stale error, the shipped fallback. Without the recovery the
        first kernel after ``_build_source`` returns is the one that dies — in
        the report, inside ``SFTTrainer.__init__``."""
        import torch

        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
        from tests.test_v07200 import _tiny_llama_dir

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)
        real_empty = torch.empty

        def _impossible_when_pinned(*args, **kwargs):
            if kwargs.get("pin_memory"):
                return real_empty(_IMPOSSIBLE_PIN_BYTES, dtype=torch.uint8, pin_memory=True)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", _impossible_when_pinned)
        console = _Console()
        source, pinned = rt._build_source(shards, index.n_layers, spec, True, console)
        monkeypatch.undo()

        assert pinned is False
        assert any("PAGEABLE" in msg for msg in console.printed)
        # The first kernel launch of the run, where the report died.
        self._launch()
        # And the pageable store is the real one: bit-identical to the shard.
        from safetensors import safe_open

        with safe_open(layer_shard_path(shards, 0), framework="pt") as handle:
            expected = handle.get_tensor("self_attn.q_proj.weight")
        assert torch.equal(source.get(0, "self_attn.q_proj.weight"), expected)

    def test_the_recovery_returns_the_failed_attempts_cached_pinned_blocks(self):
        """A pinned tensor torch frees goes back to its caching host allocator,
        not to the OS. After a partially pinned store is abandoned that cache
        holds gigabytes of page-locked memory for nothing; the recovery
        empties it."""
        import torch

        from soup_cli.utils.layer_stream_runtime import recover_from_failed_page_lock

        if not hasattr(torch.cuda, "host_memory_stats"):
            pytest.skip("this torch has no host_memory_stats to read the cache from")

        def cached_bytes() -> int:
            stats = torch.cuda.host_memory_stats()
            return int(stats["allocated_bytes.current"]) - int(stats["active_bytes.current"])

        held = [torch.empty(2**26, dtype=torch.uint8, pin_memory=True) for _ in range(4)]
        del held
        before = cached_bytes()
        assert before >= 4 * 2**26
        recover_from_failed_page_lock(device="cuda", console=None)
        assert cached_bytes() < before


@pytest.mark.gpu
class TestTheProbeCallsAnOutOfMemoryAcceleratorErrorAnOom:
    """#649's shape inside the #349 instrument. Under WDDM the allocator has often
    already spilled, so the out-of-memory surfaces later, at a synchronise, as
    ``AcceleratorError("CUDA error: out of memory")`` rather than as
    ``torch.OutOfMemoryError``. That is a RESULT — the shape does not fit — and
    the probe filed it as an instrument failure that "may have poisoned the
    context". Both refuse the run; only the OOM verdict tells the operator what
    to change (batch or max_length). In the report, with the probe on, this is
    exactly the message the reporter got."""

    @staticmethod
    def _model_raising(message: str):
        """A model whose forward raises ``torch.AcceleratorError(message)`` —
        the class torch >= 2.8 raises for a CUDA-runtime error; older torch
        (the declared floor is 2.6) spells it ``RuntimeError`` and has nothing
        to classify here, so the test skips rather than errors."""
        import torch

        accelerator_error = getattr(torch, "AcceleratorError", None)
        if accelerator_error is None:
            pytest.skip("this torch has no AcceleratorError (added in 2.8)")

        class _Model:
            def __call__(self, **kwargs):
                raise accelerator_error(message)

            @staticmethod
            def parameters():
                return iter(())

        return _Model()

    def test_an_out_of_memory_accelerator_error_is_an_oom_verdict_not_a_failure(self):
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        peak = measure_step_peak_bytes(
            self._model_raising("CUDA error: out of memory"),
            rows=1,
            seq_len=8,
            vocab_size=32,
        )
        assert peak is not None
        assert peak.oom is True
        assert peak.failed is False

    def test_any_other_accelerator_error_is_still_an_instrument_failure(self):
        from soup_cli.utils.layer_stream_runtime import measure_step_peak_bytes

        peak = measure_step_peak_bytes(
            self._model_raising("CUDA error: an illegal memory access was encountered"),
            rows=1,
            seq_len=8,
            vocab_size=32,
        )
        assert peak is not None
        assert peak.failed is True
        assert peak.oom is False
        assert peak.error == "AcceleratorError"
