"""Partial RAM source: real CPU tensors/files, bounded residency, identical backward order."""

import pytest

from soup_cli.utils.partial_ram_source import PartialRamBudget, PartialRamSource


def shards(tmp_path, count=5):
    import torch
    from safetensors.torch import save_file

    paths = []
    for idx in range(count):
        path = tmp_path / f"layer{idx}.safetensors"
        save_file(
            {
                "weight": torch.arange(16, dtype=torch.float32).reshape(4, 4) + idx,
                "offset": torch.tensor(float(idx)),
            },
            str(path),
        )
        paths.append(str(path))
    return paths, [{"weight": ((4, 4), "float32"), "offset": ((), "float32")}] * count


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("inf")])
def test_budget_rejects_invalid_bytes(bad):
    with pytest.raises(ValueError):
        PartialRamBudget(limit_bytes=bad, reserved_bytes=0, cache_bytes=64)


def test_reservations_cannot_exceed_total():
    with pytest.raises(ValueError):
        PartialRamBudget(limit_bytes=100, reserved_bytes=101, cache_bytes=64)


@pytest.mark.parametrize("cached", [0, 1, 3, 5])
def test_exact_values_both_directions_and_cleanup(tmp_path, cached):
    import torch
    from safetensors.torch import load_file

    paths, specs = shards(tmp_path)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 1000, cached * 68),
    )
    try:
        assert source.resident_indices == tuple(range(cached))
        assert source.accounted_peak_bytes <= 1_000_000
        for idx in list(range(5)) + list(reversed(range(5))) + [0, 4, 0, 4]:
            reference = load_file(paths[idx])
            for key in reference:
                assert torch.equal(source.get(idx, key), reference[key])
            source.release(idx)
        assert source.resident_bytes == cached * 68
        if cached:
            assert source.resident_tensor_bytes_served > 0
    finally:
        source.close()
    assert source.nbytes == 0
    assert source.host_allocated_bytes == 0
    source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.get(0, "weight")


def test_cached_layers_are_never_opened_by_async_reader(tmp_path, monkeypatch):
    from soup_cli.utils.async_disk_source import AsyncDiskSource

    paths, specs = shards(tmp_path)
    opened = []
    original = AsyncDiskSource._read_layer

    def observe(self, idx, region):
        opened.append(self._paths[idx])
        return original(self, idx, region)

    monkeypatch.setattr(AsyncDiskSource, "_read_layer", observe)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 0, 136),
        read_ahead=1,
    )
    try:
        for idx in list(range(5)) + list(reversed(range(5))):
            source.get(idx, "weight")
            source.release(idx)
    finally:
        source.close()
    assert opened
    assert set(opened).issubset(set(paths[2:]))
    assert not set(opened).intersection(paths[:2])


def test_staging_budget_refuses_before_allocation(tmp_path, monkeypatch):
    from soup_cli.utils.async_disk_source import AsyncDiskSource

    paths, specs = shards(tmp_path)
    allocated = []
    monkeypatch.setattr(AsyncDiskSource, "_allocate_staging", lambda *args: allocated.append(True))
    with pytest.raises(ValueError, match="budget"):
        PartialRamSource(
            str(tmp_path),
            5,
            specs,
            pin=False,
            shard_paths=paths,
            budget=PartialRamBudget(1000, 900, 68),
        )
    assert not allocated


def test_memory_reserves_load_transient_even_all_cached(tmp_path):
    paths, specs = shards(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        PartialRamSource(
            str(tmp_path),
            5,
            specs,
            pin=False,
            shard_paths=paths,
            budget=PartialRamBudget(340, 0, 340),
        )


def test_large_weights_not_selected_as_decoder_cache(tmp_path):
    paths, specs = shards(tmp_path)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        cacheable_layers=3,
        budget=PartialRamBudget(1_000_000, 0, 10000),
    )
    try:
        assert source.resident_indices == (0, 1, 2)
        assert source.disk_indices == (3, 4)
    finally:
        source.close()


def test_bad_index_cannot_alias_last_layer(tmp_path):
    paths, specs = shards(tmp_path)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 0, 68),
    )
    try:
        for idx in (-1, 5, True):
            with pytest.raises((IndexError, TypeError)):
                source.get(idx, "weight")
    finally:
        source.close()


def test_runtime_factory_routes_explicit_budget(tmp_path):
    from soup_cli.utils.layer_stream_runtime import _build_source

    paths, specs = shards(tmp_path)
    source, pinned = _build_source(
        str(tmp_path),
        5,
        specs,
        False,
        None,
        "disk",
        shard_paths=paths,
        partial_ram_budget=PartialRamBudget(1_000_000, 0, 68),
        cacheable_layers=5,
    )
    try:
        assert isinstance(source, PartialRamSource)
        assert source.resident_indices == (0,)
        assert not pinned
    finally:
        source.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_synthetic_training_matches_disk_exactly(tmp_path, device):
    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07200 import _tiny_llama_dir, _tiny_lora

    weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=4, tie=False)
    directory = str(tmp_path / "shards")
    index = shard_checkpoint(weights, directory, dtype="float32", arch="llama")
    results = []
    for budget in (
        None,
        PartialRamBudget(10_000_000, 0, 90_000 if device.startswith("cuda") else 45_000),
    ):
        model, runtime = build_streamed_model(
            model_id=weights,
            shard_dir=directory,
            index=index,
            lora_config=_tiny_lora(),
            device=device,
            dtype="float32",
            pin=device.startswith("cuda"),
            seed=3,
            tier="disk",
            read_ahead=1,
            partial_ram_budget=budget,
        )
        try:
            model.train()
            params = {
                name: value for name, value in model.named_parameters() if value.requires_grad
            }
            initial = {name: value.detach().clone() for name, value in params.items()}
            optimizer = torch.optim.AdamW(params.values(), lr=0.001)
            records = []
            for step in range(3):
                ids = ((torch.arange(8).reshape(1, 8) + step) % 64).to(device)
                optimizer.zero_grad()
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                records.append(
                    (
                        loss.detach().clone(),
                        {name: value.grad.detach().clone() for name, value in params.items()},
                    )
                )
                optimizer.step()
            final = {name: value.detach().clone() for name, value in params.items()}
            assert any(not torch.equal(initial[name], value) for name, value in final.items())
            if budget:
                assert 0 < len(runtime.source.resident_indices) < 4
                assert runtime.stats()["resident_bytes"] > 0
            results.append((records, final))
        finally:
            runtime.close()
    for (loss_a, grad_a), (loss_b, grad_b) in zip(results[0][0], results[1][0]):
        assert torch.equal(loss_a, loss_b)
        assert grad_a.keys() == grad_b.keys()
        for name in grad_a:
            assert torch.equal(grad_a[name], grad_b[name]), name
    for name in results[0][1]:
        assert torch.equal(results[0][1][name], results[1][1][name]), name


@pytest.mark.parametrize(
    "override",
    [
        {"stream_layers": False},
        {"stream_source": "auto"},
    ],
)
def test_schema_rejects_unwired_modes(override):
    from soup_cli.config.schema import SoupConfig

    training = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "stream_source": "disk",
        "stream_partial_ram": {"limit_bytes": 100000, "reserved_bytes": 100, "cache_bytes": 1000},
        **override,
    }
    with pytest.raises(ValueError, match="stream_partial_ram"):
        SoupConfig(
            base="local-model", data={"train": "./data.jsonl"}, task="sft", training=training
        )


def _partial_payload(**training):
    return {
        "base": "local-model",
        "data": {"train": "./data.jsonl"},
        "task": "sft",
        "training": {
            "batch_size": 1,
            "quantization": "none",
            "stream_layers": True,
            "stream_source": "disk",
            "stream_partial_ram": {
                "limit_bytes": 100000, "reserved_bytes": 100, "cache_bytes": 1000,
            },
            **training,
        },
    }


def _message(payload):
    from pydantic import ValidationError

    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(payload)
    return " ".join(str(excinfo.value).split())


def test_schema_refusals_name_the_gate():
    assert (
        "training.stream_partial_ram set but stream_layers is false"
        in _message(_partial_payload(stream_layers=False))
    )
    assert (
        "training.stream_partial_ram requires unquantized transformers SFT, "
        "stream_layers=true and stream_source='disk'"
        in _message(_partial_payload(stream_source="auto"))
    )
    assert "stream_partial_ram requires unquantized transformers SFT" in _message(
        _partial_payload(quantization="4bit")
    )
    payload = _partial_payload()
    payload["task"] = "dpo"
    payload["training"]["batch_size"] = 2
    assert "stream_partial_ram requires unquantized transformers SFT" in _message(payload)


@pytest.mark.parametrize(
    ("other", "value"), [("stream_ram_cache_mb", 64), ("stream_resident_layers", "auto")]
)
def test_schema_refuses_a_second_host_or_vram_planner(other, value):
    assert (
        f"training.stream_partial_ram cannot be combined with training.{other}"
        in _message(_partial_payload(**{other: value}))
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"limit_bytes": 0, "reserved_bytes": 0, "cache_bytes": 0},
        {"limit_bytes": 100, "reserved_bytes": 101, "cache_bytes": 0},
        {"limit_bytes": True, "reserved_bytes": 0, "cache_bytes": 0},
        {"limit_bytes": 100, "reserved_bytes": 0, "cache_bytes": 0, "extra": 1},
    ],
)
def test_schema_refuses_an_invalid_envelope(bad):
    from pydantic import ValidationError

    from soup_cli.config.schema import SoupConfig

    payload = _partial_payload()
    payload["training"]["stream_partial_ram"] = bad
    with pytest.raises(ValidationError):
        SoupConfig.model_validate(payload)


def test_default_config_has_no_partial_ram():
    from soup_cli.config.schema import SoupConfig

    payload = _partial_payload()
    del payload["training"]["stream_partial_ram"]
    assert SoupConfig.model_validate(payload).training.stream_partial_ram is None


def test_setup_refuses_a_non_llama_base_before_building(tmp_path, monkeypatch):
    from soup_cli.config.schema import SoupConfig
    from tests.test_v07200 import TestStreamingEndToEndSetup

    wrapper, dataset = TestStreamingEndToEndSetup()._wrapper(tmp_path, monkeypatch)
    config = wrapper.config.model_dump()
    config["training"].update(
        {
            "stream_source": "disk",
            "stream_disk_kind": "nvme",
            "stream_partial_ram": {
                "limit_bytes": 10_000_000, "reserved_bytes": 100_000, "cache_bytes": 45_000,
            },
        }
    )
    wrapper.config = SoupConfig.model_validate(config)
    monkeypatch.setattr("soup_cli.utils.layer_stream.stream_arch_of", lambda _config: "qwen3")
    with pytest.raises(
        ValueError, match="training.stream_partial_ram currently supports dense Llama"
    ):
        wrapper.setup(dataset)


def test_schema_retains_explicit_budget():
    from soup_cli.config.schema import SoupConfig

    cfg = SoupConfig(
        base="local-model",
        data={"train": "./data.jsonl"},
        task="sft",
        training={
            "batch_size": 1,
            "quantization": "none",
            "stream_layers": True,
            "stream_source": "disk",
            "stream_partial_ram": {
                "limit_bytes": 100000,
                "reserved_bytes": 100,
                "cache_bytes": 1000,
            },
        },
    )
    assert cfg.training.stream_partial_ram.cache_bytes == 1000


def test_close_waits_for_cached_transfer(tmp_path):
    paths, specs = shards(tmp_path)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 0, 68),
    )

    class Event:
        completed = False

        def query(self):
            return self.completed

        def synchronize(self):
            assert source._ram is not None
            self.completed = True

    event = Event()
    source.get(0, "weight")
    source.release(0, event)
    source.close()
    assert event.completed
    assert source._ram is None


def test_cache_load_failure_closes_disk_reader(tmp_path, monkeypatch):
    from soup_cli.utils import layer_stream_runtime
    from soup_cli.utils.async_disk_source import AsyncDiskSource

    paths, specs = shards(tmp_path)
    closed = []
    original = AsyncDiskSource.close

    def close(self):
        result = original(self)
        closed.append(not self._thread.is_alive())
        return result

    def fail(*args, **kwargs):
        raise OSError("injected cache read failure")

    monkeypatch.setattr(AsyncDiskSource, "close", close)
    monkeypatch.setattr(layer_stream_runtime.RamSource, "__init__", fail)
    with pytest.raises(OSError, match="injected"):
        PartialRamSource(
            str(tmp_path),
            5,
            specs,
            pin=False,
            shard_paths=paths,
            budget=PartialRamBudget(1_000_000, 0, 68),
        )
    assert len(closed) >= 1
    assert all(closed)


def test_cli_source_summary_separates_residency_and_staging():
    from soup_cli.trainer.stream_setup import _stream_source_line

    text = _stream_source_line(
        {
            "tier": "disk",
            "pinned": False,
            "resident_bytes": 2_000_000,
            "staging_bytes": 3_000_000,
            "host_allocated_bytes": 5_000_000,
            "read_ahead": 2,
        }
    )
    assert "2.0 MB resident decoder" in text
    assert "3.0 MB disk staging" in text
    assert "5.0 MB owned host" in text


def test_pool_failure_releases_partial_source(tmp_path, monkeypatch):
    from soup_cli.utils import layer_stream_runtime as runtime
    from soup_cli.utils.layer_shard import shard_checkpoint
    from tests.test_v07200 import _tiny_llama_dir, _tiny_lora

    weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=4, tie=False)
    directory = str(tmp_path / "shards")
    index = shard_checkpoint(weights, directory, dtype="float32", arch="llama")
    closed = []
    original = PartialRamSource.close

    def close(self):
        original(self)
        closed.append(self)

    def fail(*args, **kwargs):
        raise MemoryError("injected device buffer failure")

    monkeypatch.setattr(PartialRamSource, "close", close)
    monkeypatch.setattr(runtime, "LayerBufferPool", fail)
    with pytest.raises(MemoryError, match="injected"):
        runtime.build_streamed_model(
            model_id=weights,
            shard_dir=directory,
            index=index,
            lora_config=_tiny_lora(),
            device="cpu",
            dtype="float32",
            pin=False,
            seed=3,
            tier="disk",
            read_ahead=1,
            partial_ram_budget=PartialRamBudget(10_000_000, 0, 45_000),
        )
    assert len(closed) == 1
    assert closed[0]._closed
    assert closed[0]._ram is None


def test_pinned_rounding_is_charged_before_allocation(tmp_path, monkeypatch):
    from soup_cli.utils import layer_stream_runtime as runtime
    from soup_cli.utils.async_disk_source import AsyncDiskSource

    paths, specs = shards(tmp_path, count=1)
    allocations = []
    monkeypatch.setattr(
        AsyncDiskSource, "_allocate_staging", lambda *args: allocations.append(True)
    )
    rounded = runtime.plan_pinned_arenas([4, 64]).pinned_bytes
    assert rounded > 68
    # The cap fits tensor payload, but not the pinned allocation. The layer must
    # remain on disk; that staging also exceeds this deliberately tiny envelope.
    with pytest.raises(ValueError, match="budget"):
        PartialRamSource(
            str(tmp_path), 1, specs, pin=True, shard_paths=paths, budget=PartialRamBudget(68, 0, 68)
        )
    assert not allocations


def test_trainer_setup_consumes_budget_and_reports_cache(tmp_path, monkeypatch):
    from soup_cli.config.schema import SoupConfig
    from tests.test_v07200 import TestStreamingEndToEndSetup

    wrapper, dataset = TestStreamingEndToEndSetup()._wrapper(tmp_path, monkeypatch)
    config = wrapper.config.model_dump()
    config["training"].update(
        {
            "stream_source": "disk",
            "stream_disk_kind": "nvme",
            "stream_pin": False,
            "stream_partial_ram": {
                "limit_bytes": 10_000_000,
                "reserved_bytes": 100_000,
                "cache_bytes": 45_000,
            },
        }
    )
    wrapper.config = SoupConfig.model_validate(config)
    try:
        wrapper.setup(dataset)
        source = wrapper._stream_runtime.source
        assert isinstance(source, PartialRamSource)
        assert source.budget.reserved_bytes >= 100_000
        assert source.resident_bytes > 0
        assert source.accounted_peak_bytes <= 10_000_000
    finally:
        runtime = getattr(wrapper, "_stream_runtime", None)
        if runtime is not None:
            runtime.close()
