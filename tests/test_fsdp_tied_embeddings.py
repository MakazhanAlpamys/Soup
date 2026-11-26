"""The real Torch 2.13 failure requires joint placement, not two shard counts."""

from types import SimpleNamespace

import pytest


def test_frozen_tied_embedding_and_head_are_placed_in_one_fsdp2_group(monkeypatch):
    import torch.nn as nn
    from torch.distributed import fsdp

    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    embed = nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    head.weight = embed.weight
    embed.weight.requires_grad_(False)
    model = SimpleNamespace(get_input_embeddings=lambda: embed, get_output_embeddings=lambda: head)
    policy = object()
    plugin = SimpleNamespace(fsdp_version=2, mixed_precision_policy=policy, cpu_offload=None)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    calls = []
    monkeypatch.setattr(fsdp, "fully_shard",
                        lambda modules, **kwargs: calls.append((modules, kwargs)))
    assert prepare_fsdp2_tied_embeddings(SimpleNamespace(model=model, accelerator=accelerator))
    assert calls[0][0] == [embed, head]  # embedding alone creates the hardware failure
    assert calls[0][1]["reshard_after_forward"] is False
    assert calls[0][1]["mp_policy"] is policy
    assert embed.weight is head.weight


@pytest.mark.parametrize("variant", ["untied", "trainable", "fsdp1"])
def test_other_routes_are_not_rewritten(monkeypatch, variant):
    import torch.nn as nn
    from torch.distributed import fsdp

    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    embed = nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    if variant != "untied":
        head.weight = embed.weight
    for module in (embed, head):
        module.requires_grad_(variant == "trainable")
    model = SimpleNamespace(get_input_embeddings=lambda: embed, get_output_embeddings=lambda: head)
    plugin = SimpleNamespace(fsdp_version=1 if variant == "fsdp1" else 2)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    monkeypatch.setattr(fsdp, "fully_shard", lambda *a, **k: pytest.fail("unexpected placement"))
    assert not prepare_fsdp2_tied_embeddings(SimpleNamespace(model=model, accelerator=accelerator))


def test_non_distributed_trainer_is_a_noop():
    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    assert not prepare_fsdp2_tied_embeddings(SimpleNamespace())


def _prepare_two_cpu_ranks(rank, rendezvous, ram_efficient, precision):
    """Exercise real FSDP2 DTensors and Accelerate full-state broadcast/load."""
    from types import MethodType
    from unittest.mock import patch

    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from accelerate.state import PartialState
    from accelerate.utils import fsdp_utils
    from accelerate.utils.dataclasses import DynamoBackend
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)
            self.tie_weights()
            self.embed.requires_grad_(False)
            self.adapter = torch.nn.Linear(4, 4, bias=False)
            self.register_buffer("scratch", torch.arange(4), persistent=False)

        def tie_weights(self):
            self.head.weight = self.embed.weight

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    PartialState(cpu=True)  # Accelerate's upcast logger requires initialized state.
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = TiedModel()
        if precision == "bf16":
            model.bfloat16()
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        if precision == "bf16":
            expected["adapter.weight"] = expected["adapter.weight"].float()
        if ram_efficient and rank != 0:
            # Only rank 0 has loaded weights in this mode.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        plugin = SimpleNamespace(
            fsdp_version=2, cpu_ram_efficient_loading=ram_efficient,
            mixed_precision_policy=None, cpu_offload=None, ignored_modules=None,
            reshard_after_forward=True, auto_wrap_policy=None,
            set_auto_wrap_policy=lambda model: None, activation_checkpointing=False,
        )
        accelerator = SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=plugin,
                                  dynamo_plugin=SimpleNamespace(backend=DynamoBackend.NO)),
            device=torch.device("cpu"), fp8_backend=None, _models=[],
            torch_device_mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",)),
            parallelism_config=SimpleNamespace(fsdp_dim_names=("fsdp",)),
            mixed_precision=precision, is_main_process=rank == 0,
        )
        # Leave unrelated object/device preparation out of this CPU fixture;
        # the FSDP2 caller, full-state loader and optimizer remapping are real.
        accelerator._prepare_one = lambda obj, **kwargs: obj
        accelerator._get_named_parameters = MethodType(
            Accelerator._get_named_parameters, accelerator,
        )
        accelerator._prepare_fsdp2 = MethodType(Accelerator._prepare_fsdp2, accelerator)
        optimizer = torch.optim.AdamW([model.adapter.weight], lr=0.01)
        upstream_caller = Accelerator._prepare_fsdp2
        upstream_prepare = fsdp_utils.fsdp2_prepare_model
        original_load = fsdp_utils.fsdp2_load_full_state_dict
        loaded = []

        def checked_load(accelerator, model, full_sd, **kwargs):
            # A global shape on a DTensor is insufficient: the source must
            # still be the full ordinary CPU tensor, before any sharding.
            assert all(not isinstance(value, DTensor) for value in full_sd.values())
            assert full_sd["embed.weight"].shape == (8, 4)
            assert full_sd["adapter.weight"].dtype == torch.float32
            assert isinstance(model.embed, FSDPModule)
            assert isinstance(model.head, FSDPModule)
            loaded.append(True)
            return original_load(accelerator, model, full_sd, **kwargs)

        with patch.object(fsdp_utils, "fsdp2_load_full_state_dict", checked_load):
            trainer = SimpleNamespace(model=model, accelerator=accelerator)
            assert prepare_fsdp2_tied_embeddings(trainer)
            if ram_efficient:
                assert not isinstance(model.embed.weight, DTensor)
                assert not prepare_fsdp2_tied_embeddings(trainer)  # idempotent installation
            model, prepared_optimizer = accelerator._prepare_fsdp2(model, optimizer)

        assert len(loaded) == int(ram_efficient)
        assert prepared_optimizer is optimizer
        assert optimizer.param_groups[0]["params"][0] is model.adapter.weight
        assert Accelerator._prepare_fsdp2 is upstream_caller
        assert fsdp_utils.fsdp2_prepare_model is upstream_prepare
        assert model.embed.weight is model.head.weight
        assert model.embed._get_fsdp_state() is model.head._get_fsdp_state()
        for key, value in model.state_dict().items():
            assert isinstance(value, DTensor)
            assert value.shape == expected[key].shape
            torch.testing.assert_close(value.full_tensor(), expected[key], rtol=0, atol=0)
        assert model.embed.weight.to_local().shape == (4, 4)
        assert torch.equal(model.scratch, torch.arange(4))
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.parametrize("ram_efficient", [False, True])
@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_real_two_rank_preparation_preserves_full_weights(tmp_path, ram_efficient, precision):
    import time

    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("two-CPU-rank setup regression requires Torch distributed/Gloo")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    context = mp.spawn(_prepare_two_cpu_ranks, (rendezvous, ram_efficient, precision),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("FSDP2 setup exceeded 90 seconds (possible collective mismatch)")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_unknown_accelerate_boundary_fails_before_mutating_weights():
    import torch.nn as nn

    from soup_cli.utils.fsdp import prepare_fsdp2_tied_embeddings

    embed = nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    head.weight = embed.weight
    embed.requires_grad_(False)
    original_weight = embed.weight
    model = SimpleNamespace(get_input_embeddings=lambda: embed, get_output_embeddings=lambda: head)
    plugin = SimpleNamespace(fsdp_version=2, cpu_ram_efficient_loading=True)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin),
                                  _prepare_fsdp2=lambda *args: None)
    with pytest.raises(RuntimeError, match="Unsupported Accelerate"):
        prepare_fsdp2_tied_embeddings(SimpleNamespace(model=model, accelerator=accelerator))
    assert embed.weight is head.weight is original_weight
    assert not hasattr(accelerator, "_soup_fsdp2_tied_embeddings")
