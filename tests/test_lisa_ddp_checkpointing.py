"""LISA with native activation checkpointing needs DDP to find unused parameters."""

import hashlib
from datetime import timedelta
from pathlib import Path
from time import monotonic

import pytest


@pytest.fixture
def train_extra(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    for module in ("torch", "transformers", "accelerate", "datasets", "peft", "trl"):
        pytest.importorskip(module)


def _setup_wrapper(task, path, monkeypatch, *, lisa, checkpointing):
    from tests.test_issue341_seed_and_fullft import _tiny_llama_dir, _wrapper
    from tests.test_v07134 import _pretrain_wrapper

    path.mkdir()
    options = {
        "optimizer": "adamw_torch",
        "gradient_checkpointing": checkpointing,
        "epochs": 2,
        "warmup_ratio": 0.0,
        "rewind_log": False,
    }
    if lisa:
        options.update(lisa_enabled=True, lisa_num_layers=1, lisa_interval_steps=2)
    elif task == "sft":
        options["lora"] = {"r": 0}

    if task == "sft":
        base = _tiny_llama_dir(path, n_layers=4)
        wrapper, data = _wrapper(path, monkeypatch, base=base, **options)
    else:
        wrapper, data = _pretrain_wrapper(path, monkeypatch, **options)
        data["train"].extend({"text": row} for row in (
            "one cat ran", "two dog ran", "cat sat fast", "red cat blue",
        ))
    wrapper.setup(data)
    return wrapper


def _gloo_worker(rank, directory, task):
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback

    torch.set_num_threads(1)
    root = Path(directory)
    with pytest.MonkeyPatch.context() as patch:
        for key, value in {
            "ACCELERATE_USE_CPU": "true",
            "ACCELERATE_MIXED_PRECISION": "no",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
        }.items():
            patch.setenv(key, value)
        dist.init_process_group(
            "gloo", init_method=(root / "rendezvous").as_uri(),
            rank=rank, world_size=2, timeout=timedelta(seconds=60),
        )
        try:
            wrapper = _setup_wrapper(
                task, root / f"rank-{rank}", patch, lisa=True, checkpointing="full",
            )
            trainer = wrapper.trainer
            assert trainer.args.world_size == 2
            assert trainer.args.gradient_checkpointing is True
            expected_layers = (3, 3, 3, 3, 0, 0, 2, 2)
            active = []
            gradient_steps = []

            class Trace(TrainerCallback):
                weights_before_step = None

                def on_train_begin(self, args, state, control, model=None, **kwargs):
                    assert isinstance(
                        trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel,
                    )
                    functions = [
                        module._gradient_checkpointing_func for module in model.modules()
                        if getattr(module, "gradient_checkpointing", False)
                    ]
                    assert functions, "Native checkpointing was not enabled"
                    assert all(func.keywords.get("use_reentrant") is False for func in functions)

                def on_step_begin(self, args, state, control, model=None, **kwargs):
                    layer = expected_layers[state.global_step]
                    decoder = {
                        name: param for name, param in model.named_parameters()
                        if ".layers." in name
                    }
                    expected = {name for name in decoder if f".layers.{layer}." in name}
                    assert expected
                    trainable = {name for name, param in decoder.items() if param.requires_grad}
                    assert trainable == expected
                    active.append(expected)
                    self.weights_before_step = {
                        name: param.detach().clone() for name, param in decoder.items()
                    }

                def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                    grads = {
                        name: param.grad for name, param in model.named_parameters()
                        if ".layers." in name and param.grad is not None
                    }
                    assert grads.keys() == active[-1]
                    assert all(torch.isfinite(grad).all().item() for grad in grads.values())
                    gradient_steps.append(state.global_step + 1)

                def on_step_end(self, args, state, control, model=None, **kwargs):
                    changed = {
                        name for name, param in model.named_parameters()
                        if name in self.weights_before_step
                        and not torch.equal(param, self.weights_before_step[name])
                    }
                    assert changed and changed <= active[-1], (state.global_step, changed)

            trainer.add_callback(Trace())
            wrapper.train()
            assert trainer.state.global_step == len(expected_layers)
            assert gradient_steps == list(range(1, len(expected_layers) + 1))
            assert trainer.args.ddp_find_unused_parameters is True
            assert trainer.model_wrapped.find_unused_parameters is True

            digest = hashlib.sha256()
            for name, param in trainer.model.named_parameters():
                weights = param.detach().cpu().contiguous()
                assert torch.isfinite(weights).all().item(), name
                digest.update(name.encode("utf-8"))
                digest.update(weights.numpy().tobytes())
            hashes = [None, None]
            dist.all_gather_object(hashes, digest.hexdigest())
            assert hashes[0] == hashes[1]
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_lisa_native_checkpointing_trains_through_a_layer_switch(train_extra, tmp_path, task):
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU distributed test needs Gloo")
    ranks = torch.multiprocessing.spawn(
        _gloo_worker, args=(str(tmp_path), task), nprocs=2, join=False,
    )
    deadline = monotonic() + 180
    try:
        while not ranks.join(timeout=max(0, deadline - monotonic())):
            assert monotonic() < deadline, "Two-rank LISA training timed out"
    finally:
        for process in ranks.processes:
            if process.is_alive():
                process.kill()
            process.join()


@pytest.mark.parametrize(("task", "lisa", "checkpointing", "native"), [
    ("sft", True, False, False),
    ("pretrain", True, False, False),
    ("sft", False, "full", True),
    ("pretrain", False, "full", True),
    ("sft", True, "selective", False),
])
def test_other_training_modes_keep_ddp_default(
    train_extra, tmp_path, monkeypatch, task, lisa, checkpointing, native,
):
    wrapper = _setup_wrapper(
        task, tmp_path / "fixture", monkeypatch, lisa=lisa, checkpointing=checkpointing,
    )
    assert wrapper.trainer.args.gradient_checkpointing is native
    assert wrapper.trainer.args.ddp_find_unused_parameters is None
