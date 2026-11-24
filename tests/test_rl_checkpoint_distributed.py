"""Replicated PEFT recovery through real, bounded two-process Gloo callbacks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(120)


@pytest.fixture(autouse=True)
def training_dependencies(monkeypatch):
    # Hosted macOS stalls at Gloo group creation without an explicit local interface.
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", os.environ.get("GLOO_SOCKET_IFNAME", "lo0"))
    for module in ("torch", "peft", "transformers"):
        pytest.importorskip(module, reason=f"RL recovery tests require {module}")


def _same(first, second):
    import torch

    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _same(left, right) for left, right in zip(first, second, strict=True)
        )
    return first == second


def _fixture(rank, task="grpo", sharding=None, mock_trainer=False, **training):
    from types import SimpleNamespace

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.peft_wiring import attach_rl_callbacks

    if task == "ppo":
        from transformers import LlamaForSequenceClassification
        from trl.experimental.ppo.ppo_trainer import PolicyAndValueWrapper

    torch.manual_seed(7)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
    )), LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    callback_model = model
    if task == "ppo":
        callback_model = PolicyAndValueWrapper(model, LlamaForSequenceClassification(model.config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    callbacks = []
    trainer = SimpleNamespace(model=callback_model, beta=0.02,
                              args=SimpleNamespace(beta=0.02, kl_coef=0.02),
                              add_callback=callbacks.append)
    if mock_trainer:
        from unittest.mock import MagicMock

        trainer = MagicMock(model=callback_model, args=trainer.args, beta=0.02)
        trainer.add_callback.side_effect = callbacks.append
    if sharding is not None:
        setattr(trainer, sharding, True)
    fields = dict(reward_hack_mitigation="pid_lagrangian", reward_hack_detector="info_rm",
                  reward_hack_rollback=True, reward_hack_rollback_patience=1,
                  reward_hack_max_recovery_attempts=1, rl_checkpoint_save_every_steps=1)
    config = TrainingConfig(**(fields | training))
    attach_rl_callbacks(trainer, config, buffer=SimpleNamespace(snapshot=lambda: {}),
                        output_dir="run", task=task)
    return model, optimizer, trainer, callbacks


def _worker(rank, directory, case):
    import copy
    from datetime import timedelta
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from transformers import TrainerControl
    from transformers.trainer_callback import CallbackHandler

    os.chdir(directory)
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo", init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=15),
    )
    try:
        options = {}
        if case in {"adapter_only", "include_off_manifest", "manifest_disagreement"}:
            options["rl_checkpoint_include_optimizer"] = False
        if case == "no_target":
            options["rl_checkpoint_save_every_steps"] = 100
        if case in {"prune", "pruned_target"}:
            options["rl_checkpoint_keep_last"] = 2 if case == "prune" else 1
        if case in {"fsdp", "deepspeed"}:
            options["sharding"] = f"is_{case}_enabled"
        options["mock_trainer"] = case == "mock_flags"
        task = "ppo" if case == "unequal_ppo" else "grpo"
        model, optimizer, trainer, callbacks = _fixture(rank, task=task, **options)
        checkpoint, controller = callbacks
        handler = CallbackHandler(callbacks, trainer.model, None, optimizer, None)
        state = SimpleNamespace(global_step=1)
        control = TrainerControl()
        controller._observe = lambda snapshot, step: ({}, {"info_rm": 0.0})
        if case in {"real_detector", "binary_reward"}:
            del controller._observe
            controller.buffer.snapshot = lambda: {
                "rewards": ([0.0, 0.05, 0.95, 1.0] if case == "real_detector" else [0, 0, 1, 1]),
                "completions": ["a", "b", "c", "d"]
            }
        if case in {"fsdp", "deepspeed"}:
            with patch.object(distributed, "all_gather", side_effect=AssertionError("collective")):
                handler.on_step_end(trainer.args, state, control)
            assert checkpoint._saved == ([1] if rank == 0 else [])
            return
        if case == "step_mismatch":
            state.global_step += rank
        if case == "cadence":
            from dataclasses import replace

            checkpoint.config = replace(checkpoint.config, save_every_steps=rank + 1)
        if case == "hard_exit":
            distributed.barrier()
            if rank == 1:
                os._exit(17)
        if case in {"step_mismatch", "hard_exit"}:
            try:
                handler.on_step_end(trainer.args, state, control)
            except RuntimeError as exc:
                if case == "step_mismatch":
                    assert "RL collective exchange mismatch" in str(exc)
                    Path(f"mismatch-{rank}.txt").write_text(str(exc))
                else:
                    assert "timed out" in str(exc).lower() or "connection" in str(exc).lower()
                return
            raise AssertionError("communication failure was swallowed")
        if case == "no_target":
            controller._observe = lambda snapshot, step: ({}, {"info_rm": 1.0})
            handler.on_step_end(trainer.args, state, control)
            assert not control.should_training_stop
            assert checkpoint._saved == [] and controller._last_good_step is None
            assert controller._state.recovery_attempts == controller._hack_streak == 0
            return
        handler.on_step_end(trainer.args, state, control)
        if case == "cadence":
            assert control.should_training_stop
            assert checkpoint._saved == []
            return
        assert checkpoint._saved == [1], "every rank must learn the committed step"
        assert controller._last_good_step == 1
        if case == "binary_reward":
            before = copy.deepcopy(model.state_dict())
            wrapped = DistributedDataParallel(model)
            tokens = torch.tensor([[1, 2, 3]])
            for step in range(2, 5):
                optimizer.zero_grad()
                loss = wrapped(input_ids=tokens, labels=tokens).loss
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                state.global_step = step
                handler.on_step_end(trainer.args, state, control)
                assert not control.should_training_stop
                assert controller._state.step == step
                assert controller._state.last_signal == 0.0
                assert controller._state.recovery_attempts == controller._hack_streak == 0
                assert controller._last_good_step == step
            assert not _same(before, model.state_dict())
            torch.save({"model": model.state_dict(), "state": vars(controller._state)},
                       f"rank-{rank}.pt")
            return
        if case in {"prune", "pruned_target"}:
            state.global_step = 2
            handler.on_step_end(trainer.args, state, control)
            assert checkpoint._saved == ([1, 2] if case == "prune" else [2])
            assert controller._last_good_step == 2
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        assert saved_optimizer["state"]
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(2)
            for moments in optimizer.state.values():
                moments["exp_avg"].add_(3)
        optimizer.param_groups[0]["lr"] = 0.9
        vote = 0.0 if ((case in {"unequal_grpo", "unequal_ppo"} and rank == 0)
                       or (case == "silent_peer" and rank == 1)) else 1.0
        if case == "nan" and rank == 1:
            vote = float("nan")
        if case == "nan_root" and rank == 0:
            vote = float("nan")
        if case == "infinite" and rank == 1:
            vote = float("inf")
        signals = {} if case == "missing_signal" and rank == 1 else {"info_rm": vote}
        controller._observe = lambda snapshot, step: ({}, signals)
        if case == "real_detector":
            del controller._observe
            controller.buffer.snapshot = lambda: {
                "rewards": [0.5] * 4, "completions": ["a", "b", "c", "d"]
            }
        if case == "invalid_computed_vote" and rank == 1:
            controller._compute_vote = lambda signals: float("nan")
        if case == "silent_peer" and rank == 1:
            controller._observe = lambda snapshot, step: ({"raw_signal": None}, signals)
        if case == "missing_buffer" and rank == 1:
            controller.buffer = None
        if case == "prepare_error" and rank == 1:
            def raise_preparation(snapshot, step):
                raise OSError("injected local preparation failure")
            controller._observe = raise_preparation
        target = controller._last_good_step
        target_path = Path(f"run/rl-checkpoints/step-{target}")
        if case in {"missing_manifest", "include_off_manifest"} and rank == 0:
            (target_path / "manifest.json").unlink()
        if case == "missing_optimizer" and rank == 0:
            (target_path / "optimizer.pt").unlink()
        if case == "corrupt_optimizer" and rank == 0:
            (target_path / "optimizer.pt").write_bytes(b"corrupt")
        if case == "corrupt_manifest" and rank == 0:
            (target_path / "manifest.json").write_text("{")
        if case == "optimizer_only" and rank == 0:
            (target_path / "adapter_model.safetensors").write_bytes(b"corrupt")
        if case == "adapter_missing_key" and rank == 0:
            from safetensors.torch import load_file, save_file

            adapter_path = str(target_path / "adapter_model.safetensors")
            weights = load_file(adapter_path)
            del weights[next(iter(weights))]
            save_file(weights, adapter_path)
        if case in {"manifest_invisible", "manifest_unreadable"} and rank == 1:
            from soup_cli.utils import paths

            original_open = paths.open_no_follow
            def fail_manifest(path, flags):
                if Path(path).name == "manifest.json" and Path(path).parent.name == "step-1":
                    if case == "manifest_invisible":
                        raise FileNotFoundError("shared manifest not yet visible")
                    raise PermissionError("manifest unreadable")
                return original_open(path, flags)
            paths.open_no_follow = fail_manifest
        if case == "asymmetric_adapter" and rank == 1:
            import peft

            def fail_adapter(*args, **kwargs):
                raise OSError("injected adapter read failure")
            peft.load_peft_weights = fail_adapter
        if case == "optimizer_only_outcome" and rank == 1:
            checkpoint._restore_components = lambda **kwargs: {
                "adapter": False, "optimizer": True, "required": True
            }
        if case == "manifest_disagreement" and rank == 1:
            from dataclasses import replace

            original_manifest = checkpoint._load_manifest
            def changed_manifest(step):
                return replace(original_manifest(step), has_optimizer=True)
            checkpoint._load_manifest = changed_manifest
            checkpoint.config = replace(checkpoint.config, include_optimizer_state=True)
            torch.save(saved_optimizer, target_path / "optimizer.pt")
        if case in {"save_error", "save_after_append"} and rank == 0:
            original_save = checkpoint.save_checkpoint
            def fail_save(*args, **kwargs):
                if case == "save_after_append":
                    original_save(*args, **kwargs)
                raise OSError("injected save failure")
            checkpoint.save_checkpoint = fail_save
        if case == "apply_error" and rank == 1:
            controller._apply_coefficient = lambda value: None
        distributed.barrier()
        state.global_step += 1
        if case == "skipped_exchange":
            if rank == 1:
                handler.callbacks.remove(checkpoint)
            try:
                handler.on_step_end(trainer.args, state, control)
            except RuntimeError as exc:
                assert "RL collective exchange mismatch" in str(exc)
                assert "checkpoint.prepare" in str(exc) and "pid.prepare" in str(exc)
                Path(f"mismatch-{rank}.txt").write_text(str(exc))
                return
            raise AssertionError("a skipped checkpoint exchange must fail loudly")
        handler.on_step_end(trainer.args, state, control)
        failure_cases = {
            "nan", "nan_root", "infinite", "invalid_computed_vote", "missing_signal",
            "missing_buffer", "manifest_disagreement",
            "prepare_error", "missing_manifest", "include_off_manifest", "missing_optimizer",
            "corrupt_manifest", "manifest_invisible", "manifest_unreadable", "optimizer_only",
            "corrupt_optimizer", "adapter_missing_key", "optimizer_only_outcome",
            "asymmetric_adapter", "save_error", "save_after_append", "apply_error",
        }
        if case in failure_cases:
            assert control.should_training_stop, f"{case} must stop every rank"
            if case in {"save_error", "save_after_append"}:
                assert checkpoint._saved == [1], "failed saves cannot become targets"
            torch.save({"state": vars(controller._state), "target": controller._last_good_step,
                        "streak": controller._hack_streak, "stop": control.should_training_stop},
                       f"rank-{rank}.pt")
            if case == "asymmetric_adapter":
                assert _same(saved_model, model.state_dict()) == (rank == 0)
            return
        if case == "pruned_target":
            assert checkpoint._saved == [3] and controller._last_good_step is None
            assert controller._state.recovery_attempts == 0
            assert not control.should_training_stop
            return
        assert not control.should_training_stop
        assert _same(saved_model, model.state_dict()), "policy must restore on every rank"
        if case != "adapter_only":
            assert _same(saved_optimizer, optimizer.state_dict()), "moments must restore"
        else:
            assert not _same(saved_optimizer, optimizer.state_dict())
        assert controller._state.recovery_attempts == 1
        assert controller._state.beta > 0.02 and controller._state.integral != 0
        if task == "grpo":
            assert trainer.beta == trainer.args.beta == controller._state.beta
        else:
            assert trainer.args.kl_coef == controller._state.beta
        if case == "prune":
            assert checkpoint._saved == [2, 3] and not Path("run/rl-checkpoints/step-1").exists()
        if case == "exhausted":
            state.global_step += 1
            before = copy.deepcopy(model.state_dict())
            handler.on_step_end(trainer.args, state, control)
            assert control.should_training_stop and controller._state.recovery_attempts == 1
            assert _same(before, model.state_dict())
            return
        wrapped = DistributedDataParallel(model)
        optimizer.zero_grad()
        tokens = torch.tensor([[1, 2, 3]])
        wrapped(input_ids=tokens, labels=tokens).loss.backward()
        optimizer.step()
        assert not _same(saved_model, model.state_dict()), "the next update must change weights"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "controller": vars(controller._state)}, f"rank-{rank}.pt")
    finally:
        distributed.destroy_process_group()


def _run_workers(tmp_path, case):
    import torch

    if case == "unequal_ppo":
        pytest.importorskip("trl", reason="PPO wrapper recovery requires TRL")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("installed torch has no CPU Gloo backend")
    processes = []
    try:
        for rank in range(2):
            output = (tmp_path / f"worker-{rank}.log").open("w", encoding="utf-8")
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            processes.append((subprocess.Popen(
                [sys.executable, __file__, str(rank), str(tmp_path), case],
                stdout=output, stderr=subprocess.STDOUT, env=environment,
            ), output))
        for rank, (process, _output) in enumerate(processes):
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pytest.fail(f"Gloo worker {rank} timed out for {case}")
        logs = [(tmp_path / f"worker-{rank}.log").read_text() for rank in range(2)]
        expected = [0, 17] if case == "hard_exit" else [0, 0]
        assert [process.returncode for process, _output in processes] == expected, "\n".join(logs)
    finally:
        for process, output in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            output.close()


@pytest.mark.parametrize("case", ["restore", "unequal_grpo", "unequal_ppo", "mock_flags",
                                 "prune", "adapter_only", "real_detector", "silent_peer"])
def test_both_ranks_restore_and_make_the_same_next_update(tmp_path, case):
    import torch

    _run_workers(tmp_path, case)
    first, second = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
                     for rank in range(2)]
    assert _same(first, second)


@pytest.mark.parametrize("case", [
    "nan", "nan_root", "infinite", "invalid_computed_vote", "missing_signal",
    "missing_buffer", "prepare_error", "manifest_disagreement",
    "missing_manifest", "include_off_manifest", "missing_optimizer", "corrupt_manifest",
    "manifest_invisible", "manifest_unreadable", "optimizer_only", "asymmetric_adapter",
    "corrupt_optimizer", "adapter_missing_key", "optimizer_only_outcome",
    "save_error", "save_after_append", "apply_error",
])
def test_one_rank_failure_stops_every_rank_with_matching_controller_state(tmp_path, case):
    import torch

    _run_workers(tmp_path, case)
    first, second = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
                     for rank in range(2)]
    assert _same(first, second)


@pytest.mark.parametrize("case", ["skipped_exchange", "step_mismatch"])
def test_mismatched_exchange_fails_loudly_on_every_rank(tmp_path, case):
    _run_workers(tmp_path, case)
    messages = [(tmp_path / f"mismatch-{rank}.txt").read_text() for rank in range(2)]
    assert messages[0] == messages[1]


@pytest.mark.parametrize("case", ["no_target", "pruned_target", "exhausted", "cadence",
                                 "fsdp", "deepspeed", "hard_exit"])
def test_control_paths_and_process_loss_terminate_within_deadline(tmp_path, case):
    _run_workers(tmp_path, case)


def test_two_valued_rewards_keep_real_detector_and_ddp_training_running(tmp_path):
    import torch

    _run_workers(tmp_path, "binary_reward")
    first, second = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
                     for rank in range(2)]
    assert _same(first, second)


@pytest.mark.parametrize("environment_rank", [None, "1"])
@pytest.mark.parametrize("world_size", [None, 1])
def test_without_a_multi_rank_group_callbacks_never_exchange(
    tmp_path, monkeypatch, environment_rank, world_size,
):
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch.distributed as distributed
    from transformers import TrainerControl

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    if environment_rank is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", environment_rank)
    model, optimizer, trainer, callbacks = _fixture(0)
    checkpoint, controller = callbacks
    controller._observe = lambda snapshot, step: ({}, {"info_rm": 0.0})
    assert not distributed.is_initialized()
    if world_size == 1:
        if not distributed.is_available() or not distributed.is_gloo_available():
            pytest.skip("installed torch has no CPU Gloo backend")
        from datetime import timedelta

        distributed.init_process_group(
            "gloo", init_method=(tmp_path / "single-rank").as_uri(), rank=0, world_size=1,
            timeout=timedelta(seconds=15),
        )
    try:
        with patch.object(distributed, "all_gather", side_effect=AssertionError("collective")):
            control = TrainerControl()
            for callback in callbacks:
                callback.on_step_end(trainer.args, SimpleNamespace(global_step=1), control,
                                     model=model, optimizer=optimizer)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
    expected = [] if environment_rank == "1" and world_size is None else [1]
    assert checkpoint._saved == expected
    assert controller._last_good_step == (1 if expected else None)
    assert not control.should_training_stop


@pytest.mark.parametrize("damage", ["oversize", "wrong_step", "wrong_task", "wrong_path",
                                   "wrong_bool", "extra_field", "wrong_shape", "optimizer_missing"])
def test_manifest_loader_checks_size_shape_identity_even_without_optimizer(
    tmp_path, monkeypatch, damage,
):
    import json

    monkeypatch.chdir(tmp_path)
    model, optimizer, _trainer, callbacks = _fixture(0, rl_checkpoint_include_optimizer=False)
    checkpoint = callbacks[0]
    checkpoint.save_checkpoint(step=1, model=model, optimizer=optimizer)
    manifest = Path("run/rl-checkpoints/step-1/manifest.json")
    record = json.loads(manifest.read_text())
    if damage == "oversize":
        manifest.write_text(json.dumps(record) + " " * (64 * 1024))
    elif damage == "wrong_shape":
        manifest.write_text("[]")
    else:
        updates = {"wrong_step": {"step": 2}, "wrong_task": {"task": "ppo"},
                   "wrong_path": {"checkpoint_dir": "elsewhere"},
                   "wrong_bool": {"has_optimizer": "false"}, "extra_field": {"extra": 1},
                   "optimizer_missing": {"has_optimizer": True}}
        record.update(updates[damage])
        manifest.write_text(json.dumps(record))
    with pytest.raises((ValueError, TypeError)):
        checkpoint._restore_components(step=1, model=model, optimizer=optimizer)


@pytest.mark.requires_symlink
@pytest.mark.parametrize("linked", ["manifest.json", "adapter_model.safetensors", "step"])
def test_collective_restore_does_not_read_linked_files_or_escape_run(
    tmp_path, monkeypatch, linked,
):
    import shutil

    monkeypatch.chdir(tmp_path)
    model, optimizer, _trainer, callbacks = _fixture(0)
    checkpoint = callbacks[0]
    checkpoint.save_checkpoint(step=1, model=model, optimizer=optimizer)
    directory = Path("run/rl-checkpoints/step-1")
    if linked == "step":
        outside = tmp_path / "outside"
        shutil.move(directory, outside)
        directory.symlink_to(outside, target_is_directory=True)
    else:
        source = directory / linked
        outside = tmp_path / linked
        source.rename(outside)
        source.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        if linked == "step":
            checkpoint._load_manifest(1)
        else:
            checkpoint._restore_components(step=1, model=model, optimizer=optimizer)


def test_missing_adapter_cannot_trigger_hub_lookup(tmp_path, monkeypatch):
    from unittest.mock import patch

    monkeypatch.chdir(tmp_path)
    model, optimizer, _trainer, callbacks = _fixture(0)
    checkpoint = callbacks[0]
    checkpoint.save_checkpoint(step=1, model=model, optimizer=optimizer)
    Path("run/rl-checkpoints/step-1/adapter_model.safetensors").unlink()
    with patch("peft.load_peft_weights", side_effect=AssertionError("remote lookup")):
        with pytest.raises(ValueError, match="adapter is missing"):
            checkpoint._restore_components(step=1, model=model, optimizer=optimizer)


def test_no_distributed_backend_never_inspects_group():
    from unittest.mock import patch

    import torch.distributed as distributed

    from soup_cli.utils.rl_distributed import active_group

    with (patch.object(distributed, "is_available", return_value=False),
          patch.object(distributed, "is_initialized", side_effect=AssertionError("group query"))):
        assert active_group(True) is None


@pytest.mark.parametrize("kind", ["kl_control", "log_only"])
def test_other_mitigation_modes_do_not_enter_pid_collectives(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch.distributed as distributed
    from transformers import TrainerControl

    monkeypatch.chdir(tmp_path)
    model, optimizer, trainer, callbacks = _fixture(0, reward_hack_mitigation=kind,
                                                  reward_hack_rollback=False)
    controller = callbacks[1]
    controller._observe = lambda snapshot, step: ({}, {"info_rm": 1.0})
    with (patch.object(distributed, "is_initialized", return_value=True),
          patch.object(distributed, "get_world_size", return_value=2),
          patch.object(distributed, "all_gather", side_effect=AssertionError("collective"))):
        controller.on_step_end(trainer.args, SimpleNamespace(global_step=1), TrainerControl(),
                               model=model, optimizer=optimizer)
    if kind == "kl_control":
        assert controller._state.step == 1
    else:
        assert controller._state.step == 0


if __name__ == "__main__":
    _worker(int(sys.argv[1]), sys.argv[2], sys.argv[3])
