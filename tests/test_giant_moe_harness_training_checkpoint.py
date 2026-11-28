"""Process interruption and optimizer-state controls for the giant MoE harness."""

from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness.giant_moe_training_checkpoint import (
    find_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

IDENTITY = {"revision": "test-revision", "data_sha256": "test-data", "steps": 5}


def _adapter() -> tuple[dict[str, Any], Any]:
    import torch

    parameters = {
        "lora_A": torch.nn.Parameter(torch.tensor([0.25, -0.5])),
        "lora_B": torch.nn.Parameter(torch.tensor([0.0, 0.0])),
    }
    return parameters, torch.optim.AdamW(parameters.values(), lr=0.05)


def _train(parameters: dict[str, Any], optimizer: Any, start: int, stop: int) -> list[float]:
    import torch

    losses = []
    for step in range(start, stop):
        optimizer.zero_grad(set_to_none=True)
        row = torch.rand(2) + step
        prediction = row * parameters["lora_A"] + parameters["lora_B"]
        loss = prediction.square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_resume_matches_uninterrupted_adamw_updates(tmp_path: Path) -> None:
    import torch

    torch.manual_seed(77)
    reference, reference_optimizer = _adapter()
    expected_losses = _train(reference, reference_optimizer, 0, 5)
    expected_rng = torch.get_rng_state()

    torch.manual_seed(77)
    interrupted, optimizer = _adapter()
    prefix = _train(interrupted, optimizer, 0, 2)
    checkpoint = save_training_checkpoint(
        tmp_path,
        interrupted,
        optimizer,
        identity=IDENTITY,
        completed_steps=2,
        heldout_before=1.25,
        device="cpu",
    )
    assert checkpoint.name == "checkpoint-2.pt"

    torch.manual_seed(19)
    resumed, resumed_optimizer = _adapter()
    step, baseline = restore_training_checkpoint(
        find_training_checkpoint(tmp_path),
        resumed,
        resumed_optimizer,
        identity=IDENTITY,
        device="cpu",
    )
    assert step == 2 and baseline == 1.25
    assert prefix + _train(resumed, resumed_optimizer, step, 5) == expected_losses
    for name in reference:
        torch.testing.assert_close(resumed[name], reference[name], atol=0, rtol=0)
        state = resumed_optimizer.state[resumed[name]]
        expected = reference_optimizer.state[reference[name]]
        assert state.keys() == expected.keys()
        for key in state:
            torch.testing.assert_close(state[key], expected[key], atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_failed_save_keeps_previous_checkpoint_and_ignores_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    parameters, optimizer = _adapter()
    _train(parameters, optimizer, 0, 2)
    previous = save_training_checkpoint(
        tmp_path,
        parameters,
        optimizer,
        identity=IDENTITY,
        completed_steps=2,
        heldout_before=1.25,
        device="cpu",
    )
    previous_bytes = previous.read_bytes()
    # A killed process may leave a staging file, even when its step number is greater.
    leftover = tmp_path / ".checkpoint-4-interrupted.tmp"
    leftover.write_bytes(b"partial")

    def fail_save(_state: Any, stream: Any) -> None:
        stream.write(b"truncated archive")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="^simulated disk write failure$"):
        save_training_checkpoint(
            tmp_path,
            parameters,
            optimizer,
            identity=IDENTITY,
            completed_steps=3,
            heldout_before=1.25,
            device="cpu",
        )
    assert find_training_checkpoint(tmp_path) == previous
    assert previous.read_bytes() == previous_bytes
    assert set(tmp_path.iterdir()) == {previous, leftover}


def test_resume_rejects_different_data_or_parameter_order_before_restoring(tmp_path: Path) -> None:
    import torch

    trained, optimizer = _adapter()
    _train(trained, optimizer, 0, 2)
    checkpoint = save_training_checkpoint(
        tmp_path,
        trained,
        optimizer,
        identity=IDENTITY,
        completed_steps=2,
        heldout_before=1.25,
        device="cpu",
    )
    fresh, fresh_optimizer = _adapter()
    expected = {name: param.detach().clone() for name, param in fresh.items()}
    with pytest.raises(ValueError) as exc:
        restore_training_checkpoint(
            checkpoint,
            fresh,
            fresh_optimizer,
            identity={**IDENTITY, "data_sha256": "different"},
            device="cpu",
        )
    assert str(exc.value) == f"checkpoint run identity differs from the current run at {checkpoint}"
    with pytest.raises(ValueError) as exc:
        restore_training_checkpoint(
            checkpoint,
            dict(reversed(tuple(fresh.items()))),
            fresh_optimizer,
            identity=IDENTITY,
            device="cpu",
        )
    assert str(exc.value) == f"checkpoint adapter parameter order differs at {checkpoint}"
    assert not fresh_optimizer.state
    assert all(torch.equal(param, expected[name]) for name, param in fresh.items())


@_REQUIRES_TRAINING_FLOOR
def test_tiny_streamed_experts_resume_matches_continuous_training(tmp_path: Path) -> None:
    import torch

    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_expert_stream import _lora, _shard, _tiny_deepseek_v3

    weights, _ = _tiny_deepseek_v3(tmp_path)
    shards = tmp_path / "shards"
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    identity = {**IDENTITY, "steps": 3}

    def build(seed: int) -> tuple[Any, Any, dict[str, Any], Any]:
        model, runtime = build_streamed_model(
            model_id=weights,
            shard_dir=str(shards),
            index=index,
            lora_config=_lora(),
            device="cpu",
            dtype="float32",
            buffers=2,
            pin=False,
            seed=seed,
        )
        model.get_base_model().set_experts_implementation("eager")
        params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        return model, runtime, params, torch.optim.AdamW(params.values(), lr=1e-4)

    def update(model: Any, optimizer: Any, step: int) -> float:
        model.train()
        ids = torch.tensor([[1, 2, 3 + step, 7, 8]])
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        return loss.item()

    model, runtime, params, optimizer = build(seed=71)
    try:
        update(model, optimizer, 0)
        checkpoint = save_training_checkpoint(
            tmp_path / "checkpoints",
            params,
            optimizer,
            identity=identity,
            completed_steps=1,
            heldout_before=1.25,
            device="cpu",
        )
        expected_losses = [update(model, optimizer, step) for step in (1, 2)]
        expected_params = {name: param.detach().clone() for name, param in params.items()}
    finally:
        runtime.close()

    model, runtime, params, optimizer = build(seed=11)
    try:
        step, _ = restore_training_checkpoint(
            checkpoint,
            params,
            optimizer,
            identity=identity,
            device="cpu",
        )
        assert [update(model, optimizer, index) for index in range(step, 3)] == expected_losses
        assert runtime.stats()["expert_fetches"] > 0
        for name, param in params.items():
            torch.testing.assert_close(param, expected_params[name], rtol=0, atol=0)
    finally:
        runtime.close()
