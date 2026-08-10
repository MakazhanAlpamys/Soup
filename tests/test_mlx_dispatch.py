"""Dispatch tests for the MLX backend (PR #362).

The original bug: `backend: mlx` silently fell through to the transformers
trainer because commands/train.py branched on task only. These tests pin the
backend-first resolution through `resolve_trainer` so the CLI can never
silently ignore the backend again.

Note: the config schema already rejects `backend: mlx` for non-SFT tasks
("MLX backend only ships SFT..."), so the registry-level rejection in
`get_mlx_trainer` is a second line of defence for callers that bypass the
schema validator.
"""

import pytest


def _cfg(backend, task):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        "base: mlx-community/tiny\n"
        f"task: {task}\n"
        f"backend: {backend}\n"
        "data: {train: d.jsonl, format: chatml}\n"
        "training: {epochs: 1, lr: 2e-4, batch_size: 1}\n"
        "output: ./out\n"
    )


def test_mlx_registry_resolves_known_trainers():
    from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper
    from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper
    from soup_cli.trainer.mlx_routing import get_mlx_trainer
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    assert get_mlx_trainer("sft") is MLXSFTTrainerWrapper
    assert get_mlx_trainer("dpo") is MLXDPOTrainerWrapper
    assert get_mlx_trainer("grpo") is MLXGRPOTrainerWrapper


def test_mlx_registry_rejects_unknown_task_loudly():
    from soup_cli.trainer.mlx_routing import get_mlx_trainer

    with pytest.raises(
        ValueError, match="MLX backend does not support task 'pretrain'"
    ):
        get_mlx_trainer("pretrain")


def test_mlx_backend_resolves_sft_trainer():
    from soup_cli.trainer.mlx_routing import resolve_trainer
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cls, kwargs = resolve_trainer(_cfg("mlx", "sft"))
    assert cls is MLXSFTTrainerWrapper
    assert kwargs == {}


def test_transformers_backend_falls_through_to_task_chain():
    from soup_cli.trainer.mlx_routing import resolve_trainer

    cls, kwargs = resolve_trainer(_cfg("transformers", "sft"))
    assert cls is None
    assert kwargs == {}


def test_resolve_trainer_forwards_trainer_kwargs():
    from soup_cli.trainer.mlx_routing import resolve_trainer

    cls, kwargs = resolve_trainer(
        _cfg("mlx", "sft"), {"trust_remote_code": True, "device": "cpu"}
    )
    assert cls is not None
    assert kwargs == {"trust_remote_code": True, "device": "cpu"}


def test_schema_rejects_mlx_for_non_sft_tasks():
    """The schema gate (#270 prerequisite): mlx + dpo/grpo/ppo must not
    reach the trainer dispatch with a silent fallthrough."""
    from soup_cli.config.loader import load_config_from_string

    for task in ("dpo", "grpo", "ppo"):
        with pytest.raises(ValueError, match="MLX backend"):
            load_config_from_string(
                "base: mlx-community/tiny\n"
                f"task: {task}\n"
                "backend: mlx\n"
                "data: {train: d.jsonl, format: chatml}\n"
                "training: {epochs: 1, lr: 2e-4, batch_size: 1}\n"
                "output: ./out\n"
            )
