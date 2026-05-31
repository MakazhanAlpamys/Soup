"""MLX backend trainer registry — routes task → MLX trainer class.

All trainer imports are deferred to function bodies so the MLX classes don't
load on non-Apple Silicon machines during CLI startup (project convention:
lazy heavy imports).
"""

from __future__ import annotations


class _MLXRegistry(dict):
    """Dict-like registry that lazy-imports MLX trainer classes."""

    def _resolve(self, task: str):
        if task == "sft":
            from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

            return MLXSFTTrainerWrapper
        if task == "dpo":
            from soup_cli.trainer.mlx_dpo import MLXDPOTrainerWrapper

            return MLXDPOTrainerWrapper
        if task == "grpo":
            from soup_cli.trainer.mlx_grpo import MLXGRPOTrainerWrapper

            return MLXGRPOTrainerWrapper
        raise KeyError(task)

    def __contains__(self, key: object) -> bool:
        return key in ("sft", "dpo", "grpo")

    def __getitem__(self, key: str):
        return self._resolve(key)

    def keys(self):  # noqa: D401
        return ("sft", "dpo", "grpo")


MLX_TRAINER_REGISTRY: _MLXRegistry = _MLXRegistry()


def get_mlx_trainer(task: str):
    """Return the MLX trainer class for a task, or raise ValueError."""
    try:
        return MLX_TRAINER_REGISTRY[task]
    except KeyError as exc:
        supported = ", ".join(MLX_TRAINER_REGISTRY.keys())
        raise ValueError(
            f"MLX backend does not support task '{task}'. "
            f"Supported: {supported}. Use backend=transformers for full task coverage."
        ) from exc
