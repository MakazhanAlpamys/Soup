"""MLX GRPO trainer — Apple Silicon reasoning training (Part E of v0.25.0)."""

from __future__ import annotations

from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()


class MLXGRPOTrainerWrapper:
    """Minimal GRPO training wrapper for MLX backend."""

    def __init__(self, config: SoupConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None

    def _require_mlx(self) -> None:
        try:
            import mlx  # noqa: F401
            import mlx.core  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MLX backend requires the 'mlx' and 'mlx-lm' packages. "
                "Install with: pip install 'soup-cli[mlx]'"
            ) from exc

    def setup(self, dataset: dict) -> None:
        self._require_mlx()
        from soup_cli.utils.mlx import load_mlx_model

        cfg = self.config
        console.print(f"[dim]Loading MLX model for GRPO: {cfg.base}[/]")
        self.model, self.tokenizer = load_mlx_model(
            cfg.base, quantization=cfg.training.quantization
        )
        self._dataset = dataset

    def train(self, resume_from: Optional[str] = None) -> None:
        """MLX GRPO training loop.

        Requires a group-generation + reward-scoring loop in MLX. This wrapper
        is scaffolding for the v0.25.0 release.
        """
        self._require_mlx()
        raise NotImplementedError(
            "MLX GRPO training loop is scaffolding in v0.25.0. Full "
            "implementation depends on upstream mlx-lm GRPO support."
        )
