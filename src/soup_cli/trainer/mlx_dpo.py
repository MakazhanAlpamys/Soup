"""MLX DPO trainer — Apple Silicon preference alignment (Part E of v0.25.0)."""

from __future__ import annotations

from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()


class MLXDPOTrainerWrapper:
    """Minimal DPO training wrapper for MLX backend."""

    def __init__(self, config: SoupConfig) -> None:
        self.config = config
        self.model = None
        self.ref_model = None
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
        console.print(f"[dim]Loading MLX model for DPO: {cfg.base}[/]")
        self.model, self.tokenizer = load_mlx_model(
            cfg.base, quantization=cfg.training.quantization
        )
        # Reference model: frozen copy of base
        self.ref_model, _ = load_mlx_model(cfg.base, quantization=cfg.training.quantization)
        self._dataset = dataset

    def train(self, resume_from: Optional[str] = None) -> None:
        """MLX DPO training loop (delegates to mlx_lm.dpo when available)."""
        self._require_mlx()
        # Delegation to a real MLX DPO implementation would go here.
        raise NotImplementedError(
            "MLX DPO training loop requires mlx-lm ≥ 0.25 with DPO support. "
            "This wrapper is scaffolding for the v0.25.0 release; full "
            "training loop lands when mlx-lm upstream lands its DPO helper."
        )
