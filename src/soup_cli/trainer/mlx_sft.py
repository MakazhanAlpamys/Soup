"""MLX SFT trainer — Apple Silicon fine-tuning via mlx-lm (Part E of v0.25.0).

MLX is a separate training path from transformers/unsloth — it uses Apple's
unified memory architecture on M1-M4 chips. This trainer wraps ``mlx_lm``'s
LoRA training utilities and bridges to Soup's Rich display and SQLite tracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()


class MLXSFTTrainerWrapper:
    """High-level wrapper for MLX supervised fine-tuning."""

    def __init__(self, config: SoupConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _require_mlx(self) -> None:
        try:
            import mlx  # noqa: F401
            import mlx.core  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MLX backend requires the 'mlx' and 'mlx-lm' packages. "
                "Install with: pip install 'soup-cli[mlx]'"
            ) from exc

    def _check_unsupported(self) -> None:
        tcfg = self.config.training
        unsupported = []
        if tcfg.quantization == "8bit":
            unsupported.append("quantization=8bit (use mlx-community 4bit models)")
        if tcfg.use_galore:
            unsupported.append("GaLore")
        if tcfg.use_ring_attention:
            unsupported.append("Ring Attention")
        if tcfg.use_flash_attn:
            unsupported.append("FlashAttention (MLX has its own attention kernels)")
        if unsupported:
            console.print(
                "[yellow]MLX backend ignores: " + ", ".join(unsupported) + "[/]"
            )

    def setup(self, dataset: dict) -> None:
        """Load MLX model, configure LoRA, prepare dataset."""
        self._require_mlx()
        self._check_unsupported()

        from soup_cli.utils.mlx import load_mlx_model

        cfg = self.config
        console.print(f"[dim]Loading MLX model: {cfg.base}[/]")
        self.model, self.tokenizer = load_mlx_model(
            cfg.base, quantization=cfg.training.quantization
        )
        console.print(
            f"[green]MLX model loaded:[/] {cfg.base} "
            f"(task={cfg.task}, lora_r={cfg.training.lora.r})"
        )
        self._dataset = dataset

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run MLX training loop.

        This is a thin driver that delegates to ``mlx_lm.lora.train`` when
        available. A proper streaming Rich callback bridge can be added in a
        follow-up release once the mlx-lm training API stabilizes.
        """
        self._require_mlx()

        from mlx_lm.tuner.trainer import TrainingArgs, train  # type: ignore

        cfg = self.config
        output_dir = Path(cfg.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = (
            int(cfg.training.batch_size)
            if isinstance(cfg.training.batch_size, int)
            else 1
        )
        iters = int(
            cfg.training.epochs * max(1, len(self._dataset.get("train", [])))
        )
        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            steps_per_report=cfg.training.logging_steps,
            steps_per_eval=cfg.training.save_steps,
            adapter_file=str(output_dir / "adapters.safetensors"),
        )

        train(
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=None,
            train_dataset=self._dataset.get("train", []),
            val_dataset=self._dataset.get("val", []),
            training_callback=None,
            args=args,
        )
        console.print(f"[green]MLX training complete:[/] {output_dir}")
