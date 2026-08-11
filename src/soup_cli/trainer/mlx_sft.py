"""MLX SFT trainer — Apple Silicon fine-tuning via mlx-lm.

Rewritten for mlx-lm >= 0.31 (v0.73.0 fix, PR #362):
- ``mlx_lm.tuner.trainer.train`` no longer takes a ``tokenizer`` argument;
  datasets are built with ``mlx_lm.tuner.datasets.create_dataset``, wrapped in
  ``CacheDataset`` (bare ``ChatDataset`` lacks ``__len__``/``__getitem__``),
  and loss is reported through ``TrainingCallback.on_train_loss_report``.
- LoRA layers are applied with ``mlx_lm.tuner.utils.linear_to_lora_layers``
  after ``model.freeze()``: ``mlx_lm.load`` does NOT freeze, and without it
  every parameter is trainable and the saved "adapter" is actually a full
  fine-tune (172 tensors vs 24 LoRA tensors on a 1.2B model).
- Implements the CLI trainer contract: ``train()`` accepts the common kwargs
  (display/tracker/run_id/resume_from_checkpoint) and returns the metrics
  dict that ``commands/train.py`` expects.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()


class MLXSFTTrainerWrapper:
    """High-level wrapper for MLX supervised fine-tuning."""

    def __init__(self, config: SoupConfig, **kwargs) -> None:
        self.config = config
        # Accepted for CLI-contract parity (trainer_kwargs forward). Stored
        # intentionally unused: trust_remote_code has no MLX meaning and
        # load_mlx_model does not take it.
        self.extra_kwargs = kwargs
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
                "Install with: pip install \"soup-cli[mlx]\""
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

    def _apply_lora(self, model) -> None:
        """Convert the configured linear layers to LoRA (mlx-lm >= 0.31).

        The model must be frozen first: ``mlx_lm.load`` does NOT freeze, and
        without it every parameter is trainable and the saved "adapter" is
        actually a full fine-tune (172 tensors instead of ~64 LoRA weights).
        """
        model.freeze()
        from mlx_lm.tuner.utils import linear_to_lora_layers

        cfg = self.config
        lora_cfg = cfg.training.lora
        target_keys = lora_cfg.target_modules
        if not target_keys or target_keys in (["auto"], "auto"):
            target_keys = ["self_attn.q_proj", "self_attn.v_proj"]
            console.print(
                "[yellow]MLX backend: target_modules: auto cannot be resolved for "
                f"all architectures; defaulting to {target_keys}[/]"
            )
        keys = set(target_keys)
        num_layers = len(getattr(model, "layers", []))
        linear_to_lora_layers(
            model,
            num_layers,
            {
                "rank": int(lora_cfg.r),
                "scale": float(lora_cfg.alpha) / max(1.0, float(lora_cfg.r)),
                "dropout": float(getattr(lora_cfg, "dropout", 0.0) or 0.0),
                "keys": keys,
            },
        )

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        """Run MLX training loop via mlx-lm (mlx-lm >= 0.31 API)."""
        del display, tracker, run_id  # accepted for CLI-contract parity

        if resume_from_checkpoint is not None:
            console.print(
                "[yellow]MLX backend does not support --resume yet; "
                "starting training from scratch.[/]"
            )

        self._require_mlx()

        import mlx.optimizers as optim  # type: ignore
        from mlx_lm.tuner.callbacks import TrainingCallback
        from mlx_lm.tuner.datasets import CacheDataset, create_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train  # type: ignore

        cfg = self.config
        output_dir = Path(cfg.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "MLX backend: setup(dataset) must be called before train()"
            )

        self._apply_lora(self.model)

        batch_size = (
            int(cfg.training.batch_size)
            if isinstance(cfg.training.batch_size, int)
            else 1
        )
        train_rows = list(self._dataset.get("train", []))
        val_rows = list(self._dataset.get("val", []))
        iters = int(
            cfg.training.epochs * max(1, math.ceil(len(train_rows) / batch_size))
        )

        max_seq_length = int(getattr(cfg.data, "max_length", 2048) or 2048)
        steps_per_report = int(getattr(cfg.training, "logging_steps", 10) or 10)
        steps_per_save = int(getattr(cfg.training, "save_steps", 0) or 0)
        if steps_per_save <= 0:
            steps_per_save = iters
        # Real eval cadence when a val split exists; otherwise the value is
        # irrelevant (val_dataset stays None and mlx-lm skips evaluation).
        steps_per_eval = max(1, iters // 4) if val_rows else max(1000, iters + 1)
        args = TrainingArgs(
            batch_size=batch_size,
            iters=iters,
            max_seq_length=max_seq_length,
            steps_per_report=steps_per_report,
            steps_per_eval=steps_per_eval,
            steps_per_save=steps_per_save,
            adapter_file=str(output_dir / "adapters.safetensors"),
        )

        train_dataset = CacheDataset(create_dataset(train_rows, self.tokenizer, args))
        val_dataset = (
            CacheDataset(create_dataset(val_rows, self.tokenizer, args))
            if val_rows
            else None
        )

        optimizer = optim.AdamW(learning_rate=float(cfg.training.lr))

        captured: dict = {}

        class _Callback(TrainingCallback):
            def on_train_loss_report(self, train_info: dict) -> None:
                captured.setdefault("losses", []).append(
                    train_info.get("train_loss", 0.0)
                )

        t0 = time.time()
        train(
            model=self.model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            training_callback=_Callback(),
        )
        duration = time.time() - t0

        # mlx-lm's tuner only saves adapters.safetensors; write the
        # adapter_config.json so the output dir is directly loadable with
        # mlx_lm.load(..., adapter_path=output).
        lora_cfg = cfg.training.lora
        (output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "model": str(cfg.base),
                    "num_layers": len(getattr(self.model, "layers", [])),
                    "batch_size": batch_size,
                    "iters": iters,
                    "learning_rate": float(cfg.training.lr),
                    "steps_per_report": steps_per_report,
                    "steps_per_eval": steps_per_eval,
                    "steps_per_save": steps_per_save,
                    "max_seq_length": max_seq_length,
                    "adapter_path": str(output_dir),
                    "lora_parameters": {
                        "rank": int(lora_cfg.r),
                        "scale": float(lora_cfg.alpha)
                        / max(1.0, float(lora_cfg.r)),
                        "dropout": float(getattr(lora_cfg, "dropout", 0.0) or 0.0),
                        "keys": list(
                            lora_cfg.target_modules
                            if isinstance(lora_cfg.target_modules, list)
                            else [lora_cfg.target_modules]
                        ),
                    },
                    "mask_prompt": False,
                    "grad_checkpoint": False,
                    "grad_accumulation_steps": 1,
                },
                indent=2,
            )
        )

        losses = captured.get("losses") or [0.0]
        result = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "total_steps": iters,
            "duration_secs": duration,
            "duration": f"{duration:.0f}s",
            "output_dir": str(output_dir),
        }
        console.print(f"[green]MLX training complete:[/] {output_dir}")
        return result
