"""Native dense Llama reward training and strict scalar scoring bundles.

Subclass SFT only to reuse its dependency and LoRA implementation. SFT setup,
chat-mask diagnostics and its LM loop do not run. Task-specific scope policy
is checked by SoupConfig before setup.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console

from soup_cli.trainer.loss_summary import summarize_training_loss
from soup_cli.trainer.mlx_sft import (
    MLXSFTTrainerWrapper,
    _clipping_optimizer,
    _resolved_batch_size,
    mlx_lora_scale,
    resolve_mlx_target_keys,
)

console = Console()
BASE_DOWNLOAD_PATTERNS = (
    "config.json", "model*.safetensors", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "tiktoken.model", "*.tiktoken", "vocab.json", "vocab.txt", "merges.txt",
    "chat_template.jinja", "chat_templates/*.jinja", "hf_quant_config.json",
)
RESUME_REFUSAL = (
    "--resume is not supported for task 'reward_model' on backend 'mlx': scoring bundles "
    "do not save optimizer/RNG state. Start a fresh run in a new output directory."
)


class MLXRewardModelTrainerWrapper(MLXSFTTrainerWrapper):
    """Train ordinary LoRA and a scalar fp32 head with pair-weighted updates."""

    def setup(self, dataset: Any) -> None:
        from soup_cli.utils.mlx_reward_bundle import (
            base_identity,
            load_tokenizer,
            validate_output,
        )
        from soup_cli.utils.mlx_reward_data import prepare_pairs
        from soup_cli.utils.mlx_reward_native import (
            check_dense_llama,
            make_reward_model,
            require_mlx,
        )
        from soup_cli.utils.seeding import resolve_training_seed

        validate_output(Path(self.config.output))
        require_mlx()
        import mlx.core as mx

        from soup_cli.utils.mlx import load_mlx_model

        cfg = self.config
        base = Path(cfg.base)
        if not base.is_dir():
            from huggingface_hub import snapshot_download

            base = Path(snapshot_download(
                cfg.base, allow_patterns=list(BASE_DOWNLOAD_PATTERNS),
                ignore_patterns=["*/*.safetensors"],
            ))
        self._base_identity = base_identity(base)  # custom-code checks precede MLX-LM
        self.tokenizer = load_tokenizer(base, allow_cache_links=True)
        self._pairs, self._train_counts = prepare_pairs(
            dataset.get("train", []),
            self.tokenizer,
            cfg.data.max_length,
            "train",
        )
        validation = list(dataset.get("val", []))
        self._val_pairs, self._val_counts = ([], None)
        if validation:
            self._val_pairs, self._val_counts = prepare_pairs(
                validation,
                self.tokenizer,
                cfg.data.max_length,
                "val",
            )
        for split, counts in (("train", self._train_counts), ("val", self._val_counts)):
            if counts:
                console.print(
                    f"reward_model {split}: retained {counts['retained']}; "
                    f"skipped lost={counts['skipped_lost']}, "
                    f"collapsed={counts['skipped_collapsed']}.",
                    markup=False,
                )
                if counts["first_skip"]:
                    console.print(counts["first_skip"], markup=False)
        seed = resolve_training_seed(cfg.training)
        mx.random.seed(seed)
        backbone, _ = load_mlx_model(str(base), quantization=cfg.training.quantization)
        check_dense_llama(backbone)
        self._apply_lora(backbone)  # freezes the base before LoRA insertion
        self.model = make_reward_model(
            backbone,
            seed,
            gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        )
        lora = cfg.training.lora
        self._contract = {
            "base": self._base_identity,
            "lora": {
                "rank": int(lora.r),
                "scale": mlx_lora_scale(lora),
                "dropout": float(lora.dropout),
                "keys": resolve_mlx_target_keys(lora),
                "layers": len(backbone.layers),
            },
            "tokenizer": {"eos_id": self.tokenizer.eos_token_id, "vocab_size": len(self.tokenizer)},
            "max_length": cfg.data.max_length,
            "head": {"seed": seed, "std": 0.02, "dtype": "float32", "bias": False},
        }

    def train(
        self,
        display: Any = None,
        tracker: Any = None,
        run_id: Any = None,
        resume_from_checkpoint: Any = None,
    ) -> dict:
        if resume_from_checkpoint is not None:
            raise ValueError(RESUME_REFUSAL)
        if self.model is None:
            raise RuntimeError("MLX backend: setup(dataset) must be called before train()")
        import mlx.core as mx

        from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer
        from soup_cli.utils.mlx_reward_bundle import POLICY, publish_bundle
        from soup_cli.utils.mlx_reward_data import plan_windows
        from soup_cli.utils.mlx_reward_native import accumulated_gradients, evaluate_pairs
        from soup_cli.utils.seeding import resolve_training_seed

        tcfg = self.config.training
        seed = resolve_training_seed(tcfg)
        data_seed = seed if tcfg.data_seed is None else int(tcfg.data_seed)
        batch_size = _resolved_batch_size(tcfg)
        plan = plan_windows(
            len(self._pairs), batch_size, tcfg.gradient_accumulation_steps, tcfg.epochs, data_seed
        )
        total_updates = sum(map(len, plan))
        optimizer_plan = plan_optimizer(
            lr=tcfg.lr,
            optimizer=tcfg.optimizer,
            scheduler=tcfg.scheduler,
            warmup_ratio=tcfg.warmup_ratio,
            weight_decay=tcfg.weight_decay,
            total_updates=total_updates,
        )
        for warning in optimizer_plan.warnings:
            console.print(warning, markup=False)
        optimizer = _clipping_optimizer(build_optimizer(optimizer_plan), tcfg.max_grad_norm)
        self.optimizer = optimizer  # allows direct numerical/state qualification
        update, losses, validation, generations = 0, [], {}, []
        evaluations = []
        started = time.monotonic()
        self.model.train()
        if display is not None:
            display.start(total_updates)

        def save() -> None:
            contract = dict(
                self._contract,
                policy=POLICY,
                training={
                    "completed_updates": update,
                    "seed": seed,
                    "data_seed": data_seed,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
                    "optimizer": optimizer_plan.as_metadata(),
                    "train_counts": self._train_counts,
                    "val_counts": self._val_counts,
                },
            )
            generations.append(
                publish_bundle(
                    Path(self.config.output),
                    self.model,
                    self.tokenizer,
                    contract,
                )
            )

        for epoch, windows in enumerate(plan, start=1):
            for window in windows:
                chunks = [[self._pairs[index] for index in indices] for indices in window]
                loss, gradients = accumulated_gradients(
                    self.model,
                    chunks,
                    self.tokenizer.eos_token_id,
                )
                optimizer.update(self.model, gradients)
                mx.eval(self.model.parameters(), optimizer.state)
                update += 1
                losses.append(loss)
                if tcfg.save_steps and update % tcfg.save_steps == 0:
                    save()
                if update % tcfg.logging_steps == 0 or update == total_updates:
                    metrics = {
                        "step": update,
                        "epoch": epoch,
                        "loss": loss,
                        "lr": float(optimizer.learning_rate),
                        "speed": update / max(time.monotonic() - started, 1e-9),
                    }
                    if display is not None:
                        display.update(**metrics)
                    else:
                        console.print(
                            f"reward_model update {update}/{total_updates}: loss={loss:.6f}"
                        )
                    if tracker is not None and run_id:
                        tracker.log_metrics(run_id=run_id, **metrics)
            if self._val_pairs:
                validation = evaluate_pairs(
                    self.model,
                    self._val_pairs,
                    self.tokenizer.eos_token_id,
                    batch_size,
                )
                evaluations.append({"update": update, **validation})
        save()
        return {
            **summarize_training_loss([{"loss": value} for value in losses]),
            **validation,
            "total_steps": update,
            "train_pairs": len(self._pairs),
            "train_counts": self._train_counts,
            "val_counts": self._val_counts,
            "generations": generations,
            "evaluations": evaluations,
            "duration_secs": time.monotonic() - started,
            "duration": f"{time.monotonic() - started:.0f}s",
            "output_dir": str(self.config.output),
        }
