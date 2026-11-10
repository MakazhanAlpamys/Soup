"""Native raw-text LoRA pretraining: independent windows and token-weighted updates.

The loop is intentionally bounded to continued pretraining. Loading, LoRA target
resolution, optimizer schedules, clipping and artifact consumers share the MLX
primitives; no Transformers training packages are needed.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from rich.console import Console

from soup_cli.data.mlx_pretrain import batch_windows, plan_updates, prepare_windows
from soup_cli.trainer.mlx_sft import (
    MLXSFTTrainerWrapper,
    _clipping_optimizer,
    _resolved_batch_size,
    build_mlx_adapter_config,
    resolve_mlx_target_keys,
    scheduled_eval_updates,
)

console = Console()
WINDOW_POLICY = "independent-documents/one-context-overlap/keep-tail/v1"


def causal_loss_sum(logits, targets, mask):
    """Unnormalized float32 CE on already-shifted labels, with positional padding."""
    import mlx.core as mx
    import mlx.nn as nn

    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("MLX pretrain logits, shifted targets and mask shapes must agree")
    # The batch builder guarantees nonzero masks; refuse direct invalid calls as well.
    if not bool((mask.sum() > 0).item()):
        raise ValueError("MLX pretrain loss has no supervised targets")
    return (nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
            * mask).sum()


def evaluate_windows(model, windows, *, batch_size, pad_id):
    """Full held-out token-weighted loss, without consuming RNG or updating parameters."""
    import mlx.core as mx

    training = model.training
    model.eval()
    numerator = count = 0
    try:
        for start in range(0, len(windows), batch_size):
            x, y, mask = batch_windows(windows[start:start + batch_size], pad_id)
            loss = causal_loss_sum(model(mx.array(x)), mx.array(y), mx.array(mask))
            numerator += float(loss.item())
            count += sum(map(sum, mask))
    finally:
        model.train(training)
    if not count:
        raise ValueError("MLX pretrain validation has no supervised targets")
    loss = numerator / count
    if not math.isfinite(loss):
        raise FloatingPointError("MLX pretrain validation loss is non-finite")
    return loss


def token_weighted_update(model, optimizer, batches, *, pad_id):
    """Sum gradients first, normalize by all real targets, then clip/update once."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    def objective(m, inputs, targets, mask):
        return causal_loss_sum(m(inputs), targets, mask)

    value_grad = nn.value_and_grad(model, objective)
    total = count = 0
    gradients = None
    for windows in batches:
        x, y, mask = batch_windows(windows, pad_id)
        value, grad = value_grad(model, mx.array(x), mx.array(y), mx.array(mask))
        mx.eval(value, grad)
        total += float(value.item())
        count += sum(map(sum, mask))
        gradients = grad if gradients is None else tree_map(lambda a, b: a + b, gradients, grad)
        mx.eval(gradients)
    if not count or not math.isfinite(total):
        raise FloatingPointError("MLX pretrain update requires finite loss and real targets")
    gradients = tree_map(lambda g: g / count, gradients)
    if any(not bool(mx.all(mx.isfinite(g)).item()) for _, g in _flatten(gradients)):
        raise FloatingPointError("MLX pretrain gradient is non-finite")
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    if any(not bool(mx.all(mx.isfinite(v)).item())
           for _, v in _flatten(model.trainable_parameters())):
        raise FloatingPointError("MLX pretrain adapter became non-finite after the update")
    return total / count, int(count)


def _flatten(tree):
    from mlx.utils import tree_flatten

    return tree_flatten(tree)


class MLXPretrainTrainerWrapper(MLXSFTTrainerWrapper):
    """A frozen dense/affine-4-bit causal LM with ordinary linear LoRA."""

    def _require_mlx(self):
        super()._require_mlx()
        try:
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Native MLX pretrain requires soup-cli[mlx] on Apple Silicon",
            ) from exc

    def setup(self, dataset: dict) -> None:
        from soup_cli.config.mlx_pretrain import refuse_unsupported_pretrain
        from soup_cli.utils.mlx import is_apple_silicon, load_mlx_model

        refuse_unsupported_pretrain(self.config)
        if not is_apple_silicon():
            raise RuntimeError(
                "Native MLX pretrain requires an Apple Silicon Mac and soup-cli[mlx]",
            )
        self._require_mlx()
        if not dataset.get("train"):
            raise ValueError("MLX pretrain training split has no usable raw-text documents")
        if self.config.training.eval_steps is not None and not dataset.get("val"):
            raise ValueError("MLX pretrain training.eval_steps requires a validation split")
        started = time.perf_counter()
        self.model, self.tokenizer = load_mlx_model(
            self.config.base, quantization=self.config.training.quantization,
        )
        self._check_base()
        self.loading_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self._dataset = dataset
        self.train_windows, self.train_stats = prepare_windows(
            dataset["train"], self.tokenizer, self.config.data.max_length,
        )
        self.val_windows, self.val_stats = (prepare_windows(
            dataset["val"], self.tokenizer, self.config.data.max_length, split="validation",
        ) if dataset.get("val") else ([], None))
        hf = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.pad_id = hf.pad_token_id
        if self.pad_id is None:
            self.pad_id = hf.eos_token_id if hf.eos_token_id is not None else 0
        self.preparation_seconds = time.perf_counter() - started

    def _check_base(self):
        from soup_cli.trainer.mlx_grpo import qualify_base

        quantization = qualify_base(self.model, self.config.base, context="MLX pretrain")
        if (quantization is not None) != (self.config.training.quantization == "4bit"):
            raise ValueError(
                "MLX pretrain training.quantization does not match actual base tensors; "
                "use none for dense or 4bit for a pre-quantized 4-bit affine model",
            )
        self._base_quantization = quantization

    def _apply_lora(self, model):
        import mlx.nn as nn

        keys = set(resolve_mlx_target_keys(self.config.training.lora))
        matches = [(name, module) for parent in [*model.layers, model]
                   for name, module in parent.named_modules() if name in keys]
        if keys != {name for name, _ in matches} or any(
            not isinstance(module, (nn.Linear, nn.QuantizedLinear)) for _, module in matches
        ):
            raise ValueError(
                f"MLX pretrain requires resolved linear LoRA target keys: {sorted(keys)}",
            )
        super()._apply_lora(model)
        trainable = dict(_flatten(model.trainable_parameters()))
        if not trainable or any(k.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"}
                                for k in trainable):
            raise ValueError("MLX pretrain requires trainable LoRA tensors and a frozen base")

    def _load_checkpoint_weights(self, checkpoint_path):
        from soup_cli.utils.mlx_pretrain_checkpoint import load_snapshot

        load_snapshot(self.model, Path(checkpoint_path), self._resume_contract)

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        import mlx.core as mx
        import mlx.nn as nn

        from soup_cli.trainer.loss_summary import summarize_training_loss
        from soup_cli.trainer.mlx_grpo import write_atomically
        from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer
        from soup_cli.utils.mlx_pretrain_checkpoint import (
            base_identity,
            save_snapshot,
            tokenizer_identity,
        )
        from soup_cli.utils.seeding import resolve_training_seed

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("MLX pretrain: call setup(dataset) before train()")
        self._check_base()
        started = time.perf_counter()
        cfg, training = self.config, self.config.training
        seed = resolve_training_seed(training)
        data_seed = seed if training.data_seed is None else training.data_seed
        batch_size = _resolved_batch_size(training)
        updates = plan_updates(len(self.train_windows), batch_size,
                               training.gradient_accumulation_steps, training.epochs, data_seed)
        plan = plan_optimizer(lr=training.lr, optimizer=training.optimizer,
                              scheduler=training.scheduler, warmup_ratio=training.warmup_ratio,
                              weight_decay=training.weight_decay, total_updates=len(updates))
        for warning in plan.warnings:
            console.print(f"[yellow]{warning}[/]")
        mx.random.seed(seed)
        self._apply_lora(self.model)
        contract = {"task": "pretrain", "model": base_identity(cfg.base),
                    "tokenizer_sha256": tokenizer_identity(self.tokenizer),
                    "window_policy": WINDOW_POLICY, "max_length": cfg.data.max_length,
                    "lora_parameters": build_mlx_adapter_config(training.lora,
                                                               adapter_path="")["lora_parameters"],
                    "base_quantization": self._base_quantization}
        self._resume_contract = contract
        if resume_from_checkpoint is not None:
            self._load_checkpoint_weights(resume_from_checkpoint)
            console.print("[yellow]MLX pretrain weights-only warm start; optimizer, seed, "
                          "schedule and data position restart. Exact resume is not supported.[/]")
        optimizer = _clipping_optimizer(build_optimizer(plan), training.max_grad_norm)
        self.optimizer = optimizer
        output = Path(cfg.output)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {**build_mlx_adapter_config(training.lora, adapter_path=str(output)),
                    "model": str(cfg.base), "num_layers": len(self.model.layers),
                    "task": "pretrain", "contract": contract,
                    "base_quantization": self._base_quantization, **plan.as_metadata(),
                    "seed": seed, "data_seed": data_seed, "batch_size": batch_size,
                    "gradient_accumulation_steps": training.gradient_accumulation_steps,
                    "eval_steps": training.eval_steps, "eval_unit": "optimizer updates",
                    "save_steps": training.save_steps, "logging_steps": training.logging_steps,
                    "max_grad_norm": training.max_grad_norm, "epochs": training.epochs,
                    "data": self._dataset.get("pretrain_data_stats", {}),
                    "train_coverage": self.train_stats, "validation_coverage": self.val_stats,
                    "warm_start": resume_from_checkpoint is not None,
                    "resume_contract": "adapter weights only; new optimizer/schedule/order"}
        # This is a real evaluated graph warmup, not a fabricated compile timing.
        # The bounded loop uses eager MLX graphs (no explicit mx.compile); report that.
        warmup_started = time.perf_counter()
        initial_val = (evaluate_windows(self.model, self.val_windows, batch_size=batch_size,
                                       pad_id=self.pad_id) if self.val_windows else None)
        x, y, mask = batch_windows(self.train_windows[:batch_size], self.pad_id)
        self.model.eval()
        mx.eval(causal_loss_sum(self.model(mx.array(x)), mx.array(y), mx.array(mask)))

        def warmup_objective(model):
            return causal_loss_sum(model(mx.array(x)), mx.array(y), mx.array(mask))

        # Inference mode disables dropout, keeping adapter/data RNG untouched.
        mx.eval(nn.value_and_grad(self.model, warmup_objective)(self.model))
        warmup_seconds = time.perf_counter() - warmup_started
        mx.reset_peak_memory()
        self.model.train()
        if training.eval_steps is not None:
            eval_at = set(scheduled_eval_updates(len(updates), training.eval_steps))
        else:
            eval_at = {i for i, (epoch, _) in enumerate(updates, 1)
                       if i == len(updates) or updates[i][0] != epoch}
        losses, evaluated, consumed = [], [], 0
        update_seconds = 0.
        try:
            for step, (epoch, group) in enumerate(updates, 1):
                tick = time.perf_counter()
                batches = [[self.train_windows[i] for i in indices] for indices in group]
                loss, tokens = token_weighted_update(self.model, optimizer, batches,
                                                     pad_id=self.pad_id)
                update_seconds += time.perf_counter() - tick
                consumed += tokens
                losses.append(loss)
                val = None
                if self.val_windows and step in eval_at:
                    val = evaluate_windows(self.model, self.val_windows, batch_size=batch_size,
                                           pad_id=self.pad_id)
                    evaluated.append({"update": step, "val_loss": val})
                if step % training.logging_steps == 0 or step == len(updates):
                    console.print(f"[dim]MLX pretrain update {step}/{len(updates)}: "
                                  f"loss={loss:.6f}, supervised tokens={consumed}[/]")
                if tracker is not None and run_id is not None:
                    tracker.log_metrics(run_id, step=step, epoch=float(epoch), loss=loss,
                                        lr=float(optimizer.learning_rate.item()),
                                        **({"val_loss": val} if val is not None else {}))
                if step % training.save_steps == 0:
                    save_snapshot(self.model, output / f"{step:07d}_adapters.safetensors",
                                  {**metadata, "completed_updates": step})
            metadata.update(completed_updates=len(updates), consumed_targets=consumed,
                            initial_val_loss=initial_val, evaluations=evaluated,
                            preparation_seconds=self.preparation_seconds,
                            loading_seconds=self.loading_seconds,
                            graph_warmup_seconds=warmup_seconds, compilation="eager; no mx.compile",
                            update_seconds=update_seconds,
                            supervised_tokens_per_second=consumed / update_seconds,
                            throughput_scope="optimizer updates after one forward/backward warmup; "
                            "new shapes may incur additional runtime kernel preparation",
                            peak_memory_bytes=mx.get_peak_memory(),
                            duration_secs=time.perf_counter() - started)
            save_snapshot(self.model, output / "adapters.safetensors", metadata)
            write_atomically(output / "adapter_config.json", lambda temp: Path(temp).write_text(
                json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8",
            ))
        finally:
            if display is not None:
                display.stop()
        self.model.eval()
        console.print(f"[green]MLX pretraining complete:[/] {output}")
        return {**summarize_training_loss([{"loss": loss} for loss in losses]),
                "losses": losses, "total_steps": len(updates), "completed_updates": len(updates),
                "consumed_targets": consumed, "eval_updates": [v["update"] for v in evaluated],
                "duration_secs": metadata["duration_secs"],
                "duration": f"{metadata['duration_secs']:.0f}s", "output_dir": str(output),
                **({"val_loss": evaluated[-1]["val_loss"], "initial_val_loss": initial_val}
                   if evaluated else {})}
