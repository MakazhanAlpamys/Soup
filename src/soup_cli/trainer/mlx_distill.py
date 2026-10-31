"""Single-device, unquantized native token distillation with a LoRA student.

The dedicated loop normalizes gradients over active targets in an entire
optimizer window, including partial batches/windows. The SFT loop's equal
microbatch means and dropped remainder cannot express that contract.
"""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import version
from pathlib import Path

from rich.console import Console

from soup_cli.trainer.mlx_distill_data import build_examples, collate, compatibility_identity
from soup_cli.trainer.mlx_distill_loss import token_loss_sums
from soup_cli.trainer.mlx_optim import build_optimizer, plan_optimizer
from soup_cli.trainer.mlx_sft import (
    MLXSFTTrainerWrapper,
    _clipping_optimizer,
    _resolved_batch_size,
    build_mlx_adapter_config,
    resolve_mlx_target_keys,
)
from soup_cli.utils.mlx_distill import validate_native_distill

console = Console()


def _publish_final_files(staged, mark_complete):
    """Roll back a failed multi-file replacement, retaining earlier adapters.

    Each rename is atomic on the same filesystem. The run receipt remains
    running until all files arrive; this is not a power-loss-atomic directory
    transaction. If rollback itself fails, leave the backup directory intact.
    """
    import tempfile

    backup_dir = Path(tempfile.mkdtemp(prefix=".distillation-backup-", dir=staged[0][1].parent))
    backups, published = [], []
    try:
        for source, target in staged:
            if target.exists():
                backup = backup_dir / target.name
                target.replace(backup)
                backups.append((backup, target))
            source.replace(target)
            published.append(target)
        mark_complete()
    except BaseException:
        for target in reversed(published):
            target.unlink()
        for backup, target in reversed(backups):
            backup.replace(target)
        backup_dir.rmdir()
        raise
    for backup, _ in backups:
        backup.unlink()
    backup_dir.rmdir()


class MLXDistillTrainerWrapper(MLXSFTTrainerWrapper):
    """CLI trainer contract; teacher state never enters the optimizer or artifact."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.teacher = None
        self.teacher_tokenizer = None

    def setup(self, dataset: dict) -> None:
        validate_native_distill(self.config)
        if any(self.extra_kwargs.get(key) for key in (
            "deepspeed_config", "fsdp_config", "trust_remote_code",
        )):
            raise ValueError("MLX distillation does not support distributed or remote-code loading")
        self._require_mlx()
        self._check_eval_steps_has_rows(dataset)
        if not dataset.get("train"):
            raise ValueError("MLX distillation train split is empty")
        import mlx.core as mx
        import mlx.nn as nn

        from soup_cli.utils.mlx import load_mlx_model

        try:
            cfg = self.config
            if cfg.training.seed is not None:
                mx.random.seed(cfg.training.seed)
            self.model, self.tokenizer = load_mlx_model(cfg.base, quantization="none")
            self.teacher, self.teacher_tokenizer = load_mlx_model(
                cfg.training.teacher_model, quantization="none",
            )
            for name, model in (("student", self.model), ("teacher", self.teacher)):
                if any(isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
                       or (hasattr(module, "bits") and hasattr(module, "scales"))
                       for _, module in model.named_modules()):
                    raise ValueError(f"MLX distillation {name} artifact is actually quantized")
                model.eval()
            self.teacher.freeze()
            probe = mx.zeros((1, 2), dtype=mx.int32)
            student_output, teacher_output = self.model(probe), self.teacher(probe)
            mx.eval(student_output, teacher_output)
            self.tokenizer_identity = compatibility_identity(
                self.tokenizer, self.teacher_tokenizer,
                student_output.shape[-1], teacher_output.shape[-1],
            )
            keys = set(resolve_mlx_target_keys(cfg.training.lora))
            matches = [(name, layer) for parent in [*self.model.layers, self.model]
                       for name, layer in parent.named_modules() if name in keys]
            if keys - {name for name, _ in matches} or any(
                not isinstance(layer, nn.Linear) for _, layer in matches
            ):
                raise ValueError(
                    "MLX distillation lora.target_modules must match dense linear layers"
                )
            self._apply_lora(self.model)
            mx.eval(self.model.parameters())
            from mlx.utils import tree_flatten

            if not tree_flatten(self.model.trainable_parameters()):
                raise ValueError("MLX distillation student has no trainable LoRA tensors")
            args = dict(
                max_length=cfg.data.max_length, responses_only=cfg.data.train_on_responses_only,
            )
            self._train_rows = build_examples(list(dataset["train"]), self.tokenizer,
                                              split="train", **args)
            self._val_rows = build_examples(list(dataset.get("val") or []), self.tokenizer,
                                            split="val", **args)
            self._dataset = dataset
        except BaseException:
            self.teacher = self.teacher_tokenizer = self.model = self.tokenizer = None
            mx.clear_cache()
            raise

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "adapters.safetensors"
        expected = dict(tree_flatten(self.model.trainable_parameters()))
        try:
            weights = mx.load(str(path))
        except (ValueError, RuntimeError, OSError) as exc:
            raise ValueError(f"MLX distillation warm-start is not readable: {path}") from exc
        if set(weights) != set(expected) or any(
            value.shape != expected[name].shape or value.dtype != expected[name].dtype
            or not mx.all(mx.isfinite(value)).item() for name, value in weights.items()
        ):
            raise ValueError(
                "MLX distillation warm-start tensor keys, shapes, dtypes or values differ"
            )
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.parameters())

    def _teacher_logits(self, tokens):
        import mlx.core as mx

        self.teacher.eval()
        return mx.stop_gradient(self.teacher(tokens))

    def _loss_arguments(self):
        from soup_cli.utils.distill import validate_divergence

        tcfg = self.config.training
        return dict(
            divergence=validate_divergence(tcfg.distill_divergence or "forward_kl"),
            temperature=tcfg.distill_temperature or 2.0,
            chunk_size=tcfg.distill_chunk_size, recompute=tcfg.distill_checkpoint,
        )

    @staticmethod
    def _checked_metrics(total, metrics, gradients=None):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        checks = {name: mx.all(mx.isfinite(value)) for name, value in metrics.items()
                  if not name.endswith("_finite")}
        checks.update({name: value for name, value in metrics.items() if name.endswith("_finite")})
        checks["total_loss"] = mx.isfinite(total)
        if gradients is not None:
            checks["student_gradients"] = mx.all(mx.stack([
                mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)
            ]))
        mx.eval(total, metrics, checks, gradients)
        # One evaluation boundary per microbatch, independent of chunk count.
        for name in ("teacher_finite", "student_finite", "ce_sum", "kd_sum",
                     "total_loss", "student_gradients"):
            if name in checks and not checks[name].item():
                raise ValueError(f"MLX distillation non-finite {name}; optimizer update aborted")
        return {"loss_sum": total.item(), "ce_sum": metrics["ce_sum"].item(),
                "kd_sum": metrics["kd_sum"].item()}

    def train(self, display=None, tracker=None, run_id=None, resume_from_checkpoint=None) -> dict:
        import mlx.core as mx

        try:
            return self._train(display, tracker, run_id, resume_from_checkpoint)
        finally:
            self.teacher = self.teacher_tokenizer = None
            if display is not None:
                display.stop()
            mx.clear_cache()

    def _train(self, display, tracker, run_id, resume_from_checkpoint) -> dict:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.nn.utils import checkpoint
        from mlx.utils import tree_flatten, tree_map

        cfg, tcfg = self.config, self.config.training
        batch_size = _resolved_batch_size(tcfg)
        accumulation = tcfg.gradient_accumulation_steps
        batches_per_epoch = math.ceil(len(self._train_rows) / batch_size)
        updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
        total_updates = tcfg.epochs * updates_per_epoch
        plan = plan_optimizer(
            lr=tcfg.lr, optimizer=tcfg.optimizer, scheduler=tcfg.scheduler,
            warmup_ratio=tcfg.warmup_ratio, weight_decay=tcfg.weight_decay,
            total_updates=total_updates,
        )
        optimizer = _clipping_optimizer(build_optimizer(plan), tcfg.max_grad_norm)
        optimizer.init(self.model.trainable_parameters())
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if type(pad_id) is not int:
            raise ValueError("MLX distillation tokenizer needs an integer pad or EOS ID")
        loss_arguments = self._loss_arguments()
        # Whole-student activation recomputation preserves module/parameter names
        # and never mutates a shared layer class (which would affect the teacher).
        forward = self.model
        if tcfg.gradient_checkpointing:
            def with_random_key(tokens, key):
                # Recomputing a stochastic forward must reuse its original key,
                # not draw fresh dropout masks or advance the next update's RNG.
                mx.random.state[0][:] = mx.stop_gradient(key)
                logits = self.model(mx.stop_gradient(tokens))
                return logits, mx.random.state[0]

            checkpointed = checkpoint(self.model, with_random_key)

            def forward(tokens):
                logits, next_key = checkpointed(tokens, mx.random.state[0])
                mx.random.state[0][:] = next_key
                return logits

        def loss(tokens, labels, active, teacher_logits):
            return token_loss_sums(
                forward(tokens), teacher_logits, labels, active, **loss_arguments,
            )

        step = mx.compile(
            nn.value_and_grad(self.model, loss),
            inputs=[self.model.state, mx.random.state],
            outputs=[self.model.state, mx.random.state],
        )
        output = Path(cfg.output)
        output.mkdir(parents=True, exist_ok=True)
        status = output / "distillation_run.json"

        def write_status(state):
            temp = status.with_suffix(".json.tmp")
            temp.write_text(json.dumps({"status": state, "optimizer_updates": len(history)}))
            temp.replace(status)

        history, evaluations = [], []
        started = time.perf_counter()
        rng = np.random.default_rng(tcfg.data_seed if tcfg.data_seed is not None else tcfg.seed)

        def batch(rows):
            return collate(rows, pad_id=pad_id, max_length=cfg.data.max_length)

        def evaluate(update):
            self.model.eval()
            sums = {key: 0.0 for key in ("loss_sum", "ce_sum", "kd_sum")}
            count = 0
            for start in range(0, len(self._val_rows), batch_size):
                tokens, labels, active = batch(self._val_rows[start:start + batch_size])
                total, metrics = token_loss_sums(
                    self.model(tokens), self._teacher_logits(tokens), labels, active,
                    **loss_arguments,
                )
                values = self._checked_metrics(total, metrics)
                for key in sums:
                    sums[key] += values[key]
                count += active.size
            evaluations.append({"update": update, "active_targets": count,
                                **{("loss" if key == "loss_sum" else
                                    key.removesuffix("_sum") + "_loss"): value / count
                                   for key, value in sums.items()}})
            if tracker is not None and run_id:
                tracker.log_metrics(run_id=run_id, step=update, loss=None,
                                    val_loss=evaluations[-1]["loss"])
            self.model.train()

        try:
            write_status("running")
            if resume_from_checkpoint:
                self._load_checkpoint_weights(str(resume_from_checkpoint))
                console.print(
                    "[yellow]MLX distillation: weights-only warm start; optimizer restarts.[/]"
                )
            self.teacher.freeze()
            self.teacher.eval()
            self.model.train()
            if display is not None:
                display.start(total_updates)
            for epoch in range(tcfg.epochs):
                order = rng.permutation(len(self._train_rows)).tolist()
                for start in range(0, len(order), batch_size * accumulation):
                    window_started = time.perf_counter()
                    indices = order[start:start + batch_size * accumulation]
                    denominator = sum(self._train_rows[i].active_count for i in indices)
                    gradients = None
                    sums = {key: 0.0 for key in ("loss_sum", "ce_sum", "kd_sum")}
                    for offset in range(0, len(indices), batch_size):
                        tokens, labels, active = batch([
                            self._train_rows[i] for i in indices[offset:offset + batch_size]
                        ])
                        teacher_logits = self._teacher_logits(tokens)
                        (total, metrics), current = step(
                            tokens, labels, active, teacher_logits,
                        )
                        values = self._checked_metrics(total, metrics, current)
                        gradients = current if gradients is None else tree_map(
                            lambda left, right: left + right, gradients, current,
                        )
                        for key in sums:
                            sums[key] += values[key]
                    gradients = tree_map(lambda value: value / denominator, gradients)
                    self._checked_metrics(total, metrics, gradients)
                    optimizer.update(self.model, gradients)
                    mx.eval(self.model.parameters(), optimizer.state)
                    lr = float(optimizer.learning_rate.item())
                    update = len(history) + 1
                    entry = {
                        "update": update, "epoch": epoch + 1, "active_targets": denominator,
                        "loss": sums["loss_sum"] / denominator,
                        "ce_loss": sums["ce_sum"] / denominator,
                        "kd_loss": sums["kd_sum"] / denominator, "lr": lr,
                        "duration_secs": time.perf_counter() - window_started,
                    }
                    history.append(entry)
                    if update % max(1, tcfg.logging_steps) == 0 or update == total_updates:
                        console.print(
                            f"[dim]KD update {update}/{total_updates}: loss={entry['loss']:.6f} "
                            f"CE={entry['ce_loss']:.6f} KD={entry['kd_loss']:.6f} "
                            f"targets={denominator}[/]"
                        )
                        if tracker is not None and run_id:
                            tracker.log_metrics(run_id=run_id, step=update, epoch=epoch + 1,
                                                loss=entry["loss"], lr=lr)
                    if self._val_rows and (update == total_updates or (
                        tcfg.eval_steps is not None and update % tcfg.eval_steps == 0
                    )):
                        evaluate(update)
                    if display is not None:
                        display.update(
                            step=update, epoch=epoch + 1, loss=entry["loss"], lr=lr,
                            speed=1 / max(entry["duration_secs"], 1e-9),
                            gpu_mem=f"{mx.get_peak_memory() / 1e9:.3f} GB",
                            val_loss=evaluations[-1]["loss"] if evaluations else None,
                        )
                    if tcfg.save_steps > 0 and update % tcfg.save_steps == 0:
                        checkpoint_temp = output / f".{update:07d}_adapters.tmp.safetensors"
                        try:
                            mx.save_safetensors(str(checkpoint_temp), dict(tree_flatten(
                                self.model.trainable_parameters(),
                            )))
                            checkpoint_temp.replace(output / f"{update:07d}_adapters.safetensors")
                        finally:
                            checkpoint_temp.unlink(missing_ok=True)
            metadata = build_mlx_adapter_config(
                tcfg.lora, adapter_path=str(output), model=cfg.base, task="distill",
                num_layers=len(self.model.layers), teacher_model=tcfg.teacher_model,
                tokenizer_identity=self.tokenizer_identity,
                objective={"ce_weight": 0.5, "kd_weight": 0.5, **loss_arguments,
                           "reduction": "active causal targets per optimizer window"},
                gradient_checkpointing=tcfg.gradient_checkpointing,
                gradient_checkpointing_scope="whole student forward",
                grad_accumulation_steps=accumulation, batch_size=batch_size,
                eval_steps=tcfg.eval_steps, eval_unit="optimizer updates",
                seed=tcfg.seed, data_seed=tcfg.data_seed,
                train_on_responses_only=cfg.data.train_on_responses_only,
                max_seq_length=cfg.data.max_length, max_grad_norm=tcfg.max_grad_norm,
                parameter_dtypes=sorted({str(value.dtype) for _, value in
                                         tree_flatten(self.model.parameters())}),
                runtime={name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
                **plan.as_metadata(),
            )
            # Stage both final files; the status receipt disambiguates a failed
            # replacement from a completed run and prior recoverable checkpoints.
            weights_temp = output / ".adapters.tmp.safetensors"
            mx.save_safetensors(str(weights_temp), dict(tree_flatten(
                self.model.trainable_parameters(),
            )))
            config_temp = output / ".adapter_config.tmp.json"
            config_temp.write_text(json.dumps(metadata, indent=2))
            metrics_temp = output / ".distillation_metrics.tmp.json"
            metrics_temp.write_text(json.dumps({
                "training": history, "validation": evaluations,
            }, indent=2, allow_nan=False))
            _publish_final_files([
                (config_temp, output / "adapter_config.json"),
                (weights_temp, output / "adapters.safetensors"),
                (metrics_temp, output / "distillation_metrics.json"),
            ], lambda: write_status("complete"))
        except BaseException as exc:
            write_status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            raise
        duration = time.perf_counter() - started
        result = {
            "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
            "loss_summary_kind": "delta" if len(history) > 1 else "single",
            "total_steps": len(history), "optimizer_updates": len(history),
            "duration_secs": duration, "duration": f"{duration:.0f}s", "output_dir": str(output),
            "history": history, "evaluation_history": evaluations,
            "eval_updates": [entry["update"] for entry in evaluations],
        }
        if evaluations:
            result["val_loss"] = evaluations[-1]["loss"]
        return result
