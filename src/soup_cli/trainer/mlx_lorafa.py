"""Native frozen-A projected AdamW, pinned to PEFT 0.21.2 FP32 semantics."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

SETTINGS = {
    "reference": "peft==0.21.2",
    "regularizer": 1e-8,
    "pinv_rtol": "rank * float32_eps",
    "precision": "float32",
    "betas": [0.9, 0.999],
    "epsilon": 1e-6,
    "correct_bias": True,
    "decay_order": "after_update",
    "clipping": "raw_global_before_projection",
    "resume": "weights_only",
}


def adapter_tensors(model: Any) -> dict:
    from mlx.utils import tree_flatten

    return {n: v for n, v in tree_flatten(model.parameters()) if n.endswith((".lora_a", ".lora_b"))}


def build_lorafa_optimizer(plan: Any, model: Any) -> Any:
    """Cache a FP32 Gram pseudoinverse from loaded A, then freeze only A."""
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten

    from soup_cli.trainer.mlx_optim import build_lr_schedule

    if plan.optimizer_name != "AdamW":
        raise ValueError("MLX use_lorafa requires an AdamW optimizer plan")
    started = time.perf_counter()
    pairs = adapter_tensors(model)
    expected_a = {n[:-7] for n in pairs if n.endswith(".lora_a")}
    expected_b = {n[:-7] for n in pairs if n.endswith(".lora_b")}
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if not expected_a or expected_a != expected_b or trainable.keys() != pairs.keys():
        raise ValueError("MLX use_lorafa requires complete ordinary linear A/B pairs")
    modules = dict(model.named_modules())
    projections = {}
    for prefix in sorted(expected_a):
        a, b = pairs[prefix + ".lora_a"], pairs[prefix + ".lora_b"]
        scale = float(modules[prefix].scale)
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != b.shape[0]
            or a.dtype != mx.float32
            or b.dtype != mx.float32
            or not math.isfinite(scale)
            or scale <= 0
            or scale > math.sqrt(float(np.finfo(np.float32).max))
        ):
            raise ValueError("MLX use_lorafa requires compatible FP32 A/B and positive scale")
        mx.eval(a, b)
        a_np, b_np = np.array(a), np.array(b)
        if not np.isfinite(a_np).all() or not np.isfinite(b_np).all():
            raise ValueError("MLX use_lorafa requires finite A/B tensors")
        with np.errstate(over="ignore", invalid="ignore"):
            gram = a_np.T @ a_np + np.float32(1e-8) * np.eye(a.shape[1], dtype=np.float32)
        if not np.isfinite(gram).all():
            raise ValueError("MLX use_lorafa requires a finite FP32 Gram matrix")
        try:
            projection = np.linalg.pinv(gram, rcond=a.shape[1] * np.finfo(np.float32).eps)
        except np.linalg.LinAlgError as exc:
            raise ValueError("MLX use_lorafa Gram pseudoinverse did not converge") from exc
        projection = projection / np.float32(scale**2)
        if not np.isfinite(projection).all():
            raise ValueError("MLX use_lorafa Gram projection is not finite")
        projections[prefix + ".lora_b"] = mx.array(projection)
    # Derived state must be built after warm-start, before freezing A.
    for prefix in expected_a:
        modules[prefix].freeze(keys=["lora_a"], recurse=False)

    class ProjectedAdamW(optim.Optimizer):
        def __init__(self) -> None:
            super().__init__()
            self._maybe_schedule("learning_rate", build_lr_schedule(plan))
            self.state["lorafa"] = {
                "projections": projections,
                "beta1": mx.array(0.9),
                "beta2": mx.array(0.999),
                "epsilon": mx.array(1e-6),
                "weight_decay": mx.array(plan.weight_decay),
            }
            self.setup_seconds = time.perf_counter() - started

        def init_single(self, parameter, state):
            state["m"] = mx.zeros_like(parameter)
            state["v"] = mx.zeros_like(parameter)

        def apply_gradients(self, gradients, parameters):
            flat = dict(tree_flatten(gradients))
            projectors = self.state["lorafa"]["projections"]
            if flat.keys() != projectors.keys():
                raise ValueError("MLX use_lorafa gradients must contain exactly B tensors")
            corrected = tree_unflatten([(n, projectors[n] @ g) for n, g in flat.items()])
            return super().apply_gradients(corrected, parameters)

        def apply_single(self, gradient, parameter, state):
            settings = self.state["lorafa"]
            beta1, beta2 = settings["beta1"], settings["beta2"]
            state["m"] = beta1 * state["m"] + (1 - beta1) * gradient
            state["v"] = beta2 * state["v"] + (1 - beta2) * mx.square(gradient)
            rate = self.learning_rate * mx.sqrt(1 - beta2**self.step) / (1 - beta1**self.step)
            updated = parameter - rate * state["m"] / (mx.sqrt(state["v"]) + settings["epsilon"])
            return updated * (1 - self.learning_rate * settings["weight_decay"])

    return ProjectedAdamW()


def load_fa_checkpoint(model: Any, checkpoint: str) -> None:
    """Validate the full A/B set before weights-only warm-start changes state."""
    import mlx.core as mx

    weights, expected = mx.load(checkpoint), adapter_tensors(model)
    if weights.keys() != expected.keys() or any(
        weights[n].shape != expected[n].shape
        or weights[n].dtype != mx.float32
        or not bool(mx.all(mx.isfinite(weights[n])).item())
        for n in weights
    ):
        raise ValueError("MLX use_lorafa checkpoint requires complete compatible finite FP32 A/B")
    model.load_weights(list(weights.items()), strict=False)


def train_lorafa(
    *,
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args,
    training_callback,
    loss=None,
    iterate_batches=None,
):
    """Single-device compiled SFT with an explicit complete A/B save contract.

    mlx-lm saves only trainable tensors and has no save hook. Its batching/loss/
    evaluation/checkpointing helpers are reused; frozen A is explicitly serialized.
    The wrapper rounds iterations to complete accumulation windows before this call.
    """
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from mlx_lm.tuner import trainer as native

    if mx.distributed.init().size() != 1:
        raise ValueError("MLX use_lorafa currently requires a single device")
    loss = loss or native.default_loss
    iterate_batches = iterate_batches or native.iterate_batches
    if args.grad_checkpoint:
        native.grad_checkpoint(model.layers[0])
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state, mx.random.state]
    value_and_grad = nn.value_and_grad(model, loss)
    accumulation = args.grad_accumulation_steps

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, previous, update):
        (value, tokens), gradient = value_and_grad(model, *batch)
        if previous is not None:
            gradient = tree_map(lambda a, b: a + b, gradient, previous)
        if update:
            gradient = tree_map(lambda g: g / accumulation, gradient)
            optimizer.update(model, gradient)
            gradient = None
        return value, tokens, gradient

    def save(path):
        weights = adapter_tensors(model)
        mx.eval(weights)
        mx.save_safetensors(str(path), weights)

    model.train()
    previous, losses, tokens, count, trained_tokens = None, 0, 0, 0, 0
    started = time.perf_counter()
    for iteration, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            loop=True,
            comm_group=mx.distributed.init(),
        ),
    ):
        if val_dataset and (
            iteration == 1 or iteration % args.steps_per_eval == 0 or iteration == args.iters
        ):
            tic = time.perf_counter()
            value = native.evaluate(
                model,
                val_dataset,
                loss=loss,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            model.train()
            training_callback.on_val_loss_report(
                dict(iteration=iteration - 1, val_loss=value, val_time=time.perf_counter() - tic)
            )
        value, n_tokens, previous = step(batch, previous, iteration % accumulation == 0)
        losses, tokens, count = losses + value, tokens + n_tokens, count + 1
        mx.eval(state, losses, tokens, previous)
        native._clear_cache(args.clear_cache_threshold)
        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            duration = max(time.perf_counter() - started, 1e-9)
            trained_tokens += int(tokens.item())
            training_callback.on_train_loss_report(
                dict(
                    iteration=iteration,
                    train_loss=float(losses.item()) / count,
                    learning_rate=float(optimizer.learning_rate.item()),
                    iterations_per_second=count / duration,
                    tokens_per_second=int(tokens.item()) / duration,
                    trained_tokens=trained_tokens,
                    peak_memory=mx.get_peak_memory() / 1e9,
                )
            )
            losses, tokens, count, started = 0, 0, 0, time.perf_counter()
        if iteration % args.steps_per_save == 0:
            save(args.adapter_file)
            save(Path(args.adapter_file).parent / f"{iteration:07d}_adapters.safetensors")
    save(args.adapter_file)


def check_fa_output(output: Path, metadata: dict) -> None:
    """Do not relabel old checkpoint weights before a new run's first save."""
    import json

    if not any(output.glob("*adapters.safetensors")):
        return
    try:
        previous = json.loads((output / "adapter_config.json").read_text())
        fields = ("fine_tune_type", "lora_parameters", "model", "num_layers", "soup_lora")
        if isinstance(previous, dict) and all(previous.get(k) == metadata.get(k) for k in fields):
            return
    except (OSError, ValueError):
        pass
    raise ValueError("MLX use_lorafa requires a fresh output directory for a different recipe")
