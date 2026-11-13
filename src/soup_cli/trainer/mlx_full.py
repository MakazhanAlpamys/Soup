"""Dense native Llama SFT: token-weighted updates and standalone checkpoints (#1992)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from soup_cli.config.schema import SoupConfig

RESUME_REFUSAL = (
    "MLX full fine-tuning does not support --resume: optimizer/RNG-state "
    "resume is tracked in #1977. Start a new run from a standalone model instead."
)


def prepare_full_model(model: Any, config: SoupConfig) -> dict:
    """Validate the qualified layout, unfreeze base weights, then freeze bottom blocks."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.models.llama import Model

    if mx.distributed.init().size() != 1:
        raise ValueError("MLX full fine-tuning does not support distributed training")
    for name, module in model.named_modules():
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            raise ValueError(
                f"MLX full fine-tuning refuses quantized module {name!r} "
                f"({type(module).__name__}) even with training.quantization: none"
            )
    if (
        type(model) is not Model or model.model_type != "llama"
        or not model.model.layers
        or not isinstance(model.model.embed_tokens, nn.Embedding)
        or not isinstance(model.model.norm, nn.RMSNorm)
        or (not model.args.tie_word_embeddings and not isinstance(model.lm_head, nn.Linear))
    ):
        raise ValueError(
            "MLX full fine-tuning requires the native Llama layout: model.model.layers, "
            "model.model.embed_tokens, model.model.norm and an output head; unknown layer layout"
        )
    parameters = dict(tree_flatten(model.parameters()))
    for name, parameter in parameters.items():
        if parameter.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            raise ValueError(f"MLX full fine-tuning requires dense float weights: {name!r}")
    total_layers = len(model.model.layers)
    training = config.training
    cutoff = training.freeze_layers
    if cutoff is None:
        cutoff = int(total_layers * (training.freeze_ratio or 0))
    if not 0 <= cutoff <= total_layers:
        raise ValueError(
            f"training.freeze_layers={cutoff} is out of range for {total_layers} Llama layers"
        )
    model.unfreeze()
    for layer in model.model.layers[:cutoff]:
        layer.freeze()
    # Dense Adam updates in fp16 can underflow their moments/epsilon; keep master weights in fp32.
    model.update(tree_map(lambda parameter: parameter.astype(mx.float32),
                          model.trainable_parameters()))
    trainable = dict(tree_flatten(model.trainable_parameters()))
    trainable_count = sum(parameter.size for parameter in trainable.values())
    if trainable_count == 0:
        raise ValueError("MLX full fine-tuning freeze plan leaves nothing trainable (all-frozen)")
    frozen = parameters.keys() - trainable.keys()
    return {
        "fine_tune_type": "full", "architecture": "llama",
        "freeze_cutoff": cutoff, "num_layers": total_layers,
        "trainable_parameters": trainable_count,
        "frozen_parameters": sum(parameters[name].size for name in frozen),
        "trainable_tensors": len(trainable), "frozen_tensors": len(frozen),
        "trainable_names": sorted(trainable), "frozen_names": sorted(frozen),
        "checkpoint_parameter_dtypes": sorted({str(value.dtype) for value in parameters.values()}),
        "trainable_dtype": "float32",
        "tie_word_embeddings": model.args.tie_word_embeddings,
    }


def save_full_model(
    path: Path, model: Any, tokenizer: Any, model_config: dict, metadata: dict,
) -> None:
    """Save all parameters, including frozen tensors, without depending on the base path."""
    import json

    from mlx_lm.utils import save_config, save_model

    save_model(path, model)
    save_config(dict(model_config), path / "config.json")
    tokenizer.save_pretrained(path)
    (path / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train_full(
    *, model: Any, optimizer: Any, train_dataset: Any, val_dataset: Any,
    args: Any, training_callback: Any, save_checkpoint: Callable[[int], None],
    loss: Callable | None = None, iterate_batches: Callable | None = None,
) -> None:
    """Train dense parameters, weighting an accumulation group by supervised tokens.

    mlx-lm's LoRA loop averages micro-batch means and saves trainable tensors only.
    Dense SFT needs a token-weighted objective and complete model checkpoints.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from mlx_lm.tuner.trainer import default_loss
    from mlx_lm.tuner.trainer import iterate_batches as default_iterate

    from soup_cli.trainer.mlx_sft import evaluate_all_rows

    loss = loss or default_loss
    iterate_batches = iterate_batches or default_iterate

    def summed_loss(current_model, *batch):
        mean, tokens = loss(current_model, *batch)
        return mean * tokens, tokens

    gradient = nn.value_and_grad(model, summed_loss)
    model.train()
    gradients = None
    group_tokens = mx.array(0)
    report_loss = mx.array(0.0)
    report_tokens = mx.array(0)
    report_steps = 0
    report_time = 0.0
    trained_tokens = 0
    next_save = args.steps_per_save
    # Checkpoint the layer call locally: upstream's helper patches the class globally.
    layer_type = type(model.model.layers[0])
    original_call = layer_type.__call__
    if args.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint

        grad_checkpoint(model.model.layers[0])
    try:
        stream = iterate_batches(
            dataset=train_dataset, batch_size=args.batch_size,
            max_seq_length=args.max_seq_length, loop=True,
        )
        for iteration, batch in zip(range(1, args.iters + 1), stream):
            if val_dataset is not None and (
                iteration == 1 or iteration % args.steps_per_eval == 0
                or iteration == args.iters
            ):
                started = time.perf_counter()
                value = evaluate_all_rows(
                    model, val_dataset, batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length, loss_fn=loss, iterate_fn=iterate_batches,
                )
                training_callback.on_val_loss_report({
                    "iteration": iteration - 1, "val_loss": value,
                    "val_time": time.perf_counter() - started,
                })
            started = time.perf_counter()
            (value, tokens), current_gradients = gradient(model, *batch)
            mx.eval(value, tokens, current_gradients)
            if tokens.item() <= 0 or not mx.isfinite(value).item():
                raise FloatingPointError(
                    "MLX full fine-tuning: empty or non-finite supervised loss"
                )
            gradients = current_gradients if gradients is None else tree_map(
                lambda previous, current: previous + current, gradients, current_gradients,
            )
            del current_gradients
            group_tokens = group_tokens + tokens
            if iteration % args.grad_accumulation_steps == 0:
                normalized = tree_map(lambda value: value / group_tokens, gradients)
                optimizer.update(model, normalized)
                mx.eval(model.state, optimizer.state)
                del normalized
                gradients = None
                group_tokens = mx.array(0)
                if iteration >= next_save:
                    save_checkpoint(iteration)
                    next_save = (iteration // args.steps_per_save + 1) * args.steps_per_save
            report_loss = report_loss + value
            report_tokens = report_tokens + tokens
            report_steps += 1
            mx.eval(report_loss, report_tokens, gradients)
            report_time += time.perf_counter() - started
            if iteration % args.steps_per_report == 0 or iteration == args.iters:
                count = int(report_tokens.item())
                trained_tokens += count
                training_callback.on_train_loss_report({
                    "iteration": iteration, "train_loss": (report_loss / report_tokens).item(),
                    "learning_rate": optimizer.learning_rate.item(),
                    "iterations_per_second": report_steps / report_time,
                    "tokens_per_second": count / report_time, "trained_tokens": trained_tokens,
                    "peak_memory": mx.get_peak_memory() / 1e9,
                })
                report_loss, report_tokens = mx.array(0.0), mx.array(0)
                report_steps, report_time = 0, 0.0
    finally:
        if args.grad_checkpoint:
            layer_type.__call__ = original_call
