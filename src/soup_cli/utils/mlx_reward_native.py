"""Lazy native tensor operations shared by reward training and strict reload."""

from __future__ import annotations

from typing import Any

DENSE_REFUSAL = "MLX reward: dense Llama backbone required."


def require_mlx() -> None:
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    MLXSFTTrainerWrapper._require_mlx(None)
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MLX backend requires the 'mlx' and 'mlx-lm' packages. "
            'Install with: pip install "soup-cli[mlx]"'
        ) from exc


def check_dense_llama(backbone: Any) -> None:
    import mlx.nn as nn
    from mlx_lm.models.llama import Model

    if type(backbone) is not Model or any(
        isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        for _, module in backbone.named_modules()
    ):
        raise ValueError(DENSE_REFUSAL)


def make_reward_model(
    backbone: Any,
    seed: int,
    *,
    gradient_checkpointing: bool = False,
) -> Any:
    """Head on the final norm, never the LM vocabulary projection."""
    import mlx.core as mx
    import mlx.nn as nn

    check_dense_llama(backbone)

    class RewardModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            # nn.Linear's default RNG initialization must not move the training
            # RNG: construct the head with its one explicitly keyed parameter.
            self.score = {
                "weight": mx.random.normal(
                    (1, backbone.args.hidden_size),
                    dtype=mx.float32,
                    key=mx.random.key(seed),
                )
                * 0.02
            }

        def __call__(self, tokens: Any, mask: Any) -> Any:
            width = tokens.shape[1]
            positions = mx.arange(width)[None, :]
            order = mx.argsort(mx.where(mask, positions, width + positions), axis=1)
            compact = mx.take_along_axis(tokens, order, axis=1)
            lengths = mask.sum(axis=1).astype(mx.int32)
            if gradient_checkpointing and self.training:

                def forward(parameters: Any, ids: Any) -> Any:
                    self.backbone.model.update(parameters)
                    return self.backbone.model(mx.stop_gradient(ids))

                hidden = mx.checkpoint(forward)(
                    self.backbone.model.trainable_parameters(),
                    compact,
                )
            else:
                hidden = self.backbone.model(compact)
            final = hidden[mx.arange(tokens.shape[0]), lengths - 1].astype(mx.float32)
            return (final @ self.score["weight"].T).squeeze(-1)

    return RewardModel()


def pair_loss(model: Any, tokens: Any, mask: Any) -> Any:
    """First half chosen, second half rejected; one scalar logit per sequence."""
    import mlx.core as mx

    scores = model(tokens, mask)
    chosen, rejected = mx.split(scores, 2)
    return mx.mean(mx.logaddexp(mx.array(0.0), -(chosen - rejected)))


def pair_arrays(pairs: list, pad_id: int) -> tuple[Any, Any]:
    from soup_cli.utils.mlx_reward_data import padded_arrays

    return padded_arrays([p.chosen for p in pairs] + [p.rejected for p in pairs], pad_id)


def accumulated_gradients(model: Any, chunks: list, pad_id: int) -> tuple[float, Any]:
    """Pair-weight microbatch means; normalization precedes the single clip/update."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    loss_grad = nn.value_and_grad(model, pair_loss)
    total, count, gradients = 0.0, 0, None
    for chunk in chunks:
        loss, grad = loss_grad(model, *pair_arrays(chunk, pad_id))
        grad = tree_map(lambda value: value * len(chunk), grad)
        gradients = (
            grad
            if gradients is None
            else tree_map(
                lambda left, right: left + right,
                gradients,
                grad,
            )
        )
        mx.eval(loss, gradients)
        if not bool(mx.isfinite(loss).item()):
            raise FloatingPointError("reward_model train: non-finite loss. Check model numerics.")
        total += float(loss.item()) * len(chunk)
        count += len(chunk)
    return total / count, tree_map(lambda value: value / count, gradients)


def evaluate_pairs(model: Any, pairs: list, pad_id: int, batch_size: int) -> dict:
    """Pair-weight all rows, restoring training mode and MLX RNG even on failure."""
    import mlx.core as mx

    training = model.training
    state = list(mx.random.state)
    model.eval()
    total_loss = total_margin = correct = 0.0
    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            scores = model(*pair_arrays(chunk, pad_id))
            chosen, rejected = mx.split(scores, 2)
            margin = chosen - rejected
            loss = mx.logaddexp(mx.array(0.0), -margin)
            mx.eval(loss, margin)
            if not bool(mx.all(mx.isfinite(scores)).item()):
                raise FloatingPointError(
                    "reward_model val: non-finite score. Check model numerics."
                )
            total_loss += float(loss.sum().item())
            total_margin += float(margin.sum().item())
            correct += float((margin > 0).sum().item())
    finally:
        mx.random.state = state
        model.train(training)
    return {
        "val_loss": total_loss / len(pairs),
        "val_margin": total_margin / len(pairs),
        "val_accuracy": correct / len(pairs),
    }
