"""Bounded, isolated text-SFT packing for native full-attention Llama."""

from collections.abc import Iterator
from typing import Any


def validate_packing_config(cfg: Any) -> None:
    """Refuse combinations outside the qualified ordinary-LoRA slice."""
    if cfg.task != "sft" or cfg.modality != "text":
        raise ValueError("MLX packing supports text SFT only")
    if cfg.training.batch_size not in (1, "auto"):
        raise ValueError("MLX packing requires batch_size: 1 (one pack per microbatch)")
    if cfg.data.streaming:
        raise ValueError("MLX packing does not support streaming datasets")
    if cfg.training.gradient_checkpointing:
        raise ValueError("MLX packing requires gradient_checkpointing: false")
    lora = cfg.training.lora
    if lora.use_dora or lora.rank_pattern or lora.alpha_pattern:
        raise ValueError("MLX packing supports ordinary LoRA only")


def validate_packing_model(model: Any) -> None:
    """The packed forward below mirrors only this explicit upstream layout."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
    from packaging.version import Version

    args = getattr(model, "args", None)
    if not Version("0.31.2") <= Version(mx.__version__) < Version("0.33"):
        raise ValueError("MLX packing requires mlx >=0.31.2,<0.33")
    if not Version("0.31.3") <= Version(mlx_lm.__version__) < Version("0.33"):
        raise ValueError("MLX packing requires mlx-lm >=0.31.3,<0.33")
    if any(isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)) and
           module.bits != 4 for _, module in model.named_modules()):
        raise ValueError("MLX packing supports dense or 4-bit base weights only")
    if (type(model).__module__ != "mlx_lm.models.llama" or
            getattr(args, "model_type", None) != "llama" or
            getattr(args, "rope_scaling", None) or
            getattr(args, "sliding_window", None) or
            any(kind != "full_attention" for kind in getattr(args, "layer_types", [])) or
            any(type(layer.self_attn.rope) is not nn.RoPE for layer in model.layers)):
        raise ValueError(
            "MLX packing supports full-attention Llama with unscaled RoPE only; "
            "other architectures, including hybrid/recurrent models, are unsupported"
        )


class PackedDataset:
    """Ordered next-fit packs; no target is moved across its segment boundary."""

    def __init__(self, dataset: Any, *, kind: str, max_length: int, vocab_size: int):
        from soup_cli.trainer.mlx_pretokenized import PretokenizedDataset

        if kind not in ("labels", "mask", "span"):
            raise ValueError("packing target kind must be labels, mask or span")
        self._packs: list[tuple[list[int], list[int], list[int], list[int]]] = []
        current: tuple[list[int], list[int], list[int], list[int]] = ([], [], [], [])
        records = 0
        supervised = 0
        for index in range(len(dataset)):
            tokens, target = dataset[index]
            tokens = list(tokens[:max_length])
            if kind == "labels":
                labels = list(target[:max_length])
            elif kind == "mask":
                if any(type(value) not in (int, bool) or value not in (0, 1)
                       for value in target):
                    raise ValueError("packing token masks must be binary")
                labels = [token if keep else -100 for token, keep in zip(tokens, target)]
            else:
                if type(target) is not int or target < 0:
                    raise ValueError("packing span offset must be a nonnegative integer")
                labels = [token if pos >= target else -100 for pos, token in enumerate(tokens)]
            # A standalone first token is never a causal target. In a pack it
            # must not become a transition from the previous example.
            if labels:
                labels[0] = -100
            checked = PretokenizedDataset(
                [{"input_ids": tokens, "labels": labels}],
                vocab_size=vocab_size, max_length=max_length,
            )[0]
            tokens, labels = checked["input_ids"], checked["labels"]
            if len(current[0]) + len(tokens) > max_length:
                self._packs.append(current)
                current = ([], [], [], [])
            segment = records
            current[0].extend(tokens)
            current[1].extend(labels)
            current[2].extend([segment] * len(tokens))
            current[3].extend(range(len(tokens)))
            records += 1
            supervised += sum(label != -100 for label in labels)
        if current[0]:
            self._packs.append(current)
        if not self._packs:
            raise ValueError("MLX packing requires at least one nonempty supervised record")
        self.metadata = {
            "strategy": "ordered-next-fit-isolated-llama",
            "records": records, "packs": len(self._packs),
            "supervised_tokens": supervised,
            "input_tokens": sum(len(row[0]) for row in self._packs),
            "capacity_tokens": len(self._packs) * max_length,
        }
        self.metadata["utilization"] = (
            self.metadata["input_tokens"] / self.metadata["capacity_tokens"]
        )

    def __len__(self) -> int:
        return len(self._packs)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], list[int], list[int]]:
        return self._packs[index]


def packed_iterate_batches(
    dataset: Any, batch_size: int, max_seq_length: int, loop: bool = False,
    seed: Any = None, comm_group: Any = None, *, accumulation: int = 1,
) -> Iterator[Any]:
    """One pack per microbatch; normalize each accumulation group by its targets."""
    import mlx.core as mx
    import numpy as np

    if batch_size != 1:
        raise ValueError("MLX packing requires batch_size: 1")
    if comm_group is not None and comm_group.size() != 1:
        raise ValueError("MLX packing supports single-process training only")
    group_size = accumulation if loop else 1
    if loop and len(dataset) % group_size:
        raise ValueError("MLX packing requires pack count divisible by gradient_accumulation_steps")
    rng = np.random.default_rng(seed)
    while True:
        indices = rng.permutation(len(dataset)) if loop else range(len(dataset))
        indices = list(indices)
        for start in range(0, len(indices), group_size):
            rows = [dataset[int(index)] for index in indices[start:start + group_size]]
            count = sum(sum(label != -100 for label in row[1]) for row in rows)
            for tokens, labels, segments, positions in rows:
                if len(tokens) > max_seq_length:
                    raise ValueError("packed record exceeds the configured context")
                yield mx.array([tokens]), (
                    mx.array([labels]), mx.array([segments]), mx.array([positions]),
                    mx.array(group_size / count, dtype=mx.float32),
                )
        if not loop:
            break


def packed_logits(model: Any, tokens: Any, segments: Any, positions: Any) -> Any:
    """Native Llama forward with explicit segment mask and per-token RoPE offsets.

    Each token is temporarily a length-one sequence for the existing RoPE
    module, whose vector offset supplies its actual position. Model modules and
    parameter names are never replaced, preserving optimizer/save/reload paths.
    """
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    batch_size, length = tokens.shape
    indices = mx.arange(length)
    mask = ((segments[:, :, None] == segments[:, None, :]) &
            (indices[:, None] >= indices[None, :]))[:, None, :, :]
    hidden = model.model.embed_tokens(tokens)

    def rotate(values: Any, rope: Any) -> Any:
        # [B,H,T,D] -> [B*T,H,1,D]; offset has one position per B*T.
        count, heads, width, dims = values.shape
        flat = values.transpose(0, 2, 1, 3).reshape(count * width, heads, 1, dims)
        rotated = rope(flat, offset=positions.reshape(-1))
        return rotated.reshape(count, width, heads, dims).transpose(0, 2, 1, 3)

    for layer in model.layers:
        attention = layer.self_attn
        normalized = layer.input_layernorm(hidden)
        queries = attention.q_proj(normalized).reshape(
            batch_size, length, attention.n_heads, -1,
        ).transpose(0, 2, 1, 3)
        keys = attention.k_proj(normalized).reshape(
            batch_size, length, attention.n_kv_heads, -1,
        ).transpose(0, 2, 1, 3)
        values = attention.v_proj(normalized).reshape(
            batch_size, length, attention.n_kv_heads, -1,
        ).transpose(0, 2, 1, 3)
        output = scaled_dot_product_attention(
            rotate(queries, attention.rope), rotate(keys, attention.rope), values,
            cache=None, scale=attention.scale, mask=mask,
        ).transpose(0, 2, 1, 3).reshape(batch_size, length, -1)
        hidden = hidden + attention.o_proj(output)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    hidden = model.model.norm(hidden)
    return (model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings
            else model.lm_head(hidden))


def packed_loss(model: Any, batch: Any, payload: Any) -> tuple[Any, Any]:
    """Report the true pack mean; accumulate gradients by supervised-token count."""
    import mlx.core as mx
    import mlx.nn as nn

    labels, segments, positions, coefficient = payload
    targets = labels[:, 1:]
    selected = targets != -100
    logits = packed_logits(model, batch[:, :-1], segments[:, :-1], positions[:, :-1])
    ce = nn.losses.cross_entropy(logits, mx.maximum(targets, 0)).astype(mx.float32)
    total = (ce * selected).sum()
    count = selected.sum()
    mean = total / count
    # mlx-lm divides accumulated gradients by G. G / sum(group targets)
    # therefore gives the gradient of the group's true token-weighted mean.
    scaled = total * coefficient
    return scaled + mx.stop_gradient(mean - scaled), count
