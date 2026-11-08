"""Native vocabulary expansion and trainable new tokens for MLX text SFT (#1994).

Provides deterministic token registration on the tokenizer, additive parameter
slices for input embeddings and language-model projection heads, and artifact
serialization/loading helpers that preserve frozen base weights under MLX autograd.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_EXTENDED_CLASSES: dict[str, type] = {}


def _get_extended_classes() -> tuple[type, type]:
    """Construct ExtendedEmbedding and ExtendedLinear lazily on first access."""
    if "ExtendedEmbedding" not in _EXTENDED_CLASSES:
        import mlx.core as mx
        import mlx.nn as nn

        class ExtendedEmbedding(nn.Module):
            """Additive parameter slice for token embeddings in MLX.

            Leaves the original base embedding weights 100% frozen while keeping the
            newly added token embedding rows trainable in the autograd tree.
            """

            def __init__(self, base_embed: nn.Embedding, new_weights: mx.array) -> None:
                super().__init__()
                self.base_embed = base_embed
                self.new_weights = new_weights
                if hasattr(base_embed, "freeze"):
                    base_embed.freeze()

            def __call__(self, x: mx.array) -> mx.array:
                full_w = mx.concatenate([self.base_embed.weight, self.new_weights], axis=0)
                return full_w[x]

            def as_linear(self, x: mx.array) -> mx.array:
                """Support tied-weight models without copying base weights."""
                if hasattr(self.base_embed, "as_linear"):
                    base_logits = self.base_embed.as_linear(x)
                else:
                    base_logits = x @ self.base_embed.weight.T
                new_logits = x @ self.new_weights.T
                return mx.concatenate([base_logits, new_logits], axis=-1)

            @property
            def weight(self) -> mx.array:
                return mx.concatenate([self.base_embed.weight, self.new_weights], axis=0)

        class ExtendedLinear(nn.Module):
            """Additive parameter slice for language-model output projection in MLX.

            Leaves the original base projection weights 100% frozen while keeping the
            newly added token projection rows trainable in the autograd tree.
            """

            def __init__(self, base_head: nn.Linear, new_weights: mx.array) -> None:
                super().__init__()
                self.base_head = base_head
                self.new_weights = new_weights
                if hasattr(base_head, "freeze"):
                    base_head.freeze()

            def __call__(self, x: mx.array) -> mx.array:
                base_logits = self.base_head(x)
                new_logits = x @ self.new_weights.T
                return mx.concatenate([base_logits, new_logits], axis=-1)

            @property
            def weight(self) -> mx.array:
                return mx.concatenate([self.base_head.weight, self.new_weights], axis=0)

        _EXTENDED_CLASSES["ExtendedEmbedding"] = ExtendedEmbedding
        _EXTENDED_CLASSES["ExtendedLinear"] = ExtendedLinear

    return _EXTENDED_CLASSES["ExtendedEmbedding"], _EXTENDED_CLASSES["ExtendedLinear"]


def __getattr__(name: str) -> Any:
    if name in ("ExtendedEmbedding", "ExtendedLinear"):
        embed_cls, linear_cls = _get_extended_classes()
        return embed_cls if name == "ExtendedEmbedding" else linear_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_mlx_tokens(tokenizer: Any, data_cfg: Any) -> tuple[int, list[str]]:
    """Register regular and new special tokens on an MLX-wrapped tokenizer.

    Deduplicates input tokens, skips tokens already present in the vocabulary,
    and gives special tokens precedence if a string appears in both lists.

    Args:
        tokenizer: An MLX TokenizerWrapper or Hugging Face PreTrainedTokenizer.
        data_cfg: The data configuration object containing ``add_new_tokens``
            and ``new_special_tokens``.

    Returns:
        A tuple of ``(delta_v, added_tokens_list)`` where ``delta_v`` is the
        number of newly registered unique tokens.
    """
    regular = list(dict.fromkeys(getattr(data_cfg, "add_new_tokens", None) or []))
    special = list(dict.fromkeys(getattr(data_cfg, "new_special_tokens", None) or []))
    if not regular and not special:
        return 0, []

    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)

    existing: set[str] = set()
    if hasattr(hf_tokenizer, "get_vocab"):
        try:
            existing = set(hf_tokenizer.get_vocab().keys())
        except (AttributeError, ValueError):
            existing = set()

    # Special tokens take precedence if the same string appears in both lists.
    special = [t for t in special if t not in existing]
    special_set = set(special)
    regular = [t for t in regular if t not in existing and t not in special_set]

    added_tokens_list: list[str] = []

    if special and hasattr(hf_tokenizer, "add_special_tokens"):
        for tok in special:
            res = hf_tokenizer.add_special_tokens({"additional_special_tokens": [tok]})
            if res and int(res) > 0:
                added_tokens_list.append(tok)

    if regular and hasattr(hf_tokenizer, "add_tokens"):
        for tok in regular:
            res = hf_tokenizer.add_tokens([tok])
            if res and int(res) > 0:
                added_tokens_list.append(tok)

    return len(added_tokens_list), added_tokens_list


def attach_mlx_vocab_extension(
    model: Any,
    delta_v: int,
    init_std: float = 0.02,
) -> tuple[bool, Any, Any | None]:
    """Attach additive trainable parameter slices to an MLX causal LM.

    Args:
        model: An MLX causal language model (e.g. Llama).
        delta_v: Number of new token rows to add.
        init_std: Standard deviation for initializing new token rows.

    Returns:
        A tuple of ``(is_tied, extended_embedding, extended_linear_or_none)``.
    """
    import mlx.core as mx

    embed_cls, linear_cls = _get_extended_classes()

    inner = getattr(model, "model", model)
    if not hasattr(inner, "embed_tokens"):
        raise ValueError(f"MLX model {type(model)} has no embed_tokens attribute")

    base_embed = inner.embed_tokens
    hidden_dim = base_embed.weight.shape[1]

    # Check for tied word embeddings
    is_tied = False
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        is_tied = True
    elif hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False):
        is_tied = True
    elif not hasattr(model, "lm_head") or getattr(model, "lm_head", None) is None:
        is_tied = True
    elif getattr(model.lm_head, "weight", None) is getattr(base_embed, "weight", None):
        is_tied = True

    # Compute reasonable initialization scale
    try:
        base_std = float(mx.std(base_embed.weight).item())
        if base_std > 0:
            init_std = base_std
    except (AttributeError, ValueError):
        pass

    embed_dtype = base_embed.weight.dtype
    new_embed_w = (mx.random.normal((delta_v, hidden_dim)) * init_std).astype(embed_dtype)

    if is_tied:
        extended_embed = embed_cls(base_embed, new_embed_w)
        inner.embed_tokens = extended_embed
        extended_head = None
    else:
        head_dtype = model.lm_head.weight.dtype
        new_head_w = (mx.random.normal((delta_v, hidden_dim)) * init_std).astype(head_dtype)
        extended_embed = embed_cls(base_embed, new_embed_w)
        inner.embed_tokens = extended_embed
        extended_head = linear_cls(model.lm_head, new_head_w)
        model.lm_head = extended_head

    # Update vocab_size metadata on model if present
    base_vocab_size = base_embed.weight.shape[0]
    new_vocab_size = base_vocab_size + delta_v
    if hasattr(model, "args") and hasattr(model.args, "vocab_size"):
        model.args.vocab_size = new_vocab_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        model.config.vocab_size = new_vocab_size

    return is_tied, extended_embed, extended_head


def save_expanded_vocab_artifact(
    output_dir: Path | str,
    model: Any,
    tokenizer: Any,
    base_vocab_size: int,
    is_tied: bool,
    added_tokens: list[str],
    extended_embed: Any,
    extended_head: Any | None = None,
) -> None:
    """Save newly trained vocabulary rows, tokenizer, and expansion metadata.

    1. Writes ``added_tokens.safetensors`` containing only the newly trained rows.
    2. Writes ``added_tokens_info.json`` with vocabulary metadata.
    3. Saves tokenizer assets via ``save_pretrained``.
    4. Strips extension weights from ``adapters.safetensors`` so it contains only
       ordinary LoRA parameters.
    5. Injects expansion metadata into ``adapter_config.json``.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save added token weights
    added_tensors: dict[str, Any] = {
        "new_embed_tokens": extended_embed.new_weights
    }
    if not is_tied and extended_head is not None:
        added_tensors["new_lm_head"] = extended_head.new_weights

    mx.save_safetensors(str(output_path / "added_tokens.safetensors"), added_tensors)

    # 2. Write metadata
    info = {
        "base_vocab_size": base_vocab_size,
        "added_vocab_size": len(added_tokens),
        "is_tied": is_tied,
        "added_tokens": added_tokens,
    }
    (output_path / "added_tokens_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    # 3. Save tokenizer
    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    if hasattr(hf_tokenizer, "save_pretrained"):
        hf_tokenizer.save_pretrained(str(output_path))
    elif hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_path))

    # 4. Clean adapters.safetensors if it exists
    adapter_file = output_path / "adapters.safetensors"
    if adapter_file.is_file():
        try:
            weights = dict(tree_flatten(mx.load(str(adapter_file))))
            cleaned = {
                k: v for k, v in weights.items()
                if (
                    not k.endswith("new_weights")
                    and "new_embed_tokens" not in k
                    and "new_lm_head" not in k
                )
            }
            if len(cleaned) < len(weights):
                mx.save_safetensors(str(adapter_file), cleaned)
        except (OSError, ValueError, KeyError):
            pass

    # 5. Inject metadata into adapter_config.json
    cfg_file = output_path / "adapter_config.json"
    if cfg_file.is_file():
        try:
            cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
            cfg_dict["vocab_expansion"] = True
            cfg_dict["base_vocab_size"] = base_vocab_size
            cfg_dict["added_vocab_size"] = len(added_tokens)
            cfg_dict["added_tokens"] = added_tokens
            cfg_dict["is_tied"] = is_tied
            cfg_file.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
        except (OSError, ValueError, KeyError):
            pass


def load_expanded_vocab_artifact(
    artifact_dir: Path | str,
    model: Any,
    tokenizer: Any | None = None,
) -> tuple[bool, Any, Any | None]:
    """Restore trained vocabulary extension and attach it to an MLX model."""
    import mlx.core as mx

    embed_cls, linear_cls = _get_extended_classes()

    artifact_path = Path(artifact_dir)
    added_file = artifact_path / "added_tokens.safetensors"
    info_file = artifact_path / "added_tokens_info.json"

    if not added_file.is_file() or not info_file.is_file():
        raise FileNotFoundError(
            f"Expanded vocabulary artifact files missing in {artifact_path}"
        )

    info = json.loads(info_file.read_text(encoding="utf-8"))
    weights = mx.load(str(added_file))
    is_tied = bool(info.get("is_tied", False))

    inner = getattr(model, "model", model)
    base_embed = inner.embed_tokens
    if isinstance(base_embed, embed_cls):
        base_embed = base_embed.base_embed

    new_embed_w = weights["new_embed_tokens"].astype(base_embed.weight.dtype)
    extended_embed = embed_cls(base_embed, new_embed_w)
    inner.embed_tokens = extended_embed

    extended_head = None
    if not is_tied and "new_lm_head" in weights:
        base_head = model.lm_head
        if isinstance(base_head, linear_cls):
            base_head = base_head.base_head
        new_head_w = weights["new_lm_head"].astype(base_head.weight.dtype)
        extended_head = linear_cls(base_head, new_head_w)
        model.lm_head = extended_head

    base_vocab_size = int(info["base_vocab_size"])
    added_vocab_size = int(info["added_vocab_size"])
    new_vocab_size = base_vocab_size + added_vocab_size

    if hasattr(model, "args") and hasattr(model.args, "vocab_size"):
        model.args.vocab_size = new_vocab_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        model.config.vocab_size = new_vocab_size

    if tokenizer is not None and "added_tokens" in info:
        hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
        if hasattr(hf_tok, "add_tokens"):
            hf_tok.add_tokens(info["added_tokens"])

    return is_tied, extended_embed, extended_head
