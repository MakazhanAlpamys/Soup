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


class TokenRegistrationResult(tuple):
    """Result of token registration: (delta_v, added_tokens_list).

    Subclasses tuple for backward compatibility so callers can unpack
    ``delta_v, added = register_mlx_tokens(...)`` directly, while exposing
    ``.special_tokens`` and ``.regular_tokens`` for serialization.
    """

    delta_v: int
    added_tokens: list[str]
    special_tokens: list[str]
    regular_tokens: list[str]

    def __new__(
        cls,
        delta_v: int,
        added_tokens: list[str],
        special: list[str] | None = None,
        regular: list[str] | None = None,
    ) -> TokenRegistrationResult:
        obj = super().__new__(cls, (delta_v, added_tokens))
        obj.delta_v = delta_v
        obj.added_tokens = added_tokens
        obj.special_tokens = list(special or [])
        obj.regular_tokens = list(regular or [])
        return obj

    def __reduce__(self):
        return (
            self.__class__,
            (self.delta_v, self.added_tokens, self.special_tokens, self.regular_tokens),
        )


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
                self.new_weights = new_weights.astype(base_embed.weight.dtype)
                if hasattr(base_embed, "freeze"):
                    base_embed.freeze()

            def __call__(self, x: mx.array) -> mx.array:
                base_vocab = self.base_embed.weight.shape[0]
                is_new = x >= base_vocab
                base_ids = mx.where(is_new, 0, x)
                new_ids = mx.where(is_new, x - base_vocab, 0)
                base_out = self.base_embed(base_ids)
                new_out = self.new_weights[new_ids]
                return mx.where(is_new[..., None], new_out, base_out)

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
                self.new_weights = new_weights.astype(base_head.weight.dtype)
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


def register_mlx_tokens(tokenizer: Any, data_cfg: Any) -> TokenRegistrationResult:
    """Register regular and new special tokens on an MLX-wrapped tokenizer.

    Deduplicates input tokens, skips tokens already present in the vocabulary,
    and gives special tokens precedence if a string appears in both lists.

    Registration order:
    Special tokens are registered first, followed by regular tokens. If a token
    appears in both lists, it is registered as special. This matches the semantic
    priority of control markers over plain vocabulary.

    Args:
        tokenizer: An MLX TokenizerWrapper or Hugging Face PreTrainedTokenizer.
        data_cfg: The data configuration object containing ``add_new_tokens``
            and ``new_special_tokens``.

    Returns:
        A ``TokenRegistrationResult`` tuple of ``(delta_v, added_tokens_list)``
        with ``.special_tokens`` and ``.regular_tokens`` attributes.
    """
    regular = list(dict.fromkeys(getattr(data_cfg, "add_new_tokens", None) or []))
    special = list(dict.fromkeys(getattr(data_cfg, "new_special_tokens", None) or []))
    if not regular and not special:
        return TokenRegistrationResult(0, [], [], [])

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

    added_special_list: list[str] = []
    added_regular_list: list[str] = []

    if special and hasattr(hf_tokenizer, "add_special_tokens"):
        # Combine with existing additional special tokens in one call so that
        # Hugging Face tokenizers do not replace earlier special tokens.
        existing_specials = [
            t for t in getattr(hf_tokenizer, "all_special_tokens", [])
            if t not in (
                getattr(hf_tokenizer, "unk_token", None),
                getattr(hf_tokenizer, "bos_token", None),
                getattr(hf_tokenizer, "eos_token", None),
                getattr(hf_tokenizer, "pad_token", None),
            )
        ]
        combined = list(dict.fromkeys(existing_specials + special))
        res = hf_tokenizer.add_special_tokens({"additional_special_tokens": combined})
        if res and int(res) > 0:
            added_special_list.extend(special)

    if regular and hasattr(hf_tokenizer, "add_tokens"):
        res = hf_tokenizer.add_tokens(regular)
        if res and int(res) > 0:
            added_regular_list.extend(regular)

    all_added = added_special_list + added_regular_list
    return TokenRegistrationResult(
        len(all_added),
        all_added,
        added_special_list,
        added_regular_list,
    )


def attach_mlx_vocab_extension(
    model: Any,
    delta_v: int,
    init_std: float = 0.02,
    seed: int | None = None,
) -> tuple[bool, Any, Any | None]:
    """Attach additive trainable parameter slices to an MLX causal LM.

    Args:
        model: An MLX causal language model (e.g. Llama).
        delta_v: Number of new token rows to add.
        init_std: Standard deviation for initializing new token rows.
        seed: Optional integer seed for reproducible weight initialization.

    Returns:
        A tuple of ``(is_tied, extended_embedding, extended_linear_or_none)``.
    """
    if delta_v <= 0:
        raise ValueError(f"delta_v must be positive, got {delta_v}")

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

    if seed is not None:
        mx.random.seed(int(seed))

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
    *,
    added_special_tokens: list[str] | None = None,
    added_regular_tokens: list[str] | None = None,
) -> None:
    """Save newly trained vocabulary rows, tokenizer, and expansion metadata.

    1. Writes ``added_tokens.safetensors`` containing only the newly trained rows.
    2. Writes ``added_tokens_info.json`` with vocabulary metadata (separating
       special and regular tokens for faithful reload).
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

    try:
        mx.save_safetensors(str(output_path / "added_tokens.safetensors"), added_tensors)
    except Exception as exc:
        dest = output_path / "added_tokens.safetensors"
        raise RuntimeError(f"Failed to save added token weights to {dest}: {exc}") from exc

    # Determine special vs regular split if not explicitly supplied
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    all_specials = set(getattr(hf_tok, "all_special_tokens", []))
    if added_special_tokens is None:
        added_special_tokens = [t for t in added_tokens if t in all_specials]
    if added_regular_tokens is None:
        added_regular_tokens = [t for t in added_tokens if t not in all_specials]

    # 2. Write metadata
    info = {
        "base_vocab_size": base_vocab_size,
        "added_vocab_size": len(added_tokens),
        "is_tied": is_tied,
        "added_tokens": added_tokens,
        "added_special_tokens": added_special_tokens,
        "added_regular_tokens": added_regular_tokens,
    }
    try:
        (output_path / "added_tokens_info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write metadata to {output_path / 'added_tokens_info.json'}: {exc}"
        ) from exc

    # 3. Save tokenizer
    try:
        if hasattr(hf_tok, "save_pretrained"):
            hf_tok.save_pretrained(str(output_path))
        elif hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(output_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to save tokenizer to {output_path}: {exc}") from exc

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
        except Exception as exc:
            raise RuntimeError(
                f"Failed to strip vocabulary extension weights from {adapter_file}: {exc}"
            ) from exc

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
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inject vocabulary expansion metadata into {cfg_file}: {exc}"
            ) from exc


def load_expanded_vocab_artifact(
    artifact_dir: Path | str,
    model: Any,
    tokenizer: Any | None = None,
) -> tuple[bool, Any, Any | None]:
    """Restore trained vocabulary extension and attach it to an MLX model.

    Validates model dimensions, tie status, and tokenizer token IDs against
    saved artifact metadata, raising descriptive ValueErrors on mismatch.
    """
    import mlx.core as mx

    embed_cls, linear_cls = _get_extended_classes()

    artifact_path = Path(artifact_dir)
    added_file = artifact_path / "added_tokens.safetensors"
    info_file = artifact_path / "added_tokens_info.json"

    if not added_file.is_file():
        raise FileNotFoundError(f"Missing added token weights: {added_file}")
    if not info_file.is_file():
        raise FileNotFoundError(f"Missing added token metadata: {info_file}")

    try:
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError("added_tokens_info.json must contain a JSON object")
        base_vocab_size = int(info["base_vocab_size"])
        added_vocab_size = int(info["added_vocab_size"])
        is_tied = bool(info["is_tied"])
        if not isinstance(info.get("added_tokens"), list):
            raise TypeError("added_tokens must be a list")
        added_tokens = list(info["added_tokens"])
        if "added_special_tokens" in info and not isinstance(info["added_special_tokens"], list):
            raise TypeError("added_special_tokens must be a list")
        if "added_regular_tokens" in info and not isinstance(info["added_regular_tokens"], list):
            raise TypeError("added_regular_tokens must be a list")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Malformed or invalid metadata in {info_file}: {exc}") from exc

    if len(added_tokens) != added_vocab_size:
        raise ValueError(
            f"added_vocab_size ({added_vocab_size}) does not match length of "
            f"added_tokens ({len(added_tokens)}) in {info_file}"
        )

    try:
        weights = mx.load(str(added_file))
    except Exception as exc:
        raise ValueError(f"Failed to load weights from {added_file}: {exc}") from exc

    if "new_embed_tokens" not in weights:
        raise ValueError(f"Missing 'new_embed_tokens' in {added_file}")

    inner = getattr(model, "model", model)
    if not hasattr(inner, "embed_tokens"):
        raise ValueError(f"Model has no embed_tokens attribute: {type(model)}")

    base_embed = inner.embed_tokens
    if isinstance(base_embed, embed_cls):
        base_embed = base_embed.base_embed

    # 1. Validate base_vocab_size matches model's embedding rows
    actual_base_rows = base_embed.weight.shape[0]
    if base_vocab_size != actual_base_rows:
        raise ValueError(
            f"Artifact base_vocab_size ({base_vocab_size}) does not match model "
            f"embedding table rows ({actual_base_rows})"
        )

    hidden_dim = base_embed.weight.shape[1]

    # 2. Validate shape of new_embed_tokens
    new_embed_w = weights["new_embed_tokens"]
    expected_embed_shape = (added_vocab_size, hidden_dim)
    if tuple(new_embed_w.shape) != expected_embed_shape:
        raise ValueError(
            f"new_embed_tokens shape {tuple(new_embed_w.shape)} does not match "
            f"expected {expected_embed_shape}"
        )

    # 3. Validate is_tied matches model
    model_is_tied = False
    if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
        model_is_tied = True
    elif hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False):
        model_is_tied = True
    elif not hasattr(model, "lm_head") or getattr(model, "lm_head", None) is None:
        model_is_tied = True
    elif getattr(model.lm_head, "weight", None) is getattr(base_embed, "weight", None):
        model_is_tied = True

    if is_tied != model_is_tied:
        raise ValueError(
            f"Artifact is_tied={is_tied} does not match model tie_word_embeddings={model_is_tied}"
        )

    if not is_tied:
        if "new_lm_head" not in weights:
            raise ValueError(f"Untied artifact {added_file} is missing 'new_lm_head'")
        new_head_w = weights["new_lm_head"]
        expected_head_shape = (added_vocab_size, hidden_dim)
        if tuple(new_head_w.shape) != expected_head_shape:
            raise ValueError(
                f"new_lm_head shape {tuple(new_head_w.shape)} does not match "
                f"expected {expected_head_shape}"
            )

    # 4. Attach extension
    new_embed_w = new_embed_w.astype(base_embed.weight.dtype)
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

    new_vocab_size = base_vocab_size + added_vocab_size
    if hasattr(model, "args") and hasattr(model.args, "vocab_size"):
        model.args.vocab_size = new_vocab_size
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        model.config.vocab_size = new_vocab_size

    # 5. Restore tokenizer special and regular tokens and validate token IDs
    if tokenizer is not None:
        hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
        added_special = info.get("added_special_tokens", [])
        added_regular = info.get("added_regular_tokens", [])
        if not added_special and not added_regular:
            # Fallback if older artifact without split
            added_regular = added_tokens

        if added_special and hasattr(hf_tok, "add_special_tokens"):
            existing_specials = [
                t for t in getattr(hf_tok, "all_special_tokens", [])
                if t not in (
                    getattr(hf_tok, "unk_token", None),
                    getattr(hf_tok, "bos_token", None),
                    getattr(hf_tok, "eos_token", None),
                    getattr(hf_tok, "pad_token", None),
                )
            ]
            combined = list(dict.fromkeys(existing_specials + added_special))
            hf_tok.add_special_tokens({"additional_special_tokens": combined})

        if added_regular and hasattr(hf_tok, "add_tokens"):
            hf_tok.add_tokens(added_regular)

        # Validate token IDs match base_vocab_size + position
        if hasattr(hf_tok, "convert_tokens_to_ids"):
            for i, tok in enumerate(added_tokens):
                expected_id = base_vocab_size + i
                actual_id = hf_tok.convert_tokens_to_ids(tok)
                if actual_id != expected_id:
                    raise ValueError(
                        f"Token {tok!r} was assigned ID {actual_id}, but expected {expected_id} "
                        f"(base_vocab_size {base_vocab_size} + index {i})"
                    )

    return is_tied, extended_embed, extended_head
