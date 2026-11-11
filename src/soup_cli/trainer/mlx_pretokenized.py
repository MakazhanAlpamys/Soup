"""Validated cached IDs and explicit labels for native text SFT."""

from collections.abc import Mapping, Sequence


def _integers(value, field, size=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a nonempty integer sequence")
    if size is not None and len(value) != size:
        raise ValueError(f"{field} length must match input_ids")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{field} must contain integer values, not bools or floats")
    return list(value)


class PretokenizedDataset:
    """Validate every row before training; processing never invokes a tokenizer.

    Labels are explicit target IDs, not just a boolean selection of input IDs.
    Right-padding must already have ignored labels; it cannot become context for
    a later real token. Split/row order is retained for the existing native sampler.
    """

    def __init__(self, data: Sequence, *, vocab_size: int, max_length: int):
        from soup_cli.data.loss_mask import ensure_causal_loss_target

        self._data = []
        for index, row in enumerate(data, 1):
            try:
                if not isinstance(row, Mapping):
                    raise ValueError("cached record must be a mapping")
                tokens = _integers(row.get("input_ids"), "input_ids")
                labels = _integers(row.get("labels"), "labels", len(tokens))
                if any(token < 0 or token >= vocab_size for token in tokens) or any(
                    label != -100 and (label < 0 or label >= vocab_size) for label in labels
                ):
                    raise ValueError("input_ids/labels are outside the model vocabulary")
                checked = {**row, "input_ids": tokens, "labels": labels}
                if "attention_mask" in row:
                    attention = _integers(row["attention_mask"], "attention_mask", len(tokens))
                    if any(value not in (0, 1) for value in attention):
                        raise ValueError("attention_mask must be binary")
                    if any(left < right for left, right in zip(attention, attention[1:])):
                        raise ValueError("cached attention_mask requires right padding only")
                    if any(not keep and label != -100 for keep, label in zip(attention, labels)):
                        raise ValueError("padding must have ignored labels (-100)")
                    checked["attention_mask"] = attention
                ensure_causal_loss_target(labels[:max_length], max_length=max_length)
                self._data.append(checked)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"pre_tokenized row {index}: {exc}") from exc

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def process(self, row):
        return row["input_ids"], row["labels"]


def labeled_iterate_batches(*args, **kwargs):
    """Use the native sampler, padding target labels with their ignore sentinel."""
    from soup_cli.trainer.mlx_masking import masked_iterate_batches

    return masked_iterate_batches(*args, **kwargs, _target_pad=-100)


def labeled_loss(model, batch, labels):
    """Causal cross-entropy over the stored targets, with -100 ignored."""
    import mlx.core as mx
    import mlx.nn as nn

    targets = labels[:, 1:]
    mask = targets != -100
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(logits, mx.maximum(targets, 0))
    count = mask.sum()
    return (ce * mask).astype(mx.float32).sum() / mx.maximum(count, 1), count
