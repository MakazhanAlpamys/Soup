"""Native scalar reward scoring. Importing this public facade uses only stdlib.

Bundles are scoring artifacts, never language-model adapters or resume states.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCORING_ONLY = "MLX reward: scoring-only bundle; use load_reward_scorer/score_completions."


def refuse_reward_artifact(path: str | Path) -> None:
    """Refuse a scoring root/payload or an LM adapter recording one as its base.

    Inspect the recorded local base before consumers create output directories
    or resolve remote models. This is bounded metadata inspection, not loading;
    malformed LM metadata remains the existing consumer's responsibility.
    """
    from soup_cli.utils.mlx_reward_bundle import read_json

    def recognized(root: Path) -> bool:
        if os.path.lexists(root / "reward_model.json") or os.path.lexists(
            root / "reward.safetensors"
        ):
            return True
        try:
            manifest = read_json(root / "manifest.json")
        except (OSError, ValueError):
            return False
        return manifest.get("artifact_type") == "mlx_reward_model"

    root = Path(path)
    if recognized(root):
        raise ValueError(SCORING_ONLY)
    try:
        metadata = read_json(root / "adapter_config.json")
    except (OSError, ValueError):
        return
    for key in ("base_model_name_or_path", "model"):
        base = metadata.get(key)
        if isinstance(base, str) and base and recognized(Path(base)):
            raise ValueError(SCORING_ONLY)


@dataclass
class NativeRewardScorer:
    """Loaded base, trained LoRA/head and the bundled tokenizer; reuse across calls."""

    model: Any
    tokenizer: Any
    max_length: int


def load_reward_scorer(
    artifact: str | Path,
    *,
    base: str | Path | None = None,
) -> NativeRewardScorer:
    """Load a checked immutable generation and identical local/cache-only base."""
    from soup_cli.utils.mlx_reward_bundle import load_scorer

    return load_scorer(artifact, base=base)


def score_completions(
    scorer: NativeRewardScorer,
    items: Sequence[tuple[str, str]],
    *,
    batch_size: int = 1,
) -> list[float]:
    """Score in input order, validating the entire request before any forward."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(
            "MLX reward: batch_size must be a positive integer, not bool. Set batch_size >= 1."
        )
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError(
            "MLX reward: items must be a sequence of (prompt, completion) pairs. "
            "Supply a list or tuple of pairs."
        )
    from soup_cli.utils.mlx_reward_data import encode_completion, padded_arrays

    encoded = []
    for index, item in enumerate(items):
        prefix = f"MLX reward item {index}"
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                f"{prefix}: expected a (prompt, completion) pair. Supply exactly two strings."
            )
        prompt, completion = item
        for field, value in zip(("prompt", "completion"), item, strict=True):
            if not isinstance(value, str):
                raise ValueError(
                    f"{prefix} {field}: expected a string. Convert this field to text."
                )
        if not completion.strip():
            raise ValueError(f"{prefix} completion: blank text. Supply a nonblank completion.")
        try:
            ids, _lost, count = encode_completion(
                scorer.tokenizer,
                prompt,
                completion,
                scorer.max_length,
            )
        except ValueError as exc:
            if str(exc) != "no non-EOS content tokens":
                raise
            raise ValueError(
                f"{prefix} completion: no non-EOS content tokens. Supply a meaningful completion."
            ) from exc
        if count > scorer.max_length:
            raise ValueError(
                f"{prefix}: {count} tokens including terminal EOS exceed saved "
                f"max_length={scorer.max_length}. Shorten the input or use a bundle "
                "trained with a larger data.max_length."
            )
        encoded.append(ids)
    if not encoded:
        return []
    scores = []
    for start in range(0, len(encoded), batch_size):
        arrays = padded_arrays(encoded[start : start + batch_size], scorer.tokenizer.eos_token_id)
        values = scorer.model(*arrays).tolist()
        for offset, value in enumerate(values):
            if not math.isfinite(value):
                raise ValueError(
                    f"MLX reward item {start + offset}: non-finite score. "
                    "Check the bundle and model numerics."
                )
            scores.append(float(value))
    return scores
