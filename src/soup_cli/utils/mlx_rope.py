"""Native MLX RoPE scaling configuration and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_checkpoint_config(
    model_or_config: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Read model configuration from a dictionary, local directory, or HF Hub model."""
    if isinstance(model_or_config, dict):
        return dict(model_or_config), "in-memory configuration"

    model_path = Path(model_or_config)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise ValueError(
                f"Base checkpoint directory {str(model_path)!r} is missing config.json"
            )
        try:
            return json.loads(config_path.read_text(encoding="utf-8")), str(model_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to read config.json in {str(model_path)!r}: {exc}") from exc

    # Treat as HF Hub repo id
    repo_id = str(model_or_config)
    try:
        from huggingface_hub import hf_hub_download

        # Try offline cache first, then fall back to normal download
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename="config.json", local_files_only=True
            )
        except Exception:
            downloaded = hf_hub_download(repo_id=repo_id, filename="config.json")
        return json.loads(Path(downloaded).read_text(encoding="utf-8")), repo_id
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve base checkpoint configuration for {repo_id!r}: {exc}"
        ) from exc


def resolve_mlx_rope_scaling(
    model_or_config: str | Path | dict[str, Any],
    target_length: int,
    rope_scaling_type: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve and validate native RoPE scaling configuration for MLX text SFT.

    Returns:
        (adapter_rope_metadata, model_config_overrides)
        If rope_scaling_type is None or target_length <= original length, returns (None, None).
    """
    if rope_scaling_type is None:
        return None, None

    if rope_scaling_type != "linear":
        raise ValueError(
            f"training.rope_scaling_type={rope_scaling_type!r} is not applied on backend='mlx': "
            "MLX text SFT currently supports rope_scaling_type='linear' only. "
            "Use rope_scaling_type: linear and training.quantization: none, "
            "or backend: transformers, or remove the key."
        )

    config, model_desc = _read_checkpoint_config(model_or_config)

    orig_len = config.get("max_position_embeddings")
    if (
        orig_len is None
        or isinstance(orig_len, bool)
        or not isinstance(orig_len, int)
        or orig_len <= 0
    ):
        raise ValueError(
            f"Base checkpoint {model_desc!r} original context length ('max_position_embeddings') "
            f"is missing or invalid (got {orig_len!r}); "
            "positive integer required for MLX RoPE scaling."
        )

    # Check for pre-scaled or per-layer RoPE configurations
    existing = config.get("rope_scaling")
    if existing is None:
        existing = config.get("rope_parameters")

    if isinstance(existing, dict):
        if any(isinstance(v, dict) for v in existing.values()):
            raise ValueError(
                f"Base checkpoint {model_desc!r} has per-layer or nested RoPE configuration; "
                "not supported on MLX."
            )
        rope_type = existing.get("rope_type") or existing.get("type")
        if rope_type not in (None, "default"):
            raise ValueError(
                f"Base checkpoint {model_desc!r} already carries RoPE scaling ({existing!r}); "
                "MLX linear RoPE context extension applies only to checkpoints with "
                "unscaled native RoPE."
            )

    # If target length is at or below the base model's context window, leave RoPE untouched
    if target_length <= orig_len:
        return None, None

    factor = float(target_length / orig_len)
    rope_scaling = {
        "type": "linear",
        "factor": factor,
        "original_max_position_embeddings": orig_len,
    }
    model_config = {
        "rope_scaling": {"type": "linear", "factor": factor},
        "max_position_embeddings": target_length,
    }
    return rope_scaling, model_config
