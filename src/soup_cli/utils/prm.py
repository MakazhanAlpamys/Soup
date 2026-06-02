"""PRM (Process Reward Model) — v0.50.0 Part E + v0.53.3 #129.

Schema helpers for the new ``task='prm'`` stepwise-supervised trainer.
The PRM data format (``data.format='prm'``) was schema-locked in v0.42.0
Part A; v0.50.0 promotes it to a first-class task with cross-validators.

v0.53.3 #129 extends :func:`validate_vision_grpo_compat` with an optional
``base`` model name probe (``KNOWN_VLM_REGEX``) so a config that pairs
``vision_grpo: true`` with a non-VLM checkpoint is rejected at schema-load
with an actionable message naming a known VLM family.

The actual PRM trainer wrapper (``soup_cli/trainer/prm.py``) is deferred
to v0.50.1 — mirrors v0.27.0 MII / v0.37.0 multipack / v0.41.0 LLaMA Pro /
v0.45.0 plugins / v0.49.0 LongLoRA stub-then-live pattern.

Security:
- Pure schema-time validation; no filesystem touch.
- All validators raise ``ValueError`` with actionable messages.
- Name-regex probe rejects null-byte / non-string / oversize inputs by
  returning ``False`` (no exception — mirrors v0.39.0 ``is_gemma4_model``
  / v0.44.0 ``is_llama4_model`` / v0.49.0 ``is_llama_model`` policy).
"""

from __future__ import annotations

import re

# v0.53.3 #129 — case-insensitive name allowlist for VLM bases.
# Each alternative uses word-style boundaries so substring noise like
# ``"my-pixtralish"`` does not match. The list is deliberately small and
# additive — extending it does not break callers because callers always
# pass through :func:`is_known_vlm_base`.
_VLM_PATTERNS = (
    r"(?:^|[^a-z0-9])qwen[\d.]*-vl(?:[^a-z0-9]|$)",   # Qwen2-VL / Qwen2.5-VL
    r"(?:^|[^a-z0-9])qvq(?:[^a-z0-9]|$)",              # QVQ-72B
    r"(?:^|[^a-z0-9])pixtral(?:[^a-z0-9]|$)",          # Pixtral
    r"(?:^|[^a-z0-9])internvl[\d._]*(?:[^a-z0-9]|$)",  # InternVL/InternVL2_5/InternVL3
    # Llama-3.2-Vision (any size in between, e.g. Llama-3.2-11B-Vision)
    r"(?:^|[^a-z0-9])llama-?3\.?2[a-z0-9._-]*vision(?:[^a-z0-9]|$)",
    r"(?:^|[^a-z0-9])llava(?:[^a-z0-9]|$)",            # LLaVA
    r"(?:^|[^a-z0-9])minicpm-?v(?:[^a-z0-9]|$)",       # MiniCPM-V
    r"(?:^|[^a-z0-9])idefics[\d]*(?:[^a-z0-9]|$)",     # Idefics
    r"(?:^|[^a-z0-9])sharegpt4v(?:[^a-z0-9]|$)",       # ShareGPT4V
    r"(?:^|[^a-z0-9])fuyu(?:[^a-z0-9]|$)",             # Fuyu
)
KNOWN_VLM_REGEX = re.compile("|".join(_VLM_PATTERNS), re.IGNORECASE)

_MAX_BASE_NAME_LEN = 512


def is_known_vlm_base(name: object) -> bool:
    """Best-effort check whether ``name`` matches a known VLM family.

    Returns ``False`` (never raises) on any of: non-string, empty, null
    byte, length > 512. Match is case-insensitive with word boundaries so
    substring noise (``"my-pixtralish"``) does not false-positive — mirrors
    v0.39.0 / v0.44.0 / v0.49.0 model-detection policy.
    """
    if isinstance(name, bool):
        return False
    if not isinstance(name, str):
        return False
    if not name:
        return False
    if "\x00" in name:
        return False
    if len(name) > _MAX_BASE_NAME_LEN:
        return False
    return KNOWN_VLM_REGEX.search(name) is not None


def validate_prm_compat(
    *,
    task: str,
    data_format: str,
    backend: str,
    modality: str,
) -> None:
    """Schema-time gate for ``task='prm'``.

    Rejects:
    - non-PRM task (the function is intended to be called only when
      ``task == 'prm'``; defence-in-depth).
    - ``data.format`` not in ``{'prm', 'auto'}`` — PRM requires the
      stepwise-supervised data shape from v0.42.0 Part A.
    - ``backend='mlx'`` — PRM trainer is HF Trainer-specific.
    - ``modality != 'text'`` — vision/audio PRM not modelled.
    """
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if task != "prm":
        raise ValueError(
            f"validate_prm_compat called with task={task!r} (expected 'prm')"
        )
    if not isinstance(data_format, str) or not data_format:
        raise ValueError("data.format must be a non-empty string")
    if data_format not in ("prm", "auto"):
        raise ValueError(
            f"task='prm' requires data.format in ('prm', 'auto'); "
            f"got data.format={data_format!r}"
        )
    if backend == "mlx":
        raise ValueError(
            "task='prm' is not supported on backend=mlx in v0.50.0"
        )
    if modality != "text":
        raise ValueError(
            f"task='prm' requires modality='text'; got modality={modality!r}"
        )


def validate_vision_grpo_compat(
    *,
    task: str,
    modality: str,
    backend: str,
    base: str | None = None,
) -> None:
    """Schema-time gate for ``vision_grpo=True``.

    Rejects on:
    - task not in {'grpo', 'ppo'} (vision RL is only meaningful for RL);
    - modality != 'vision' (the whole point of the flag);
    - backend == 'mlx' (no VLM-RL on MLX);
    - v0.53.3 #129: ``base`` (when supplied, non-empty) does not match a
      known VLM family — the runtime trainer error would be cryptic
      ("module has no attribute 'vision_tower'") so we surface a friendly
      schema-load rejection naming the expected families instead.

    ``base=None`` or empty-string skips the probe (backwards-compatible —
    legacy callers from v0.50.0 Part E pass no ``base`` kwarg).
    """
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if task not in ("grpo", "ppo"):
        raise ValueError(
            f"vision_grpo requires task in ('grpo', 'ppo'); got task={task!r}"
        )
    if modality != "vision":
        raise ValueError(
            f"vision_grpo requires modality='vision'; got modality={modality!r}"
        )
    if backend == "mlx":
        raise ValueError(
            "vision_grpo is not supported on backend=mlx in v0.50.0"
        )
    # v0.53.3 #129 — name-regex probe (deliberately permissive: empty /
    # None / non-string skips the probe).
    if isinstance(base, str) and base and not is_known_vlm_base(base):
        # Truncate the echoed value to keep adversarial / int bases from
        # bloating error logs (security review fix; mirrors v0.34.0 crash
        # redaction policy).
        safe_base = base if len(base) <= 64 else base[:61] + "..."
        raise ValueError(
            f"vision_grpo=True requires a known VLM base; got base={safe_base!r}. "
            "Expected one of the Qwen2-VL / Pixtral / InternVL / "
            "Llama-3.2-Vision / LLaVA / MiniCPM-V families. If your base "
            "is a legitimate VLM not in the allowlist, omit vision_grpo "
            "until a future release adds a runtime config-probe path."
        )


def build_prm_trainer(*, config, **kwargs):
    """Live PRM trainer factory (v0.53.11 #126).

    Returns a configured :class:`PRMTrainerWrapper`. Rejects unknown kwargs
    with ``TypeError`` so the factory contract is explicit (matches
    v0.53.2 build_distill_trainer policy).
    """
    from soup_cli.trainer.prm import PRMTrainerWrapper

    allowed = {
        "device",
        "report_to",
        "deepspeed_config",
        "fsdp_config",
        "trust_remote_code",
    }
    unknown = set(kwargs) - allowed
    if unknown:
        raise TypeError(
            f"build_prm_trainer got unexpected kwargs: {sorted(unknown)}"
        )
    return PRMTrainerWrapper(config, **kwargs)


def compute_prm_loss(predictions, labels, *, mask=None):
    """Per-step PRM MSE loss kernel (v0.53.11 #126).

    Used by ``PRMTrainerWrapper`` to score per-step rewards against scalar
    labels. ``predictions`` and ``labels`` are aligned 1-D / 2-D tensors;
    ``mask`` (optional) zeros out padding positions.

    Returns a scalar torch tensor — mean of squared residuals over
    non-masked positions.
    """
    if not hasattr(predictions, "shape") or not hasattr(labels, "shape"):
        raise TypeError("predictions and labels must be tensors")
    if predictions.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: predictions={tuple(predictions.shape)} vs "
            f"labels={tuple(labels.shape)}"
        )
    diff = (predictions - labels) ** 2
    if mask is None:
        return diff.mean()
    if mask.shape != predictions.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != predictions shape "
            f"{tuple(predictions.shape)}"
        )
    masked = diff * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom
