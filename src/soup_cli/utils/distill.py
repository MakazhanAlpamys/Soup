"""v0.52.0 Part C — Knowledge Distillation schema helpers.

Schema-only support for ``task='distill'`` — teacher/student training.
Four divergence options are recognised, mirroring axolotl's distillation
plugin:

* ``kl`` (forward KL — student KL teacher, standard distillation)
* ``forward_kl`` (alias for ``kl``)
* ``reverse_kl`` (teacher KL student)
* ``js`` (Jensen-Shannon, symmetric)

The live distillation trainer lands in v0.52.1; this module exposes pure
validators so the schema gate can fail fast on misconfiguration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_DIVERGENCE_ALIASES: Mapping[str, str] = MappingProxyType({
    "kl": "forward_kl",
    "forward_kl": "forward_kl",
    "reverse_kl": "reverse_kl",
    "js": "js",
})

# Public, derived from the alias map so adding a new alias updates both the
# accepted-input set and the error message in lockstep.
DIVERGENCES: frozenset[str] = frozenset(_DIVERGENCE_ALIASES)

# v0.71.12 #145 — cross-tokenizer distillation modes.
#   * token    — column-aligned logit KL (default; requires same tokenizer, or
#                training.uld_strategy for the cross-tokenizer logit-level path).
#   * sequence — sequence-level KD (Kim & Rush 2016): the teacher GENERATES a
#                completion per prompt and the student does plain CE on the
#                re-tokenised teacher output. Works across ANY tokenizer pair.
SUPPORTED_DISTILL_MODES: frozenset[str] = frozenset({"token", "sequence"})

_MAX_TEACHER_LEN: int = 512
_MAX_DIVERGENCE_LEN: int = 16
_MAX_DISTILL_MODE_LEN: int = 16
_MIN_TEMPERATURE: float = 0.05
_MAX_TEMPERATURE: float = 100.0
# Default teacher generation budget for sequence-level KD. Clamped to
# ``data.max_length`` by the trainer; bounded here as a DoS guard.
_DEFAULT_SEQ_MAX_NEW_TOKENS: int = 256
_MAX_SEQ_MAX_NEW_TOKENS: int = 4096


@dataclass(frozen=True)
class DivergenceSpec:
    """Metadata for a divergence kernel. Frozen so callers cannot mutate."""

    name: str
    description: str
    symmetric: bool
    live_wired: bool


_DIVERGENCE_METADATA: Mapping[str, DivergenceSpec] = MappingProxyType({
    "forward_kl": DivergenceSpec(
        name="forward_kl",
        description="Forward KL (standard distillation)",
        symmetric=False,
        live_wired=False,
    ),
    "reverse_kl": DivergenceSpec(
        name="reverse_kl",
        description="Reverse KL (mode-seeking)",
        symmetric=False,
        live_wired=False,
    ),
    "js": DivergenceSpec(
        name="js",
        description="Jensen-Shannon (symmetric KL)",
        symmetric=True,
        live_wired=False,
    ),
})


def validate_divergence(name: object) -> str:
    """Validate a divergence name and return the canonical form.

    Accepts ``kl`` as an alias for ``forward_kl``. Mirrors v0.41.0
    ``validate_optimizer_name`` policy.
    """
    if isinstance(name, bool):
        raise TypeError(f"distill_divergence must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(
            f"distill_divergence must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("distill_divergence must be non-empty")
    if "\x00" in name:
        raise ValueError("distill_divergence must not contain null bytes")
    if len(name) > _MAX_DIVERGENCE_LEN:
        raise ValueError(
            f"distill_divergence too long (max {_MAX_DIVERGENCE_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in _DIVERGENCE_ALIASES:
        supported = ", ".join(sorted(DIVERGENCES))
        raise ValueError(
            f"distill_divergence {name!r} not supported. Supported: {supported}"
        )
    return _DIVERGENCE_ALIASES[canonical]


def get_divergence_spec(name: str) -> DivergenceSpec:
    """Return the frozen :class:`DivergenceSpec` for ``name`` or raise."""
    canonical = validate_divergence(name)
    return _DIVERGENCE_METADATA[canonical]


def validate_distill_mode(value: object) -> str:
    """Validate ``training.distill_mode`` and return the canonical form.

    Accepts ``token`` (default) or ``sequence`` (case-insensitive). Mirrors
    the v0.41.0 / v0.52.0 validator policy (bool-first, null-byte, oversize,
    case-insensitive normalisation).
    """
    if isinstance(value, bool):
        raise TypeError(f"distill_mode must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"distill_mode must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("distill_mode must be non-empty")
    if "\x00" in value:
        raise ValueError("distill_mode must not contain null bytes")
    if len(value) > _MAX_DISTILL_MODE_LEN:
        raise ValueError(
            f"distill_mode too long (max {_MAX_DISTILL_MODE_LEN} chars)"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_DISTILL_MODES:
        supported = ", ".join(sorted(SUPPORTED_DISTILL_MODES))
        raise ValueError(
            f"distill_mode {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_distill_temperature(value: object) -> float:
    """Validate a distillation temperature scalar.

    Bounds [0.05, 100.0]. Rejects bool, NaN, ±inf.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"distill_temperature must not be bool, got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"distill_temperature must be float, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(
            f"distill_temperature must be finite, got {value!r}"
        )
    if fval < _MIN_TEMPERATURE:
        raise ValueError(
            f"distill_temperature must be >= {_MIN_TEMPERATURE}, got {fval}"
        )
    if fval > _MAX_TEMPERATURE:
        raise ValueError(
            f"distill_temperature must be <= {_MAX_TEMPERATURE}, got {fval}"
        )
    return fval


def validate_teacher_model(value: object) -> str:
    """Validate a teacher model string (HF repo id or local path).

    Mirrors the v0.40.5 ``reward_model`` field validator: null-byte
    rejection + 512-char cap.
    """
    if isinstance(value, bool):
        raise TypeError(f"teacher_model must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"teacher_model must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("teacher_model must be non-empty")
    if "\x00" in value:
        raise ValueError("teacher_model must not contain null bytes")
    if len(value) > _MAX_TEACHER_LEN:
        raise ValueError(
            f"teacher_model too long (max {_MAX_TEACHER_LEN} chars)"
        )
    return value


def validate_distill_compat(
    *,
    task: str,
    backend: str,
    teacher_model: object,
) -> None:
    """Schema-time gate for ``task='distill'``.

    Rejects:
    - non-distill task.
    - ``backend == 'mlx'`` (no MLX teacher-load path yet).
    - missing teacher_model — distillation is meaningless without one.
    """
    for name, value in (("task", task), ("backend", backend)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if task != "distill":
        raise ValueError(
            f"validate_distill_compat called with task={task!r} "
            "(expected 'distill')"
        )
    if backend == "mlx":
        raise ValueError(
            "task='distill' is not supported on backend=mlx in v0.52.0"
        )
    if teacher_model is None:
        raise ValueError(
            "task='distill' requires training.teacher_model to be set"
        )
    # Reuse the standard validator — null-byte / oversize / type check.
    validate_teacher_model(teacher_model)


def extract_prompt_messages(messages: object) -> list:
    """Return the prompt portion of a chat-messages list (v0.71.12 #145).

    The "prompt" is everything up to (and excluding) a trailing assistant
    turn — the part the teacher should be conditioned on for sequence-level
    KD. If the final message is not an assistant turn (prompt-only dataset)
    the whole list is returned.

    Raises:
        TypeError: ``messages`` is not a list.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"messages must be a list, got {type(messages).__name__}"
        )
    prompt: list = list(messages)
    # Strip a single trailing assistant turn (the teacher will regenerate it).
    while prompt and isinstance(prompt[-1], dict) and prompt[-1].get("role") == "assistant":
        prompt = prompt[:-1]
    return prompt


def _resolve_seq_max_new_tokens(value: object) -> int:
    """Clamp the sequence-KD generation budget to ``[1, _MAX_SEQ_MAX_NEW_TOKENS]``."""
    if value is None:
        return _DEFAULT_SEQ_MAX_NEW_TOKENS
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_new_tokens must be int")
    if value < 1:
        raise ValueError("max_new_tokens must be positive")
    return min(value, _MAX_SEQ_MAX_NEW_TOKENS)


def build_sequence_distill_rows(
    rows,
    teacher,
    teacher_tokenizer,
    *,
    max_new_tokens: object = None,
    device: str | None = None,
) -> list:
    """Sequence-level KD dataset builder (v0.71.12 #145 — Kim & Rush 2016).

    For each ``{"messages": [...]}`` row, the teacher GENERATES a completion
    from the prompt portion (its own tokenizer + chat template), and a new
    student row is emitted with the teacher's text as the assistant turn::

        {"messages": prompt + [{"role": "assistant", "content": <teacher text>}]}

    The student then trains with plain CE on the re-tokenised teacher output —
    which works across ANY tokenizer pair. The teacher is used for generation
    only; it is NOT needed in the student loss loop.

    Args:
        rows: iterable of dataset rows (each a dict with a ``messages`` list).
        teacher: a loaded causal-LM with ``.generate`` and ``.parameters()``.
        teacher_tokenizer: the teacher's tokenizer (its own chat template).
        max_new_tokens: generation budget (clamped to a DoS-safe ceiling).
        device: optional device override; defaults to the teacher's device.

    Returns:
        A list of student ``{"messages": [...]}`` rows.
    """
    import torch

    budget = _resolve_seq_max_new_tokens(max_new_tokens)
    try:
        teacher_device = next(teacher.parameters()).device
    except (StopIteration, AttributeError):
        teacher_device = device or "cpu"
    pad_id = getattr(teacher_tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(teacher_tokenizer, "eos_token_id", None)

    out_rows: list = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        prompt_msgs = extract_prompt_messages(messages)
        if not prompt_msgs:
            continue
        input_ids = teacher_tokenizer.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(teacher_device)
        prompt_len = int(input_ids.shape[-1])
        with torch.no_grad():
            generated = teacher.generate(
                input_ids=input_ids,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = generated[0][prompt_len:]
        teacher_text = teacher_tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        out_rows.append(
            {
                "messages": list(prompt_msgs)
                + [{"role": "assistant", "content": teacher_text}]
            }
        )
    return out_rows


def build_distill_trainer(
    config: object, **kwargs: object
) -> object:
    """Live distillation trainer factory (v0.53.2 #133).

    Returns a :class:`DistillTrainerWrapper`. Lazy import keeps the heavy
    transformers/peft surface out of schema-only import paths.
    """
    from soup_cli.trainer.distill import DistillTrainerWrapper

    return DistillTrainerWrapper(config, **kwargs)  # type: ignore[arg-type]
