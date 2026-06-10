"""Multi-turn agent rollout backends — v0.50.0 Part C / v0.71.21 #125 live.

Closed allowlist for multi-turn agent RL rollout backends shipped for
unsloth + axolotl parity:

- ``art``      — OpenPipe ART (multi-turn agent rollouts; unsloth)
- ``ruler``    — RULER agent eval framework (unsloth)
- ``nemo_gym`` — NVIDIA NeMo Gym (unsloth, axolotl)
- ``openenv``  — Generic ``rollout_func`` protocol (unsloth, axolotl)

v0.71.21 (#125) lifts :func:`launch_rollout` to live:

- ``openenv`` runs fully live on CPU — ``training.rollout_func`` names a
  user-supplied ``module.path:function_name`` callable (mirrors the
  v0.42.0 ``data.prompt_strategy`` design) that receives the seed prompts
  and returns rollout rows (``{"prompt": ..., "answer"?: ...}``). The
  rows replace the GRPO prompt dataset.
- ``art`` / ``ruler`` / ``nemo_gym`` are lazy-import gated: a friendly
  ImportError names the pip package when it is missing; when present, an
  honest "adapter not yet validated" RuntimeError fires instead of
  shipping unrun integration code (mirrors the v0.71.20 TTS-codec /
  onebitllms BETA-gate policy). ``_EXTERNAL_ROLLOUT_RUNNERS`` is the
  injectable runner seam (mirrors ``cloud/modal._MODAL_SUBMIT_OVERRIDE``).

Security:
- Closed allowlist; arbitrary string at schema level rejected.
- ``_BACKEND_METADATA`` wrapped in MappingProxyType (matches v0.36.0
  _REGISTRY policy).
- ``validate_rollout_backend`` rejects empty / null-byte / non-string /
  oversize inputs; ``rollout_func`` is regex-validated (``module:fn``
  shape, no null bytes / oversize) before any import is attempted.
- Rollout rows are capped at ``_MAX_ROLLOUT_ROWS`` and normalised to
  ``{prompt, answer?}``-only dicts so a misbehaving rollout callable
  cannot smuggle arbitrary payloads into the training dataset.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

_MAX_BACKEND_NAME_LEN = 32
_MAX_ROLLOUT_FUNC_LEN = 260
_MAX_ROLLOUT_ROWS = 100_000
_MAX_ROLLOUT_STEPS = 100_000

# ``module.path:function_name`` — mirrors v0.42.0 data.prompt_strategy.
_ROLLOUT_FUNC_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]{0,127}:[A-Za-z_][A-Za-z0-9_]{0,127}$"
)

# Import names per external backend (pip name differs — see
# ``required_package`` on the spec).
_EXTERNAL_IMPORT_NAMES = types.MappingProxyType({
    "art": "art",
    "ruler": "ruler",
    "nemo_gym": "nemo_gym",
})

# Injectable runner seam for the external backends — tests and advanced
# operators may install ``name -> callable(**kwargs) -> rows`` here.
# NOT a public API (mirrors cloud/modal._MODAL_SUBMIT_OVERRIDE).
_EXTERNAL_ROLLOUT_RUNNERS: dict[str, Callable[..., Any]] = {}

SUPPORTED_ROLLOUT_BACKENDS: frozenset[str] = frozenset({
    "art",
    "ruler",
    "nemo_gym",
    "openenv",
})


@dataclass(frozen=True)
class RolloutBackendSpec:
    """Metadata for a multi-turn agent rollout backend."""

    name: str
    description: str
    required_package: str | None
    live_wired: bool


_BACKEND_METADATA = types.MappingProxyType({
    "art": RolloutBackendSpec(
        name="art",
        description="OpenPipe ART — multi-turn agent rollouts",
        required_package="openpipe-art",
        live_wired=False,
    ),
    "ruler": RolloutBackendSpec(
        name="ruler",
        description="RULER agent evaluation framework",
        required_package="ruler-eval",
        live_wired=False,
    ),
    "nemo_gym": RolloutBackendSpec(
        name="nemo_gym",
        description="NVIDIA NeMo Gym single/multi-turn rollout backend",
        required_package="nemo-gym",
        live_wired=False,
    ),
    "openenv": RolloutBackendSpec(
        name="openenv",
        description="Generic OpenEnv rollout_func protocol",
        required_package=None,
        live_wired=True,  # v0.71.21 #125 — fully live on CPU.
    ),
})


def validate_rollout_backend(name: object) -> str:
    """Validate and normalise a rollout backend name."""
    if isinstance(name, bool):
        raise ValueError("rollout_backend must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"rollout_backend must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("rollout_backend must be a non-empty string")
    if "\x00" in name:
        raise ValueError("rollout_backend must not contain null bytes")
    if len(name) > _MAX_BACKEND_NAME_LEN:
        raise ValueError(
            f"rollout_backend exceeds {_MAX_BACKEND_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_ROLLOUT_BACKENDS:
        raise ValueError(
            f"rollout_backend={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_ROLLOUT_BACKENDS)}"
        )
    return normalised


def get_rollout_backend_spec(name: str) -> RolloutBackendSpec:
    """Return the :class:`RolloutBackendSpec` for ``name``."""
    normalised = validate_rollout_backend(name)
    return _BACKEND_METADATA[normalised]


def required_rollout_package(name: str) -> str | None:
    """Return the pip-installable package name (or None for openenv)."""
    spec = get_rollout_backend_spec(name)
    return spec.required_package


def list_rollout_backends() -> tuple[str, ...]:
    """Return a sorted tuple of supported rollout backend names."""
    return tuple(sorted(SUPPORTED_ROLLOUT_BACKENDS))


def validate_rollout_func(value: object) -> Optional[str]:
    """Validate a ``module.path:function_name`` rollout-func spec.

    ``None`` passes through (field unset). Mirrors the v0.42.0
    ``validate_prompt_strategy`` policy with rollout-func-named errors.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"rollout_func must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("rollout_func must not be empty")
    if "\x00" in value:
        raise ValueError("rollout_func must not contain null bytes")
    if len(value) > _MAX_ROLLOUT_FUNC_LEN:
        raise ValueError(
            f"rollout_func must be <= {_MAX_ROLLOUT_FUNC_LEN} chars"
        )
    if not _ROLLOUT_FUNC_RE.match(value):
        raise ValueError(
            "rollout_func must match 'module.path:function_name' "
            "(letters / digits / underscore / dot only)."
        )
    return value


def resolve_rollout_func(spec: str) -> Callable[..., Any]:
    """Resolve a validated ``module:fn`` spec into the rollout callable.

    Mirrors the v0.53.7 ``resolve_prompt_strategy`` runtime resolver.
    Trusted-input policy applies: the spec names operator-controlled code
    (same trust level as ``data.prompt_strategy`` / custom reward files).
    """
    validated = validate_rollout_func(spec)
    if validated is None:
        raise ValueError("rollout_func must not be None")
    module_path, _, fn_name = validated.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"rollout_func module {module_path!r} could not be imported: {exc}"
        ) from exc
    if not hasattr(module, fn_name):
        raise ValueError(
            f"rollout_func {validated!r}: module {module_path!r} has no "
            f"attribute {fn_name!r}"
        )
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(
            f"rollout_func {validated!r}: resolved attribute is not callable "
            f"({type(fn).__name__})"
        )
    return fn


def _spec_exists(name: str) -> bool:
    """``importlib.util.find_spec`` wrapper that never raises.

    Mirrors the v0.71.14 ``kv_cache._spec_exists`` policy —
    ``ModuleNotFoundError`` / ``ValueError`` from a half-installed parent
    package count as "not available".
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _normalise_rollout_rows(raw: object, backend: str) -> tuple[dict, ...]:
    """Validate + normalise rollout output to ``{prompt, answer?}`` dicts.

    Message-list prompts are deep-copied (the rollout callable must not be
    able to mutate training rows through a retained alias — immutability
    policy); a non-string ``answer`` is rejected LOUDLY, matching the
    prompt-side rejection (a silently dropped answer starves the accuracy
    reward of its reference).
    """
    if raw is None or isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError(
            f"rollout backend {backend!r} must return an iterable of row "
            f"mappings, got {type(raw).__name__}"
        )
    rows: list[dict] = []
    for index, row in enumerate(raw):
        if len(rows) >= _MAX_ROLLOUT_ROWS:
            raise ValueError(
                f"rollout backend {backend!r} produced more than "
                f"{_MAX_ROLLOUT_ROWS} rows"
            )
        if not isinstance(row, Mapping):
            raise ValueError(
                f"rollout row {index} must be a mapping, "
                f"got {type(row).__name__}"
            )
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            if not prompt:
                raise ValueError(f"rollout row {index} has an empty prompt")
        elif isinstance(prompt, list):
            if not prompt:
                raise ValueError(f"rollout row {index} has an empty prompt")
            # Break the alias with the rollout callable's return value.
            prompt = [
                dict(message) if isinstance(message, Mapping) else message
                for message in prompt
            ]
        else:
            raise ValueError(
                f"rollout row {index} must carry a 'prompt' (str or message "
                f"list), got {type(prompt).__name__}"
            )
        normalised: dict = {"prompt": prompt}
        answer = row.get("answer")
        if answer is not None:
            if not isinstance(answer, str):
                raise ValueError(
                    f"rollout row {index} 'answer' must be a string, "
                    f"got {type(answer).__name__}"
                )
            if answer:
                normalised["answer"] = answer
        rows.append(normalised)
    if not rows:
        raise ValueError(
            f"rollout backend {backend!r} produced no rows — refusing to "
            "train on an empty prompt set"
        )
    return tuple(rows)


@dataclass(frozen=True)
class RolloutResult:
    """Normalised rollout output: ``rows`` feed the GRPO prompt dataset."""

    backend: str
    rows: tuple[dict, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend", validate_rollout_backend(self.backend)
        )
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        # Re-validate through the shared normaliser so a directly
        # constructed RolloutResult cannot smuggle malformed rows.
        object.__setattr__(
            self, "rows", _normalise_rollout_rows(self.rows, self.backend)
        )


def launch_rollout(
    name: str,
    *,
    prompts: Optional[Sequence[Any]] = None,
    rollout_func: Optional[str] = None,
    model: Any = None,
    tokenizer: Any = None,
    reward_fn: Any = None,
    max_steps: int = 64,
) -> RolloutResult:
    """Run the named rollout backend and return its normalised result.

    Live since v0.71.21 (#125):

    - ``openenv`` resolves ``rollout_func`` (``module:fn``) and calls it
      with the seed ``prompts`` list. The callable returns an iterable of
      ``{"prompt": str | messages, "answer"?: str}`` rows.
    - ``art`` / ``ruler`` / ``nemo_gym`` raise a friendly ImportError when
      the backend package is missing, and an honest "not yet validated"
      RuntimeError when it is present (lazy-import + advisory — the live
      integration adapters are dep-gated; inject a runner via the
      ``_EXTERNAL_ROLLOUT_RUNNERS`` seam to use them today).

    ``model`` / ``tokenizer`` / ``reward_fn`` are forwarded to injected
    external runners (forward-compat per the v0.50.0 planned signature);
    the openenv contract intentionally stays minimal (prompts only).
    """
    normalised_name = validate_rollout_backend(name)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError(
            f"max_steps must be an int, got {type(max_steps).__name__}"
        )
    if not 1 <= max_steps <= _MAX_ROLLOUT_STEPS:
        raise ValueError(
            f"max_steps must be in [1, {_MAX_ROLLOUT_STEPS}], got {max_steps}"
        )
    if prompts is not None and (
        isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence)
    ):
        raise TypeError(
            f"prompts must be a sequence, got {type(prompts).__name__}"
        )
    seed_prompts = list(prompts or [])

    if normalised_name == "openenv":
        if rollout_func is None:
            raise ValueError(
                "rollout_backend='openenv' requires training.rollout_func "
                "('module.path:function_name')"
            )
        fn = resolve_rollout_func(rollout_func)
        raw = fn(seed_prompts)
        rows = _normalise_rollout_rows(raw, normalised_name)
        return RolloutResult(backend=normalised_name, rows=rows)

    # External backends: injected runner seam first, then lazy-import gate.
    runner = _EXTERNAL_ROLLOUT_RUNNERS.get(normalised_name)
    if runner is not None:
        raw = runner(
            prompts=seed_prompts,
            model=model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            max_steps=max_steps,
        )
        rows = _normalise_rollout_rows(raw, normalised_name)
        return RolloutResult(backend=normalised_name, rows=rows)

    import_name = _EXTERNAL_IMPORT_NAMES[normalised_name]
    spec = _BACKEND_METADATA[normalised_name]
    if not _spec_exists(import_name):
        raise ImportError(
            f"rollout_backend={normalised_name!r} requires the "
            f"{spec.required_package!r} package: "
            f"pip install {spec.required_package}"
        )
    raise RuntimeError(
        f"rollout_backend={normalised_name!r}: the {spec.required_package!r} "
        "integration adapter is not yet validated on any environment "
        "(BETA gate). Use rollout_backend='openenv' with "
        "training.rollout_func, or inject a runner via "
        "soup_cli.utils.agent_rollout._EXTERNAL_ROLLOUT_RUNNERS."
    )
