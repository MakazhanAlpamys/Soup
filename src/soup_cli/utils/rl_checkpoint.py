"""Mid-epoch checkpoint for PPO / GRPO — v0.70.0 Part D.

Optimizer-state serialization for int RL runs that need to survive
preemption mid-rollout. TorchTune explicitly punts this (their README
notes "we expect users to resume from the start of the most recent
epoch"); Soup ships a real mid-epoch save/load surface here.

Composes with:
- v0.32 spike-recovery (the recovery policy can hop back to the most
  recent rl checkpoint instead of the start of the epoch).
- v0.40.0 ref-model regen (ref-model state is captured in the
  checkpoint manifest so a resume restarts with the same reference
  distribution).

Schema-only release; live save_state / load_state HF Trainer callback
deferred to v0.70.1. Mirrors v0.50.0 / v0.62.0 / v0.69.0 stub-then-live
cadence.

Security:
- ``RLCheckpointConfig`` is frozen + per-field-validated.
- Bool / non-int rejection on every numeric (bool-before-int policy).
- ``RLCheckpointState.task`` allowlisted to RL tasks only.
- ``checkpoint_dir`` shape-validated (null-byte / oversize) — cwd
  containment + symlink rejection happen at v0.70.1 disk-write time
  (matches v0.69.0 build_dag deferred-write-containment policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MAX_SAVE_EVERY_STEPS = 10_000_000
_MIN_KEEP_LAST = 1
_MAX_KEEP_LAST = 100
_MAX_DIR_LEN = 4096
_MAX_SOUP_VERSION_LEN = 32

# RL tasks whose mid-epoch checkpoint makes sense. Other tasks already
# have HF Trainer's per-epoch checkpoint and don't need the rollout
# / ref-model state captured.
_RL_TASKS: frozenset[str] = frozenset({"grpo", "ppo"})


def validate_save_every_steps(value: object) -> int:
    """Bool-rejecting + bounded validator for save_every_steps.

    Range: ``[1, _MAX_SAVE_EVERY_STEPS=10_000_000]``.
    """
    if isinstance(value, bool):
        raise ValueError("save_every_steps must not be bool")
    if not isinstance(value, int):
        raise ValueError(
            f"save_every_steps must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"save_every_steps must be >= 1, got {value}")
    if value > _MAX_SAVE_EVERY_STEPS:
        raise ValueError(
            f"save_every_steps={value} exceeds {_MAX_SAVE_EVERY_STEPS} cap"
        )
    return value


def _validate_keep_last(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("keep_last must not be bool")
    if not isinstance(value, int):
        raise ValueError(
            f"keep_last must be int, got {type(value).__name__}"
        )
    if value < _MIN_KEEP_LAST or value > _MAX_KEEP_LAST:
        raise ValueError(
            f"keep_last must be in [{_MIN_KEEP_LAST}, {_MAX_KEEP_LAST}], "
            f"got {value}"
        )
    return value


def _validate_bool_flag(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool, got {type(value).__name__}")
    return value


def _validate_dir_shape(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be str, got bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_DIR_LEN:
        raise ValueError(f"{field} exceeds {_MAX_DIR_LEN} chars")
    return value


@dataclass(frozen=True)
class RLCheckpointConfig:
    """Frozen RL-checkpoint config.

    - ``save_every_steps``: int >= 1; ``[1, 10_000_000]`` cap.
    - ``include_optimizer_state``: include AdamW / Lion state in the
      checkpoint. Default True (the whole point of mid-epoch RL ckpt).
    - ``include_ref_model``: include the frozen reference model state.
      Default False (ref model can usually be reconstructed from
      ``cfg.base``).
    - ``include_rollout_buffer``: include the replay / rollout buffer
      so resumed runs don't lose collected experience.
    - ``keep_last``: number of recent checkpoints to retain. Older
      checkpoints get pruned at write time.
    """

    save_every_steps: int
    include_optimizer_state: bool = True
    include_ref_model: bool = False
    include_rollout_buffer: bool = False
    keep_last: int = 3

    def __post_init__(self) -> None:
        validate_save_every_steps(self.save_every_steps)
        _validate_bool_flag(self.include_optimizer_state, "include_optimizer_state")
        _validate_bool_flag(self.include_ref_model, "include_ref_model")
        _validate_bool_flag(self.include_rollout_buffer, "include_rollout_buffer")
        _validate_keep_last(self.keep_last)


@dataclass(frozen=True)
class RLCheckpointState:
    """Frozen state manifest persisted at each mid-epoch save.

    Written to ``<checkpoint_dir>/manifest.json``. Allows a resume
    handler in v0.70.1 to verify the checkpoint shape before loading
    the (potentially large) optimizer state.

    - ``step``: training step at which the checkpoint was taken.
      Non-negative int (bool rejected per project policy).
    - ``checkpoint_dir``: directory holding the saved tensors. Shape
      validated only (cwd-containment deferred to v0.70.1 disk hook).
    - ``task``: must be in the RL allowlist (``grpo`` / ``ppo``).
    - ``has_optimizer``/``has_ref_model``/``has_rollout_buffer``: real
      bools (no str/int coercion).
    - ``soup_version``: capped at 32 chars; null-byte rejected.
    """

    step: int
    checkpoint_dir: str
    task: str
    has_optimizer: bool
    has_ref_model: bool
    has_rollout_buffer: bool
    soup_version: str

    def __post_init__(self) -> None:
        if isinstance(self.step, bool):
            raise ValueError("step must not be bool")
        if not isinstance(self.step, int):
            raise TypeError(f"step must be int, got {type(self.step).__name__}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        _validate_dir_shape(self.checkpoint_dir, "checkpoint_dir")
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task must be a non-empty string")
        if self.task not in _RL_TASKS:
            raise ValueError(
                f"task={self.task!r} must be one of {sorted(_RL_TASKS)}"
            )
        _validate_bool_flag(self.has_optimizer, "has_optimizer")
        _validate_bool_flag(self.has_ref_model, "has_ref_model")
        _validate_bool_flag(self.has_rollout_buffer, "has_rollout_buffer")
        if not isinstance(self.soup_version, str) or not self.soup_version:
            raise ValueError("soup_version must be a non-empty string")
        if "\x00" in self.soup_version:
            raise ValueError("soup_version must not contain null bytes")
        if len(self.soup_version) > _MAX_SOUP_VERSION_LEN:
            raise ValueError(
                f"soup_version exceeds {_MAX_SOUP_VERSION_LEN} chars"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict for the manifest file."""
        return {
            "step": self.step,
            "checkpoint_dir": self.checkpoint_dir,
            "task": self.task,
            "has_optimizer": self.has_optimizer,
            "has_ref_model": self.has_ref_model,
            "has_rollout_buffer": self.has_rollout_buffer,
            "soup_version": self.soup_version,
        }


def build_rl_checkpoint_callback(config):
    """Live HF Trainer callback for mid-epoch RL checkpoints.

    Deferred to v0.70.1. Validates the config type at the public
    boundary so misconfigured callers fail fast (mirrors v0.50.0 /
    v0.62.0 / v0.67.0 / v0.69.0 deferred-live policy).
    """
    if not isinstance(config, RLCheckpointConfig):
        raise TypeError(
            f"config must be RLCheckpointConfig, got {type(config).__name__}"
        )
    raise NotImplementedError(
        "Live mid-epoch RL checkpoint callback is deferred to v0.70.1. "
        "v0.70.0 ships the schema + state manifest only."
    )
