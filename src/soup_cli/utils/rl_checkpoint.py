"""Mid-epoch checkpoint for PPO / GRPO — v0.70.0 Part D.

Optimizer-state serialization for long RL runs that need to survive
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

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


def _get_trainer_callback_base():
    """Lazy-resolve ``transformers.TrainerCallback`` (mirror v0.53.11)."""
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:
        return object


_TrainerCallbackBase = _get_trainer_callback_base()


def _step_number(name: str) -> int:
    """Extract the integer step from a ``step-<N>`` checkpoint dir name."""
    try:
        return int(name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


class RLCheckpointCallback(_TrainerCallbackBase):  # type: ignore[misc, valid-type]
    """Live HF TrainerCallback for mid-epoch RL checkpoints (v0.71.11 #238).

    Saves an RL-aware checkpoint every ``save_every_steps`` steps under
    ``<output_dir>/rl-checkpoints/step-<N>/``:

    - the policy adapter (``model.save_pretrained``),
    - the optimizer state (when ``include_optimizer_state``), and
    - a ``manifest.json`` (:class:`RLCheckpointState`).

    Older checkpoints beyond ``keep_last`` are pruned at write time. The
    optimizer is read from the ``optimizer`` kwarg HF Trainer passes to
    callbacks, so this works for any HF-Trainer-based RL loop (GRPO/PPO).
    """

    def __init__(
        self,
        config: RLCheckpointConfig,
        *,
        output_dir: str,
        task: str = "grpo",
        soup_version: Optional[str] = None,
    ) -> None:
        if not isinstance(config, RLCheckpointConfig):
            raise TypeError(
                f"config must be RLCheckpointConfig, got {type(config).__name__}"
            )
        self.config = config
        self.output_dir = _validate_dir_shape(output_dir, "output_dir")
        # Containment: the run dir (and everything we write under it) must
        # stay under cwd (security review LOW). is_under_cwd is realpath-based
        # so it works before the dir exists; the per-file atomic_write_text on
        # the manifest adds the symlink-rejection at write time.
        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(self.output_dir):
            raise ValueError("output_dir must stay under the current directory")
        if task not in _RL_TASKS:
            raise ValueError(f"task={task!r} must be one of {sorted(_RL_TASKS)}")
        self.task = task
        if soup_version is None:
            from soup_cli import __version__ as _v

            soup_version = _v
        self.soup_version = soup_version
        self._saved: list[int] = []

    def _ckpt_root(self) -> str:
        import os

        return os.path.join(self.output_dir, "rl-checkpoints")

    def save_checkpoint(self, *, step: int, model, optimizer) -> str:
        """Write a checkpoint for ``step``; returns the directory path.

        Pure of HF Trainer state — exercised directly by tests with a tiny
        peft model + a real torch optimizer.
        """
        import os

        from soup_cli.utils.paths import atomic_write_text

        root = self._ckpt_root()
        ckpt_dir = os.path.join(root, f"step-{int(step)}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if model is not None and hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt_dir)

        has_optimizer = False
        if self.config.include_optimizer_state and optimizer is not None:
            opt_out = os.path.join(ckpt_dir, "optimizer.pt")
            # Refuse to write THROUGH a pre-placed symlink (write-to-arbitrary
            # path in a shared checkpoint dir; security-review LOW #6).
            if os.path.islink(opt_out):
                has_optimizer = False
            else:
                try:
                    import torch

                    torch.save(optimizer.state_dict(), opt_out)
                    has_optimizer = True
                except Exception:  # noqa: BLE001 — best-effort, manifest reflects it
                    has_optimizer = False

        manifest = RLCheckpointState(
            step=int(step),
            checkpoint_dir=ckpt_dir,
            task=self.task,
            has_optimizer=has_optimizer,
            has_ref_model=bool(self.config.include_ref_model),
            has_rollout_buffer=bool(self.config.include_rollout_buffer),
            soup_version=self.soup_version,
        )
        atomic_write_text(
            json.dumps(manifest.to_dict(), indent=2),
            os.path.join(ckpt_dir, "manifest.json"),
        )
        self._saved.append(int(step))
        self._prune()
        return ckpt_dir

    def restore_checkpoint(self, *, step: int, model, optimizer) -> bool:
        """Reload a saved checkpoint's PEFT adapter + optimizer state (v0.71.26).

        Used by the reward-hack mitigation rollback ladder to hop back to the
        last-good checkpoint. Reloads the adapter via
        ``set_peft_model_state_dict`` and the optimizer via ``load_state_dict``.
        Best-effort — returns ``True`` when at least one artifact was restored,
        ``False`` otherwise, and NEVER raises (rollback must not crash the run).
        """
        import os

        ckpt_dir = os.path.join(self._ckpt_root(), f"step-{int(step)}")
        if not os.path.isdir(ckpt_dir):
            return False
        restored = False
        if model is not None:
            try:
                from peft import (
                    PeftModel,
                    load_peft_weights,
                    set_peft_model_state_dict,
                )

                if isinstance(model, PeftModel):
                    set_peft_model_state_dict(model, load_peft_weights(ckpt_dir))
                    restored = True
            except Exception:  # noqa: BLE001 — best-effort restore, never crash
                pass
        if optimizer is not None:
            opt_path = os.path.join(ckpt_dir, "optimizer.pt")
            # SECURITY (review HIGH #1): torch.load(weights_only=False) executes
            # arbitrary pickle. Refuse a SYMLINKED optimizer.pt — an attacker
            # with write access to a shared checkpoint dir could swap the file
            # for a symlink to a malicious pickle between save and restore (RCE).
            # opt_path is otherwise contained (output_dir is_under_cwd-verified in
            # __init__ + int-cast step); the symlink check closes the TOCTOU.
            if os.path.islink(opt_path):
                logger.warning(
                    "refusing to restore optimizer state from a symlinked "
                    "optimizer.pt (%s) — possible tampering",
                    opt_path,
                )
            elif os.path.isfile(opt_path):
                try:
                    import torch

                    optimizer.load_state_dict(
                        torch.load(
                            opt_path, map_location="cpu", weights_only=False
                        )
                    )
                    restored = True
                except Exception:  # noqa: BLE001
                    pass
        return restored

    def _prune(self) -> None:
        """Keep only the ``keep_last`` most-recent step checkpoints."""
        import os
        import shutil

        root = self._ckpt_root()
        if not os.path.isdir(root):
            return
        entries = []
        for name in os.listdir(root):
            if not name.startswith("step-"):
                continue
            full = os.path.join(root, name)
            if os.path.islink(full) or not os.path.isdir(full):
                continue
            entries.append((_step_number(name), full))
        entries.sort(key=lambda t: t[0], reverse=True)
        for step_num, path in entries[self.config.keep_last:]:
            try:
                shutil.rmtree(path)
            except OSError:
                continue
            # Keep the in-memory ledger in sync with disk so a rollback target
            # (max(_saved)) can never point at a deleted checkpoint (v0.71.26
            # code-review HIGH — the reward-hack rollback ladder reads _saved).
            if step_num in self._saved:
                self._saved.remove(step_num)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Per-step hook — save a checkpoint on the configured cadence."""
        try:
            step = int(getattr(state, "global_step", 0) or 0)
            if step <= 0 or step % self.config.save_every_steps != 0:
                return control
            optimizer = kwargs.get("optimizer")
            self.save_checkpoint(step=step, model=model, optimizer=optimizer)
        except Exception:  # noqa: BLE001 — checkpoint failure must not crash run
            pass
        return control


def build_rl_checkpoint_callback(
    config,
    *,
    output_dir: Optional[str] = None,
    task: str = "grpo",
    soup_version: Optional[str] = None,
) -> "RLCheckpointCallback":
    """Build the live mid-epoch RL checkpoint callback (v0.71.11 #238).

    Lifts the v0.70.0 ``NotImplementedError`` stub. ``output_dir`` is the
    trainer's run directory under which ``rl-checkpoints/`` is created.
    Validates config type at the public boundary (fail-fast policy).
    """
    if not isinstance(config, RLCheckpointConfig):
        raise TypeError(
            f"config must be RLCheckpointConfig, got {type(config).__name__}"
        )
    if output_dir is None:
        raise ValueError("output_dir is required to build the RL checkpoint callback")
    return RLCheckpointCallback(
        config,
        output_dir=output_dir,
        task=task,
        soup_version=soup_version,
    )
