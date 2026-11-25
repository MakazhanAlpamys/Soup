"""LISA — Layerwise Importance Sampled AdamW (v0.71.34 #267).

LISA (arXiv:2403.17919) gives full-FT quality at LoRA-like memory: every N
steps it freezes all decoder layers except a small randomly-sampled set
(embeddings + language-model head + final norm trainable throughout by default;
``train_embeddings=False`` freezes that group too, #377). The dynamic cousin of
Spectrum's static ``unfrozen_parameters`` selection.

Correctness invariant (see ``utils/peft_wiring.apply_lisa_setup``, shared by the
SFT and pretrain trainers since #307): the trainer leaves the model **fully
trainable at prep time** so HF's ``create_optimizer`` (called
*before* ``on_train_begin``) includes every decoder parameter in its param
groups. This callback then toggles ``requires_grad`` — frozen parameters produce
``grad=None`` and AdamW skips them, and their optimizer state is cleared on
re-freeze — so peak optimizer memory ≈ (embed + head + ``num_layers`` active)
rather than the whole model.

No top-level torch/transformers. Torch is only imported to identify a compiler
wrapper when present. The HF ``TrainerCallback`` and ``ExportableState`` bases are resolved lazily
(mirrors ``monitoring/curriculum_callback.py``) so the module stays import-cheap
while ``LisaCallback`` still inherits the no-op defaults for every Trainer event
(HF dispatches every event with ``getattr(cb, event)`` and no ``hasattr`` guard,
so a bare duck-typed callback would crash on ``on_epoch_begin``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Parameter-name substrings that must stay trainable for every LISA interval:
# input embeddings, the LM head, and the final norm (LISA keeps them always on).
_ALWAYS_ON = (
    "embed_tokens",
    "embed_out",
    "wte",
    "wpe",
    "lm_head",
    "model.norm.",
    "ln_f.",
    "final_layernorm",
)

_LAYER_RE = re.compile(r"(?:layers|h)\.(\d+)\.")
_CHECKPOINT_VERSION = 1
_MISSING_STATE = (
    "checkpoint has no saved LISA state. "
    "Use its model weights to start a fresh run in a new output directory."
)


class LisaResumeError(ValueError):
    """The checkpoint cannot continue the configured LISA schedule."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"LISA resume: {reason}")


@dataclass(frozen=True)
class LisaPolicy:
    """Immutable LISA schedule."""

    num_layers: int
    interval_steps: int
    reset_optimizer: bool = True
    seed: int = 0
    # False freezes the always-on group (embeddings + LM head + final norm) so
    # only the sampled decoder layers are trainable — the always-on set is 70.7%
    # of what LISA trains at 8B, so this is where the memory actually is (#377).
    train_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in ("num_layers", "interval_steps", "seed"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"LisaPolicy.{name} must be int, not {type(val).__name__}")
        if self.num_layers < 1:
            raise ValueError("LisaPolicy.num_layers must be >= 1")
        if self.interval_steps < 1:
            raise ValueError("LisaPolicy.interval_steps must be >= 1")
        if self.seed < 0:
            raise ValueError("LisaPolicy.seed must be >= 0")
        if not isinstance(self.reset_optimizer, bool):
            raise TypeError("LisaPolicy.reset_optimizer must be bool")
        if not isinstance(self.train_embeddings, bool):
            raise TypeError("LisaPolicy.train_embeddings must be bool")


def _try_import_callback_base():
    """Return HF ``TrainerCallback`` (or ``object`` when transformers is absent)."""
    try:
        from transformers import TrainerCallback  # noqa: PLC0415

        return TrainerCallback
    except Exception:  # noqa: BLE001 — transformers optional in slim test envs.
        return object


def _try_import_callback_bases():
    """Return the HF callback bases, or ``object`` in a slim environment.

    Imported inside the function so the module has no top-level transformers
    dependency; the class below still inherits every no-op event stub the HF
    dispatch loop requires. Mirrors ``monitoring/curriculum_callback.py``.
    """
    try:
        from transformers.trainer_callback import ExportableState, TrainerCallback  # noqa: PLC0415

        return TrainerCallback, ExportableState
    except Exception:  # noqa: BLE001 — transformers optional in slim test envs.
        return (object,)


def _is_always_on(name: str) -> bool:
    return any(sub in name for sub in _ALWAYS_ON)


def locate_decoder_layer_indices(model: Any) -> list[int]:
    """Sorted-unique decoder-layer indices from parameter names."""
    seen: set[int] = set()
    for name, _ in model.named_parameters():
        m = _LAYER_RE.search(name)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def _lisa_named_parameters(model: Any) -> dict[str, Any]:
    """Keep checkpoint names stable when Trainer replaces the model with a compiled wrapper."""
    if hasattr(model, "_orig_mod"):
        from torch._dynamo.eval_frame import OptimizedModule

        if isinstance(model, OptimizedModule):
            model = model._orig_mod
    return dict(model.named_parameters())


def _parameter_layout(parameters: dict[str, Any]) -> str:
    """Identify parameter names, shapes and order without reading tensor data."""
    # Optimizer state is loaded by parameter position within each group.
    shapes = [(name, list(param.shape)) for name, param in parameters.items()]
    return hashlib.sha256(json.dumps(shapes, separators=(",", ":")).encode("utf-8")).hexdigest()


def _decoder_mask(parameters: dict[str, Any], chosen: set[int]) -> tuple[set[str], list[str]]:
    names, skipped = set(), []
    for name, param in parameters.items():
        if _is_always_on(name):
            continue
        match = _LAYER_RE.search(name)
        if match is not None and int(match.group(1)) in chosen:
            if param.is_floating_point() or param.is_complex():
                names.add(name)
            else:
                skipped.append(name)
    return names, skipped


def _decode_rng(raw: Any) -> tuple:
    probe = random.Random(0)
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or type(raw[0]) is not int
        or raw[0] != probe.VERSION
        or not isinstance(raw[1], (list, tuple))
        or len(raw[1]) != len(probe.getstate()[1])
        or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in raw[1])
        or not any(raw[1][:-1])  # An all-zero MT state stays zero. The last entry is the cursor.
        or raw[2] is not None
    ):
        raise LisaResumeError("invalid private RNG state")
    rng_state = (raw[0], tuple(raw[1]), None)
    try:
        probe.setstate(rng_state)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LisaResumeError("invalid private RNG state") from exc
    return rng_state


class _LisaCallback_body:  # type: ignore[misc]  # noqa: N801
    """HF ``TrainerCallback`` implementing LISA layer sampling.

    Subclasses the lazily-resolved ``TrainerCallback`` so it inherits the no-op
    default for every Trainer event; only ``on_train_begin`` /
    ``on_step_end`` are overridden.
    """

    def __init__(self, policy: LisaPolicy | dict, console: Any = None) -> None:
        # ExportableState passes its JSON constructor arguments back as a dict
        # when a caller explicitly opts into Trainer's callback restoration.
        self.policy = LisaPolicy(**policy) if isinstance(policy, dict) else policy
        self.console = console
        self._rng = random.Random(self.policy.seed)
        # decoder param objects currently trainable — tracked so we can clear
        # optimizer state for the ones that get re-frozen on the next sample.
        self._active_decoder_params: list[Any] = []
        self._active_decoder_names: set[str] = set()
        self._active_layers: set[int] = set()
        self._layout: str | None = None
        self._checkpoint_step: int | None = None
        self._resume_checkpoint: dict | None = None
        self.fire_count = 0

    def state(self) -> dict:
        """Export the selection in effect after this optimizer step."""
        policy = asdict(self.policy)
        checkpoint = self._resume_checkpoint
        if self._checkpoint_step is not None:
            checkpoint = {
                "version": _CHECKPOINT_VERSION,
                "step": self._checkpoint_step,
                "policy": policy,
                "layout": self._layout,
                "layers": sorted(self._active_layers),
                "active_decoder_names": sorted(self._active_decoder_names),
                "rng": self._rng.getstate(),
                "fire_count": self.fire_count,
            }
        return {"args": {"policy": policy}, "attributes": {"_resume_checkpoint": checkpoint}}

    def prepare_checkpoint(self, model: Any, checkpoint_dir: str | None) -> None:
        """Capture the logical layout and validate a resume before Trainer runs."""
        parameters = _lisa_named_parameters(model)
        self._layout = _parameter_layout(parameters)
        self._resume_checkpoint = None
        if checkpoint_dir is None:
            return

        path = Path(checkpoint_dir) / "trainer_state.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LisaResumeError(_MISSING_STATE) from exc
        except (OSError, ValueError) as exc:
            raise LisaResumeError(f"could not read trainer state at {path}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("stateful_callbacks", {}), dict):
            raise LisaResumeError("invalid trainer state")
        entry = data.get("stateful_callbacks", {}).get("LisaCallback")
        if entry is None:
            raise LisaResumeError(_MISSING_STATE)
        attributes = entry.get("attributes") if isinstance(entry, dict) else None
        if not isinstance(attributes, dict):
            raise LisaResumeError("invalid saved callback entry")
        checkpoint = attributes.get("_resume_checkpoint")
        if checkpoint is None:
            raise LisaResumeError(_MISSING_STATE)
        self._validate_checkpoint(checkpoint, data.get("global_step"), parameters)
        expected_args = {"policy": asdict(self.policy)}
        if (
            json.dumps(entry.get("args"), sort_keys=True)
            != json.dumps(expected_args, sort_keys=True)
            or set(attributes) != {"_resume_checkpoint"}
        ):
            raise LisaResumeError("invalid saved callback entry")
        self._resume_checkpoint = checkpoint

    def _validate_checkpoint(
        self, checkpoint: Any, step: Any, parameters: dict[str, Any]
    ) -> tuple[set[int], set[str], tuple]:
        if not isinstance(checkpoint, dict):
            raise LisaResumeError("invalid checkpoint state")
        version = checkpoint.get("version")
        if type(version) is not int or version != _CHECKPOINT_VERSION:
            raise LisaResumeError("unsupported checkpoint state version")
        saved_step = checkpoint.get("step")
        if type(step) is not int or step < 0 or type(saved_step) is not int or saved_step != step:
            raise LisaResumeError("checkpoint step does not match Trainer step")
        saved_policy = checkpoint.get("policy")
        current_policy = asdict(self.policy)
        if not isinstance(saved_policy, dict) or saved_policy.keys() != current_policy.keys():
            raise LisaResumeError("invalid checkpoint policy")
        try:
            policy = LisaPolicy(**saved_policy)
        except (TypeError, ValueError) as exc:
            raise LisaResumeError("invalid checkpoint policy") from exc
        if policy != self.policy:
            raise LisaResumeError("checkpoint policy does not match the current LISA policy")
        if checkpoint.get("layout") != self._layout:
            raise LisaResumeError("checkpoint parameter layout does not match the model")
        fire_count = checkpoint.get("fire_count")
        if type(fire_count) is not int or fire_count != step // self.policy.interval_steps:
            raise LisaResumeError("invalid checkpoint sample count")

        indices = {int(match.group(1)) for name in parameters if (match := _LAYER_RE.search(name))}
        layers = checkpoint.get("layers")
        count = min(self.policy.num_layers, len(indices))
        if (
            not indices
            or not isinstance(layers, list)
            or len(layers) != count
            or any(type(layer) is not int or layer not in indices for layer in layers)
            or len(set(layers)) != count
        ):
            raise LisaResumeError("invalid sampled decoder layers")
        chosen = set(layers)
        active, _ = _decoder_mask(parameters, chosen)
        names = checkpoint.get("active_decoder_names")
        if (
            not isinstance(names, list)
            or len(names) != len(active)
            or any(not isinstance(name, str) for name in names)
            or set(names) != active
        ):
            raise LisaResumeError("invalid active decoder parameter names")
        return chosen, set(names), _decode_rng(checkpoint.get("rng"))

    # -- HF callback hooks --------------------------------------------------
    def on_train_begin(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if model is None:
            return control
        parameters = _lisa_named_parameters(model)
        if self._layout is None:
            self._layout = _parameter_layout(parameters)
        step = getattr(state, "global_step", 0)
        checkpoint = self._resume_checkpoint
        if checkpoint is None:
            if step:
                raise LisaResumeError(_MISSING_STATE)
            self._rng.seed(self.policy.seed)
            self.fire_count = 0
            self._resample(model, optimizer=None)
        else:
            chosen, active, rng_state = self._validate_checkpoint(checkpoint, step, parameters)
            self._rng.setstate(rng_state)
            self.fire_count = checkpoint["fire_count"]
            self._active_layers = chosen
            self._apply_mask(parameters, active)
            optimizer = kwargs.get("optimizer")
            if optimizer is not None and self.policy.reset_optimizer:
                frozen = [
                    param
                    for name, param in parameters.items()
                    if name not in active and not _is_always_on(name) and _LAYER_RE.search(name)
                ]
                self._clear_optimizer_state(optimizer, frozen)
            self._resume_checkpoint = None
        self._checkpoint_step = step
        return control

    def on_step_end(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        global_step = int(getattr(state, "global_step", 0) or 0)
        model = kwargs.get("model")
        if global_step <= 0 or model is None:
            return control
        self._checkpoint_step = global_step
        if global_step % self.policy.interval_steps != 0:
            return control
        self._resample(model, optimizer=kwargs.get("optimizer"))
        self.fire_count += 1
        if self.console is not None:
            try:
                self.console.print(
                    f"[cyan]LISA[/cyan] re-sampled {self.policy.num_layers} "
                    f"decoder layer(s) at step {global_step}"
                )
            except Exception:  # noqa: BLE001 — console.print is best-effort
                pass
        return control

    # -- core ---------------------------------------------------------------
    def _resample(self, model: Any, optimizer: Any) -> None:
        indices = locate_decoder_layer_indices(model)
        if not indices:
            # Fail loud rather than silently full-fine-tune with none of LISA's
            # memory savings: the callback left the whole model trainable and
            # cannot select a decoder subset for this architecture.
            raise RuntimeError(
                "LISA: could not detect numbered decoder layers "
                "('layers.N.' / 'h.N.') in the model — LISA layer sampling "
                "cannot be applied to this architecture. Disable lisa_enabled "
                "or use a supported decoder LM."
            )
        k = min(self.policy.num_layers, len(indices))
        chosen = set(self._rng.sample(indices, k))

        prev_active = self._active_decoder_params
        parameters = _lisa_named_parameters(model)
        active, skipped_non_float = _decoder_mask(parameters, chosen)
        self._active_layers = chosen
        self._apply_mask(parameters, active)

        # Clear optimizer state for decoder params that were active and are now
        # frozen (avoids stale Adam moments; mirrors ReLoRACallback reset).
        if optimizer is not None and self.policy.reset_optimizer:
            new_ids = {id(p) for p in self._active_decoder_params}
            re_frozen = [p for p in prev_active if id(p) not in new_ids]
            self._clear_optimizer_state(optimizer, re_frozen)

        if skipped_non_float:
            logger.warning(
                "LISA: %d matched parameter(s) are non-float (e.g. quantized) "
                "and cannot be trained — skipped. LISA requires "
                "quantization: none. First skipped: %s",
                len(skipped_non_float),
                skipped_non_float[0],
            )

    def _apply_mask(self, parameters: dict[str, Any], active: set[str]) -> None:
        self._active_decoder_names = active
        self._active_decoder_params = []
        for name, param in parameters.items():
            if _is_always_on(name):
                param.requires_grad = self.policy.train_embeddings
            else:
                param.requires_grad = name in active
                if param.requires_grad:
                    self._active_decoder_params.append(param)

    @staticmethod
    def _clear_optimizer_state(optimizer: Any, params: list[Any]) -> None:
        try:
            state = getattr(optimizer, "state", None)
            if state is None:
                return
            for param in params:
                if param in state:
                    state[param] = {}
        except Exception:  # noqa: BLE001 — optimizer state shape varies (DS/FSDP)
            return


_LAZY_CALLBACKS = {
    "LisaCallback": _LisaCallback_body,
}
_BODY_SKIP = frozenset(("__dict__", "__weakref__"))


def __getattr__(name: str):  # PEP 562
    body = _LAZY_CALLBACKS.get(name)
    if body is not None:
        bases = _try_import_callback_bases()
        ns = {k: v for k, v in vars(body).items() if k not in _BODY_SKIP}
        cls = type(name, bases, ns)
        cls.__module__ = __name__
        cls.__qualname__ = name
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
