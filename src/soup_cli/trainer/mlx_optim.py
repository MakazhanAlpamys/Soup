"""Optimizer and LR-schedule construction for the MLX SFT path (#686).

The MLX backend validated ``warmup_ratio``, ``scheduler``, ``weight_decay`` and
``optimizer`` and then built ``optim.AdamW(learning_rate=lr)`` -- a bare scalar.
All four were accepted and dropped without a word, so an MLX run silently
trained a different recipe from the one the config described.

**The step unit is optimizer updates, not iterations.** Measured rather than
assumed: MLX calls a callable ``learning_rate`` with ``optimizer.step``, and
that counter advances once per ``optimizer.update()`` -- which mlx-lm invokes
only when ``it % grad_accumulation_steps == 0``. So a schedule built against
``iters`` would stretch its warmup by a factor of ``grad_accumulation_steps``
and never reach the cosine floor. Everything here counts updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

# Soup scheduler names -> how to build them from `mlx.optimizers`. The repo
# passes `training.scheduler` straight to HuggingFace as `lr_scheduler_type`
# on the transformers path, so these names are HF's.
_SUPPORTED_SCHEDULERS = ("cosine", "linear", "constant", "constant_with_warmup")

# Soup optimizer names -> MLX optimizer classes. `training.optimizer` is
# validated against `utils.optimizer_zoo`, an allowlist far wider than anything
# MLX ships, so most valid Soup names have no equivalent here. Today every one
# of them silently becomes AdamW; the ones below are the ones that genuinely
# correspond, and everything else is refused by name.
_OPTIMIZER_MAP = {
    "adamw_torch": "AdamW",
    "adamw_hf": "AdamW",
    "adamw": "AdamW",
    "adamw_torch_fused": "AdamW",
    "adam": "Adam",
    "sgd": "SGD",
    "lion": "Lion",
    "adafactor": "Adafactor",
    "adagrad": "Adagrad",
    "adamax": "Adamax",
    "rmsprop": "RMSprop",
    "adadelta": "AdaDelta",
    "muon": "Muon",
}

# Optimizers whose MLX constructor takes no `weight_decay`. Passing it would
# TypeError, and silently dropping it would be the bug this module exists to
# fix, so a non-default decay on one of these is refused instead.
# Derived from the real constructor signatures, and pinned by
# `test_the_weight_decay_table_matches_the_real_signatures` so it cannot drift
# when mlx-lm changes. It is a table rather than runtime introspection so the
# refusal message can be produced on a machine without MLX installed.
_NO_WEIGHT_DECAY = frozenset({"Adam", "Adagrad", "Adamax", "AdaDelta", "RMSprop"})


class MlxOptimizerError(ValueError):
    """Raised for a validated Soup setting the MLX backend cannot honour.

    Explicit refusal, not a silent fallback: a run that quietly substitutes
    AdamW for the optimizer you asked for is indistinguishable from one that
    honoured it, and that is the defect #686 reports.
    """


@dataclass(frozen=True)
class OptimizerPlan:
    """What will actually be built, resolved before anything is constructed."""

    optimizer_name: str
    """The MLX class name, e.g. ``"AdamW"``."""

    scheduler: str
    warmup_updates: int
    total_updates: int
    weight_decay: float
    peak_lr: float
    warnings: List[str] = field(default_factory=list)

    def as_metadata(self) -> Dict[str, Any]:
        """The effective configuration, for the adapter's metadata."""
        return {
            "optimizer": self.optimizer_name,
            "scheduler": self.scheduler,
            "warmup_updates": self.warmup_updates,
            "total_updates": self.total_updates,
            "weight_decay": self.weight_decay,
            "peak_lr": self.peak_lr,
        }


def resolve_optimizer_name(name: str) -> str:
    """Map a Soup optimizer name to an MLX class name, or refuse it."""
    key = str(name).strip().lower()
    if key in _OPTIMIZER_MAP:
        return _OPTIMIZER_MAP[key]
    raise MlxOptimizerError(
        f"training.optimizer={name!r} has no MLX equivalent. MLX supports "
        f"{', '.join(sorted(set(_OPTIMIZER_MAP)))}. Choose one of those for the "
        "MLX backend, or run this recipe on the transformers backend -- the "
        "previous behaviour silently substituted AdamW."
    )


def plan_optimizer(
    *,
    lr: float,
    optimizer: str,
    scheduler: str,
    warmup_ratio: float,
    weight_decay: float,
    total_updates: int,
) -> OptimizerPlan:
    """Resolve every optimizer-affecting setting, or refuse with a reason.

    ``total_updates`` must already be in optimizer-update units
    (``iters // gradient_accumulation_steps``); see the module docstring.
    """
    mlx_name = resolve_optimizer_name(optimizer)
    sched = str(scheduler).strip().lower()
    if sched not in _SUPPORTED_SCHEDULERS:
        raise MlxOptimizerError(
            f"training.scheduler={scheduler!r} is not available on the MLX "
            f"backend. Supported: {', '.join(_SUPPORTED_SCHEDULERS)}. The "
            "previous behaviour ran a constant learning rate regardless."
        )

    total = max(1, int(total_updates))
    warmup = int(float(warmup_ratio) * total)
    warnings: List[str] = []

    # A benchmark-sized run can round the warmup away entirely (0.03 * 12 = 0).
    # That is legitimate arithmetic, but a run that silently gets no warmup
    # when one was configured is the failure shape this issue is about, so it
    # is said out loud rather than inferred from the LR curve afterwards.
    if float(warmup_ratio) > 0 and warmup == 0:
        warnings.append(
            f"warmup_ratio={warmup_ratio} over {total} optimizer update(s) "
            "rounds to 0 warmup steps; training starts at the peak learning "
            "rate. Raise warmup_ratio, or lower gradient_accumulation_steps "
            "to produce more updates."
        )
    if warmup >= total:
        warnings.append(
            f"warmup_ratio={warmup_ratio} covers all {total} optimizer "
            "update(s); the learning rate never leaves warmup and no decay "
            "phase runs."
        )
        warmup = max(0, total - 1)

    decay = float(weight_decay)
    if mlx_name in _NO_WEIGHT_DECAY and decay:
        raise MlxOptimizerError(
            f"training.weight_decay={decay} cannot be applied to MLX "
            f"{mlx_name}, whose constructor takes no weight_decay. Use an "
            "optimizer that supports it (adamw), or set weight_decay to 0."
        )

    return OptimizerPlan(
        optimizer_name=mlx_name,
        scheduler=sched,
        warmup_updates=warmup,
        total_updates=total,
        weight_decay=decay,
        peak_lr=float(lr),
        warnings=warnings,
    )


def build_lr_schedule(plan: OptimizerPlan) -> Union[float, Callable[[Any], Any]]:
    """Build the learning-rate schedule described by ``plan``.

    Returns a plain float when the result is genuinely constant with no warmup,
    so the common case constructs exactly what the old code did and the change
    is provably confined to the configurations that asked for something else.
    """
    peak, warmup, total = plan.peak_lr, plan.warmup_updates, plan.total_updates
    decay_steps = max(1, total - warmup)

    # A constant rate with no warmup is a plain float and needs nothing from
    # MLX, so it is answered before the import. That keeps the function
    # callable -- and testable -- on a machine with no MLX, which is every CI
    # runner this project has; a module-wide import here made a genuinely
    # platform-independent behaviour fail on Windows.
    if warmup <= 0 and plan.scheduler in ("constant", "constant_with_warmup"):
        return peak

    import mlx.optimizers as optim

    if plan.scheduler in ("constant", "constant_with_warmup"):
        body: Union[float, Callable[[Any], Any]] = peak
    elif plan.scheduler == "cosine":
        body = optim.cosine_decay(peak, decay_steps)
    else:  # linear
        body = optim.linear_schedule(peak, 0.0, decay_steps)

    if warmup <= 0:
        return body
    warm = optim.linear_schedule(0.0, peak, warmup)
    if not callable(body):
        # `join_schedules` needs two callables; a constant is one that ignores
        # its argument. Written out rather than lambda'd so the closure is
        # obvious in a traceback.
        const = float(body)

        def body(_step, _const=const):
            return _const

    return optim.join_schedules([warm, body], [warmup])


def build_optimizer(plan: OptimizerPlan):
    """Construct the MLX optimizer described by ``plan``."""
    import mlx.optimizers as optim

    cls = getattr(optim, plan.optimizer_name)
    kwargs: Dict[str, Any] = {"learning_rate": build_lr_schedule(plan)}
    if plan.optimizer_name not in _NO_WEIGHT_DECAY:
        kwargs["weight_decay"] = plan.weight_decay
    return cls(**kwargs)
