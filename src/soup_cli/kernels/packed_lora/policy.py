# SPDX-License-Identifier: Apache-2.0
#
# Origin: Soup Kernel Challenge (Decentrathon 6.0), the submission of team Luck
# (BAITC-Hacks/hack-e79529fc-luck@b3f1225f); the int8 gate/up packs follow ideas seen in the
# solutions of teams Null Pointers, 21, DarkDaur and OrbiTeam. Modified for Soup: the module
# switches and KZ_* environment variables of the original became this per-run policy.
"""What one run was asked for, and the numerics state that goes with it.

``Policy.from_block`` turns the run's ``PackedLoraConfig`` into a frozen value; nothing here reads
the environment or keeps module-level state, so two runs in one process cannot see each
other. No torch: the CPU suite and the CLI start-up path import this module.

The int8 backward
-----------------
The MLP's three input-gradient GEMMs (``g @ W_down``, ``d_gate @ W_gate``, ``d_up @ W_up``)
can run on int8 tensor cores. ``int8_backward`` chooses how:

``auto``   on when a start-up gradient check passes (``int8check.py``), else off
``on``     always on
``stash``  only the gate/up activations are stored as int8; the GEMMs run in the compute dtype
``off``    the exact mode: no int8 anywhere (about 21-24 % slower)

and ``int8_rounding`` how the gradient quantisers round. Round-to-nearest zeroes every entry
below half a step of its row's largest value; that loss is a bias, not noise, and it showed
as a training loss drifting above plain Soup's (+0.68 % after 500 steps). ``grad`` rounds the
two gradient quantisers stochastically (``floor(x / scale + u)``, ``u`` uniform in [0, 1)),
which is unbiased in expectation and brought the drift to the exact mode's level (+0.04 % train
loss, -0.005 % held-out eval loss). ``stochastic`` also dithers the int8 weights; it measured no
different from ``grad``. The default is ``grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

ACTIVATIONS = ("auto", "resident", "offload")
INT8_BACKWARD_MODES = ("auto", "on", "stash", "off")
INT8_ROUNDINGS = ("nearest", "grad", "stochastic")
ON_UNSUPPORTED = ("error", "stock")

#: Cosine between the int8 and the compute-dtype ``d_x`` of the start-up check at which the
#: int8 path stays on. Round-to-nearest quantisers measure about 0.9995; stochastic rounding is
#: unbiased but noisier per step (about 0.9991), so its bar is lower. On the Kaggle T4 (fp16)
#: the check passed with cosine 0.99881 against 0.998: do not raise this without data.
COSINE_MIN = 0.999
COSINE_MIN_SR = 0.998

_MIB = 1 << 20


@dataclass(frozen=True)
class Policy:
    """The settings of ``training.packed_lora`` for one run."""

    gpu_budget_bytes: Optional[int] = None
    activations: str = "auto"
    int8_backward: str = "auto"
    int8_rounding: str = "grad"
    on_unsupported: str = "error"

    def __post_init__(self) -> None:
        for name, allowed in (("activations", ACTIVATIONS),
                              ("int8_backward", INT8_BACKWARD_MODES),
                              ("int8_rounding", INT8_ROUNDINGS),
                              ("on_unsupported", ON_UNSUPPORTED)):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"packed_lora {name} must be one of {allowed}, not {value!r}")
        budget = self.gpu_budget_bytes
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int)
                                   or budget <= 0):
            raise ValueError(f"packed_lora gpu_budget_bytes must be a positive int, not {budget!r}")

    @classmethod
    def from_block(cls, block: Any) -> "Policy":
        """From a ``config.PackedLoraConfig`` (any object with its attributes)."""
        mib = block.gpu_budget_mib
        return cls(
            gpu_budget_bytes=None if mib is None else int(mib) * _MIB,
            activations=block.activations,
            int8_backward=block.int8_backward,
            int8_rounding=block.int8_rounding,
            on_unsupported=block.on_unsupported,
        )

    def rounding(self) -> "Rounding":
        """A fresh rounding state for the quantisers of one run."""
        return Rounding(self.int8_rounding)

    def int8_switch(self) -> "Int8Switch":
        """A fresh int8 decision for one run."""
        return Int8Switch(self.int8_backward)


class Rounding:
    """Rounding of the int8 quantisers for one run, and the seeds of their random numbers.

    ``grad``: the gradient quantisers round stochastically. ``stochastic``: the int8 weights
    too. Seeds come from a counter that is hashed, so runs are reproducible and no torch random
    state is touched.
    """

    def __init__(self, mode: str = "grad") -> None:
        if mode not in INT8_ROUNDINGS:
            raise ValueError(f"int8 rounding must be one of {INT8_ROUNDINGS}, not {mode!r}")
        self.mode = mode
        self.grad = mode != "nearest"
        self.weights = mode == "stochastic"
        self._count = 0

    def next_seed(self) -> int:
        """A new 31-bit seed for one quantiser launch."""
        self._count += 1
        return (self._count * 2654435761 + 12345) % 2147483647

    def grad_seed(self) -> int:
        """Seed for a gradient quantiser launch (0 where rounding is to nearest)."""
        return self.next_seed() if self.grad else 0

    def weight_seed(self) -> int:
        """Seed for an int8 weight decode (0 where the weights round to nearest)."""
        return self.next_seed() if self.weights else 0

    @property
    def cosine_min(self) -> float:
        """The start-up check's bar for this rounding."""
        return COSINE_MIN_SR if self.grad else COSINE_MIN


class Int8Switch:
    """Whether the backward may use the int8 GEMMs, for one run."""

    def __init__(self, mode: str = "auto") -> None:
        if mode not in INT8_BACKWARD_MODES:
            raise ValueError(f"int8_backward must be one of {INT8_BACKWARD_MODES}, not {mode!r}")
        self.mode = mode
        self.decided: Optional[bool] = None  # the start-up check's verdict ("auto" only)
        self.check: Optional[dict] = None  # {"cosine", "rel_err"} of that check

    @property
    def needs_check(self) -> bool:
        """Whether the start-up gradient check still has to run."""
        return self.mode == "auto" and self.decided is None

    def enabled(self) -> bool:
        """Whether int8 may be used now. Before the "auto" check has run: yes (its layers are
        sized for the int8 stash)."""
        if self.mode != "auto":
            return self.mode != "off"
        return True if self.decided is None else self.decided

    def gemm_enabled(self) -> bool:
        """Whether the backward's MLP input-gradient GEMMs run in int8 (not with stash / off)."""
        return self.enabled() and self.mode != "stash"

    def decide(self, passed: bool, cosine: Optional[float] = None,
               rel_err: Optional[float] = None) -> bool:
        """Record the start-up check's verdict ("auto" only)."""
        self.decided = bool(passed)
        if cosine is not None:
            self.check = {"cosine": cosine, "rel_err": rel_err}
        return self.decided
