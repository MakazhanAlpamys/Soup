"""Training paths without a live eval-gate callback (#1811).

Kept import-light so config validation and the train CLI shortcut can refuse
the same setting before hardware, data or model initialization.
"""

from __future__ import annotations


def eval_gate_refusal(*, task: str, backend: str) -> str | None:
    """Explain why an enabled training gate cannot run, or return None."""
    if task in ("prm", "unlearn", "moe_lora_routing"):
        return (
            f"training.eval_gate is not supported for task={task!r} "
            f"because {task!r} does not attach a live training callback"
        )
    if backend == "mlx":
        return (
            "training.eval_gate is not supported for backend='mlx' "
            "because MLX does not attach a live training callback"
        )
    return None
