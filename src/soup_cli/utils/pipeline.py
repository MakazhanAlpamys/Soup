"""Pipeline parallelism config validation (v0.27.0).

v0.27.0 ships pipeline config wiring only — actual execution is deferred to
v0.27.1. Validation here prevents misconfiguration from reaching a trainer
that cannot execute the requested strategy.
"""

from __future__ import annotations


def validate_pipeline_config(
    parallelism: str,
    pipeline_stages: int,
    device: str,
    gpu_count: int,
) -> list[str]:
    """Validate pipeline-parallel settings.

    Args:
        parallelism: ``"data"`` or ``"pipeline"``.
        pipeline_stages: Number of pipeline stages.
        device: Training device (``cuda`` / ``cpu`` / ``mps``).
        gpu_count: Detected GPU count.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []
    if parallelism != "pipeline":
        return errors

    if pipeline_stages < 2:
        errors.append(
            "parallelism='pipeline' requires pipeline_stages >= 2 "
            f"(got {pipeline_stages})."
        )
    if device != "cuda":
        errors.append(
            f"Pipeline parallelism requires CUDA GPUs. Current device: {device}."
        )
    if gpu_count < pipeline_stages:
        errors.append(
            f"Pipeline parallelism needs >= {pipeline_stages} GPUs "
            f"(got {gpu_count})."
        )
    return errors
