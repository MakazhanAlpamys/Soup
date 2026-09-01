"""Model-free contracts for Soup's future offline AutoDistill pipeline."""

from soup_cli.autodistill.contract import (
    AUTODISTILL_PLAN_SCHEMA,
    CAPTURE_TOKEN_SCHEMA,
    CONSUMPTION_EVENT_SCHEMA,
    SHARD_MANIFEST_SCHEMA,
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    build_plan_estimate,
)

__all__ = [
    "AUTODISTILL_PLAN_SCHEMA",
    "CAPTURE_TOKEN_SCHEMA",
    "CONSUMPTION_EVENT_SCHEMA",
    "SHARD_MANIFEST_SCHEMA",
    "AutoDistillPlan",
    "CaptureToken",
    "ShardManifest",
    "build_plan_estimate",
]
