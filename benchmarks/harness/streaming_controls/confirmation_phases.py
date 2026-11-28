"""Prospective confirmation phase limits, validated locally before launch."""

from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness.streaming_controls.setup_supervisor import supervise, validate_bounds


@dataclass(frozen=True)
class Phase:
    max_seconds: int
    minimum_remaining_seconds: int
    purpose: str = "setup"


PHASES = {
    "fetch": Phase(1200, 6000),
    "setup": Phase(1200, 5700),
    "controls": Phase(600, 5400),
    "prepare": Phase(1800, 5100),
    "measurement": Phase(5400, 5100, "measurement"),
}


def validate_plan(deadline_unix_s: float) -> None:
    """Exercise the actual supervisor admission for every phase before launch."""
    for phase in PHASES.values():
        validate_bounds(phase.max_seconds, deadline_unix_s, purpose=phase.purpose)


def supervise_phase(
    name: str, script: Path, sha256: str, output: Path, deadline_unix_s: float
) -> dict:
    phase = PHASES[name]
    return supervise(
        script, sha256, output, phase.max_seconds, deadline_unix_s, purpose=phase.purpose
    )
