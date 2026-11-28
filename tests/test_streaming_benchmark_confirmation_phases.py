"""The exact prospective phase plan must be executable on a CPU host."""

import hashlib
import os
import time

import pytest

from benchmarks.harness.streaming_controls.confirmation_phases import (
    PHASES,
    supervise_phase,
    validate_plan,
)


def test_whole_phase_plan_is_admitted():
    validate_plan(time.time() + 7200)


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_whole_phase_plan_runs_through_real_supervisor(tmp_path):
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    sha = hashlib.sha256(script.read_bytes()).hexdigest()
    for name in PHASES:
        result = supervise_phase(name, script, sha, tmp_path / name, time.time() + 10)
        assert result["status"] == "PASS"
        assert result["group_cleanup_confirmed"]


def test_expired_plan_is_not_admitted():
    with pytest.raises(ValueError, match="^Unexpired deadline and setup bound <=1800s required$"):
        validate_plan(time.time() - 1)
