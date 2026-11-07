"""The `Warm HF Hub cache` step of the `test` job downloads over plain HTTP, on a short budget.

Measured 2026-10-19 over the last 12 pull-request runs: on macOS and Windows the step took 6.3
minutes in every cell (6.25 to 6.37) and the cache behind it was never saved. The Hub answers
404 on `/api/models/gpt2/xet-read-token/<sha>` -- the token the `hf_xet` download path asks for
-- and the retry loop treats a 404 like a 429 until its budget runs out. About 57 such cells a
day, 362 runner-minutes a day, since about 2026-10-16 17Z. Linux never hit it: its restored
cache already held the file.

`HF_HUB_DISABLE_XET` makes huggingface_hub's `is_xet_available()` false (constants.py and
utils/_runtime.py, read at v2.2.0, the version `.github/constraints/pinned-upstream.txt`
installs), so the download takes the plain HTTP path and never asks for that token. The budget
is 120 s, not 360: it is spent in full only when the Hub refuses a repo, and the step never
fails the job either way.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
WARM_STEP = "Warm HF Hub cache (models the tests load)"
SAVE_STEP = "Save HF Hub cache"


def _test_steps() -> list[dict[str, Any]]:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]["test"]["steps"]


def _step(name: str) -> dict[str, Any]:
    steps = [step for step in _test_steps() if step.get("name") == name]
    assert len(steps) == 1, (name, steps)
    return steps[0]


class TestTheWarmStepAvoidsTheXetTokenPath:
    def test_the_step_disables_xet_for_its_own_downloads(self) -> None:
        env = _step(WARM_STEP).get("env") or {}
        assert env.get("HF_HUB_DISABLE_XET") == "1", env

    def test_it_is_a_python_step_whose_downloads_read_that_environment(self) -> None:
        """Scoped to this step, not the job: a job-level `env:` would also change how every test
        that downloads fetches its model."""
        step = _step(WARM_STEP)
        assert step.get("shell") == "python"
        assert "snapshot_download" in str(step["run"])
        job_env = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]["test"].get("env") or {}
        assert "HF_HUB_DISABLE_XET" not in job_env

    def test_the_variable_is_a_name_huggingface_hub_still_reads(self) -> None:
        """A renamed variable would be ignored silently, and the step would go back to 6 minutes."""
        hub = pytest.importorskip("huggingface_hub")
        if int(hub.__version__.split(".")[0]) < 1:
            pytest.skip("hf_xet is a hard dependency from huggingface_hub 1.0")
        from huggingface_hub import constants

        assert hasattr(constants, "HF_HUB_DISABLE_XET"), hub.__version__


class TestTheWarmStepBudget:
    def _deadline(self) -> int:
        script = str(_step(WARM_STEP)["run"])
        match = re.search(r"deadline\s*=\s*time\.monotonic\(\)\s*\+\s*(\d+)", script)
        assert match is not None, "the warm step has no single retry budget"
        return int(match.group(1))

    def test_the_whole_list_shares_one_budget_of_120_seconds(self) -> None:
        assert self._deadline() == 120

    def test_it_never_fails_the_job_and_saves_only_a_complete_warm(self) -> None:
        script = str(_step(WARM_STEP)["run"])
        assert "except Exception" in script
        assert "partial" in script and "all" in script
        condition = str(_step(SAVE_STEP).get("if", ""))
        assert "steps.warm.outputs.warmed == 'all'" in condition, condition
