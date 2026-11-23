"""#3241: judge_model assigned after suite load is re-checked by _require_gate_tasks.

The load-time validator (GateTask._valid_judge_url) refuses a private IP
literal (https) or a non-loopback http host, but only when the suite is
parsed. An SDK caller that assigns ``suite.tasks[i].judge_model`` afterwards
used to reach ``run_gate`` un-checked. ``_require_gate_tasks`` now applies the
same rule before any generator call.
"""

import pytest
from pydantic import ValidationError

from soup_cli.eval.gate import EvalSuite, GateTask, _require_gate_tasks


def _suite_with_judge(judge_model: str) -> EvalSuite:
    suite = EvalSuite(
        suite="s",
        tasks=[GateTask(type="judge", name="t", threshold=1.0, judge_model="ollama://m")],
    )
    suite.tasks[0].judge_model = judge_model
    return suite


class TestJudgeModelRecheckedAfterLoad:
    def test_private_ip_https_assigned_after_load_is_refused(self):
        with pytest.raises(ValueError, match="judge_model URL"):
            _require_gate_tasks(_suite_with_judge("https://169.254.169.254/model"))

    def test_non_loopback_http_assigned_after_load_is_refused(self):
        with pytest.raises(ValueError, match="disallowed scheme"):
            _require_gate_tasks(_suite_with_judge("http://192.168.1.10:8080/model"))

    def test_ollama_control_still_runs(self):
        _require_gate_tasks(_suite_with_judge("ollama://llama3.1"))

    def test_public_https_control_still_runs(self):
        _require_gate_tasks(_suite_with_judge("https://api.example.com/v1"))

    def test_loopback_http_control_still_runs(self):
        _require_gate_tasks(_suite_with_judge("http://localhost:11434/v1"))

    def test_none_judge_model_is_left_alone(self):
        suite = EvalSuite(
            suite="s",
            tasks=[GateTask(type="custom", name="t", threshold=1.0, scorer="exact")],
        )
        _require_gate_tasks(suite)

    def test_load_time_validator_unchanged(self):
        with pytest.raises(ValidationError, match="judge_model URL"):
            GateTask(
                type="judge", name="t", threshold=1.0,
                judge_model="https://10.0.0.5/model",
            )
