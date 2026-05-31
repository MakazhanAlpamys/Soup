"""soup diagnose — post-training model report card (v0.56.0).

Six independent failure-mode probes scored against a base reference:

- forgetting        — catastrophic forgetting on held-out tasks
- refusal           — refusal-rate regression on advbench / xstest
- format            — JSON / regex / tool-call validity drift
- mode_collapse     — self-BLEU + diversity at T=0 and T=1
- memorization      — training-prefix echo on partial-prompt probes
- contamination     — overlap of training data with public benchmarks

Each probe returns a ``FailureScore`` with ``score in [0, 1]`` (lower is
worse) and a verdict OK / MINOR / MAJOR — same taxonomy as v0.26.0
Quant-Lobotomy.

All probes accept caller-supplied generator callables so the CLI stays
unit-testable without a GPU; live model-loading factories land in
v0.56.1 (matches v0.27.0 MII / v0.37.0 multipack / v0.50.0 GRPO Plus /
v0.54.0 probe runner stub-then-live cadence).
"""

from __future__ import annotations

from soup_cli.utils.diagnose.report import (
    FAILURE_MODES,
    VERDICTS,
    FailureReport,
    FailureScore,
    classify_score,
    compose_report,
    overall_verdict,
)

__all__ = [
    "FAILURE_MODES",
    "VERDICTS",
    "FailureReport",
    "FailureScore",
    "classify_score",
    "compose_report",
    "overall_verdict",
]
