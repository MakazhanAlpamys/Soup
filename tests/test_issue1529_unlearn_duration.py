"""#1529 — a finished ``task: unlearn`` run crashed the completion panel
with ``KeyError: 'duration'``: the unlearn wrapper returned only
``duration_secs``, and the panel indexed ``result['duration']`` directly.

The fix has two halves, exactly as the issue proposes:

- the unlearn wrapper now returns the pre-formatted ``duration`` string
  next to ``duration_secs`` like every other wrapper;
- the completion panel is tolerant: it prefers ``result["duration"]`` and
  falls back to a string built from ``duration_secs``, so the next wrapper
  that forgets the key cannot crash a finished run.

The scan test keeps the second wrapper from regressing: any trainer
module that reports ``duration_secs`` must also report ``duration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.commands.train import _completion_duration_display

TRAINER_DIR = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"


class TestCompletionPanelDuration:
    def test_prefers_the_preformatted_duration_string(self):
        result = {"duration": "1h 5m", "duration_secs": 3900.0}
        assert _completion_duration_display(result) == "1h 5m"

    def test_falls_back_to_duration_secs_when_string_missing(self):
        # This is the unlearn result shape on main: the panel raised
        # KeyError instead of printing the completed run.
        result = {"duration_secs": 3900.0}
        assert _completion_duration_display(result) == "1h 5m"

    def test_fallback_formats_sub_hour_durations_as_minutes(self):
        assert _completion_duration_display({"duration_secs": 321.0}) == "5m"
        assert _completion_duration_display({"duration_secs": 7320.0}) == "2h 2m"

    def test_missing_both_keys_does_not_raise(self):
        assert _completion_duration_display({}) == "unknown"

    @pytest.mark.parametrize(
        "result",
        [
            {"duration_secs": 3900.0},  # mutation: unlearn shape on main
            {},  # mutation: wrapper returned neither key
        ],
    )
    def test_panel_code_path_never_raises_on_missing_key(self, result):
        # The helper replaces the panel's direct result['duration'] index,
        # so exercising it with key-less dicts is the mutation check.
        assert isinstance(_completion_duration_display(result), str)


class TestEveryDurationSecsWrapperAlsoReportsDuration:
    def test_scan_trainer_modules(self):
        offenders = []
        for path in sorted(TRAINER_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            if '"duration_secs"' in source and '"duration"' not in source:
                offenders.append(path.name)
        assert not offenders, (
            "trainer module(s) report duration_secs without a pre-formatted "
            f"duration string: {offenders}"
        )
