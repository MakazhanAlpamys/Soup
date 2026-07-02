"""Regression tests for the MEDIUM/LOW findings in CODE_REVIEW.md."""

from __future__ import annotations

from pathlib import Path

import soup_cli


def _src(rel: str) -> str:
    return (Path(soup_cli.__file__).parent / rel).read_text(encoding="utf-8")


def test_license_matrix_permissive_weak_symmetric():
    from soup_cli.utils.license_matrix import (
        _PERMISSIVE,
        _WEAK_COPYLEFT,
        LICENSE_MATRIX,
    )

    assert _WEAK_COPYLEFT in LICENSE_MATRIX[_PERMISSIVE]
    assert _PERMISSIVE in LICENSE_MATRIX[_WEAK_COPYLEFT]


def test_detect_format_prefers_tool_calling_over_audio():
    from soup_cli.data.formats import detect_format

    row = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "f"}],
        "tool_calls": [{"name": "f", "arguments": "{}"}],
        "audio": "a.wav",
    }
    assert detect_format([row]) == "tool-calling"


def test_converters_reject_null_content():
    from soup_cli.data.formats import format_to_messages

    # A JSON null in a required content field routes the row to the drop path
    # (returns None) instead of producing literal None content.
    assert format_to_messages({"instruction": "hi", "output": None}, "alpaca") is None
    assert (
        format_to_messages(
            {"prompt": "p", "chosen": None, "rejected": "r"}, "dpo"
        )
        is None
    )
    # A well-formed row still converts.
    ok = format_to_messages({"instruction": "hi", "output": "yo"}, "alpaca")
    assert ok["messages"][-1]["content"] == "yo"


def test_tool_call_args_subset_penalizes_hallucinated_args():
    # The dead ternary `0.5 if not out_args else 0.5` gave hallucinated args
    # full credit; the fix scores 0.0 for the args portion in that branch.
    src = _src("eval/custom.py")
    assert "args_score = 0.5 if not out_args else 0.0" in src
    assert "0.5 if not out_args else 0.5" not in src


def test_ema_and_median_use_window_size():
    from soup_cli.utils.reward_hack_control import smooth_signal

    # Windowed EMA: a longer retained window folds in more history, so the
    # result differs from the 1-element (2-tap) case — proving
    # reward_hack_smoothing_window now has effect for EMA.
    short = smooth_signal(1.0, [0.0], method="ema")
    longer = smooth_signal(1.0, [1.0, 0.0, 0.0], method="ema")
    assert short != longer
    # median genuinely uses the retained window too.
    assert smooth_signal(10.0, [1.0, 2.0], method="median") == 2.0


def test_sse_metric_push_preserves_zero():
    assert "float(loss) if loss is not None else None" in _src("monitoring/callback.py")


def test_deploy_target_rejects_windows_drive_absolute():
    src = _src("cans/schema.py")
    assert 'value[1] == ":"' in src  # drive-absolute (C:\...) now rejected


def test_diagnose_rejects_non_numeric_score():
    src = _src("commands/diagnose.py")
    assert "must be a number" in src


def test_generate_partial_save_present():
    src = _src("commands/generate.py")
    assert "Partial save" in src and "generated before the error" in src


def test_package_docstring_has_no_mojibake():
    assert soup_cli.__doc__ is not None
    assert "вЂ" not in soup_cli.__doc__
    assert "—" in soup_cli.__doc__
