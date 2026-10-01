"""#1529 — a finished unlearn run must not die in the completion panel.

``soup train`` ends every run by printing a panel that reads ``result['duration']``.
Every wrapper that reaches that panel returned a pre-formatted ``duration`` string
next to ``duration_secs`` except ``UnlearnTrainerWrapper``, which returned only the
latter. ``finish_run()`` had already recorded the run as complete at that point, so
the user got a ``KeyError: 'duration'`` traceback instead of the panel.

Three layers here, because the issue asks for both and one of them turned out to be
load-bearing on its own:

* the wrapper contract — an AST scan over ``trainer/*.py`` that asserts every
  wrapper returning ``duration_secs`` also returns ``duration``, so the next
  wrapper cannot repeat the omission;
* the formatter — ``_format_training_duration`` renders from ``duration_secs`` when
  the string form is missing;
* the assembled panel — the KeyError surfaced there, after the run was recorded as
  complete, so the panel is rendered and asserted directly rather than inferred
  from the formatter's own tests.

No test trains. The panel tests call the extracted builder, and the contract test
parses the trainer modules instead of importing them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from soup_cli.commands.train import (
    _format_training_complete_panel,
    _format_training_duration,
)

TRAINER_DIR = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"


def _returns_key(node: ast.AST, key: str) -> bool:
    """True if a dict literal in `node` has `key` as a string key."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Dict):
            continue
        for k in child.keys:
            if isinstance(k, ast.Constant) and k.value == key:
                return True
    return False


def _wrappers_returning(path: Path, key: str) -> list[ast.FunctionDef]:
    """Functions in `path` with a `return` whose dict carries `key`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                if _returns_key(sub.value, key):
                    found[node.name] = node
                    break
    return list(found.values())


# ---------------------------------------------------------------------------
# layer 1: every wrapper that reports duration_secs also reports duration
# ---------------------------------------------------------------------------


def test_unlearn_wrapper_reports_both_duration_forms():
    """The fix itself: `unlearn` returns both keys."""
    wrappers = _wrappers_returning(TRAINER_DIR / "unlearn.py", "duration_secs")
    assert wrappers, "expected unlearn.py to return duration_secs"

    for fn in wrappers:
        returns = [
            sub.value
            for sub in ast.walk(fn)
            if isinstance(sub, ast.Return) and sub.value is not None
        ]
        assert any(_returns_key(value, "duration") for value in returns), (
            f"unlearn.{fn.name}() returns duration_secs without duration, so the "
            "completion panel raises KeyError on a finished run"
        )


def test_every_wrapper_returning_duration_secs_also_returns_duration():
    """The contract: the omission cannot come back in another wrapper."""
    offenders: list[str] = []
    scanned = 0

    for path in sorted(TRAINER_DIR.glob("*.py")):
        with_secs = _wrappers_returning(path, "duration_secs")
        if not with_secs:
            continue
        scanned += 1
        for fn in with_secs:
            returns = [
                sub.value
                for sub in ast.walk(fn)
                if isinstance(sub, ast.Return) and sub.value is not None
            ]
            if not any(_returns_key(value, "duration") for value in returns):
                offenders.append(f"{path.name}::{fn.name}")

    assert scanned >= 10, f"scan only reached {scanned} wrappers; expected the full set"
    assert not offenders, (
        "these wrappers return duration_secs but not duration, so the completion "
        f"panel would raise KeyError on a finished run: {offenders}"
    )


# ---------------------------------------------------------------------------
# layer 2: the formatter degrades instead of crashing
# ---------------------------------------------------------------------------


def test_formatter_uses_the_preformatted_duration_when_present():
    assert (
        _format_training_duration({"duration": "3h 7m", "duration_secs": 12420})
        == "3h 7m"
    )


def test_formatter_falls_back_to_duration_secs_when_the_string_form_is_missing():
    # The #1529 shape: a finished run whose wrapper forgot `duration`.
    assert _format_training_duration({"duration_secs": 7542}) == "2h 5m"


@pytest.mark.parametrize(
    ("duration_secs", "expected"),
    [(0, "0m"), (59, "0m"), (60, "1m"), (3600, "1h 0m"), (7325, "2h 2m")],
)
def test_formatter_fallback_matches_the_wrapper_string_format(duration_secs, expected):
    """The fallback must render exactly what the wrappers put in `duration`.

    Every wrapper builds `f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"`
    from the same seconds value, so a panel that disagreed would print a different
    duration than the run recorded.
    """
    hours = int(duration_secs // 3600)
    minutes = int((duration_secs % 3600) // 60)
    wrapper_string = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    assert expected == wrapper_string
    assert _format_training_duration({"duration_secs": duration_secs}) == wrapper_string


def test_formatter_reports_unknown_instead_of_raising_on_an_empty_result():
    assert _format_training_duration({}) == "unknown"


# ---------------------------------------------------------------------------
# layer 3: the assembled panel, which is where the KeyError actually surfaced
# ---------------------------------------------------------------------------


def _render_panel(result: dict) -> str:
    from rich.console import Console

    panel = _format_training_complete_panel(result, "run-7", "merge hint")
    console = Console(width=120, no_color=True, force_terminal=False)
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


# The shape #1529 produced: a finished unlearn run, whose wrapper returned
# duration_secs and no duration, while the panel indexed result['duration'].
UNLEARN_RESULT = {
    "output_dir": "/tmp/soup-unlearn",
    "duration_secs": 7542,
    "initial_loss": 1.5,
    "final_loss": 0.25,
}


def test_panel_renders_a_duration_for_a_result_without_the_string_form():
    """Regression test for the reported crash: it must not raise."""
    rendered = _render_panel(dict(UNLEARN_RESULT))

    assert "Duration: 2h 5m" in rendered
    assert "/tmp/soup-unlearn" in rendered
    assert "run-7" in rendered


def test_panel_prefers_the_preformatted_duration_when_the_wrapper_supplied_one():
    rendered = _render_panel({**UNLEARN_RESULT, "duration": "9h 41m"})

    assert "Duration: 9h 41m" in rendered


def test_panel_degrades_to_text_rather_than_raising_when_duration_is_absent_entirely():
    """A finished run with no duration information at all still gets a panel.

    `loss_summary_kind` is the real "the wrapper measured no loss" shape, which
    also keeps this test off the loss formatter's own pre-existing key
    requirements -- that is a separate concern from #1529.
    """
    rendered = _render_panel(
        {"output_dir": "/tmp/out", "loss_summary_kind": "unavailable"}
    )

    assert "Duration: unknown" in rendered
    assert "Loss: unavailable" in rendered
    assert "/tmp/out" in rendered
