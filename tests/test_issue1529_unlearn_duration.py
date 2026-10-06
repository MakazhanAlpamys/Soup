"""#1529 — unlearn's trainer result must carry the panel's 'duration' key.

``soup train --task unlearn`` crashed with ``KeyError: 'duration'`` in the
"Training Complete!" panel — AFTER the adapter was saved and the tracker row
was written — because ``UnlearnTrainerWrapper`` was the one transformers-backend
wrapper that returned only ``duration_secs``. These tests pin both halves of
the fix: the wrapper now formats the string like every other wrapper, and the
panel tolerates a result dict that lacks the key anyway.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from soup_cli.commands.train import _format_duration_display
from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

TRAINER_DIR = Path(__file__).parent.parent / "src" / "soup_cli" / "trainer"


def _run_ready_unlearn_wrapper(tmp_path, monkeypatch) -> UnlearnTrainerWrapper:
    """A wrapper past setup() with an empty forget set (a zero-step run).

    setup() itself refuses an empty forget set, so this bypasses it and
    arranges exactly what a real call leaves behind: ``_setup_called`` set,
    mock model/tokenizer, and an output dir under the (tmp) cwd that the
    ``_validated_output_dir`` gate accepts. ``train()`` then clears the
    optimizer's gradients, finds no forget row to step over, and exercises
    the save + result path.
    """
    monkeypatch.chdir(tmp_path)
    cfg = MagicMock()
    cfg.training.epochs = 1
    cfg.output = "unlearn-out"
    wrapper = UnlearnTrainerWrapper(cfg)
    wrapper._setup_called = True
    wrapper.model = MagicMock()
    wrapper.tokenizer = MagicMock()
    wrapper._optimizer = MagicMock()  # #1564: train() touches it even at zero steps
    return wrapper


@pytest.mark.parametrize(("elapsed", "expected"), [(90, "1m"), (3720, "1h 2m")])
def test_unlearn_train_returns_formatted_duration(
    tmp_path, monkeypatch, elapsed, expected
):
    """train() returns the pre-formatted 'duration' alongside duration_secs."""
    wrapper = _run_ready_unlearn_wrapper(tmp_path, monkeypatch)
    ticks = iter([0, elapsed])

    with mock_patch(
        "soup_cli.trainer.unlearn.time.monotonic", side_effect=lambda: next(ticks)
    ):
        result = wrapper.train()

    assert result["duration"] == expected
    assert result["duration_secs"] == elapsed


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"duration": "3m", "duration_secs": 185.0}, "3m"),
        # The #1529 shape: seconds only, no pre-formatted string.
        ({"duration_secs": 90.0}, "1m"),
        ({"duration_secs": 3720.0}, "1h 2m"),
        # A measured zero stays 0m; an absent measurement is not a zero.
        ({"duration_secs": 0.0}, "0m"),
        # Neither key: a finished run still gets its summary panel.
        ({}, "unknown"),
    ],
)
def test_completion_panel_duration_falls_back_to_seconds(result, expected):
    """The panel reads 'duration' when present, else formats duration_secs."""
    assert _format_duration_display(result) == expected


def test_completion_panel_never_indexes_result_duration_directly():
    """#1529 surfaced in the completion panel, after finish_run() had already run.

    The helper tests above never reach that call site, so reverting the panel
    line to ``result['duration']`` would leave them green. Pin the site itself.
    """
    import ast

    train_py = TRAINER_DIR.parent / "commands" / "train.py"
    tree = ast.parse(train_py.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "result"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "duration"
    ]
    assert not offenders, (
        f"commands/train.py indexes result['duration'] at line(s) {offenders}; "
        "go through _format_duration_display(result) so a wrapper that omits the key "
        "cannot crash a finished run"
    )


def test_every_trainer_reporting_duration_secs_also_formats_duration():
    """Guard the whole wrapper family, not just unlearn (#1529).

    Any trainer module that reports ``duration_secs`` but never emits the
    ``"duration"`` string key crashes the train completion panel after the
    run is otherwise done. unlearn.py was the original offender; this scan
    catches the next wrapper that repeats the mistake.
    """
    offenders = []
    for path in sorted(TRAINER_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "duration_secs" not in src:
            continue
        if '"duration"' not in src and "'duration'" not in src:
            offenders.append(path.name)
    assert not offenders, f"trainer module(s) missing the duration key: {offenders}"
