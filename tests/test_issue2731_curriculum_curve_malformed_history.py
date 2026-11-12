"""#2731: ``soup runs curriculum-curve`` refuses every malformed history the same way.

Weights that do not sum to 1 already got "history malformed" (exit 2). A history
that mixes bucket counts, which two runs with different ``curriculum_buckets``
write into one output directory, a truncated JSON line, and a row that is not an
object all crashed instead (``Error: ValueError`` / ``JSONDecodeError`` /
``AttributeError``, exit 1).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def _row(step, weights):
    return json.dumps({"step": step, "weights": weights})


MIXED_BUCKETS = "\n".join(
    [_row(s, [0.25, 0.25, 0.25, 0.25]) for s in (5, 10, 15, 20)]
    + [_row(s, [0.4, 0.3, 0.3]) for s in (5, 10, 15, 20)]
) + "\n"
TRUNCATED = _row(5, [0.5, 0.5]) + "\n" + '{"step": 10, "weights": [0.5' + "\n"
LIST_ROW = "[0.5, 0.5]\n"


@pytest.fixture()
def curve(tmp_path, monkeypatch):
    import soup_cli.experiment.tracker as et

    monkeypatch.chdir(tmp_path)

    class FakeTracker:
        def get_run(self, run_id):
            return {"run_id": run_id, "output_dir": "."}

    monkeypatch.setattr(et, "ExperimentTracker", FakeTracker)

    def _run(text):
        history = tmp_path / "h.jsonl"
        history.write_text(text, encoding="utf-8")
        return runner.invoke(app, ["runs", "curriculum-curve", "stub", "--history",
                                   str(history)])

    return _run


@pytest.mark.parametrize(
    "text, detail",
    [
        (MIXED_BUCKETS, "num_buckets"),
        (TRUNCATED, "line 2 is not JSON"),
        (LIST_ROW, "Mapping"),
    ],
    ids=["mixed-bucket-counts", "truncated-line", "list-row"],
)
def test_a_malformed_history_is_refused_not_crashed(curve, text, detail):
    result = curve(text)
    out = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, (out, repr(result.exception))
    assert "history malformed" in out
    assert detail in out
    for crash in ("Error: ValueError", "Error: JSONDecodeError", "Error: AttributeError"):
        assert crash not in out


def test_a_well_formed_history_still_renders(curve):
    result = curve("\n".join(_row(s, [0.25, 0.25, 0.25, 0.25]) for s in (5, 10)) + "\n")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "history malformed" not in strip_ansi(result.output)
