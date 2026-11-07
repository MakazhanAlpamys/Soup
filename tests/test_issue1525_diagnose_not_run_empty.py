"""#1525 (part 2) — a probe that measured nothing reads NOT_RUN, not OK 1.00.

``score_memorization`` answered ``OK 1.00 "no rows with text+suffix; nothing to
check"`` for a row list it could not split into a prefix and a suffix (one
character of text, a single token under ``--tokenizer``, all-empty ``messages``,
or an empty list). ``score_format`` and ``score_mode_collapse`` did the same for
an empty prompt list. A score of 1.00 with an OK verdict is indistinguishable
from a clean measurement, so a gate reading it cannot tell a checked run from an
unchecked one.
"""

from __future__ import annotations

import json

import soup_cli.utils.diagnose.live as live
from soup_cli.utils.diagnose.format import score_format
from soup_cli.utils.diagnose.memorization import score_memorization
from soup_cli.utils.diagnose.mode_collapse import score_mode_collapse


def _noop_generator(_prompt: str) -> str:
    return "irrelevant completion"


def test_empty_prompt_list_is_not_run_for_format() -> None:
    score = score_format([], _noop_generator, kind="json")

    assert score.mode == "format"
    assert score.verdict == "NOT_RUN"
    assert score.score == 0.0
    assert "no prompts" in score.evidence


def test_empty_prompt_list_is_not_run_for_mode_collapse() -> None:
    score = score_mode_collapse([], lambda _prompt, k: ["x", "y"][:k], k=2)

    assert score.mode == "mode_collapse"
    assert score.verdict == "NOT_RUN"
    assert score.score == 0.0
    assert "no prompts" in score.evidence


def test_one_character_text_cannot_split_is_not_run() -> None:
    score = score_memorization([{"text": "x"}], _noop_generator)

    assert score.mode == "memorization"
    assert score.verdict == "NOT_RUN"
    assert score.score == 0.0
    assert "prefix and a suffix" in score.evidence


def test_empty_row_list_is_not_run_for_memorization() -> None:
    score = score_memorization([], _noop_generator)

    assert score.verdict == "NOT_RUN"
    assert "prefix and a suffix" in score.evidence


def test_all_empty_messages_cannot_split_is_not_run() -> None:
    rows = [{"messages": [{"role": "user", "content": ""}]} for _ in range(3)]

    score = score_memorization(rows, _noop_generator)

    assert score.verdict == "NOT_RUN"


def test_a_split_table_row_still_scores_normally() -> None:
    """Control: the NOT_RUN guard must not swallow a measurable row."""
    rows = [{"text": "alpha beta gamma delta epsilon"}]

    score = score_memorization(rows, lambda _prompt: "unrelated text here")

    assert score.verdict == "OK"
    assert score.score == 1.0


def test_live_path_reports_short_text_rows_as_not_run(tmp_path, monkeypatch) -> None:
    """The same rows through the live runner: NOT_RUN, and the whole report with it."""
    rows = [{"prompt": "Q", "text": "x"} for _ in range(12)]
    (tmp_path / "d.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    gens = {
        "base_gen": lambda _prompt: "ok",
        "adapter_gen": lambda _prompt: "ok",
        "base_multi": lambda _prompt, k: [f"b{i}" for i in range(k)],
        "adapter_multi": lambda _prompt, k: [f"a{i}" for i in range(k)],
    }
    monkeypatch.setattr(live, "load_adapter_pair", lambda *a, **kw: gens)

    report = live.run_live_diagnose(
        run_id="r", base="b", adapter="a", dataset_path="d.jsonl"
    )

    score = report.scores["memorization"]
    assert score.verdict == "NOT_RUN"
    assert "prefix and a suffix" in score.evidence
    assert report.overall == "NOT_RUN"
