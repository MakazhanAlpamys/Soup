"""Identical latest thumbs must not become training preferences."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.utils import local_rl


def _record(rows: list[tuple[str, str, str]]) -> None:
    local_rl.init_local_rl_db("thumbs.db")
    # Explicit timestamps exercise chronological selection on an existing database.
    with closing(sqlite3.connect("thumbs.db")) as connection, connection:
        connection.executemany(
            "INSERT INTO thumbs (ts, prompt, response, thumb) VALUES (?, ?, ?, ?)",
            [(index, *row) for index, row in enumerate(rows)],
        )


@pytest.mark.parametrize("consumer", ["sdk", "cli", "loop"])
@pytest.mark.parametrize(("rows", "expected"), [
    pytest.param(
        [("Q", "Same", "up"), ("Q", "Same", "down")], [], id="identical-only",
    ),
    pytest.param(
        [("Q", "Older good", "up"), ("Q", "Same", "up"), ("Q", "Same", "down")],
        [], id="latest-up-conflicts",
    ),
    pytest.param(
        [("Q", "Older bad", "down"), ("Q", "Same", "down"), ("Q", "Same", "up")],
        [], id="latest-down-conflicts",
    ),
    pytest.param(
        [("Q", "Same", "up"), ("Q", "Same", "down"),
         ("B", "Good B", "up"), ("B", "Bad B", "down"),
         ("A", "Good A", "up"), ("A", "Bad A", "down"),
         ("Unpaired", "Only up", "up")],
        [("A", "Good A", "Bad A"), ("B", "Good B", "Bad B")], id="mixed-prompts",
    ),
    pytest.param(
        [("Q", "Older good", "up"), ("Q", "Older bad", "down"),
         ("Q", "Newest bad", "down"), ("Q", "Newest good", "up")],
        [("Q", "Newest good", "Newest bad")], id="latest-distinct-control",
    ),
])
def test_latest_thumbs_define_a_distinct_preference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str,
    rows: list[tuple[str, str, str]], expected: list[tuple[str, str, str]],
) -> None:
    monkeypatch.chdir(tmp_path)
    _record(rows)
    expected_rows = [
        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        for prompt, chosen, rejected in expected
    ]

    if consumer == "sdk":
        actual = [dataclasses.asdict(pair) for pair in local_rl.harvest_dpo_pairs("thumbs.db")]
    elif consumer == "cli":
        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "local-rl", "harvest", "--db", "thumbs.db", "--output", "pairs.jsonl",
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        actual = [json.loads(line) for line in Path("pairs.jsonl").read_text().splitlines()]
    else:
        from soup_cli.utils.loop_stages import harvest_from_traces
        from soup_cli.utils.loop_state import LoopState

        monkeypatch.delenv("SOUP_LOOP_THUMBS_DB", raising=False)
        monkeypatch.delenv("SOUP_LOOP_TRACE_DIR", raising=False)
        state = LoopState(
            served_model="registry://demo", eval_suite="eval", baseline="base",
            thumbs_db="thumbs.db",
        )
        result = harvest_from_traces(state)
        assert result["thumbs_pairs"] == result["pairs_harvested"] == len(expected)
        if not expected:
            assert result["pairs_path"] is None
            actual = []
        else:
            actual = [
                json.loads(line) for line in Path(result["pairs_path"]).read_text().splitlines()
            ]
            expected_rows = [dict(row, source="soup-serve-thumbs") for row in expected_rows]
    assert actual == expected_rows


def test_nightly_training_counts_only_distinct_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _record([("Q", "Same", "up"), ("Q", "Same", "down")])
    calls = []
    config = local_rl.LocalRLConfig(
        backend="ollama", model="org/model", db_path="thumbs.db", train_method="dpo",
    )
    result = local_rl.run_nightly_train(
        config, min_pairs=1, train_fn=lambda **kwargs: calls.append(kwargs),
    )
    assert result.status == "skipped_insufficient_pairs"
    assert result.num_pairs == 0
    assert calls == []
