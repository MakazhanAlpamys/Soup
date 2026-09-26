"""#1218: a local ``.parquet`` dataset loads the same rows as the same data in JSONL.

pandas turned every list cell into a ``numpy.ndarray``, so each converter that checks
``isinstance(..., list)`` dropped every row, and ``soup data validate`` / ``inspect``
crashed on the array's truth value. Each test writes identical rows to JSONL and to
parquet (with pandas, as users do) and requires the two loads to agree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import format_to_messages
from soup_cli.data.loader import load_dataset, load_raw_data
from tests.conftest import strip_ansi

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

_ROWS = 10


def _messages(i: int) -> list[dict]:
    return [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": f"a{i}"},
    ]


def _alpaca(count: int) -> list[dict]:
    return [
        {"instruction": f"i{n}", "input": "", "output": f"o{n}"}
        for n in range(count)
    ]

_BUILDERS = {
    "chatml": lambda i: {"messages": _messages(i)},
    "sharegpt": lambda i: {
        "conversations": [
            {"from": "human", "value": f"q{i}"},
            {"from": "gpt", "value": f"a{i}"},
        ]
    },
    "tool-calling": lambda i: {
        "messages": [{"role": "user", "content": f"weather in c{i}?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": f"c{i}"}),
                }
            }
        ],
    },
    "audio": lambda i: {"audio": f"clip{i}.wav", "messages": _messages(i)},
    "video": lambda i: {"video": f"clip{i}.mp4", "messages": _messages(i)},
    "multimodal": lambda i: {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"q{i}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]},
        ]
    },
    "prm": lambda i: {
        "prompt": f"p{i}",
        "completions": [f"step{i}a", f"step{i}b"],
        "labels": [True, False],
    },
    "input_output": lambda i: {
        "segments": [
            {"text": f"q{i}", "label": False},
            {"text": f"a{i}", "label": True},
        ]
    },
    "raft": lambda i: {
        "query": f"q{i}",
        "golden_doc": f"golden{i}",
        "distractor_docs": [] if i == 0 else [f"d{i}a", f"d{i}b"],
        "answer": f"a{i}",
    },
    # Conversational DPO: trl only treats the row as conversational when these are lists.
    "dpo": lambda i: {
        "prompt": [{"role": "user", "content": f"q{i}"}],
        "chosen": [{"role": "assistant", "content": f"good{i}"}],
        "rejected": [{"role": "assistant", "content": f"bad{i}"}],
    },
}


def _write_both(tmp_path: Path, name: str, rows: list[dict]) -> tuple[Path, Path]:
    jsonl = tmp_path / f"{name}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    parquet = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    return jsonl, parquet


def _ndarray_paths(value: object, path: str = "row") -> list[str]:
    """Every location under ``value`` that holds a numpy array."""
    if isinstance(value, np.ndarray):
        return [path]
    if isinstance(value, dict):
        return [
            p
            for k, v in value.items()
            for p in _ndarray_paths(v, f"{path}[{k!r}]")
        ]
    if isinstance(value, (list, tuple)):
        return [
            p
            for i, v in enumerate(value)
            for p in _ndarray_paths(v, f"{path}[{i}]")
        ]
    return []


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text))


@pytest.mark.parametrize("fmt", sorted(_BUILDERS))
def test_parquet_rows_match_jsonl_rows(tmp_path: Path, fmt: str) -> None:
    rows = [_BUILDERS[fmt](i) for i in range(_ROWS)]
    jsonl, parquet = _write_both(tmp_path, fmt, rows)

    from_parquet = load_raw_data(parquet)

    assert _ndarray_paths(from_parquet) == []
    assert from_parquet == load_raw_data(jsonl) == rows

    converted = [format_to_messages(row, fmt) for row in from_parquet]
    assert None not in converted, f"{fmt}: a parquet row was dropped"
    assert converted == [format_to_messages(row, fmt) for row in rows]


def test_nested_shapes_come_back_as_python_lists(tmp_path: Path) -> None:
    tool = load_raw_data(
        _write_both(
            tmp_path,
            "tool",
            [_BUILDERS["tool-calling"](0)],
        )[1]
    )[0]
    assert tool["tools"][0]["function"]["parameters"]["required"] == ["city"]
    assert type(tool["tools"][0]["function"]["parameters"]["required"]) is list

    prm = load_raw_data(
        _write_both(
            tmp_path,
            "prm",
            [_BUILDERS["prm"](0)],
        )[1]
    )[0]
    assert prm["labels"] == [True, False]
    assert all(type(label) is bool for label in prm["labels"])

    raft = load_raw_data(
        _write_both(
            tmp_path,
            "raft",
            [_BUILDERS["raft"](0)],
        )[1]
    )[0]
    assert raft["distractor_docs"] == []
    assert type(raft["distractor_docs"]) is list


def test_load_dataset_keeps_every_parquet_chatml_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_both(
        tmp_path,
        "chat",
        [_BUILDERS["chatml"](i) for i in range(_ROWS)],
    )

    for name in ("chat.jsonl", "chat.parquet"):
        loaded = load_dataset(DataConfig(train=name, val_split=0.0))
        assert len(loaded["train"]) == _ROWS, name


def test_data_validate_and_inspect_agree_on_parquet(tmp_path: Path) -> None:
    jsonl, parquet = _write_both(
        tmp_path,
        "chat",
        [_BUILDERS["chatml"](i) for i in range(_ROWS)],
    )
    runner = CliRunner()

    for path in (jsonl, parquet):
        result = runner.invoke(app, ["data", "validate", str(path)])
        assert result.exit_code == 0, (
            path.name,
            result.output,
            repr(result.exception),
        )
        assert "10/10 rows valid for chatml format" in _plain(result.output), path.name

        result = runner.invoke(app, ["data", "inspect", str(path)])
        assert result.exit_code == 0, (
            path.name,
            result.output,
            repr(result.exception),
        )
        assert re.search(r"Total samples\W*10\b", _plain(result.output)), path.name


def test_hidden_pandas_index_is_not_a_column(tmp_path: Path) -> None:
    frame = pd.DataFrame(_alpaca(6)).sample(frac=1, random_state=0)
    path = tmp_path / "shuffled.parquet"
    frame.to_parquet(path)

    rows = load_raw_data(path)

    assert [sorted(row) for row in rows] == [
        ["input", "instruction", "output"]
    ] * 6
    assert rows == frame.to_dict(orient="records")


def test_named_index_stays_a_column(tmp_path: Path) -> None:
    path = tmp_path / "named.parquet"
    pd.DataFrame(_alpaca(3)).set_index("instruction").to_parquet(path)

    assert [row["instruction"] for row in load_raw_data(path)] == [
        "i0",
        "i1",
        "i2",
    ]
