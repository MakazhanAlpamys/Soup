"""chatml / audio / video converters must drop a non-dict message (#676).

``format_to_messages`` documents a drop contract: a malformed row returns
``None`` so one bad JSONL line is skipped rather than corrupting the
dataset. ``_convert_chatml``, ``_convert_audio`` and ``_convert_video``
passed a non-dict ``messages`` element through verbatim, so a row like
``{"messages": ["hello"]}`` survived into training.

The assertions are ``is None``, not merely "did not raise": replacing the
guard with ``continue`` (keep the row, skip the element) must fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import format_to_messages
from soup_cli.data.loader import load_dataset
from soup_cli.data.validator import validate_and_stats

from .conftest import strip_ansi

NON_DICT_MESSAGES = ("hello", 42, ["role", "user"], None)

VALID_CHATML = {
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
}
VALID_AUDIO = {
    "audio": "clip.wav",
    "messages": [
        {"role": "user", "content": "transcribe"},
        {"role": "assistant", "content": "hello"},
    ],
}
VALID_VIDEO = {
    "video": "clip.mp4",
    "messages": [{"role": "user", "content": "describe"}],
}


def _row_for(fmt: str, msg: object) -> dict:
    if fmt == "chatml":
        return {"messages": [msg]}
    if fmt == "audio":
        return {"audio": "clip.wav", "messages": [msg]}
    return {"video": "clip.mp4", "messages": [msg]}


@pytest.mark.parametrize("fmt", ["chatml", "audio", "video"])
@pytest.mark.parametrize("msg", list(NON_DICT_MESSAGES))
def test_non_dict_message_is_dropped(fmt: str, msg: object) -> None:
    # Must be None, not a kept row. ``continue`` over the bad element would
    # return {"messages": []} (or the original list) and fail this.
    assert format_to_messages(_row_for(fmt, msg), fmt) is None


def test_valid_chatml_audio_video_rows_convert_unchanged() -> None:
    assert format_to_messages(VALID_CHATML, "chatml") == {
        "messages": VALID_CHATML["messages"]
    }
    assert format_to_messages(VALID_AUDIO, "audio") == {
        "messages": VALID_AUDIO["messages"],
        "audio": "clip.wav",
    }
    assert format_to_messages(VALID_VIDEO, "video") == {
        "video": "clip.mp4",
        "messages": VALID_VIDEO["messages"],
    }


def test_validate_and_load_dataset_agree_on_chatml_file(tmp_path: Path) -> None:
    rows = [
        VALID_CHATML,
        {"messages": ["hello"]},
        {"messages": [{"role": "user", "content": "ok"}]},
        {"messages": [42]},
        {"messages": [None]},
    ]
    path = tmp_path / "chatml.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    stats = validate_and_stats(rows, expected_format="chatml")
    loaded = load_dataset(
        DataConfig(train=str(path), format="chatml", val_split=0.0)
    )
    assert stats["valid_rows"] == 2
    assert len(loaded["train"]) == stats["valid_rows"]


def test_validate_does_not_greenlight_multimodal_rows_the_loader_rejects() -> None:
    # ``soup data validate --format multimodal`` used to report every row
    # valid because it only checked keys, while load_dataset raised
    # AttributeError on a non-dict message. Validator must not green-light
    # that file. load_dataset still raises until the neighbouring multimodal
    # converter guard lands; that mismatch is deliberate.
    rows = [{"messages": [msg]} for msg in NON_DICT_MESSAGES]
    rows.append({"messages": [{"role": "user", "content": "ok"}]})
    stats = validate_and_stats(rows, expected_format="multimodal")
    assert stats["valid_rows"] == 1
    assert stats["valid_rows"] < stats["total"]

    with pytest.raises(AttributeError):
        format_to_messages({"messages": ["hello"]}, "multimodal")


def test_soup_data_validate_reports_dropped_chatml_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    rows = [VALID_CHATML, {"messages": ["hello"]}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["data", "validate", str(path), "--format", "chatml"]
    )
    assert result.exit_code == 0, result.output
    assert "1/2 rows valid" in strip_ansi(result.output)
