"""Regression tests for issue #1424: ``format: auto`` read only the first row.

``detect_format`` used ``sample = data[0]`` and returned the first format whose
required keys were a subset of that single row. A first row missing an optional
key therefore picked the poorer converter for the WHOLE file and silently
dropped that key from every row that had it — a 4-row LLaVA file whose first
row is text-only loaded with 0 of 4 images kept, and a ``messages``/``tools``
file whose first row had no ``tools`` was converted as ``chatml`` and lost the
tool schema on every row.

Detection now reads a bounded prefix (``MAX_DETECT_ROWS``). A uniform file
detects exactly as before (controls below). A mixed file upgrades to the richer
format when its converter can also read the poorer rows (tool-calling over
chatml), and otherwise raises a message naming both shapes, the row each was
first seen at, and the ``data.format`` escape hatch — rather than silently
converting everything with the poorer format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import (
    FORMAT_SIGNATURES,
    MAX_DETECT_ROWS,
    detect_format,
    format_to_messages,
)
from soup_cli.data.loader import load_dataset

_CHAT = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
_SHAREGPT = [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]
_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

# The poorer row of each richer/poorer family, and the richer row that shares
# the family (same signature plus one key).
_SHAREGPT_ROW = {"conversations": _SHAREGPT}
_LLAVA_ROW = {"image": "i.png", "conversations": _SHAREGPT}
_CHATML_ROW = {"messages": _CHAT}
_TOOL_ROW = {"messages": _CHAT, "tools": [_TOOL]}
_AUDIO_ROW = {"audio": "a.wav", "messages": _CHAT}
_PLAINTEXT_ROW = {"text": "raw text"}
_ASR_ROW = {"audio": "a.wav", "text": "transcript"}

# One valid row per format `detect_format` can return; the expected detected
# name is listed separately because sharegpt4v and llava share a signature
# (llava is checked first, so detection never returns "sharegpt4v").
_UNIFORM_ROWS = {
    "alpaca": {"instruction": "q", "output": "a"},
    "sharegpt": _SHAREGPT_ROW,
    "chatml": _CHATML_ROW,
    "dpo": {"prompt": "q", "chosen": "a", "rejected": "b"},
    "kto": {"prompt": "q", "completion": "a", "label": True},
    "llava": _LLAVA_ROW,
    "sharegpt4v": _LLAVA_ROW,  # same signature as llava; llava wins the check order
    "embedding": {"anchor": "q", "positive": "a"},
    "tool-calling": _TOOL_ROW,
    "audio": _AUDIO_ROW,
    "asr": _ASR_ROW,
    "plaintext": _PLAINTEXT_ROW,
}
_UNIFORM_EXPECTED = {fmt: fmt for fmt in _UNIFORM_ROWS}
_UNIFORM_EXPECTED["sharegpt4v"] = "llava"


def _load(path: Path, data_format: str = "auto") -> dict:
    return load_dataset(DataConfig(train=str(path), format=data_format, val_split=0.0))


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# Controls — every uniform format detects exactly as before (#1424 step 3)
# ---------------------------------------------------------------------------


def test_fixture_covers_every_format_signature():
    """A new signature must join the uniform control sweep."""
    assert set(_UNIFORM_ROWS) == set(FORMAT_SIGNATURES)


@pytest.mark.parametrize("fmt", sorted(_UNIFORM_ROWS))
def test_uniform_file_detects_exactly_as_before(fmt):
    # A whole file of one shape keeps detecting to that shape (sharegpt4v's
    # rows detect as llava, today's behaviour, pinned here as a control).
    rows = [_UNIFORM_ROWS[fmt]] * 4
    assert detect_format(rows) == _UNIFORM_EXPECTED[fmt]


# ---------------------------------------------------------------------------
# The four mixed cases from the issue
# ---------------------------------------------------------------------------


def test_mixed_sharegpt_and_llava_refuses_naming_both_shapes():
    # Row 0 has no `image`; converting the file as sharegpt would drop the
    # image from row 1. llava's converter refuses a row without an image, so
    # the fix must raise rather than silently return 'sharegpt'.
    data = [_SHAREGPT_ROW, _LLAVA_ROW]
    with pytest.raises(ValueError) as exc:
        detect_format(data)
    message = str(exc.value)
    assert "'sharegpt' (row 0)" in message
    assert "'llava' (row 1)" in message
    assert "data.format" in message


def test_mixed_messages_and_tools_upgrades_to_tool_calling():
    # The chatml row converts fine as tool-calling (tools absent -> no schema
    # turn), so the richer converter covers the whole file.
    assert detect_format([_CHATML_ROW, _TOOL_ROW]) == "tool-calling"
    # Order must not matter: the poorer row first, or the richer row first.
    assert detect_format([_TOOL_ROW, _CHATML_ROW]) == "tool-calling"


def test_mixed_chatml_and_audio_refuses_naming_both_shapes():
    data = [_CHATML_ROW, _AUDIO_ROW]
    with pytest.raises(ValueError) as exc:
        detect_format(data)
    message = str(exc.value)
    assert "'chatml' (row 0)" in message
    assert "'audio' (row 1)" in message
    assert "data.format" in message


def test_mixed_plaintext_and_asr_refuses_naming_both_shapes():
    data = [_PLAINTEXT_ROW, _ASR_ROW]
    with pytest.raises(ValueError) as exc:
        detect_format(data)
    message = str(exc.value)
    assert "'plaintext' (row 0)" in message
    assert "'asr' (row 1)" in message
    assert "data.format" in message


def test_unrelated_shapes_refuse_naming_both_rows():
    # alpaca and chatml share no richer/poorer relation, so neither may win.
    data = [{"instruction": "q", "output": "a"}, _CHATML_ROW]
    with pytest.raises(ValueError) as exc:
        detect_format(data)
    message = str(exc.value)
    assert "'alpaca' (row 0)" in message
    assert "'chatml' (row 1)" in message
    assert "data.format" in message


def test_upgrade_keeps_the_tools_schema_on_the_tools_row():
    # The whole point of upgrading: the tool schema survives instead of being
    # dropped by the chatml converter.
    rows = [_CHATML_ROW, _TOOL_ROW]
    assert detect_format(rows) == "tool-calling"
    converted = [format_to_messages(row, "tool-calling") for row in rows]
    assert all(result is not None for result in converted)
    schema_blob = json.dumps(converted[1])
    assert "get_weather" in schema_blob
    # The chatml-only row still converts (as a plain conversation turn).
    assert converted[0]["messages"] == _CHAT


# ---------------------------------------------------------------------------
# Bounded prefix + unrecognised keys
# ---------------------------------------------------------------------------


def test_detection_reads_a_bounded_prefix_not_the_whole_file():
    # A richer row sitting past the cap cannot flip the file: the prefix of
    # chatml rows decides, exactly as the pre-fix single-row read did for row 0.
    data = [_CHATML_ROW] * MAX_DETECT_ROWS + [_LLAVA_ROW]
    assert detect_format(data) == "chatml"
    # One row inside the cap is enough to trigger mixed detection.
    inside = [_CHATML_ROW] * (MAX_DETECT_ROWS - 1) + [_LLAVA_ROW]
    with pytest.raises(ValueError):
        detect_format(inside)


def test_unrecognised_row_is_skipped_not_fatal():
    # #1217: a row matching no signature is skipped and reported, so one bad row
    # must not stop the load. The prefix scan resolves from the rows that do
    # match, and only raises when NOTHING in the prefix matches.
    data = [_CHATML_ROW, {"unrelated": "keys"}]
    assert detect_format(data) == "chatml"


def test_detection_raises_when_no_row_in_the_prefix_matches():
    with pytest.raises(ValueError) as exc:
        detect_format([{"unrelated": "keys"}, {"also": "unrelated"}])
    assert "Cannot detect format at row 0" in str(exc.value)


def test_non_dict_row_in_the_prefix_is_skipped_not_fatal():
    # A JSONL line may be any JSON value. The old code read only row 0, so a
    # bare scalar later in the file never reached .keys(); scanning the prefix
    # must not turn that into an AttributeError that the loader (which catches
    # only ValueError) would surface as a traceback.
    for bad in ("a bare json string", 1, 1.5, None, ["a", "list"], True):
        assert detect_format([_CHATML_ROW, bad]) == "chatml"
        assert detect_format([bad, _CHATML_ROW]) == "chatml"


def test_all_non_dict_rows_raise_valueerror_not_attributeerror():
    for data in ([1, "x"], ["a string"], [[1, 2], [3, 4]], [None]):
        with pytest.raises(ValueError) as exc:
            detect_format(data)
        assert "not an object" in str(exc.value)


def test_empty_dataset_still_raises():
    with pytest.raises(ValueError, match="Empty"):
        detect_format([])


# ---------------------------------------------------------------------------
# End-to-end: the acceptance criteria through load_dataset
# ---------------------------------------------------------------------------


def test_mixed_llava_file_refuses_or_keeps_every_image(tmp_path):
    # The issue's repro: a 4-row LLaVA file whose first row is text-only used
    # to load with 0 of 4 images kept. Now it refuses.
    path = tmp_path / "mixed_llava.jsonl"
    rows = [_SHAREGPT_ROW] + [{"image": f"i{i}.png", "conversations": _SHAREGPT} for i in range(3)]
    _write(path, rows)
    with pytest.raises(ValueError, match="llava"):
        _load(path)


def test_mixed_tools_file_loads_every_row_with_the_schema(tmp_path):
    path = tmp_path / "mixed_tools.jsonl"
    _write(path, [_CHATML_ROW, _TOOL_ROW])
    loaded = _load(path)["train"]
    assert len(loaded) == 2
    # The tools row keeps its schema turn; the plain row stays a plain chat.
    assert "get_weather" in json.dumps(loaded)
    assert loaded[0]["messages"] == _CHAT


def test_mixed_audio_file_refuses(tmp_path):
    path = tmp_path / "mixed_audio.jsonl"
    _write(path, [_CHATML_ROW, _AUDIO_ROW])
    with pytest.raises(ValueError, match="audio"):
        _load(path)


def test_mixed_asr_file_refuses(tmp_path):
    path = tmp_path / "mixed_asr.jsonl"
    _write(path, [_PLAINTEXT_ROW, _ASR_ROW])
    with pytest.raises(ValueError, match="asr"):
        _load(path)


# ---------------------------------------------------------------------------
# Pinned by review: behaviours the refusal message promises
# ---------------------------------------------------------------------------


def test_upgrade_needs_the_richer_converter_to_accept_every_poorer_row():
    # Row 0 converts fine as tool-calling, row 1 does not (list content is refused
    # by the tool-calling converter). One accepted row must not be enough to upgrade.
    multipart = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": "a"},
    ]
    data = [
        {"messages": _CHAT},
        {"messages": multipart},
        {"messages": _CHAT, "tools": [_TOOL]},
    ]
    with pytest.raises(ValueError) as exc:
        detect_format(data)
    message = str(exc.value)
    assert "'chatml' (row 0)" in message
    assert "'tool-calling' (row 2)" in message
    assert "data.format" in message


def test_refusal_names_the_first_row_of_each_shape():
    sharegpt = {"conversations": _SHAREGPT}
    llava = {"image": "i.png", "conversations": _SHAREGPT}
    with pytest.raises(ValueError) as exc:
        detect_format([sharegpt, sharegpt, sharegpt, llava, llava, llava])
    message = str(exc.value)
    assert "'sharegpt' (row 0)" in message
    assert "'llava' (row 3)" in message


def test_inspect_endpoint_reports_a_mixed_file_instead_of_a_500(tmp_path):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("FastAPI not installed")
    from unittest.mock import patch

    from soup_cli.ui.app import create_app, get_auth_token

    path = tmp_path / "mixed.jsonl"
    rows = [{"conversations": _SHAREGPT}] + [{"image": "i.png", "conversations": _SHAREGPT}] * 3
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    client = TestClient(create_app(), raise_server_exceptions=False)
    with patch("soup_cli.ui.app.Path.cwd", return_value=tmp_path):
        response = client.post(
            "/api/data/inspect",
            json={"path": str(path), "limit": 10},
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 4
    assert body["format"] == "unknown"
    assert "mixes" in body["format_error"]


def test_upgrade_refuses_when_the_richer_converter_would_drop_a_field():
    # A chatml row whose messages carry `name`/`weight` converts through the
    # chatml converter with those fields and through the tool-calling converter
    # without them, so the "upgrade" would silently drop them. The converter
    # accepts the row rather than rejecting it, so acceptance alone is not enough.
    extras = {
        "messages": [
            {"role": "user", "content": "q", "name": "alice", "weight": 0.5},
            {"role": "assistant", "content": "a"},
        ]
    }
    data = [{"messages": _CHAT}, extras, _TOOL_ROW]
    with pytest.raises(ValueError) as exc:
        detect_format(data)
    message = str(exc.value)
    assert "'chatml' (row 0)" in message
    assert "'tool-calling' (row 2)" in message
    assert "data.format" in message


def test_upgrade_still_happens_when_the_converters_agree():
    # The plain case must still upgrade: no field is lost.
    assert detect_format([{"messages": _CHAT}, _TOOL_ROW]) == "tool-calling"
