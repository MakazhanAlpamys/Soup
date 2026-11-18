"""Regression tests for Unicode line separators in soup-serve JSONL records."""

import json

import pytest

from soup_cli.data.traces.parsers import parse_soup_serve


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_separators_inside_json_response_are_preserved(tmp_path, separator):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    expected = f"before{separator}after"
    rows = [
        {"id": "first", "prompt": "Question", "response": expected, "signal": "thumbs_up"},
        {"id": "second", "prompt": "Question", "response": "Other", "signal": "thumbs_down"},
    ]
    (trace_dir / "serve.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    traces = list(parse_soup_serve(str(trace_dir)))

    assert [(trace.trace_id, trace.output, trace.signal) for trace in traces] == [
        ("first", expected, "thumbs_up"),
        ("second", "Other", "thumbs_down"),
    ]


def test_utf8_bom_and_malformed_jsonl_lines_still_work(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    record = {"id": "valid", "prompt": "Question", "response": "Answer"}
    (trace_dir / "serve.jsonl").write_text(
        json.dumps(record) + "\nnot json\n", encoding="utf-8-sig"
    )

    traces = list(parse_soup_serve(str(trace_dir)))

    assert len(traces) == 1
    assert traces[0].trace_id == "valid"
