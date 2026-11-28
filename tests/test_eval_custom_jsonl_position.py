"""#3326: a JSONL syntax error in a custom eval file names one physical position.

The loader strips each row before decoding it, so the decoder's own position is
row-local and shifted left by the row's indentation. The message carries the
physical line and the column in the original line, plus the decoder's reason.
"""

from pathlib import Path

import pytest

from soup_cli.eval.custom import load_eval_tasks


def _load(tmp_path: Path, content: bytes):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_bytes(content)
    return load_eval_tasks(tasks)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        # The issue's fixture: two blank lines, then an indented row.
        (
            b'{"prompt":"valid","expected":"a"}\n\n\n  {"prompt": }\n',
            "Invalid JSON on line 4, column 14: Expecting value",
        ),
        (
            b'{"prompt":"valid","expected":"a"}\r\n  {"prompt": }\r\n',
            "Invalid JSON on line 2, column 14: Expecting value",
        ),
        (b'\t{"prompt": }\n', "Invalid JSON on line 1, column 13: Expecting value"),
        # An incomplete row: the error sits right after its last physical
        # character, trailing whitespace included (the decoder saw the end of
        # the stripped row).
        (
            b'{"prompt": "x"   \n',
            "Invalid JSON on line 1, column 18: Expecting ',' delimiter",
        ),
        (
            b'  {"prompt": \t  \r\n',
            "Invalid JSON on line 1, column 17: Expecting value",
        ),
    ],
)
def test_a_syntax_error_names_one_physical_position(tmp_path, content, message):
    with pytest.raises(ValueError) as excinfo:
        _load(tmp_path, content)
    assert str(excinfo.value) == message


def test_a_valid_file_with_blank_lines_and_indentation_still_loads(tmp_path):
    tasks = _load(
        tmp_path,
        b'{"prompt":"a","expected":"1"}\n\n  {"prompt":"b","expected":"2"}\r\n',
    )
    assert [task.prompt for task in tasks] == ["a", "b"]
