"""Tests for Issue #1434: ``soup expect`` fails closed on unreadable JSONL input.

A UTF-8 BOM must be stripped like the training loader does, malformed JSON
lines and non-object rows must be reported (not silently dropped), and a run
that checked zero rows must not pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

PII_LINE = b'{"text": "mail jane.doe@example.com"}\n'
SUITE_YAML = "expectations:\n  - name: expect_no_pii\n"

FILES = {
    "bom.jsonl": b"\xef\xbb\xbf" + PII_LINE + b'{"text": "hi"}\n',
    "nobom.jsonl": PII_LINE + b'{"text": "hi"}\n',
    "array_per_line.jsonl": b'[{"text": "mail jane.doe@example.com"}]\n',
    "non_object_rows.jsonl": b'"mail jane.doe@example.com"\n42\nnull\n',
    "all_malformed.jsonl": b"{not json\n{still: not}\n",
    "empty.jsonl": b"",
}


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, content: bytes):
    (tmp_path / "suite.yaml").write_text(SUITE_YAML, encoding="utf-8")
    (tmp_path / name).write_bytes(content)
    monkeypatch.chdir(tmp_path)
    return CliRunner().invoke(app, ["expect", name, "suite.yaml"])


def test_bom_file_matches_no_bom_control(tmp_path, monkeypatch):
    """The BOM-prefixed file reports the PII in row 1 and exits 2, like the control."""
    with_bom = _run(tmp_path, monkeypatch, "bom.jsonl", FILES["bom.jsonl"])
    assert with_bom.exit_code == 2
    clean = " ".join(strip_ansi(with_bom.output).split())
    assert "jane.doe@example.com" in clean or "FAIL" in clean

    control = _run(tmp_path, monkeypatch, "nobom.jsonl", FILES["nobom.jsonl"])
    assert control.exit_code == 2
    clean = " ".join(strip_ansi(control.output).split())
    assert "jane.doe@example.com" in clean or "FAIL" in clean


@pytest.mark.parametrize(
    "name, expected_fragment",
    [
        ("empty.jsonl", "no checkable rows"),
        ("all_malformed.jsonl", "malformed JSON"),
        ("array_per_line.jsonl", "not a JSON object"),
        ("non_object_rows.jsonl", "not a JSON object"),
    ],
)
def test_uncheckable_files_fail_nonzero(tmp_path, monkeypatch, name, expected_fragment):
    """Files with no checkable row exit non-zero and say why lines could not be checked."""
    result = _run(tmp_path, monkeypatch, name, FILES[name])
    assert result.exit_code in (2, 3)
    clean = " ".join(strip_ansi(result.output).split())
    assert expected_fragment in clean
    assert "PASS" not in clean


def test_single_malformed_line_reported_with_line_number(tmp_path, monkeypatch):
    """One malformed line among valid rows is reported with its line number."""
    content = b'{"text": "hi"}\n{broken\n{"text": "ok"}\n'
    result = _run(tmp_path, monkeypatch, "mixed.jsonl", content)
    assert result.exit_code in (2, 3)
    clean = " ".join(strip_ansi(result.output).split())
    assert "line 2" in clean
    assert "malformed JSON" in clean
