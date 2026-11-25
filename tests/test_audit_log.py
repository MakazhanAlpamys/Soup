"""``read_audit_tail`` and ``soup audit-log tail`` on logs with non-record lines (#2460)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.audit_log import read_audit_tail
from tests.conftest import strip_ansi

_GOOD = {
    "timestamp": "2026-11-03T10:00:00Z",
    "command": "train",
    "exit_code": 0,
    "operator_id": "op-1",
    "host_id": "host-1",
}

# Valid JSON on every line, but none of them is an object.
_NON_OBJECT_LINES = ["[1, 2, 3]", '"just a string"', "42", "null", "true"]


def _write_log(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestReadAuditTailNonObjectLines:
    def test_skips_json_values_that_are_not_objects(self, tmp_path):
        log = _write_log(tmp_path / "audit.jsonl", [json.dumps(_GOOD), *_NON_OBJECT_LINES])

        assert read_audit_tail(log) == [_GOOD]

    def test_limit_counts_records_not_stray_lines(self, tmp_path):
        older = dict(_GOOD, command="eval")
        log = _write_log(
            tmp_path / "audit.jsonl",
            [json.dumps(older), json.dumps(_GOOD), *_NON_OBJECT_LINES],
        )

        assert read_audit_tail(log, limit=1) == [_GOOD]
        assert read_audit_tail(log, limit=2) == [older, _GOOD]

    def test_only_non_object_lines_gives_no_records(self, tmp_path):
        log = _write_log(tmp_path / "audit.jsonl", _NON_OBJECT_LINES)

        assert read_audit_tail(log) == []


class TestAuditLogTailCliNonObjectLines:
    def test_table_shows_the_record_and_exits_zero(self, tmp_path):
        log = _write_log(tmp_path / "audit.jsonl", [json.dumps(_GOOD), "[1, 2, 3]", "42"])

        result = CliRunner().invoke(app, ["audit-log", "tail", "--path", log])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = strip_ansi(result.output)
        assert "train" in out
        assert "host-1" in out
        assert "No audit records." not in out

    def test_json_output_has_only_the_record(self, tmp_path):
        log = _write_log(tmp_path / "audit.jsonl", ["[1, 2, 3]", json.dumps(_GOOD), '"x"'])

        result = CliRunner().invoke(app, ["audit-log", "tail", "--path", log, "--json"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert [json.loads(ln) for ln in lines] == [_GOOD]

    def test_limit_one_with_trailing_stray_lines_shows_the_record(self, tmp_path):
        log = _write_log(tmp_path / "audit.jsonl", [json.dumps(_GOOD), *_NON_OBJECT_LINES])

        result = CliRunner().invoke(
            app, ["audit-log", "tail", "--path", log, "--limit", "1", "--json"]
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert [json.loads(ln) for ln in lines] == [_GOOD]
