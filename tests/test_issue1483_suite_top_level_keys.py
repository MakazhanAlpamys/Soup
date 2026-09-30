"""#1483: ``soup expect`` refuses every top-level suite key other than ``expectations``.

#1428 / #1461 made ``parse_suite_spec`` refuse unknown keys inside an expectation
entry. One level up it still read only ``raw.get("expectations")``, so a suite
with ``fail_fast: true`` beside ``expectations:`` loaded, ran and exited 0 with no
message about the key.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.expectations import parse_suite_yaml

_TOP_LEVEL = "fail_fast: true\nexpectations:\n  - name: expect_no_pii\n"
_CONTROL = "expectations:\n  - name: expect_no_pii\n"


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _run(tmp_path: Path, suite_text: str) -> tuple[int, str, str]:
    (tmp_path / "data.jsonl").write_text('{"text": "a short clean row"}\n', encoding="utf-8")
    (tmp_path / "suite.yaml").write_text(suite_text, encoding="utf-8")
    result = CliRunner().invoke(app, ["expect", "data.jsonl", "suite.yaml"])
    return result.exit_code, _plain(result.output), result.output


class TestTopLevelKeys:
    def test_a_key_beside_expectations_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="fail_fast") as info:
            parse_suite_yaml(_TOP_LEVEL)
        assert "expectations" in str(info.value)

    def test_every_extra_key_is_named(self) -> None:
        text = "fail_fast: true\nseverity: high\n" + _CONTROL
        with pytest.raises(ValueError) as info:
            parse_suite_yaml(text)
        assert "fail_fast" in str(info.value) and "severity" in str(info.value)

    def test_control_the_plain_form_still_loads(self) -> None:
        assert [e.name for e in parse_suite_yaml(_CONTROL).expectations] == ["expect_no_pii"]


class TestSoupExpectCli:
    def test_exits_3_before_the_data_is_opened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import soup_cli.commands.expect as expect_mod

        def _never(_path: str) -> list:
            raise AssertionError("the data file was opened before the suite was refused")

        monkeypatch.setattr(expect_mod, "_load_jsonl_rows", _never)
        monkeypatch.chdir(tmp_path)
        code, out, _ = _run(tmp_path, _TOP_LEVEL)
        assert code == 3, out
        assert "fail_fast" in out

    def test_control_the_plain_form_exits_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, out, _ = _run(tmp_path, _CONTROL)
        assert code == 0, out

    def test_a_terminal_escape_in_a_top_level_key_does_not_reach_the_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # YAML's ``\e`` escape yields a real ESC byte in the key once parsed.
        code, out, raw = _run(tmp_path, '"\\e[2Jfast": true\n' + _CONTROL)
        assert code == 3, out
        assert "fast" in out  # the key is echoed ...
        assert "\x1b[2J" not in raw  # ... but never as a raw control sequence
