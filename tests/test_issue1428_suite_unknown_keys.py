"""#1428: ``soup expect`` refuses suite keys and argument names an expectation does not take.

Before this fix ``parse_suite_spec`` read ``name`` and ``args`` and dropped every other
key, and ``_dispatch_expectation`` read arguments with ``args.get(key, default)``. The
``docs/compliance.md`` example, which wrote ``min_tokens`` / ``max_tokens`` beside
``name``, therefore ran with the defaults (1 .. 1,048,576 tokens) and passed a
5000-token row. A misspelled argument did the same. The gate that ran was not the
gate the user wrote, and nothing said so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import expectations
from soup_cli.utils.expectations import parse_suite_yaml

_ROOT = Path(__file__).resolve().parents[1]
_COMPLIANCE_MD = _ROOT / "docs" / "compliance.md"
_DATA_MD = _ROOT / "docs" / "data.md"

# The docs/compliance.md example, verbatim as shipped before the fix (issue body).
_SIBLING_FORM = (
    "expectations:\n"
    "  - name: expect_no_pii\n"
    "  - name: expect_token_length_between\n"
    "    min_tokens: 1\n"
    "    max_tokens: 512\n"
)
_MISSPELLED = "expectations:\n  - name: expect_token_length_between\n    args: {max_token: 512}\n"
_CONTROL = "expectations:\n  - name: expect_token_length_between\n    args: {max_tokens: 512}\n"
_LONG_ROW = json.dumps({"text": "word " * 5000}) + "\n"


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _compliance_suite() -> str:
    """The ``expectations.yaml`` block of docs/compliance.md, read from the file."""
    blocks = re.findall(
        r"```yaml\n(expectations:\n.*?)```", _COMPLIANCE_MD.read_text(encoding="utf-8"), re.S
    )
    assert len(blocks) == 1, blocks
    return blocks[0]


def _data_md_suite() -> str:
    """The ``suite.yaml`` heredoc of docs/data.md (the ``args:`` form), read from the file."""
    match = re.search(
        r"^expectations:\n((?:  - .*\n)+)EOF$", _DATA_MD.read_text(encoding="utf-8"), re.M
    )
    assert match is not None
    return "expectations:\n" + match.group(1)


def _run(tmp_path: Path, suite_text: str, row: str) -> tuple[int, str]:
    (tmp_path / "data.jsonl").write_text(row, encoding="utf-8")
    (tmp_path / "suite.yaml").write_text(suite_text, encoding="utf-8")
    result = CliRunner().invoke(app, ["expect", "data.jsonl", "suite.yaml"])
    return result.exit_code, _plain(result.output)


class TestKeysBesideName:
    def test_the_compliance_md_sibling_form_names_both_keys_and_args(self) -> None:
        with pytest.raises(ValueError, match=r"expectations\[1\]") as info:
            parse_suite_yaml(_SIBLING_FORM)
        message = str(info.value)
        assert "min_tokens" in message and "max_tokens" in message
        assert "under 'args:'" in message

    def test_judge_threshold_beside_name_is_refused(self) -> None:
        text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    threshold: 0.9\n"
        )
        with pytest.raises(ValueError, match="threshold") as info:
            parse_suite_yaml(text)
        assert "under 'args:'" in str(info.value)

    def test_a_key_that_is_no_argument_anywhere_is_refused(self) -> None:
        text = "expectations:\n  - name: expect_no_pii\n    severity: high\n"
        with pytest.raises(ValueError, match="severity") as info:
            parse_suite_yaml(text)
        assert "name" in str(info.value) and "args" in str(info.value)


class TestUnknownArgumentNames:
    def test_misspelled_argument_is_refused_with_the_nearest_name(self) -> None:
        with pytest.raises(ValueError, match="max_token") as info:
            parse_suite_yaml(_MISSPELLED)
        assert "did you mean 'max_tokens'" in str(info.value)

    def test_an_argument_on_an_expectation_taking_none_is_refused(self) -> None:
        text = "expectations:\n  - name: expect_no_pii\n    args: {threshold: 0.5}\n"
        with pytest.raises(ValueError, match="threshold") as info:
            parse_suite_yaml(text)
        assert "expect_no_pii" in str(info.value)

    def test_the_first_unknown_argument_is_reported_by_index(self) -> None:
        text = (
            "expectations:\n"
            "  - name: expect_no_pii\n"
            "  - name: expect_token_length_between\n"
            "    args: {min_tokens: 1, max_tokenz: 5}\n"
        )
        with pytest.raises(ValueError, match=r"expectations\[1\]"):
            parse_suite_yaml(text)


class TestValuesAreCheckedAtParseTime:
    def test_min_above_max_is_refused_before_any_data_is_read(self) -> None:
        text = (
            "expectations:\n"
            "  - name: expect_token_length_between\n"
            "    args: {min_tokens: 10, max_tokens: 5}\n"
        )
        with pytest.raises(ValueError, match="min_tokens"):
            parse_suite_yaml(text)

    def test_out_of_range_threshold_is_refused(self) -> None:
        text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {threshold: 1.5}\n"
        )
        with pytest.raises(ValueError, match="threshold"):
            parse_suite_yaml(text)

    def test_bool_token_bound_is_refused(self) -> None:
        text = (
            "expectations:\n  - name: expect_token_length_between\n    args: {max_tokens: true}\n"
        )
        with pytest.raises(TypeError, match="max_tokens"):
            parse_suite_yaml(text)


class TestControls:
    def test_the_args_form_still_loads(self) -> None:
        spec = parse_suite_yaml(_CONTROL)
        assert dict(spec.expectations[0].args) == {"max_tokens": 512}

    def test_no_args_expectations_still_load(self) -> None:
        text = (
            "expectations:\n"
            "  - name: expect_no_pii\n"
            "  - name: expect_no_refusal_pattern\n"
            "  - name: expect_token_length_between\n"
            "    args: {}\n"
        )
        assert [e.name for e in parse_suite_yaml(text).expectations] == [
            "expect_no_pii",
            "expect_no_refusal_pattern",
            "expect_token_length_between",
        ]

    def test_the_docs_data_md_suite_still_loads(self) -> None:
        spec = parse_suite_yaml(_data_md_suite())
        assert len(spec.expectations) == 3

    def test_every_supported_expectation_declares_its_arguments(self) -> None:
        assert set(expectations._EXPECTATION_ARGS) == set(expectations.SUPPORTED_EXPECTATIONS)


class TestSoupExpectCli:
    def test_sibling_form_exits_3_before_the_data_is_opened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import soup_cli.commands.expect as expect_mod

        def _never(_path: str) -> list:
            raise AssertionError("the data file was opened before the suite was refused")

        monkeypatch.setattr(expect_mod, "_load_jsonl_rows", _never)
        monkeypatch.chdir(tmp_path)
        code, out = _run(tmp_path, _SIBLING_FORM, _LONG_ROW)
        assert code == 3, out
        assert "min_tokens" in out and "args" in out

    def test_misspelled_argument_exits_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, out = _run(tmp_path, _MISSPELLED, _LONG_ROW)
        assert code == 3, out
        assert "max_token" in out and "max_tokens" in out

    def test_control_args_form_still_fails_the_long_row_with_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, out = _run(tmp_path, _CONTROL, _LONG_ROW)
        assert code == 2, out
        assert "FAIL" in out

    def test_control_no_args_suite_passes_a_clean_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        text = "expectations:\n  - name: expect_no_pii\n  - name: expect_no_refusal_pattern\n"
        code, out = _run(tmp_path, text, '{"text": "a short clean row"}\n')
        assert code == 0, out

    def test_control_token_length_without_args_runs_on_its_default_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        suite = "expectations:\n  - name: expect_token_length_between\n"
        code, out = _run(tmp_path, suite, _LONG_ROW)
        assert code == 0, out

    def test_a_terminal_escape_in_a_key_name_does_not_reach_the_terminal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # YAML's ``\e`` escape yields a real ESC byte in the key once parsed.
        text = 'expectations:\n  - name: expect_no_pii\n    "\\e[2Jsev": 1\n'
        (tmp_path / "data.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
        (tmp_path / "suite.yaml").write_text(text, encoding="utf-8")
        result = CliRunner().invoke(app, ["expect", "data.jsonl", "suite.yaml"])
        assert result.exit_code == 3, result.output
        assert "sev" in _plain(result.output)  # the key is echoed ...
        assert "\x1b[2J" not in result.output  # ... but never as a raw control sequence


class TestTheDocsExampleIsAGateThatGates:
    def test_compliance_md_suite_fails_a_5000_token_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, out = _run(tmp_path, _compliance_suite(), _LONG_ROW)
        assert code == 2, out
        assert "FAIL" in out

    def test_compliance_md_suite_passes_a_short_clean_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, out = _run(tmp_path, _compliance_suite(), '{"text": "a short clean row"}\n')
        assert code == 0, out
