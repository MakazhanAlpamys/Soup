"""Issue #889 — MCP data_doctor format validation matches data_validate."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"instruction": "q", "output": "a"}) + "\n",
        encoding="utf-8",
    )
    return path


def _stub_doctor_runtime(monkeypatch):
    """Keep these tests on format routing, not the optional tokenizer stack."""
    from soup_cli.utils import data_doctor as doctor_module

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    monkeypatch.setattr(doctor_module, "resolve_tokenizer", lambda model, **kwargs: object())

    class Report:
        def __init__(self, fmt):
            self.fmt = fmt

        def to_dict(self):
            return {"format": self.fmt}

    monkeypatch.setattr(
        doctor_module,
        "run_doctor",
        lambda rows, tok, *, fmt, max_length, sample_size: Report(fmt),
    )


class TestInvalidFormatIsRefused:
    @pytest.mark.parametrize("bad_format", ["", "  ", "bogus", "ALPACA", "alpaca "])
    def test_blank_and_unknown_values_are_rejected(self, dataset, bad_format):
        from soup_cli.mcp_server.registry import McpToolError, tool_data_doctor

        with pytest.raises(McpToolError) as exc_info:
            tool_data_doctor(
                {"data": str(dataset), "model": "fake/model", "format": bad_format}
            )

        message = str(exc_info.value)
        assert f"unknown format {bad_format!r}" in message

    def test_error_matches_data_validate_and_names_the_real_allowlist(self, dataset):
        from soup_cli.data.formats import VALID_FORMATS
        from soup_cli.mcp_server.registry import (
            McpToolError,
            tool_data_doctor,
            tool_data_validate,
        )

        with pytest.raises(McpToolError) as validate_exc:
            tool_data_validate({"data": str(dataset), "format": "bogus"})
        with pytest.raises(McpToolError) as doctor_exc:
            tool_data_doctor(
                {"data": str(dataset), "model": "fake/model", "format": "bogus"}
            )

        message = str(doctor_exc.value)
        assert message == str(validate_exc.value)
        assert "auto" in message
        assert all(fmt in message for fmt in VALID_FORMATS)
        assert str(dataset) not in message
        assert len(message) < 2048


class TestSharedAllowlistAndAutoDetection:
    def test_every_valid_format_reaches_the_doctor(self, dataset, monkeypatch):
        from soup_cli.data.formats import VALID_FORMATS
        from soup_cli.mcp_server.registry import tool_data_doctor

        _stub_doctor_runtime(monkeypatch)
        for fmt in sorted(VALID_FORMATS):
            result = tool_data_doctor(
                {"data": str(dataset), "model": "fake/model", "format": fmt}
            )
            assert result["format"] == fmt

    def test_new_allowlist_member_needs_no_handler_change(self, dataset, monkeypatch):
        from soup_cli.data import formats as formats_module
        from soup_cli.mcp_server.registry import tool_data_doctor

        _stub_doctor_runtime(monkeypatch)
        monkeypatch.setattr(
            formats_module,
            "VALID_FORMATS",
            set(formats_module.VALID_FORMATS) | {"zzz_new"},
        )
        result = tool_data_doctor(
            {"data": str(dataset), "model": "fake/model", "format": "zzz_new"}
        )
        assert result["format"] == "zzz_new"

    def test_omitted_format_still_auto_detects(self, dataset, monkeypatch):
        from soup_cli.mcp_server.registry import tool_data_doctor

        _stub_doctor_runtime(monkeypatch)
        result = tool_data_doctor({"data": str(dataset), "model": "fake/model"})
        assert result["format"] == "alpaca"
