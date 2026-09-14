"""Issue #878 — the MCP `data_validate` tool and `soup data validate` disagreed.

#866 was fixed for the CLI by #869: `soup data validate --format bogus` exits 1
instead of reporting every row valid. The MCP surface kept the old behaviour,
so one dataset got two different verdicts depending on which surface asked —
the CLI-vs-MCP divergence class already recorded in the v0.73.2 notes.

The mechanism: `validator.py` computes

    check_format = bool(expected_format and expected_format in VALID_FORMATS)

so an unrecognised format silently turns the format check *off*, and a
`None` format never turns it on. The CLI guards both before it gets there; the
MCP handler passed the raw value straight through.

The property worth pinning is not the exit code but the agreement: the same
file and the same format argument must produce the same verdict on both
surfaces.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


ALPACA_ROWS = [{"instruction": "q0", "output": "a0"}, {"instruction": "q1", "output": "a1"}]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Two valid alpaca rows, reachable from a cwd-contained path."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ALPACA_ROWS), encoding="utf-8")
    return path


class TestAnUnrecognisedFormatIsRefused:
    def test_it_raises_rather_than_reporting_every_row_valid(self, dataset):
        from soup_cli.mcp_server.registry import McpToolError, tool_data_validate

        with pytest.raises(McpToolError):
            tool_data_validate({"data": str(dataset), "format": "bogus"})

    def test_the_error_names_the_accepted_values(self, dataset):
        """As the CLI's does — an error a caller cannot act on is half a fix."""
        from soup_cli.data.formats import VALID_FORMATS
        from soup_cli.mcp_server.registry import McpToolError, tool_data_validate

        with pytest.raises(McpToolError) as exc:
            tool_data_validate({"data": str(dataset), "format": "bogus"})

        message = str(exc.value)
        assert "alpaca" in message and "sharegpt" in message, message
        assert "auto" in message, "auto is accepted too, so it has to be listed"
        missing = sorted(f for f in VALID_FORMATS if f not in message)
        assert not missing, f"accepted values missing from the message: {missing}"

    @pytest.mark.parametrize("bogus", ["", "  ", "ALPACA", "alpaca ", "json", "csv"])
    def test_near_misses_are_refused_too(self, dataset, bogus):
        """Case and whitespace are not silently normalised into a valid format,
        and an empty string must not read as "omitted"."""
        from soup_cli.mcp_server.registry import McpToolError, tool_data_validate

        with pytest.raises(McpToolError):
            tool_data_validate({"data": str(dataset), "format": bogus})


class TestTheAllowlistIsTheRealOne:
    """Acceptance criterion 4: `VALID_FORMATS`, not a second hand-written
    tuple. A copy is how the CLI and the MCP tool drifted apart to begin
    with."""

    def test_every_valid_format_is_accepted(self, dataset):
        from soup_cli.data.formats import VALID_FORMATS
        from soup_cli.mcp_server.registry import McpToolError, tool_data_validate

        refused = []
        for fmt in sorted(VALID_FORMATS):
            try:
                tool_data_validate({"data": str(dataset), "format": fmt})
            except McpToolError:
                refused.append(fmt)
        assert refused == [], f"real formats refused by the handler: {refused}"

    def test_a_format_added_to_the_allowlist_is_accepted_without_a_code_change(
        self, dataset, monkeypatch
    ):
        """The point of reusing the set rather than copying it."""
        from soup_cli.data import formats as formats_module
        from soup_cli.mcp_server.registry import tool_data_validate

        monkeypatch.setattr(
            formats_module, "VALID_FORMATS", set(formats_module.VALID_FORMATS) | {"zzz_new"}
        )
        result = tool_data_validate({"data": str(dataset), "format": "zzz_new"})
        assert result["total"] == 2


class TestAnOmittedFormatAutoDetects:
    """Acceptance criterion 3, and the second bug in the same call: the schema
    says "omit to auto-detect" and nothing detected anything — `None` made
    `check_format` False, so the answer was 2/2 valid for a reason unrelated to
    the data."""

    def test_the_detected_format_is_reported_back(self, dataset):
        from soup_cli.mcp_server.registry import tool_data_validate

        result = tool_data_validate({"data": str(dataset)})
        assert result.get("format") == "alpaca", (
            "the caller cannot tell which format was checked"
        )

    def test_auto_is_accepted_explicitly_as_the_cli_accepts_it(self, dataset):
        from soup_cli.mcp_server.registry import tool_data_validate

        assert tool_data_validate({"data": str(dataset), "format": "auto"})["format"] == (
            "alpaca"
        )

    def test_undetectable_data_is_refused_not_silently_passed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "junk.jsonl"
        path.write_text(json.dumps({"nothing": "recognisable"}) + "\n", encoding="utf-8")

        from soup_cli.mcp_server.registry import McpToolError, tool_data_validate

        with pytest.raises(McpToolError):
            tool_data_validate({"data": str(path)})

    def test_the_schema_description_matches_what_the_handler_does(self):
        """The description promised auto-detect while the code did nothing."""
        from soup_cli.mcp_server.registry import _readonly_specs

        spec = next(t for t in _readonly_specs() if t.name == "data_validate")
        description = spec.input_schema["properties"]["format"]["description"]
        assert "auto" in description.lower(), description


class TestTheTwoSurfacesAgree:
    """Acceptance criterion 5, and the property that actually broke."""

    def _cli(self, dataset, *args):
        from typer.testing import CliRunner

        from soup_cli.commands.data import app

        return CliRunner(env={"COLUMNS": "200"}).invoke(
            app, ["validate", str(dataset), *args]
        )

    def test_both_refuse_an_unrecognised_format(self, dataset):
        from soup_cli.mcp_server.registry import McpToolError, tool_data_validate

        cli = self._cli(dataset, "--format", "bogus")
        assert cli.exit_code != 0, cli.output

        with pytest.raises(McpToolError):
            tool_data_validate({"data": str(dataset), "format": "bogus"})

    @pytest.mark.parametrize("fmt", ["alpaca", "sharegpt", "chatml"])
    def test_both_reach_the_same_valid_row_count(self, dataset, fmt):
        """Including the formats these rows do NOT satisfy — agreeing only on
        the happy path would not have caught the original bug."""
        from soup_cli.data.validator import validate_and_stats
        from soup_cli.mcp_server.registry import tool_data_validate

        mcp = tool_data_validate({"data": str(dataset), "format": fmt})
        cli_equivalent = validate_and_stats(ALPACA_ROWS, expected_format=fmt)

        assert mcp["valid_rows"] == cli_equivalent["valid_rows"]
        assert mcp["total"] == cli_equivalent["total"] == 2

    def test_a_real_but_wrong_format_still_reports_rather_than_raising(self, dataset):
        """Control: the guard is about *unrecognised* values. A recognised
        format the rows fail is a report of 0 valid rows, not an error —
        that distinction is the whole point of the tool."""
        from soup_cli.mcp_server.registry import tool_data_validate

        result = tool_data_validate({"data": str(dataset), "format": "sharegpt"})
        assert result["valid_rows"] == 0
        assert result["total"] == 2
