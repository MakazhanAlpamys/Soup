"""Tests for tool-calling / agentic fine-tuning pipeline (Part B of v0.25.0)."""

import json
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Format detection + normalization
# ---------------------------------------------------------------------------


class TestToolCallingFormat:
    """Format detection + normalization for tool-calling data."""

    @staticmethod
    def _row():
        return {
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}',
                    },
                }
            ],
        }

    def test_detect_format(self):
        from soup_cli.data.formats import detect_format

        row = self._row()
        assert detect_format([row]) == "tool-calling"

    def test_normalize_row(self):
        from soup_cli.data.formats import format_to_messages

        result = format_to_messages(self._row(), "tool-calling")
        assert result is not None
        messages = result["messages"]
        # First message is system with embedded tool schema
        assert messages[0]["role"] == "system"
        assert "get_weather" in messages[0]["content"]
        # User message is preserved
        assert any(m["role"] == "user" and "Tokyo" in m["content"] for m in messages)
        # Assistant message with tool_calls appended
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "tool_calls" in assistant_msgs[0]
        assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_detect_prefers_tool_calling_over_chatml(self):
        """tool-calling check must come before chatml, since chatml sig is a subset."""
        from soup_cli.data.formats import detect_format

        row = self._row()
        assert detect_format([row]) == "tool-calling"

    def test_normalize_invalid_tools_returns_none(self):
        from soup_cli.data.formats import format_to_messages

        bad = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": "not-a-list",
            "tool_calls": [],
        }
        assert format_to_messages(bad, "tool-calling") is None

    def test_normalize_invalid_tool_calls_args_returns_none(self):
        """Non-JSON parseable tool_call arguments are rejected."""
        from soup_cli.data.formats import format_to_messages

        bad = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
            "tool_calls": [{"function": {"name": "f", "arguments": "not-json-content{"}}],
        }
        assert format_to_messages(bad, "tool-calling") is None


# ---------------------------------------------------------------------------
# Synth data template
# ---------------------------------------------------------------------------


class TestToolCallingTemplate:
    def test_build_prompt_contains_domains_and_count(self):
        from soup_cli.data.templates.tool_calling import build_prompt

        prompt = build_prompt(count=5, fmt="tool-calling", format_spec="{...}", domain="weather")
        assert "5" in prompt
        assert "weather" in prompt.lower()
        assert "function" in prompt.lower() or "tool" in prompt.lower()

    def test_build_prompt_default_domain(self):
        from soup_cli.data.templates.tool_calling import build_prompt

        prompt = build_prompt(count=3, fmt="tool-calling", format_spec="{}")
        assert "3" in prompt

    def test_available_domains(self):
        from soup_cli.data.templates.tool_calling import TEMPLATE_SPEC

        for key in ("weather", "search", "database", "filesystem"):
            assert key in TEMPLATE_SPEC["domains"]


# ---------------------------------------------------------------------------
# DataConfig literal accepts tool-calling
# ---------------------------------------------------------------------------


class TestToolCallingConfig:
    def test_dataconfig_accepts_tool_calling(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl", format="tool-calling")
        assert cfg.format == "tool-calling"

    def test_soupconfig_tool_calling(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: tool-calling
training:
  epochs: 1
  lr: 2e-4
output: ./output
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.data.format == "tool-calling"


# ---------------------------------------------------------------------------
# Init template
# ---------------------------------------------------------------------------


class TestToolCallingInitTemplate:
    def test_tool_calling_template_in_templates(self):
        from soup_cli.config.schema import TEMPLATES

        assert "tool-calling" in TEMPLATES

    def test_tool_calling_template_loads(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.config.schema import TEMPLATES

        cfg = load_config_from_string(TEMPLATES["tool-calling"])
        assert cfg.data.format == "tool-calling"
        assert cfg.task == "sft"

    def test_init_with_tool_calling_template(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "init",
                "--template",
                "tool-calling",
            ],
        )
        assert result.exit_code == 0
        soup_yaml = tmp_path / "soup.yaml"
        assert soup_yaml.exists()
        content = soup_yaml.read_text(encoding="utf-8")
        assert "tool-calling" in content


# ---------------------------------------------------------------------------
# Eval scoring functions
# ---------------------------------------------------------------------------


class TestToolCallScoring:
    """Scoring functions for tool-call evaluation."""

    def test_tool_call_match_exact(self):
        from soup_cli.eval.custom import tool_call_match

        output = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        )
        expected = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        )
        assert tool_call_match(output, expected) is True

    def test_tool_call_match_wrong_name(self):
        from soup_cli.eval.custom import tool_call_match

        output = json.dumps(
            {
                "function": {"name": "get_time", "arguments": '{"city": "Tokyo"}'},
            }
        )
        expected = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        )
        assert tool_call_match(output, expected) is False

    def test_tool_call_match_wrong_args(self):
        from soup_cli.eval.custom import tool_call_match

        output = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Kyoto"}'},
            }
        )
        expected = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        )
        assert tool_call_match(output, expected) is False

    def test_tool_call_name_match_only_name(self):
        from soup_cli.eval.custom import tool_call_name_match

        output = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Osaka"}'},
            }
        )
        expected = json.dumps(
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        )
        assert tool_call_name_match(output, expected) is True

    def test_tool_call_args_subset_partial(self):
        from soup_cli.eval.custom import tool_call_args_subset

        # Expected args is a subset of output args — partial credit
        output = json.dumps(
            {
                "function": {
                    "name": "search",
                    "arguments": '{"query": "cats", "limit": 10}',
                },
            }
        )
        expected = json.dumps(
            {
                "function": {
                    "name": "search",
                    "arguments": '{"query": "cats"}',
                },
            }
        )
        score = tool_call_args_subset(output, expected)
        assert score > 0.9  # name matches + query matches

    def test_tool_call_args_subset_no_match(self):
        from soup_cli.eval.custom import tool_call_args_subset

        output = json.dumps(
            {
                "function": {"name": "search", "arguments": '{"query": "dogs"}'},
            }
        )
        expected = json.dumps(
            {
                "function": {"name": "search", "arguments": '{"query": "cats"}'},
            }
        )
        score = tool_call_args_subset(output, expected)
        assert score < 1.0

    def test_tool_call_match_invalid_json_returns_false(self):
        from soup_cli.eval.custom import tool_call_match

        assert tool_call_match("not-json", "{}") is False

    def test_tool_call_scoring_registered(self):
        """tool_call_* scorings are registered for use as eval 'scoring' strings."""
        from soup_cli.eval.custom import VALID_SCORING

        assert "tool_call_match" in VALID_SCORING
        assert "tool_call_name_match" in VALID_SCORING
        assert "tool_call_args_subset" in VALID_SCORING


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


class TestToolCallingRecipes:
    def test_qwen3_8b_tools_recipe(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("qwen3-8b-tools")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.data.format == "tool-calling"
        assert cfg.task == "sft"

    def test_llama4_scout_tools_recipe(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("llama4-scout-tools")
        assert recipe is not None
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.data.format == "tool-calling"


# ---------------------------------------------------------------------------
# Round-trip and security
# ---------------------------------------------------------------------------


class TestToolCallingRoundTrip:
    def test_validate_and_stats_tool_calling(self, tmp_path):
        from soup_cli.data.loader import load_raw_data
        from soup_cli.data.validator import validate_and_stats

        row = {
            "messages": [{"role": "user", "content": "Query weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_calls": [
                {
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        }

        data_file = tmp_path / "tools.jsonl"
        data_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

        data = load_raw_data(data_file)
        stats = validate_and_stats(data, "tool-calling")
        assert stats["total"] == 1
        # Render via StringIO console (project convention)
        console = Console(file=StringIO())
        console.print(f"Stats: {stats}")
        assert "total" in console.file.getvalue()

    def test_empty_tool_calls_with_partial_schema(self):
        """Partial tool schema with empty tool_calls still produces normalized messages."""
        from soup_cli.data.formats import format_to_messages

        row = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function"}],
            "tool_calls": [],
        }
        result = format_to_messages(row, "tool-calling")
        assert result is not None
        assert "messages" in result
        # Only the synthesized system + the user message
        assert any(m["role"] == "system" for m in result["messages"])

    def test_tool_call_non_dict_rejected(self):
        """A non-dict tool_calls entry is rejected."""
        from soup_cli.data.formats import format_to_messages

        row = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_calls": ["not-a-dict"],
        }
        assert format_to_messages(row, "tool-calling") is None

    def test_tool_call_referencing_unknown_function_still_normalizes(self):
        """Cross-reference: tool_calls may name a function not in tools.

        v0.25.0 does not enforce strict tools↔tool_calls cross-reference — the
        format normalizer passes the call through so the model learns what the
        user actually asked for. This test documents that contract.
        """
        from soup_cli.data.formats import format_to_messages

        row = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_calls": [{"function": {"name": "search_web", "arguments": "{}"}}],
        }
        result = format_to_messages(row, "tool-calling")
        assert result is not None
        # The assistant turn carries the original (unknown) function name
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "search_web"


# ---------------------------------------------------------------------------
# Data validator accepts tool-calling format
# ---------------------------------------------------------------------------


class TestToolCallingValidator:
    def test_validator_accepts_tool_calling_format(self, tmp_path):
        import json as json_mod

        from soup_cli.data.loader import load_raw_data
        from soup_cli.data.validator import validate_and_stats

        rows = [
            {
                "messages": [{"role": "user", "content": f"q{i}"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
            }
            for i in range(3)
        ]
        path = tmp_path / "toolcalls.jsonl"
        path.write_text("\n".join(json_mod.dumps(r) for r in rows) + "\n", encoding="utf-8")

        data = load_raw_data(path)
        stats = validate_and_stats(data, "tool-calling")
        assert stats["total"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
