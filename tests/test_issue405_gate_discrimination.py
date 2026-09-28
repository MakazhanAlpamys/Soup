"""#405 — bundled tool selection must discriminate away from both rails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.eval.gate_suites import load_suite_items, tool_names_in_prompt

RECORD = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "gate-v0.76.0-tool-call-discrimination.md"
)


def _no_tool_rows() -> tuple[dict, ...]:
    return tuple(
        item for item in load_suite_items("mini_tool_call") if item["expected"] == "NO_TOOL"
    )


class TestToolCallFixtureHasMeasuredHeadroom:
    def test_fixture_keeps_40_rows_with_a_balanced_no_tool_axis(self):
        items = load_suite_items("mini_tool_call")
        no_tool = _no_tool_rows()

        assert len(items) == 40
        assert len(no_tool) == 16
        assert len(items) - len(no_tool) == 24

    def test_prompts_are_unique(self):
        prompts = [item["prompt"] for item in load_suite_items("mini_tool_call")]
        assert len(prompts) == len(set(prompts))

    def test_every_row_retains_original_hand_authored_provenance(self):
        for item in load_suite_items("mini_tool_call"):
            assert item["source"].startswith("hand-authored, original")

    def test_no_tool_rows_offer_real_distractors_and_state_the_contract(self):
        rows = _no_tool_rows()
        assert rows
        for item in rows:
            assert "reply exactly NO_TOOL" in item["prompt"]
            assert len(tool_names_in_prompt(item["prompt"])) == 8

    def test_new_positive_rows_include_semantically_close_alternatives(self):
        expected_distractor = {
            "search_contacts": "make_call",
            "make_call": "search_contacts",
            "send_message": "send_email",
            "send_email": "send_message",
            "file_search": "web_search",
            "web_search": "file_search",
            "get_stock_price": "web_search",
            "get_weather": "get_climate",
        }
        items = load_suite_items("mini_tool_call")[-8:]

        assert len(items) == len(expected_distractor)
        for item in items:
            expected = json.loads(item["expected"])["function"]["name"]
            assert expected_distractor[expected] in tool_names_in_prompt(item["prompt"])

    def test_reference_measurements_are_recorded(self):
        text = RECORD.read_text(encoding="utf-8")
        assert "Qwen/Qwen2.5-7B-Instruct" in text
        assert "HuggingFaceTB/SmolLM2-135M-Instruct" in text
        assert "v0.73.2 shipped fixture" in text
        assert "final fixture (16 legacy + 24 candidate rows)" in text
        assert "mini_safety" in text
        assert "Exact-abstention packaging cost" in text


class TestNoToolScoring:
    def test_exact_no_tool_scores_every_no_tool_row(self):
        from soup_cli.eval.gate_suites import _score_tool_call

        assert _score_tool_call(_no_tool_rows(), lambda _prompt: "NO_TOOL") == 1.0

    def test_answering_in_prose_does_not_satisfy_the_exact_contract(self):
        from soup_cli.eval.gate_suites import _score_tool_call

        rows = _no_tool_rows()
        assert rows
        assert _score_tool_call(rows, lambda _prompt: "Paris") == 0.0

    @pytest.mark.parametrize(
        "output",
        [
            "NO_TOOL. I can answer directly.",
            "NO_TOOL\nI can answer directly.",
            "NO_TOOL.",
            "```NO_TOOL```",
            "```\nNO_TOOL\n```",
            "```text\nNO_TOOL\n```",
            "`NO_TOOL`",
            "no_tool",
        ],
    )
    def test_near_miss_no_tool_outputs_do_not_satisfy_exact_contract(self, output):
        from soup_cli.eval.gate_suites import _score_tool_call

        rows = _no_tool_rows()
        assert rows
        assert _score_tool_call(rows, lambda _prompt: output) == 0.0

    def test_exact_no_tool_contract_ignores_surrounding_whitespace(self):
        from soup_cli.eval.gate_suites import _score_tool_call

        rows = _no_tool_rows()
        assert rows
        assert _score_tool_call(rows, lambda _prompt: "  NO_TOOL\n") == 1.0

    def test_calling_a_visible_distractor_fails_every_no_tool_row(self):
        from soup_cli.eval.gate_suites import _score_tool_call

        rows = _no_tool_rows()
        assert rows

        def call_first_tool(prompt: str) -> str:
            first = tool_names_in_prompt(prompt)[0]
            return json.dumps({"function": {"name": first, "arguments": {}}})

        assert _score_tool_call(rows, call_first_tool) == 0.0

    def test_no_tool_inside_a_tool_call_envelope_does_not_score(self):
        from soup_cli.eval.gate_suites import _score_tool_call

        rows = _no_tool_rows()
        assert rows
        output = json.dumps({"function": {"name": "NO_TOOL", "arguments": {}}})
        assert _score_tool_call(rows, lambda _prompt: output) == 0.0

    def test_no_tool_embedded_in_prose_does_not_score(self):
        from soup_cli.eval.gate_suites import _score_tool_call

        rows = _no_tool_rows()
        assert rows
        output = "I think the answer is NO_TOOL because I can answer directly."
        assert _score_tool_call(rows, lambda _prompt: output) == 0.0

    def test_every_expected_answer_still_scores(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        expected = {
            item["prompt"]: item["expected"]
            for item in load_suite_items("mini_tool_call")
        }
        assert score_bundled_suite("mini_tool_call", expected.__getitem__) == 1.0


def test_fixture_scale_change_bumps_baseline_provenance_revision():
    from soup_cli.eval.gate_suites import BUNDLED_SCORER_REVISION

    assert BUNDLED_SCORER_REVISION >= 2

def test_no_tool_scorer_mutation_moves_fingerprint(monkeypatch):
    """Breaking the NO_TOOL branch must move the provenance fingerprint."""
    from soup_cli.eval import gate_suites as suites
    from soup_cli.eval.custom import tool_call_name_match

    before = suites.bundled_scorer_fingerprint()
    assert before == suites.BUNDLED_SCORER_FINGERPRINT

    def broken_tool_call(items, gen):
        def selected(item, output):
            expected = item.get("expected", "")
            if expected == suites.NO_TOOL_RESPONSE:
                return False
            return tool_call_name_match(suites._unwrap_tool_call(output), expected)

        return suites._fraction_passing(items, gen, selected)

    monkeypatch.setitem(suites._EXTENDED_SCORERS, "tool_call", broken_tool_call)
    monkeypatch.setattr(suites, "_fingerprint_responses", None)

    assert suites.bundled_scorer_fingerprint() != before
