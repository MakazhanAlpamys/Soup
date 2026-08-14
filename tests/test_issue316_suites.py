"""#316 — the two leg-2 behavioural suites that measured NOTHING on a real model.

Measured on an H100 against ``Meta-Llama-3.1-8B-Instruct`` (see the task brief):

- ``mini_tool_call`` scored **0.000/40**. The prompts were bare user questions
  (``"What's the weather in Paris right now?"``) with **no tool schema anywhere**,
  so a fully tool-capable model correctly answered in prose and **0 of 40 outputs
  contained any JSON at all**. That is a harness defect, not a model failure: the
  suite cannot distinguish a model that lost tool-calling from one that was never
  asked to tool-call, so its contribution to the regression gate was zero.
- ``mini_format_json`` scored **0.000/40**. ``_is_json_container`` ran
  ``json.loads()`` over the WHOLE output; 38 of 40 answers were wrapped in a
  ```json fence. The model emitted correct JSON and the **envelope** failed the
  parse — the suite scored packaging, not validity.
- The generation budget (``max_new_tokens=64``) truncated 31/40 tool calls and
  15/40 JSON fences. 31 of those 32 tool-call failures were **a single missing
  closing brace**. Even a corrected suite is unscoreable at that cap.

Each suite has exactly 40 items, so ONE item is 2.5% against ``soup ship``'s
0.05 regression threshold — a suite pinned at 0.000 contributes nothing but it
also cannot *fall*, which is the specific way this failure hides.

Every fix here carries a CONTROL, because each one has a symmetric failure mode
that is *worse* than the bug: a tool schema that hands the model the answer, or
a JSON extractor so eager that every output contains "some JSON" and the suite
becomes a constant 1.0. A suite pinned at 1.0 detects exactly as much as one
pinned at 0.0.
"""

import json
import time

import pytest

# ---------------------------------------------------------------------------
# Defect 1 — mini_tool_call had no tool schema, so it measured nothing
# ---------------------------------------------------------------------------


def _expected_name(item: dict) -> str:
    """The function name the scorer will require for this item."""
    return json.loads(item["expected"])["function"]["name"]


class TestToolCallSuiteShowsATooolSchema:
    def test_every_prompt_shows_the_tool_the_scorer_requires(self):
        """The load-bearing repair: the model must be able to SEE the tool.

        Without the schema in the prompt, ``tool_call_name_match`` is asking the
        model to guess an internal function name from a bare user question. A
        correct model answers in prose and scores 0 — which is what was measured.
        """
        from soup_cli.eval.gate_suites import load_suite_items

        items = load_suite_items("mini_tool_call")
        assert items, "fixture empty"
        for item in items:
            name = _expected_name(item)
            assert name in item["prompt"], (
                f"tool {name!r} is required by the scorer but never shown to the "
                f"model; prompt={item['prompt'][:120]!r}"
            )

    def test_every_prompt_states_the_output_shape_the_scorer_parses(self):
        """A schema alone is not enough — the scorer parses a specific envelope.

        ``tool_call_name_match`` reads ``{"function": {"name": ...}}``. A model
        told only "you have tools" may answer ``{"tool": "get_weather"}``, which
        is a correct tool call and a scorer miss. The prompt must name the shape.
        """
        from soup_cli.eval.gate_suites import load_suite_items

        for item in load_suite_items("mini_tool_call"):
            prompt = item["prompt"]
            assert '"function"' in prompt, prompt[:160]
            assert '"arguments"' in prompt, prompt[:160]

    def test_prompts_offer_distractors_so_the_suite_measures_selection(self):
        """CONTROL for over-correction: a schema must not hand over the answer.

        If each prompt showed only the one correct tool, the suite would measure
        transcription, not tool selection, and would sit at a ceiling no
        regression could move. Every prompt must offer several candidates.
        """
        from soup_cli.eval.gate_suites import load_suite_items

        for item in load_suite_items("mini_tool_call"):
            names = set(json.loads(_catalogue_json(item["prompt"])))
            assert len(names) >= 4, (
                f"only {len(names)} candidate tools shown — the suite measures "
                f"transcription, not selection: {sorted(names)}"
            )
            assert _expected_name(item) in names

    def test_a_model_that_always_picks_one_tool_cannot_score_well(self):
        """The behavioural half of the control above.

        A degenerate model that emits the same tool call for every prompt must
        score near the floor. If the schema leaked the answer (one tool per
        prompt) this would be impossible to distinguish from a good model.
        """
        from soup_cli.eval.gate_suites import load_suite_items, score_bundled_suite

        items = load_suite_items("mini_tool_call")
        fixed = _expected_name(items[0])
        score = score_bundled_suite(
            "mini_tool_call",
            lambda p: json.dumps({"function": {"name": fixed, "arguments": {}}}),
        )
        # get_weather appears twice in 40 items; anything near 1.0 means the
        # scorer stopped discriminating.
        assert score < 0.15, f"degenerate one-tool model scored {score}"

    def test_prompts_are_unique(self):
        """Prompt text is the dict key every caller uses to look an item up.

        ``tests/test_v07138.py`` builds ``{it["prompt"]: it["expected"]}``; two
        items sharing a prompt would silently collapse and the suite would score
        a different number of items than it reports.
        """
        from soup_cli.eval.gate_suites import load_suite_items

        prompts = [it["prompt"] for it in load_suite_items("mini_tool_call")]
        assert len(prompts) == len(set(prompts))

    def test_prompts_fit_the_generator_prompt_budget(self):
        """A schema long enough to be truncated is a schema that does not exist.

        ``live_eval.make_generator`` tokenises with ``truncation=True,
        max_length=1024`` and HF truncates from the RIGHT, so an over-long
        catalogue would cut off the user question at the end of the prompt and
        silently restore the 0.000 score with a *different* root cause.
        """
        from soup_cli.eval.gate_suites import load_suite_items

        for item in load_suite_items("mini_tool_call"):
            # ~4 chars/token; 1024 tokens is the cap. Stay under a quarter of it.
            assert len(item["prompt"]) < 1000, len(item["prompt"])

    def test_expected_calls_still_parse_as_the_scorer_needs(self):
        """Fixture integrity: rewriting prompts must not disturb the answers."""
        from soup_cli.eval.gate_suites import load_suite_items

        items = load_suite_items("mini_tool_call")
        assert len(items) == 40
        for item in items:
            parsed = json.loads(item["expected"])
            assert isinstance(parsed["function"]["name"], str)
            assert isinstance(parsed["function"]["arguments"], dict)
            assert isinstance(item.get("source"), str) and item["source"]

    def test_echoing_the_expected_call_still_scores_one(self):
        """Back-compat: the generator still receives ``item["prompt"]`` verbatim.

        Existing callers key a lookup off the fixture's own prompt string; if the
        schema were injected at scoring time instead of living in the fixture,
        every such lookup would miss and score 0.
        """
        from soup_cli.eval.gate_suites import load_suite_items, score_bundled_suite

        items = load_suite_items("mini_tool_call")
        by_prompt = {it["prompt"]: it["expected"] for it in items}
        assert score_bundled_suite(
            "mini_tool_call", lambda p: by_prompt[p]
        ) == pytest.approx(1.0)


def _catalogue_json(prompt: str) -> str:
    """Pull the tool-name list out of a rendered prompt, for assertions."""
    from soup_cli.eval.gate_suites import tool_names_in_prompt

    return json.dumps(tool_names_in_prompt(prompt))


# ---------------------------------------------------------------------------
# Defect 2 — _is_json_container scored the envelope, not the JSON
# ---------------------------------------------------------------------------


class TestJsonContainerExtraction:
    def test_fenced_object_is_scored_on_validity_not_packaging(self):
        """The measured case: 38/40 answers arrived inside a ```json fence."""
        from soup_cli.eval.gate_suites import _is_json_container

        assert _is_json_container('```json\n{"name": "Ada", "age": 36}\n```')
        assert _is_json_container("```\n[1, 2, 3]\n```")

    def test_unclosed_fence_still_scores(self):
        """15/40 fences were unclosed — truncated mid-answer by the 64-token cap.

        The JSON itself is complete; only the closing ``` is missing. Scoring
        that as invalid measures the generation budget, not the model.
        """
        from soup_cli.eval.gate_suites import _is_json_container

        assert _is_json_container('```json\n{"title": "Dune", "author": "Herbert"}')

    def test_preamble_before_the_object_is_tolerated(self):
        from soup_cli.eval.gate_suites import _is_json_container

        assert _is_json_container('Sure! Here is the JSON:\n{"ok": true}')

    def test_bare_container_still_passes(self):
        """CONTROL (required): the pre-existing bare-JSON path must not regress."""
        from soup_cli.eval.gate_suites import _is_json_container

        assert _is_json_container('{"ok": true}')
        assert _is_json_container("[1, 2, 3]")
        assert _is_json_container('  {"padded": 1}  ')

    def test_genuinely_invalid_json_is_still_rejected(self):
        """CONTROL (required): an extractor too eager makes the suite constant 1.0.

        Each payload below *looks* like it contains a container but has no valid
        one. If any of these passed, ``mini_format_json`` would score ~1.0 for
        every model and detect nothing — the same blindness as scoring 0.0.
        """
        from soup_cli.eval.gate_suites import _is_json_container

        for bad in (
            "not json at all",
            '{"unterminated": ',
            "{'single': 'quotes'}",
            '{"trailing": 1,}',
            "{key: value}",
            "```json\nnope, I cannot do that\n```",
            "{",
            "[",
            '{"a": undefined}',
        ):
            assert not _is_json_container(bad), bad

    def test_bare_scalar_still_rejected(self):
        """A scalar is valid JSON but not the structured object asked for."""
        from soup_cli.eval.gate_suites import _is_json_container

        assert not _is_json_container("42")
        assert not _is_json_container("true")
        assert not _is_json_container('"just a string"')

    def test_null_byte_and_oversize_still_rejected(self):
        from soup_cli.eval.gate_suites import _MAX_OUTPUT_LEN, _is_json_container

        assert not _is_json_container('{"a": 1}\x00')
        assert not _is_json_container("x" * (_MAX_OUTPUT_LEN + 1))
        assert not _is_json_container(None)  # type: ignore[arg-type]

    def test_extraction_is_bounded_not_unlimited(self):
        """CONTROL: the scan must not hunt the whole output for any bracket pair.

        A model that emits 4 KB of prose and then some JSON did not answer
        "respond with JSON". An unbounded scan would credit it, and would also
        credit incidental braces in ordinary prose.
        """
        from soup_cli.eval.gate_suites import _is_json_container

        assert not _is_json_container("word " * 2000 + '{"buried": true}')

    def test_pathological_nesting_is_rejected_quickly(self):
        """The bounded scan must not turn a RecursionError probe into a hang."""
        from soup_cli.eval.gate_suites import _is_json_container

        start = time.monotonic()
        assert not _is_json_container("[" * 20000)
        assert time.monotonic() - start < 5.0

    def test_suite_score_reflects_the_repair(self):
        from soup_cli.eval.gate_suites import score_bundled_suite

        fenced = score_bundled_suite(
            "mini_format_json", lambda p: '```json\n{"k": "v"}\n```'
        )
        assert fenced == pytest.approx(1.0)
        # ... and the floor still exists.
        assert score_bundled_suite("mini_format_json", lambda p: "sorry, no") == 0.0


class TestToolCallScoringToleratesTheSameEnvelope:
    """The tool-call scorer parses the whole output too — same packaging bug."""

    def test_fenced_tool_call_scores(self):
        from soup_cli.eval.gate_suites import load_suite_items, score_bundled_suite

        by_prompt = {
            it["prompt"]: f"```json\n{it['expected']}\n```"
            for it in load_suite_items("mini_tool_call")
        }
        assert score_bundled_suite(
            "mini_tool_call", lambda p: by_prompt[p]
        ) == pytest.approx(1.0)

    def test_fenced_call_with_the_wrong_name_still_fails(self):
        """CONTROL: unwrapping must not become unconditional crediting."""
        from soup_cli.eval.gate_suites import score_bundled_suite

        wrong = '```json\n{"function": {"name": "definitely_not_a_tool", "arguments": {}}}\n```'
        assert score_bundled_suite("mini_tool_call", lambda p: wrong) == 0.0


# ---------------------------------------------------------------------------
# Defect 3 — the generation budget truncated the answers being scored
# ---------------------------------------------------------------------------


class TestBehaviouralGenerationBudget:
    def test_budget_is_declared_and_fits_a_real_tool_call(self):
        """64 tokens left 31/40 tool calls one closing brace short.

        The behavioural suites need a budget of their own: the MCQ suites answer
        in one letter, so the shared 64-token default was never sized for them.
        """
        from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS

        assert isinstance(BEHAVIOURAL_MAX_NEW_TOKENS, int)
        assert not isinstance(BEHAVIOURAL_MAX_NEW_TOKENS, bool)
        assert BEHAVIOURAL_MAX_NEW_TOKENS >= 128, (
            "64 truncated 31/40 tool calls and 15/40 JSON fences; a budget that "
            "does not clear that with margin re-creates the defect"
        )

    def test_budget_clears_the_longest_expected_answer_with_margin(self):
        """Pinned against the fixture, so a future longer item cannot silently
        outgrow the budget."""
        from soup_cli.eval.gate_suites import (
            BEHAVIOURAL_MAX_NEW_TOKENS,
            load_suite_items,
        )

        longest = max(len(it["expected"]) for it in load_suite_items("mini_tool_call"))
        # ~4 chars/token, doubled for the fence + preamble models actually emit.
        assert BEHAVIOURAL_MAX_NEW_TOKENS >= (longest / 4) * 2


# ---------------------------------------------------------------------------
# The suites must stay offline, deterministic and import-light
# ---------------------------------------------------------------------------


class TestSuitesStayOfflineAndDeterministic:
    def test_scoring_is_deterministic_across_runs(self):
        from soup_cli.eval.gate_suites import load_suite_items, score_bundled_suite

        by_prompt = {
            it["prompt"]: it["expected"] for it in load_suite_items("mini_tool_call")
        }
        runs = {score_bundled_suite("mini_tool_call", lambda p: by_prompt[p])
                for _ in range(3)}
        assert len(runs) == 1

    def test_module_imports_no_network_or_torch(self):
        """The offline promise: ``soup ci init``'s core-only install must work."""
        import subprocess
        import sys

        code = (
            "import sys; import soup_cli.eval.gate_suites as g;"
            "assert 'torch' not in sys.modules;"
            "assert 'requests' not in sys.modules;"
            "assert 'urllib.request' not in sys.modules;"
            "print('ok')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, (out.stdout, out.stderr)
        assert "ok" in out.stdout


class TestTheBudgetReachesTheGenerator:
    """Defect 3, wired at the construction site.

    `gate_suites` can only DECLARE the budget; `commands/ship.py::_resolve_generators`
    is what builds the generators, and it was taking `make_generator`'s default of
    64. Measured through the shipped code: a correct tool call **one closing brace
    short** scores 0.000, and that was 31 of 40 items.

    Extraction deliberately does not repair truncated JSON — a decoder lenient
    enough to guess the missing brace is the "extracts too eagerly" failure the
    other controls exist to prevent. So the budget is the only fix, and this test
    exists because declaring a constant nobody reads is indistinguishable from not
    having one.
    """

    def _capture(self, monkeypatch):
        from soup_cli.utils import live_eval

        calls = []

        def _fake(model_id, adapter=None, device=None, max_new_tokens=64, **kwargs):
            calls.append(max_new_tokens)
            return lambda prompt: ""

        monkeypatch.setattr(live_eval, "make_generator", _fake)
        return calls

    def test_every_generator_gets_the_behavioural_budget(self, monkeypatch):
        from soup_cli.commands.ship import _resolve_generators
        from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS

        calls = self._capture(monkeypatch)
        _resolve_generators(base="b", adapter="a", tuned=None, device="cpu")
        assert calls, "no generator was built, so this asserts nothing"
        assert all(n == BEHAVIOURAL_MAX_NEW_TOKENS for n in calls), calls

    def test_the_budget_fits_a_real_tool_call(self):
        """CONTROL on the constant itself: 64 was the measured failure and any
        value near it reintroduces the truncation this fix exists to remove."""
        from soup_cli.eval.gate_suites import BEHAVIOURAL_MAX_NEW_TOKENS

        assert BEHAVIOURAL_MAX_NEW_TOKENS >= 128, BEHAVIOURAL_MAX_NEW_TOKENS
