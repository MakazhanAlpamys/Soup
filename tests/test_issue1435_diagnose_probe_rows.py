"""Tests for Issue #1435 (PR 2): live diagnose probe inputs from real dataset formats.

PR 1 reports a dataset the row builders cannot read as NOT_RUN. These tests pin that the
common training formats (ShareGPT, Alpaca with ``input``, plaintext, mixed) now reach the
probes as the model saw them, that vision rows and over-long prompts are handled
explicitly, and that skipped rows are reported.
"""

from __future__ import annotations

import json

import pytest

from soup_cli.utils.diagnose.memorization import split_prefix
from tests.conftest import strip_ansi
from tests.test_issue1435_diagnose_not_run import DATASET_MODES, _cli, _gens, _run_live

SKIPPABLE = ("forgetting", "format", "mode_collapse")


def _words(tag: str, count: int = 24) -> str:
    return " ".join(f"{tag}word{j}" for j in range(count))


def _sharegpt(i: int, answer: str | None = None) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": f"question {i} {_words(f'q{i}', 12)}"},
            {"from": "gpt", "value": answer or f"answer {i} {_words(f'a{i}', 12)}"},
        ]
    }


def _not_run_modes(report) -> set:
    return {m for m, s in report.scores.items() if s.verdict == "NOT_RUN"}


class TestPlaintext:
    TEXTS = [_words(f"topic{i}") for i in range(12)]

    def _regurgitating(self):
        suffix = dict(split_prefix(text) for text in self.TEXTS)
        return _gens(adapter_gen=lambda prompt: suffix.get(prompt, "ok"))

    def test_memorization_runs_and_catches_regurgitation(self, tmp_path, monkeypatch):
        rows = [{"text": text} for text in self.TEXTS]
        report, _ = _run_live(tmp_path, monkeypatch, rows, self._regurgitating())
        assert report.scores["memorization"].verdict == "MAJOR"
        assert report.overall == "MAJOR"

    def test_probes_needing_a_prompt_answer_split_say_so(self, tmp_path, monkeypatch):
        rows = [{"text": text} for text in self.TEXTS]
        report, _ = _run_live(tmp_path, monkeypatch, rows, self._regurgitating())
        for mode in SKIPPABLE:
            score = report.scores[mode]
            assert score.verdict == "NOT_RUN", mode
            assert "plaintext" in score.evidence

    def test_cli_exit_2_on_regurgitating_plaintext(self, tmp_path, monkeypatch):
        rows = [{"text": text} for text in self.TEXTS]
        result, _, _ = _cli(tmp_path, monkeypatch, rows, self._regurgitating())
        assert result.exit_code == 2


class TestShareGPT:
    def test_probes_run(self, tmp_path, monkeypatch):
        rows = [_sharegpt(i) for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in DATASET_MODES:
            assert report.scores[mode].verdict != "NOT_RUN", mode
        assert report.overall != "NOT_RUN"

    def test_conversation_text_reaches_memorization(self, tmp_path, monkeypatch):
        rows = [_sharegpt(i) for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert "scanned=" in report.scores["memorization"].evidence

    def test_json_targets_are_seen_by_the_format_probe(self, tmp_path, monkeypatch):
        rows = [_sharegpt(i, answer=json.dumps({"n": i})) for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())  # adapter never emits JSON
        assert report.scores["format"].verdict == "MAJOR"


class TestAlpaca:
    def test_input_is_joined_to_the_instruction(self, tmp_path, monkeypatch):
        seen: list = []

        def base_gen(prompt):
            seen.append(prompt)
            return "ok"

        rows = [
            {"instruction": "Translate to French:", "input": f"Good morning {i}",
             "output": f"Bonjour {i}"}
            for i in range(12)
        ]
        _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=base_gen))
        assert "Translate to French:\nGood morning 0" in seen

    def test_alpaca_without_input_still_works(self, tmp_path, monkeypatch):
        rows = [{"instruction": f"Say {i}", "output": f"Said {i}"} for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert not _not_run_modes(report)


class TestFlatAndMixedRows:
    def test_flat_question_answer_rows_keep_working(self, tmp_path, monkeypatch):
        rows = [{"question": f"What is {i}?", "answer": f"It is {i}."} for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert not _not_run_modes(report)

    def test_flat_rows_reach_memorization(self, tmp_path, monkeypatch):
        rows = [
            {"question": f"What is {_words(f'n{i}', 10)}?", "answer": _words(f"a{i}", 12)}
            for i in range(12)
        ]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert "scanned=" in report.scores["memorization"].evidence

    def test_mixed_formats_in_one_file(self, tmp_path, monkeypatch):
        seen: list = []

        def base_gen(prompt):
            seen.append(prompt)
            return "ok"

        rows = []
        for i in range(6):
            rows.append(_sharegpt(i))
            rows.append({"instruction": f"Say {i}", "input": "now", "output": f"Said {i}"})
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=base_gen))
        assert not _not_run_modes(report)
        # Both halves reach the probes, not just the format that used to be readable.
        assert any(prompt.startswith("question 0") for prompt in seen)
        assert any(prompt.startswith("Say 0\nnow") for prompt in seen)


class TestVision:
    def test_llava_rows_are_not_run_with_the_image_reason(self, tmp_path, monkeypatch):
        rows = [dict(_sharegpt(i), image=f"img{i}.jpg") for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in DATASET_MODES:
            score = report.scores[mode]
            assert score.verdict == "NOT_RUN", mode
            assert "image" in score.evidence


class TestLongPrompts:
    @staticmethod
    def _rows():
        return [
            {"prompt": f"Write story number {i}.", "completion": f"Once upon a time {i}."}
            for i in range(12)
        ]

    def test_one_long_prompt_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        rows = self._rows()
        rows[0]["prompt"] = "x" * 9000
        collapsed = _gens(adapter_multi=lambda prompt, k: ["same reply"] * k)
        report, _ = _run_live(tmp_path, monkeypatch, rows, collapsed)
        assert report.scores["mode_collapse"].verdict == "MAJOR"  # other rows still scored

    def test_all_prompts_too_long_is_not_run_with_the_limit(self, tmp_path, monkeypatch):
        rows = [dict(row, prompt="x" * 9000) for row in self._rows()]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in SKIPPABLE:
            score = report.scores[mode]
            assert score.verdict == "NOT_RUN", mode
            assert "8192" in score.evidence

    def test_prompt_limit_constant_matches_require_prompts(self):
        from soup_cli.utils.diagnose import _common

        assert _common.MAX_PROMPT_CHARS == 8192
        with pytest.raises(ValueError, match="8192"):
            _common.require_prompts(["x" * (_common.MAX_PROMPT_CHARS + 1)])


class TestSkippedRows:
    def _rows(self):
        rows = [
            {"prompt": f"Write story number {i}.", "completion": f"Once upon a time {i}."}
            for i in range(12)
        ]
        rows[0]["prompt"] = "x" * 9000
        rows += [{"foo": 1}, {"foo": 2}]
        return rows

    def test_skips_are_counted_in_extras(self, tmp_path, monkeypatch):
        report, _ = _run_live(tmp_path, monkeypatch, self._rows(), _gens())
        assert report.extras["rows_skipped"] == "2 unreadable, 1 over-long"

    def test_no_extras_when_nothing_is_skipped(self, tmp_path, monkeypatch):
        rows = [{"prompt": f"Story {i}.", "completion": f"Once {i}."} for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert "rows_skipped" not in report.extras

    def test_cli_prints_the_skipped_note(self, tmp_path, monkeypatch):
        _, out, _ = _cli(tmp_path, monkeypatch, self._rows(), _gens())
        assert "Rows skipped: 2 unreadable, 1 over-long" in strip_ansi(out)


def test_looks_like_json_dataset_keeps_its_call_shape():
    from soup_cli.utils.diagnose.live import _looks_like_json_dataset

    assert _looks_like_json_dataset([{"output": '{"a": 1}'}] * 5) is True
    assert _looks_like_json_dataset([{"output": "plain"}] * 5) is False
    # ShareGPT targets are read through the converters too.
    sharegpt_json = [_sharegpt(i, answer=json.dumps({"n": i})) for i in range(5)]
    assert _looks_like_json_dataset(sharegpt_json) is True


TOOL_ROW = {
    "messages": [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
        {"role": "assistant", "content": "It is sunny in Paris."},
    ],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "d",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
}


def _recording(seen: list):
    def gen(prompt):
        seen.append(prompt)
        return "ok"

    return gen


class TestRobustness:
    def test_malformed_tool_schema_row_does_not_crash(self, tmp_path, monkeypatch):
        bad = {"messages": [{"role": "user", "content": "hi"}],
               "tools": [{"function": "get_weather"}]}
        rows = [
            {"prompt": f"Write story number {i}.", "completion": f"Once upon a time {i}."}
            for i in range(12)
        ] + [bad]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert report.extras["rows_skipped"] == "1 without an answer"
        assert not _not_run_modes(report)

    def test_rows_without_an_answer_are_counted(self, tmp_path, monkeypatch):
        rows = [{"prompt": f"only a prompt {i}"} for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert report.extras["rows_skipped"] == "12 without an answer"
        for mode in SKIPPABLE:
            assert report.scores[mode].verdict == "NOT_RUN"

    def test_tool_calling_rows_give_a_pair_from_the_final_reply(self, tmp_path, monkeypatch):
        seen: list = []
        rows = [TOOL_ROW] * 12
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=_recording(seen)))
        assert not _not_run_modes(report)
        assert any("Weather in Paris?" in prompt for prompt in seen)
        assert "rows_skipped" not in report.extras


class TestOtherFormats:
    def test_audio_rows_are_not_run_with_the_audio_reason(self, tmp_path, monkeypatch):
        rows = [{"audio": f"a{i}.wav", "messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
            for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in DATASET_MODES:
            score = report.scores[mode]
            assert score.verdict == "NOT_RUN", mode
            assert "audio" in score.evidence

    def test_dpo_rows_probe_with_the_prompt(self, tmp_path, monkeypatch):
        seen: list = []
        rows = [{"prompt": f"Pick one {i}", "chosen": f"good {i}", "rejected": f"bad {i}"}
                for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=_recording(seen)))
        assert "Pick one 0" in seen
        assert not _not_run_modes(report)

    def test_kto_rows_probe_with_the_prompt(self, tmp_path, monkeypatch):
        seen: list = []
        rows = [{"prompt": f"Rate {i}", "completion": f"fine {i}", "label": True}
                for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=_recording(seen)))
        assert "Rate 0" in seen
        assert not _not_run_modes(report)

    def test_multi_turn_prompt_stops_at_the_first_assistant_reply(self, tmp_path, monkeypatch):
        seen: list = []
        rows = [{"messages": [
            {"role": "user", "content": f"first question {i}"},
            {"role": "assistant", "content": f"first answer {i}"},
            {"role": "user", "content": f"second question {i}"},
            {"role": "assistant", "content": f"second answer {i}"}]} for i in range(12)]
        _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=_recording(seen)))
        assert "first question 0" in seen
        assert not any("second question" in prompt for prompt in seen)

    def test_alpaca_system_text_is_part_of_the_prompt(self, tmp_path, monkeypatch):
        seen: list = []
        rows = [{"instruction": f"Say {i}", "output": f"Said {i}", "system": "Be terse."}
                for i in range(12)]
        _run_live(tmp_path, monkeypatch, rows, _gens(base_gen=_recording(seen)))
        assert "Be terse.\nSay 0" in seen

    def test_raft_rows_still_feed_citation_the_raw_rows(self, tmp_path, monkeypatch):
        from unittest import mock

        from soup_cli.utils.diagnose.runner import neutral_score

        rows = [{"query": f"What is {i}?", "golden_doc": f"doc {i}", "answer": f"It is {i}."}
                for i in range(12)]
        with mock.patch(
            "soup_cli.utils.diagnose.citation.score_citation",
            return_value=neutral_score("citation", "mock"),
        ) as citation:
            report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        passed = citation.call_args.args[0]
        assert passed[0].get("golden_doc") == "doc 0"  # raw RAFT rows, not converted ones
        assert not _not_run_modes(report)  # the flat query/answer pair is still built


class TestReasons:
    def test_memorization_not_run_reason_names_the_missing_text(self, tmp_path, monkeypatch):
        rows = [{"foo": f"bar {i}"} for i in range(12)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert "no row has text" in report.scores["memorization"].evidence

    def test_dominant_cause_picks_the_reason(self, tmp_path, monkeypatch):
        rows = [dict(_sharegpt(0), image="i.jpg")] + [
            {"text": _words(f"t{i}")} for i in range(11)
        ]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in SKIPPABLE:
            evidence = report.scores[mode].evidence
            assert "plaintext" in evidence and "image" not in evidence

    def test_no_answer_rows_can_be_the_dominant_cause(self, tmp_path, monkeypatch):
        long_row = {"prompt": "x" * 9000, "completion": "done"}
        rows = [long_row] + [{"prompt": f"only a prompt {i}"} for i in range(11)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in SKIPPABLE:
            evidence = report.scores[mode].evidence
            assert "no prompt/answer pair" in evidence and "8192" not in evidence

    def test_no_answer_beats_a_lone_plaintext_row(self, tmp_path, monkeypatch):
        rows = [{"text": _words("t0")}] + [
            {"messages": [{"role": "user", "content": f"question {i}"}]} for i in range(11)
        ]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        for mode in SKIPPABLE:
            evidence = report.scores[mode].evidence
            assert "no prompt/answer pair" in evidence and "plaintext" not in evidence

    def test_unreadable_rows_beat_a_lone_media_row_for_memorization(
        self, tmp_path, monkeypatch
    ):
        rows = [dict(_sharegpt(0), image="i.jpg")] + [{"foo": f"bar {i}"} for i in range(11)]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        evidence = report.scores["memorization"].evidence
        assert "no row has text" in evidence and "image" not in evidence

    def test_mixed_media_kinds_are_named_together(self, tmp_path, monkeypatch):
        rows = [dict(_sharegpt(i), image=f"i{i}.jpg") for i in range(6)] + [
            {"audio": f"a{i}.wav", "messages": [
                {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
            for i in range(6)
        ]
        report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())
        assert report.extras["rows_skipped"] == "12 need audio or image"
        assert "an image or audio" in report.scores["forgetting"].evidence

    def test_over_long_rows_still_reach_memorization(self, tmp_path, monkeypatch):
        seen: list = []
        long_prompt = " ".join(["word"] * 3000)  # ~15000 chars, over the prompt limit
        rows = [{"prompt": long_prompt, "completion": "done"} for _ in range(12)]
        _run_live(tmp_path, monkeypatch, rows, _gens(adapter_gen=_recording(seen)))
        assert max(len(prompt) for prompt in seen) > 3000


def test_dpo_target_is_the_chosen_answer(tmp_path, monkeypatch):
    # JSON in `chosen`, prose in `rejected`: the format probe only fires if chosen is the target.
    rows = [
        {"prompt": f"Give json {i}", "chosen": json.dumps({"n": i}), "rejected": f"plain {i}"}
        for i in range(12)
    ]
    report, _ = _run_live(tmp_path, monkeypatch, rows, _gens())  # adapter never emits JSON
    assert report.scores["format"].verdict == "MAJOR", report.scores["format"]


def test_rows_are_read_before_the_model_loads(tmp_path, monkeypatch):
    import soup_cli.utils.diagnose.live as live

    order: list = []
    real_analyse = live._analyse_rows

    def analyse(rows):
        order.append("rows")
        return real_analyse(rows)

    def fake_pair(base, adapter=None, **kw):
        order.append("model")
        return _gens()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "d.jsonl").write_text(
        "\n".join(json.dumps({"prompt": f"p{i}", "completion": f"c{i}"}) for i in range(12)),
        encoding="utf-8",
    )
    monkeypatch.setattr(live, "_analyse_rows", analyse)
    monkeypatch.setattr(live, "load_adapter_pair", fake_pair)
    live.run_live_diagnose(run_id="r", base="b", adapter="a", dataset_path="d.jsonl")
    assert order == ["rows", "model"]


def test_a_null_byte_prompt_is_skipped_not_fatal(tmp_path, monkeypatch):
    rows = [
        {"prompt": f"Write story number {i}.", "completion": f"Once upon a time {i}."}
        for i in range(12)
    ]
    rows[0]["prompt"] = "nul\x00byte"
    collapsed = _gens(adapter_multi=lambda prompt, k: ["same reply"] * k)
    report, _ = _run_live(tmp_path, monkeypatch, rows, collapsed)
    assert report.scores["mode_collapse"].verdict == "MAJOR", report.scores["mode_collapse"]
    assert report.extras["rows_skipped"] == "1 unreadable"
