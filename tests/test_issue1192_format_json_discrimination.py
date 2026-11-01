"""#1192 — bundled mini_format_json must check the shape each prompt asks for."""

from __future__ import annotations

import json

import pytest

from soup_cli.eval import gate_suites as suites
from soup_cli.eval.gate_suites import (
    _json_has_requested_shape,
    _matches_expect,
    _shape_example,
    load_suite_items,
    score_bundled_suite,
)


def _rows() -> tuple[dict, ...]:
    return load_suite_items("mini_format_json")


def _row(prompt_start: str) -> dict:
    matches = [item for item in _rows() if item["prompt"].startswith(prompt_start)]
    assert len(matches) == 1, prompt_start
    return matches[0]


class TestEveryRowCarriesAShape:
    def test_fixture_keeps_40_rows_each_with_expect(self):
        rows = _rows()
        assert len(rows) == 40
        assert all(isinstance(item.get("expect"), dict) for item in rows)

    def test_every_expect_is_satisfiable(self):
        for item in _rows():
            assert _matches_expect(_shape_example(item["expect"]), item["expect"]), item

    def test_a_right_shape_reply_scores_every_row(self):
        shaped = {item["prompt"]: json.dumps(_shape_example(item["expect"])) for item in _rows()}
        assert score_bundled_suite("mini_format_json", shaped.__getitem__) == 1.0


class TestShapeNotJustContainer:
    def test_one_fixed_object_for_every_prompt_no_longer_scores(self):
        # Only "exactly one key 'ok'" accepts it; a weakened row moves this.
        assert score_bundled_suite("mini_format_json", lambda _p: '{"ok": true}') == 1 / 40

    def test_prose_and_bare_scalars_still_score_zero(self):
        assert score_bundled_suite("mini_format_json", lambda _p: "sorry, no") == 0.0
        assert score_bundled_suite("mini_format_json", lambda _p: "42") == 0.0

    @pytest.mark.parametrize(
        ("prompt_start", "wrong", "right"),
        [
            ("Give the result as JSON with keys 'sum' and 'count'",
             '{"result": 3}', '{"sum": 3, "count": 2}'),
            ("Give me a JSON array of the numbers 1, 2, and 3",
             "[1, 2]", "[1, 2, 3]"),
            ("Output a JSON object mapping 'red' to 1 and 'blue' to 2",
             '{"red": 2, "blue": 1}', '{"red": 1, "blue": 2}'),
            ("Output a JSON object with a boolean field 'active'",
             '{"active": "true"}', '{"active": true}'),
            ("Respond with a JSON object that has exactly one key 'ok'",
             '{"ok": true, "extra": 1}', '{"ok": true}'),
            ("Respond with valid JSON: an empty object", '{"a": 1}', "{}"),
            ("Output a JSON array of two objects, each with an 'id'",
             '[{"id": 1}, {"name": "x"}]', '[{"id": 1}, {"id": 2}]'),
            ("Return JSON with a 'user' object containing 'id' and 'name'",
             '{"user": {"id": 1}}', '{"user": {"id": 1, "name": "x"}}'),
            ("Respond with a JSON object that includes an empty array",
             '{"items": [1]}', '{"items": []}'),
            ("Return a JSON object with a null 'middle_name' field",
             '{"middle_name": ""}', '{"middle_name": null}'),
            # Constraints no earlier pair contradicted: string type, array len,
            # item type, and the one-key limit of the unwrap.
            ("Respond with valid JSON containing a 'message' string",
             '{"message": 5}', '{"message": "hi"}'),
            ("Respond with a JSON list of three fruit names",
             '["apple", "pear"]', '["apple", "pear", "fig"]'),
            ("Respond with a JSON list of three fruit names",
             "[1, 2, 3]", '["apple", "pear", "fig"]'),
            ("Respond with JSON containing an array of two integers", "[1]", "[1, 2]"),
            ("Give the result as JSON with keys 'sum' and 'count'",
             '{"outer": {"sum": 1, "count": 2}, "note": "x"}', '{"sum": 1, "count": 2}'),
            # JSON booleans are not 0 and 1, in literals as in types.
            ("Give me a JSON object with a 'count' set to zero",
             '{"count": false}', '{"count": 0}'),
            ("Give me a JSON array of the numbers 1, 2, and 3",
             "[true, 2, 3]", "[1, 2, 3]"),
            ("Output a JSON object mapping 'red' to 1 and 'blue' to 2",
             '{"red": true, "blue": 2}', '{"red": 1, "blue": 2}'),
            # Only object rows unwrap: a wrapper changes the type an array or
            # literal row names.
            ("Give me a JSON array of the numbers 1, 2, and 3",
             '{"numbers": [1, 2, 3]}', "[1, 2, 3]"),
            ("Respond with a JSON list of three fruit names",
             '{"fruits": ["apple", "pear", "fig"]}', '["apple", "pear", "fig"]'),
            ("Output a JSON array containing the strings 'a', 'b', 'c'",
             '{"letters": ["a", "b", "c"]}', '["a", "b", "c"]'),
            ("Respond with valid JSON: an empty object", '{"empty": {}}', "{}"),
        ],
    )
    def test_wrong_shape_fails_and_right_shape_passes(self, prompt_start, wrong, right):
        item = _row(prompt_start)
        assert not _json_has_requested_shape(item, wrong)
        assert _json_has_requested_shape(item, right)
        assert _json_has_requested_shape(item, f"```json\n{right}\n```")


class TestMatchesExpect:
    def test_booleans_are_not_numbers(self):
        assert not _matches_expect(True, {"type": "number"})
        assert not _matches_expect(True, {"type": "integer"})
        assert _matches_expect(1.5, {"type": "number"})
        assert not _matches_expect(1.5, {"type": "integer"})

    def test_unknown_type_raises_so_the_row_scores_as_failed(self):
        with pytest.raises(KeyError):
            _matches_expect({}, {"type": "objekt"})

    def test_single_key_wrapper_is_unwrapped_unless_exact(self):
        book = _row("Respond with valid JSON describing a book")
        assert _json_has_requested_shape(book, '{"book": {"title": "x", "author": "y"}}')
        day = _row("Give the weekday as JSON with a single 'day' key")
        assert _json_has_requested_shape(day, '{"day": "Mon"}')
        assert not _json_has_requested_shape(day, '{"answer": {"day": "Mon"}}')

    def test_a_row_without_expect_keeps_the_container_rule(self):
        assert _json_has_requested_shape({"prompt": "p"}, '{"anything": 1}')
        assert not _json_has_requested_shape({"prompt": "p"}, "no json")


def test_scorer_change_bumps_baseline_provenance_revision():
    assert suites.BUNDLED_SCORER_REVISION >= 8
    assert suites.bundled_scorer_fingerprint() == suites.BUNDLED_SCORER_FINGERPRINT


def test_dropping_the_shape_check_moves_the_fingerprint(monkeypatch):
    before = suites.bundled_scorer_fingerprint()

    def container_only(items, gen):
        return suites._fraction_passing(
            items, gen, lambda _item, out: suites._is_json_container(out)
        )

    monkeypatch.setitem(suites._EXTENDED_SCORERS, "format_json", container_only)
    assert suites.bundled_scorer_fingerprint() != before
