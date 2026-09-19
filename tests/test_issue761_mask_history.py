"""#761: ``data.mask_history`` was declared, validated, documented — and read by
nothing, so a config that asked to train on the last turn trained on every turn.

The maintainer's ruling (2026-09-18) was to wire it, not to refuse it: the
machinery for excluding spans from the loss already exists in ``data/loss_mask.py``
and ``train_on_responses_only`` is the worked example one axis over.

These tests count **trained tokens**, never a flag. The precedent is v0.73.3, where
an assistant-only label mask built from a mapping's key strings produced 0 trained
tokens, showed an ordinary loss curve, and raised nothing: a boolean assertion
would not have caught it.

Both label-building branches are covered. ``build_assistant_only_labels`` prefers
the template's own ``{% generation %}`` mask and falls back to incremental
rendering when the template has none, and the two reach the labels by different
code, so ``mask_history`` has to be applied on both.
"""

import pytest

from soup_cli.data.loss_mask import IGNORE_INDEX

_SPECIALS = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>"]
_WORDS = [
    "Hi", "Hello", "there", "What", "is", "two", "plus", "Four", "Three",
    "and", "one", "more", "Thanks", "Welcome", "you", "are",
]

_BODY = (
    "{% for m in messages %}<|{{ m['role'] }}|> "
    "{{ m['content'] }} <|end|> {% endfor %}"
)
# The same rendering, with the markers that let the template report its own
# assistant mask -- which is the branch ``_apply_template_with_mask`` prefers.
_BODY_WITH_GENERATION = (
    "{% for m in messages %}<|{{ m['role'] }}|> "
    "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}"
    "{% endgeneration %}{% else %}{{ m['content'] }}{% endif %}"
    " <|end|> {% endfor %}"
)
_TEMPLATES = {"generation_markers": _BODY_WITH_GENERATION, "no_markers": _BODY}

_TWO_TURN = [
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hello there"},
    {"role": "user", "content": "What is two plus two"},
    {"role": "assistant", "content": "Four"},
]
_THREE_TURN = _TWO_TURN + [
    {"role": "user", "content": "and one more"},
    {"role": "assistant", "content": "Thanks you are Welcome"},
]


def _tokenizer(chat_template):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = chat_template
    return tok


def _trained(labels):
    return [label for label in labels if label != IGNORE_INDEX]


def _labels(tok, messages, *, mask_history):
    from soup_cli.data.loss_mask import build_assistant_only_labels

    return build_assistant_only_labels(
        messages, tok, max_length=2048, mask_history=mask_history
    )["labels"]


def _turn_token_count(tok, content):
    return len(tok(content, add_special_tokens=False)["input_ids"])


def _spans(labels):
    """The contiguous trained runs, as ``(start, end)`` half-open pairs."""
    spans, run_start = [], None
    for index, label in enumerate(labels):
        if label != IGNORE_INDEX and run_start is None:
            run_start = index
        elif label == IGNORE_INDEX and run_start is not None:
            spans.append((run_start, index))
            run_start = None
    if run_start is not None:
        spans.append((run_start, len(labels)))
    return spans


# Measured on the fixtures below, per branch. The two branches disagree because
# the template's own ``{% generation %}`` mask covers the assistant *content*
# while incremental rendering attributes the whole delta -- role marker, content
# and terminator -- to the turn that produced it. Both are pre-existing
# behaviour; what #761 adds is that only the last span survives.
_EXPECTED = {
    # template          conversation   trained off   trained on
    ("generation_markers", "two"): (3, 1),
    ("generation_markers", "three"): (7, 4),
    ("no_markers", "two"): (7, 3),
    ("no_markers", "three"): (13, 6),
}


@pytest.mark.parametrize("template", sorted(_TEMPLATES))
class TestTrainedTokenCounts:
    """The numbers, per branch. Each asserts how many label positions survive,
    not that a flag was honoured."""

    def test_two_turns_keeps_only_the_second_assistant_turn(self, template):
        tok = _tokenizer(_TEMPLATES[template])
        off = _labels(tok, _TWO_TURN, mask_history=False)
        on = _labels(tok, _TWO_TURN, mask_history=True)
        expected_off, expected_on = _EXPECTED[(template, "two")]

        assert len(_trained(off)) == expected_off, "sanity: both turns train"
        assert len(_trained(on)) == expected_on
        assert len(_spans(off)) == 2, "sanity: two assistant turns, two spans"
        assert _spans(on) == [_spans(off)[-1]], "the survivor is the last turn"

    def test_three_turns_removes_both_earlier_assistant_turns(self, template):
        tok = _tokenizer(_TEMPLATES[template])
        off = _labels(tok, _THREE_TURN, mask_history=False)
        on = _labels(tok, _THREE_TURN, mask_history=True)
        expected_off, expected_on = _EXPECTED[(template, "three")]

        assert len(_trained(off)) == expected_off
        assert len(_trained(on)) == expected_on
        assert len(_spans(off)) == 3, "sanity: three assistant turns"
        assert _spans(on) == [_spans(off)[-1]]

    def test_a_single_turn_is_unchanged(self, template):
        """The field only means something for a multi-turn shape. One turn is
        already the last one, so the loss must be identical -- a fix that always
        trimmed something would fail here."""
        tok = _tokenizer(_TEMPLATES[template])
        single = _TWO_TURN[:2]

        assert _labels(tok, single, mask_history=True) == _labels(
            tok, single, mask_history=False
        )

    def test_it_never_adds_a_trained_token(self, template):
        """Whatever it does, it is a narrowing: every surviving position was
        already trained without it, in the same place."""
        tok = _tokenizer(_TEMPLATES[template])
        off = _labels(tok, _THREE_TURN, mask_history=False)
        on = _labels(tok, _THREE_TURN, mask_history=True)

        assert len(on) == len(off)
        for index, (a, b) in enumerate(zip(on, off)):
            assert a == IGNORE_INDEX or a == b, f"position {index} changed: {a} != {b}"
        assert any(a == IGNORE_INDEX and b != IGNORE_INDEX for a, b in zip(on, off))


class TestBothBranchesRan:
    """The parametrisation above is worthless if both ids resolve to the same
    branch, so pin which one each template takes."""

    def test_the_generation_template_reports_its_own_mask(self):
        from soup_cli.data.loss_mask import _apply_template_with_mask

        tok = _tokenizer(_BODY_WITH_GENERATION)
        assert _apply_template_with_mask(tok, _TWO_TURN) is not None

    def test_the_plain_template_falls_back_to_incremental_rendering(self):
        from soup_cli.data.loss_mask import _apply_template_with_mask

        tok = _tokenizer(_BODY)
        assert _apply_template_with_mask(tok, _TWO_TURN) is None


class TestTheHelper:
    def test_all_masked_labels_are_returned_unchanged(self):
        """A template that reported no assistant span must not be turned into a
        different silent no-op.

        No mutation of the early return can be killed from here: an all-masked
        list is equal to an all-masked list of the same length however it is
        built. The branch is kept for the object identity and the intent, and is
        reported as an equivalent mutant rather than covered by a test that only
        looks like it covers it."""
        from soup_cli.data.loss_mask import keep_only_the_last_assistant_turn

        labels = [IGNORE_INDEX] * 5
        assert keep_only_the_last_assistant_turn(labels) == labels
        assert keep_only_the_last_assistant_turn([]) == []

    def test_a_trailing_span_that_reaches_the_end_is_kept(self):
        from soup_cli.data.loss_mask import keep_only_the_last_assistant_turn

        labels = [7, 8, IGNORE_INDEX, 9, 10]
        assert keep_only_the_last_assistant_turn(labels) == [
            IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 9, 10
        ]

    def test_a_span_that_starts_at_position_zero_is_kept_whole(self):
        """The only span with nothing masked in front of it. Slicing it with an
        off-by-one leaves a one-token row and no test above notices, because
        every rendered conversation starts with a user turn."""
        from soup_cli.data.loss_mask import keep_only_the_last_assistant_turn

        assert keep_only_the_last_assistant_turn([9, 10]) == [9, 10]

    def test_a_span_followed_by_masked_tail_is_still_the_last_one(self):
        from soup_cli.data.loss_mask import keep_only_the_last_assistant_turn

        labels = [7, IGNORE_INDEX, 9, 10, IGNORE_INDEX]
        assert keep_only_the_last_assistant_turn(labels) == [
            IGNORE_INDEX, IGNORE_INDEX, 9, 10, IGNORE_INDEX
        ]

    def test_a_non_bool_is_refused(self):
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _tokenizer(_BODY)
        with pytest.raises(TypeError, match="mask_history must be bool"):
            build_assistant_only_labels(_TWO_TURN, tok, mask_history=1)


class TestTheConfigRefusals:
    """Ask 4: the interaction is stated, and the undefined combinations are
    refused at parse naming both fields."""

    def test_without_train_on_responses_only_it_is_refused(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import DataConfig

        with pytest.raises(ValidationError) as excinfo:
            DataConfig(
                train="./d.jsonl",
                mask_history=True,
                train_on_responses_only=False,
            )
        message = str(excinfo.value)
        assert "mask_history" in message and "train_on_responses_only" in message

    def test_the_per_message_train_field_combination_is_refused_transitively(self):
        """No clause of its own: ``train_on_messages_with_train_field`` is already
        exclusive with ``train_on_responses_only``, which ``mask_history``
        requires, so neither order of the two can load. A mutation run is what
        found the second clause I first wrote here to be unreachable."""
        from pydantic import ValidationError

        from soup_cli.config.schema import DataConfig

        for extra in ({"train_on_responses_only": True}, {"train_on_responses_only": False}):
            with pytest.raises(ValidationError) as excinfo:
                DataConfig(
                    train="./d.jsonl",
                    mask_history=True,
                    train_on_messages_with_train_field=True,
                    **extra,
                )
            assert "mutually exclusive" in str(excinfo.value) or "mask_history" in str(
                excinfo.value
            )

    def test_the_supported_combination_loads(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(
            train="./d.jsonl", mask_history=True, train_on_responses_only=True
        )
        assert cfg.mask_history is True

    def test_mask_history_off_is_unaffected_by_the_refusal(self):
        """Control: the refusal is about the combination, not about the path."""
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="./d.jsonl", train_on_responses_only=False)
        assert cfg.mask_history is False


class TestTheLiveWiring:
    """The field has to reach the label builder through the real factory, not just
    through a keyword argument nothing passes -- that was the #761 defect."""

    def test_build_format_row_honours_it(self):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        tok = _tokenizer(_BODY)
        rows = {"messages": _THREE_TURN}
        on = build_format_row(
            tok,
            DataConfig(
                train="./d.jsonl",
                format="chatml",
                max_length=2048,
                train_on_responses_only=True,
                mask_history=True,
            ),
        )(rows)["labels"]
        off = build_format_row(
            tok,
            DataConfig(
                train="./d.jsonl",
                format="chatml",
                max_length=2048,
                train_on_responses_only=True,
                mask_history=False,
            ),
        )(rows)["labels"]

        assert (len(_trained(off)), len(_trained(on))) == _EXPECTED[
            ("no_markers", "three")
        ], "data.mask_history reached the label builder through the factory"

    def test_a_config_stand_in_without_the_field_still_builds(self):
        """`build_format_row` is handed duck-typed stand-ins by several suites
        (`test_v0532.py`, `test_issue532_supervised_token_guard.py`), carrying only
        the fields that existed when they were written. Reading
        `data_cfg.mask_history` directly turned six of those into
        `AttributeError` at setup -- found by the full suite, not by any targeted
        run. The read is a `getattr` with the schema default."""
        import types

        tok = _tokenizer(_BODY)
        stand_in = types.SimpleNamespace(
            train_on_responses_only=True,
            train_on_messages_with_train_field=False,
            max_length=2048,
            chat_template=None,
            prompt_strategy=None,
        )
        from soup_cli.data.sft_format import build_format_row

        labels = build_format_row(tok, stand_in)({"messages": _THREE_TURN})["labels"]

        assert len(_trained(labels)) == _EXPECTED[("no_markers", "three")][0], (
            "absent field behaves as mask_history: false"
        )
