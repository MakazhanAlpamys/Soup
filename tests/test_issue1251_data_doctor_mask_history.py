"""#1251: ``soup data doctor`` ignored ``data.mask_history`` — the compat
report and the ``--show-mask`` preview trained every assistant turn while a
``mask_history: true`` run trains only the last.

The fix threads a ``mask_history`` flag through ``_build_row_labels`` and
every caller that already takes ``train_on_responses_only``, adds a
``--mask-history/--no-mask-history`` CLI flag (default off, matching
``DataConfig``), and refuses ``--mask-history`` together with
``--no-train-on-responses-only`` — mirroring the soup.yaml schema rule
``data.mask_history requires data.train_on_responses_only: true``.

Like #761's tests, these count **trained label positions**, never a flag: a
boolean assertion would not catch the flag being accepted and silently
ignored (the exact bug this issue reports).
"""

import json

import pytest

from soup_cli.data.loss_mask import IGNORE_INDEX

pytestmark = pytest.mark.unit

_SPECIALS = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>"]
_WORDS = [
    "Hi", "Hello", "there", "What", "is", "two", "plus", "Four", "Three",
    "and", "one", "more", "Thanks", "Welcome", "you", "are",
]

# Plain template (no {% generation %}) and the generation-marker variant —
# build_assistant_only_labels reaches its labels by different code on the two.
_BODY = (
    "{% for m in messages %}<|{{ m['role'] }}|> "
    "{{ m['content'] }} <|end|> {% endfor %}"
)
_BODY_WITH_GENERATION = (
    "{% for m in messages %}<|{{ m['role'] }}|> "
    "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}"
    "{% endgeneration %}{% else %}{{ m['content'] }}{% endif %}"
    " <|end|> {% endfor %}"
)
_TEMPLATES = ["generation_markers", "no_markers"]

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


def _row(messages):
    return {"messages": [dict(m) for m in messages]}


def _trained_positions(tok, messages, *, mask_history):
    """Count trained label positions via the --show-mask preview path."""
    from soup_cli.utils.data_doctor import render_mask_preview

    previews = render_mask_preview(
        [_row(messages)], tok, fmt="chatml", n=1, max_length=2048,
        train_on_responses_only=True, mask_history=mask_history,
    )
    assert len(previews) == 1
    return sum(1 for token in previews[0].tokens if token.trained)


def _assistant_token_count(tok, content):
    return len(tok(content, add_special_tokens=False)["input_ids"])


def _expected_trained_positions(messages, tok, *, mask_history):
    """Ground truth from the SAME builder the trainer uses."""
    from soup_cli.data.loss_mask import build_assistant_only_labels

    labels = build_assistant_only_labels(
        messages, tok, max_length=2048, mask_history=mask_history
    )["labels"]
    return sum(1 for label in labels if label != IGNORE_INDEX)


class TestShowMaskPreviewHonorsMaskHistory:
    @pytest.mark.parametrize("template", _TEMPLATES)
    def test_two_turn_preview_trains_only_the_last_assistant_span(self, template):
        tok = _tokenizer(
            _BODY_WITH_GENERATION if template == "generation_markers" else _BODY
        )
        with_flag = _trained_positions(tok, _TWO_TURN, mask_history=True)
        without_flag = _trained_positions(tok, _TWO_TURN, mask_history=False)
        assert with_flag == _expected_trained_positions(
            _TWO_TURN, tok, mask_history=True
        )
        assert with_flag < without_flag

    @pytest.mark.parametrize("template", _TEMPLATES)
    def test_three_turn_preview_trains_only_the_last_assistant_span(self, template):
        tok = _tokenizer(
            _BODY_WITH_GENERATION if template == "generation_markers" else _BODY
        )
        with_flag = _trained_positions(tok, _THREE_TURN, mask_history=True)
        without_flag = _trained_positions(tok, _THREE_TURN, mask_history=False)
        assert with_flag == _expected_trained_positions(
            _THREE_TURN, tok, mask_history=True
        )
        assert with_flag < without_flag

    def test_default_off_trains_every_assistant_turn(self):
        """The flag's default is off (matches DataConfig): without it the
        preview must keep training every assistant turn. This is also the
        no-op detector — if mask_history were accepted and ignored, the
        two/three-turn assertions above would fail because with_flag would
        equal without_flag."""
        tok = _tokenizer(_BODY)
        without_flag = _trained_positions(tok, _TWO_TURN, mask_history=False)
        assert without_flag == _expected_trained_positions(
            _TWO_TURN, tok, mask_history=False
        )
        assert without_flag > _trained_positions(tok, _TWO_TURN, mask_history=True)

    def test_trained_spans_match_build_assistant_only_labels_bit_for_bit(self):
        """The preview and the trainer's label builder must agree exactly."""
        tok = _tokenizer(_BODY_WITH_GENERATION)
        assert _trained_positions(tok, _THREE_TURN, mask_history=True) == (
            _expected_trained_positions(_THREE_TURN, tok, mask_history=True)
        )


class TestEngineRefusesMaskHistoryWithoutResponsesOnly:
    def test_render_mask_preview_refuses_the_combination(self):
        from soup_cli.utils.data_doctor import render_mask_preview

        tok = _tokenizer(_BODY)
        with pytest.raises(ValueError) as exc_info:
            render_mask_preview(
                [_row(_TWO_TURN)], tok, fmt="chatml", n=1,
                train_on_responses_only=False, mask_history=True,
            )
        message = str(exc_info.value)
        assert "mask_history" in message
        assert "train_on_responses_only" in message

    def test_run_doctor_refuses_the_combination(self):
        from soup_cli.utils.data_doctor import run_doctor

        tok = _tokenizer(_BODY)
        with pytest.raises(ValueError) as exc_info:
            run_doctor(
                [_row(_TWO_TURN)], tok, fmt="chatml",
                train_on_responses_only=False, mask_history=True,
            )
        message = str(exc_info.value)
        assert "mask_history" in message
        assert "train_on_responses_only" in message


class TestCliRefusesMaskHistoryWithoutResponsesOnly:
    def test_combo_is_refused_and_names_both_flags(self, tmp_path, monkeypatch):
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"messages": _TWO_TURN}) + "\n", encoding="utf-8"
        )
        result = CliRunner().invoke(
            app,
            [
                "data", "doctor", str(path), "--model", "fake/model",
                "--mask-history", "--no-train-on-responses-only",
            ],
        )
        assert result.exit_code == 3
        text = re.compile(r"\x1b\[[0-9;]*m").sub("", result.output)
        assert "--mask-history" in text
        assert "--train-on-responses-only" in text
