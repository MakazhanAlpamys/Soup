"""Tests for assistant-only loss masking (v0.36.0 Part A).

Closes the silent-failure mode where Soup relied on TRL's heuristics for
multi-turn loss masking. Mirrors:
- LlamaFactory `processor/supervised.py:88` (IGNORE_INDEX on non-assistant)
- Axolotl `prompt_strategies/chat_template.py:151+` (per-message train field)
"""

from __future__ import annotations

import pytest


class _FakeTokenizer:
    """Character-level fake tokenizer.

    Renders messages as ``<role>:<content>\\n`` and tokenizes each char to
    ``ord(c) % 256``. Two modes for ``apply_chat_template``:

    - ``return_assistant_tokens_mask=True``: returns dict with
      ``{"input_ids": [...], "assistant_masks": [0/1, ...]}`` (preferred path).
    - default: returns string (when ``tokenize=False``) or list[int].

    The ``supports_assistant_mask`` flag toggles whether the dict path is
    available — lets us exercise both the preferred and fallback strategies.
    """

    eos_token_id = 0
    pad_token_id = 0
    chat_template = "fake"

    def __init__(self, supports_assistant_mask: bool = True):
        self.supports_assistant_mask = supports_assistant_mask

    def _render(self, messages):
        parts: list[tuple[str, bool]] = []
        for msg in messages:
            prefix = f"<{msg['role']}>:"
            content = msg["content"]
            suffix = "\n"
            parts.append((prefix, False))
            parts.append((content, msg["role"] == "assistant"))
            parts.append((suffix, False))
        text = "".join(p for p, _ in parts)
        ids = [ord(c) % 256 for c in text]
        mask = []
        for piece, is_assistant in parts:
            mask.extend([1 if is_assistant else 0] * len(piece))
        return text, ids, mask

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_assistant_tokens_mask: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        text, ids, mask = self._render(messages)
        if not tokenize:
            return text
        if return_assistant_tokens_mask and return_dict:
            if not self.supports_assistant_mask:
                raise TypeError(
                    "this tokenizer does not support return_assistant_tokens_mask"
                )
            return {"input_ids": ids, "assistant_masks": mask}
        return ids


class _StaticTokenizer:
    def __init__(self, output):
        self.output = output

    def apply_chat_template(self, *_args, **_kwargs):
        return self.output


class _BatchEncodingTokenizer(_FakeTokenizer):
    def __init__(self, *, zero_mask: bool = False):
        super().__init__()
        self.zero_mask = zero_mask
        self.fallback_calls = 0

    def apply_chat_template(self, messages, **kwargs):
        from transformers import BatchEncoding

        _text, ids, mask = self._render(messages)
        if kwargs.get("return_assistant_tokens_mask"):
            if self.zero_mask:
                mask = [0] * len(ids)
            return BatchEncoding(
                {"input_ids": ids, "assistant_masks": mask}
            )
        self.fallback_calls += 1
        return BatchEncoding({"input_ids": ids})


# ---------------------------------------------------------------------------
# Schema field
# ---------------------------------------------------------------------------


class TestSchemaFields:
    def test_train_on_responses_only_default_true(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl")
        assert cfg.train_on_responses_only is True

    def test_train_on_messages_with_train_field_default_false(self):
        from soup_cli.config.schema import DataConfig

        cfg = DataConfig(train="data.jsonl")
        assert cfg.train_on_messages_with_train_field is False

    def test_train_field_requires_responses_only_disabled(self):
        """Per-message 'train' field is mutually exclusive with response-only mode."""
        from soup_cli.config.schema import DataConfig

        with pytest.raises(ValueError, match="mutually exclusive"):
            DataConfig(
                train="data.jsonl",
                train_on_responses_only=True,
                train_on_messages_with_train_field=True,
            )


# ---------------------------------------------------------------------------
# Loss-mask module
# ---------------------------------------------------------------------------


class TestIgnoreIndex:
    def test_ignore_index_is_minus_100(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX

        assert IGNORE_INDEX == -100


class TestPreferredPath:
    """When tokenizer supports ``return_assistant_tokens_mask=True``."""

    def test_assistant_only_single_turn(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _FakeTokenizer(supports_assistant_mask=True)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        out = build_assistant_only_labels(messages, tok, max_length=2048)
        assert "input_ids" in out
        assert "labels" in out
        assert "attention_mask" in out
        assert len(out["labels"]) == len(out["input_ids"])
        # Exactly the assistant content tokens should NOT be -100.
        # Render: "<user>:Hello\n<assistant>:Hi\n" — "Hi" is 2 chars.
        non_masked = [lab for lab in out["labels"] if lab != IGNORE_INDEX]
        assert len(non_masked) == 2

    def test_assistant_only_multi_turn(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _FakeTokenizer(supports_assistant_mask=True)
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        out = build_assistant_only_labels(messages, tok)
        non_masked = [lab for lab in out["labels"] if lab != IGNORE_INDEX]
        # "A1" + "A2" = 4 chars
        assert len(non_masked) == 4

    def test_truncation_to_max_length(self):
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _FakeTokenizer(supports_assistant_mask=True)
        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ]
        out = build_assistant_only_labels(messages, tok, max_length=128)
        assert len(out["input_ids"]) == 128
        assert len(out["labels"]) == 128
        assert len(out["attention_mask"]) == 128

    def test_real_batch_encoding_is_accepted(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _BatchEncodingTokenizer()
        out = build_assistant_only_labels(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            tok,
        )

        assert tok.fallback_calls == 0
        assert any(label != IGNORE_INDEX for label in out["labels"])
        assert all(type(token_id) is int for token_id in out["input_ids"])

    def test_all_zero_mask_with_assistant_falls_back(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _BatchEncodingTokenizer(zero_mask=True)
        out = build_assistant_only_labels(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            tok,
        )

        assert tok.fallback_calls > 0
        assert any(label != IGNORE_INDEX for label in out["labels"])

    def test_all_zero_mask_without_assistant_is_valid(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _BatchEncodingTokenizer(zero_mask=True)
        out = build_assistant_only_labels(
            [{"role": "user", "content": "Q"}], tok
        )

        assert tok.fallback_calls == 0
        assert all(label == IGNORE_INDEX for label in out["labels"])


class TestFallbackPath:
    """When tokenizer does NOT support ``return_assistant_tokens_mask``."""

    def test_fallback_single_turn(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _FakeTokenizer(supports_assistant_mask=False)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        out = build_assistant_only_labels(messages, tok)
        non_masked = [lab for lab in out["labels"] if lab != IGNORE_INDEX]
        # The fallback marks the *delta* tokens of each assistant turn — that
        # delta is `<assistant>:World\n` = 17 chars (prefix+content+newline).
        # The exact count depends on the renderer, but non-masked must contain
        # 'W','o','r','l','d' chars at minimum.
        assert len(non_masked) >= 5
        # All masked positions must be inside the assistant turn region.
        labels = out["labels"]
        # Check that the user content is masked. user "Hello" chars are at
        # positions 7..12 (after "<user>:"). Verify those are -100.
        for pos in range(7, 12):
            assert labels[pos] == IGNORE_INDEX

    def test_fallback_no_assistant_returns_all_masked(self):
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _FakeTokenizer(supports_assistant_mask=False)
        messages = [{"role": "user", "content": "no answer"}]
        out = build_assistant_only_labels(messages, tok)
        assert all(lab == IGNORE_INDEX for lab in out["labels"])

    def test_fallback_strict_assistant_only(self):
        """Fallback may include extra prefix tokens; strict mode keeps only
        the *content* delta against the next user/system turn."""
        from soup_cli.data.loss_mask import IGNORE_INDEX, build_assistant_only_labels

        tok = _FakeTokenizer(supports_assistant_mask=False)
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Q2"},
        ]
        out = build_assistant_only_labels(messages, tok)
        # The user "Q2" must remain masked.
        # Render: "<user>:Q\n<assistant>:A\n<user>:Q2\n"
        # Last 4 chars are "<user>:Q2\n" prefix+content+newline part of the
        # tail — those must all be -100.
        labels = out["labels"]
        assert labels[-1] == IGNORE_INDEX  # newline
        assert labels[-2] == IGNORE_INDEX  # '2'
        assert labels[-3] == IGNORE_INDEX  # 'Q'


class TestTokenIdNormalisation:
    def test_real_batch_encoding_returns_values_not_mapping_keys(self):
        from transformers import BatchEncoding

        from soup_cli.data.loss_mask import _tokenize_only

        encoded = BatchEncoding({"input_ids": [1, 2, 3]})
        assert not isinstance(encoded, dict)

        ids = _tokenize_only(
            _StaticTokenizer(encoded), [{"role": "user", "content": "Q"}]
        )
        assert ids == [1, 2, 3]
        assert all(type(token_id) is int for token_id in ids)

    def test_tensor_ids_are_normalised_to_python_ints(self):
        import torch

        from soup_cli.data.loss_mask import _tokenize_only

        ids = _tokenize_only(
            _StaticTokenizer(torch.tensor([4, 5, 6])),
            [{"role": "user", "content": "Q"}],
        )
        assert ids == [4, 5, 6]
        assert all(type(token_id) is int for token_id in ids)

    def test_missing_input_ids_raises(self):
        from transformers import BatchEncoding

        from soup_cli.data.loss_mask import _tokenize_only

        with pytest.raises(ValueError, match="input_ids"):
            _tokenize_only(
                _StaticTokenizer(BatchEncoding({"attention_mask": [1, 1]})),
                [{"role": "user", "content": "Q"}],
            )

    def test_non_integer_token_ids_raise(self):
        from transformers import BatchEncoding

        from soup_cli.data.loss_mask import _tokenize_only

        with pytest.raises(ValueError, match="non-integer input_ids"):
            _tokenize_only(
                _StaticTokenizer(
                    BatchEncoding({"input_ids": ["input_ids"]})
                ),
                [{"role": "user", "content": "Q"}],
            )


class TestTokenizerMappingAudit:
    @staticmethod
    def _messages():
        return [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]

    def test_show_mask_path_accepts_real_batch_encoding(self):
        from soup_cli.utils.data_doctor import _build_row_labels

        messages = self._messages()
        built = _build_row_labels(
            _BatchEncodingTokenizer(),
            messages,
            max_length=16,
            train_on_responses_only=False,
            train_on_messages_with_train_field=False,
            include_eot=False,
        )

        assert built["input_ids"]
        assert all(type(token_id) is int for token_id in built["input_ids"])

    def test_generation_marker_check_accepts_real_batch_encoding(self):
        from soup_cli.utils.data_doctor import check_generation_markers

        assert check_generation_markers(_BatchEncodingTokenizer()).verdict == "OK"

    def test_truncation_check_measures_batch_encoding_values(self):
        from soup_cli.utils.data_doctor import check_truncation_risk

        result = check_truncation_risk(
            _BatchEncodingTokenizer(),
            [{"messages": self._messages()}],
            max_length=1,
        )

        assert result.verdict == "MAJOR"
        assert "p95 length = 1 tokens" not in result.message


class TestPerMessageTrainField:
    def test_train_field_overrides_default(self):
        from soup_cli.data.loss_mask import (
            IGNORE_INDEX,
            build_per_message_train_labels,
        )

        tok = _FakeTokenizer(supports_assistant_mask=False)
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1", "train": False},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "train": True},
        ]
        out = build_per_message_train_labels(messages, tok)
        # Only A2's content tokens should NOT be IGNORE_INDEX
        non_masked = [lab for lab in out["labels"] if lab != IGNORE_INDEX]
        # At minimum 'A','2' (both content chars).
        assert len(non_masked) >= 2

    def test_train_field_default_when_missing(self):
        """Missing 'train' field → role==assistant default."""
        from soup_cli.data.loss_mask import (
            IGNORE_INDEX,
            build_per_message_train_labels,
        )

        tok = _FakeTokenizer(supports_assistant_mask=False)
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},  # no 'train' field
        ]
        out = build_per_message_train_labels(messages, tok)
        # Default-include assistant when no flag.
        non_masked = [lab for lab in out["labels"] if lab != IGNORE_INDEX]
        assert len(non_masked) >= 1


class TestEdgeCases:
    def test_empty_messages_raises(self):
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _FakeTokenizer()
        with pytest.raises(ValueError, match="empty"):
            build_assistant_only_labels([], tok)

    def test_max_length_must_be_positive(self):
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _FakeTokenizer()
        messages = [{"role": "user", "content": "x"}]
        with pytest.raises(ValueError, match="max_length"):
            build_assistant_only_labels(messages, tok, max_length=0)

    def test_max_length_rejects_bool(self):
        """`bool` is a subclass of `int` — guard like v0.30.0 Candidate."""
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _FakeTokenizer()
        messages = [{"role": "user", "content": "x"}]
        with pytest.raises(ValueError, match="max_length"):
            build_assistant_only_labels(messages, tok, max_length=True)

    def test_per_message_max_length_truncates(self):
        from soup_cli.data.loss_mask import build_per_message_train_labels

        tok = _FakeTokenizer(supports_assistant_mask=False)
        messages = [
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500, "train": True},
        ]
        out = build_per_message_train_labels(messages, tok, max_length=32)
        assert len(out["input_ids"]) == 32
        assert len(out["labels"]) == 32
        assert len(out["attention_mask"]) == 32

    def test_tokenizer_without_chat_template_raises(self):
        """Hard-fail when tokenizer has no chat_template."""
        from soup_cli.data.loss_mask import build_assistant_only_labels

        class _NoTemplate:
            chat_template = None

            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("tokenizer has no chat_template")

        with pytest.raises(ValueError, match="chat_template"):
            build_assistant_only_labels(
                [{"role": "user", "content": "x"}], _NoTemplate()
            )


# ---------------------------------------------------------------------------
# build_format_row factory (sft.py wiring)
# ---------------------------------------------------------------------------


class TestBuildFormatRow:
    @staticmethod
    def _row():
        return {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ]
        }

    def test_default_responses_only_returns_input_ids_labels(self):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        tok = _FakeTokenizer(supports_assistant_mask=True)
        cfg = DataConfig(train="data.jsonl")  # default train_on_responses_only=True
        fn = build_format_row(tok, cfg, console=None)
        out = fn(self._row())
        assert "input_ids" in out
        assert "labels" in out
        assert "attention_mask" in out

    def test_per_message_train_field_path(self):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        tok = _FakeTokenizer(supports_assistant_mask=False)
        cfg = DataConfig(
            train="data.jsonl",
            train_on_responses_only=False,
            train_on_messages_with_train_field=True,
        )
        fn = build_format_row(tok, cfg, console=None)
        out = fn(self._row())
        assert "labels" in out

    def test_legacy_text_path_when_both_false(self):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        tok = _FakeTokenizer(supports_assistant_mask=True)
        cfg = DataConfig(
            train="data.jsonl",
            train_on_responses_only=False,
            train_on_messages_with_train_field=False,
        )
        fn = build_format_row(tok, cfg)
        out = fn(self._row())
        assert "text" in out
        assert "input_ids" not in out

    def test_no_chat_template_calling_format_row_raises(self):
        """v0.36.0 Part C: previous silent fallback now raises ValueError."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        class _NoTemplate:
            chat_template = None

            def apply_chat_template(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("must not be called")

        cfg = DataConfig(train="data.jsonl")  # default responses_only=True
        # Factory still returns a callable; calling it on a templateless
        # tokenizer raises. (Factory falls back to legacy path which now
        # hard-errors instead of building f-string concat.)
        fn = build_format_row(_NoTemplate(), cfg, console=None)
        with pytest.raises(ValueError, match="chat_template"):
            fn(self._row())

    def test_max_length_threaded_through(self):
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        tok = _FakeTokenizer(supports_assistant_mask=True)
        cfg = DataConfig(train="data.jsonl", max_length=64)
        fn = build_format_row(tok, cfg)
        long_row = {
            "messages": [
                {"role": "user", "content": "x" * 1000},
                {"role": "assistant", "content": "y" * 1000},
            ]
        }
        out = fn(long_row)
        assert len(out["input_ids"]) == 64
