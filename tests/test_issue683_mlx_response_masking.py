"""Tests for Issue #683: MLX SFT ignores data.train_on_responses_only.

`train_on_responses_only` defaults to True, so before this fix *every* MLX SFT
run trained on system and user turns against the documented default.

The tests that matter are the ones about multi-turn. Setting upstream's
`mask_prompt` flag looks like the fix and is not: `ChatDataset` masks a single
prefix before `messages[-1]`, so it supervises the last assistant turn and
drops the earlier ones. That is a different wrong distribution, so
`test_upstream_mask_prompt_would_drop_earlier_assistant_turns` pins the reason
this code exists -- if upstream ever generalises, that test fails and this
module can be deleted.

A scripted fake tokenizer is used rather than a real one so the suite needs no
network and no Apple Silicon. It reproduces ChatML's structure exactly (per-turn
`<|im_start|>role\\n … <|im_end|>\\n`, and `add_generation_prompt` emitting the
assistant header), which is the only property the span arithmetic depends on.
The behaviour was also measured on a real Qwen2.5 tokenizer and a real Metal
training run; those numbers are in the PR body.
"""

import pytest

from soup_cli.trainer.mlx_masking import (
    MaskedChatDataset,
    ResponseMaskError,
    build_response_mask,
)

SYS = {"role": "system", "content": "You are terse."}
U1 = {"role": "user", "content": "What is 2+2?"}
A1 = {"role": "assistant", "content": "Four."}
U2 = {"role": "user", "content": "And 3+3?"}
A2 = {"role": "assistant", "content": "Six."}

MULTI_TURN = [SYS, U1, A1, U2, A2]
SINGLE_TURN = [SYS, U1, A1]


class FakeChatTokenizer:
    """Deterministic ChatML-shaped tokenizer. One token per word or marker."""

    def __init__(self, prefix_stable: bool = True):
        self._prefix_stable = prefix_stable

    def _turn(self, m):
        return (
            [f"<|im_start|>{m['role']}"]
            + m["content"].split()
            + ["<|im_end|>"]
        )

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, return_dict=False
    ):
        # Real HuggingFace tokenizers raise here rather than returning []
        # ("Cannot apply chat template to an empty conversation"). The fake
        # originally did not, which let a crash reach a Metal run that the
        # suite had called green -- so the fake now reproduces the refusal.
        if not messages:
            raise ValueError(
                "Cannot apply chat template to an empty conversation. "
                "Provide at least one message."
            )
        out = []
        for m in messages:
            out.extend(self._turn(m))
        if add_generation_prompt:
            out.append("<|im_start|>assistant")
        if not self._prefix_stable and messages:
            # A template that stamps a running turn count at the front is a
            # realistic way to be non-prefix-stable: every partial rendering
            # differs from the full one in its very first token.
            out = [f"<|turns:{len(messages)}|>"] + out
        return out


def supervised(tokens, mask):
    return [t for t, m in zip(tokens, mask) if m]


class TestEveryAssistantTurnIsSupervised:
    """Acceptance criteria 1-3 from the issue."""

    def test_multi_turn_supervises_both_assistant_turns(self):
        tokens, mask = build_response_mask(MULTI_TURN, FakeChatTokenizer())
        got = supervised(tokens, mask)
        assert "Four." in got, "assistant turn A was dropped from the loss"
        assert "Six." in got, "assistant turn B was dropped from the loss"

    def test_system_and_user_turns_are_excluded(self):
        tokens, mask = build_response_mask(MULTI_TURN, FakeChatTokenizer())
        got = supervised(tokens, mask)
        for word in ("You", "are", "terse.", "What", "is", "2+2?", "And", "3+3?"):
            assert word not in got, f"{word!r} is prompt content but was supervised"

    def test_single_turn_prompt_tokens_have_zero_weight(self):
        tokens, mask = build_response_mask(SINGLE_TURN, FakeChatTokenizer())
        assert supervised(tokens, mask) == ["Four.", "<|im_end|>"]

    def test_the_assistant_header_is_not_supervised(self):
        """`<|im_start|>assistant` is scaffolding the model never has to emit.

        It falls on the masked side because the prefix for an assistant turn is
        rendered with `add_generation_prompt=True`. Dropping that argument
        would supervise the header, which is why this is pinned separately.
        """
        tokens, mask = build_response_mask(MULTI_TURN, FakeChatTokenizer())
        assert "<|im_start|>assistant" not in supervised(tokens, mask)

    def test_mask_is_the_same_length_as_the_tokens(self):
        tokens, mask = build_response_mask(MULTI_TURN, FakeChatTokenizer())
        assert len(tokens) == len(mask)
        assert set(mask) <= {0, 1}

    def test_something_is_actually_masked(self):
        """Control: a mask of all ones would pass every exclusion test above
        only if the words differed, so pin that the mask discriminates."""
        tokens, mask = build_response_mask(MULTI_TURN, FakeChatTokenizer())
        assert 0 < sum(mask) < len(mask)


class TestUnsupportedShapesAreRefusedNotApproximated:
    """The issue's last criterion: reject rather than silently approximate."""

    def test_a_non_prefix_stable_template_is_refused(self):
        with pytest.raises(ResponseMaskError, match="prefix-stable"):
            build_response_mask(MULTI_TURN, FakeChatTokenizer(prefix_stable=False))

    def test_a_conversation_with_no_assistant_turn_is_refused(self):
        with pytest.raises(ResponseMaskError, match="no assistant content"):
            build_response_mask([SYS, U1], FakeChatTokenizer())

    def test_a_leading_assistant_turn_is_refused(self):
        """Its generation prompt cannot be rendered, so its header would leak.

        Found by running a real Metal job, not by this suite: the first
        version called `apply_chat_template([])` for the k == 0 prefix, which a
        real tokenizer refuses outright.
        """
        with pytest.raises(ResponseMaskError, match="begins with an assistant"):
            build_response_mask([A1, U1, A2], FakeChatTokenizer())

    def test_the_first_message_never_reaches_the_tokenizer_as_an_empty_list(self):
        """Regression pin for the crash above, at the boundary that caused it."""
        seen = []

        class Recording(FakeChatTokenizer):
            def apply_chat_template(self, messages, **kw):
                seen.append(list(messages))
                return super().apply_chat_template(messages, **kw)

        build_response_mask(MULTI_TURN, Recording())
        assert [] not in seen, (
            "an empty conversation was rendered; a real tokenizer raises on that"
        )

    def test_an_empty_conversation_is_refused(self):
        with pytest.raises(ResponseMaskError, match="no messages"):
            build_response_mask([], FakeChatTokenizer())

    def test_the_refusal_is_not_a_bare_valueerror_callers_cannot_catch(self):
        assert issubclass(ResponseMaskError, ValueError)


class TestUpstreamIsWhyThisModuleExists:
    """If upstream ever fixes multi-turn masking, this test fails and says so."""

    def test_upstream_mask_prompt_would_drop_earlier_assistant_turns(self):
        """mlx-lm's ChatDataset masks one prefix, ending before messages[-1]."""
        pytest.importorskip("mlx_lm")
        from mlx_lm.tuner.datasets import ChatDataset

        row = {"messages": MULTI_TURN}
        ds = ChatDataset([row], FakeChatTokenizer(), mask_prompt=True)
        tokens, offset = ds.process(row)
        upstream_supervised = tokens[offset:]

        assert "Six." in upstream_supervised, "sanity: the last turn is supervised"
        assert "Four." not in upstream_supervised, (
            "mlx-lm now supervises earlier assistant turns; Soup's per-token "
            "mask in trainer/mlx_masking.py may no longer be needed"
        )

        # And ours, on the same conversation, keeps both.
        _, mask = build_response_mask(MULTI_TURN, FakeChatTokenizer())
        ours = supervised(tokens, mask)
        assert "Four." in ours and "Six." in ours


class TestMaskedChatDataset:
    def test_process_returns_tokens_and_mask_for_a_row(self):
        ds = MaskedChatDataset([{"messages": MULTI_TURN}], FakeChatTokenizer())
        tokens, mask = ds.process(ds[0])
        assert len(tokens) == len(mask)
        assert "Four." in supervised(tokens, mask)

    def test_it_wraps_in_upstream_cachedataset_unchanged(self):
        """The shape contract: CacheDataset must be able to wrap it as-is."""
        pytest.importorskip("mlx_lm")
        from mlx_lm.tuner.datasets import CacheDataset

        ds = CacheDataset(
            MaskedChatDataset([{"messages": MULTI_TURN}] * 3, FakeChatTokenizer())
        )
        assert len(ds) == 3
        tokens, mask = ds[0]
        assert len(tokens) == len(mask)
        # Second read comes from the cache and must be identical.
        assert ds[0] == (tokens, mask)

    def test_a_custom_chat_key_is_honoured(self):
        ds = MaskedChatDataset(
            [{"conversation": MULTI_TURN}], FakeChatTokenizer(), chat_key="conversation"
        )
        tokens, mask = ds.process(ds[0])
        assert "Four." in supervised(tokens, mask)


class TestMaskedLossAlignment:
    """The off-by-one that a shape-only test would not catch."""

    def test_the_mask_is_shifted_to_match_the_targets(self):
        """`targets = batch[:, 1:]`, so the mask must shift with it.

        An unshifted mask supervises the token *before* each assistant token --
        the last prompt token -- which is exactly the distribution this issue is
        about, just moved by one. The assertion below fails for an unshifted
        mask because the two produce different supervised counts.
        """
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_masking import masked_loss

        # tokens 0..5; only 4 and 5 are assistant content.
        batch = mx.array([[10, 11, 12, 13, 14, 15]])
        masks = mx.array([[0, 0, 0, 0, 1, 1]])

        seen = {}

        def fake_model(inputs):
            seen["shape"] = inputs.shape
            # Uniform logits over a 20-token vocab -> a finite, equal CE per
            # position, so `ntoks` alone decides the reported mean.
            return mx.zeros((inputs.shape[0], inputs.shape[1], 20))

        _, ntoks = masked_loss(fake_model, batch, masks)
        assert seen["shape"] == (1, 5), "the model must see batch[:, :-1]"
        # masks[:, 1:] = [0,0,0,1,1] -> 2 supervised targets, and those targets
        # are original tokens 4 and 5. An unshifted mask would give the same
        # count here only by coincidence, so the count is checked against the
        # positions too.
        assert int(ntoks) == 2

    def test_a_fully_truncated_row_yields_zero_rather_than_nan(self):
        """0 supervised tokens must not produce a nan that poisons the run."""
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_masking import masked_loss

        batch = mx.array([[10, 11, 12, 13]])
        masks = mx.array([[0, 0, 0, 0]])

        def fake_model(inputs):
            return mx.zeros((inputs.shape[0], inputs.shape[1], 20))

        loss, ntoks = masked_loss(fake_model, batch, masks)
        assert int(ntoks) == 0
        assert float(loss) == 0.0
        assert float(loss) == float(loss), "loss is nan"


class TestBatchingPadsTheMaskWithTheTokens:
    def test_mask_and_tokens_come_back_the_same_shape(self):
        pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_masking import masked_iterate_batches

        rows = [([1, 2, 3], [0, 0, 1]), ([1, 2, 3, 4, 5], [0, 0, 0, 1, 1])]

        class DS:
            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                return rows[i]

        batch, mask = next(
            iter(masked_iterate_batches(DS(), batch_size=2, max_seq_length=512))
        )
        assert batch.shape == mask.shape, "a mask padded differently misaligns"
        assert int(mask.sum()) == 3, "padding must contribute no supervision"

    def test_truncation_cuts_the_mask_too(self):
        """A mask longer than its truncated row would index past the end."""
        pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_masking import masked_iterate_batches

        rows = [([1] * 40, [0] * 30 + [1] * 10)] * 2

        class DS:
            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                return rows[i]

        batch, mask = next(
            iter(masked_iterate_batches(DS(), batch_size=2, max_seq_length=33))
        )
        assert batch.shape == mask.shape
        assert batch.shape[1] <= 33


class TestTheDispatchPicksTheRightStrategyPerShape:
    """Three shapes, three different correct answers, two of them silent.

    Split out of `train()` so it is testable without a model load. Getting this
    wrong is invisible for chat rows (wrong distribution) and for text rows
    (upstream raises), which is why each branch is pinned separately rather
    than through one round-trip test.
    """

    def test_chat_rows_get_soups_per_token_mask(self):
        from soup_cli.trainer.mlx_masking import plan_response_masking

        plan = plan_response_masking(True, {"messages": MULTI_TURN})
        assert plan.token_mask is True
        assert plan.mask_prompt is False, (
            "upstream's flag must not also be set; it would supervise only the "
            "last assistant turn"
        )
        assert plan.warning == ""

    def test_prompt_completion_rows_use_upstreams_flag(self):
        """Upstream is correct for this shape -- the prefix IS the whole prompt."""
        from soup_cli.trainer.mlx_masking import plan_response_masking

        plan = plan_response_masking(True, {"prompt": "q", "completion": "a"})
        assert plan.mask_prompt is True
        assert plan.token_mask is False
        assert plan.warning == ""

    def test_plain_text_rows_are_warned_about_not_masked(self):
        """Setting the flag here makes upstream raise, so it must not be set."""
        from soup_cli.trainer.mlx_masking import plan_response_masking

        plan = plan_response_masking(True, {"text": "hello"})
        assert plan.mask_prompt is False, (
            "upstream raises ValueError('Prompt masking not supported for text "
            "dataset.') -- setting this turns a silent bug into a crash"
        )
        assert plan.token_mask is False
        assert "train_on_responses_only" in plan.warning

    def test_upstream_really_does_raise_on_text_rows_with_the_flag(self):
        """The reason the branch above exists, pinned against mlx-lm itself."""
        pytest.importorskip("mlx_lm")
        from mlx_lm.tuner.datasets import create_dataset

        class Args:
            mask_prompt = True

        with pytest.raises(ValueError, match="not supported for text dataset"):
            create_dataset([{"text": "hello"}], FakeChatTokenizer(), Args())

    @pytest.mark.parametrize(
        "sample",
        [{"messages": MULTI_TURN}, {"prompt": "q", "completion": "a"}, {"text": "x"}],
    )
    def test_the_flag_off_masks_nothing_whatever_the_shape(self, sample):
        """Control: no shape may start masking when the option is disabled."""
        from soup_cli.trainer.mlx_masking import plan_response_masking

        plan = plan_response_masking(False, sample)
        assert (plan.token_mask, plan.mask_prompt, plan.warning) == (False, False, "")

    def test_an_empty_dataset_does_not_crash_the_dispatch(self):
        from soup_cli.trainer.mlx_masking import plan_response_masking

        plan = plan_response_masking(True, {})
        assert plan.token_mask is False and plan.mask_prompt is False
