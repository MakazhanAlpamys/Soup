"""Exact native message targets and public-wrapper selective-mask dispatch."""

import json
import sys

import pytest

from soup_cli.trainer.mlx_masking import MaskedChatDataset, ResponseMaskError
from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx
from tests.test_issue683_mlx_response_masking import FakeChatTokenizer, supervised


def conversation():
    return [
        {"role": "system", "content": "policy", "train": True},
        {"role": "developer", "content": "ignored", "train": False},
        {"role": "user", "content": "répété", "train": True},
        {"role": "assistant", "content": "répété", "train": False},
        {"role": "tool", "content": "tool-output", "train": True},
        {"role": "assistant", "content": "earlier", "train": None},
        {"role": "assistant", "content": "last"},
    ]


class IntegerTokenizer(FakeChatTokenizer):
    """Token ids with a hand-written whole-turn target oracle, including headers."""

    chat_template = "fixture"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = {}
        self.words = {}

    def apply_chat_template(self, messages, **kwargs):
        kwargs.pop("tokenize", None)
        kwargs.pop("add_special_tokens", None)
        result = super().apply_chat_template(messages, **kwargs)
        for word in result:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
                self.words[self.vocab[word]] = word
        return [self.vocab[word] for word in result]


def selected(messages, **mode):
    tokenizer = IntegerTokenizer()
    dataset = MaskedChatDataset([], tokenizer, **mode)
    tokens, mask = dataset.process({"messages": messages})
    return [tokenizer.words[token] for token, keep in zip(tokens, mask) if keep]


def test_per_message_preserves_role_defaults_and_explicit_nonassistant_targets():
    assert selected(conversation(), train_field=True) == [
        "<|im_start|>system", "policy", "<|im_end|>",
        "<|im_start|>user", "répété", "<|im_end|>",
        "<|im_start|>tool", "tool-output", "<|im_end|>",
        "<|im_start|>assistant", "earlier", "<|im_end|>",
        "<|im_start|>assistant", "last", "<|im_end|>",
    ]


def test_history_selects_actual_last_assistant_with_consecutive_and_later_tool_turns():
    messages = conversation() + [{"role": "tool", "content": "after"}]
    assert selected(messages, mask_history=True) == ["last", "<|im_end|>"]


def test_history_checks_selected_boundaries_when_template_rewrites_ignored_reasoning():
    """Nemotron drops old reasoning after a later user turn; the final span is exact."""
    class RewritesEarlierReasoning(IntegerTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            rewritten = [dict(message) for message in messages]
            for i, message in enumerate(rewritten):
                if message["role"] == "assistant" and any(
                    later["role"] == "user" for later in rewritten[i + 1:]
                ):
                    message["content"] = "omitted"
            return super().apply_chat_template(rewritten, **kwargs)

    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "old-reasoning"},
        {"role": "user", "content": "next-question"},
        {"role": "assistant", "content": "last"},
    ]
    tokenizer = RewritesEarlierReasoning()
    tokens, mask = MaskedChatDataset([], tokenizer, mask_history=True).process(
        {"messages": messages},
    )
    assert [tokenizer.words[token] for token, keep in zip(tokens, mask) if keep] == [
        "last", "<|im_end|>",
    ]
    with pytest.raises(ResponseMaskError, match="prefix-stable"):
        MaskedChatDataset([], tokenizer).process({"messages": messages})


@pytest.mark.parametrize("flag", [0, 1, "false", [], {}])
def test_malformed_flags_refused_before_any_template_call(flag):
    with pytest.raises(ResponseMaskError, match="train.*bool"):
        selected([{"role": "user", "content": "question", "train": flag}], train_field=True)


@pytest.mark.parametrize("mode", [{"train_field": True}, {"mask_history": True}])
def test_unsafe_templates_refused(mode):
    dataset = MaskedChatDataset([], IntegerTokenizer(prefix_stable=False), **mode)
    with pytest.raises(ResponseMaskError, match="prefix-stable"):
        dataset.process({"messages": conversation()})


def test_zero_selected_targets_are_refused():
    with pytest.raises(ResponseMaskError, match="no causal-loss target"):
        selected([{"role": "user", "content": "question"}], train_field=True)


@pytest.mark.parametrize("mode", [{"train_field": True}, {"mask_history": True}])
def test_selected_turn_truncated_away_refuses_instead_of_falling_back(mode):
    messages = [
        {"role": "user", "content": "long long long long long"},
        {"role": "assistant", "content": "answer"},
    ]
    dataset = MaskedChatDataset([], IntegerTokenizer(), max_length=3, **mode)
    with pytest.raises(ResponseMaskError, match="no causal-loss target.*truncation"):
        dataset.process({"messages": messages})
    dataset.max_length = 32
    tokens, mask = dataset.process({"messages": messages})
    assert any(mask[1:]) and len(tokens) == len(mask)


@pytest.mark.parametrize("messages", [None, [], {}, [1], [{"role": 1, "content": "x"}]])
def test_malformed_messages_refused(messages):
    dataset = MaskedChatDataset([], IntegerTokenizer(), train_field=True)
    with pytest.raises(ResponseMaskError, match="messages|roles"):
        dataset.process({"messages": messages})


@pytest.mark.parametrize("sample", [{"text": "plain"}, {"prompt": "q", "completion": "a"}])
@pytest.mark.parametrize("mode", [{"train_field": True}, {"mask_history": True}])
def test_selective_modes_refuse_shapes_without_message_boundaries(sample, mode):
    from soup_cli.trainer.mlx_masking import plan_response_masking

    with pytest.raises(ResponseMaskError, match="requires chat messages"):
        plan_response_masking(not mode.get("train_field", False), sample, **mode)


def test_verified_message_targets_agree_with_shared_labels_on_stable_template():
    from soup_cli.data.loss_mask import IGNORE_INDEX, build_per_message_train_labels

    tokenizer = IntegerTokenizer()
    messages = conversation()
    shared = build_per_message_train_labels(messages, tokenizer)
    tokens, mask = MaskedChatDataset([], tokenizer, train_field=True).process(
        {"messages": messages},
    )
    assert tokens == shared["input_ids"]
    actual_labels = [token if keep else IGNORE_INDEX for token, keep in zip(tokens, mask)]
    assert actual_labels == shared["labels"]


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("mode", ["mask_history", "train_on_messages_with_train_field"])
def test_selective_nontext_is_reported_and_refused_before_model_loading(modality, mode):
    from soup_cli.config.backend_support import REJECTED, check_config
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(
        base="local-base", backend="mlx", modality=modality,
        data={"train": "unused.jsonl", mode: True,
              "train_on_responses_only": mode == "mask_history"},
        training={"quantization": "none"},
    )
    gaps = {entry.field: entry.status for entry in check_config(cfg)}
    assert gaps[f"data.{mode}"] == REJECTED
    with pytest.raises(ValueError, match="selective.*text SFT only"):
        MLXSFTTrainerWrapper(cfg)._check_unsupported()
    cfg.modality = "text"
    assert f"data.{mode}" not in {entry.field for entry in check_config(cfg)}
    MLXSFTTrainerWrapper(cfg)._check_unsupported()


@pytest.mark.parametrize("mode", ["train_field", "mask_history"])
def test_public_wrapper_trains_the_selected_tokens(tmp_path, monkeypatch, mode):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    _install_fake_mlx(monkeypatch)
    captured = {}

    def train(**kwargs):
        dataset = kwargs["train_dataset"]
        assert isinstance(dataset, MaskedChatDataset)
        tokens, mask = dataset.process(dataset[0])
        captured["selected"] = supervised(tokens, mask)
        kwargs["training_callback"].on_train_loss_report({"train_loss": 0.5})

    monkeypatch.setattr(sys.modules["mlx_lm.tuner.trainer"], "train", train)
    data = {"train": "unused.jsonl", "format": "chatml"}
    if mode == "train_field":
        data.update(train_on_responses_only=False, train_on_messages_with_train_field=True)
    else:
        data.update(train_on_responses_only=True, mask_history=True)
    wrapper = MLXSFTTrainerWrapper(SoupConfig(
        base="local-base", backend="mlx", data=data, output=str(tmp_path),
        training={"quantization": "none", "epochs": 1, "batch_size": 1},
    ))
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = IntegerTokenizer()
    wrapper._dataset = {"train": [{"messages": conversation()}] * 4, "val": []}
    wrapper.train()
    words = [wrapper.tokenizer.words[token] for token in captured["selected"]]
    assert words == selected(conversation(), **{mode: True})
    metadata = json.loads((tmp_path / "adapter_config.json").read_text())
    assert metadata["mask_history"] is (mode == "mask_history")
    assert metadata["train_on_messages_with_train_field"] is (mode == "train_field")
