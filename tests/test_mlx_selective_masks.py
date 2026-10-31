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


@pytest.mark.parametrize('responses,train_field,history,valid,token_mask', [
    (False, False, False, True, False),
    (False, False, True, False, None),
    (False, True, False, True, True),
    (False, True, True, False, None),
    (True, False, False, True, True),
    (True, False, True, True, True),
    (True, True, False, False, None),
    (True, True, True, False, None),
])
def test_selective_mode_matrix(responses, train_field, history, valid, token_mask):
    from soup_cli.trainer.mlx_masking import plan_response_masking

    args = (responses, {'messages': conversation()})
    options = dict(train_field=train_field, mask_history=history)
    if not valid:
        with pytest.raises(ResponseMaskError, match='incompatible'):
            plan_response_masking(*args, **options)
    else:
        plan = plan_response_masking(*args, **options)
        assert plan.token_mask is token_mask
        assert plan.mask_prompt is False and plan.warning == ''


@pytest.mark.parametrize('train_field,history', [(False, False), (False, True),
                                               (True, False), (True, True)])
def test_dataset_selective_modes_are_independently_checked(train_field, history):
    dataset = MaskedChatDataset([], IntegerTokenizer(), train_field=train_field,
                                mask_history=history)
    if train_field and history:
        with pytest.raises(ResponseMaskError, match='incompatible train-field and mask_history'):
            dataset.process({'messages': conversation()})
    else:
        tokens, mask = dataset.process({'messages': conversation()})
        words = [dataset.tokenizer.words[token] for token, keep in zip(tokens, mask) if keep]
        if history:
            assert words == ['last', '<|im_end|>']
        elif train_field:
            assert '<|im_start|>user' in words and 'tool-output' in words
            assert '<|im_start|>assistant' in words and 'last' in words
        else:
            assert words == ['répété', '<|im_end|>', 'earlier', '<|im_end|>', 'last', '<|im_end|>']


@pytest.mark.parametrize('flag', [0, 1, 'false', [], {}])
def test_verified_helper_rejects_malformed_train_flag_directly(flag):
    from soup_cli.data.loss_mask import build_per_message_train_labels

    messages = [{'role': 'user', 'content': 'question', 'train': flag},
                {'role': 'assistant', 'content': 'answer'}]
    with pytest.raises(ValueError, match='message 0 train must be bool or None'):
        build_per_message_train_labels(messages, IntegerTokenizer(), verified_prefix=True)
    for valid in (False, True, None):
        messages[0]['train'] = valid
        tokenizer = IntegerTokenizer()
        actual = build_per_message_train_labels(messages, tokenizer, verified_prefix=True)
        assert len(actual['labels']) == len(actual['input_ids'])
        assert actual['labels'][-2:] == actual['input_ids'][-2:]
        assert (actual['labels'][1] == actual['input_ids'][1]) is (valid is True)


@pytest.mark.parametrize('through_dataset', [False, True])
def test_verified_message_tools_affect_actual_tokens_and_targets(through_dataset):
    from soup_cli.data.loss_mask import IGNORE_INDEX, build_per_message_train_labels

    class ToolTokenizer(IntegerTokenizer):
        def apply_chat_template(self, messages, *, tools=None, **kwargs):
            name = tools[0]['function']['name'] if tools else 'no-tools'
            rendered = [{**m, 'content': name + ' ' + m['content']} for m in messages]
            return super().apply_chat_template(rendered, tools=tools, **kwargs)

    messages = [{'role': 'user', 'content': 'question', 'train': False},
                {'role': 'assistant', 'content': 'answer', 'train': True}]
    tools = [{'type': 'function', 'function': {'name': 'lookup'}}]
    tokenizer = ToolTokenizer()
    if through_dataset:
        tokens, mask = MaskedChatDataset([], tokenizer, train_field=True).process(
            {'messages': messages, 'tools': tools},
        )
        labels = [token if keep else IGNORE_INDEX for token, keep in zip(tokens, mask)]
    else:
        result = build_per_message_train_labels(messages, tokenizer, verified_prefix=True,
                                                 tools=tools)
        tokens, labels = result['input_ids'], result['labels']
    assert [tokenizer.words[token] for token in tokens] == [
        '<|im_start|>user', 'lookup', 'question', '<|im_end|>',
        '<|im_start|>assistant', 'lookup', 'answer', '<|im_end|>',
    ]
    assert [tokenizer.words[token] for token in labels if token != IGNORE_INDEX] == [
        '<|im_start|>assistant', 'lookup', 'answer', '<|im_end|>',
    ]


@pytest.mark.parametrize('modality', ['vision', 'audio'])
@pytest.mark.parametrize('field', ['use_flash_attn', 'use_liger', 'moe_lora'])
def test_nontext_doctor_omits_settings_refused_by_schema(modality, field):
    from soup_cli.config.backend_support import check_config, unsupported_for
    from soup_cli.config.schema import SoupConfig

    kwargs = dict(base='local-base', backend='mlx', modality=modality,
                  data={'train': 'unused.jsonl'})
    with pytest.raises(ValueError, match=field):
        SoupConfig(**kwargs, training={field: True})
    control = SoupConfig(**kwargs, training={field: False})
    assert f'training.{field}' not in {entry.field for entry in check_config(control)}
    assert f'training.{field}' not in {
        entry.field for entry in unsupported_for('sft', 'mlx', modality)
    }
