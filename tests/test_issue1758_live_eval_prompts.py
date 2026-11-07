"""Public prompt preparation must match both live generators (#1758)."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import live_eval

_TEMPLATE = (
    "{{ bos_token }}{{ messages[0]['content'] }} "
    "{% if enable_thinking is not defined or enable_thinking %}think{% else %}answer{% endif %}"
    "{% if add_generation_prompt %} assistant{% endif %}"
)


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {word: index for index, word in enumerate(
        ["<unk>", "<bos>", "<eos>", "hello", "world", "think", "answer", "assistant"]
    )}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        pad_token="<eos>", unk_token="<unk>",
    )
    tok.chat_template = _TEMPLATE
    return tok


@pytest.mark.parametrize("factory", ["make_generator", "make_multi_generator"])
def test_generators_share_public_encoding_and_snapshot_options(tokenizer, factory):
    import torch

    class Recorder:
        sent = None

        def generate(self, input_ids, **kwargs):
            self.sent = input_ids.tolist()[0]
            return torch.cat([input_ids, input_ids[:, -1:]], dim=1)

    model = Recorder()
    options = {"enable_thinking": False}
    generate = getattr(live_eval, factory)(
        "unused", loaded=(model, tokenizer, "cpu"), chat_template_kwargs=options
    )
    options["enable_thinking"] = True
    if factory == "make_multi_generator":
        generate("hello", 1)
    else:
        generate("hello")
    encoded = live_eval.encode_prompt(
        tokenizer, "hello", chat_template_kwargs={"enable_thinking": False}, return_tensors=None
    )
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert model.sent == encoded["input_ids"] == expected
    assert model.sent.count(tokenizer.bos_token_id) == 1
    assert "think" not in tokenizer.decode(model.sent)


@pytest.mark.parametrize("template", [None, "{{ broken( }}"])
def test_default_fallback_preserves_raw_special_tokens(tokenizer, template):
    tokenizer.chat_template = template
    encoded = live_eval.encode_prompt(tokenizer, "hello", return_tensors=None)
    assert encoded["input_ids"] == tokenizer("hello")["input_ids"]
    with pytest.raises(ValueError, match="chat.template"):
        live_eval.encode_prompt(tokenizer, "hello", chat_template_kwargs={"enable_thinking": False})


def test_public_prompt_budget_and_batch_padding(tokenizer):
    encoded = [
        live_eval.encode_prompt(tokenizer, text, return_tensors=None)
        for text in ["hello " * 2000, "world"]
    ]
    assert len(encoded[0]["input_ids"]) == live_eval.MAX_PROMPT_TOKENS == 1024
    assert encoded[0]["input_ids"][-1] == tokenizer.convert_tokens_to_ids("hello")
    assert tokenizer.convert_tokens_to_ids("assistant") not in encoded[0]["input_ids"]
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    assert batch["input_ids"].shape == (2, 1024)
    assert batch["attention_mask"][1].sum() == len(encoded[1]["input_ids"])


@pytest.mark.parametrize("options", [
    [], "bad", {1: False}, {"conversation": []}, {"tokenize": True},
    {"add_generation_prompt": False}, {"continue_final_message": True},
    {"return_tensors": "pt"}, {"return_dict": True}, {"padding": True},
    {"truncation": True}, {"max_length": 1}, {"chat_template": "replacement"},
    {"return_assistant_tokens_mask": True},
])
def test_invalid_options_fail_before_loading(monkeypatch, tokenizer, options):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid options must fail before loading")

    monkeypatch.setattr(live_eval, "load_model_and_tokenizer", unexpected)
    for factory in [live_eval.make_generator, live_eval.make_multi_generator]:
        with pytest.raises(ValueError, match="chat_template_kwargs"):
            factory("unused", chat_template_kwargs=options)
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        live_eval.encode_prompt(tokenizer, "hello", chat_template_kwargs=options)


def test_real_model_cpu_uses_public_prompt_contract(tokenizer):
    from tests._tiny_hf_models import build_llama

    model = build_llama(tokenizer).eval()
    result = live_eval.make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=2,
        chat_template_kwargs={"enable_thinking": False},
    )("hello")
    assert isinstance(result, str)


@pytest.mark.parametrize("adapter", [None, "adapter"])
def test_ship_forwards_options_to_base_and_tuned(monkeypatch, adapter):
    from soup_cli.commands.ship import _resolve_generators

    calls = []

    def factory(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return lambda prompt: "answer"

    monkeypatch.setattr(live_eval, "make_generator", factory)
    _resolve_generators("base", "tuned" if adapter is None else None, adapter, "cpu",
                        chat_template_kwargs={"enable_thinking": False})
    assert len(calls) == 2
    assert all(c[1]["chat_template_kwargs"] == {"enable_thinking": False} for c in calls)


@pytest.mark.parametrize("value", ["[]", "null", "bad", '{"tokenize":true}',
                                  '{"enable_thinking":NaN}', '{"chat_template":"replacement"}',
                                  '{"return_assistant_tokens_mask":true}'])
def test_ship_invalid_json_options_are_usage_errors(value):
    result = CliRunner().invoke(app, ["ship", "--chat-template-kwargs", value])
    assert result.exit_code == 3
    assert "--chat-template-kwargs" in result.output


def test_ship_cli_forwards_valid_json(monkeypatch):
    import soup_cli.commands.ship as ship_module

    seen = {}

    def verdict(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after flag parsing")

    monkeypatch.setattr(ship_module, "_verdict_live", verdict)
    result = CliRunner().invoke(app, ["ship", "--base", "base", "--tuned", "tuned",
                                    "--task-eval", "tasks.jsonl", "--chat-template-kwargs",
                                    json.dumps({"enable_thinking": False})])
    assert isinstance(result.exception, RuntimeError)
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_ship_refuses_template_options_in_offline_mode():
    result = CliRunner().invoke(app, ["ship", "--evidence", "missing.json",
                                    "--chat-template-kwargs", "{}"])
    assert result.exit_code == 3
    assert "live" in result.output


def test_live_ship_cli_options_reach_both_generators(monkeypatch, tmp_path):
    calls = []

    def factory(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return lambda prompt: "answer" if model_id == "tuned" else "wrong"

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.jsonl").write_text(json.dumps(
        {"prompt": "hello", "expected": "answer", "scoring": "exact"}
    ) + "\n")
    monkeypatch.setattr(live_eval, "make_generator", factory)
    result = CliRunner().invoke(app, ["ship", "--base", "base", "--tuned", "tuned",
                                    "--device", "cpu", "--task-eval", "tasks.jsonl",
                                    "--chat-template-kwargs", '{"enable_thinking":false}'])
    assert result.exit_code == 0, result.output
    assert [call[0] for call in calls] == ["base", "tuned"]
    assert all(call[1]["chat_template_kwargs"] == {"enable_thinking": False} for call in calls)


@pytest.mark.parametrize("factory", ["make_generator", "make_multi_generator"])
def test_generators_snapshot_nested_template_options(tokenizer, factory):
    import torch

    tokenizer.chat_template = (
        "{{ bos_token }}{{ messages[0]['content'] }} "
        "{% if settings.flags.enable_thinking %}think{% else %}answer{% endif %}"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )

    class Recorder:
        sent = None

        def generate(self, input_ids, **kwargs):
            self.sent = input_ids.tolist()[0]
            return torch.cat([input_ids, input_ids[:, -1:]], dim=1)

    model = Recorder()
    options = {"settings": {"flags": {"enable_thinking": False}}}
    generate = getattr(live_eval, factory)(
        "unused", loaded=(model, tokenizer, "cpu"), chat_template_kwargs=options,
    )
    options["settings"]["flags"]["enable_thinking"] = True
    if factory == "make_multi_generator":
        generate("hello", 1)
    else:
        generate("hello")
    assert tokenizer.convert_tokens_to_ids("answer") in model.sent
    assert tokenizer.convert_tokens_to_ids("think") not in model.sent


def test_template_error_keeps_the_underlying_diagnostic(tokenizer):
    tokenizer.chat_template = '{{ raise_exception("named rendering cause") }}'
    with pytest.raises(ValueError, match="cannot render chat template.*named rendering cause"):
        live_eval.encode_prompt(tokenizer, "hello", chat_template_kwargs={"enable_thinking": False})


def test_ship_deeply_nested_json_is_a_usage_error():
    value = '{"nested":' + '[' * 2000 + '0' + ']' * 2000 + '}'
    result = CliRunner().invoke(app, ["ship", "--chat-template-kwargs", value])
    assert result.exit_code == 3
    assert "--chat-template-kwargs" in result.output
