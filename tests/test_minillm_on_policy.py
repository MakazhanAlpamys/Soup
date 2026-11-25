"""MiniLLM prompt boundaries, separate collation and ragged autoregressive state."""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from soup_cli.config.loader import load_config_from_string  # noqa: E402
from soup_cli.trainer.distill import DistillTrainerWrapper  # noqa: E402
from soup_cli.utils.minillm import (  # noqa: E402
    MiniLLMConfig,
    minillm_on_policy_rollout,
)

_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|> "
    "{{ message['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
_ROW = {"messages": [
    {"role": "user", "content": "What is the capital of France ?"},
    {"role": "assistant", "content": "Paris ."},
]}


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = ["<unk>", "<s>", "</s>", "<pad>", "<|user|>",
                "<|assistant|>", "<|system|>", "<|end|>"]
    words = ["What", "is", "the", "capital", "of", "France", "?", "Paris", ".", "hi"]
    backend = Tokenizer(models.WordLevel(
        vocab={word: index for index, word in enumerate(specials + words)},
        unk_token="<unk>",
    ))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(specials)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>", chat_template=_TEMPLATE,
    )


def _model(seed, architecture="llama"):
    torch.manual_seed(seed)
    if architecture == "gpt2":
        config = transformers.GPT2Config(
            vocab_size=18, n_embd=16, n_layer=2, n_head=2, n_positions=256,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        return transformers.GPT2LMHeadModel(config).eval()
    config = transformers.LlamaConfig(
        vocab_size=18, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
        attention_dropout=0.0,
    )
    return transformers.LlamaForCausalLM(config).eval()


@pytest.fixture
def wrapper_factory(tmp_path, monkeypatch):
    original_post_init = transformers.TrainingArguments.__post_init__

    def cpu_args(args):
        args.use_cpu = True
        args.dataloader_pin_memory = False
        original_post_init(args)

    monkeypatch.setattr(transformers.TrainingArguments, "__post_init__", cpu_args)
    monkeypatch.setattr("soup_cli.trainer.distill.bf16_fp16_flags", lambda device: (False, False))
    tokenizer = _tokenizer()
    for name, seed in [("student", 11), ("teacher", 17)]:
        directory = tmp_path / name
        _model(seed).save_pretrained(directory)
        tokenizer.save_pretrained(directory)

    def build(rows, *, validation=None, mode="on_policy", full_sequence=False,
              checkpointing=False, anchor_path=None):
        training = {
            "teacher_model": str(tmp_path / "teacher"), "quantization": "none",
            "batch_size": 2, "epochs": 1, "warmup_ratio": 0.0,
            "gradient_checkpointing": checkpointing, "minillm_rollout_length": 3,
            "minillm_enabled": True, "minillm_on_policy": True,
            "lora": {"r": 2, "target_modules": ["q_proj", "v_proj"]},
        }
        if anchor_path is not None:
            training["minillm_pretrain_anchor_path"] = str(anchor_path)
            training["minillm_pretrain_anchor_weight"] = 0.2
        if mode != "on_policy":
            training.pop("minillm_rollout_length")
            training["minillm_on_policy"] = False
            training["minillm_enabled"] = mode == "offline"
            if mode == "offline":
                training["minillm_teacher_mix_ratio"] = 0.3
        config = load_config_from_string(json.dumps({
            "base": str(tmp_path / "student"), "task": "distill",
            "data": {"train": "unused.jsonl", "format": "chatml", "max_length": 64,
                     "val_split": 0.0, "train_on_responses_only": not full_sequence},
            "training": training, "output": str(tmp_path / "output"),
        }))
        wrapper = DistillTrainerWrapper(config, device="cpu")
        wrapper.setup({"train": rows, "val": validation or []})
        wrapper.model.eval()
        return wrapper

    return build


def _mapped(rows, *, full_sequence=False, training=None, tokenizer=None):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.trainer.distill import _map_distill_rows

    tokenizer = tokenizer or _tokenizer()
    data = DataConfig(train="unused.jsonl", max_length=64,
                      train_on_responses_only=not full_sequence)
    formatter = build_format_row(tokenizer, data, training_cfg=training, include_messages=True)
    return _map_distill_rows(rows, format_row=formatter, split="train", max_length=64,
                             rollout_tokenizer=tokenizer)


@pytest.mark.parametrize("format_name", ["alpaca", "sharegpt", "chatml", "auto"])
def test_normalized_format_boundaries_preserve_ce(format_name):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.formats import detect_format, format_to_messages
    from soup_cli.data.sft_format import build_format_row

    raw = {
        "alpaca": {"instruction": "What is the capital", "input": "of France ?",
                   "output": "Paris .", "system": "hi"},
        "sharegpt": {"conversations": [
            {"from": "system", "value": "hi"},
            {"from": "human", "value": "What is the capital of France ?"},
            {"from": "gpt", "value": "Paris ."},
        ]},
        "chatml": {"messages": [{"role": "system", "content": "hi"}, *_ROW["messages"]]},
    }
    source = raw["chatml" if format_name == "auto" else format_name]
    detected = detect_format([source]) if format_name == "auto" else format_name
    normalized = format_to_messages(source, detected)
    tokenizer = _tokenizer()
    actual = _mapped([normalized], tokenizer=tokenizer)[0]
    expected = tokenizer.apply_chat_template(normalized["messages"][:-1],
        tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
    assert actual["rollout_input_ids"] == expected
    assert actual["rollout_attention_mask"] == [1] * len(expected)
    baseline = build_format_row(tokenizer, DataConfig(train="unused.jsonl", max_length=64))(
        normalized
    )
    assert {name: actual[name] for name in baseline} == baseline
    assert "messages" not in actual


@pytest.mark.parametrize("full_sequence", [False, True])
def test_multiturn_boundary_is_independent_of_label_mask(full_sequence):
    messages = [*_ROW["messages"], {"role": "user", "content": "France ?"},
                {"role": "assistant", "content": "Paris ."}]
    tokenizer = _tokenizer()
    actual = _mapped([{"messages": messages}], full_sequence=full_sequence, tokenizer=tokenizer)[0]
    expected = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
        add_generation_prompt=True, return_dict=True)["input_ids"]
    assert actual["rollout_input_ids"] == expected
    assert tokenizer.decode(expected).count("Paris") == 1


def test_per_message_training_flags_do_not_choose_the_rollout_boundary():
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.trainer.distill import _map_distill_rows

    messages = [*_ROW["messages"], {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "France .", "train": False}]
    tokenizer = _tokenizer()
    data = DataConfig(train="unused.jsonl", max_length=64, train_on_responses_only=False,
                      train_on_messages_with_train_field=True)
    formatter = build_format_row(tokenizer, data, include_messages=True)
    row = _map_distill_rows([{"messages": messages}], format_row=formatter, split="train",
                           max_length=64, rollout_tokenizer=tokenizer)[0]
    expected = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
        add_generation_prompt=True, return_dict=True)["input_ids"]
    assert row["rollout_input_ids"] == expected
    assert tokenizer.convert_tokens_to_ids("Paris") in row["labels"]
    assert tokenizer.convert_tokens_to_ids("France") not in row["labels"]


def test_prompt_only_keeps_all_turns_and_existing_ce_guard():
    row = {"messages": _ROW["messages"][:-1]}
    tokenizer = _tokenizer()
    actual = _mapped([row], full_sequence=True, tokenizer=tokenizer)[0]
    assert actual["rollout_input_ids"] == tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True, return_dict=True,
    )["input_ids"]
    with pytest.raises(ValueError, match=r"train row 1: no causal-loss target"):
        _mapped([row])


def test_prompt_truncation_keeps_generation_suffix_and_ce_guard():
    row = {"messages": [{"role": "user", "content": "hi " * 100},
                        _ROW["messages"][-1]]}
    tokenizer = _tokenizer()
    actual = _mapped([row], full_sequence=True, tokenizer=tokenizer)[0]
    expected = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
        add_generation_prompt=True, return_dict=True)["input_ids"]
    assert actual["rollout_input_ids"] == expected[-64:]
    assert len(actual["rollout_input_ids"]) == 64
    assert actual["rollout_input_ids"][-1] == tokenizer.convert_tokens_to_ids("<|assistant|>")
    assert actual["input_ids"] != actual["rollout_input_ids"]
    with pytest.raises(ValueError, match=r"train row 1: no causal-loss target.*max_length=64"):
        _mapped([row])


def test_row_transforms_run_once_for_both_representations(monkeypatch):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.trainer.distill import _map_distill_rows
    from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

    calls = []

    def transform(spec, row):
        calls.append(spec)
        return {"messages": [{"role": "system", "content": "hi"}, *row["messages"]]}

    monkeypatch.setattr("soup_cli.utils.data_pipeline.resolve_prompt_strategy", lambda spec: None)
    monkeypatch.setattr("soup_cli.utils.data_pipeline.apply_prompt_strategy", transform)
    tokenizer = _tokenizer()
    data = DataConfig(train="unused.jsonl", max_length=64, prompt_strategy="local:transform")
    training = SimpleNamespace(reasoning_effort="high", train_on_eot=False)
    formatter = build_format_row(tokenizer, data, training_cfg=training, include_messages=True)
    actual = _map_distill_rows([_ROW], format_row=formatter, split="train", max_length=64,
                              rollout_tokenizer=tokenizer)[0]
    assert calls == ["local:transform"]
    transformed = apply_reasoning_effort_prefix(
        [{"role": "system", "content": "hi"}, *_ROW["messages"]], "high",
    )
    expected = tokenizer.apply_chat_template(transformed[:-1], tokenize=True,
        add_generation_prompt=True, return_dict=True)["input_ids"]
    assert actual["rollout_input_ids"] == expected
    baseline = build_format_row(tokenizer, DataConfig(train="unused.jsonl", max_length=64))(
        {"messages": transformed}
    )
    assert {name: actual[name] for name in baseline} == baseline


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("system_prefix", [False, True], ids=["assistant-only", "system-only"])
def test_empty_prompt_refusal_names_split_and_row(wrapper_factory, split, system_prefix):
    assistant_only = {"messages": [_ROW["messages"][-1]]}
    if system_prefix:
        assistant_only["messages"].insert(0, {"role": "system", "content": "hi"})
    rows = [_ROW, assistant_only]
    expected = (f"{split} row 2: MiniLLM on-policy requires a non-empty prompt before the "
                "target response; supply explicit prompt and response rows")
    with pytest.raises(ValueError) as caught:
        wrapper_factory(rows if split == "train" else [_ROW],
                        validation=rows if split == "val" else None)
    assert str(caught.value) == expected


@pytest.mark.parametrize("mode", ["plain", "offline"])
@pytest.mark.parametrize("system_prefix", [False, True], ids=["assistant-only", "system-only"])
def test_assistant_only_rows_still_train_without_on_policy(wrapper_factory, mode, system_prefix):
    row = {"messages": [_ROW["messages"][-1]]}
    if system_prefix:
        row["messages"].insert(0, {"role": "system", "content": "hi"})
    wrapper = wrapper_factory([row, row], validation=[row], mode=mode)
    assert "rollout_input_ids" not in wrapper.trainer.train_dataset.column_names
    before = {name: parameter.detach().clone()
              for name, parameter in wrapper.model.named_parameters() if parameter.requires_grad}
    outcome = wrapper.trainer.train()
    assert outcome.global_step == 1 and math.isfinite(outcome.training_loss)
    assert any(not torch.equal(before[name], parameter) for name, parameter in
               wrapper.model.named_parameters() if name in before)
    assert math.isfinite(wrapper.trainer.evaluate()["eval_loss"])


@pytest.mark.parametrize("split", ["train", "val"])
def test_system_user_prompt_keeps_existing_rollout_ids(wrapper_factory, split):
    row = {"messages": [{"role": "system", "content": "hi"}, *_ROW["messages"]]}
    wrapper = wrapper_factory([row], validation=[row])
    dataset = (wrapper.trainer.train_dataset if split == "train"
               else wrapper.trainer.eval_dataset)
    expected = wrapper.tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=True, add_generation_prompt=True, return_dict=True,
    )["input_ids"]
    assert dataset[0]["rollout_input_ids"] == expected
    assert dataset[0]["rollout_attention_mask"] == [1] * len(expected)


def test_no_rendered_prompt_tokens_names_row(monkeypatch):
    tokenizer = _tokenizer()
    original = tokenizer.apply_chat_template

    def render(*args, **kwargs):
        if kwargs.get("add_generation_prompt"):
            return {"input_ids": []}
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    message = "train row 1: MiniLLM on-policy prompt has no usable tokens"
    with pytest.raises(ValueError, match=message):
        _mapped([_ROW], tokenizer=tokenizer)


def test_rollout_prefix_excludes_target(wrapper_factory, monkeypatch):
    changed = copy.deepcopy(_ROW)
    changed["messages"][-1]["content"] = "France ."
    wrapper = wrapper_factory([_ROW, changed])
    batch = next(iter(wrapper.trainer.get_train_dataloader()))
    captured = []

    def capture(student, teacher, input_ids, attention_mask, **kwargs):
        captured.extend(input_ids.tolist())
        return next(parameter for parameter in student.parameters()
                    if parameter.requires_grad).sum() * 0, 3

    monkeypatch.setattr("soup_cli.utils.minillm.minillm_on_policy_rollout", capture)
    wrapper.trainer.compute_loss(wrapper.model, batch)
    assert captured[0] == captured[1]
    expected = wrapper.tokenizer.apply_chat_template(
        _ROW["messages"][:-1], tokenize=True, add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    assert captured == [expected, expected]
    assert not torch.equal(batch["labels"][0], batch["labels"][1])


def _rollout(student, teacher, input_ids, attention_mask, *, use_cache):
    sampled = []

    def greedy(probabilities):
        tokens = probabilities.argmax(-1, keepdim=True)
        sampled.append(tokens.detach().clone())
        return tokens

    loss, steps = minillm_on_policy_rollout(
        student, teacher, input_ids, attention_mask,
        config=MiniLLMConfig(on_policy=True, teacher_mix_ratio=0.3),
        max_new_tokens=4, sample_fn=greedy, use_cache=use_cache,
    )
    loss.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in student.named_parameters()
                 if parameter.grad is not None}
    assert steps == 4 and gradients
    assert all(parameter.grad is None for parameter in teacher.parameters())
    return loss.detach(), torch.cat(sampled, dim=1), gradients


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_ragged_rollout_matches_singletons(use_cache, architecture, padding_side):
    student, teacher = _model(91, architecture), _model(92, architecture)
    sequences = [[1, 2, 3], [1, 2, 3, 4, 5]]
    singletons = [_rollout(copy.deepcopy(student), copy.deepcopy(teacher),
                          torch.tensor([sequence]), torch.ones(1, len(sequence), dtype=torch.long),
                          use_cache=use_cache) for sequence in sequences]
    padded = [([0, 0] + sequences[0]) if padding_side == "left"
              else (sequences[0] + [0, 0]), sequences[1]]
    masks = [[0, 0, 1, 1, 1] if padding_side == "left" else [1, 1, 1, 0, 0], [1] * 5]
    loss, tokens, gradients = _rollout(student, teacher, torch.tensor(padded),
                                      torch.tensor(masks), use_cache=use_cache)
    torch.testing.assert_close(loss, torch.stack([result[0] for result in singletons]).mean())
    assert torch.equal(tokens, torch.cat([result[1] for result in singletons]))
    for name, gradient in gradients.items():
        expected = torch.stack([result[2][name] for result in singletons]).mean(0)
        torch.testing.assert_close(gradient, expected, atol=2e-7, rtol=2e-4)


@pytest.mark.parametrize("mode", ["plain", "offline"])
def test_prompt_fields_do_not_change_ce_or_offline_loss(wrapper_factory, mode):
    wrapper = wrapper_factory([_ROW], mode=mode)
    clean = next(iter(wrapper.trainer.get_train_dataloader()))
    carrying = {**clean, "rollout_input_ids": torch.tensor([[1, 2]]),
                "rollout_attention_mask": torch.tensor([[1, 1]])}

    def no_metadata(module, args, kwargs):
        assert "rollout_input_ids" not in kwargs and "rollout_attention_mask" not in kwargs

    handles = [model.register_forward_pre_hook(no_metadata, with_kwargs=True)
               for model in [wrapper.model, wrapper.teacher]]
    wrapper.model.zero_grad()
    expected_loss, expected_outputs = wrapper.trainer.compute_loss(
        wrapper.model, clean, return_outputs=True,
    )
    expected_loss.backward()
    expected_gradients = {name: parameter.grad.clone()
                          for name, parameter in wrapper.model.named_parameters()
                          if parameter.grad is not None}
    wrapper.model.zero_grad()
    loss, outputs = wrapper.trainer.compute_loss(wrapper.model, carrying, return_outputs=True)
    loss.backward()
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(outputs.logits, expected_outputs.logits, rtol=0, atol=0)
    for name, parameter in wrapper.model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    assert "rollout_input_ids" in carrying


def test_real_collator_and_evaluation_use_separate_ragged_prefixes(wrapper_factory, monkeypatch):
    short = {"messages": [{"role": "user", "content": "hi"}, _ROW["messages"][-1]]}
    wrapper = wrapper_factory([_ROW, short], validation=[_ROW, short])
    features = [wrapper.trainer.train_dataset[index] for index in range(2)]
    original_features = copy.deepcopy(features)
    batch = wrapper.trainer.data_collator(features)
    assert features == original_features
    assert wrapper.tokenizer.padding_side == "right"
    assert batch["rollout_attention_mask"].tolist()[1] == [0] * 6 + [1] * 4
    assert batch["attention_mask"].tolist()[1] == [1] * 7 + [0] * 6
    assert batch["rollout_input_ids"].shape == (2, 10)
    assert batch["input_ids"].shape == (2, 13)
    seen = []
    original_rollout = minillm_on_policy_rollout

    def record(student, teacher, input_ids, attention_mask, **kwargs):
        seen.append((input_ids.clone(), attention_mask.clone(), torch.is_grad_enabled()))
        return original_rollout(student, teacher, input_ids, attention_mask, **kwargs)

    def no_metadata(module, args, kwargs):
        assert "rollout_input_ids" not in kwargs and "rollout_attention_mask" not in kwargs

    hooks = [model.register_forward_pre_hook(no_metadata, with_kwargs=True)
             for model in [wrapper.model, wrapper.teacher]]
    monkeypatch.setattr("soup_cli.utils.minillm.minillm_on_policy_rollout", record)
    try:
        loss, outputs = wrapper.trainer.compute_loss(wrapper.model, batch, return_outputs=True)
        assert torch.isfinite(loss) and outputs.logits.shape[:2] == batch["input_ids"].shape
        metrics = wrapper.trainer.evaluate()
    finally:
        for hook in hooks:
            hook.remove()
    assert math.isfinite(metrics["eval_loss"])
    assert len(seen) == 2 and seen[0][2] and not seen[1][2]
    for ids, mask, _grad_enabled in seen:
        assert torch.equal(ids, batch["rollout_input_ids"])
        assert torch.equal(mask, batch["rollout_attention_mask"])
    assert all(parameter.grad is None for parameter in wrapper.teacher.parameters())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_on_policy_real_optimizer_update(wrapper_factory, checkpointing):
    wrapper = wrapper_factory([_ROW, _ROW], checkpointing=checkpointing)
    before = {name: parameter.detach().clone()
              for name, parameter in wrapper.model.named_parameters() if parameter.requires_grad}
    result = wrapper.trainer.train()
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    assert all(torch.isfinite(parameter).all() for parameter in wrapper.model.parameters())
    assert any(not torch.equal(before[name], parameter) for name, parameter in
               wrapper.model.named_parameters() if name in before)
    assert all(parameter.grad is None for parameter in wrapper.teacher.parameters())


def test_on_policy_ce_rollout_and_anchor_weights(wrapper_factory, tmp_path, monkeypatch):
    from soup_cli.utils.minillm import build_minillm_callback

    monkeypatch.chdir(tmp_path)
    anchor_path = tmp_path / "anchor.jsonl"
    anchor_path.write_text(json.dumps({"text": "hi Paris ."}) + "\n")
    wrapper = wrapper_factory([_ROW], anchor_path=anchor_path)
    batch = next(iter(wrapper.trainer.get_train_dataloader()))
    monkeypatch.setattr("soup_cli.utils.minillm._multinomial_sample",
                        lambda probabilities: probabilities.argmax(-1, keepdim=True))
    callback = build_minillm_callback(MiniLLMConfig(
        on_policy=True, rollout_length=3, pretrain_anchor_path=str(anchor_path),
        pretrain_anchor_weight=0.2,
    ), tokenizer=wrapper.tokenizer, temperature=2.0)
    outputs = wrapper.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ce = torch.nn.functional.cross_entropy(outputs.logits[:, :-1].reshape(-1, 18),
                                           batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
    rollout = callback.on_policy_term(wrapper.model, wrapper.teacher,
                                      batch["rollout_input_ids"], batch["rollout_attention_mask"])
    anchor = callback.anchor_term(wrapper.model)
    assert anchor > 0
    actual = wrapper.trainer.compute_loss(wrapper.model, batch)
    torch.testing.assert_close(actual, 0.5 * ce + 0.5 * rollout + anchor)


def test_rollout_keeps_fixed_budget_after_eos():
    sampled = []

    def sample_eos(probabilities):
        token = torch.full((probabilities.shape[0], 1), 2, dtype=torch.long)
        sampled.append(token)
        return token

    loss, steps = minillm_on_policy_rollout(_model(91), _model(92), torch.tensor([[1, 2, 3]]),
        None, config=MiniLLMConfig(on_policy=True), max_new_tokens=4, sample_fn=sample_eos)
    assert steps == len(sampled) == 4 and torch.isfinite(loss)


class _IgnoreCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.widths = []

    def forward(self, input_ids, attention_mask, position_ids=None,
                past_key_values=None, use_cache=False):
        self.widths.append(input_ids.shape[1])
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=position_ids, use_cache=False)


@pytest.mark.parametrize("ignored_cache", ["student", "teacher", "both"])
def test_ragged_rollout_cache_fallback(ignored_cache):
    student, teacher = _model(91), _model(92)
    ids = torch.tensor([[2, 2, 1, 2, 3], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1] * 5])
    reference = _rollout(copy.deepcopy(student), copy.deepcopy(teacher), ids, mask, use_cache=False)
    if ignored_cache in {"student", "both"}:
        student = _IgnoreCache(student)
    if ignored_cache in {"teacher", "both"}:
        teacher = _IgnoreCache(teacher)
    actual = _rollout(student, teacher, ids, mask, use_cache=True)
    torch.testing.assert_close(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])
    for name, gradient in actual[2].items():
        reference_name = name.removeprefix("model.") if isinstance(student, _IgnoreCache) else name
        torch.testing.assert_close(gradient, reference[2][reference_name])
    for model in [student, teacher]:
        if isinstance(model, _IgnoreCache):
            assert model.widths == [5, 6, 7, 8]


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_rollout_positions_and_cache_stay_aligned(architecture):
    student, teacher = _model(91, architecture), _model(92, architecture)
    ids = torch.tensor([[2, 2, 1, 2, 3], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1] * 5])
    reference = _rollout(copy.deepcopy(student), copy.deepcopy(teacher), ids, mask, use_cache=False)
    records = {"student": [], "teacher": []}

    def recorder(name):
        def capture(module, args, kwargs):
            past = kwargs.get("past_key_values")
            records[name].append({
                "positions": kwargs["position_ids"].detach().clone(),
                "mask": kwargs["attention_mask"].detach().clone(),
                "width": kwargs["input_ids"].shape[1],
                "cache_length": past.get_seq_length() if past is not None else 0,
                "grad_enabled": torch.is_grad_enabled(),
            })
        return capture

    hooks = [model.register_forward_pre_hook(recorder(name), with_kwargs=True)
             for name, model in [("student", student), ("teacher", teacher)]]
    try:
        actual = _rollout(student, teacher, ids, mask, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])
    for name, gradient in actual[2].items():
        torch.testing.assert_close(gradient, reference[2][name], atol=2e-7, rtol=2e-4)
    for name, calls in records.items():
        assert [call["width"] for call in calls] == [5, 1, 1, 1]
        assert [call["cache_length"] for call in calls] == [0, 5, 6, 7]
        assert all(call["grad_enabled"] == (name == "student") for call in calls)
        assert calls[0]["positions"].tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]
        for step, call in enumerate(calls[1:]):
            assert call["positions"].tolist() == [[3 + step], [5 + step]]
            assert call["mask"].tolist() == [[0, 0] + [1] * (4 + step), [1] * (6 + step)]


def test_equal_lengths_and_shared_pad_eos_control():
    student, teacher = _model(91, "gpt2"), _model(92, "gpt2")
    for model in [student, teacher]:
        model.config.pad_token_id = model.config.eos_token_id = 2
    sequence = torch.tensor([[1, 2, 3]])
    single = _rollout(copy.deepcopy(student), copy.deepcopy(teacher), sequence,
                      torch.ones_like(sequence), use_cache=True)
    equal = _rollout(copy.deepcopy(student), copy.deepcopy(teacher), sequence.repeat(2, 1),
                     torch.ones(2, 3, dtype=torch.long), use_cache=True)
    ragged = _rollout(student, teacher, torch.tensor([[2, 2, 1, 2, 3], [1, 2, 3, 4, 5]]),
                      torch.tensor([[0, 0, 1, 1, 1], [1] * 5]), use_cache=True)
    torch.testing.assert_close(equal[0], single[0])
    assert torch.equal(equal[1], single[1].repeat(2, 1))
    assert torch.equal(ragged[1][:1], single[1])


@pytest.mark.parametrize("ids,mask,message", [
    ([[]], [[]], "non-empty"),
    ([[1]], [[0]], "non-empty prompt per row"),
    ([[1]], [[2]], "binary mask"),
    ([[1, 2]], [[1]], "must match"),
])
def test_invalid_rollout_prefix_refuses(ids, mask, message):
    with pytest.raises(ValueError, match=message):
        minillm_on_policy_rollout(_model(91), _model(92), torch.tensor(ids), torch.tensor(mask),
            config=MiniLLMConfig(on_policy=True), max_new_tokens=2)
