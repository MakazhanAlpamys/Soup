"""Regression tests for #804: chat-template overrides must reach every chat trainer."""

from __future__ import annotations

import inspect
from importlib import import_module
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

_WIRED_METHODS = {
    "bco": ("soup_cli.trainer.bco", "BCOTrainerWrapper", "setup"),
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", "setup"),
    "grpo": ("soup_cli.trainer.grpo", "GRPOTrainerWrapper", "setup"),
    "ipo": ("soup_cli.trainer.ipo", "IPOTrainerWrapper", "setup"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", "setup"),
    "online_dpo": (
        "soup_cli.trainer.online_dpo",
        "OnlineDPOTrainerWrapper",
        "_setup_transformers",
    ),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", "setup"),
    "reward_model": (
        "soup_cli.trainer.reward_model",
        "RewardModelTrainerWrapper",
        "setup",
    ),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", "setup"),
}

_RUNTIME_WRAPPERS = {
    task: spec for task, spec in _WIRED_METHODS.items() if task != "online_dpo"
}


def _requires_train_extra() -> None:
    for module in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(module, reason=f"{module} is only in the [train] extra")


def _tiny_llama_dir(tmp_path):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True)
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))

    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "SHIPPED": 4,
        "OVERRIDE": 5,
        "user": 6,
        "assistant": 7,
        "hi": 8,
        "good": 9,
        "bad": 10,
        "answer": 11,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        chat_template=(
            "{% for message in messages %}SHIPPED {{ message['role'] }} "
            "{{ message['content'] }} {% endfor %}"
        ),
    )
    fast.save_pretrained(str(weights))
    return weights


def _dpo_config(weights, output):
    from soup_cli.config.loader import load_config_from_string

    marker_template = (
        "{% for message in messages %}OVERRIDE {{ message['role'] }} "
        "{{ message['content'] }} {% endfor %}"
    )
    return load_config_from_string(
        f"base: {weights}\n"
        "task: dpo\n"
        "backend: transformers\n"
        "data:\n"
        "  train: train.jsonl\n"
        "  max_length: 64\n"
        f"  chat_template: \"{marker_template}\"\n"
        "training:\n"
        "  batch_size: 1\n"
        "  quantization: none\n"
        "  epochs: 1\n"
        "  lora:\n"
        "    r: 4\n"
        "    alpha: 8\n"
        "    target_modules: [q_proj, v_proj]\n"
        f"output: {output}\n"
    )


class TestDpoUsesTheOverrideForLiveRendering:
    def test_marker_reaches_prompt_ids_and_saved_tokenizer(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from transformers import AutoTokenizer

        from soup_cli.trainer.dpo import DPOTrainerWrapper

        weights = _tiny_llama_dir(tmp_path)
        cfg = _dpo_config(weights, tmp_path / "out")
        monkeypatch.chdir(tmp_path)
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        rows = [
            {
                "prompt": [{"role": "user", "content": "hi"}],
                "chosen": [{"role": "assistant", "content": "good answer"}],
                "rejected": [{"role": "assistant", "content": "bad answer"}],
            }
            for _ in range(4)
        ]
        wrapper.setup({"train": rows})

        prepared = wrapper.trainer.train_dataset[0]
        prompt_ids = prepared.get("prompt_input_ids", prepared.get("prompt_ids"))
        assert prompt_ids is not None, prepared.keys()
        rendered = wrapper.tokenizer.decode(prompt_ids)
        assert "OVERRIDE" in rendered
        assert "SHIPPED" not in rendered

        saved = tmp_path / "saved-tokenizer"
        wrapper.tokenizer.save_pretrained(saved)
        reloaded = AutoTokenizer.from_pretrained(saved)
        assert "OVERRIDE" in reloaded.chat_template


class TestEveryConversationalTrainerIsWired:
    @pytest.mark.parametrize("task", tuple(_WIRED_METHODS))
    def test_override_is_applied_after_tokenizer_setup(self, task):
        module_name, class_name, method_name = _WIRED_METHODS[task]
        wrapper_class = getattr(import_module(module_name), class_name)
        source = inspect.getsource(getattr(wrapper_class, method_name))
        assert "apply_chat_template_override" in source, task

    @pytest.mark.parametrize("task", tuple(_RUNTIME_WRAPPERS))
    def test_setup_applies_the_override_to_the_loaded_tokenizer(
        self, task, monkeypatch
    ):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.utils import trust_remote

        class StopAfterTemplateError(RuntimeError):
            pass

        class ModelSentinel:
            def get_nb_trainable_parameters(self):
                raise StopAfterTemplateError

        marker = "{% for message in messages %}OVERRIDE{% endfor %}"
        training = {"quantization": "none", "batch_size": 2}
        if task == "grpo":
            training.update({"num_generations": 2, "reward_fn": "format"})
        cfg = SoupConfig(
            base="model",
            task=task,
            data={"train": "data.jsonl", "chat_template": marker},
            training=training,
        )

        monkeypatch.setattr(
            trust_remote, "model_requires_trust_remote_code", lambda _base: False
        )
        monkeypatch.setattr(
            trust_remote,
            "resolve_trust_remote_code",
            lambda *_args, **_kwargs: False,
        )
        module_name, class_name, _method_name = _RUNTIME_WRAPPERS[task]
        wrapper_class = getattr(import_module(module_name), class_name)

        def fake_setup(instance, _cfg, _tcfg):
            instance.tokenizer = SimpleNamespace(chat_template="SHIPPED")
            instance.model = ModelSentinel()

        monkeypatch.setattr(wrapper_class, "_setup_transformers", fake_setup)
        wrapper = wrapper_class(cfg, device="cpu")
        with pytest.raises(StopAfterTemplateError):
            wrapper.setup({"train": []})
        assert wrapper.tokenizer.chat_template == marker, task

    def test_grpo_applies_override_before_fallback(self):
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        source = inspect.getsource(GRPOTrainerWrapper.setup)
        assert source.index("apply_chat_template_override") < source.index(
            "if not getattr(self.tokenizer, \"chat_template\""
        )

    def test_online_dpo_applies_override_before_fallback(self):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        source = inspect.getsource(OnlineDPOTrainerWrapper._setup_transformers)
        assert source.index("apply_chat_template_override") < source.index(
            "if self.tokenizer.chat_template is None"
        )

    def test_online_dpo_override_beats_fallback_on_a_loaded_tokenizer(self, tmp_path):
        _requires_train_extra()
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        weights = _tiny_llama_dir(tmp_path)
        marker = "{% for message in messages %}OVERRIDE{% endfor %}"
        cfg = SoupConfig(
            base=str(weights),
            task="online_dpo",
            data={"train": "data.jsonl", "chat_template": marker},
            training={
                "quantization": "none",
                "online_dpo_judge": "ollama://judge",
                "lora": {
                    "r": 4,
                    "alpha": 8,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
        )
        wrapper = OnlineDPOTrainerWrapper(cfg, device="cpu")
        wrapper._setup_transformers(cfg, cfg.training)
        assert wrapper.tokenizer.chat_template == marker


class TestNonChatTasksRejectTheField:
    @pytest.mark.parametrize(
        "task", ("pretrain", "embedding", "classifier", "reranker", "cross_encoder")
    )
    def test_task_is_named_in_error(self, task):
        from soup_cli.config.schema import SoupConfig

        training = {"num_labels": 2} if task in {"classifier", "reranker", "cross_encoder"} else {}
        with pytest.raises(ValidationError, match=rf"task='{task}'"):
            SoupConfig(
                base="model",
                task=task,
                data={"train": "data.jsonl", "chat_template": "chatml"},
                training=training,
            )

    @pytest.mark.parametrize("task", ("pretrain", "embedding"))
    def test_streaming_incompatibility_keeps_its_existing_error_precedence(self, task):
        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError, match="stream_layers"):
            SoupConfig(
                base="model",
                task=task,
                data={"train": "data.jsonl", "chat_template": "chatml"},
                training={
                    "stream_layers": True,
                    "batch_size": 1,
                    "quantization": "none",
                },
            )
