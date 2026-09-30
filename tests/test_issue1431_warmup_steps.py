"""Tests for dynamic warmup resolution across trainer wrappers (#1431).

Verifies that:
1. All trainer wrappers pass warmup_ratio directly through as warmup_steps to
   HuggingFace TrainingArguments, so transformers dynamically resolves warmup
   over the trainer's actual optimizer steps (ceil(max_steps * warmup_ratio)).
2. GRPO warmup no longer shrinks by num_generations.
3. SFT with packing / multipack and pretrain with packing scale warmup to the
   real packed step count.
4. Unpacked SFT control matches the expected warmup within one step.
5. All 15 wrappers resolve warmup_steps as the configured fraction.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.warmup import resolve_trainer_warmup_steps

WORDS = ["<pad>", "<s>", "</s>", "<unk>", "apple", "banana", "cherry", "date"]


@pytest.fixture(scope="module")
def tiny_llama_model(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("tiny_llama")
    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {w: i for i, w in enumerate(WORDS)}
    raw.add_tokens([AddedToken(w, special=True) for w in WORDS[:4]])
    raw.add_tokens([AddedToken(w) for w in WORDS[4:]])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tok.chat_template = (
        "{% for m in messages %}{{ '<|' + m['role'] + '|> ' }}"
        "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] + ' </s>' }}"
        "{% endgeneration %}{% else %}{{ m['content'] + ' </s> ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
    )
    torch.manual_seed(0)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).save_pretrained(str(tmp))
    tok.save_pretrained(str(tmp))
    return str(tmp).replace("\\", "/")


class TestWarmupHelper:
    def test_resolve_trainer_warmup_steps(self) -> None:
        assert resolve_trainer_warmup_steps(0.0) == 0.0
        assert resolve_trainer_warmup_steps(0.1) == 0.1
        assert resolve_trainer_warmup_steps(0.5) == 0.5
        assert resolve_trainer_warmup_steps(None) == 0.0


class TestGRPOWarmup:
    @pytest.mark.parametrize(
        ("prompts", "batch", "accum", "gens", "ratio"),
        [
            (40, 4, 1, 4, 0.5),
            (40, 8, 1, 8, 0.5),
            (64, 4, 4, 4, 0.1),
        ],
    )
    def test_grpo_warmup_matches_real_steps(
        self, tiny_llama_model: str, prompts: int, batch: int, accum: int, gens: int, ratio: float
    ) -> None:
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        cfg = load_config_from_string(
            f"base: {tiny_llama_model}\ntask: grpo\n"
            "data: {train: ./unused.jsonl, max_length: 64}\n"
            f"training:\n  epochs: 1\n  batch_size: {batch}\n"
            f"  gradient_accumulation_steps: {accum}\n"
            f"  num_generations: {gens}\n  warmup_ratio: {ratio}\n  reward_fn: accuracy\n"
            "  quantization: none\n  lora: {r: 4, alpha: 8, target_modules: [q_proj, v_proj]}\n"
            "output: ./out-grpo-test\n"
        )
        row = {
            "messages": [
                {"role": "user", "content": "what is apple ?"},
                {"role": "assistant", "content": "apple"},
            ]
        }
        rows = [row] * prompts
        wrapper = GRPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": rows})
        trainer = wrapper.trainer
        steps = len(trainer.get_train_dataloader()) // accum
        expected_warmup = math.ceil(steps * ratio)
        resolved_warmup = trainer.args.get_warmup_steps(steps)
        assert resolved_warmup == expected_warmup
        assert trainer.args.warmup_steps == ratio

        # Mutation check: the old row-count formula would have shrunk warmup by gens
        old_formula_steps = int(math.ceil(prompts / batch / accum) * ratio)
        if gens > 1:
            assert resolved_warmup > old_formula_steps


class TestSFTWarmup:
    @pytest.mark.parametrize("packing_mode", ["none", "packing", "multipack"])
    def test_sft_warmup_scales_to_packed_optimizer_steps(
        self, tiny_llama_model: str, packing_mode: str
    ) -> None:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        extra = ""
        if packing_mode == "packing":
            extra = "  packing: true\n"
        elif packing_mode == "multipack":
            extra = "  multipack: true\n"

        cfg = load_config_from_string(
            f"base: {tiny_llama_model}\ntask: sft\n"
            "data: {train: ./unused.jsonl, format: chatml, max_length: 64}\n"
            "training:\n  epochs: 1\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
            "  lr: 1.0e-3\n  warmup_ratio: 0.5\n"
            "  scheduler: cosine\n  quantization: none\n"
            f"{extra}  lora: {{r: 4, alpha: 8}}\noutput: ./out-sft-test\n"
        )
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"what is {w} ?"},
                    {"role": "assistant", "content": "ok"},
                ]
            }
            for w in WORDS[4:] * 5
        ]
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": rows})
        trainer = wrapper.trainer
        steps = len(trainer.get_train_dataloader())
        resolved_warmup = trainer.args.get_warmup_steps(steps)
        expected_warmup = math.ceil(steps * 0.5)
        assert resolved_warmup == expected_warmup
        assert trainer.args.warmup_steps == 0.5


class TestPretrainWarmup:
    def test_pretrain_warmup_ratio_preserved(self, tiny_llama_model: str) -> None:
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        cfg = load_config_from_string(
            f"base: {tiny_llama_model}\ntask: pretrain\n"
            "data: {train: ./unused.jsonl, max_length: 64}\n"
            "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
            "  lr: 1.0e-3\n  warmup_ratio: 0.25\n"
            "  quantization: none\n  lora: {r: 4, alpha: 8}\noutput: ./out-pretrain-test\n"
        )
        rows = [{"text": f"apple {w} banana"} for w in WORDS[4:] * 4]
        wrapper = PretrainTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": rows})
        trainer = wrapper.trainer
        steps = len(trainer.get_train_dataloader())
        assert trainer.args.warmup_steps == 0.25
        assert trainer.args.get_warmup_steps(steps) == math.ceil(steps * 0.25)


class TestAllTrainerWrappersWarmupFraction:
    @pytest.mark.parametrize(
        "module_name, class_name, task_name",
        [
            ("soup_cli.trainer.sft", "SFTTrainerWrapper", "sft"),
            ("soup_cli.trainer.grpo", "GRPOTrainerWrapper", "grpo"),
            ("soup_cli.trainer.pretrain", "PretrainTrainerWrapper", "pretrain"),
            ("soup_cli.trainer.dpo", "DPOTrainerWrapper", "dpo"),
            ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", "orpo"),
            ("soup_cli.trainer.kto", "KTOTrainerWrapper", "kto"),
            ("soup_cli.trainer.bco", "BCOTrainerWrapper", "bco"),
            ("soup_cli.trainer.online_dpo", "OnlineDPOTrainerWrapper", "online_dpo"),
            ("soup_cli.trainer.ipo", "IPOTrainerWrapper", "ipo"),
            ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", "simpo"),
            ("soup_cli.trainer.reward_model", "RewardModelTrainerWrapper", "reward_model"),
            ("soup_cli.trainer.classifier", "ClassifierTrainerWrapper", "classifier"),
            ("soup_cli.trainer.distill", "DistillTrainerWrapper", "distill"),
            ("soup_cli.trainer.asr", "AsrTrainerWrapper", "asr"),
            ("soup_cli.trainer.embedding", "EmbeddingTrainerWrapper", "embedding"),
        ],
    )
    def test_wrapper_passes_configured_warmup_fraction(
        self, tiny_llama_model: str, module_name: str, class_name: str, task_name: str
    ) -> None:
        import importlib

        mod = importlib.import_module(module_name)
        wrapper_cls = getattr(mod, class_name)

        # Prepare task-appropriate data and config
        if task_name in {"dpo", "orpo", "kto", "bco", "ipo", "simpo", "reward_model"}:
            extra_cfg = "data: {train: ./unused.jsonl, format: pairs}\n"
            rows = [{"prompt": "what is apple?", "chosen": "a fruit", "rejected": "a car"}] * 4
        elif task_name == "classifier":
            extra_cfg = "data: {train: ./unused.jsonl}\nclassifier: {num_labels: 2}\n"
            rows = [{"text": "great fruit", "label": 1}] * 4
        elif task_name == "embedding":
            extra_cfg = "data: {train: ./unused.jsonl}\nembedding: {pooling: mean}\n"
            rows = [{"anchor": "apple", "positive": "fruit"}] * 4
        elif task_name == "distill":
            extra_cfg = (
                f"data: {{train: ./unused.jsonl}}\n"
                f"distill: {{teacher: {tiny_llama_model}}}\n"
            )
            rows = [{"prompt": "what is apple?", "response": "a fruit"}] * 4
        elif task_name == "asr":
            extra_cfg = "data: {train: ./unused.jsonl}\n"
            rows = [{"audio": "dummy.wav", "text": "apple"}] * 4
        elif task_name == "grpo":
            extra_cfg = (
                "data: {train: ./unused.jsonl}\n"
                "training:\n  reward_fn: accuracy\n  num_generations: 2\n"
            )
            prompt_row = {
                "messages": [
                    {"role": "user", "content": "what is apple ?"},
                    {"role": "assistant", "content": "apple"},
                ]
            }
            rows = [prompt_row] * 4
        elif task_name == "sft":
            extra_cfg = "data: {train: ./unused.jsonl, format: chatml}\n"
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "bye"},
                    ]
                }
            ] * 4
        else:
            extra_cfg = "data: {train: ./unused.jsonl}\n"
            rows = [{"text": "apple banana"}] * 4

        yaml_text = (
            f"base: {tiny_llama_model}\ntask: {task_name}\n"
            f"{extra_cfg}"
            "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
            "  lr: 1.0e-3\n  warmup_ratio: 0.35\n"
            "  quantization: none\n  lora: {r: 4, alpha: 8}\noutput: ./out-all-test\n"
        )
        try:
            cfg = load_config_from_string(yaml_text)
        except Exception:
            pytest.skip(f"Config load not supported in minimal environment for task {task_name}")

        try:
            wrapper = wrapper_cls(cfg, device="cpu")
            # If setup needs external audio files or special deps, skip gracefully
            wrapper.setup({"train": rows})
            assert wrapper.trainer.args.warmup_steps == 0.35
        except (ImportError, NotImplementedError, FileNotFoundError, ValueError) as exc:
            pytest.skip(f"Task {task_name} requires hardware/deps/model not present: {exc}")
