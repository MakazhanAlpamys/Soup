"""Tests for dynamic warmup resolution across trainer wrappers (#1431).

Verifies that:
1. All trainer wrappers pass warmup_ratio directly through as warmup_steps to
   HuggingFace TrainingArguments, so transformers dynamically resolves warmup
   over the trainer's actual optimizer steps (ceil(max_steps * warmup_ratio)).
2. GRPO warmup no longer shrinks by num_generations.
3. SFT with packing / multipack and pretrain with packing scale warmup to the
   real packed step count.
4. Unpacked SFT control matches the expected warmup within one step.
5. All 15 wrappers resolve warmup_steps as the configured fraction without multiplying
   warmup_ratio into a step count.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.warmup import resolve_trainer_warmup_steps
from tests import test_trl_preference_config_contract as contract

WORDS = ["<pad>", "<s>", "</s>", "<unk>", "apple", "banana", "cherry", "date"]
TRAINER_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer"
WARMUP_WRAPPERS = (
    "asr",
    "bco",
    "classifier",
    "distill",
    "dpo",
    "embedding",
    "grpo",
    "ipo",
    "kto",
    "online_dpo",
    "orpo",
    "pretrain",
    "reward_model",
    "sft",
    "simpo",
)


@pytest.fixture(scope="module")
def tiny_llama_model(tmp_path_factory: pytest.TempPathFactory) -> str:
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

    def test_invalid_ratio_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            resolve_trainer_warmup_steps(-0.01)
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            resolve_trainer_warmup_steps(1.0)
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            resolve_trainer_warmup_steps(1.5)


class TestGRPOWarmup:
    @pytest.mark.parametrize(
        ("prompts", "batch", "accum", "gens", "ratio", "epochs"),
        [
            (40, 4, 1, 4, 0.5, 1),
            (40, 8, 1, 8, 0.5, 1),
            (64, 4, 4, 4, 0.1, 1),
            (40, 4, 1, 4, 0.25, 2),
        ],
    )
    def test_grpo_warmup_matches_real_steps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_llama_model: str,
        prompts: int,
        batch: int,
        accum: int,
        gens: int,
        ratio: float,
        epochs: int,
    ) -> None:
        from soup_cli.trainer.grpo import GRPOTrainerWrapper

        monkeypatch.chdir(tmp_path)
        out_dir = str(tmp_path / "out-grpo").replace("\\", "/")
        cfg = load_config_from_string(
            f"base: {tiny_llama_model}\ntask: grpo\n"
            "data: {train: ./unused.jsonl, max_length: 64}\n"
            f"training:\n  epochs: {epochs}\n  batch_size: {batch}\n"
            f"  gradient_accumulation_steps: {accum}\n"
            f"  num_generations: {gens}\n  warmup_ratio: {ratio}\n  reward_fn: accuracy\n"
            "  quantization: none\n  lora: {r: 4, alpha: 8, target_modules: [q_proj, v_proj]}\n"
            f"output: {out_dir}\n"
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
        steps = (len(trainer.get_train_dataloader()) // accum) * epochs
        expected_warmup = math.ceil(steps * ratio)
        resolved_warmup = trainer.args.get_warmup_steps(steps)
        assert resolved_warmup == expected_warmup
        assert trainer.args.warmup_steps == ratio

        # Mutation check: the old row-count formula would have shrunk warmup by gens
        old_formula_steps = int(math.ceil(prompts / batch / accum) * epochs * ratio)
        if gens > 1:
            assert resolved_warmup > old_formula_steps


class TestSFTWarmup:
    @pytest.mark.parametrize("packing_mode", ["none", "packing", "multipack"])
    @pytest.mark.parametrize("epochs", [1, 2])
    def test_sft_warmup_scales_to_packed_optimizer_steps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_llama_model: str,
        packing_mode: str,
        epochs: int,
    ) -> None:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        monkeypatch.chdir(tmp_path)
        out_dir = str(tmp_path / f"out-sft-{packing_mode}-{epochs}").replace("\\", "/")
        extra = ""
        if packing_mode == "packing":
            extra = "  packing: true\n"
        elif packing_mode == "multipack":
            extra = "  multipack: true\n"

        cfg = load_config_from_string(
            f"base: {tiny_llama_model}\ntask: sft\n"
            "data: {train: ./unused.jsonl, format: chatml, max_length: 64}\n"
            f"training:\n  epochs: {epochs}\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
            "  lr: 1.0e-3\n  warmup_ratio: 0.5\n"
            "  scheduler: cosine\n  quantization: none\n"
            f"{extra}  lora: {{r: 4, alpha: 8}}\noutput: {out_dir}\n"
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
        steps = len(trainer.get_train_dataloader()) * epochs
        resolved_warmup = trainer.args.get_warmup_steps(steps)
        expected_warmup = math.ceil(steps * 0.5)
        assert resolved_warmup == expected_warmup
        assert trainer.args.warmup_steps == 0.5


class TestPretrainWarmup:
    def test_pretrain_warmup_ratio_preserved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tiny_llama_model: str,
    ) -> None:
        from soup_cli.trainer.pretrain import PretrainTrainerWrapper

        monkeypatch.chdir(tmp_path)
        out_dir = str(tmp_path / "out-pretrain").replace("\\", "/")
        cfg = load_config_from_string(
            f"base: {tiny_llama_model}\ntask: pretrain\n"
            "data: {train: ./unused.jsonl, max_length: 64}\n"
            "training:\n  epochs: 1\n  batch_size: 2\n  gradient_accumulation_steps: 1\n"
            "  lr: 1.0e-3\n  warmup_ratio: 0.25\n"
            f"  quantization: none\n  lora: {{r: 4, alpha: 8}}\noutput: {out_dir}\n"
        )
        rows = [{"text": f"apple {w} banana"} for w in WORDS[4:] * 4]
        wrapper = PretrainTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": rows})
        trainer = wrapper.trainer
        steps = len(trainer.get_train_dataloader())
        assert trainer.args.warmup_steps == 0.25
        assert trainer.args.get_warmup_steps(steps) == math.ceil(steps * 0.25)


@pytest.mark.parametrize("task", contract._ALL_SIX)
def test_live_preference_trainer_carries_the_warmup_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    wrapper = contract._build(tmp_path, monkeypatch, task)
    assert wrapper.trainer.args.warmup_steps == pytest.approx(0.03), (
        task,
        wrapper.trainer.args.warmup_steps,
    )


@pytest.mark.parametrize("name", WARMUP_WRAPPERS)
def test_wrapper_source_passes_the_ratio_not_a_row_count_estimate(name: str) -> None:
    tree = ast.parse((TRAINER_DIR / f"{name}.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "resolve_trainer_warmup_steps"
    ]
    assert len(calls) == 1, f"{name}: expected one resolve_trainer_warmup_steps call"
    assert ast.unparse(calls[0].args[0]) == "tcfg.warmup_ratio", name
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            assert "warmup_ratio" not in ast.unparse(node), (
                f"{name}: warmup_ratio is multiplied into a step count again"
            )
