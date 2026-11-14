"""``training.packed_lora`` end to end through ``SFTTrainerWrapper`` (GPU).

``pytest -m gpu --no-cov tests/test_packed_lora_e2e_gpu.py`` on a Linux CUDA machine (WSL2
included); skipped everywhere else. A tiny random Qwen2 checkpoint is trained by the real set-up
and train path of Soup twice from one seed, once as plain Soup and once with
``training.packed_lora``, and the runs are compared: the loss of every step, the columns TRL logs,
the rewind rows, and the adapter each one saved (reloaded onto a fresh 4-bit base). Also here: the
hook is off the model and the memory is back after training, two packed runs in one process agree,
and a step that cannot be packed (a left-padded batch) runs the stock forward and is counted.

The kernels against PyTorch are ``test_packed_lora_kernels_gpu.py`` and
``test_packed_lora_layers_gpu.py``; the real Qwen2.5-1.5B is ``test_packed_lora_smoke_gpu.py``.
"""

from __future__ import annotations

import gc
import json
import os
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")
pytest.importorskip("trl")

pytestmark = pytest.mark.gpu

STEPS = 20


def rows(n=64, seed=5):
    """Short chats from a few questions and answers (the tiny tokenizer knows these words), in an
    order that mixes their lengths so that a batch is padded."""
    generator = random.Random(seed)
    questions = ("hi", "yo", "hello world", "the cat sat on the mat")
    answers = ("hello world", "good answer", "the cat sat on the mat", "bad soup question",
               "hi yo")
    return [{"messages": [{"role": "user", "content": generator.choice(questions)},
                          {"role": "assistant", "content": generator.choice(answers)}]}
            for _ in range(n)]


def write_tokenizer(directory):
    """A byte-level BPE tokenizer trained on the words of ``rows()``, as ``vocab.json`` and
    ``merges.txt``. (A directory with a Qwen2 ``config.json`` loads as ``Qwen2Tokenizer``, which
    reads those two files and ignores a ``tokenizer.json``.)"""
    from tokenizers import ByteLevelBPETokenizer

    texts = ["hello world", "the cat sat on the mat", "good answer", "bad soup question", "hi yo",
             "user", "assistant", "system", "<|im_start|>user", "<|im_start|>assistant",
             "<|im_end|>"]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts * 8, vocab_size=400, min_frequency=1,
                            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    bpe.save_model(directory)
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|endoftext|>",
                   "pad_token": "<|endoftext|>", "model_max_length": 512,
                   "clean_up_tokenization_spaces": False}, handle)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny random Qwen2 in bf16 on disk, with an offline tokenizer (4-bit block counts of every
    projection are multiples of 256, which the kernels' NF4 decoder needs)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    directory = tmp_path_factory.mktemp("tiny_qwen2")
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=False, rms_norm_eps=1e-6)
    Qwen2ForCausalLM(config).to(torch.bfloat16).save_pretrained(str(directory))
    write_tokenizer(str(directory))
    return str(directory)


def make_wrapper(checkpoint, tmp_path, name, packed, *, lr=1e-3, left_padded=False, **training):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    tmp_path.mkdir(parents=True, exist_ok=True)
    tcfg = {"batch_size": 4, "gradient_accumulation_steps": 1, "quantization": "4bit",
            "epochs": 1, "logging_steps": 1, "save_steps": 100000, "lr": lr, "seed": 3407,
            "gradient_checkpointing": True,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.0}, **training}
    if packed is not None:
        tcfg["packed_lora"] = packed
    cfg = load_config_from_string(yaml.safe_dump({
        "base": checkpoint, "task": "sft", "backend": "transformers", "modality": "text",
        "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml",
                 "val_split": 0},
        "training": tcfg, "output": str(tmp_path / name)}))
    wrapper = SFTTrainerWrapper(cfg, device="cuda")
    wrapper.setup({"train": rows()})
    if left_padded:
        pad_on_the_left(wrapper.trainer)
    wrapper.trainer.args.max_steps = STEPS
    return wrapper


def pad_on_the_left(trainer):
    """Make the collator pad on the left: the tokens of each row move to the end of the row, so
    the batch is not right-padded."""
    inner = trainer.data_collator

    def collate(features):
        batch = inner(features)
        width = batch["attention_mask"].shape[1]
        lengths = batch["attention_mask"].sum(1).tolist()
        for key in ("input_ids", "attention_mask", "labels"):
            batch[key] = torch.stack([
                torch.roll(row, shifts=width - int(length), dims=0)
                for row, length in zip(batch[key], lengths)])
        return batch

    trainer.data_collator = collate


def finish(wrapper):
    result = wrapper.train()
    history = wrapper.trainer.state.log_history
    losses = [entry["loss"] for entry in history if "loss" in entry]
    rewind_path = os.path.join(wrapper._output_dir, "rewind.jsonl")
    rewind = None
    if os.path.exists(rewind_path):
        with open(rewind_path, encoding="utf-8") as handle:
            rewind = [json.loads(line) for line in handle if line.strip()]
    return {"result": result, "losses": losses, "history": history, "rewind": rewind,
            "output_dir": wrapper._output_dir}


def release(*objects):
    del objects
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def arms(checkpoint, tmp_path_factory):
    """Plain Soup and packed Soup, from one seed, through the real set-up and train path."""
    tmp_path = tmp_path_factory.mktemp("arms")
    out = {}
    plain = make_wrapper(checkpoint, tmp_path, "plain", None)
    out["plain"] = finish(plain)
    out["plain"]["eval_loss"] = plain.trainer.evaluate(
        eval_dataset=plain.trainer.train_dataset)["eval_loss"]
    out["plain"]["columns"] = [e for e in plain.trainer.state.log_history if "loss" in e]
    release(plain)
    packed = make_wrapper(checkpoint, tmp_path, "packed", {"enabled": True})
    out["run"] = packed._packed_lora_run
    out["hooked_model"] = packed.model.get_base_model()
    out["packed"] = finish(packed)
    out["packed"]["after_train_forward_on_instance"] = "forward" in out["hooked_model"].__dict__
    out["packed"]["columns"] = [e for e in packed.trainer.state.log_history if "loss" in e]
    release(packed)
    return out


class TestAgainstPlainSoup:
    def test_the_hook_ran_every_step(self, arms):
        run = arms["run"]
        assert run.steps_packed == STEPS and run.steps_stock == 0

    def test_step_one_is_the_same_loss(self, arms):
        plain, packed = arms["plain"]["losses"], arms["packed"]["losses"]
        assert len(plain) == len(packed) == STEPS
        assert packed[0] == pytest.approx(plain[0], rel=2e-3)

    def test_the_loss_curve_follows_plain_soups(self, arms):
        plain, packed = arms["plain"]["losses"], arms["packed"]["losses"]
        diffs = [abs(a - b) / abs(a) for a, b in zip(plain, packed)]
        print(f"[e2e] per-step relative loss difference: max {max(diffs):.4f}, "
              f"mean {sum(diffs) / len(diffs):.5f}")
        assert max(diffs) < 0.02
        assert sum(diffs) / len(diffs) < 0.005

    def test_the_columns_trl_logs_are_there_with_trls_numbers(self, arms):
        plain, packed = arms["plain"]["columns"], arms["packed"]["columns"]
        assert len(plain) == len(packed) == STEPS
        for want, got in zip(plain, packed):
            assert got.keys() == want.keys()
            assert {"entropy", "num_tokens", "mean_token_accuracy", "grad_norm"} <= got.keys()
            assert got["num_tokens"] == want["num_tokens"] > 0
        assert packed[0]["entropy"] == pytest.approx(plain[0]["entropy"], rel=5e-3)
        assert packed[0]["mean_token_accuracy"] == pytest.approx(
            plain[0]["mean_token_accuracy"], abs=0.02)
        assert packed[0]["grad_norm"] == pytest.approx(plain[0]["grad_norm"], rel=0.05)

    def test_the_rewind_log_has_the_same_rows(self, arms):
        want = [e for e in arms["plain"]["rewind"] if e["kind"] == "batch"]
        got = [e for e in arms["packed"]["rewind"] if e["kind"] == "batch"]
        assert len(want) == len(got) == STEPS
        for a, b in zip(want[:3], got[:3]):
            assert b["rows"] == a["rows"] and b["row_tokens"] == a["row_tokens"]
            assert b["row_loss"] == pytest.approx(a["row_loss"], rel=5e-3)

    def test_the_evaluation_loss_is_close_after_training(self, arms):
        assert arms["plain"]["eval_loss"] > 0
        assert arms["packed"]["losses"][-1] == pytest.approx(arms["plain"]["losses"][-1], rel=0.03)


class TestAfterTraining:
    def test_the_hook_is_off_the_model(self, arms):
        assert arms["packed"]["after_train_forward_on_instance"] is False

    def test_the_run_is_closed_and_summarised(self, arms):
        assert arms["run"].closed

    def test_the_adapter_reloads_and_scores_like_plain_softs(self, arms, checkpoint):
        """The adapter each arm saved, put on a fresh 4-bit base, scores a fixed batch alike."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        batch = tokenizer(
            ["hello world the cat sat on the mat good answer", "hi yo soup question bad"],
            return_tensors="pt", padding=True).to("cuda")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        losses = {}
        for arm in ("plain", "packed"):
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
            base = AutoModelForCausalLM.from_pretrained(
                checkpoint, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base, arms[arm]["output_dir"])
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                losses[arm] = model(**batch, labels=labels).loss.item()
            release(model, base)
        print(f"[e2e] reloaded adapter loss: plain {losses['plain']:.4f}, "
              f"packed {losses['packed']:.4f}")
        assert losses["packed"] == pytest.approx(losses["plain"], rel=0.03)


class TestRunsInOneProcess:
    def test_two_packed_runs_agree_and_give_the_memory_back(self, checkpoint, tmp_path):
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        first_wrapper = make_wrapper(checkpoint, tmp_path, "one", {"enabled": True})
        first = finish(first_wrapper)
        release(first_wrapper)
        after_first = torch.cuda.memory_allocated()
        second_wrapper = make_wrapper(checkpoint, tmp_path, "two", {"enabled": True})
        second = finish(second_wrapper)
        release(second_wrapper)
        after_second = torch.cuda.memory_allocated()
        # same seed, same data: the first steps are the same (stochastic rounding differs only
        # from the second backward on, and by a counter that restarts with the run)
        assert second["losses"][0] == pytest.approx(first["losses"][0], rel=1e-4)
        assert max(abs(a - b) / abs(a) for a, b in zip(first["losses"], second["losses"])) < 0.01
        print(f"[e2e] allocated before {base >> 20} MiB, after one run {after_first >> 20} MiB, "
              f"after two {after_second >> 20} MiB")
        assert after_second <= after_first + (16 << 20)
        assert after_first <= base + (64 << 20)


class TestAStepThatCannotBePacked:
    def test_a_left_padded_batch_runs_the_stock_forward_and_is_counted(
            self, checkpoint, tmp_path):
        wrapper = make_wrapper(checkpoint, tmp_path, "left", {"enabled": True},
                               left_padded=True)
        run = wrapper._packed_lora_run
        result = finish(wrapper)
        release(wrapper)
        assert run.steps_stock > 0
        assert run.stock_reasons.get("a batch that is not right-padded", 0) == run.steps_stock
        assert run.steps_packed + run.steps_stock == STEPS
        assert len(result["losses"]) == STEPS and all(v > 0 for v in result["losses"])
