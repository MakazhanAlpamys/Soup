"""``training.curriculum: true`` trains the sorted rows in their sorted order.

The SFT wrapper sorts ``dataset["train"]`` short to long, but the Trainer's
default sampler is random, so the sorted list was dealt out in a seeded random
order and a curriculum run saw no easier-to-harder progression at all. Driven
for real on CPU over a tiny checkpoint: the rewind flight recorder logs which
rows each step consumed, so the order the model saw is read back from the run
itself rather than inferred from the configuration.

After the sort, some paths replace or reorder the list (#2863): RAFT, with or without
``raft_epoch_shuffle``, keeps it; a pre-tokenised dataset, ``packing`` (TRL's ``bfd``
re-bins the rows longest first) and ``multipack`` (its FFD sampler owns the dataloader)
keep the default sampler, as before. Nothing is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: Answer lengths in words, deliberately out of order and all distinct, so the
#: sorted order is unique and a random permutation of it is not sorted.
_WORD_COUNTS = (5, 1, 7, 3, 8, 2, 6, 4)


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")


# A real, tiny checkpoint on disk. Duplicated from
# test_issue1223_evaluate_val_split.py deliberately, as the suites around it do,
# so this file stays runnable on its own.
def _tiny_llama_dir(root: Path, vocab: int = 64, hidden: int = 64) -> str:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = root / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))

    words = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "good", "answer", "bad"):
        words[word] = len(words)
    tok = Tokenizer(models.WordLevel(vocab=words, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    ).save_pretrained(str(weights))
    return str(weights)


def _rows() -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": " ".join(["hello"] * count)},
            ]
        }
        for count in _WORD_COUNTS
    ]


def _raft_rows() -> list[dict]:
    return [
        {
            "query": "hi",
            "golden_doc": "good answer",
            "distractor_docs": ["bad", "world"],
            "answer": " ".join(["hello"] * count),
        }
        for count in _WORD_COUNTS
    ]


def _build(tmp_path, monkeypatch, *, curriculum: bool, data=None, training=None, rows=None):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = _tiny_llama_dir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "max_length": 64,
                    "chat_template": "chatml",
                    **(data or {}),
                },
                "training": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "epochs": 1,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "curriculum": curriculum,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                    **(training or {}),
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": rows if rows is not None else _rows()})
    return wrapper


def _consumed(output_dir: str) -> list[tuple[int, int]]:
    """``(row, supervised tokens)`` in the order the run consumed them."""
    order = []
    for line in (Path(output_dir) / "rewind.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "batch":
            order.extend(zip(record["rows"], record["row_tokens"]))
    return order


class TestCurriculumTrainingOrder:
    def test_a_curriculum_run_consumes_rows_short_to_long(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, curriculum=True)
        wrapper.train()

        consumed = _consumed(wrapper._output_dir)
        rows = [row for row, _ in consumed]
        tokens = [count for _, count in consumed]
        assert len(rows) == len(_WORD_COUNTS)
        # Row ids index the sorted list the trainer was handed, so the
        # curriculum order is exactly 0, 1, 2, ...
        assert rows == list(range(len(_WORD_COUNTS)))
        assert tokens == sorted(tokens)
        assert len(set(tokens)) == len(tokens)

    def test_without_curriculum_the_sampler_stays_random(self, tmp_path, monkeypatch):
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, curriculum=False)

        assert wrapper.trainer.args.train_sampling_strategy == "random"


class TestWhichPathsKeepTheOrder:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ({}, True),
            ({"is_raft": True}, True),
            # RAFT's plain Trainer reads neither flag.
            ({"is_raft": True, "packing": True}, True),
            ({"is_raft": True, "multipack": True}, True),
            ({"pretokenized": True}, False),
            ({"pretokenized": True, "is_raft": True}, False),
            ({"packing": True}, False),
            ({"multipack": True}, False),
            ({"curriculum": False}, False),
        ],
        ids=[
            "text-vision-audio", "raft", "raft-packing", "raft-multipack", "pretokenised",
            "pretokenised-raft", "packing", "multipack", "no-curriculum",
        ],
    )
    def test_each_path(self, path, expected):
        from soup_cli.trainer.sft import curriculum_keeps_sorted_order

        flags = {
            "curriculum": True, "pretokenized": False, "is_raft": False,
            "packing": False, "multipack": False, **path,
        }
        assert curriculum_keeps_sorted_order(**flags) is expected

    def test_raft_epoch_shuffle_reads_the_sorted_rows_in_order(self, tmp_path, monkeypatch):
        """The shuffle re-permutes the documents inside each row per epoch; the rows
        themselves come in the sorted order, shortest answer first."""
        _requires_train_extra()
        wrapper = _build(
            tmp_path, monkeypatch, curriculum=True,
            data={"format": "raft", "raft_epoch_shuffle": True, "max_length": 128},
            rows=_raft_rows(),
        )

        assert wrapper.trainer.args.train_sampling_strategy == "sequential"
        lengths = [
            int(batch["input_ids"].shape[-1])
            for batch in wrapper.trainer.get_train_dataloader()
        ]
        assert len(lengths) == len(_WORD_COUNTS)
        assert lengths == sorted(lengths)
        assert len(set(lengths)) == len(lengths)

    def test_packing_keeps_the_random_sampler(self, tmp_path, monkeypatch):
        """``bfd`` packs the longest rows first, so a sequential read would run long to
        short: the sampler stays random, as before."""
        _requires_train_extra()
        wrapper = _build(tmp_path, monkeypatch, curriculum=True, training={"packing": True})

        assert wrapper.trainer.args.train_sampling_strategy == "random"
