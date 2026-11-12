"""``soup rewind`` previews a ``training.curriculum: true`` run.

With ``curriculum`` on, the SFT wrapper sorts the train rows short to long
before the trainer and the flight recorder see them, so the logged row ids and
the dataset fingerprint are taken over that sorted list. ``load_row_previews``
rebuilt the rows in file order, the fingerprints never matched, and every
curriculum run was told its unchanged dataset "differs from the one this run
trained on", with no previews. Driven for real: a tiny checkpoint trains on CPU
and writes the log, then the previews are rebuilt from the config the tracker
records for the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: Answer lengths in words, deliberately out of file order and all distinct, so
#: the sorted list differs from the file at every position but the shortest.
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


def _answer(count: int) -> str:
    return " ".join(["hello"] * count)


def _train_curriculum_run(tmp_path, monkeypatch):
    """Train for real; return the tracker's config dict and the rewind log path."""
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = _tiny_llama_dir(tmp_path)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": _answer(count)},
            ]
        }
        for count in _WORD_COUNTS
    ]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": str(tmp_path / "train.jsonl"),
                    "format": "chatml",
                    "val_split": 0.0,
                    "max_length": 64,
                    "chat_template": "chatml",
                },
                "training": {
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "epochs": 1,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "curriculum": True,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    # What `soup train` hands the wrapper, and what its tracker records.
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup(load_dataset(cfg.data))
    wrapper.train()
    recorded = json.loads(json.dumps(cfg.model_dump(), default=str))
    return recorded, Path(wrapper._output_dir) / "rewind.jsonl"


def _read_log(path: Path) -> tuple[str, list[tuple[int, int]]]:
    """The header's fingerprint, and every ``(row, supervised tokens)`` logged."""
    fingerprint = ""
    logged = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "header":
            fingerprint = record["dataset_fingerprint"]
        elif record.get("kind") == "batch":
            logged.extend(zip(record["rows"], record["row_tokens"]))
    return fingerprint, logged


class TestRewindPreviewsForCurriculumRuns:
    def test_an_unchanged_dataset_is_previewed_in_the_order_the_run_trained(
        self, tmp_path, monkeypatch
    ):
        _requires_train_extra()
        from soup_cli.utils.rewind import load_row_previews

        config, log_path = _train_curriculum_run(tmp_path, monkeypatch)
        fingerprint, logged = _read_log(log_path)
        assert sorted(row for row, _ in logged) == list(range(len(_WORD_COUNTS)))

        row_ids = [row for row, _ in logged]
        previews = load_row_previews(config, row_ids, expected_fingerprint=fingerprint)

        assert previews is not None, "an unchanged dataset was reported as changed"
        # Row i of a curriculum run is the i-th shortest answer. The token count
        # the recorder measured for each row agrees with the previewed text.
        for row in row_ids:
            assert previews[row] == _answer(sorted(_WORD_COUNTS)[row])
        by_tokens = sorted(logged, key=lambda pair: pair[1])
        assert [row for row, _ in by_tokens] == list(range(len(_WORD_COUNTS)))
