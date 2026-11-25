"""``training.prefix_tree`` end to end: the config key, the trainer wiring, the
model check, and a real ``SFTTrainerWrapper`` run (resident and layer-streamed).

The arithmetic of a prefix-tree row is in ``tests/test_prefix_tree.py``. Here:

- the schema accepts the key where it is wired and names what to change where it
  is not;
- the trainer subclass replaces the TRAINING dataloader only, and only once
  state is attached;
- ``check_prefix_tree_model`` picks the mask form a model needs and refuses a
  model that does not compute the rows faithfully;
- a real run trains on 4D-mask batches whose loss, token count and entropy
  metric, step by step, are those of the ordinary batch holding the same
  examples, with one row per step and with two.
"""

from __future__ import annotations

import io
import re
from array import array
from typing import Any

import pytest

from soup_cli.utils.prefix_tree import IGNORE_INDEX

X = IGNORE_INDEX
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


# --------------------------------------------------------------------------
# the config key
# --------------------------------------------------------------------------
#: LoRA dropout (0.05 by default) is refused beside prefix_tree; every config
#: below that means to be accepted turns it off.
_NO_DROPOUT = "  lora:\n    dropout: 0.0\n"


def _config(training: str = "", data: str = "", top: str = "task: sft\n") -> Any:
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        f"base: some/model\n{top}backend: transformers\n"
        f"data:\n  train: data.jsonl\n{data}"
        f"training:\n  prefix_tree: true\n{_NO_DROPOUT}{training}"
    )


class TestConfig:
    def test_off_by_default(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string("base: some/model\ntask: sft\ndata:\n  train: d.jsonl\n")
        assert cfg.training.prefix_tree is False

    def test_accepted_for_text_sft(self):
        assert _config().training.prefix_tree is True

    def test_accepted_with_layer_streaming(self):
        """Layer streaming refuses packing and multipack; a prefix-tree step is
        still ``batch_size`` rows of ``max_length`` tokens, so it is allowed."""
        cfg = _config("  stream_layers: true\n  batch_size: 1\n")
        assert cfg.training.stream_layers is True and cfg.training.prefix_tree is True

    @pytest.mark.parametrize("other", ["packing", "multipack"])
    def test_refused_beside_another_row_layout(self, other):
        with pytest.raises(ValueError, match="decides which examples share a row"):
            _config(f"  {other}: true\n")

    def test_refused_beside_the_chunked_streaming_loss(self):
        with pytest.raises(ValueError, match="stream_loss_chunk"):
            _config("  stream_layers: true\n  batch_size: 1\n  stream_loss_chunk: 64\n")

    def test_refused_for_another_task(self):
        with pytest.raises(ValueError, match="supports task='sft' only"):
            _config(top="task: pretrain\n", data="  format: plaintext\n")

    def test_refused_for_raft(self):
        with pytest.raises(ValueError, match="raft"):
            _config(data="  format: raft\n")

    def test_refused_with_replay(self):
        with pytest.raises(ValueError, match="data.replay"):
            _config(data="  replay: old.jsonl\n")

    def test_refused_with_interleave(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="data.interleave"):
            load_config_from_string(
                "base: some/model\ntask: sft\nbackend: transformers\n"
                "data:\n  train: [a.jsonl, b.jsonl]\n  interleave: concat\n"
                "training:\n  prefix_tree: true\n" + _NO_DROPOUT
            )

    @pytest.mark.parametrize("backend", ["mlx", "unsloth"])
    def test_refused_on_a_backend_other_than_transformers(self, backend):
        """Only the HF Trainer path is wired; a switch accepted on another
        backend would silently do nothing."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="backend='transformers' only") as exc:
            load_config_from_string(
                f"base: some/model\ntask: sft\nbackend: {backend}\n"
                "data:\n  train: data.jsonl\ntraining:\n  prefix_tree: true\n" + _NO_DROPOUT
            )
        assert f"backend='{backend}'" in str(exc.value)
        assert "set prefix_tree: false" in str(exc.value)

    def test_refused_for_a_modality_that_is_not_text(self):
        with pytest.raises(ValueError, match="supports modality='text' only"):
            _config(top="task: sft\nmodality: vision\n", data="  format: llava\n")

    def test_refused_with_neftune(self):
        """NEFTune's noise is scaled by the row length and drawn once per row
        token: on a prefix-tree row it is not the ordinary batch's noise."""
        with pytest.raises(ValueError, match="training.neftune_alpha") as exc:
            _config("  neftune_alpha: 5.0\n")
        assert "Remove one of the two" in str(exc.value)
        assert _config("  neftune_alpha: null\n").training.prefix_tree is True

    @pytest.mark.parametrize("value", ["1", "0", "2", "'yes'", "'true'"])
    def test_a_value_that_is_not_a_bool_is_refused(self, value):
        """YAML ``1`` and a quoted ``'yes'`` would be coerced to ``true`` by a
        plain ``bool`` field; this switch takes a bool only."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="prefix_tree must be true or false"):
            load_config_from_string(
                "base: some/model\ntask: sft\ndata:\n  train: d.jsonl\n"
                f"training:\n  prefix_tree: {value}\n"
            )

    @pytest.mark.parametrize("lora", ["", "  lora:\n    dropout: 0.1\n"], ids=["default", "0.1"])
    def test_refused_beside_a_lora_dropout(self, lora):
        """In an ordinary batch each example draws its own dropout mask over a
        shared beginning; a prefix-tree row computes that beginning once, so in
        train mode the loss would differ by more than rounding. The default
        0.05 is refused too."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="cannot be combined with a LoRA dropout") as exc:
            load_config_from_string(
                "base: some/model\ntask: sft\nbackend: transformers\n"
                "data:\n  train: data.jsonl\n"
                f"training:\n  prefix_tree: true\n{lora}"
            )
        assert str(exc.value).rstrip().endswith(
            "Set training.lora.dropout: 0 or training.prefix_tree: false."
        )

    def test_a_full_fine_tune_needs_no_dropout_setting(self):
        """Without an adapter the LoRA dropout is never applied (lora.r 0)."""
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: some/model\ntask: sft\nbackend: transformers\n"
            "data:\n  train: data.jsonl\n"
            "training:\n  prefix_tree: true\n  quantization: none\n  lora:\n    r: 0\n"
        )
        assert cfg.training.prefix_tree and cfg.training.lora.dropout == 0.05

    def test_refused_with_curriculum(self):
        """The sampler decides the order of the examples; a curriculum order
        would be dropped without a word."""
        with pytest.raises(ValueError, match="training.curriculum") as exc:
            _config("  curriculum: true\n")
        assert "Remove one of the two" in str(exc.value)

    def test_the_task_helper(self):
        from soup_cli.utils.prefix_tree_trainer import supports_prefix_tree

        assert supports_prefix_tree("sft")
        assert not supports_prefix_tree("pretrain")
        assert not supports_prefix_tree("dpo")


# --------------------------------------------------------------------------
# the trainer subclass
# --------------------------------------------------------------------------
def _example(prompt, answer):
    return list(prompt) + list(answer), [X] * len(prompt) + list(answer)


def _collate(features):
    import torch

    width = max(len(ids) for ids, _ in features)
    ids = torch.zeros((len(features), width), dtype=torch.long)
    mask = torch.zeros((len(features), width), dtype=torch.long)
    labels = torch.full((len(features), width), X, dtype=torch.long)
    for row, (tokens, row_labels) in enumerate(features):
        ids[row, : len(tokens)] = torch.tensor(tokens)
        mask[row, : len(tokens)] = 1
        labels[row, : len(tokens)] = torch.tensor(row_labels)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class _BaseTrainer:
    """The slice of ``transformers.Trainer`` the subclass touches."""

    def __init__(self, dataset, processes=1):
        self.train_dataset = dataset
        self.data_collator = _collate
        self.args = type(
            "Args", (), {"dataloader_num_workers": 0, "dataloader_pin_memory": False}
        )()
        self.accelerator = type("Accelerator", (), {"num_processes": processes})()

    def get_train_dataloader(self):
        return "the base dataloader"

    def get_eval_dataloader(self):
        return "the base eval dataloader"


def _shared_dataset():
    system = list(range(1, 21))
    return [_example(system + [30 + n], [40 + n, 50 + n]) for n in range(6)]


class TestTrainerClass:
    def test_cached_and_named(self):
        from soup_cli.utils.prefix_tree_trainer import make_prefix_tree_trainer_class

        cls = make_prefix_tree_trainer_class(_BaseTrainer)
        assert cls is make_prefix_tree_trainer_class(_BaseTrainer)
        assert cls.__name__ == "PrefixTree_BaseTrainer" and issubclass(cls, _BaseTrainer)
        assert cls.soup_prefix_tree is True

    def test_without_state_it_is_the_base_class(self):
        from soup_cli.utils.prefix_tree_trainer import make_prefix_tree_trainer_class

        trainer = make_prefix_tree_trainer_class(_BaseTrainer)(_shared_dataset())
        assert trainer.get_train_dataloader() == "the base dataloader"

    def test_with_state_it_yields_prefix_tree_batches(self):
        pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree_trainer import (
            TREE_MASK_KEY,
            attach_prefix_tree_state,
            examples_from_dataset,
            make_prefix_tree_trainer_class,
            prefix_tree_state,
        )

        dataset = _shared_dataset()
        trainer = make_prefix_tree_trainer_class(_BaseTrainer)(dataset)
        state = attach_prefix_tree_state(
            trainer,
            examples=examples_from_dataset(dataset, _collate),
            max_tokens=32,
            batch_size=1,
            pad_token_id=0,
            seed=1,
        )
        assert prefix_tree_state(trainer) is state
        # 20 shared + (1 + 2) for the first example, 1 + 3 per further one.
        assert state.sampler.stats.rows == 2 and state.sampler.stats.ratio > 2.0
        loader = trainer.get_train_dataloader()
        batches = list(loader)
        assert len(batches) == len(loader) == 2
        seen = 0
        tokens = 0
        for batch in batches:
            assert set(batch) == {
                "input_ids", "attention_mask", "position_ids", "labels", TREE_MASK_KEY
            }
            rows, width = batch["input_ids"].shape
            assert rows == 1 and width <= 32
            assert batch[TREE_MASK_KEY].shape == (rows, 1, width, width)
            # 2D for the trainer's books: examples per token, 0 for padding.
            assert batch["attention_mask"].shape == (rows, width)
            tokens += int(batch["attention_mask"].sum())
            seen += int((batch["labels"] != X).sum())
        assert seen == 12, "every trained token of every example, once"
        assert tokens == sum(len(ids) for ids, _ in dataset), "the ordinary batch's tokens"
        assert state.collator.examples == 6
        assert state.collator.tree_tokens < state.collator.source_tokens
        assert trainer.get_eval_dataloader() == "the base eval dataloader"

    def test_the_model_gets_the_4d_mask_and_the_trainer_keeps_the_2d_one(self):
        """``compute_loss`` of the base class reads ``inputs["attention_mask"]``
        after the forward (TRL: token count, entropy weights). The model's call
        must see the 4D mask; the base class must still see the 2D one."""
        torch = pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree_trainer import (
            TREE_MASK_KEY,
            make_prefix_tree_trainer_class,
        )

        seen = {}

        class Model(torch.nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                seen["model"] = attention_mask
                seen["model_kwargs"] = sorted(kwargs)
                return "outputs"

        class Base(_BaseTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                outputs = model(**inputs)
                seen["trainer"] = inputs["attention_mask"]
                seen["passed"] = (return_outputs, num_items_in_batch)
                return ("loss", outputs) if return_outputs else "loss"

        trainer = make_prefix_tree_trainer_class(Base)(_shared_dataset())
        model = Model()
        counts = torch.tensor([[2, 1, 1]])
        tree = torch.ones(1, 1, 3, 3, dtype=torch.bool).tril()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": counts,
            "position_ids": torch.tensor([[0, 1, 1]]),
            TREE_MASK_KEY: tree,
        }
        assert trainer.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=7) == (
            "loss",
            "outputs",
        )
        assert seen["model"] is tree and seen["trainer"] is counts
        assert seen["model_kwargs"] == ["position_ids"], "the key itself never reaches the model"
        assert seen["passed"] == (True, 7)
        assert not model._forward_pre_hooks, "the hook outlived the step"

        # An ordinary batch (evaluation) goes straight through.
        plain = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        assert trainer.compute_loss(model, plain) == "loss"
        assert seen["model"] is plain["attention_mask"]

        # A forward that raises leaves no hook behind either.
        class Raising(Model):
            def forward(self, **kwargs):
                raise RuntimeError("boom")

        broken = Raising()
        with pytest.raises(RuntimeError, match="boom"):
            trainer.compute_loss(broken, dict(inputs, **{TREE_MASK_KEY: tree}))
        assert not broken._forward_pre_hooks

    def test_several_processes_are_refused(self):
        pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree_trainer import (
            attach_prefix_tree_state,
            make_prefix_tree_trainer_class,
        )

        dataset = _shared_dataset()
        trainer = make_prefix_tree_trainer_class(_BaseTrainer)(dataset, processes=2)
        attach_prefix_tree_state(
            trainer, examples=dataset, max_tokens=32, batch_size=1, pad_token_id=0
        )
        with pytest.raises(NotImplementedError, match="single-process") as exc:
            trainer.get_train_dataloader()
        assert str(exc.value).endswith("Set training.prefix_tree: false.")

    def test_bad_state_is_refused(self):
        from soup_cli.utils.prefix_tree_trainer import attach_prefix_tree_state

        trainer = _BaseTrainer(_shared_dataset())
        with pytest.raises(ValueError, match="max_tokens"):
            attach_prefix_tree_state(
                trainer, examples=_shared_dataset(), max_tokens=0, batch_size=1, pad_token_id=0
            )
        with pytest.raises(TypeError, match="pad_token_id"):
            attach_prefix_tree_state(
                trainer, examples=_shared_dataset(), max_tokens=8, batch_size=1, pad_token_id=None
            )


class TestExamplesFromDataset:
    def test_reads_every_row_through_the_collator(self):
        pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree_trainer import examples_from_dataset

        dataset = _shared_dataset() + [_example([7], [8])] * 70  # more than one read chunk
        examples = examples_from_dataset(dataset, _collate)
        assert [(list(ids), list(labels)) for ids, labels in examples] == [
            (list(ids), list(labels)) for ids, labels in dataset
        ]
        # Compact: 4-byte ints, not Python lists of int objects.
        assert all(
            isinstance(part, array) and part.typecode == "i" for pair in examples for part in pair
        )

    def test_a_dataset_without_a_length_is_refused(self):
        from soup_cli.utils.prefix_tree_trainer import examples_from_dataset

        with pytest.raises(ValueError, match="needs a dataset with a length") as exc:
            examples_from_dataset(iter([1, 2]), _collate)
        assert str(exc.value).endswith(
            "Set data.streaming: false or training.prefix_tree: false."
        )
        with pytest.raises(ValueError, match="no training dataset"):
            examples_from_dataset(None, _collate)

    def test_a_collator_that_merges_rows_is_refused(self):
        pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree_trainer import examples_from_dataset

        def merging(features):
            return _collate(features[:1])

        with pytest.raises(RuntimeError, match="one row per example") as exc:
            examples_from_dataset(_shared_dataset(), merging)
        assert str(exc.value).endswith("Set training.prefix_tree: false.")

    def test_planning_is_announced_only_when_it_is_long(self):
        from soup_cli.utils.prefix_tree_trainer import PLAN_NOTICE_TOKENS, planning_notice

        assert planning_notice(10_000, 512) is None
        assert planning_notice(PLAN_NOTICE_TOKENS // 512, 512) is None
        said = planning_notice(1_000_000, 512)
        assert "1,000,000 training examples" in said and "512,000,000 tokens" in said
        assert "8.2 GB" in said and "17 min" in said


# --------------------------------------------------------------------------
# the model check
# --------------------------------------------------------------------------
class TestModelCheck:
    def _model(self):
        from tests.test_prefix_tree import _tiny_model

        return _tiny_model("llama")

    def test_a_plain_decoder_takes_the_boolean_mask(self):
        from soup_cli.utils.prefix_tree_trainer import check_prefix_tree_model

        verdict = check_prefix_tree_model(self._model(), pad_token_id=0)
        assert verdict.exact and verdict.mask == "bool"

    def test_a_model_that_adds_its_mask_gets_the_additive_form(self):
        """An attention that ADDS the mask to its scores turns a boolean mask into
        +1 for allowed pairs and +0 for forbidden ones: nothing is masked."""
        import torch

        from soup_cli.utils.prefix_tree_trainer import check_prefix_tree_model

        model = self._model()
        real = model.forward

        def forward(input_ids=None, attention_mask=None, **kwargs):
            if attention_mask is not None and attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    # what "scores + mask" does to a boolean mask
                    attention_mask = torch.ones_like(attention_mask)
                else:
                    attention_mask = attention_mask == 0
            return real(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        model.forward = forward
        verdict = check_prefix_tree_model(model, pad_token_id=0)
        assert verdict.exact and verdict.mask == "additive"

    def test_a_model_that_cannot_is_refused_before_any_step(self):
        from soup_cli.utils.prefix_tree_trainer import check_prefix_tree_model

        model = self._model()
        real = model.forward

        def forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
            return real(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        model.forward = forward
        with pytest.raises(ValueError, match="does not compute a prefix-tree row") as exc:
            check_prefix_tree_model(model, pad_token_id=0)
        assert "prefix_tree: false" in str(exc.value)
        assert "with a boolean mask" in str(exc.value)
        assert "with an additive mask" in str(exc.value)

    def test_real_examples_come_from_the_first_shared_row(self):
        from soup_cli.utils.prefix_tree_trainer import verification_examples

        examples = [_example([n], [n + 1]) for n in range(1, 11, 2)]
        assert verification_examples(examples, [(4,), (1, 3, 0, 2), (2,)]) == [
            (list(examples[i][0]), list(examples[i][1])) for i in (1, 3, 0)
        ]
        # No row with two members: the first two examples, packed as two trees.
        assert verification_examples(examples, [(3,), (0,)]) == [
            (list(examples[i][0]), list(examples[i][1])) for i in (0, 1)
        ]


def test_the_check_row_stays_within_the_budget_when_no_planned_row_holds_two():
    """The first two examples each fill ``max_tokens`` and share no beginning,
    so no planned row holds two (``batch_size: 1`` plans one per row) and the
    fallback pair is used. Packed whole it would be one row of twice the
    budget; the check gets shorter copies whose row fits, and still a row of
    two trees."""
    from soup_cli.utils.prefix_tree import pack_prefix_tree
    from soup_cli.utils.prefix_tree_trainer import verification_examples

    max_tokens = 64
    first = ([3 + k % 40 for k in range(max_tokens)], [X] * 32 + [5] * 32)
    second = ([50 + k % 40 for k in range(max_tokens)], [X] * 32 + [6] * 32)
    rows = [(0,), (1,)]
    whole = pack_prefix_tree([first, second], max_tokens=2 * max_tokens)
    assert len(whole) == 1 and len(whole[0].input_ids) == 2 * max_tokens, "the defect"

    chosen = verification_examples([first, second], rows, max_tokens=max_tokens)
    assert len(chosen) == 2
    for (ids, labels), (full_ids, full_labels) in zip(chosen, [first, second]):
        assert ids == full_ids[: len(ids)] and labels == full_labels[: len(labels)]
    total = sum(len(ids) for ids, _ in chosen)
    assert total <= max_tokens
    (row,) = pack_prefix_tree(chosen, max_tokens=total)
    assert len(row.input_ids) <= max_tokens and len(row.paths) == 2
    # Members of a planned row are used as they are.
    assert verification_examples([first, second], [(0, 1)], max_tokens=max_tokens) == [
        first, second
    ]


def test_the_rewind_notice_names_prefix_tree():
    from soup_cli.trainer.sft import rewind_skip_reason

    kwargs = dict(
        task="sft", pretokenized=False, is_raft=False, multipack=False, vision=False, audio=False
    )
    assert "prefix-tree" in rewind_skip_reason(prefix_tree=True, **kwargs)
    assert "prefix-tree" not in rewind_skip_reason(**kwargs)


# --------------------------------------------------------------------------
# a real run
# --------------------------------------------------------------------------
_SYSTEM = "the cat sat on the mat hello world hi yo the cat sat on the mat hello world"
_TURNS = (
    ("hi", "hello world"),
    ("hi", "yo yo"),
    ("the cat", "sat on the mat"),
    ("hello", "the cat sat"),
    ("yo", "hi"),
    ("world", "hello hello world"),
)
#: Enough for several rows of 64 tokens, so that a step of two rows exists.
_MORE_TURNS = _TURNS + (
    ("mat", "the cat"),
    ("sat", "on the mat hello"),
    ("hello world", "yo"),
    ("on", "world world hi"),
    ("cat", "hello yo hi"),
    ("the", "mat mat"),
    ("yo yo", "the cat sat on"),
    ("hi hi", "world"),
)


def _rows(turns=_TURNS):
    return [
        {
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
        for prompt, answer in turns
    ]


def _wrapper(
    root,
    monkeypatch,
    *,
    stream,
    prefix_tree=True,
    batch_size=1,
    turns=_TURNS,
    max_length=64,
    responses_only=True,
):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_v07204 import _tiny_llama_dir, _write_tiny_tokenizer

    weights, _, _ = _tiny_llama_dir(root, n_layers=2, tie=True, hidden=64, vocab=64)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(root / "cache"))
    monkeypatch.chdir(root)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": weights,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {
                    "train": "train.jsonl",
                    "max_length": max_length,
                    "chat_template": "chatml",
                    "train_on_responses_only": responses_only,
                },
                "training": {
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "stream_layers": stream,
                    "prefix_tree": prefix_tree,
                    "epochs": 2,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "lr": 1e-3,
                    # No LoRA dropout: the test compares a train-mode step's loss
                    # with a forward of the ordinary batch on the same weights.
                    "lora": {
                        "r": 4,
                        "alpha": 8,
                        "dropout": 0.0,
                        "target_modules": ["q_proj", "v_proj"],
                    },
                },
                "output": str(root / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": _rows(turns)})
    return wrapper


def _entropy(logits, weights):
    """TRL's entropy metric: per-token entropy averaged with the 2D mask."""
    import torch

    per_token = torch.distributions.Categorical(logits=logits.float()).entropy()
    return float((per_token * weights).sum() / weights.sum())


def test_no_row_is_wider_than_data_max_length(tmp_path):
    """Every VRAM pre-flight sizes a step as ``batch_size`` rows of
    ``data.max_length`` tokens. A prefix-tree row is planned within that width,
    so the pre-flights hold with the option on: here the examples together are
    several times ``data.max_length`` and every batch is still within it."""
    pytest.importorskip("torch")
    from soup_cli.utils.prefix_tree_trainer import TREE_MASK_KEY, prefix_tree_state
    from tests.test_stream_resume_exact import _cpu_only

    max_length = 64
    with _cpu_only() as monkeypatch:
        wrapper = _wrapper(
            tmp_path, monkeypatch, stream=False, batch_size=2, turns=_MORE_TURNS,
            max_length=max_length,
        )
        state = prefix_tree_state(wrapper.trainer)
        batches = list(wrapper.trainer.get_train_dataloader())
    assert state.max_tokens == max_length
    assert state.sampler.stats.tree_tokens > 3 * max_length, "vacuous: one row holds it all"
    widths = [int(batch["input_ids"].shape[1]) for batch in batches]
    assert max(widths) <= max_length, widths
    assert max(widths) > max_length // 2, "vacuous: rows far below the bound"
    for batch in batches:
        rows, width = batch["input_ids"].shape
        assert rows <= 2
        assert batch[TREE_MASK_KEY].shape[-1] == width <= max_length


def test_setup_warns_when_the_examples_share_nothing(tmp_path):
    """Trained on every token (``train_on_responses_only: false``), the examples
    have nothing to share but their first token, and the setup says so. (Its
    absence where a prompt is shared is asserted in ``TestARealRun``.)"""
    pytest.importorskip("torch")
    from rich.console import Console

    import soup_cli.trainer.sft as sft_module
    from tests.test_stream_resume_exact import _cpu_only

    buffer = io.StringIO()
    with _cpu_only() as monkeypatch:
        monkeypatch.setattr(sft_module, "console", Console(file=buffer, width=400))
        _wrapper(tmp_path, monkeypatch, stream=False, responses_only=False)
    said = _plain(buffer.getvalue())
    assert "these examples share almost no untrained beginning" in said, said


class TestDropoutInModel:
    def test_a_model_without_dropout_has_none(self):
        from soup_cli.utils.prefix_tree_trainer import dropout_in_model
        from tests.test_prefix_tree import _tiny_model

        assert dropout_in_model(_tiny_model("llama")) == []
        assert dropout_in_model(_tiny_model("gpt2")) == []

    def test_dropout_modules_and_config_rates_are_found(self):
        import transformers

        from soup_cli.utils.prefix_tree_trainer import dropout_in_model

        gpt2 = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=32)
        )
        found = dropout_in_model(gpt2)
        assert any("attn_dropout (p=0.1)" in item for item in found), found
        llama = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
                attention_dropout=0.1,
            )
        )
        assert dropout_in_model(llama) == ["config.attention_dropout=0.1"]


def test_setup_refuses_a_model_that_applies_dropout(tmp_path):
    """The schema refuses LoRA dropout; a model whose own config drops
    activations in training is refused by the setup, before any step."""
    pytest.importorskip("torch")
    import transformers

    from tests.test_stream_resume_exact import _cpu_only

    real_init = transformers.LlamaForCausalLM.__init__

    def init(self, config, *args, **kwargs):
        config.attention_dropout = 0.1
        real_init(self, config, *args, **kwargs)

    with _cpu_only() as monkeypatch:
        monkeypatch.setattr(transformers.LlamaForCausalLM, "__init__", init)
        with pytest.raises(ValueError, match="applies no dropout in training") as exc:
            _wrapper(tmp_path, monkeypatch, stream=False)
    assert "config.attention_dropout=0.1" in str(exc.value)
    assert str(exc.value).endswith("set training.prefix_tree: false.")


def test_setup_checks_the_model_within_data_max_length(tmp_path):
    """Examples that each fill ``data.max_length`` and share no beginning
    (trained on every token): every planned row holds one, so the setup checks
    the model on the fallback pair, which must fit one row of the budget."""
    pytest.importorskip("torch")
    import soup_cli.utils.prefix_tree_trainer as trainer_module
    from soup_cli.utils.prefix_tree_trainer import prefix_tree_state
    from tests.test_stream_resume_exact import _cpu_only

    words = ("hello", "world", "the", "cat", "sat", "on", "mat", "hi", "yo")
    turns = tuple(
        (words[k], " ".join(words[(k + j) % len(words)] for j in range(80)))
        for k in range(4)
    )
    seen: list = []
    real_check = trainer_module.check_prefix_tree_model

    def spy(model, *, pad_token_id, examples):
        seen.append([list(ids) for ids, _ in examples])
        return real_check(model, pad_token_id=pad_token_id, examples=examples)

    with _cpu_only() as monkeypatch:
        monkeypatch.setattr(trainer_module, "check_prefix_tree_model", spy)
        wrapper = _wrapper(tmp_path, monkeypatch, stream=False, turns=turns, responses_only=False)
        state = prefix_tree_state(wrapper.trainer)
    max_length = 64
    assert all(len(row) == 1 for row in state.sampler.rows), "vacuous: a row holds two"
    (checked,) = seen
    assert len(checked) == 2
    assert sum(len(ids) for ids in checked) <= max_length, [len(ids) for ids in checked]


def test_setup_refuses_a_model_that_cannot_before_any_step(tmp_path):
    """The check is wired into the wrapper's setup: a model that drops
    ``position_ids`` (so a prefix-tree row would see the wrong positions) is
    refused there, before a trainer step exists."""
    pytest.importorskip("torch")
    import transformers

    from tests.test_stream_resume_exact import _cpu_only

    real = transformers.LlamaForCausalLM.forward

    def forward(self, *args, position_ids=None, **kwargs):
        return real(self, *args, **kwargs)

    with _cpu_only() as monkeypatch:
        monkeypatch.setattr(transformers.LlamaForCausalLM, "forward", forward)
        with pytest.raises(ValueError, match="does not compute a prefix-tree row") as exc:
            _wrapper(tmp_path, monkeypatch, stream=False)
    assert "prefix_tree: false" in str(exc.value)


@pytest.mark.parametrize("stream", [False, True], ids=["resident", "streamed"])
class TestARealRun:
    @pytest.mark.parametrize(
        "batch_size, turns", [(1, _TURNS), (2, _MORE_TURNS)], ids=["one-row", "two-rows"]
    )
    def test_trains_on_prefix_tree_batches_with_the_ordinary_loss(
        self, tmp_path, stream, batch_size, turns
    ):
        import torch

        from soup_cli.utils.prefix_tree_trainer import TREE_MASK_KEY, prefix_tree_state
        from tests.test_stream_resume_exact import _cpu_only

        with _cpu_only() as monkeypatch:
            buffer = io.StringIO()
            from rich.console import Console

            import soup_cli.trainer.sft as sft_module

            monkeypatch.setattr(sft_module, "console", Console(file=buffer, width=200))
            if batch_size == 1:
                # The planning notice, printed by the setup: lowered so these few
                # examples cross it. The two-row case keeps the default and so
                # must not print it.
                import soup_cli.utils.prefix_tree_trainer as trainer_module

                monkeypatch.setattr(trainer_module, "PLAN_NOTICE_TOKENS", 1)
            wrapper = _wrapper(
                tmp_path, monkeypatch, stream=stream, batch_size=batch_size, turns=turns
            )
            try:
                trainer = wrapper.trainer
                assert type(trainer).__name__.startswith("PrefixTree")
                assert (getattr(wrapper, "_stream_runtime", None) is not None) is stream
                state = prefix_tree_state(trainer)
                assert state is not None
                stats = state.sampler.stats
                assert stats.examples == len(turns)
                assert stats.ratio > 1.5, stats
                said = _plain(buffer.getvalue())
                assert "Prefix-tree batches enabled" in said
                assert f"{stats.ratio:.2f}x fewer" in said
                notice = f"planning reads all {len(turns)} training examples"
                assert (notice in said) is (batch_size == 1), said
                assert "share almost no untrained beginning" not in said

                # Record the ordinary batch each prefix-tree batch was built from.
                plain_batches: list = []
                real_collator = trainer.data_collator

                def recording(features):
                    batch = real_collator(features)
                    plain_batches.append({key: value.clone() for key, value in batch.items()})
                    return batch

                trainer.data_collator = recording
                seen: list = []
                real_step = trainer.training_step

                # What the model itself was called with (after any pre-hook).
                model_masks: list = []

                def record(_module, _args, kwargs, _output):
                    model_masks.append(tuple(kwargs["attention_mask"].shape))

                recorder = wrapper.model.register_forward_hook(record, with_kwargs=True)

                def spy(model, inputs, *args, **kwargs):
                    plain = plain_batches[len(seen)]
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        ordinary = model(**plain)
                        want = float(ordinary.loss)
                        want_entropy = _entropy(ordinary.logits, plain["attention_mask"])
                    model.train(was_training)
                    del model_masks[:]
                    tokens_before = trainer._total_train_tokens
                    tree_mask = tuple(inputs[TREE_MASK_KEY].shape)
                    loss = real_step(model, inputs, *args, **kwargs)
                    seen.append(
                        {
                            "shape": tuple(inputs["input_ids"].shape),
                            "tree_mask": tree_mask,
                            "model_masks": list(model_masks),
                            "has_positions": "position_ids" in inputs,
                            "examples": int(plain["input_ids"].shape[0]),
                            "plain_tokens": int(plain["attention_mask"].sum()),
                            "loss": float(loss.detach()),
                            "ordinary_loss": want,
                            "num_tokens": trainer._total_train_tokens - tokens_before,
                            "entropy": trainer._metrics["train"]["entropy"][-1],
                            "ordinary_entropy": want_entropy,
                        }
                    )
                    return loss

                trainer.training_step = spy
                wrapper.train()
                recorder.remove()
                assert not wrapper.model._forward_pre_hooks, "a step left its hook behind"
            finally:
                wrapper._close_stream_runtime()

        assert len(seen) == 2 * len(state.sampler), "two epochs of the planned batches"
        for step in seen:
            rows, width = step["shape"]
            assert 1 <= rows <= batch_size and width <= 64
            assert step["tree_mask"] == (rows, 1, width, width)
            assert step["model_masks"] == [(rows, 1, width, width)], "the model saw a 2D mask"
            assert step["has_positions"]
            # LoRA dropout is 0, so the two forwards differ only in layout.
            assert step["loss"] == pytest.approx(step["ordinary_loss"], rel=1e-4, abs=1e-5)
            # The trainer's books are the ordinary batch's: every token of every
            # example counted once, and the entropy averaged over those tokens.
            assert step["num_tokens"] == step["plain_tokens"]
            assert step["entropy"] == pytest.approx(step["ordinary_entropy"], rel=1e-4)
        if batch_size > 1:
            assert any(step["shape"][0] == batch_size for step in seen), "vacuous: one row a step"
        assert sum(step["examples"] for step in seen) == 2 * len(turns)
        assert any(step["examples"] > 1 for step in seen), "vacuous: one example per row"
        packed = sum(step["shape"][1] for step in seen)
        ordinary = sum(step["plain_tokens"] for step in seen)
        assert packed < ordinary, (packed, ordinary)
        assert seen[-1]["loss"] != seen[0]["loss"], "vacuous: nothing trained"

    def test_off_is_the_ordinary_trainer(self, tmp_path, stream):
        from soup_cli.utils.prefix_tree_trainer import prefix_tree_state
        from tests.test_stream_resume_exact import _cpu_only

        with _cpu_only() as monkeypatch:
            wrapper = _wrapper(tmp_path, monkeypatch, stream=stream, prefix_tree=False)
            try:
                assert not type(wrapper.trainer).__name__.startswith("PrefixTree")
                assert prefix_tree_state(wrapper.trainer) is None
            finally:
                wrapper._close_stream_runtime()
