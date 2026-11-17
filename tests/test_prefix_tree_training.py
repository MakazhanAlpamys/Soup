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
def _config(training: str = "", data: str = "", top: str = "task: sft\n") -> Any:
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        f"base: some/model\n{top}backend: transformers\n"
        f"data:\n  train: data.jsonl\n{data}"
        f"training:\n  prefix_tree: true\n{training}"
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
                "training:\n  prefix_tree: true\n"
            )

    def test_refused_on_the_mlx_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="not supported on the mlx backend"):
            load_config_from_string(
                "base: some/model\ntask: sft\nbackend: mlx\n"
                "data:\n  train: data.jsonl\ntraining:\n  prefix_tree: true\n"
            )

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
        with pytest.raises(NotImplementedError, match="single-process"):
            trainer.get_train_dataloader()

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

        with pytest.raises(ValueError, match="needs a dataset with a length"):
            examples_from_dataset(iter([1, 2]), _collate)
        with pytest.raises(ValueError, match="no training dataset"):
            examples_from_dataset(None, _collate)

    def test_a_collator_that_merges_rows_is_refused(self):
        pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree_trainer import examples_from_dataset

        def merging(features):
            return _collate(features[:1])

        with pytest.raises(RuntimeError, match="one row per example"):
            examples_from_dataset(_shared_dataset(), merging)

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


def _wrapper(root, monkeypatch, *, stream, prefix_tree=True, batch_size=1, turns=_TURNS):
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
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
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
