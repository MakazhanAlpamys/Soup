"""``training.prefix_tree`` end to end: the config key, the trainer wiring, the
model check, and a real ``SFTTrainerWrapper`` run (resident and layer-streamed).

The arithmetic of a prefix-tree row is in ``tests/test_prefix_tree.py``. Here:

- the schema accepts the key where it is wired and names what to change where it
  is not;
- the trainer subclass replaces the TRAINING dataloader only, and only once
  state is attached;
- ``check_prefix_tree_model`` picks the mask form a model needs and refuses a
  model that does not compute the rows faithfully;
- a real run trains on 4D-mask batches whose loss, step by step, is the loss of
  the ordinary batch holding the same examples.
"""

from __future__ import annotations

import io
import re
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
        for batch in batches:
            assert set(batch) == {"input_ids", "attention_mask", "position_ids", "labels"}
            rows, width = batch["input_ids"].shape
            assert rows == 1 and width <= 32
            assert batch["attention_mask"].shape == (rows, 1, width, width)
            seen += int((batch["labels"] != X).sum())
        assert seen == 12, "every trained token of every example, once"
        assert state.collator.examples == 6
        assert state.collator.tree_tokens < state.collator.source_tokens
        assert trainer.get_eval_dataloader() == "the base eval dataloader"

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
        assert examples == [(list(ids), list(labels)) for ids, labels in dataset]

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
        assert "boolean mask off by" in str(exc.value)


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


def _rows():
    return [
        {
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
        for prompt, answer in _TURNS
    ]


def _wrapper(root, monkeypatch, *, stream, prefix_tree=True, batch_size=1):
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
    wrapper.setup({"train": _rows()})
    return wrapper


@pytest.mark.parametrize("stream", [False, True], ids=["resident", "streamed"])
class TestARealRun:
    def test_trains_on_prefix_tree_batches_with_the_ordinary_loss(
        self, tmp_path, stream, capsys
    ):
        import torch

        from soup_cli.utils.prefix_tree_trainer import prefix_tree_state
        from tests.test_stream_resume_exact import _cpu_only

        with _cpu_only() as monkeypatch:
            buffer = io.StringIO()
            from rich.console import Console

            import soup_cli.trainer.sft as sft_module

            monkeypatch.setattr(sft_module, "console", Console(file=buffer, width=200))
            wrapper = _wrapper(tmp_path, monkeypatch, stream=stream)
            try:
                trainer = wrapper.trainer
                assert type(trainer).__name__.startswith("PrefixTree")
                assert (getattr(wrapper, "_stream_runtime", None) is not None) is stream
                state = prefix_tree_state(trainer)
                assert state is not None
                stats = state.sampler.stats
                assert stats.examples == len(_TURNS)
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

                def spy(model, inputs, *args, **kwargs):
                    plain = plain_batches[len(seen)]
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        want = float(model(**plain).loss)
                    model.train(was_training)
                    loss = real_step(model, inputs, *args, **kwargs)
                    seen.append(
                        {
                            "shape": tuple(inputs["input_ids"].shape),
                            "mask": tuple(inputs["attention_mask"].shape),
                            "has_positions": "position_ids" in inputs,
                            "examples": int(plain["input_ids"].shape[0]),
                            "plain_tokens": int(plain["attention_mask"].sum()),
                            "loss": float(loss.detach()),
                            "ordinary_loss": want,
                        }
                    )
                    return loss

                trainer.training_step = spy
                wrapper.train()
            finally:
                wrapper._close_stream_runtime()

        assert len(seen) == 2 * len(state.sampler), "two epochs of the planned batches"
        for step in seen:
            rows, width = step["shape"]
            assert rows == 1 and width <= 64
            assert step["mask"] == (rows, 1, width, width)
            assert step["has_positions"]
            # LoRA dropout is 0, so the two forwards differ only in layout.
            assert step["loss"] == pytest.approx(step["ordinary_loss"], rel=1e-4, abs=1e-5)
        assert sum(step["examples"] for step in seen) == 2 * len(_TURNS)
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
