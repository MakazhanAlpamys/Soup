"""#1850 — ``training.seed`` reaches MLX SFT.

``mlx_lm.tuner.trainer.train()`` calls ``iterate_batches`` without a seed, so
the shuffle draws from numpy's global RNG, and LoRA ``A`` init plus dropout
draw from ``mx.random``. The ``mlx_lm.lora`` CLI seeds both before training;
Soup called ``train()`` directly and seeded neither, so two runs of one
``soup.yaml`` trained on different batches in a different order.

These run the real wrapper on a real (tiny, random) mlx-lm model and replace
only upstream's ``train()``, which records what the RNGs produce at the moment
training would start, and the LoRA ``A`` matrices it would train from.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def _tiny_llama():
    from mlx_lm.models import llama

    return llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            tie_word_embeddings=True,
        )
    )


def _start_of_training(tmp_path, monkeypatch, **training) -> dict:
    """What a run sees when upstream's ``train()`` is entered."""
    import mlx.core as mx
    import mlx_lm.tuner.trainer  # noqa: F401 — registered before it is patched
    import numpy as np
    from mlx.utils import tree_flatten

    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    seen: dict = {}

    def _recording_train(*, model, **_kwargs):
        # The shuffle mlx-lm's iterate_batches would draw, and the next
        # mx.random draw (dropout masks come from the same generator).
        seen["batch_order"] = np.random.permutation(1000).tolist()
        seen["mx_draw"] = mx.random.uniform(shape=(8,)).tolist()
        seen["lora_a"] = [
            v.tolist() for k, v in tree_flatten(model.trainable_parameters()) if "lora_a" in k
        ]

    monkeypatch.setattr(sys.modules["mlx_lm.tuner.trainer"], "train", _recording_train)

    training.setdefault("epochs", 1)
    training.setdefault("batch_size", 1)
    cfg = SoupConfig(
        base="mlx-community/tiny",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="plaintext"),
        training=TrainingConfig(**training),
        output=str(tmp_path),
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.model = _tiny_llama()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hi"}] * 8, "val": []}
    wrapper.train()
    return seen


def _scramble_global_rngs():
    """Whatever ran before training must not decide what training sees."""
    import mlx.core as mx
    import numpy as np

    np.random.seed(None)
    mx.random.seed(int(np.random.randint(0, 2**31 - 1)))


class TestTheSeedReachesTraining:
    def test_same_seed_same_batch_order_dropout_and_lora_init(self, tmp_path, monkeypatch):
        _scramble_global_rngs()
        first = _start_of_training(tmp_path / "a", monkeypatch, seed=7)
        _scramble_global_rngs()
        second = _start_of_training(tmp_path / "b", monkeypatch, seed=7)

        assert first["batch_order"] == second["batch_order"]
        assert first["mx_draw"] == second["mx_draw"]
        assert first["lora_a"] and first["lora_a"] == second["lora_a"]

    def test_different_seeds_differ(self, tmp_path, monkeypatch):
        first = _start_of_training(tmp_path / "a", monkeypatch, seed=1)
        second = _start_of_training(tmp_path / "b", monkeypatch, seed=2)

        assert first["batch_order"] != second["batch_order"]
        assert first["lora_a"] != second["lora_a"]

    def test_an_unset_seed_uses_the_schema_default(self, tmp_path, monkeypatch):
        """The panel prints ``Seed: 42 (unset default)``; an unset run must be
        the run that line describes, not an unseeded one."""
        _scramble_global_rngs()
        unset = _start_of_training(tmp_path / "a", monkeypatch)
        _scramble_global_rngs()
        explicit = _start_of_training(tmp_path / "b", monkeypatch, seed=42)

        assert unset["batch_order"] == explicit["batch_order"]
        assert unset["lora_a"] == explicit["lora_a"]

    def test_seed_zero_is_a_seed(self, tmp_path, monkeypatch):
        """0 is a valid seed, not "unset": it must not fall back to 42."""
        zero_a = _start_of_training(tmp_path / "a", monkeypatch, seed=0)
        zero_b = _start_of_training(tmp_path / "b", monkeypatch, seed=0)
        default = _start_of_training(tmp_path / "c", monkeypatch)

        assert zero_a["batch_order"] == zero_b["batch_order"]
        assert zero_a["batch_order"] != default["batch_order"]

    def test_data_seed_zero_is_a_seed(self, tmp_path, monkeypatch):
        """``data_seed: 0`` is read with ``is None``, not truthiness: it orders
        the batches like seed 0 does, not like the ``seed`` it sits next to."""
        zero = _start_of_training(tmp_path / "a", monkeypatch, seed=7, data_seed=0)
        seed_zero = _start_of_training(tmp_path / "b", monkeypatch, seed=0)
        seed_seven = _start_of_training(tmp_path / "c", monkeypatch, seed=7)

        assert zero["batch_order"] == seed_zero["batch_order"]
        assert zero["batch_order"] != seed_seven["batch_order"]
        assert zero["lora_a"] == seed_seven["lora_a"]

    def test_data_seed_owns_the_batch_order(self, tmp_path, monkeypatch):
        """As on the transformers path: ``data_seed`` decides the data order,
        ``seed`` everything else."""
        a = _start_of_training(tmp_path / "a", monkeypatch, seed=1, data_seed=9)
        b = _start_of_training(tmp_path / "b", monkeypatch, seed=2, data_seed=9)

        assert a["batch_order"] == b["batch_order"]
        assert a["lora_a"] != b["lora_a"]


# --------------------------------------------------------------------------
# End to end: mlx-lm's real train() loop, nothing replaced
# --------------------------------------------------------------------------


class _CharTokenizer:
    """Just what mlx-lm's text dataset reads: ``encode`` and ``eos_token_id``.
    Ids stay inside the tiny model's 64-token vocabulary."""

    eos_token_id = 2

    def encode(self, text):
        return [1] + [3 + ord(c) % 60 for c in text]


def _train_for_real(tmp_path, **training):
    import mlx.core as mx

    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    training.setdefault("epochs", 2)
    training.setdefault("batch_size", 2)
    training.setdefault("lr", 1e-2)
    cfg = SoupConfig(
        base="mlx-community/tiny",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="plaintext"),
        training=TrainingConfig(logging_steps=1, gradient_accumulation_steps=1, **training),
        output=str(tmp_path),
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    # A checkpoint on disk is the same model every run; the tiny stand-in is
    # built from mx.random, so pin its weights. Then scramble every RNG: what
    # training sees must come from training.seed alone.
    mx.random.seed(1234)
    wrapper.model = _tiny_llama()
    _scramble_global_rngs()
    wrapper.tokenizer = _CharTokenizer()
    texts = [f"sample number {i} " + "abc" * i for i in range(12)]
    wrapper._dataset = {"train": [{"text": t} for t in texts], "val": []}
    result = wrapper.train()
    weights = mx.load(str(tmp_path / "adapters.safetensors"))
    return result, {k: v.tolist() for k, v in weights.items()}


class TestSameSeedSameLosses:
    """Mirrors #1846's MLX DPO check on the SFT path: two runs at one seed
    report the same losses and save the same adapter; another seed does not."""

    def test_same_seed_reproduces_losses_and_adapter(self, tmp_path):
        first, first_weights = _train_for_real(tmp_path / "a", seed=5)
        second, second_weights = _train_for_real(tmp_path / "b", seed=5)

        assert first["initial_loss"] == second["initial_loss"]
        assert first["final_loss"] == second["final_loss"]
        assert first_weights and first_weights == second_weights

    def test_another_seed_trains_differently(self, tmp_path):
        first, first_weights = _train_for_real(tmp_path / "a", seed=5)
        second, second_weights = _train_for_real(tmp_path / "b", seed=6)

        assert first_weights != second_weights
