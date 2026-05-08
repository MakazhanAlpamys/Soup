"""Tests for v0.40.3 Part B — #65 MultipackBatchSampler in HF Trainer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from soup_cli.utils.multipack_trainer import (
    attach_multipack_state,
    detect_arch_name,
    lengths_from_dataset,
    make_multipack_trainer_class,
)


class TestLengthsFromDataset:
    def test_none_returns_empty(self):
        assert lengths_from_dataset(None) == []

    def test_input_ids(self):
        ds = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [1, 2]},
        ]
        assert lengths_from_dataset(ds) == [3, 2]

    def test_length_fallback(self):
        ds = [{"length": 5}, {"length": 7}]
        assert lengths_from_dataset(ds) == [5, 7]

    def test_zero_for_missing_keys(self):
        ds = [{"foo": "bar"}, {}]
        assert lengths_from_dataset(ds) == [0, 0]

    def test_non_dict_row(self):
        ds = ["not-a-dict", 42]
        assert lengths_from_dataset(ds) == [0, 0]

    def test_non_iterable(self):
        assert lengths_from_dataset(123) == []

    def test_alternate_key(self):
        ds = [{"tokens": [1, 2, 3, 4]}]
        assert lengths_from_dataset(ds, key="tokens") == [4]

    def test_bool_length_falls_back(self):
        # bool is a subclass of int — must NOT be silently accepted as a
        # length value (matches v0.30.0 Candidate policy).
        ds = [{"length": True}]
        # bool is rejected via the explicit guard => fallback to 0.
        assert lengths_from_dataset(ds) == [0]


class TestDetectArchName:
    def test_none_returns_none(self):
        assert detect_arch_name(None) is None

    def test_from_config_architectures(self):
        model = SimpleNamespace(
            config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        )
        assert detect_arch_name(model) == "LlamaForCausalLM"

    def test_from_class_name_fallback(self):
        class FakeModel:
            pass

        assert detect_arch_name(FakeModel()) == "FakeModel"

    def test_empty_architectures_falls_back(self):
        class FakeModel:
            pass

        m = FakeModel()
        m.config = SimpleNamespace(architectures=[])
        assert detect_arch_name(m) == "FakeModel"

    def test_non_string_first_falls_back(self):
        class FakeModel:
            pass

        m = FakeModel()
        m.config = SimpleNamespace(architectures=[123])
        assert detect_arch_name(m) == "FakeModel"


class TestMakeMultipackTrainerClass:
    def test_returns_subclass(self):
        class Base:
            def _get_train_sampler(self):
                return "default"

        sub_cls = make_multipack_trainer_class(Base)
        assert issubclass(sub_cls, Base)
        assert sub_cls.__name__ == "MultipackBase"
        assert sub_cls.soup_multipack is True

    def test_falls_back_to_super_when_state_missing(self):
        class Base:
            def __init__(self):
                pass

            def _get_train_sampler(self):
                return "default-sampler"

        sub_cls = make_multipack_trainer_class(Base)
        instance = sub_cls()
        assert instance._get_train_sampler() == "default-sampler"

    def test_get_train_sampler_accepts_extra_args_for_hf_compat(self):
        """HF Trainer >=4.41 calls ``_get_train_sampler(train_dataset)``.

        Our override accepts ``*args, **kwargs`` so the signature works on
        both old and new transformers.
        """
        class Base:
            def __init__(self):
                pass

            def _get_train_sampler(self, *args, **kwargs):
                return ("default", args, kwargs)

        sub_cls = make_multipack_trainer_class(Base)
        instance = sub_cls()
        # Without state attached, the override delegates to super with the
        # same args — passes positional and keyword through.
        result = instance._get_train_sampler("some_dataset", flag=True)
        assert result[0] == "default"
        assert result[1] == ("some_dataset",)
        assert result[2] == {"flag": True}

    def test_returns_multipack_sampler_when_state_set(self):
        class Base:
            def __init__(self):
                pass

            def _get_train_sampler(self):
                return "default-sampler"

        sub_cls = make_multipack_trainer_class(Base)
        instance = sub_cls()
        attach_multipack_state(
            instance,
            lengths=[10, 20, 30, 40],
            max_seq_len=128,
            batch_size=2,
            seed=42,
        )
        sampler = instance._get_train_sampler()
        # Must be a MultipackBatchSampler (not the string default).
        from soup_cli.utils.multipack_sampler import MultipackBatchSampler

        assert isinstance(sampler, MultipackBatchSampler)


class TestAttachMultipackState:
    def _trainer(self):
        return MagicMock()

    def test_happy_path(self):
        t = self._trainer()
        attach_multipack_state(
            t, lengths=[1, 2, 3], max_seq_len=64, batch_size=4, seed=7,
        )
        assert t._soup_multipack_lengths == [1, 2, 3]
        assert t._soup_multipack_max_seq_len == 64
        assert t._soup_multipack_batch_size == 4
        assert t._soup_multipack_seed == 7

    def test_rejects_non_int_max_seq_len(self):
        with pytest.raises(TypeError):
            attach_multipack_state(
                self._trainer(),
                lengths=[1],
                max_seq_len="64",  # type: ignore[arg-type]
                batch_size=1,
            )

    def test_rejects_bool_batch_size(self):
        with pytest.raises(TypeError):
            attach_multipack_state(
                self._trainer(),
                lengths=[1],
                max_seq_len=64,
                batch_size=True,  # type: ignore[arg-type]
            )

    def test_rejects_zero_max_seq_len(self):
        with pytest.raises(ValueError):
            attach_multipack_state(
                self._trainer(), lengths=[1], max_seq_len=0, batch_size=1,
            )

    def test_rejects_negative_batch_size(self):
        with pytest.raises(ValueError):
            attach_multipack_state(
                self._trainer(), lengths=[1], max_seq_len=64, batch_size=-1,
            )

    def test_rejects_bool_seed(self):
        with pytest.raises(TypeError):
            attach_multipack_state(
                self._trainer(),
                lengths=[1],
                max_seq_len=64,
                batch_size=1,
                seed=True,  # type: ignore[arg-type]
            )

    def test_rejects_empty_lengths(self):
        with pytest.raises(ValueError, match="empty"):
            attach_multipack_state(
                self._trainer(), lengths=[], max_seq_len=64, batch_size=1,
            )


class TestSftAndPretrainWiringDeferred:
    """v0.40.3 deferred #65 live wiring after the adversarial review surfaced
    a HF Trainer DataLoader sampler-vs-batch-sampler shape mismatch. The SFT
    and Pretrain wrappers print a yellow advisory and fall back to the
    standard sampler when ``multipack: true``. Live wiring lands in v0.40.4.
    """

    def test_sft_emits_deferred_advisory(self):
        text = Path("soup_cli/trainer/sft.py").read_text(encoding="utf-8")
        # Case-insensitive — comment uses DEFERRED, console string uses deferred.
        assert "v0.40.4" in text and "deferred" in text.lower()
        # The active wiring (factory call) MUST NOT appear in the SFT path
        # — the multipack subclass is not built or instantiated.
        assert "make_multipack_trainer_class(SFTTrainer)" not in text

    def test_pretrain_emits_deferred_advisory(self):
        text = Path("soup_cli/trainer/pretrain.py").read_text(encoding="utf-8")
        assert "v0.40.4" in text and "deferred" in text.lower()
        assert "make_multipack_trainer_class(SFTTrainer)" not in text


class TestSamplerRespectsArchitectureAllowlist:
    def test_unknown_arch_loud_fails(self):
        from soup_cli.utils.multipack_sampler import (
            validate_multipack_architecture,
        )

        with pytest.raises(ValueError):
            validate_multipack_architecture("NotAModelForCausalLM")


class TestFactoryCaching:
    """Confirm `lru_cache` caches the dynamically-created subclass."""

    def test_same_base_returns_same_subclass(self):
        class Base:
            pass

        sub1 = make_multipack_trainer_class(Base)
        sub2 = make_multipack_trainer_class(Base)
        assert sub1 is sub2

    def test_different_bases_produce_different_subclasses(self):
        class A:
            pass

        class B:
            pass

        assert make_multipack_trainer_class(A) is not make_multipack_trainer_class(B)


class TestLengthsAllZeroLogsWarning:
    def test_all_zero_logs_warning(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="soup_cli.utils.multipack_trainer")
        ds = [{"foo": "bar"}, {"baz": "qux"}]
        result = lengths_from_dataset(ds)
        assert result == [0, 0]
        assert any("all zeros" in rec.message for rec in caplog.records)

    def test_partial_zero_does_not_log(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="soup_cli.utils.multipack_trainer")
        ds = [{"input_ids": [1, 2, 3]}, {}]
        result = lengths_from_dataset(ds)
        assert result == [3, 0]
        assert not any("all zeros" in rec.message for rec in caplog.records)


class TestRealTrainerIntegration:
    """Mix the override into the real `transformers.Trainer` MRO if available."""

    def test_real_trainer_subclass_constructs(self):
        try:
            from transformers import Trainer
        except ImportError:
            pytest.skip("transformers not installed")
        sub = make_multipack_trainer_class(Trainer)
        assert issubclass(sub, Trainer)
        # MRO must place our override above Trainer.
        assert sub.__mro__[1] is Trainer

