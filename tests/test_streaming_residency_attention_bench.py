"""Construction selection must be explicit and scoped; CPU tests do not qualify CUDA."""

import sys
from types import SimpleNamespace

import pytest

from benchmarks.harness.streaming_residency_attention_bench import forced_attention_backend


@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
@pytest.mark.parametrize("fail", [False, True])
def test_constructor_selection_and_descriptor_restoration(monkeypatch, backend, fail):
    calls = []

    class Factory:
        @classmethod
        def from_config(cls, config, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(config=SimpleNamespace(
                _attn_implementation=kwargs["attn_implementation"],
            ))

    descriptor = Factory.__dict__["from_config"]
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=Factory))
    with pytest.raises(RuntimeError) if fail else __import__("contextlib").nullcontext():
        with forced_attention_backend(backend) as state:
            Factory.from_config(object(), torch_dtype="sentinel")
            assert state["constructed_backends"] == [backend]
            if fail:
                raise RuntimeError("after construction")
    assert Factory.__dict__["from_config"] is descriptor
    assert calls == [{"attn_implementation": backend, "torch_dtype": "sentinel"}]


def test_existing_attention_override_is_refused(monkeypatch):
    class Factory:
        @classmethod
        def from_config(cls, *_args, **_kwargs):
            raise AssertionError("must fail before constructing")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=Factory))
    with forced_attention_backend("sdpa"):
        with pytest.raises(ValueError, match="override already present"):
            Factory.from_config(object(), attn_implementation="eager")
