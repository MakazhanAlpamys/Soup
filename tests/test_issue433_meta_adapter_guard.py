"""#433 — adapter materialisation must be verified by its postcondition.

PEFT 0.18 creates LoRA parameters on ``meta`` in a streamed build, while PEFT
0.19 creates them with real storage.  ``materialize_meta_adapters`` therefore
returns zero for the newer, healthy path as well as for any path where no
parameter was repaired.  The count cannot distinguish those outcomes.

These tests assert the decision with the PEFT behaviour stubbed.  In
particular, the negative control deliberately leaves a trainable LoRA parameter
on ``meta`` and proves the streamed build refuses it by name.  That keeps the
guard testable on the repository's pinned PEFT and on CPU-only CI.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest


def _adapter_model(*, adapter_device: str, adapter_trainable: bool = True):
    import torch
    import torch.nn as nn

    class _AdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = nn.Parameter(
                torch.empty(1, device="meta"), requires_grad=False
            )
            self.lora_A = nn.Parameter(
                torch.empty(2, 2, device=adapter_device),
                requires_grad=adapter_trainable,
            )

    return _AdapterModel()


class TestAdapterMaterializationPostcondition:
    def test_zero_materialized_is_valid_when_adapter_is_already_real(self):
        from soup_cli.utils.layer_stream_runtime import (
            assert_trainable_adapters_materialized,
            materialize_meta_adapters,
        )

        model = _adapter_model(adapter_device="cpu")
        assert materialize_meta_adapters(model, device="cpu") == 0
        assert_trainable_adapters_materialized(model)

    def test_frozen_meta_base_is_deliberately_ignored(self):
        from soup_cli.utils.layer_stream_runtime import (
            assert_trainable_adapters_materialized,
        )

        model = _adapter_model(adapter_device="cpu")
        assert model.base_weight.is_meta
        assert_trainable_adapters_materialized(model)

    def test_trainable_meta_adapter_fails_and_names_the_parameter(self):
        from soup_cli.utils.layer_stream_runtime import (
            assert_trainable_adapters_materialized,
        )

        model = _adapter_model(adapter_device="meta")
        with pytest.raises(RuntimeError, match=r"lora_A"):
            assert_trainable_adapters_materialized(model)

    def test_frozen_meta_adapter_is_not_part_of_the_training_invariant(self):
        from soup_cli.utils.layer_stream_runtime import (
            assert_trainable_adapters_materialized,
        )

        model = _adapter_model(adapter_device="meta", adapter_trainable=False)
        assert_trainable_adapters_materialized(model)

    def test_trainable_meta_non_lora_parameter_is_out_of_scope(self):
        import torch
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import (
            assert_trainable_adapters_materialized,
        )

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier_weight = nn.Parameter(
                    torch.empty(2, 2, device="meta"), requires_grad=True
                )

        assert_trainable_adapters_materialized(_Model())


def test_streamed_build_refuses_a_materializer_that_skips_a_meta_adapter(monkeypatch):
    """Negative control: stub the capability and force the failure state.

    This test drives the real ``build_streamed_model`` call site.  If the
    postcondition is removed or placed before PEFT attaches the adapter, the
    install sentinel is reached and the test fails.
    """
    import torch
    import torch.nn as nn

    import soup_cli.utils.layer_stream_runtime as runtime

    class _Skeleton(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = nn.Parameter(torch.empty(1, device="meta"))

    model = _Skeleton()

    def fake_get_peft_model(base, _config):
        base.lora_A = nn.Parameter(torch.empty(2, 2, device="meta"))
        return base

    fake_peft = ModuleType("peft")
    fake_peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(runtime, "build_meta_skeleton", lambda *_a, **_kw: model)
    monkeypatch.setattr(
        runtime,
        "materialize_extras",
        lambda *_a, **_kw: SimpleNamespace(codes={}),
    )
    monkeypatch.setattr(runtime, "materialize_meta_adapters", lambda *_a, **_kw: 0)

    install_reached = False

    def fake_install(*_args, **_kwargs):
        nonlocal install_reached
        install_reached = True
        return object()

    monkeypatch.setattr(runtime, "install_streaming", fake_install)

    with pytest.raises(RuntimeError, match=r"lora_A"):
        runtime.build_streamed_model(
            model_id="fake/model",
            shard_dir="unused",
            index=object(),
            lora_config=object(),
            device="cpu",
            dtype="float32",
        )
    assert install_reached is False
