"""Default PEFT saves and real resume paths for streamed boundary adapters.

#1048 item 1, following the build-time guard contributed in #2082 by @ybai08.
The base remains meta; exporting its frozen bytes must never read the shared slot.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from tests.test_issue1048_lora_head_save_resume import (
    _build_streamed,
    _perturb_lora,
    _requires_train_extra,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _adapters(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def _assert_adapters(model: Any, expected: dict[str, Any]) -> None:
    import torch

    actual = _adapters(model)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_streamed_embedding_starts_as_the_unchanged_base(tmp_path: Path) -> None:
    _requires_train_extra()
    import torch
    from transformers import LlamaForCausalLM

    model, runtime = _build_streamed(
        tmp_path, tie=False, target_modules=["q_proj", "v_proj", "embed_tokens"], seed=3
    )
    try:
        parameters = _adapters(model)
        embedding_a = next(
            value for name, value in parameters.items() if "lora_embedding_A" in name
        )
        embedding_b = next(
            value for name, value in parameters.items() if "lora_embedding_B" in name
        )
        assert torch.count_nonzero(embedding_a) == 0
        assert embedding_b.std() > 0.5  # PEFT's normal B init, rather than linear's Kaiming A.
        resident = LlamaForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.float32).eval()
        model.eval()
        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            assert torch.equal(
                model(input_ids=inputs, use_cache=False).logits,
                resident(input_ids=inputs, use_cache=False).logits,
            )
    finally:
        runtime.close()


def test_reference_adapter_copy_includes_embedding_suffix_parameters(tmp_path: Path) -> None:
    _requires_train_extra()
    import copy

    import torch

    from soup_cli.utils.layer_stream_runtime import materialize_meta_adapter_copy

    model, runtime = _build_streamed(
        tmp_path, tie=False, target_modules=["q_proj", "v_proj", "lm_head", "embed_tokens"], seed=3
    )
    try:
        _perturb_lora(model, seed=1234)
        original = _adapters(model)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        cloned = copy.deepcopy(model)
        cloned.add_adapter("ref", copy.deepcopy(cloned.peft_config["default"]))
        assert set(model.peft_config) == {"default"}
        assert materialize_meta_adapter_copy(cloned) == 12
        parameters = dict(cloned.named_parameters())
        for name, value in original.items():
            ref_name = name.replace(".default.", ".ref.")
            if name.endswith(".default"):
                ref_name = name[:-len("default")] + "ref"
            assert not parameters[ref_name].is_meta
            assert not parameters[ref_name].requires_grad
            assert torch.equal(parameters[ref_name], value), ref_name
        assert {
            name for name, value in cloned.named_parameters() if value.requires_grad
        } == trainable
        _assert_adapters(model, original)
    finally:
        runtime.close()


def test_adapter_injection_exception_restores_the_streaming_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _requires_train_extra()
    import copy

    from soup_cli.utils.layer_stream_runtime import _attach_streamed_boundary_adapter_bridge

    model, runtime = _build_streamed(
        tmp_path, tie=False, target_modules=["q_proj", "lm_head", "embed_tokens"], seed=3
    )
    before = (model.get_input_embeddings(), model.get_output_embeddings())
    try:
        def fail_after_replacing_input(self: Any, *args: Any, **kwargs: Any) -> None:
            input_module = self.get_input_embeddings()
            assert input_module is before[0].inner
            self.set_input_embeddings(copy.deepcopy(input_module))
            raise RuntimeError("injected adapter setup failure")

        monkeypatch.setattr(type(model), "add_adapter", fail_after_replacing_input)
        _attach_streamed_boundary_adapter_bridge(model)
        with pytest.raises(RuntimeError, match="injected adapter setup failure"):
            model.add_adapter("ref", model.peft_config["default"])
        assert model.get_input_embeddings() is before[0]
        assert model.get_output_embeddings() is before[1]
        assert before[0].pool is runtime.large_pool
        assert before[1].pool is runtime.large_pool
        assert before[0].weight.is_meta
        assert before[1].weight.is_meta
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "targets",
    [
        ["q_proj", "v_proj", "lm_head"],
        ["q_proj", "v_proj", "embed_tokens"],
        ".*(q_proj|v_proj|lm_head)",
        ".*(q_proj|v_proj|embed_tokens)",
        ["q_proj", "v_proj", "model.embed_tokens"],
        ["q_proj", "v_proj", "embed_tokens", "lm_head"],
        ["q_proj", "v_proj"],
    ],
)
@pytest.mark.parametrize("save_embeddings", ["default", True, False])
def test_public_save_load_and_strict_round_trip(
    tmp_path: Path, targets: list[str] | str, save_embeddings: bool | str
) -> None:
    _requires_train_extra()
    import torch
    from safetensors.torch import load_file

    model, runtime = _build_streamed(
        tmp_path / "original", tie=False, target_modules=targets, seed=3
    )
    try:
        _perturb_lora(model, seed=1234)
        expected = _adapters(model)
        target_text = str(targets)
        expected_count = 8 + 2 * ("lm_head" in target_text) + 2 * ("embed_tokens" in target_text)
        assert len(expected) == expected_count
        model.eval()
        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            expected_logits = model(input_ids=inputs, use_cache=False).logits.clone()
        state = model.state_dict()
        assert not any(".inner." in name for name in state)
        # The head is what normally occupies the slot at a save after forward.
        runtime.large_pool.load_async("lm_head.weight", runtime.source)
        before = runtime.large_pool.buffer.clone()
        checkpoint = tmp_path / "adapter"
        options = {} if save_embeddings == "default" else {"save_embedding_layers": save_embeddings}
        model.save_pretrained(checkpoint, **options)
        artifact = load_file(str(checkpoint / "adapter_model.safetensors"))
        assert len([name for name in artifact if "lora_" in name]) == len(expected)
        if save_embeddings is True:
            # PEFT includes an adapted boundary's base, and may omit the
            # opposite unadapted boundary. Preserve that selection policy.
            if "embed_tokens" in target_text:
                assert any("embed_tokens" in name and "lora_" not in name for name in artifact)
            if "lm_head" in target_text:
                assert any("lm_head" in name and "lora_" not in name for name in artifact)
        elif save_embeddings is False:
            assert all("lora_" in name for name in artifact)
        elif "lm_head" in target_text:
            assert "base_model.model.lm_head.base_layer.weight" in artifact
        elif "embed_tokens" in targets:
            assert "base_model.model.model.embed_tokens.base_layer.weight" in artifact
        assert torch.equal(runtime.large_pool.buffer, before)
        assert model.get_input_embeddings().weight.is_meta
        assert model.get_output_embeddings().weight.is_meta
    finally:
        runtime.close()

    fresh, fresh_runtime = _build_streamed(
        tmp_path / "fresh", tie=False, target_modules=targets, seed=99
    )
    try:
        _perturb_lora(fresh, seed=9876)
        fresh.load_adapter(checkpoint, adapter_name="default", is_trainable=True)
        _assert_adapters(fresh, expected)
        for assign in (False, True):
            _perturb_lora(fresh, seed=9876)
            with warnings.catch_warnings(record=True) as emitted:
                result = fresh.load_state_dict(state, strict=True, assign=assign)
            assert not any("meta parameter" in str(item.message) for item in emitted)
            assert result.missing_keys == []
            assert result.unexpected_keys == []
            _assert_adapters(fresh, expected)
            assert fresh.get_input_embeddings().weight.is_meta
            assert fresh.get_output_embeddings().weight.is_meta
            fresh.eval()
            with torch.no_grad():
                actual_logits = fresh(input_ids=inputs, use_cache=False).logits
            assert torch.equal(actual_logits, expected_logits)
    finally:
        fresh_runtime.close()


@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_frozen_boundary_export_is_independent_and_conflicting_base_is_refused(
    tmp_path: Path, tier: str
) -> None:
    _requires_train_extra()
    import torch
    from safetensors.torch import load_file

    model, runtime = _build_streamed(
        tmp_path,
        tie=False,
        target_modules=["q_proj", "lm_head", "embed_tokens"],
        seed=3,
        tier=tier,
    )
    try:
        original = load_file(str(tmp_path / "model" / "model.safetensors"))
        runtime.large_pool.load_async("lm_head.weight", runtime.source)
        state = model.state_dict()
        boundaries = {
            "base_model.model.model.embed_tokens.base_layer.weight": "model.embed_tokens.weight",
            "base_model.model.lm_head.base_layer.weight": "lm_head.weight",
        }
        for name, canonical in boundaries.items():
            assert state[name].device.type == "cpu"
            assert torch.equal(state[name], original[canonical]), name
            assert state[name].data_ptr() != runtime.large_pool.buffer.data_ptr()
        runtime.large_pool.buffer.fill_(123)
        for name, canonical in boundaries.items():
            assert torch.equal(state[name], original[canonical]), name
        # Export is not a view into the resident RAM store either.
        for name, canonical in boundaries.items():
            source_value = runtime.source.get(
                runtime.large_pool.source_indices[canonical], canonical
            )
            assert state[name].data_ptr() != source_value.data_ptr()
        corrupted = dict(state)
        name = "base_model.model.lm_head.base_layer.weight"
        corrupted[name] = state[name] + 1
        with pytest.raises(ValueError, match="frozen.*lm_head|lm_head.*frozen"):
            model.load_state_dict(corrupted, strict=True)
    finally:
        runtime.close()


@pytest.mark.parametrize("target", ["lm_head", "model.embed_tokens"])
def test_modules_to_save_is_refused_before_peft_or_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    _requires_train_extra()
    import peft

    from soup_cli.utils import layer_stream_runtime
    from soup_cli.utils.layer_shard import shard_checkpoint
    from tests.test_issue1048_lora_head_save_resume import _lora, _tiny_llama_dir

    weights = _tiny_llama_dir(tmp_path, tie=False)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    lora = _lora(["q_proj"])
    lora.modules_to_save = [target]

    def must_not_run(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unsupported trainable boundary reached PEFT or buffer allocation")

    monkeypatch.setattr(peft, "get_peft_model", must_not_run)
    monkeypatch.setattr(layer_stream_runtime, "install_streaming", must_not_run)
    with pytest.raises(ValueError, match="modules_to_save.*frozen streamed boundary"):
        layer_stream_runtime.build_streamed_model(
            model_id=weights,
            shard_dir=shards,
            index=index,
            lora_config=lora,
            device="cpu",
            dtype="float32",
            pin=False,
        )


@pytest.mark.parametrize("target", ["lm_head", "embed_tokens"])
@pytest.mark.parametrize("task", ["sft", "dpo"])
def test_trainer_checkpoint_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, task: str
) -> None:
    _requires_train_extra()
    import torch
    from transformers import TrainerCallback

    from tests.test_layer_stream_boundary_lora import _setup

    targets = ["q_proj", "v_proj", target]
    wrapper = _setup(
        tmp_path / "run",
        monkeypatch,
        tie=False,
        target_modules=targets,
        task=task,
        device=None,
        save_steps=1,
    )
    wrapper.trainer.args.max_steps = 1
    try:
        wrapper.train()
        trained = _adapters(wrapper.model)
        assert len(trained) == (20 if task == "dpo" else 10)
    finally:
        wrapper._close_stream_runtime()
    checkpoint = tmp_path / "run" / "out" / "checkpoint-1"
    fresh = _setup(
        tmp_path / "fresh",
        monkeypatch,
        tie=False,
        target_modules=targets,
        task=task,
        device=None,
        save_steps=1,
    )
    try:
        _perturb_lora(fresh.model, seed=9876)
        before = {name: value.clone() for name, value in trained.items()}

        class AssertRestoredBeforeTheNextStep(TrainerCallback):
            checked = False

            def on_train_begin(
                self,
                args: Any,
                state: Any,
                control: Any,
                model: Any = None,
                optimizer: Any = None,
                **kwargs: Any,
            ) -> None:
                _assert_adapters(model, trained)
                assert state.global_step == 1
                assert optimizer.state
                assert all(float(item["step"]) == 1 for item in optimizer.state.values())
                self.checked = True

        restored = AssertRestoredBeforeTheNextStep()
        fresh.trainer.add_callback(restored)
        fresh.trainer.args.max_steps = 2
        fresh.train(resume_from_checkpoint=str(checkpoint))
        assert restored.checked
        assert fresh.trainer.state.global_step == 2
        after = _adapters(fresh.model)
        assert any(not torch.equal(after[name], value) for name, value in before.items())
        assert (tmp_path / "fresh" / "out" / "checkpoint-2" / "adapter_model.safetensors").is_file()
    finally:
        fresh._close_stream_runtime()
