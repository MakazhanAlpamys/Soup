"""Ambiguous boundary checkpoint keys must never silently pick one spelling.

This pins #1048 item 3's load-side collision guard only. These tests make no
claim about serialization or save/resume support; item 1 landed in #2159.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_issue1048_lora_head_save_resume import (
    _build_streamed,
    _perturb_lora,
    _requires_train_extra,
)


def _boundary_prefix(target: str) -> str:
    module = "model.embed_tokens" if target == "embed_tokens" else "lm_head"
    return f"base_model.model.{module}."


@pytest.mark.parametrize("target", ["lm_head", "embed_tokens"])
@pytest.mark.parametrize("tensor", ["base", "adapter_a", "adapter_b"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("redirected_first", [False, True])
def test_both_checkpoint_spellings_are_refused_before_boundary_load(
    tmp_path: Path, target: str, tensor: str, strict: bool, redirected_first: bool,
) -> None:
    _requires_train_extra()
    import torch

    model, runtime = _build_streamed(
        tmp_path, tie=False, target_modules=["q_proj", target], seed=3,
    )
    try:
        _perturb_lora(model, seed=42)
        state = model.state_dict()
        suffixes = {
            "base": "base_layer.weight",
            "adapter_a": "lora_A.default.weight" if target == "lm_head"
            else "lora_embedding_A.default",
            "adapter_b": "lora_B.default.weight" if target == "lm_head"
            else "lora_embedding_B.default",
        }
        prefix = _boundary_prefix(target)
        canonical = prefix + suffixes[tensor]
        redirected = prefix + "inner." + suffixes[tensor]
        assert canonical in state
        assert redirected not in state
        # Equal values are ambiguous too: the checkpoint format must have one
        # spelling, regardless of insertion order or strict=False.
        duplicate = state[canonical].clone()
        if redirected_first:
            state = {redirected: duplicate, **state}
        else:
            state[redirected] = duplicate
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and not parameter.is_meta
        }
        with pytest.raises(ValueError, match="checkpoint contains both") as error:
            model.load_state_dict(state, strict=strict)
        assert repr(canonical) in str(error.value)
        assert repr(redirected) in str(error.value)
        assert "re-save the adapter" in str(error.value)
        current = dict(model.named_parameters())
        assert before
        assert all(torch.equal(current[name], value) for name, value in before.items())
    finally:
        runtime.close()


@pytest.mark.parametrize("target", ["lm_head", "embed_tokens"])
@pytest.mark.parametrize("spelling", ["canonical", "redirected"])
@pytest.mark.parametrize("strict", [False, True])
def test_one_checkpoint_spelling_restores_every_adapter_tensor(
    tmp_path: Path, target: str, spelling: str, strict: bool,
) -> None:
    _requires_train_extra()
    import torch

    model, runtime = _build_streamed(
        tmp_path, tie=False, target_modules=["q_proj", target], seed=3,
    )
    try:
        _perturb_lora(model, seed=42)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        saved = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if "lora_" in name
        }
        if spelling == "redirected":
            prefix = _boundary_prefix(target)
            state = {
                prefix + "inner." + name[len(prefix):] if name.startswith(prefix) else name: value
                for name, value in state.items()
            }
        _perturb_lora(model, seed=99)
        assert any(
            not torch.equal(parameter, saved[name])
            for name, parameter in model.named_parameters() if name in saved
        )
        result = model.load_state_dict(state, strict=strict)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        loaded = dict(model.named_parameters())
        assert saved
        assert all(torch.equal(loaded[name], value) for name, value in saved.items())
    finally:
        runtime.close()
