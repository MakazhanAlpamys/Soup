"""Real tiny-model routing/gradients and registry lifetime, using a CPU stand-in."""

import copy

import pytest

from benchmarks.harness.streaming_attention_candidate import native_masks_external_unmasked


@pytest.fixture
def attention_api():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip(
        "transformers.integrations.sdpa_attention", reason="pinned research attention API required"
    )
    pytest.importorskip(
        "transformers.masking_utils", reason="pinned research mask API required"
    )
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not callable(getattr(ALL_MASK_ATTENTION_FUNCTIONS, "register", None)):
        pytest.skip("the research harness requires the pinned mask registry API")
    return torch, sdpa_attention_forward, ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS


@pytest.mark.parametrize("padded", [False, True])
def test_tiny_model_logits_loss_and_gradients_keep_native_arithmetic(attention_api, padded):
    torch, stand_in, _, _ = attention_api
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(792)
    config = LlamaConfig(
        vocab_size=47, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, attention_dropout=0.0, use_cache=False,
    )
    config._attn_implementation = "sdpa"
    native = LlamaForCausalLM(config).train()
    candidate = copy.deepcopy(native)
    candidate.config._attn_implementation = "flash_attention_2"
    ids = torch.arange(12).reshape(2, 6) + 3
    mask = torch.ones_like(ids)
    if padded:
        mask[1, :2] = 0
        ids[1, :2] = 0
    batch = {
        "input_ids": ids, "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
        "labels": ids.clone().masked_fill(mask == 0, -100),
    }
    with native_masks_external_unmasked(fixture_external=stand_in) as counts:
        a, b = native(**batch), candidate(**batch)
        assert torch.equal(a.logits, b.logits)
        assert torch.equal(a.loss, b.loss)
        a.loss.backward()
        b.loss.backward()
        assert all(
            torch.equal(x.grad, y.grad)
            for x, y in zip(native.parameters(), candidate.parameters(), strict=True)
        )
        assert counts["fixture_external"]
        assert counts["native_explicit_mask_calls"] == (2 if padded else 0)
        assert counts["external_unmasked_calls"] == (0 if padded else 2)


def test_real_external_route_rejects_cpu_instead_of_using_a_stand_in(attention_api):
    torch, _, attention, masks = attention_api
    before = attention["flash_attention_2"], masks["flash_attention_2"]
    q = torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="restricted to CUDA BF16"):
        with native_masks_external_unmasked():
            attention["flash_attention_2"](None, q, q, q, None)
    assert (attention["flash_attention_2"], masks["flash_attention_2"]) == before


def test_explicit_masks_require_the_native_prepared_rank(attention_api):
    torch, stand_in, attention, masks = attention_api
    before = attention["flash_attention_2"], masks["flash_attention_2"]
    q = torch.zeros((1, 1, 4, 8))
    with pytest.raises(ValueError, match="prepared 4D mask"):
        with native_masks_external_unmasked(fixture_external=stand_in):
            attention["flash_attention_2"](None, q, q, q, torch.ones(1, 4))
    assert (attention["flash_attention_2"], masks["flash_attention_2"]) == before


def test_nested_installation_cannot_replace_the_outer_registry(attention_api):
    _, stand_in, attention, masks = attention_api
    before = attention["flash_attention_2"], masks["flash_attention_2"]
    with native_masks_external_unmasked(fixture_external=stand_in):
        installed = attention["flash_attention_2"], masks["flash_attention_2"]
        with pytest.raises(ValueError, match="already installed"):
            with native_masks_external_unmasked(fixture_external=stand_in):
                pytest.fail("nested installation accepted")
        assert (attention["flash_attention_2"], masks["flash_attention_2"]) == installed
    assert (attention["flash_attention_2"], masks["flash_attention_2"]) == before
