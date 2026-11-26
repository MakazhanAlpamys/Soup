"""The cached decoder forward: equal to the plain forward, and reused only when valid."""

import copy

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
)
from soup_cli.utils.frozen_prefix_forward import (
    FrozenPrefixRunner,
    frozen_prefix_session,
    install_frozen_prefix_cache,
)

COMMON = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    attention_dropout=0.0,
    sliding_window=None,
)


def tiny_decoder(**overrides):
    torch.manual_seed(42)
    return MistralModel(MistralConfig(**{**COMMON, **overrides})).eval().requires_grad_(False)


def make_runner(decoder, cache_dir, cutoff=2):
    configuration = fingerprint_prefix_configuration(decoder, cutoff)

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny-mistral",
            frozen_prefix_fingerprint="runner-computed",
            config_fingerprint=configuration,
            cutoff=cutoff,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    return FrozenPrefixRunner(
        decoder=decoder,
        cache=FrozenPrefixCache(cache_dir),
        cutoff=cutoff,
        metadata_factory=metadata_factory,
    )


def batch(length=12, offset=1):
    inputs = torch.tensor([list(range(offset, offset + length))], dtype=torch.long)
    return inputs, torch.ones_like(inputs), torch.arange(length).unsqueeze(0)


def test_cached_forward_equals_the_plain_forward_on_a_miss_and_on_a_hit(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    with torch.no_grad():
        reference = decoder(
            input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False
        ).last_hidden_state
        cold = runner.forward(
            input_ids=inputs, attention_mask=mask, position_ids=positions
        ).last_hidden_state
        warm = runner.forward(
            input_ids=inputs, attention_mask=mask, position_ids=positions
        ).last_hidden_state

    assert (runner.misses, runner.hits) == (1, 1)
    torch.testing.assert_close(cold, reference, rtol=0, atol=0)
    torch.testing.assert_close(warm, reference, rtol=0, atol=0)


def test_a_hit_does_not_run_the_frozen_layers(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()
    with torch.no_grad():
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)

    calls = []
    handles = [
        layer.register_forward_pre_hook(lambda module, args: calls.append(module))
        for layer in decoder.layers[:2]
    ]
    try:
        with torch.no_grad():
            runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)
    finally:
        for handle in handles:
            handle.remove()

    assert runner.hits == 1
    assert calls == []


def test_missing_position_ids_equal_the_default_positions(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    with torch.no_grad():
        implicit = runner.forward(input_ids=inputs, attention_mask=mask).last_hidden_state
        explicit = runner.forward(
            input_ids=inputs, attention_mask=mask, position_ids=positions
        ).last_hidden_state

    assert (runner.misses, runner.hits) == (1, 1), "the implicit positions are the default ones"
    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


def test_changed_inputs_are_a_miss_not_a_stale_hit(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    with torch.no_grad():
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)
        other_ids = batch(offset=40)[0]
        other = runner.forward(
            input_ids=other_ids, attention_mask=mask, position_ids=positions
        ).last_hidden_state
        masked = mask.clone()
        masked[0, -3:] = 0
        runner.forward(input_ids=inputs, attention_mask=masked, position_ids=positions)
        shifted = runner.forward(
            input_ids=inputs, attention_mask=mask, position_ids=positions + 5
        ).last_hidden_state
        reference_other = decoder(
            input_ids=other_ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).last_hidden_state
        reference_shifted = decoder(
            input_ids=inputs, attention_mask=mask, position_ids=positions + 5, use_cache=False
        ).last_hidden_state

    assert (runner.misses, runner.hits) == (4, 0)
    torch.testing.assert_close(other, reference_other, rtol=0, atol=0)
    torch.testing.assert_close(shifted, reference_shifted, rtol=0, atol=0)


def test_changed_rope_theta_misses_a_shared_persistent_cache(tmp_path):
    first = tiny_decoder(rope_theta=10000.0)
    second = tiny_decoder(rope_theta=1000.0)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    # Some versions persist inv_freq, so loading may copy the other RoPE state.
    second.rotary_emb = type(second.rotary_emb)(second.config)
    second.eval().requires_grad_(False)

    assert fingerprint_frozen_prefix(first, 2) != fingerprint_frozen_prefix(second, 2)

    inputs, mask, positions = batch(length=24)
    shared = tmp_path / "shared"

    def run(decoder):
        runner = make_runner(decoder, shared)
        with torch.no_grad():
            reference = decoder(
                input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False
            ).last_hidden_state
            cold = runner.forward(
                input_ids=inputs, attention_mask=mask, position_ids=positions
            ).last_hidden_state
            warm = runner.forward(
                input_ids=inputs, attention_mask=mask, position_ids=positions
            ).last_hidden_state
        assert (runner.misses, runner.hits) == (1, 1)
        torch.testing.assert_close(cold, reference, rtol=0, atol=0)
        torch.testing.assert_close(warm, reference, rtol=0, atol=0)
        return reference

    old_output = run(first)
    new_output = run(second)

    assert not torch.equal(old_output, new_output), "the negative control must differ"
    assert len(list(shared.glob("*.safetensors"))) == 2


def test_changing_a_frozen_weight_mid_run_is_refused(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    with torch.no_grad():
        next(decoder.layers[0].parameters()).add_(0.01)

        with pytest.raises(RuntimeError, match="frozen-prefix weights changed during training"):
            runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)


def test_half_precision_hit_equals_the_plain_forward(tmp_path):
    decoder = tiny_decoder().to(torch.bfloat16)
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()
    seen = []
    handle = decoder.layers[2].register_forward_pre_hook(
        lambda module, args: seen.append(args[0].dtype)
    )
    try:
        with torch.no_grad():
            reference = decoder(
                input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False
            ).last_hidden_state
            seen.clear()
            runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)
            warm = runner.forward(
                input_ids=inputs, attention_mask=mask, position_ids=positions
            ).last_hidden_state
    finally:
        handle.remove()

    assert (runner.misses, runner.hits) == (1, 1)
    assert seen == [torch.bfloat16, torch.bfloat16]
    torch.testing.assert_close(warm, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"use_cache": True}, "does not support KV caching"),
        ({"output_router_logits": True}, "output_router_logits=True"),
        ({"output_attentions": True}, "output_attentions=True"),
        ({"output_hidden_states": True}, "output_hidden_states=True"),
        ({"surprise": 1}, "unsupported forward arguments"),
    ],
)
def test_unsupported_forward_arguments_are_refused(tmp_path, kwargs, message):
    runner = make_runner(tiny_decoder(), tmp_path / "cache")
    inputs, mask, positions = batch()

    with pytest.raises(ValueError, match=message):
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions, **kwargs)


def test_embeddings_input_is_refused(tmp_path):
    runner = make_runner(tiny_decoder(), tmp_path / "cache")

    with pytest.raises(ValueError, match="requires input_ids"):
        runner.forward(inputs_embeds=torch.zeros(1, 4, 64))


def test_runner_refuses_a_trainable_prefix(tmp_path):
    decoder = tiny_decoder()
    decoder.layers[0].requires_grad_(True)

    with pytest.raises(ValueError, match="requires a fully frozen prefix"):
        make_runner(decoder, tmp_path / "cache")


def test_runner_refuses_an_active_dropout_in_the_prefix(tmp_path):
    decoder = tiny_decoder(attention_dropout=0.1).train()

    with pytest.raises(ValueError, match="deterministic frozen prefix"):
        make_runner(decoder, tmp_path / "cache")


@pytest.mark.parametrize("cutoff", [0, 4, 9])
def test_runner_refuses_an_empty_prefix_or_upper_stack(tmp_path, cutoff):
    decoder = tiny_decoder()

    with pytest.raises(ValueError, match="Invalid frozen-prefix cutoff"):
        FrozenPrefixRunner(
            decoder=decoder,
            cache=FrozenPrefixCache(tmp_path / "cache"),
            cutoff=cutoff,
            metadata_factory=lambda **_: {},
        )


def test_install_replaces_the_decoder_forward_and_restores_it(tmp_path):
    model = MistralForCausalLM(MistralConfig(**COMMON)).eval().requires_grad_(False)
    decoder = model.model
    original = decoder.forward
    configuration = fingerprint_prefix_configuration(decoder, 2)

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny",
            frozen_prefix_fingerprint="runner-computed",
            config_fingerprint=configuration,
            cutoff=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    inputs, mask, _ = batch()
    with install_frozen_prefix_cache(
        model, FrozenPrefixCache(tmp_path / "cache"), 2, metadata_factory
    ) as runner:
        with torch.no_grad():
            model(input_ids=inputs, attention_mask=mask, use_cache=False)
            model(input_ids=inputs, attention_mask=mask, use_cache=False)

    assert (runner.misses, runner.hits) == (1, 1)
    assert decoder.forward == original


def test_install_refuses_a_trainable_embedding(tmp_path):
    model = MistralForCausalLM(MistralConfig(**COMMON)).eval().requires_grad_(False)
    model.model.embed_tokens.requires_grad_(True)

    with pytest.raises(ValueError, match="requires frozen input embeddings"):
        with install_frozen_prefix_cache(
            model, FrozenPrefixCache(tmp_path / "cache"), 2, lambda **_: {}
        ):
            pass


def test_install_refuses_an_unsupported_decoder(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
        )
    )

    with pytest.raises(ValueError, match="supports Mistral and OLMoE models only"):
        with install_frozen_prefix_cache(
            model, FrozenPrefixCache(tmp_path / "cache"), 2, lambda **_: {}
        ):
            pass


def test_session_needs_a_frozen_layer(tmp_path):
    model = MistralForCausalLM(MistralConfig(**COMMON))

    with pytest.raises(ValueError, match="needs at least one frozen decoder layer"):
        with frozen_prefix_session(
            model, cutoff=0, output_dir=tmp_path, model_revision="tiny"
        ):
            pass


def test_session_refuses_data_parallel(tmp_path):
    model = MistralForCausalLM(MistralConfig(**COMMON))

    with pytest.raises(RuntimeError, match="does not support torch.nn.DataParallel"):
        with frozen_prefix_session(
            model, cutoff=2, output_dir=tmp_path, model_revision="tiny", n_gpu=2
        ):
            pass


def test_session_writes_only_under_the_output_directory(tmp_path):
    model = MistralForCausalLM(MistralConfig(**COMMON)).eval().requires_grad_(False)
    inputs, mask, _ = batch()

    with frozen_prefix_session(
        model, cutoff=2, output_dir=tmp_path, model_revision="tiny"
    ) as runner:
        with torch.no_grad():
            model(input_ids=inputs, attention_mask=mask, use_cache=False)

    assert runner.misses == 1
    entries = list(tmp_path.rglob("*.safetensors"))
    assert len(entries) == 1
    assert entries[0].parent.parent == tmp_path
