"""What can make a cached prefix activation stale or unwritable, and the guards for it."""

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_prefix_configuration,
)
from soup_cli.utils.frozen_prefix_forward import (
    FrozenPrefixRunner,
    estimate_cache_bytes,
    execution_signature,
    frozen_prefix_session,
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


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch):
    """The disk guard reads the real volume; most of these tests are about something else."""
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "soup_cli.utils.frozen_prefix_cache.shutil.disk_usage",
        lambda path: usage(10**13, 0, 10**13),
    )


def tiny_decoder(seed=42):
    torch.manual_seed(seed)
    return MistralModel(MistralConfig(**COMMON)).eval().requires_grad_(False)


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


def batch(length=12):
    inputs = torch.tensor([list(range(1, 1 + length))], dtype=torch.long)
    return inputs, torch.ones_like(inputs), torch.arange(length).unsqueeze(0)


class Console:
    def __init__(self):
        self.lines = []

    def print(self, text, **kwargs):
        self.lines.append(str(text))


# --- hooks (NEFTune and anything like it) ---


def test_runner_refuses_a_hook_on_the_embeddings(tmp_path):
    decoder = tiny_decoder()
    decoder.embed_tokens.register_forward_hook(lambda module, args, output: output)

    with pytest.raises(ValueError, match="forward hook on the input embeddings") as caught:
        make_runner(decoder, tmp_path / "cache")

    assert "found on embed_tokens" in str(caught.value)


@pytest.mark.parametrize("kind", ["forward", "pre"])
def test_runner_refuses_a_hook_on_a_frozen_layer_or_inside_it(tmp_path, kind):
    decoder = tiny_decoder()
    target = decoder.layers[1].mlp
    if kind == "forward":
        target.register_forward_hook(lambda module, args, output: output)
    else:
        target.register_forward_pre_hook(lambda module, args: None)

    with pytest.raises(ValueError, match="found on layers.1.mlp"):
        make_runner(decoder, tmp_path / "cache")


def test_runner_allows_a_hook_on_an_upper_layer(tmp_path):
    decoder = tiny_decoder()
    decoder.layers[3].register_forward_hook(lambda module, args, output: output)

    make_runner(decoder, tmp_path / "cache")


def test_a_hook_installed_after_the_runner_is_refused_before_the_cache_is_used(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    def add_noise(module, args, output):  # what NEFTune does to the embeddings
        return output + 0.1

    decoder.embed_tokens.register_forward_hook(add_noise)

    with torch.no_grad(), pytest.raises(ValueError, match="NEFTune installs one"):
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)

    assert (runner.misses, runner.hits) == (0, 0)
    assert list((tmp_path / "cache").glob("*.safetensors")) == []


# --- the key ---


def test_the_runner_fingerprint_separates_decoders_of_one_configuration(tmp_path):
    """Removing the line that sets the weights fingerprint must fail this test."""
    first = tiny_decoder(seed=1)
    second = tiny_decoder(seed=2)
    inputs, mask, positions = batch()
    shared = tmp_path / "shared"

    with torch.no_grad():
        first_runner = make_runner(first, shared)
        first_runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)
        second_runner = make_runner(second, shared)
        served = second_runner.forward(
            input_ids=inputs, attention_mask=mask, position_ids=positions
        ).last_hidden_state
        reference = second(
            input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False
        ).last_hidden_state

    assert (second_runner.misses, second_runner.hits) == (1, 0)
    torch.testing.assert_close(served, reference, rtol=0, atol=0)
    assert len(list(shared.glob("*.safetensors"))) == 2


def test_execution_signature_changes_with_each_input_to_the_numbers():
    config = MistralConfig(**COMMON)
    base = torch.zeros(1, 2, 4, dtype=torch.float32)
    original = execution_signature(base, config)

    assert execution_signature(base.to(torch.bfloat16), config) != original
    assert execution_signature(base.to("meta"), config) != original
    other = MistralConfig(**COMMON)
    other._attn_implementation = "eager" if config._attn_implementation != "eager" else "sdpa"
    assert execution_signature(base, other) != original
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert execution_signature(base, config) != original
    assert execution_signature(base, config) == original


def test_autocast_is_a_miss_not_a_hit_on_the_plain_entry(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    with torch.no_grad():
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)

    assert (runner.misses, runner.hits) == (2, 0)


def test_a_changed_attention_implementation_is_a_miss(tmp_path):
    decoder = tiny_decoder()
    runner = make_runner(decoder, tmp_path / "cache")
    inputs, mask, positions = batch()

    with torch.no_grad():
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)
        decoder.config._attn_implementation = "eager"
        runner.forward(input_ids=inputs, attention_mask=mask, position_ids=positions)

    assert (runner.misses, runner.hits) == (2, 0)


# --- disk: announcement and the stop-writing guard through a session ---


def test_estimate_is_rows_times_tokens_times_hidden_times_bytes():
    assert estimate_cache_bytes(1000, 2048, 4096, 2) == 1000 * 2048 * 4096 * 2


def test_session_announces_directory_size_and_free_space(tmp_path):
    model = MistralForCausalLM(MistralConfig(**COMMON)).eval().requires_grad_(False)
    console = Console()

    with frozen_prefix_session(
        model,
        cutoff=2,
        output_dir=tmp_path,
        model_revision="tiny",
        rows=100,
        max_length=64,
        console=console,
    ):
        pass

    text = " ".join(console.lines)
    assert "Frozen-prefix cache directory:" in text
    assert "_frozen_prefix_cache" in text
    assert f"{100 * 64 * 64 * 4 / 1e9:.1f} GB for 100 rows of 64 tokens" in text
    assert "free on that volume" in text and "The folder is kept after the run" in text


def test_session_warns_when_the_estimate_exceeds_the_room(tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "soup_cli.utils.frozen_prefix_cache.shutil.disk_usage",
        lambda path: usage(100 * 1024**3, 97 * 1024**3, 3 * 1024**3),
    )
    model = MistralForCausalLM(MistralConfig(**COMMON)).eval().requires_grad_(False)
    console = Console()

    with frozen_prefix_session(
        model,
        cutoff=2,
        output_dir=tmp_path,
        model_revision="tiny",
        rows=10**6,
        max_length=2048,
        console=console,
    ):
        pass

    assert any("will stop writing before the first epoch ends" in line for line in console.lines)


def test_the_notice_reaches_the_console_when_writing_stops(tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "soup_cli.utils.frozen_prefix_cache.shutil.disk_usage",
        lambda path: usage(100 * 1024**3, 99 * 1024**3, 1 * 1024**3),
    )
    model = MistralForCausalLM(MistralConfig(**COMMON)).eval().requires_grad_(False)
    console = Console()
    inputs, mask, _ = batch()

    with frozen_prefix_session(
        model, cutoff=2, output_dir=tmp_path, model_revision="tiny", console=console
    ) as runner:
        with torch.no_grad():
            first = model(input_ids=inputs, attention_mask=mask, use_cache=False)
            second = model(input_ids=inputs, attention_mask=mask, use_cache=False)

    assert (runner.misses, runner.hits) == (2, 0), "no entry was written, so nothing can hit"
    assert sum("stopped writing" in line for line in console.lines) == 1
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
