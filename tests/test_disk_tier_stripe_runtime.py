"""R4 — a striped cache reads back the same bytes through every source and every path."""

import os
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.async_disk_source import AsyncDiskSource  # noqa: E402
from soup_cli.utils.layer_shard import layer_paths, shard_checkpoint  # noqa: E402
from soup_cli.utils.layer_stream_runtime import DiskSource, RamSource  # noqa: E402
from soup_cli.utils.stripe_roots import STRIPE_DIRS_ENV  # noqa: E402


def _tiny_llama(tmp_path, n_layers=3, tie=True):
    from transformers import AutoModelForCausalLM, LlamaConfig

    weights = tmp_path / "model"
    torch.manual_seed(7)
    # hidden 64: see tests/test_v07203.py::_tiny_stream — at 32 the bnb CPU NF4 path breaks.
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    weights.mkdir()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        # Untied (tie=False) keeps lm_head.weight so the sharder sees both large-layer
        # roles and actually splits them — the tied path pops it, as before.
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


@pytest.fixture
def caches(tmp_path):
    """The same checkpoint sharded twice: one root, and striped over a second folder."""
    weights = _tiny_llama(tmp_path)
    stripe = tmp_path / "stripe"
    stripe.mkdir()
    single_dir = str(tmp_path / "c1" / "tiny")
    striped_dir = str(tmp_path / "c2" / "tiny")
    single = shard_checkpoint(weights, single_dir, dtype="float32", arch="llama")
    striped = shard_checkpoint(
        weights, striped_dir, dtype="float32", arch="llama", stripe_roots=(str(stripe),)
    )
    assert striped.layer_roots == (0, 1, 0)
    return weights, (single_dir, single), (striped_dir, striped), str(stripe)


def _raw(tensor):
    return tensor.reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("kind", ["ram", "disk", "async"])
def test_every_tensor_matches_the_single_root_cache(caches, kind):
    _, (single_dir, single), (striped_dir, striped), _ = caches
    single_paths = layer_paths(single_dir, single)
    striped_paths = layer_paths(striped_dir, striped)
    spec = RamSource.layer_specs_from_paths(striped_paths)
    assert spec == RamSource.layer_specs_from_paths(single_paths)

    def build(shard_dir, paths, roots):
        if kind == "ram":
            return RamSource(shard_dir, len(paths), spec, pin=False, shard_paths=paths)
        if kind == "disk":
            return DiskSource(shard_dir, len(paths), spec, shard_paths=paths)
        return AsyncDiskSource(
            shard_dir,
            len(paths),
            spec,
            pin=False,
            shard_paths=paths,
            layer_roots=roots,
            read_ahead=3,
        )

    theirs = build(single_dir, single_paths, None)
    mine = build(striped_dir, striped_paths, striped.layer_roots)
    try:
        for idx in range(len(striped_paths)):
            for name in spec[idx]:
                assert torch.equal(_raw(mine.get(idx, name)), _raw(theirs.get(idx, name))), (
                    idx,
                    name,
                )
    finally:
        for source in (mine, theirs):
            close = getattr(source, "close", None)
            if close is not None:
                close()


def test_async_source_refuses_a_placement_of_the_wrong_length(caches):
    _, _, (striped_dir, striped), _ = caches
    paths = layer_paths(striped_dir, striped)
    spec = RamSource.layer_specs_from_paths(paths)
    with pytest.raises(ValueError, match="expected 3 layer roots"):
        AsyncDiskSource(striped_dir, 3, spec, pin=False, shard_paths=paths, layer_roots=[0, 1])


def _stream(weights, shard_dir, index, *, read_ahead, **extra):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    return build_streamed_model(
        model_id=weights,
        shard_dir=shard_dir,
        index=index,
        lora_config=lora,
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=3,
        tier="disk",
        quant="none",
        read_ahead=read_ahead,
        **extra,
    )


def _grads_after_one_step(model):
    gen = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
    ids = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(5))
    model.train()
    model(input_ids=ids, labels=ids).loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if "lora_" in n}


def test_a_striped_training_step_is_bit_identical_to_one_root(caches):
    weights, (single_dir, single), (striped_dir, striped), _ = caches
    one, runtime_one = _stream(weights, single_dir, single, read_ahead=2)
    two, runtime_two = _stream(weights, striped_dir, striped, read_ahead=3)
    try:
        want, got = _grads_after_one_step(one), _grads_after_one_step(two)
        assert want.keys() == got.keys() and want
        for name in want:
            assert torch.equal(got[name], want[name]), name
    finally:
        for runtime in (runtime_one, runtime_two):
            close = getattr(runtime, "close", None)
            if close is not None:
                close()


def test_a_missing_stripe_folder_is_refused_by_name(caches):
    """Review Focus 1: an unmounted drive must say which drive, not FileNotFoundError."""
    weights, _, (striped_dir, striped), stripe = caches
    shutil.rmtree(stripe)
    with pytest.raises(RuntimeError, match=STRIPE_DIRS_ENV) as caught:
        _stream(weights, striped_dir, striped, read_ahead=3)
    assert os.path.realpath(stripe) in str(caught.value) and "[1]" in str(caught.value)


def test_an_untied_striped_training_step_is_bit_identical_to_one_root(tmp_path):
    """R4 review finding: the ``caches`` fixture always ties embeddings, so ``large_keys``
    is always ``()`` in every other test in this file and the root-0 padding term in
    ``install_streaming`` (``source_roots = placement + (0,) * len(large_keys) ...``) has no
    witness under striping. An untied checkpoint gives the sharder both the
    ``model.embed_tokens.weight`` and ``lm_head.weight`` large-layer roles, so this is the
    first striped run where that padding actually matters."""
    weights = _tiny_llama(tmp_path, tie=False)
    stripe = tmp_path / "stripe"
    stripe.mkdir()
    single_dir = str(tmp_path / "c1" / "tiny")
    striped_dir = str(tmp_path / "c2" / "tiny")
    single = shard_checkpoint(weights, single_dir, dtype="float32", arch="llama")
    striped = shard_checkpoint(
        weights, striped_dir, dtype="float32", arch="llama", stripe_roots=(str(stripe),)
    )
    assert single.large_keys and striped.large_keys
    assert striped.layer_roots == (0, 1, 0)

    one, runtime_one = _stream(weights, single_dir, single, read_ahead=2)
    two, runtime_two = _stream(weights, striped_dir, striped, read_ahead=3)
    try:
        want, got = _grads_after_one_step(one), _grads_after_one_step(two)
        assert want.keys() == got.keys() and want
        for name in want:
            assert torch.equal(got[name], want[name]), name
    finally:
        for runtime in (runtime_one, runtime_two):
            close = getattr(runtime, "close", None)
            if close is not None:
                close()


def test_the_read_ahead_decision_reaches_the_source_builder(caches, monkeypatch):
    """Fix wave, finding A: the page-lock advice lives in `_build_source`, so the setup's
    decision has to travel build_streamed_model -> install_streaming -> _build_source."""
    import soup_cli.utils.layer_stream_runtime as rt
    from soup_cli.utils.stripe_roots import ReadAheadDecision

    weights, _, (striped_dir, striped), _ = caches
    seen = {}
    real = rt._build_source

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rt, "_build_source", spy)
    decision = ReadAheadDecision(depth=3, configured=2, n_roots=2)
    _model, runtime = _stream(
        weights, striped_dir, striped, read_ahead=3, read_ahead_decision=decision
    )
    try:
        assert seen["read_ahead_decision"] is decision
    finally:
        runtime.close()
