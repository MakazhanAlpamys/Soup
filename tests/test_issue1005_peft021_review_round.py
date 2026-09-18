"""#1005, review round — two things the first ten tests could not see.

The review of the canonical-names fix (PR #1010) found one surviving mutant and one
false invariant, both invisible from inside a suite that never changes device or
dtype and never streams a large layer:

* ``_apply`` — since the fix, ``named_children()`` no longer lists ``inner``, and
  torch's ``_apply`` (behind ``to()`` / ``cuda()`` / ``half()``, i.e. behind
  ``Trainer._move_model_to_device`` and ``accelerate.prepare_model``) recurses over
  ``children()``. The explicit ``self.inner._apply(...)`` in the wrapper is therefore
  the ONLY path to the adapters; deleting it left 624 CPU tests green while
  ``model.to(torch.float64)`` cast 0 of 8 LoRA parameters. The tests here cast (and,
  with a card, move) a streamed model and check every adapter followed.
* ``StreamedLargeLayer`` — an untied checkpoint streams ``embed_tokens`` and
  ``lm_head`` through it, and its inner module is an ``nn.Embedding`` / ``nn.Linear``
  that owns ``weight`` DIRECTLY and has no children. transformers'
  ``get_parameter_names`` (the weight-decay grouping) walks ``named_children()`` and
  reads each visited module's own ``_parameters``, so it never lists those two
  weights. The first test file asserted decay-name equality with a resident model
  and passed only because its fixture was TIED (no large layer existed). The real
  invariant is narrower and is pinned here: the difference is exactly the frozen
  meta large-layer weights, and the optimizer groups ``Trainer.create_optimizer``
  builds — which filter on ``requires_grad`` — equal the resident model's.

Every test holds on peft 0.20 AND 0.21; the resident LoRA model on the same
checkpoint is the reference.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _requires_train_extra():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")


# --------------------------------------------------------------------------
# fixtures (standalone, mirroring tests/test_issue1005_peft021_canonical_names.py,
# with the tie switch exposed)
# --------------------------------------------------------------------------


def _tiny_llama_dir(tmp_path, n_layers=2, tie=True):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _tiny_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed_cpu(tmp_path, tie):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(tmp_path, tie=tie)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=_tiny_lora(),
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=3,
    )
    return model, runtime, weights


def _build_plain_peft(weights_dir):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.float32)
    return get_peft_model(base, _tiny_lora())


def _pair(tmp_path, tie):
    _requires_train_extra()
    streamed, runtime, weights = _build_streamed_cpu(tmp_path, tie=tie)
    plain = _build_plain_peft(weights)
    try:
        yield streamed, plain
    finally:
        runtime.close()


@pytest.fixture
def tied_pair(tmp_path):
    yield from _pair(tmp_path, tie=True)


@pytest.fixture
def untied_pair(tmp_path):
    yield from _pair(tmp_path, tie=False)


def _lora(model):
    return {name: param for name, param in model.named_parameters() if "lora_" in name}


def _meta_names(model):
    return {name for name, param in model.named_parameters() if param.is_meta}


_LARGE_PATHS = ("base_model.model.model.embed_tokens", "base_model.model.lm_head")


def _large_wrappers(model):
    """The two large-layer wrappers of an untied streamed model, found structurally
    (``get_submodule``), because ``modules()`` enumerates their inner module instead."""
    return {path: model.get_submodule(path) for path in _LARGE_PATHS}


# --------------------------------------------------------------------------
# F2 — `_apply` must still reach the adapters
# --------------------------------------------------------------------------
class TestApplyReachesTheAdapters:
    def test_to_dtype_casts_every_adapter_and_leaves_the_base_on_meta(self, tied_pair):
        """`model.to(dtype)` is `_apply`. With `inner` gone from `children()`, the
        wrapper's explicit recursion is the only way the cast reaches the LoRA
        matrices inside the decoder layers; the streamed base weights are meta
        placeholders that `_skip_meta` must leave alone."""
        import torch

        streamed, plain = tied_pair
        lora_before = _lora(streamed)
        assert len(lora_before) == len(_lora(plain)) > 0
        assert all(p.dtype == torch.float32 for p in lora_before.values())
        meta_before = _meta_names(streamed)
        assert any(".layers." in name for name in meta_before), "fixture: no streamed base"

        streamed.to(torch.float64)

        lora_after = _lora(streamed)
        not_cast = sorted(n for n, p in lora_after.items() if p.dtype != torch.float64)
        assert not_cast == [], f"adapters the cast did not reach: {not_cast[:3]}"
        assert set(lora_after) == set(lora_before)
        assert _meta_names(streamed) == meta_before, "a meta placeholder was materialised"
        left_behind = sorted(
            n for n, p in streamed.named_parameters() if not p.is_meta and p.dtype != torch.float64
        )
        assert left_behind == [], f"real parameters the cast skipped: {left_behind[:3]}"

    def test_to_dtype_reaches_a_tensor_the_inner_layer_owns_directly(self, tied_pair):
        """With `named_children()` fixed, torch's default `_apply` would already
        reach inner's CHILDREN through `children()`; the explicit
        `self.inner._apply(...)` is what reaches a tensor `inner` owns ITSELF.
        No supported decoder layer owns one today, and that is not a contract,
        so plant one and check the cast finds it."""
        import torch

        streamed, _ = tied_pair
        layers = streamed.get_submodule("base_model.model.model.layers")
        wrapper = layers[0]
        assert type(wrapper).__name__ == "StreamedDecoderLayer"
        wrapper.inner.register_buffer("probe_owned_directly", torch.zeros(3))
        nested = wrapper.inner.self_attn.q_proj
        nested.register_buffer("probe_owned_by_a_child", torch.zeros(3))
        streamed.to(torch.float64)
        assert nested.probe_owned_by_a_child.dtype == torch.float64, "children unreached"
        assert wrapper.inner.probe_owned_directly.dtype == torch.float64, (
            "a tensor owned by the inner layer itself did not follow .to()"
        )

    def test_named_buffers_match_the_resident_model_exactly(self, tied_pair):
        """`named_buffers()` derives from `named_modules()` too (torch's
        `_named_members`); pin it rather than leave the derivation implicit."""
        streamed, plain = tied_pair
        streamed_names = {name for name, _ in streamed.named_buffers()}
        plain_names = {name for name, _ in plain.named_buffers()}
        assert not [n for n in streamed_names if ".inner." in n]
        assert streamed_names == plain_names

    @pytest.mark.gpu
    def test_to_device_moves_every_adapter_and_leaves_the_base_on_meta(self, tied_pair):
        """The same path as `Trainer._move_model_to_device`: every adapter lands on
        the card, every streamed base weight stays a meta placeholder."""
        import torch

        streamed, _ = tied_pair
        meta_before = _meta_names(streamed)
        assert meta_before

        streamed.to(torch.device("cuda"))

        lora = _lora(streamed)
        assert lora
        stranded = sorted(n for n, p in lora.items() if p.device.type != "cuda")
        assert stranded == [], f"adapters left on the host: {stranded[:3]}"
        assert _meta_names(streamed) == meta_before


# --------------------------------------------------------------------------
# F1 — an untied checkpoint streams the large layers
# --------------------------------------------------------------------------
class TestUntiedCheckpointsStreamTheLargeLayers:
    def test_the_fixture_streams_both_large_layers(self, untied_pair):
        """Without this the class below tests nothing: `tie=False` must put
        `embed_tokens` AND `lm_head` behind a `StreamedLargeLayer`, enumerated
        (like the decoder wrapper) as its inner module at its own name."""
        import torch

        streamed, _ = untied_pair
        wrappers = _large_wrappers(streamed)
        assert all(type(w).__name__ == "StreamedLargeLayer" for w in wrappers.values())
        assert isinstance(wrappers[_LARGE_PATHS[0]].inner, torch.nn.Embedding)
        assert isinstance(wrappers[_LARGE_PATHS[1]].inner, torch.nn.Linear)
        enumerated = {id(m) for m in streamed.modules()}
        assert all(id(w) not in enumerated for w in wrappers.values())
        assert all(id(w.inner) in enumerated for w in wrappers.values())
        by_name = dict(streamed.named_modules())
        assert all(by_name[path] is wrappers[path].inner for path in _LARGE_PATHS)

    def test_names_and_keys_are_canonical_on_the_large_layers(self, untied_pair):
        """The #1005 fix mirrored onto `StreamedLargeLayer`: the streamed large
        weights are named as the resident model names them, the names are
        state_dict keys, and peft's adapter selection sees the same keys."""
        from peft import get_peft_model_state_dict

        streamed, plain = untied_pair
        streamed_names = {name for name, _ in streamed.named_parameters()}
        plain_names = {name for name, _ in plain.named_parameters()}
        assert not [n for n in streamed_names if ".inner." in n]
        assert streamed_names == plain_names
        assert {f"{path}.weight" for path in _LARGE_PATHS} <= streamed_names
        missing = sorted(streamed_names - set(streamed.state_dict().keys()))
        assert missing == [], f"parameter names absent from state_dict: {missing[:3]}"
        streamed_keys = set(get_peft_model_state_dict(streamed).keys())
        plain_keys = set(get_peft_model_state_dict(plain).keys())
        assert plain_keys and streamed_keys == plain_keys

    def test_decay_names_differ_only_by_the_frozen_meta_large_weights(self, untied_pair):
        """THE documented exception. `get_parameter_names` never lists the two
        streamed large weights (their wrapper yields no children and owns no
        parameters), so the decay-NAME set is not the resident model's — and that
        is safe only because both are frozen meta placeholders the optimizer never
        sees. Pin the exact difference so a third case cannot hide in it."""
        import torch
        from transformers.trainer_pt_utils import get_parameter_names

        streamed, plain = untied_pair
        # `get_parameter_names` lists `_parameters` KEYS, and `nn.Linear(bias=False)`
        # registers `bias` as None -- so the resident head contributes a `lm_head.bias`
        # name no parameter carries. Compare decay NAMES over real parameters only,
        # which is the join `create_optimizer` makes.
        streamed_decay = set(get_parameter_names(streamed, [torch.nn.LayerNorm])) & {
            name for name, _ in streamed.named_parameters()
        }
        plain_decay = set(get_parameter_names(plain, [torch.nn.LayerNorm])) & {
            name for name, _ in plain.named_parameters()
        }
        wrappers = _large_wrappers(streamed)
        owned_by_large_inners = {
            f"{path}.{name}"
            for path, wrapper in wrappers.items()
            for name, _ in wrapper.inner.named_parameters()
        }
        assert owned_by_large_inners == {f"{path}.weight" for path in _LARGE_PATHS}
        assert plain_decay - streamed_decay == owned_by_large_inners
        assert streamed_decay - plain_decay == set()
        params = dict(streamed.named_parameters())
        for name in owned_by_large_inners:
            assert params[name].is_meta, f"{name} is not a meta placeholder"
            assert not params[name].requires_grad, f"{name} is trainable"
        assert _lora(streamed).keys() <= streamed_decay

    def test_optimizer_groups_equal_the_resident_models(self, untied_pair):
        """What `Trainer.create_optimizer` actually builds: the decay and
        no-decay groups over TRAINABLE parameters. These must equal the
        resident model's exactly, or a streamed run trains a different
        recipe than the same config resident (default `weight_decay` 0.01)."""
        import torch
        from transformers.trainer_pt_utils import get_parameter_names

        def groups(model):
            decay = set(get_parameter_names(model, [torch.nn.LayerNorm]))
            with_decay = {n for n, p in model.named_parameters() if n in decay and p.requires_grad}
            without = {n for n, p in model.named_parameters() if n not in decay and p.requires_grad}
            return with_decay, without

        streamed, plain = untied_pair
        streamed_groups = groups(streamed)
        plain_groups = groups(plain)
        assert streamed_groups[0], "no trainable parameter in the decay group"
        assert streamed_groups == plain_groups

    def test_train_eval_and_apply_reach_the_large_inner_modules(self, untied_pair):
        streamed, _ = untied_pair
        wrappers = _large_wrappers(streamed)
        streamed.eval()
        assert all(not w.training and not w.inner.training for w in wrappers.values())
        streamed.train()
        assert all(w.training and w.inner.training for w in wrappers.values())
        seen = []
        streamed.apply(seen.append)
        assert all(w in seen and w.inner in seen for w in wrappers.values())
