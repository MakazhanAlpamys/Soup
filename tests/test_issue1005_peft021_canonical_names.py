"""#1005 — a layer-streamed LoRA adapter saved as ZERO tensors under peft 0.21.0.

peft 0.21 selects the adapter entries of a state dict STRUCTURALLY: it collects key
prefixes from ``model.named_modules()`` (every tuner layer's attribute names) and keeps
the ``state_dict()`` entries under those prefixes. peft 0.20 filtered by the ``lora_``
substring. The streaming wrapper (``StreamedDecoderLayer``) held the real layer as a
child named ``inner``, so its module and parameter NAMES carried ``.inner.`` while its
``state_dict()`` KEYS were canonical (v0.72.1's serialisation-only delegation) — two
spellings of one tensor, and under 0.21 the prefixes were built from one and applied to
the other, so ``get_peft_model_state_dict`` returned ``{}`` and nothing raised.

The fix makes the wrapper's NAMES canonical too: ``named_modules()`` yields the inner
layer's subtree at the wrapper's own prefix and ``named_children()`` yields the inner
layer's children, so names and keys agree everywhere a consumer joins them — peft's
prefix filter, and transformers' ``get_parameter_names`` (the weight-decay grouping,
which walks ``named_children()``; Soup's default ``weight_decay`` is 0.01, so a
mismatch there would silently drop decay on every streamed run). ``train()`` /
``apply()`` / ``_apply()`` still reach the inner layer itself. Naming only: no bytes
on the streamed path change.

Every test here must hold on peft 0.20 AND 0.21 — the resident (non-streamed) LoRA
model on the same checkpoint is the reference, not a version table.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _requires_train_extra():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")


# --------------------------------------------------------------------------
# fixtures (standalone, mirroring tests/test_v07201.py)
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


def _build_streamed_cpu(tmp_path, n_layers=2):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(tmp_path, n_layers=n_layers)
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


@pytest.fixture
def pair(tmp_path):
    _requires_train_extra()
    streamed, runtime, weights = _build_streamed_cpu(tmp_path)
    plain = _build_plain_peft(weights)
    try:
        yield streamed, plain
    finally:
        runtime.close()


def _lora_names(model):
    return {name for name, _ in model.named_parameters() if "lora_" in name}


def _wrappers(model):
    """The streamed layer wrappers, found STRUCTURALLY (the container's entries),
    because `modules()` deliberately enumerates the inner layers in their place."""
    layers = model.get_submodule("base_model.model.model.layers")
    return [layer for layer in layers if type(layer).__name__ == "StreamedDecoderLayer"]


# --------------------------------------------------------------------------
# names and keys are ONE spelling
# --------------------------------------------------------------------------
class TestNamesAreCanonical:
    def test_named_modules_match_the_resident_model_exactly(self, pair):
        """The streamed tree names every module the way the resident tree does —
        no `.inner.` segment anywhere, nothing missing, nothing extra. peft 0.21
        derives its adapter prefixes from these names."""
        streamed, plain = pair
        streamed_names = {name for name, _ in streamed.named_modules()}
        plain_names = {name for name, _ in plain.named_modules()}
        assert not [n for n in streamed_names if ".inner." in n or n.endswith(".inner")]
        assert streamed_names == plain_names

    def test_named_parameters_match_the_resident_model_exactly(self, pair):
        streamed, plain = pair
        streamed_names = {name for name, _ in streamed.named_parameters()}
        plain_names = {name for name, _ in plain.named_parameters()}
        assert not [n for n in streamed_names if ".inner." in n]
        assert streamed_names == plain_names

    def test_every_parameter_name_is_a_state_dict_key(self, pair):
        """The two spellings peft 0.21 joins: the NAME (from named_modules /
        named_parameters) and the KEY (from state_dict). They must be the same
        string, or the adapter filter matches nothing."""
        streamed, _ = pair
        keys = set(streamed.state_dict().keys())
        names = {name for name, _ in streamed.named_parameters()}
        missing = sorted(names - keys)
        assert missing == [], f"parameter names absent from state_dict: {missing[:3]}"
        lora_keys = {k for k in keys if "lora_" in k}
        assert lora_keys == _lora_names(streamed)

    def test_the_real_decoder_layers_are_still_reachable_through_modules(self, pair):
        """Yielding the inner layer's subtree at the wrapper's name must not hide
        the inner layer OBJECT: `modules()` still visits it (isinstance scans
        over `modules()` are how transformers and peft find decoder layers)."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        streamed, plain = pair
        n_streamed = sum(isinstance(m, LlamaDecoderLayer) for m in streamed.modules())
        n_plain = sum(isinstance(m, LlamaDecoderLayer) for m in plain.modules())
        assert n_streamed == n_plain == 2


# --------------------------------------------------------------------------
# the consumers that join names to keys
# --------------------------------------------------------------------------
class TestConsumersAgree:
    def test_adapter_state_dict_matches_a_resident_lora_model(self, pair):
        """THE #1005 symptom. Same base, same LoRA config: the adapter state
        dict of the streamed model must carry exactly the keys the resident
        one does — non-empty, and no wrapper segment."""
        from peft import get_peft_model_state_dict

        streamed, plain = pair
        streamed_keys = set(get_peft_model_state_dict(streamed).keys())
        plain_keys = set(get_peft_model_state_dict(plain).keys())
        assert plain_keys, "the resident reference itself saved nothing — fixture broken"
        assert streamed_keys == plain_keys

    def test_saved_adapter_round_trips_into_the_resident_model(self, pair, tmp_path):
        """save_pretrained -> load into a plain model of the same base: every
        tensor lands by name and by value (lora_B is written non-zero first,
        because a dropped tensor is byte-identical to a fresh zero one)."""
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM

        streamed, plain = pair
        with torch.no_grad():
            for name, param in streamed.named_parameters():
                if "lora_B" in name:
                    param.copy_(torch.full_like(param, 0.05))
                elif "lora_A" in name:
                    param.copy_(torch.full_like(param, 0.025))
        out = tmp_path / "adapter"
        streamed.save_pretrained(str(out))
        saved = load_file(os.path.join(str(out), "adapter_model.safetensors"))
        assert saved, "the adapter file carries no tensors"
        assert not [k for k in saved if ".inner." in k]

        base = AutoModelForCausalLM.from_pretrained(
            plain.get_base_model().config._name_or_path, dtype=torch.float32
        )
        reloaded = PeftModel.from_pretrained(base, str(out))
        reloaded_b = [p for n, p in reloaded.named_parameters() if "lora_B" in n]
        assert reloaded_b and all(torch.all(p == 0.05) for p in reloaded_b), (
            "every lora_B should be 0.05 after the round trip; a zero one was dropped on load"
        )
        reloaded_a = [p for n, p in reloaded.named_parameters() if "lora_A" in n]
        assert reloaded_a and all(torch.all(p == 0.025) for p in reloaded_a), (
            "every lora_A should be 0.025 after the round trip"
        )

    def test_decay_parameter_names_reach_the_lora_parameters(self, pair):
        """transformers builds the weight-decay group by walking named_children()
        and joining the result with named_parameters(). If the two disagree on
        the streamed model, every LoRA parameter silently lands in the no-decay
        group (Soup's default weight_decay is 0.01)."""
        import torch
        from transformers.trainer_pt_utils import get_parameter_names

        streamed, plain = pair
        streamed_decay = set(get_parameter_names(streamed, [torch.nn.LayerNorm]))
        plain_decay = set(get_parameter_names(plain, [torch.nn.LayerNorm]))
        assert _lora_names(streamed) <= streamed_decay
        assert streamed_decay == plain_decay

    def test_the_wrapper_is_reachable_structurally_but_not_enumerated(self, pair):
        """The design choice, pinned: `named_modules()` enumerates the inner layer
        in the wrapper's place (an attribute-transparent wrapper answering
        `hasattr` in transformers' gradient-checkpointing walk would take the
        `setattr` that belongs to the real layer), while the wrapper itself
        stays where structural walks find it — the container's children,
        `get_submodule`, attribute access."""
        streamed, _ = pair
        wrappers = _wrappers(streamed)
        assert len(wrappers) == 2
        enumerated = {id(m) for m in streamed.modules()}
        assert all(id(w) not in enumerated for w in wrappers)
        assert all(id(w.inner) in enumerated for w in wrappers)
        assert streamed.get_submodule("base_model.model.model.layers.0") is wrappers[0]
        by_name = dict(streamed.named_modules())
        assert by_name["base_model.model.model.layers.0"] is wrappers[0].inner

    def test_train_and_eval_reach_the_inner_layer(self, pair):
        """named_children() no longer lists `inner`, so Module.train() would skip
        it unless the wrapper forwards the call; the inner layer's `training`
        flag must follow the model's."""
        streamed, _ = pair
        wrappers = _wrappers(streamed)
        assert len(wrappers) == 2
        streamed.eval()
        assert all(not w.training and not w.inner.training for w in wrappers)
        streamed.train()
        assert all(w.training and w.inner.training for w in wrappers)

    def test_apply_reaches_the_inner_layer_and_the_wrapper(self, pair):
        streamed, _ = pair
        seen = []
        streamed.apply(lambda m: seen.append(m))
        wrappers = [m for m in seen if type(m).__name__ == "StreamedDecoderLayer"]
        assert len(wrappers) == 2
        assert all(w.inner in seen for w in wrappers)
