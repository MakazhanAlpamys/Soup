"""#1165 — two Fast-LoRA patchers on one model compose by a stated precedence and
unpatch back to the model they started from.

The group kernels own their projections: shared-X Q/K/V owns q/k/v and the fused
SwiGLU MLP owns gate/up/down, whichever was patched first. The single-projection
kernel keeps every other LoRA linear. Unpatching any of them, in any order,
leaves no ``forward`` instance attribute and no marker behind, and unpatching a
kernel that is not installed changes nothing.
"""

from __future__ import annotations

import itertools
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from soup_cli.utils import fast_lora, fast_lora_mlp, fast_lora_qkv  # noqa: E402

PATCHERS = {
    "single": (
        fast_lora.patch_fast_lora_single_projection,
        fast_lora.unpatch_fast_lora_single_projection,
    ),
    "qkv": (fast_lora_qkv.patch_fast_lora_qkv, fast_lora_qkv.unpatch_fast_lora_qkv),
    "mlp": (fast_lora_mlp.patch_fast_lora_mlp, fast_lora_mlp.unpatch_fast_lora_mlp),
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _model():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=TARGETS
    )
    model = get_peft_model(LlamaForCausalLM(config), lora)
    # A trained adapter, so the fast kernels compute something peft would not
    # reproduce by accident with lora_B at zero.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.1)
    return model


def _layer(model):
    return model.base_model.model.model.layers[0]


def _structure(model):
    overrides = sorted(n for n, m in model.named_modules() if "forward" in vars(m))
    markers = sorted(
        (n, k) for n, m in model.named_modules() for k in vars(m) if k.startswith("_soup_")
    )
    return overrides, markers


def _kernels(model):
    """The autograd node each projection's output, and the MLP's, comes from."""
    layer = _layer(model)
    x = torch.randn(1, 3, 32, requires_grad=True)
    out = {
        name: type(getattr(layer.self_attn, name)(x).grad_fn).__name__
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    out["mlp"] = type(layer.mlp(x).grad_fn).__name__
    return out


def _logits(model):
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (1, 5))
    return model(input_ids=ids).logits.detach()


PAIRS = [("single", "qkv"), ("single", "mlp"), ("qkv", "mlp")]


@pytest.mark.parametrize(
    "patched, unpatched",
    [
        (patch_order, unpatch_order)
        for pair in PAIRS
        for patch_order in itertools.permutations(pair)
        for unpatch_order in itertools.permutations(pair)
    ],
)
def test_two_kernels_unpatch_to_the_starting_model(patched, unpatched):
    model = _model()
    start, start_kernels, start_logits = _structure(model), _kernels(model), _logits(model)
    for name in patched:
        PATCHERS[name][0](model)
    for name in unpatched:
        PATCHERS[name][1](model)
    # No module keeps an instance forward or a marker attribute.
    assert _structure(model) == start == ([], [])
    assert _kernels(model) == start_kernels
    assert torch.allclose(_logits(model), start_logits, atol=1e-6)


def test_the_single_projection_kernel_alone_leaves_no_instance_forward():
    model = _model()
    start = _structure(model)
    assert fast_lora.patch_fast_lora_single_projection(model) == len(TARGETS)
    assert fast_lora.unpatch_fast_lora_single_projection(model) == len(TARGETS)
    assert _structure(model) == start == ([], [])


def test_an_instance_forward_set_before_patching_comes_back():
    model = _model()
    q_proj = _layer(model).self_attn.q_proj
    q_proj.forward = q_proj.forward  # a caller's own override, as a hook would leave
    own = q_proj.__dict__["forward"]
    fast_lora.patch_fast_lora_single_projection(model)
    fast_lora.unpatch_fast_lora_single_projection(model)
    assert q_proj.__dict__["forward"] is own


def test_a_forward_installed_over_the_single_projection_kernel_stays():
    model = _model()
    fast_lora.patch_fast_lora_single_projection(model)
    q_proj = _layer(model).self_attn.q_proj

    def replacement(self, x, *args, **kwargs):
        return x

    q_proj.forward = types.MethodType(replacement, q_proj)
    assert fast_lora.unpatch_fast_lora_single_projection(model) == len(TARGETS)
    # The kernel gives up its markers, not a forward that is no longer its own.
    assert q_proj.__dict__["forward"].__func__ is replacement
    assert not [k for k in vars(q_proj) if k.startswith("_soup_")]


def test_mlp_leaves_projections_another_group_owns():
    model = _model()
    mlp = _layer(model).mlp
    setattr(mlp.gate_proj, fast_lora._GROUP_PATCH_OWNER_MARKER, "another-group")
    assert fast_lora_mlp.patch_fast_lora_mlp(model) == 0
    assert "forward" not in vars(mlp)
    assert getattr(mlp.gate_proj, fast_lora._GROUP_PATCH_OWNER_MARKER) == "another-group"
    for proj in (mlp.up_proj, mlp.down_proj):
        assert not [k for k in vars(proj) if k.startswith("_soup_")]


@pytest.mark.parametrize("order", [("single", "qkv"), ("qkv", "single")])
def test_qkv_owns_q_k_v_in_either_order(order):
    model = _model()
    for name in order:
        PATCHERS[name][0](model)
    kernels = _kernels(model)
    assert {kernels[p] for p in ("q_proj", "k_proj", "v_proj")} == {"_FastLoraQKVBackward"}
    assert kernels["o_proj"] == "_FastLoraSingleProjectionBackward"


@pytest.mark.parametrize("order", [("single", "mlp"), ("mlp", "single")])
def test_mlp_owns_gate_up_down_in_either_order(order):
    model = _model()
    for name in order:
        PATCHERS[name][0](model)
    mlp = _layer(model).mlp
    assert _kernels(model)["mlp"] == "_FastLoraSwiGLUBackward"
    for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        assert "forward" not in vars(proj), "a single-projection override under the MLP kernel"


def test_the_group_kernel_takes_over_from_the_single_projection_one():
    model = _model()
    assert fast_lora.patch_fast_lora_single_projection(model) == 7
    assert fast_lora_qkv.patch_fast_lora_qkv(model) == 1
    assert fast_lora_mlp.patch_fast_lora_mlp(model) == 1
    # Only o_proj is still the single-projection kernel's to restore.
    assert fast_lora.unpatch_fast_lora_single_projection(model) == 1


@pytest.mark.parametrize(
    "group, taken",
    [
        ("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("mlp", ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")),
    ],
)
def test_unpatching_a_group_kernel_does_not_bring_the_single_one_back(group, taken):
    model = _model()
    start_kernels = _kernels(model)
    fast_lora.patch_fast_lora_single_projection(model)
    PATCHERS[group][0](model)
    assert PATCHERS[group][1](model) == 1

    layer = _layer(model)
    for path in taken:
        proj = layer.get_submodule(path)
        assert "forward" not in vars(proj), path
        assert not [k for k in vars(proj) if k.startswith("_soup_")], path
    kernels = _kernels(model)
    # What the group kernel took is peft's again; o_proj is still the
    # single-projection kernel's, with the other group's three projections.
    for name in ("q_proj", "k_proj", "v_proj") if group == "qkv" else ("mlp",):
        assert kernels[name] == start_kernels[name]
    assert kernels["o_proj"] == "_FastLoraSingleProjectionBackward"
    assert fast_lora.unpatch_fast_lora_single_projection(model) == 4


def test_the_single_projection_kernel_skips_what_a_group_kernel_owns():
    model = _model()
    fast_lora_qkv.patch_fast_lora_qkv(model)
    fast_lora_mlp.patch_fast_lora_mlp(model)
    assert fast_lora.patch_fast_lora_single_projection(model) == 1


@pytest.mark.parametrize(
    "installed, absent",
    [("qkv", "single"), ("single", "qkv"), ("mlp", "single"), ("single", "mlp"), ("qkv", "mlp")],
)
def test_unpatching_a_kernel_that_is_not_installed_changes_nothing(installed, absent):
    model = _model()
    PATCHERS[installed][0](model)
    # A forward through the QKV kernel leaves its cache markers on the attention
    # module, so the snapshot is taken after one.
    before_kernels = _kernels(model)
    before = _structure(model)
    assert PATCHERS[absent][1](model) == 0
    assert _structure(model) == before
    assert _kernels(model) == before_kernels
